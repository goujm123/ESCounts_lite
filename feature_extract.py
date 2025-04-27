import numpy as np
import os
import torch
import cv2
from Rep_count import Rep_count
from decord import VideoReader, cpu, gpu
from video_mae_cross_full_attention import SupervisedMAE
from util.config import load_config
import argparse
import tqdm
from torchvision.transforms import Resize, CenterCrop, Normalize, Compose
from mediapipe.python.solutions import pose as mp_pose
import einops

torch.manual_seed(0)

BLAZEPOSE_CONNECTIONS = frozenset([(0, 1), (1, 2), (2, 3), (3, 7),
                              (0, 4), (4, 5), (5, 6), (6, 8),
                              (9, 10),
                              (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
                              (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)])


def get_args_parser():
    parser = argparse.ArgumentParser('MAE encoding', add_help=False)
    parser.add_argument('--use_v1', default=False, help='use the v1 variant of the encoder')
    parser.add_argument('--config', default='configs/pretrain_config.yaml', help="config file")

    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--num_gpus', default=1, type=int)
    parser.add_argument('--pretrained_encoder', default='pretrained_models/VIT_B_16x4_MAE_PT.pth', type=str)
    parser.add_argument('--dataset', default='RepCount', help='choose from [RepCount, Countix, UCFRep]', type=str)
    parser.add_argument('--model', default='VideoMAE', help="VideoMAE, VideoSwin")
    parser.add_argument('--encodings', default='mae', help="mae, swin, resnext")
    parser.add_argument('--data_path', default='D:/datasets/RepCount/video', help='data path for the dataset')
    return parser


def preprocess(tensor, min_size=224, crop_size=224, video_mean=[0.485, 0.456, 0.406], video_std=[0.229, 0.224, 0.225]):
    T, C, H, W = tensor.shape

    crop = CenterCrop(crop_size)
    resize = Resize(min_size, antialias=False)
    normalize = Normalize(mean=video_mean, std=video_std)

    data = tensor
    data = normalize(data)
    data = resize(data)
    data = crop(data)

    return data


def draw_blazepose_landmarks(image, landmarks, draw_connections=True):
    """
    在RGB图像上绘制BlazePose关键点和骨架连线
    :param image: 原始RGB图像 (H, W, 3)
    :param landmarks: 33个landmark的list, 每个为[x, y, z, visibility]
    :param draw_connections: 是否绘制骨架连线
    :return: 带可视化的图像
    """
    h = image.shape[0]
    w = image.shape[1]
    img = np.zeros((h, w, 3), dtype=np.uint8)       # 当前是三通道
    landmarks = np.array(landmarks)

    # 绘制关键点
    for idx, landmark in enumerate(landmarks):
        x, y, z, v = landmark
        if v < 0.5:
            continue
        cv2.circle(img, (int(x), int(y)), radius=3, color=[255, 255, 255], thickness=-1)

    # 绘制骨架连线
    if draw_connections:
        for start, end in BLAZEPOSE_CONNECTIONS:
            v_start = landmarks[start][3]
            v_end = landmarks[end][3]
            if v_start < 0.5 or v_end < 0.5:
                continue
            x1, y1 = int(landmarks[start][0]), int(landmarks[start][1])
            x2, y2 = int(landmarks[end][0]), int(landmarks[end][1])
            cv2.line(img, (x1, y1), (x2, y2), [255, 255, 255], 2)

    return img


def save_tokens(dataloaders, model, args):
    '''
    This function extracts the encodings for each video using windows of 64 frames and then sampling 16 frames uniformly from these windows.
    We save the encodings in npz format. The input to the encoder is B*3x16xHxW, where B is the batch size and each batch comprises overlapping windows in each video.
    The output is spatio-temporal tokens of shape Bx(T'H'W')xC. We save these encodings as BxCxT'xH'xW'.
    inputs: a dict consisting of 'train', 'val' and 'test' dataloaders,
         the pretrained model,
         other parameters needed
    '''

    num_frames = 16
    splits = ['train']

    target_dir = f'd:/datasets/RepCount/tokens'
    if not os.path.isdir(target_dir):
        print('Creating folder')
        os.makedirs(target_dir)

    # cuda内存使用统计复位
    # torch.cuda.reset_peak_memory_stats()

    with mp_pose.Pose() as pose_estimator:
        for split in splits:
            for item in tqdm.tqdm(dataloaders[split], total=len(dataloaders[split])):
                C, Total, H, W = item[0]  # Total: # of total frames

                if Total == 0:
                    continue

                video_name = item[-1][0]
                base_name = os.path.basename(video_name)[:-4]

                if os.path.exists(target_dir + '/' + base_name + '.npz'):
                    continue

                vr = VideoReader(video_name, ctx=cpu(0))
                rgb = []
                pose = []

                for j in range(0, Total, 16):  #### 75% overlap， 每个窗口覆盖64帧，滑动窗口step为16
                    idx = np.linspace(j, j + 64, num_frames + 1)[:num_frames].astype(int)  ### sample 16 frames from windows of 64 frames

                    valid_idx = [i for i in idx if i < Total]
                    padding_idx = [i for i in idx if i >= Total]
                    batch_clip = vr.get_batch(valid_idx).asnumpy() # (B, H, W, 3)


                    clip_valid = torch.from_numpy(batch_clip)  # (B, H, W, C)
                    clip_valid = clip_valid.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
                    clip_padding = torch.zeros([len(padding_idx), C, H, W]).to(dtype=torch.uint8)

                    clip = torch.cat((clip_valid, clip_padding), dim=0) if len(padding_idx) > 0 else clip_valid
                    # 可以先进行 pose 处理（注意 BlazePose一次只能处理一帧RGB图像，需要for each in batch）
                    pose_input = pose_feature_extract(clip, pose_estimator)
                    pose_input = preprocess(pose_input / 255.)
                    pose_input = torch.unsqueeze(pose_input, 0).permute(0, 2, 1, 3, 4).to("cuda")
                    
                    # rgb_feature_extract
                    rgb_input = preprocess(clip / 255.)  # Norm, resize & crop valid frames first. shape: H*W -> 224*224
                    rgb_input = torch.unsqueeze(rgb_input, 0).permute(0, 2, 1, 3, 4).to("cuda")
                    

                    # 这里的模型为VideoMAE
                    with torch.no_grad():
                        try:
                            encoded_pose, thw_pose = model(pose_input)
                            encoded_rgb, thw_rgb = model(rgb_input)                 ### get encodings from pose & rgb
                        except:
                            print(f"video: {video_name} exception")
                            raise Exception(1)

                        encoded_rgb = encoded_rgb.transpose(1, 2).reshape(encoded_rgb.shape[0], encoded_rgb.shape[-1], thw_rgb[0], thw_rgb[1], thw_rgb[2])  # reshape to B x C x T x H x W
                        encoded_pose = encoded_pose.transpose(1, 2).reshape(encoded_pose.shape[0], encoded_pose.shape[-1], thw_pose[0], thw_pose[1], thw_pose[2])  # reshape to B x C x T x H x W

                    rgb.append(encoded_rgb.cpu().numpy())
                    pose.append(encoded_pose.cpu().numpy())

                    torch.cuda.empty_cache()
                    del clip, clip_valid, clip_padding, batch_clip, pose_input, rgb_input
                
                # 按照 float存放，每个clip的16帧图像数据编码为 (768,8,14,14)，约4.8MB。300帧的图像，分成 19个 clip（或segment），内存能承受
                merged = np.concatenate(rgb, 0)
                np.savez('{}/{}_rgb.npz'.format(target_dir, base_name), merged)  ### saving as npz
                merged = np.concatenate(pose, 0)
                np.savez('{}/{}_pose.npz'.format(target_dir, base_name), merged)
                
                

                # max_reserved = torch.cuda.max_memory_reserved()
                # max_alloc = torch.cuda.max_memory_allocated()
                # print(f" {video_name}: shape[{C}, {T}, {H}, {W}], max reserved: {max_reserved / 1e9:.2f} GB, max allocated: {max_alloc / 1e9:.2f} GB")
                # torch.cuda.reset_peak_memory_stats()


def pose_feature_extract(clip, pose_estimator):
    B, C, H, W = clip.shape
    images = []
    
    for i in range(clip.shape[0]):
        frame = clip[i, :]                    # C, H, W
        frame = frame.permute(1, 2, 0).numpy()      # H, W, C
        pose_result = pose_estimator.process(image=frame)
        pose_landmarks = pose_result.pose_landmarks
        
        if pose_landmarks:
            frame_height, frame_width = frame.shape[0], frame.shape[1]
            # landmarks 维度：33*4
            landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width, lmk.visibility] for lmk in pose_landmarks.landmark], dtype=np.float32)
            assert landmarks.shape == (33, 4), 'Unexpected landmarks shape: {}'.format(landmarks.shape)
            vis_img = draw_blazepose_landmarks(frame, landmarks)        # ndarray
            images.append(vis_img)
            
            # cv2.imshow("BlazePose Visualization", vis_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else:
            images.append(np.zeros((H, W, C), dtype=np.uint8))
    
    # result = torch.tensor(images).permute(0, 3, 1, 2)
    result = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
    
    return result


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    args.opts = None

    cfg = load_config(args)

    model = SupervisedMAE(cfg=cfg, just_encode=True, use_precomputed=False, encodings=args.encodings).cuda()
    if args.pretrained_encoder:
        state_dict = torch.load(args.pretrained_encoder)
        if 'model_state' in state_dict.keys():
            state_dict = state_dict['model_state']
        else:
            state_dict = state_dict['model']
    else:
        print("You should download VIT_B_16x4_MAE_PT.pyth manually.")

    # 替换为 demo 中的参数加载代码
    for name in model.state_dict().keys():
        if 'decode' in name:
            continue
        matched = 0

        for name_, param in state_dict.items():
            if name_ == name:
                model.state_dict()[name].copy_(param)
                matched = 1
                break
            elif '.qkv.' in name and 'blocks' in name:
                q_name = name.replace('.qkv.', '.q.').replace('module.', '')
                k_name = name.replace('.qkv.', '.k.').replace('module.', '')
                v_name = name.replace('.qkv.', '.v.').replace('module.', '')
                params = torch.cat([state_dict[q_name], state_dict[k_name], state_dict[v_name]])
                model.state_dict()[name].copy_(params)
                matched = 1
                break

    model.eval()

    if args.dataset == 'RepCount':
        dataset_train = Rep_count(cfg=cfg, split="train", data_dir=args.data_path, sampling_interval=1, encode_only=True)
        dataset_val = Rep_count(cfg=cfg, split="valid", data_dir=args.data_path, sampling_interval=1, encode_only=True)
        dataset_test = Rep_count(cfg=cfg, split="test", data_dir=args.data_path, sampling_interval=1, encode_only=True)

    dataloaders = {'train': torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                                        num_workers=1,
                                                        shuffle=False,
                                                        pin_memory=True,
                                                        drop_last=False),
                   'val': torch.utils.data.DataLoader(dataset_val,
                                                      batch_size=args.batch_size,
                                                      num_workers=1,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      drop_last=False),
                   'test': torch.utils.data.DataLoader(dataset_test,
                                                       batch_size=args.batch_size,
                                                       num_workers=1,
                                                       shuffle=False,
                                                       pin_memory=True,
                                                       drop_last=False),
                   }

    save_tokens(dataloaders, model, args)


if __name__ == '__main__':
    main()
