import numpy as np
import os
import torch
from Rep_count import Rep_count
from decord import VideoReader, cpu, gpu
from video_mae_cross_full_attention import SupervisedMAE
from util.config import load_config
import argparse
import tqdm
from torchvision.transforms import Resize, CenterCrop, Normalize, Compose
import einops

torch.manual_seed(0)


def get_args_parser():
    parser = argparse.ArgumentParser('MAE encoding', add_help=False)
    parser.add_argument('--use_v1', default=False, help='use the v1 variant of the encoder')
    parser.add_argument('--config', default='configs/pretrain_config.yaml', help="config file")

    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--num_gpus', default=1, type=int)
    parser.add_argument('--pretrained_encoder', default='pretrained_models/VIT_B_16x4_MAE_PT.pth', type=str)
    parser.add_argument('--save_exemplar_encodings', default=True, type=bool)
    parser.add_argument('--dataset', default='RepCount', help='choose from [RepCount, Countix, UCFRep]', type=str)
    parser.add_argument('--model', default='VideoMAE', help="VideoMAE, VideoSwin")
    parser.add_argument('--encodings', default='mae', help="mae, swin, resnext")
    parser.add_argument('--data_path', default='D:/datasets/RepCount/video', help='data path for the dataset')
    return parser


def read_video_timestamps(video_filename, timestamps, duration=0):
    """ 
    summary

    Args:
        video_filename (string): full filepath of the video
        timestamps (list): list of ints for the temporal points to load from the file

    Returns:
        frames: tensor of shape C x T x H x W
        totfps: float for the video segment length (in secs) 实际上返回的是一个最后面的帧号
    """
    try:
        assert os.path.isfile(video_filename), f"VideoLoader: {video_filename} does not exist"
    except:
        print(f"{video_filename} does not exist")

    # 用 decord 实现, cpu结果是准确的，但 gpu 还有问题
    vr = VideoReader(video_filename, ctx=cpu(0))
    frames2 = vr.get_batch(timestamps)
    frames2 = torch.from_numpy(frames2.asnumpy())
    video_frames = frames2.permute((3, 0, 1, 2)).to(torch.float32)

    return video_frames, timestamps[-1]


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


def save_exemplar(dataloaders, model, args):
    '''
    This function extracts the encodings for every repetition in each video by uniformly sampling 16 frames
    within the repetition segments and saves these encodings as npz format. The input to the encoder is
    B*3xTxHxW, where B is the total number of repetitions in the selected video. The output is spatio-temporal
    tokens of shape Bx(T'H'W')xC. We save these encodigns as BxCxT'xH'xW'.
    inputs: a dict consisting of 'train', 'val' and 'test' dataloaders,
         the pretrained model,
         other parameters needed
    '''

    num_frames = 16
    splits = ['train', 'val', 'test']

    target_dir = f'd:/datasets/ESCount/exemplar_{args.model}tokens_new_{args.dataset}'
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    for split in splits:
        # Get file list in each split
        # video_list = os.listdir(os.path.join(args.data_path, split))
        for item in tqdm.tqdm(dataloaders[split], total=len(dataloaders[split])):
            C, Total, H, W = item[0]            # Total: # of total frames
            if Total == 0:
                continue

            video_name = item[-1][0]
            starts = item[-3][0]            # Get 1 exemplar only
            ends = item[-2][0]
            base_name = os.path.basename(video_name)[:-4]

            if os.path.exists(target_dir + '/' + base_name + '.npz'):
                continue

            vr = VideoReader(video_name, ctx=cpu(0))
            
            chunk_size = 16
            clip_list = []
            merge = []
            num_exemplars = len(starts)
            
            for j in range(num_exemplars):
                s = starts[j].item()  ## start times of each repetition
                e = ends[j].item()  ## end times of each repetition
                if s == e:
                    continue
                idx = np.linspace(s, min(e, Total - 1), num_frames + 1)[:num_frames].astype(int)
                ###sample 16 frames from the repetition segment defined by the start and end

                clip = torch.from_numpy(vr.get_batch(idx).asnumpy()).to(dtype=torch.float32)    # (B, H, W, C)
                clip = clip.permute(0, 3, 1, 2).contiguous()                        # (B, C, H, W)
                clip = preprocess(clip / 255.)
                clip_list.append(clip)

                if len(clip_list) >= chunk_size or j == num_exemplars-1:
                    data = torch.stack(clip_list).permute(0, 2, 1, 3, 4).to("cuda")
                    with torch.no_grad():
                        if args.model == 'VideoMAE':
                            try:
                                encoded, thw = model(data)  ## extract encodings
                            except:
                                print(f"video: {video_name} exception")
                                continue
                            encoded = encoded.transpose(1, 2).reshape(encoded.shape[0], encoded.shape[-1], thw[0], thw[1], thw[2])  # reshape to B x C x T x H x W
                        else:
                            encoded = model(data)

                    enc_np = encoded.cpu().numpy()
                    merge.append(enc_np)

                    del encoded, data
                    clip_list.clear()
                    torch.cuda.empty_cache()
                    

            if len(merge) == 0:
                print("video with 0 examplar: ", video_name)
                continue

            # 一个clip对应的维度 (768,8,14,14) ，1个示例样本就一个clip，占用内存不大
            merged = np.concatenate(merge, 0)
            np.savez('{}/{}.npz'.format(target_dir, base_name), merged)  ##save as npz
            del merged


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
    splits = ['train', 'val', 'test']

    target_dir = f'd:/datasets/ESCount/saved_{args.model}tokens_new_{args.dataset}'
    if not os.path.isdir(target_dir):
        print('Creating folder')
        os.makedirs(target_dir)

    # cuda内存使用统计复位
    # torch.cuda.reset_peak_memory_stats()

    for split in splits:
        for item in tqdm.tqdm(dataloaders[split], total=len(dataloaders[split])):
            C, Total, H, W = item[0]            # Total: # of total frames

            if Total == 0:
                continue

            video_name = item[-1][0]
            base_name = os.path.basename(video_name)[:-4]

            if os.path.exists(target_dir + '/' + base_name + '.npz'):
                continue

            vr = VideoReader(video_name, ctx=cpu(0))
            clip_list = []
            merge = []
            chunk_size = 16  # 16个clip一个批次

            for j in range(0, Total, 16):  #### 75% overlap
                idx = np.linspace(j, j + 64, num_frames + 1)[:num_frames].astype(int)  ### sample 16 frames from windows of 64 frames
                
                valid_idx = [i for i in idx if i < Total]
                padding_idx = [i for i in idx if i >= Total]
                
                clip_valid = torch.from_numpy(vr.get_batch(valid_idx).asnumpy()).to(dtype=torch.float32)    # (B, H, W, C)
                clip_valid = clip_valid.permute(0, 3, 1, 2).contiguous()                                    # (B, C, H, W)
                clip_valid = preprocess(clip_valid / 255.)                                  # Norm, resize & crop valid frames first. shape: H*W -> 224*224

                clip_padding = torch.zeros([len(padding_idx), C, 224, 224])                 # H, W are changed to 224, 224

                clip = torch.cat((clip_valid, clip_padding), dim=0) if len(padding_idx) > 0 else clip_valid
                clip_list.append(clip)

                # 数据太大，GPU显存/内存装不下，分块处理。一个chunk包含多个clip，或者说多个 segment
                if len(clip_list) >= chunk_size or j+16 >= Total:
                    data = torch.stack(clip_list).permute(0, 2, 1, 3, 4).to("cuda")

                    with torch.no_grad():
                        if args.model == 'VideoMAE':
                            try:
                                encoded, thw = model(data)  ### extract encodings
                            except:
                                print(f"video: {video_name} exception")
                                raise Exception(1)

                            encoded = encoded.transpose(1, 2).reshape(encoded.shape[0], encoded.shape[-1], thw[0], thw[1], thw[2])  # reshape to B x C x T x H x W
                        else:
                            encoded = model(data)

                    enc_np = encoded.cpu().numpy()

                    del encoded, thw, data
                    torch.cuda.empty_cache()

                    merge.append(enc_np)
                    clip_list.clear()
            
            # 按照 float存放，每个clip的16帧图像数据编码为 (768,8,14,14)，约4.8MB。300帧的图像，分成 19个 clip（或segment），内存能承受
            merged = np.concatenate(merge, 0)

            np.savez('{}/{}.npz'.format(target_dir, base_name), merged)  ### saving as npz
            del merge, merged

            # max_reserved = torch.cuda.max_memory_reserved()
            # max_alloc = torch.cuda.max_memory_allocated()
            # print(f" {video_name}: shape[{C}, {T}, {H}, {W}], max reserved: {max_reserved / 1e9:.2f} GB, max allocated: {max_alloc / 1e9:.2f} GB")
            # torch.cuda.reset_peak_memory_stats()


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

    # model = nn.parallel.DataParallel(model, device_ids=[i for i in range(args.num_gpus)])

    for name in model.state_dict().keys():
        if 'decoder' in name or 'decode_heads' in name:
            continue

        matched = 0

        for name_, param in state_dict.items():
            # if args.num_gpus > 1:
            if 'encoder.' in name_:
                name_ = name_.replace('encoder.', '')
            name_ = f'module.{name_}'

            # pdb.set_trace()
            if name_ == name:
                model.state_dict()[name].copy_(param)
                matched = 1
                break
        if matched == 0 and '.qkv.' in name:
            if not args.use_v1:
                q_name = name.replace('.qkv.', '.q.').replace('module.', '')
                k_name = name.replace('.qkv.', '.k.').replace('module.', '')
                v_name = name.replace('.qkv.', '.v.').replace('module.', '')
                params = torch.cat([state_dict[q_name], state_dict[k_name], state_dict[v_name]])
                model.state_dict()[name].copy_(params)
                matched = 1
                break
            else:
                if '.qkv.bias' in name:
                    q_name = name.replace('.qkv.', '.q_').replace('module.', 'encoder.')
                    v_name = name.replace('.qkv.', '.v_').replace('module.', 'encoder.')
                    params = torch.cat([state_dict[q_name], torch.zeros_like(state_dict[v_name], requires_grad=False), state_dict[v_name]])
                    model.state_dict()[name].copy_(params)
                    matched = 1
                    break
        if matched == 0:
            print(f"parameters {name} not found")

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

    if args.save_exemplar_encodings:
        save_exemplar(dataloaders, model, args)
    else:
        save_tokens(dataloaders, model, args)


if __name__ == '__main__':
    main()
