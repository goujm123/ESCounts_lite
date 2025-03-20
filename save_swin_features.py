import numpy as np
import os
import torch
from Rep_count import Rep_count

from video_mae_cross_full_attention import SupervisedMAE
from util.config import load_config
import argparse
import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser('MAE encoding', add_help=False)
    parser.add_argument('--use_v1', default=False, help='use the v1 variant of the encoder')
    parser.add_argument('--config', default='configs/pretrain_config.yaml', help="config file")

    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--num_gpus', default=1, type=int)
    parser.add_argument('--pretrained_encoder', default='pretrained_models/VIT_B_16x4_MAE_PT.pth', type=str)
    parser.add_argument('--save_exemplar_encodings', default=False, type=bool)
    parser.add_argument('--dataset', default='RepCount', help='choose from [RepCount, Countix, UCFRep]', type=str)
    parser.add_argument('--model', default='VideoMAE', help="VideoMAE, VideoSwin")
    parser.add_argument('--encodings', default='mae', help="mae, swin, resnext")
    parser.add_argument('--data_path', default='D:/datasets/RepCount/video', help='data path for the dataset')
    return parser


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

    target_dir = f'd:/datasets/ESCount/exemplar_{args.model}tokens_{args.dataset}'
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    for split in splits:
        for item in tqdm.tqdm(dataloaders[split], total=len(dataloaders[split])):
            video = item[0].squeeze(0)
            starts = item[-3][0]
            ends = item[-2][0]
            video_name = item[-1][0]
            C, T, H, W = video.shape

            clip_list = []
            num_exemplars = len(starts)
            for j in range(num_exemplars):
                s = starts[j].item()  ## start times of each repetition
                e = ends[j].item()  ## end times of each repetition
                if s == e:
                    continue
                idx = np.linspace(s, min(e, video.shape[1] - 1), num_frames + 1)[:num_frames].astype(int)
                ###sample 16 frames from the repetition segment defined by the start and end
                clips = video[:, idx]
                clip_list.append(clips)

            # 数据太大，GPU显存装不下，分块处理
            chunk_size = 16
            merge = []
            for i in range(0, len(clip_list), chunk_size):
                chunk = clip_list[i:i + chunk_size]

                data = torch.stack(chunk).cuda()  ### batch of repetitions
                with torch.no_grad():
                    if args.model == 'VideoMAE':
                        try:
                            encoded, thw = model(data)  ## extract encodings
                        except:
                            print(f"video: {video_name} exception, shape[{C}, {T}, {H}, {W}]")
                            continue
                        encoded = encoded.transpose(1, 2).reshape(encoded.shape[0], encoded.shape[-1], thw[0], thw[1], thw[2])  # reshape to B x C x T x H x W
                    else:
                        encoded = model(data)

                enc_np = encoded.cpu().numpy()
                merge.append(enc_np)

                del encoded, data
                torch.cuda.empty_cache()

            if len(merge) == 0:
                print("video with 0 examplar: ", video_name)
                continue

            # 一个clip对应的维度 (768,8,14,14) ，1个示例样本就一个clip，占用内存不大
            merged = np.concatenate(merge, 0)
            np.savez('{}/{}.npz'.format(target_dir, video_name), merged)  ##save as npz
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

    target_dir = f'd:/datasets/ESCount/saved_{args.model}tokens_{args.dataset}'
    if not os.path.isdir(target_dir):
        print('Creating folder')
        os.makedirs(target_dir)

    # cuda内存使用统计复位
    # torch.cuda.reset_peak_memory_stats()

    for split in splits:
        for item in tqdm.tqdm(dataloaders[split], total=len(dataloaders[split])):
            video = item[0].squeeze(0)
            video_name = item[-1][0]
            C, T, H, W = video.shape
            padding = torch.zeros([C, 64, H, W])  ### add padding of zeros at the end
            video = torch.cat([video, padding], 1)

            clip_list = []
            for j in range(0, T, 16):  #### 75% overlap
                idx = np.linspace(j, j + 64, num_frames + 1)[:num_frames].astype(int)  ### sample 16 frames from windows of 64 frames
                clips = video[:, idx]
                clip_list.append(clips)

            # 数据太大，GPU显存装不下，分块处理。一个chunk包含多个clip，或者说多个 segment
            chunk_size = 16  # 16个clip一个批次
            merge = []
            for i in range(0, len(clip_list), chunk_size):
                chunk = clip_list[i:i + chunk_size]
                data = torch.stack(chunk).cuda()

                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=True):
                        if args.model == 'VideoMAE':
                            try:
                                encoded, thw = model(data)  ### extract encodings
                            except:
                                print(f"video: {video_name} exception, shape[{C}, {T}, {H}, {W}]")
                                continue

                            encoded = encoded.transpose(1, 2).reshape(encoded.shape[0], encoded.shape[-1], thw[0], thw[1], thw[2])  # reshape to B x C x T x H x W
                        else:
                            encoded = model(data)

                enc_np = encoded.cpu().numpy()

                del encoded, thw, data
                torch.cuda.empty_cache()

                merge.append(enc_np)

            # 按照 float存放，每个clip的16帧图像数据编码为 (768,8,14,14)，约4.8MB。300帧的图像，分成 19个 clip（或segment），内存能承受
            merged = np.concatenate(merge, 0)
            np.savez('{}/{}.npz'.format(target_dir, video_name), merged)  ### saving as npz
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
