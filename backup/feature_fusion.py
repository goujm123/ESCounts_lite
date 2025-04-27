
import torch
import torch.nn as nn
import numpy as np
import cv2
from decord import VideoReader
from mediapipe.python.solutions import pose as mp_pose
from backup.pose_feature_extract import BLAZEPOSE_CONNECTIONS


class CrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, rgb_feat, pose_feat):
        # CrossAttention: Query=rgb, Key/Value=pose
        attn_out, _ = self.cross_attn(query=rgb_feat, key=pose_feat, value=pose_feat)
        x = self.norm(rgb_feat + attn_out)
        x_ffn = self.ffn(x)
        out = self.norm2(x + x_ffn)
        return out


def draw_blazepose_landmarks(image, landmarks, draw_connections=True):
    h, w = image.shape[:2]
    img = np.zeros((h, w), dtype=np.uint8)
    # landmarks: 33x4
    for idx, landmark in enumerate(landmarks):
        x, y, z, v = landmark
        if v < 0.5:
            continue
        cv2.circle(img, (int(x), int(y)), 3, [255], -1)

    # 绘制关键点
    for idx, landmark in enumerate(landmarks):
        x, y, z, v = landmark
        if v < 0.5:
            continue
        cv2.circle(img, (int(x), int(y)), 3, [255], -1)

    # 绘制骨架连线
    if draw_connections:
        for start, end in BLAZEPOSE_CONNECTIONS:
            v_start = landmarks[start][3]
            v_end = landmarks[end][3]
            if v_start < 0.5 or v_end < 0.5:
                continue
            x1, y1 = int(landmarks[start][0]), int(landmarks[start][1])
            x2, y2 = int(landmarks[end][0]), int(landmarks[end][1])
            cv2.line(img, (x1, y1), (x2, y2), [128], 2)
            
    return img

def extract_pose_landmarks(frame, pose_estimator):
    result = pose_estimator.process(image=frame)
    pose_landmarks = result.pose_landmarks
    if pose_landmarks is not None:
        h, w = frame.shape[:2]
        lmks = np.array([[lmk.x * w, lmk.y * h, lmk.z * w, lmk.visibility] for lmk in pose_landmarks.landmark], dtype=np.float32)
        return lmks
    else:
        return None

def process_video_clip(video_path, videomae_model, fusion_model, device='cuda'):
    vr = VideoReader(video_path)
    num_frames = len(vr)
    clip_len = 16
    all_fused_features = []

    with mp_pose.Pose() as pose_estimator:
        for start in range(0, num_frames - clip_len + 1, clip_len):
            rgb_clip = []
            pose_clip = []
            for idx in range(start, start + clip_len):
                frame = vr[idx].asnumpy()
                rgb_clip.append(frame)
                lmks = extract_pose_landmarks(frame, pose_estimator)
                if lmks is not None:
                    pose_img = draw_blazepose_landmarks(frame, lmks)
                else:
                    pose_img = np.zeros(frame.shape[:2], dtype=np.uint8)
                pose_clip.append(pose_img)

            rgb_clip = np.stack(rgb_clip)  # [16, H, W, 3]
            pose_clip = np.stack(pose_clip)  # [16, H, W]

            # [B, C, T, H, W]，假设batch=1
            rgb_tensor = torch.from_numpy(rgb_clip).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
            pose_tensor = torch.from_numpy(pose_clip).unsqueeze(1).permute(1, 0, 2, 3).unsqueeze(0).float() / 255.0
            rgb_tensor = rgb_tensor.to(device)
            pose_tensor = pose_tensor.to(device)

            with torch.no_grad():
                rgb_feature = videomae_model(rgb_tensor)  # [1, 768, 8, 14, 14]
                pose_feature = videomae_model(pose_tensor)  # [1, 768, 8, 14, 14]

                # 对空间维度做平均池化，得到 [B, 768, 8]
                rgb_feature = rgb_feature.mean(dim=[3, 4])  # [1, 768, 8]
                pose_feature = pose_feature.mean(dim=[3, 4])  # [1, 768, 8]

                # 转为 [B, 8, 768]
                rgb_feature = rgb_feature.permute(0, 2, 1)  # [1, 8, 768]
                pose_feature = pose_feature.permute(0, 2, 1)  # [1, 8, 768]

                fused_feature = fusion_model(rgb_feature, pose_feature)  # [1, 8, 768]

            all_fused_features.append(fused_feature.cpu())

    # 拼接所有clip的特征
    all_fused_features = torch.cat(all_fused_features, dim=1)  # [1, 总帧数, 768]
    return all_fused_features

device = 'cuda' if torch.cuda.is_available() else 'cpu'
videomae_model = ...  # 加载VideoMAE模型
fusion_model = CrossAttentionFusion(feature_dim=768).to(device)
video_path = 'your_video.mp4'
fused_features = process_video_clip(video_path, videomae_model, fusion_model, device)

