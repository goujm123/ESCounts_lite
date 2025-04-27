# 使用 BlazePose 提取每帧图像的关键点。关键点数量：33
# 每个视频保存为一个csv文件，文件名后半部分包含总帧数信息。 csv每行包含帧id、每个landmark的映射到图像坐标上的xyz坐标值及visibility

import os
import csv
from decord import VideoReader
import numpy as np
from tqdm import tqdm
from mediapipe.python.solutions import pose as mp_pose

import cv2
from PIL import Image, ImageDraw, ImageFont

video_root = "D:/datasets/RepCount/video"
out_csv_dir = "d:/datasets/RepCount/pose_csv"
out_pose_feature_dir = "d:/datasets/RepCount/pose_feature"
os.makedirs(out_csv_dir, exist_ok=True)
os.makedirs(out_pose_feature_dir, exist_ok=True)

splits = ['train', 'valid', 'test']

# BlazePose 33个关键点的骨架连线定义（每对为起点和终点的索引）
BLAZEPOSE_CONNECTIONS = frozenset([(0, 1), (1, 2), (2, 3), (3, 7),
                              (0, 4), (4, 5), (5, 6), (6, 8),
                              (9, 10),
                              (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
                              (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)])

# 每个clip包含16帧，要和 RGB 特征提取的帧一致
CLIP_LEN = 16


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
    img = np.zeros((h, w), dtype=np.uint8)
    landmarks = np.array(landmarks)

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


for split in splits:
    print("split: ", split)
    video_dir = os.path.join(video_root, split)

    # Initialize fresh pose tracker and run it
    with mp_pose.Pose() as pose_estimator:
        for i, video_file in enumerate(os.listdir(video_dir)):
            video = os.path.join(video_dir, video_file)
            vr = VideoReader(video)
            num_frames = len(vr)
            print(f"{i}, {video} has number of frames: {num_frames}")

            os.makedirs(os.path.join(out_csv_dir, split), exist_ok=True)
            out_csv_path = os.path.join(out_csv_dir, split, video_file) + f'_{num_frames}.csv'
            out_csv = []  # write all poses in a video to a file, one pose per frame

            for j in tqdm(range(num_frames)):   # BlazePose 一次只能处理一帧，不支持batch
                frame = vr[j].asnumpy()
                result = pose_estimator.process(image=frame)
                pose_landmarks = result.pose_landmarks      # 这里是归一化坐标

                # Save landmarks if pose was detected (it is normal if a frame has not detected pose)
                if pose_landmarks is not None:
                    frame_height, frame_width = frame.shape[0], frame.shape[1]
                    
                    # landmarks 维度：33*4
                    lmks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width, lmk.visibility] for lmk in pose_landmarks.landmark], dtype=np.float32)

                    assert lmks.shape == (33, 4), 'Unexpected landmarks shape: {}'.format(lmks.shape)

                    str_temp = [j] + lmks.flatten().astype(str).tolist()  # add frame index info
                    out_csv.append(str_temp)

                    # for visualization
                    # lmks2 = [(lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width, lmk.visibility) for lmk in pose_landmarks.landmark]
                    # vis_img = draw_blazepose_landmarks(frame, lmks2)
                    # cv2.imshow("BlazePose Visualization", vis_img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    
            # save landmarks to a file
            # with open(out_csv_path, 'w', newline='') as csv_file:
            #     csv_out_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            #     csv_out_writer.writerows(out_csv)
