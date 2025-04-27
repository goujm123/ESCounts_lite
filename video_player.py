import sys
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QSlider, QLabel, QCheckBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QSettings

REPCOUNT_VIDEO_PATH = "d:/datasets/RepCount/video"


def load_annotation(split='train'):
    csv_path = f"d:/datasets/RepCount/annotation/{split}.csv"
    indices = []
    types = []
    names = []
    counts = []
    starts = []
    ends = []

    with open(csv_path, 'r') as f:
        lines = f.readlines()
        header = lines[0].strip().split(',')
        lines = lines[1:]  # remove header from lines
        for line in lines:
            values = line.strip().split(',')
            indices.append(int(values[0]))
            types.append(values[1])
            names.append(values[2])
            counts.append(values[3] if values[3].isdigit() else 0)
            # 只保留可以转换为整数的项
            starts.append([int(x) for x in values[4::2] if x.isdigit()])
            ends.append([int(x) for x in values[5::2] if x.isdigit()])

    # 标注的动作类别名称比较混乱
    print(f'{len(set(types))} action types in [{split}]:\n {set(types)}"\n\n')

    # 规范化动作名称
    normalized_types = []
    type_mapping = {
        'squant': 'squat',
        'benchpressing': 'bench_pressing',
        'jumpjacks': 'jump_jack',
        'pushups': 'push_up',
        'pullups': 'pull_up',
        'frontraise': 'front_raise'
    }
    for t in types:
        normalized_types.append(type_mapping.get(t, t))

    print(f'{len(set(normalized_types))} normalized action types in [{split}]:\n {set(normalized_types)}"\n\n')

    return {'split': split, 'index': indices, 'type': types, 'name': names, 'count': counts, 'start': starts, 'end': ends}


class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RepCount播放器")
        self.settings = QSettings("RgbPose", "VideoPlayer")
        # 恢复窗口位置和大小
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        else:
            self.setFixedSize(1920, 1080)
            self.move(100, 100)

        # 视频相关变量
        self.cap = None
        self.frames = []
        self.current_frame_idx = 0
        self.total_frames = 0
        self.fps = 30
        self.reverse_mode = False
        self.last_opened_path = ""
        self.settings = QSettings("RgbPose", "VideoPlayer")
        self.last_opened_path = self.settings.value("last_path", "", type=str)

        self.anno_train = load_annotation(split='train')
        self.anno_valid = load_annotation(split='valid')
        self.anno_test = load_annotation(split='test')

        # 播放控制变量
        self.playing = False
        self.play_speed = 1.0  # 1.0为正常速度

        # 初始化UI
        self.init_ui()

    def init_ui(self):
        # 主窗口布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # 视频显示区域
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            """
            QLabel {
                background-color: #f0f0f0;
                border: 2px solid #7f7f7f;
                border-radius: 2px;
                padding: 5px;
                color: #333;
                font-weight: bold;
            }
            """
        )
        layout.addWidget(self.video_label, 1)

        # Meta信息显示区域
        meta_container = QWidget()
        meta_layout = QHBoxLayout(meta_container)
        meta_layout.setContentsMargins(0, 0, 0, 0)

        # meta_label大小设置
        self.meta_label = QLabel()
        self.meta_label.setMinimumWidth(1800)
        self.meta_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.meta_label.setWordWrap(True)
        self.meta_label.setStyleSheet(
            """
            QLabel {
                background-color: #f0f0f0;
                border: 2px solid #3498db;
                border-radius: 2px;
                padding: 5px;
                color: #333;
                font-weight: bold;
                qproperty-alignment: 'AlignLeft | AlignVCenter';
            }
            """
        )
        meta_layout.addWidget(self.meta_label)

        layout.addWidget(meta_container)

        # 控制区域
        control_layout = QHBoxLayout()

        # 打开文件按钮
        self.open_btn = QPushButton("打开文件")
        self.open_btn.clicked.connect(self.open_file_dialog)
        control_layout.addWidget(self.open_btn)

        # 播放/暂停按钮
        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self.toggle_play)
        control_layout.addWidget(self.play_btn)

        # 上一帧按钮
        self.prev_btn = QPushButton("上一帧")
        self.prev_btn.clicked.connect(self.prev_frame)
        control_layout.addWidget(self.prev_btn)

        # 下一帧按钮
        self.next_btn = QPushButton("下一帧")
        self.next_btn.clicked.connect(self.next_frame)
        control_layout.addWidget(self.next_btn)

        # 速度控制
        self.speed_label = QLabel("速度: 1.0x")
        control_layout.addWidget(self.speed_label)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 4)  # 1=0.25x, 2=0.5x, 3=1x, 4=2x
        self.speed_slider.setValue(3)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(1)
        self.speed_slider.valueChanged.connect(self.change_speed)
        control_layout.addWidget(self.speed_slider)

        # 倒序播放复选框
        self.reverse_check = QCheckBox("倒序播放")
        self.reverse_check.stateChanged.connect(self.toggle_reverse)
        control_layout.addWidget(self.reverse_check)

        layout.addLayout(control_layout)

        # 进度条
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.sliderMoved.connect(self.set_frame_position)
        layout.addWidget(self.progress_slider)

        central_widget.setLayout(layout)

        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def load_video(self, file_path):
        self.last_opened_path = file_path
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            print("无法打开视频文件")
            return False

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 读取所有帧到内存，这个很耗内存。如有可能，替换为分段加载
        self.frames = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frames.append(frame)

        self.cap.release()

        # 如果是RepCount数据集的视频，显示action相关的信息
        self.update_meta_info("")
        if file_path.lower().startswith(REPCOUNT_VIDEO_PATH.lower()):
            split = file_path.split('/')[-2]
            anno_split = getattr(self, f'anno_{split}')
            select = anno_split['name'].index(os.path.basename(file_path))
            self.action_count = anno_split['count'][select]
            self.action_type = anno_split['type'][select]
            action = ""
            for i, start in enumerate(anno_split['start'][select]):
                end = anno_split['end'][select][i]
                action += f"{start}-{end}, "
            self.update_meta_info(f"【{self.action_type}, {self.action_count}】, actions: {action}")

        self.progress_slider.setRange(0, len(self.frames) - 1)
        self.show_frame(self.frames[0])
        self.current_frame_idx = 0
        self.progress_slider.setRange(0, len(self.frames) - 1)
        self.show_frame(self.frames[0])
        self.setWindowTitle(f"RepCount播放器 - {os.path.basename(file_path)}")
        return True

    def update_meta_info(self, text):
        self.meta_label.setText(text)
        self.meta_label.setToolTip(text)

    def show_frame(self, frame):
        # 在帧上添加帧序号
        frame_copy = frame.copy()

        # 转换为Pillow格式
        pil_img = Image.fromarray(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # 加载系统字体或默认字体
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()

        # 绘制文本
        draw.text((10, 10), f"Frame: {self.current_frame_idx}", font=font, fill=(255, 255, 255))

        # 转换回OpenCV格式
        frame_copy = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 调整帧大小以适应窗口
        h, w, _ = frame_copy.shape
        target_w, target_h = 1920, 1080

        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        if scale > 1:  # 不放大
            scale = 1

        # 计算新尺寸
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 缩放
        resized = cv2.resize(frame_copy, (new_w, new_h))

        # 转换为Qt格式
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(target_w, target_h, Qt.KeepAspectRatio)

        # 显示
        self.video_label.setPixmap(QPixmap.fromImage(p))

    def update_frame(self):
        if not self.playing:
            return

        if len(self.frames) == 0:
            return

        if self.reverse_mode:
            self.current_frame_idx -= 1
            if self.current_frame_idx < 0:
                self.current_frame_idx = len(self.frames) - 1
        else:
            self.current_frame_idx += 1
            if self.current_frame_idx >= len(self.frames):
                self.current_frame_idx = 0

        self.progress_slider.setValue(self.current_frame_idx)
        self.show_frame(self.frames[self.current_frame_idx])

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("暂停" if self.playing else "播放")

        if self.playing:
            interval = int(1000 / (self.fps * self.play_speed))
            self.timer.start(interval)
        else:
            self.timer.stop()

    def prev_frame(self):
        self.current_frame_idx -= 1
        if self.current_frame_idx < 0:
            self.current_frame_idx = len(self.frames) - 1
        self.progress_slider.setValue(self.current_frame_idx)
        self.show_frame(self.frames[self.current_frame_idx])

    def next_frame(self):
        self.current_frame_idx += 1
        if self.current_frame_idx >= len(self.frames):
            self.current_frame_idx = 0
        self.progress_slider.setValue(self.current_frame_idx)
        self.show_frame(self.frames[self.current_frame_idx])

    def change_speed(self, value):
        speeds = {1: 0.25, 2: 0.5, 3: 1.0, 4: 2.0}
        self.play_speed = speeds[value]
        self.speed_label.setText(f"速度: {self.play_speed}x")

        if self.playing:
            self.timer.stop()
            interval = int(1000 / (self.fps * self.play_speed))
            self.timer.start(interval)

    def toggle_reverse(self, state):
        self.reverse_mode = state == Qt.Checked

    def set_frame_position(self, position):
        self.current_frame_idx = position
        self.show_frame(self.frames[self.current_frame_idx])

    def closeEvent(self, event):
        if self.cap and self.cap.isOpened():
            self.cap.release()

        # 保存当前路径
        if self.last_opened_path:
            self.settings.setValue("last_path", self.last_opened_path)

        # 只保存窗口位置
        self.settings.setValue("geometry", self.saveGeometry())

        event.accept()

    def open_file_dialog(self):
        from PyQt5.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", self.last_opened_path, "视频文件 (*.mp4 *.avi *.mov);;所有文件 (*.*)")
        if file_path:
            self.load_video(file_path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()

    if len(sys.argv) > 1:
        player.load_video(sys.argv[1])

    player.show()
    sys.exit(app.exec_())
