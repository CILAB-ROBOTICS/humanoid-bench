import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QSlider,
    QVBoxLayout, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer


class TactileRGBViewer(QWidget):
    def __init__(self, tactile_data: dict, rgb_frames: np.ndarray, fps=2):
        super().__init__()
        self.tactile_data = tactile_data
        self.rgb_frames = rgb_frames
        self.fps = fps
        self.frame_idx = 0
        self.total_frames = rgb_frames.shape[0]

        self.rgb_height, self.rgb_width = rgb_frames.shape[1:3]

        # 해상도 기반 비율 설정
        self.hand_canvas_height = 250
        self.hand_canvas_width = int(self.rgb_width / self.rgb_height * self.hand_canvas_height)

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        self.init_ui()
        self.update_frame()

    def init_ui(self):
        self.setWindowTitle("Tactile + RGB Viewer")
        main_layout = QVBoxLayout()

        # RGB 화면
        self.rgb_label = QLabel()
        self.rgb_label.setFixedSize(424, 240)
        main_layout.addWidget(self.rgb_label)

        # 손 히트맵
        hand_layout = QHBoxLayout()
        self.left_hand_label = QLabel()
        self.right_hand_label = QLabel()
        self.left_hand_label.setFixedSize(self.hand_canvas_width, self.hand_canvas_height)
        self.right_hand_label.setFixedSize(self.hand_canvas_width, self.hand_canvas_height)
        hand_layout.addWidget(self.left_hand_label)
        hand_layout.addWidget(self.right_hand_label)
        main_layout.addLayout(hand_layout)

        # 컨트롤 바
        control_layout = QHBoxLayout()
        self.play_button = QPushButton("▶️")
        self.play_button.clicked.connect(self.toggle_play)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.total_frames - 1)
        self.slider.valueChanged.connect(self.slider_changed)

        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.slider)
        main_layout.addLayout(control_layout)

        self.setLayout(main_layout)
        self.resize(800, 850)

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("▶️")
        else:
            self.timer.start(1000 // self.fps)
            self.play_button.setText("⏸️")

    def next_frame(self):
        self.frame_idx = (self.frame_idx + 1) % self.total_frames
        self.slider.setValue(self.frame_idx)
        self.update_frame()

    def slider_changed(self, value):
        self.frame_idx = value
        self.update_frame()

    def update_frame(self):
        # RGB 업데이트
        rgb_img = self.rgb_frames[self.frame_idx]
        rgb_resized = cv2.resize(rgb_img, (424, 240))
        rgb_qimg = QImage(rgb_resized.data, rgb_resized.shape[1], rgb_resized.shape[0],
                          3 * rgb_resized.shape[1], QImage.Format_RGB888).rgbSwapped()
        self.rgb_label.setPixmap(QPixmap.fromImage(rgb_qimg))

        # 손 히트맵 업데이트
        left_img = self.draw_hand_heatmap('left')
        right_img = self.draw_hand_heatmap('right')
        self.left_hand_label.setPixmap(QPixmap.fromImage(left_img))
        self.right_hand_label.setPixmap(QPixmap.fromImage(right_img))

        self.setWindowTitle(f"Frame {self.frame_idx + 1}/{self.total_frames}")

    def draw_hand_heatmap(self, side):
        canvas = np.ones((self.hand_canvas_height, self.hand_canvas_width, 3), dtype=np.uint8) * 255
        layout = self.get_hand_layout(side)

        for part, (x, y) in layout.items():
            key = f"{side}_{part}"
            if key in self.tactile_data:
                data = self.tactile_data[key][self.frame_idx]
                norm = np.clip((data - np.min(data)) / (np.ptp(data) + 1e-6), 0, 1) * 255
                norm = norm.astype(np.uint8)
                heat = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
                heat = cv2.resize(heat, (80, 80))
                if y + 80 <= canvas.shape[0] and x + 80 <= canvas.shape[1]:
                    canvas[y:y+80, x:x+80] = heat

        h, w, ch = canvas.shape
        bytes_per_line = ch * w
        return QImage(canvas.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

    def get_hand_layout(self, side):
        return {
            "thumb_tip": (20, 185), "thumb_nail": (20, 145), "thumb_middle_section": (20, 105), "thumb_pad": (20, 65),
            "index_finger_tip": (80, 20), "index_finger_nail": (80, 60), "index_finger_pad": (80, 100),
            "middle_finger_tip": (130, 15), "middle_finger_nail": (130, 55), "middle_finger_pad": (130, 95),
            "ring_finger_tip": (180, 20), "ring_finger_nail": (180, 60), "ring_finger_pad": (180, 100),
            "little_finger_tip": (230, 25), "little_finger_nail": (230, 65), "little_finger_pad": (230, 105),
            "palm": (120, 160)
        }


def load_data(npz_path: str):
    data = np.load(npz_path)
    tactile_dict = {}
    rgb = None

    for k in data:
        if k.startswith("tactile."):
            key = k.replace("tactile.", "")
            tactile_dict[key] = data[k]
        elif k == "rgb":
            rgb = data[k]

    return tactile_dict, rgb


def main():
    app = QApplication(sys.argv)
    tactile_dict, rgb_frames = load_data("output/episode_0033.npz")
    viewer = TactileRGBViewer(tactile_dict, rgb_frames)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()