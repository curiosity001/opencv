import sys
import cv2
import os
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class CameraThread(QThread):
    frame_signal = pyqtSignal(object)

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.cap = cv2.VideoCapture(self.url)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.frame_signal.emit(frame)

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Camera App")
        self.setGeometry(100, 100, 800, 600)

        self.camera_label = QLabel("Camera IP:")
        self.camera_ip = QLineEdit("192.168.31.59")
        self.password_label = QLabel("Password:")
        self.password_input = QLineEdit("Aust12345")

        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.start_camera)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.finish_button = QPushButton("Finish")
        self.finish_button.clicked.connect(self.finish_capture)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.camera_label)
        self.layout.addWidget(self.camera_ip)
        self.layout.addWidget(self.password_label)
        self.layout.addWidget(self.password_input)
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.finish_button)

        self.central_widget = QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        self.points = []
        self.box_name = ""
        self.capturing = False

        self.camera_thread = None

    def start_camera(self):
        ip = self.camera_ip.text()
        password = self.password_input.text()

        if ip:
            url = f"rtsp://admin:{password}@{ip}:554/live"
            self.camera_thread = CameraThread(url)
            self.camera_thread.frame_signal.connect(self.update_camera)
            self.camera_thread.start()

    def update_camera(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

        if self.capturing:
            self.draw_points(frame)

    def draw_points(self, frame):
        for point in self.points:
            cv2.circle(frame, point, 5, (0, 255, 0), -1)

        if len(self.points) >= 2:
            for i in range(len(self.points) - 1):
                cv2.line(frame, self.points[i], self.points[i + 1], (0, 0, 255), 2)
            if len(self.points) >= 3:
                cv2.line(frame, self.points[-1], self.points[0], (0, 0, 255), 2)

        if self.box_name:
            cv2.putText(frame, self.box_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        q_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = q_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(q_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if self.capturing:
            x = event.x()
            y = event.y()
            self.points.append((x, y))
            self.update_camera()

    def finish_capture(self):
        self.capturing = False
        box_filename = self.save_points_to_file()
        self.box_name = ""

    def save_points_to_file(self):
        if not os.path.exists("captured_points"):
            os.makedirs("captured_points")

        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        box_filename = f"captured_points/{self.camera_ip.text().replace('.', '_')}_{current_time}.txt"

        with open(box_filename, "w") as f:
            for point in self.points:
                f.write(f"{point[0]},{point[1]}\n")

        self.points = []
        return box_filename

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
