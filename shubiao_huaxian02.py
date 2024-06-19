import sys
import cv2
import os
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QComboBox
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

        self.region_label = QLabel("Region:")
        self.region_combo = QComboBox()
        self.region_combo.addItem("")  # Add an empty item to indicate no region selected
        self.region_combo.currentTextChanged.connect(self.select_region)

        self.finish_button = QPushButton("Finish")
        self.finish_button.clicked.connect(self.finish_capture)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_capture)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.camera_label)
        self.layout.addWidget(self.camera_ip)
        self.layout.addWidget(self.password_label)
        self.layout.addWidget(self.password_input)
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.region_label)
        self.layout.addWidget(self.region_combo)
        self.layout.addWidget(self.finish_button)
        self.layout.addWidget(self.cancel_button)

        self.region_name_labels = []
        self.region_name_inputs = []

        num_regions = 3  # Change this to the number of region input boxes you want

        for i in range(num_regions):
            label = QLabel(f"Region {i + 1} Name:")
            input_box = QLineEdit()
            self.region_name_labels.append(label)
            self.region_name_inputs.append(input_box)
            self.layout.addWidget(label)
            self.layout.addWidget(input_box)

        self.done_button = QPushButton("Done")
        self.done_button.clicked.connect(self.save_regions)
        self.layout.addWidget(self.done_button)

        self.central_widget = QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        self.current_region = ""
        self.regions = {}  # Dictionary to store regions and their points
        self.points = []
        self.capturing = False
        self.camera_thread = None

        self.image_label.installEventFilter(self)

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
        if self.current_region in self.regions:
            for point in self.regions[self.current_region]:
                cv2.circle(frame, point, 5, (0, 255, 0), -1)

            if len(self.regions[self.current_region]) >= 2:
                for i in range(len(self.regions[self.current_region]) - 1):
                    cv2.line(frame, self.regions[self.current_region][i], self.regions[self.current_region][i + 1],
                             (0, 0, 255), 2)
                if len(self.regions[self.current_region]) >= 3:
                    cv2.line(frame, self.regions[self.current_region][-1], self.regions[self.current_region][0],
                             (0, 0, 255), 2)

        q_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = q_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(q_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    def select_region(self, region_name):
        self.current_region = region_name
        self.points = []
        self.update_camera(self.camera_thread.cap.read()[1])  # Update the camera view

    def eventFilter(self, obj, event):
        if obj == self.image_label and event.type() == Qt.QEvent.MouseButtonPress:
            if self.capturing and self.current_region:
                x = event.pos().x()
                y = event.pos().y()
                self.points.append((x, y))
                self.update_camera(self.camera_thread.cap.read()[1])  # Update the camera view
            return True
        return False

    def finish_capture(self):
        if self.current_region:
            self.regions[self.current_region] = self.points
            self.points = []
            self.update_regions_combo()
            self.current_region = ""
            self.update_camera(self.camera_thread.cap.read()[1])  # Update the camera view

    def cancel_capture(self):
        if self.current_region:
            self.points = []
            self.current_region = ""
            self.update_camera(self.camera_thread.cap.read()[1])  # Update the camera view

    def update_regions_combo(self):
        self.region_combo.clear()
        self.region_combo.addItem("")  # Add an empty item to indicate no region selected
        self.region_combo.addItems(self.regions.keys())

    def save_regions(self):
        camera_ip = self.camera_ip.text()
        password = self.password_input.text()
        current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"{camera_ip}_{password}_{current_date}"

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        for i, input_box in enumerate(self.region_name_inputs):
            region_name = input_box.text()
            if region_name:
                points = self.regions.get(region_name, [])
                points_filename = os.path.join(folder_name, f"{region_name}_points.txt")
                with open(points_filename, "w") as f:
                    for point in points:
                        f.write(f"{point[0]},{point[1]}\n")

        self.reset_regions()

    def reset_regions(self):
        self.regions = {}
        self.points = []
        for input_box in self.region_name_inputs:
            input_box.clear()
        self.update_regions_combo()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
