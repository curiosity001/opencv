import sys
import cv2
import os
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap ,QMouseEvent
from PyQt5.QtCore import Qt, QThread, pyqtSignal

points123=[]

class CameraThread(QThread):
    frame_signal = pyqtSignal(object)

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.cap = cv2.VideoCapture(self.url)
        self.cap.set(cv2.CAP_PROP_FPS, 30)


    def run(self):
        while True:
            global points123
            ret, frame = self.cap.read()
            for point in points123:
                cv2.circle(frame, point, 5, (0, 255, 0), -1)

            if len(points123) >= 2:
                for i in range(len(points123) - 1):
                    cv2.line(frame, points123[i], points123[i + 1], (0, 0, 255), 2)
                if len(points123) >= 3:
                    cv2.line(frame, points123[-1], points123[0], (0, 0, 255), 2)
            if ret:
                self.frame_signal.emit(frame)

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("摄像头区域绘制界面")
        self.setGeometry(100, 100, 800, 600)

        self.camera_label = QLabel("摄像头 IP:")
        self.camera_ip = QLineEdit("192.168.31.59")
        self.password_label = QLabel("登录密码:")
        self.password_input = QLineEdit("Aust12345")

        self.start_button = QPushButton("打开摄像头")
        self.start_button.clicked.connect(self.start_camera)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        self.image_label.mousePressEvent = self.mousePressEvent

        self.variable_name_label = QLabel("区域命名:")
        self.variable_name_input = QLineEdit("J1_m01")  # Input for variable name
        self.save_variable_button = QPushButton("保存区域名")
        self.save_variable_button.clicked.connect(self.area_names)


        self.finish_button = QPushButton("保存区域坐标")
        self.finish_button.clicked.connect(self.finish_capture)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.camera_label)
        self.layout.addWidget(self.camera_ip)
        self.layout.addWidget(self.password_label)
        self.layout.addWidget(self.password_input)
        self.layout.addWidget(self.start_button)

        self.layout.addWidget(self.variable_name_label)
        self.layout.addWidget(self.variable_name_input)
        self.layout.addWidget(self.save_variable_button)

        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.finish_button)


        self.central_widget = QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        self.box_name = ""
        self.capturing = True

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


    def mousePressEvent(self, event: QMouseEvent):
        if event and event.button() == Qt.LeftButton:
            self.current_coordinate = event.pos()
            self.process_coordinate(self.current_coordinate)

    def process_coordinate(self, coordinate):
        global points123
        x, y = coordinate.x(), coordinate.y()
        points123.append((x, y))

        print("Processed coordinate:", (x, y))
        print(points123)

    def finish_capture(self):
        self.capturing = False
        box_filename = self.save_points_to_file()
        self.box_name = ""


    def area_names(self):
        self.area_names = self.variable_name_input.text()

    def save_points_to_file(self):
        global points123
        if not os.path.exists("captured_points"):
            os.makedirs("captured_points")

        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d")
        box_filename = f"captured_points/{self.camera_ip.text().replace('.', '_')}_{current_time}.txt"

        variable_name = self.area_names
        coordinates = points123

        existing_data = {}  # To store existing data from the file (if any)

        if os.path.exists(box_filename):
            with open(box_filename, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        var_name, coords_str = line.split("=")
                        existing_data[var_name.strip()] = eval(coords_str.strip())  # Read existing data

        existing_data[variable_name] = coordinates  # Update or add the new data

        with open(box_filename, "w") as f:
            for var_name, coords in existing_data.items():
                f.write(f"{var_name} = {coords}\n")

        points123 = []
        return box_filename

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
