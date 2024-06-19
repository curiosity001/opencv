import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtGui import QPainter, QMouseEvent
from PyQt5.QtCore import Qt

class CustomWidget(QWidget):
    def __init__(self):
        super().__init__()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            print("Left button clicked at:", event.pos())
        elif event.button() == Qt.RightButton:
            print("Right button clicked at:", event.pos())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Mouse Event Example")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = CustomWidget()
        self.setCentralWidget(self.central_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
