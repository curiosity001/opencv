import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtCore import Qt

class CustomWidget(QWidget):
    def __init__(self):
        super().__init__()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            coordinate = event.pos()
            print("Left button clicked at:", coordinate)
            self.process_coordinate(coordinate)

    def process_coordinate(self, coordinate):
        x, y = coordinate.x(), coordinate.y()
        print("Processed coordinate:", (x, y))

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
