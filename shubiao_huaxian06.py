# OpenCVPyqt08.py
# Demo07 of GUI by PyQt5
# Copyright 2023 Youcans, XUPT
# Crated：2023-02-12

import sys
import cv2 as cv
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QPoint, QRect, qDebug, Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from uiDemo8 import Ui_MainWindow  # 导入 uiDemo8.py 中的 Ui_MainWindow 界面类


class MyLabel(QLabel):
    def __init__(self,parent=None):
        super(MyLabel, self).__init__(parent)
        self.x0 = 0
        self.y0 = 0
        self.x1 = 1
        self.y1 = 1
        self.flag = False

    # 鼠标点击事件
    def mousePressEvent(self, event):
        self.flag = True  # 鼠标点击状态
        self.x0 = event.x()
        self.y0 = event.y()

    # 鼠标释放事件
    def mouseReleaseEvent(self, event):
        self.flag = False  # 鼠标释放状态
        self.x1 = event.x()
        self.y1 = event.y()

    # 鼠标移动事件
    def mouseMoveEvent(self, event):
        if self.flag:
            self.x1 = event.x()
            self.y1 = event.y()
            self.update()

    # 绘制事件
    def paintEvent(self, event):
        super().paintEvent(event)
        rect = QRect(self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0))
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        painter.drawRect(rect)


class MyMainWindow(QMainWindow, Ui_MainWindow):  # 继承 QMainWindow 类和 Ui_MainWindow 界面类
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)  # 初始化父类
        self.setupUi(self)  # 继承 Ui_MainWindow 界面类
        self.label_1 = MyLabel(self.centralwidget)
        self.label_1.setGeometry(QRect(20, 20, 400, 320))
        self.label_1.setAlignment(Qt.AlignCenter)
        self.label_1.setObjectName("label_1")

        # 菜单栏
        self.actionOpen.triggered.connect(self.openSlot)  # 连接并执行 openSlot 子程序
        self.actionSave.triggered.connect(self.saveSlot)  # 连接并执行 saveSlot 子程序
        self.actionHelp.triggered.connect(self.trigger_actHelp)  # 连接并执行 trigger_actHelp 子程序
        self.actionQuit.triggered.connect(self.close)  # 连接并执行 trigger_actHelp 子程序

        # 通过 connect 建立信号/槽连接，点击按钮事件发射 triggered 信号，执行相应的子程序 click_pushButton
        self.pushButton_1.clicked.connect(self.click_pushButton_1)  # 按钮触发：导入图像
        self.pushButton_2.clicked.connect(self.click_pushButton_2)  # # 按钮触发：灰度显示
        self.pushButton_3.clicked.connect(self.click_pushButton_3)  # # 按钮触发：框选图像
        self.pushButton_4.clicked.connect(self.trigger_actHelp)  # # 按钮触发：调整色阶
        self.pushButton_5.clicked.connect(self.close)  # 点击 # 按钮触发：关闭

        # 初始化
        self.img1 = np.ndarray(())  # 初始化图像 ndarry，用于存储图像
        self.img2 = np.ndarray(())  # 初始化图像 ndarry，用于存储图像
        self.img1 = cv.imread("../images/Lena.tif")  # OpenCV 读取图像
        self.refreshShow(self.img1, self.label_1)
        # self.refreshShow(self.img1, self.label_2)
        return

    def click_pushButton_1(self):  # 点击 pushButton_1 触发
        self.img1 = self.openSlot()  # 读取图像
        self.img2 = self.img1.copy()
        print("click_pushButton_1", self.img1.shape)
        self.refreshShow(self.img1, self.label_1)  # 刷新显示
        return

    def click_pushButton_2(self):  # 点击 pushButton_2 触发
        print("pushButton_2")
        self.img2 = cv.cvtColor(self.img2, cv.COLOR_BGR2GRAY)  # 图片格式转换：BGR -> Gray
        self.refreshShow(self.img2, self.label_2)  # 刷新显示
        return

    def click_pushButton_3(self):  # 点击 pushButton_3 触发 框选图像
        print("pushButton_3")
        self.label_1.setGeometry(QRect(20, 20, 400, 320))
        hImg, wImg = self.img1.shape[:2]
        wLabel = self.label_1.width()
        hLabel = self.label_1.height()
        x0 = self.label_1.x0 * wImg//wLabel
        y0 = self.label_1.y0 * hImg//hLabel
        x1 = self.label_1.x1 * wImg//wLabel
        y1 = self.label_1.y1 * hImg//hLabel
        print("hImg,wImg=({},{}), x1,y1=({},{})".format(hImg, wImg, hLabel, wLabel))
        print("x0,y0=({},{}), x1,y1=({},{})".format(x0, y0, x1, y1))
        self.img2 = np.zeros((self.img1.shape), np.uint8)
        self.img2[y0:y1, x0:x1, :] = self.img1[y0:y1, x0:x1, :]
        print(self.img2.shape)
        # cv.imshow("Demo", self.img2)
        # key = cv.waitKey(0)  # 等待下一个按键命令

        # self.gRightLayout.removeWidget(self.label_2)  # 删除原有 labelCover 控件及显示图表
        # sip.delete(self.labelCover)  # 删除控件 labelCover
        # self.img2 = np.zeros(self.img1.shape, np.int8)
        self.refreshShow(self.img2, self.label_2)  # 刷新显示
        return

    def refreshShow(self, img, label):
        print(img.shape, label)
        qImg = self.cvToQImage(img)  # OpenCV 转为 PyQt 图像格式
        # label.setScaledContents(False)  # 需要在图片显示之前进行设置
        label.setPixmap((QPixmap.fromImage(qImg)))  # 加载 PyQt 图像
        return

    def openSlot(self, flag=1):  # 读取图像文件
        # OpenCV 读取图像文件
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", "../images/", "*.png *.jpg *.tif")
        if flag==0 or flag=="gray":
            img = cv.imread(fileName, cv.IMREAD_GRAYSCALE)  # 读取灰度图像
        else:
            img = cv.imread(fileName, cv.IMREAD_COLOR)  # 读取彩色图像
        print(fileName, img.shape)
        return img

    def saveSlot(self):  # 保存图像文件
        # 选择存储文件 dialog
        fileName, tmp = QFileDialog.getSaveFileName(self, "Save Image", "../images/", '*.png; *.jpg; *.tif')
        if self.img1.size == 1:
            return
        # OpenCV 写入图像文件
        ret = cv.imwrite(fileName, self.img1)
        if ret:
            print(fileName, self.img.shape)
        return

    def cvToQImage(self, image):
        # 8-bits unsigned, NO. OF CHANNELS=1
        if image.dtype == np.uint8:
            channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 3:  # CV_8UC3
            # Create QImage with same dimensions as input Mat
            qImg = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
            return qImg.rgbSwapped()
        elif channels == 1:
            # Create QImage with same dimensions as input Mat
            qImg = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Indexed8)
            return qImg
        else:
            qDebug("ERROR: numpy.ndarray could not be converted to QImage. Channels = %d" % image.shape[2])
            return QImage()

    def qPixmapToCV(self, qPixmap):  # PyQt图像 转换为 OpenCV图像
        qImg = qPixmap.toImage()  # QPixmap 转换为 QImage
        shape = (qImg.height(), qImg.bytesPerLine() * 8 // qImg.depth())
        shape += (4,)
        ptr = qImg.bits()
        ptr.setsize(qImg.byteCount())
        image = np.array(ptr, dtype=np.uint8).reshape(shape)  # 定义 OpenCV 图像
        image = image[..., :3]
        return image

    def trigger_actHelp(self):  # 动作 actHelp 触发
        QMessageBox.about(self, "About",
                          """数字图像处理工具箱 v1.0\nCopyright YouCans, XUPT 2023""")
        return

if __name__ == '__main__':
    app = QApplication(sys.argv)  # 在 QApplication 方法中使用，创建应用程序对象
    myWin = MyMainWindow()  # 实例化 MyMainWindow 类，创建主窗口
    myWin.show()  # 在桌面显示控件 myWin
    sys.exit(app.exec_())  # 结束进程，退出程序


