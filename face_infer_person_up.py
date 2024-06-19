import os
import sys
import serial
import numpy
from matplotlib.path import Path
import cv2
from numpy import random
import csv
import datetime
import time
import struct
import socket
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import numpy as np
import torch
import threading
from copy import deepcopy
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import os
import pygame
import cv2 as cv
import modbus_tk.modbus_tcp as mt
import modbus_tk.defines as md
metalocA = threading.Lock()

master=mt.TcpMaster("192.168.31.39",999) #向某个地址发送数据
#设置响应等待时间
master.set_timeout(5.0)

pygame.mixer.init()#初始化

device = select_device('0')
half = device.type != 'cpu'  # half precision only supported on CUDA

weights1='person2.pt'
#加载Float32模型，确保用户设定的输入图片分辨率能整除32(如不能则调整为能整除并返回)
model1 = attempt_load(weights1, map_location=device)  # 人员检测模型加载
stride1 = int(model1.stride.max())  # model stride
imgsz1 = check_img_size(640, s=stride1)  # check img_size
#获取类别名字
names1 = model1.module.names if hasattr(model1, 'module') else model1.names
colors1 = [[random.randint(0, 255) for _ in range(3)] for _ in names1]


weights='face_detection.pt'
# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# 设置Float16
if half: #半精度模型
    model.half()
    model1.half()
# cap=cv2.VideoCapture('3.mp4')
# url1='rtsp://admin:Aust12345@192.168.31.57:554/live'
# url1 = 0
# url2='rtsp://admin:Aust12345@192.168.31.59:554/live'
# url4='22/9.mp4'
# url5='22/4.mp4'
# url3='rtsp://admin:Aust12345@192.168.31.60:554/live'
# url6='22/6.mp4'
# url4='rtsp://admin:Aust12345@192.168.31.61:554/live'
# url5='rtsp://admin:Aust12345@192.168.31.62:554/live'
# url6='rtsp://admin:Aust12345@192.168.31.63:554/live'
url2='3.mp4'
# url1='1.jpg'
# url2='3.mp4'
# url2='22/2.mp4'
# url3='22/1.mp4'
video_weight=1920
video_hight=1080

#输出信号设置
sign=[0,0,0]

sign_sleep=[0,0]

sign4 = 0
sign5 = 0
sign6 = 0
sign7 = 0
sign8 = 0

Time11 = 0
Time12 = 0
Time13 = 0
Time14 = 0
Time15 = 0
Time16 = 0
Time17 = 0
Time18 = 0
Time19 = 0
Time20 = 0

it6 = 0
it7 = 0
it8 = 0
it9 = 0
it10 = 0

#变量设置
coust=0

real_time_start4 = [0,0]
real_time_start7 = [0,0]
real_time_start8 = [0,0]

#电子围栏设置
m4 = Path([(524, 101), (524 ,912),(1445, 911),(1445,100)])
# 4上井口
m7 = Path([(730 ,6), (730, 268),(1134, 266),(1132 ,6)])
n7 = Path([(680 , 327), (159 , 1058),(1441 , 1061),(1167 , 326)])
# 7.1-8.1双罐进车上井口
m8 = Path([(876 ,5), (876 ,351),(1477, 350),(1477, 6)])
n8 = Path([(777 , 418), (357 , 1060),(1719 , 1061),(1545 , 455)])
# # 7.2-8.2双罐出车上井口

def cv2AddChineseText(img, text, position, textColor, textSize):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) #简而言之，就是实现array到image的转换
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

"""
cv2.cvtColor(src, code[, dst[, dstCn]])
src:它是要更改其色彩空间的图像。
code:它是色彩空间转换代码。
dst:它是与src图像大小和深度相同的输出图像。它是一个可选参数。
dstCn:它是目标图像中的频道数。如果参数为0，则通道数自动从src和代码得出。它是一个可选参数。
"""

def image_inference4(frame,model1,device,half,imgsz,names,colors):
    t1 = time.time()
    global sign4, Time11, Time12,it6
    # 截取第一个区域
    frame = cv2.line(frame, (524, 101), (524 ,912), (0, 255, 0), 4)  #绘制直线
    frame = cv2.line(frame, (524 ,912), (1445, 911), (0, 255, 0), 4)
    frame = cv2.line(frame, (1445, 911), (1445,100), (0, 255, 0), 4)
    frame = cv2.line(frame, (1445,100), (524, 101), (0, 255, 0), 4)
    """
    主要有cv2.line()//画线， cv2.circle()//画圆， cv2.rectangle()//长方形，cv2.ellipse()//椭圆， cv2.putText()//文字绘制

    主要参数

    img：源图像
    color：需要传入的颜色
    thickness：线条的粗细，默认值是1
    linetype：线条的类型，8 连接，抗锯齿等。默认情况是 8 连接。cv2.LINE_AA 为抗锯齿，这样看起来会非常平滑。
    """

    img = letterbox(frame, imgsz, stride=32)[0] #letterbox操作：在对图片进行resize时，保持原图的长宽比进行等比例缩放，当长边 resize 到需要的长度时，短边剩下的部分采用灰色填充。
    print(img)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416  图片通道转换
    img = np.ascontiguousarray(img) #ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快（归一化处理）
    img = torch.from_numpy(img).to(device) # 就是torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。  这段代码的意思就是将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。
    img = img.half() if half else img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0 归一化
# 如果图片是3维(RGB) 就在前面添加一个维度1当中batch_size=1
# 因为输入网络的图片需要是4为的 [batch_size, channel, w, h]
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
#因为验证集的时候，我们只是想看一下训练的效果，并不是想通过验证集来更新网络时，就可以使用with torch.no_grad()
    with torch.no_grad():
        pred = model1(img, augment=False)[0] #augment表示是否进行数据增强
        """
        pred的shape是(1, num_boxes, 5+num_class)
        num_boxes为模型在３个特征图上预测出来的框的个数，num_boxes = h/32 * w/32 + h/16 * w/16 + h/8 * w/8
        5+nnum_class的值为：
        pred[…, 0:4]为预测框坐标,预测框坐标为xywh(中心点+宽长)格式
        pred[…, 4]为objectness置信度
        pred[…, 5:-1]为分类结果
        预测得到的这一堆框送入后面的NMS函数。
        ————————————————
        版权声明：本文为CSDN博主「下大禹了」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
        原文链接：https://blog.csdn.net/weixin_43541325/article/details/108884363
        """
        pred = non_max_suppression(pred, 0.45, 0.45, classes=None, agnostic=False)[0] #非极大值抑制
        path = os.path.abspath(os.path.dirname(sys.argv[0]))
        time1 = time.strftime('%Y%m%d', time.localtime(time.time()))

    if len(pred):
        global coust,t9,t10 #定义全局变量
        coust=1
        if len(pred) and coust == 1:
            # Rescale boxes from img_size to im0 size
            #im0是原图， img是经过letterbox()函数后的图
            # 将预测信息（相对img_size 640）映射回原图 img0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round() #pred[:, :4] 是从预测结果张量 pred 中提取每个检测框的前四个元素（即边界框坐标）。在这里，: 表示选取所有行，:4 表示选取前四列。这是一个使用 NumPy 切片操作提取部分张量数据的示例。

                                                                                        #假设 pred 张量的形状为 (N, M)，其中 N 是检测框的数量，M 是每个检测框的特征数量。每个检测框的前四个特征通常表示边界框的坐标，例如 (x1, y1, x2, y2)，分别表示边界框的左上角和右下角坐标。

                                                                                        #pred[:, :4] 将创建一个新的张量，包含每个检测框的前四个特征，即所有边界框的坐标。这在之后的计算中用于将这些坐标从缩放后的图像坐标空间映射回原始图像坐标空间。
            for *xyxy, conf, cls in reversed(pred): #reversed表示反向遍历
                label = f'{names[int(cls)]} {conf:.2f}' #字符串格式的方法
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)  #*xyxy 和 xyxy 的区别在于 Python 解包（unpacking）语法中使用的星号（*）。

                                                                                                #在 for *xyxy, conf, cls in reversed(pred): 这行代码中，*xyxy 使用了星号，这意味着我们希望将 pred 的前几个元素解包为单独的变量。在这种情况下，pred 的前四个元素（边界框坐标）被解包为一个名为 xyxy 的元组。而 conf 和 cls 变量分别表示置信度和类别。

                                                                                                #如果我们只使用 xyxy（不带星号），那么在遍历过程中，xyxy 变量将直接包含边界框坐标，而没有进行解包。这将导致在后续处理中可能需要额外的索引操作来访问这些坐标值。

                                                                                                #总之，*xyxy 允许我们在遍历 pred 时直接解包边界框坐标，使得后续处理更简洁。而不带星号的 xyxy 将保留原始数据结构（例如列表或元组），可能需要额外的索引操作。



                                                                                                #解包：这种语法就是在某个变量面前加一个星号，而且这个星号可以放在任意变量，每个变量都分配一个元素后，剩下的元素都分配给这个带星号的变量
                                                                                                  #  >> > a, *b, c = [1, 2, 3, 4]
                                                                                                  #  >> > a
                                                                                                  #  1
                                                                                                  #  >> > b
                                                                                                  #  [2, 3]
                                                                                                  #  >> > c
                                                                                                  #  4



            for *xyxy, conf, cls in reversed(pred):
                if (m4.contains_point(
                        (int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                    cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,lineType=cv2.LINE_AA)
                    sign4 = 1
                    Time11 = time.time()
                    if it6 == 0:
                        it6 =time.time()
                        real_time_start4[0] = datetime.now().strftime('%Y-%m-%d %H:%M:%S') #python datetime模块用strftime 格式化时间
                        if pygame.mixer.music.get_busy():
                            pass
                        else:
                            pygame.mixer.music.load('C:\yolov7\wav\sound.wav')
                            pygame.mixer.music.set_volume(0.5)
                            pygame.mixer.music.play()
                else:
                    if sign4 == 1:
                        Time12 = time.time()
                        if Time12 - Time11 > 5:
                            sign4 = 0
                            it6 = 0
                            real_time_start4[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            a = str(real_time_start4)
                            send('上井口' + a + 'person', 777)
    else:
        sign4 = 0
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    # print(6)
    cv2.putText(frame, 'FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    # print(7)
    return frame
# 4上井口
def image_inference7(frame,model1,device,half,imgsz,names,colors):
    t1 = time.time()
    global sign5,sign6, Time13, Time14, Time15, Time16,it7,it8

    frame = cv2.line(frame, (730 ,6), (730, 268), (0, 255, 0), 4)
    frame = cv2.line(frame, (730, 268), (1134, 266), (0, 255, 0), 4)
    frame = cv2.line(frame, (1134, 266), (1132 ,6), (0, 255, 0), 4)
    frame = cv2.line(frame, (1132 ,6), (730 ,6), (0, 255, 0), 4)

    frame = cv2.line(frame, (680 , 327), (159 , 1058), (0, 255, 0), 4)
    frame = cv2.line(frame, (159 , 1058), (1441 , 1061), (0, 255, 0), 4)
    frame = cv2.line(frame, (1441 , 1061), (1167 , 326), (0, 255, 0), 4)
    frame = cv2.line(frame, (1167 , 326), (680 , 327), (0, 255, 0), 4)
    img = letterbox(frame, imgsz, stride=32)[0]
    # img = cv2.resize(frame, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    with torch.no_grad():
        pred = model1(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.45, 0.45, classes=None, agnostic=False)[0]
        path = os.path.abspath(os.path.dirname(sys.argv[0]))
        time1 = time.strftime('%Y%m%d', time.localtime(time.time()))
    if len(pred):
        global coust, none6,t15,t16
        coust=1
        if len(pred) and coust == 1:
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(pred):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
            for *xyxy, conf, cls in reversed(pred):
                if (m7.contains_point(
                        (int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                    cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,lineType=cv2.LINE_AA)
                    sign5 = 1
                    Time13 = time.time()
                    if it7 == 0:
                        it7 =time.time()
                        real_time_start7[0] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        if pygame.mixer.music.get_busy():
                            pass
                        else:
                            pygame.mixer.music.load('C:\yolov7\wav\sound.wav')
                            pygame.mixer.music.set_volume(0.5)
                            pygame.mixer.music.play()
                else:
                    if sign5 == 1:
                        Time14 = time.time()
                        if Time14 - Time13 > 10:
                            sign5 = 0
                            it7 = 0
                            real_time_start7[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            a = str(real_time_start7)
                            send('上井口双罐进车' + a + 'person', 777)
#--------------------------------------------------------------------------------------------------------------------
                if (n7.contains_point(
                        (int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                    cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,lineType=cv2.LINE_AA)
                    sign6 = 1
                    Time15 = time.time()
                    if it8 == 0:
                        it8 =time.time()
                        real_time_start8[0] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        if pygame.mixer.music.get_busy():
                            pass
                        else:
                            pygame.mixer.music.load('C:\yolov7\wav\sound.wav')
                            pygame.mixer.music.set_volume(0.5)
                            pygame.mixer.music.play()
                else:
                    if sign6 == 1:
                        Time16 = time.time()
                        if Time16 - Time15 > 10:
                            sign6 = 0
                            it8 = 0
                            real_time_start8[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            a = str(real_time_start8)
                            send('上井口双罐进车' + a + 'person', 777)
#----------------------------------------------------------------------------------------------------------------------
    else:
        sign5 = 0
        sign6 = 0
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, 'FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame
# 7.1-8.1双罐进车上井口
def image_inference8(frame,model1,device,half,imgsz,names,colors):
    t1 = time.time()
    global sign5,sign6,sign7,sign8,Time17, Time18, Time19, Time20,it9,it10
    frame = cv2.line(frame, (876 ,5), (876 ,351), (0, 255, 0), 4)
    frame = cv2.line(frame, (876 ,351), (1477, 350), (0, 255, 0), 4)
    frame = cv2.line(frame, (1477, 350), (1477, 6), (0, 255, 0), 4)
    frame = cv2.line(frame, (1477, 6), (876 ,5), (0, 255, 0), 4)

    frame = cv2.line(frame, (777 , 418), (357 , 1060), (0, 255, 0), 4)
    frame = cv2.line(frame, (357 , 1060), (1719 , 1061), (0, 255, 0), 4)
    frame = cv2.line(frame, (1719 , 1061), (1545 , 455), (0, 255, 0), 4)
    frame = cv2.line(frame, (1545 , 455), (777 , 418), (0, 255, 0), 4)

    img = letterbox(frame, imgsz, stride=32)[0]

    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    with torch.no_grad():
        pred = model1(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.45, 0.45, classes=None, agnostic=False)[0]

        path = os.path.abspath(os.path.dirname(sys.argv[0]))
        time1 = time.strftime('%Y%m%d', time.localtime(time.time()))

    if len(pred):
        global coust, none7,t17,t18
        coust=1
        if len(pred) and coust == 1:
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(pred):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
            for *xyxy, conf, cls in reversed(pred):
                if (m8.contains_point((int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                    cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,lineType=cv2.LINE_AA)
                    sign7 = 1
                    Time17 = time.time()
                    if it9 == 0:
                        it9 =time.time()
                        real_time_start7[0] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        if pygame.mixer.music.get_busy():
                            pass
                        else:
                            pygame.mixer.music.load('C:\yolov7\wav\sound.wav')
                            pygame.mixer.music.set_volume(0.5)
                            pygame.mixer.music.play()
                else:
                    if sign7 == 1:
                        Time18 = time.time()
                        if Time18 - Time17 > 10:
                            sign7 = 0
                            it9 = 0
                            real_time_start7[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            a = str(real_time_start7)
                            send('上井口双罐出车' + a + 'person', 777)
#-----------------------------------------------------------------------------------------------------------------------
                if (n8.contains_point((int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                    cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,lineType=cv2.LINE_AA)
                    sign8 = 1
                    Time19 = time.time()
                    if it10 == 0:
                        it10 = time.time()
                        real_time_start8[0] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        if pygame.mixer.music.get_busy():
                            pass
                        else:
                            pygame.mixer.music.load('C:\yolov7\wav\sound.wav')
                            pygame.mixer.music.set_volume(0.5)
                            pygame.mixer.music.play()
                else:
                    if sign8 == 1:
                        Time20 = time.time()
                        if Time20 - Time19 > 10:
                            sign8 = 0
                            it10 = 0
                            real_time_start8[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            a = str(real_time_start8)
                            send('上井口双罐出车' + a + 'person', 777)
#-----------------------------------------------------------------------------------------------------------------------
    else:
        sign7 = 0
        sign8 = 0
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, 'FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    # print(frame.shape)
    return frame
# 7.2-8.2双罐出车上井口
def image_inference(frame,model,device,imgsz,names,colors):
    t1 = time.time()
    face_status=[]
    img = letterbox(frame, imgsz, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    # img = img.half()
    # img=img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    with torch.no_grad():
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.25, classes=None, agnostic=False)[0]
    if len(pred):
        # Rescale boxes from img_size to im0 size
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
        # Write results
        for *xyxy, conf, cls in reversed(pred):
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
            face_status.append(int(cls))
            # print(face_status)
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, 'FPS=%s' % FPS, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame,face_status

def send(data, port=6666):
    client = socket.socket()
    # 连接服务器
    addr = ('192.168.31.13', port)
    client.connect(addr)
    # 发送数据
    client.send(data.encode('utf-8'))
    client.close()
"""
这段代码定义了一个名为send的函数，该函数可以向指定的服务器的指定端口发送数据。

在函数内部，代码首先使用socket.socket函数创建一个客户端 socket 对象。然后，将服务器的 IP 地址和端口号保存在addr变量中，调用connect方法将客户端 socket 连接到服务器的指定端口上。

接着，使用encode方法将字符串类型的数据转换为字节流，并使用send方法将其发送给服务器。最后，调用close方法关闭客户端 socket 对象。

需要注意的是，该函数使用了默认的字符编码utf-8，如果要发送的数据不是字符串类型，需要先将其转换为字节流再发送。另外，服务器的 IP 地址和端口号需要根据实际情况进行修改。
"""

class Carame_Accept_Object:
    def __init__(self):
        self.resolution = (1280,720)  # 分辨率
        self.img_fps = 15  # 每秒传输多少帧数

def check_option(object, client): #函数的主要功能是从客户端接收并解码二进制数据，确定帧数和分辨率，并将它们存储在object对象的属性img_fps和resolution中
    # 按格式解码，确定帧数和分辨率
    info = struct.unpack('lhh', client.recv(8))
    if info[0] > 888:
        object.img_fps = int(info[0]) - 888  # 获取帧数
        object.resolution = list(object.resolution)
        # 获取分辨率
        object.resolution[0] = info[1]
        object.resolution[1] = info[2]
        object.resolution = tuple(object.resolution)
        return 1
    else:
        return 0

def RT_Image(img, client):
    img_param = [int(cv2.IMWRITE_JPEG_QUALITY), object.img_fps]  # 设置传送图像格式、帧数
    time.sleep(0.2)  # 推迟线程运行0.2s
    # _, object.img = camera.read()  # 读取视频每一帧
    # 核心代码在这里，图像从这里传过来，
    # print(object.resolution)
    object.img = cv2.resize(img, object.resolution)  # 按要求调整图像大小(resolution必须为元组)
    _, img_encode = cv2.imencode('.jpg', object.img, img_param)  # 按格式生成图片
    img_code = numpy.array(img_encode)  # 转换成矩阵
    object.img_data = img_code.tobytes()  #
    # 按照相应的格式进行打包发送图片
    client.send(
        struct.pack("lhh", len(object.img_data), object.resolution[0], object.resolution[1]) + object.img_data)

def create_client(port=8880):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 端口可复用
    server.bind(("", port))
    server.listen(5)
    client, D_addr = server.accept()
    return client

#这段代码定义了一个名为create_client的函数，该函数可以创建一个服务器 socket，并将其绑定到指定端口上。然后，它开始监听客户端连接请求，并接受客户端连接请求，返回一个连接对象，用于与客户端进行通信。

#在函数内部，代码首先使用socket.socket函数创建一个基于 IPv4 的 TCP socket，并将其设置为可复用。然后，将服务器 socket 绑定到指定的端口上，这里使用空字符串表示在本地所有可用的 IP 地址上监听该端口。接着，调用listen方法开始监听客户端连接请求，参数5表示在等待队列中最多可排队5个连接请求。最后，调用accept方法接受客户端连接请求，返回一个连接对象和客户端地址，其中连接对象可以用于与客户端进行通信。

#最后，函数返回连接对象，用于与客户端进行通信。


object = Carame_Accept_Object()
# 这里的顺序要和receive里面一致
client1 = create_client(8880)
client2 = create_client(8881)
# client3 = create_client(8882)
# client4 = create_client(8883)
# client7 = create_client(8884)
# client8 = create_client(8885)
# port_list = [8880, 8881,8882,8883,8884,8885]
port_list = [8880, 8881]
# clients = [client1, client2,client3, client4, client7, client8]
clients = [client1, client2]
metalocA = threading.Lock()

class Kaiguanliang(threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name

    def run(self):
        global sign4,sign5,sign6,sign7,sign8,sign,sign_sleep
        while True:
            sign[0] = sign4
            if sign5==1 or sign7==1:
                sign[1]=1
            else:
                sign[1]=0
            if sign6==1 or sign8==1:
                sign[2]=1
            else:
                sign[2]=0
            print(sign)
            try: #尝试将 sign 和 sign_sleep 的值写入到指定的寄存器中，如果超时则打印错误信息
                master.execute(slave=1, function_code=md.WRITE_MULTIPLE_REGISTERS, starting_address=9,
                                     quantity_of_x=3, output_value=sign)
                master.execute(slave=1, function_code=md.WRITE_MULTIPLE_REGISTERS, starting_address=12,
                               quantity_of_x=2, output_value=sign_sleep)
            except:
                print("time out")
            time.sleep(2)

class Video_receive_thread(threading.Thread):
    def __init__(self, threadID, name,url,frame_name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.url=url
        self.frame_name=frame_name
        # float = np.float32()
        # self.frame = np.zeros(shape=(1080, 1920, 3), dtype=float)
        self.frame=None
        self.cap = cv2.VideoCapture(self.url)
    def run(self):
        # while (self.cap.isOpened()):
        while True:
            self.ret, self.frame = self.cap.read()
            if self.ret == False:
                self.cap = cv2.VideoCapture(self.url)
                continue
            # print(self.ret)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # self.cap.release()
        cv2.destroyAllWindows()

# thread_read1 = Video_receive_thread(1, "video_read1",url1,'frame1')
thread_read2 = Video_receive_thread(2, "video_read2",url2,'frame2')
# thread_read3 = Video_receive_thread(3, "video_read3",url3,'frame3')
#
# thread_read4 = Video_receive_thread(4, "video_read4",url4,'frame4')
# thread_read7 = Video_receive_thread(7, "video_read7",url5,'frame7')
# thread_read8 = Video_receive_thread(8, "video_read8",url6,'frame8')
thread_Kaiguan = Kaiguanliang(22,"USB_Kaiguan")
class Video_analyze_thread(threading.Thread):
    def __init__(self, threadID, name,thread_num,frame_name,img_infer,model,client=None):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        # self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
        self.frame_read=None
        self.thread_num=thread_num
        self.frame_name=frame_name
        self.img_infer = img_infer
        # self.model = deepcopy(model)
        self.model = model
        self.client = client

    def run(self):
        while True:
            # time.sleep(0.5)
            try:
                self.frame_read = self.img_infer(self.thread_num.frame, self.model, device, half, imgsz1, names1, colors1)#使用图像推断函数对视频帧进行推断
                RT_Image(self.frame_read, self.client)# 将推断后的图像显示到客户端上
            except:
                # self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
                self.frame_read = np.random.rand(1080,1920,3)*255
                print("Data Error",self.thread_num)
                self.client.close()
                index = clients.index(self.client)
                self.client = create_client(port=port_list[index])
                clients[index] = self.client
            # self.frame_read=cv2.resize(self.frame_read,(1920,1080))
            # print(self.frame_read.shape)
            # cv2.namedWindow(self.frame_name, cv2.WINDOW_NORMAL)
            # cv2.imshow(self.frame_name, self.frame_read)
            # print(self.frame_name)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
"""
该类是用于处理视频分析任务的线程类，继承自 threading.Thread 类。具体注释如下：

threadID：线程ID。
name：线程名称。
frame_read：读取的视频帧，初始值为 None。
thread_num：线程数量。
frame_name：视频窗口名称。
img_infer：图像推断函数。
model：模型。
client：客户端连接对象，初始值为 None。
run()：重写父类的 run() 方法。在循环中调用 img_infer() 对视频帧进行推断，并将推断后的图像通过客户端连接对象显示到客户端上。如果数据出错，则随机生成一张图像，并关闭当前客户端连接，创建新的客户端连接对象，并将新对象替换旧对象。循环等待按键事件，按下“q”键退出循环。最后，关闭窗口。
"""

face_location1="绞车室一楼"
face_location2="绞车室二楼"
class Video_analyze_thread1(threading.Thread):
    def __init__(self, threadID, name,thread_num,frame_name,img_infer,model,face_location,client=None):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        # self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
        self.frame_read=None
        self.thread_num=thread_num
        self.frame_name=frame_name
        self.img_infer = img_infer
        self.model=model
        self.client = client
        self.face_status=[]
        self.face_location=face_location
        #睡岗计时器
        self.time1=None
        self.time2=None
        #疲劳计时器
        self.time3=None
        self.time4=None
        self.a = None
        self.b = None
        self.c = None
        self.d = None

        self.sleep_flag = False
        self.sleep_flag1 = False

    def run(self):
        while True:
            # time.sleep(0.5)
            try:
                self.frame_read,self.face_status = self.img_infer(self.thread_num.frame, self.model, device, imgsz1, names1, colors1)
                # print(len(self.face_status))
                #进行睡岗检查，检查不到人脸，或者检查到人脸但是没有眼睛
                if (len(self.face_status)==1 and 0 in self.face_status) or (len(self.face_status)==0):
                    #第一次监测到睡岗的情况，记录一下时间，这个时间可以被在此赋值为None
                    if self.time1==None:
                        self.time1=time.time()
                    else:
                        self.time2=time.time()
                        if self.time2-self.time1 > 15:
                            # print("睡岗")
                            if self.sleep_flag==False:
                                self.a=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                self.sleep_flag=True
                                # send(a+'睡岗',777)
                            # self.frame_read = cv2AddChineseText(self.frame_read, "睡岗", (1720, 100), (255, 0, 0), 80)
                            # if pygame.mixer.music.get_busy():
                            #     pass
                            # else:
                            #     pygame.mixer.music.load('./wav/sleep.wav')
                            #     pygame.mixer.music.set_volume(0.5)
                            #     pygame.mixer.music.play(1)
                else:
                    if self.sleep_flag:
                        self.b = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        send(self.a+' '+self.b+self.face_location+'睡岗',777)
                    self.sleep_flag=False
                    self.time1=None
                    self.time2=None
                if len(self.face_status) > 1:
                    if (1 in self.face_status) or (4 in self.face_status):
                        if self.time3==None:
                            self.time3=time.time()
                        else:
                            self.time4=time.time()
                            time_tied=self.time4-self.time3
                            if time_tied>=3 and time_tied<5:
                                if self.sleep_flag1 == False:
                                    self.c = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    self.sleep_flag1 = True
                                self.frame_read = cv2AddChineseText(self.frame_read, "一级", (1720, 100), (255, 0, 0), 80)
                                if pygame.mixer.music.get_busy():
                                    pass
                                else:
                                    pygame.mixer.music.load('./wav/one.wav')
                                    pygame.mixer.music.set_volume(0.5)
                                    pygame.mixer.music.play(1)
                            if time_tied>=5 and time_tied<8:
                                # print("二级")
                                self.frame_read = cv2AddChineseText(self.frame_read, "二级", (1720, 100), (255, 0, 0), 80)
                                if pygame.mixer.music.get_busy():
                                    pass
                                else:
                                    pygame.mixer.music.load('./wav/two.wav')
                                    pygame.mixer.music.set_volume(0.5)
                                    pygame.mixer.music.play(1)
                            if time_tied>=8 and time_tied<10:
                                # print("三级")
                                self.frame_read = cv2AddChineseText(self.frame_read, "三级", (1720, 100), (255, 0, 0), 80)
                                if pygame.mixer.music.get_busy():
                                    pass
                                else:
                                    pygame.mixer.music.load('./wav/three.wav')
                                    pygame.mixer.music.set_volume(0.5)
                                    pygame.mixer.music.play(1)
                            if time_tied>=10 and time_tied<15:
                                # print("四级")
                                self.frame_read = cv2AddChineseText(self.frame_read, "四级", (1720, 100), (255, 0, 0), 80)
                                if pygame.mixer.music.get_busy():
                                    pass
                                else:
                                    pygame.mixer.music.load('./wav/four.wav')
                                    pygame.mixer.music.set_volume(0.5)
                                    pygame.mixer.music.play(1)
                            if time_tied>=15:
                                # print("四级")
                                self.frame_read = cv2AddChineseText(self.frame_read, "五级", (1720, 100), (255, 0, 0), 80)
                                if pygame.mixer.music.get_busy():
                                    pass
                                else:
                                    pygame.mixer.music.load('./wav/five.wav')
                                    pygame.mixer.music.set_volume(0.5)
                                    pygame.mixer.music.play(1)
                    else:
                        if self.sleep_flag1:
                            self.d = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            send(self.c +' '+ self.d + self.face_location + '疲劳', 777)
                        self.sleep_flag1 = False
                        self.time3=None
                        self.time4=None
                RT_Image(self.frame_read, self.client)
            except:
                # self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
                self.frame_read = np.random.rand(480,640,3)*255
                print("Data Error",self.thread_num)

                self.client.close()
                index = clients.index(self.client)
                self.client = create_client(port=port_list[index])
                clients[index] = self.client
            # cv2.namedWindow(self.frame_name, cv2.WINDOW_NORMAL)
            # cv2.imshow(self.frame_name, self.frame_read)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

#不分级报警
class Video_analyze_thread2(threading.Thread):
    def __init__(self, threadID, name, thread_num, frame_name, img_infer, model,address,face_location,client=None):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        # self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
        self.frame_read = None
        self.face_location = face_location
        self.thread_num = thread_num
        self.frame_name = frame_name
        self.img_infer = img_infer
        self.model1 = model
        self.face_status = []
        self.client = client
        self.sleep_flag = False
        self.sleep_flag1 = False
        self.e = None
        #这个线程对应的继电器地址
        self.address=address
        # 睡岗计时器
        self.time1 = None
        self.time2 = None
        # 疲劳计时器
        self.time3 = None
        self.time4 = None
        self.e = None
        self.f = None
        self.i = None
        self.g = None


    def run(self):
        global sign_sleep
        while True:
            # time.sleep(0.5)
            try:
                self.frame_read, self.face_status = self.img_infer(self.thread_num.frame, self.model1, device, imgsz1,
                                                                   names1, colors1)
                if (len(self.face_status)==1 and 0 in self.face_status) or (len(self.face_status)==0):
                    #第一次监测到睡岗的情况，记录一下时间，这个时间可以被在此赋值为None
                    if self.time1==None:
                        self.time1=time.time()
                    else:
                        self.time2=time.time()
                        if self.time2-self.time1 > 15:
                            if self.sleep_flag == False:
                                self.e = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                self.sleep_flag = True
                            # self.frame_read = cv2AddChineseText(self.frame_read, "睡岗", (1720, 100), (255, 0, 0), 80)
                else:
                    self.sleep_flag = False
                    self.f = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    send(self.e +' '+ self.f + self.face_location + '睡岗', 777)
                    self.time1=None
                    self.time2=None

                if len(self.face_status) > 1:

                    if (1 in self.face_status) or (4 in self.face_status):
                        if self.time3==None:
                            self.time3=time.time()
                        else:
                            self.time4=time.time()
                            time_tied=self.time4-self.time3
                            if time_tied>=5:
                                if self.sleep_flag1 == False:
                                    self.i = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    self.sleep_flag1 = True
                                self.frame_read = cv2AddChineseText(self.frame_read, "疲劳", (1720, 100), (255, 0, 0), 80)
                                # master.execute(slave=1, function_code=md.WRITE_MULTIPLE_REGISTERS, starting_address=self.address,
                                #                quantity_of_x=1, output_value=[1])
                                if self.address==12:
                                    sign_sleep[0]=1
                                if self.address==13:
                                    sign_sleep[1]=1
                    else:
                        self.sleep_flag1 = False
                        self.g = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        send(self.i + ' '+self.g + self.face_location + '疲劳', 777)
                        self.time3 = None
                        self.time4 = None
                        # master.execute(slave=1, function_code=md.WRITE_MULTIPLE_REGISTERS, starting_address=self.address,
                        #                quantity_of_x=1, output_value=[0])
                        if self.address == 12:
                            sign_sleep[0] = 0
                        if self.address == 13:
                            sign_sleep[1] = 0
                RT_Image(self.frame_read, self.client)
            except:
                # self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
                self.frame_read = np.random.rand(480, 640, 3) * 255
                print("Data Error", self.thread_num)
                self.client.close()
                index = clients.index(self.client)
                self.client = create_client(port=port_list[index])
                clients[index] = self.client
            # self.frame_read = cv2.resize(self.frame_read, (1920, 1080))
            # print(self.frame_read.shape)
            # cv2.namedWindow(self.frame_name, cv2.WINDOW_NORMAL)
            # cv2.imshow(self.frame_name, self.frame_read)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
if __name__ == "__main__":
    # thread_ana4 = Video_analyze_thread(15, "video_ana4", thread_read4, 'frame4', image_inference4, model1,client4)
    # thread_ana7 = Video_analyze_thread(16, "video_ana7", thread_read7, 'frame7', image_inference7, model1,client7)
    # thread_ana8 = Video_analyze_thread(17, "video_ana8", thread_read8, 'frame8', image_inference8, model1,client8)
    #
    # thread_ana1 = Video_analyze_thread1(7, "video_ana1", thread_read1, 'frame1', image_inference, model,face_location1,client1)
    thread_ana2 = Video_analyze_thread2(8, "video_ana2", thread_read2, 'frame2', image_inference,model,12,face_location2,client2)
    # thread_ana3 = Video_analyze_thread2(9, "video_ana3", thread_read3, 'frame3', image_inference, model,13,face_location2,client3)

    # thread_read1.start()
    thread_read2.start()
    # thread_read3.start()
    # thread_read4.start()
    # thread_read7.start()
    # thread_read8.start()

    time.sleep(2)
    # thread_ana1.start()
    thread_ana2.start()
    # thread_ana3.start()
    # thread_ana4.start()
    # thread_ana7.start()
    # thread_ana8.start()
    thread_Kaiguan.start()

    # thread_read1.join()
    thread_read2.join()
    # thread_read3.join()
    # thread_read4.join()
    # thread_read7.join()
    # thread_read8.join()

    # thread_ana1.join()
    thread_ana2.join()
    # thread_ana3.join()
    # thread_ana4.join()
    # thread_ana7.join()
    # thread_ana8.join()

    thread_Kaiguan.join()

