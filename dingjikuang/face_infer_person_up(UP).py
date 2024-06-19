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
from queue import Queue
metalocA = threading.Lock()
lock = threading.Lock()


master = mt.TcpMaster("192.168.65.206", 8234)
master1 = mt.TcpMaster("192.168.65.204", 8234)
# 设置响应等待时间
master.set_timeout(2.0)
master1.set_timeout(2.0)

pygame.mixer.init()  # 初始化

device = select_device('0')
half = device.type != 'cpu'  # half precision only supported on CUDA

weights = 'best456.pt'
# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

weights1 = 'face_detection.pt'
model1 = attempt_load(weights1, map_location=device)  # 人员检测模型加载
stride1 = int(model1.stride.max())  # model stride
imgsz1 = check_img_size(640, s=stride1)  # check img_size
names1 = model1.module.names if hasattr(model1, 'module') else model1.names
colors1 = [[random.randint(0, 255) for _ in range(3)] for _ in names1]


if half:
    model.half()
    model1.half()
# cap=cv2.VideoCapture('3.mp4')
url4 = 'rtsp://admin:hy123456@192.168.4.166:554/live'
url5 = 'rtsp://admin:cs123456@192.168.4.247:554/live'
# url4='22/9.mp4'
# url5='22/4.mp4'
url6 = 'rtsp://admin:cs123456@192.168.4.249:554/live'
# url6='22/6.mp4'
url1 = 'rtsp://admin:hy123456@192.168.65.65:554/live'
url2 = 'rtsp://admin:hy123456@192.168.4.139:554/live'
url3 = 'rtsp://admin:hy123456@192.168.4.140:554/live'
# url1='22/1.mp4'
# url2='22/2.mp4'
# url3='22/1.mp4'
video_weight = 1920
video_hight = 1080

# 输出信号设置
sign = [0, 0, 0]

sign_sleep = [0, 0]

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

# 变量设置
coust = 0

real_time_start4 = [0, 0]
real_time_start7 = [0, 0]
real_time_start8 = [0, 0]

# 电子围栏设置
m4 = Path([(524, 101), (524, 912), (1445, 911), (1445, 100)])
# 4上井口
m7 = Path([(730, 6), (730, 268), (1134, 266), (1132, 6)])
n7 = Path([(680, 327), (159, 1058), (1441, 1061), (1167, 326)])
# 7.1-8.1双罐进车上井口
m8 = Path([(876, 5), (876, 351), (1477, 350), (1477, 6)])
n8 = Path([(777, 418), (357, 1060), (1719, 1061), (1545, 455)])


# # 7.2-8.2双罐出车上井口

def cv2AddChineseText(img, text, position, textColor, textSize):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def image_inference4(frame, model1, device, half, imgsz, names, colors):
    t1 = time.time()
    global sign4, Time11, Time12, it6
    # 截取第一个区域
    frame = cv2.line(frame, (524, 101), (524, 912), (0, 255, 0), 4)
    frame = cv2.line(frame, (524, 912), (1445, 911), (0, 255, 0), 4)
    frame = cv2.line(frame, (1445, 911), (1445, 100), (0, 255, 0), 4)
    frame = cv2.line(frame, (1445, 100), (524, 101), (0, 255, 0), 4)
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
        global coust, t9, t10
        coust = 1
        if len(pred) and coust == 1:
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(pred):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
            for *xyxy, conf, cls in reversed(pred):
                if m4.contains_point((int(xyxy[-4]),int(xyxy[-3]))) or m4.contains_point((int(xyxy[-2]),int(xyxy[-1]))) or m4.contains_point((int(xyxy[-4]),int(xyxy[-1]))) or m4.contains_point((int(xyxy[-2]),int(xyxy[-1]))):
                    cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,
                                lineType=cv2.LINE_AA)
                    sign4 = 1
                    Time11 = time.time()
                    if it6 == 0:
                        it6 = time.time()
                        real_time_start4[0] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
                            send('上井口' + a + 'person')
    else:
        sign4 = 0
        it6 = 0
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    # print(6)
    cv2.putText(frame, 'FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    # print(7)
    return frame


# 4上井口
def image_inference7(frame, model1, device, half, imgsz, names, colors):
    t1 = time.time()
    global sign5, sign6, Time13, Time14, Time15, Time16, it7, it8

    frame = cv2.line(frame, (730, 6), (730, 268), (0, 255, 0), 4)
    frame = cv2.line(frame, (730, 268), (1134, 266), (0, 255, 0), 4)
    frame = cv2.line(frame, (1134, 266), (1132, 6), (0, 255, 0), 4)
    frame = cv2.line(frame, (1132, 6), (730, 6), (0, 255, 0), 4)

    frame = cv2.line(frame, (680, 327), (159, 1058), (0, 255, 0), 4)
    frame = cv2.line(frame, (159, 1058), (1441, 1061), (0, 255, 0), 4)
    frame = cv2.line(frame, (1441, 1061), (1167, 326), (0, 255, 0), 4)
    frame = cv2.line(frame, (1167, 326), (680, 327), (0, 255, 0), 4)
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
        global coust, none6, t15, t16
        coust = 1
        if len(pred) and coust == 1:
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(pred):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
            for *xyxy, conf, cls in reversed(pred):
                # if (m7.contains_point(
                #         (int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                if m7.contains_point((int(xyxy[-4]), int(xyxy[-3]))) or m7.contains_point((int(xyxy[-2]), int(
                        xyxy[-1]))) or m7.contains_point((int(xyxy[-4]), int(xyxy[-1]))) or m7.contains_point(
                    (int(xyxy[-2]), int(xyxy[-1]))):
                    cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,
                                lineType=cv2.LINE_AA)
                    sign5 = 1
                    Time13 = time.time()
                    if it7 == 0:
                        it7 = time.time()
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
                            send('上井口双罐进车' + a + 'person')
                # --------------------------------------------------------------------------------------------------------------------
                # if (n7.contains_point(
                #         (int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                if n7.contains_point((int(xyxy[-4]), int(xyxy[-3]))) or n7.contains_point((int(xyxy[-2]), int(
                            xyxy[-1]))) or n7.contains_point((int(xyxy[-4]), int(xyxy[-1]))) or n7.contains_point(
                    (int(xyxy[-2]), int(xyxy[-1]))):
                    cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,
                                lineType=cv2.LINE_AA)
                    sign6 = 1
                    Time15 = time.time()
                    if it8 == 0:
                        it8 = time.time()
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
                            send('上井口双罐进车' + a + 'person')
    # ----------------------------------------------------------------------------------------------------------------------
    else:
        sign5 = 0
        sign6 = 0
        it7 = 0
        it8 = 0
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, 'FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame


# 7.1-8.1双罐进车上井口
def image_inference8(frame, model1, device, half, imgsz, names, colors):
    t1 = time.time()
    global  sign7, sign8, Time17, Time18, Time19, Time20, it9, it10
    frame = cv2.line(frame, (876, 5), (876, 351), (0, 255, 0), 4)
    frame = cv2.line(frame, (876, 351), (1477, 350), (0, 255, 0), 4)
    frame = cv2.line(frame, (1477, 350), (1477, 6), (0, 255, 0), 4)
    frame = cv2.line(frame, (1477, 6), (876, 5), (0, 255, 0), 4)

    frame = cv2.line(frame, (777, 418), (357, 1060), (0, 255, 0), 4)
    frame = cv2.line(frame, (357, 1060), (1719, 1061), (0, 255, 0), 4)
    frame = cv2.line(frame, (1719, 1061), (1545, 455), (0, 255, 0), 4)
    frame = cv2.line(frame, (1545, 455), (777, 418), (0, 255, 0), 4)

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
        global coust, none7, t17, t18
        coust = 1
        if len(pred) and coust == 1:
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(pred):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
            for *xyxy, conf, cls in reversed(pred):
                # if (m8.contains_point(
                #         (int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                if m8.contains_point((int(xyxy[-4]), int(xyxy[-3]))) or m8.contains_point((int(xyxy[-2]), int(
                        xyxy[-1]))) or m8.contains_point((int(xyxy[-4]), int(xyxy[-1]))) or m8.contains_point((
                    int(xyxy[-2]), int(xyxy[-1]))):
                    cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,
                                lineType=cv2.LINE_AA)
                    sign7 = 1
                    Time17 = time.time()
                    if it9 == 0:
                        it9 = time.time()
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
                            send('上井口双罐出车' + a + 'person')
                # -----------------------------------------------------------------------------------------------------------------------
                # if (n8.contains_point(
                #         (int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                if n8.contains_point((int(xyxy[-4]), int(xyxy[-3]))) or n8.contains_point((int(xyxy[-2]), int(
                            xyxy[-1]))) or n8.contains_point((int(xyxy[-4]), int(xyxy[-1]))) or n8.contains_point((
                        int(xyxy[-2]), int(xyxy[-1]))):
                    cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,
                                lineType=cv2.LINE_AA)
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
                            send('上井口双罐出车' + a + 'person')
    # -----------------------------------------------------------------------------------------------------------------------
    else:
        sign7 = 0
        sign8 = 0
        it9 = 0
        it10 = 0
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, 'FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    # print(frame.shape)
    return frame


# 7.2-8.2双罐出车上井口
def image_inference(frame, model, device, imgsz, names, colors):
    t1 = time.time()
    face_status = []
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
    return frame, face_status


def send(data, port=999):
    try:
        client = socket.socket()
        # 连接服务器
        addr = ('192.168.65.205', port)
        client.connect(addr)
        # 发送数据
        client.send(data.encode('utf-8'))
        client.close()
    except:
        pass


class Carame_Accept_Object:
    def __init__(self):
        self.resolution = (1280, 720)  # 分辨率
        self.img_fps = 15  # 每秒传输多少帧数


def check_option(object, client):
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
    time.sleep(0.2)  # 推迟线程运行0.1s
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


object = Carame_Accept_Object()
# 这里的顺序要和receive里面一致
client1 = create_client(8880)
client2 = create_client(8881)
client3 = create_client(8882)
client4 = create_client(8883)
client7 = create_client(8884)
client8 = create_client(8885)
port_list = [8880, 8881, 8882, 8883, 8884, 8885]
clients = [client1, client2, client3, client4, client7, client8]
metalocA = threading.Lock()


class Kaiguanliang(threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name

    def _send(self, data, port=999):
        try:
            client = socket.socket()
            # 连接服务器
            addr = ('192.168.65.205', port)
            client.connect(addr)
            # 发送数据
            client.send(data.encode('utf-8'))
            client.close()
        except:
            pass
    def run(self):
        global sign4, sign5, sign6, sign7, sign8, sign, sign_sleep
        while True:
            sign[0] = sign4
            if (thread_ana4.videodie > 5) or (thread_read4.video_flag==False):
                sign[0]=0

            if sign5 == 1 or sign7 == 1:
                sign[1] = 1
            else:
                sign[1] = 0
            if (thread_ana7.videodie > 5)  or (thread_read7.video_flag==False):
                sign[1]=0
            if (thread_ana8.videodie > 5)  or (thread_read8.video_flag==False):
                sign[1]=0

            if sign6 == 1 or sign8 == 1:
                sign[2] = 1
            else:
                sign[2] = 0
            if (thread_ana7.videodie > 5) or (thread_read7.video_flag==False):
                sign[2]=0
            if (thread_ana8.videodie > 5) or (thread_read8.video_flag==False):
                sign[2]=0
            print(sign)

            if (thread_ana2.videodie > 5) or (thread_read2.video_flag==False):
                sign_sleep[0]=0
                
            if (thread_ana3.videodie > 5) or (thread_read3.video_flag==False):
                sign_sleep[1]=0
            self._send('master1;3;'+str([sign[0]]))
            self._send('master;3;'+str([sign[1], sign[2], sign_sleep[0], sign_sleep[1]]))
            # try:
            #     master.execute(slave=1, function_code=md.WRITE_MULTIPLE_REGISTERS, starting_address=3,
            #                    quantity_of_x=1, output_value=[sign[0]])
            # except:
            #     print("time out")
            #     master = mt.TcpMaster("192.168.65.206", 8234)
            #     # 设置响应等待时间
            #     master.set_timeout(2.0)
            #
            # try:
            #     master1.execute(slave=1, function_code=md.WRITE_MULTIPLE_REGISTERS, starting_address=3,
            #                     quantity_of_x=4, output_value=[sign[1], sign[2], sign_sleep[0], sign_sleep[1]])
            # except:
            #     print("time out1")
            #     master1 = mt.TcpMaster("192.168.65.204", 8234)
            #     # 设置响应等待时间
            #     master1.set_timeout(2.0)
            time.sleep(2)


class Video_receive_thread(threading.Thread):
    def __init__(self, threadID, name, url, frame_name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.url = url
        self.frame_name = frame_name
        # float = np.float32()
        # self.frame = np.zeros(shape=(1080, 1920, 3), dtype=float)
        self.frame = None
        self.cap = cv2.VideoCapture(self.url)
        self.que=Queue()
        self.video_flag=True

    def run(self):
        # while (self.cap.isOpened()):
        while True:
            self.ret, self.frame = self.cap.read()
            self.que.put(self.frame)
            if self.que.qsize()>2:
                self.que.get()
            if self.ret == False:
                self.video_flag=False
                self.cap = cv2.VideoCapture(self.url)
                continue
            else:
                self.video_flag = True
            # print(self.ret)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # self.cap.release()
        cv2.destroyAllWindows()


thread_read1 = Video_receive_thread(1, "video_read1", url1, 'frame1')
thread_read2 = Video_receive_thread(2, "video_read2", url2, 'frame2')
thread_read3 = Video_receive_thread(3, "video_read3", url3, 'frame3')

thread_read4 = Video_receive_thread(4, "video_read4", url4, 'frame4')
thread_read7 = Video_receive_thread(7, "video_read7", url5, 'frame7')
thread_read8 = Video_receive_thread(8, "video_read8", url6, 'frame8')
thread_Kaiguan = Kaiguanliang(22, "USB_Kaiguan")


class Video_analyze_thread(threading.Thread):
    def __init__(self, threadID, name, thread_num, frame_name, img_infer, model, client=None):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        # self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
        self.frame_read = None
        self.thread_num = thread_num
        self.frame_name = frame_name
        self.img_infer = img_infer
        # self.model = deepcopy(model)
        self.model = model
        self.client = client
        self.videodie = 0

    def run(self):
        while True:
            # time.sleep(0.5)
            try:
                self.frame_read = self.img_infer(self.thread_num.que.get(), self.model, device, half, imgsz, names,
                                                 colors)
                RT_Image(self.frame_read, self.client)
                self.videodie = 0
            except:
                # self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
                self.frame_read = np.random.rand(1080, 1920, 3) * 255
                print("Data Error", self.thread_num)
                self.videodie +=1
                self.client.close()
                index = clients.index(self.client)
                self.client = create_client(port=port_list[index])
                clients[index] = self.client
                time.sleep(1)


            # self.frame_read=cv2.resize(self.frame_read,(1920,1080))
            # print(self.frame_read.shape)
            # cv2.namedWindow(self.frame_name, cv2.WINDOW_NORMAL)
            # cv2.imshow(self.frame_name, self.frame_read)
            # print(self.frame_name)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


face_location1 = "绞车室一楼"
face_location2 = "下口单罐打点硐室"


class Video_analyze_thread1(threading.Thread):
    def __init__(self, threadID, name, thread_num, frame_name, img_infer, model, face_location, client=None):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        # self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
        self.frame_read = None
        self.thread_num = thread_num
        self.frame_name = frame_name
        self.img_infer = img_infer
        self.model = model
        self.client = client
        self.face_status = []
        self.face_location = face_location
        # 睡岗计时器
        self.time1 = None
        self.time2 = None
        # 疲劳计时器
        self.time3 = None
        self.time4 = None
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
                self.frame_read, self.face_status = self.img_infer(self.thread_num.que.get()[0:1080, 450:1250], self.model, device, imgsz1,
                                                                   names1, colors1)
                # print(len(self.face_status))
                # 进行睡岗检查，检查不到人脸，或者检查到人脸但是没有眼睛
                if (len(self.face_status) == 1 and 0 in self.face_status) or (len(self.face_status) == 0):
                    # 第一次监测到睡岗的情况，记录一下时间，这个时间可以被在此赋值为None
                    if self.time1 == None:
                        self.time1 = time.time()
                    else:
                        self.time2 = time.time()
                        if self.time2 - self.time1 > 15:
                            # print("睡岗")
                            if self.sleep_flag == False:
                                self.a = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                self.sleep_flag = True
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
                        send(self.a + ' ' + self.b + self.face_location + '睡岗')
                        self.a=None
                    self.sleep_flag = False
                    self.time1 = None
                    self.time2 = None
                if len(self.face_status) > 1:
                    if (self.face_status.count(1)==2) or (4 in self.face_status):
                        if self.time3 == None:
                            self.time3 = time.time()
                        else:
                            self.time4 = time.time()
                            time_tied = self.time4 - self.time3
                            if time_tied >= 3 and time_tied < 5:
                                if self.sleep_flag1 == False:
                                    self.c = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    self.sleep_flag1 = True
                                self.frame_read = cv2AddChineseText(self.frame_read, "一级", (600, 600), (255, 0, 0),
                                                                    80)
                                if pygame.mixer.music.get_busy():
                                    pass
                                else:
                                    pygame.mixer.music.load('./wav/one.wav')
                                    pygame.mixer.music.set_volume(0.5)
                                    pygame.mixer.music.play(1)
                            if time_tied >= 5 and time_tied < 8:
                                # print("二级")
                                self.frame_read = cv2AddChineseText(self.frame_read, "二级", (600, 600), (255, 0, 0),
                                                                    80)
                                if pygame.mixer.music.get_busy():
                                    pass
                                else:
                                    pygame.mixer.music.load('./wav/two.wav')
                                    pygame.mixer.music.set_volume(0.5)
                                    pygame.mixer.music.play(1)
                            if time_tied >= 8 and time_tied < 10:
                                # print("三级")
                                self.frame_read = cv2AddChineseText(self.frame_read, "三级", (600, 600), (255, 0, 0),
                                                                    80)
                                if pygame.mixer.music.get_busy():
                                    pass
                                else:
                                    pygame.mixer.music.load('./wav/three.wav')
                                    pygame.mixer.music.set_volume(0.5)
                                    pygame.mixer.music.play(1)
                            if time_tied >= 10 and time_tied < 15:
                                # print("四级")
                                self.frame_read = cv2AddChineseText(self.frame_read, "四级", (600, 600), (255, 0, 0),
                                                                    80)
                                if pygame.mixer.music.get_busy():
                                    pass
                                else:
                                    pygame.mixer.music.load('./wav/four.wav')
                                    pygame.mixer.music.set_volume(0.5)
                                    pygame.mixer.music.play(1)
                            if time_tied >= 15:
                                # print("四级")
                                self.frame_read = cv2AddChineseText(self.frame_read, "五级", (600, 600), (255, 0, 0),
                                                                    80)
                                if pygame.mixer.music.get_busy():
                                    pass
                                else:
                                    pygame.mixer.music.load('./wav/five.wav')
                                    pygame.mixer.music.set_volume(0.5)
                                    pygame.mixer.music.play(1)
                    else:
                        if self.sleep_flag1:
                            self.d = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            send(self.c + ' ' + self.d + self.face_location + '疲劳')
                        self.sleep_flag1 = False
                        self.time3 = None
                        self.time4 = None
                RT_Image(self.frame_read, self.client)
            except:
                # self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
                self.frame_read = np.random.rand(480, 640, 3) * 255
                print("Data Error", self.thread_num)

                self.client.close()
                index = clients.index(self.client)
                self.client = create_client(port=port_list[index])
                clients[index] = self.client


            # cv2.namedWindow(self.frame_name, cv2.WINDOW_NORMAL)
            # cv2.imshow(self.frame_name, self.frame_read)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


# 不分级报警
class Video_analyze_thread2(threading.Thread):
    def __init__(self, threadID, name, thread_num, frame_name, img_infer, model, address, face_location, client=None):
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
        # 这个线程对应的继电器地址
        self.address = address
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
        self.videodie = 0

    def run(self):
        global sign_sleep
        while True:
            # time.sleep(0.5)
            try:
                self.frame_read, self.face_status = self.img_infer(self.thread_num.que.get()[300:1000, 600:1400], self.model1, device, imgsz1,
                                                                   names1, colors1)
                if (len(self.face_status) == 1 and 0 in self.face_status) or (len(self.face_status) == 0):
                    # 第一次监测到睡岗的情况，记录一下时间，这个时间可以被在此赋值为None
                    if self.time1 == None:
                        self.time1 = time.time()
                    else:
                        self.time2 = time.time()
                        if self.time2 - self.time1 > 15:
                            if self.sleep_flag == False:
                                self.e = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                self.sleep_flag = True
                            # self.frame_read = cv2AddChineseText(self.frame_read, "睡岗", (1720, 100), (255, 0, 0), 80)
                else:
                    self.sleep_flag = False
                    self.f = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    if self.e!=None:
                        send(self.e + ' ' + self.f + self.face_location + '睡岗')
                        self.e=None
                    self.time1 = None
                    self.time2 = None

                if len(self.face_status) > 1:

                    if (self.face_status.count(1)==2) or (4 in self.face_status):
                        if self.time3 == None:
                            self.time3 = time.time()
                        else:
                            self.time4 = time.time()
                            time_tied = self.time4 - self.time3
                            if time_tied >= 5:
                                if self.sleep_flag1 == False:
                                    self.i = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    self.sleep_flag1 = True
                                self.frame_read = cv2AddChineseText(self.frame_read, "疲劳", (600, 600), (255, 0, 0),
                                                                    80)
                                # master.execute(slave=1, function_code=md.WRITE_MULTIPLE_REGISTERS, starting_address=self.address,
                                #                quantity_of_x=1, output_value=[1])
                                if self.address == 12:
                                    sign_sleep[0] = 1
                                if self.address == 13:
                                    sign_sleep[1] = 1
                    else:
                        self.sleep_flag1 = False
                        self.g = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        if self.i!=None:
                            send(self.i + ' ' + self.g + self.face_location + '疲劳')
                            self.i=None
                        self.time3 = None
                        self.time4 = None
                        # master.execute(slave=1, function_code=md.WRITE_MULTIPLE_REGISTERS, starting_address=self.address,
                        #                quantity_of_x=1, output_value=[0])
                        if self.address == 12:
                            sign_sleep[0] = 0
                        if self.address == 13:
                            sign_sleep[1] = 0
                else:
                    if self.address == 12:
                        sign_sleep[0] = 0
                    if self.address == 13:
                        sign_sleep[1] = 0
                RT_Image(self.frame_read, self.client)
                self.videodie = 0
            except:
                # self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
                self.frame_read = np.random.rand(480, 640, 3) * 255
                print("Data Error", self.thread_num)
                self.videodie +=1
                self.client.close()
                index = clients.index(self.client)
                self.client = create_client(port=port_list[index])
                clients[index] = self.client
                time.sleep(1)
            # self.frame_read = cv2.resize(self.frame_read, (1920, 1080))
            # print(self.frame_read.shape)
            # cv2.namedWindow(self.frame_name, cv2.WINDOW_NORMAL)
            # cv2.imshow(self.frame_name, self.frame_read)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    thread_ana4 = Video_analyze_thread(15, "video_ana4", thread_read4, 'frame4', image_inference4, model, client4)
    thread_ana7 = Video_analyze_thread(16, "video_ana7", thread_read7, 'frame7', image_inference7, model, client7)
    thread_ana8 = Video_analyze_thread(17, "video_ana8", thread_read8, 'frame8', image_inference8, model, client8)

    thread_ana1 = Video_analyze_thread1(7, "video_ana1", thread_read1, 'frame1', image_inference, model1, face_location1,
                                        client1)
    thread_ana2 = Video_analyze_thread2(8, "video_ana2", thread_read2, 'frame2', image_inference, model1, 12,
                                        face_location2, client2)
    thread_ana3 = Video_analyze_thread2(9, "video_ana3", thread_read3, 'frame3', image_inference, model1, 13,
                                        face_location2, client3)

    thread_read1.start()
    thread_read2.start()
    thread_read3.start()
    thread_read4.start()
    thread_read7.start()
    thread_read8.start()

    time.sleep(2)
    thread_ana1.start()
    thread_ana2.start()
    thread_ana3.start()
    thread_ana4.start()
    thread_ana7.start()
    thread_ana8.start()
    thread_Kaiguan.start()

    thread_read1.join()
    thread_read2.join()
    thread_read3.join()
    thread_read4.join()
    thread_read7.join()
    thread_read8.join()

    thread_ana1.join()
    thread_ana2.join()
    thread_ana3.join()
    thread_ana4.join()
    thread_ana7.join()
    thread_ana8.join()

    thread_Kaiguan.join()
