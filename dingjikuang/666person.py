# -*- coding: utf-8 -*-
# from __init__ import *
# from step_defss.scenario_steps import *
#接后续代码
import os
from queue import Queue
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

# master=mt.TcpMaster("192.168.65.206",8234)
# master1=mt.TcpMaster("192.168.65.204",8234)
#设置响应等待时间
# master.set_timeout(2.0)
# master1.set_timeout(2.0)
device = select_device('0')
half = device.type != 'cpu'  # half precision only supported on CUDA

weights='best456.pt'
model = attempt_load(weights, map_location=device)  # 人员检测模型加载
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

if half:
    model.half()



pygame.mixer.init()

video_weight=1920
video_hight=1080

#输出信号设置
sign=[0,0,0,0,0]
sign1 = 0
sign2 = 0
sign3 = 0
sign4 = 0
sign5 = 0
sign6 = 0



Time1 = 0
Time2 = 0
Time3 = 0
Time4 = 0
Time5 = 0
Time6 = 0
Time7 = 0
Time8 = 0
Time9 = 0
Time10 = 0
Time11 = 0
Time12 = 0

it1 = 0
it2 = 0
it3 = 0
it4 = 0
it5 = 0
it6 = 0
#变量设置
coust=0

real_time_start1 = [0,0]
real_time_start2 = [0,0]
real_time_start3 = [0,0]

#电子围栏设置
m1 = Path([(965,1067), (512,257),(1908,320),(1908,1070)])
#111
# m2 = Path([(714 ,1069), (682 ,336),(1307 ,441),(1910 ,942)])
m2 = Path([(658,1069), (683 ,335),(1304 ,443),(1917 ,897),(1913,1068)])

# 222
m3 = Path([(2 , 740), (1156 , 151),(1446 , 182),(1257 , 1072), (6,1073)])
# 333

def image_inference1(frame,model1,device,half,imgsz,names,colors):
    t1 = time.time()
    global sign1,sign2,Time1,Time2,it1
    frame = cv2.line(frame, (965,1067), (512,257), (0, 255, 0), 4)
    frame = cv2.line(frame, (512,257), (1908,320), (0, 255, 0), 4)
    frame = cv2.line(frame, (1908,320), (1908,1070), (0, 255, 0), 4)
    frame = cv2.line(frame, (1908,1070), (965,1067), (0, 255, 0), 4)
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
    if len(pred):
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
        for *xyxy, conf, cls in reversed(pred):
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
        for *xyxy, conf, cls in reversed(pred):
#-----------------------------------------------------------------------------------------------------------------------输出1.1
            if m1.contains_point((int(xyxy[-4]), int(xyxy[-3]))) or m1.contains_point((int(xyxy[-2]), int(xyxy[-1]))) or m1.contains_point((int(xyxy[-4]), int(xyxy[-1])))or m1.contains_point((int(xyxy[-2]), int(xyxy[-1]))):
                cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,lineType=cv2.LINE_AA)
                sign1 = 1
                Time1 = time.time()
                if it1 == 0:
                    it1 =time.time()
                    real_time_start1[0] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            else:
                if sign1 == 1:
                    Time2 = time.time()
                    if Time2 - Time1 > 5:
                        sign1 = 0
                        it1 = 0
                        real_time_start1[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        a = str(real_time_start1)
                        print(000)
                        send('下进口东门调车机车厂'+a+'person')
                        print(000)
#-----------------------------------------------------------------------------------------------------------------------输出2.1
    else:
        sign1 = 0
        it1=0
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, '1FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame
# 111
def image_inference2(frame,model1,device,half,imgsz,names,colors):
    t1 = time.time()
    global sign2,Time3,Time4,it2
    frame = cv2.line(frame, (658 ,1069), (683 ,335), (0, 255, 0), 4)
    frame = cv2.line(frame, (683 ,335), (1304 ,443), (0, 255, 0), 4)
    frame = cv2.line(frame, (1304 ,443), (1917 ,897), (0, 255, 0), 4)
    frame = cv2.line(frame, (1917 ,897), (1913 ,1068), (0, 255, 0), 4)
    # frame = cv2.line(frame, (1902 ,876), (1904 ,988), (0, 255, 0), 4)
    frame = cv2.line(frame, (1913 ,1068), (658 ,1069), (0, 255, 0), 4)

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
    if len(pred):
        # Rescale boxes from img_size to im0 size
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
        for *xyxy, conf, cls in reversed(pred):
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
        for *xyxy, conf, cls in reversed(pred):
# -----------------------------------------------------------------------------------------------------------------------输出1.2
#             if (m2.contains_point((int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
            if m2.contains_point((int(xyxy[-4]), int(xyxy[-3]))) or m2.contains_point((int(xyxy[-2]),
                                                                        int(xyxy[-1]))) or m2.contains_point(
                (int(xyxy[-4]), int(xyxy[-1]))) or m2.contains_point((int(xyxy[-2]), int(xyxy[-1]))):
                cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,lineType=cv2.LINE_AA)
                sign2 = 1
                Time3 = time.time()
                if it2 == 0:
                    it2 =time.time()
                    real_time_start1[0] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            #         if pygame.mixer.music.get_busy():
            #             pass
            #         else:
            #             pygame.mixer.music.load('C:\yolov7\wav\sound.wav')
            #             pygame.mixer.music.set_volume(0.5)
            #             pygame.mixer.music.play()
            else:
                if sign2 == 1:
                    Time4 = time.time()
                    if Time4 - Time3 > 5:
                        sign2 = 0
                        it2 = 0
                        real_time_start1[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        a = str(real_time_start1)
                        print(000)
                        send('上井口进车场' + a + 'person')
                        print(000)
#-----------------------------------------------------------------------------------------------------------------------输出2.2
    else:
        sign2 = 0
        it2=0
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, 'FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame
# 222
def image_inference3(frame,model1,device,half,imgsz,names,colors):
    t1 = time.time()
    global sign3, Time5, Time6,it3
    frame = cv2.line(frame, (2 , 740), (1156 , 151), (0, 255, 0), 4)
    frame = cv2.line(frame, (1156 , 151), (1446 , 182), (0, 255, 0), 4)
    frame = cv2.line(frame, (1446 , 182), (1257 , 1072), (0, 255, 0), 4)
    frame = cv2.line(frame, (1257 , 1072), (6 , 1073), (0, 255, 0), 4)
    frame = cv2.line(frame, (6 , 1073), (2, 740), (0, 255, 0), 4)

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
    if len(pred) :
        # Rescale boxes from img_size to im0 size
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
        for *xyxy, conf, cls in reversed(pred):
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
        for *xyxy, conf, cls in reversed(pred):
            # if (m3.contains_point(
            #         (int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
            if m3.contains_point((int(xyxy[-4]), int(xyxy[-3]))) or m3.contains_point((int(xyxy[-2]),int(xyxy[-1]))) \
                    or m3.contains_point((int(xyxy[-4]), int(xyxy[-1])))\
                    or m3.contains_point((int(xyxy[-2]), int(xyxy[-1]))):
                cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,lineType=cv2.LINE_AA)
                sign3 = 1
                Time5 = time.time()
                if it3 == 0:
                    it3 =time.time()
                    real_time_start3[0] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    # if pygame.mixer.music.get_busy():
                    #     pass
                    # else:
                    #     pygame.mixer.music.load('C:\yolov7\wav\sound.wav')
                    #     pygame.mixer.music.set_volume(0.5)
                    #     pygame.mixer.music.play()
            else:
                if sign3 == 1:
                    Time6 = time.time()
                    if Time10 - Time9 > 5:
                        sign3= 0
                        it3 = 0
                        real_time_start3[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        a = str(real_time_start3)
                        print(000)
                        send('上井口出车场' + a + 'person')
                        print(000)
    else:
        sign3=0
        it3=0
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, '1FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame
# 333
def image_inference_person1(frame,model,device,half,imgsz,names,colors):
    global sign4,Time7,Time8,it4
    t1 = time.time()
    img = letterbox(frame, imgsz, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    with torch.no_grad():
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.25, classes=None, agnostic=False)[0]
    if len(pred):
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
        for *xyxy, conf, cls in reversed(pred):
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
        cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3, lineType=cv2.LINE_AA)
        sign4 = 1
        Time7 = time.time()
        if it4 == 0:
            it4 = time.time()
            real_time_start3[0] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    else:
        if sign4 == 1:
            Time8 = time.time()
            if Time8 - Time7 > 5:
                sign4 = 0
                it4 = 0
                real_time_start3[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                a = str(real_time_start3)
                print(111)
                send('某区域' + a + 'person')
                print(111)
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, 'FPS=%s' % FPS, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame

def image_inference_person2(frame,model,device,half,imgsz,names,colors):
    global sign5,Time9,Time10,it5
    t1 = time.time()
    img = letterbox(frame, imgsz, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    with torch.no_grad():
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.25, classes=None, agnostic=False)[0]
    if len(pred):
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
        for *xyxy, conf, cls in reversed(pred):
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
        cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3, lineType=cv2.LINE_AA)
        sign5 = 1
        Time9 = time.time()
        if it5 == 0:
            it5 = time.time()
            real_time_start3[0] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    else:
        if sign5 == 1:
            Time10 = time.time()
            if Time10 - Time9 > 5:
                sign5 = 0
                it5 = 0
                real_time_start3[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                a = str(real_time_start3)
                print(222)
                send('某区域' + a + 'person')
                print(222)
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, 'FPS=%s' % FPS, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame

def image_inference_person3(frame,model,device,half,imgsz,names,colors):
    global sign6,Time11,Time12,it6
    t1 = time.time()
    img = letterbox(frame, imgsz, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    with torch.no_grad():
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.25, classes=None, agnostic=False)[0]
    if len(pred):
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
        for *xyxy, conf, cls in reversed(pred):
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
        cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3, lineType=cv2.LINE_AA)
        sign6 = 1
        Time11 = time.time()
        if it6 == 0:
            it6 = time.time()
            real_time_start3[0] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    else:
        if sign6 == 1:
            Time12 = time.time()
            if Time12 - Time11 > 5:
                sign6 = 0
                it6 = 0
                real_time_start3[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                a = str(real_time_start3)
                print(333)
                send('某区域' + a + 'person')
                print(333)
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, 'FPS=%s' % FPS, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame

def send(data, port=5555):
    client = socket.socket()
    # 连接服务器
    addr = ('192.168.65.205', port)
    client.connect(addr)
    # 发送数据
    client.send(data.encode('utf-8'))
    # client.send(data)
    client.close()

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
    time.sleep(0.1)  # 推迟线程运行0.1s
    # _, object.img = camera.read()  # 读取视频每一帧
    # 核心代码在这里，图像从这里传过来，
    object.img = cv2.resize(img, object.resolution)  # 按要求调整图像大小(resolution必须为元组)
    _, img_encode = cv2.imencode('.jpg', object.img, img_param)  # 按格式生成图片
    img_code = numpy.array(img_encode)  # 转换成矩阵
    object.img_data = img_code.tobytes()  # 生成相应的字符串
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
print('111111')
object = Carame_Accept_Object()
# 这里的顺序要和receive里面一致
client1 = create_client(8898)
print('122222')
client2 = create_client(8899)
print('122222')
client3 = create_client(8900)
print('122222')
client4 = create_client(8901)
print('122222')
client7 = create_client(8902)
print('122222')
client8 = create_client(8903)
print('122222')
port_list = [8898, 8899,8900,8901,8902,8903]
clients = [client1, client2,client3, client4, client7, client8]
metalocA = threading.Lock()

# url1='C:/yolov7/22/3.mp4'
# url2='C:/yolov7/22/5.mp4'
# url1='C:/yolov7-main-dj/1-1.mp4'
# url2='C:/yolov7-main-dj/1-2.mp4'
# url3='C:/yolov7-main-dj/1-3.mp4'
# url3='C:/yolov7/22/10.mp4'
url1='rtsp://admin:hy123456@192.168.4.182:554/live'
url2='rtsp://admin:hy123456@192.168.7.142:554/live'
url3='rtsp://admin:hy123456@192.168.7.235:554/live'
# url4='C:/yolov7-main-dj/1-4.mp4'
# url7='C:/yolov7-main-dj/2-1.mp4'
# url8='C:/yolov7-main-dj/2-2.mp4'
url4='rtsp://admin:hy123456@192.168.4.253:554/live'
url7='rtsp://admin:hy123456@192.168.4.254:554/live'
url8='rtsp://admin:admin123@192.168.7.230:554/live'

# url1='22/3.mp4'
# url2='22/4.mp4'
# url3='22/5.mp4'
# url4='22/6.mp4'
# url7='22/7.mp4'
# url8='22/8.mp4'

#疲劳检测

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
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

class Kaiguanliang(threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
    def _send(self,data, port=5555):
        client = socket.socket()
        # 连接服务器
        addr = ('192.168.65.205', port)
        # print('send success')
        client.connect(addr)
        # 发送数据
        client.send(data.encode('utf-8'))
        # client.send(data)
        client.close()
    def run(self):

        global sign1,sign2,sign3,sign4,sign5,sign
        while True:
            sign[0] = sign1
            if (thread_ana1.videodie>5) or (thread_read1.video_flag==False):
                sign[0] = 0

            sign[1] = sign2
            if (thread_ana2.videodie > 5) or (thread_read2.video_flag==False):
                sign[1] = 0

            sign[2] = sign3
            if (thread_ana3.videodie > 5) or (thread_read3.video_flag==False):
                sign[2] = 0

            if sign4==1 or sign5==1:
                sign[3]=1
            else:
                sign[3]=0
            if (thread_ana4.videodie > 5) or (thread_read4.video_flag==False):
                sign[3] = 0
            if (thread_ana7.videodie > 5) or (thread_read7.video_flag==False):
                sign[3] = 0
                
            sign[4] = sign6
            if (thread_ana8.videodie > 5) or (thread_read8.video_flag==False):
                sign[4] = 0
            print(sign)
            try:
                self._send('master1;8;'+str(sign))
            except:
                pass
            try:
                self._send('master;8;'+str(sign))
            except:
                pass
            # try:
            #     master1.execute(slave=1, function_code=md.WRITE_MULTIPLE_REGISTERS, starting_address=8,
            #                         quantity_of_x=5, output_value=sign)
            # except:
            #     master1 = mt.TcpMaster("192.168.65.204", 8234)
            #     master1.set_timeout(2.0)
            #     print("time out")
            # try:
            #     master.execute(slave=1, function_code=md.WRITE_MULTIPLE_REGISTERS, starting_address=8,
            #                    quantity_of_x=5, output_value=sign)
            # except:
            #     master = mt.TcpMaster("192.168.65.206", 8234)
            #     master.set_timeout(2.0)
            #     print("time out")
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
        self.cap = cv2.VideoCapture(self.url,cv2.CAP_INTEL_MFX)
        self.que= Queue()
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
            # cv2.imshow(self.frame_name, self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # self.cap.release()
        cv2.destroyAllWindows()


thread_Kaiguan = Kaiguanliang(22,"USB_Kaiguan")
thread_read1 = Video_receive_thread(1, "video_read1",url1,'frame1')
thread_read2 = Video_receive_thread(2, "video_read2",url2,'frame2')
thread_read3 = Video_receive_thread(3, "video_read3",url3,'frame3')
thread_read4 = Video_receive_thread(4, "video_read4",url4,'frame4')
thread_read7 = Video_receive_thread(7, "video_read7",url7,'frame7')
thread_read8 = Video_receive_thread(8, "video_read8",url8,'frame8')

#-----------------------------------------------------------------

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
        self.videodie=0

    def run(self):
        while True:
            # time.sleep(0.5)
            try:
                self.frame_read = self.img_infer(self.thread_num.que.get(), self.model, device, half, imgsz, names, colors)
                RT_Image(self.frame_read, self.client)
                self.videodie = 0
            except:
                # self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
                self.frame_read = np.random.rand(1080,1920,3)*255
                print("Data Error",self.thread_num)
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


# csv 写入报警数据信息
path = os.path.abspath(os.path.dirname(sys.argv[0]))
time1 = time.strftime('%Y%m%d', time.localtime(time.time()))

if __name__ == "__main__":
    thread_ana1 = Video_analyze_thread(12, "video_ana1",thread_read1,'frame1',image_inference1,model,client1)
    thread_ana2 = Video_analyze_thread(13, "video_ana2",thread_read2,'frame2',image_inference2,model,client2)
    thread_ana3 = Video_analyze_thread(14, "video_ana3", thread_read3, 'frame3',image_inference3,model,client3)
    # ----------------------------------------------------------------------------------------------------------
    thread_ana4 = Video_analyze_thread(15, "video_ana4", thread_read4, 'frame4', image_inference_person1, model,client4)
    thread_ana7 = Video_analyze_thread(16, "video_ana7", thread_read7, 'frame7', image_inference_person2, model,client7)
    thread_ana8 = Video_analyze_thread(17, "video_ana8", thread_read8, 'frame8', image_inference_person3, model,client8)


    thread_read1.start()
    thread_read2.start()
    thread_read3.start()
    thread_read4.start()
    thread_read7.start()
    thread_read8.start()
    time.sleep(3)


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

