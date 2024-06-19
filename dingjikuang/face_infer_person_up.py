# -*- coding: utf-8 -*-
# from __init__ import *
# from step_defss.scenario_steps import *
# 接后续代码
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

master = mt.TcpMaster("192.168.65.206", 8234)
master1 = mt.TcpMaster("192.168.65.204", 8234)
# 设置响应等待时间
master.set_timeout(2.0)
master1.set_timeout(2.0)
device = select_device('0')
half = device.type != 'cpu'  # half precision only supported on CUDA

weights = 'best456.pt'
model = attempt_load(weights, map_location=device)  # 人员检测模型加载
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# weights='YOLO-fs.pt'
weights1 = 'face_detection.pt'
# Load model
model1 = attempt_load(weights1, map_location=device)  # load FP32 model

stride1 = int(model1.stride.max())  # model stride
imgsz1 = check_img_size(640, s=stride1)  # check img_size

names1 = model1.module.names if hasattr(model1, 'module') else model1.names
colors1 = [[random.randint(0, 255) for _ in range(3)] for _ in names1]

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))

if half:
    model.half()
    model1.half()

pygame.mixer.init()

video_weight = 1920
video_hight = 1080

# 输出信号设置
sign = [0, 0, 0]
sign1 = 0
sign2 = 0
sign3 = 0
sign4 = 0
sign5 = 0

sign_sleep = [0, 0]

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

it1 = 0
it2 = 0
it3 = 0
it4 = 0
it5 = 0

# 变量设置
coust = 0

real_time_start1 = [0, 0]
real_time_start2 = [0, 0]
real_time_start3 = [0, 0]

# 电子围栏设置
m1 = Path([(788, 9), (788, 278), (1108, 278), (1107, 10)])
n1 = Path([(778, 291), (448, 1061), (1457, 1061), (1136, 295)])
# 1.1-2.1上井口单罐进车
m2 = Path([(640, 9), (640, 374), (1144, 373), (1145, 10)])
n2 = Path([(660, 460), (482, 1063), (1415, 1063), (1218, 460)])
# # 1.2-2.2上井口单罐出车
m3 = Path([(109, 8), (108, 927), (1597, 927), (1597, 9)])


# 3上井口二平台双罐东

def image_inference1(frame, model1, device, half, imgsz, names, colors):
    t1 = time.time()
    global sign1, sign2, Time1, Time2, Time3, Time4, it1, it2
    frame = cv2.line(frame, (788, 9), (788, 278), (0, 255, 0), 4)
    frame = cv2.line(frame, (788, 278), (1108, 278), (0, 255, 0), 4)
    frame = cv2.line(frame, (1108, 278), (1107, 10), (0, 255, 0), 4)
    frame = cv2.line(frame, (1107, 10), (788, 9), (0, 255, 0), 4)

    frame = cv2.line(frame, (778, 291), (448, 1061), (0, 255, 0), 4)
    frame = cv2.line(frame, (448, 1061), (1457, 1061), (0, 255, 0), 4)
    frame = cv2.line(frame, (1457, 1061), (1136, 295), (0, 255, 0), 4)
    frame = cv2.line(frame, (1136, 295), (778, 291), (0, 255, 0), 4)
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
        global coust, none, t3, t4
        coust = 1
        if len(pred) and coust == 1:
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(pred):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
            for *xyxy, conf, cls in reversed(pred):
                # -----------------------------------------------------------------------------------------------------------------------输出1.1
                if m1.contains_point((int(xyxy[-4]), int(xyxy[-3]))) or m1.contains_point(
                        (int(xyxy[-2]), int(xyxy[-1]))) or m1.contains_point(
                        (int(xyxy[-4]), int(xyxy[-1]))) or m1.contains_point((int(xyxy[-2]), int(xyxy[-1]))):
                    cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,
                                lineType=cv2.LINE_AA)
                    sign1 = 1
                    Time1 = time.time()
                    if it1 == 0:
                        it1 = time.time()
                        real_time_start1[0] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        # if pygame.mixer.music.get_busy():
                        #     pass
                        # else:
                        #     pygame.mixer.music.load('C:\yolov7\wav\sound.wav')
                        #     pygame.mixer.music.set_volume(0.5)
                        #     pygame.mixer.music.play()
                else:
                    if sign1 == 1:
                        Time2 = time.time()
                        if Time2 - Time1 > 5:
                            sign1 = 0
                            it1 = 0
                            real_time_start1[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            a = str(real_time_start1)
                            send('上井口单罐进车' + a + 'person')
                # -----------------------------------------------------------------------------------------------------------------------输出2.1
                # if (n1.contains_point((int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                if n1.contains_point((int(xyxy[-4]), int(xyxy[-3]))) or n1.contains_point(
                        (int(xyxy[-2]), int(xyxy[-1]))) or n1.contains_point(
                    (int(xyxy[-4]), int(xyxy[-1]))) or n1.contains_point((int(xyxy[-2]), int(xyxy[-1]))):
                    cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,
                                lineType=cv2.LINE_AA)
                    sign2 = 1
                    Time3 = time.time()
                    if it2 == 0:
                        it2 = time.time()
                        real_time_start2[0] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        # if pygame.mixer.music.get_busy():
                        #     pass
                        # else:
                        #     pygame.mixer.music.load('C:\yolov7\wav\sound.wav')
                        #     pygame.mixer.music.set_volume(0.5)
                        #     pygame.mixer.music.play()
                else:
                    if sign2 == 1:
                        Time4 = time.time()
                        if Time4 - Time3 > 5:
                            sign2 = 0
                            it2 = 0
                            real_time_start2[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            a = str(real_time_start2)
                            send('上井口单罐进车' + a + 'person')
    else:
        sign1 = 0
        sign2 = 0
        it1 = 0
        it2 = 0
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, '1FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame


# 1.1-2.1上井口单罐进车
def image_inference2(frame, model1, device, half, imgsz, names, colors):
    t1 = time.time()
    global sign3, sign4, Time5, Time6, Time7, Time8, it3, it4
    frame = cv2.line(frame, (640, 9), (640, 374), (0, 255, 0), 4)
    frame = cv2.line(frame, (640, 374), (1144, 373), (0, 255, 0), 4)
    frame = cv2.line(frame, (1144, 373), (1145, 10), (0, 255, 0), 4)
    frame = cv2.line(frame, (1145, 10), (640, 9), (0, 255, 0), 4)

    frame = cv2.line(frame, (660, 460), (482, 1063), (0, 255, 0), 4)
    frame = cv2.line(frame, (482, 1063), (1415, 1063), (0, 255, 0), 4)
    frame = cv2.line(frame, (1415, 1063), (1218, 460), (0, 255, 0), 4)
    frame = cv2.line(frame, (1218, 460), (660, 460), (0, 255, 0), 4)

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
        global coust, none1, t5, t6
        coust = 1
        if len(pred) and coust == 1:
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(pred):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
            for *xyxy, conf, cls in reversed(pred):
                # -----------------------------------------------------------------------------------------------------------------------输出1.2
                #                 if (m2.contains_point((int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                if m2.contains_point((int(xyxy[-4]), int(xyxy[-3]))) or m2.contains_point(
                        (int(xyxy[-2]), int(xyxy[-1]))) or m2.contains_point(
                        (int(xyxy[-4]), int(xyxy[-1]))) or m2.contains_point((int(xyxy[-2]), int(xyxy[-1]))):
                    cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,
                                lineType=cv2.LINE_AA)
                    sign3 = 1
                    Time5 = time.time()
                    if it3 == 0:
                        it3 = time.time()
                        real_time_start1[0] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        # if pygame.mixer.music.get_busy():
                        #     pass
                        # else:
                        #     pygame.mixer.music.load('C:\yolov7\wav\sound.wav')
                        #     pygame.mixer.music.set_volume(0.5)
                        #     pygame.mixer.music.play()
                else:
                    if sign3 == 1:
                        Time6 = time.time()
                        if Time6 - Time5 > 5:
                            sign3 = 0
                            it3 = 0
                            real_time_start1[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            a = str(real_time_start1)
                            send('上井口单罐出车' + a + 'person')
                # -----------------------------------------------------------------------------------------------------------------------输出2.2
                # if (n2.contains_point((int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                if n2.contains_point((int(xyxy[-4]), int(xyxy[-3]))) or n2.contains_point(
                        (int(xyxy[-2]), int(xyxy[-1]))) or n2.contains_point(
                    (int(xyxy[-4]), int(xyxy[-1]))) or n2.contains_point((int(xyxy[-2]), int(xyxy[-1]))):
                    cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,
                                lineType=cv2.LINE_AA)
                    sign4 = 1
                    Time7 = time.time()
                    if it4 == 0:
                        it4 = time.time()
                        real_time_start2[0] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        # if pygame.mixer.music.get_busy():
                        #     pass
                        # else:
                        #     pygame.mixer.music.load('C:\yolov7\wav\sound.wav')
                        #     pygame.mixer.music.set_volume(0.5)
                        #     pygame.mixer.music.play()
                else:
                    if sign4 == 1:
                        Time8 = time.time()
                        if Time8 - Time7 > 5:
                            sign4 = 0
                            it4 = 0
                            real_time_start2[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            a = str(real_time_start2)
                            send('上井口单罐出车' + a + 'person', 777)
    else:
        sign3 = 0
        sign4 = 0
        it3 = 0
        it4 = 0
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, 'FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame


# 1.2-2.2上井口单罐出车
def image_inference3(frame, model1, device, half, imgsz, names, colors):
    t1 = time.time()
    global sign5, Time9, Time10, it5
    frame = cv2.line(frame, (109, 8), (108, 927), (0, 255, 0), 4)
    frame = cv2.line(frame, (108, 927), (1597, 927), (0, 255, 0), 4)
    frame = cv2.line(frame, (1597, 927), (1597, 9), (0, 255, 0), 4)
    frame = cv2.line(frame, (1597, 9), (109, 8), (0, 255, 0), 4)

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
        global coust, none2, t7, t8
        coust = 1
        if len(pred) and coust == 1:
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(pred):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
            for *xyxy, conf, cls in reversed(pred):
                # if (m3.contains_point(
                #         (int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                if m3.contains_point((int(xyxy[-4]), int(xyxy[-3]))) or m3.contains_point(
                        (int(xyxy[-2]), int(xyxy[-1]))) or m3.contains_point(
                    (int(xyxy[-4]), int(xyxy[-1]))) or m3.contains_point((int(xyxy[-2]), int(xyxy[-1]))):
                    cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,
                                lineType=cv2.LINE_AA)
                    sign5 = 1
                    Time9 = time.time()
                    if it5 == 0:
                        it5 = time.time()
                        real_time_start3[0] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        # if pygame.mixer.music.get_busy():
                        #     pass
                        # else:
                        #     pygame.mixer.music.load('C:\yolov7\wav\sound.wav')
                        #     pygame.mixer.music.set_volume(0.5)
                        #     pygame.mixer.music.play()
                else:
                    if sign5 == 1:
                        Time10 = time.time()
                        if Time10 - Time9 > 5:
                            sign5 = 0
                            it5 = 0
                            real_time_start3[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            a = str(real_time_start3)
                            send('上井口二平台双罐东' + a + 'person', 777)
    else:
        sign5 = 0
        it5 = 0
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, '1FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame


# 3上井口二平台双罐东

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


def image_inference_person(frame, model, device, imgsz, names, colors):
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


def send(data, port=6666):
    try:
        client = socket.socket()
        # 连接服务器
        addr = ('192.168.65.205', port)
        client.connect(addr)
        # 发送数据
        client.send(data.encode('utf-8'))
        # client.send(data)
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


print(111)
object = Carame_Accept_Object()
# 这里的顺序要和receive里面一致
client1 = create_client(8892)
print(222)
client2 = create_client(8893)
print(333)
client3 = create_client(8894)
client4 = create_client(8895)
client7 = create_client(8896)
client8 = create_client(8897)

port_list = [8892, 8893, 8894, 8895, 8896, 8897]
clients = [client1, client2, client3, client4, client7, client8]
metalocA = threading.Lock()

# url1='C:/yolov7/22/3.mp4'
# url2='C:/yolov7/22/5.mp4'
# url3='rtsp://admin:Aust12345@192.168.31.58:554/live'
# url3='rtsp://admin:Aust12345@192.168.31.69:554/live'
# url3='rtsp://admin:Aust12345@192.168.31.60:554/live'
# url3='C:/yolov7/22/10.mp4'
url1 = 'rtsp://admin:cs123456@192.168.4.246:554/live'
url2 = 'rtsp://admin:cs123456@192.168.4.248:554/live'
url3 = 'rtsp://admin:hy123456@192.168.4.137:554/live'
# url4='C:/yolov7/22/1.mp4'
# url7='C:/yolov7/22/2.mp4'
# url8='C:/yolov7/22/1.mp4'
url4 = 'rtsp://admin:hy123456@192.168.65.66:554/live'
url7 = 'rtsp://admin:hy123456@192.168.4.225:554/live'
url8 = 'rtsp://admin:hy123456@192.168.4.164:554/live'


# url1='22/3.mp4'
# url2='22/4.mp4'
# url3='22/5.mp4'
# url4='22/6.mp4'
# url7='22/7.mp4'
# url8='22/8.mp4'

# 疲劳检测

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

    def _send(self, data, port=6666):
        try:
            client = socket.socket()
            # 连接服务器
            addr = ('192.168.65.205', port)
            client.connect(addr)
            # print('send success')
            # 发送数据
            client.send(data.encode('utf-8'))
            # client.send(data)
            client.close()
        except:
            pass
    def run(self):
        global sign1, sign2, sign3, sign4, sign5, sign, sign_sleep
        while True:
            if sign1 == 1 or sign3 == 1:
                sign[0] = 1
            else:
                sign[0] = 0
            if (thread_ana1.videodie > 5) or (thread_read1.video_flag==False):
                sign[0] = 0
            if (thread_ana2.videodie > 5) or (thread_read2.video_flag==False):
                sign[0] = 0

            if sign2 == 1 or sign4 == 1:
                sign[1] = 1
            else:
                sign[1] = 0
            if (thread_ana1.videodie > 5) or (thread_read1.video_flag==False):
                sign[1] = 0
            if (thread_ana2.videodie > 5) or (thread_read2.video_flag==False):
                sign[1] = 0

            sign[2] = sign5
            if (thread_ana3.videodie > 5) or (thread_read3.video_flag==False):
                sign[2] = 0

            if (thread_ana7.videodie > 5) or (thread_read7.video_flag==False):
                sign_sleep[0] = 0

            if (thread_ana8.videodie > 5) or (thread_read8.video_flag==False):
                sign_sleep[1] = 0
            print([sign[0], sign[1], sign_sleep[0], sign_sleep[1]])
            self._send('master1;4;' + str([sign[0], sign[1], sign_sleep[0], sign_sleep[1]]))
            self._send('master;7;' + str([sign[2]]))
            # try:
            #     master.execute(slave=1, function_code=md.WRITE_MULTIPLE_REGISTERS, starting_address=4,
            #                    quantity_of_x=4, output_value=[sign[0],sign[1],sign_sleep[0],sign_sleep[1]])
            # except:
            #     print("time out")
            #     master = mt.TcpMaster("192.168.65.203", 8234)
            #     master.set_timeout(2.0)
            # try:
            #     master1.execute(slave=1, function_code=md.WRITE_MULTIPLE_REGISTERS, starting_address=7,
            #                         quantity_of_x=1, output_value=[sign[2]])
            # except:
            #     print("time out")
            #     master1 = mt.TcpMaster("192.168.65.204", 8234)
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
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_INTEL_MFX)
        self.que = Queue()
        self.video_flag=True

    def run(self):
        # while (self.cap.isOpened()):
        while True:
            self.ret, self.frame = self.cap.read()
            self.que.put(self.frame)
            if self.que.qsize() > 2:
                self.que.get()
            if self.ret == False:
                self.video_flag=False
                self.cap = cv2.VideoCapture(self.url)
                continue
            else:
                self.video_flag=True
            # print(self.ret)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # self.cap.release()
        cv2.destroyAllWindows()


thread_Kaiguan = Kaiguanliang(22, "USB_Kaiguan")
thread_read1 = Video_receive_thread(1, "video_read1", url1, 'frame1')
thread_read2 = Video_receive_thread(2, "video_read2", url2, 'frame2')
thread_read3 = Video_receive_thread(3, "video_read3", url3, 'frame3')
thread_read4 = Video_receive_thread(4, "video_read4", url4, 'frame4')
thread_read7 = Video_receive_thread(7, "video_read7", url7, 'frame7')
thread_read8 = Video_receive_thread(8, "video_read8", url8, 'frame8')


# thread_read9 = Video_receive_thread(9, "video_read8",url9,'frame9')
# thread_read10 = Video_receive_thread(10, "video_read8",url10,'frame10')
# -----------------------------------------------------------------

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
                self.videodie += 1
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
face_location2 = "打点硐室"


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
                self.frame_read, self.face_status = self.img_infer(self.thread_num.que.get()[0: 1080, 200: 950],
                                                                   self.model, device, imgsz1, names1, colors1)
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
                        if self.a != None:
                            send(self.a + ' ' + self.b + self.face_location + '睡岗')
                            self.a = None
                    self.sleep_flag = False
                    self.time1 = None
                    self.time2 = None
                if len(self.face_status) > 1:
                    if (self.face_status.count(1) == 2) or (4 in self.face_status):
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
                self.frame_read, self.face_status = self.img_infer(self.thread_num.que.get()[0: 1080, 0: 1000],
                                                                   self.model1, device, imgsz1,
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
                    self.f = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    if self.e != None:
                        send(self.e + ' ' + self.f + self.face_location + '睡岗')
                        self.e = None
                    self.sleep_flag = False
                    self.time1 = None
                    self.time2 = None

                if len(self.face_status) > 1:

                    if (self.face_status.count(1) == 2) or (4 in self.face_status):
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
                                if self.address == 14:
                                    sign_sleep[0] = 1
                                if self.address == 15:
                                    sign_sleep[1] = 1
                                # master.execute(slave=1, function_code=md.WRITE_MULTIPLE_REGISTERS, starting_address=self.address,
                                #                quantity_of_x=1, output_value=[1])
                    else:
                        self.sleep_flag1 = False
                        self.g = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        if self.i != None:
                            send(self.i + ' ' + self.g + self.face_location + '疲劳')
                            self.i = None
                        self.time3 = None
                        self.time4 = None
                        if self.address == 14:
                            sign_sleep[0] = 0
                        if self.address == 15:
                            sign_sleep[1] = 0
                else:
                    if self.address == 14:
                        sign_sleep[0] = 0
                    if self.address == 15:
                        sign_sleep[1] = 0
                        # master.execute(slave=1, function_code=md.WRITE_MULTIPLE_REGISTERS, starting_address=self.address,
                        #                quantity_of_x=1, output_value=[0])
                RT_Image(self.frame_read, self.client)
                self.videodie = 0
            except:
                # self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
                self.frame_read = np.random.rand(480, 640, 3) * 255
                print("Data Error", self.thread_num)
                self.videodie += 1
                self.client.close()
                index = clients.index(self.client)
                self.client = create_client(port=port_list[index])
                clients[index] = self.client
                time.sleep(1)
            # self.frame_read = cv2.resize(self.frame_read, (1920, 1080))
            # print(self.frame_read.shape)
            # cv2.imshow(self.frame_name, self.frame_read)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


# csv 写入报警数据信息
path = os.path.abspath(os.path.dirname(sys.argv[0]))
time1 = time.strftime('%Y%m%d', time.localtime(time.time()))

if __name__ == "__main__":
    thread_ana1 = Video_analyze_thread(12, "video_ana1", thread_read1, 'frame1', image_inference1, model, client1)
    thread_ana2 = Video_analyze_thread(13, "video_ana2", thread_read2, 'frame2', image_inference2, model, client2)
    thread_ana3 = Video_analyze_thread(14, "video_ana3", thread_read3, 'frame3', image_inference3, model, client3)
    # ----------------------------------------------------------------------------------------------------------
    thread_ana4 = Video_analyze_thread1(15, "video_ana4", thread_read4, 'frame4', image_inference, model1,
                                        face_location1, client4)
    thread_ana7 = Video_analyze_thread2(16, "video_ana7", thread_read7, 'frame7', image_inference, model1, 14,
                                        face_location2, client7)
    thread_ana8 = Video_analyze_thread2(17, "video_ana8", thread_read8, 'frame8', image_inference, model1, 15,
                                        face_location2, client8)

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
