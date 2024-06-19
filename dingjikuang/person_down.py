# -*- coding: utf-8 -*-
# from __init__ import *
# from step_defss.scenario_steps import *
#接后续代码
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
master=mt.TcpMaster("192.168.65.206",8234)
master1=mt.TcpMaster("192.168.65.204",8234)
#设置响应等待时间
master.set_timeout(2.0)
master1.set_timeout(2.0)
metalocA = threading.Lock()

device = select_device('0')
half = device.type != 'cpu'  # half precision only supported on CUDA

weights='best456.pt'
model = attempt_load(weights, map_location=device)  # 人员检测模型加载
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]



def send(data, port=777):
    try:
        client = socket.socket()
        # 连接服务器
        addr = ('192.168.65.205', port)
        # addr = ('192.168.31.180', port)
        client.connect(addr)
        # 发送数据
        client.send(data.encode('utf-8'))
        client.close()
    except:
        pass

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))

if half:
    model.half()
pygame.mixer.init()

video_weight=1920
video_hight=1080

#输出信号设置
sign=[0,0,0,0,0,0]
sign1=0
sign2=0
sign3=0
sign4=0
sign5=0
sign6=0
sign7=0
sign8=0
sign9=0
sign10=0

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
Time13 = 0
Time14 = 0
Time15 = 0
Time16 = 0
Time17 = 0
Time18 = 0
Time19 = 0
Time20 = 0

it1 = 0
it2 = 0
it3 = 0
it4 = 0
it5 = 0
it6 = 0
it7 = 0
it8 = 0
it9 = 0
it10 = 0


#变量设置
coust=0
none =0
have =0
none1=0
have1=0
none2=0
have2=0
none3=0
have3=0
none4=0
have4=0
none5=0
have5=0
none6=0
have6=0
none7=0
have7=0

real_time_start1 = [0,0]
real_time_start2 = [0,0]
real_time_start3 = [0,0]
real_time_start4 = [0,0]
real_time_start7 = [0,0]
real_time_start8 = [0,0]

#电子围栏设置
m1 = Path([(555 ,57), (553 ,545),(1033 ,546),(1030, 59)])
# 5下井口出车
m2 = Path([(228, 7), (227 ,690),(1585 ,691),(1587, 9)])
# 6下井口二平台双罐
m3 = Path([(869 , 110), (871 , 543),(1332 , 545),(1331 , 111)])
n3 = Path([(754 , 613), (498 , 1061),(1640 , 1061),(1439 , 668)])
# 9.1-10.1下井口双罐进车
m4 = Path([(756 , 78), (756 , 442),(1435 , 441),(1435 , 78)])
n4 = Path([(698 , 555), (111 , 1060),(1593 , 1060),(1504 , 551)])
# 9.2-10.2下井口双罐出车警戒
m7 = Path([(1173 , 94), (1173 , 493),(1490 , 492),(1490 , 92)])
n7 = Path([(1112 , 579), (536 , 1043),(1457 , 1037),(1499 , 607)])
# 11.1-12.1下井口单罐进车
m8 = Path([(660 , 163), (712 , 595),(1212 , 530),(1154 , 106)])
n8 = Path([(741 , 718), (740 , 1059),(1469 , 1063),(1241 , 634)])
# 11.2-12.2下井口单罐出车

def image_inference1(frame,model1,device,half,imgsz,names,colors):
    t1 = time.time()
    global sign1,Time1,Time2,Time3,Time4,it1
    frame = cv2.line(frame, (555, 57), (553, 545), (0, 255, 0), 4)
    frame = cv2.line(frame, (553, 545), (1033, 546), (0, 255, 0), 4)
    frame = cv2.line(frame, (1033, 546), (1030, 59), (0, 255, 0), 4)
    frame = cv2.line(frame, (1030, 59), (555, 57), (0, 255, 0), 4)
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
        global coust, none,t3,t4
        coust=1
        if len(pred) and coust == 1:
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(pred):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
            for *xyxy, conf, cls in reversed(pred):
#-----------------------------------------------------------------------------------------------------------------------输出1.1
                if m1.contains_point((int(xyxy[-4]), int(xyxy[-3]))) or m1.contains_point((int(xyxy[-2]), int(xyxy[-1])))or\
                        m1.contains_point((int(xyxy[-4]), int(xyxy[-1])))or\
                        m1.contains_point((int(xyxy[-2]), int(xyxy[-1]))):
                    cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,lineType=cv2.LINE_AA)
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
                            a=str(real_time_start1)
                            send('上井口单罐进车'+a+'person',666)
    else:
        sign1=0
        it1=0
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, '1FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame
# 5下井口出车
def image_inference2(frame,model1,device,half,imgsz,names,colors):
    t1 = time.time()
    global sign2,Time5,Time6,Time7,Time8,it2
    frame = cv2.line(frame, (228, 7), (227, 690), (0, 255, 0), 4)
    frame = cv2.line(frame, (227, 690), (1585, 691), (0, 255, 0), 4)
    frame = cv2.line(frame, (1585, 691), (1587, 9), (0, 255, 0), 4)
    frame = cv2.line(frame, (1587, 9), (228, 7), (0, 255, 0), 4)

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
        global coust, none1,t5,t6
        coust = 1
        if len(pred) and coust==1:
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(pred):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
            for *xyxy, conf, cls in reversed(pred):
# -----------------------------------------------------------------------------------------------------------------------输出1.2
#                 if (m2.contains_point((int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                if m2.contains_point((int(xyxy[-4]), int(xyxy[-3]))) or m2.contains_point((int(xyxy[-2]), int(xyxy[-1])))or \
                        m2.contains_point((int(xyxy[-4]), int(xyxy[-1])))or\
                        m2.contains_point((int(xyxy[-2]), int(xyxy[-1]))):
                    cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,lineType=cv2.LINE_AA)
                    sign2 = 1
                    Time5 = time.time()
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
                        Time6 = time.time()
                        if Time6 - Time5 > 5:
                            sign2 = 0
                            it2 = 0
                            real_time_start2[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            a = str(real_time_start2)
                            send('上井口单罐出车'+a+'person', 666)
    else:
        sign2 = 0
        it2=0
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, 'FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame
# 6下井口二平台双罐
def image_inference3(frame,model1,device,half,imgsz,names,colors):
    t1 = time.time()
    global sign3,sign4, Time9, Time10,it3,it4
    frame = cv2.line(frame, (869, 110), (871, 543), (0, 255, 0), 4)
    frame = cv2.line(frame, (871, 543), (1332, 545), (0, 255, 0), 4)
    frame = cv2.line(frame, (1332, 545), (1331, 111), (0, 255, 0), 4)
    frame = cv2.line(frame, (1331, 111), (869, 110), (0, 255, 0), 4)

    frame = cv2.line(frame, (754, 613), (498, 1061), (0, 255, 0), 4)
    frame = cv2.line(frame, (498, 1061), (1640, 1061), (0, 255, 0), 4)
    frame = cv2.line(frame, (1640, 1061), (1439, 668), (0, 255, 0), 4)
    frame = cv2.line(frame, (1439, 668), (754, 613), (0, 255, 0), 4)

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
        global coust, none2,t7,t8
        coust=1
        if len(pred) and coust == 1:
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(pred):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
            for *xyxy, conf, cls in reversed(pred):
                # if (m3.contains_point(
                #         (int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                if m3.contains_point((int(xyxy[-4]), int(xyxy[-3]))) or m3.contains_point((int(xyxy[-2]), int(
                        xyxy[-1]))) or m3.contains_point((int(xyxy[-4]), int(xyxy[-1]))) or \
                        m3.contains_point((int(xyxy[-2]), int(xyxy[-1]))):
                    cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,lineType=cv2.LINE_AA)
                    sign3 = 1
                    Time9 = time.time()
                    if it3 == 0:
                        it3 = time.time()
                        real_time_start3[0] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        # if pygame.mixer.music.get_busy():
                        #     pass
                        # else:
                        #     pygame.mixer.music.load('C:\yolov7\wav\sound.wav')
                        #     pygame.mixer.music.set_volume(0.5)
                        #     pygame.mixer.music.play()
                else:
                    if sign3 == 1:
                        Time10 = time.time()
                        if Time10 - Time9 > 5:
                            sign3 = 0
                            it3 = 0
                            real_time_start3[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            a = str(real_time_start3)
                            send('上井口二平台双罐东'+a+'person', 666)
                # if (n3.contains_point(
                #         (int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                if n3.contains_point((int(xyxy[-4]), int(xyxy[-3]))) or n3.contains_point((int(xyxy[-2]), int(
                            xyxy[-1]))) or n3.contains_point((int(xyxy[-4]), int(xyxy[-1]))) or \
                            n3.contains_point((int(xyxy[-2]), int(xyxy[-1]))):
                    cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,lineType=cv2.LINE_AA)
                    sign4 = 1
                    Time9 = time.time()
                    if it4 == 0:
                        it4 = time.time()
                        real_time_start4[0] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        # if pygame.mixer.music.get_busy():
                        #     pass
                        # else:
                        #     pygame.mixer.music.load('C:\yolov7\wav\sound.wav')
                        #     pygame.mixer.music.set_volume(0.5)
                        #     pygame.mixer.music.play()
                else:
                    if sign4 == 1:
                        Time10 = time.time()
                        if Time10 - Time9 > 5:
                            sign4 = 0
                            it4 = 0
                            real_time_start4[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            a = str(real_time_start4)
                            send('上井口二平台双罐东'+a+'person', 666)
    else:
        sign3 = 0
        sign4 = 0
        it3=0
        it4=0
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, '1FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame
# 9.1-10.1下井口双罐进车
def image_inference4(frame,model1,device,half,imgsz,names,colors):
    t1 = time.time()
    global sign5,sign6, Time11, Time12,it5,it6
    # 截取第一个区域
    frame = cv2.line(frame, (756, 78), (756, 442), (0, 255, 0), 4)
    frame = cv2.line(frame, (756, 442), (1435, 441), (0, 255, 0), 4)
    frame = cv2.line(frame, (1435, 441), (1435, 78), (0, 255, 0), 4)
    frame = cv2.line(frame, (1435, 78), (756, 78), (0, 255, 0), 4)

    frame = cv2.line(frame, (698, 555), (111, 1060), (0, 255, 0), 4)
    frame = cv2.line(frame, (111, 1060), (1593, 1060), (0, 255, 0), 4)
    frame = cv2.line(frame, (1593, 1060), (1504, 551), (0, 255, 0), 4)
    frame = cv2.line(frame, (1504, 551), (698, 555), (0, 255, 0), 4)
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
        pred = non_max_suppression(pred, 0.6, 0.45, classes=None, agnostic=False)[0]
        path = os.path.abspath(os.path.dirname(sys.argv[0]))
        time1 = time.strftime('%Y%m%d', time.localtime(time.time()))

    if len(pred):
        global coust, none3,t9,t10
        coust=1
        if len(pred) and coust == 1:
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(pred):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
            for *xyxy, conf, cls in reversed(pred):
                # if (m4.contains_point(
                #         (int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                if m4.contains_point((int(xyxy[-4]), int(xyxy[-3]))) or m4.contains_point((int(xyxy[-2]), int(
                        xyxy[-1]))) or m4.contains_point((int(xyxy[-4]), int(xyxy[-1]))) or \
                        m4.contains_point((int(xyxy[-2]), int(xyxy[-1]))):
                    cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,lineType=cv2.LINE_AA)
                    sign5 = 1
                    Time11 = time.time()
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
                        Time12 = time.time()
                        if Time12 - Time11 > 5:
                            sign5 = 0
                            it5 = 0
                            real_time_start3[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            a = str(real_time_start3)
                            send('上井口'+a+'person', 666)
                # if (n4.contains_point(
                #         (int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                if n4.contains_point((int(xyxy[-4]), int(xyxy[-3]))) or n4.contains_point((int(xyxy[-2]), int(
                            xyxy[-1]))) or n4.contains_point((int(xyxy[-4]), int(xyxy[-1]))) or \
                            n4.contains_point((int(xyxy[-2]), int(xyxy[-1]))):
                    cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,lineType=cv2.LINE_AA)
                    sign6 = 1
                    Time11 = time.time()
                    if it6 == 0:
                        it6 = time.time()
                        real_time_start4[0] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        # if pygame.mixer.music.get_busy():
                        #     pass
                        # else:
                        #     pygame.mixer.music.load('C:\yolov7\wav\sound.wav')
                        #     pygame.mixer.music.set_volume(0.5)
                        #     pygame.mixer.music.play()
                else:
                    if sign6 == 1:
                        Time12 = time.time()
                        if Time12 - Time11 > 5:
                            sign6 = 0
                            it6 = 0
                            real_time_start4[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            a = str(real_time_start4)
                            send('上井口'+a+'person', 666)
    else:
        sign5 = 0
        sign6 = 0
        it5=0
        it6=0
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    # print(6)
    cv2.putText(frame, 'FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    # print(7)
    return frame
# 9.2-10.2下井口双罐出车警戒
def image_inference7(frame,model1,device,half,imgsz,names,colors):
    t1 = time.time()
    global sign7,sign8, Time13, Time14, Time15, Time16,it7,it8

    frame = cv2.line(frame, (1173, 94), (1173, 493), (0, 255, 0), 4)
    frame = cv2.line(frame, (1173, 493), (1490, 492), (0, 255, 0), 4)
    frame = cv2.line(frame, (1490, 492), (1490, 92), (0, 255, 0), 4)
    frame = cv2.line(frame, (1490, 92), (1173, 94), (0, 255, 0), 4)

    frame = cv2.line(frame, (1112, 579), (536, 1043), (0, 255, 0), 4)
    frame = cv2.line(frame, (536, 1043), (1457, 1037), (0, 255, 0), 4)
    frame = cv2.line(frame, (1457, 1037), (1499, 607), (0, 255, 0), 4)
    frame = cv2.line(frame, (1499, 607), (1112, 579), (0, 255, 0), 4)
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
                # if (m7.contains_point(
                #         (int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                if m7.contains_point((int(xyxy[-4]), int(xyxy[-3]))) or m7.contains_point((int(xyxy[-2]), int(
                        xyxy[-1]))) or m7.contains_point((int(xyxy[-4]), int(xyxy[-1]))) or \
                        m7.contains_point((int(xyxy[-2]), int(xyxy[-1]))):
                    cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,lineType=cv2.LINE_AA)
                    sign7 = 1
                    Time13 = time.time()
                    if it7 == 0:
                        it7 = time.time()
                        real_time_start7[0] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        # if pygame.mixer.music.get_busy():
                        #     pass
                        # else:
                        #     pygame.mixer.music.load('C:\yolov7\wav\sound.wav')
                        #     pygame.mixer.music.set_volume(0.5)
                        #     pygame.mixer.music.play()
                else:
                    if sign7 == 1:
                        Time14 = time.time()
                        if Time14 - Time13 > 5:
                            sign7 = 0
                            it7 = 0
                            real_time_start7[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            a=str(real_time_start7)
                            send('双罐进车上井口'+a+'person',666)
#--------------------------------------------------------------------------------------------------------------------
                # if (n7.contains_point(
                #         (int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                if n7.contains_point((int(xyxy[-4]), int(xyxy[-3]))) or n7.contains_point((int(xyxy[-2]), int(
                            xyxy[-1]))) or n7.contains_point((int(xyxy[-4]), int(xyxy[-1]))) or \
                            n7.contains_point((int(xyxy[-2]), int(xyxy[-1]))):
                    cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,lineType=cv2.LINE_AA)
                    sign8 = 1
                    Time15 = time.time()
                    if it8 == 0:
                        it8 = time.time()
                        real_time_start8[0] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        # if pygame.mixer.music.get_busy():
                        #     pass
                        # else:
                        #     pygame.mixer.music.load('C:\yolov7\wav\sound.wav')
                        #     pygame.mixer.music.set_volume(0.5)
                        #     pygame.mixer.music.play()
                else:
                    if sign8 == 1:
                        Time16 = time.time()
                        if Time16 - Time15 > 5:
                            sign8 = 0
                            it8 = 0
                            real_time_start8[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            a = str(real_time_start7)
                            send('双罐进车上井口'+a+'person', 666)
#----------------------------------------------------------------------------------------------------------------------
    else:
        sign7 = 0
        sign8 = 0
        it7=0
        it8=0
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, 'FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame
# 11.1-12.1下井口单罐进车
def image_inference8(frame,model1,device,half,imgsz,names,colors):
    t1 = time.time()
    global sign9,sign10,Time17, Time18, Time19, Time20,it9,it10
    frame = cv2.line(frame, (660, 163), (712, 595), (0, 255, 0), 4)
    frame = cv2.line(frame, (712, 595), (1212, 530), (0, 255, 0), 4)
    frame = cv2.line(frame, (1212, 530), (1154, 106), (0, 255, 0), 4)
    frame = cv2.line(frame, (1154, 106), (660, 163), (0, 255, 0), 4)

    frame = cv2.line(frame, (741, 718), (740, 1059), (0, 255, 0), 4)
    frame = cv2.line(frame, (740, 1059), (1469, 1063), (0, 255, 0), 4)
    frame = cv2.line(frame, (1469, 1063), (1241, 634), (0, 255, 0), 4)
    frame = cv2.line(frame, (1241, 634), (741, 718), (0, 255, 0), 4)

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
        pred = non_max_suppression(pred, 0.6, 0.45, classes=None, agnostic=False)[0]

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
                # if (m8.contains_point((int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                if m8.contains_point((int(xyxy[-4]), int(xyxy[-3]))) or m8.contains_point((int(xyxy[-2]), int(
                        xyxy[-1]))) or m8.contains_point((int(xyxy[-4]), int(xyxy[-1]))) or \
                        m8.contains_point((int(xyxy[-2]), int(xyxy[-1]))):
                    cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,lineType=cv2.LINE_AA)
                    sign9 = 1
                    Time17 = time.time()
                    if it9 == 0:
                        it9 = time.time()
                        real_time_start7[0] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        # if pygame.mixer.music.get_busy():
                        #     pass
                        # else:
                        #     pygame.mixer.music.load('C:\yolov7\wav\sound.wav')
                        #     pygame.mixer.music.set_volume(0.5)
                        #     pygame.mixer.music.play()
                else:
                    if sign9 == 1:
                        Time18 = time.time()
                        if Time18 - Time17 > 5:
                            sign9 = 0
                            it9 = 0
                            real_time_start7[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            a=str(real_time_start7)
                            send('双罐出车上井口'+a+'person',666)
#-----------------------------------------------------------------------------------------------------------------------
                # if (n8.contains_point((int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                if n8.contains_point((int(xyxy[-4]), int(xyxy[-3]))) or n8.contains_point((int(xyxy[-2]), int(
                            xyxy[-1]))) or n8.contains_point((int(xyxy[-4]), int(xyxy[-1]))) or \
                            n8.contains_point((int(xyxy[-2]), int(xyxy[-1]))):
                    cv2.putText(frame, 'person', (1520, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3,lineType=cv2.LINE_AA)
                    sign10 = 1
                    Time19 = time.time()
                    if it10 == 0:
                        it10 = time.time()
                        real_time_start8[0] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        # if pygame.mixer.music.get_busy():
                        #     pass
                        # else:
                        #     pygame.mixer.music.load('C:\yolov7\wav\sound.wav')
                        #     pygame.mixer.music.set_volume(0.5)
                        #     pygame.mixer.music.play()
                else:
                    if sign10 == 1:
                        Time20 = time.time()
                        if Time20 - Time19 > 5:
                            sign10 = 0
                            it10 = 0
                            real_time_start8[1] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            a = str(real_time_start8)
                            send('双罐出车上井口'+a + 'person', 666)
#-----------------------------------------------------------------------------------------------------------------------
    else:
        sign9 = 0
        sign10 = 0
        it9=0
        it10=0
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, 'FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    # print(frame.shape)
    return frame
# 11.2-12.2下井口单罐出车



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
    # time.sleep(0.1)  # 推迟线程运行0.1s
    # _, object.img = camera.read()  # 读取视频每一帧
    # 核心代码在这里，图像从这里传过来，
    object.img = cv2.resize(img, object.resolution)  # 按要求调整图像大小(resolution必须为元组)
    _, img_encode = cv2.imencode('.jpg', object.img, img_param)  # 按格式生成图片
    img_code = numpy.array(img_encode)  # 转换成矩阵
    object.img_data = img_code.tobytes()  # 生成相应的字符串
    # 按照相应的格式进行打包发送图片
    # s1 = time.time()
    client.send(
        struct.pack("lhh", len(object.img_data), object.resolution[0], object.resolution[1]) + object.img_data)
    # print(time.time()-s1)

def create_client(port=8880):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 端口可复用
    server.bind(("", port))
    server.listen(5)
    client, D_addr = server.accept()
    return client

print('1111')
object = Carame_Accept_Object()
# 这里的顺序要和receive里面一致
client1 = create_client(8886)
client2 = create_client(8887)
client3 = create_client(8888)
client4 = create_client(8889)
client7 = create_client(8890)
client8 = create_client(8891)
port_list = [8886, 8887,8888,8889,8890,8891]
clients = [client1, client2,client3, client4, client7, client8]
metalocA = threading.Lock()

print('2222')


url1='rtsp://admin:admin123@192.168.4.200:554/live'
url2='rtsp://admin:hy123456@192.168.4.138:554/live'
url3='rtsp://admin:hy123456@192.168.4.196:554/live'
# url1='C:/yolov7/22/11.mp4'
# url2='C:/yolov7/22/14.mp4'
# url3='C:/yolov7/22/7.mp4'
url4='rtsp://admin:hy123456@192.168.4.198:554/live'
url7='rtsp://admin:hy123456@192.168.4.197:554/live'
url8='rtsp://admin:hy123456@192.168.4.199:554/live'
# url4='C:/yolov7/22/12.mp4'
# url7='C:/yolov7/22/8.mp4'
# url8='C:/yolov7/22/13.mp4'
# url1 = '3_1.mp4'
# url2 = '4_1.mp4'
# url3 = '4_2.mp4'
# url4 = '5_1.mp4'
# url7 = '6_1.mp4'
# url8 = '7_1.mp4'
# url9='rtsp://admin:Aust12345@192.168.31.61:554/live'
# url10='rtsp://admin:Aust12345@192.168.31.62:554/live'
# url11='rtsp://admin:Aust12345@192.168.31.63:554/live'
#人员检测

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

    def _send(self, data, port=777):
        try:
            client = socket.socket()
            # 连接服务器
            addr = ('192.168.65.205', port)
            # addr = ('192.168.31.180', port)
            client.connect(addr)
            # 发送数据
            client.send(data.encode('utf-8'))
            client.close()
        except:
            pass
    def run(self):
        global sign1,sign2,sign3,sign4,sign5,sign6,sign7,sign8,sign9,sign10,sign
        while True:
            sign[0] = sign1
            if (thread_ana1.videodie > 5) or (thread_read1.video_flag==False):
                sign[0]=0

            sign[1] = sign2
            if (thread_ana2.videodie > 5) or (thread_read2.video_flag==False):
                sign[1]=0

            if sign3==1 or sign5==1:
                sign[2] = 1
            else:
                sign[2] = 0
            if (thread_ana3.videodie > 5) or (thread_read3.video_flag==False):
                sign[2]=0
            if (thread_ana4.videodie > 5) or (thread_read4.video_flag==False):
                sign[2]=0

            if sign4==1 or sign6==1:
                sign[3] = 1
            else:
                sign[3] = 0
            if (thread_ana3.videodie > 5) or (thread_read3.video_flag==False):
                sign[3]=0
            if (thread_ana4.videodie > 5) or (thread_read4.video_flag==False):
                sign[3]=0

            if sign7==1 or sign9==1:
                sign[4] = 1
            else:
                sign[4] = 0
            if (thread_ana7.videodie > 5) or (thread_read7.video_flag==False):
                sign[4]=0
            if (thread_ana8.videodie > 5) or (thread_read8.video_flag==False):
                sign[4] = 0

            if sign8==1 or sign10==1:
                sign[5] = 1
            else:
                sign[5] = 0
            if (thread_ana7.videodie > 5) or (thread_read7.video_flag==False):
                sign[5]=0
            if (thread_ana8.videodie > 5) or (thread_read8.video_flag==False):
                sign[5] = 0
            print(sign)
            self._send('master;0;'+str([sign[1], sign[2],sign[3]]))
            self._send('master1;0;'+str([sign[0], sign[4], sign[5]]))
            # try:
            #     master1.execute(slave=1, function_code=md.WRITE_MULTIPLE_REGISTERS, starting_address=0,
            #                     quantity_of_x=3, output_value=[sign[1], sign[2],sign[3]])
            # except:
            #     master1 = mt.TcpMaster("192.168.65.204", 8234)
            #     master1.set_timeout(2.0)
            #     print('time out1')
            #
            # try:
            #     master.execute(slave=1, function_code=md.WRITE_MULTIPLE_REGISTERS, starting_address=0,
            #                    quantity_of_x=3, output_value=[sign[0], sign[4], sign[5]])
            # except:
            #     master = mt.TcpMaster("192.168.65.206", 8234)
            #     master.set_timeout(2.0)
            #     print('time out')
            time.sleep(3)

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
        self.que = Queue()
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

thread_read1 = Video_receive_thread(1, "video_read1",url1,'frame1')
thread_read2 = Video_receive_thread(2, "video_read2",url2,'frame2')
thread_read3 = Video_receive_thread(3, "video_read3",url3,'frame3')
thread_read4 = Video_receive_thread(4, "video_read4",url4,'frame4')
thread_read7 = Video_receive_thread(7, "video_read7",url7,'frame7')
thread_read8 = Video_receive_thread(8, "video_read8",url8,'frame8')
#-----------------------------------------------------------------
thread_Kaiguan = Kaiguanliang(21,"USB_Kaiguan")

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
        self.model = deepcopy(model)
        self.client = client
        self.videodie = 0


    def run(self):
        while True:
            # time.sleep(0.5)
            try:
                self.frame_read = self.img_infer(self.thread_num.que.get(), self.model, device, half, imgsz, names, colors)
                # self.frame_read = self.thread_num.frame
                # cv2.imshow(self.frame_name, self.frame_read)
                RT_Image(self.frame_read, self.client)
                # print(self.frame_name)
                self.videodie = 0
            except:
                # self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
                self.frame_read = np.random.rand(1080,1920,3)*255
                print("Data Error",self.thread_num)
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

# csv 写入报警数据信息
path = os.path.abspath(os.path.dirname(sys.argv[0]))
time1 = time.strftime('%Y%m%d', time.localtime(time.time()))

if __name__ == "__main__":
    thread_ana1 = Video_analyze_thread(9, "video_ana1",thread_read1,'frame1',image_inference1,model,client1)
    thread_ana2 = Video_analyze_thread(10, "video_ana2",thread_read2,'frame2',image_inference2,model,client2)
    thread_ana3 = Video_analyze_thread(11, "video_ana3", thread_read3, 'frame3',image_inference3,model,client3)
    thread_ana4 = Video_analyze_thread(12, "video_ana4", thread_read4, 'frame4',image_inference4,model,client4)
    thread_ana7 = Video_analyze_thread(15, "video_ana7", thread_read7, 'frame7',image_inference7,model,client7)
    thread_ana8 = Video_analyze_thread(16, "video_ana8", thread_read8, 'frame8',image_inference8,model,client8)
    #----------------------------------------------------------------------------------------------------------


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

