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
# master=mt.TcpMaster("192.168.31.99",8234)
#设置响应等待时间
# master.set_timeout(5.0)

metalocA = threading.Lock()

device = select_device('0')
half = device.type != 'cpu'  # half precision only supported on CUDA

weights='person.pt'
model = attempt_load(weights, map_location=device)  # 人员检测模型加载
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

weights1='face_detection.pt'
model1 = attempt_load(weights1, map_location=device)  # 疲劳检测模型加载
stride1 = int(model1.stride.max())  # model stride
imgsz1 = check_img_size(640, s=stride1)  # check img_size
names1 = model1.module.names if hasattr(model1, 'module1') else model1.names
colors1 = [[random.randint(0, 255) for _ in range(3)] for _ in names1]

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))

if half:
    model.half()
    model1.half()# to FP16
pygame.mixer.init()

video_weight=1920
video_hight=1080

#输出信号设置
sign=[0,0,0,0,0,0]
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

# s1 = serial.Serial('com14', 9600)
# s2 = serial.Serial('com14', 9600)
# s3 = serial.Serial('com14', 9600)
# s4 = serial.Serial('com14', 9600)
# s7 = serial.Serial('com14', 9600)
# s8 = serial.Serial('com14', 9600)

#电子围栏设置
m1 = Path([(788 , 9), (788 , 278),(1108 , 278),(1107 , 10)])
n1 = Path([(778 , 291), (448 , 1061),(1457 , 1061),(1136 , 295)])
# 1.1-2.1上井口单罐进车
m2 = Path([(640,9), (640,374),(1144,373),(1145,10)])
n2 = Path([(660 , 460), (482 , 1063),(1415 , 1063),(1218 , 460)])
# # 1.2-2.2上井口单罐出车
m3 = Path([(109, 8), (108 ,927),(1597 ,927),(1597, 9)])
# 3上井口二平台双罐东
m4 = Path([(524, 101), (524 ,912),(1445, 911),(1445,100)])
# 4上井口
m7 = Path([(730 ,6), (730, 268),(1134, 266),(1132 ,6)])
n7 = Path([(680 , 327), (159 , 1058),(1441 , 1061),(1167 , 326)])
# 7.1-8.1双罐进车上井口
m8 = Path([(876 ,5), (876 ,351),(1477, 350),(1477, 6)])
n8 = Path([(777 , 418), (357 , 1060),(1719 , 1061),(1545 , 455)])
# # 7.2-8.2双罐出车上井口

def image_inference1(frame,model1,device,half,imgsz,names,colors):
    t1 = time.time()
    global Time1,Time2,Time3,Time4
    frame = cv2.line(frame, (788 , 9), (788 , 278), (0, 255, 0), 4)
    frame = cv2.line(frame, (788 , 278), (1108 , 278), (0, 255, 0), 4)
    frame = cv2.line(frame, (1108 , 278), (1107 , 10), (0, 255, 0), 4)
    frame = cv2.line(frame, (1107 , 10), (788 , 9), (0, 255, 0), 4)

    frame = cv2.line(frame, (778 , 291), (448 , 1061), (0, 255, 0), 4)
    frame = cv2.line(frame, (448 , 1061), (1457 , 1061), (0, 255, 0), 4)
    frame = cv2.line(frame, (1457 , 1061), (1136 , 295), (0, 255, 0), 4)
    frame = cv2.line(frame, (1136 , 295), (778 , 291), (0, 255, 0), 4)
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
        if have==none and coust ==1:
            real_time_start1[0]=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            t3 = time.time()
            none=1
        if len(pred) and coust == 1:
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(pred):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
            for *xyxy, conf, cls in reversed(pred):
#-----------------------------------------------------------------------------------------------------------------------输出1.1
                if (m1.contains_point((int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                    cv2.putText(frame, 'person', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 0, 255], 2,lineType=cv2.LINE_AA)
                    sign[0] = 1
                    Time1 = time.time()
                else:
                    if sign[0] == 1:
                        Time2 = time.time()
                        if Time2 - Time1 > 10:
                            sign[0] = 0
#-----------------------------------------------------------------------------------------------------------------------输出2.1
                if (n1.contains_point((int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                    cv2.putText(frame, 'person', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 0, 255], 2,
                                lineType=cv2.LINE_AA)
                    sign[1] = 1
                    Time3 = time.time()
                else:
                    if sign[1] == 1:
                        Time4 = time.time()
                        if Time4 - Time3 > 10:
                            sign[1] = 0
            if pygame.mixer.music.get_busy():
                pass
            else:
                pygame.mixer.music.load('wav/1.wav')
                pygame.mixer.music.set_volume(0.5)
                pygame.mixer.music.play()
    else:
        if have!=none:
            real_time_start1[1]=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            t4=time.time()
            if(t4-t3>3):
                with open(path + '\\' + str(time1) + '-1.csv', 'a', encoding='utf-8') as cvs_File:
                    # cvs_File = open('time.csv', 'a', newline='')
                    writer = csv.writer(cvs_File)
                    writer.writerow((['1号监控区域异常开始时间'],['1号监控区域异常结束时间']))
                    writer.writerow(real_time_start1)
                    metalocA.acquire()
                    # try:
                    #     my.thread.getdata('1号监控区域异常开始时间,1号监控区域异常结束时间')
                    #     my.thread.getdata(str(real_time_start))
                    # except:
                    #     pass
                    metalocA.release()
                    # print('1',real_time_start1)
                    # my.write('1'+real_time_start)
                none = 0
                #print(3)
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, '1FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame
# 1.1-2.1上井口单罐进车
def image_inference2(frame,model1,device,half,imgsz,names,colors):
    t1 = time.time()
    global Time5,Time6,Time7,Time8
    frame = cv2.line(frame, (640, 9), (640, 374), (0, 255, 0), 4)
    frame = cv2.line(frame, (640, 374), (1144, 373), (0, 255, 0), 4)
    frame = cv2.line(frame, (1144, 373), (1145, 10), (0, 255, 0), 4)
    frame = cv2.line(frame, (1145, 10), (640, 9), (0, 255, 0), 4)

    frame = cv2.line(frame, (660 , 460), (482 , 1063), (0, 255, 0), 4)
    frame = cv2.line(frame, (482 , 1063), (1415 , 1063), (0, 255, 0), 4)
    frame = cv2.line(frame, (1415 , 1063), (1218 , 460), (0, 255, 0), 4)
    frame = cv2.line(frame, (1218 , 460), (660 , 460), (0, 255, 0), 4)

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
        if have1==none1 and coust ==1:
            real_time_start2[0]=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            t5 = time.time()
            none1=1
        if len(pred) and coust==1:
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(pred):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
            for *xyxy, conf, cls in reversed(pred):
# -----------------------------------------------------------------------------------------------------------------------输出1.2
                if (m2.contains_point((int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                    cv2.putText(frame, 'person', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 0, 255], 2,
                                lineType=cv2.LINE_AA)
                    sign[0] = 1
                    Time5 = time.time()
                else:
                    if sign[0] == 1:
                        Time6 = time.time()
                        if Time6 - Time5 > 10:
                            sign[0] = 0
#-----------------------------------------------------------------------------------------------------------------------输出2.2
                if (n2.contains_point((int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                    cv2.putText(frame, 'person', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 0, 255], 2,
                                lineType=cv2.LINE_AA)
                    sign[1] = 1
                    Time7 = time.time()
                else:
                    if sign[1] == 1:
                        Time8 = time.time()
                        if Time8 - Time7 > 10:
                            sign[1] = 0
            # frame = cv2AddChineseText(frame, "有人", (100, 100), (255, 0, 0), 100)
            if pygame.mixer.music.get_busy():
                pass
            else:
                pygame.mixer.music.load('wav/1.wav')
                pygame.mixer.music.set_volume(0.5)
                pygame.mixer.music.play()
    else:
        if have1!=none1:
            real_time_start2[1]=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            t6 = time.time()
            if(t6-t5>3):
                with open(path + '\\' + str(time1) + '-2.csv', 'a', encoding='utf-8') as cvs_File:
                    # cvs_File = open('time.csv', 'a', newline='')
                    writer = csv.writer(cvs_File)
                    writer.writerow((['2号监控区域异常开始时间'],['2号监控区域异常结束时间']))
                    writer.writerow(real_time_start2)
                    metalocA.acquire()

                    # try:
                    #     my.thread.getdata('2号监控区域异常开始时间,2号监控区域异常结束时间')
                    #     my.thread.getdata(str(real_time_start1))
                    # except:
                    #     pass
                    metalocA.release()
                    print('2',real_time_start2)
                    # my.write('2'+ real_time_start)
                none1 = 0
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, 'FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame
# 1.2-2.2上井口单罐出车
def image_inference3(frame,model1,device,half,imgsz,names,colors):
    t1 = time.time()
    global sign3, Time9, Time10
    frame = cv2.line(frame, (109, 8), (108 ,927), (0, 255, 0), 4)
    frame = cv2.line(frame, (108 ,927), (1597 ,927), (0, 255, 0), 4)
    frame = cv2.line(frame, (1597 ,927), (1597, 9), (0, 255, 0), 4)
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
        global coust, none2,t7,t8
        coust=1
        if have2==none2 and coust ==1:
            real_time_start3[0]=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            t7 = time.time()
            none2=1
        if len(pred) and coust == 1:
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(pred):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
            for *xyxy, conf, cls in reversed(pred):
                if (m3.contains_point(
                        (int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                    cv2.putText(frame, 'person', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 0, 255], 2,
                                lineType=cv2.LINE_AA)
                    sign[2] = 1
                    Time9 = time.time()
                else:
                    if sign[2] == 1:
                        Time10 = time.time()
                        if Time10 - Time9 > 10:
                            sign[2] = 0
            if pygame.mixer.music.get_busy():
                pass
            else:
                pygame.mixer.music.load('wav/1.wav')
                pygame.mixer.music.set_volume(0.5)
                pygame.mixer.music.play()
    else:
        if have2!=none2:
            real_time_start3[1]=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            t8=time.time()
            if(t8-t7>3):
                with open(path + '\\' + str(time1) + '-3.csv', 'a', encoding='utf-8') as cvs_File:
                    # cvs_File = open('time.csv', 'a', newline='')
                    writer = csv.writer(cvs_File)
                    writer.writerow((['3号监控区域异常开始时间'],['3号监控区域异常结束时间']))
                    writer.writerow(real_time_start3)
                    metalocA.acquire()
                    # try:
                    #     my.thread.getdata('3号监控区域异常开始时间,3号监控区域异常结束时间')
                    #     my.thread.getdata(str(real_time_start2))
                    # except:
                    #     pass
                    metalocA.release()
                    # print('3',real_time_start3)
                none2 = 0
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, '1FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame

# 3上井口二平台双罐东

def image_inference4(frame,model1,device,half,imgsz,names,colors):
    t1 = time.time()
    global sign4, Time11, Time12
    # 截取第一个区域
    frame = cv2.line(frame, (524, 101), (524 ,912), (0, 255, 0), 4)
    frame = cv2.line(frame, (524 ,912), (1445, 911), (0, 255, 0), 4)
    frame = cv2.line(frame, (1445, 911), (1445,100), (0, 255, 0), 4)
    frame = cv2.line(frame, (1445,100), (524, 101), (0, 255, 0), 4)
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
        global coust, none3,t9,t10
        coust=1
        if have3==none3 and coust ==1:
            real_time_start4[0]=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            t9 = time.time()
            none3=1
            # print(1)
        if len(pred) and coust == 1:
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(pred):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
            for *xyxy, conf, cls in reversed(pred):
                if (m4.contains_point(
                        (int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                    cv2.putText(frame, 'person', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 0, 255], 2,
                                lineType=cv2.LINE_AA)
                    sign[3] = 1
                    Time11 = time.time()
                else:
                    if sign[3] == 1:
                        Time12 = time.time()
                        if Time12 - Time11 > 10:
                            sign[3] = 0
            if pygame.mixer.music.get_busy():
                pass
            else:
                pygame.mixer.music.load('wav/1.wav')
                pygame.mixer.music.set_volume(0.5)
                pygame.mixer.music.play()
    else:
        #print(1)
        if have3!=none3:
            real_time_start4[1]=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            t10=time.time()
            if(t10-t9>3):
                with open(path + '\\' + str(time1) + '-4.csv', 'a', encoding='utf-8') as cvs_File:
                    # cvs_File = open('time.csv', 'a', newline='')
                    writer = csv.writer(cvs_File)
                    writer.writerow((['4号监控区域异常开始时间'],['4号监控区域异常结束时间']))
                    writer.writerow(real_time_start4)
                    metalocA.acquire()
                    # try:
                    #     my.thread.getdata('4号监控区域异常开始时间,4号监控区域异常结束时间')
                    #     my.thread.getdata(str(real_time_start3))
                    # except:
                    #     pass
                    metalocA.release()
                    print('4',real_time_start4)
                none3 = 0
                #print(3)
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    # print(6)
    cv2.putText(frame, 'FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    # print(7)
    return frame
# 4上井口
def image_inference7(frame,model1,device,half,imgsz,names,colors):
    t1 = time.time()
    global sign7,sign8, Time13, Time14, Time15, Time16

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
        if have6==none6 and coust ==1:
            real_time_start7[0]=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            t15 = time.time()
            none6=1
        if len(pred) and coust == 1:
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(pred):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
            for *xyxy, conf, cls in reversed(pred):
                if (m7.contains_point(
                        (int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                    cv2.putText(frame, 'person', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 0, 255], 2,
                                lineType=cv2.LINE_AA)
                    sign[4] = 1
                    Time13 = time.time()
                else:
                    if sign[4] == 1:
                        Time14 = time.time()
                        if Time14 - Time13 > 10:
                            sign[4] = 0
#--------------------------------------------------------------------------------------------------------------------
                if (n7.contains_point(
                        (int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                    cv2.putText(frame, 'person', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 0, 255], 2,
                                lineType=cv2.LINE_AA)
                    sign[5] = 1
                    Time15 = time.time()
                else:
                    if sign[5] == 1:
                        Time16 = time.time()
                        if Time16 - Time15 > 10:
                            sign[5] = 0
#----------------------------------------------------------------------------------------------------------------------
            if pygame.mixer.music.get_busy():
                pass
            else:
                pygame.mixer.music.load('wav/1.wav')
                pygame.mixer.music.set_volume(0.5)
                pygame.mixer.music.play()
    else:
        if have6!=none6:
            real_time_start7[1]=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            t16=time.time()
            if(t16-t15>3):
                with open(path + '\\' + str(time1) + '-7.csv', 'a', encoding='utf-8') as cvs_File:
                    # cvs_File = open('time.csv', 'a', newline='')
                    writer = csv.writer(cvs_File)
                    writer.writerow((['7号监控区域异常开始时间'],['7号监控区域异常结束时间']))
                    writer.writerow(real_time_start7)
                    metalocA.acquire()
                    # try:
                    #     my.thread.getdata('7号监控区域异常开始时间,7号监控区域异常结束时间')
                    #     my.thread.getdata(str(real_time_start6))
                    # except:
                    #     pass
                    metalocA.release()
                    print('7',real_time_start7)
                none6 = 0
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, 'FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame
# 7.1-8.1双罐进车上井口
def image_inference8(frame,model1,device,half,imgsz,names,colors):
    t1 = time.time()
    global Time17, Time18, Time19, Time20
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
        if have7==none7 and coust ==1:
            real_time_start8[0]=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            t17 = time.time()
            none7=1
        if len(pred) and coust == 1:
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(pred):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
            for *xyxy, conf, cls in reversed(pred):
                if (m8.contains_point((int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                    cv2.putText(frame, 'person', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 0, 255], 2,
                                lineType=cv2.LINE_AA)
                    sign[4] = 1
                    Time17 = time.time()
                else:
                    if sign[4] == 1:
                        Time18 = time.time()
                        if Time18 - Time17 > 10:
                            sign[4] = 0
#-----------------------------------------------------------------------------------------------------------------------
                if (n8.contains_point((int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                    cv2.putText(frame, 'person', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 0, 255], 2,
                                lineType=cv2.LINE_AA)
                    sign[5] = 1
                    Time19 = time.time()
                else:
                    if sign[5] == 1:
                        Time20 = time.time()
                        if Time20 - Time19 > 10:
                            sign[5] = 0
#-----------------------------------------------------------------------------------------------------------------------
            if pygame.mixer.music.get_busy():
                pass
            else:
                pygame.mixer.music.load('wav/1.wav')
                pygame.mixer.music.set_volume(0.5)
                pygame.mixer.music.play()
    else:
        frame = cv2AddChineseText(frame, "无人", (100, 100), (0, 255, 0), 100)
        if have7!=none7:
            real_time_start8[1]=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            t18=time.time()
            if(t18-t17>3):
                with open(path + '\\' + str(time1) + '-8.csv', 'a', encoding='utf-8') as cvs_File:
                    # cvs_File = open('time.csv', 'a', newline='')
                    writer = csv.writer(cvs_File)
                    writer.writerow((['8号监控区域异常开始时间'],['8号监控区域异常结束时间']))
                    writer.writerow(real_time_start8)
                    # try:
                    #     my.thread.getdata('8号监控区域异常开始时间,8号监控区域异常结束时间')
                    #     my.thread.getdata(str(real_time_start7))
                    # except:
                    #     pass
                    # print('8', real_time_start7)
                none7 = 0
                #print(3)
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, 'FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    # print(frame.shape)
    return frame
# 7.2-8.2双罐出车上井口
def image_inference_face1(frame,model,device,imgsz,names,colors):
    t1 = time.time()
    face_status=[]
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
        # Rescale boxes from img_size to im0 size
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
        # Write results
        for *xyxy, conf, cls in reversed(pred):
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
            face_status.append(int(cls))
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, 'FPS=%s' % FPS, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame,face_status
# 第一路疲劳检测推理
def image_inference_face2(frame,model,device,imgsz,names,colors):
    t1 = time.time()
    face_status=[]
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
        # Rescale boxes from img_size to im0 size
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
        # Write results
        for *xyxy, conf, cls in reversed(pred):
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
            face_status.append(int(cls))
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, 'FPS=%s' % FPS, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame,face_status
# 第二路疲劳检测推理
def image_inference_face3(frame,model,device,imgsz,names,colors):
    t1 = time.time()
    face_status=[]
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
        # Rescale boxes from img_size to im0 size
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
        # Write results
        for *xyxy, conf, cls in reversed(pred):
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
            face_status.append(int(cls))
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, 'FPS=%s' % FPS, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame,face_status
# 第三路疲劳检测推理



def send(data, port=6666):
    client = socket.socket()
    # 连接服务器
    addr = ('192.168.10.13', port)
    client.connect(addr)
    # 发送数据
    client.send(data.encode('utf-8'))
    client.close()

class Carame_Accept_Object:
    def __init__(self):
        self.resolution = (640, 480)  # 分辨率
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
    object.img_data = img_code.tostring()  # 生成相应的字符串
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
# client1 = create_client(8880)
# client2 = create_client(8881)
# client3 = create_client(8882)
# client4 = create_client(8883)
# client7 = create_client(8884)
# client8 = create_client(8885)
# port_list = [8880, 8881,8882,8883,8884,8885]
# clients = [client1, client2,client3, client4, client7, client8]
metalocA = threading.Lock()

url1='22/3.mp4'
url2='22/4.mp4'
url3='22/5.mp4'
url4='22/6.mp4'
url7='22/7.mp4'
url8='22/8.mp4'
#人员检测
url9="22/1.mp4"
url10="22/2.mp4"
url11="22/1.mp4"
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

    def run(self):
        while True:
            # master.execute(slave=1, function_code=md.WRITE_MULTIPLE_REGISTERS, starting_address=0,
            #                     quantity_of_x=5, output_value=sign)
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

thread_read1 = Video_receive_thread(1, "video_read1",url1,'frame1')
thread_read2 = Video_receive_thread(2, "video_read2",url2,'frame2')
thread_read3 = Video_receive_thread(3, "video_read3",url3,'frame3')
thread_read4 = Video_receive_thread(4, "video_read4",url4,'frame4')
thread_read7 = Video_receive_thread(7, "video_read7",url7,'frame7')
thread_read8 = Video_receive_thread(8, "video_read8",url8,'frame8')
#-----------------------------------------------------------------
thread_read9 = Video_receive_thread(9, "video_read9",url9,'frame9')
thread_read10 = Video_receive_thread(10, "video_read10",url10,'frame10')
thread_read11 = Video_receive_thread(11, "video_read11",url11,'frame11')
thread_Kaiguan = Kaiguanliang(1,"USB_Kaiguan")

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

    def run(self):
        while True:
            try:
                self.frame_read = self.img_infer(self.thread_num.frame, self.model, device, half, imgsz, names, colors)
                # RT_Image(self.frame_read, self.client)
            except:
                # self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
                self.frame_read = np.random.rand(1080,1920,3)*255
                print("Data Error",self.thread_num)
                # self.client.close()
                # index = clients.index(self.client)
                # self.client = create_client(port=port_list[index])
                # clients[index] = self.client
            # self.frame_read=cv2.resize(self.frame_read,(1920,1080))
            # print(self.frame_read.shape)
            # cv2.namedWindow(self.frame_name, cv2.WINDOW_NORMAL)
            # cv2.imshow(self.frame_name, self.frame_read)
            print(self.frame_name)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

class Video_face_analyze_thread(threading.Thread):
    def __init__(self, threadID, name,thread_num,frame_name,img_infer,model1):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        # self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
        self.frame_read=None
        self.thread_num=thread_num
        self.frame_name=frame_name
        self.img_infer = img_infer
        self.model1=model1

        self.face_status=[]
        #睡岗计时器
        self.time1=None
        self.time2=None
        #疲劳计时器
        self.time3=None
        self.time4=None
    def run(self):
        while True:
            try:
                self.frame_read, self.face_status = self.img_infer(self.thread_num.frame, self.model1, device, imgsz1,
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
                            print("睡岗")
                else:
                    self.time1 = None
                    self.time2 = None
                if len(self.face_status) > 1:
                    if (1 in self.face_status) or (4 in self.face_status):
                        if self.time3 == None:
                            self.time3 = time.time()
                        else:
                            self.time4 = time.time()
                            time_tied = self.time4 - self.time3
                            if time_tied >= 5 and time_tied < 30:
                                print("一级")
                                self.frame_read = cv2AddChineseText(self.frame_read, "一级", (100, 100), (255, 0, 0),
                                                                    100)
                            if time_tied >= 30 and time_tied < 45:
                                print("二级")
                            if time_tied >= 45 and time_tied < 60:
                                print("三级")
                            if time_tied >= 60:
                                print("四级")
                    else:
                        self.time3 = None
                        self.time4 = None
            except:
                # self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
                self.frame_read = np.random.rand(1080,1920,3)*255
                print("Data Error",self.thread_num)
            # self.frame_read=cv2.resize(self.frame_read,(1920,1080))
            print(self.frame_name)
            # cv2.imshow(self.frame_name, self.frame_read)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
# csv 写入报警数据信息
path = os.path.abspath(os.path.dirname(sys.argv[0]))
time1 = time.strftime('%Y%m%d', time.localtime(time.time()))

if __name__ == "__main__":
    thread_ana1 = Video_analyze_thread(9, "video_ana1",thread_read1,'frame1',image_inference1,model)
    thread_ana2 = Video_analyze_thread(10, "video_ana2",thread_read2,'frame2',image_inference2,model)
    thread_ana3 = Video_analyze_thread(11, "video_ana3", thread_read3, 'frame3',image_inference3,model)
    thread_ana4 = Video_analyze_thread(12, "video_ana4", thread_read4, 'frame4',image_inference4,model)
    thread_ana7 = Video_analyze_thread(15, "video_ana7", thread_read7, 'frame7',image_inference7,model)
    thread_ana8 = Video_analyze_thread(16, "video_ana8", thread_read8, 'frame8',image_inference8,model)
    #----------------------------------------------------------------------------------------------------------
    thread_ana9 = Video_face_analyze_thread(17, "video_ana9", thread_read9, 'frame9', image_inference_face1, model1)
    thread_ana10 = Video_face_analyze_thread(18, "video_ana10", thread_read10, 'frame10', image_inference_face2, model1)
    thread_ana11 = Video_face_analyze_thread(19, "video_ana11", thread_read11, 'frame11', image_inference_face3, model1)

    thread_read1.start()
    thread_read2.start()
    thread_read3.start()
    thread_read4.start()
    thread_read7.start()
    thread_read8.start()
    thread_read9.start()
    thread_read10.start()
    thread_read11.start()
    time.sleep(3)


    thread_ana1.start()
    thread_ana2.start()
    thread_ana3.start()
    thread_ana4.start()
    thread_ana7.start()
    thread_ana8.start()
    thread_ana9.start()
    thread_ana10.start()
    thread_ana11.start()
    thread_Kaiguan.start()

    thread_read1.join()
    thread_read2.join()
    thread_read3.join()
    thread_read4.join()
    thread_read7.join()
    thread_read8.join()
    thread_read9.join()
    thread_read10.join()
    thread_read11.join()

    thread_ana1.join()
    thread_ana2.join()
    thread_ana3.join()
    thread_ana4.join()
    thread_ana7.join()
    thread_ana8.join()
    thread_ana9.join()
    thread_ana10.join()
    thread_ana11.join()
    thread_Kaiguan.join()

