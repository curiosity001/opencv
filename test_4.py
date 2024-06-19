# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import warnings
import pickle
import cv2
#from cv2 import VideoCapture
# import math
import copy
import numpy as np
# import time
import logging
from PIL import Image
from skimage import exposure
#import panda as pd
import serial
import time
from matplotlib.path import Path
import cv2
from numpy import random
from utils.datasets import letterbox
from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import numpy as np
import torch
import threading
s = serial.Serial('com14', 9600)
sign1 =0
weights2='person.pt'
device = select_device('0')
half = device.type != 'cpu'  # half precision only supported on CUDA
model2 = attempt_load(weights2, map_location=device)  # load FP32 model
stride = int(model2.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size
names = model2.module.names if hasattr(model2, 'module') else model2.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
if half:
    model2.half()  # to FP16
Time1=0
Time2=0
m1 = Path([(780, 8), (780, 240),(1097, 240),(1097 ,8)])
n1 = Path([(406 , 182), (383 , 497),(762 , 489),(790 , 171)])
def image_inference1(frame,model,device,half,imgsiz,names,colors):
    t1 = time.time()
    global sign1,Time1,Time2
    # frame = frame[420:1500, 0:1080]
    # m = Path([(200,200), (200, 800),(1200, 800),(1200,400)])
    video_weight = 1920
    video_hight = 1080
    frame = cv2.line(frame, (406 , 182), (383 , 497), (0, 255, 0), 4)
    frame = cv2.line(frame, (383 , 497), (762 , 489), (0, 255, 0), 4)
    frame = cv2.line(frame, (762 , 489), (790 , 171), (0, 255, 0), 4)
    frame = cv2.line(frame, (790 , 171), (406 , 182), (0, 255, 0), 4)
    # img = cv2.resize(frame, (640, 640))
    img = letterbox(frame, imgsz, stride=32)[0]

    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    with torch.no_grad():
        pred2=model(img, augment=False)[0]
        pred2 = non_max_suppression(pred2, 0.40, 0.45, classes=0, agnostic=False)[0]

    if len(pred2):
        # Rescale boxes from img_size to im0 size
        pred2[:, :4] = scale_coords(img.shape[2:], pred2[:, :4], frame.shape).round()
        for *xyxy, conf, cls in reversed(pred2):
            if (n1.contains_point((int(xyxy[-4]+(xyxy[-2]-xyxy[-4])/2), int(xyxy[-3]+(xyxy[-1]-xyxy[-3])/2)))):
                label = f'{names[int(cls)]} {conf:.2f}'
                frame=plot_one_box(xyxy,frame,label=label,color=colors[int(cls)], line_thickness=3)
                sign1 = 1
                Time1 = time.time()
            else:
                if sign1==1:
                    Time2 = time.time()
                    if Time2 - Time1 > 10:
                         sign1 = 0
    t2 = time.time()
    FPS=int(1/(t2 - t1))
    print("1FPS:", 1/(t2 - t1))
    cv2.putText(frame, 'FPS=%s' %FPS, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0],2, lineType=cv2.LINE_AA)
    return frame

m2 = Path([(640,9), (640,374),(1144,373),(1145,10)])
n2 = Path([(651 ,402), (493 ,877),(1380, 876),(1202, 403)])
def image_inference2(frame,model,device,half,imgsz,names,colors):
    t1 = time.time()
    video_weight = 1920
    video_hight = 1080
    frame = cv2.line(frame, (640,9), (640,374), (0, 255, 0), 4)
    frame = cv2.line(frame, (640,374), (1144,373), (0, 255, 0), 4)
    frame = cv2.line(frame, (1144,373), (1145,10), (0, 255, 0), 4)
    frame = cv2.line(frame, (1145,10), (640,9), (0, 255, 0), 4)

    frame = cv2.line(frame, (651 ,402), (493 ,877), (0, 255, 0), 4)
    frame = cv2.line(frame, (493 ,877), (1380, 876), (0, 255, 0), 4)
    frame = cv2.line(frame, (1380, 876), (1202, 403), (0, 255, 0), 4)
    frame = cv2.line(frame, (1202, 403), (651 ,402), (0, 255, 0), 4)
    # img=cv2.resize(frame,(640,640))
    img = letterbox(frame, imgsz, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    with torch.no_grad():
        pred2=model(img, augment=False)[0]
        pred2 = non_max_suppression(pred2, 0.40, 0.45, classes=0, agnostic=False)[0]
    if len(pred2):
        # Rescale boxes from img_size to im0 size
        pred2[:, :4] = scale_coords(img.shape[2:], pred2[:, :4], frame.shape).round()
        for *xyxy, conf, cls in reversed(pred2):
            label = f'{names[int(cls)]} {conf:.2f}'
            frame=plot_one_box(xyxy,frame,label=label,color=colors[int(cls)], line_thickness=3)
    t2 = time.time()
    FPS=int(1/(t2 - t1))
    print("2FPS:", 1/(t2 - t1))
    cv2.putText(frame, 'FPS=%s' %FPS, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0],2, lineType=cv2.LINE_AA)
    return frame

url1 = 'D:/yolov7-main-dj/1.mp4'
url2 = 'D:/yolov7-main-dj/1.mp4'

class Kaiguanliang(threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name

    def run(self):
        while True:
            global sign1
            print("detecting")
            if sign1==1:
                s.close()
                s.open()
                d1 = b'\xA0\x01\x01\xA2'
                s.write(d1)
                print("someone inner")
            else:
                s.close()
                s.open()
                d2 = b'\xA0\x01\x00\xA1'
                s.write(d2)
                print("noone inner")
                s.close()

class Video_receive_thread(threading.Thread):
    def __init__(self, threadID, name,url,frame_name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.url=url
        self.frame_name=frame_name
        self.frame = np.zeros(shape=(1080, 1920, 3), dtype=float)
        self.cap = cv2.VideoCapture(self.url)

    def run(self):
        while (self.cap.isOpened()):
            self.ret, self.frame = self.cap.read()
            time.sleep(0.02)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

thread_read1 = Video_receive_thread(1, "video_read1",url1,'frame1')
thread_read2 = Video_receive_thread(2, "video_read2",url2,'frame2')
thread_Kaiguan = Kaiguanliang(1,"USB_Kaiguan")

class Video_analyze_thread(threading.Thread):
    def __init__(self, threadID, name,thread_num,frame_name,img_infer):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
        self.thread_num=thread_num
        self.frame_name=frame_name
        self.img_infer=img_infer

    def run(self):
        while True:
            # self.frame_read = thread1.frame
            self.frame_read = self.img_infer(self.thread_num.frame, model2, device, half, imgsz, names, colors)
            cv2.namedWindow(self.frame_name, cv2.WINDOW_NORMAL)
            cv2.imshow(self.frame_name, self.frame_read)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    #with torch.no_grad():
    thread_ana1 = Video_analyze_thread(5, "video_ana1",thread_read1,'frame1',image_inference1)
    #time.sleep(1)
    thread_ana2 = Video_analyze_thread(6, "video_ana2",thread_read2,'frame2',image_inference2)
    #thread_ana3 = Video_analyze_thread(7, "video_ana3", thread_read3, 'frame3')
    #thread_ana4 = Video_analyze_thread(8, "video_ana4", thread_read4, 'frame4')

    thread_read1.start()
    thread_read2.start()
    #time.sleep(1)

    thread_ana1.start()
    thread_ana2.start()
    thread_Kaiguan.start()
    #thread_ana3.start()
    #thread_ana4.start()


    thread_read1.join()
    #time.sleep(1)
    thread_read2.join()
    thread_Kaiguan.join()
    #thread_read3.join()
    #thread_read4.join()
    thread_ana1.join()
    #time.sleep(1)
    thread_ana2.join()
    #thread_ana3.join()
    #thread_ana4.join()

