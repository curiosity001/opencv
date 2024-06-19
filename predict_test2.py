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
metalocA = threading.Lock()
weights='person.pt'
device = select_device('0')
half = device.type != 'cpu'  # half precision only supported on CUDA
# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model

stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

if half:
    model.half()

# cap=cv2.VideoCapture('3.mp4')
url1='22/5.mp4'
video_weight=1920
video_hight=1080

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

have =0
real_time_start1 = [0,0]
sign=[0,0,0,0,0,0]
coust=0
none =0
have =0

pygame.mixer.init()

m1 = Path([(788 , 9), (788 , 278),(1108 , 278),(1107 , 10)])
n1 = Path([(778 , 291), (448 , 1061),(1457 , 1061),(1136 , 295)])
m4 = Path([(524, 101), (524 ,912),(1445, 911),(1445,100)])
none3=0
have3=0
real_time_start4 = [0,0]



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
    # img = img.float()
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
                    # frame = cv2AddChineseText(frame, "有人", (100, 100), (255, 0, 0), 100)
                    cv2.putText(frame,'person',(100,100),cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
                    sign[0] = 1
                    Time1 = time.time()
                else:
                    if sign[0] == 1:
                        Time2 = time.time()
                        if Time2 - Time1 > 10:
                            sign[0] = 0
#-----------------------------------------------------------------------------------------------------------------------输出2.1
                if (n1.contains_point((int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                    # frame = cv2AddChineseText(frame, "有人", (100, 100), (255, 0, 0), 100)
                    cv2.putText(frame, 'person', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2,lineType=cv2.LINE_AA)
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
                    print('1',real_time_start1)
                    # my.write('1'+real_time_start)
                none = 0
                #print(3)
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, '1FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame

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
            print(1)
        if len(pred) and coust == 1:
            # Rescale boxes from img_size to im0 size
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            print(2)
            for *xyxy, conf, cls in reversed(pred):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
            for *xyxy, conf, cls in reversed(pred):
                if (m4.contains_point(
                        (int(xyxy[-4] + (xyxy[-2] - xyxy[-4]) / 2), int(xyxy[-3] + (xyxy[-1] - xyxy[-3]) / 2)))):
                    print(3)
                    # label = f'{names[int(cls)]} {conf:.2f}'
                    print(4)
                    # frame = plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
                    print(5)
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
        frame = cv2AddChineseText(frame, "无人", (100, 100), (0, 255, 0), 100)
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
    print(6)
    cv2.putText(frame, 'FPS=%s' % FPS, (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    print(7)
    return frame

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

class Video_analyze_thread(threading.Thread):
    def __init__(self, threadID, name,thread_num,frame_name,img_infer,model):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        # self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
        self.frame_read=None
        self.thread_num=thread_num
        self.frame_name=frame_name
        self.img_infer = img_infer
        self.model=model
    def run(self):
        while True:
            self.frame_read = self.img_infer(self.thread_num.frame, self.model, device, half, imgsz, names, colors)
            cv2.imshow(self.frame_name, self.frame_read)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    thread_ana1 = Video_analyze_thread(9, "video_ana1",thread_read1,'frame1',image_inference1,model)
    thread_read1.start()
    time.sleep(2)
    thread_ana1.start()

    thread_read1.join()
    thread_ana1.join()