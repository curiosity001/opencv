import time
import cv2
from numpy import random
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
import numpy as np
import torch
import threading
from PIL import Image, ImageDraw, ImageFont
import socket
from queue import Queue
import subprocess as sp
import requests
import json
import base64
import datetime

#权重、设备选择
weights = 'tied_detection.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载模型
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

#摄像头选择/域名
# url0 = 0
url1='rtsp://192.168.16.38:554/user=admin&password=&channel=1&stream=0.sdp?'   #5009
url2='rtsp://admin:admin123@192.168.12.148:554/live'   #5010
# url2='rtsp://admin:Aust12345@192.168.31.60:554/live'

rtmpUrl1 = "rtmp://192.168.16.35:1935/live/6"
rtmpUrl2 = "rtmp://192.168.16.35:1935/live/2"

# push_url1 = "rtsp://192.168.12.154:554/room7"
# push_url2 = "rtsp://192.168.12.154:554/room8"

event_url='http://192.168.16.34:8081/getmessage/responseAlarm'

width = int(1280)
height = int(960)


fps = int(12)
# print("width", width, "height", height, "fps：", fps)

command = ['ffmpeg', # linux不用指定
    '-hwaccel','cuvid',
    '-y', '-an',
    '-f', 'rawvideo',
    '-vcodec','rawvideo',
    '-pix_fmt', 'bgr24', #像素格式
    '-s', "{}x{}".format(width, height),
    '-r', str(fps), # 自己的摄像头的fps是0，若用自己的notebook摄像头，设置为15、20、25都可。
    '-i', '-',
    '-c:v', 'h264_nvenc',  # 视频编码方式
    '-pix_fmt', 'yuv420p',
    # '-preset', 'ultrafast',
    '-f', 'flv', #  flv rtsp
    # '-rtmp_transport', 'tcp',  # 使用TCP推流，linux中一定要有这行
    rtmpUrl1] # rtsp rtmp
command2 = ['ffmpeg', # linux不用指定
    '-hwaccel','cuvid',
    '-y', '-an',
    '-f', 'rawvideo',
    '-vcodec','rawvideo',
    '-pix_fmt', 'bgr24', #像素格式
    '-s', "{}x{}".format(width, height),
    '-r', str(fps), # 自己的摄像头的fps是0，若用自己的notebook摄像头，设置为15、20、25都可。
    '-i', '-',
    '-c:v', 'h264_nvenc',  # 视频编码方式
    '-pix_fmt', 'yuv420p',
    # '-preset', 'ultrafast',
    '-f', 'flv', #  flv rtsp
    # '-rtmp_transport', 'tcp',  # 使用TCP推流，linux中一定要有这行
    rtmpUrl2] # rtsp rtmp

pipe1 = sp.Popen(command, shell=False, stdin=sp.PIPE)
pipe2 = sp.Popen(command2, shell=False, stdin=sp.PIPE)

#视频大小
# video_weight = 1920
# video_hight = 1080

#udp发送地址设置
udp_addr = ('192.168.12.153', 9989)
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


#继电器选择/离岗疲劳选择（此处是需要将列表里面的字符作为变量传入线程参数中）
region = ["J1_L15", "J1_L16", "J1_P17","J1_P18", "J2_L01", "J2_L02", "J2_L03", "J2_P04", "J2_P05", "J2_P06"]


video_queue1=Queue()
video_queue2=Queue()

def events_trans(url,data_dict):
    res=requests.post(url=url,json=data_dict,timeout=5)
    content_str=str(res.content,encoding="utf-8")
    print(content_str)

#一个摄像头处理一个人员的图片推理函数
def image_inference(frame, model, device, imgsz, names, colors):
    t1 = time.time()
    face_status = []
    img = letterbox(frame, imgsz, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
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
            face_status.append(int(cls))
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, 'FPS=%s' % FPS, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 0, 0], 2, lineType=cv2.LINE_AA)

    sleep_getaway = 0
    face_tired = 0
    # 睡岗/离开状态（若是没有检测到人脸或只检测到人脸）
    if (len(face_status) == 1 and 0 in face_status) or (len(face_status) == 0):
        sleep_getaway = 1
    else:
        sleep_getaway = 0
    # 判断疲劳状态（若是判断面部信息>1，如检测到眼睛，嘴巴则执行下面代码）:
    if len(face_status) > 1:
        if (1 in face_status) or (4 in face_status):  # 若是检测到眼睛有处于闭合状态或嘴巴张开时；
            face_tired = 1
        else:
            face_tired = 0

    return frame, sleep_getaway, face_tired



#一个摄像头处理两个人员的图片推理函数
def image_inference_2(frame, model, device, imgsz, names, colors, param):
    t1 = time.time()
    face_status1 = []
    face_status2 = []
    img = letterbox(frame, imgsz, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
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
            if (xyxy[0] < param):
                face_status2.append(int(cls))
            else:
                face_status1.append(int(cls))
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, 'FPS=%s' % FPS, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 0, 0], 2, lineType=cv2.LINE_AA)

    sleep_getaway1 = 0
    face_tired1 = 0
    # 睡岗/离开状态（若是没有检测到人脸或只检测到人脸）
    if (len(face_status1) == 1 and 0 in face_status1) or (len(face_status1) == 0):
        sleep_getaway1 = 1
    else:
        sleep_getaway1 = 0
    # 判断疲劳状态（若是判断面部信息>1，如检测到眼睛，嘴巴则执行下面代码）:
    if len(face_status1) > 1:
        if (1 in face_status1) or (4 in face_status1):  # 若是检测到眼睛有处于闭合状态或嘴巴张开时；
            face_tired1 = 1
        else:
            face_tired1 = 0

    sleep_getaway2 = 0
    face_tired2 = 0
    # 睡岗/离开状态（若是没有检测到人脸或只检测到人脸）
    if (len(face_status2) == 1 and 0 in face_status2) or (len(face_status2) == 0):
        sleep_getaway2 = 1
    else:
        sleep_getaway2 = 0
    # 判断疲劳状态（若是判断面部信息>1，如检测到眼睛，嘴巴则执行下面代码）:
    if len(face_status2) > 1:
        if (1 in face_status2) or (4 in face_status2):  # 若是检测到眼睛有处于闭合状态或嘴巴张开时；
            face_tired2 = 1
        else:
            face_tired2 = 0

    return frame, sleep_getaway1, face_tired1, sleep_getaway2, face_tired2




# 视频接收线程（所有视频均从此处接收）
class Video_receive_thread(threading.Thread):
    def __init__(self, threadID, name, url, frame_name,queue):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.url = url
        self.frame_name = frame_name
        self.frame = 0
        self.cap = cv2.VideoCapture(self.url)
        self.q = queue
        self.ret = True

    def run(self):
        while True:
            try:
                self.ret, self.frame = self.cap.read()
                # print(self.frame.shape)
                self.q.put(self.frame)
                if self.q.qsize() > 1:  # 如果队列中的帧数大于1，表示队列已满，需要移除最早的一帧
                    self.q.get()
                if self.ret == False:
                    self.cap = cv2.VideoCapture(self.url)
                    continue
            except:
                time.sleep(5)


# 不分级疲劳判断线程（一个摄像头检测一个人）
class Video_analyze_thread(threading.Thread):
    def __init__(self, threadID, name, thread_num, frame_name, img_infer, model, queue, pipe, camera_id, region1, region2):
        threading.Thread.__init__(self)
        #分析线程需要传入的参数（region1、region2是udp发送信息中继电器/疲劳或离岗状态选择）
        self.threadID = threadID
        self.name = name
        self.thread_num = thread_num
        self.frame_name = frame_name
        self.img_infer = img_infer
        self.model1 = model
        self.region1 = region1
        self.region2 = region2
        self.udp_socket = udp_socket
        self.camera_id = camera_id

        #image_inference返回值初始为0
        self.face_status = []
        self.sleep_getaway = 0
        self.face_tired = 0
        self.frame_read = 0

        #计时器初始值为0
        self.fatigue_timer = 0
        self.fatigue_timer1 = 0

        #离岗/疲劳标志位初始值置[0]
        self.sleep_getaway_list = [0]
        self.sleep_getaway_list_before = [0]
        self.face_tired_list = [0]
        self.face_tired_list_before = [0]

        #摄像头掉线标志位置0
        self.Camera_drop_off = 0

        #人员从离岗到在线、疲劳到清醒标志位初始值置0
        self.person_online_flag = 0
        self.person_sober_flag = 0

        self.q=queue
        self.count=0
        self.pipe = pipe

        self.data_dict = {
            'positionid': '0001',  # 煤矿ID
            'cameraid': self.camera_id,  # 摄像头ID
            'starttime': '',
            'endtime': '',
            'currenttimes': '',  # 报警触发时间
            'eventtype': '12002',  # 事件类型
            'image': "0",
            'taskid': '12',  # 任务ID
            'mask': "人员疲劳"  # 报警事件描述
        }



    def run(self):
        self.udp_socket.sendto(f"{self.region1}_{0}".encode('utf-8'), udp_addr)
        self.udp_socket.sendto(f"{self.region2}_{0}".encode('utf-8'), udp_addr)

        while True:
            #摄像头掉线判断
            if self.q.qsize() == 0:
                time.sleep(0.15)
                self.count += 1
                if self.count == 20:
                    self.count = 0
                    self.udp_socket.sendto(f"{self.region1}_{0}".encode('utf-8'),udp_addr)
                    self.udp_socket.sendto(f"{self.region2}_{0}".encode('utf-8'),udp_addr)
                self.pipe.stdin.write(self.frame_read.tostring())
            #当摄像头正常运行时，执行疲劳分析代码
            else:
                try:
                    self.count=0
                    #将上一帧图片的离岗/疲劳标志位赋给 self.sleep_getaway_list_before，用于当前帧的标志位与上一帧进行比较
                    self.sleep_getaway_list_before = self.sleep_getaway_list.copy()
                    self.face_tired_list_before = self.face_tired_list.copy()

                    #将image_inference里面的返回值进行赋值
                    self.frame_read, self.sleep_getaway, self.face_tired = self.img_infer(self.q.get(),
                                                                                self.model1, device, imgsz, names,colors)
                    #当出现人员离岗时执行下列代码，向对应继电器的对应位置发送信息
                    if self.sleep_getaway == 1:
                        if self.fatigue_timer == 0:
                            self.fatigue_timer = time.time()
                        else:
                            self.time_away = time.time() - self.fatigue_timer
                            if self.time_away >= 5:
                                self.frame_read = cv2.putText(self.frame_read, "sleep/getaway", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0),2 , lineType=cv2.LINE_AA)
                                self.sleep_getaway_list = [1]
                            else:
                                self.sleep_getaway_list = [0]
                            if self.sleep_getaway_list != self.sleep_getaway_list_before:
                                if self.sleep_getaway_list == [1]:
                                    self.udp_socket.sendto(f"{self.region1}_{self.sleep_getaway}".encode('utf-8'), udp_addr)
                                    self.person_online_flag = 1
                    #当检测到人员时且是从离岗到工作时，发送人员在岗信息
                    else:
                        if self.person_online_flag == 1:
                            self.udp_socket.sendto(f"{self.region1}_{self.sleep_getaway}".encode('utf-8'), udp_addr)
                            self.person_online_flag = 0
                        self.fatigue_timer = 0

                    #当出现人员疲劳时执行下列代码，向对应继电器的对应位置发送信息
                    if self.face_tired == 1:
                        if self.fatigue_timer1 == 0:
                            self.fatigue_timer1 = time.time()
                        else:
                            self.time_tired = time.time() - self.fatigue_timer1
                            if self.time_tired >= 4:
                                # self.frame_read = cv2AddChineseText(self.frame_read, "疲劳", (50, 70), (255, 0, 0), 50)
                                self.frame_read = cv2.putText(self.frame_read, "tired", (50, 240),
                                                              cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2,
                                                              lineType=cv2.LINE_AA)
                                self.face_tired_list = [1]
                            else:
                                self.face_tired_list = [0]
                            if self.face_tired_list != self.face_tired_list_before:
                                if self.face_tired_list == [1]:
                                    self.udp_socket.sendto(f"{self.region2}_{self.face_tired}".encode('utf-8'), udp_addr)

                                    self.str_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    self.data_dict['currenttimes'] = self.str_time
                                    threading.Thread(target=events_trans, args=(event_url, self.data_dict),
                                                     daemon=True).start()
                                    self.person_sober_flag = 1
                    #当人员清醒时且从疲劳到清醒状态时，发送人员清醒信息
                    else:
                        if self.person_sober_flag == 1:
                            self.udp_socket.sendto(f"{self.region2}_{self.face_tired}".encode('utf-8'), udp_addr)
                            self.person_sober_flag = 0
                        self.fatigue_timer1 = 0
                    self.pipe.stdin.write(self.frame_read.tostring())
                except:
                    time.sleep(0.15)
                    self.pipe.stdin.write(self.frame_read.tostring())




# 不分级疲劳判断线程（一个摄像头检测两个人）
class Video_analyze_thread1(threading.Thread):
    def __init__(self, threadID, name, thread_num, frame_name, img_infer, model, queue, pipe, camera_id, region1,  region2, region3, region4):
        threading.Thread.__init__(self)
        #一个摄像头检测两个人的传入参数初始值（region1、2、3、4分别代表：右边人员离岗/在线继电器选择；右边人员疲劳/清醒继电器选择；
        # 左边人员离岗/在线继电器选择；左边人员疲劳/清醒继电器选择）
        self.threadID = threadID
        self.name = name
        self.thread_num = thread_num
        self.frame_name = frame_name
        self.img_infer = img_infer
        self.model1 = model
        self.q = queue
        self.region1 = region1
        self.region2 = region2
        self.region3 = region3
        self.region4 = region4

        self.face_status = []
        self.udp_socket = udp_socket

        #从image_inference_2里面返回的参数初始值置0
        self.frame_read1 = 0
        self.sleep_getaway1 = 0
        self.sleep_getaway2 = 0
        self.face_tired1 = 0
        self.face_tired2 = 0

        #计时器初始值置0
        self.fatigue_timer = 0
        self.fatigue_timer1 = 0
        self.fatigue_timer2 = 0
        self.fatigue_timer3 = 0

        #左右人员当前与上一帧离岗状态标志位初始值置0
        self.sleep_getaway_list = [0]
        self.sleep_getaway_list1 = [0]
        self.sleep_getaway_list_before = [0]
        self.sleep_getaway_list_1_before = [0]
        #左右人员当前与上一帧疲劳状态标志位初始值置0
        self.face_tired_list = [0]
        self.face_tired_list1 = [0]
        self.face_tired_list_before = [0]
        self.face_tired_list_1_before = [0]

        #右边人员在线及清醒状态标志位初始值置0
        self.person_online_flag_1 = 0
        self.person_sober_flag_1 = 0
        #左边人员在线及清醒状态标志位初始值置0
        self.person_online_flag_2 = 0
        self.person_sober_flag_2 = 0

        #shexiangtouduanxianjishiqi
        self.count = 0
        self.Camera_drop_off = 0
        self.pipe = pipe

        #chuangli diction
        self.data_dict={
            'position_id':'0001',
            'camera_id':camera_id,
            'start_time':'',
            'end_time':'',
            'current_time':'',
            'event_type':'55002',
            'task_id':'5000',
        }


    def run(self):
        self.udp_socket.sendto(f"{self.region1}_{0}".encode('utf-8'), udp_addr)
        self.udp_socket.sendto(f"{self.region2}_{0}".encode('utf-8'), udp_addr)
        self.udp_socket.sendto(f"{self.region3}_{0}".encode('utf-8'), udp_addr)
        self.udp_socket.sendto(f"{self.region4}_{0}".encode('utf-8'), udp_addr)

        while True:
            # 摄像头掉线判断
            if self.q.qsize() == 0:
                time.sleep(0.15)
                self.count += 1
                if self.count == 20:
                    self.count = 0
                    self.udp_socket.sendto(f"{self.region1}_{0}".encode('utf-8'),udp_addr)
                    self.udp_socket.sendto(f"{self.region2}_{0}".encode('utf-8'),udp_addr)
                    self.udp_socket.sendto(f"{self.region3}_{0}".encode('utf-8'),udp_addr)
                    self.udp_socket.sendto(f"{self.region4}_{0}".encode('utf-8'),udp_addr)
                self.pipe.stdin.write(self.frame_read1.tostring())
            else:
                try:
                    self.count = 0
                    self.sleep_getaway_list_before = self.sleep_getaway_list.copy()
                    self.sleep_getaway_list_1_before = self.sleep_getaway_list1.copy()
                    self.face_tired_list_before = self.face_tired_list.copy()
                    self.face_tired_list_1_before = self.face_tired_list1.copy()

                    self.frame_read1, self.sleep_getaway1, self.face_tired1, self.sleep_getaway2, self.face_tired2 = \
                        self.img_infer(self.q.get(), self.model1, device, imgsz, names, colors, 1280)

                    # zuo边人员的在岗/离岗状态判断
                    if self.sleep_getaway1 == 1:
                        if self.fatigue_timer == 0:
                            self.fatigue_timer = time.time()
                        else:
                            self.time_away = time.time() - self.fatigue_timer
                            if self.time_away >= 5:
                                # self.frame_read1 = cv2AddChineseText(self.frame_read1, "睡着/离开", (400, 70), (255, 0, 0), 50)
                                self.frame_read1 = cv2.putText(self.frame_read1, "sleep/getaway", (1280, 240),
                                                              cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, lineType=cv2.LINE_AA)
                                self.sleep_getaway_list = [1]
                            else:
                                self.sleep_getaway_list = [0]
                            if self.sleep_getaway_list != self.sleep_getaway_list_before:
                                if self.sleep_getaway_list == [1]:
                                    self.udp_socket.sendto(f"{self.region1}_{self.sleep_getaway1}".encode('utf-8'), udp_addr)
                                    self.person_online_flag_1 = 1


                    #zuo边人员在在岗状态信息发送
                    else:
                        if self.person_online_flag_1 == 1:
                            self.udp_socket.sendto(f"{self.region1}_{self.sleep_getaway1}".encode('utf-8'), udp_addr)
                            self.person_online_flag_1 = 0


                        self.fatigue_timer = 0

                    # zuo边人员的疲劳状态判断
                    if self.face_tired1 == 1:
                        if self.fatigue_timer1 == 0:
                            self.fatigue_timer1 = time.time()
                        else:
                            self.time_tired = time.time() - self.fatigue_timer1
                            if self.time_tired >= 3:
                                # self.frame_read1 = cv2AddChineseText(self.frame_read1, "疲劳", (500, 70), (255, 0, 0), 50)
                                self.frame_read1 = cv2.putText(self.frame_read1, "tired", (1280, 240),
                                                              cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, lineType=cv2.LINE_AA)
                                self.face_tired_list = [1]
                            else:
                                self.face_tired_list = [0]
                            if self.face_tired_list != self.face_tired_list_before:
                                if self.face_tired_list == [1]:
                                    self.udp_socket.sendto(f"{self.region2}_{self.face_tired1}".encode('utf-8'), udp_addr)
                                    self.person_sober_flag_1 = 1

                    #zuo边人员清醒状态信息发送
                    else:
                        if self.person_sober_flag_1 == 1:
                            self.udp_socket.sendto(f"{self.region2}_{self.face_tired1}".encode('utf-8'), udp_addr)
                            self.person_sober_flag_1 = 0

                        self.fatigue_timer1 = 0

                    # you边人员离岗状态判断
                    if self.sleep_getaway2 == 1:
                        if self.fatigue_timer2 == 0:
                            self.fatigue_timer2 = time.time()
                        else:
                            self.time_away1 = time.time() - self.fatigue_timer2
                            if self.time_away1 >= 5:
                                # self.frame_read1 = cv2AddChineseText(self.frame_read1, "睡着/离开", (50, 70), (255, 0, 0), 50)
                                self.frame_read1 = cv2.putText(self.frame_read1, "sleep/getaway", (50, 240),
                                                              cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, lineType=cv2.LINE_AA)
                                self.sleep_getaway_list1 = [1]
                            else:
                                self.sleep_getaway_list1 = [0]
                            if self.sleep_getaway_list1 != self.sleep_getaway_list_1_before:
                                if self.sleep_getaway_list1 == [1]:
                                    self.udp_socket.sendto(f"{self.region3}_{self.sleep_getaway2}".encode('utf-8'), udp_addr)
                                    self.person_online_flag_2 = 1


                    #you边人员在岗状态信息发送
                    else:
                        if self.person_online_flag_2 == 1:
                            self.udp_socket.sendto(f"{self.region3}_{self.sleep_getaway2}".encode('utf-8'), udp_addr)
                            self.person_online_flag_2 = 0


                        self.fatigue_timer2 = 0

                    # you边人员的疲劳状态判断
                    if self.face_tired2 == 1:
                        if self.fatigue_timer3 == 0:
                            self.fatigue_timer3 = time.time()
                        else:
                            self.time_tired1 = time.time() - self.fatigue_timer3
                            if self.time_tired1 >= 3:
                                # self.frame_read1 = cv2AddChineseText(self.frame_read1, "疲劳", (50, 70), (255, 0, 0), 50)
                                self.frame_read1 = cv2.putText(self.frame_read1, "tired", (50, 240),
                                                              cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, lineType=cv2.LINE_AA)
                                self.face_tired_list1 = [1]
                            else:
                                self.face_tired_list1 = [0]
                            if self.face_tired_list1 != self.face_tired_list_1_before:
                                if self.face_tired_list1 == [1]:
                                    self.udp_socket.sendto(f"{self.region4}_{self.face_tired2}".encode('utf-8'), udp_addr)
                                    self.person_sober_flag_2 = 1


                    #you边人员清醒状态信息发送
                    else:
                        if self.person_sober_flag_2 == 1:
                            self.udp_socket.sendto(f"{self.region4}_{self.face_tired2}".encode('utf-8'), udp_addr)
                            self.person_sober_flag_2 = 0


                        self.fatigue_timer3 = 0

                    self.pipe.stdin.write(self.frame_read1.tostring())
                except:
                    time.sleep(0.15)
                    self.pipe.stdin.write(self.frame_read1.tostring())




if __name__ == "__main__":



#视频接收线程(one people)
    # thread_read0 = Video_receive_thread(1, "video_read0", url0, 'frame0',video_queue1)
    thread_read1 = Video_receive_thread(1, "video_read1", url1, 'frame1',video_queue1)
    # thread_read2 = Video_receive_thread(2, "video_read2", url2, 'frame2',video_queue1)
    # thread_read3 = Video_receive_thread(3, "video_read3", url3, 'frame3',video_queue1)
    # thread_read4 = Video_receive_thread(4, "video_read4", url4, 'frame4',video_queue1)
    # thread_read5 = Video_receive_thread(5, "video_read5", url5, 'frame5',video_queue1)

#视频接收线程(two people)
    # thread_read0 = Video_receive_thread(1, "video_read0", url0, 'frame0',video_queue2)
    # thread_read1 = Video_receive_thread(1, "video_read1", url1, 'frame1',video_queue1)
    # thread_read2 = Video_receive_thread(2, "video_read2", url2, 'frame2',video_queue2)
    # thread_read3 = Video_receive_thread(3, "video_read3", url3, 'frame3',video_queue2)
    # thread_read4 = Video_receive_thread(4, "video_read4", url4, 'frame4',video_queue2)
    # thread_read5 = Video_receive_thread(5, "video_read5", url5, 'frame5',video_queue2)


#离岗/疲劳分析线程（一个摄像头检测一个人）
    thread_ana1 = Video_analyze_thread(6, "video_ana1", thread_read1, 'frame1', image_inference, model, video_queue1, pipe1, 37, region[0], region[3])
    # thread_ana2 = Video_analyze_thread(7, "video_ana2", thread_read2, 'frame2', image_inference, model, video_queue1, pipe1,region[0], region[3])
    # thread_ana3 = Video_analyze_thread(8, "video_ana3", thread_read3, 'frame3', image_inference, model, video_queue1,pipe1,region[0], region[3])
    # thread_ana4 = Video_analyze_thread(9, "video_ana4", thread_read4, 'frame4', image_inference, model, video_queue1,pipe1,region[0], region[3])
    # thread_ana5 = Video_analyze_thread(10, "video_ana5", thread_read5, 'frame5', image_inference, model, video_queue1, pipe1,region[0], region[3])


#离岗/疲劳分析线程（一个摄像头检测两个人）
    # thread_ana6 = Video_analyze_thread1(11, "video_ana6", thread_read1, 'frame6', image_inference_2, model, video_queue1, pipe1,'5009', region[5], region[8], region[6], region[9])  #shuang guan siji shi
    # thread_ana7 = Video_analyze_thread1(12, "video_ana7", thread_read2, 'frame7', image_inference_2, model, video_queue2, pipe2,'5010', region[0], region[2], region[1], region[3])  #dan guan siji shi
    # thread_ana8 = Video_analyze_thread1(13, "video_ana8", thread_read3, 'frame8', image_inference_2, model, video_queue2, pipe2, region[1], region[4], region[2], region[5])
    # thread_ana9 = Video_analyze_thread1(14, "video_ana9", thread_read4, 'frame9', image_inference_2, model, video_queue2, pipe2, region[1], region[4], region[2], region[5])
    # thread_ana10 = Video_analyze_thread1(15, "video_ana10", thread_read5, 'frame10', image_inference_2, model, video_queue2, pipe2, region[1], region[4], region[2], region[5])

#视频接收线程启动
    thread_read1.start()
    # thread_read2.start()
    # thread_read3.start()
    # thread_read4.start()
    # thread_read5.start()

    time.sleep(2)

#离岗/疲劳分析线程（一个摄像头检测一个人）启动
    thread_ana1.start()
    # thread_ana2.start()
    # thread_ana3.start()
    # thread_ana4.start()
    # thread_ana5.start()
#离岗/疲劳分析线程（一个摄像头检测两个人）启动
    # thread_ana6.start()
    # thread_ana7.start()
    # thread_ana8.start()
    # thread_ana9.start()
    # thread_ana10.start()

#视频显示
    cv2.namedWindow(thread_ana1.frame_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(thread_ana2.frame_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(thread_ana3.frame_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(thread_ana4.frame_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(thread_ana5.frame_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(thread_ana6.frame_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(thread_ana7.frame_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(thread_ana8.frame_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(thread_ana9.frame_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(thread_ana10.frame_name, cv2.WINDOW_NORMAL)

    while True:
        cv2.imshow(thread_ana1.frame_name, thread_ana1.frame_read)
        # cv2.imshow(thread_ana2.frame_name, thread_ana2.frame_read)
        # cv2.imshow(thread_ana3.frame_name, thread_ana3.frame_read)
        # cv2.imshow(thread_ana4.frame_name, thread_ana4.frame_read)
        # cv2.imshow(thread_ana5.frame_name, thread_ana5.frame_read)
        # cv2.imshow(thread_ana6.frame_name, thread_ana6.frame_read1)
        # cv2.imshow(thread_ana7.frame_name, thread_ana7.frame_read1)
        # cv2.imshow(thread_ana8.frame_name, thread_ana8.frame_read1)
        # cv2.imshow(thread_ana9.frame_name, thread_ana9.frame_read1)
        # cv2.imshow(thread_ana10.frame_name, thread_ana10.frame_read1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()



    thread_read1.join()
    # thread_read2.join()
    # thread_read3.join()
    # thread_read4.join()
    # thread_read5.join()

    thread_ana1.join()
    # thread_ana2.join()
    # thread_ana3.join()
    # thread_ana4.join()
    # thread_ana5.join()

    # thread_ana6.join()
    # thread_ana7.join()
    # thread_ana8.join()
    # thread_ana9.join()
    # thread_ana10.join()

