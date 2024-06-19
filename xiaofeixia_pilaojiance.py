import time
from pathlib import Path
import socket

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import numpy as np
import torch
import threading
from PIL import Image, ImageDraw, ImageFont
import pygame
import modbus_tk.modbus_tcp as mt
import Jetson.GPIO as GPIO




pygame.mixer.init()#初始化

master=mt.TcpMaster("10.4.47.128",999) #向某个地址发送数据




# weights='YOLO-fs.pt'
weights='tied_detection.pt'
# device = select_device('cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model

stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

udp_addr = ('192.168.31.49', 8081)
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# cap=cv2.VideoCapture('3.mp4')
url1=0
# url1='3.mp4'
video_weight=1920
video_hight=1080

# led灯控制函数
def control_led(num_leds):
    # 清除停止LED闪烁事件
    stop_led.clear()
    while not stop_led.is_set():
        # 将所有 LED 灯关闭
        for pin in led_pins:
            GPIO.output(pin, GPIO.LOW)

        # 打开对应数量的 LED 灯
        for i in range(num_leds):
            GPIO.output(led_pins[i], GPIO.HIGH)
            time.sleep(0.1)  # 等待 0.5 秒
            GPIO.output(led_pins[i], GPIO.LOW)
            time.sleep(0.1)  # 等待 0.5 秒

# 蜂鸣器控制函数
def buzzer_alert():
    stop_buzzer.clear()
    # 触发蜂鸣器报警
    while not stop_buzzer.is_set():
        GPIO.output(buzzer_pin, GPIO.LOW)  # 蜂鸣器开始响
        time.sleep(0.1)  # 响铃持续时间
        GPIO.output(buzzer_pin, GPIO.HIGH)  # 蜂鸣器停止响
        time.sleep(0.1)  # 停止响铃后的等待时间




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

def image_inference(frame,model,device,imgsz,names,colors):
    t1 = time.time()
    face_status=[]
    img = letterbox(frame, imgsz, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    # img = img.half() if half else img.float()
    # img = img.half()
    img=img.float()
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
            # ss
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, 'FPS=%s' % FPS, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame,face_status

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
# current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

class Video_analyze_thread(threading.Thread):
    def __init__(self, threadID, name,thread_num,frame_name,img_infer,model):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        # self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
        self.frame_read=0
        self.thread_num=thread_num
        self.frame_name=frame_name
        self.img_infer = img_infer
        self.model1=model
        self.sleep_getaway_list_before = [0]
        self.sleep_getaway_list = [0]
        self.face_tired_list_before = [0]
        self.face_tired_list = [0]
        self.face_status=[]
        self.person_online_flag = 0
        #睡岗计时器
        self.time1=None
        self.time2=None
        #疲劳计时器
        self.time3=None
        self.time4=None
        self.udp_socket = udp_socket
        self.current_time = 0

    def run(self):
        while True:
            self.sleep_getaway_list_before = self.sleep_getaway_list.copy()
            self.face_tired_list_before = self.face_tired_list.copy()
            self.frame_read,self.face_status = self.img_infer(self.thread_num.frame, self.model1, device, imgsz, names, colors)

            #进行睡岗检查，检查不到人脸，或者检查到人脸但是没有眼睛
            if (len(self.face_status)==1 and 0 in self.face_status) or (len(self.face_status)==0):
                #第一次监测到睡岗的情况，记录一下时间，这个时间可以被在此赋值为None
                if self.time1==None:
                    self.time1=time.time()
                else:
                    self.time2=time.time()
                    if self.time2-self.time1 >= 5:
                        self.frame_read = cv2AddChineseText(self.frame_read, "睡着/离开", (50, 70), (255, 0, 0), 50)
                        self.sleep_getaway_list = [1]
                    else:
                        self.sleep_getaway_list = [0]
                    if self.sleep_getaway_list != self.sleep_getaway_list_before:
                        if self.sleep_getaway_list == [1]:
                            self.current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                            self.udp_socket.sendto(f"face_处于睡着/离开状态_{self.current_time}_detection".encode('utf-8'), udp_addr)
                            self.person_online_flag = 1

            else:
                self.time1=None
                self.time2=None



            if len(self.face_status) > 1:
                if (1 in self.face_status) or (4 in self.face_status):
                    if self.time3==None:
                        self.time3=time.time()
                    else:
                        self.time4=time.time()
                        time_tied=self.time4-self.time3

                        if time_tied>=2 and time_tied<5:
                            self.frame_read = cv2AddChineseText(self.frame_read, "一级疲劳", (50, 70), (255, 0, 0), 50)
                            self.face_tired_list = [1]
                            if self.face_tired_list != self.face_tired_list_before:
                                if self.face_tired_list == [1]:
                                    self.current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                                    self.udp_socket.sendto(f"face_处于一级疲劳状态_{self.current_time}_detection".encode('utf-8'), udp_addr)
                                    self.person_online_flag = 1

                        if time_tied>=5 and time_tied<8:
                            # print("二级")
                            self.frame_read = cv2AddChineseText(self.frame_read, "二级疲劳", (50, 70), (255, 0, 0), 50)
                            self.face_tired_list = [2]
                            if self.face_tired_list != self.face_tired_list_before:
                                if self.face_tired_list == [2]:
                                    self.current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                                    self.udp_socket.sendto(f"face_处于二级疲劳状态_{self.current_time}_detection".encode('utf-8'), udp_addr)
                                    self.person_online_flag = 1

                        if time_tied>=8 and time_tied<11:
                            # print("三级")
                            self.frame_read = cv2AddChineseText(self.frame_read, "三级疲劳", (50, 70), (255, 0, 0), 50)
                            self.face_tired_list = [3]
                            if self.face_tired_list != self.face_tired_list_before:
                                if self.face_tired_list == [3]:
                                    self.current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                                    self.udp_socket.sendto(f"face_处于三级疲劳状态_{self.current_time}_detection".encode('utf-8'), udp_addr)
                                    self.person_online_flag = 1

                        if time_tied>=11:
                            # print("四级")
                            self.frame_read = cv2AddChineseText(self.frame_read, "四级疲劳", (50, 70), (255, 0, 0), 50)
                            self.face_tired_list = [4]
                            if self.face_tired_list != self.face_tired_list_before:
                                if self.face_tired_list == [4]:
                                    self.current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                                    self.udp_socket.sendto(f"face_处于四级疲劳状态_{self.current_time}_detection".encode('utf-8'), udp_addr)
                                    self.person_online_flag = 1
                else:
                    self.time3=None
                    self.time4=None

            if (2 in self.face_status) and (0 in self.face_status):
                if self.person_online_flag == 1:
                    self.current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    self.udp_socket.sendto(f"face_处于工作状态_{self.current_time}_detection".encode('utf-8'), udp_addr)
                self.person_online_flag = 0
                self.time1 = None
                self.time2 = None
                self.time3 = None
                self.time4 = None


            _, send_data = cv2.imencode('.jpg', self.frame_read, [cv2.IMWRITE_JPEG_QUALITY, 50])
            self.udp_socket.sendto(send_data, udp_addr)

        #     cv2.imshow(self.frame_name, self.frame_read)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    thread_ana1 = Video_analyze_thread(9, "video_ana1", thread_read1, 'frame1', image_inference, model)
    thread_read1.start()

    time.sleep(2)

    thread_ana1.start()

    cv2.namedWindow(thread_ana1.frame_name, cv2.WINDOW_NORMAL)
    while True:
        cv2.imshow(thread_ana1.frame_name, thread_ana1.frame_read)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    thread_read1.join()
    thread_ana1.join()

