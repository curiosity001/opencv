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
import matplotlib.pyplot as plt

weights = 'tied_detection.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

url0 = 0
# url1 = 'rtsp://admin:Aust12345@192.168.31.62:554/live'
# url2 = 'rtsp://admin:Aust12345@192.168.31.59:554/live'
# url3 = 'rtsp://admin:Aust12345@192.168.31.60:554/live'
# url4 = 'rtsp://admin:Aust12345@192.168.31.62:554/live'
# url5 = 'rtsp://admin:Aust12345@192.168.31.63:554/live'
# url6 = 'rtsp://admin:Aust12345@192.168.31.68:554/live'
# url7 = 'rtsp://admin:Aust12345@192.168.31.68:554/live'
# url8 = 'rtsp://admin:Aust12345@192.168.31.68:554/live'

url_select = [0, 1, 2, 3, 4, 5, 6, 7, 8]

video_weight = 1920
video_hight = 1080

udp_addr = ('192.168.31.116', 9999)
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


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

def image_inference_2(frame, model, device, imgsz, names, colors,param):
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
            if(xyxy[0]<param):
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


# 视频接收线程（一个摄像头只处理一个人）
class Video_receive_thread(threading.Thread):
    def __init__(self, threadID, name, url, frame_name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.url = url
        self.frame_name = frame_name
        self.frame = 0
        self.cap = cv2.VideoCapture(self.url)
        self.q = Queue()

    def run(self):
        while True:
            self.ret, self.frame = self.cap.read()
            self.q.put(self.frame)
            if self.q.qsize() > 2:  # 如果队列中的帧数大于1，表示队列已满，需要移除最早的一帧
                self.q.get()
            if self.ret == False:
                self.cap = cv2.VideoCapture(self.url)
                continue

# 不分级疲劳判断线程（一个摄像头检测一个人）
class Video_analyze_thread(threading.Thread):
    def __init__(self, threadID, name, thread_num, frame_name, img_infer, model, url_select):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.url_select = url_select
        self.frame_read = 0
        self.thread_num = thread_num
        self.frame_name = frame_name
        self.img_infer = img_infer
        self.model1 = model
        self.face_status = []
        self.udp_socket = udp_socket
        self.sleep_getaway = 0
        self.face_tired = 0
        self.fatigue_timer = 0
        self.fatigue_timer1 = 0
        self.sleep_getaway_list = [0]
        self.sleep_getaway_list_before = [0]
        self.face_tired_list = [0]
        self.face_tired_list_before = [0]
        self.Camera_drop_off = 0
        self.person_online_flag = 0
        self.person_sober_flag = 0
    def run(self):
        while True:

            if self.thread_num.q.qsize == 0:
                time.sleep(0.15)
                self.count += 1
                self.frame_read = np.random.rand(1080, 1920, 3) * 255
                if self.count == 20:
                    self.count = 0
                    self.Camera_drop_off = 1
                    self.udp_socket.sendto(f"{self.url_select}目前处于掉线状态{self.Camera_drop_off}".encode('utf-8'),
                                           udp_addr)

            else:
                self.sleep_getaway_list_before = self.sleep_getaway_list.copy()
                self.face_tired_list_before = self.face_tired_list.copy()
                self.frame_read, self.sleep_getaway, self.face_tired = self.img_infer(self.thread_num.q.get(),
                                                                                      self.model1, device, imgsz, names,
                                                                                      colors)

                if self.sleep_getaway == 1:
                    if self.fatigue_timer == 0:
                        self.fatigue_timer = time.time()
                    else:
                        self.time_away = time.time() - self.fatigue_timer
                        if self.time_away >= 5:
                            self.frame_read = cv2AddChineseText(self.frame_read, "睡着/离开", (50, 70), (255, 0, 0), 50)
                            self.sleep_getaway_list = [1]
                        else:
                            self.sleep_getaway_list = [0]
                        if self.sleep_getaway_list != self.sleep_getaway_list_before:
                            if self.sleep_getaway_list == [1]:
                                self.udp_socket.sendto(f"{self.url_select}目前处于睡着或离开状态{self.sleep_getaway}".
                                                       encode('utf-8'), udp_addr)
                                self.person_online_flag = 1

                else:
                    if self.person_online_flag == 1:
                        self.udp_socket.sendto(f"{self.url_select}目前处于在线或到岗状态{self.sleep_getaway}".
                                               encode('utf-8'), udp_addr)
                        self.person_online_flag = 0
                    self.fatigue_timer = 0

                if self.face_tired == 1:
                    if self.fatigue_timer1 == 0:
                        self.fatigue_timer1 = time.time()
                    else:
                        self.time_tired = time.time() - self.fatigue_timer1
                        if self.time_tired >= 4:
                            self.frame_read = cv2AddChineseText(self.frame_read, "疲劳", (50, 70), (255, 0, 0), 50)
                            self.face_tired_list = [1]
                        else:
                            self.face_tired_list = [0]
                        if self.face_tired_list != self.face_tired_list_before:
                            if self.face_tired_list == [1]:
                                self.udp_socket.sendto(f"{self.url_select}目前处于疲劳状态{self.face_tired}".
                                                       encode('utf-8'), udp_addr)
                                self.person_sober_flag = 1
                else:
                    if self.person_sober_flag == 1:
                        self.udp_socket.sendto(f"{self.url_select}目前处于清醒状态{self.face_tired}".
                                               encode('utf-8'), udp_addr)
                        self.person_sober_flag = 0
                    self.fatigue_timer1 = 0


# 不分级疲劳判断线程（一个摄像头检测两个人）
class Video_analyze_thread1(threading.Thread):
    def __init__(self, threadID, name, thread_num, frame_name, img_infer, model):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.frame_read1 = 0
        self.thread_num = thread_num
        self.frame_name = frame_name
        self.img_infer = img_infer
        self.model1 = model
        self.face_status = []
        self.udp_socket = udp_socket
        self.sleep_getaway1 = 0
        self.sleep_getaway2 = 0
        self.face_tired1 = 0
        self.face_tired2 = 0
        self.fatigue_timer = 0
        self.fatigue_timer1 = 0
        self.fatigue_timer2 = 0
        self.fatigue_timer3 = 0
        self.sleep_getaway_list = [0]
        self.sleep_getaway_list1 = [0]
        self.sleep_getaway_list_before = [0]
        self.sleep_getaway_list_1_before = [0]
        self.face_tired_list = [0]
        self.face_tired_list1 = [0]
        self.face_tired_list_before = [0]
        self.face_tired_list_1_before = [0]
        self.person_online_flag_1 = 0
        self.person_sober_flag_1 = 0
        self.person_online_flag_2 = 0
        self.person_sober_flag_2 = 0

    def run(self):
        while True:
            self.sleep_getaway_list_before = self.sleep_getaway_list.copy()
            self.sleep_getaway_list_1_before = self.sleep_getaway_list1.copy()
            self.face_tired_list_before = self.face_tired_list.copy()
            self.face_tired_list_1_before = self.face_tired_list1.copy()

            self.frame_read1, self.sleep_getaway1, self.face_tired1,self.sleep_getaway2, self.face_tired2 = \
                self.img_infer(self.thread_num.q.get(),self.model1,device, imgsz, names, colors,320)
            print(self.frame_read1.shape)

            # 右边人员的在岗/离岗状态判断
            if self.sleep_getaway1 == 1:
                if self.fatigue_timer == 0:
                    self.fatigue_timer = time.time()
                else:
                    self.time_away = time.time() - self.fatigue_timer
                    if self.time_away >= 5:
                        self.frame_read1 = cv2AddChineseText(self.frame_read1, "睡着/离开", (400, 70), (255, 0, 0), 50)
                        self.sleep_getaway_list = [1]
                    else:
                        self.sleep_getaway_list = [0]
                    if self.sleep_getaway_list != self.sleep_getaway_list_before:
                        if self.sleep_getaway_list == [1]:
                            self.udp_socket.sendto(f"右边人员目前处于睡着或离开状态".encode('utf-8'), udp_addr)
                            self.person_online_flag_1 = 1
            else:
                if self.person_online_flag_1 == 1:
                    self.udp_socket.sendto(f"右边人员目前处于在线或到岗状态".encode('utf-8'), udp_addr)
                    self.person_online_flag_1 = 0
                self.fatigue_timer = 0

            # 右边人员的疲劳状态判断
            if self.face_tired1 == 1:
                if self.fatigue_timer1 == 0:
                    self.fatigue_timer1 = time.time()
                else:
                    self.time_tired = time.time() - self.fatigue_timer1
                    if self.time_tired >= 4:
                        self.frame_read1 = cv2AddChineseText(self.frame_read1, "疲劳", (500, 70), (255, 0, 0), 50)
                        self.face_tired_list = [1]
                    else:
                        self.face_tired_list = [0]
                    if self.face_tired_list != self.face_tired_list_before:
                        if self.face_tired_list == [1]:
                            self.udp_socket.sendto(f"右边人员目前处于疲劳状态".encode('utf-8'), udp_addr)
                            self.person_sober_flag_1 = 1
            else:
                if self.person_sober_flag_1 == 1:
                    self.udp_socket.sendto(f"右边人员目前处于清醒状态".encode('utf-8'), udp_addr)
                    self.person_sober_flag_1 = 0
                self.fatigue_timer1 = 0

            # 左边人员的在岗/离岗状态判断
            if self.sleep_getaway2 == 1:
                if self.fatigue_timer2 == 0:
                    self.fatigue_timer2 = time.time()
                else:
                    self.time_away1 = time.time() - self.fatigue_timer2
                    if self.time_away1 >= 5:
                        self.frame_read1 = cv2AddChineseText(self.frame_read1, "睡着/离开", (50, 70), (255, 0, 0), 50)
                        self.sleep_getaway_list1 = [1]
                    else:
                        self.sleep_getaway_list1 = [0]
                    if self.sleep_getaway_list1 != self.sleep_getaway_list_1_before:
                        if self.sleep_getaway_list1 == [1]:
                            self.udp_socket.sendto(f"左边人员目前处于睡着或离开状态".encode('utf-8'), udp_addr)
                            self.person_online_flag_2 = 1
            else:
                if self.person_online_flag_2 == 1:
                    self.udp_socket.sendto(f"左边人员目前处于在线或到岗状态".encode('utf-8'), udp_addr)
                    self.person_online_flag_2 = 0
                self.fatigue_timer2 = 0

            # 左边人员的疲劳状态判断
            if self.face_tired2 == 1:
                if self.fatigue_timer3 == 0:
                    self.fatigue_timer3 = time.time()
                else:
                    self.time_tired1 = time.time() - self.fatigue_timer3
                    if self.time_tired1 >= 4:
                        self.frame_read1 = cv2AddChineseText(self.frame_read1, "疲劳", (50, 70), (255, 0, 0), 50)
                        self.face_tired_list1 = [1]
                    else:
                        self.face_tired_list1 = [0]
                    if self.face_tired_list1 != self.face_tired_list_1_before:
                        if self.face_tired_list1 == [1]:
                            self.udp_socket.sendto(f"左边人员目前处于疲劳状态".encode('utf-8'), udp_addr)
                            self.person_sober_flag_2 = 1
            else:
                if self.person_sober_flag_2 == 1:
                    self.udp_socket.sendto(f"左边人员目前处于清醒状态".encode('utf-8'), udp_addr)
                    self.person_sober_flag_2 = 0
                self.fatigue_timer3 = 0


if __name__ == "__main__":
    thread_read1 = Video_receive_thread(1, "video_read1", url0, 'frame1')
    # thread_read2 = Video_receive_thread(2, "video_read2", url2, 'frame2')
    # thread_read3 = Video_receive_thread(3, "video_read3", url3, 'frame3')
    # thread_read4 = Video_receive_thread(4, "video_read4", url4, 'frame4')
    # thread_read5 = Video_receive_thread(5, "video_read5", url5, 'frame5')

    # thread_read2 = Video_receive_thread1(2, "video_read2", url2, 'frame1', 'frame2')

    thread_ana1 = Video_analyze_thread(6, "video_ana1", thread_read1, 'frame1', image_inference, model, url_select[0])
    # thread_ana2 = Video_analyze_thread(7, "video_ana2", thread_read2, 'frame2', image_inference, model)
    # thread_ana3 = Video_analyze_thread(8, "video_ana3", thread_read3, 'frame3', image_inference, model)
    # thread_ana4 = Video_analyze_thread(9, "video_ana4", thread_read4, 'frame4', image_inference, model)
    # thread_ana5 = Video_analyze_thread(10, "video_ana5", thread_read5, 'frame5', image_inference, model)

    # thread_ana2 = Video_analyze_thread(9, "video_ana2", thread_read1, 'frame1&2', image_inference, model)

    thread_read1.start()
    # thread_read2.start()
    # thread_read3.start()
    # thread_read4.start()
    # thread_read5.start()

    time.sleep(2)

    thread_ana1.start()
    # thread_ana2.start()
    # thread_ana3.start()
    # thread_ana4.start()
    # thread_ana5.start()

    cv2.namedWindow(thread_ana1.frame_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(thread_ana2.frame_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(thread_ana3.frame_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(thread_ana4.frame_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(thread_ana5.frame_name, cv2.WINDOW_NORMAL)

    while True:
        cv2.imshow(thread_ana1.frame_name, thread_ana1.frame_read)
        # cv2.imshow(thread_ana2.frame_name, thread_ana2.frame_read1)
        # cv2.imshow(thread_ana3.frame_name, thread_ana3.frame_read)
        # cv2.imshow(thread_ana4.frame_name, thread_ana4.frame_read)
        # cv2.imshow(thread_ana5.frame_name, thread_ana5.frame_read)
        # cv2.imshow(thread_ana6.frame_name, thread_ana6.frame_read)
        # cv2.imshow(thread_ana7.frame_name, thread_ana7.frame_read)
        # cv2.imshow(thread_ana8.frame_name, thread_ana8.frame_read)
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
