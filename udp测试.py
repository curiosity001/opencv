import sys

from matplotlib.path import Path
import cv2
from numpy import random
import time
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import numpy as np
import torch
import threading
import socket
from queue import Queue
import subprocess as sp
import requests
import json
import base64
import datetime
import ast

weights = 'best456.pt'
device = select_device('0')
half = device.type != 'cpu'  # half precision only supported on CUDA
# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model

stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

udp_addr = ('192.168.65.201', 9989)
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

if half:
    model.half()

J1_m03 = [(524,101),(524,912),(1445,911),(1445,100)]  #4.166
J2_m03 = [(730,6),(730,268),(1134,266),(1132,6)] #4.247
J2_m04 = [(680,327),(159,1058),(1441,1061),(1167,326)] #4.247
J2_n03 = [(876,5),(876,351),(1447,350),(1477,6)] #4.249
J2_n04 = [(777,418),(357,1060),(1719,1061),(1545,455)] #4.249


url1 = 'rtsp://admin:hy123456@192.168.4.166:554/live' # 1920*1080
url2 = 'rtsp://admin:cs123456@192.168.4.247:554/live'# 1920*1080
url3 = 'rtsp://admin:cs123456@192.168.4.249:554/live'# 1920*1080


rtmpUrl1 = "rtmp://192.168.65.201:1935/live/1"
rtmpUrl2 = "rtmp://192.168.65.201:1935/live/2"
rtmpUrl3 = "rtmp://192.168.65.201:1935/live/3"


event_url = 'http://192.168.65.205:8081/getmessage/responseAlarm'

video_que1 = Queue()
video_que2 = Queue()
video_que3 = Queue()
data_queue = Queue()



width1 = int(1920)
height1 = int(1080)

fps = int(0)

command1 = ['ffmpeg',  # linux不用指定
            '-hwaccel', 'cuvid',
            '-y', '-an',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',  # 像素格式
            '-s', "{}x{}".format(width1, height1),
            '-r', str(fps),  # 自己的摄像头的fps是0，若用自己的notebook摄像头，设置为15、20、25都可。
            '-i', '-',
            '-c:v', 'h264_nvenc',  # 视频编码方式
            '-pix_fmt', 'yuv420p',
            # '-preset', 'ultrafast',
            '-f', 'flv',  # flv rtsp
            # '-rtmp_transport', 'tcp',  # 使用TCP推流，linux中一定要有这行
            rtmpUrl1]  # rtsp rtmp
command2 = ['ffmpeg',  # linux不用指定
            '-hwaccel', 'cuvid',
            '-y', '-an',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',  # 像素格式
            '-s', "{}x{}".format(width1, height1),
            '-r', str(fps),  # 自己的摄像头的fps是0，若用自己的notebook摄像头，设置为15、20、25都可。
            '-i', '-',
            '-c:v', 'h264_nvenc',  # 视频编码方式
            '-pix_fmt', 'yuv420p',
            # '-preset', 'ultrafast',
            '-f', 'flv',  # flv rtsp
            # '-rtmp_transport', 'tcp',  # 使用TCP推流，linux中一定要有这行
            rtmpUrl2]  # rtsp rtmp
command3 = ['ffmpeg',  # linux不用指定
            '-hwaccel', 'cuvid',
            '-y', '-an',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',  # 像素格式
            '-s', "{}x{}".format(width1, height1),
            '-r', str(fps),  # 自己的摄像头的fps是0，若用自己的notebook摄像头，设置为15、20、25都可。
            '-i', '-',
            '-c:v', 'h264_nvenc',  # 视频编码方式
            '-pix_fmt', 'yuv420p',
            # '-preset', 'ultrafast',
            '-f', 'flv',  # flv rtsp
            # '-rtmp_transport', 'tcp',  # 使用TCP推流，linux中一定要有这行
            rtmpUrl3]  # rtsp rtmp

pipe1 = sp.Popen(command1, shell=False, stdin=sp.PIPE)
pipe2 = sp.Popen(command2, shell=False, stdin=sp.PIPE)
pipe3 = sp.Popen(command3, shell=False, stdin=sp.PIPE)

areas_names = {key: value for key, value in globals().items() if value in (J1_m03, J2_m03, J2_m04, J2_n03, J2_n04)}


def events_trans(url, data_dict):
    res = requests.post(url=url, json=data_dict, timeout=5)
    content_str = str(res.content, encoding="utf-8")
    print(content_str)

# todo image_inference
def image_inference(frame, model1, device, half, imgsz, names, colors, area_list):
    base_x, base_y, step_y = 50, 50, 50  # 初始坐标和步长
    next_y = base_y  # 下一个文本的y坐标

    for area in area_list:
        for i in range(len(area)):
            start_point = area[i]
            end_point = area[(i + 1) % len(area)]
            frame = cv2.line(frame, start_point, end_point, (0, 255, 0), 4)

    img = letterbox(frame, imgsz, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0

    sign = [0] * len(area_list)

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

        for i, selected_area in enumerate(area_list):
            area_name = next(key for key, value in areas_names.items() if value == area_list[i])
            path_area = Path(selected_area)
            for *xyxy, conf, cls in reversed(pred):
                if path_area.contains_point(
                        (int(xyxy[0] + (xyxy[2] - xyxy[0]) / 2), int(xyxy[1] + (xyxy[3] - xyxy[1]) / 2))):
                    sign[i] = 1
                    text_pos = (base_x, next_y)
                    next_y += step_y
                    cv2.putText(frame, f"{area_name}_Person", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                lineType=cv2.LINE_AA)
                    break
    return frame, sign


# todo Video_receive_thread
class Video_receive_thread(threading.Thread):
    def __init__(self, threadID, name, url, frame_name, queue):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.url = url
        self.frame_name = frame_name
        self.frame = None
        self.cap = cv2.VideoCapture(self.url)
        self.queue = queue
        # self.video_flag = True

    def run(self):
        while True:
            try:
                self.ret, self.frame = self.cap.read()
                # print(self.frame.shape)
                self.queue.put(self.frame)
                if self.queue.qsize() > 1:  # 如果队列中的帧数大于1，表示队列已满，需要移除最早的一帧
                    self.queue.get()
                if self.ret == False:
                    # self.video_flag=False
                    self.cap = cv2.VideoCapture(self.url)
                    continue
            except:
                time.sleep(5)
            # else:
            # self.video_flag = True

# todo Video_analyze_thread
class Video_analyze_thread(threading.Thread):
    def __init__(self, threadID, name, thread_num, frame_name, img_infer, model, queue, pipe, video_read_thread,
                 camera_id, select_list):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        # self.frame_read = None
        self.frame_read = self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
        self.thread_num = thread_num
        self.frame_name = frame_name
        self.img_infer = img_infer
        self.model = model
        self.select_list = select_list
        self.udp_socket = udp_socket
        self.sign_before = [0] * len(self.select_list)
        self.sign = [0] * len(self.select_list)
        self.start_time = [None] * len(self.select_list)  # 每个区域的计时器
        self.count = 0
        self.queue = queue
        self.pipe = pipe
        self.camera_id = camera_id
        self.video_read_thread = video_read_thread
        self.is_running = True

        self.data_dict = {
            'positionid': '0001',  # 煤矿ID
            'cameraid': self.camera_id,  # 摄像头ID
            'starttime': '',
            'endtime': '',
            'currenttimes': '',  # 报警触发时间
            'eventtype': '12001',  # 事件类型
            'image': "0",
            'taskid': '12',  # 任务ID
            'mask': "区域闯入"  # 报警事件描述
        }

    def run(self):

        for n in range(len(self.select_list)):
            area_name = next(key for key, value in areas_names.items() if value == self.select_list[n])
            self.udp_socket.sendto(f"{area_name}_{0}".encode('utf-8'), udp_addr)

        while self.is_running:
            if self.queue.qsize() == 0:
                time.sleep(0.15)
                # time.sleep(5)

                self.count += 1
                # self.frame_read = np.random.rand(1080, 1920, 3) * 255
                if self.count == 20:
                    self.sign = [0] * len(self.select_list)
                    self.count = 0
                    for n in range(len(self.select_list)):
                        area_name = next(key for key, value in areas_names.items() if value == self.select_list[n])
                        self.udp_socket.sendto(f"{area_name}_{0}".encode('utf-8'), udp_addr)

                self.pipe.stdin.write(self.frame_read.tostring())
            else:
                try:
                    self.count = 0
                    self.sign_before = self.sign.copy()
                    self.frame_read, self.sign = self.img_infer(self.queue.get(), self.model, device, half, imgsz,
                                                                names,
                                                                colors, area_list=self.select_list)
                    for n, status in enumerate(self.sign):
                        if self.sign_before[n] != status:
                            # 如果计时器还没开始，就开始计时
                            if self.start_time[n] is None:
                                self.start_time[n] = time.time()
                            # 如果计时器已经开始，说明在2秒内状态又改变了，重置计时器
                            else:
                                self.start_time[n] = None

                    for n in range(len(self.select_list)):
                        # 如果计时器正在计时
                        if self.start_time[n] is not None:
                            elapsed_time = time.time() - self.start_time[n]
                            area_name = next(key for key, value in areas_names.items() if value == self.select_list[n])
                            # 时间判断
                            # if self.sign[n] == 1:  # 进入区域
                            if self.sign[n] == 1:  # 进入区域
                                self.udp_socket.sendto(f"{area_name}_{self.sign[n]}".encode('utf-8'), udp_addr)
                                self.start_time[n] = None

                                self.str_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                self.data_dict['currenttimes'] = self.str_time
                                threading.Thread(target=events_trans, args=(event_url, self.data_dict),
                                                 daemon=True).start()

                            if self.sign[n] == 0:  # 离开区域
                                if elapsed_time >= 2:
                                    self.udp_socket.sendto(f"{area_name}_{self.sign[n]}".encode('utf-8'), udp_addr)
                                    self.start_time[n] = None  # 重置计时器

                    self.pipe.stdin.write(self.frame_read.tostring())
                except:
                    time.sleep(0.15)
                    print("data error")
                    # self.frame_read = np.random.rand(1080, 1920, 3) * 255
                    self.pipe.stdin.write(self.frame_read.tostring())

    def stop(self):
        self.is_running = False

class UDPReceiver:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind((self.ip, self.port))

    def start(self):
        print("UDP 服务器已启动，等待接收数据...")
        self.receive_thread = threading.Thread(target=self._udp_receive)
        self.receive_thread.start()

    def _udp_receive(self):
        while True:
            data, addr = self.udp_socket.recvfrom(1024)
            print(f"收到来自 {addr} 的数据: {data.decode('utf-8')}")
            data_queue.put(data.decode('utf-8'))
            time.sleep(1)



# todo main
if __name__ == "__main__":
    thread_read1 = Video_receive_thread(1, "video_read1", url1, 'frame1', video_que1)
    thread_read2 = Video_receive_thread(2, "video_read2", url2, 'frame2', video_que2)
    thread_read3 = Video_receive_thread(3, "video_read3", url3, 'frame3', video_que3)
    # thread_read4 = Video_receive_thread(4, "video_read4", url4, 'frame4', video_que4)
    # thread_read5 = Video_receive_thread(5, "video_read5", url5, 'frame5', video_que5)
    # thread_read6 = Video_receive_thread(6, "video_read6", url6, 'frame6', video_que6)

    thread_ana1 = Video_analyze_thread(7, "video_ana1", thread_read1, 'frame1', image_inference, model, video_que1, pipe1, thread_read1, 166, select_list=[J1_m03])
    thread_ana2 = Video_analyze_thread(8, "video_ana2", thread_read2, 'frame2', image_inference, model, video_que2, pipe2, thread_read2, 247, select_list=[J2_m03, J2_m04])
    thread_ana3 = Video_analyze_thread(9, "video_ana3", thread_read3, 'frame3', image_inference, model, video_que3, pipe3, thread_read3, 249, select_list=[J2_n03, J2_n04])
    # thread_ana4 = Video_analyze_thread(10, "video_ana4", thread_read4, 'frame4', image_inference, model, video_que4, pipe4, thread_read4, 65, select_list=[J3_m05, J1_m09])
    # thread_ana5 = Video_analyze_thread(11, "video_ana5", thread_read5, 'frame5', image_inference, model, video_que5, pipe5, thread_read5, 139, select_list=[J3_m05, J1_m09])
    # thread_ana6 = Video_analyze_thread(12, "video_ana6", thread_read6, 'frame6', image_inference, model, video_que6, pipe6, thread_read6, 140, select_list=[J3_m05, J1_m09])

    udp_receiver = UDPReceiver("127.0.0.1", 8080)
    udp_receiver.start()
    time.sleep(1)

    thread_read1.start()
    thread_read2.start()
    thread_read3.start()
    # thread_read4.start()
    # thread_read5.start()
    # thread_read6.start()

    time.sleep(1)

    thread_ana1.start()
    thread_ana2.start()
    thread_ana3.start()
    # thread_ana4.start()
    # thread_ana5.start()
    # thread_ana6.start()

    while True:
        time.sleep(1)
        if not data_queue.empty():
            data = data_queue.get()
            data_queue.task_done()
            m4 = ast.literal_eval(data)
            thread_ana1.stop()
            time.sleep(1)
            thread_ana1 = Video_analyze_thread(7, "video_ana1", thread_read1, 'frame1', image_inference, model,
                                               video_que1, pipe1, thread_read1, 166, select_list=[J1_m03])
            thread_ana1.start()

    # cv2.namedWindow(thread_ana1.frame_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(thread_ana2.frame_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(thread_ana3.frame_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(thread_ana4.frame_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(thread_ana5.frame_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(thread_ana6.frame_name, cv2.WINDOW_NORMAL)

    # while True:
        # cv2.imshow(thread_ana1.frame_name, thread_ana1.frame_read)
        # cv2.imshow(thread_ana2.frame_name, thread_ana2.frame_read)
        # cv2.imshow(thread_ana3.frame_name, thread_ana3.frame_read)
    #     cv2.imshow(thread_ana4.frame_name, thread_ana4.frame_read)
    #     cv2.imshow(thread_ana5.frame_name, thread_ana5.frame_read)
        # cv2.imshow(thread_ana6.frame_name, thread_ana6.frame_read)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()

    thread_read1.join()
    thread_read2.join()
    thread_read3.join()
    # thread_read4.join()
    # thread_read5.join()
    # thread_read6.join()

    thread_ana1.join()
    thread_ana2.join()
    thread_ana3.join()
    # thread_ana4.join()
    # thread_ana5.join()
    # thread_ana6.join()
