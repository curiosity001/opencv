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


# Load model
weights = 'person.pt'
device = select_device('0')
half = device.type != 'cpu'  # half precision only supported on CUDA
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]


# udp_addr = ('192.168.31.49', 9999)
# udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

if half:
    model.half()

url1 = 0
# url1='rtsp://admin:Aust12345@192.168.31.62:554/live'
# url2='rtsp://admin:Aust12345@192.168.31.59:554/live'
# url3='rtsp://admin:Aust12345@192.168.31.60:554/live'
# url4='rtsp://admin:Aust12345@192.168.31.62:554/live'
# url5='rtsp://admin:Aust12345@192.168.31.63:554/live'
# url6='rtsp://admin:Aust12345@192.168.31.68:554/live'
# url7='rtsp://admin:Aust12345@192.168.31.68:554/live'
# url8='rtsp://admin:Aust12345@192.168.31.68:554/live'


def image_inference(frame, model1, device, half, imgsz, names, colors):

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
            path_area = Path(selected_area)
            for *xyxy, conf, cls in reversed(pred):
                if path_area.contains_point(
                        (int(xyxy[0] + (xyxy[2] - xyxy[0]) / 2), int(xyxy[1] + (xyxy[3] - xyxy[1]) / 2))):
                    sign[i] = 1
                    break
    return frame, sign


class Video_receive_thread(threading.Thread):
    def __init__(self, threadID, name, url, frame_name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.url = url
        self.frame_name = frame_name
        self.frame = None
        self.cap = cv2.VideoCapture(self.url)
        self.q = Queue()
    def run(self):
        while True:
            self.ret, self.frame = self.cap.read()
            self.q.put(self.frame)
            if self.q.qsize() > 2:# 如果队列中的帧数大于1，表示队列已满，需要移除最早的一帧
                self.q.get()
            if self.ret == False:
                self.cap = cv2.VideoCapture(self.url)
                continue

class Video_analyze_thread(threading.Thread):
    def __init__(self, threadID, name, thread_num, frame_name, img_infer, model):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.frame_read = self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
        self.thread_num = thread_num
        self.frame_name = frame_name
        self.img_infer = img_infer
        self.model = model
        self.sign = 0

    def run(self):
        while True:
            self.frame_read, self.sign = self.img_infer(self.thread_num.q.get(), self.model, device, half, imgsz,
                                                        names,colors)




if __name__ == "__main__":
    thread_read1 = Video_receive_thread(1, "video_read1", url1, 'frame1')
    # thread_read2 = Video_receive_thread(2, "video_read2", url2, 'frame2')
    # thread_read3 = Video_receive_thread(3, "video_read3", url3, 'frame3')
    # thread_read4 = Video_receive_thread(4, "video_read4", url4, 'frame4')
    # thread_read5 = Video_receive_thread(5, "video_read5", url5, 'frame5')
    # thread_read6 = Video_receive_thread(6, "video_read6", url6, 'frame6')
    # thread_read7 = Video_receive_thread(7, "video_read7", url7, 'frame7')
    # thread_read8 = Video_receive_thread(8, "video_read8", url8, 'frame8')
    thread_ana1 = Video_analyze_thread(1, "video_ana1", thread_read1, 'frame1', image_inference, model)
    # thread_ana2 = Video_analyze_thread(2, "video_ana2", thread_read2, 'frame2', image_inference, model, select_list=[url_62_m4, url_62_m6])
    # thread_ana3 = Video_analyze_thread(3, "video_ana3", thread_read3, 'frame3', image_inference, model, select_list=[url_62_m4, url_62_m6])
    # thread_ana4 = Video_analyze_thread(4, "video_ana4", thread_read4, 'frame4', image_inference, model, select_list=[url_62_m4, url_62_m6])
    # thread_ana5 = Video_analyze_thread(5, "video_ana5", thread_read5, 'frame5', image_inference, model, select_list=[url_62_m4, url_62_m6])
    # thread_ana6 = Video_analyze_thread(6, "video_ana6", thread_read6, 'frame6', image_inference, model, select_list=[url_62_m4, url_62_m6])
    # thread_ana7 = Video_analyze_thread(7, "video_ana7", thread_read7, 'frame7', image_inference, model, select_list=[url_62_m4, url_62_m6])
    # thread_ana8 = Video_analyze_thread(8, "video_ana8", thread_read8, 'frame8', image_inference, model, select_list=[url_62_m4, url_62_m6])

    thread_read1.start()
    # thread_read2.start()
    # thread_read3.start()
    # thread_read4.start()
    # thread_read5.start()
    # thread_read6.start()
    # thread_read7.start()
    # thread_read8.start()
    time.sleep(1)
    thread_ana1.start()
    # thread_ana2.start()
    # thread_ana3.start()
    # thread_ana4.start()
    # thread_ana5.start()
    # thread_ana6.start()
    # thread_ana7.start()
    # thread_ana8.start()


    cv2.namedWindow(thread_ana1.frame_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(thread_ana2.frame_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(thread_ana3.frame_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(thread_ana4.frame_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(thread_ana5.frame_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(thread_ana6.frame_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(thread_ana7.frame_name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(thread_ana8.frame_name, cv2.WINDOW_NORMAL)

    while True:
        cv2.imshow(thread_ana1.frame_name, thread_ana1.frame_read)
        # cv2.imshow(thread_ana2.frame_name, thread_ana2.frame_read)
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
    # thread_read6.join()
    # thread_read7.join()
    # thread_read8.join()
    # thread_ana1.join()
    # thread_ana2.join()
    # thread_ana3.join()
    # thread_ana4.join()
    # thread_ana5.join()
    # thread_ana6.join()
    # thread_ana7.join()
    # thread_ana8.join()
