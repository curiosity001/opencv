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

metalocA = threading.Lock()
weights = 'person.pt'
device = select_device('0')
half = device.type != 'cpu'  # half precision only supported on CUDA
# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model

stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

udp_addr = ('192.168.31.47', 9999)
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

if half:
    model.half()

url1 = 0

# m1 = [(788, 9), (788, 278), (1108, 278), (1107, 10)]
# m2 = [(778, 291), (448, 1061), (1457, 1061), (1136, 295)]
# m3 = [(524, 101), (524, 912), (1445, 911), (1445, 100)]
m4 = [(20, 510), (20, 380), (300, 380), (300, 50)]
m5 = [(350, 110), (350, 400), (600, 400), (600, 10)]
# m6 = [(10, 30), (10, 400), (600, 400), (600, 30)]


def image_inference(frame, model1, device, half, imgsz, names, colors, area_list):
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

    def run(self):
        while True:
            self.ret, self.frame = self.cap.read()
            if self.ret == False:
                self.cap = cv2.VideoCapture(self.url)
                continue
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


class Video_analyze_thread(threading.Thread):
    def __init__(self, threadID, name, thread_num, frame_name, img_infer, model, select_list):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.frame_read = None
        self.thread_num = thread_num
        self.frame_name = frame_name
        self.img_infer = img_infer
        self.model = model
        self.select_list = select_list
        self.udp_socket = udp_socket
        self.sign_before = [0] * len(self.select_list)
        self.sign = [0] * len(self.select_list)


    def run(self):
        while True:
            self.sign_before = self.sign.copy()

            self.frame_read, self.sign = self.img_infer(self.thread_num.frame, self.model, device, half, imgsz, names,
                                                        colors, area_list=self.select_list)



            for n, status in enumerate(self.sign):
                if self.sign_before[n] != status:
                    self.udp_socket.sendto(f"index_{n}_changed".encode('utf-8'), udp_addr)

            cv2.imshow(self.frame_name, self.frame_read)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    thread_read1 = Video_receive_thread(1, "video_read1", url1, 'frame1')
    thread_ana1 = Video_analyze_thread(1, "video_ana1", thread_read1, 'frame1', image_inference, model,
                                       select_list=[m4, m5])

    thread_read1.start()
    time.sleep(2)
    thread_ana1.start()

    thread_read1.join()
    thread_ana1.join()
