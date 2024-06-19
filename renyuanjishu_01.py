#1线程读取H.264码流 2线程处理数据
#import panda as pd
import time
from pathlib import Path
import cv2
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
from queue import Queue

weights2='person.pt'
device = select_device('0')
half = device.type != 'cpu'  # half precision only supported on CUDA
model2 = attempt_load(weights2, map_location=device)  # load FP32 model
stride = int(model2.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size
names = model2.module.names if hasattr(model2, 'module') else model2.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))

video_queue1=Queue()
video_queue2=Queue()

url1=0

#视频大小
video_weight = 1920
video_hight = 1080

def image_inference(frame, model, device, imgsz, names, colors):
    t1 = time.time()
    status = []
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
            status.append(int(cls))
    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, 'FPS=%s' % FPS, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 0, 0], 2, lineType=cv2.LINE_AA)
    return frame

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
            self.ret, self.frame = self.cap.read()
            self.q.put(self.frame)
            if self.q.qsize() > 1:  # 如果队列中的帧数大于1，表示队列已满，需要移除最早的一帧
                self.q.get()
            if self.ret == False:
                self.cap = cv2.VideoCapture(self.url)
                continue

class Video_analyze_thread(threading.Thread):
    def __init__(self, threadID, name,thread_num,frame_name,img_infer,queue):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
        self.thread_num=thread_num
        self.frame_name=frame_name
        self.img_infer=img_infer
        self.q = queue
        # 计数相关变量
        self.count = 0  # 进入电梯的人员数量
        self.entered_elevator = set()  # 用于记录已经进入电梯的人员ID（避免重复计数）
    def run(self):
        while True:
            frame = self.q.get()
            self.frame_read = self.img_infer(self.q.get(), model2, device, imgsz, names, colors)
            # 计数逻辑：检查检测到的目标框的中心点位置，只考虑status为0的全身检测框
            for *xyxy, _, cls in reversed(model2(frame, size=imgsz)[0]):
                if cls == 0:  # 只考虑status为0的全身检测框
                    x_center = (xyxy[0] + xyxy[2]) / 2
                    y_center = (xyxy[1] + xyxy[3]) / 2
                    if y_center < frame.shape[0] // 2:  # 在图像的上半部分，判断为人员进入
                        person_id = int(cls)
                        if person_id not in self.entered_elevator:
                            self.count += 1
                            self.entered_elevator.add(person_id)
if __name__ == "__main__":
    thread_read1 = Video_receive_thread(1, "video_read1", url1, 'frame1', video_queue1)
    thread_ana1 = Video_analyze_thread(5, "video_ana1",thread_read1,'frame1',image_inference,video_queue1)
    thread_read1.start()
    thread_ana1.start()
    cv2.namedWindow(thread_ana1.frame_name, cv2.WINDOW_NORMAL)
    while True:
        cv2.imshow(thread_ana1.frame_name, thread_ana1.frame_read)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    thread_read1.join()
    thread_ana1.join()


