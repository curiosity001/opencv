import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from cv2.gapi.streaming import timestamp
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
from threading import Timer
weights='last.pt'
device = select_device('cpu')
# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model

stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# cap=cv2.VideoCapture('3.mp4')
url1='6.mp4'
video_weight=1920
video_hight=1080

start_timer = time.time()
# Execute task
time.sleep(15)
timer = time.time() - start_timer
def image_inference(frame, model, device, imgsz, names, colors):
    t1 = time.time()
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
        startTime = 0
        timeDiff = 0
        for *xyxy, conf, cls in reversed(pred):
            label = f'{names[int(cls)]} {conf:.2f}'
            print(label)
            plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
            if label[0:8] == "open_eye":
                timer = 0  # 重新计时
            elif label[0:9] == "close_eye":
                start_timer = timestamp
            elif (timestamp - start_timer) >= 5 and (timestamp - start_timer) < 15:
                fatigue_level = 1
                print("Fatigue level one")
            elif (timestamp - start_timer) >= 15 and (timestamp - start_timer) < 30:
                fatigue_level = 2
                print("Fatigue level two")
            elif (timestamp - start_timer) >= 30 and (timestamp - start_timer) < 45:
                fatigue_level = 3
                print("Fatigue level three")
            elif (timestamp - start_timer) >= 45 and (timestamp - start_timer) < 60:
                fatigue_level = 4
                print("Fatigue level four")
            elif (timestamp - start_timer) >= 60:
                fatigue_level = 5
                print("you are sleeping")
                break# 进入睡岗或离岗状态

    t2 = time.time()
    FPS = int(1 / (t2 - t1))
    cv2.putText(frame, 'FPS=%s' % FPS, (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
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
        self.model1=model

    def run(self):
        while True:
            try:
                self.frame_read = self.img_infer(self.thread_num.frame, self.model1, device, imgsz, names, colors)
            except:
                # self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
                self.frame_read = np.random.rand(1080,1920,3)*255
                print("Data Error",self.thread_num)
            #self.frame_read=cv2.resize(self.frame_read,(1920,1080))
            # print(self.frame_read.shape)
            cv2.imshow(self.frame_name, self.frame_read)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    thread_ana1 = Video_analyze_thread(9, "video_ana1", thread_read1, 'frame1', image_inference, model)
    thread_read1.start()
    thread_ana1.start()


