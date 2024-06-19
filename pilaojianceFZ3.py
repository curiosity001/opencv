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

weights = 'tied_detection.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
url1 = 0
video_weight = 1920
video_hight = 1080


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

fatigue_timer = 0
fatigue_timer1 = 0

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

#定义全局定时器初始值fatigue_timer = 0 ， fatigue_timer1 = 0
    global fatigue_timer
    global fatigue_timer1

#睡岗/离开状态（若是没有检测到人脸或只检测到人脸）
    if (len(face_status) == 1 and 0 in face_status) or (len(face_status) == 0):
        if fatigue_timer == 0:
            fatigue_timer = time.time()
        else:
            time_tied = time.time() - fatigue_timer
            if time_tied > 5:
                frame = cv2AddChineseText(frame, "睡着/离开", (50, 70), (255, 0, 0), 50)
    else:
        fatigue_timer = 0

#判断疲劳状态（若是判断面部信息>1，如检测到眼睛，嘴巴则执行下面代码）
    if len(face_status) > 1:
        if (1 in face_status) or (4 in face_status):#若是检测到眼睛有处于闭合状态，则启动定时器进行计时；
            if fatigue_timer1 == 0:
                fatigue_timer1 = time.time()
            else:
                time_tied = time.time() - fatigue_timer1
                if time_tied >= 3:
                    frame= cv2AddChineseText(frame, "疲劳", (50, 70), (255, 0, 0), 50)
        else:
            fatigue_timer1 = 0
    return frame, face_status



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
thread_read1 = Video_receive_thread(1, "video_read1", url1, 'frame1')


# 不分级报警
class Video_analyze_thread(threading.Thread):
    def __init__(self, threadID, name, thread_num, frame_name, img_infer, model):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.frame_read = None
        self.thread_num = thread_num
        self.frame_name = frame_name
        self.img_infer = img_infer
        self.model1 = model
        self.face_status = []

    def run(self):
        while True:
            self.frame_read, self.face_status = self.img_infer(self.thread_num.frame, self.model1, device, imgsz, names, colors)
            cv2.imshow(self.frame_name, self.frame_read)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    thread_ana1 = Video_analyze_thread(8, "video_ana1", thread_read1, 'frame1', image_inference, model)
    thread_read1.start()
    time.sleep(2)
    thread_ana1.start()
    thread_read1.join()
    thread_ana1.join()