import cv2
import tensorrt as trt
import torch
import numpy as np
import time
from collections import OrderedDict, namedtuple
import threading
import Jetson.GPIO as GPIO

LABELS = ["face", "close_eye", "open_eye", "close_mouth", "open_mouth"]

# 设置 GPIO 模式为 BOARD
GPIO.setmode(GPIO.BOARD)

# 定义 led 要使用的 GPIO 引脚号
led_pins = [12, 16, 18, 22]

# 定义蜂鸣器使用的 GPIO 引脚号
buzzer_pin = 24

# 设置 led 引脚为输出模式
for pin in led_pins:
    GPIO.setup(pin, GPIO.OUT)


# 设置蜂鸣器引脚为输出模式
GPIO.setup(buzzer_pin, GPIO.OUT)

# 默认设置为 HIGH，这样低电平触发的蜂鸣器不会响起
GPIO.output(buzzer_pin, GPIO.HIGH)

# 创建一个蜂鸣器报警的线程
buzzer_thread = threading.Thread()
led_thread = threading.Thread()

# 定义一个事件，用于在主线程中通知子线程停止报警
stop_buzzer = threading.Event()
stop_led = threading.Event()


class TRT_engine():
    def __init__(self, weight):
        self.imgsz = [640, 640]
        self.weight = weight
        self.device = torch.device('cuda:0')
        self.init_engine()

    def init_engine(self):
        # Infer TensorRT Engine
        self.Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        self.logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(self.weight, 'rb') as self.f, trt.Runtime(self.logger) as self.runtime:
            self.model = self.runtime.deserialize_cuda_engine(self.f.read())
        self.bindings = OrderedDict()
        self.fp16 = False
        for index in range(self.model.num_bindings):
            self.name = self.model.get_binding_name(index)
            self.dtype = trt.nptype(self.model.get_binding_dtype(index))
            self.shape = tuple(self.model.get_binding_shape(index))
            self.data = torch.from_numpy(np.empty(self.shape, dtype=np.dtype(self.dtype))).to(self.device)
            self.bindings[self.name] = self.Binding(self.name, self.dtype, self.shape, self.data,
                                                    int(self.data.data_ptr()))
            if self.model.binding_is_input(index) and self.dtype == np.float16:
                self.fp16 = True
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = self.model.create_execution_context()

    def letterbox(self, im, color=(114, 114, 114), auto=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        new_shape = self.imgsz
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # Scale ratio (new / old)
        self.r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            self.r = min(self.r, 1.0)
        # Compute padding
        new_unpad = int(round(shape[1] * self.r)), int(round(shape[0] * self.r))
        self.dw, self.dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            self.dw, self.dh = np.mod(self.dw, stride), np.mod(self.dh, stride)  # wh padding
        self.dw /= 2  # divide padding into 2 sides
        self.dh /= 2
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(self.dh - 0.1)), int(round(self.dh + 0.1))
        left, right = int(round(self.dw - 0.1)), int(round(self.dw + 0.1))
        self.img = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return self.img, self.r, self.dw, self.dh

    def preprocess(self, image):
        self.img, self.r, self.dw, self.dh = self.letterbox(image)
        self.img = self.img.transpose((2, 0, 1))
        self.img = np.expand_dims(self.img, 0)
        self.img = np.ascontiguousarray(self.img)
        self.img = torch.from_numpy(self.img).to(self.device)
        self.img = self.img.float()
        return self.img

    def predict(self, img, threshold):
        img = self.preprocess(img)
        self.binding_addrs['images'] = int(img.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        nums = self.bindings['num_dets'].data[0].tolist()
        boxes = self.bindings['det_boxes'].data[0].tolist()
        scores = self.bindings['det_scores'].data[0].tolist()
        classes = self.bindings['det_classes'].data[0].tolist()
        num = int(nums[0])
        new_bboxes = []
        face_status = []
        for i in range(num):
            if scores[i] < threshold:
                continue
            xmin = (boxes[i][0] - self.dw) / self.r
            ymin = (boxes[i][1] - self.dh) / self.r
            xmax = (boxes[i][2] - self.dw) / self.r
            ymax = (boxes[i][3] - self.dh) / self.r
            new_bboxes.append([classes[i], scores[i], xmin, ymin, xmax, ymax])
            face_status.append(int(classes[i]))

        sleep_getaway = None
        face_tired = None
        # 睡岗/离岗状态（若是没有检测到人脸或只检测到人脸）
        if (len(face_status) == 1 and 0 in face_status) or (len(face_status) == 0):
            sleep_getaway = 1
        else:
            sleep_getaway = 0
        # 判断疲劳状态（若是判断面部信息>1，如检测到眼睛，嘴巴则执行下面代码）
        if len(face_status) > 1:
            if (1 in face_status) or (4 in face_status):  # 若是检测到眼睛有处于闭合状态或嘴巴张开时，则启动定时器进行计时；
                face_tired = 1
            else:
                face_tired = 0

        return new_bboxes, sleep_getaway, face_tired


def visualize(img, bbox_array):
    for temp in bbox_array:
        xmin = int(temp[2])
        ymin = int(temp[3])
        xmax = int(temp[4])
        ymax = int(temp[5])
        clas = int(temp[0])
        score = temp[1]

        label_name = LABELS[clas]

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (105, 237, 249), 2)

        img = cv2.putText(img, "{} {:.2f}".format(label_name, score), (xmin, int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                          0.5,
                          (105, 237, 249), 1)

        # img = cv2.putText(img, "class:"+str(clas)+" "+str(round(score,2)), (xmin,int(ymin)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (105, 237, 249), 1)
    return img


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


trt_engine = TRT_engine("/home/xuzhi/Downloads/YOLOv7_Tensorrt-master/tied_detection1.engine")

cap = cv2.VideoCapture(0)
cv2.namedWindow("Frame",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame",2560,1440)

fatigue_level = 0
sleep_getaway_active = False
face_tired_active = False


fatigue_timer = None
fatigue_timer1 = None


while True:
    ret, frame = cap.read()
    if not ret:
        continue

    t1 = time.time()

    results, sleep_getaway, face_tired = trt_engine.predict(frame, 0.5)
    frame = visualize(frame, results)

    if sleep_getaway == 1:
        if fatigue_timer == None:
            fatigue_timer = time.time()
        else:
            time_away = time.time() - fatigue_timer
            if time_away >= 5:
                frame = cv2.putText(frame, "Sleeping/Leaving", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 0, 255], 2)
                if not led_thread.is_alive():
                    led_thread = threading.Thread(target=control_led, args=(4,))
                    led_thread.start()
                if not buzzer_thread.is_alive():
                    stop_buzzer.clear()  # clear the event before starting a new thread
                    buzzer_thread = threading.Thread(target=buzzer_alert)
                    buzzer_thread.start()
    else:
        fatigue_timer = None
        stop_led.set()
        stop_buzzer.set()

    if face_tired == 1:
        if fatigue_timer1 == None:
            fatigue_timer1 = time.time()
        else:
            time_tired = time.time() - fatigue_timer1
            if time_tired >= 2 and time_tired < 4:
                frame = cv2.putText(frame, "Fatigue Level 1", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 0, 255], 2)
                if not led_thread.is_alive():
                    led_thread = threading.Thread(target=control_led, args=(1,))
                    led_thread.start()
                if not buzzer_thread.is_alive():
                    stop_buzzer.clear()  # clear the event before starting a new thread
                    buzzer_thread = threading.Thread(target=buzzer_alert)
                    buzzer_thread.start()
            elif time_tired >= 4 and time_tired < 6:
                frame = cv2.putText(frame, "Fatigue Level 2", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 0, 255], 2)
                if not led_thread.is_alive():
                    led_thread = threading.Thread(target=control_led, args=(2,))
                    led_thread.start()
                if not buzzer_thread.is_alive():
                    stop_buzzer.clear()  # clear the event before starting a new thread
                    buzzer_thread = threading.Thread(target=buzzer_alert)
                    buzzer_thread.start()
            elif time_tired >= 6 and time_tired < 8:
                frame = cv2.putText(frame, "Fatigue Level 3", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 0, 255], 2)
                if not led_thread.is_alive():
                    led_thread = threading.Thread(target=control_led, args=(3,))
                    led_thread.start()
                if not buzzer_thread.is_alive():
                    stop_buzzer.clear()
                    buzzer_thread = threading.Thread(target=buzzer_alert)
                    buzzer_thread.start()
            elif time_tired >= 8:
                frame = cv2.putText(frame, "Fatigue Level 4", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 0, 255], 2)
                if not led_thread.is_alive():
                    led_thread = threading.Thread(target=control_led, args=(4,))
                    led_thread.start()
                if not buzzer_thread.is_alive():
                    stop_buzzer.clear()
                    buzzer_thread = threading.Thread(target=buzzer_alert)
                    buzzer_thread.start()
    else:
        fatigue_timer1 = None
        stop_led.set()
        stop_buzzer.clear()

    t2 = time.time()

    print("time:", t2 - t1)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
