import time
from pathlib import Path
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

pygame.mixer.init()#初始化
# master=mt.TcpMaster("10.33.83.147",999) #向某个地址发送数据
# weights='YOLO-fs.pt'
weights='tied_detection.pt'
# device = select_device('cpu')7
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model

stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# cap=cv2.VideoCapture('3.mp4')
url1=0
# url1='3.mp4'
video_weight=1920
video_hight=1080

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

def image_inference(frame,model,device,imgsz,names,colors):  #定义一个函数image_inference，它接受6个参数：frame表示输入的图像，model表示使用的模型，device表示使用的设备（CPU或GPU），imgsz表示输入图像的大小，names表示模型预测的类别名列表，colors表示每个类别的颜色。
    t1 = time.time()  #记录函数开始执行的时间。
    face_status=[]  #定义一个空列表，用于存储检测到的人脸的状态。
    img = letterbox(frame, imgsz, stride=32)[0]  #将输入的图像frame按照imgsz的大小进行缩放，并进行填充，使得图像的长和宽都是stride的倍数。返回值是一个列表，取第一个元素。
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416  将图像从BGR格式转换为RGB格式，并将它的维度从HWC变为CHW。
    img = np.ascontiguousarray(img)  #创建一个具有相同数据的新数组，并按照内存中的C顺序存储它。
    img = torch.from_numpy(img).to(device)  #将数组转换为PyTorch张量，并将它移到指定的设备上。
    # img = img.half() if half else img.float()
    # img = img.half()
    img=img.float()   #将张量的数据类型转换为float。
    img /= 255.0  # 0 - 255 to 0.0 - 1.0   #将张量中的数值从0-255缩放到0.0-1.0。
    if img.ndimension() == 3:  #如果输入张量的维度为3，则将它的第0维度增加一个维度，变成4维。
        img = img.unsqueeze(0)
    with torch.no_grad():   #使用torch.no_grad()上下文管理器，表示在计算图中不需要计算梯度。
        pred = model(img, augment=False)[0]   #使用模型对输入图像进行推理，得到预测结果pred，并取第0个元素。augment表示是否使用数据增强。
        pred = non_max_suppression(pred, 0.25, 0.25, classes=None, agnostic=False)[0]  #对预测结果进行非极大值抑制，得到检测结果pred，并取第0个元素。0.25表示IoU的阈值，classes表示要保留的类别，agnostic表示是否对类别不敏感。
    if len(pred):  #如果检测结果不为空。
        # Rescale boxes from img_size to im0 size
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()  #将检测结果的坐标缩放回原始图像的大小，并四舍五入取整数值。
        # Write results
        for *xyxy, conf, cls in reversed(pred):  #遍历检测结果中的每个检测框，每个检测框由4个坐标和置信度conf、类别cls组成。使用reversed函数，使得遍历时先遍历置信度高的检测框。
            label = f'{names[int(cls)]} {conf:.2f}'  #根据类别cls和置信度conf，生成一个标签label。
            plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)  #在原始图像上画出检测框，并标上标签。plot_one_box是一个自定义的函数，用于画出矩形框。
            face_status.append(int(cls))   #将检测到的人脸的类别加入到face_status列表中。
            # print(face_status)
            # ss
    t2 = time.time()   #记录函数执行结束的时间。
    FPS = int(1 / (t2 - t1))   #计算函数的帧率。
    cv2.putText(frame, 'FPS=%s' % FPS, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 0, 0], 2, lineType=cv2.LINE_AA)   #在图像上写上帧率的信息。
    return frame,face_status   #返回处理后的图像和人脸状态列表。

class Video_receive_thread(threading.Thread):  #这段代码定义了一个继承自threading.Thread的类Video_receive_thread，用于在一个线程中读取网络视频流。
    def __init__(self, threadID, name,url,frame_name):  #定义了类的构造函数，其中threadID、name、url和frame_name是输入参数。
        threading.Thread.__init__(self)  #调用父类threading.Thread的构造函数。
        self.threadID = threadID  #给类的实例添加了一个threadID属性，并将输入参数threadID的值赋给它。
        self.name = name  #给类的实例添加了一个name属性，并将输入参数name的值赋给它。
        self.url=url  #给类的实例添加了一个url属性，并将输入参数url的值赋给它。
        self.frame_name=frame_name  #给类的实例添加了一个frame_name属性，并将输入参数frame_name的值赋给它。
        # float = np.float32()
        # self.frame = np.zeros(shape=(1080, 1920, 3), dtype=float)
        self.frame=None  #给类的实例添加了一个frame属性，并将其值设为None。
        self.cap = cv2.VideoCapture(self.url)  #创建了一个OpenCV的视频捕捉对象，并将输入参数url作为其输入。
    def run(self):  #定义了一个类方法run()，用于在线程中运行。
        # while (self.cap.isOpened()):
        while True:  #创建了一个无限循环，表示读取视频流的操作一直进行。
            self.ret, self.frame = self.cap.read()  #从视频捕捉对象中读取一帧图像，self.ret表示是否读取成功，self.frame表示读取到的图像。
            if self.ret == False:   #如果读取失败，则重新从视频流中读取。
                self.cap = cv2.VideoCapture(self.url)
                continue
            # print(self.ret)
            if cv2.waitKey(1) & 0xFF == ord('q'):  #如果按下了键盘上的q键，则跳出循环。
                break
        # self.cap.release()
        cv2.destroyAllWindows()  #释放所有的窗口资源。

thread_read1 = Video_receive_thread(1, "video_read1",url1,'frame1')   #创建了一个Video_receive_thread的实例thread_read1，并将threadID设为1，name设为video_read1，url设为url1，frame_name设为frame1。

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

        self.face_status=[]
        #睡岗计时器
        self.time1=None
        self.time2=None
        #疲劳计时器
        self.time3=None
        self.time4=None
    def run(self):
        while True:
            # try:
            self.frame_read,self.face_status = self.img_infer(self.thread_num.frame, self.model1, device, imgsz, names, colors)

            # print(len(self.face_status))
            # print(self.face_status)
            # print(self.img_infer(self.thread_num.frame, self.model1, device, imgsz, names, colors))
            #进行睡岗检查，检查不到人脸，或者检查到人脸但是没有眼睛
            if (len(self.face_status)==1 and 0 in self.face_status) or (len(self.face_status)==0):
                #第一次监测到睡岗的情况，记录一下时间，这个时间可以被在此赋值为None
                if self.time1==None:
                    self.time1=time.time()
                else:
                    self.time2=time.time()
                    if self.time2-self.time1 > 5:
                        # print("睡岗")
                        self.frame_read = cv2AddChineseText(self.frame_read, "睡着/离开", (50, 70), (255, 0, 0), 50)
                        if pygame.mixer.music.get_busy():
                            pass
                        else:
                            pygame.mixer.music.load('./wav/sleep.wav')
                            pygame.mixer.music.set_volume(0.5)
                            pygame.mixer.music.play(1)
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
                            # print("一级")
                            self.frame_read = cv2AddChineseText(self.frame_read, "一级疲劳", (50, 70), (255, 0, 0), 50)
                            if pygame.mixer.music.get_busy():
                                pass
                            else:
                                pygame.mixer.music.load('./wav/one.wav')
                                pygame.mixer.music.set_volume(0.5)
                                pygame.mixer.music.play(1)
                        if time_tied>=5 and time_tied<8:
                            # print("二级")
                            self.frame_read = cv2AddChineseText(self.frame_read, "二级疲劳", (50, 70), (255, 0, 0), 50)
                            if pygame.mixer.music.get_busy():
                                pass
                            else:
                                pygame.mixer.music.load('./wav/two.wav')
                                pygame.mixer.music.set_volume(0.5)
                                pygame.mixer.music.play(1)
                        if time_tied>=8 and time_tied<11:
                            # print("三级")
                            self.frame_read = cv2AddChineseText(self.frame_read, "三级疲劳", (50, 70), (255, 0, 0), 50)
                            if pygame.mixer.music.get_busy():
                                pass
                            else:
                                pygame.mixer.music.load('./wav/three.wav')
                                pygame.mixer.music.set_volume(0.5)
                                pygame.mixer.music.play(1)
                        if time_tied>=11:
                            # print("四级")
                            self.frame_read = cv2AddChineseText(self.frame_read, "四级疲劳", (50, 70), (255, 0, 0), 50)
                            if pygame.mixer.music.get_busy():
                                pass
                            else:
                                pygame.mixer.music.load('./wav/four.wav')
                                pygame.mixer.music.set_volume(0.5)
                                pygame.mixer.music.play(1)
            else:
                self.time3=None
                self.time4=None
            # except:
            #     # self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
            #     self.frame_read = np.random.rand(480,640,3)*255
            #     print("Data Error",self.thread_num)
            #self.frame_read=cv2.resize(self.frame_read,(1920,1080))
            # print(self.frame_read.shape)
            cv2.imshow(self.frame_name, self.frame_read)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    thread_ana1 = Video_analyze_thread(9, "video_ana1", thread_read1, 'frame1', image_inference, model)
    thread_read1.start()
    time.sleep(2)
    thread_ana1.start()
    thread_read1.join()
    thread_ana1.join()