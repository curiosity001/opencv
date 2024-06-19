# from __init__ import *
# from step_defss.scenario_steps import *
import sys
import os
# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)
# print(sys.path)

#接后续代码
import os
import sys
import serial
import subprocess
import warnings
import pickle
import cv2
#from cv2 import VideoCapture
# import math
import copy
import numpy as np
# import time
import logging
from PIL import Image
# from skimage import exposure
import time
from pathlib import Path
import cv2
from numpy import random
import csv
import datetime
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
from copy import deepcopy
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from datetime import datetime
import os
# from base_readCSV import QmyMainWindow

import cv2 as cv
# import dlib
from scipy.spatial import distance
# from ssdface_detect_module import ssdface
metalocA = threading.Lock()
weights='D:\guanlong\yolov7-main\yolov5.pt'
weights2='D:\guanlong\yolov7-main\yolov7.pt'
device = select_device('0')
half = device.type != 'cpu'  # half precision only supported on CUDA

model = attempt_load(weights, map_location=device)  # load FP32 model
model2 = attempt_load(weights2, map_location=device)  # load FP32 model

stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
if half:
    model.half()  # to FP16
    model2.half()
import pygame
# import time
pygame.mixer.init()

# video_weight=1920
# video_hight=1080
# url1 = 'rtsp://admin:hy123456@192.168.4.151:554/live'
url1='rtsp://admin:hy123456@192.168.4.138:554/live'#双下（上层）
url2='rtsp://admin:hy123456@192.168.4.137:554/live'#双上（上层）




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
    def __init__(self, threadID, name,thread_num,frame_name,img_infer,model1,model2):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        # self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
        self.frame_read=None
        self.thread_num=thread_num
        self.frame_name=frame_name
        self.img_infer = img_infer
        self.model1 = deepcopy(model1)
        self.model2 = deepcopy(model2)


    def run(self):
        while True:
            try:
                self.frame_read = self.img_infer(self.thread_num.frame, self.model1,self.model2, device, half, imgsz, names, colors)
            except:
                # self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
                self.frame_read = np.random.rand(1080,1920,3)*255
                print("Data Error",self.thread_num)
            #self.frame_read=cv2.resize(self.frame_read,(1920,1080))
            # print(self.frame_read.shape)
            # cv2.imshow(self.frame_name, self.frame_read)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()






if __name__ == "__main__":

    #with torch.no_grad():
    thread_ana1 = Video_analyze_thread(9, "video_ana1",thread_read1,'frame1',image_inference1,model,model2)
    time.sleep(2)
    thread_read1.start()
    time.sleep(3)
    thread_ana1.start()

    thread_read1.join()
    thread_ana1.join()
