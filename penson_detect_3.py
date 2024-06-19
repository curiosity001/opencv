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



# # 定义 YOLOv7 模型的权重文件名
# weights2 = 'yolov7.pt'
#
# # 选择设备（例如 GPU）来运行模型
# device = select_device('0')
#
# # 判断是否使用半精度浮点数（FP16），仅在使用 CUDA 设备时支持
# half = device.type != 'cpu'
#
# # 加载 FP32 模型
# model2 = attempt_load(weights2, map_location=device)
#
# # 获取模型的步长
# stride = int(model2.stride.max())
#
# # 检查图像尺寸是否符合要求
# imgsz = check_img_size(640, s=stride)
#
# # 获取模型的类别名称
# names = model2.module.names if hasattr(model2, 'module') else model2.names
#
# # 为每个类别生成随机颜色
# colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
#
# # 创建用于对比度受限自适应直方图均衡化 (CLAHE) 的对象
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
#
# # 如果使用半精度浮点数 (FP16)，将模型转换为半精度
# if half:
#     model2.half()
#
# # 定义图像推理函数
# def image_inference1(frame, model, device, half, imgsz, names, colors):
#     # 记录开始时间
#     t1 = time.time()
#
#     # 对输入图像进行预处理
#     img = letterbox(frame, imgsz, stride=32)[0] #自适应图片缩放
#     img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR 转 RGB，然后转换为 3x416x416
#     img = np.ascontiguousarray(img)  # 将不连续的内存数组转换为连续的内存数组，以提高运行速度
#
#     # 将 NumPy 数组转换为 PyTorch 张量并将其发送到设备
#     img = torch.from_numpy(img).to(device)
#
#     # 根据是否使用半精度浮点数 (FP16) 转换图像张量的数据类型
#     img = img.half() if half else img.float()
#
#     # 归一化图像数据（0-255 范围转换为 0.0-1.0 范围）
#     img /= 255.0
#
#     # 如果图像张量的维度为 3，添加一个批量维度
#     if img.ndimension() == 3:
#         img = img.unsqueeze(0)
#
#     # 使用模型进行推理
#     with torch.no_grad():
#         pred2 = model(img, augment=False)[0]
#
#         # 使用非极大值抑制 (NMS) 来过滤检测框
#         pred2 = non_max_suppression(pred2, 0.45, 0.45, classes=0, agnostic=False)[0]
#
#
#     # 如果存在检测结果
#     if len(pred2):
#         # 绘制检测框和类别标签
#         for *xyxy, conf, cls in reversed(pred2):  #这是一个 for 循环，用于遍历 pred2（一组目标检测结果）中的每个检测框   *xyxy：这是一个包含检测框左上角和右下角坐标（x1, y1, x2, y2）的元组。  conf：这是检测框的置信度得分，表示模型对检测结果的置信程度。  cls：这是检测框所属类别的索引（例如，对于目标检测任务，类别可能是 'person'、'car' 等）
#             # 获取类别名称和置信度
#             label = f'{names[int(cls)]} {conf:.2f}'
#
#             # 在图像上绘制检测框和标签
#             frame = plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
#
#     # 计算运行时间并获取 FPS（每秒帧数）
#     t2 = time.time()
#     FPS = int(1 / (t2 - t1))
#
#     # 在图像上显示 FPS
#     cv2.putText(frame, 'FPS=%s' % FPS, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0], 2, lineType=cv2.LINE_AA)
#
#     # 返回带有检测框和标签的图像
#     return frame

"""
这段代码执行模型的推理过程，并使用非极大值抑制（NMS）来过滤检测结果。

with torch.no_grad():：在此上下文管理器中，PyTorch 不会计算梯度，这可以降低内存使用并提高推理速度。
pred2 = model(img, augment=False)[0]：将图像输入模型进行推理。augment=False 表示不使用数据增强。[0] 表示获取模型输出的第一个元素，即检测结果。
pred2 = non_max_suppression(pred2, 0.45, 0.45, classes=0, agnostic=False)[0]：对检测结果应用非极大值抑制（NMS）算法，以过滤掉与较高置信度检测框重叠且较低置信度的检测框。
0.45（第一个参数）：NMS 的 IOU 阈值，表示如果两个检测框的交并比（Intersection over Union）大于此阈值，则保留置信度较高的检测框，删除较低置信度的检测框。
0.45（第二个参数）：置信度阈值，表示只保留置信度大于此阈值的检测框。
classes=0：指定要过滤的类别索引。在这里，只过滤类别 0（例如，'person' 类别）的检测框。如果要对所有类别应用 NMS，可以设置为 None。
agnostic=False：表示 NMS 不是类别无关的，即仅在同一类别的检测框之间应用 NMS。如果设置为 True，则在所有类别的检测框之间应用 NMS。
最后，[0] 表示获取 NMS 结果的第一个元素。这是经过 NMS 处理的检测框列表。

"""



"""
pred2[0] 代表经过非极大值抑制（NMS）处理后的第一个元素。这是因为 non_max_suppression 函数返回一个列表，其中每个元素对应于一个输入图像（在这种情况下，只有一个输入图像）。因此，pred2[0] 表示 NMS 处理后的检测框列表，这些检测框是针对给定输入图像的过滤后的目标检测结果。

要澄清的是，在这里，pred2 是经过 NMS 处理的检测框列表，而 pred2[0] 是第一个（且唯一）输入图像对应的经过 NMS 处理的检测框列表。由于这个例子中只有一个输入图像，所以可以直接使用 pred2[0] 获取 NMS 结果。"

在这个特定的代码示例中，pred2[1] 并没有实际意义，因为只有一个输入图像，所以只有一个 NMS 结果，即 pred2[0]。在这种情况下，尝试访问 pred2[1] 会导致索引错误，因为列表中没有第二个元素。

然而，在一个多输入图像的场景中，pred2[1] 将代表第二个输入图像经过非极大值抑制（NMS）处理后的检测框列表。在这种情况下，pred2 列表将包含与输入图像数量相等的元素，每个元素都是一个经过 NMS 处理的检测框列表。这样，pred2[i] 将表示第 i+1 个输入图像的 NMS 处理结果。
"""





"""
检测框列表是一个包含多个检测框的列表，每个检测框代表一个检测到的目标。在目标检测任务中，模型的目的是识别图像中的不同目标并为每个目标提供一个边界框。这些边界框称为检测框。

检测框列表中的每个检测框通常包含以下信息：

边界框坐标：这是一个包含四个值（x1, y1, x2, y2）的元组，分别表示检测框左上角和右下角的坐标。

置信度得分：这是一个浮点值，表示模型对检测框的置信程度。通常在 0 到 1 之间，值越高表示模型对该检测结果的置信度越高。

类别索引：这是一个整数值，表示检测框所属类别的索引。例如，在一个目标检测任务中，类别可以是 'person'、'car' 等。类别索引是这些类别在类别列表中的位置。

pred2[0] 返回经过 NMS 处理的检测框列表，意味着列表中的检测框已经经过了过滤。NMS 算法消除了多个高度重叠的检测框，并保留了具有较高置信度得分的检测框。因此，pred2[0] 包含的检测框列表可以被视为模型对输入图像的最终预测结果，其中每个检测框都表示一个检测到的目标。
"""

"""
如果有三张图片输入，检测框列表的形式可以表示为一个外层列表，其中每个元素是一个与上述格式相同的检测框列表，对应于每张输入图片的检测结果。这个外层列表的形式如下：
[
    [
        [x1_1_1, y1_1_1, x2_1_1, y2_1_1, conf_1_1, cls_1_1],
        [x1_1_2, y1_1_2, x2_1_2, y2_1_2, conf_1_2, cls_1_2],
        ...
        [x1_1_n1, y1_1_n1, x2_1_n1, y2_1_n1, conf_1_n1, cls_1_n1]
    ],
    [
        [x1_2_1, y1_2_1, x2_2_1, y2_2_1, conf_2_1, cls_2_1],
        [x1_2_2, y1_2_2, x2_2_2, y2_2_2, conf_2_2, cls_2_2],
        ...
        [x1_2_n2, y1_2_n2, x2_2_n2, y2_2_n2, conf_2_n2, cls_2_n2]
    ],
    [
        [x1_3_1, y1_3_1, x2_3_1, y2_3_1, conf_3_1, cls_3_1],
        [x1_3_2, y1_3_2, x2_3_2, y2_3_2, conf_3_2, cls_3_2],
        ...
        [x1_3_n3, y1_3_n3, x2_3_n3, y2_3_n3, conf_3_n3, cls_3_n3]
    ]
]
在这个形式中，外层列表包含 3 个元素，每个元素对应一张输入图片。每个元素都是一个检测框列表，包含 n 个检测框，每个检测框由一个包含 6 个元素的子列表表示。
子列表的前四个元素（x1_i_j, y1_i_j, x2_i_j, y2_i_j）表示检测框的左上角和右下角坐标，第五个元素（conf_i_j）表示置信度得分，第六个元素（cls_i_j）表示类别索引。其中 i 表示输入图片的索引，j 表示检测框在该图片中的索引。


上面这个列表的第一个（索引为 0 的）元素是与第一张输入图片对应的检测框列表。这个检测框列表包含了第一张图片中所有检测到的目标的边界框坐标、置信度得分和类别索引。这个子列表的形式如下：
[
    [x1_1_1, y1_1_1, x2_1_1, y2_1_1, conf_1_1, cls_1_1],
    [x1_1_2, y1_1_2, x2_1_2, y2_1_2, conf_1_2, cls_1_2],
    ...
    [x1_1_n1, y1_1_n1, x2_1_n1, y2_1_n1, conf_1_n1, cls_1_n1]
]


"""



"""

这段代码检查输入图像张量 img 的维度，如果它是一个 3 维张量，那么将在第一个维度上添加一个额外的维度，从而将其转换为一个 4 维张量。

让我们分析这段代码：

if img.ndimension() == 3:：这里，img.ndimension() 函数返回输入张量 img 的维度数。如果维度数为 3（例如，通常的图像张量形状为 (channels, height, width)），则执行下一行代码。

img = img.unsqueeze(0)：img.unsqueeze(0) 函数在输入张量 img 的第一个维度上添加一个额外的维度。这将图像张量从形状 (channels, height, width) 转换为形状 (1, channels, height, width)。这通常是为了满足深度学习模型的输入要求，因为许多模型期望输入为批量形式（即具有一个批量大小维度）。

在这个例子中，通过将图像张量转换为一个批量大小为 1 的 4 维张量，可以使其兼容预期输入为批量形式的深度学习模型。

"""




weights2='yolov7.pt'
device = select_device('0')
half = device.type != 'cpu'  # half precision only supported on CUDA
model2 = attempt_load(weights2, map_location=device)  # load FP32 model
stride = int(model2.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size
names = model2.module.names if hasattr(model2, 'module') else model2.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
if half:
    model2.half()  # to FP16


url1=0

def image_inference1(frame,model,device,half,imgsz,names,colors):
    t1 = time.time()
    img = letterbox(frame, imgsz, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img) #ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快（归一化处理）
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    with torch.no_grad():
        pred2=model(img, augment=False)[0]
        pred2 = non_max_suppression(pred2, 0.45, 0.45, classes=0, agnostic=False)[0]
        #print(pred2.shape) #torch.Size([0, 6])
    if len(pred2):
        for *xyxy, conf, cls in reversed(pred2):
            # print(*xyxy[0])
            # print(conf.shape)
            # print(names[int(cls)]) #person
            # ss
            label = f'{names[int(cls)]} {conf:.2f}'
            # print(label)
            # ss
            frame=plot_one_box(xyxy,frame,label=label,color=colors[int(cls)], line_thickness=3)
    t2 = time.time()
    FPS=int(1/(t2 - t1))
    cv2.putText(frame, 'FPS=%s' %FPS, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0],2, lineType=cv2.LINE_AA)
    return frame

# def image_inference2(frame,model,device,half,imgsz,names,colors):
#     t1 = time.time()
#     img = letterbox(frame, imgsz, stride=32)[0]
#     img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
#     img = np.ascontiguousarray(img)
#     img = torch.from_numpy(img).to(device)
#     img = img.half() if half else img.float()
#     img /= 255.0  # 0 - 255 to 0.0 - 1.0
#     if img.ndimension() == 3:
#         img = img.unsqueeze(0)
#     with torch.no_grad():
#         pred2=model(img, augment=False)[0]
#         pred2 = non_max_suppression(pred2, 0.25, 0.45, classes=0, agnostic=False)[0]
#     if len(pred2):
#         # Rescale boxes from img_size to im0 size
#         pred2[:, :4] = scale_coords(img.shape[2:], pred2[:, :4], frame.shape).round()
#         # Write results
#         for *xyxy, conf, cls in reversed(pred2):
#             # print(int(cls)) #person
#             # ss
#             label = f'{names[int(cls)]} {conf:.2f}'
#             frame=plot_one_box(xyxy,frame,label=label,color=colors[int(cls)], line_thickness=3)
#     t2 = time.time()
#     FPS=int(1/(t2 - t1))
#     # print("FPS:", 1/(t2 - t1))
#     cv2.putText(frame, 'FPS=%s' %FPS, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, [225, 0, 0],2, lineType=cv2.LINE_AA)
#     return frame


class Video_receive_thread(threading.Thread):
    def __init__(self, threadID, name,url,frame_name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.url=url
        self.frame_name=frame_name
        self.frame = np.zeros(shape=(1080, 1920, 3), dtype=float)
        self.cap = cv2.VideoCapture(self.url)

    def run(self):
        while (self.cap.isOpened()):
            self.ret, self.frame = self.cap.read()
            time.sleep(0.02)
            #cv2.rectangle(frame, (start_col,start_row), (end_col, end_row), (255, 0, 0), 2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

thread_read1 = Video_receive_thread(1, "video_read1",url1,'frame1')
# thread_read2 = Video_receive_thread(2, "video_read2",url2,'frame2')

class Video_analyze_thread(threading.Thread):
    def __init__(self, threadID, name,thread_num,frame_name,img_infer):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.frame_read = np.zeros(shape=(1080, 1920, 3), dtype=float)
        self.thread_num=thread_num
        self.frame_name=frame_name
        self.img_infer=img_infer
    def run(self):
        while True:
            # self.frame_read = thread1.frame
            self.frame_read = self.img_infer(self.thread_num.frame, model2, device, half, imgsz, names, colors)
            cv2.imshow(self.frame_name, self.frame_read)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    #with torch.no_grad():
    thread_ana1 = Video_analyze_thread(5, "video_ana1",thread_read1,'frame1',image_inference1)
    # thread_ana2 = Video_analyze_thread(6, "video_ana2",thread_read2,'frame2',image_inference2)
    thread_read1.start()
    # thread_read2.start()
    thread_ana1.start()
    # thread_ana2.start()
    thread_read1.join()
    # thread_read2.join()
    thread_ana1.join()
    # thread_ana2.join()

