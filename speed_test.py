import numpy as np

import torch.utils.data
import yaml

from models.yolo import Model
import time
from torchsummaryX import summary
from torchvision.transforms import Compose,Resize
import cv2
from torch.cuda import amp
from torch.cuda.amp import autocast as autocast
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from models.common import SPPCFSPC, SPPCSPC
cfg_path='cfg/training/yolov7-fs.yaml'
hyp_path='data/hyp.scratch.p5.yaml'

with open("coco/train2017.txt",'r') as f:
    lines=f.readlines()

img_path=[]
for i in range(len(lines)):
    img_path.append('coco/'+lines[i][1:-1])

device = select_device('cpu')
with open(hyp_path) as f:
    hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
model = Model(cfg_path, ch=3, nc=80, anchors=hyp.get('anchors')).to(device)
tensor=torch.rand(1,3,416,416).cpu()
flops=summary(model,tensor)

# resize=Compose([Resize((416,416))])
# img_list=[]
# for i in range(len(img_path)):
#     img = cv2.imread(img_path[i])
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = img.astype(np.float32)
#     img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
#     img = resize(img)
#     img_list.append(img)
#
# model.eval()
# t1=time.time()
#
# for i in range(len(img_path)):
#     img_l=img_list[i]
#     with torch.no_grad():
#         out=model(img_l)
# t2=time.time()
# print("time:",t2-t1,"FPS:",(len(img_list)/(t2-t1)))



# tensor=torch.rand(1,1024,20,20)
# model=SPPCSPC(1024,512)
# flops=summary(model,tensor)