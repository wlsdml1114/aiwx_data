# -*- coding: EUC-KR -*- 
import os
import numpy as np
import torch
import math
import sys
import time
import argparse
from PIL import Image
import torchvision
from torch import nn, Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from typing import List, Tuple, Dict, Optional
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import utils
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import configparser
from collections import Counter

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                width, _ = F._get_image_size(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)
                if "keypoints" in target:
                    keypoints = target["keypoints"]
                    keypoints = _flip_coco_person_keypoints(keypoints, width)
                    target["keypoints"] = keypoints
        return image, target

def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data

class ToTensor(nn.Module):
    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.to_tensor(image)
        return image, target

def delete_yellow_line_coms(pil_image):
    indx = [1,1,-1,-1]
    indy = [1,-1,1,-1]
    temp = pil_image
    for i in range(temp.size[0]):
        for j in range(temp.size[1]):
            r,g,b = temp.getpixel((i,j))
            if  (r==255) & (g==255) & (b==0):
                re_r, re_g, re_b = 0,0,0
                count = 0
                for k in range(4):
                    try :
                        r,g,b = temp.getpixel((i+indx[k],j+indy[k]))
                        if  not((r==255) & (g==255) & (b==0)):
                            re_r += r
                            re_g += g
                            re_b += b
                            count+=1
                    except :
                        continue
                if count != 0 :
                    temp.putpixel((i,j),(int(re_r/count),int(re_g/count),int(re_b/count)))
                else : 
                    temp.putpixel((i,j),(0,0,0))
    return temp

def delete_yellow_line_gk2a(pil_image):
    indx = [1,1,-1,-1]
    indy = [1,-1,1,-1]
    temp = pil_image
    for i in range(temp.size[0]):
        for j in range(temp.size[1]):
            r,g,b,a = temp.getpixel((i,j))
            if  (r==255) & (g==255) & (b==0):
                re_r, re_g, re_b = 0,0,0
                count = 0
                for k in range(4):
                    try :
                        r,g,b,a = temp.getpixel((i+indx[k],j+indy[k]))
                        if  not((r==255) & (g==255) & (b==0)):
                            re_r += r
                            re_g += g
                            re_b += b
                            count+=1
                    except :
                        continue
                if count != 0 :
                    temp.putpixel((i,j),(int(re_r/count),int(re_g/count),int(re_b/count)))
                else : 
                    temp.putpixel((i,j),(0,0,0))
    return temp

class CustomDataset(object):
    def __init__(self, root,anno, transforms,mode, num=0):
        self.root = root
        self.transforms = transforms
        self.anno = anno    
        #self.sats = np.concatenate([sat,rad[:len(sat)]],axis=1)
        if mode == 'sat' :
            self.sats = np.load(os.path.join(root,anno,'sat.npy'))
        elif mode == 'rad' :
            self.sats = np.load(os.path.join(root,anno,'rad.npy'))
        else :
            print('wrong name')
        self.masks = np.load(os.path.join(root,anno,'mask.npy'))
        print('loading')
        #self.sats = np.transpose(self.sats,(0,2,3,1))
        #print(self.sats.shape)
        #self.sats = self.sats[:100]
        #self.masks = self.masks[:100]
        # ???? ?????? ???????? ????, ????????
        # ???????? ???? ?????? ?????? ??????????
        #self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        #self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        
    def __getitem__(self, idx):
        # ???????? ???????? ??????????
        #img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        #mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        
        img = self.sats[idx]
        mask = self.masks[idx]
        # ???? ???????? RGB?? ???????? ?????? ??????????
        # ???????? ?? ?????? ???? ?????????? ????????, 0?? ?????? ??????????
        #mask = Image.open(mask_path)
        # numpy ?????? PIL ???????? ??????????
        #mask = np.array(mask)
        # ???????????? ???? ?????? ?????? ???? ????????.
        obj_ids = np.unique(mask)
        # ?????? id ?? ???????? ??????????
        obj_ids = obj_ids[1:]

        # ???? ???????? ???????? ???????? ?????? ?????? ????????
        masks = mask == obj_ids[:, None, None]

        # ?? ???????? ?????? ???? ?????? ????????
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if xmin == xmax:
                xmax = xmax+1
            if ymin == ymax :
                ymax = ymax+1
            boxes.append([xmin, ymin, xmax, ymax])

        # ???? ???? torch.Tensor ???????? ??????????
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # ???? ?????? ?? ?????? ??????????(??????: ?????????? ???????? ??????????)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        #area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # ???? ?????????? ????(crowd) ?????? ?????? ??????????
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = torch.tensor(100)
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.sats)

def get_model_instance_segmentation(num_classes):
    # COCO ???? ???? ?????? ???????? ???? ?????? ??????????
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # ?????? ???? ???? ???? ?????? ????????
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # ???? ?????? ?????? ?????? ?????? ????????
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # ?????? ???????? ???? ???? ???????? ?????? ????????
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # ?????? ???????? ?????? ?????? ????????
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    return Compose(transforms)


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        target = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if target[0]['boxes'].size()[0]==0 :
            continue
        
        loss_dict = model(images, target)

        losses = sum(loss for loss in loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

config = configparser.ConfigParser()    
config.read('setting.ini', encoding='CP949') 

#set root
path = config['path']['test_path']
data_path = os.path.join(config['path']['test_path'],'processed_data')
model_path = config['path']['model_path']
anno = config['train']['anno']
mode = config['train']['mode']
train_set_rate = config['train']['test_set_rate']
val_set_rate = config['train']['val_set_rate']
test_set_rate = config['train']['test_set_rate']
batch_size = config['train']['batch_size']
#Origin_path = '/mnt/nasmnt/sat'

# ?????? GPU?? ???????? GPU?? ???????? ?????? CPU?? ??????
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ???? ?????????? ?? ???? ???????? ???????? - ?????? ????
num_classes = 2
# ?????????? ?????? ???????? ??????????
dataset = CustomDataset(data_path,anno, get_transform(train=True),mode)

# ?????????? ???????? ???????????? ????????(??????: ???????? ?????? 50???? ????????, ???????? ?????? ??????????)

#dataset = torch.utils.data.Subset(dataset, list(range(int(len(dataset)*0.8))))
#dataset_test = torch.utils.data.Subset(dataset, list(range(int(len(dataset)*0.8),len(dataset))))

indices = torch.randperm(len(dataset)).tolist()
dataset_tr = torch.utils.data.Subset(dataset, indices[:int(train_set_rate/100)])
dataset_test = torch.utils.data.Subset(dataset, indices[int(train_set_rate/100):])

# ?????? ?????? ???????? ?????????? ??????????
data_loader = torch.utils.data.DataLoader(
    dataset_tr, batch_size=batch_size, shuffle=True, num_workers=2,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, shuffle=False, num_workers=2,
    collate_fn=utils.collate_fn)
    
# ???? ?????? ?????? ?????? ??????????
model = get_model_instance_segmentation(num_classes)

# ?????? GPU?? CPU?? ????????
model.to(device)

# ??????????(Optimizer)?? ????????
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# ?????? ?????????? ????????
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

# 10 ?????????? ????????????
num_epochs = 15

for epoch in range(num_epochs):
    # 1 ?????????? ????????, 10?? ???? ??????????
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
    # ???????? ???????? ??????
    lr_scheduler.step()


torch.save(model,os.path.join(model_path,anno+'_'+mode+'.pt'))
