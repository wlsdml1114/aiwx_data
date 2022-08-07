# -*- coding: EUC-KR -*- 
import os
import numpy as np
import torch
import math
import sys
import torchvision
from torch import nn, Tensor
from torchvision.transforms import functional as F
from typing import List, Tuple, Dict, Optional
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import utils
import configparser

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(nn.Module):
    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.to_tensor(image)
        return image, target

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
        # 모든 이미지 파일들을 읽고, 정렬하여
        # 이미지와 분할 마스크 정렬을 확인합니다
        #self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        #self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        
    def __getitem__(self, idx):
        # 이미지와 마스크를 읽어옵니다
        #img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        #mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        
        img = self.sats[idx]
        mask = self.masks[idx]
        # 분할 마스크는 RGB로 변환하지 않음을 유의하세요
        # 왜냐하면 각 색상은 다른 인스턴스에 해당하며, 0은 배경에 해당합니다
        #mask = Image.open(mask_path)
        # numpy 배열을 PIL 이미지로 변환합니다
        #mask = np.array(mask)
        # 인스턴스들은 다른 색들로 인코딩 되어 있습니다.
        obj_ids = np.unique(mask)
        # 첫번째 id 는 배경이라 제거합니다
        obj_ids = obj_ids[1:]

        # 컬러 인코딩된 마스크를 바이너리 마스크 세트로 나눕니다
        masks = mask == obj_ids[:, None, None]

        # 각 마스크의 바운딩 박스 좌표를 얻습니다
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

        # 모든 것을 torch.Tensor 타입으로 변환합니다
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # 객체 종류는 한 종류만 존재합니다(역자주: 예제에서는 사람만이 대상입니다)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        #area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # 모든 인스턴스는 군중(crowd) 상태가 아님을 가정합니다
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
    # COCO 에서 미리 학습된 인스턴스 분할 모델을 읽어옵니다
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # 분류를 위한 입력 특징 차원을 얻습니다
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 미리 학습된 헤더를 새로운 것으로 바꿉니다
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 마스크 분류기를 위한 입력 특징들의 차원을 얻습니다
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # 마스크 예측기를 새로운 것으로 바꿉니다
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

# 학습을 GPU로 진행하되 GPU가 가용하지 않으면 CPU로 합니다
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 우리 데이터셋은 두 개의 클래스만 가집니다 - 배경과 사람
num_classes = 2
# 데이터셋과 정의된 변환들을 사용합니다
dataset = CustomDataset(data_path,anno, get_transform(train=True),mode)

# 데이터셋을 학습용과 테스트용으로 나눕니다(역자주: 여기서는 전체의 50개를 테스트에, 나머지를 학습에 사용합니다)

#dataset = torch.utils.data.Subset(dataset, list(range(int(len(dataset)*0.8))))
#dataset_test = torch.utils.data.Subset(dataset, list(range(int(len(dataset)*0.8),len(dataset))))

indices = torch.randperm(len(dataset)).tolist()
dataset_tr = torch.utils.data.Subset(dataset, indices[:int(train_set_rate/100)])
dataset_test = torch.utils.data.Subset(dataset, indices[int(train_set_rate/100):])

# 데이터 로더를 학습용과 검증용으로 정의합니다
data_loader = torch.utils.data.DataLoader(
    dataset_tr, batch_size=batch_size, shuffle=True, num_workers=2,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, shuffle=False, num_workers=2,
    collate_fn=utils.collate_fn)
    
# 도움 함수를 이용해 모델을 가져옵니다
model = get_model_instance_segmentation(num_classes)

# 모델을 GPU나 CPU로 옮깁니다
model.to(device)

# 옵티마이저(Optimizer)를 만듭니다
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# 학습률 스케쥴러를 만듭니다
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

# 10 에포크만큼 학습해봅시다
num_epochs = 15

for epoch in range(num_epochs):
    # 1 에포크동안 학습하고, 10회 마다 출력합니다
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
    # 학습률을 업데이트 합니다
    lr_scheduler.step()


torch.save(model,os.path.join(model_path,anno+'_'+mode+'.pt'))
