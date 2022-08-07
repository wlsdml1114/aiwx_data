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
        # ��� �̹��� ���ϵ��� �а�, �����Ͽ�
        # �̹����� ���� ����ũ ������ Ȯ���մϴ�
        #self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        #self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        
    def __getitem__(self, idx):
        # �̹����� ����ũ�� �о�ɴϴ�
        #img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        #mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        
        img = self.sats[idx]
        mask = self.masks[idx]
        # ���� ����ũ�� RGB�� ��ȯ���� ������ �����ϼ���
        # �ֳ��ϸ� �� ������ �ٸ� �ν��Ͻ��� �ش��ϸ�, 0�� ��濡 �ش��մϴ�
        #mask = Image.open(mask_path)
        # numpy �迭�� PIL �̹����� ��ȯ�մϴ�
        #mask = np.array(mask)
        # �ν��Ͻ����� �ٸ� ����� ���ڵ� �Ǿ� �ֽ��ϴ�.
        obj_ids = np.unique(mask)
        # ù��° id �� ����̶� �����մϴ�
        obj_ids = obj_ids[1:]

        # �÷� ���ڵ��� ����ũ�� ���̳ʸ� ����ũ ��Ʈ�� �����ϴ�
        masks = mask == obj_ids[:, None, None]

        # �� ����ũ�� �ٿ�� �ڽ� ��ǥ�� ����ϴ�
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

        # ��� ���� torch.Tensor Ÿ������ ��ȯ�մϴ�
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # ��ü ������ �� ������ �����մϴ�(������: ���������� ������� ����Դϴ�)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        #area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # ��� �ν��Ͻ��� ����(crowd) ���°� �ƴ��� �����մϴ�
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
    # COCO ���� �̸� �н��� �ν��Ͻ� ���� ���� �о�ɴϴ�
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # �з��� ���� �Է� Ư¡ ������ ����ϴ�
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # �̸� �н��� ����� ���ο� ������ �ٲߴϴ�
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # ����ũ �з��⸦ ���� �Է� Ư¡���� ������ ����ϴ�
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # ����ũ �����⸦ ���ο� ������ �ٲߴϴ�
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

# �н��� GPU�� �����ϵ� GPU�� �������� ������ CPU�� �մϴ�
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# �츮 �����ͼ��� �� ���� Ŭ������ �����ϴ� - ���� ���
num_classes = 2
# �����ͼ°� ���ǵ� ��ȯ���� ����մϴ�
dataset = CustomDataset(data_path,anno, get_transform(train=True),mode)

# �����ͼ��� �н���� �׽�Ʈ������ �����ϴ�(������: ���⼭�� ��ü�� 50���� �׽�Ʈ��, �������� �н��� ����մϴ�)

#dataset = torch.utils.data.Subset(dataset, list(range(int(len(dataset)*0.8))))
#dataset_test = torch.utils.data.Subset(dataset, list(range(int(len(dataset)*0.8),len(dataset))))

indices = torch.randperm(len(dataset)).tolist()
dataset_tr = torch.utils.data.Subset(dataset, indices[:int(train_set_rate/100)])
dataset_test = torch.utils.data.Subset(dataset, indices[int(train_set_rate/100):])

# ������ �δ��� �н���� ���������� �����մϴ�
data_loader = torch.utils.data.DataLoader(
    dataset_tr, batch_size=batch_size, shuffle=True, num_workers=2,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size, shuffle=False, num_workers=2,
    collate_fn=utils.collate_fn)
    
# ���� �Լ��� �̿��� ���� �����ɴϴ�
model = get_model_instance_segmentation(num_classes)

# ���� GPU�� CPU�� �ű�ϴ�
model.to(device)

# ��Ƽ������(Optimizer)�� ����ϴ�
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# �н��� �����췯�� ����ϴ�
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

# 10 ����ũ��ŭ �н��غ��ô�
num_epochs = 15

for epoch in range(num_epochs):
    # 1 ����ũ���� �н��ϰ�, 10ȸ ���� ����մϴ�
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
    # �н����� ������Ʈ �մϴ�
    lr_scheduler.step()


torch.save(model,os.path.join(model_path,anno+'_'+mode+'.pt'))
