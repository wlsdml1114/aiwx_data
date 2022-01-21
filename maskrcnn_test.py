import torch
import argparse
from PIL import Image
import os
import numpy as np
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
from datetime import datetime, timedelta
from collections import Counter
import utils
import configparser
print((datetime.now()+timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S'))
print("mask rcnn test start..")

class CustomDataset(object):
    def __init__(self, root,anno, transforms,mode, num=0):
        self.root = root
        self.transforms = transforms
        self.anno = anno    
        if mode == 'sat' :
            self.sats = np.load(os.path.join(root,anno,'sat.npy'))
        elif mode == 'rad' :
            self.sats = np.load(os.path.join(root,anno,'rad.npy'))
        else :
            print('wrong name')
        self.masks = np.load(os.path.join(root,anno,'mask.npy'))
        new_sat = self.sats
        new_mask = self.masks
        '''
        count = 0
        for mask in self.masks :
            c = Counter(mask.flatten())
            if c[1] >= 100 :
                new_sat.append(self.sats[count])
                new_mask.append(self.masks[count])
            count+=1
        '''
        self.sats = np.array(new_sat)
        self.sats = np.transpose(self.sats,(0,2,3,1))
        self.masks = np.array(new_mask)
        
    def __getitem__(self, idx):
        
        img = self.sats[idx]
        mask = self.masks[idx]
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
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

def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    return Compose(transforms)

class ToTensor(nn.Module):
    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.to_tensor(image)
        return image, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cuda = torch.device('cuda')

config = configparser.ConfigParser()    
config.read('setting.ini', encoding='utf-8') 

START_DATE = datetime.strptime(config['date']['start_date'],'%Y-%m-%d %H:%M:%S')
END_DATE = datetime.strptime(config['date']['end_date'],'%Y-%m-%d %H:%M:%S')
path = config['path']['test_path']
model_path = config['path']['model_path']


annos = ['LLJ', 'W_SNOW','E_SNOW','WET_SN','CUM_SN','COLD_FRONT','WARM_FRONT','OCC_FRONT','H_POINT','L_POINT'
        ,'HLJ', 'TYPOON', 'R_START', 'R_STOP','RA_SN','HAIL']
modes = ['sat','rad']

model_dir = model_path
data_path = os.path.join(path,'processed_data')
eval_path = os.path.join(path,'processed_data')

for anno in annos:
    for mode in modes:
        model = torch.load(os.path.join(model_dir,anno+'_'+mode+'.pt'))
        model = model.cuda()
        model.eval()

        dataset = CustomDataset(data_path,anno, get_transform(train=True),mode)

        data_loader_test = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=2,
            collate_fn=utils.collate_fn)

        target_arr = []
        preds = []
        print(anno+mode+'data loading finish..')

        for images, targets in data_loader_test:
            images = list(image.to(device) for image in images)
            target = [{k: v.to(device) for k, v in t.items()} for t in targets]

            output = model(images)

            for j in range(len(images)):
                try:
                    temp = output[j]['masks'][0].cpu().detach()[0].numpy()
                except Exception as e:
                    temp = np.zeros((900,900))
                try:
                    temp_target = target[j]['masks'][0].cpu().detach().numpy()
                except:
                    temp_target = np.zeros((900,900))

                pred = np.mean(temp.reshape(20,45,20,45),axis=(1,3))
                temp_target = np.mean(temp_target.reshape(20,45,20,45),axis=(1,3))
                preds.append(pred)
                target_arr.append(temp_target)

        np.save(os.path.join(eval_path,anno+'_'+mode+'_pred.npy'),preds)
        np.save(os.path.join(eval_path,anno+'_'+mode+'_target.npy'),target_arr)

