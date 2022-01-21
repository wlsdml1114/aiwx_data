# -*- coding: EUC-KR -*- 
from matplotlib import image
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.conv import Conv2d, ConvTranspose2d
from torch.nn.modules.pooling import MaxPool2d
from torch.utils.data import DataLoader, random_split
import configparser
import pytorch_lightning as pl
from pl_examples import _DATASETS_PATH, cli_lightning_logo
from pl_examples.basic_examples.mnist_datamodule import MNIST
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE
from pytorch_lightning import loggers as pl_loggers
from PIL import Image, ImageDraw
from datetime import date, datetime,timedelta
import numpy as np
import os
from tqdm import tqdm

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms

from datetime import datetime, timedelta
print((datetime.now()+timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S'))
print("rad image feature extraction..")

class LitAutoEncoder(pl.LightningModule):
    """
    >>> LitAutoEncoder()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LitAutoEncoder(
      (encoder): ...
      (decoder): ...
    )
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,8,kernel_size=5, padding='same'), 
            nn.ReLU(), 
            nn.MaxPool2d(4,4),
            nn.Conv2d(8,16,kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(5,5),
            nn.Conv2d(16,32,kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(5,5)
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32,16,kernel_size=5,stride = 5),
            nn.ReLU(),
            nn.ConvTranspose2d(16,8,kernel_size=5,stride = 5),
            nn.ReLU(),
            nn.ConvTranspose2d(8,1,kernel_size=5,stride = 4, padding=1, output_padding=1),
            nn.ReLU()
            )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("my_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("valid_loss", loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("test_loss", loss, on_step=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch
        z = self.encoder(x)
        return self.decoder(z)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

cuda = torch.device('cuda')

config = configparser.ConfigParser()    
config.read('setting.ini', encoding='utf-8') 

test_path = config['path']['test_path']
model_path = config['path']['model_path']

model = LitAutoEncoder.load_from_checkpoint(os.path.join(model_path,'rad_ae.ckpt'))
npy = np.load(os.path.join(test_path,'processed_data/rad_images_ar.npy'))

model = model.cuda()
images = torch.tensor(npy).cuda()

outputs = []

for i in tqdm(range(len(images))):
    output = model.forward(images[i:i+1]/256)
    outputs.append(output.cpu().detach().numpy().flatten())

np.save(os.path.join(test_path,'processed_data/rad_images_feature.npy'),np.array(outputs))