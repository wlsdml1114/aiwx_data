# -*- coding: EUC-KR -*- 

import warnings
warnings.simplefilter('ignore',UserWarning)

from pytorch_lightning.trainer.trainer import Trainer
import torch
import torchmetrics
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.conv import Conv2d, ConvTranspose2d
from torch.nn.modules.pooling import MaxPool2d
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pl_examples import _DATASETS_PATH, cli_lightning_logo
from pl_examples.basic_examples.mnist_datamodule import MNIST
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.utilities.imports import _TORCHVISION_AVAILABLE
from pytorch_lightning import loggers as pl_loggers
from PIL import Image, ImageDraw
from datetime import datetime,timedelta
import numpy as np
import os
import pandas as pd
import configparser
if _TORCHVISION_AVAILABLE:
    from torchvision import transforms


config = configparser.ConfigParser()    
config.read('setting.ini', encoding='CP949') 
test_path = config['path']['test_path']

train_set_rate = int(config['train']['train_set_rate'])
val_set_rate = int(config['train']['val_set_rate'])
test_set_rate = int(config['train']['test_set_rate'])

class TimeseriesDataset(Dataset):   

    def __init__(self, X: np.ndarray,Y: np.ndarray, seq_len: int = 24):
        self.X = torch.tensor(X).float()
        self.Y = torch.tensor(Y).float()
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len)

    def __getitem__(self, index):
        return (self.X[index:index+self.seq_len], self.Y[index+self.seq_len])

class lstm_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = input_size, num_layers = num_layers, batch_first=True)
        self.linear = nn.Sequential(
            nn.BatchNorm1d(8560),
            nn.Dropout(0.2),
            nn.Linear(8560, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 9),
            nn.Sigmoid()
        )
        #self.linear = nn.Linear(hidden_size, input_size)   
    def forward(self, x_input):
        lstm_out, self.hidden = self.lstm(x_input)
        output = self.linear(lstm_out[:,-1])
        #weather = F.sigmoid(output[-9:])
        #output = torch.cat((output[:-9],weather))
        return output,lstm_out, self.hidden

class lstm_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = input_size,num_layers = num_layers, batch_first=True)
        self.linear = nn.Sequential(
            nn.BatchNorm1d(8560),
            nn.Dropout(0.2),
            nn.Linear(8560, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 9),
            nn.Sigmoid()
        )
        #self.linear = nn.Linear(hidden_size, input_size)           

    def forward(self, x_input, encoder_hidden_states):
        lstm_out, self.hidden = self.lstm(x_input, encoder_hidden_states)
        output = self.linear(lstm_out[:,-1])
        #weather = F.sigmoid(output[-9:])
        #output = torch.cat((output[:-9],weather))
        
        return output,lstm_out, self.hidden

class LitAutoEncoder(pl.LightningModule):

    def __init__(self, hidden_size: int = 16, input_size: int=8560):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = lstm_encoder(input_size = input_size, hidden_size = hidden_size)
        self.decoder = lstm_decoder(input_size = input_size, hidden_size = hidden_size)

        self.criterion = F.binary_cross_entropy

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        lstm_out, state,_ = self.encoder(x)
        return lstm_out,state, _

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat,state,_ = self(x)
        loss = self.criterion(y_hat, y)
        #loss = torch.cat((loss,F.binary_cross_entropy(y_hat[-9:], y[-9:])))
        self.log("my_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat,state,_ = self(x)
        loss = self.criterion(y_hat, y)
        #loss = torch.cat((loss,F.binary_cross_entropy(y_hat[-9:], y[-9:])))
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat,state,_ = self(x)
        loss = self.criterion(y_hat, y)
        #loss = torch.cat((loss,F.binary_cross_entropy(y_hat[-9:], y[-9:])))
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def first_predict(self, x):
        lstm_out,state, _ = self.encoder(x)
        return lstm_out,state, _

    def second_predict(self, x, hidden):
        lstm_out,state, _ = self.decoder(x,hidden)
        return lstm_out,state, _

class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 16,seq_len = 24,num_workers=16):
        super().__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_workers = num_workers
        
        dataset = np.load(os.path.join(test_path,'processed_data/LSTM_X.npy'))
        target = np.load(os.path.join(test_path,'processed_data/LSTM_rain.npy'))
        
        cut_1 = train_set_rate/(train_set_rate+val_set_rate+test_set_rate)
        cut_2 = (train_set_rate+val_set_rate)/(train_set_rate+val_set_rate+test_set_rate)

        self.X_train = dataset[:int(len(dataset)*cut_1)]
        self.X_val = dataset[int(len(dataset)*cut_1):int(len(dataset)*cut_2)]
        self.X_test = dataset[int(len(dataset)*cut_2):]

        self.Y_train = target[:int(len(dataset)*cut_1)]
        self.Y_val = target[int(len(dataset)*cut_1):int(len(dataset)*cut_2)]
        self.Y_test = target[int(len(dataset)*cut_2):]

    def train_dataloader(self):
        train_dataset = TimeseriesDataset(self.X_train, self.Y_train,
                                          seq_len=self.seq_len)
        train_loader = DataLoader(train_dataset, 
                                  batch_size = self.batch_size, 
                                  shuffle = False, 
                                  num_workers = self.num_workers)
        
        return train_loader

    def val_dataloader(self):
        val_dataset = TimeseriesDataset(self.X_val, self.Y_val,
                                        seq_len=self.seq_len)
        val_loader = DataLoader(val_dataset, 
                                batch_size = self.batch_size, 
                                shuffle = False, 
                                num_workers = self.num_workers)

        return val_loader

    def test_dataloader(self):
        test_dataset = TimeseriesDataset(self.X_test, self.Y_test,
                                         seq_len=self.seq_len)
        test_loader = DataLoader(test_dataset, 
                                 batch_size = self.batch_size, 
                                 shuffle = False, 
                                 num_workers = self.num_workers)

        return test_loader
    '''
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
    '''

def cli_main():
    cli = LightningCLI(
        LitAutoEncoder, MyDataModule, seed_everything_default=1234, save_config_overwrite=True
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best")
    #predictions = cli.trainer.predict(ckpt_path="best")
    cli.trainer.save_checkpoint(os.path.join(test_path,'model/rain.ckpt'))
    #print(predictions[0])

if __name__ == "__main__":
    if not(os.path.exists(os.path.join(test_path,'model'))):
        os.system('mkdir -p '+os.path.join(test_path,'model') )
    cli_main()

#python LSTM_training.py --trainer.gpus 1 --trainer.max_epochs 10
