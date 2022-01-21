import argparse
import configparser
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
from tqdm import tqdm, trange
import pandas as pd
if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
    
cuda = torch.device('cuda')

print((datetime.now()+timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S'))
print("LSTM evaluation..")

class TimeseriesDataset(Dataset):   
    '''
    Custom Dataset subclass. 
    Serves as input to DataLoader to transform X 
      into sequence data using rolling window. 
    DataLoader using this dataset will output batches 
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs. 
    '''
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
    """
    >>> LitAutoEncoder()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LitAutoEncoder(
      (encoder): ...
      (decoder): ...
    )
    """

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


config = configparser.ConfigParser()    
config.read('setting.ini', encoding='utf-8') 

test_path = config['path']['test_path']
model_path = config['path']['model_path']

names = ['rain','heat','snow']

for name in names :
    #set root
    timestep = 24
    seq_len = 24
    print('dataset X loading..')
    dataset = np.load(os.path.join(test_path,'processed_data/LSTM_X.npy'))

    print(name,'target loading..')
    target = np.load(os.path.join(test_path,'processed_data/LSTM_'+name+'.npy'))

    print(name,'model loading..')
    model = LitAutoEncoder.load_from_checkpoint(os.path.join(model_path,name+'.ckpt'))
    model = model.cuda()
    model.eval()

    original_list = []
    result_list = []

    print(name,'predict start')
    for start in trange(len(dataset)-timestep-seq_len-1):
        test_set = dataset[start:start+timestep]
        original = target[start+timestep:start+timestep+seq_len]
        original_list.append(original.flatten())
        test_set = torch.Tensor(test_set).cuda()
        test_set = test_set.reshape(1,test_set.shape[0],test_set.shape[1])


        output,state,hidden = model.first_predict(test_set)
        temp_result = output.cpu().detach().numpy()
        state= state[:,-1]
        test_set = torch.cat([test_set[:,1:],state.reshape(1,state.shape[0],state.shape[1])],dim=1)

        for i in range(seq_len-1):
            output,state,hidden = model.second_predict(test_set,hidden)
            temp_result = np.concatenate([temp_result,output.cpu().detach().numpy()])
            state= state[:,-1]
            test_set = torch.cat([test_set[:,1:],state.reshape(1,state.shape[0],state.shape[1])],dim=1)
        
        result_list.append(np.array(temp_result).flatten())
    np.save(os.path.join(test_path,'processed_data/'+name+'_result.npy'),np.array(result_list))
    np.save(os.path.join(test_path,'processed_data/'+name+'_origin.npy'),np.array(original_list))