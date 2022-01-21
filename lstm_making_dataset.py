# -*- coding: EUC-KR -*- 
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import configparser

print((datetime.now()+timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S'))
print("LSTM dataset make..")

# test set
config = configparser.ConfigParser()    
config.read('setting.ini', encoding='utf-8') 

start_date = datetime.strptime(config['date']['start_date'],'%Y-%m-%d %H:%M:%S')
end_date = datetime.strptime(config['date']['end_date'],'%Y-%m-%d %H:%M:%S')

test_path = config['path']['test_path']
model_path = config['path']['model_path']

sat_mis = np.load(os.path.join(test_path,'processed_data/sat_missing_date_ar.npy'),allow_pickle=True)
rad_mis = np.load(os.path.join(test_path,'processed_data/rad_missing_date_ar.npy'),allow_pickle=True)
sat =  np.load(os.path.join(test_path,'processed_data/sat_images_feature.npy'))
rad =  np.load(os.path.join(test_path,'processed_data/rad_images_feature.npy'))
aws = pd.read_csv(os.path.join(test_path,'aws/aws.csv'))
aws['�ð�'] = pd.to_datetime(aws['�ð�'])
df = pd.read_csv(os.path.join('./target.csv'))
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.loc[:,df.columns[1:]]

weather = ['��ǳ','ǳ��','ȣ��','�뼳','����','��ǳ����','����','��ǳ','Ȳ��',
          '����','��������','ȭ����','�޺�ǳ']
region = ['�����õ����⵵','�λꡤ��ꡤ��󳲵�','�뱸�����ϵ�',
'���֡����󳲵�','����ϵ�','��������������û����','��û�ϵ�','������','���ֵ�']

# making dataset
first = True
count = 0

rad_count = 0
sat_count = 0

sat_mean = np.mean(sat)
sat_std = np.std(sat)
rad_mean = np.mean(rad)
rad_std = np.std(rad)
temp_mean = np.mean(aws.loc[:,'��� ���'].values)
temp_std = np.std(aws.loc[:,'��� ���'].values)
rain_mean = np.mean(aws.loc[:,'�հ� ������'].values)
rain_std = np.std(aws.loc[:,'�հ� ������'].values)

with tqdm(total = 365*24) as pbar:
    while start_date <= end_date:
        pbar.update(1)
        check = False
        
        rad_c = Counter(rad_mis==start_date)
        sat_c = Counter(sat_mis==start_date)

        
        try :
            temperature = aws.loc[(aws['�ð�'] == start_date),'��� ���'].values
            temperature = (temperature-temp_mean)/temp_std
            rain = aws.loc[(aws['�ð�'] == start_date),'�հ� ������'].values
            rain = (rain-rain_mean)/rain_std
            temp_sat = (sat[sat_count].flatten() - sat_mean)/sat_std
            temp_rad = (rad[rad_count].flatten() - rad_mean)/rad_std

            temp = []
            for reg in region:
                temp.append(df.loc[(df['datetime']==start_date),reg+'_ȣ��'].values[0])
            heavy_rain = np.array(temp)
            heavy_rain = heavy_rain.astype(np.float64)
            #label smoothing
            heavy_rain[heavy_rain>0] = 0.9
            heavy_rain[heavy_rain<=0] = 0.1

            temp = []
            for reg in region:
                temp.append(df.loc[(df['datetime']==start_date),reg+'_����'].values[0])
            heat = np.array(temp)
            heat = heat.astype(np.float64)
            #label smoothing
            heat[heat>0] = 0.9
            heat[heat<=0] = 0.1

            temp = []
            for reg in region:
                temp.append(df.loc[(df['datetime']==start_date),reg+'_�뼳'].values[0])
            snow = np.array(temp)
            snow = snow.astype(np.float64)
            #label smoothing
            snow[snow>0] = 0.9
            snow[snow<=0] = 0.1

            comb = np.concatenate([rain,temperature,temp_sat,temp_rad])

            sat_count+=1
            rad_count+=1

            if (sat_c[True] != 0) :
                sat_count-=1
                check = True
            if  (rad_c[True] != 0) : 
                rad_count-=1
                check = True
            if check :
                start_date = start_date + timedelta(hours=1)
                continue
            if first :
                X = comb
                rains = heavy_rain
                heats = heat
                snows = snow
                first = False
            else : 
                X = np.vstack([X,comb])
                rains = np.vstack([rains,heavy_rain])
                heats = np.vstack([heats,heat])
                snows = np.vstack([snows,snow])
        except Exception as e :
            print(e)
            print(start_date)
        start_date = start_date + timedelta(hours=1)
        
np.save(os.path.join(test_path,'processed_data/LSTM_X.npy'),X)
np.save(os.path.join(test_path,'processed_data/LSTM_rain.npy'),rains)
np.save(os.path.join(test_path,'processed_data/LSTM_heat.npy'),heats)
np.save(os.path.join(test_path,'processed_data/LSTM_snow.npy'),snows)