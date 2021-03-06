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
from dateutil.relativedelta import relativedelta

print((datetime.now()+timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S'))
print("LSTM dataset make..")

# test set
config = configparser.ConfigParser()    
config.read('setting.ini', encoding='CP949') 

START_DATE = datetime.strptime(config['date']['start_date'],'%Y-%m-%d %H:%M:%S')
END_DATE = datetime.strptime(config['date']['end_date'],'%Y-%m-%d %H:%M:%S')
INTERVAL_DATE = END_DATE-START_DATE

test_path = config['path']['test_path']
model_path = config['path']['model_path']
data_path = config['path']['data_path']

sat_mis = np.load(os.path.join(test_path,'processed_data/sat_missing_date_ar.npy'),allow_pickle=True)
rad_mis = np.load(os.path.join(test_path,'processed_data/rad_missing_date_ar.npy'),allow_pickle=True)
sat =  np.load(os.path.join(test_path,'processed_data/sat_images_feature.npy'))
rad =  np.load(os.path.join(test_path,'processed_data/rad_images_feature.npy'))

st_num = np.load('./st_num.npy')
data_path = config['path']['data_path']

aws_path = 'obs/aws/%Y/%Y%m.csv'
TEMP_DATE = START_DATE
first = True

while TEMP_DATE <= END_DATE:
    print(TEMP_DATE.strftime('%Y%m'),'AWS data loading..')
    try:
        temp = pd.read_csv(os.path.join(data_path,TEMP_DATE.strftime(aws_path)))
    except Exception as e :
        if 'Resource' in str(e):
            continue

    if first :
        aws = temp
        first = False
    else : 
        aws = aws.append(temp)
    TEMP_DATE = TEMP_DATE + relativedelta(months=1)

print('AWS preprocessing.. (it will take time)')
first = True
for st in st_num:
    temp = aws.loc[aws['지점번호']==st]
    if first :
        df = temp
        first = False
    else : 
        df = df.append(temp)

aws = df
aws['시각'] = pd.to_datetime(aws['시각'])
aws = aws.fillna(method='ffill')
aws = aws.fillna(method='bfill')
df = pd.read_csv(os.path.join('./target.csv'))
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.loc[:,df.columns[1:]]

weather = ['강풍','풍랑','호우','대설','건조','폭풍해일','한파','태풍','황사',
          '폭염','지진해일','화산재','급변풍']
region = ['서울·인천·경기도','부산·울산·경상남도','대구·경상북도',
'광주·전라남도','전라북도','대전·세종·충청남도','충청북도','강원도','제주도']

# making dataset
first = True
count = 0

rad_count = 0
sat_count = 0

sat_mean = np.mean(sat)
sat_std = np.std(sat)
rad_mean = np.mean(rad)
rad_std = np.std(rad)
temp_mean = np.mean(aws.loc[:,'평균 기온'].values)
temp_std = np.std(aws.loc[:,'평균 기온'].values)
rain_mean = np.mean(aws.loc[:,'합계 강수량'].values)
rain_std = np.std(aws.loc[:,'합계 강수량'].values)

with tqdm(total = (INTERVAL_DATE.days*24 + int(INTERVAL_DATE.seconds/3600) +1)) as pbar:
    while START_DATE <= END_DATE:
        pbar.update(1)
        check = False
        
        rad_c = Counter(rad_mis==START_DATE)
        sat_c = Counter(sat_mis==START_DATE)

        
        try :
            temperature = aws.loc[(aws['시각'] == START_DATE),'평균 기온'].values
            temperature = (temperature-temp_mean)/temp_std
            rain = aws.loc[(aws['시각'] == START_DATE),'합계 강수량'].values
            rain = (rain-rain_mean)/rain_std
            temp_sat = (sat[sat_count].flatten() - sat_mean)/sat_std
            temp_rad = (rad[rad_count].flatten() - rad_mean)/rad_std

            temp = []
            for reg in region:
                temp.append(df.loc[(df['datetime']==START_DATE),reg+'_호우'].values[0])
            heavy_rain = np.array(temp)
            heavy_rain = heavy_rain.astype(np.float64)
            #label smoothing
            heavy_rain[heavy_rain>0] = 0.9
            heavy_rain[heavy_rain<=0] = 0.1

            temp = []
            for reg in region:
                temp.append(df.loc[(df['datetime']==START_DATE),reg+'_폭염'].values[0])
            heat = np.array(temp)
            heat = heat.astype(np.float64)
            #label smoothing
            heat[heat>0] = 0.9
            heat[heat<=0] = 0.1

            temp = []
            for reg in region:
                temp.append(df.loc[(df['datetime']==START_DATE),reg+'_대설'].values[0])
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
                START_DATE = START_DATE + timedelta(hours=1)
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
            pass
        START_DATE = START_DATE + timedelta(hours=1)
        
np.save(os.path.join(test_path,'processed_data/LSTM_X.npy'),X)
np.save(os.path.join(test_path,'processed_data/LSTM_rain.npy'),rains)
np.save(os.path.join(test_path,'processed_data/LSTM_heat.npy'),heats)
np.save(os.path.join(test_path,'processed_data/LSTM_snow.npy'),snows)