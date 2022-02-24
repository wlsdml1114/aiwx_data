# -*- coding: EUC-KR -*-
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix
import argparse
import os
import pandas as pd
import configparser
from datetime import datetime, timedelta
print((datetime.now()+timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S'))
print("LSTM score calculation")

config = configparser.ConfigParser()    
config.read('setting.ini', encoding='utf-8') 

test_path = config['path']['test_path']
output_path = config['path']['output_path']

names = ['rain','heat','snow']

if not(os.path.exists(output_path)):
    os.system('mkdir -p ',output_path)

for name in names:

    print(name,' data loading..')
    pred = np.load(os.path.join(test_path,'processed_data/%s_result.npy'%(name)))
    target = np.load(os.path.join(test_path,'processed_data/%s_origin.npy'%(name)))

    target[target >0.5] = 1
    target[target <=0.5] = 0

    thr = 0.517
    pred[pred >thr] = 1
    pred[pred <=thr] = 0
    tn, fp, fn, tp = confusion_matrix(target.flatten(), pred.flatten()).ravel()
    print(confusion_matrix(target.flatten(), pred.flatten()))
    print('acc : ',(tn+tp)/(tn+fp+fn+tp))
    print('f1-score : ',(tn+tp)/(tn+fp+fn+tp))

    region = ['서울·인천·경기도',
    '부산·울산·경상남도',
    '대구·경상북도',
    '광주·전라남도',
    '전라북도',
    '대전·세종·충청남도',
    '충청북도',
    '강원도',
    '제주도']
    columns = []
    for i in range(24):
        for reg in region:
            columns.append(str(i+1)+'hours'+'_'+reg)
            
    pred_csv = pd.DataFrame(pred, columns=columns)
    target_csv = pd.DataFrame(target, columns=columns)
    pred_csv.to_csv(os.path.join(output_path,'%s_prediction.csv'%(name)),index=False)
    target_csv.to_csv(os.path.join(output_path,'%s_target.csv'%(name)),index=False)
print((datetime.now()+timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S'))
