import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from tqdm import tqdm, trange
import numpy as np
from collections import Counter
from PIL import Image
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import configparser
print((datetime.now()+timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S'))
print("mask rcnn score calculation start..")

config = configparser.ConfigParser()    
config.read('setting.ini', encoding='utf-8') 

START_DATE = datetime.strptime(config['date']['start_date'],'%Y-%m-%d %H:%M:%S')
END_DATE = datetime.strptime(config['date']['end_date'],'%Y-%m-%d %H:%M:%S')
output_path = config['path']['output_path']
path = config['path']['test_path']

names = np.array(['LLJ', 'W_SNOW','E_SNOW','WET_SN','CUM_SN','COLD_FRONT','WARM_FRONT','OCC_FRONT','H_POINT','L_POINT'
                ,'HLJ', 'TYPOON', 'R_START', 'R_STOP','RA_SN','HAIL'])

modes = ['sat','rad']

eval_path = os.path.join(path,'processed_data')

column =[]
for i in range(400):
    column.append('gridcell'+str(i+1))

for name in names:
    for mode in modes:
        print(name,mode,'data_loading..')
        pred = np.load(os.path.join(eval_path,name+'_'+mode+'_pred.npy'))
        target = np.load(os.path.join(eval_path,name+'_'+mode+'_target.npy'))
        print('test dataset length :',len(pred))
        target[target >0.5] = 1
        target[target <=0.5] = 0
        precision, recall, thresholds = precision_recall_curve(target.flatten(), pred.flatten())
        f1 = 2*(precision*recall)/(precision+recall)
        f1 = np.nan_to_num(f1)
        print('f1-score :',np.max(f1))
        thr = thresholds[np.argmax(f1)]

        pred[pred >thr] = 1
        pred[pred <=thr] = 0
        try :
            tn, fp, fn, tp = confusion_matrix(target.flatten(), pred.flatten()).ravel()
        except :
            tn = confusion_matrix(target.flatten(), pred.flatten()).ravel()[0]
            fp=0
            fn=0
            tp=0
        res = np.array([[tn,fp],[fn,tp]])

        print(res)
        print('acc : ',(tn+tp)/(tn+fp+fn+tp))

        target = target.reshape(target.shape[0],400)
        pred = pred.reshape(pred.shape[0],400)
        target_csv = pd.DataFrame(target, columns=column)
        pred_csv = pd.DataFrame(pred, columns=column)

        target_csv.to_csv(os.path.join(output_path,'%s_%s_target.csv'%(name,mode)),index=False)
        pred_csv.to_csv(os.path.join(output_path,'%s_%s_prediction.csv'%(name,mode)),index=False)
        

print((datetime.now()+timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S'))
print("all task finish..")
