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
from datetime import datetime, timedelta
print((datetime.now()+timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S'))
print("sat image preprocessing..")

def delete_yellow_line_gk2a(pil_image):
    indx = [1,1,-1,-1]
    indy = [1,-1,1,-1]
    temp = pil_image
    for i in range(temp.size[0]):
        for j in range(temp.size[1]):
            r,g,b,a = temp.getpixel((i,j))
            if  (r==255) & (g==255) & (b==0):
                re_r, re_g, re_b = 0,0,0
                count = 0
                for k in range(4):
                    try :
                        r,g,b,a = temp.getpixel((i+indx[k],j+indy[k]))
                        if  not((r==255) & (g==255) & (b==0)):
                            re_r += r
                            re_g += g
                            re_b += b
                            count+=1
                    except :
                        continue
                if count != 0 :
                    temp.putpixel((i,j),(int(re_r/count),int(re_g/count),int(re_b/count)))
                else : 
                    temp.putpixel((i,j),(0,0,0))
    return temp


# test set 
config = configparser.ConfigParser()    
config.read('setting.ini', encoding='utf-8') 

START_DATE = datetime.strptime(config['date']['start_date'],'%Y-%m-%d %H:%M:%S')
END_DATE = datetime.strptime(config['date']['end_date'],'%Y-%m-%d %H:%M:%S')
Origin_path = config['path']['test_path']

gk2a_path = "sat/png/%Y%m/%Y%m%d%H%M_KOMPSAT2A_KO_IR112.png"

if os.path.exists(os.path.join(Origin_path,'processed_data/sat_images_ar.npy')):
    print('preprocessed image already exist')
    sys.exit()

NOW_DATE = START_DATE
images = []
use_date = []
missing_date=[]
with tqdm(total = 365*24) as pbar:
    while NOW_DATE <= END_DATE:
        pbar.update(1)
        try :
            img = Image.open(os.path.join(Origin_path,NOW_DATE.strftime(gk2a_path)))
            img = img.crop((0,22,900,922))
            img = delete_yellow_line_gk2a(img)
            img = img.convert('L')
            img = img.resize((900,900))
            imgarr = np.array(img)
            imgarr = imgarr.reshape(1,imgarr.shape[0],imgarr.shape[1])
            images.append(imgarr)
            use_date.append(NOW_DATE)
        except Exception as e :
            missing_date.append(NOW_DATE)
        NOW_DATE = NOW_DATE + timedelta(hours=1)

np.save(os.path.join(Origin_path,'processed_data/sat_images_ar.npy'),np.array(images))
np.save(os.path.join(Origin_path,'processed_data/sat_missing_date_ar.npy'),np.array(missing_date))
np.save(os.path.join(Origin_path,'processed_data/sat_use_date_ar.npy'),np.array(use_date))
