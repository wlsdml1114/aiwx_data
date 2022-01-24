import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from tqdm import tqdm, trange
import numpy as np
from collections import Counter
from PIL import Image
import configparser
print((datetime.now()+timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S'))
print("mask rcnn dataset making..")

config = configparser.ConfigParser()    
config.read('setting.ini', encoding='utf-8') 

START_DATE = datetime.strptime(config['date']['start_date'],'%Y-%m-%d %H:%M:%S')
END_DATE = datetime.strptime(config['date']['end_date'],'%Y-%m-%d %H:%M:%S')
path = config['path']['test_path']


names = np.array(['LLJ', 'W_SNOW','E_SNOW','WET_SN','CUM_SN','COLD_FRONT','WARM_FRONT','OCC_FRONT','H_POINT','L_POINT'
                ,'HLJ', 'TYPOON', 'R_START', 'R_STOP','RA_SN','HAIL'])

in_dir = os.path.join(path,'processed_data/mask')
out_dir = os.path.join(path,'processed_data')

miss_sat = np.load(os.path.join(path,'processed_data/sat_missing_date_ar.npy'),allow_pickle=True)
miss_rad = np.load(os.path.join(path,'processed_data/rad_missing_date_ar.npy'),allow_pickle=True)
date_sat = np.load(os.path.join(path,'processed_data/sat_use_date_ar.npy'),allow_pickle=True)
date_rad = np.load(os.path.join(path,'processed_data/rad_use_date_ar.npy'),allow_pickle=True)

sat_image = np.load(os.path.join(path,'processed_data/sat_images_ar.npy'))
#sat_image.shape
#(60578, 1, 900, 900)

rad_image = np.load(os.path.join(path,'processed_data/rad_images_ar.npy'))
#rad_image.shape
#(61134, 1, 600, 600)


for name in names :
    if not(os.path.exists(os.path.join(out_dir,name))):
        os.system('mkdir -p '+os.path.join(out_dir,name))

for name in names :
    START_DATE = datetime.strptime(config['date']['start_date'],'%Y-%m-%d %H:%M:%S')
    END_DATE = datetime.strptime(config['date']['end_date'],'%Y-%m-%d %H:%M:%S')  

    while START_DATE <= END_DATE :
        sats = []
        rads = []
        masks = []
        dates = []
        files = os.listdir(os.path.join(in_dir,name,START_DATE.strftime('%Y%m')))    
        if len(files) == 0:
            START_DATE = START_DATE+relativedelta(months=1)
            continue
        for i in trange(len(files),desc='name : %s, date : %s'%(name,START_DATE.strftime('%Y%m'))):
            file = files[i]
            now_date = datetime.strptime(file,'%Y%m%d%H%M.npy')
            if Counter(now_date == miss_sat)[True] == 1:
                continue
            if Counter(now_date == miss_rad)[True] == 1:
                continue
            sat_index = np.where(now_date == date_sat)[0][0]
            rad_index = np.where(now_date == date_rad)[0][0]
            temp = np.load(os.path.join(in_dir,name,START_DATE.strftime('%Y%m'),file))
            rad = rad_image[rad_index]
            rad[rad==250]=0
            rad[rad==255]=0   
            temp_rad = np.zeros((1,900,900))
            if now_date < datetime(2019,1,7):
                temp_rad[0,265:842,235:760] = np.array(Image.fromarray(rad[0]).resize((525,577)))
            else :
                temp_rad[0,322:834,230:740] = np.array(Image.fromarray(rad[0]).resize((510,512)))
            temp_rad = temp_rad.astype(np.uint8)
            dates.append(now_date)
            sats.append(sat_image[sat_index])
            rads.append(temp_rad)
            masks.append(temp)
        
        np.save(os.path.join(out_dir,name,START_DATE.strftime('%Y%m_dates.npy')),np.array(dates))
        np.save(os.path.join(out_dir,name,START_DATE.strftime('%Y%m_sat.npy')),np.array(sats))
        np.save(os.path.join(out_dir,name,START_DATE.strftime('%Y%m_rad.npy')),np.array(rads))
        np.save(os.path.join(out_dir,name,START_DATE.strftime('%Y%m_mask.npy')),np.array(masks))
        
        START_DATE = START_DATE+relativedelta(months=1)

#month to all
for name in names :
    START_DATE = datetime.strptime(config['date']['start_date'],'%Y-%m-%d %H:%M:%S')
    END_DATE = datetime.strptime(config['date']['end_date'],'%Y-%m-%d %H:%M:%S')
    first = True
    while START_DATE <= END_DATE :
        print(name,START_DATE)
        try :
            sat = np.load(os.path.join(out_dir,name,START_DATE.strftime('%Y%m_rad.npy')))
            if first :
                first = False
                final = sat
            else :
                final = np.vstack([final,sat])
        except Exception as e:
            pass
        START_DATE = START_DATE+relativedelta(months=1)
    np.save(os.path.join(out_dir,name,'rad.npy'),final)

    
for name in names :
    START_DATE = datetime.strptime(config['date']['start_date'],'%Y-%m-%d %H:%M:%S')
    END_DATE = datetime.strptime(config['date']['end_date'],'%Y-%m-%d %H:%M:%S')
    first = True
    while START_DATE <= END_DATE :
        print(name,START_DATE)
        try :
            sat = np.load(os.path.join(out_dir,name,START_DATE.strftime('%Y%m_mask.npy')))
            if first :
                first = False
                final = sat
            else :
                final = np.vstack([final,sat])
        except Exception as e:
            pass
        START_DATE = START_DATE+relativedelta(months=1)
    np.save(os.path.join(out_dir,name,'mask.npy'),final)

    
for name in names :
    START_DATE = datetime.strptime(config['date']['start_date'],'%Y-%m-%d %H:%M:%S')
    END_DATE = datetime.strptime(config['date']['end_date'],'%Y-%m-%d %H:%M:%S')
    first = True
    while START_DATE <= END_DATE :
        print(name,START_DATE)
        try :
            sat = np.load(os.path.join(out_dir,name,START_DATE.strftime('%Y%m_sat.npy')))
            if first :
                first = False
                final = sat
            else :
                final = np.vstack([final,sat])
        except Exception as e:
            pass
        START_DATE = START_DATE+relativedelta(months=1)
    np.save(os.path.join(out_dir,name,'sat.npy'),final)
