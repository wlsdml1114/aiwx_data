import numpy as np
import json, os
import cv2
from datetime import datetime, timedelta
from PIL import Image, ImageDraw
from dateutil.relativedelta import relativedelta
from tqdm import tqdm, trange
import configparser
print((datetime.now()+timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S'))
print("geojson to mask preprocessing..")

def cal_dist(mlat, mlon, olat, olon):
    return np.sqrt((mlat - olat)**2 + (mlon - olon)**2)

def find_nearest(olat, olon, mlat, mlon):
    dist = cal_dist(mlat, mlon, olat, olon)
    nx, ny = np.unravel_index(dist.argmin(), dist.shape)
    return ny, nx


def make_line(pos_list, array):
    img_array = Image.fromarray(array)
    draw = ImageDraw.Draw(img_array)
    draw.line((pos_list), fill='white', width=100)
    re_array = np.array(img_array)
    re_array[np.where(re_array>100.0)]=1
    return(re_array)

def find_value(sdate, edate, in_dir, out_dir):
    fmt = '%Y%m'
    dt_sdate = datetime.strptime(sdate, fmt)
    dt_edate = datetime.strptime(edate, fmt)
    now = dt_sdate
    grid = np.load('./ea_2km_latlong.npy')
    mlat = grid[0,:,:]
    mlon = grid[1,:,:]
    names = np.array(['LLJ', 'W_SNOW','E_SNOW','WET_SN','CUM_SN','COLD_FRONT','WARM_FRONT','OCC_FRONT','H_POINT','L_POINT'
            ,'HLJ', 'TYPOON', 'R_START', 'R_STOP','RA_SN','HAIL'])
    

    while now <= dt_edate:
        print(now)
        str_now = datetime.strftime(now, '%Y%m')
        file_list = os.listdir('%s/%s'%(in_dir, str_now))
        count = 0
        for name in names :
            if not(os.path.exists(os.path.join(out_dir,name,str_now))):
                os.system('mkdir -p '+os.path.join(out_dir,name,str_now))
        if len(file_list) == 0:
            continue
        else:
            for i in trange(len(file_list)):
                file_name = file_list[i]
                check = np.repeat(0,16)
                masks = [np.zeros((3000,2600)) for i in range(16)]
                with open('%s/%s/%s'%(in_dir, str_now, file_name), 'r') as json_file:
                    json_data = json.load(json_file)
                '''
                now_color = 1
                name_count = np.repeat(0,16)
                for datas in json_data['features']:
                    now_name =  datas['properties']['name']
                    now_index = np.where(names == now_name)[0][0]
                    name_count[now_index] +=1
                if len(np.where(name_count>=2)[0]) == 0 :
                    continue
                allow_name = names[np.where(name_count>=2)[0]]

                for datas in json_data['features']:
                    now_name =  datas['properties']['name']
                    if len(np.where(allow_name == now_name)[0]) == 0 :
                        continue
                    now_index = np.where(names == now_name)[0][0]
                '''
                now_color = 1
                for datas in json_data['features']:
                    now_name =  datas['properties']['name']
                    now_index = np.where(names == now_name)[0][0]
                    now_type = datas['geometry']['type']
                    spot_list = datas['geometry']['coordinates']
                    pos_list = []
                    if now_type == 'Point' :
                        spot_list = [spot_list]
                    if now_type == 'Polygon' :
                        spot_list = spot_list[0]
                    try : 
                        for olon, olat in spot_list:
                            pos_list.append(find_nearest(olat, olon, mlat, mlon))
                    except Exception as e:
                        print(e)
                        print(str_now,file_name)
                    pos_list = np.array(pos_list)
                    check[now_index] +=1
                    if now_type == 'LineString':
                        masks[now_index] = cv2.polylines(masks[now_index],[pos_list],isClosed=False,color=now_color,thickness=50)
                    elif now_type == 'Point' : 
                        masks[now_index] = cv2.circle(masks[now_index],pos_list[0],50,now_color,-1)
                    elif now_type == 'Polygon' :
                        masks[now_index] = cv2.fillPoly(masks[now_index],[pos_list],now_color)
                    now_color+=1
                for i in range(len(names)):
                    if check[i]>=1:
                        tempmask = masks[i]
                        tempmask = np.rot90(tempmask)
                        tempmask = Image.fromarray(tempmask)
                        tempmask = tempmask.crop((1050,850,1950,1750))
                        tempmask = np.array(tempmask)
                        tempmask = tempmask.astype(np.uint8)
                        np.save('%s/%s/%s/%s.npy'%(out_dir, names[i],str_now,file_name[:12]), tempmask)
        now = now+relativedelta(months=1)

config = configparser.ConfigParser()    
config.read('setting.ini', encoding='utf-8') 

START_DATE = datetime.strptime(config['date']['start_date'],'%Y-%m-%d %H:%M:%S')
END_DATE = datetime.strptime(config['date']['end_date'],'%Y-%m-%d %H:%M:%S')
path = config['path']['test_path']

sdate = START_DATE.strftime('%Y%m')
edate = END_DATE.strftime('%Y%m')
in_dir = os.path.join(path,'geojson')
out_dir = os.path.join(path,'processed_data/mask')
find_value(sdate, edate, in_dir, out_dir)
