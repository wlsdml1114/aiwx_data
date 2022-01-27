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
print("rad image preprocessing..")

def delete_black_line(pil_image):
    temp = pil_image
    indx = [1,1,-1,-1]
    indy = [1,-1,1,-1]
    for i in range(temp.size[0]):
        for j in range(temp.size[1]):
            r,g,b = temp.getpixel((i,j))
            if  (r==140) &(g==140)&(b==140):
                temp.putpixel((i,j),(250,250,250))
            elif  (r==163) &(g==163)&(b==163):
                temp.putpixel((i,j),(250,250,250))
            elif  (r==224) &(g==224)&(b==224):
                temp.putpixel((i,j),(250,250,250))
    
    for i in range(temp.size[0]):
        for j in range(temp.size[1]):
            r,g,b = temp.getpixel((i,j))
            if (r==0) &(g==0)&(b==0):
                re_r, re_g, re_b = 0,0,0
                count = 0
                for k in range(4):
                    try :
                        r,g,b = temp.getpixel((i+indx[k],j+indy[k]))
                        if  not((r==0) & (g==0) & (b==0)):
                            re_r += r
                            re_g += g
                            re_b += b
                            count+=1
                    except :
                        continue
                if count != 0 :
                    temp.putpixel((i,j),(int(re_r/count),int(re_g/count),int(re_b/count)))
                else : 
                    temp.putpixel((i,j),(250,250,250))
    return temp
def delete_black_line_(pil_image):
    temp = pil_image
    indx = [1,1,-1,-1]
    indy = [1,-1,1,-1]
    pxl_list = []
    for i in range(temp.size[0]):
        for j in range(temp.size[1]):
            r = temp.getpixel((i,j))
            if  (r==28) :
                temp.putpixel((i,j),(0))
                pxl_list.append((i,j))
            elif  (r==25) :
                temp.putpixel((i,j),(0))
    for i in range(len(pxl_list)):
        re= 0,0,0
        count = 0
        for k in range(4):
            try:
                r = temp.getpixel((pxl_list[i][0]+indx[k],pxl_list[i][1]+indy[k]))
            except :
                continue
        try:
            if count != 0 :
                temp.putpixel((i,j),(int(re/count)))
            else : 
                temp.putpixel((i,j),(0))
        except:
            continue
    return temp
def fill_black(array, coor_x, coor_y):
    index = [1,1,-1,-1]
    indey = [1,-1,1,-1]
    for i in range(len(coor_x)):
        cum = 0
        count = 0
        for k in range(4):
            try :
                temp = array[coor_x[i]+index[k],coor_y[i]+indey[k]]
                cum+= temp
                count +=1
            except :
                continue
        try:
            array[coor_x[i],coor_y[i]] = int(cum/count)
        except:
            continue
    return array


# test set
config = configparser.ConfigParser()    
config.read('setting.ini', encoding='utf-8') 

START_DATE = datetime.strptime(config['date']['start_date'],'%Y-%m-%d %H:%M:%S')
END_DATE = datetime.strptime(config['date']['end_date'],'%Y-%m-%d %H:%M:%S')
path = config['path']['test_path']

if os.path.exists((os.path.join(path,'processed_data/rad_images_ar.npy')):
    sys.exit()

NOW_DATE = START_DATE
images = []
missing_date = []
use_date =[]
before_crop = (1,1,526,575)
after_crop = (1,21,601,621)

img = Image.open('./rad_template_1.png')
img_1 = np.array(img)
x_1,y_1 = np.where(img_1 == 0)

img = Image.open('./rad_template_2.png')
img_2 = np.array(img)
x_2,y_2 = np.where(img_2 == 28)

img = Image.open('./rad_template_3.png')
img_3 = np.array(img)
x_3,y_3 = np.where(img_3 == 0)


with tqdm(total = 365*24) as pbar:
    NOW_DATE = START_DATE
    while NOW_DATE <= END_DATE:
        pbar.update(1)
        if NOW_DATE <= datetime(2019,1,7,9,0):
            crop_tuple = before_crop
            template = img_1
            indx = x_1
            indy = y_1
            preprocessing = delete_black_line
        elif (NOW_DATE >= datetime(2019,1,7,10,0)) & (NOW_DATE <= datetime(2020,10,7,9,0)):
            crop_tuple = after_crop
            template = img_2
            indx = x_2
            indy = y_2
            preprocessing = delete_black_line_
        else:
            crop_tuple = after_crop
            template = img_3
            indx = x_3
            indy = y_3
            preprocessing = delete_black_line
        try:
            
            img = Image.open(os.path.join(path,NOW_DATE.strftime('com/%Y%m/%Y%m%d%H%M_RAD_COMP.png')))
            img = img.crop(crop_tuple)
            img = preprocessing(img)
            img = img.convert('L')
            img = img.resize((600,600))
            imgarr = np.array(img)
            imgarr = fill_black(imgarr,indx,indy)
            imgarr = imgarr.reshape(1,imgarr.shape[0],imgarr.shape[1])
            images.append(imgarr)
            use_date.append(NOW_DATE)

        except :
            missing_date.append(NOW_DATE)

        NOW_DATE = NOW_DATE+timedelta(hours=1)

np.save(os.path.join(path,'processed_data/rad_images_ar.npy'),np.array(images))
np.save(os.path.join(path,'processed_data/rad_missing_date_ar.npy'),np.array(missing_date))
np.save(os.path.join(path,'processed_data/rad_use_date_ar.npy'),np.array(use_date))
