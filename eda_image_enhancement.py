# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 16:43:51 2021

@author: tawate
"""

import tensorflow as tf
import os
import pandas as pd
import numpy as np
import seaborn as sns
import cv2
import random
import ffmpeg
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import logging 
from IPython.display import Video
from tqdm import tqdm
import warnings

def RecoverCLAHE(sceneRadiance):
    clahe = cv2.createCLAHE(clipLimit=7, tileGridSize=(14, 14))
    for i in range(3):
        sceneRadiance[:, :, i] = clahe.apply((sceneRadiance[:, :, i]))
    return sceneRadiance


direc = 'C:/Users/tawate/OneDrive - SAS/01_Training/08_Kaggle/Barrier_Reef_Ob_Detection/Data/'
image_dir = 'C:/Users/tawate/OneDrive - SAS/01_Training/08_Kaggle/Barrier_Reef_Ob_Detection/Data/train_images/'

train = pd.read_csv(direc + 'train.csv')
test = pd.read_csv(direc + 'test.csv')


# EDA for Image Preprocessing

cm = sns.light_palette("green", as_cmap = True)
pd.option_context('display.max_colwidth', 100)

# SEED EVERYTHING
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)

# Check out Training Data
img_og = plt.imread(image_dir + 'video_1/9101.jpg')
img_9101 = cv2.imread(image_dir + 'video_1/9101.jpg')

train['image_path'] = image_dir + 'video_' + train['video_id'].astype(str) + '/' + train['video_frame'].astype(str) + '.jpg'

train.head().style.set_properties(**{'background-color': 'black',
                           'color': 'lawngreen',
                           'border-color': 'white'})

train.info()

train[train.annotations.str.len() > 2].head(5)
train['annotations'] = train['annotations'].apply(eval)
train_v2 = train[train.annotations.str.len() > 0].reset_index(drop = True)


# Data Preparation
dest_path1 = direc + "train_images/clahe_img"
os.mkdir(dest_path1)

for img_path in tqdm(train_v2["image_path"][0:400]):

    image = plt.imread(img_path)
    image_cv = cv2.imread(img_path)
    img_clahe = RecoverCLAHE(image_cv)
    file_name = img_path.split("/")[-1]
    
    cv2.imwrite(dest_path1+"/"+file_name, img_clahe)

dest_path2 = direc + 'train_images/annot_img'
os.mkdir(dest_path2)

idx = 0
for img_idx in tqdm(train_v2["image_path"][0:400]):
    file_name = img_idx.split("/")[-1] 
    img_path = os.path.join(dest_path1,file_name)
    image = plt.imread(img_path)

#     idx = int(img_path.split("/")[-1].split(".")[0])
    for i in range(len(train_v2["annotations"][idx])):
        file_name = img_path.split('/')
        b_boxs = train_v2["annotations"][idx][i]
        x,y,w,h = b_boxs["x"],b_boxs["y"],b_boxs["width"],b_boxs["height"]

        image = cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 3)
        image = cv2.putText(image, 'starfish', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imwrite(dest_path2+"/"+file_name, image)
    idx +=1
#     print(img_path)



# Image Analysis
plt.figure(figsize = (12,15))

plt.rcParams["figure.figsize"] = [20.00, 10.50]
plt.rcParams["figure.autolayout"] = True

fig, ax = plt.subplots(figsize=(20,6))
ax.imshow(plt.imread(image_dir + 'video_0/40.jpg'))
newax = fig.add_axes([0.3,0.3,0.6,0.7], anchor='NE', zorder=1)

newax.axis('off')
plt.show();

# Check if images have same size
img_sizes = []
for i in train_v2["image_path"]:
    img_sizes.append(plt.imread(i).shape)

np.unique(img_sizes)

# Image with Annotations
plt.figure(figsize = (12,15))

plt.rcParams["figure.figsize"] = [20.00, 10.50]
plt.rcParams["figure.autolayout"] = True

fig, ax = plt.subplots(figsize=(20,6))
ax.imshow(plt.imread(dest_path2 + "/40.jpg"))
newax = fig.add_axes([0.26,0.2,0.6,0.6], anchor='NE', zorder=1)

newax.axis('off')
plt.show()

# 80 % of data is not annotated
count_bbox = []
for i in train["annotations"]:
    count_bbox.append(len(i))
    
from collections import defaultdict


bbox_dict = defaultdict(int)

for val in count_bbox:
    bbox_dict[val] += 1
    
# data = {'milk': 60, 'water': 10}
names = list(bbox_dict.keys())
values = list(bbox_dict.values())

N = len(list(bbox_dict.values()))
menMeans = list(bbox_dict.values())
ind = np.arange(N)

plt.rcParams["figure.figsize"] = [20.00, 10.50]
plt.rcParams["figure.autolayout"] = True
fig, ax = plt.subplots(figsize=(20,6))
# plt.figure(figsize=(20,6))
ax.bar(ind,menMeans,width=0.4)
plt.xticks(np.arange(0, N, step=1))
plt.title("Number of bounding box VS Count of Bounding Box",fontsize=20)
# fig.suptitle('test title', fontsize=20)
plt.xlabel('Number of bounding box', fontsize=18)
plt.ylabel('Count', fontsize=16)
for index,data in enumerate(menMeans):
    plt.text(x=index , y =data+1 , s=f"{data}" , fontdict=dict(fontsize=20))
newax = fig.add_axes([0.3,0.35,0.6,0.5], anchor='NE', zorder=1)

newax.axis('off')
plt.show()

# Applying Histogram Equalization which increaes the contrast of the image
def he_hsv(img_demo):
    img_hsv = cv2.cvtColor(img_demo, cv2.COLOR_RGB2HSV)
    
    #Histogram equalisation on the V-Channel
    img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
    
    # Convert image back from HSV to RGB
    image_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    
    return image_hsv

# Applying Histogram Eq on each Channel with repo code
def RecoverHE(sceneRadiance):
    for i in range(3):
        sceneRadiance[:, :, i] = cv2.equalizeHist(sceneRadiance[:, :, i])
    return sceneRadiance

# Applying CLAHE (Contrast Limited AHE) instead of Histogram EQ. Used to prevent the overamplification of noise that rises from Hist EQ.
def clahe_hsv(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    h, s, v = hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]
    clahe = cv2.createCLAHE(clipLimit = 15.0, tileGridSize = (20,20))
    v = clahe.apply(v)
    
    hsv_img = np.dstack((h,s,v))
    
    rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    return rgb

# Clahe with rep code
def RecoverCLAHE(sceneRadiance):
    clahe = cv2.createCLAHE(clipLimit=7, tileGridSize=(14, 14))
    for i in range(3):

        # sceneRadiance[:, :, i] =  cv2.equalizeHist(sceneRadiance[:, :, i])
        sceneRadiance[:, :, i] = clahe.apply((sceneRadiance[:, :, i]))


    return sceneRadiance

# Plotting the transformed image
def plot_img(img_dir, num_items, func, mode):
    img_list = random.sample(os.listdir(img_dir), num_items)
    
    for i in range(len(img_list)):
        full_path = img_dir + '/' + img_list[i]
        img_temp1 = plt.imread(full_path)
        img_temp_cv = cv2.imread(full_path)
        plt.figure(figsize = (20,15))
        plt.subplot(1,2,1)
        plt.imshow(img_temp1)
        plt.subplot(1,2,2)
        if mode == 'plt':
            plt.imshow(func(img_temp1));
        elif mode == 'cv2':
            plt.imshow(func(img_temp_cv));
            
            
            
def plot_img_tf(img_dir,num_items,func,mode):
    img_list = random.sample(os.listdir(img_dir), num_items)
    full_path = img_dir + '/' + img_list[0]
    img_temp_plt = plt.imread(full_path)
    img_temp_cv = cv2.imread(full_path)
    if mode=="plt":
        
        img_stack = np.hstack((img_temp_plt,func(img_temp_plt)))
        plt.figure(figsize=(20,15))
        plt.imshow(img_stack);
        plt.title("Original Image VS Enhanced Image",fontsize=25)
        plt.axis("off")
        plt.show()
    if mode=="cv2":
        
        img_stack = np.hstack((img_temp_cv,func(img_temp_cv)))
        plt.figure(figsize=(20,15))
        plt.imshow(img_stack);
        plt.title("Original Image VS Enhanced Image",fontsize=25)
        plt.axis("off")
        plt.show()

    for i in range(1, len(img_list)):
        full_path = img_dir + '/' + img_list[i]
        img_temp_plt = plt.imread(full_path)
        img_temp_cv = cv2.imread(full_path)
        if mode=="plt":
            img_stack = np.hstack((img_temp_plt,func(img_temp_plt)));
            plt.figure(figsize=(20,15))
            plt.imshow(img_stack);
            plt.axis("off")
            plt.show()
        if mode=="cv2":
            img_stack = np.hstack((img_temp_cv,func(img_temp_cv)));
            plt.figure(figsize=(20,15))
            plt.imshow(img_stack);
            plt.axis("off")
            plt.show()
            


vid_0_dir = image_dir + 'video_0'
num_items1 = 4
num_items = 4

plot_img(vid_0_dir, num_items1, he_hsv, 'cv2')
plot_img(vid_0_dir, num_items1, RecoverHE, 'cv2')
plot_img(vid_0_dir, num_items1, clahe_hsv, 'plt')
plot_img(vid_0_dir, num_items1, RecoverCLAHE, 'cv2')

plot_img_tf(vid_0_dir,num_items,tfa.image.equalize,"plt")

"""
Adding code for git commit
"""
    
    
    
    

