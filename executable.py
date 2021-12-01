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
dest_path1 = "./clahe_img"
os.mkdir(dest_path1)

for img_path in tqdm(train_v2["image_path"][0:400]):

    image = plt.imread(img_path)
    image_cv = cv2.imread(img_path)
    img_clahe = RecoverCLAHE(image_cv)
    file_name = img_path.split("/")[-1]
    
    cv2.imwrite(dest_path1+"/"+file_name, img_clahe)

dest_path1 = "./annot_img"
os.mkdir(dest_path1)

idx = 0
for img_idx in tqdm(train_v2["image_path"][0:400]):
    file_name = img_idx.split("/")[-1] 
    img_path = os.path.join("./clahe_img",file_name)
    image = plt.imread(img_path)

#     idx = int(img_path.split("/")[-1].split(".")[0])
    for i in range(len(train_v2["annotations"][idx])):
        file_name = img_path.split('/')[-1]
        b_boxs = train_v2["annotations"][idx][i]
        x,y,w,h = b_boxs["x"],b_boxs["y"],b_boxs["width"],b_boxs["height"]

        image = cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 3)
        image = cv2.putText(image, 'starfish', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imwrite(dest_path1+"/"+file_name, image)
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
ax.imshow(plt.imread("./annot_img/40.jpg"))
newax = fig.add_axes([0.26,0.2,0.6,0.6], anchor='NE', zorder=1)


newax.axis('off')
plt.show()






