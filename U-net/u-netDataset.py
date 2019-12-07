#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread, imread_collection
from skimage.transform import resize
import os
from tqdm import tqdm
import tensorflow as tf
import time

#%%
DATASET_PATH='D:\\Dataset\\nuke'
TRAIN_PATH = DATASET_PATH+"\\stage1_train"
IMG_SIZE = 128
#%%
cases = os.listdir(TRAIN_PATH)
images = np.zeros((len(cases), IMG_SIZE, IMG_SIZE,3), dtype=np.uint8)
labels = np.zeros((len(cases), IMG_SIZE, IMG_SIZE,1), dtype=np.bool)

for i, c in enumerate(cases):
    #get and resize image
    img = imread(TRAIN_PATH+"\\{}\\images\\{}.png".format(c,c))[:,:,:3]
    img = resize(img, (IMG_SIZE, IMG_SIZE, 3))
    print(img.shape)
    images[i] =img

    #get and resize mask
    label = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.bool)
    masks = imread_collection(TRAIN_PATH+ "\\{}\\masks\\*".format(c)).concatenate()
    print(masks.shape)
    masks = np.expand_dims(masks,-1)
    print(masks.shape)
    for j, m in enumerate(masks):
        m = resize(m, (IMG_SIZE, IMG_SIZE, 1), mode='constant')
        label = np.maximum(label,m)
    print(label.shape)
    plt.imshow(label, cmap='gray')
    plt.show()
    break
    labels[i] =label
#%%
