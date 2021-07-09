'''
@author Will Anderson

This file takes a full folder of images and splits them on an 80-10-10 split between training, testing, and validation and stores them in a new directory
'''

from pandas.core.frame import DataFrame
import torch
import torch.nn as nn
from torch.utils.data import dataloader
import torchvision
from torchvision import datasets
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import fnmatch
import os
from sklearn.model_selection import train_test_split
import cv2 as cv
import random
import pandas as pd


data_dir = '/home/williamanderson/Train Images/trainImages/'
file_list = os.listdir(data_dir)
# print(file_list)
image_paths = []
steering_angles = []
train_data = DataFrame(columns=['path', 'SA'])
test_data = DataFrame(columns=['path', 'SA'])
val_data = DataFrame(columns=['path', 'SA'])
pattern = "*.jpg"

for filename in file_list:
    print(filename)
    if fnmatch.fnmatch(filename, pattern):
        image_paths.append(os.path.join(data_dir, filename))
        if(filename[-13] == '-'):
            angle = float(filename[-13:-4])
        else:
            angle = float(filename[-12:-4])
        steering_angles.append(angle)
        origPath = data_dir + filename
        img = cv.imread(origPath)
        rand = random.uniform(0,1)
        if  rand >= 0.2:
            path = '/home/williamanderson/driveImages/trainImages/' + filename
            train_data.loc[len(train_data.index)] = [filename, angle]
            cv.imwrite(path, img)
        elif rand >= 0.1:
            path = '/home/williamanderson/driveImages/testImages/' + filename
            test_data.loc[len(test_data.index)] = [filename, angle]
            cv.imwrite(path, img)
        else:
            path = '/home/williamanderson/driveImages/valImages/' + filename
            val_data.loc[len(val_data.index)] = [filename, angle]
            cv.imwrite(path, img)


train_data.to_csv('/home/williamanderson/driveImages/trainImages/train.csv')
test_data.to_csv('/home/williamanderson/driveImages/testImages/test.csv')
val_data.to_csv('/home/williamanderson/driveImages/valImages/val.csv')

