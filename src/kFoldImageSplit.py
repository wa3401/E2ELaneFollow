'''
@author Will Anderson

This file takes a directory of images with steering angle and creates a csv file with filenames and steering angles
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
        path = '/home/williamanderson/kFold/' + filename
        train_data.loc[len(train_data.index)] = [filename, angle]
        cv.imwrite(path, img)


train_data.to_csv('/home/williamanderson/kFold/data.csv')
