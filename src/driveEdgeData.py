'''
@author Will Anderson

Define the DriveData which extends the Dataset class for training a lane following neural network
'''

import torch
import os
import pandas as pd
import cv2 as cv
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class DriveEdgeData(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.hmin, self.hmax, self.smin, self.smax, self.vmin, self.vmax = 0, 255, 0, 80, 180, 255

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        #print(img_path)
        startImage = cv.imread(img_path)
        rgbImage = cv.cvtColor(startImage, cv.COLOR_RGBA2RGB)
        yuvImage = cv.cvtColor(rgbImage, cv.COLOR_RGB2YUV)
        channels = cv.split(yuvImage)
        channels[0] = cv.equalizeHist(channels[0])
        merge = cv.merge(channels, yuvImage)
        normalized = cv.cvtColor(merge, cv.COLOR_YUV2RGB)
        hsvImage = cv.cvtColor(normalized, cv.COLOR_RGB2HSV)
        lower = (self.hmin, self.smin, self.vmin)
        upper = (self.hmax, self.smax, self.vmax)
        colorFilter = cv.inRange(hsvImage, lower, upper)
        edgeImage = cv.Canny(rgbImage, 60, 120)
        finalImage = colorFilter & edgeImage
        image = cv.resize(finalImage, (200, 75))
        # plt.imshow(image)
        # cv.waitKey(0)
        # plt.show()
        # image = cv.GaussianBlur(image, (3,3), 0)
        # image = cv.resize(image, (200, 75))
        label = torch.tensor(self.img_labels.iloc[idx, 2], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        
        return image, label