'''
@author Will Anderson

Define the DriveData which extends the Dataset class for training a lane following neural network
'''

import torch
import os
import pandas as pd
import cv2 as cv
from torch.utils.data import Dataset

class DriveData(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        #print(img_path)
        startImage = cv.imread(img_path)
        image = cv.cvtColor(startImage, cv.COLOR_RGB2GRAY)
        label = torch.tensor(self.img_labels.iloc[idx, 2], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        
        return image, label