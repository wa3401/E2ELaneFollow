'''
@author Will Anderson

This file is designed to train a neural network to follow a lane 
based on Images capured from an f1tenth car
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.io import read_image
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from model1 import NvidiaModel
from driveData1 import DriveData
from driveEdgeData import DriveEdgeData
from ryanModel import RyanModel
#from torch.utils.tensorboard import SummaryWriter

#If a cuda device is avaiable, set that as device
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparamaters
num_epochs = 100
batch_size = 4
learning_rate = 0.1

# Define Transforms
transform1 = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

transform2 = transforms.Compose(
    [transforms.ToTensor()]
)

#Define Train Dataset
train_dir = '/home/williamanderson/procDriveImages/trainImages/'
train_csv = train_dir + 'train.csv'
train_set = DriveData(annotations_file=train_csv, img_dir=train_dir, transform=transform2)
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Define test dataset
test_dir = '/home/williamanderson/procDriveImages/testImages/'
test_csv = test_dir + 'test.csv'
test_set = DriveData(annotations_file=test_csv, img_dir=test_dir, transform=transform2)
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

#Define validation dataset
val_dir = '/home/williamanderson/procDriveImages/valImages/'
val_csv = val_dir + 'val.csv'
val_set = DriveData(annotations_file=val_csv, img_dir=val_dir, transform=transform2)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)




# Prints the shapes of feature being fed into the network
images, labels = next(iter(train_dataloader))
print(f"Train Feature batch shape: {images.size()}")
print(f"Train Labels batch shape: {labels.size()}")
images, labels = next(iter(val_dataloader))
print(f"Val Feature batch shape: {images.size()}")
print(f"Val Labels batch shape: {labels.size()}")

#Define model
model = RyanModel().to(device)
#model.load_state_dict(torch.load('/home/williamanderson/LaneFollower/savedModel.ipynb'))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

n_total_steps = len(train_set)
num_total_steps=len(train_dataloader)
num_total_steps_in_test=len(test_dataloader)
for epoch in range(num_epochs):
    averageLoss = 0.0
    numInRange = 0
    for i, (images, angles) in enumerate(train_dataloader):
        images = images.to(device)
        angles = angles.to(device)
        angles = angles.view(-1, 1)
        outputs = model(images)
        #print(outputs)
        loss = criterion(outputs,angles)
        averageLoss +=loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss < 0.0025:
            numInRange = numInRange+1

    averageLoss/=num_total_steps
    print(f"Epoch: {epoch} Loss: {averageLoss} Num in range: {numInRange/num_total_steps}")

    with torch.no_grad():
        totalInRange = 0
        for i, (images, angles) in enumerate(test_dataloader):
            images = images.to(device)
            angles = angles.to(device)
            angles = angles.view(-1, 1)
            outputs = model(images)
            loss = criterion(outputs, angles)
            if loss < 0.0025:
                totalInRange = totalInRange + 1
        print(f"Accuracy: {100*totalInRange/num_total_steps_in_test}%")


print('Finished Training')

PATH = './train_network.pth'
torch.save(model.state_dict(), PATH)