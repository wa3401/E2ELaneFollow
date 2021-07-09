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
from model import NvidiaModel
from driveData import DriveData
#from torch.utils.tensorboard import SummaryWriter

#If a cuda device is avaiable, set that as device
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparamaters
num_epochs = 500
batch_size = 64
learning_rate = 0.01

# Define Transforms
transform1 = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

transform2 = transforms.Compose(
    [transforms.ToTensor()]
)

#Define Train Dataset
train_dir = '/home/williamanderson/driveImages/trainImages/'
train_csv = train_dir + 'train.csv'
train_set = DriveData(annotations_file=train_csv, img_dir=train_dir, transform=transform1)
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Define test dataset
test_dir = '/home/williamanderson/driveImages/testImages/'
test_csv = test_dir + 'test.csv'
test_set = DriveData(annotations_file=test_csv, img_dir=test_dir, transform=transform1)
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

#Define validation dataset
val_dir = '/home/williamanderson/driveImages/valImages/'
val_csv = val_dir + 'val.csv'
val_set = DriveData(annotations_file=val_csv, img_dir=val_dir, transform=transform1)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)




# Prints the shapes of feature being fed into the network
images, labels = next(iter(train_dataloader))
print(f"Train Feature batch shape: {images.size()}")
print(f"Train Labels batch shape: {labels.size()}")
images, labels = next(iter(val_dataloader))
print(f"Val Feature batch shape: {images.size()}")
print(f"Val Labels batch shape: {labels.size()}")

#Define model
model = NvidiaModel().to(device)
#model.load_state_dict(torch.load('/home/williamanderson/LaneFollower/savedModel.ipynb'))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

#Learning Rate Scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, verbose=True)

n_total_steps = len(train_dataloader)
min_valid_loss = np.inf
step = 0
saved_epoch = 0

#Training Loop
for epoch in range(num_epochs):
    train_loss = 0.0
    model.train()
    for i, (images, labels) in enumerate(train_dataloader):

        #Send tensors to device and shape properly
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.view(-1, 1)

        #forward pass
        outputs = model(images)

        #loss calculation
        loss = criterion(outputs, labels)

        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss = loss.item() * images.size(0)
    scheduler.step()
        

    valid_loss = 0.0
    model.eval()
    for i, (images, labels) in enumerate(val_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.view(-1, 1)
        #print(f'Val image size: {images.size()}')
        #print(f'Val label size: {labels.size()}')

        outputs = model(images)

        loss = criterion(outputs, labels)

        valid_loss = loss.item() * images.size(0)
        #writer.add_scalar('Validation Loss', valid_loss, global_step=step)
        #step += 1

    with torch.no_grad():
        model.eval()
        n_correct = 0
        n_samples = 0

        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            for i in range(len(outputs)):
                n_samples += 1
                label = labels[i]
                #print(f'Test label: {label}')
                pred = outputs[i]
                #print(f'Test Predicted: {pred}')
                if abs(label - pred) < 0.01:
                    n_correct += 1

        print(f'Num Samples: {n_samples}')
        print(f'Num Correct: {n_correct}')
        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy for the network: {acc:.4f} %')
    model.train()

    

    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_loss / len(val_dataloader)}')
      
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        torch.save(model.state_dict(), '/home/williamanderson/LaneFollower/savedModel.ipynb')
        min_valid_loss = valid_loss
        saved_epoch = epoch

print(f'Finished Training, Saved Epoch {saved_epoch}')
print('Loading best Model')
model.load_state_dict(torch.load('/home/williamanderson/LaneFollower/savedModel.ipynb'))


with torch.no_grad():
    model.eval()
    n_correct = 0
    n_samples = 0

    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        for i in range(len(outputs)):
            n_samples += 1
            label = labels[i]
            #print(f'Test label: {label}')
            pred = outputs[i]
            #print(f'Test Predicted: {pred}')
            if abs(label - pred) < 0.01:
                n_correct += 1

    print(f'Num Samples: {n_samples}')
    print(f'Num Correct: {n_correct}')
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy for the network: {acc:.4f} %')
model.train()
