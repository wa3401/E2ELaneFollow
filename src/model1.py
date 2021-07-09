'''
-@author Will Anderson
-
-This file defines a neural network which was designed originally by nvidia that is designed for deep learning for lane following in an autonomous car
-'''

import torch
import torch.nn as nn
class NvidiaModel(nn.Module):
    def __init__(self):
        super(NvidiaModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 5, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=(5, 4), stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(6, 16,kernel_size=(3, 5), stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 10, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Flatten()
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=1540, out_features=100),
            #nn.Linear(in_features=1188, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, input):
        #output = self.conv1(input)
        #print(f'Conv1 Output Shape: {output.shape}')
        #output = self.conv2(output)
        #print(f'Conv2 Output Shape: {output.shape}')
        output = self.conv3(input)
        #print(f'Conv3 Output Shape: {output.shape}')
        output = self.conv4(output)
        #print(f'Conv4 Output Shape: {output.shape}')
        output = self.conv5(output)
        #print(f'Conv5 Output Shape: {output.shape}')
        output = self.linear_layers(output)
        #print(f'Forward Pass Output Shape: {output.shape}')
        return output