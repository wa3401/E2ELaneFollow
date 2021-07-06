import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision
from torchvision import datasets
from torchvision.io import read_image
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import fnmatch
import os
import cv2 as cv
import matplotlib.pyplot as plt
#from torch.utils.tensorboard import SummaryWriter

device = ('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparamaters
num_epochs = 50
batch_size = 32
learning_rate = 0.001

def getLabels(data_dir):
    file_list = os.listdir(data_dir)
    # print(file_list)
    image_paths = []
    steering_angles = []
    pattern = "*.jpg"
    for filename in file_list:
        #print(filename)
        if fnmatch.fnmatch(filename, pattern):
            image_paths.append(os.path.join(data_dir, filename))
            if(filename[-13] == '-'):
                angle = float(filename[-13:-4])
            else:
                angle = float(filename[-12:-4])
            steering_angles.append(angle)

    return steering_angles

class CustomImageDataset(Dataset):
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
        image = cv.imread(img_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2YUV)
        image = cv.GaussianBlur(image, (3,3), 0)
        image = cv.resize(image, (200, 66))
        label = torch.tensor(self.img_labels.iloc[idx, 2], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        
        return image, label

mean = torch.Tensor([0.4312, 0.4975, 0.5023])
std = torch.Tensor([0.3043, 0.0235, 0.0123])

transform1 = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

transform2 = transforms.Compose(
    [transforms.ToTensor()]
)

train_dir = '/home/williamanderson/driveImages/trainImages/'
train_csv = train_dir + 'train.csv'
train_set = CustomImageDataset(annotations_file=train_csv, img_dir=train_dir, transform=transform1)
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_dir = '/home/williamanderson/driveImages/testImages/'
test_csv = test_dir + 'test.csv'
test_set = CustomImageDataset(annotations_file=test_csv, img_dir=test_dir, transform=transform1)
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

val_dir = '/home/williamanderson/driveImages/valImages/'
val_csv = val_dir + 'val.csv'
val_set = CustomImageDataset(annotations_file=val_csv, img_dir=val_dir, transform=transform1)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)





images, labels = next(iter(train_dataloader))
print(f"Train Feature batch shape: {images.size()}")
print(f"Train Labels batch shape: {labels.size()}")
images, labels = next(iter(val_dataloader))
print(f"Val Feature batch shape: {images.size()}")
print(f"Val Labels batch shape: {labels.size()}")

# img = train_features[0]
# img_sq = torch.squeeze(img)
# label = train_labels[0]



# plt.imshow(img.permute(1, 2, 0))
# plt.show()
# print(f"Label: {label}")

class NvidiaModel(nn.Module):
    def __init__(self):
        super(NvidiaModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            #nn.AvgPool2d(3, stride=2),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            #nn.AvgPool2d(3, stride=2),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            #nn.AvgPool2d(3, stride=2),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            #nn.AvgPool2d(3, stride=2),
            nn.Conv2d(64, 64, 3),
            nn.Flatten(),
            nn.Dropout(0.5)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=1152, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, input):
        output = self.conv_layers(input)
        output = self.linear_layers(output)
        #print(f'Forward Pass Output Shape: {output.shape}')
        return output

model = NvidiaModel().to(device)
#model.load_state_dict(torch.load('/home/williamanderson/LaneFollower/savedModel.ipynb'))



criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
#writer = SummaryWriter(f'runs/LaneFollow/data')
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1, verbose=True)

n_total_steps = len(train_dataloader)
min_valid_loss = np.inf
step = 0
saved_epoch = 0
for epoch in range(num_epochs):
    train_loss = 0.0
    model.train()
    for i, (images, labels) in enumerate(train_dataloader):

        images = images.to(device)
        labels = labels.to(device)
        labels = labels.view(-1, 1)
        #print(f'Training Label Shape: {labels.shape}')

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()
        train_loss = loss.item() * images.size(0)
        #writer.add_scalar('Training Loss', train_loss, global_step=step)

        # _, predictions = outputs.max(1)
        # num_correct = (abs(predictions - outputs) < 0.01).sum()
        # running_train_acc = float(num_correct)/float(images.shape[0])

        # writer.add_scalar('Training Accuracy', running_train_acc, global_step=step)
        

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

    

    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_loss / len(val_dataloader)}')
      
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        torch.save(model.state_dict(), '/home/williamanderson/LaneFollower/savedModel.ipynb')
        min_valid_loss = valid_loss
        saved_epoch = epoch

print(f'Finished Training, Saved Epoch {saved_epoch}')


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
