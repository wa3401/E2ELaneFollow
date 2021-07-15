import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model1 import NvidiaModel
from driveData1 import DriveData
import optuna
from optuna.trial import TrialState
from driveEdgeData import DriveEdgeData
from ryanModel import RyanModel

# If a cuda device is avaiable, set that as device
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

# Define Transforms
transform1 = transforms.Compose(
    [transforms.ToTensor()]
)

transform2 = transforms.Compose(
    [transforms.ToTensor()]
)


def objective(trial):
    model = RyanModel().to(DEVICE)

    num_epochs = trial.suggest_int("num_epochs", 100, 350)
    batch_size = trial.suggest_int("batch_size", 8, 128)
    
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = Adam(model.parameters(), lr=lr)

    criterion = nn.MSELoss()


    # Define Train Dataset
    train_dir = '/home/williamanderson/procDriveImages/trainImages/'
    train_csv = train_dir + 'train.csv'
    train_set = DriveData(annotations_file=train_csv,
                        img_dir=train_dir, transform=transform1)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Define test dataset
    test_dir = '/home/williamanderson/procDriveImages/testImages/'
    test_csv = test_dir + 'test.csv'
    test_set = DriveData(annotations_file=test_csv,
                        img_dir=test_dir, transform=transform1)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # Define validation dataset
    val_dir = '/home/williamanderson/procDriveImages/valImages/'
    val_csv = val_dir + 'val.csv'
    val_set = DriveEdgeData(annotations_file=val_csv,
                        img_dir=val_dir, transform=transform1)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    #Training Loop
    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):

            #Send tensors to device and shape properly
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
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
            
        #scheduler.step()
        #train_loss_graph.append(loss.item())

        valid_loss = 0.0
        model.eval()
        for i, (images, labels) in enumerate(val_dataloader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            labels = labels.view(-1, 1)
            #print(f'Val image size: {images.size()}')
            #print(f'Val label size: {labels.size()}')

            outputs = model(images)

            loss = criterion(outputs, labels)

            valid_loss = loss.item() * images.size(0)
            
            #writer.add_scalar('Validation Loss', valid_loss, global_step=step)
            #step += 1
        #val_loss_graph.append(valid_loss)
        #val_idx_graph.append(i * 7.7)

        with torch.no_grad():
            model.eval()
            n_correct = 0
            n_samples = 0

            for images, labels in test_dataloader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)

                for i in range(len(outputs)):
                    n_samples += 1
                    label = labels[i]
                    #print(f'Test label: {label}')
                    pred = outputs[i]
                    #print(f'Test Predicted: {pred}')
                    if abs(label - pred) < 0.01:
                        n_correct += 1

            #print(f'Num Samples: {n_samples}')
            #print(f'Num Correct: {n_correct}')
            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy for the network: {acc:.4f} %')
        

        

        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_loss / len(val_dataloader)}')
        trial.report(acc, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return acc

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
