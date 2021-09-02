# End-To-End Lane Following

This package was an attempt to solve the issue of lane following using just a convolutional Neural Network

The algorithms used in this implementation are based on this paper published by Nvidia
  - [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)

## Methods

- Images were taken using a hard-coded lane following algorithm, then tied to a steering angle for training, validation, and testing
- Multiple approaches were tested for training, including different network stuctures, a vast array of hyper parameters, and k-fold corss validation
### Using unprocessed image in the Nvidia Network described in the paper
  - This method was the easiest to implement but it did not show good preformance
  - Nvidia used over 200,000 hours of recorded video with steering angles and we had maybe 5 minutes of useable training data
  - This meant that we needed to try other things
### Training using Optuna Hyperparamater Tuning
  - This method we slightly more dificult to use, especially with the implementation of the Optuna API
  - This was run on the Crane Servers because of the long runtime on a computer without cuda capabilities
  - This ended up finding great hyperparamaters for training, which improved accuracy by around 10%, but this still only brought it up to around 70%
### K-Fold Cross Validation
  - This method took multiple different splits of training to validation sets
![Kfold Pic](https://user-images.githubusercontent.com/69916271/131913673-7aaff8b1-4a2d-42ec-bcb1-54fe80ed7763.png)
  - This method also proved to work better than both the originial method when implementing the hyperparamaters from optuna
  - However, still only achived ~75% accuracy

## Conclusions

- Our goal is to build a safe and relaiable system to research the verrification of Neural Networks in Autonomus Vehicles.
- A ~75% accuracy will not accomplish this goal
- After doing more research on how companies such as tesla and waymo control their cars we decided to move into creating a Semantic Segemtnation network for our cars
- All of the code for this is housed In my Semantic Segemtnation Respository
