import torch
import torch.nn as nn
import torchvision
from time import time


fashion_train = torchvision.datasets.FashionMNIST('./data', train=True, download=True)
x_train = fashion_train.data.reshape(-1, 1, 28, 28).float()
y_train = torch.zeros((fashion_train.targets.shape[0], 10))
y_train[torch.arange(fashion_train.targets.shape[0]), fashion_train.targets] = 1

fashion_test = torchvision.datasets.FashionMNIST('./data', train=False, download=True)
x_test = fashion_test.data.reshape(-1, 1, 28, 28).float()
y_test = torch.zeros((fashion_test.targets.shape[0], 10))
y_test[torch.arange(fashion_test.targets.shape[0]), fashion_test.targets] = 1

# Normalization of inputs
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

batches = 600
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)

class FashionModel(nn.Module):
    def __init__(self):
        super(FashionModel, self).__init__()

        self.logits = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 10)
        )

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


model = FashionModel()
epochs = 10
lr = 0.0005
optimizer = torch.optim.Adam(model.parameters(), lr)
time0 = time()
print("Learning rate: %f" % lr)
print("Model: %s" % model.logits)

for epoch in range(epochs):
    for batch in range(len(x_train_batches)):
        model.loss(x_train_batches[batch], y_train_batches[batch]).backward()
        optimizer.step()
        optimizer.zero_grad()

    print("Accuracy epoch #%i = %f" % (epoch + 1, model.accuracy(x_test, y_test).item()))

time1 = time()
print("Calculation time for %i epochs: %f minutes" % (epochs, (time1 - time0) / 60))

""" 
Learning rate: 0.000500
Model: Sequential(
  (0): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU()
  (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (4): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (6): ReLU()
  (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (8): Flatten(start_dim=1, end_dim=-1)
  (9): Linear(in_features=3136, out_features=10, bias=True)
)
Accuracy epoch #1 = 0.866800
Accuracy epoch #2 = 0.882500
Accuracy epoch #3 = 0.891800
Accuracy epoch #4 = 0.897300
Accuracy epoch #5 = 0.900300
Accuracy epoch #6 = 0.904000
Accuracy epoch #7 = 0.904600
Accuracy epoch #8 = 0.905600
Accuracy epoch #9 = 0.904800
Accuracy epoch #10 = 0.905700
Calculation time for 10 epochs: 7.024704 minutes
"""