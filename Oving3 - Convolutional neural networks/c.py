import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout2d
import torchvision
from time import time


mnist_train = torchvision.datasets.MNIST('./data', train=True)
x_train = torchvision.datasets.MNIST('./data', train=True, download=True).data.reshape(-1, 1, 28, 28).float()
y_train = torch.zeros((mnist_train.targets.shape[0], 10))
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1

mnist_test = torchvision.datasets.MNIST('./data', train=False)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float()
y_test = torch.zeros((mnist_test.targets.shape[0], 10))
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1

# Normalization of inputs
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

batches = 600
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)

class SequentialModel(nn.Module):
    def __init__(self):
        super(SequentialModel, self).__init__()

        self.logits = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2), 
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Dropout2d(),

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1 * 1024, 10)
        )

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


model = SequentialModel()
epochs = 10
lr = 0.0004
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
Learning rate: 0.000100
Model:
SequentialModel(
  (logits): Sequential(
    (0): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Flatten(start_dim=1, end_dim=-1)
    (5): Linear(in_features=3136, out_features=1024, bias=True)
    (6): ReLU()
    (7): Linear(in_features=1024, out_features=10, bias=True)
  )
)

Accuracy epoch #1 = 0.938600
Accuracy epoch #2 = 0.963600
Accuracy epoch #3 = 0.974800
Accuracy epoch #4 = 0.979800
Accuracy epoch #5 = 0.982900
Accuracy epoch #6 = 0.984400
Accuracy epoch #7 = 0.985700
Accuracy epoch #8 = 0.986400
Accuracy epoch #9 = 0.986800
Accuracy epoch #10 = 0.987100
Calculation time for 10 epochs: 5.057017 minutes
"""