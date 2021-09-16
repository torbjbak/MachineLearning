import torch
import torch.nn as nn
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
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 1024),
            nn.Flatten(),
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
lr = 0.0001
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
Accuracy epoch #1 = 0.943100
Accuracy epoch #2 = 0.966500
Accuracy epoch #3 = 0.975500
Accuracy epoch #4 = 0.980200
Accuracy epoch #5 = 0.983300
Accuracy epoch #6 = 0.984000
Accuracy epoch #7 = 0.984700
Accuracy epoch #8 = 0.985000
Accuracy epoch #9 = 0.984900
Accuracy epoch #10 = 0.985500
Calculation time for 10 epochs: 5.274205 minutes 
"""