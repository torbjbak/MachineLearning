import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import torch

matplotlib.rcParams.update({'font.size': 11})

W_init = np.array([[-0.2], [0.53]]).reshape(-1, 1)
b_init = np.array([[3.1]])


class LinearRegressionModel:
    def __init__(self, W=W_init.copy(), b=b_init.copy()):
        self.W = torch.tensor(W, requires_grad=True)
        self.b = torch.tensor(b, requires_grad=True)

    # predictor
    def f(self, x):
        return x @ self.W + self.b

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


model = LinearRegressionModel()

# observed/training input and output
data = np.array(pd.read_csv("day_length_weight.csv", names=['day', 'length', 'weight']))

x_in = data[1:, 1:].astype(float)
y_in = data[1:, :1].astype(float)

x_train = torch.tensor(x_in[:800]).reshape(-1, 2)
y_train = torch.tensor(y_in[:800]).reshape(-1, 1)

x_test = x_in[800:]
y_test = y_in[800:]

lr = 0.00011
epochs = 5000

optimizer = torch.optim.SGD([model.W, model.b], lr)
for epoch in range(epochs):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step
    
print("W = %s\nb = %f\nloss =%f" % (
    model.W.data.numpy(), 
    model.b.item(), 
    model.loss(x_train, y_train).item()
    ))

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.set_xlabel('length [cm]')
ax.set_ylabel('weight [kg]')
ax.set_zlabel('age [days]')

x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])

ax.plot3D(np.linspace(0, 1800, 1800), model.f(x))

ax.scatter3D(x_test[:, 0], x_test[:, 1], y_test[:, 0])

plt.show()

