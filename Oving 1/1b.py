import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch

matplotlib.rcParams.update({'font.size': 11})

W_init = np.array([[0.0], [0.0]]).reshape(-1, 1)
b_init = np.array([[0.0]])


class LinearRegressionModel:
    def __init__(self, W=W_init.copy(), b=b_init.copy()):
        self.W = torch.tensor(W, requires_grad=True)
        self.b = torch.tensor(b, requires_grad=True)

    # predictor
    def f(self, input):
        return input @ self.W + self.b

    # Uses Mean Squared Error
    def loss(self, input, output):
        return torch.mean(torch.square(self.f(input) - output))


model = LinearRegressionModel()

# observed/training input and output
data = np.array(pd.read_csv("day_length_weight.csv", names=['day', 'length', 'weight']))

input_in = data[1:, 1:].astype(float)
output_in = data[1:, :1].astype(float)

input_train = torch.tensor(input_in[:800]).reshape(-1, 2)
output_train = torch.tensor(output_in[:800]).reshape(-1, 1)

input_test = input_in[800:]
output_test = output_in[800:]

lr = 0.00011
epochs = 10000

optimizer = torch.optim.SGD([model.W, model.b], lr)
for epoch in range(epochs):
    model.loss(input_train, output_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step
    
print("Learnig rate = %f\nEpochs = %i\nW =\n%s\nb = %f\nloss = %f" % (
    lr,
    epochs,
    model.W.data.numpy(), 
    model.b.item(), 
    model.loss(input_train, output_train).item()
    ))

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.set_xlabel('length [cm]')
ax.set_ylabel('weight [kg]')
ax.set_zlabel('age [days]')

min_x = torch.min(input_train[:, 0])
max_x = torch.max(input_train[:, 0])
min_y = torch.min(input_train[:, 1])
max_y = torch.max(input_train[:, 1])

x = torch.tensor([min_x, max_x]).reshape(-1, 1)
y = torch.tensor([min_y, max_y]).reshape(-1, 1)

xy = torch.cat((x, y), 1)

X, Y = np.meshgrid(x, y)
Z = model.f(xy).detach().numpy()

ax.plot_surface(X, Y, Z, color="yellow")

ax.scatter3D(input_test[:, 0], input_test[:, 1], output_test[:, 0])

plt.show()

