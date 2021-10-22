import torch as tr
import numpy as np
import matplotlib.pyplot as plt

W_init = np.array([[8.5]])
b_init = np.array([[-4.0]])
lr = 0.0001
epochs = 10000

class SigmoidModel:
    def __init__(self, W=W_init.copy(), b=b_init.copy()):
        self.W = tr.tensor(W, requires_grad=True)
        self.b = tr.tensor(b, requires_grad=True)

    def f(self, x):
        return -tr.sigmoid(x @ self.W + self.b) + 1

    def loss(self, x, y):
        return -tr.mean(
            tr.multiply(y, tr.log(self.f(x))) + 
            tr.multiply((1 - y), tr.log(1 - self.f(x)))
        )

model = SigmoidModel()

in_train = tr.tensor(np.array([[0], [1]]).astype(float))
out_train = tr.tensor(np.array([[1], [0]]).astype(float))

opt = tr.optim.Adam([model.W, model.b], lr)
for epoch in range(epochs):
    model.loss(in_train, out_train).backward()
    opt.step()
    opt.zero_grad()

print("Learning rate = %f\nEpochs = %i\nW = %f\nb = %f\nLoss = %f" % (
    lr, 
    epochs, 
    model.W.item(), 
    model.b.item(),
    model.loss(in_train, out_train).item()
))

# Plotting model based on training data
plt.plot(in_train, out_train, 'o', label='NOT values')
plt.xlabel('Input')
plt.ylabel('Output')

x = tr.tensor(np.linspace(-0.25, 1.25, 1000, dtype=float)).reshape(-1, 1)
plt.plot(x, model.f(x).detach(), label='Model')

plt.legend()
plt.show()