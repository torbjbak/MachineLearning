import torch as tr
import numpy as np
import matplotlib.pyplot as plt


W_init = np.array([[8.5], [8.5]]).astype(float).reshape(-1, 1)
b_init = np.array([[-4.0]]).astype(float)
lr = 0.0001
epochs = 10000

class SigmoidModel:
    def __init__(self, W=W_init.copy(), b=b_init.copy()):
        self.W = tr.tensor(W, requires_grad=True)
        self.b = tr.tensor(b, requires_grad=True)

    def f(self, x):
        return tr.sigmoid((-x + 1) @ self.W + self.b)

    def loss(self, x, y):
        return -tr.mean(
            tr.multiply(y, tr.log(self.f(x))) + 
            tr.multiply((1 - y), tr.log(1 - self.f(x)))
        )

model = SigmoidModel()

in_train = tr.tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).astype(float))
out_train = tr.tensor(np.array([[1], [1], [1], [0]]).astype(float))

opt = tr.optim.Adam([model.W, model.b], lr)
for epoch in range(epochs):
    model.loss(in_train, out_train).backward()
    opt.step()
    opt.zero_grad()

print("Learning rate = %f\nEpochs = %i\nW =\n%s\nb = %f\nLoss = %f" % (
    lr, 
    epochs, 
    model.W.data.numpy(), 
    model.b.item(),
    model.loss(in_train, out_train).item()
))

# Plotting model based on training data
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.set_xlabel('Input 1')
ax.set_ylabel('Input 2')
ax.set_zlabel('Output')



x = np.linspace(-0.25, 1.25, 50, dtype=float).reshape(-1, 1)
y = np.linspace(-0.25, 1.25, 50, dtype=float).reshape(-1, 1)

X, Y = np.meshgrid(x.flatten(), y.flatten())

Z = np.empty((50, 50))
for i in range(50):
    for j in range(50):
        Z[i, j] = model.f(tr.tensor([[X[i, j], Y[i, j]]])).detach().numpy()

ax.plot_wireframe(X, Y, Z, color="orange")

ax.scatter3D(in_train[:, 0], in_train[:, 1], out_train[:, 0])

plt.show()