import torch as tr
import numpy as np
import matplotlib.pyplot as plt


DATA_SET_SIZE = 100
W1_init = np.array([[10, -10], [10, -10]]).astype(float)
b1_init = np.array([[-5, 15]]).astype(float)
W2_init = np.array([[10], [10]]).astype(float)
b2_init = np.array([[-15]]).astype(float)

""" W1_init = np.array([[0, 0], [0, 0]]).astype(float)
b1_init = np.array([[0, 0]]).astype(float)
W2_init = np.array([[0], [0]]).astype(float)
b2_init = np.array([[0]]).astype(float) """

lr = 0.0001
epochs = 10000

class SigmoidModel:
    def __init__(self, W1=W1_init.copy(), W2=W2_init.copy(), b1=b1_init.copy(), b2=b2_init.copy()):
        self.W1 = tr.tensor(W1, requires_grad=True)
        self.W2 = tr.tensor(W2, requires_grad=True)
        self.b1 = tr.tensor(b1, requires_grad=True)
        self.b2 = tr.tensor(b2, requires_grad=True)

    def f1(self, x):
        return tr.sigmoid(x @ self.W1 + self.b1)

    def f2(self, h):
        return tr.sigmoid(h @ self.W2 + self.b2)

    def f(self, x):
        return self.f2(self.f1(x))
   
    def loss(self, x, y):
        return -tr.mean(
            tr.multiply(y, tr.log(self.f(x))) + 
            tr.multiply((1 - y), tr.log(1 - self.f(x)))
        )

model = SigmoidModel()

in_train = tr.tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).astype(float))
out_train = tr.tensor(np.array([[0], [1], [1], [0]]).astype(float))

opt = tr.optim.Adam([model.W1, model.W2, model.b1, model.b2], lr)
for epoch in range(epochs):
    model.loss(in_train, out_train).backward()
    opt.step()
    opt.zero_grad()

print("Learning rate = %f\nEpochs = %i\nW1 =\n%s\nW2 =\n%s\nb1 = %s\nb1 = %f\nLoss = %f" % (
    lr, 
    epochs, 
    model.W1.data.numpy(),
    model.W2.data.numpy(),  
    model.b1.data.numpy()[0],
    model.b2.item(),
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