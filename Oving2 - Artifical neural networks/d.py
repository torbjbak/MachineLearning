import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset = datasets.MNIST('./data', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST('./data', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next() 

# Model creation with neural net Sequential model
model=nn.Sequential(nn.Linear(784,128),     # 1 layer:- 784 input 128 o/p
                    nn.ReLU(),              # Defining Regular linear unit as activation
                    nn.Linear(128,64),      # 2 Layer:- 128 Input and 64 O/p
                    nn.Tanh(),              # Defining Regular linear unit as activation
                    nn.Linear(64,10),       # 3 Layer:- 64 Input and 10 O/P as (0-9)
                    nn.LogSoftmax(dim=1)    # Defining the log softmax to find the probablities for the last output unit
                  ) 

criterion = nn.NLLLoss() 

images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images)
loss = criterion(logps, labels)
loss.backward()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

images, labels = next(iter(trainloader))
images.resize_(64, 784)

optimizer.zero_grad()

output = model(images)
loss = criterion(output, labels)
loss.backward()

time0 = time()
epochs = 15
running_loss_list= []
epochs_list = []

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)   # Flatenning MNIST images with size [64,784]
        optimizer.zero_grad()                       # defining gradient in each epoch as 0
        output = model(images)                      # modeling for each image batch
        loss = criterion(output, labels)            # calculating the loss
        loss.backward()                             # This is where the model learns by backpropagating
        optimizer.step()                            # And optimizes its weights here
        running_loss += loss.item()                 # calculating the loss
        
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))

print("\nTraining Time (in minutes) =",(time()-time0)/60)

torch.save(model, './my_mnist_model.pt')

model = torch.load('./my_mnist_model.pt')


W = (model[4].weight @ model[2].weight @ model[0].weight)
for i in range(10):
    plt.title(i)
    plt.imshow(W.detach().numpy()[i, :].reshape(28, 28))
    plt.show()

def classify(img, ps):
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    

images, labels = next(iter(testloader))

correct_count, all_count = 0, 0
for images,labels in testloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)

    with torch.no_grad():
        logps = model(img)

    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count)) 