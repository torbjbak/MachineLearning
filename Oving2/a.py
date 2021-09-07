import torch as tr
import numpy as np
from random import randrange as rr

DATA_SET_SIZE = 1000

inputData = [] 
outputData = []

for i in range(DATA_SET_SIZE):
    x = rr(2)
    y = rr(2)
    inputData.append([x, y])

print(inputData[:10])