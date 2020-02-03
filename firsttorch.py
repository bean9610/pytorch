import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

x=torch.tensor(1.,requires_grad=True)
w=torch.tensor(2.,requires_grad=True)
b=torch.tensor(3.,requires_grad=True)

y=w*x+b
y.backward()

print(x.grad)
print(b.grad)
print(w.grad)

x=torch.randn(10,3)
y=torch.randn(10,2)

print(x)

linear=nn.Linear(3,2)
print("w:",linear.weight)
print("b:",linear.bias)
