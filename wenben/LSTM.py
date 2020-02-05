import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

lstm=nn.LSTM(3,3)
inputs=[torch.randn(1,3) for _ in range(5)]
print(inputs)

hidden = (torch.randn(1,1,3),
          torch.randn(1,1,3))
for i in inputs:
    out,hidden=lstm(i.view(1,1,-1),hidden)
print(out)
print(hidden)
inputs=torch.cat(inputs).view(len(inputs),1,-1)
hidden=(torch.randn(1,1,3),torch.randn(1,1,3))
out,hidden=lstm(inputs,hidden)
print(out)
print(hidden)