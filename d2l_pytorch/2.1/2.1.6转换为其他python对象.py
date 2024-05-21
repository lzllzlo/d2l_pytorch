import torch
import numpy

x = torch.tensor([1.0])
a = x.numpy()
b = torch.tensor(a)
print(x, a, b)
print(type(a), type(b))

a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))