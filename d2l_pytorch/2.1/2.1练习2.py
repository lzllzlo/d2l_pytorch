import torch

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print('a=', a)
print('b=', b)

print(a + b)

a = torch.tensor([1, 3, 5, 7, 9])
b = torch.tensor([[2], [4], [6], [8], [10]])
print('a=', a)
print('b=', b)

print(a + b)