import torch

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x ** y)

print(torch.exp(x))

x = torch.arange(12,dtype=torch.float32).reshape((3, 4))
y = torch.tensor([[2.0, 1, 4, 3],[1, 2, 3, 4],[4, 3, 2, 1]])
print(torch.cat((x, y), dim=0))
print(torch.cat((x, y),dim=1))

print(x == y)
print(x.sum())