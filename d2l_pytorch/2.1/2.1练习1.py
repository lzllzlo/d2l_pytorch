import torch

x = torch.arange(12,dtype=torch.float32).reshape((3, 4))
y = torch.tensor([[2.0, 1, 4, 3],[1, 2, 3, 4],[4, 3, 2, 1]])
print(torch.cat((x, y), dim=0))
print(torch.cat((x, y),dim=1))

print('x=', x)
print('y=', y)
print(x == y)
print(x > y)
print(x < y)
