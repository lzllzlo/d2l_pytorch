import torch

x = torch.zeros(3, 4) + 2
y = torch.ones(3, 4)
print(x)
print(y)

before = id(y)
y = y + x
print('y=', y)
print(id(y) == before)

z = torch.zeros_like(y)
print('z=', z, "\n", 'id(z):', id(z))
z[:] = x + y
print('z=', z, "\n", 'id(z):', id(z))

before = id(x)
x += y
print(id(x) == before)

before = id(x)
x[:] = x + y
print(id(x) == before)