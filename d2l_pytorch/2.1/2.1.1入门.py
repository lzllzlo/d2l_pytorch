import torch

x = torch.arange(12)
print(x)
print(x.shape)
print(x.numel())

x = x.reshape(3, -1)
print(x)

x = torch.zeros((2, 3, 4))
print(x)

x = torch.ones((2, 3, 4))
print(x)

x = torch.randn(3, 4)
print(x)

x = torch.tensor([[0, 1, 5, 9],[9, 8, 9, 4],[5, 8, 9, 1]])
print(x)