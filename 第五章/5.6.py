# 5.6 GPU
# 使用nvidia-smi命令 来查看显卡信息
# !nvidia-smi

# 5.6.1 计算设备
import torch
from torch import nn

# 在PyTorch中，CPU和GPU可以用torch.device('cpu') 和torch.device('cuda')表示。
# 应该注意的是，cpu设备意味着所有物理CPU和内存，这意味着PyTorch的计算将尝试使用所有CPU核心。
# 然而，gpu设备只代表一个卡和相应的显存。
# 如果有多个GPU，我们使用torch.device(f'cuda:{i}') 来表示第i块GPU(i从0开始)。
# 另外，cuda:0和cuda是等价的。
print(torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1'))

# 查询可用gpu的数量
print(torch.cuda.device_count())


# 定义了两个方便的函数
# 这两个函数允许我们在不存在所需所有GPU的情况下运行代码
def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
        for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


print(try_gpu(), try_gpu(10), try_all_gpus())
print('\n')

# 5.6.2 张量与GPU
# 查询张量所在的设备。默认情况下，张量是在CPU上创建的。
x = torch.tensor([1, 2, 3])
print(x.device)

# 存储在GPU上
# 我们可以在创建张量时指定存储设备。
# 接下来，我们在第一个gpu上 创建张量变量X。在GPU上创建的张量只消耗这个GPU的显存。
# 我们可以使用nvidia-smi命令查看显存使用情况。
# 一般来说，我们需要确保不创建超过GPU显存限制的数据。
X = torch.ones(2, 3, device=try_gpu())
print(X)

# 假设我们至少有两个GPU，
# 下面的代码将在第二个GPU上创建一个随机张量。
Y = torch.rand(2, 3, device=try_gpu(1))
print(Y)

# 复制

# Z = X.cuda(1)
# print(X)
# print(Z)
# print(Y+Z)

# print(Z.cuda(1) is Z)

# 5.6.3 神经网络与GPU
print('5.6.3 神经网络与GPU')
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
print(net(X))
print(net[0].weight.data.device)










