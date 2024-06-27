# 5.2 参数管理

import torch
from torch import nn
net = nn.Sequential(nn.Linear(4, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1))
X = torch.rand(size=(2, 4))

print(net(X))

# 5.2.1 参数访问
# 检查第二个全连接层的参数
print(net[2].state_dict())

# 目标参数
# 从第二个全连接层(即第三个神经网络层)提取偏置，
# 提取后返回的是一个参数类实例，并进一步访问该参数的值
print('\n')
print(type(net[2].bias))
print('\n')
print(net[2].bias)
print('\n')
print(net[2].bias.data)
print('\n')
print(net[2].weight.grad == None)
print('\n')

# 一次性访问所有参数
# 通过演示来比较访问第一个全连接层的参数和访问所有层。
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
print(net.state_dict()['2.bias'].data)
print('\n')


# 从嵌套块收集参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net


rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet(X))
print(rgnet)

# 访问第一个主要的块中、第二个子块的第一层的偏置项。
print(rgnet[0][1][0].bias.data)
print('\n')

# 5.2.2 参数初始化
# 内置初始化
# 将所有权重参数初始化为标准差为0.01的高斯随机变量，
# 且将偏置参数设置为0。
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


net.apply(init_normal)
print(net[0].weight.data[0], net[0].bias.data[0])


# 将所有参数初始化为给定的常数，比如初始化为1。
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


net.apply(init_constant)
print(net[0].weight.data[0], net[0].bias.data[0])
print('\n')


# 对某些块应用不同的初始化方法。
# 例如，下面我们使用Xavier初始化方法初始化第一个神经网络层，
# 然后将第三个神经网络层初始化为常量值42。
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


net[0].apply(init_xavier)
net[2].apply(init_42)

print(net[0].weight.data[0])
print(net[2].weight.data)
print('\n')


# 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print('Init', *[(name, param.shape)
                        for name, param
                        in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5


net.apply(my_init)
print(net[0].weight[:2])


#  5.2.3 参数绑定
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8),
                    nn.ReLU(),
                    shared,
                    nn.ReLU(),
                    shared,
                    nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
