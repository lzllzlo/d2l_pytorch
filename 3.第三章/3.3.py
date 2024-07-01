# 3.3 线性回归的简洁实现

import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
# nn是神经网络的缩写
from torch import nn


# 3.3.1 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# 3.3.2 读取数据集
def load_array(data_arrays, batch_size, is_train=True):   #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)
print(next(iter(data_iter)))


# 3.3.3 定义模型

net = nn.Sequential(nn.Linear(2, 1))
# 定义网络结构
# 我们将两个参数传递到nn.Linear中。
# 第一个指定输入特征形状，即2，
# 第二个指定输出特征形状，输出特征形状为单个标量，因此为1。

# 3.3.4 初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
# 通过net[0]选择网络中的第一个图层
# 指定每个权重参数应该从均值为0、标准差为0.01的正态分布中随机采样，
# 偏置参数将初始化为零。
print(net[0].weight.data)
print(net[0].bias.data.fill_(0))

# 3.3.5 定义损失函数
loss = nn.MSELoss()
# 计算均方误差使用的是MSELoss类，也称为平方L2范数。
# 默认情况下，它返回所有样本损失的平均值。

# 3.3.6 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
# 小批量随机梯度下降算法是一种优化神经网络的标准工具，PyTorch在optim模块中实现了该算法的许多变种。
# 当我们实例化一个SGD实例时，我们要指定优化的参数(可通过net.parameters()从我们的模型中获得)
# 以及优化算法所需的
# 超参数（在开始学习过程之前设置值的参数，而不是通过训练得到的参数数据。
# 包括：学习率、批量大小、训练轮数、激活函数、网络架构/隐藏层和单元数、
# Dropout、权重初始化、优化器、正则化等）字典。
# 小批量随机梯度下降只需要设置学习率lr值，这里设置为0.03。

# 3.3.7 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)  # 通过调用net(X)生成预测并计算损失l(前向传播)。
        trainer.zero_grad()  # trainer是选用的的优化算法
        # 清空模型参数梯度的函数，它将模型参数的梯度缓存设置为0。
        # 在进行反向传播时，梯度会累加，如果不清空梯度，会影响后续的梯度计算。
        l.backward()  # 通过进行反向传播来计算梯度。
        trainer.step()  # 通过调用优化器来更新模型参数。
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')


w = net[0].weight.data
print('真实的w：', true_w, '算出的w：', w)
print('w的估计误差:', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('真实的b：', true_b, '算出的b：', b)
print('b的估计误差:', true_b - b)


