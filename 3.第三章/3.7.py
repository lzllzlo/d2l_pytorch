# 3.7 softmax回归的简洁实现
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# print(train_iter, len(train_iter))
# print(train_iter, len(test_iter))

# 3.7.1 初始化模型参数
# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层(flatten)，来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        # 以均值0和标准差0.01随机初始化权重


net.apply(init_weights)

# 3.7.2 重新审视Softmax的实现
# 在继续softmax计算之前，先从所有ok 中减去max(ok )。
# 这里可以看到每个ok按常数进行的移动不会改变softmax的返回值
loss = nn.CrossEntropyLoss(reduction='none')

# 3.7.3 优化算法
# 使用学习率为0.1的小批量随机梯度下降作为优化算法。
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# 3.7.4 训练
# 接下来我们调用 3.6节中定义的训练函数来训练模型。
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

d2l.plt.show()
