# 4.4 模型选择、欠拟合和过拟合
# 4.4.4 多项式回归
import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

# 生成数据集
max_degree = 20  # 多项式的最大阶数
n_train, n_test = 100, 100  # 训练和测试数据集大小
true_w = np.zeros(max_degree)  # 分配大量的空间,生成一个20个0的向量
# print('true_w=', true_w)
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])  # 给第0-3个向量赋值
# print('true_w=', true_w)

features = np.random.normal(size=(n_train + n_test, 1))  # 随机生成一个200行，1列的特征向量
# print(features)
np.random.shuffle(features)  # 打乱特征向量
# print(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))  # nn.power() 幂函数
# 对第所有维的特征取0次方、1次方、2次方...19次方
# print(poly_features)
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!
    # i次方的特征除以(i+1)阶乘
    # labels的维度:(n_train+n_test,)
labels = np.dot(poly_features, true_w)
# 根据多项式生成y，即生成真实的labels
labels += np.random.normal(scale=0.1, size=labels.shape)
# 对真实labels加噪音进去

# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
    torch.float32) for x in [true_w, features, poly_features, labels]]

print('x=', features[:2], '\nx^=', poly_features[:2, :], '\ny=', labels[:2])
# 看一下前两个样本的x、x的所有次方、x对应的y


# 对模型进行训练和测试
def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2)  # 两个数的累加器
    for X, y in data_iter:  # 从迭代器中拿出对应特征和标签
        out = net(X)
        y = y.reshape(out.shape)  # 将真实标签改为网络输出标签的形式，统一形式
        l = loss(out, y)  # 计算网络输出的预测值与真实值之间的损失差值
        metric.add(l.sum(), l.numel())  # 总量除以个数，等于平均
    return metric[0] / metric[1]  # 返回数据集的平均损失


# 定义训练函数
def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss()
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))  # 单层线性回归
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1, 1)), batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log', xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss), evaluate_loss(net, test_iter, loss)))
    print('weight', net[0].weight.data.numpy())  # 训练完后打印，打印最终学到的weight值


# 三阶多项式函数拟合(正常)
# 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train],
      labels[n_train:])  # 最后返回的weight值和公式真实weight值很接近
d2l.plt.show()

# 线性函数拟合(欠拟合)
# 从多项式特征中选择前2个维度，即1和x
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:])
# 一阶多项式函数拟合(欠拟合)
# 这里相当于用一阶多项式拟合真实的三阶多项式，欠拟合了，损失很高，根本就没降

# 高阶多项式函数拟合(过拟合)
# 从多项式特征中选取所有维度
# 十九阶多项式函数拟合(过拟合)
# 这里相当于用十九阶多项式拟合真实的三阶多项式，过拟合了
train(poly_features[:n_train, :], poly_features[n_train:, :],
      labels[:n_train], labels[n_train:], num_epochs=1500)


