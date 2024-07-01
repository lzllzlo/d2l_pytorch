# 6.2 图像卷积
import torch
from torch import nn
from d2l import torch as d2l


# 6.2.1 互相关运算
def corr2d(X, K):
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print('X=', X, '\nK=', K, '\nY=', corr2d(X, K))


# 6.2.2 卷积层
class Conv2D(nn.Module):  # 卷积层
    def __init__(self, kernel_size):
        super().__init__()
        # 随机初始化卷积核权重
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))
        # 将weight和bias声明为两个模型参数。
        # 前向传播函数调用corr2d函数并添加偏置。

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


# 6.2.3 图像中目标的边缘检测

# 构造
# 一个6 × 8像素的黑白图像。
# 间四列为黑色(0)，其余像素为白色(1)。
X = torch.ones((6, 8))
X[:, 2:6] = 0
print('X=', X)
# 构造一个高度为1、宽度为2的卷积核K。
# 当进行互相关运算时，
# 如果水平相邻的两元素相同，则输出为零，否则输出为非零。
K = torch.tensor([[1.0, -1.0]])
print('K=', K)
# 对参数X(输入)和K(卷积核)执行互相关运算。
# 如下所示，
# 输出Y中的1代表从白色到黑色的边缘，
# ‐1代表从黑色到白色的边缘，
# 其他情况的输出为0。
Y = corr2d(X, K)

print('Y=', Y)
# 将输入的二维图像转置，再进行如上的互相关运算。
# 其输出如下，之前检测到的垂直边缘消失了。
# 不出所料，这个卷积核K只可以检测垂直边缘，无法检测水平边缘。
print('corr2d(X.t(), K)=', corr2d(X.t(), K))
print('\n')


# 6.2.4 学习卷积核
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))
print('X=', X)
# X是原始图像
Y = Y.reshape((1, 1, 6, 7))
print('Y=', Y)
# Y是结果图像
lr = 3e-2  # 学习率

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')

print(conv2d.weight.data.reshape((1, 2)))




