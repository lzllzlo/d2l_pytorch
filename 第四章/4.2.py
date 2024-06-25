# 4.2 多层感知机的从零开始实现
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 4.2.1 初始化模型参数
# num_inputs, num_outputs, num_hiddens = 784, 10, 256
# print(num_inputs, num_outputs, num_hiddens)

# W1 = nn.Parameter(torch.randn(
#     num_inputs, num_hiddens, requires_grad=True) * 0.01)  # 从输入层到隐藏层
# b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
#
# W2 = nn.Parameter(torch.randn(
#     num_hiddens, num_outputs, requires_grad=True) * 0.01)  # 从隐藏层到输出层
# b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
# params = [W1, b1, W2, b2]

# 初始化模型参数
num_inputs, num_outputs, num_hiddens, num_hiddens2 = 784, 10, 256, 64  # 输入784 输出10 一个隐藏层包含256个隐藏单元
# 增加隐藏层数 模型变差 Loss变大
W1 = nn.Parameter(torch.randn(  # 输入层权重
    num_inputs, num_hiddens, requires_grad=True) * 0.01)  # 输入 隐藏层 梯度下降  考虑为什么要随机？ 试试全为0或1
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))  # 偏差 设成0 偏差和输出宽度保持一致
W2 = nn.Parameter(  # 第二层
    torch.randn(num_hiddens, num_hiddens2, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_hiddens2, requires_grad=True))  # 偏差是输出
W3 = nn.Parameter(  # 输出层
    torch.randn(num_hiddens2, num_outputs, requires_grad=True) * 0.01)
b3 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))  # 偏差是输出

params = [W1, b1, W2, b2, W3, b3]  # W1 b1 第一层 W2 b2第二层


# print(params)


# 4.2.2 激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


# 4.2.3 模型
# 使用reshape将每个二维图像转换为一个长度为num_inputs的向量
# def net(X):
#     X = X.reshape((-1, num_inputs))
#     # -1的意思是，后面用一个值替换-1，
#     # 这个值等于(batch_size * height * width * channels)/num_inputs
#     # 就是batch_size
#     H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
#     return H@W2 + b2
# 模型
def net(X):
    X = X.reshape((-1, num_inputs))  # 先把X拉成二维矩阵 = 28*28 拉成784矩阵

    H1 = relu(X @ W1 + b1)  # 这里“@”代表矩阵乘法
    H2 = relu(H1 @ W2 + b2)  # 添加隐藏层
    return (H2 @ W3 + b3)


"""使用 `X.reshape((-1, num_inputs))` 并不会将输入的图片自动变成 `num_inputs` 大小，
而是会调整数组的形状，使得数据可以进行矩阵乘法运算。
在这里，`num_inputs` 通常指的是输入层的神经元数量，也就是每个输入样本的特征数量。

让我们更详细地看看这个过程：

1. **输入数据**：假设 `X` 是一个包含多个图片的数据集。
每张图片可能有不同的尺寸，比如 `(batch_size, height, width, channels)`。
其中，`batch_size` 是图片的数量，`height` 和 `width` 是图片的高和宽，`channels` 是通道数（例如RGB图像有3个通道）。

2. **展平图片**：在进行神经网络前向传播之前，图片通常需要展平（flatten）成一维数组。
如果每张图片的形状是 `(height, width, channels)`，展平后它将变成一维数组，长度为 `height * width * channels`。

3. **重塑数据**：`reshape((-1, num_inputs))` 的作用是在保持总元素数量不变的情况下重塑数据的形状。
`-1` 表示自动计算新形状的第一维度大小。假设每张展平后的图片有 `num_inputs` 个特征（即 `num_inputs = height * width * channels`），
那么重塑后的形状将是 `(batch_size, num_inputs)`。

具体地说，如果输入图片的原始形状是 `(batch_size, height, width, channels)`，
通过展平操作，形状变成 `(batch_size, height * width * channels)`，
这时 `num_inputs` 就等于 `height * width * channels`。

举个例子：

- 原始输入数据 `X` 的形状是 `(32, 28, 28, 3)`，表示有32张28x28的RGB图片。
- 展平操作后，每张图片的大小是 `28 * 28 * 3 = 2352`，所以展平后的数据形状是 `(32, 2352)`.
- 使用 `X.reshape((-1, 2352))`，这里的 `-1` 自动计算出第一维度，结果保持 `(32, 2352)` 形状。

因此，`reshape((-1, num_inputs))` 并不会改变图片的大小，
而是将输入数据展平成适合矩阵乘法运算的形状，确保每个输入样本具有 `num_inputs` 个特征。
输入图片的大小（即每张图片的高、宽和通道数）需要预先固定，才能确定 `num_inputs` 的值。
如果输入图片的尺寸不同，通常需要在预处理步骤中将其调整到相同的大小。"""

# 4.2.4 损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 4.2.5 训练
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
d2l.plt.show()

d2l.predict_ch3(net, test_iter)
d2l.plt.show()



