# 6.5 汇聚层
# 汇聚(pooling)层,池化层，它具有双重目的:
# 降低卷积层对位置的敏感性，
# 同时降低对空间降采样表示的敏感性。

# 6.5.1 最大汇聚层和平均汇聚层
# 不同于卷积层中的输入与卷积核之间的互相关计算，汇聚层不包含参数。
# 池运算是确定性的，我们通常计算汇聚窗口中所有元素的最大值或平均值。
# 这些操作分别称为最大汇聚层(maximum pooling)和平均汇聚层(average pooling)。
# 与互相关运算符一样，汇聚窗口从输入张量的左上角开始，
# 从左往右、从上往下的在输入张量内滑动。
# 在汇聚窗口到达的每个位置，它计算该窗口中输入子张量的最大值或平均值。
# 计算最大值或平均值是取决于使用了最大汇聚层还是平均汇聚层。

# 在下面的代码中的pool2d函数，我们实现汇聚层的前向传播。
# 这类似于6.2节中的corr2d函数。
# 然而，这里我们没有卷积核，输出为输入中每个区域的最大值或平均值。
import torch
from torch import nn
from d2l import torch as d2l


def pool2d(X, pool_size, mode='max'):   # 本函数已保存在d2lzh包中方便以后使用
    h, w = pool_size  # 取出K的行数和列数
    # print("h=", h, "w=", w)
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))  # 生成输出特征图的大小并且初始化为0
    # print("X.shape[0]", X.shape[0])  # X.shape[0]是X 3*3的行数3 X.shape[1]是X 3*3的列数3
    # print("Y=", Y)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                # print(X[i: i + h, j: j + w], "**")
                Y[i, j] = X[i: i + h, j: j + w].max()
                # X[i: i + h, j: j + w]取出对应区域
            elif mode == 'avg':
                # print(X[i: i + h, j: j + w], "**")
                Y[i, j] = X[i: i + h, j: j + w].mean()
    return Y


X = torch.tensor([[0.0, 1.0, 2.0],
                  [3.0, 4.0, 5.0],
                  [6.0, 7.0, 8.0]])
print(pool2d(X, (2, 2)))
print(pool2d(X, (2,2), 'max'))
print(pool2d(X, (2,2), 'avg'))
print('\n')

# 6.5.2 填充和步幅
# 构造一个输入张量X，它有四个维度，其中样本数和通道数都是1。
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
print(X)

# 使用形状为(3, 3)的汇聚窗口，
# 那么默认情况下，得到的步幅形状为(3, 3)。
pool2d = nn.MaxPool2d(3)
print(pool2d(X))
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
print(pool2d(X))
print('\n')


# 6.5.3 多个通
# 在处理多通道输入数据时，
# 汇聚层在每个输入通道上单独运算，
# 而不是像卷积层一样在通道上对输入进行汇总。
X = torch.cat((X, X + 1), 1)  # 创建一个向量
print(X)

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))





