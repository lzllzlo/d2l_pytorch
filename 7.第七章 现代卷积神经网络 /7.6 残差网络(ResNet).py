# 7.6 残差网络(ResNet)
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 7.6.2 残差块
"""
定义残差网络
每个残差块的具体逻辑：
1、3*3卷积层操作
2、批量规范化
3、relu激活函数
4、3*3卷积层操作
5、批量规范化
称以上5步的操作为f(x)函数
若未指定1*1卷积操作，则输出返回 x + f(x)
否则返回 conv3(x) + f(x)， 其中conv3()代表1*1卷积层
"""


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)   # 3*3卷积操作
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)                   # 3*3卷积操作
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,           # 是否使用1*1卷积层操作
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)  # 两个批量规范化操作

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))  # 对样本数据进行两次卷积操作，得到g(x)
        if self.conv3:
            X = self.conv3(X)
        Y += X  # 加上x，即 f(x) = g(x) + x
        return F.relu(Y)


# 下面我们来查看输入和输出形状一致的情况。
# 注意，当未使用1*1卷积层时，输入通道数和输出通道数要保持一致，
# 否则会出现 X 与 Y 形状不一致相加出现错误的现象
blk = Residual(3, 3)
X = torch.rand(4, 3, 6, 6)  # 定义X数据集为4个样本数，3个通道，每个图片为 6*6
Y = blk(X)
print(Y.shape)

# 我们也可以在增加输出通道数的同时，减半输出的高和宽。
blk = Residual(3, 6, use_1x1conv=True, strides=2)
print(blk(X).shape)

# 7.6.3 ResNet模型
# ResNet的前两层跟之前介绍的GoogLeNet中的一样:
# 在输出通道数为64、步幅为2的7 × 7卷积层后，接步幅为2的3 × 3的最大汇聚层。
# 不同之处在于ResNet每个卷积层后增加了批量规范化层。
# 定义b1环节模型，包含一个 7 × 7 的卷积层、批量规范化层、relu激活函数、最大汇聚层(池化层)。
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                  nn.BatchNorm2d(64), nn.ReLU(),
                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


# 定义残差块，输入参数分别为输入、输出通道数，残差网络数目
def resent_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []  # 定义残差网络列表

    for i in range(num_residuals):

        if i == 0 and not first_block:
            # 如果是第一个残差网络，则将宽高减半
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))

        else:
            # 后续的残差网络
            blk.append(Residual(num_channels, num_channels))

    return blk


# 接着在ResNet加入所有残差块，这里每个模块使用2个残差块。
b2 = nn.Sequential(*resent_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resent_block(64, 128, 2))
b4 = nn.Sequential(*resent_block(128, 256, 2))
b5 = nn.Sequential(*resent_block(256, 512, 2))

# 最后，在ResNet中加入全局平均汇聚层，以及全连接层输出。
net = nn.Sequential(b1, b2, b3, b4, b5,
                   nn.AdaptiveAvgPool2d((1, 1)),
                   nn.Flatten(), nn.Linear(512, 10))

# 观察网络结构49
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

# 7.6.4 训练模型
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
