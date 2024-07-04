# 7.4 含并行连结的网络(GoogLeNet)
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 7.4.1 Inception块


# 定义GoogLeNet的Inception块
class Inception(nn.Module):

    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)

        # 线路1， 单1*1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)

        # 线路2， 1*1卷积层后接3*3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

        # 线路3， 1*1卷积层后接5*5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)

        # 线路4， 3*3最大汇聚层后接1*1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    # 前向传播函数，计算神经网络的结果
    def forward(self, x):
        p1 = F.relu(self.p1_1(x))  # 线路1的卷积结果
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))  # 线路2的卷积结果
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))  # 线路3的卷积结果
        p4 = F.relu(self.p4_2(self.p4_1(x)))  # 线路4的卷积结果

        # 在通道的维数上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)


# 7.4.2 GoogLeNet模型
# 第一个模块使用64个通道、7 × 7卷积层。
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 第二个模块使用两个卷积层:第一个卷积层是64个通道、1 × 1卷积层;
# 第二个卷积层使用将通道数量增加三倍的3 × 3卷积层。
# 这对应于Inception块中的第二条路径。
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 第三个模块串联两个完整的Inception块。
# 第一个Inception块的输出通道数为64 + 128 + 32 + 32 = 256，
# 四个路径之间的输出通道数量比为64 : 128 : 32 : 32 = 2 : 4 : 1 : 1。
# 第二个和第三个路径首先将输入通道的数量分别减少到96/192 = 1/2和16/192 = 1/12，
# 然后连接第二个卷积层。第二个Inception块的输出通道数增加到128+192+96+64 = 480，
# 四个路径之间的输出通道数量比为128 : 192 : 96 : 64 = 4 : 6 : 3 : 2。
# 第二条和第三条路径首先将输入通道的数量分别减少到128/256 = 1/2和32/256 = 1/8。
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 第四模块更加复杂，它串联了5个Inception块，
# 其输出通道数分别是192 + 208 + 48 + 64 = 512、
# 160 + 224 + 64+64 = 512、
# 128+256+64+64 = 512、
# 112+288+64+64 = 528
# 和256+320+128+128 = 832。
# 这些路径的通道数分配和第三模块中的类似，
# 首先是含3×3卷积层的第二条路径输出最多通道，
# 其次是仅含1×1卷 积层的第一条路径，
# 之后是含5×5卷积层的第三条路径
# 和含3×3最大汇聚层的第四条路径。
# 其中第二、第三条 路径都会先按比例减小通道数。
# 这些比例在各个Inception块中都略有不同。
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 第五模块包含输出通道数为256 + 320 + 128 + 128 = 832和384 + 384 + 128 + 128 = 1024的两个Inception块。
# 其中每条路径通道数的分配思路和第三、第四模块中的一致，只是在具体数值上有所不同。
# 需要注意的是，第五模块的后面紧跟输出层，该模块同NiN一样使用全局平均汇聚层，将每个通道的高和宽变成1。
# 最后我们将输出变成二维数组，再接上一个输出个数为标签类别数的全连接层。
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())
net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

# GoogLeNet模型的计算复杂，而且不如VGG那样便于修改通道数。
# 为了使Fashion‐MNIST上的训练短小精悍，我们将输入的高和宽从224降到96，这简化了计算。
# 下面演示各个模块输出的形状变化。
X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

# 7.4.3 训练模型
# 我们使用Fashion‐MNIST数据集来训练我们的模型。
# 在训练之前，我们将图片转换为96 × 96分辨率。
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())



