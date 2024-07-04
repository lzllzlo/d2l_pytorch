# 7.3 网络中的网络(NiN)

# 7.3.1 NiN块
import torch                                        #引入依赖包
from torch import nn
from d2l import torch as d2l


# 定义nin网络块
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        # 定义卷积层块
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())


# 7.3.2 NiN模型
# 定义网络中的网络NiN模型
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),  # 定义一个NiN块
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),  # 使用暂退法减低模型复杂度

    # 标签类别数是 10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),  # 自定义池化层，最终样本的像素被压缩为(1, 1)

    # 将四维的输出转成二维的输出，其形状为(批量大小， 10)
    nn.Flatten())

# 创建一个数据样本来查看每个块的输出形状。
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

# 7.3.3 训练模型
# 使用Fashion‐MNIST来训练模型。
# 训练NiN与训练AlexNet、VGG时相似。
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

















