# 5.4 自定义层
import torch
import torch.nn.functional as F
from torch import nn


# 5.4.1 不带参数的层

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,X):
        return X - X.mean()


layer = CenteredLayer()
# 测试
print(layer(torch.FloatTensor([1,2,3,4,5])))

#  将层作为组件组合到更加复杂的模型中
net = nn.Sequential(nn.Linear(8,128),CenteredLayer())

Y = net(torch.rand(4, 8))
# 在向该网络发送随机数据后，检查均值是否为0。
# 由于我们处理的是浮点数，因为存储精度的原因，
# 我们仍然可能会看到一个非常小的非零数。
print(Y.mean())

# 5.4.2 带参数的层
# 自定义版本的全连接层。
# 回想一下，该层需要两个参数，一个用于表示权重，另一个用于表示偏置项。
# 在此实现中，我们使用修正线性单元作为激活函数。
# 该层需要输入参数:in_units和units，分 别表示输入数和输出数。


# 定义自带参数的层  输入和输出
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        # 初始化权重参数
        self.weight = nn.Parameter(torch.randn(in_units, units))
        # 初始化偏置
        self.bias = nn.Parameter(torch.randn(units, ))

    def forward(self, X):
        #  计算线性层
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


# 实例化MyLinear类并访问其模型参数
#  权重矩阵  5 x 3
linear = MyLinear(5, 3)
print(linear.weight)

print(linear(torch.randn(2, 5)))  # 两个样本 五个特征

# 使用自定义层构建模型，就像使用内置的全连接层一样使用自定义层。
net = nn.Sequential(MyLinear(64, 8),
                    MyLinear(8, 1))
print(net(torch.rand(2, 64)))














