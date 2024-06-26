# 4.6 暂退法(Dropout)
# 暂退法(Dropout)就是在训练的过程中以一定的概率随机灭活神经元,从而达到避免过拟合的效果。
# 4.6.4 从零开始实现
import torch
from torch import nn
from d2l import torch as d2l


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1  # 概率p必须大于等于0，小于等于 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:  # 全部灭活
        return torch.zeros_like(X)  # 在本情况中，所有元素都被保留
        # 将所有X重置为0，此时无论X与哪个矩阵作乘法，始终为0，可视为消去了所有隐藏单元
    if dropout == 0:  # 全部保留
        return X  # 即不对X作任何改变，不使用暂退法
    # 返回一个符合概率p的(0,1)分布张量，大于p的x为1, 小于p的x为0，这样就实现了一般情况
    # 小于概率p的x置为0， 大于概率p的x置为1
    mask = (torch.rand(X.shape) > dropout).float()  # 随机灭活
    # print('mask=', mask)
    return mask * X / (1.0 - dropout)  # 返回应用过暂退法(dropout)后的X隐藏单元


X = torch.arange(16, dtype = torch.float32).reshape((2, 8))
print('X=', X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))

# 定义模型参数
# 定义输入特征个数、输出类别个数、隐藏层1的隐神经单元数量、隐藏层2的隐神经单元数量
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

# 定义模型
dropout1, dropout2 = 0.2, 0.5  # 设置隐层1、隐层2的暂退概率


class Net(nn.Module):
    # 神经网络的初始化操作
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):

        super(Net, self).__init__()  # 父类初始化
        self.num_inputs = num_inputs  # 输入特征数量
        self.is_training = is_training  # 输入训练标志

        self.lin1 = nn.Linear(num_inputs, num_hiddens1)  # 线性模型1
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)  # 线性模型2
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)  # 线性模型3

        self.relu = nn.ReLU()  # 激活函数

    def forward(self, X):

        # 计算出隐层H1的隐单元值
        H1 = self.relu(self.lin1(X.reshape((-1, num_inputs))))
        # 只有在训练模型时才使用dropout
        # 在第一个全连接层后添加暂退dropout层
        if self.is_training == True:
            H1 = dropout_layer(H1, dropout1)  # 为隐藏层H1添加暂退概率，减少模型的复杂度
            # 在第一个全连接层之后添加一个dropout层
        H2 = self.relu(self.lin2(H1))
        # 在第二个全连接层后添加暂退dropout层
        if self.is_training == True:
            H2 = dropout_layer(H2, dropout2)  # 为隐藏层H2添加暂退概率，减少模型的复杂度
        out = self.lin3(H2)  # 输出分类的类别分布概率
        return out


# 深度学习模型net
net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

# 训练和测试
num_epochs, lr, batch_size = 10, 0.5, 256
# 定义迭代次数10，学习率0.5, 小批量样本数据集为256
loss = nn.CrossEntropyLoss(reduction='none')
# 定义交叉熵损失函数
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 获取训练集迭代器和测试集迭代器
updater = torch.optim.SGD(net.parameters(), lr)
# 定义优化函数，更新权重

# 开始训练数据集
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
# d2l.plt.show()

# 4.6.5 简洁实现
# 使用pytorch高级API直接定义神经网络net
net_concise = nn.Sequential(nn.Flatten(),  # 降维打击函数，把高阶的图片张量数据展开为一阶
                            nn.Linear(num_inputs, num_hiddens1),  # 定义第一个线性模型，从输入特征层到隐层1层的模型
                            nn.ReLU(),  # 应用激活函数
                            nn.Dropout(dropout1),  # 应用暂退函数
                            nn.Linear(num_hiddens1, num_hiddens2),  # 定义第二个线性模型，从隐层1层到隐层2层的模型
                            nn.ReLU(),  # 应用激活函数
                            nn.Dropout(dropout2),  # 应用暂退函数
                            nn.Linear(num_hiddens2, num_outputs))  # 定义第三个线性模型，从隐层2层到输出层的模型


# 以正态分布初始化神经网络权重
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net_concise.apply(init_weights)

updater_concise = torch.optim.SGD(net_concise.parameters(), lr)
d2l.train_ch3(net_concise, train_iter, test_iter, loss, num_epochs, updater_concise)
d2l.plt.show()



