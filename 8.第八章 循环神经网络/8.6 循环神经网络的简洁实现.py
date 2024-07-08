# 8.6 循环神经网络的简洁实现
# 虽然 8.5节 对了解循环神经网络的实现方式具有指导意义，但并不方便。
# 本节将展示如何使用深度学习框架的高级API提供的函数更有效地实现相同的语言模型。
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# 我们仍然从读取时光机器数据集开始。

# 8.6.1 定义模型
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)  # 定义rnn循环神经网络层数的两个参数(字典、隐单元个数)

# 我们使用张量来初始化隐状态，它的形状是(隐藏层数，批量大小，隐藏单元数)。
# 初始化隐状态
state = torch.zeros((1, batch_size, num_hiddens))
print(state.shape)

# 通过一个隐状态和一个输入，我们就可以用更新后的隐状态计算输出。
# 需要强调的是，rnn_layer的“输出”(Y)不涉及输出层的计算:
# 它是指每个时间步的隐状态，这些隐状态可以用作后续输出层的输入。
X = torch.rand(size=(num_steps, batch_size, len(vocab)))    # X的形状为(时间步数、批量大小、词典长度)

# 其中rnn_layer接收的两个参数
# (X:sequence_length、batch_size、vocab_size， state:num_layers、batch_size、num_hiddens)
# 输出的两个参数为
# (X:sequence_length、batch_size、num_hiddens， state:num_layers、batch_size、num_hiddens)
Y, state_new = rnn_layer(X, state)
print(Y.shape, state_new.shape)


# 与 8.5节类似，我们为一个完整的循环神经网络模型定义了一个RNNModel类。
# 注意，rnn_layer只包含隐藏的循环层，我们还需要创建一个单独的输出层。
# 定义RNN循环神经网络模型
class RNNModel(nn.Module):
    """循环神经网络模型"""

    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)

        self.rnn = rnn_layer  # 定义 rnn 循环层
        self.vocab_size = vocab_size  # 定义词典大小为 vocab_size
        self.num_hiddens = self.rnn.hidden_size  # 定义隐单元个数为 hidden_size

        # 如果RNN是双向的(之后将介绍)，num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)  # 定义单层的输出层
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)  # 双向 rnn 循环神经网络

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)  # 将inputs进行one-hot编码，结果形状为(35, 32, 28)
        X = X.to(torch.float32)  # 将X放入device设备之中

        # 计算循环层的结果，得到Y和state
        # 输出的结果形状分别为 X:(35, 32, 256)、state:(1, 32, 256)
        Y, state = self.rnn(X, state)

        # 全连接层首先将Y的形状改为(时间步数*批量大小，隐单元数量)
        # 它的输出形状为(时间步数*批量大小，词表大小)
        # 最后得到的输出结果为(35*32, 28)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))

        # 返回最终的结果和记忆状态
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态(单向传播)
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens))
        else:
            # nn.LSTM以元组作为隐状态(双向传播)
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))


# 8.6.2 训练与预测
device = d2l.try_gpu()  # 选择设备
net = RNNModel(rnn_layer, vocab_size=len(vocab))    # 实例化RNN模型
net = net.to(device)
print(d2l.predict_ch8('time traveller', 10, net, vocab, device))   # 预测数据

# 接下来，我们使用 8.5节中定义的超参数调用train_ch8，并且使 用高级API训练模型。
num_epochs, lr = 500, 1
print(d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device))
d2l.plt.show()

# 小结
# • 深度学习框架的高级API提供了循环神经网络层的实现。
# • 高级API的循环神经网络层返回一个输出和一个更新后的隐状态，我们还需要计算整个模型的输出层。
# • 相比从零开始实现的循环神经网络，使用高级API实现可以加速训练。

