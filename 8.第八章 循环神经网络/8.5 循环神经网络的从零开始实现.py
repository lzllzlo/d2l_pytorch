# 8.5 循环神经网络的从零开始实现
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 定义数据批量大小为 32， 时间步数为 35，
# 即迭代器train_iter内的X, y的形状均为(32, 35), vocab为词表
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 8.5.1 独热编码
# 索引为0和2的独热向量如下所示
print(F.one_hot(torch.tensor([0, 2]), len(vocab)))
X = torch.arange(10).reshape((2, 5))
print(F.one_hot(X.T, 28).shape)


# 8.5.2 初始化模型参数
def get_params(vocab_size, num_hiddens, device):
    # 词表的大小，输入大小，输出大小相同
    num_inputs = num_outputs = vocab_size

    # 生成初始化分布值
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))  # 输入时的权重
    W_hh = normal((num_hiddens, num_hiddens))  # 序列记忆的权重
    b_h = torch.zeros(num_hiddens, device=device)  # 偏置量

    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))  # 输出时的权重
    b_q = torch.zeros(num_outputs, device=device)  # 输出的偏置量

    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)

    # 返回参数
    return params


# 8.5.3 循环神经网络模型
# 初始化rnn循环神经网络的隐状态
def init_rnn_state(batch_size, num_hiddens, device):
    # 返回一个记忆状态信息
    return (torch.zeros((batch_size, num_hiddens), device=device), )


# 前向传播函数，并计算状态值 H
def rnn(inputs, state, params):
    # inputs的形状: (时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params

    H, = state  # 记忆状态
    outputs = []  # 初始化输出列表

    # X的形状: (批量大小，词表大小)g
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)  # 计算结果记忆H
        Y = torch.mm(H, W_hq) + b_q  # 计算输出结果Y
        outputs.append(Y)  # 追加 结果Y 至 output输出中

    return torch.cat(outputs, dim=0), (H,)  # 返回记忆H与结果Y


# 定义了所有需要的函数之后，接下来我们创建一个类来包装这些函数，
# 并存储从零开始实现的循环神经网络模型的参数。
class RNNModelScratch:
    """从零开始实现的循环神经网络模型"""

    def __init__(self, vocab_size, num_hiddens, device, get_params,
                 init_state, forward_in):
        # 定义词表大小，隐单元个数
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens

        # 定义参数
        self.params = get_params(vocab_size, num_hiddens, device)

        # 定义初始化隐状态，前向传播函数
        self.init_state, self.forward_in = init_state, forward_in

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_in(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


# 让我们检查输出是否具有正确的形状。例如，隐状态的维数是否保持不变。
num_hiddens = 512   # 定义隐单元个数512

# 定义网络
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                     init_rnn_state, rnn)

# 初始化记忆隐状态
state = net.begin_state(X.shape[0], d2l.try_gpu())

d2l.try_gpu()
# 返回Y的值和状态state的值
Y, new_state = net(X.to(d2l.try_gpu()), state)

print(Y.shape, len(new_state), new_state[0].shape)
# 我们可以看到输出形状是(时间步数×批量大小，词表大小)，
# 而隐状态形状保持不变，即(批量大小，隐藏单元数)。


# 8.5.4 预测
def predict_ch8(prefix, num_preds, net, vocab, device):
    """在prefix后面生成新字符"""
    # 生成最初的隐状态数据，形状为(1, 512)
    state = net.begin_state(batch_size=1, device=device)

    # 首先获取第一个字母对应的字典idx，初始化输出列表
    outputs = [vocab[prefix[0]]]

    # 初始化输入列表，输入数据为输出列表的最后一个数据
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))

    # 预热期
    for y in prefix[1:]:
        _, state = net(get_input(), state)  # 记录当前的隐状态
        outputs.append(vocab[y])  # 输出集合追加y在字典中的索引

    # 预测num_preds步
    for _ in range(num_preds):
        y, state = net(get_input(), state)  # 进行预测操作，返回预测得到的结果及隐状态
        outputs.append(int(y.argmax(dim=1).reshape(1)))  # 追加预测所得结果new_y至输出列表中

    return ''.join([vocab.idx_to_token[i] for i in outputs])  # 输出预测所得的新数据


# 现在我们可以测试predict_ch8函数。我们将前缀指定为time traveller，并基于这个前缀生成10个后续字符。
# 鉴于我们还没有训练网络，它会生成荒谬的预测结果。
print(predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu()))
print('\n')


# 8.5.5 梯度裁剪
# 下面我们定义一个函数来裁剪模型的梯度，模型是从零开始实现的模型或由高级API构建的模型。
# 我们在此计算了所有模型参数的梯度的范数。
def grad_clipping(net, theta):
    """裁剪梯度"""
    # 首先获取神经网络中的参数
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params

    # 计算参数梯度的第二范式
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))

    # 若norm > theta， 则对网络的所有参数进行梯度裁剪
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


# 8.5.6 训练
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代的周期"""
    state, timer = None, d2l.Timer()  # 定义初始化状态，时间类
    metric = d2l.Accumulator(2)  # 训练损失之和， 词元数量

    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state(32, 512)
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state 对于 nn.GRU 是个张量
                state.detach_()

            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()

        y = Y.T.reshape(-1)  # 将y展开为一维数组
        X, y = X.to(device), y.to(device)  # 把x，y放入选定设备中
        y_hat, state = net(X, state)  # 预测y_hat(1120,28)以及记忆state(32,512)
        l = loss(y_hat, y.long()).mean()  # 计算损失值

        if isinstance(updater, torch.optim.Optimizer):  # torch包内的优化器
            updater.zero_grad()  # 清空梯度
            l.backward()  # 反向传播函数
            grad_clipping(net, 1)  # 梯度裁剪
            updater.step()  # 更新参数
        else:
            l.backward()  # 自定义的优化器
            grad_clipping(net, 1)  # 梯度裁剪
            # 因为已经调用了mean函数
            updater(batch_size=1)  # 更新梯度

        metric.add(l * y.numel(), y.numel())  # 累计损失值与样本数量

    # 返回平均损失与训练时长
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


# 循环神经网络模型的训练函数既支持从零开始实现，也可以使用高级API来实现。
# 定义RNN循环神经网络的训练函数
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型"""
    loss = nn.CrossEntropyLoss()  # 定义交叉熵损失函数

    animator = d2l.Animator(xlabel='epoch', ylabel='proplexity',
                            legend=['train'], xlim=[10, num_epochs])

    # 初始化，定义优化函数
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)

    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)

    # 训练和预测
    for epoch in range(num_epochs):
        # 训练模型，获取平均损失以及训练速度
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)

        # 每迭代十次后，绘图
        if (epoch + 1) % 10 == 0:
            print(predict('time traverller'))
            animator.add(epoch + 1, [ppl])

    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))


num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
d2l.plt.show()


# 最后，让我们检查一下使用随机抽样方法的结果。
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True)
d2l.plt.show()

# 小结
# • 我们可以训练一个基于循环神经网络的字符级语言模型，根据用户提供的文本的前缀生成后续文本。
# • 一个简单的循环神经网络语言模型包括输入编码、循环神经网络模型和输出生成。
# • 循环神经网络模型在训练以前需要初始化状态，不过随机抽样和顺序划分使用初始化方法不同。
# • 当使用顺序划分时，我们需要分离梯度以减少计算量。
# • 在进行任何预测之前，模型通过预热期进行自我更新(例如，获得比初始值更好的隐状态)。
# • 梯度裁剪可以防止梯度爆炸，但不能应对梯度消失。

