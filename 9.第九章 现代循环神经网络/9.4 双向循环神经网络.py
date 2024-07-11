# 9.4 双向循环神经网络

# 9.4.3 双向循环神经网络的错误应用

# 由于双向循环神经网络使用了过去的和未来的数据，
# 所以我们不能盲目地将这一语言模型应用于任何预测任务。
# 尽管模型产出的困惑度是合理的，
# 该模型预测未来词元的能力却可能存在严重缺陷。
# 我们用下面的示例代码引以为戒，
# 以防在错误的环境中使用它们。

import torch
from torch import nn
from d2l import torch as d2l

# 加载数据
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 通过设置“bidirective=True”来定义双向LSTM模型
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
# 训练模型
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

d2l.plt.show()

# 小结
# • 在双向循环神经网络中，每个时间步的隐状态由当前时间步的前后数据同时决定。
# • 双向循环神经网络与概率图模型中的“前向‐后向”算法具有相似性。
# • 双向循环神经网络主要用于序列编码和给定双向上下文的观测估计。
# • 由于梯度链更长，因此双向循环神经网络的训练代价非常高。
