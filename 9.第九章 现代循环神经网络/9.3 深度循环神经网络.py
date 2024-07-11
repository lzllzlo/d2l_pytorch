# 9.3 深度循环神经网络

# 9.3.2 简洁实现

import torch
from torch import nn
from d2l import torch as d2l

# 加载数据集
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# 像选择超参数这类架构决策也跟 9.2节中的决策非常相似。
# 因为我们有不同的词元，所以输入和输出都选择相同数量，即vocab_size。
# 隐藏单元的数量仍然是256。
# 唯一的区别是，我们现在通过num_layers的值来设定隐藏层数。
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = d2l.try_gpu()
# 这里改变了层数
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)

num_epochs, lr = 500, 2
d2l.train_ch8(model, train_iter, vocab, lr*1.0, num_epochs, device)
d2l.plt.show()

# 小结
# • 在深度循环神经网络中，隐状态的信息被传递到当前层的下一时间步和下一层的当前时间步。
# • 有许多不同风格的深度循环神经网络，如长短期记忆网络、门控循环单元、或经典循环神经网络。这些
# 模型在深度学习框架的高级API中都有涵盖。
# • 总体而言，深度循环神经网络需要大量的调参(如学习率和修剪)来确保合适的收敛，模型的初始化也
# 需要谨慎。
