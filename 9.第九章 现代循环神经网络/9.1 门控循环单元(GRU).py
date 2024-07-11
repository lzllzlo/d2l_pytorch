# 9.1 门控循环单元(GRU)
import torch
from torch import nn
from d2l import torch as d2l
# 9.1.2 从零开始实现
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)


# 初始化模型参数
def get_params(vocab_size, num_hiddens, device):
    # 定义输入和输出的维度，通常等于词汇表的大小
    num_inputs = num_outputs = vocab_size

    # 辅助函数：生成服从正态分布的随机张量
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 辅助函数：生成门控循环单元（GRU）相关参数
    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    # 初始化更新门参数（W_xz, W_hz, b_z）
    W_xz, W_hz, b_z = three()
    # 初始化重置门参数（W_xr, W_hr, b_r）
    W_xr, W_hr, b_r = three()
    # 初始化候选隐状态参数（W_xh, W_hh, b_h）
    W_xh, W_hh, b_h = three()
    # 初始化输出层参数（W_hq, b_q）
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # 将所有参数放入列表
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]

    # 设置所有参数的梯度属性为True，以便进行反向传播
    for param in params:
        param.requires_grad_(True)

    # 返回包含所有模型参数的列表
    return params


# 定义模型初始化函数
# 返回一个形状为（批量大小，隐藏单元个数）的张量，张量的值全部为零。
def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )


# 现在我们准备定义门控循环单元模型，
# 模型的架构与基本的循环神经网络单元是相同的，
# 只是权重更新公式更为复杂。
# 与公式一一对应。
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


# 训练与预测
# 训练和预测的工作方式与 8.5节完全相同。
# 训练结束后，我们分别打印输出训练集的困惑度，
# 以及前缀“time traveler”和“traveler”的预测序列上的困惑度。
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                            init_gru_state, gru)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
d2l.plt.show()

# 总结
# 门控循环神经网络可以更好地捕获时间步距离很长的序列上的依赖关系。
# 重置门有助于捕获序列中的短期依赖关系。
# 更新门有助于捕获序列中的长期依赖关系。
# 重置门打开时，门控循环单元包含基本循环神经网络；更新门打开时，门控循环单元可以跳过子序列。
