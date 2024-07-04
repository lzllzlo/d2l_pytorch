# 8.1 序列模型
import torch
from torch import nn
from d2l import torch as d2l

# 8.1.2 训练
# 我们生成一些数据:使用正弦函数和一些可加性噪声来生成序列数据，时间步为1, 2, . . . , 1000。
T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)  # 产生从 1 到 1000 的所有数据点
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T, ))  # 定义所有time对应的x, 每一个x还会添加对应的噪声分布
d2l.plot(time, [x], 'time', xlim=[1, 1000], figsize=(6, 3))  # 绘图
d2l.plt.show()  # 显示绘制的图像

# 接下来，我们将这个序列转换为模型的特征-标签(feature‐label)对。
# 基于嵌入维度τ ，我们将数据映射为数据对yt = xt 和xt = [xt−τ , . . . , xt−1 ]。
# 这比我们提供的数据样本少了τ个，因为我们没有足够的历史记录来描述前τ个数据样本。
# 一个简单的解决办法是:如果拥有足够长的序列就丢弃这几项;另一个方法是用零填充序列。
# 在这里，我们仅使用前600个“特征-标签”对进行训练。
tau = 4  # 嵌入维度等于4
features = torch.zeros((T - tau, tau))  # 即特征的样本量为996行，4列数据

# 生成996行的样本数
for i in range(tau):
    features[:, i] = x[i: T - tau + i]  # 其中每行对应每个标签, features对应每个样本

# labels相当于标签结果
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600  # 设置批量数据大小为16, 训练集大小为600

# 只有前 n_train 个样本用于训练
# 加载数据集为一个训练集的迭代器
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)


# 在这里，我们使用一个相当简单的架构训练模型:
# 一个拥有两个全连接层的多层感知机，ReLU激活函数和平方损失。
# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                       nn.ReLU(),
                       nn.Linear(10, 1))
    net.apply(init_weights)
    return net


# 平方损失。注意: MSELoss计算平方误差时不带系数1/2
loss = nn.MSELoss(reduction='none')


# 现在，准备训练模型了。实现下面的训练代码的方式与前面几节(如 3.3节)中的循环训练基本相同。
# 定义训练函数
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)  # 定义Adam优化器

    # 多次迭代训练模型
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()  # 清空梯度
            l = loss(net(X), y)  # 计算损失
            l.sum().backward()  # 调用反向传播函数计算梯度
            trainer.step()  # 更新梯度

        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')


net = get_net()
train(net, train_iter, loss, 5, 0.01)

# 8.1.3 预测
# 由于训练损失很小，因此我们期望模型能有很好的工作效果。
# 让我们看看这在实践中意味着什么。
# 首先是检查模型预测下一个时间步的能力，也就是单步预测(one‐step‐ahead prediction)。
onestep_preds = net(features)   # 定义一个网络，输出一步长的预测
d2l.plot([time, time[tau:]],    # 绘图预测
         [x.detach().numpy(), onestep_preds.detach().numpy()],
         'time', 'x',
         legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))
d2l.plt.show()


multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]

# 此为(k)多步预测，使用预测的样本值再次预测新数据
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)))

d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()],
         'time', 'x',
         legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))
d2l.plt.show()
# 来24小时的天气预报往往相当准确，但超过这一点，精度就会迅速下降。

# 基于k = 1, 4, 16, 64，通过对整个序列预测的计算，让我们更仔细地看一下k步预测的困难。
max_steps = 64  # 设置的最大步长数为64

features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))

# 列 i (i < tau) 是来自x的观测，其时间步从(i)到(i+T-tau-max_steps +1)
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# 使用预测得出的数据再次预测新的数据
# 列 i (i >= tau) 是来自(i-tau+1)步的预测，其时间步从 (i) 到 (i+T-tau-max_steps+1)
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

# 步数元组为(1, 4, 16, 64)
steps = (1, 4, 16, 64)
d2l.plot([time[tau + i -1: T - max_steps + i] for i in steps],  # 绘制出4个x的取值范围(4, 937)、(7, 940)、(19, 952)、(67, 1000)
        [features[:, (tau + i - 1)].detach().numpy() for i in steps],   # 绘制出它们对应的预测值
        'time', 'x', legend=[f'{i}-step preds' for i in steps],
        xlim=[5, 1000], figsize=(6, 3))
d2l.plt.show()
# 小结
# • 内插法(在现有观测值之间进行估计)和外推法(对超出已知观测范围进行预测)在实践的难度上差别很大。
# 因此，对于所拥有的序列数据，在训练时始终要尊重其时间顺序，即最好不要基于未来的数据进 行训练。
# • 序列模型的估计需要专门的统计工具，两种较流行的选择是自回归模型和隐变量自回归模型。
# • 对于时间是向前推进的因果模型，正向估计通常比反向估计更容易。
# • 对于直到时间步t的观测序列，其在时间步t + k的预测输出是“k步预测”。
# 随着我们对预测时间k值的 增加，会造成误差的快速累积和预测质量的极速下降。

