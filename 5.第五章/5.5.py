# 5.5 读写文件
import torch
from torch import nn
from torch.nn import functional as F

# 5.5.1 加载和保存张量
x = torch.arange(4)
print('x=', x)
torch.save(x, 'x-file')  # 保存张量
x2 = torch.load('x-file')  # 加载张量
print('x2=', x2)

# 存储一个张量列表，然后把它们读回内存。
y = torch.zeros(4)
print('y=', y)
torch.save([x, y], 'x-files')
x2, y2 = torch.load('x-files')
print('x2=', x2, 'y2=', y2)

# 写入或读取从字符串映射到张量的字典。
# 当我们要读取或写入模型中的所有权重时，这很方便。
mydict = {'x': x, 'y': y}
print('mydict=', mydict)
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print('mydict2=', mydict2)
print('\n')


# 5.5.2 加载和保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


net = MLP()
X = torch.randn(size=(2, 20))
print('X=', X)
Y = net(X)
print('Y=', Y)

# 将模型的参数存储在一个叫做“mlp.params”的文件中。
torch.save(net.state_dict(), 'mlp.params')

# 为了恢复模型，我们实例化了原始多层感知机模型的一个备份。
# 这里我们不需要随机初始化模型参数，
# 而是直接读取文件中存储的参数。
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())

# 由于两个实例具有相同的模型参数，
# 在输入相同的X时，两个实例的计算结果应该相同。
# 让我们来验证一下。
print('X=', X)
Y_clone = clone(X)
print('Y_clone=', Y_clone)
print('Y=', Y)
print(Y_clone == Y)










