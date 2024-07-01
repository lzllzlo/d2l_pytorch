import random
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt


def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('features:', features[999], '\nlabel:', labels[999])

d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)

plt.show()


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    print('num_examples=', num_examples)
    indices = list(range(num_examples))  # indices在这里是一个向量
    """
    numbers=list(range(1,11))
    print(numbers)
 
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    """
    # 这些样本是随机读取的，没有特定的顺序
    print('打乱前：', indices)
    random.shuffle(indices)  # 随机打乱列表
    print('打乱后：', indices)
    for i in range(0, num_examples, batch_size):  # range(start,stop[,step])    []代表不是必须
        batch_indices = torch.tensor(  # a = torch.tensor([1, 2, 3, 4, 5]) 创建向量
            indices[i:min(i + batch_size, num_examples)]
            # indices是一个向量
            # 获取向量的前三个元素可表示为：first_three_elements = v[0:3]
        )
        print('抽取：', batch_indices)
        yield features[batch_indices], labels[batch_indices]


batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# 3.2.3 初始化模型参数
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
#  当requires_grad 设置为 True 时,表示这个张量的梯度将会被计算并存储在张量的 grad 属性中
print('w=', w)
b = torch.zeros(1, requires_grad=True)
print('b=', b)


# 3.2.4 定义模型
def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b


# 3.2.5 定义损失函数
def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 3.2.6 定义优化算法
def sgd(params, lr, batch_size): #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        '''with torch.no_grad()是一个用于禁用梯度的上下文管理器。
        禁用梯度计算对于推理是很有用的，
        当我们确定不会调用 Tensor.backward()时，它将减少计算的内存消耗。
        因为在此模式下，即使输入为 requires_grad=True，
        每次计算的结果也将具有requires_grad=False。
        总的来说， with torch.no_grad() 可以理解为，
        在管理器外产生的与原参数有关联的参数requires_grad属性都默认为True，
        而在该管理器内新产生的参数的requires_grad属性都将置为False。'''
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 3.2.7 训练
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss


for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y) # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
