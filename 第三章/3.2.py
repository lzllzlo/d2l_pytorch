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

# plt.show()


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



