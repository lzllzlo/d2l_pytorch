# 6.4 多输入多输出通道

# 6.4.1 多输入通道
# 对每个通道输入的二维张量和卷积核的二维张量进行互相关运算，
# 再对通道求和(将ci 的结果相加)得到二维张量。
# 这是多通道输入和多输入通道卷积核之间进行二维互相关运算的结果。

# 对每个通道执行互相关操作，
# 然后将结果相加。
import torch
from d2l import torch as d2l


def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))


X = torch.tensor([[[0.0, 1.0, 2.0],
                   [3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0],
                   [2.0, 3.0]],
                  [[1.0, 2.0],
                   [3.0, 4.0]]])

print(corr2d_multi_in(X, K))
print('\n')


# 6.4.2 多输出通道
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


K = torch.stack((K, K + 1, K + 2), 0)
# 将核张量K与K+1(K中每个元素加1)和K+2连接起来，
# 构造了一个具有3个输出通道的卷积核。
print(K)
print(K.shape)
print(corr2d_multi_in_out(X, K))
print('\n')


# 6.4.3 1×1卷积层

def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))


X = torch.normal(0, 1, (3, 3, 3))  # 从0～1正态分布中创建矩阵3*3*3
K = torch.normal(0, 1, (2, 3, 1, 1))  # 从0～1正态分布中创建矩阵2*3*1*1
print('X=', X)
print('K=', K)
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
# 结果是一样的
print('Y1=', Y1)
print('Y2=', Y2)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6





















