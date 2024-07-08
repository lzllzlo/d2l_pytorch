# 8.4 循环神经网络
import torch
from d2l import torch as d2l
# 8.4.2 有隐状态的循环神经网络
# 首先，我们定义矩阵X、 W_xh、H和W_hh，它们的形状分别为(3􏰀1)、(1􏰀4)、(3􏰀4)和(4􏰀4)。
# 分别将X乘以W_xh，将H乘以W_hh，然后将这两个乘法相加，我们得到一个形状为(3􏰀4)的矩阵。
X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))
print(torch.matmul(X, W_xh) + torch.matmul(H, W_hh))

# 现在，我们沿列(轴1)拼接矩阵X和H，沿行(轴0)拼接矩阵W_xh和W_hh。
# 这两个拼接分别产生形状(3, 5)和 形状(5, 4)的矩阵。
# 再将这两个拼接的矩阵相乘，我们得到与上面相同形状(3, 4)的输出矩阵。
print(torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0)))




