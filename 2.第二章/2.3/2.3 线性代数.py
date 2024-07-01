import torch


# 2.3.1 标量
x = torch.tensor(3.0)
y = torch.tensor(2.0)

print(x + y, x * y, x / y, x ** y)

# 2.3.2 向量
x = torch.arange(4)

print(x)
print(x[0], x[1], x[2], x[3])

print('len(x)=', len(x))

print('x.shape=', x.shape)

# 2.3.3 矩阵
A = torch.arange(20).reshape(5, 4)
print('A=', A)

print('A.T=', A.T)

B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print('B=', B)
print('B == B.T', B == B.T)

# 2.3.4 张量
X = torch.arange(24).reshape(2, 3, 4)
print('X = ', X)

# 2.3.5 张量算法的基本性质
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
print('A = ', A, '\nA + B = ', A + B)
print('A * B = ', A * B)

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print('a + X =', a + X, '\n(a + X).shape=', (a + X).shape)

# 2.3.6 降维
x = torch.arange(4, dtype=torch.float32)
print('x = ', x, '\nx.sum() = ', x.sum())

print('\nA = ', A, '\nA.shape = ', A.shape, '\nA.sum() = ', A.sum())

A_sum_axis0 = A.sum(axis = 0)  # 按行降维
print('\nA = ', A, '\nA_sum_axis0 = ', A_sum_axis0, '\nA_sum_axis0.shape = ', A_sum_axis0.shape)

A_sum_axis1 = A.sum(axis = 1) # 按列降维
print('\nA = ', A, '\nA_sum_axis1 = ', A_sum_axis1, '\nA_sum_axis1.shape = ', A_sum_axis1.shape)

print('A.sum(axis=[0, 1]) = ', A.sum(axis = [0, 1])) # 同时按行和列降维

print(A.mean(), A.sum() / A.numel())
print(A.mean(axis=0), A.sum(axis=0) / A.shape[0])

# 非降维求和
sum_A = A.sum(axis=1, keepdims=True)
print(sum_A)
print(A)
print(A / sum_A)

print(A)
print(A.cumsum(axis=0))  # 按行累计，不降维

# 2.3.7 点积
print('\n')
y = torch.ones(4, dtype=torch.float32)
print('x=', x, 'y=', y, '\ntorch.dot(x, y) =', torch.dot(x, y))
print('torch.sum(x * y) = ', torch.sum(x * y))
# 点乘要求两个矩阵结构一样，把两个矩阵中相同位置的元素相乘，然后将其全部相加得到一个标量值（一个数字）。

# 2.3.8 矩阵-向量积
print('\n')
print(A, A.shape)
print(x, x.shape)
print(torch.mv(A, x))  # 拿x每个元素，依次乘以A的每行的每个元素，结果按行相加显示。

# 2.3.9 矩阵-矩阵乘法
B = torch.ones(4, 3)
print(A)
print(B)
print(torch.mm(A, B))
# 矩阵乘以矩阵，只需要第一个矩阵的行等于第二个矩阵的列，对应行乘以列，在相加，得到的结果是一个矩阵。

# 2.3.10 范数
u = torch.tensor([3.0, -4.0])
print('\n', torch.norm(u))  # L2范数：顶点距离
print(torch.abs(u).sum())  # L1范数：坐标值之和
print(torch.norm(torch.ones((4, 9))))  # Lp范数：矩阵元素平方和的平方根
