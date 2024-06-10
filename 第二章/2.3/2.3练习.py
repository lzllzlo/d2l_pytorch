import torch

# 练习1
A = torch.arange(12).reshape(3, 4)
print(A)
B = A.T
print(B)
print(B.T == A)
print(A.T.T == A, '\n')

# 练习2
A = torch.arange(12).reshape(3, 4)
B = torch.ones(3, 4)
print(A)
print(B)
print(A.T + B.T == (A + B).T)

# 练习3
A = torch.arange(9).reshape(3, 3)
print(A)
print(A.T)
print(A + A.T)  # 因为是方阵所以本体与转置相加总得出对称的结果

# 练习4
X = torch.arange(24).reshape(2, 3, 4)
print(X)
print(len(X))

# 练习5
# 第一个轴

# 练习6
A = torch.arange(20, dtype=torch.float32).reshape(5,4)
print('\n', A)
print(A.sum(axis=0))
print(A/A.sum(axis=0))
# 要把axis=1改成0 要不然没法除

# 练习7
X = torch.arange(24).reshape(2, 3, 4)
print('\n')
print(X)
print('axis=0:', X.sum(axis=0))
print('axis=1:', X.sum(axis=1))
print('axis=2:', X.sum(axis=2))

# 练习8
X = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
print('\n')
print(X)
print(torch.linalg.norm(X, dim=0))
# L2范数，对应元素的平方和的平方根
print((X * X).sum() ** 0.5)
print(torch.norm(X))  # 所有元素的平方和的平方根
print(torch.abs(X).sum())
