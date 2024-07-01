import torch

# 2.5.1 一个简单的例子

x = torch.arange(4.0)
print(x)

x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
print(x)
print(x.grad)  # 默认值是None

y = 2 * torch.dot(x, x)
print(y)

y.backward()
print(x)
print(x.grad)

print(x.grad == 4 * x)

x.grad.zero_()
y = x.sum()
y.backward()
print(x)
print(x.grad)

# 2.5.2 非标量变量的反向传播
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
print('y=', y)
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
print('x.grad=', x.grad)

# 2.5.3 分离结算
print('\n')
print(x.grad)
x.grad.zero_()
print(x.grad)
print(y)
y = x * x
print(y)
u = y.detach()
print(u)
print(x)
z = u * x
print(z)

z.sum().backward()
print(x.grad == u)

x.grad.zero_()
y.sum().backward()
print(x)
print(x.grad)
print(x.grad == 2 * x)


# 2.5.4 Python控制流的梯度计算
print('2.5.4')


def f(a):
    b = a * 2
    while b.norm() < 1000:
        print('b=', b)
        print('b.norm=', b.norm)
        print('b.sum=', b.sum)
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad)
print(a.grad == d / a)
