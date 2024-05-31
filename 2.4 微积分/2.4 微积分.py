%matplotlib inline
import numpy as np
form matplotlib_inline import backend_inline
forem d2l import torch as d2l

# 2.4.1 导数和微分
def f(x):
    return 3 * x ** 2 - 4 * x
def numerical_lim(f, x, h):
    return (f(x+h) - f(x))/h
h = 0.1
for i in range(5)
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1
