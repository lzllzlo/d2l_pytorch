# 4.7 前向传播、反向传播和计算图
# 前向传播(forward propagation或forward pass)指的是:
# 按顺序(从输入层到输出层)计算和存储神经网络中每层的结果。
# 反向传播(backward propagation或backpropagation)指的是计算神经网络参数梯度的方法。
# 简言之，该方 法根据微积分中的链式规则，按相反的顺序从输出层到输入层遍历网络。
# 该算法存储了计算某些参数梯度时 所需的任何中间变量(偏导数)。
# 在训练神经网络时，前向传播和反向传播相互依赖。
# 对于前向传播，我们沿着依赖的方向遍历计算图并计算其路径上的所有变量。
# 然后将这些用于反向传播，其中计算顺序与计算图的相反。
