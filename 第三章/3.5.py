# 3.5 图像分类数据集

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
import matplotlib.pyplot as plt

d2l.use_svg_display()

# 3.5.1 读取数据集
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0~1之间
trans = transforms.ToTensor()
print(trans)
# 通过框架中的内置函数将Fashion‐MNIST数据集下载并读取到内存中。
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)

# print(mnist_train, mnist_test)
print(len(mnist_train), len(mnist_test))

# 每个输入图像的高度和宽度均为28像素。数据集由灰度图像组成，其通道数为1。
print(mnist_train[0][0].shape)


# 以下函数用于在数字标签索引及其文本名称之间进行转换。
def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 创建一个函数来可视化这些样本。
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


# 以下是训练数据集中前18个样本的图像及其相应的标签。
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
print('X=', X, '\ny=', y)
print('X[16]=', X[16])
print('y[0]=', y[0])
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
plt.show()
