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
# print('X=', X, '\ny=', y)
# print('X[16]=', X[16])
# print('y[0]=', y[0])
# X是图片，y是标签
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
# plt.show()

# 3.5.2 读取小批量
batch_size = 256


def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4


train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')


def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]  # 规定数据会进行什么变换
    if resize:  # 如果resize这个参数不为0我们就会重置它的分辨率
        trans.insert(0, transforms.Resize(resize))  # 运行Resize函数，重置分辨率
    trans = transforms.Compose(trans)  # 把这一系列的操作组合在一起
    mnist_train = torchvision.datasets.FashionMNIST(  # 获取训练数据集
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(  # 获取测试数据集
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))
# 对数据进行一系列的操作，
# trans就可以理解为这些操作的一个集合
# resize就是重新定义它的分辨率
# 在pytorch中，我们的一切数据都需要分装成dataLoader，
# 给定一个数据集，指定batch_size,再指定是否打乱，
# 指定几个进程去读。这个都是用的很多的


train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
