# 4.10 实战Kaggle比赛:预测房价

# 4.10.1 下载和缓存数据集
import hashlib
import os
import tarfile
import zipfile
import requests

# 如果没有安装pandas，请取消下一行的注释
# !pip install pandas

# 导入预处理所需的包
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

# 建立字典DATA_HUB，它可以将数据集名称的字符串映射到数据集相关的二元组上，
# 这个二元组包含数据 集的url和验证文件完整性的sha‐1密钥。
# 所有类似的数据集都托管在地址为DATA_URL的站点上。
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'  # 设置下载路径


# 下载函数
# download函数用来下载数据集，
# 将数据集缓存在本地目录(默认情况下为../data)中，并返回下载文件的名称。
# 如果缓存目录中已经存在此数据集文件，并且其sha‐1与存储在DATA_HUB中的相匹配，
# 我们将使用缓存的文件，以避免重复的下载。
def download(name, cache_dir=os.path.join('..', 'data')):
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


# 解压函数
def download_extract(name, folder=None):  #@save
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all():
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)


# 4.10.3 访问和读取数据集


# 将对应数据集注册成指定的命名
# 实际Kaggle数据集到特定项目页面下载即可
DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
# 分别下载测试集和训练集
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
# 查看训练集和测试集的尺寸
print(train_data.shape, test_data.shape)
# 查看训练集的开头四列和最后三列
print(train_data.iloc[0:4,[0,1,2,3,-3,-2,-1]])

# 数据中的Id列只是数据的编号，属于无用特征，于是在测试和训练集中去掉该列，
# 并将两个集合合并为所有特征。
all_features = pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))
print(all_features.shape)

# 4.10.4 数据预处理
# 数据中的缺失项补充
# 数字缺失项补充
# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 通过将特征重新缩放到零均值和单位方差来标准化数据
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# 补充NA项
# “Dummy_na=True”将“na”(缺失值)视为有效的特征值，并为其创建指示符特征
# 字符缺失项补充
# 处理离散值。这包括诸如“MSZoning”之类的特征。
# 我们用独热编码替换它们，方法与前面 将多类别标签转换为向量的方式相同
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)

# 将数据转化为张量格式
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)


# 4.10.5 训练
# 损失函数和网络
loss = nn.MSELoss()
in_features = train_features.shape[1]


def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net


# 降低房价误差大对模型的影响，即房价高的房子，预测值和实际值误差肯定比房价低的高，从而导致房价高的房子的权重更高。
# 于是考虑将误差转为百分比表示，真实值-预测值/真实值
def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    # torch.clamp将元素值压缩到1到无穷，这样做log都是正数，计算预测值的log与实际值的log的损失
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()


# 与前面的部分不同，我们的训练函数将借助Adam优化器(我们将在后面章节更详细地描述它)。
# Adam优化器的主要吸引力在于它对初始学习率不那么敏感。
# 训练网络
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# 4.10.6 K折交叉验证
# 交叉验证既可以解决数据集的数据量不够大问题，也可以解决参数调优的问题
def get_k_fold_data(k, i, X, y):
    # i:当前第几折
    # k肯定大于1，小于则报错
    assert k > 1
    # “//”是一个算术运算符，表示整数除法，它可以返回商的整数部分（向下取整）。
    # 不管分为几折和测试数据有多少个，都不会出现小数
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j*fold_size,(j+1)*fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
        # 当前折扣和i相等，则表明到了验证集部分，设定验证集
            X_valid, y_valid = X_part, y_part
        # train是None，说明还未赋值过，则将分割的赋值
        elif X_train is None:
            X_train, y_train = X_part, y_part
        # 否则将train和part合并
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


# 在K折交叉验证中训练次后，返回训练和验证误差的平均值。
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

# 4.10.7 模型选择
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')
d2l.plt.show()


# 4.10.8 提交Kaggle预测
# 提交预测结果
# 将预测保存在CSV文件中可 以简化将结果上传到Kaggle的过程。
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)

d2l.plt.show()
