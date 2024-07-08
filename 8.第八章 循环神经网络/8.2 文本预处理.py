# 8.2 文本预处理
# 本节中，我们将解析文本的常见预处理步骤。这些步骤通常包括:
# 1. 将文本作为字符串加载到内存中。
# 2. 将字符串拆分为词元(如单词和字符)。
# 3. 建立一个词表，将拆分的词元映射到数字索引。
# 4. 将文本转换为数字索引序列，方便模型操作。
import collections
import re
from d2l import torch as d2l

# 8.2.1 读取数据集
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')


# 定义读取数据集的函数
def read_time_machine():
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()

    # 返回一个句子列表，只保留大小写字母信息，去除句子两旁的空格，并全部转化为小写
    return [re.sub('[^A-Za-z]', ' ', line).strip().lower() for line in lines]


lines = read_time_machine()  # 获取一个文本列表，其中每一个元素代表一个句子
print(f'# 文本总行数: {len(lines)}')  # 输出文本总行数
print(lines[0])
print(lines[10])


# 8.2.2 词元化
# 文本数据词元化
def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误: 未知词元类型: ' + token)


tokens = tokenize(lines)  # 把文本数据转化为token词元类型

for i in range(11):  # 打印前十一行单词
    print(tokens[i])

# 8.2.3 词表
# 文本数据词元化
def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误: 未知词元类型: ' + token)


tokens = tokenize(lines)  # 把文本数据转化为token词元类型

for i in range(11):  # 打印前11行单词
    print(tokens[i])
print('\n')

# 8.2.3 词表
# 词典类
class Vocab:
    """
    文本词表，其中三个参数，
    tokes表示待统计的词元列表，
    reserved_tokens表示特殊标记的词(如未知词元)，
    min_freq表示最低出现的频率
    """

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):

        if tokens is None:
            tokens = []

        if reserved_tokens is None:
            reserved_tokens = []

        # 按照出现的概率进行排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens  # 初始化id对token和token对id
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        # 通过循环依次添加token和id，分别添加至idx_to_token列表中和token_to_idx字典中
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)  # 追加到idx_to_token列表中
                self.token_to_idx[token] = len(self.idx_to_token) - 1  # 追加到token_to_idx字典之中

    def __len__(self):
        return len(self.idx_to_token)  # 返回字典的长度

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):  # 若不是列表或者元组，则返回单个下标
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]  # 若是列表或者元组，返回列表形式的下标

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]  # 若不是列表或者元组，返回单个token
        return [self.idx_to_token[index] for index in indices]  # 若是列表或元组，则返回多个token

    # 代表变量
    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


# 统计corpus语料
def count_corpus(tokens):
    """统计词元的频率"""
    # 这里的tokens是1D列表或者2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]

    return collections.Counter(tokens)


vocab = Vocab(tokens)   # 生成字典对象
print(list(vocab.token_to_idx.items())[:10])    # 生成前10个出现频率最高的词

# 将每一条文本行转换成一个数字索引列表
for i in [0, 10]:
    print('文本:', tokens[i])
    print('索引:', vocab[tokens[i]])


# 8.2.4 整合所有功能
# 加载machine数据集，并返回corpus词元索引列表和vocab词表
def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)

    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]

    if max_tokens > 0:
        corpus = corpus[:max_tokens]

    return corpus, vocab  # 返回corpus语料库与vocab字典


corpus, vocab = load_corpus_time_machine()
print(len(corpus), len(vocab))

# 小结
# • 文本是序列数据的一种最常见的形式之一。
# • 为了对文本进行预处理，我们通常将文本拆分为词元，构建词表将词元字符串映射为数字索引，
# 并将文本数据转换为词元索引以供模型操作。

