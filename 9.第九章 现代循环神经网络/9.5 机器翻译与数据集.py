# 9.5 机器翻译与数据集
# 下面，我们看一下如何将预处理 后的数据加载到小批量中用于训练。
import os
import torch
from d2l import torch as d2l

# 9.5.1 下载和预处理数据集
# 首先，下载一个由Tatoeba项目的双语句子对113组成的“英-法”数据集，
# 数据集中的每一行都是制表符分隔的文本序列对，
# 序列对由英文文本序列和翻译后的法语文本序列组成。
# 请注意，每个文本序列可以是一个句子，
# 也可以是包含多个句子的一个段落。
# 在这个将英语翻译成法语的机器翻译问题中，
# 英语是源语言(source language)，
# 法语是目标语言(target language)。

d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')


def read_data_nmt():
    """载入“英语－法语”数据集"""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
             encoding='utf-8') as f:
        return f.read()


raw_text = read_data_nmt()
print(raw_text[:75])


# 下载数据集后，原始文本数据需要经过几个预处理步骤。
# 例如，我们用空格代替不间断空格(non‐breaking space)，
# 使用小写字母替换大写字母，
# 并在单词和标点符号之间插入空格。
def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


text = preprocess_nmt(raw_text)
print(text[:80])


# 9.5.2 词元化
# 之前都是字符词元化，现在是单词词元化。
# 每个词元要不是个单词，要不是个符号。
# 此函数返回两个词元列表：source和target：
# source[i]是源语言（这里是英语）第 i 个文本序列的词元列表，
# target[i]是目标语言（这里是法语）第 i 个文本序列的词元列表。
def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


source, target = tokenize_nmt(text)
print(source[:6], target[:6])


# 让我们绘制每个文本序列所包含的词元数量的直方图。
# 在这个简单的“英-法”数据集中，
# 大多数文本序列的词元数量少于20个。
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)


show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', source, target)

d2l.plt.show()

# 9.5.3 词表
# 这里我们将出现次数少于2次的低频率词元视为相同的未知(“<unk>”)词元。
# 除此之外，我们还指定了额外的特定词元，
# 例如在小批量时用于将序列填充到相同长度的填充词元(“<pad>”)，
# 以及序列的开始词元(“<bos>”)和结束词元(“<eos>”)。
# 这些特殊词元在自然语言处理任务中比较常用。
src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
print(len(src_vocab))


# 9.5.4 加载数据集
# 为了相同的长度，我们要对序列样本有一个固定的长度。通过截断(truncation)和填充(padding)
# 截断：只取其前num_steps 个词元，并且丢弃剩余的词元。
# 填充：文本序列的词元数目少于num_steps时，我们将继续在其末尾添加特定的（<pad>）词元。
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


print(truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>']))


# 转换成小批量输入数据
# 将特定的（<eos>）词元添加到所有序列的末尾，用于表示序列的结束。
def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


# 9.5.5 训练模型
# 定义迭代器
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


# 尝试读出这个小批量数据
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('X的有效长度:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('Y的有效长度:', Y_valid_len)
    break

# 小结
# (continued from previous page)
# • 机器翻译指的是将文本序列从一种语言自动翻译成另一种语言。
# • 使用单词级词元化时的词表大小，将明显大于使用字符级词元化时的词表大小。
# 为了缓解这一问题，我们可以将低频词元视为相同的未知词元。
# • 通过截断和填充文本序列，可以保证所有的文本序列都具有相同的长度，以便以小批量的方式加载。
