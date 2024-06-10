import os
import pandas as pd
import torch

# 创建包含更多行和列的原始数据集
os.makedirs(os.path.join('../..', 'data'), exist_ok=True)
data_file = os.path.join('../..', 'data', 'house_tiny_practice.cav')
with open(data_file, 'w') as f:
    f.write('number,NumRooms,Alley,Price\n')
    f.write('1,NA,Pave,127500\n')
    f.write('2,2,NA,106000\n')
    f.write('3,4,NA,178100\n')
    f.write('4,NA,NA,14000\n')
    f.write('5,1,NA,10086\n')
    f.write('6,NA,NA,NA\n')

data = pd.read_csv(data_file)
print(data)
# 删除缺失值最多的列
data.isnull(), data.isnull().sum(), data.isnull().sum().idxmax() # 得到缺失值最多列的索引
data = data.drop(data.isnull().sum().idxmax(), axis=1) # 删除
print(data)

# 将与处理后的数据转换为张量格式
data = data.fillna(data.mean())
print(data)
