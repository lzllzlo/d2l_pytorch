import os
import pandas as pd

# 2.2.1 读取数据集
os.makedirs(os.path.join('../..', 'data'), exist_ok=True)
data_file = os.path.join('../..', 'data', 'house_tiny.cav')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,14000\n')

data = pd.read_csv(data_file)
print(data)

# 2.2.2 处理缺失值
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean(numeric_only=True))
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

