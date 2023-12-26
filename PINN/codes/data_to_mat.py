import pandas as pd
from scipy import io

# 创建一个空的DataFrame用于存储合并后的数据

csv_files=['../data/x.csv','../data/t.csv','../data/usol.csv']
#定义字典
data_dict={}
# 遍历CSV文件列表
for file in csv_files:
    # 读取CSV文件
    if file == '../data/x.csv':
        data = pd.read_csv(file)
        data_dict['x'] =data.values

    elif file== '../data/t.csv':
        data = pd.read_csv(file)
        data_dict['t'] = data.values

    elif file== '../data/usol.csv':
        data = pd.read_csv(file)
        data_dict['usol'] = data.values



print(data_dict['x'])
print(data_dict['t'])
#保存为mat文件
io.savemat('../data/output.mat', data_dict)