import numpy as np

def normalize_aaindex(data):
    # 计算属性的均值和标准差
    mean = np.mean(data, axis=0)

    std = np.std(data, axis=0)
    normalized_data = (data - mean) / std

    return normalized_data

# 假设你有一个包含多个AAindex属性的数据集
aaindex_data = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])

# 对AAindex数据进行正则化处理
normalized_data = normalize_aaindex(aaindex_data)

# 输出正则化后的数据
print("Normalized data:")
print(normalized_data)