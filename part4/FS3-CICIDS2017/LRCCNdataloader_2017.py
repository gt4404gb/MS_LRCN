import torch
import torch.nn.functional as F
import time
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split

# 原始映射: {0: 'BENIGN', 1: 'FTP-Patator', 2: 'SSH-Patator', 3: 'DoS slowloris',
    # 4: 'DoS Slowhttptest', 5: 'DoS Hulk', 6: 'DoS GoldenEye', 7: 'Heartbleed',
    # 8: 'Infiltration', 9: 'Web Attack – Brute Force', 10: 'Web Attack – XSS',
    # 11: 'Web Attack – Sql Injection', 12: 'DDoS', 13: 'PortScan', 14: 'Bot'}
# 定义原始到大类标签的映射（不包括13和14，后面单独处理）
mapping_dict = {
    0: 0,  # BENIGN -> 0
    1: 3,  # FTP-Patator -> 暴力破解 -> 3
    2: 3,  # SSH-Patator -> 暴力破解 -> 3
    3: 1,  # DoS slowloris -> DoS -> 1
    4: 1,  # DoS Slowhttptest -> DoS -> 1
    5: 1,  # DoS Hulk -> DoS -> 1
    6: 1,  # DoS GoldenEye -> DoS -> 1
    7: 4,  # Heartbleed -> 其他 -> 4
    8: 4,  # Infiltration -> 其他 -> 4
    9: 2,  # Web Attack – Brute Force -> Web 攻击 -> 2
    10: 2,  # Web Attack – XSS -> Web 攻击 -> 2
    11: 2,  # Web Attack – Sql Injection -> Web 攻击 -> 2
    12: 1,  # DDoS -> DoS -> 1
}

def map_labels(y, is_test=False):
    """
    对标签进行映射：
    - 对于测试集，当标签为13或14时，返回 -1 表示未知类别；
    - 对于训练集，则不保留13和14的样本。
    - 其他类别按照 mapping_dict 进行映射。
    """
    mapped = []
    for label in y:
        if label in [13, 14]:
            # 测试集中将13和14映射为-1，训练集中后续会过滤掉
            mapped.append(-1 if is_test else label)
        else:
            mapped.append(mapping_dict[label])
    return np.array(mapped)

def LRCCNdataload(proportion=0.9, device='cpu', batch_size=1024):
    # ————————————————————————————数据集加载————————————————————————————————
    x = pd.read_csv("../../data/CICIDS2017/cicids2017.csv", low_memory=False)
    num_cols = x.shape[1]  # 获取列数
    # 提取出最后一列为 y，并转换为 numpy 数组
    y = x.pop(x.columns[-1]).values

    # 设置使用数据集比例
    use_proportion = 0.01 * proportion
    # 随机划分，比例为80%的训练数据和20%的测试数据
    x_train, x_test, y_train, y_test = train_test_split(
        x.values, y, train_size=80 * use_proportion,
        test_size=20 * use_proportion, random_state=42
    )

    # 对训练集进一步划分，得到带全部数据的训练集和部分“带标签”的子集
    x_train, x_train2, y_train, y_train2 = train_test_split(
        x_train, y_train, train_size=0.7, random_state=42
    )

    # 对训练集数据：首先剔除原始标签为 13 和 14 的样本（PortScan和Bot）
    mask_train = (y_train != 13) & (y_train != 14)
    x_train = x_train[mask_train]
    y_train = y_train[mask_train]

    mask_train2 = (y_train2 != 13) & (y_train2 != 14)
    x_train2 = x_train2[mask_train2]
    y_train2 = y_train2[mask_train2]

    # 对训练集标签做映射（训练集中13,14已剔除）
    y_train = map_labels(y_train, is_test=False)
    y_train2 = map_labels(y_train2, is_test=False)

    # 对测试集标签做映射：对于13,14，映射为 -1，其余按规则映射
    y_test = map_labels(y_test, is_test=True)

    print("训练集整体形状：", x_train.shape)
    print("带标签子集形状：", x_train2.shape)
    print("测试集形状：", x_test.shape)

    # 如有需要，可以统计并打印训练集中带标签的数据分布
    if True:
        y_values = [int(label) for label in y_train2]
        y_distribution = pd.Series(y_values).value_counts().sort_index()
        print("带标签数据集的数据分布：")
        print(y_distribution)

    # 转换为 PyTorch 张量并移动到指定设备
    x_train = torch.from_numpy(x_train).float().to(device)
    y_train = torch.from_numpy(y_train).long().to(device)
    x_train2 = torch.from_numpy(x_train2).float().to(device)
    y_train2 = torch.from_numpy(y_train2).long().to(device)
    x_test = torch.from_numpy(x_test).float().to(device)
    y_test = torch.from_numpy(y_test).long().to(device)

    # 定义数据加载器
    train_dataset = TensorDataset(x_train, y_train)
    train_dataset2 = TensorDataset(x_train2, y_train2)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader2 = DataLoader(train_dataset2, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, train_loader2, test_loader

# 使用示例
if __name__ == '__main__':
    train_loader, train_loader2, test_loader = LRCCNdataload(proportion=0.9, device='cpu', batch_size=1024)