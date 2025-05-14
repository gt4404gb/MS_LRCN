import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split


def LRCCNdataload(proportion=0.9,device='cpu',batch_size=1024):
    # 随机划分训练集，取50%作为训练集

    #——————————————————————————数据集加载————————————————————————————————
    x_train = pd.read_csv("../data/UNSW-NB15/kaggle_UNSW_NB15_full_training.csv", low_memory=False)
    num_cols = x_train.shape[1] #获取列数
    #提取出最后一列为y
    y_train = x_train.pop(x_train.columns[-1]).values

    x_test = pd.read_csv("../data/UNSW-NB15/kaggle_UNSW_NB15_full_edit_testing.csv", low_memory=False)
    num_cols = x_test.shape[1] #获取列数
    y_test = x_test.pop(x_test.columns[-1]).values

    # 将读取的数据转化为np格式方便后续训练
    x_train = np.array(x_train, dtype=np.float32) # 将数据转换为float32类型
    y_train = np.array(y_train, dtype=np.int64) # 将数据转换为int64类型
    x_test = np.array(x_test, dtype=np.float32) # 将数据转换为float32类型
    y_test = np.array(y_test, dtype=np.int64) # 将数据转换为int64类型


    X_train, _, Y_train, _ = train_test_split(x_train, y_train, train_size=proportion, random_state=42)
    #在这里，我们将训练集分为两部分，一部分用于训练模型，另一部分用于训练标签编码器，比例由train_size参数控制
    x_train, x_train2, y_train, y_train2 = train_test_split(X_train, Y_train, train_size=0.9, random_state=42)

    # 定义未知类别和映射
    unknown_classes = [0, 1, 8, 9]
    known_classes = sorted(set(range(10)) - set(unknown_classes))

    mapping_dict = {orig_label: new_label for new_label, orig_label in enumerate(known_classes)}

    # 筛选train2，删除未知类别
    filtered_indices = ~np.isin(y_train2, unknown_classes)
    x_train2 = x_train2[filtered_indices]
    y_train2 = y_train2[filtered_indices]

    # 对y_train2进行映射
    y_train2 = np.vectorize(mapping_dict.get)(y_train2)

    print(x_train.shape, x_train2.shape, x_test.shape)

    x_train = torch.from_numpy(x_train).float().to(device)
    y_train = torch.from_numpy(y_train).long().to(device)
    x_train2 = torch.from_numpy(x_train2).float().to(device)
    y_train2 = torch.from_numpy(y_train2).long().to(device)
    x_test = torch.from_numpy(x_test).float().to(device)
    y_test = torch.from_numpy(y_test).long().to(device)

    # 测试集中的类别0,1,8,9设为未知(-1)，其余进行映射
    y_test_mapped = torch.full_like(y_test, -1)
    for orig, new in mapping_dict.items():
        y_test_mapped[y_test == orig] = new

    # 数据加载器
    train_dataset = TensorDataset(x_train, y_train)
    train_dataset2 = TensorDataset(x_train2, y_train2)

    # 输出映射后的数据分布
    y_values = [label.item() for _, label in train_dataset2]
    y_distribution = pd.Series(y_values).value_counts()
    print("映射后的带标签数据集数据分布：")
    print(y_distribution)

    test_dataset = TensorDataset(x_test, y_test_mapped)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader2 = DataLoader(train_dataset2, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, train_loader2, test_loader


#-------------------------------------------