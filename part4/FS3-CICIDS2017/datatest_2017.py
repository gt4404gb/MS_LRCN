import torch
import torch.nn.functional as F
import time
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
import umap
import copy
import matplotlib.pyplot as plt

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
    x = pd.read_csv("../data/CICIDS2017/cicids2017.csv", low_memory=False)
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


    # 对测试集标签做映射：对于13,14，映射为 -1，其余按规则映射
    y_test = map_labels(y_test, is_test=True)

    print("测试集形状：", x_test.shape)

    x_test = torch.from_numpy(x_test).float().to(device)
    y_test = torch.from_numpy(y_test).long().to(device)

    test_dataset = TensorDataset(x_test, y_test)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader


def sample_data(X, y, max_samples_per_class=1000):
    """
    利用 Pandas 的 groupby 从每个类别中采样最多 max_samples_per_class 个样本
    """
    # 构造 DataFrame，其中一列为标签
    df = pd.DataFrame(X)
    df['label'] = y
    # 每个类别采样
    df_sampled = df.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), max_samples_per_class)))
    # 分离标签和特征
    y_sampled = df_sampled.pop('label').values
    X_sampled = df_sampled.values
    return X_sampled, y_sampled


def drawplt(y_hat_total, y_true, title="Unamed"):
    """
    绘制UMAP降维后的散点图，并使用中文标签和指定字体显示图例。

    Args:
        y_hat_total (np.ndarray): 预测结果或特征向量 (N, Features)。
        y_true (np.ndarray): 真实的类别标签 (N,)。应包含映射后的大类标签 (0-4)。
        title (str): 图表的标题。
    """
    # 类别标签到中文名称的映射
    grouped_label_to_chinese = {
        -1: '异常（OOD样本）',
        0: '正常',
        1: '拒绝服务攻击',
        2: 'Web攻击',
        3: '暴力破解攻击',
        4: '其他攻击'
        # 你可以根据实际情况添加更多映射
    }

    # --- 字体设置 ---
    # 设置宋体为中文备选字体
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 全局中文字体为宋体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # --- 字体设置结束 ---

    # 采样
    # 确保 y_true 中的标签是映射后的标签 (0, 1, 2, 3, 4)
    # 如果 y_true 包含的是原始标签 (0-14)，你需要先进行映射
    # 示例映射过程（如果需要的话）：
    # mapping_dict_provided = { ... } # 你提供的原始到大类的映射
    # y_true_grouped = np.array([mapping_dict_provided.get(label, 4) for label in y_true]) # 假设未映射的归为'其他'(4)
    # y_pred_sampled, y_true_sampled = sample_data(y_hat_total, y_true_grouped, max_samples_per_class=200)

    # 假设 y_true 已经是分组后的标签 (0-4)
    y_pred_sampled, y_true_sampled = sample_data(y_hat_total, y_true, max_samples_per_class=200)
    print("Sampled data shape:", y_pred_sampled.shape, y_true_sampled.shape)

    # 检查采样后的标签是否都在映射中
    unique_sampled_labels = np.unique(y_true_sampled)
    for label in unique_sampled_labels:
        if label not in grouped_label_to_chinese:
            print(
                f"警告: 采样数据中包含标签 {label}, 但该标签未在 grouped_label_to_chinese 中定义。图例可能不完整或出错。")

    # UMAP 降维
    reducer = umap.UMAP(n_components=5,n_neighbors=5, min_dist=0.8, random_state=42)
    y_pred_umap = reducer.fit_transform(y_pred_sampled)

    # 确保类别与颜色映射一致
    # unique_labels = np.unique(y_true_sampled) # 现在从采样后的数据获取唯一标签
    cmap = plt.get_cmap('tab10')  # 使用 'tab10' 颜色映射
    # 创建颜色映射时，确保只为存在的标签创建条目
    label_to_color = {label: cmap(i / len(unique_sampled_labels)) for i, label in enumerate(unique_sampled_labels)}

    # 绘制散点图
    colors = [label_to_color[label] for label in y_true_sampled]
    scatter = plt.scatter(
        y_pred_umap[:, 0],
        y_pred_umap[:, 1],
        c=colors,
        s=8,  # 点大小
        alpha=0.7
    )

    # 创建类别图示框 (使用中文标签)
    handles = []
    # 按标签顺序（例如 0, 1, 2, 3, 4）创建图例，保证顺序一致性
    sorted_labels = sorted(label_to_color.keys())
    for label in sorted_labels:
        color = label_to_color[label]
        chinese_label = grouped_label_to_chinese.get(label, f"未知类别 {label}")  # 获取中文名，如果找不到则显示未知
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label=chinese_label,
                                  markerfacecolor=color, markersize=8))

    # 添加图例，标题设为 "类别"
    plt.legend(handles=handles, title="类别", loc='best')  # 将 title 改为中文

    # 设置标题和标签
    plt.title(title)  # 标题字体会根据 rcParams 设置
    plt.xlabel("UMAP 1")  # X轴标签
    plt.ylabel("UMAP 2")  # Y轴标签
    plt.tight_layout()  # 调整布局防止重叠

    # 保存图像
    # 确保目录存在
    save_dir_path = "..\\result\\"
    import os
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    save_path = os.path.join(save_dir_path, title + "_" + str(time.time()) + ".png")
    # 替换文件名中不安全/不支持的字符（例如，如果标题包含特殊字符）
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_', '-')).rstrip()
    save_path = os.path.join(save_dir_path, safe_title + "_" + str(time.time()) + ".png")

    print(f"Saving plot to: {save_path}")
    plt.savefig(save_path, dpi=300)  # 提高保存图片的分辨率
    plt.show()

# 使用示例
if __name__ == '__main__':
    test_loader = LRCCNdataload(proportion=0.9, device='cpu', batch_size=1024)

    # 从 test_loader 中提取所有数据和标签
    all_features = []
    all_labels = []
    for features, labels in test_loader:
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 调用绘图函数，直接对原始测试数据进行降维与绘制
    drawplt(all_features, all_labels, title="原始测试集分布")
