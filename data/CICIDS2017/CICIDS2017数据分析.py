import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# =============== 1. 字体全局设置 ===============
# 让 matplotlib 默认使用 Times New Roman，并在需要显示中文时回退到 SimSun
matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

# =============== 2. 读取预处理后的 csv 文件 ===============
data = pd.read_csv('cicids2017.csv', header=None)

# 假设最后一列是标签，给所有列设置名称：feature_0, feature_1, ..., Label
num_features = data.shape[1] - 1
feature_names = [f'feature_{i}' for i in range(num_features)] + ['Label']
data.columns = feature_names

# =============== 3. 定义映射字典并统计分布 ===============
label_mapping = {
    0: 'BENIGN',
    1: 'FTP-Patator',
    2: 'SSH-Patator',
    3: 'DoS slowloris',
    4: 'DoS Slowhttptest',
    5: 'DoS Hulk',
    6: 'DoS GoldenEye',
    7: 'Heartbleed',
    8: 'Infiltration',
    9: 'Web Attack – Brute Force',
    10: 'Web Attack – XSS',
    11: 'Web Attack – Sql Injection',
    12: 'DDoS',
    13: 'PortScan',
    14: 'Bot'
}

label_counts = data['Label'].value_counts().sort_index()
mapped_labels = label_counts.index.map(lambda x: label_mapping.get(x, str(x)))

# =============== 4. 绘制水平柱状图 ===============
plt.figure(figsize=(8, 6), dpi=300)
bars = plt.barh(mapped_labels, label_counts.values)
plt.xlabel("标签计数", fontsize=14)
# 如果标题中含有中文，matplotlib 会自动回退到 SimSun 显示中文。

# 让 x 轴采用对数刻度（可根据需要注释/取消）
plt.xscale('log')

plt.tight_layout()

# =============== 5. 防止数字溢出：在柱状图上添加计数并动态判断位置 ===============
for bar in bars:
    width = bar.get_width()  # 每个柱子的宽度(即数值大小)
    label_y = bar.get_y() + bar.get_height() / 2

    # 如果柱子非常长，就把文字放到柱子内部(避免溢出)
    # 这里的阈值可根据实际情况调整，比如 width > 10 万 或者更大
    if width > 1e6:
        plt.text(width * 0.8, label_y, f'{int(width)}',
                 va='center', ha='right', color='white', fontsize=10)
    else:
        # 否则正常放在柱子右侧
        plt.text(width, label_y, f'{int(width)}',
                 va='center', ha='left', fontsize=10)

plt.show()
