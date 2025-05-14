import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('kaggle_UNSW_NB15_full_testing.csv')  # 替换为你的CSV文件路径

# 分离出分类为6的数据
df_class_6 = df[df.iloc[:, -1] == 6]

# 分离出分类不为6的数据
df_not_class_6 = df[df.iloc[:, -1] != 6]

# 对分类为6的数据随机选择20%保留
df_class_6_reduced = df_class_6.sample(frac=0.1, random_state=42)  # random_state确保可复现的随机结果

# 将减少后的分类为6的数据与分类不为6的数据合并
df_reduced = pd.concat([df_not_class_6, df_class_6_reduced])

# 将处理后的数据保存到新的CSV文件中
df_reduced.to_csv('kaggle_UNSW_NB15_full_edit_testing.csv', index=False)  # 替换为你想要保存的新CSV文件路径