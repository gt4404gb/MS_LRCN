import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# === 设置字体 ===
ch_font = fm.FontProperties(fname='C:/Windows/Fonts/simsun.ttc', size=20)  # 中文宋体
en_font = fm.FontProperties(fname='C:/Windows/Fonts/times.ttf', size=20)   # 英文字体 Times New Roman

# 原始百分比标签
x_labels = ['1%', '5%', '10%', '20%', '50%']

# 等距的 x 轴位置（索引）
x = np.arange(len(x_labels))  # [0, 1, 2, 3, 4]

# 将原始数据乘以100转换为百分数
lrcn = [x * 100 for x in [0.8811, 0.8853, 0.8912, 0.8982, 0.9065]]
transformer = [x * 100 for x in [0.8702, 0.8588, 0.8682, 0.8887, 0.8978]]
lstm = [x * 100 for x in [0.8554, 0.8439, 0.8622, 0.8728, 0.8914]]
gru = [x * 100 for x in [0.8628, 0.8798, 0.8753, 0.8851, 0.8876]]
bonus = [x * 100 for x in [0.8612, 0.8665, 0.8891, 0.8941, 0.9043]]

# === 创建画布 ===
plt.figure(figsize=(10, 8), dpi=300)

# 绘制五条折线
plt.plot(x, lrcn, color='#1f77b4', marker='o', linewidth=2, markersize=6, label='LRCN')
plt.plot(x, transformer, color='#ff7f0e', marker='s', linewidth=2, markersize=6, label='Transformer')
plt.plot(x, lstm, color='#2ca02c', marker='^', linewidth=2, markersize=6, label='LSTM')
plt.plot(x, gru, color='#d62728', marker='d', linewidth=2, markersize=6, label='GRU')
plt.plot(x, bonus, color='#9467bd', marker='*', linewidth=2, markersize=7, label='BONUS')

# 为每个数据点添加标注（显示两位小数并添加%）
def annotate_points(x_vals, y_vals):
    for xi, yi in zip(x_vals, y_vals):
        plt.text(xi, yi + 0.01, f'{yi:.2f}%', ha='center', va='bottom', fontproperties=en_font, fontsize=14)

annotate_points(x, lrcn)
annotate_points(x, transformer)
annotate_points(x, lstm)
annotate_points(x, gru)
annotate_points(x, bonus)

# 设置 x 轴刻度
plt.xticks(x, x_labels, fontproperties=en_font, fontsize=24)

# === 设置坐标轴标签 ===
plt.xlabel('训练集使用比例', fontproperties=ch_font, fontsize=24)
plt.ylabel('F1分数 (%)', fontproperties=ch_font, fontsize=24)  # 在y轴标签添加%

# === Y轴范围 & 刻度：调整为百分数范围 ===
plt.ylim([84, 91])  # 从84%到91%
yticks = np.arange(85, 92, 1)  # 从85%到91%，步长1%
plt.yticks(yticks, [f'{yt:.0f}%' for yt in yticks], fontproperties=en_font, fontsize=20)

# === 图例 ===
plt.legend(prop=en_font, loc='lower right', fontsize=16)

# === 布局并显示 ===
plt.tight_layout()
plt.show()