import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# 1) 设置字体：中文用宋体、英文用 Times New Roman
# ---------------------------
plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置全局中文字体为宋体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号无法显示问题

# ---------------------------
# 2) 准备示例数据
# ---------------------------
methods = ["GRU", "LSTM", "Transformer", "BONUS", "LRCN"]
training_time = [6166.66, 6146.42, 4982.09, 7170.23, 3841.12]
testing_time = [87.02, 89.43, 38.45,57.57, 29.27]

# 将 x 轴设置为方法的索引
x = np.arange(len(methods))
# 设置柱状图的宽度
bar_width = 0.4

# ---------------------------
# 3) 创建画布和坐标轴
# ---------------------------
fig, ax1 = plt.subplots(figsize=(10, 8), dpi=300)  # dpi=300 使图像更清晰

# ---------------------------
# 4) 绘制左侧柱状图（训练时间）
# ---------------------------
bar1 = ax1.bar(
    x - bar_width/2,             # 柱子在 x 轴上的位置
    training_time,
    width=bar_width,
    label='训练时间',
    color='tab:blue'
)

ax1.set_ylim(0, max(training_time) +1000)  # 10 times the max to spread the numbers out
# 设置左 y 轴标签及刻度颜色（英文用 Times New Roman）
ax1.set_ylabel("训练时间（秒）", color='tab:blue', fontsize=22)
ax1.tick_params(axis='y', labelcolor='tab:blue')


# ---------------------------
# 5) 在同一张图上添加右侧坐标轴（测试时间）
# ---------------------------
ax2 = ax1.twinx()
bar2 = ax2.bar(
    x + bar_width/2,
    testing_time,
    width=bar_width,
    label='测试时间',
    color='tab:orange'
)

ax2.set_ylim(0, max(testing_time) * 3)  # 10 times the max to spread the numbers out

# 设置右 y 轴标签及刻度颜色（英文用 Times New Roman）
ax2.set_ylabel("测试时间（秒）", color='tab:orange', fontsize=22)
ax2.tick_params(axis='y', labelcolor='tab:orange')

# ---------------------------
# 6) 设置 x 轴刻度标签（英文用 Times New Roman）
# ---------------------------
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontname='Times New Roman', fontsize=20)

# 如果需要中文标题，可使用宋体
ax1.set_xlabel("模型", fontsize=22)

# ---------------------------
# 7) 为柱子顶部添加数值标注（可选）
# ---------------------------
# 左侧柱子
for i, v in enumerate(training_time):
    ax1.text(x[i] - bar_width/2, v + 50, f"{v:.2f}",
             ha='center', va='bottom',
             fontname='Times New Roman', fontsize=18)

# 右侧柱子
for i, v in enumerate(testing_time):
    ax2.text(x[i] + bar_width/2, v + 5, f"{v:.2f}",
             ha='center', va='bottom',
             fontname='Times New Roman', fontsize=18)

# 调整 y 轴刻度标签的字体大小
ax1.tick_params(axis='y', labelsize=16)  # 左侧 y 轴
ax2.tick_params(axis='y', labelsize=16)  # 右侧 y 轴

# ---------------------------
# 8) 调整布局并显示
# ---------------------------
plt.tight_layout()
plt.show()

# 如果需要保存为高分辨率图片：
# plt.savefig("double_axis_bar.png", dpi=300, bbox_inches='tight')
