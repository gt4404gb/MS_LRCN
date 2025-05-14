import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# 1) 设置字体：中文用宋体、英文用 Times New Roman
# ---------------------------
plt.rcParams['font.sans-serif'] = ['SimSun']  # 全局中文使用宋体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ---------------------------
# 2) 准备数据
# ---------------------------
models = ["Transformer", "SimCLR", "VAE", "MS-LRCN"]
train_time = [4836, 2133, 1910, 4116]

x = np.arange(len(models))   # x 轴索引
bar_width = 0.5              # 柱状图宽度

# ---------------------------
# 3) 创建画布并绘图
# ---------------------------
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# 绘制训练耗时柱状图
bars = ax.bar(
    x,
    train_time,
    width=bar_width,
    color="tab:blue"
)

# ---------------------------
# 4) 设置坐标轴标签、标题和刻度
# ---------------------------
ax.set_xticks(x)
ax.set_xticklabels(models, fontname='Times New Roman', fontsize=20)

ax.set_xlabel("模型", fontsize=22)
ax.set_ylabel("训练耗时（秒）", fontsize=22)

# 根据最大值适当调整 y 轴范围，让柱子更美观
ax.set_ylim(0, max(train_time) * 1.1)

# 调整 y 轴刻度字体大小
ax.tick_params(axis='y', labelsize=16)

# ---------------------------
# 5) 在柱子顶部添加数值标注
# ---------------------------
for i, v in enumerate(train_time):
    # 在每个柱子的顶部显示具体耗时数值，稍作偏移以免与柱子重叠
    ax.text(
        x[i],
        v + 0.005 * max(train_time),
        f"{v}",
        ha='center',
        va='bottom',
        fontname='Times New Roman',
        fontsize=18
    )

# ---------------------------
# 6) 布局和显示
# ---------------------------
plt.tight_layout()
plt.show()

# 如果需要保存高分辨率图片，可使用：
# plt.savefig("training_time_bar.png", dpi=300, bbox_inches='tight')
