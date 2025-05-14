import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# 1) 设置字体：中文用宋体、英文用 Times New Roman
# ---------------------------
plt.rcParams['font.sans-serif'] = ['SimSun']  # 全局中文字体为宋体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ---------------------------
# 2) 准备数据
# ---------------------------
models = ["LOF", "IForest", "OCSVM", "MS-LRCN"]
test_time = [371, 21, 9029, 11]

x = np.arange(len(models))   # x 轴索引
bar_width = 0.5              # 柱状图宽度

# ---------------------------
# 3) 创建画布和坐标轴
# ---------------------------
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# 绘制测试耗时柱状图
bars = ax.bar(
    x,
    test_time,
    width=bar_width,
    label="测试耗时（s）",
    color="tab:orange"
)

# ---------------------------
# 4) 设置坐标轴标签、标题和刻度
# ---------------------------
ax.set_xticks(x)
ax.set_xticklabels(models, fontname='Times New Roman', fontsize=20)

ax.set_xlabel("模型", fontsize=22)
ax.set_ylabel("测试耗时（秒）", fontsize=22)

# 根据最大值适当调整 y 轴范围
ax.set_ylim(0, max(test_time)*1.1)

# 调整 y 轴刻度字体大小
ax.tick_params(axis='y', labelsize=16)

# ---------------------------
# 5) 为柱子顶部添加数值标注
# ---------------------------
for i, v in enumerate(test_time):
    # 这里在柱顶基础上增加一个偏移量，使数字显示在柱子上方
    ax.text(
        x[i],
        v + 0.005 * max(test_time),
        f"{v}",
        ha='center',
        va='bottom',
        fontname='Times New Roman',
        fontsize=18
    )

plt.tight_layout()
plt.show()

# 如果需要保存高分辨率图片，可使用：
# plt.savefig("test_time_bar.png", dpi=300, bbox_inches='tight')
