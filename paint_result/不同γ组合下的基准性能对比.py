import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# 1) 设置字体
# ---------------------------
plt.rcParams['font.sans-serif'] = ['SimSun']  # 全局中文使用宋体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ---------------------------
# 2) 数据准备
# ---------------------------
params = [
    "[0.1,\n0.1,\n0.1]",
    "[0.3,\n0.3,\n0.3]",
    "[0.5,\n0.5,\n0.5]",
    "[0.7,\n0.7,\n0.7]",
    "[0.9,\n0.9,\n0.9]",
    "[0.1,\n0.3,\n0.5]",
    "[0.3,\n0.5,\n0.7]",
    "[0.1,\n0.5,\n0.9]"
]

# 将原始数据乘以100转换为百分数
accuracy = [x * 100 for x in [0.8926, 0.89, 0.8839, 0.8917, 0.8909, 0.8968, 0.8938, 0.8813]]
f1_score = [x * 100 for x in [0.8836, 0.8816, 0.8724, 0.8748, 0.8777, 0.8853, 0.8848, 0.8649]]

x = np.arange(len(params))  # x 轴刻度位置
bar_width = 0.3            # 柱状图宽度

# ---------------------------
# 3) 创建画布并绘图
# ---------------------------
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

# 绘制“准确率”柱子
bars_acc = ax.bar(
    x - bar_width/2,
    accuracy,
    width=bar_width,
    color='tab:blue',
    label='准确率'
)

# 绘制“F1分数”柱子
bars_f1 = ax.bar(
    x + bar_width/2,
    f1_score,
    width=bar_width,
    color='orange',
    label='F1分数'
)

# ---------------------------
# 4) 设置坐标轴标签、标题、刻度等
# ---------------------------
ax.set_xticks(x)
ax.set_xticklabels(params, rotation=0, ha='center', fontname='Times New Roman', fontsize=14)
ax.set_ylabel("F1分数（%）", fontname='SimSun', fontsize=16)     # 在y轴标签添加%
ax.set_xlabel("γ参数组", fontname='SimSun', fontsize=16)

# 调整y轴范围为百分数
plt.ylim([85, 90])
yticks = np.arange(85, 91, 1)  # 从85%到90%，步长1%
plt.yticks(yticks, [f'{yt:.0f}%' for yt in yticks], fontproperties='Times New Roman', fontsize=14)

# ---------------------------
# 5) 为柱状图添加数值标注
# ---------------------------
for rect in bars_acc:
    height = rect.get_height()
    ax.text(
        rect.get_x() + rect.get_width()/2,
        height + 0.1,  # 调整标注位置
        f"{height:.2f}%",  # 显示两位小数并添加%
        ha='center', va='bottom',
        fontname='Times New Roman', fontsize=12
    )

for rect in bars_f1:
    height = rect.get_height()
    ax.text(
        rect.get_x() + rect.get_width()/2 + 0.13,
        height + 0.1,  # 调整标注位置
        f"{height:.2f}%",  # 显示两位小数并添加%
        ha='center', va='bottom',
        fontname='Times New Roman', fontsize=12
    )

# ---------------------------
# 6) 将图例放在图外的底部
# ---------------------------
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.18),
    ncol=2,
    prop={'family': 'SimSun', 'size': 16}
)

# 调整子图边距
plt.subplots_adjust(bottom=0.2)

# ---------------------------
# 7) 布局和显示
# ---------------------------
plt.show()

# 如果需要保存：
# plt.savefig("bar_chart_percentage.png", dpi=300, bbox_inches='tight')