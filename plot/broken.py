import matplotlib.pyplot as plt
import numpy as np
from sympy.simplify.simplify import bottom_up

# 数据
# data = [4421.711, 4421.639, 4418.638, 8498.72, 320867.95]
# data1 = [6621.711, 6621.639, 6618.638, 10930.77, 335236.15]

data = [4418.638, 8498.72, 320867.95]
data1 = [2200, 2432.05, 14368.2]

# labels = ['7B-ARC-C', '7B-ARC-E', '7B-SQuAD', '13B-Human', '70B-LEval']

labels = ['7B-SQuAD', '13B-Human', '70B-LEval']

# 创建画布
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 4]})
plt.xticks(fontsize=20)

# 绘制第一部分柱状图（断点以上部分）
# ax1.bar(labels, data, color=['green', 'green', 'green', 'blue', 'red'])
# ax1.bar([0, 1, 2], data[0:3], width=0.5, color='green', hatch="\\", edgecolor='w')
# ax1.bar([3], data[3], width=0.5, color='blue', hatch=".", edgecolor='w')
# ax1.bar([4], data[4], width=0.5, color='red', hatch="/", edgecolor='w')
x = [1.8 * i for i in range(len(labels))]
x_ = [i + 0.2 for i in x]
ax1.bar(x, data, width=0.5, color='red', hatch="/", edgecolor='w')
ax1.bar(x, data1, bottom=data, width=0.5, color='green', hatch="\\", edgecolor='w')

ax1.set_ylim(300000, 340000)  # 设置第一部分的 y 轴范围
ax1.spines['bottom'].set_visible(False)  # 隐藏底部边框
ax1.xaxis.tick_top()  # 将 x 轴刻度放在顶部
ax1.set_xticks([i + 0.1 for i in x], labels, fontproperties='Times New Roman', size=20, weight='bold')
ax1.tick_params(labeltop=False, axis='y', labelsize=20)  # 不显示顶部的 x 轴标签

# 绘制第二部分柱状图（断点以下部分）
# ax2.bar(labels, data, color=['green', 'green', 'green', 'blue', 'red'])
# ax2.bar(labels[0:3], data[0:3], width=0.5, color='green', hatch="\\", edgecolor='w')
# ax2.bar(labels[3], data[3], width=0.5, color='blue', hatch=".", edgecolor='w')
# ax2.bar(labels[4], data[4], width=0.5, color='red', hatch="/", edgecolor='w')

ax2.bar(x, data, width=0.5, color='red', hatch="/", edgecolor='w', label='Disk to CPU')
ax2.bar(x, data1, bottom=data, width=0.5, color='green', hatch="\\", edgecolor='w', label='CPU to GPU')


ax2.set_ylim(0, 11000)  # 设置第二部分的 y 轴范围
ax2.spines['top'].set_visible(False)  # 隐藏顶部边框
ax2.tick_params(labeltop=False, axis='y', labelsize=20)  # 不显示顶部的 x 轴标签

# 添加断点斜线
d = 0.015  # 斜线长度
# kwargs = dict(marker=[(-1, -d), (1, d)], markersize=5, linestyle='none', color='k', mec='k', mew=1, clip_on=False)
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)

ax1.plot((-d, +d), (-d, +d), **kwargs)        # 左下角斜线
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # 右下角斜线

kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # 左上角斜线
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # 右上角斜线

# 添加网格线
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# 设置标题和标签
# ax2.set_xlabel('Model & Dataset',size=20,alpha=0.8)
ax2.set_ylabel('Latency',size=20,alpha=0.8)
# ax1.set_ylabel('Values')

# 调整布局
plt.tight_layout()
plt.legend(loc='upper left', fontsize=20)
# 显示图像
# plt.show()
# plt.savefig('./pics/mutil-model-load-time.png', format='png')
plt.savefig('pics/mutil-model-load-time.pdf', format='pdf', bbox_inches='tight', pad_inches=0.26)