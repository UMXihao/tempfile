import matplotlib.pyplot as plt
import numpy as np

# 数据
data1 = [6621.711, 6621.639, 6618.638, 10930.77, 335236.15]
data2 = [21.71, 21.6, 18, 333, 526]
labels = ['A', 'B', 'C', 'D', 'E']

# 创建画布
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [1, 4, 1]})

# 绘制第一部分柱状图（断点以上部分）
ax1.bar(labels, data1, color=['blue', 'green', 'red', 'purple', 'orange'], label='Data 1')
ax1.bar(labels, data2, color=['cyan', 'lime', 'pink', 'violet', 'yellow'], label='Data 2', alpha=0.7)
ax1.set_ylim(33000, 340000)  # 设置第一部分的 y 轴范围
ax1.spines['bottom'].set_visible(False)  # 隐藏底部边框
ax1.xaxis.tick_top()  # 将 x 轴刻度放在顶部
ax1.tick_params(labeltop=False)  # 不显示顶部的 x 轴标签
ax1.legend(loc='upper right')

# 绘制第二部分柱状图（中间部分）
ax2.bar(labels, data1, color=['blue', 'green', 'red', 'purple', 'orange'], label='Data 1')
ax2.bar(labels, data2, color=['cyan', 'lime', 'pink', 'violet', 'yellow'], label='Data 2', alpha=0.7)
ax2.set_ylim(11000, 33000)  # 设置第二部分的 y 轴范围
ax2.spines['top'].set_visible(False)  # 隐藏顶部边框
ax2.spines['bottom'].set_visible(False)  # 隐藏底部边框
ax2.legend(loc='upper right')

# 绘制第三部分柱状图（断点以下部分）
ax3.bar(labels, data1, color=['blue', 'green', 'red', 'purple', 'orange'], label='Data 1')
ax3.bar(labels, data2, color=['cyan', 'lime', 'pink', 'violet', 'yellow'], label='Data 2', alpha=0.7)
ax3.set_ylim(0, 11000)  # 设置第三部分的 y 轴范围
ax3.spines['top'].set_visible(False)  # 隐藏顶部边框
ax3.legend(loc='upper right')

# 添加断点斜线
d = 0.015  # 斜线长度
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)        # 左下角斜线
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # 右下角斜线

kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # 左上角斜线
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # 右上角斜线
ax2.plot((-d, +d), (-d, +d), **kwargs)        # 左下角斜线
ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # 右下角斜线

kwargs = dict(transform=ax3.transAxes, color='k', clip_on=False)
ax3.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # 左上角斜线
ax3.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # 右上角斜线

# 添加网格线
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax3.grid(axis='y', linestyle='--', alpha=0.7)

# 设置标题和标签
ax3.set_xlabel('Categories', fontsize=14)
ax2.set_ylabel('Values', fontsize=14)

# 调整布局
plt.tight_layout()

# 显示图像
plt.show()