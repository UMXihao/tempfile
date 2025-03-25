import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
model1 = np.array([1.147831623,0.918822254,0.503805096,0.392943239,0.389868374,0.360693465,0.353197028,0.341084511,0.318233539,0.295382568])
Ours = np.array([75038.1406,4342.6079,73.737,6.4624,5.2203,5.6659,10.0562,5.3622,5.1343,4.9064])

linewidth = 2.0
markersize = 7

# 创建图形和第一个坐标轴
fig, ax1 = plt.subplots(figsize=(6, 4))
# 绘制第一条曲线（左侧Y轴）
ax1.plot(x, model1, marker='s', markersize=markersize, color="blue", label="Perplexity", linewidth=linewidth)

# ax1.set_ylabel('Perplexity')

# 创建第二个坐标轴（共享X轴）
ax2 = ax1.twinx()

# 绘制第二条曲线（右侧Y轴）
ax2.plot(x, Ours, marker='X', markersize=markersize, color="tomato", label="Uncertainty", linewidth=linewidth)
# ax2.set_ylabel('Uncertainty')

# 添加图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# plt.title('双Y坐标轴示例')
# plt.show()
plt.savefig(f'ppl-uncertainty.png', format='png')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
