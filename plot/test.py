import matplotlib.pyplot as plt
import numpy as np

# 数据准备
categories = [100, 500, 1000, 2000, 4000]
class1_data = [10, 15, 7, 12, 9]  # 第一类数据
class2_data = [8, 11, 9, 14, 10]   # 第二类数据
class3_data = [12, 9, 6, 10, 8]   # 第三类数据

class1_data = [34.08, 120.34, 281.4, 497.9, 938.78]
class2_data = [56.35, 177.7, 349.5, 748.86, 1563.5]
class3_data = [56.35, 177.7, 349.5, 748.86, 1563.5]

ours = [22.07, 22.09, 22.37, 22.93, 24.06]
llama_cpp = [27.28, 27.82, 28.85, 30.61, 32.89]
power_infer = [120.85, 123.05, 124.66, 157.46, 157.85]

x = np.arange(len(categories))  # x轴位置
width = 0.2  # 柱状图的宽度

# 创建画布和子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 绘制第一个柱状图
ax1.bar(x - width, class1_data, width, label='Class 1', color='royalblue', hatch=".")
ax1.bar(x, class2_data, width, label='Class 2', color='tomato', hatch="/")
ax1.bar(x + width, class3_data, width, label='Class 3', color='gold', hatch="\\")
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.set_ylabel('Latency (ms)')
ax1.set_title('TTFT')
ax1.legend()

# 绘制第二个柱状图
ax2.bar(x - width, ours, width, label='Class 1', color='royalblue', hatch=".")
ax2.bar(x, llama_cpp, width, label='Class 2', color='tomato', hatch="/")
ax2.bar(x + width, power_infer, width, label='Class 3', color='gold', hatch="\\")
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.set_ylabel('Latency (ms)')
ax2.set_title('TPOT')
ax2.legend()

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()
fig.savefig('./pics/two_bars.png', format='png')