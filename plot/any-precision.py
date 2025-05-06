# 8 6.67GB {3,4,8} 7.7 GB 13.7 GB
import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ['Q8', 'Any-Precision', 'Separate Deployment']  # 分类标签
group1 = [6.67, 7.7, 13.7]

# 设置柱状图的位置和宽度
x = np.arange(len(categories))  # 分类标签的位置
width = 0.25  # 柱子的宽度

# 绘制柱状图
plt.bar(x, group1, width, label='Memory', color='#009aff')

# 添加标题和标签
# plt.xlabel('Categories')
plt.ylabel('Memory size')
plt.title('Memory size for deployment of Llama-2-7B')
plt.xticks(x, categories)  # 设置x轴刻度标签
# plt.legend()  # 添加图例

# 显示图形
plt.savefig('pics/mutil-model.pdf', format='pdf', bbox_inches='tight', pad_inches=0.26)
# plt.show()