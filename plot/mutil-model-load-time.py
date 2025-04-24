import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ['7B-SQuAD', '13B-Human', '70B-LEval']
data = [6618.638, 10930.77, 335236.15]
data1 = [4418.638, 8498.72, 320867.95]
group3 = [5, 7, 9, 10]    # 第三组数据

# 设置柱状图的位置
x = np.arange(len(categories))  # 类别位置

# 绘制柱状图
plt.bar(x, data, label='Group 1', color='b')  # 第一组
plt.bar(x, data1, bottom=data, label='Group 2', color='r')  # 第二组，叠加在第一组上
# plt.bar(x, group3, bottom=np.array(group1) + np.array(group2), label='Group 3', color='g')  # 第三组，叠加在前两组上

# 添加标签和标题
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Stacked Bar Chart')
plt.xticks(x, categories)  # 设置x轴刻度标签
plt.legend()  # 添加图例

# 显示图形
plt.show()