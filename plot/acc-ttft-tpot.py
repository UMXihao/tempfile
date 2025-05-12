import matplotlib.pyplot as plt
import numpy as np
group_labels = ['Llama-2-7B', '13B', '70B']
# Accuracy
acc_squad = np.array([67.2, 72.9, 80.7])
acc_human = np.array([12.8, 18.3, 29.9])
acc_leval = np.array([29.75, 30.49, 44.4312])
# TTFT
squad = np.array([18.638, 37.69, 4586.691])
human = np.array([72.79, 130.77, 19766.63])
leval = np.array([2093.85, 8107.264, 362433.67])
# Accuracy
tpot_squad = np.array([10.797, 24.562, 1126.855])
tpot_human = np.array([11.09, 26.74, 1151.89])
tpot_leval = np.array([16.47, 50.57, 1255.14])
# 创建一个画布
plt.figure(figsize=(15, 5))  # 设置画布大小，宽度为20英寸，高度为5英寸

# 绘制第一个折线图
plt.subplot(1, 3, 1)  # 表示一行三列的第1个位置
plt.plot(group_labels, acc_squad, label='SQuAD', color='blue', marker='o')  # 第一组数据
plt.plot(group_labels, acc_human, label='Human_Eval', color='red', marker='s')   # 第二组数据
plt.plot(group_labels, acc_leval, label='L-Eval', color='orange', marker='^')   # 第三组数据
plt.title('Accuracy', fontsize=24)  # 设置标题
# plt.xlabel('Model', fontsize=24)  # 设置X轴标签
# plt.ylabel('Accuracy', fontsize=24)  # 设置Y轴标签
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# 绘制第二个折线图
plt.subplot(1, 3, 2)  # 表示一行三列的第2个位置
plt.plot(group_labels, squad, label='SQuAD', color='blue', marker='o')  # 第一组数据
plt.plot(group_labels, human, label='Human_Eval', color='red', marker='s') # 第二组数据
plt.plot(group_labels, leval, label='L-Eval', color='orange', marker='^')   # 第三组数据
plt.title('TTFT', fontsize=24)  # 设置标题
# plt.ylabel('Latency(ms)', fontsize=24)  # 设置Y轴标签
# plt.xlabel('Model', fontsize=24)  # 设置X轴标签
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# 绘制第三个折线图
plt.subplot(1, 3, 3)  # 表示一行三列的第3个位置
plt.plot(group_labels, tpot_squad, label='SQuAD', color='blue', marker='o')  # 第一组数据
plt.plot(group_labels, tpot_human, label='Human_Eval', color='red', marker='s')   # 第二组数据
plt.plot(group_labels, tpot_leval, label='L-Eval', color='orange', marker='^')   # 第三组数据
plt.title('TPOT', fontsize=24)  # 设置标题
# plt.ylabel('Latency(ms)', fontsize=24)  # 设置Y轴标签
# plt.xlabel('Model', fontsize=24)  # 设置X轴标签

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# 添加图例
plt.legend(fontsize=20)

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
# plt.show()
plt.savefig('pics/acc-ttft-tpot.pdf', format='pdf', bbox_inches='tight', pad_inches=0.26)
