import matplotlib.pyplot as plt

# Orac-mini-3B
llama7_x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
acc_y = [67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2,67.2]

# Llama-2-7B
llama7_y = [0,67.7344,67.5899,65.3593,69.3791,62.3507,67.05019,66.9659,67.5988,64.6749,65.4849,71.6797,60.3739,61.0499,67.9754,65.6709,68.1726,64.4009,63.9879,65.2679,65.8319,63.7529,68.9499,66.2849,66.0419,63.0819,61.3639,66.8089,61.9849,68.6859,10.34,3.421]

# 创建图形
plt.figure(figsize=(6, 5))

# 绘制三条线
plt.plot(llama7_x, acc_y, label='All-Layer', marker='o', color='#FF420E', linestyle='--')  # 添加标记点
plt.plot(llama7_x, llama7_y, label='Skip-Layer', marker='s', color='#004586')  # 添加标记点

plt.rcParams["font.family"] = "Arial"
# plt.tick_params(axis='both', labelsize=10)  # 'both' 表示同时设置 x 轴和 y 轴的标尺字体大小

# 添加标题和标签
# plt.title('sample')
plt.xlabel('Layer', fontsize=24)
plt.ylabel('SQuAD F1 Score', fontsize=24)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# 添加图例
plt.legend(fontsize=20)

# 显示网格
# plt.grid(True)
plt.tight_layout()
# 显示图形
# plt.show()
plt.savefig('pics/middle-layer-accuracy.pdf', format='pdf', bbox_inches='tight', pad_inches=0.26)
