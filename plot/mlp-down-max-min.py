import matplotlib.pyplot as plt

orac_x = [1,2,3,4,5,6,7,8]
min = [6.1245,6.6683,7.732,8.2492,16.8925,18.6041,39.5714,54.8694]
max = [7.7638,8.5449,9.5437,18.07,21.3996,199.5495,261.428,328.495]

# 创建图形
plt.figure(figsize=(6, 5))

# 绘制三条线
plt.plot(orac_x, min, label='The Smallest', marker='o', color='#004586')  # 添加标记点
plt.plot(orac_x, max, label='The Largest', marker='s', color='#FF420E')  # 添加标记点

plt.rcParams["font.family"] = "Arial"
# plt.tick_params(axis='both', labelsize=10)  # 'both' 表示同时设置 x 轴和 y 轴的标尺字体大小

# 添加标题和标签
# plt.title('sample')
plt.xlabel('Layer', fontsize=24)
plt.ylabel('Perplexity', fontsize=24)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# 添加图例
plt.legend(fontsize=20)

# 显示网格
# plt.grid(True)
plt.tight_layout()
# 显示图形
# plt.show()
plt.savefig('pics/mlp-down-max-min.pdf', format='pdf', bbox_inches='tight', pad_inches=0.26)
