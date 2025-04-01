import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns
import pylustrator
pylustrator.start()

def plot_ablation_bar_in_one():

    models = ['Base', '$\\neg$ A', '$\\neg$ B', '$\\neg$ C', "Ours"]

    x_label_size = 35
    rotation = 0
    # 从这里开始选择数据

    labels = models

    # plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 如果要显示中文字体,则在此处设为：SimHei
    # plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    # # plt.rcParams['savefig.dpi'] = 300  # 图片像素
    # plt.rcParams['figure.dpi'] = 300  # 分辨率

    plt.figure(figsize=(40, 7))
    colors = ['blue', 'green', 'red']

    low = 0.34
    up = 0.391
    Beauty = [0.3488, 0.3687, 0.3688, 0.3546, 0.3761]
    data=Beauty
    plt.subplot(141)
    plt.ylim(low, up)
    position = (up - low) * 0.9 + low
    plt.text(1.4, position, 'Beauty', fontsize=40)
    plt.xticks(fontsize=x_label_size, rotation=rotation)

    my_y_ticks = np.arange(low, up, 0.01)
    plt.yticks(my_y_ticks, fontsize=40)
    # plt.xlabel("Meituan", fontsize=20, fontweight='bold')
    # hatches = ["\\"] + ["."] * 4 + ["/"]
    # /,  \, |, -, +, x, o, O,., * 。
    plt.bar([0], data[0], width=0.5, color=colors[0], hatch=".", edgecolor='w')
    plt.bar([1, 2, 3], data[1:4], width=0.5, color=colors[1], hatch=".", edgecolor='w')
    plt.bar([4], data[4], width=0.5, color=colors[2], hatch=".", edgecolor='w')
    plt.xticks([0, 1, 2, 3, 4], labels)
    # plt.bar(np.arange(len(data)), data, width=0.5, tick_label=labels, color=colors, hatches=hatches)

    low = 0.32
    up = 0.361
    Sport = [0.328, 0.3385, 0.3346, 0.3335, 0.3473]
    data = Sport

    plt.subplot(142)
    plt.ylim(low, up)
    position = (up - low) * 0.9 + low
    plt.text(1.4, position, 'Sports', fontsize=40)
    plt.xticks(fontsize=x_label_size, rotation=rotation)
    my_y_ticks = np.arange(low, up, 0.01)
    plt.yticks(my_y_ticks, fontsize=40)
    # plt.xlabel("Beauty", fontsize=20, fontweight='bold')
    # colors = ['darkorange'] * 1 + ['royalblue'] * 4 + ['red']
    # /,  \, |, -, +, x, o, O,., * 。
    # plt.bar(np.arange(len(data)), data, hatch=2 * '.', width=0.5,
    #         tick_label=labels, color=colors,
    #         # edgecolor='k'
    #         )
    plt.bar([0], data[0], width=0.5, color=colors[0], hatch="\\", edgecolor='w')
    plt.bar([1, 2, 3], data[1:4], width=0.5, color=colors[1], hatch=".", edgecolor='w')
    plt.bar([4], data[4], width=0.5, color=colors[2], hatch="/", edgecolor='w')
    plt.xticks([0, 1, 2, 3, 4], labels)

    low = 0.34
    up = 0.391

    Toys = [0.3455, 0.3671, 0.3588, 0.3611, 0.3749]

    data = Toys

    plt.subplot(143)
    plt.ylim(low, up)
    position = (up - low) * 0.9 + low
    plt.text(1.4, position, 'Toys', fontsize=40)
    plt.xticks(fontsize=x_label_size, rotation=rotation)
    my_y_ticks = np.arange(low, up, 0.01)
    plt.yticks(my_y_ticks, fontsize=40)
    # plt.xlabel("Sports", fontsize=20, fontweight='bold')
    # colors = ['darkorange'] * 1 + ['royalblue'] * 4 + ['red']
    # /,  \, |, -, +, x, o, O,., * 。
    # plt.bar(np.arange(len(data)), data, hatch=2 * '.', width=0.5,
    #         tick_label=labels, color=colors,
    #         # edgecolor='k'
    #         )
    plt.bar([0], data[0], width=0.5, color=colors[0], hatch="\\", edgecolor='w')
    plt.bar([1, 2, 3], data[1:4], width=0.5, color=colors[1], hatch=".", edgecolor='w')
    plt.bar([4], data[4], width=0.5, color=colors[2], hatch="/", edgecolor='w')
    plt.xticks([0, 1, 2, 3, 4], labels)

    low = 0.49
    up = 0.54

    Yelp = [0.496, 0.523, 0.5231, 0.5167, 0.53]
    data = Yelp
    plt.subplot(144)
    plt.ylim(low, up)
    position = (up - low) * 0.9 + low
    plt.text(1.6, position, 'Yelp', fontsize=40)
    plt.xticks(fontsize=x_label_size, rotation=rotation)
    my_y_ticks = np.arange(low, up, 0.01)
    plt.yticks(my_y_ticks, fontsize=40)
    # plt.xlabel("Toys", fontsize=20, fontweight='bold')
    # colors = ['darkorange'] * 1 + ['royalblue'] * 4 + ['red']
    # /,  \, |, -, +, x, o, O,., * 。
    # plt.bar(np.arange(len(data)), data, hatch=2 * '.', width=0.5,
    #         tick_label=labels, color=colors,
    #         # edgecolor='k'
    #         )
    plt.bar([0], data[0], width=0.5, color=colors[0], hatch="\\", edgecolor='w')
    plt.bar([1, 2, 3], data[1:4], width=0.5, color=colors[1], hatch=".", edgecolor='w')
    plt.bar([4], data[4], width=0.5, color=colors[2], hatch="/", edgecolor='w')
    plt.xticks([0, 1, 2, 3, 4], labels)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0)
    # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
    plt.savefig('./pics/ablation.png', format='png')
    plt.show()

plot_ablation_bar_in_one()