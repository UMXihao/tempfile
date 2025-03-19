import numpy as np
import matplotlib
import seaborn as sns
import pylustrator
pylustrator.start()

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

def plot_line():

    x = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])
    model1 = np.array([12.180522, 39716.90792, 636297.28, 82283.59567, 921605.51, 35350.10095, 409624.9221, 666186.0596, 164659.4689, 265087.5404])
    Ours = np.array([12.180522, 5188409.425, 2753229.097, 2533102.589, 13980053.02, 14224090.81, 2861948.2, 2133021.95, 2573965.881, 742626.5432])

    # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-  --   -.  :    ,
    # marker：.  ,   o   v    <    *    +    1
    plt.figure(figsize=(5, 4))
    # linestyle = "-"
    # plt.grid(linestyle="-.")  # 设置背景网格线为虚线
    ax = plt.gca()
    # ax.spines['top'].set_visible(False)  # 去掉上边框
    # ax.spines['right'].set_visible(False)  # 去掉右边框

    linewidth = 2.0
    markersize = 7

    plt.plot(x, model1, marker='s', markersize=markersize, color="blue", label="MHA", linewidth=linewidth)
    plt.plot(x, Ours, marker='X', markersize=markersize, color="tomato", label="MLP", linewidth=linewidth)


    # group_labels = ['-', '20%', '40%', '60%', '80%']
    # plt.xticks(x, group_labels, fontsize=15)  # 默认字体大小为10
    # y_ticks = [0.10, 0.15, 0.20, 0.25, 0.30]
    # y_lables = ['0.10', '0.15', '0.20', '0.25', '0.30']
    # plt.yticks(np.array(y_ticks), y_lables, fontsize=15)
    # plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
    # plt.text(1, label_position, dataset,fontsize=25, fontweight='bold')
    # plt.xlabel("Edge Miss Rate", fontsize=15)
    plt.ylabel(f"Perplexity", fontsize=15)
    # plt.xlim(0.5, 5.5)  # 设置x轴的范围
    # plt.ylim(0.08, 0.30)

    # plt.legend()
    # 显示各曲线的图例 loc=3 lower left
    plt.legend(loc=3, numpoints=1, ncol=2)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=15)
    # plt.setp(ltext, fontsize=25, fontweight='bold')  # 设置图例字体的大小和粗细
    plt.tight_layout()
    # plt.savefig(f'ppl-mlp-mha.png', format='png')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
    plt.savefig('ppl-mlp-mha.pdf', format='pdf')
    #% start: automatic generated code from pylustrator
    plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
    import matplotlib as mpl
    getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
    plt.figure(1).axes[0].legend(loc=(0.2723, 1.008), handlelength=1.9, handletextpad=0.7, ncols=2)
    plt.figure(1).axes[0].set(position=[0.1336, 0.09681, 0.8344, 0.8278])
    #% end: automatic generated code from pylustrator
    plt.show()

def plot_bar_and_line():
    fontsize=20
    result1 = [0.1967, 0.2103, 0.2398, 0.2446, 0.2387]
    l = [i for i in range(5)]

    lx = ['2', '3', '4', '5', '6']

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.bar(l, result1, alpha=0.3, color='blue', label='HR@20')

    # left_axis.set_ylim(0.80, 0.96)
    # left_axis.set_yticks(np.arange(0.80, 0.97, 0.04))
    ax1.set_ylim([0.18, 0.26])
    ax1.set_yticks(np.arange(0.18, 0.26, 0.015))
    # ax1.set_ylabel('AUC', fontsize=fontsize)
    plt.legend(loc="upper left", prop={'size': 15})
    plt.xticks(l, lx, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    result2 = [0.0823, 0.0976, 0.1054, 0.1185, 0.1045]

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(l, result2, 'or-', label='NDCG@20', color='green')
    ax2.legend(loc=2)
    ax2.set_ylim([0.07, 0.13])
    ax2.set_yticks(np.arange(0.07, 0.13, 0.01))
    # ax2.set_ylabel('Log-loss', fontsize=fontsize)
    plt.text(1.5, 0.06, "Num", fontsize=20)
    plt.legend(loc="upper right", prop={'size': 15})
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    # , bbox_inches='tight', pad_inches=0.05, dpi=100
    plt.savefig('pics/bar_and_line.png', format='png')
    plt.show()

if __name__ == '__main__':
    plot_line()