import matplotlib.pyplot as plt
import os

# 手动设置 LaTeX 路径
os.environ["PATH"] += ":/usr/local/texlive/2024/bin/universal-darwin"

# 调整布局
# plt.tight_layout()

# 设置 Matplotlib 参数以适应 LaTeX 风格
plt.rcParams.update({
    "text.usetex": True,  # 启用 LaTeX 渲染文本
    # "font.family": "serif",  # 使用衬线字体
    # "font.serif": ["Computer Modern"],  # 使用 Computer Modern 字体
    "font.size": 10,  # 设置字体大小
    "figure.figsize": (5, 3),  # 设置图形大小
    "axes.labelsize": 10,  # 坐标轴标签字体大小
    "xtick.labelsize": 8,  # x 轴刻度字体大小
    "ytick.labelsize": 8,  # y 轴刻度字体大小
    "legend.fontsize": 8,  # 图例字体大小
    "lines.linewidth": 1.5,  # 线条宽度
    "lines.markersize": 4,  # 标记大小
    "grid.linestyle": "--",  # 网格线样式
    "grid.alpha": 0.5,  # 网格线透明度
})

# 数据
# x = np.linspace(0, 10, 100)
# y1 = np.sin(x)
# y2 = np.cos(x)
model_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # 模型规模（单位：B）
prompt_lengths = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]  # Prompt 长度（单位：tokens）
ppl = [75038.1406, 4342.6079, 73.737, 10.0562, 6.4624, 5.6659, 5.3622, 5.2203, 5.1343, 5.0523]  # 模型精度（与模型规模相关）
sim = [0.6688083680410392,0.5379551473160019,0.7475348196716701,0.6784418539774661,0.8306842165986217,
                          0.785276340799404,0.8121119945319599,0.6128736664150488,0.7348773597914541,0.6353043594245164,
                          0.8535280848194907,0.8512710176059688,0.7815897231380406,0.8206665189885614,0.755444771085966,
                          0.8740433864034259,0.8925264218708977,0.893495863663352,0.8171567668301358, 0.99]  # 模型精度（与 Prompt 长度相关）


# 创建图形
fig, ax = plt.subplots()

# 绘制折线图
ax.plot(model_sizes, ppl, label="$\sin(x)$", color="blue", marker="o", linestyle="-")
# ax.plot(prompt_lengths, sim, label="$\sin(x)$", color="blue", marker="o", linestyle="-")
# ax.plot(x, y2, label="$\cos(x)$", color="red", marker="s", linestyle="--")

# 添加标题和标签
# ax.set_title("Example of a Scientific Plot", fontsize=12)
ax.set_xlabel("Model Size", fontsize=12)
ax.set_ylabel("Perplexity", fontsize=12)
# ax.set_xlabel("Prompt Length", fontsize=12)
# ax.set_ylabel("Consistency", fontsize=12)

# 添加图例
# ax.legend(loc="upper right")

# 添加网格
# ax.grid(True)

# 显示图形
# plt.show()

# 保存为 PDF 文件，适合 Overleaf 使用
plt.savefig("fig-ppl-m.pdf", format="pdf", bbox_inches="tight")
# plt.savefig("fig-con-p.pdf", format="pdf", bbox_inches="tight")
