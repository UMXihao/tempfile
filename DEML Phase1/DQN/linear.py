import numpy as np

# 示例数据：模型大小（MB）和延迟（ms）
model_sizes = np.arange(0, 100, 10)  # 模型规模
prompt_lengths = np.arange(0, 2000, 100)  # Prompt长度
delays = np.array([99.1, 101.33, 108.56, 111.42, 111.95, 121.48, 123.06, 130.25, 134.27, 138.4])  # 延迟
sim = np.array([0.737549, 0.653158, 0.798446, 0.587124, 0.622443, 0.8642, 0.845584, 0.685215, 0.868587, 0.844477, 0.843288, 0.883459, 0.897513, 0.8487745, 0.703606, 0.762648, 0.825139, 0.827052, 0.872399, 0.84564])

# 使用 numpy 的 polyfit 进行线性回归
m, b = np.polyfit(prompt_lengths, sim, 1)  # 1 表示线性拟合

print(f"线性表达式: y = {m:.2f}x + {b:.2f}")

import matplotlib.pyplot as plt

# 绘制数据点
plt.scatter(prompt_lengths, sim, color='blue', label='Data points')

# 绘制拟合直线
y_pred = m * prompt_lengths + b
plt.plot(prompt_lengths, y_pred, color='red', label='Fitted line')

# 添加标签和图例
plt.xlabel('Model Size')
plt.ylabel('TTFT(ms)')
plt.title('Linear relationship between model size and latency')
plt.legend()
plt.show()