import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 示例数据：模型大小和精度
model_sizes = np.array([100, 200, 300, 400, 500])  # 模型大小
accuracies = np.array([0.8, 0.85, 0.88, 0.9, 0.92])  # 对应的精度

# 线性回归拟合
model = LinearRegression()
model.fit(model_sizes.reshape(-1, 1), accuracies)

# 获取拟合结果
slope = model.coef_[0]  # 斜率
intercept = model.intercept_  # 截距
print(f"拟合的线性关系式为：精度 = {slope:.4f} * 模型大小 + {intercept:.4f}")

# 绘制拟合曲线
plt.scatter(model_sizes, accuracies, color='blue', label='实际数据')
plt.plot(model_sizes, model.predict(model_sizes.reshape(-1, 1)), color='red', label='拟合曲线')
plt.xlabel('模型大小')
plt.ylabel('精度')
plt.legend()
plt.show()