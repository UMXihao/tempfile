import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 示例数据集
x = np.arange(0, 100, 10)  # 模型规模
y = np.arange(0, 1000, 100)  # Prompt长度
z = np.array([99.1, 115.91, 143.42, 174.96, 223.03, 261.96, 289.94, 327.23, 380.98, 438.64])  # 目标变量 z

# 使用 Scikit-learn 进行多元线性回归
X = np.column_stack((x, y))  # 特征矩阵

model = LinearRegression()
model.fit(X, z)

a, b = model.coef_
c = model.intercept_

print(f"拟合结果: z = {a:.2f}x + {b:.2f}y + {c:.2f}")

# 评估拟合效果
z_pred = model.predict(X)
mse = mean_squared_error(z, z_pred)
r2 = r2_score(z, z_pred)

print(f"均方误差 (MSE): {mse:.2f}")
print(f"决定系数 (R²): {r2:.2f}")

# 可视化真实值 vs 预测值
plt.scatter(z, z_pred, color='blue', label='真实值 vs 预测值')
plt.plot([min(z), max(z)], [min(z), max(z)], color='red', linestyle='--', label='理想拟合')
plt.xlabel('真实值 (z)')
plt.ylabel('预测值 (z_pred)')
plt.title('真实值 vs 预测值')
plt.legend()
plt.show()