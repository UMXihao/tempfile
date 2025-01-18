import torch

# 假设 A 是一个 4096 x 4096 的张量
# A = torch.randn(5, 5)
#
# print(A)
#
# # 保持前 512 列不变，将其他列置为零
# A[:, 2:] = 0
# print(A)
# A[2:, :] = 0
# print(A)

# # 创建一个示例张量
# tensor = torch.tensor([
#     [0, 0, 1, 4, 0],
#     [0, 0, 2, 5, 0],
#     [0, 0, 3, 6, 0]
# ])
tensor = torch.tensor([
    [1, 2, 3, 4, 5],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])

# 指定列的范围
n = 2
m = 3

# 判断第n列到第m列的所有元素是否为0
# 注意：列索引在Python中是从0开始的，所以第n列实际上是索引n-1
columns_zero = torch.all(tensor[n:m, :] == 0)

print(f'第{n}列到第{m}列的所有元素是否为0: {columns_zero.item()}')

print(tensor)