import numpy as np
import time

# 随机初始化一个 4096x4096 的张量
tensor = np.random.rand(4096, 4096)

result = 0
# 多位偏移访问第一列
for i in range(1000):
    start_time = time.time()
    for i in range(128):
        first_column_origin = tensor[:, 4095]
        # first_column_origin = tensor[0]
    end_time = time.time()
    elapsed_time = end_time - start_time
    if i != 0:
        result += elapsed_time
    # print("First column of the transposed tensor:")
    # print(first_column)
    # print(f"first_column_origin time: {elapsed_time:.6f} seconds")
print(f"{result/999:.6f}")

# 转置后访问第一列
# for i in range(100):
#     start_time1 = time.time()
#     for i in range(128):
#         transposed_tensor = tensor.T
#         first_column = transposed_tensor[0]
#     end_time1 = time.time()
#     elapsed_time1 = end_time1 - start_time1
# # # 打印结果
# # # print("First column of the transposed tensor:")
# # # print(first_column)
# # print(f"first_column time: {elapsed_time1:.6f} seconds")
#     print(f"{elapsed_time1:.6f}")