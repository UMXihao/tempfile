import random

import torch
import torch.nn as nn
import math
from transformers import AutoModel, AutoTokenizer

def attention_scores(Q, K, V, mask=None):
    """
    计算注意力得分。

    参数:
    - Q: 查询张量 (query)，形状为 (batch_size, seq_len, dim)
    - K: 键张量 (key)，形状为 (batch_size, seq_len, dim)
    - V: 值张量 (value)，形状为 (batch_size, seq_len, dim)
    - mask: 可选的掩码张量，形状为 (batch_size, seq_len)，用于屏蔽某些位置的注意力

    返回:
    - 经过注意力加权的值张量
    """
    # 计算Q和K的点积
    scores = torch.matmul(Q, torch.transpose(K.unsqueeze(0), 0, 1).squeeze(0)) / math.sqrt(Q.size(-1))

    print("score:", scores.shape)
    # 如果有掩码，应用掩码
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 应用softmax获取注意力权重
    attention_weights = torch.softmax(scores, dim=-1)
    print("attention_weights:", attention_weights.shape)
    # 使用注意力权重对V进行加权
    output = torch.matmul(attention_weights, V)

    return output, attention_weights

# 模型输入
model_name = '../../models/Llama-2-7b-hf'
# model_name = '../../models/Llama-2-7b-hf-P10'
# model_name = '../../models/all-false-lora-P10'
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained('../../models/Llama-2-7b-hf-P20')
model = AutoModel.from_pretrained(model_name)

# for param in model.parameters():
#     print(param)

# for name, param in model.named_parameters():
#     # 冻结除了指定层之外的所有参数
#     if "self_attn" not in name:
#         param.requires_grad = False
#
# for name, param in model.named_parameters():
#     print(name, " requires_grad = ", param.requires_grad)
#
for layer in model.layers:
    layer.self_attn.q_proj.weight.data[:, 896:] = 0
    layer.self_attn.k_proj.weight.data[:, 896:] = 0
    layer.self_attn.v_proj.weight.data[:, 896:] = 0
    layer.self_attn.o_proj.weight.data[896:, :] = 0
# #
# state_dict = model.state_dict()

# print(state_dict['layers.0.self_attn.q_proj.weight'])

model.save_pretrained('../../models/Llama-2-7b-hf-P20')

# 重新排序权重
'''
self.attn qkvo 32 * 128 进行head拆分，计算多组数据集输入进行重要性排序
排序Query weight，Key weight、value weight暂时保持不动
计算XQi与全部KV的注意力得分，最高的将优先加载
'''
# 输入X可以遍历取值embed_tokens.weight
# embed = state_dict['embed_tokens.weight']
# inputs = embed[random.randint(1, 32000)]
#
# inputs = torch.transpose(inputs.unsqueeze(0), 0, 1).squeeze(0)
# print("input:", inputs)
#
# # 按照head头进行权重拆分
# query = state_dict['layers.0.self_attn.q_proj.weight']
# key = state_dict['layers.0.self_attn.k_proj.weight']
# value = state_dict['layers.0.self_attn.v_proj.weight']
#
# print("Q weight:", query.shape)
# print("K weight:", key.shape)
# print("V weight:", value.shape)

# attention score
# Q = torch.matmul(query, inputs)
# K = torch.matmul(key, inputs)
# V = torch.matmul(value, inputs)

# print("Q:", Q)
# print("K:", K)
# print("V:", V)
# # 计算注意力得分
# output, attention_weights = attention_scores(Q, K, V)
#
# print("Attention Weights:", attention_weights)
# print("Output:", output)

# 读取FFN权重
'''
mlp gate、up、down 也按照32个头进行拆分 32 * 344
'''

# 重新排序权重

# 重置模型