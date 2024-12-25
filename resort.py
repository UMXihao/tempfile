import random

import torch
import torch.nn.functional as F
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

    print("score:", scores)
    # 如果有掩码，应用掩码
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 应用softmax获取注意力权重
    attention_weights = F.softmax(scores, dim=-1)
    print("attention_weights:", attention_weights)
    # 使用注意力权重对V进行加权
    output = torch.matmul(attention_weights, V)

    return output, attention_weights

# 模型输入
model_name = '../../models/Llama-2-7b-hf'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 读取QKV权重
'''
# model weights

embed_tokens.weight 32000*4096

layers.0.self_attn.q_proj.weight 4096*4096
layers.0.self_attn.k_proj.weight 4096*4096
layers.0.self_attn.v_proj.weight 4096*4096
layers.0.self_attn.o_proj.weight 4096*4096

layers.0.mlp.gate_proj.weight 11008*4096
layers.0.mlp.up_proj.weight 11008*4096
layers.0.mlp.down_proj.weight 4096*11008

layers.0.input_layernorm.weight 4096
layers.0.post_attention_layernorm.weight 4096

... layers

norm.weight 4096
'''
state_dict = model.state_dict()

# 重新排序权重
'''
self.attn qkvo 32 * 128 进行head拆分，计算多组数据集输入进行重要性排序
排序Query weight，Key weight、value weight暂时保持不动
计算XQi与全部KV的注意力得分，最高的将优先加载
'''
# 输入X可以遍历取值embed_tokens.weight
embed = state_dict['embed_tokens.weight']
inputs = embed[random.randint(1, 32000)]

print("input:", inputs)

dims = 4096

# 按照head头进行权重拆分
query = state_dict['layers.0.self_attn.q_proj.weight']
key = state_dict['layers.0.self_attn.k_proj.weight']
value = state_dict['layers.0.self_attn.v_proj.weight']

# attention score
Q = torch.matmul(inputs, query)
K = torch.matmul(inputs, key)
V = torch.matmul(inputs, value)

print("Q:", Q)
print("K:", K)
print("V:", V)
# 计算注意力得分
output, attention_weights = attention_scores(Q, K, V)

print("Attention Weights:", attention_weights)
print("Output:", output)

# 读取FFN权重
'''
mlp gate、up、down 也按照32个头进行拆分 32 * 344
'''

# 重新排序权重

# 重置模型