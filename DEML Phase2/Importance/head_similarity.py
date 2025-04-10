#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Xihao Sun
@contact:sunxh2016@gmail.com
@version: 1.0.0
@file: head_similarity.py
@time: 4/10/25 2:31 PM

对每一层的Head重要性进行分析，即对每一层的attn进行head数量的拆分
需要查看模型的配置，确定head的数量
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import cosine_similarity

# 加载模型和分词器
# model_name = "/home/yandong/Documents/um-data/models/Llama-2-7b-hf"
# model_name = "/home/yandong/Documents/um-data/models/Orac-mini-3B"
# model_name = "/home/yandong/Documents/um-data/models/Vicuna-7B"
# model_name = "/home/yandong/Documents/um-data/models/MPT-7B-Chat"
model_name = "/home/yandong/Documents/um-data/models/InternLM2-chat-7B"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)  # InternLM2
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)  # InternLM2

importance_head = []

hidden_state_input = []
hidden_state_attn = []


def compute_attention_head_similarity(inputs, outputs, num_heads):
    """
    计算注意力层中不同 head 的输出相似度
    :param outputs: 注意力层的输出，形状为 (batch_size, seq_len, hidden_size)
    :param num_heads: 注意力头的数量
    :return: 相似度矩阵，形状为 (num_heads, num_heads)
    """
    batch_size, seq_len, hidden_size = outputs.shape
    head_dim = hidden_size // num_heads
    # 将输出重新排列为 (batch_size, seq_len, num_heads, head_dim)
    inputs = inputs.view(batch_size, seq_len, num_heads, head_dim)
    outputs = outputs.view(batch_size, seq_len, num_heads, head_dim)
    # 调整维度以便计算相似度
    # 将张量按 num_heads 划分
    reshaped_inputs = inputs.permute(2, 0, 1, 3).reshape(num_heads, -1)
    reshaped_outputs = outputs.permute(2, 0, 1, 3).reshape(num_heads, -1)

    similarities = []
    for i in range(num_heads):
        # 获取当前头和下一个头的一维向量
        current_head = reshaped_inputs[i]
        next_head = reshaped_outputs[i]

        # 计算余弦相似度
        cos_sim = cosine_similarity(current_head.unsqueeze(0), next_head.unsqueeze(0), dim=1)
        similarities.append(cos_sim.item())

    return similarities
    # outputs = outputs.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
    # outputs = outputs.reshape(batch_size * num_heads, seq_len * head_dim)
    # # 计算余弦相似度
    # similarity_matrix = F.cosine_similarity(outputs.unsqueeze(1), outputs.unsqueeze(0), dim=-1)
    # similarity_matrix = similarity_matrix.view(batch_size, num_heads, batch_size, num_heads)
    # # 取对角线上的相似度矩阵，即同一个 batch 内的 head 之间的相似度
    # similarity_matrix = similarity_matrix.diagonal(dim1=0, dim2=2)
    # return similarity_matrix

class SimilarityRecorder(nn.Module):
    def __init__(self, module):
        super(SimilarityRecorder, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        # llama2: self_attn的输入kwargs.hidden_states, mlp的输入args
        # if self.module.__class__.__name__ == 'LlamaSdpaAttention':
        #     output = self.module(*args, **kwargs)  # 执行原始模块
        #     hidden_state_input.append(kwargs.get('hidden_states'))
        #     hidden_state_attn.append(output[0])
        # else:
        #     output = self.module(*args, **kwargs)  # 执行原始模块
        # if self.module.__class__.__name__ == 'MptAttention':
        #     output = self.module(*args, **kwargs)  # 执行原始模块
        #     hidden_state_input.append(args[0])
        #     hidden_state_attn.append(output[0])
        # else:
        #     output = self.module(*args, **kwargs)  # 执行原始模块

        if self.module.__class__.__name__ == 'InternLM2Attention':
            output = self.module(*args, **kwargs)  # 执行原始模块
            hidden_state_input.append(kwargs.get('hidden_states'))
            hidden_state_attn.append(output[0])
        else:
            output = self.module(*args, **kwargs)  # 执行原始模块
        return output


# Llama2-7B/Orac-mini-3B/Vicuna
# for i in range(len(model.model.layers)):
#     layer = model.model.layers[i]
#     layer.self_attn = SimilarityRecorder(layer.self_attn)
#     layer.mlp = SimilarityRecorder(layer.mlp)

# MPT
# for i in range(len(model.transformer.blocks)):
#     layer = model.transformer.blocks[i]
#     layer.attn = SimilarityRecorder(layer.attn)
#     layer.ffn = SimilarityRecorder(layer.ffn)

# InternLM
for i in range(len(model.model.layers)):
    layer = model.model.layers[i]
    layer.attention = SimilarityRecorder(layer.attention)
    layer.feed_forward = SimilarityRecorder(layer.feed_forward)

# 准备输入
prompts = ['What sits on top of the Main Building at Notre Dame?',
           'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
           'What is in front of the Notre Dame Main Building?',
           'The Basilica of the Sacred heart at Notre Dame is beside to which structure?',
           'What is the Grotto at Notre Dame?',
           'When did the Scholastic Magazine of Notre dame begin publishing?',
           "How often is Notre Dame's the Juggler published?"]
input_text = 'What is the Grotto at Notre Dame?'
inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    model(inputs['input_ids'])

num_heads = 32
for i in range(len(hidden_state_input)):
    importance_head.append(compute_attention_head_similarity(hidden_state_input[i], hidden_state_attn[i], num_heads))
print(importance_head)