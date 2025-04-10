#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Xihao Sun
@contact:sunxh2016@gmail.com
@version: 1.0.0
@file: head_imp_valid.py
@time: 4/10/25 4:20 PM

最重要：读取原始模型，将layer12的head13的attn相关的权重全部置零，输入模型
最不重要：读取原始模型，将layer5的head13的attn相关的权重全部置零，输入模型
"""

from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig

origin_model = "/home/yandong/Documents/um-data/models/Llama-2-7b-hf"

# 需要修改保存目录
target_dir = "/home/yandong/Documents/um-data/models/Llama-2-7b-hf-unimp"

# 加载模型配置
config = AutoConfig.from_pretrained(origin_model)

print(config.head_dim)
print(config.num_attention_heads)

# save tokenizer
tokenizer = AutoTokenizer.from_pretrained(origin_model)
tokenizer.save_pretrained(target_dir)

model = LlamaForCausalLM.from_pretrained(origin_model)  # 读取或者导出模型需要明确指定LlamaForCausalLM类型，否则自动识别为LlamaModel

# print(model)
target_layer = 5
target_head = 13
head_dim = config.head_dim

layer = model.model.layers[target_layer - 1]
start_head = head_dim * (target_head - 1)
end_head = head_dim * target_head
layer.self_attn.q_proj.weight.data[:, start_head:end_head] = 0
layer.self_attn.k_proj.weight.data[:, start_head:end_head] = 0
layer.self_attn.v_proj.weight.data[:, start_head:end_head] = 0
layer.self_attn.o_proj.weight.data[start_head:end_head, :] = 0

model.save_pretrained(target_dir)
