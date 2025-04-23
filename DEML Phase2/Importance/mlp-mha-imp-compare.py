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
import numpy as np
from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig

origin_model_path = "/home/yandong/Documents/um-data/models/Llama-2-7b-hf"
imp_dir = "/home/yandong/Documents/um-data/models/Llama-2-7b-hf-mlp-com3"


def load_config(model_path):
    # 加载模型配置
    config = AutoConfig.from_pretrained(model_path)
    # print(config.head_dim)
    # print(config.num_attention_heads)
    return config


def load_model(model_path):
    model = LlamaForCausalLM.from_pretrained(model_path)  # 读取或者导出模型需要明确指定LlamaForCausalLM类型，否则自动识别为LlamaModel
    # print(model)
    return model


def save_tokenizer(model_path, directory):
    # save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(directory)


def save_model(model, directory):
    model.save_pretrained(directory)


def set_mha_zero(model, layer_num, head_num, head_dim):
    layer = model.model.layers[layer_num - 1]
    start_head = head_dim * (head_num - 1)
    end_head = head_dim * head_num
    layer.self_attn.q_proj.weight.data[:, start_head:end_head] = 0
    layer.self_attn.k_proj.weight.data[:, start_head:end_head] = 0
    layer.self_attn.v_proj.weight.data[:, start_head:end_head] = 0
    layer.self_attn.o_proj.weight.data[start_head:end_head, :] = 0


def set_mlp_zero(model, layer_num, head_num, head_dim):
    layer = model.model.layers[layer_num - 1]
    start_head = head_dim * (head_num - 1)
    end_head = head_dim * head_num
    layer.mlp.gate_proj.weight.data[:, start_head:end_head] = 0
    layer.mlp.up_proj.weight.data[:, start_head:end_head] = 0
    layer.mlp.down_proj.weight.data[start_head:end_head, :] = 0


'''对重要性层进行掩码处理'''
origin_model = load_model(origin_model_path)
# config = load_config(origin_model_path)
# head = [12, 17, 27]
# for i in head:
#     set_mha_zero(origin_model, 5, i, config.head_dim)
mlp = [4, 13, 3]
for i in mlp:
    set_mlp_zero(origin_model, 5, i, 344)
save_tokenizer(origin_model_path, imp_dir)
save_model(origin_model, imp_dir)