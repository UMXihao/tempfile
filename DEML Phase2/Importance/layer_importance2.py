import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
model_name = "/home/yandong/Documents/um-data/models/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 准备输入数据
input_text = "This is a sample input text for the LLaMA2-7B model."
inputs = tokenizer(input_text, return_tensors="pt")

# 定义一个函数来计算输入和输出之间的相似度
def calculate_similarity(input_tensor, output_tensor):
    # 使用余弦相似度作为相似度指标
    similarity = F.cosine_similarity(input_tensor, output_tensor, dim=-1)
    return similarity.mean().item()

# 遍历模型的每一层，计算Attention和FFN的重要性
for i, layer in enumerate(model.transformer.h):
    # 获取当前层的输入
    layer_input = inputs["input_ids"]

    # 计算Attention模块的输出
    attention_output = layer.attention(layer_input)[0]

    # 计算FFN模块的输出
    ffn_output = layer.feed_forward(attention_output)

    # 计算输入和Attention输出之间的相似度
    attention_similarity = calculate_similarity(layer_input, attention_output)

    # 计算输入和FFN输出之间的相似度
    ffn_similarity = calculate_similarity(layer_input, ffn_output)

    print(f"Layer {i+1}:")
    print(f"  Attention Similarity: {attention_similarity:.4f}")
    print(f"  FFN Similarity: {ffn_similarity:.4f}")