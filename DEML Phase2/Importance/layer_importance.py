import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
model_name = "/home/yandong/Documents/um-data/models/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 准备输入数据
input_text = "This is a sample input text."
inputs = tokenizer(input_text, return_tensors="pt")

# 获取模型的每一层
layers = model.model.layers

# 存储每一层的输入和输出
layer_inputs = []
layer_outputs = []

# 遍历每一层，获取输入和输出
with torch.no_grad():
    hidden_states = inputs["input_ids"]
    for i, layer in enumerate(layers):
        # 获取当前层的输入
        layer_inputs.append(hidden_states)
        # 获取当前层的输出
        hidden_states = layer(hidden_states)[0]
        layer_outputs.append(hidden_states)

# 计算每一层的输入和输出之间的余弦相似度
similarities = []
for i in range(len(layer_inputs)):
    input_tensor = layer_inputs[i]
    output_tensor = layer_outputs[i]
    # 展平张量以计算相似度
    input_flat = input_tensor.view(-1)
    output_flat = output_tensor.view(-1)
    similarity = F.cosine_similarity(input_flat.unsqueeze(0), output_flat.unsqueeze(0))
    similarities.append(similarity.item())

# 打印每一层的相似度
for i, similarity in enumerate(similarities):
    print(f"Layer {i+1} similarity: {similarity}")