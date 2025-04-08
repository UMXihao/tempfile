import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# 加载模型和分词器
# model_name = "/home/yandong/Documents/um-data/models/Llama-2-7b-hf"
model_name = "/home/yandong/Documents/um-data/models/Orac-mini-3B"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 准备输入数据
input_text = "This is a sample input text for the LLaMA2-7B model."
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# response = model.generate(inputs['input_ids'], max_length=100)
# print(tokenizer.decode(response[0]))


# 计算每一层的输入输出相似度
def compute_cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    a = a.view(a.size(0), -1)  # 展平
    b = b.view(b.size(0), -1)  # 展平
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)
    similarity = torch.mm(a_norm, b_norm.t())
    return similarity.item()


# 获取模型的每一层的输入输出
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # 包含每一层的输入输出


# 遍历每一层
for i in range(len(hidden_states) - 1):
    input_layer = hidden_states[i]
    output_layer = hidden_states[i + 1]
    similarity = compute_cosine_similarity(input_layer, output_layer)
    # print(f"Layer {i + 1} Input-Output Similarity: {similarity:.4f}")
    print(similarity)