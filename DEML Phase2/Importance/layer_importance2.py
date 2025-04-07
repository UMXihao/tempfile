import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# 加载模型和分词器
model_name = "/home/yandong/Documents/um-data/models/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 准备输入数据
input_text = "This is a sample input text for the LLaMA2-7B model."
inputs = tokenizer(input_text, return_tensors="pt").to(device)

response = model.generate(inputs['input_ids'], max_length=100)
# print(response)
print(tokenizer.decode(response[0]))