import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
model_id = "meta-llama/Llama-2-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# 定义一个函数来记录每层attention和ffn的执行耗时
def record_layer_times(input_ids):
    # 获取模型的transformer层
    transformer_layers = model.transformer.layers

    # 初始化一个字典来存储每层的执行耗时
    layer_times = {"attention": [], "ffn": []}

    # 遍历每一层
    for i, layer in enumerate(transformer_layers):
        # 获取attention模块和ffn模块
        attention_module = layer.attention
        ffn_module = layer.feed_forward

        # 记录attention模块的执行耗时
        start_time = time.time()
        attention_output = attention_module(input_ids)
        end_time = time.time()
        layer_times["attention"].append(end_time - start_time)

        # 记录ffn模块的执行耗时
        start_time = time.time()
        ffn_output = ffn_module(attention_output)
        end_time = time.time()
        layer_times["ffn"].append(end_time - start_time)

        # 更新输入
        input_ids = ffn_output

    return layer_times

# 输入文本
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# 记录每层的执行耗时
layer_times = record_layer_times(inputs["input_ids"])

# 打印结果
for layer_type, times in layer_times.items():
    print(f"{layer_type.capitalize()} layer times: {times}")