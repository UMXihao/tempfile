import math
from transformers import AutoModel, AutoTokenizer

origin_model = '../../models/Llama-2-7b-hf'

# 需要修改保存目录
target_dir = '../../models/Llama-2-7b-hf-P30'

tokenizer = AutoTokenizer.from_pretrained(origin_model)
tokenizer.save_pretrained(target_dir)
model = AutoModel.from_pretrained(origin_model)

# 需要修改稀疏性
sparsity = 0.1
gradient = math.ceil(sparsity * 32) * 128

for layer in model.layers:
    layer.self_attn.q_proj.weight.data[:, gradient:] = 0
    layer.self_attn.k_proj.weight.data[:, gradient:] = 0
    layer.self_attn.v_proj.weight.data[:, gradient:] = 0
    layer.self_attn.o_proj.weight.data[gradient:, :] = 0

if sparsity > 0.1:
    # 融合上个梯度的模型,需要修改
    last_sparsity = sparsity - 0.1
    last_gradient = math.ceil(last_sparsity * 32) * 128
    last_dir = '../../models/alpaca-p20'
    last_model = AutoModel.from_pretrained(origin_model)
    for idx in range(len(model.layers)):
        last_layer = last_model.layers[idx]
        layer = model.layers[idx]
        layer.self_attn.q_proj.weight.data[:, :last_gradient] = last_layer.self_attn.q_proj.weight.data[:, :last_gradient]
        layer.self_attn.k_proj.weight.data[:, :last_gradient] = last_layer.self_attn.q_proj.weight.data[:, :last_gradient]
        layer.self_attn.v_proj.weight.data[:, :last_gradient] = last_layer.self_attn.q_proj.weight.data[:, :last_gradient]
        layer.self_attn.o_proj.weight.data[:last_gradient, :] = last_layer.self_attn.o_proj.weight.data[:last_gradient, :]

model.save_pretrained(target_dir)