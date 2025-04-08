import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import cosine_similarity

# 加载模型和分词器
# model_name = "/home/yandong/Documents/um-data/models/Llama-2-7b-hf"
model_name = "/home/yandong/Documents/um-data/models/Orac-mini-3B"
# model_name = "/home/yandong/Documents/um-data/models/MPT-7B-Chat"z
# model_name = "/home/yandong/Documents/um-data/models/InternLM2-chat-7B"
# model_name = "/home/yandong/Documents/um-data/models/Vicuna-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

importance_attn = []
importance_ffn = []


def cal_similarity(layer_input, layer_output):
    similarity = cosine_similarity(layer_input.view(-1), layer_output.view(-1), dim=0)
    # 进行重要性归一化处理
    norm_similarity = (similarity + 1) / 2
    return norm_similarity.mean().item()


class SimilarityRecorder(nn.Module):
    def __init__(self, module):
        super(SimilarityRecorder, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        # llama2: self_attn的输入kwargs.hidden_states, mlp的输入args
        if self.module.__class__.__name__ == 'LlamaSdpaAttention':
            layer_input = kwargs.get('hidden_states')
            output = self.module(*args, **kwargs)  # 执行原始模块
            importance_attn.append(cal_similarity(layer_input, output[0]))
        else:
            layer_input = args[0]
            output = self.module(*args, **kwargs)  # 执行原始模块
            importance_ffn.append(cal_similarity(layer_input, output))
        return output


# Llama2-7B/Orac-mini-3B
for i in range(len(model.model.layers)):
    layer = model.model.layers[i]
    layer.self_attn = SimilarityRecorder(layer.self_attn)
    layer.mlp = SimilarityRecorder(layer.mlp)

# MPT
# for i in range(len(model.transformer.blocks)):
#     layer = model.transformer.blocks[i]
#     layer.attn = TimeRecorder(layer.attn)
#     layer.ffn = TimeRecorder(layer.ffn)

# InternLM
# for i in range(len(model.model.layers)):
#     layer = model.model.layers[i]
#     layer.attention = TimeRecorder(layer.attention)
#     layer.feed_forward = TimeRecorder(layer.feed_forward)

# Vicuna
# for i in range(len(model.model.layers)):
#     layer = model.model.layers[i]
#     layer.self_attn = TimeRecorder(layer.self_attn)
#     layer.mlp = TimeRecorder(layer.mlp)

# 准备输入
prompts = ['What sits on top of the Main Building at Notre Dame?',
           'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
           'What is in front of the Notre Dame Main Building?',
           'The Basilica of the Sacred heart at Notre Dame is beside to which structure?',
           'What is the Grotto at Notre Dame?',
           'When did the Scholastic Magazine of Notre dame begin publishing?',
           "How often is Notre Dame's the Juggler published?"]
input_text = "How often is Notre Dame's the Juggler published?"
inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    model(inputs['input_ids'])

print("attn:", importance_attn, "\nffn:", importance_ffn)