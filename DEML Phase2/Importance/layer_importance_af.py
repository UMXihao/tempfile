import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import cosine_similarity

# 加载模型和分词器
model_name = "/home/yandong/Documents/um-data/models/Llama-2-7b-hf"
# model_name = "/home/yandong/Documents/um-data/models/Orac-mini-3B"
# model_name = "/home/yandong/Documents/um-data/models/MPT-7B-Chat"
# model_name = "/home/yandong/Documents/um-data/models/InternLM2-chat-7B"
# model_name = "/home/yandong/Documents/um-data/models/Vicuna-7B"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)  # InternLM2
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)  # InternLM2

importance_attn = []
importance_ffn = []

hidden_state_attn = []
hidden_state_ffn = []


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
            output = self.module(*args, **kwargs)  # 执行原始模块
            print(kwargs.get('hidden_states').shape)
            importance_attn.append(cal_similarity(kwargs.get('hidden_states'), output[0]))
            hidden_state_attn.append(output[0])
        else:
            output = self.module(*args, **kwargs)  # 执行原始模块
            importance_ffn.append(cal_similarity(args[0], output))
            hidden_state_ffn.append(output)

        # if self.module.__class__.__name__ == 'MptAttention':
        #     output = self.module(*args, **kwargs)  # 执行原始模块
        #     importance_attn.append(cal_similarity(args[0], output[0]))
        # else:
        #     output = self.module(*args, **kwargs)  # 执行原始模块
        #     importance_ffn.append(cal_similarity(args[0], output[0]))

        # if self.module.__class__.__name__ == 'InternLM2Attention':
        #     output = self.module(*args, **kwargs)  # 执行原始模块
        #     importance_attn.append(cal_similarity(kwargs.get('hidden_states'), output[0]))
        # else:
        #     output = self.module(*args, **kwargs)  # 执行原始模块
        #     importance_ffn.append(cal_similarity(args[0], output))
        return output


# Llama2-7B/Orac-mini-3B/Vicuna
for i in range(len(model.model.layers)):
    layer = model.model.layers[i]
    layer.self_attn = SimilarityRecorder(layer.self_attn)
    layer.mlp = SimilarityRecorder(layer.mlp)

# MPT
# for i in range(len(model.transformer.blocks)):
#     layer = model.transformer.blocks[i]
#     layer.attn = SimilarityRecorder(layer.attn)
#     layer.ffn = SimilarityRecorder(layer.ffn)

# InternLM
# for i in range(len(model.model.layers)):
#     layer = model.model.layers[i]
#     layer.attention = SimilarityRecorder(layer.attention)
#     layer.feed_forward = SimilarityRecorder(layer.feed_forward)

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

# print("attn:", importance_attn, "\nffn:", importance_ffn)

# for i in range(1, len(hidden_state_attn) - 1):
#     input_layer = hidden_state_attn[i]
#     output_layer = hidden_state_attn[i + 1]
#     similarity = cal_similarity(input_layer, output_layer)
#     print(similarity)

# for i in range(1, len(hidden_state_ffn) - 1):
#     input_layer = hidden_state_ffn[i]
#     output_layer = hidden_state_ffn[i + 1]
#     similarity = cal_similarity(input_layer, output_layer)
#     print(similarity)