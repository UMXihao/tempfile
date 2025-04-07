import torch
import torch.nn as nn
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
# model_name = "/home/yandong/Documents/um-data/models/Llama-2-7b-hf"
# model_name = "/home/yandong/Documents/um-data/models/Orac-mini-3B"
# model_name = "/home/yandong/Documents/um-data/models/MPT-7B-Chat"z
# model_name = "/home/yandong/Documents/um-data/models/InternLM2-chat-7B"
model_name = "/home/yandong/Documents/um-data/models/Vicuna-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# print(model)

latency_attn = []
latency_ffn = []


# 定义一个包装类，用于记录每层 self-attn 和 MLP 的执行耗时
class TimeRecorder(nn.Module):
    def __init__(self, module):
        super(TimeRecorder, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        start_time = time.time()  # 开始时间
        output = self.module(*args, **kwargs)  # 执行原始模块
        end_time = time.time()  # 结束时间
        duration = end_time - start_time  # 计算耗时
        # if self.module.__class__.__name__ == 'MptAttention':
        # if self.module.__class__.__name__ == 'LlamaSdpaAttention':
        if self.module.__class__.__name__ == 'InternLM2Attention':
            latency_attn.append(duration)
        else:
            latency_ffn.append(duration)
        # print(f"{self.module.__class__.__name__} 耗时: {duration:.6f} 秒")
        return output


'''
LLama2-7B模型结构
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
'''
# 遍历模型的每一层，为 self-attn 和 MLP 添加时间记录器
# for i in range(len(model.model.layers)):
#     layer = model.model.layers[i]
#     layer.self_attn = TimeRecorder(layer.self_attn)
#     layer.mlp = TimeRecorder(layer.mlp)

'''
MptForCausalLM(
  (transformer): MptModel(
    (wte): Embedding(50432, 4096)
    (blocks): ModuleList(
      (0-31): 32 x MptBlock(
        (norm_1): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (attn): MptAttention(
          (Wqkv): Linear(in_features=4096, out_features=12288, bias=False)
          (out_proj): Linear(in_features=4096, out_features=4096, bias=False)
        )
        (norm_2): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (ffn): MptMLP(
          (up_proj): Linear(in_features=4096, out_features=16384, bias=False)
          (act): GELU(approximate='none')
          (down_proj): Linear(in_features=16384, out_features=4096, bias=False)
        )
        (resid_attn_dropout): Dropout(p=0, inplace=False)
      )
    )
    (norm_f): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=4096, out_features=50432, bias=False)
)
'''
# for i in range(len(model.transformer.blocks)):
#     layer = model.transformer.blocks[i]
#     layer.attn = TimeRecorder(layer.attn)
#     layer.ffn = TimeRecorder(layer.ffn)

'''
InternLM2ForCausalLM(
  (model): InternLM2Model(
    (tok_embeddings): Embedding(92544, 4096, padding_idx=2)
    (layers): ModuleList(
      (0-31): 32 x InternLM2DecoderLayer(
        (attention): InternLM2Attention(
          (wqkv): Linear(in_features=4096, out_features=6144, bias=False)
          (wo): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): InternLM2DynamicNTKScalingRotaryEmbedding()
        )
        (feed_forward): InternLM2MLP(
          (w1): Linear(in_features=4096, out_features=14336, bias=False)
          (w3): Linear(in_features=4096, out_features=14336, bias=False)
          (w2): Linear(in_features=14336, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (attention_norm): InternLM2RMSNorm()
        (ffn_norm): InternLM2RMSNorm()
      )
    )
    (norm): InternLM2RMSNorm()
  )
  (output): Linear(in_features=4096, out_features=92544, bias=False)
)
'''
for i in range(len(model.model.layers)):
    layer = model.model.layers[i]
    layer.attention = TimeRecorder(layer.attention)
    layer.feed_forward = TimeRecorder(layer.feed_forward)

# 准备输入数据
prompts = ['What sits on top of the Main Building at Notre Dame?',
           'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
           'What is in front of the Notre Dame Main Building?',
           'The Basilica of the Sacred heart at Notre Dame is beside to which structure?',
           'What is the Grotto at Notre Dame?',
           'When did the Scholastic Magazine of Notre dame begin publishing?',
           "How often is Notre Dame's the Juggler published?"]
prompt = "How often is Notre Dame's the Juggler published?"
inputs = tokenizer(prompt, return_tensors='pt')
# 执行模型推理
with torch.no_grad():
    model(inputs['input_ids'])
print("attn:", latency_attn, "\nffn:", latency_ffn)
