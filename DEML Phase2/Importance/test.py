import torch
import torch.nn as nn
import time
from transformers import AutoModelForCausalLM

# 加载 LLaMA-2-7B-hf 模型
model_name = "/home/yandong/Documents/um-data/models/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)

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
        print(f"{self.module.__class__.__name__} 耗时: {duration:.6f} 秒")
        return output

# 遍历模型的每一层，为 self-attn 和 MLP 添加时间记录器
for i in range(len(model.model.layers)):
    layer = model.model.layers[i]
    layer.self_attn = TimeRecorder(layer.self_attn)
    layer.mlp = TimeRecorder(layer.mlp)

# 准备输入数据
input_ids = torch.randint(0, 100, (1, 10))  # 示例输入

# 执行模型推理
with torch.no_grad():
    model(input_ids)

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