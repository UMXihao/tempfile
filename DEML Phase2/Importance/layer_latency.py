import torch
import torch.nn as nn
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载 LLaMA-2-7B-hf 模型
model_name = "/home/yandong/Documents/um-data/models/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

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
        if self.module.__class__.__name__ == 'LlamaSdpaAttention':
            latency_attn.append(duration)
        else:
            latency_ffn.append(duration)
        # print(f"{self.module.__class__.__name__} 耗时: {duration:.6f} 秒")
        return output


# 遍历模型的每一层，为 self-attn 和 MLP 添加时间记录器
for i in range(len(model.model.layers)):
    layer = model.model.layers[i]
    layer.self_attn = TimeRecorder(layer.self_attn)
    layer.mlp = TimeRecorder(layer.mlp)

# 准备输入数据
prompts = ['What is the Grotto at Notre Dame?',
           'What sits on top of the Main Building at Notre Dame?',
           'When did the Scholastic Magazine of Notre dame begin publishing?',
           "How often is Notre Dame's the Juggler published?",
           'What is the daily student paper at Notre Dame called?',
           'How many student news papers are found at Notre Dame?',
           'In what year did the student paper Common Sense begin publication at Notre Dame?',
           'Where is the headquarters of the Congregation of the Holy Cross?',
           'What is the primary seminary of the Congregation of the Holy Cross?',
           'What is the oldest structure at Notre Dame?',
           'What individuals live at Fatima House at Notre Dame?',
           'Which prize did Frederick Buechner create?',
           'How many BS level degrees are offered in the College of Engineering at Notre Dame?',
           'In what year was the College of Engineering at Notre Dame formed?',
           'Before the creation of the College of Engineering similar studies were carried out at which Notre Dame college?',
           'How many departments are within the Stinson-Remick Hall of Engineering?',
           'The College of Science began to offer civil engineering courses beginning at what time at Notre Dame?',
           'What entity provides help with the management of time for new students at Notre Dame?',
           'How many colleges for undergraduates are at Notre Dame?',
           'What was created at Notre Dame in 1962 to assist first year students?',
           'Which organization declared the First Year of Studies program at Notre Dame "outstanding?"',
           'The granting of Doctorate degrees first occurred in what year at Notre Dame?',
           'What type of degree is an M.Div.?',
           'Which program at Notre Dame offers a Master of Education degree?',
           'In what year was a Master of Arts course first offered at Notre Dame?',
           'Which department at Notre Dame is the only one to not offer a PhD program?',
           'What institute at Notre Dame studies  the reasons for violent conflict?',
           "What is the title of Notre Dame's Theodore Hesburgh?",
           'In what year was the Joan B. Kroc Institute for International Peace Studies founded?',
           'To whom was John B. Kroc married?',
           'What company did Ray Kroc own?'
           ]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors='pt')
    # 执行模型推理
    with torch.no_grad():
        model(inputs['input_ids'])
    print("attn:", latency_attn, "\nffn:", latency_ffn)

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
