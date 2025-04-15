import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# 加载模型和分词器
model_name = "/home/yandong/Documents/um-data/models/Llama-2-7b-hf"
# model_name = "/home/yandong/Documents/um-data/models/Llama-2-13b-chat-hf"
# model_name = "/home/yandong/Documents/um-data/models/Orac-mini-3B"
# model_name = "/home/yandong/Documents/um-data/models/MPT-7B-Chat"
# model_name = "/home/yandong/Documents/um-data/models/InternLM2-chat-7B"
# model_name = "/home/yandong/Documents/um-data/models/Vicuna-7B"

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompts = ['What sits on top of the Main Building at Notre Dame?',
           'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
           'What is in front of the Notre Dame Main Building?',
           'The Basilica of the Sacred heart at Notre Dame is beside to which structure?',
           'What is the Grotto at Notre Dame?',
           'When did the Scholastic Magazine of Notre dame begin publishing?',
           "How often is Notre Dame's the Juggler published?"]
from datasets import load_dataset
squad_val = load_dataset("squad", split="validation")
context = squad_val[2]["context"]
question = squad_val[2]["question"]
answers = squad_val[2]["answers"]
input_text = f"Context: {context}\nQuestion: {question}\nAnswer: "
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
    # outputs = model.generate(inputs['input_ids'], max_length=20)
    # outputs = model.generate(inputs['input_ids'], max_length=20, return_dict_in_generate=True, output_hidden_states=True)

    # GenerateDecoderOnlyOutput(sequences=tensor([[1, 1128, 4049, 338, 24337, 360, 420, 29915, 29879, 278,
    #                                              12028, 29887, 1358, 6369, 29973, 13, 3664, 276, 360, 420]]),
    #                           scores=None, logits=None, attentions=None, hidden_states=

    hidden_states = outputs.hidden_states  # 包含每一层的输入输出
    # print(hidden_states[0].shape)
    # print(hidden_states[1].shape)
    # print(hidden_states[32].shape)
    # print("output:", outputs.keys())


# 遍历每一层
for i in range(1, len(hidden_states) - 1):
    input_layer = hidden_states[i]
    output_layer = hidden_states[i + 1]
    # print("layer", i, " ", input_layer, output_layer)
    similarity = compute_cosine_similarity(input_layer, output_layer)
    print(similarity)
