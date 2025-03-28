from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained('/Users/sunxihao/Documents/code/llama.cpp/models/Llama-3.2-1B-Instruct')

# 计算每个prompt的token长度
def calculate_prompt_token_length(prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    return inputs['input_ids'].shape[1]

# print(calculate_prompt_token_length(squad_train[0]["question"]))

# for data in squad_train:
#     print(calculate_prompt_token_length(data["question"]))

# 查看示例
# print(len(squad_train)) # length 87599
# {
#   "id": "5733be284776f41900661182",
#   "title": "University_of_Notre_Dame",
#   "context": "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.",
#   "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
#   "answers": {
#     "text": [
#       "Saint Bernadette Soubirous"
#     ],
#     "answer_start": [
#       515
#     ]
#   }
# }

# LLaVA-OneVision-Data 格式
# {
#   "id": "unique_id",
#   "image": "path/to/image.jpg",
#   "conversations": [
#     {
#       "from": "human",
#       "value": "Describe this image in detail."
#     },
#     {
#       "from": "gpt",
#       "value": "The image shows a sunny day with a blue sky..."
#     }
#   ]
# }

# import json

# def squad_to_llava(squad_data, output_file):
#     llava_data = []
#     for content in squad_data:
#         # 构造 LLaVA 格式的对话
#         conversation = [
#             {"from": "human", "value": content["question"]},
#             {"from": "gpt", "value": content["answers"]["text"][0]}  # 取第一个答案
#         ]
#         # 构造 LLaVA 格式的条目
#         llava_entry = {
#             "id": content["id"],
#             # "image": "dummy.jpg",  # 虚拟图像路径
#             "conversations": conversation
#         }
#         llava_data.append(llava_entry)
#
#     # 保存转换后的数据
#     with open(output_file, "w") as f:
#         json.dump(llava_data, f, indent=2)

# 执行转换
# squad_to_llava(squad_train, "squad_llava_format.json")

def squad_token_length():
    # 加载训练集和验证集
    squad_train = load_dataset("squad", split="train")
    squad_val = load_dataset("squad", split="validation")

    token_length = []
    prompt = []
    for i in tqdm(range(len(squad_train))):
        token_length.append(calculate_prompt_token_length(squad_train[i]["question"]))
        prompt.append(squad_train[i]["question"])

    data = {
        'ID': token_length,
        'prompt': prompt
    }
    df = pd.DataFrame(data)
    # 查看数据的前几行
    print("start to write file...")
    # 将数据写入Excel文件
    output_file = 'squad.xlsx'
    df.to_excel(output_file, index=False, engine='openpyxl')

# squad_token_length()

def human_token_length():
    human_eval = load_dataset("openai_humaneval", split="test")
    token_length = []
    prompt = []
    for i in tqdm(range(len(human_eval))):
        token_length.append(calculate_prompt_token_length(human_eval[i]["prompt"]))
        prompt.append(human_eval[i]["prompt"])

    data = {
        'ID': token_length,
        'prompt': prompt
    }
    df = pd.DataFrame(data)
    # 查看数据的前几行
    print("start to write file...")
    # 将数据写入Excel文件
    output_file = 'humaneval.xlsx'
    df.to_excel(output_file, index=False, engine='openpyxl')

# human_token_length()

def mbpp_token_length():
    dataset = load_dataset('mbpp', split='test')

    token_length = []
    prompt = []
    for i in tqdm(range(len(dataset))):
        token_length.append(calculate_prompt_token_length(dataset[i]["text"]))
        prompt.append(dataset[i]["text"])

    data = {
        'ID': token_length,
        'prompt': prompt
    }
    df = pd.DataFrame(data)
    # 查看数据的前几行
    print("start to write file...")
    # 将数据写入Excel文件
    output_file = 'mbpp.xlsx'
    df.to_excel(output_file, index=False, engine='openpyxl')

mbpp_token_length()

# arc_c = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='train')
# arc_e = load_dataset('allenai/ai2_arc', 'ARC-Easy', split='train')
def arc_token_length(dataset, name):
    token_length = []
    prompt = []
    for i in tqdm(range(len(dataset))):
        token_length.append(calculate_prompt_token_length(dataset[i]["question"]))
        prompt.append(dataset[i]["question"])

    data = {
        'ID': token_length,
        'prompt': prompt
    }
    df = pd.DataFrame(data)
    # 查看数据的前几行
    print("start to write file...")
    # 将数据写入Excel文件
    output_file = name + '.xlsx'
    df.to_excel(output_file, index=False, engine='openpyxl')

# arc_token_length(arc_c, 'arc-c')
# arc_token_length(arc_e, 'arc-e')

def trivia_qa_token_length():
    dataset = load_dataset('mandarjoshi/trivia_qa', 'rc.wikipedia')
    print(dataset)

# trivia_qa_token_length()