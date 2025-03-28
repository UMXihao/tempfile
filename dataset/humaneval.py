from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained('/Users/sunxihao/Documents/code/llama.cpp/models/Llama-3.2-1B-Instruct')

# 计算每个prompt的token长度
def calculate_prompt_token_length(prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    return inputs['input_ids'].shape[1]

human_eval = load_dataset("openai_humaneval", split="test")

# DatasetDict({
#     test: Dataset({
#         features: ['task_id', 'prompt', 'canonical_solution', 'test', 'entry_point'],
#         num_rows: 164
#     })
# })
# print(human_eval[0])
# for row in human_eval:
#     row["prompt"]

token_length = []
for i in tqdm(range(len(human_eval))):
    token_length.append(calculate_prompt_token_length(human_eval[i]["prompt"]))

data = {
    'ID': token_length
}

df = pd.DataFrame(data)

# 查看数据的前几行
print("start to write file...")

# 将数据写入Excel文件
output_file = 'humaneval.xlsx'
df.to_excel(output_file, index=False, engine='openpyxl')