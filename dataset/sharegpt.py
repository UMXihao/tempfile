from datasets import load_dataset

# 加载训练集和验证集
dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")
print(dataset[0])
# anon8231489123/ShareGPT_Vicuna_unfiltered