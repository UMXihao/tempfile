"""
start llama.cpp server
build/bin/llama-server -m /Users/sunxihao/Documents/code/llama.cpp/models/meta-llama-3-8B-Q4.gguf
"""

# origin: {'exact_match': 0.8, 'f1': 10.712394806511215} 2550.10 seconds
# mha-com1: {'exact_match': 0.7, 'f1': 9.611282249657261} 2619.58 second
# mha-com2: {'exact_match': 1.5, 'f1': 11.080792751055608} 2626.62 seconds
# mha-com3: {'exact_match': 1.1, 'f1': 10.805290867036286} 2640.59 seconds
# mlp-com1: {'exact_match': 1.0, 'f1': 10.930289111442235} 2599.92 seconds
# mlp-com2: {'exact_match': 1.4, 'f1': 10.674141385073654} 2673.28 seconds
# mlp-com3: {'exact_match': 1.4, 'f1': 9.465017859774878} 2685.24 seconds

import requests
import time
from tqdm import tqdm
from datasets import load_dataset
import evaluate

url = "http://127.0.0.1:8080/completion"
# url = "http://127.0.0.1:8080/v1/chat/completions"
headers = {"Content-Type": "application/json"}
# data = {
#     "model": "meta-llama-3-8B-Q4",
#     "messages": [{"role": "user", "content": "Hello, how are you?"}],
#     "max_tokens": 128
# }

squad_val = load_dataset("squad", split="validation")

predictions = []
references = []

start_time = time.time()
for i in tqdm(range(1000)):
    # for i in tqdm(range(len(dataset))):
    context = squad_val[i]["context"]
    question = squad_val[i]["question"]
    answers = squad_val[i]["answers"]
    question_id = squad_val[i]["id"]

    # 构造输入文本
    input_text = f"Context: {context}\nQuestion: {question}\nAnswer: "
    data = {"prompt": input_text, "n_predict": 128, "stop": "\n"}

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    reference = {'answers': answers, 'id': question_id}
    prediction = {'prediction_text': response.json().get("content"), 'id': question_id}

    references.append(reference)
    predictions.append(prediction)

result = evaluate.load("squad").compute(predictions=predictions, references=references)
print(result)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"requests: {elapsed_time:.2f} seconds")