#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Xihao Sun
@contact:sunxh2016@gmail.com
@version: 1.0.0
@file: squad.py
@time: 4/10/25 6:57 PM
"""
from tqdm import tqdm
from datasets import load_dataset
import evaluate
from llama_cpp import Llama


def load_squad():
    # 加载 SQuAD 数据集
    squad_val = load_dataset("squad", split="validation")
    return squad_val


def load_human_eval():
    human_eval = load_dataset("bigcode/humanevalpack", "python")["test"]
    return human_eval

'''
squad_val example
{
  "id": "5733be284776f41900661182",
  "title": "University_of_Notre_Dame",
  "context": "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.",
  "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
  "answers": {
    "text": [
      "Saint Bernadette Soubirous"
    ],
    "answer_start": [
      515
    ]
  }
}
'''

llm = Llama(
    # model_path="/home/yandong/Documents/um-data/models/Llama-2-7b-hf-gguf/Llama-2-7b-hf-gguf.gguf",
    model_path="/home/yandong/Documents/um-data/models/Llama-2-7b-hf-imp-gguf/Llama-2-7b-hf-imp.gguf",
    # model_path="/home/yandong/Documents/um-data/models/Llama-2-7b-hf-imp-gguf/Llama-2-7b-hf-unimp.gguf",
    n_gpu_layer=-1,
    n_ctx=4096,
    seed=1337,
    verbose=False
)


# 准备评估函数
def evaluate_model(dataset, metric):
    predictions = []
    references = []

    for i in tqdm(range(100)):
    # for i in tqdm(range(len(dataset))):
        context = dataset[i]["context"]
        question = dataset[i]["question"]
        answers = dataset[i]["answers"]
        question_id = dataset[i]["id"]

        # 构造输入文本
        input_text = f"Context: {context}\nQuestion: {question}\nAnswer: "

        # 生成回答
        output = llm(
            input_text,
            max_tokens=128,
            stop=['.', '\n'],
            echo=False
        )
        reference = {'answers': answers, 'id': question_id}
        prediction = {'prediction_text': output['choices'][0]['text'], 'id': question_id}

        references.append(reference)
        predictions.append(prediction)

    result = metric.compute(predictions=predictions, references=references)
    return result


'''
squad_metric = load("squad_v2")
predictions = [{'prediction_text': 'predicted_answer', 'id': 'question_id'}]
references = [{'answers': {'answer_start': [start_index], 'text': ['true_answer']}, 'id': 'question_id'}]
results = squad_metric.compute(predictions=predictions, references=references)
'''

'''
version 1 bug:
Expected format: {'predictions': {'id': Value(dtype='string', id=None), 'prediction_text': Value(dtype='string', id=None)}, 
'references': {'id': Value(dtype='string', id=None), 'answers': Sequence(feature={'text': Value(dtype='string', id=None), 'answer_start': Value(dtype='int32', id=None)}, length=-1, id=None)}},
Input predictions: ['\u200b\u200bThe Broncos won the Super Bowl for the third time, and first time since Super Bowl XXXII in 1998'],
Input references: [['Denver Broncos', 'Denver Broncos', 'Denver Broncos']]
'''


# data = load_squad()
# squad_metric = evaluate.load("squad")
# print(evaluate_model(data, squad_metric))

data = load_human_eval()
print(data)
# human_metric = evaluate.load("humaneval")
# print(evaluate_model(data, human_metric))

'''
single:
origin:      {'exact_match': 0.0, 'f1': 10.526315789473683}
important:   {'exact_match': 0.0, 'f1': 50.0}
unimportant: {'exact_match': 0.0, 'f1': 0.0}
100:
origin:      {'exact_match': 4.0, 'f1': 15.462865038772478}
important:   {'exact_match': 7.0, 'f1': 21.733007051428107}
unimportant: {'exact_match': 5.0, 'f1': 17.378484522013935}
all:
origin:      {'exact_match': 4.0, 'f1': 15.462865038772478}
important:   {'exact_match': 3.0, 'f1': 12.247457770987182}
unimportant: {'exact_match': 3.0, 'f1': 19.46046053190016}
'''