#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Xihao Sun
@contact:sunxh2016@gmail.com
@version: 1.0.0
@file: head_similarity.py
@time: 4/10/25 2:31 PM

对每一层的Head重要性进行分析，即对每一层的attn进行head数量的拆分
需要查看模型的配置，确定head的数量
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import cosine_similarity

# 加载模型和分词器
model_name = "/home/yandong/Documents/um-data/models/Llama-2-7b-hf"
# model_name = "/home/yandong/Documents/um-data/models/Orac-mini-3B"
# model_name = "/home/yandong/Documents/um-data/models/Vicuna-7B"
# model_name = "/home/yandong/Documents/um-data/models/MPT-7B-Chat"
# model_name = "/home/yandong/Documents/um-data/models/InternLM2-chat-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)  # InternLM2
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)  # InternLM2

importance_head = []

hidden_state_input = []
hidden_state_attn = []


def compute_attention_head_similarity(inputs, outputs, num_heads):
    """
    计算注意力层中不同 head 的输出相似度
    :param outputs: 注意力层的输出，形状为 (batch_size, seq_len, hidden_size)
    :param num_heads: 注意力头的数量
    :return: 相似度矩阵，形状为 (num_heads, num_heads)
    """
    batch_size, seq_len, hidden_size = outputs.shape
    head_dim = hidden_size // num_heads
    # 将输出重新排列为 (batch_size, seq_len, num_heads, head_dim)
    inputs = inputs.view(batch_size, seq_len, num_heads, head_dim)
    outputs = outputs.view(batch_size, seq_len, num_heads, head_dim)
    # 调整维度以便计算相似度
    # 将张量按 num_heads 划分
    reshaped_inputs = inputs.permute(2, 0, 1, 3).reshape(num_heads, -1)
    reshaped_outputs = outputs.permute(2, 0, 1, 3).reshape(num_heads, -1)

    similarities = []
    for i in range(num_heads):
        # 获取当前头和下一个头的一维向量
        current_head = reshaped_inputs[i]
        next_head = reshaped_outputs[i]

        # 计算余弦相似度
        cos_sim = cosine_similarity(current_head.unsqueeze(0), next_head.unsqueeze(0), dim=1)
        similarities.append(cos_sim.item())

    return similarities
    # outputs = outputs.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
    # outputs = outputs.reshape(batch_size * num_heads, seq_len * head_dim)
    # # 计算余弦相似度
    # similarity_matrix = F.cosine_similarity(outputs.unsqueeze(1), outputs.unsqueeze(0), dim=-1)
    # similarity_matrix = similarity_matrix.view(batch_size, num_heads, batch_size, num_heads)
    # # 取对角线上的相似度矩阵，即同一个 batch 内的 head 之间的相似度
    # similarity_matrix = similarity_matrix.diagonal(dim1=0, dim2=2)
    # return similarity_matrix

class SimilarityRecorder(nn.Module):
    def __init__(self, module):
        super(SimilarityRecorder, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        # llama2: self_attn的输入kwargs.hidden_states, mlp的输入args
        if self.module.__class__.__name__ == 'LlamaSdpaAttention':
            output = self.module(*args, **kwargs)  # 执行原始模块
            hidden_state_input.append(kwargs.get('hidden_states'))
            hidden_state_attn.append(output[0])
        else:
            output = self.module(*args, **kwargs)  # 执行原始模块
            # hidden_state_input.append(args[0])
            # hidden_state_attn.append(output)

        # if self.module.__class__.__name__ == 'MptAttention':
        #     output = self.module(*args, **kwargs)  # 执行原始模块
        #     hidden_state_input.append(args[0])
        #     hidden_state_attn.append(output[0])
        # else:
        #     output = self.module(*args, **kwargs)  # 执行原始模块

        # if self.module.__class__.__name__ == 'InternLM2Attention':
        #     output = self.module(*args, **kwargs)  # 执行原始模块
        #     hidden_state_input.append(kwargs.get('hidden_states'))
        #     hidden_state_attn.append(output[0])
        # else:
        #     output = self.module(*args, **kwargs)  # 执行原始模块
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
from datasets import load_dataset


input_text = 'What sits on top of the Main Building at Notre Dame?'

# input_text = '''
# from typing import List
#
#
# def has_close_elements(numbers: List[float], threshold: float) -> bool:
#     """ Check if in given list of numbers, are any two numbers closer to each other than
#     given threshold.
#     >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
#     False
#     >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
#     True
#     """
# '''

# input_text = '''(CNN) It was an oddly familiar O.J. Simpson that for a little while filled TV screens across America on Thursday. A Nevada parole board decided Simpson should be freed after the former NFL star apologized for his role in a 2007 armed robbery, said he'd been a model prisoner, and promised that he'd have no conflicts if released. Simpson's appearance before the board garnered wall-to-wall coverage from cable news shows, coverage that recalled the "trial of the century," and the many months more than 20 years ago when it transfixed a nation. "I've done my time," Simpson said. "I've done it as well and as respectfully as I think anyone can." Now 70, Simpson's energy seemed little affected by his time behind bars. He was alert, engaged, and quick to smile, even letting out a hearty laugh when parole board Chairman Connie Bisbee accidentally said he was aged 90. "I feel like it," he said. Simpson has served nine years of a nine-to-33-year sentence for an armed robbery and kidnapping in Las Vegas. He is expected to be released as early as October -- and said he plans to move to his home in Florida. An entire generation of Americans has come of age since Simpson seemed an almost inescapable public figure. But for one afternoon, it felt like 1995 again. That was the year he was acquitted of murder charges in the grisly slayings of his ex-wife Nicole Brown Simpson and her friend Ronald Goldman. Thursday's parole hearing followed renewed interest in his story, which was explored last year in the award-winning documentary "O.J.: Made in America" and the FX true-crime drama "The People v. O.J. Simpson." Though it's been 22 years since the not guilty verdict, the murder trial's themes of criminal justice and race, trust in police, celebrity and domestic violence remain remarkably resonant in modern culture. "We talk about O.J. as though the story is O.J.," journalist Celia Farber says toward the end of the "Made in America" documentary. "The story is O.J. and us." Parole board vote unanimous For his part, Simpson seemed remarkably unaltered. He repeatedly avoided taking full responsibility for the Vegas crime. At one point, he said he had lived a "conflict-free life," a statement that perhaps bemused anyone whose memory stretches back more than two decades. "Juice," as he was known in his heyday, said associates misled him during the Vegas robbery and then turned on him in court. One of those associates is Tom Riccio. Simpson testified that Riccio is the one who called him, persuading him to take part in the robbery. Simpson said Thursday he regretted ever taking Riccio's call. But, according to Riccio, Simpson did a lot more than that. "He should wish he didn't make all those calls after my call," Riccio told CNN. "After he took my call he did a lot of things he shouldn't have done." Riccio added that Simpson was the one who orchestrated the robbery. "He plotted it all and gathered up men with guns." Simpson said Riccio avoided punishment by throwing him under the bus. "Unfortunately, they got a get-out-of-jail-free card when they said 'O.J. told me (to do it),'" Simpson said. "Nothing I can do about that." Sufficient remorse is not a relevant factor for parole, and Simpson ticked off several mitigating factors that made him a good candidate for release. He was discipline-free in prison, he has stable release plans, he has family and community support -- and, of course, he has no prior criminal convictions. The four parole board members voted unanimously to grant him parole, and board member Tony Corda said he was graded a "low risk to reoffend." Simpson smiled, said "thank you," and then put his head down for a few moments silently. O.J. Simpson wins parole after serving nine years for armed robbery and kidnapping https://t.co/di9NLhRSI6 https://t.co/G3ECaWR0u2 — CNN (@CNN) July 20, 2017 'My best friend' Simpson said in closing remarks that he had been a peacemaker in the prison and had been a model prisoner. "I've spent nine years making no excuses about anything. I am sorry that things turned out the way they did. I had no intent to commit a crime." The parole hearing featured testimony from Arnelle Simpson, the former football great's oldest daughter, who said her father was "my best friend and my rock." Simpson also said he has taken two "Alternative to Violence" classes, which he said was "the most important course any person in this prison can take." In addition, robbery victim Bruce Fromong testified that he had forgiven Simpson for the crime at that Las Vegas hotel room, and advocated for his release. Simpson had also been described by authorities as a model prisoner at Lovelock Correctional Center, a medium-security prison in the Nevada desert. The robbery Simpson and an associate were convicted of kidnapping, armed robbery and assault with a deadly weapon for attempting to steal pieces of Simpson sports memorabilia at gunpoint. At his 2008 sentencing, the Hall of Fame running back said he went to the room in the Palace Station Hotel & Casino in Las Vegas to reclaim family heirlooms and other personal items that had been taken from him. He also claimed he didn't know his associates were armed. "I wasn't there to hurt anybody," Simpson said. "I just wanted my personal things, and I realize now that was stupid of me. I am sorry." The case, which featured a colorful cast of seedy characters, secret recordings and a Las Vegas heist, read like a low-budget parody of "Ocean's Eleven," CNN wrote at the time Photos: The rise and fall of O.J. Simpson Photos: The rise and fall of O.J. Simpson O.J. Simpson reacts after learning he was granted parole at Lovelock Correctional Center on Thursday, July 20, in Lovelock, Nevada. Simpson is serving a nine-to-33-year prison term for a 2007 armed robbery and kidnapping conviction. Click through the gallery to see moments from the notable life of the former football and media star. Hide Caption 1 of 23 Photos: The rise and fall of O.J. Simpson As a University of Southern California running back, Simpson accepts the Heisman Trophy in December 1968. Hide Caption 2 of 23 Photos: The rise and fall of O.J. Simpson Simpson, pictured in 1974, was a running back for the Buffalo Bills from 1969 to 1977. Hide Caption 3 of 23 Photos: The rise and fall of O.J. Simpson Simpson with his wife, Marguerite Whitley, his daughter Arnelle and son Jason, circa 1974. The couple were married from 1967 to 1979. They had another daughter, Aaren, who died as a toddler in a drowning accident. Hide Caption 4 of 23 Photos: The rise and fall of O.J. Simpson Simpson in action during a Buffalo Bills game against the New York Jets. Hide Caption 5 of 23 Photos: The rise and fall of O.J. Simpson Simpson married Nicole Brown Simpson in 1985. Here the couple appears at a Los Angeles nightclub around 1976. Hide Caption 6 of 23 Photos: The rise and fall of O.J. Simpson Coach Lou Sabin and O.J. Simpson Hide Caption 7 of 23 Photos: The rise and fall of O.J. Simpson Simpson branched out into acting. He appears with Bill Murray, left, Laraine Newman and Garrett Morris in a "Saturday Night Live" skit in 1978. Hide Caption 8 of 23 Photos: The rise and fall of O.J. Simpson As a running back for the San Francisco 49ers, Simpson carries the ball against the Oakland Raiders during a preseason game circa 1978. Hide Caption 9 of 23 Photos: The rise and fall of O.J. Simpson Simpson becomes a commentator on ABC's "Monday Night Football" in the mid-'80s. He appears with Joe Namath, left, and Frank Gifford. Hide Caption 10 of 23 Photos: The rise and fall of O.J. Simpson Simpson and his children attend Nicole Brown Simpson's funeral in June 1994. Hide Caption 11 of 23 Photos: The rise and fall of O.J. Simpson Ronald Goldman was slain with Simpson's ex-wife Nicole Brown Simpson on June 12, 1994. Hide Caption 12 of 23 Photos: The rise and fall of O.J. Simpson In footage seen on TV screens around the world, police chase a white Ford Bronco with a fugitive Simpson inside on the Los Angeles freeways on June 17, 1994. The Bronco eventually returned to Simpson's home in the Brentwood section of Los Angeles, and he surrendered to police on murder charges in the deaths of his ex-wife and Ronald Goldman. Hide Caption 13 of 23 Photos: The rise and fall of O.J. Simpson Simpson's mug shot after his arrest on murder charges. Hide Caption 14 of 23 Photos: The rise and fall of O.J. Simpson Fans leave signs of support outside Simpson's house in June 1994. Hide Caption 15 of 23 Photos: The rise and fall of O.J. Simpson Lead defense attorney Johnnie Cochran Jr. and prosecutor Marcia Clark face off during a hearing in the murder trial that riveted a nation. Hide Caption 16 of 23 Photos: The rise and fall of O.J. Simpson "If it doesn't fit, you must acquit" was defense attorney Cochran's mantra during the trial. Here, Simpson tries on a leather glove tied to the crime scene at his murder trial on June 15, 1995. Hide Caption 17 of 23 Photos: The rise and fall of O.J. Simpson Simpson cheers with his attorneys F. Lee Bailey, left, and Johnnie Cochan Jr. on October 3, 1995, after being found not guilty of killing Nicole Brown Simpson and Ronald Goldman. Though cleared of criminal charges, a civil jury later slapped the former football star with a $33 million wrongful death judgment, and attorneys for the Goldman family have doggedly pursued his assets. Hide Caption 18 of 23 Photos: The rise and fall of O.J. Simpson Simpson continued to encounter legal problems, including a "road rage" trial in the Miami area in October 2001. He was found not guilty on charges stemming from a traffic altercation with another motorist. Hide Caption 19 of 23 Photos: The rise and fall of O.J. Simpson Simpson appears in court with attorneys Gabriel Grasso, left, and Yale Galanter before sentencing in the sports memorabilia case in December 2008 in Las Vegas. Simpson contended he was retrieving personal items that had been stolen from him and were being sold as memorabilia. He later accused Galanter of having a conflict of interest and failing to mount an effective defense. Hide Caption 20 of 23 Photos: The rise and fall of O.J. Simpson The Palace Station hotel room, the scene of Simpson's reported confrontation with sports memorabilia dealers, is displayed on a monitor during Simpson's trial in September 2008. Hide Caption 21 of 23 Photos: The rise and fall of O.J. Simpson Simpson embraces his sister, Carmelita Durio, while his friend Tom Scotto looks on in court after a guilty verdict was reached in October 2008. Simpson was convicted of leading a group of associates into a room at the Palace Station Hotel and Casino and using threats, guns and force to take back items from two dealers. Hide Caption 22 of 23 Photos: The rise and fall of O.J. Simpson Disgraced football star O.J. Simpson appears in court on May 13, 2013, seeking to get his robbery, assault and kidnapping convictions thrown out after spending more than four years in prison. He argued that bad legal advice led to his arrest and conviction in a confrontation with sports memorabilia dealers. His 2008 conviction came 13 years after his acquittal on murder charges in the deaths of ex-wife Nicole Brown Simpson and Ronald Goldman. Hide Caption 23 of 23 Simpson's legal team argued that the nine-to-33-year sentence did not match the crime and that it was, in fact, a form of payback for his controversial acquittal in the deaths of Brown and Goldman. Even Bruce Fromong, a victim in the robbery, agreed. "It wasn't about justice," Fromong said in "O.J.: Made in America." "They wanted the guy that got away with murder in 1994." Simpson has always denied he killed Brown and Goldman. Their families won a wrongful death civil judgment against him in 1997. At a parole hearing in 2013, Simpson said he regretted the Las Vegas kidnapping and robbery. "I just wish I had never gone to that room. I wish I had just said, 'Keep it,' and not worry about it," he said. These crawls are part of an effort to archive pages as they are created and archive the pages that they refer to. That way, as the pages that are referenced are changed or taken from the web, a link to the version that was live when the page was written will be preserved.Then the Internet Archive hopes that references to these archived pages will be put in place of a link that would be otherwise be broken, or a companion link to allow people to see what was originally intended by a page's authors.The goal is to fix all broken links on the web . Crawls of supported "No More 404" sites. Tweet with a location You can add location information to your Tweets, such as your city or precise location, from the web and via third-party applications. You always have the option to delete your Tweet location history. Learn more'''

inputs = tokenizer(input_text, return_tensors="pt")
with torch.no_grad():
    # model(inputs['input_ids'])
    model.generate(inputs['input_ids'], max_length=20)

num_heads = 32
for i in range(len(hidden_state_input)):
    importance_head.append(compute_attention_head_similarity(hidden_state_input[i], hidden_state_attn[i], num_heads))
print(importance_head)