import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

# 加载模型和分词器
checkpoint = "/Users/sunxihao/Documents/code/llama.cpp/models/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

# 准备输入数据
prompt = "Alice and Bob"
inputs = tokenizer(prompt, return_tensors="pt")

# 修改模型结构以跳过第一层 FFN
class SkipFirstFFNModel(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model

    def forward(self, *args, **kwargs):
        # 获取原始模型的每一层
        layers = list(self.original_model.transformer.layers.children())

        # 修改第一层以跳过 FFN
        class SkipFFNLayer(torch.nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.self_attn = layer.self_attn
                self.norm1 = layer.norm1
                self.norm2 = layer.norm2
                # 跳过 FFN，直接将输入传递到下一层
                self.ffn = lambda x: x

            def forward(self, x, *args, **kwargs):
                x = self.norm1(x)
                x = self.self_attn(x, *args, **kwargs)[0]
                x = self.norm2(x)
                x = self.ffn(x)
                return x

        layers[0] = SkipFFNLayer(layers[0])

        # 重新组装模型
        self.original_model.transformer.layers = torch.nn.Sequential(*layers)
        return self.original_model(*args, **kwargs)


# 包装模型以跳过第一层 FFN
model = SkipFirstFFNModel(model)

# 运行模型
outputs = model.generate(**inputs)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)