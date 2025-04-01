import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# 加载模型和分词器
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")


# 定义自定义 forward 函数（跳过第一层 FFN）
def forward_skip_first_ffn(input_ids, attention_mask=None):
    with torch.no_grad():
        # 获取输入嵌入
        inputs_embeds = model.model.embed_tokens(input_ids)

        # 逐层处理
        for layer_idx, layer in enumerate(model.model.layers):
            # 原始 hidden_states
            hidden_states = inputs_embeds if layer_idx == 0 else hidden_states

            # 第一层跳过 FFN
            if layer_idx == 0:
                # 仅执行 Self-Attention
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
                hidden_states = layer.self_attn(
                    hidden_states,
                    attention_mask=attention_mask,
                )[0]
                hidden_states = residual + hidden_states
                # 跳过 mlp（FFN）
            else:
                # 正常执行所有操作
                hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]

        # 后续处理（最终 LayerNorm + LM Head）
        hidden_states = model.model.norm(hidden_states)
        logits = model.lm_head(hidden_states)
        return logits


# 测试
input_text = "Hello, how are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

# 原始输出
original_output = model(input_ids).logits

# 跳过第一层 FFN 的输出
modified_output = forward_skip_first_ffn(input_ids)

# 比较差异
diff = torch.mean(torch.abs(original_output - modified_output))
print(f"输出差异均值: {diff.item()}")