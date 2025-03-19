#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Dong
@contact:liuyandong1988@gmail.com
@version: 1.0.0
@file: structure.py
@time: 1/16/25 11:21 AM
[INFO|configuration_utils.py:746] 2025-01-16 11:18:07,604 >> Model config LlamaConfig {
  "_name_or_path": "/home/yandong/Documents/um-data/sunxh/PycharmProjects/models/Llama-2-7b-hf-P10",
  "architectures": [
    "LlamaModel"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "sparsity": 0.1,
  "tie_word_embeddings": false,
  "torch_dtype": "float32",
  "transformers_version": "4.46.1",
  "use_cache": true,
  "vocab_size": 32000
}

[INFO|tokenization_utils_base.py:2209] 2025-01-16 11:18:07,605 >> loading file tokenizer.model
[INFO|tokenization_utils_base.py:2209] 2025-01-16 11:18:07,605 >> loading file tokenizer.json
[INFO|tokenization_utils_base.py:2209] 2025-01-16 11:18:07,606 >> loading file added_tokens.json
[INFO|tokenization_utils_base.py:2209] 2025-01-16 11:18:07,606 >> loading file special_tokens_map.json
[INFO|tokenization_utils_base.py:2209] 2025-01-16 11:18:07,606 >> loading file tokenizer_config.json
[INFO|2025-01-16 11:18:07] llamafactory.data.template:157 >> Add pad token: </s>
[INFO|2025-01-16 11:18:07] llamafactory.data.loader:157 >> Loading dataset identity.json...
training example:
input_ids:
[1, 518, 25580, 29962, 7251, 518, 29914, 25580, 29962, 15043, 29991, 306, 626, 8620, 978, 11656, 385, 319, 29902, 20255, 8906, 491, 8620, 8921, 27243, 1128, 508, 306, 6985, 366, 9826, 29973, 2]
inputs:
<s> [INST] hi [/INST] Hello! I am {{name}}, an AI assistant developed by {{author}}. How can I assist you today?</s>
label_ids:
[-100, -100, -100, -100, -100, -100, -100, -100, -100, 15043, 29991, 306, 626, 8620, 978, 11656, 385, 319, 29902, 20255, 8906, 491, 8620, 8921, 27243, 1128, 508, 306, 6985, 366, 9826, 29973, 2]
labels:
Hello! I am {{name}}, an AI assistant developed by {{author}}. How can I assist you today?</s>
[INFO|configuration_utils.py:677] 2025-01-16 11:18:08,633 >> loading configuration file /home/yandong/Documents/um-data/sunxh/PycharmProjects/models/Llama-2-7b-hf-P10/config.json
[INFO|configuration_utils.py:746] 2025-01-16 11:18:08,634 >> Model config LlamaConfig {
  "_name_or_path": "/home/yandong/Documents/um-data/sunxh/PycharmProjects/models/Llama-2-7b-hf-P10",
  "architectures": [
    "LlamaModel"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "sparsity": 0.1,
  "tie_word_embeddings": false,
  "torch_dtype": "float32",
  "transformers_version": "4.46.1",
  "use_cache": true,
  "vocab_size": 32000
}

[INFO|modeling_utils.py:3935] 2025-01-16 11:18:08,660 >> loading weights file /home/yandong/Documents/um-data/sunxh/PycharmProjects/models/Llama-2-7b-hf-P10/model.safetensors.index.json
[INFO|modeling_utils.py:1671] 2025-01-16 11:18:08,660 >> Instantiating LlamaForCausalLM model under default dtype torch.bfloat16.
[INFO|configuration_utils.py:1096] 2025-01-16 11:18:08,661 >> Generate config GenerationConfig {
  "bos_token_id": 1,
  "eos_token_id": 2
}

Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:33<00:00,  5.51s/it]
[INFO|modeling_utils.py:4801] 2025-01-16 11:18:45,731 >> All model checkpoint weights were used when initializing LlamaForCausalLM.

[WARNING|modeling_utils.py:4803] 2025-01-16 11:18:45,731 >> Some weights of LlamaForCausalLM were not initialized from the model checkpoint at /home/yandong/Documents/um-data/sunxh/PycharmProjects/models/Llama-2-7b-hf-P10 and are newly initialized: ['lm_head.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
[INFO|modeling_utils.py:4273] 2025-01-16 11:18:45,734 >> Generation config file not found, using a generation config created from the model config.
[INFO|2025-01-16 11:18:46] llamafactory.model.model_utils.checkpointing:157 >> Gradient checkpointing enabled.
[INFO|2025-01-16 11:18:46] llamafactory.model.model_utils.attention:157 >> Using torch SDPA for faster training and inference.
[INFO|2025-01-16 11:18:46] llamafactory.model.adapter:157 >> Upcasting trainable params to float32.
[INFO|2025-01-16 11:18:46] llamafactory.model.adapter:157 >> Fine-tuning method: LoRA
[INFO|2025-01-16 11:18:46] llamafactory.model.model_utils.misc:157 >> Found linear modules: down_proj,k_proj,gate_proj,up_proj,q_proj,o_proj,v_proj
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): lora.Linear(
            (base_layer): Linear(in_features=4096, out_features=4096, bias=False)
            (lora_dropout): ModuleDict(
              (default): Identity()
            )
            (lora_A): ModuleDict(
              (default): Linear(in_features=4096, out_features=8, bias=False)
            )
            (lora_B): ModuleDict(
              (default): Linear(in_features=8, out_features=4096, bias=False)
            )
            (lora_embedding_A): ParameterDict()
            (lora_embedding_B): ParameterDict()
            (lora_magnitude_vector): ModuleDict()
          )
          (k_proj): lora.Linear(
            (base_layer): Linear(in_features=4096, out_features=4096, bias=False)
            (lora_dropout): ModuleDict(
              (default): Identity()
            )
            (lora_A): ModuleDict(
              (default): Linear(in_features=4096, out_features=8, bias=False)
            )
            (lora_B): ModuleDict(
              (default): Linear(in_features=8, out_features=4096, bias=False)
            )
            (lora_embedding_A): ParameterDict()
            (lora_embedding_B): ParameterDict()
            (lora_magnitude_vector): ModuleDict()
          )
          (v_proj): lora.Linear(
            (base_layer): Linear(in_features=4096, out_features=4096, bias=False)
            (lora_dropout): ModuleDict(
              (default): Identity()
            )
            (lora_A): ModuleDict(
              (default): Linear(in_features=4096, out_features=8, bias=False)
            )
            (lora_B): ModuleDict(
              (default): Linear(in_features=8, out_features=4096, bias=False)
            )
            (lora_embedding_A): ParameterDict()
            (lora_embedding_B): ParameterDict()
            (lora_magnitude_vector): ModuleDict()
          )
          (o_proj): lora.Linear(
            (base_layer): Linear(in_features=4096, out_features=4096, bias=False)
            (lora_dropout): ModuleDict(
              (default): Identity()
            )
            (lora_A): ModuleDict(
              (default): Linear(in_features=4096, out_features=8, bias=False)
            )
            (lora_B): ModuleDict(
              (default): Linear(in_features=8, out_features=4096, bias=False)
            )
            (lora_embedding_A): ParameterDict()
            (lora_embedding_B): ParameterDict()
            (lora_magnitude_vector): ModuleDict()
          )
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): lora.Linear(
            (base_layer): Linear(in_features=4096, out_features=11008, bias=False)
            (lora_dropout): ModuleDict(
              (default): Identity()
            )
            (lora_A): ModuleDict(
              (default): Linear(in_features=4096, out_features=8, bias=False)
            )
            (lora_B): ModuleDict(
              (default): Linear(in_features=8, out_features=11008, bias=False)
            )
            (lora_embedding_A): ParameterDict()
            (lora_embedding_B): ParameterDict()
            (lora_magnitude_vector): ModuleDict()
          )
          (up_proj): lora.Linear(
            (base_layer): Linear(in_features=4096, out_features=11008, bias=False)
            (lora_dropout): ModuleDict(
              (default): Identity()
            )
            (lora_A): ModuleDict(
              (default): Linear(in_features=4096, out_features=8, bias=False)
            )
            (lora_B): ModuleDict(
              (default): Linear(in_features=8, out_features=11008, bias=False)
            )
            (lora_embedding_A): ParameterDict()
            (lora_embedding_B): ParameterDict()
            (lora_magnitude_vector): ModuleDict()
          )
          (down_proj): lora.Linear(
            (base_layer): Linear(in_features=11008, out_features=4096, bias=False)
            (lora_dropout): ModuleDict(
              (default): Identity()
            )
            (lora_A): ModuleDict(
              (default): Linear(in_features=11008, out_features=8, bias=False)
            )
            (lora_B): ModuleDict(
              (default): Linear(in_features=8, out_features=4096, bias=False)
            )
            (lora_embedding_A): ParameterDict()
            (lora_embedding_B): ParameterDict()
            (lora_magnitude_vector): ModuleDict()
          )
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
"""

'''
FinetuningArguments(use_swanlab=False, swanlab_project='llamafactory', swanlab_workspace=None, swanlab_run_name=None, 
swanlab_mode='cloud', swanlab_api_key=None, use_badam=False, badam_mode='layer', badam_start_block=None,
badam_switch_mode='ascending', badam_switch_interval=50, badam_update_ratio=0.05, badam_mask_mode='adjacent',
badam_verbose=0, use_galore=False, galore_target=['all'], galore_rank=16, galore_update_interval=200, galore_scale=0.25, 
galore_proj_type='std', galore_layerwise=False, pref_beta=0.1, pref_ftx=0.0, pref_loss='sigmoid', dpo_label_smoothing=0.0,
kto_chosen_weight=1.0, kto_rejected_weight=1.0, simpo_gamma=0.5, ppo_buffer_size=1, ppo_epochs=4, ppo_score_norm=False, 
ppo_target=6.0, ppo_whiten_rewards=False, ref_model=None, ref_model_adapters=None, ref_model_quantization_bit=None,
reward_model=None, reward_model_adapters=None, reward_model_quantization_bit=None, reward_model_type='lora',
additional_target=None, lora_alpha=16, lora_dropout=0, lora_rank=8, lora_target=['all'], loraplus_lr_ratio=None,
loraplus_lr_embedding=1e-06, use_rslora=False, use_dora=False, pissa_init=False, pissa_iter=16, pissa_convert=False,
create_new_adapter=False, freeze_trainable_layers=2, freeze_trainable_modules=['all'], freeze_extra_modules=None, 
pure_bf16=False, stage='sft', finetuning_type='lora', use_llama_pro=False, use_adam_mini=False, freeze_vision_tower=True, 
train_mm_proj_only=False, compute_accuracy=False, disable_shuffling=False, plot_loss=True, include_effective_tokens_per_second=False)
'''
'''
/home/yandong/anaconda3/envs/llamaFactory/lib/python3.11/site-packages/transformers/optimization.py
'''

"""
[INFO|2025-01-18 10:39:05] parser.py:355 >> Process rank: 0, device: cuda:0, n_gpu: 1, distributed training: False, compute dtype: torch.bfloat16

[INFO|2025-01-18 10:39:05] configuration_utils.py:677 >> loading configuration file /home/yandong/Documents/um-data/sunxh/PycharmProjects/models/Llama-2-7b-hf-P10/config.json

[INFO|2025-01-18 10:39:05] configuration_utils.py:746 >> Model config LlamaConfig {
  "_name_or_path": "/home/yandong/Documents/um-data/sunxh/PycharmProjects/models/Llama-2-7b-hf-P10",
  "architectures": [
    "LlamaModel"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "sparsity": 0.1,
  "tie_word_embeddings": false,
  "torch_dtype": "float32",
  "transformers_version": "4.46.1",
  "use_cache": true,
  "vocab_size": 32000
}


[INFO|2025-01-18 10:39:05] tokenization_utils_base.py:2209 >> loading file tokenizer.model

[INFO|2025-01-18 10:39:05] tokenization_utils_base.py:2209 >> loading file tokenizer.json

[INFO|2025-01-18 10:39:05] tokenization_utils_base.py:2209 >> loading file added_tokens.json

[INFO|2025-01-18 10:39:05] tokenization_utils_base.py:2209 >> loading file special_tokens_map.json

[INFO|2025-01-18 10:39:05] tokenization_utils_base.py:2209 >> loading file tokenizer_config.json

[INFO|2025-01-18 10:39:05] configuration_utils.py:677 >> loading configuration file /home/yandong/Documents/um-data/sunxh/PycharmProjects/models/Llama-2-7b-hf-P10/config.json

[INFO|2025-01-18 10:39:05] configuration_utils.py:746 >> Model config LlamaConfig {
  "_name_or_path": "/home/yandong/Documents/um-data/sunxh/PycharmProjects/models/Llama-2-7b-hf-P10",
  "architectures": [
    "LlamaModel"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "sparsity": 0.1,
  "tie_word_embeddings": false,
  "torch_dtype": "float32",
  "transformers_version": "4.46.1",
  "use_cache": true,
  "vocab_size": 32000
}


[INFO|2025-01-18 10:39:05] tokenization_utils_base.py:2209 >> loading file tokenizer.model

[INFO|2025-01-18 10:39:05] tokenization_utils_base.py:2209 >> loading file tokenizer.json

[INFO|2025-01-18 10:39:05] tokenization_utils_base.py:2209 >> loading file added_tokens.json

[INFO|2025-01-18 10:39:05] tokenization_utils_base.py:2209 >> loading file special_tokens_map.json

[INFO|2025-01-18 10:39:05] tokenization_utils_base.py:2209 >> loading file tokenizer_config.json

[INFO|2025-01-18 10:39:05] logging.py:157 >> Add pad token: </s>

[INFO|2025-01-18 10:39:05] logging.py:157 >> Loading dataset identity.json...

[INFO|2025-01-18 10:39:06] configuration_utils.py:677 >> loading configuration file /home/yandong/Documents/um-data/sunxh/PycharmProjects/models/Llama-2-7b-hf-P10/config.json

[INFO|2025-01-18 10:39:06] configuration_utils.py:746 >> Model config LlamaConfig {
  "_name_or_path": "/home/yandong/Documents/um-data/sunxh/PycharmProjects/models/Llama-2-7b-hf-P10",
  "architectures": [
    "LlamaModel"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "sparsity": 0.1,
  "tie_word_embeddings": false,
  "torch_dtype": "float32",
  "transformers_version": "4.46.1",
  "use_cache": true,
  "vocab_size": 32000
}


[INFO|2025-01-18 10:39:06] modeling_utils.py:3935 >> loading weights file /home/yandong/Documents/um-data/sunxh/PycharmProjects/models/Llama-2-7b-hf-P10/model.safetensors.index.json

[INFO|2025-01-18 10:39:06] modeling_utils.py:1671 >> Instantiating LlamaForCausalLM model under default dtype torch.bfloat16.

[INFO|2025-01-18 10:39:06] configuration_utils.py:1096 >> Generate config GenerationConfig {
  "bos_token_id": 1,
  "eos_token_id": 2
}


[INFO|2025-01-18 10:39:35] modeling_utils.py:4801 >> All model checkpoint weights were used when initializing LlamaForCausalLM.


[WARNING|2025-01-18 10:39:35] modeling_utils.py:4803 >> Some weights of LlamaForCausalLM were not initialized from the model checkpoint at /home/yandong/Documents/um-data/sunxh/PycharmProjects/models/Llama-2-7b-hf-P10 and are newly initialized: ['lm_head.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

[INFO|2025-01-18 10:39:35] modeling_utils.py:4273 >> Generation config file not found, using a generation config created from the model config.

[INFO|2025-01-18 10:39:35] logging.py:157 >> Gradient checkpointing enabled.

[INFO|2025-01-18 10:39:35] logging.py:157 >> Using torch SDPA for faster training and inference.

[INFO|2025-01-18 10:39:35] logging.py:157 >> Upcasting trainable params to float32.

[INFO|2025-01-18 10:39:35] logging.py:157 >> Fine-tuning method: LoRA

[INFO|2025-01-18 10:39:35] logging.py:157 >> Found linear modules: k_proj,v_proj,o_proj,q_proj,down_proj,gate_proj,up_proj

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.0.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[-0.0062, -0.0148, -0.0022,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0142, -0.0043,  0.0028,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0146,  0.0126,  0.0005,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0013,  0.0109, -0.0003,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0256,  0.0102,  0.0032,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0134, -0.0066,  0.0018,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[ 1.0071e-02, -1.2741e-03, -1.2268e-02,  ...,  1.4709e-02,
          8.4839e-03, -3.2959e-03],
        [-8.8501e-04,  8.1787e-03,  5.7068e-03,  ...,  1.3367e-02,
          1.4648e-02, -5.0354e-03],
        [ 7.1106e-03, -9.2773e-03,  9.2773e-03,  ..., -2.5177e-03,
          5.6458e-03, -1.4893e-02],
        ...,
        [-2.5635e-03,  7.6599e-03, -7.2632e-03,  ..., -6.0120e-03,
         -1.2512e-02,  1.3611e-02],
        [ 1.0742e-02, -4.7684e-04, -6.0120e-03,  ..., -8.1062e-05,
         -6.7520e-04,  1.3123e-02],
        [-7.2632e-03, -5.6839e-04,  1.3123e-03,  ..., -1.0925e-02,
         -6.3171e-03, -3.7231e-03]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.0.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[-0.0162,  0.0079, -0.0013,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0192,  0.0015,  0.0036,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0236, -0.0217,  0.0017,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0128, -0.0007, -0.0008,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0056,  0.0173, -0.0032,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0037, -0.0021,  0.0013,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.0.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[-0.0079, -0.0031,  0.0143,  ...,  0.0131, -0.0116,  0.0101],
        [-0.0003,  0.0150,  0.0029,  ..., -0.0129, -0.0143,  0.0026],
        [-0.0051,  0.0079,  0.0125,  ...,  0.0121,  0.0099, -0.0110],
        ...,
        [-0.0146,  0.0023, -0.0022,  ..., -0.0053,  0.0047,  0.0047],
        [-0.0028,  0.0129, -0.0148,  ...,  0.0063, -0.0032, -0.0080],
        [-0.0039,  0.0141,  0.0096,  ..., -0.0023, -0.0005,  0.0111]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.0.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.0.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[ 0.0008, -0.0006,  0.0019,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0069, -0.0005, -0.0077,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0018,  0.0096,  0.0010,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0063, -0.0057,  0.0103,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0031,  0.0048, -0.0010,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0001,  0.0025,  0.0056,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0076, -0.0041, -0.0150,  ..., -0.0047,  0.0089, -0.0033],
        [-0.0029, -0.0049,  0.0006,  ..., -0.0098, -0.0106, -0.0139],
        [ 0.0083, -0.0128, -0.0099,  ...,  0.0129,  0.0077,  0.0012],
        ...,
        [-0.0085, -0.0094, -0.0114,  ..., -0.0120,  0.0011, -0.0048],
        [ 0.0053,  0.0099,  0.0150,  ...,  0.0064, -0.0062, -0.0101],
        [ 0.0025, -0.0083,  0.0109,  ..., -0.0034,  0.0084, -0.0004]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.0.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[-1.6212e-05, -1.9226e-03,  4.8828e-03,  ...,  5.9204e-03,
          3.4485e-03, -9.5215e-03],
        [ 2.7618e-03,  1.8463e-03, -1.2970e-03,  ..., -1.0300e-03,
          1.8082e-03,  6.2561e-03],
        [ 2.3346e-03, -2.7275e-04,  9.2697e-04,  ..., -1.6556e-03,
         -5.7373e-03, -6.3705e-04],
        ...,
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00]], device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.0.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[-1.3184e-02,  1.0254e-02,  1.5320e-02,  ..., -4.0283e-03,
         -7.5073e-03,  1.4221e-02],
        [ 5.3406e-03, -8.6670e-03,  2.1057e-03,  ..., -9.6436e-03,
          1.4038e-02, -1.2329e-02],
        [ 1.0986e-02, -1.0681e-02, -7.2937e-03,  ..., -2.8229e-04,
         -4.4556e-03,  3.2663e-05],
        ...,
        [ 5.7983e-03, -6.6833e-03,  3.5400e-03,  ...,  1.4771e-02,
          3.5248e-03,  1.2390e-02],
        [-4.9133e-03, -1.2817e-02, -1.9531e-03,  ..., -4.3335e-03,
          1.2939e-02, -7.3242e-04],
        [ 5.4932e-03, -8.1635e-04,  1.1597e-02,  ...,  2.5482e-03,
          6.9580e-03,  6.4087e-04]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.0.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.1.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[-0.0125,  0.0073, -0.0381,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0006, -0.0082,  0.0079,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0306,  0.0325,  0.0205,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0002,  0.0018,  0.0036,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0021, -0.0038, -0.0042,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0003,  0.0048,  0.0067,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.1.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[-8.2016e-05, -7.8735e-03,  3.7384e-03,  ...,  6.1646e-03,
         -1.3184e-02,  4.6997e-03],
        [ 2.9755e-03, -1.2451e-02,  3.1891e-03,  ..., -1.5625e-02,
         -1.4954e-02,  9.5825e-03],
        [-6.2866e-03,  6.9580e-03,  1.3794e-02,  ...,  1.3367e-02,
          1.3062e-02, -1.0010e-02],
        ...,
        [-1.0376e-02, -2.8839e-03, -9.0942e-03,  ...,  4.4861e-03,
          5.7678e-03,  1.0681e-02],
        [ 6.6833e-03,  3.9673e-03,  4.1199e-03,  ...,  1.0254e-02,
          1.0620e-02, -1.6479e-03],
        [ 2.3651e-03, -6.8665e-03, -7.5989e-03,  ..., -1.5259e-02,
         -9.8877e-03,  1.5137e-02]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.1.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.1.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[-0.0248, -0.0025,  0.0383,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0295,  0.0046, -0.0114,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0250,  0.0294, -0.0649,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0111,  0.0189, -0.0015,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0080, -0.0192,  0.0040,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0080,  0.0147,  0.0007,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.1.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[-0.0028,  0.0070, -0.0082,  ...,  0.0085, -0.0089,  0.0151],
        [-0.0094, -0.0126, -0.0020,  ...,  0.0018,  0.0120,  0.0045],
        [ 0.0091, -0.0152,  0.0025,  ..., -0.0106,  0.0077,  0.0137],
        ...,
        [ 0.0009, -0.0142, -0.0006,  ...,  0.0018, -0.0063,  0.0153],
        [-0.0030,  0.0017,  0.0135,  ..., -0.0156,  0.0082,  0.0058],
        [-0.0139, -0.0039,  0.0038,  ..., -0.0012, -0.0138, -0.0105]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.1.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.1.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[ 0.0068, -0.0084, -0.0041,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0036,  0.0024, -0.0024,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0059, -0.0118,  0.0109,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0053,  0.0027,  0.0046,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0063, -0.0056,  0.0094,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0010,  0.0016,  0.0042,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.1.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[-1.1169e-02, -1.7853e-03, -7.9956e-03,  ..., -5.8594e-03,
          4.0283e-03,  3.9062e-03],
        [-4.8523e-03,  1.3733e-02,  4.8523e-03,  ..., -2.6855e-03,
         -8.5449e-03, -1.5442e-02],
        [-4.9744e-03,  6.6528e-03, -3.6469e-03,  ...,  2.0599e-03,
          4.6387e-03, -5.5847e-03],
        ...,
        [-3.2959e-03, -8.8120e-04, -7.0496e-03,  ..., -1.0803e-02,
         -4.5166e-03,  7.1716e-03],
        [ 6.4392e-03, -5.0354e-03, -9.4604e-03,  ..., -1.2939e-02,
         -8.9111e-03,  2.0447e-03],
        [-1.1353e-02,  5.2929e-05,  2.5177e-03,  ..., -7.2632e-03,
          9.6436e-03,  1.2146e-02]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.1.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.1.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[ 0.0048, -0.0194,  0.0142,  ..., -0.0050, -0.0001,  0.0003],
        [-0.0030, -0.0054, -0.0132,  ..., -0.0007, -0.0027,  0.0032],
        [ 0.0223,  0.0126, -0.0120,  ...,  0.0047,  0.0020, -0.0022],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.1.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0014, -0.0049, -0.0029,  ...,  0.0123,  0.0076, -0.0028],
        [ 0.0076,  0.0082, -0.0082,  ..., -0.0059,  0.0132, -0.0023],
        [-0.0063,  0.0009, -0.0058,  ..., -0.0019,  0.0066,  0.0058],
        ...,
        [-0.0117, -0.0045,  0.0083,  ...,  0.0112,  0.0075, -0.0101],
        [-0.0045, -0.0135,  0.0028,  ...,  0.0067, -0.0046, -0.0052],
        [-0.0058, -0.0104, -0.0039,  ...,  0.0140, -0.0143, -0.0005]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.1.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.2.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[-0.0233, -0.0091,  0.0077,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0148,  0.0107, -0.0374,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0143,  0.0330, -0.0256,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0476,  0.0135, -0.0226,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0024, -0.0093,  0.0017,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0066, -0.0200,  0.0253,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.2.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[ 1.2695e-02, -1.5381e-02, -1.6937e-03,  ...,  8.1787e-03,
         -1.4465e-02,  1.8463e-03],
        [-8.3618e-03,  1.1719e-02,  7.9346e-03,  ...,  1.0620e-02,
         -1.5442e-02,  1.2695e-02],
        [-1.4038e-02,  1.0376e-02,  1.1368e-03,  ..., -1.2146e-02,
         -6.2866e-03, -1.0824e-04],
        ...,
        [-1.2878e-02, -9.8267e-03, -1.1169e-02,  ..., -5.1575e-03,
         -8.2397e-03, -1.5198e-02],
        [ 3.5706e-03, -1.3184e-02,  9.8267e-03,  ...,  1.3306e-02,
         -4.6387e-03,  1.2695e-02],
        [ 1.3062e-02,  1.0315e-02, -1.1841e-02,  ...,  9.7046e-03,
          1.8239e-05, -1.1292e-02]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.2.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.2.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[-0.0011,  0.0087, -0.0071,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0186, -0.0118, -0.0044,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0135,  0.0128,  0.0354,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0339,  0.0491, -0.0284,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0061, -0.0123, -0.0007,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0059, -0.0156, -0.0045,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.2.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0071, -0.0073, -0.0073,  ..., -0.0091, -0.0081, -0.0121],
        [-0.0079, -0.0154,  0.0092,  ...,  0.0074, -0.0087, -0.0002],
        [-0.0057,  0.0106, -0.0001,  ...,  0.0001,  0.0137, -0.0090],
        ...,
        [ 0.0061,  0.0001, -0.0045,  ...,  0.0120, -0.0005,  0.0016],
        [-0.0116, -0.0048, -0.0046,  ..., -0.0027,  0.0087,  0.0113],
        [ 0.0132,  0.0156,  0.0015,  ...,  0.0142,  0.0062, -0.0095]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.2.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.2.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[-0.0045,  0.0008,  0.0212,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0028,  0.0141, -0.0219,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0007,  0.0152, -0.0013,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0004, -0.0306, -0.0190,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0366, -0.0006,  0.0178,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0122,  0.0168,  0.0013,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.2.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[-3.0060e-03, -1.2817e-02,  1.2207e-03,  ..., -6.3171e-03,
         -1.9684e-03, -7.8735e-03],
        [-1.4526e-02, -9.3460e-04,  6.1951e-03,  ..., -1.5442e-02,
          4.2419e-03, -1.6861e-03],
        [ 1.0559e-02,  5.9814e-03,  1.7071e-04,  ...,  5.4321e-03,
          1.2817e-02,  9.1553e-04],
        ...,
        [-6.7139e-03,  1.3550e-02,  5.0964e-03,  ...,  3.4027e-03,
          1.0803e-02,  1.1414e-02],
        [-1.4282e-02, -3.0823e-03,  1.9226e-03,  ..., -1.0193e-02,
         -4.7302e-03, -7.9346e-04],
        [-7.9956e-03, -7.6294e-03,  5.5552e-05,  ...,  2.3041e-03,
         -1.3306e-02,  1.3428e-02]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.2.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.2.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[-0.0018,  0.0153,  0.0092,  ...,  0.0181,  0.0033, -0.0280],
        [-0.0010, -0.0184,  0.0059,  ...,  0.0047, -0.0119, -0.0208],
        [-0.0204,  0.0032, -0.0145,  ..., -0.0187, -0.0085,  0.0096],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.2.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[-0.0156, -0.0076, -0.0070,  ...,  0.0075, -0.0057,  0.0089],
        [-0.0083, -0.0027,  0.0013,  ..., -0.0036, -0.0042, -0.0107],
        [ 0.0146,  0.0071, -0.0039,  ...,  0.0120,  0.0090, -0.0049],
        ...,
        [-0.0154, -0.0048,  0.0054,  ...,  0.0103,  0.0065,  0.0116],
        [ 0.0114,  0.0027,  0.0007,  ...,  0.0144,  0.0155,  0.0092],
        [ 0.0118, -0.0023,  0.0106,  ...,  0.0077, -0.0119,  0.0091]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.2.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.3.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[ 0.0072,  0.0124,  0.0007,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0113,  0.0168, -0.0194,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0154,  0.0062, -0.0200,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0703, -0.0684,  0.0415,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0801,  0.0388, -0.0036,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0154, -0.0510,  0.0713,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.3.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[-0.0041,  0.0021, -0.0098,  ..., -0.0045, -0.0075,  0.0123],
        [ 0.0018,  0.0121, -0.0027,  ..., -0.0096, -0.0153, -0.0060],
        [ 0.0051,  0.0009,  0.0066,  ..., -0.0071, -0.0124, -0.0045],
        ...,
        [ 0.0067, -0.0027,  0.0139,  ...,  0.0089, -0.0102, -0.0074],
        [ 0.0010, -0.0071,  0.0018,  ..., -0.0118, -0.0042, -0.0035],
        [-0.0077, -0.0146, -0.0108,  ..., -0.0020, -0.0037, -0.0019]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.3.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.3.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[ 0.0096, -0.0084, -0.0079,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0122,  0.0121, -0.0237,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0002, -0.0081, -0.0022,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0894, -0.0850,  0.0037,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0864,  0.0277,  0.0435,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0011, -0.0510,  0.0923,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.3.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[-2.1362e-03, -5.9509e-03,  1.1292e-02,  ...,  7.6294e-03,
         -3.5400e-03, -6.6223e-03],
        [ 1.6479e-03,  1.1902e-02,  3.2663e-05,  ...,  1.3245e-02,
         -8.7738e-04,  1.4038e-02],
        [ 1.1047e-02,  8.9111e-03, -4.7913e-03,  ...,  6.6376e-04,
          3.9062e-03, -2.1210e-03],
        ...,
        [ 7.4768e-04,  6.8665e-03, -7.6904e-03,  ...,  1.2329e-02,
         -1.4343e-02, -1.1841e-02],
        [-2.8839e-03, -7.6599e-03, -1.3657e-03,  ...,  4.4861e-03,
         -1.2817e-03,  1.4404e-02],
        [ 3.5095e-03, -1.8082e-03,  3.4485e-03,  ..., -8.6670e-03,
          5.4932e-03,  9.3384e-03]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.3.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.3.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[-0.0014,  0.0118, -0.0049,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0101,  0.0251, -0.0173,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0026, -0.0021,  0.0029,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0077, -0.0098,  0.0025,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0036,  0.0001,  0.0068,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0091, -0.0064,  0.0103,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.3.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0049,  0.0151, -0.0008,  ..., -0.0004, -0.0155, -0.0048],
        [ 0.0143,  0.0110, -0.0136,  ..., -0.0041,  0.0038,  0.0037],
        [-0.0108,  0.0058,  0.0059,  ..., -0.0076, -0.0050,  0.0085],
        ...,
        [-0.0147,  0.0117, -0.0148,  ..., -0.0037, -0.0140,  0.0082],
        [-0.0152,  0.0075, -0.0033,  ..., -0.0127, -0.0090,  0.0120],
        [-0.0085, -0.0140, -0.0066,  ..., -0.0089, -0.0012, -0.0011]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.3.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.3.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[-0.0273,  0.0070, -0.0172,  ...,  0.0052,  0.0069, -0.0029],
        [ 0.0270, -0.0281, -0.0084,  ...,  0.0076, -0.0031,  0.0005],
        [-0.0013, -0.0106, -0.0010,  ...,  0.0005,  0.0029,  0.0060],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.3.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[-0.0133, -0.0060, -0.0124,  ...,  0.0020,  0.0117, -0.0063],
        [ 0.0092, -0.0139,  0.0013,  ...,  0.0026, -0.0002, -0.0103],
        [-0.0093, -0.0134,  0.0074,  ...,  0.0068, -0.0039, -0.0124],
        ...,
        [-0.0023,  0.0007,  0.0013,  ..., -0.0060, -0.0051,  0.0032],
        [ 0.0018,  0.0007,  0.0137,  ..., -0.0027, -0.0091,  0.0145],
        [-0.0115, -0.0036, -0.0128,  ...,  0.0131, -0.0080, -0.0092]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.3.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.4.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[-0.0176,  0.0045, -0.0019,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0231, -0.0128, -0.0090,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0034, -0.0280,  0.0143,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0226, -0.0038, -0.0037,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0312,  0.0138,  0.0269,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0121, -0.0339,  0.0649,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.4.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[ 1.7624e-03,  1.0803e-02,  3.7079e-03,  ...,  9.8877e-03,
         -1.5259e-02,  7.6599e-03],
        [ 1.0254e-02,  9.3384e-03,  1.1292e-02,  ..., -7.5684e-03,
          9.6436e-03,  6.6528e-03],
        [-5.6458e-03, -5.8289e-03, -1.0010e-02,  ...,  2.5024e-03,
         -4.6387e-03, -9.7656e-03],
        ...,
        [-2.4567e-03, -6.6528e-03, -1.1414e-02,  ..., -1.0132e-02,
          5.1270e-03,  3.6926e-03],
        [ 5.3711e-03, -8.3618e-03,  7.4768e-04,  ...,  5.4932e-03,
          5.0964e-03,  1.0071e-02],
        [ 3.1281e-03,  6.0797e-05,  8.0872e-04,  ...,  6.8054e-03,
          7.7515e-03,  1.2756e-02]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.4.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.4.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[-0.0085,  0.0123, -0.0120,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0295, -0.0049,  0.0010,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0166, -0.0002, -0.0141,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0098,  0.1206,  0.0586,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0337,  0.0500, -0.0183,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0091,  0.0610,  0.0165,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.4.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[-1.0681e-03,  2.1210e-03, -1.1780e-02,  ...,  7.3853e-03,
         -1.3489e-02,  1.5503e-02],
        [-4.9438e-03,  8.5068e-04,  4.8828e-03,  ..., -1.0681e-02,
          1.2146e-02, -1.0550e-05],
        [ 7.9155e-05, -1.1978e-03, -1.8387e-03,  ...,  6.9046e-04,
         -5.7983e-04,  9.9487e-03],
        ...,
        [ 6.5308e-03, -1.5259e-02, -1.4038e-02,  ...,  1.4038e-02,
         -8.1787e-03, -3.6163e-03],
        [ 1.4893e-02,  1.0925e-02, -4.2419e-03,  ..., -2.1057e-03,
         -4.3640e-03, -1.4404e-02],
        [-3.6926e-03, -4.9438e-03, -1.0132e-02,  ...,  5.9509e-03,
          2.9755e-03,  3.2196e-03]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.4.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.4.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[ 0.0006, -0.0016, -0.0080,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0034,  0.0249,  0.0068,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0140,  0.0012, -0.0366,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0019, -0.0094, -0.0212,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0036, -0.0063, -0.0006,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0028, -0.0192, -0.0016,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.4.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0006, -0.0109, -0.0070,  ..., -0.0064, -0.0097,  0.0104],
        [ 0.0118,  0.0121,  0.0017,  ..., -0.0012,  0.0005, -0.0015],
        [-0.0096, -0.0120, -0.0062,  ...,  0.0013, -0.0030, -0.0044],
        ...,
        [ 0.0093, -0.0101, -0.0084,  ..., -0.0058, -0.0105, -0.0063],
        [-0.0055,  0.0091,  0.0143,  ..., -0.0030,  0.0140, -0.0100],
        [ 0.0020, -0.0113, -0.0046,  ...,  0.0063, -0.0019, -0.0013]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.4.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.4.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[ 3.4424e-02,  9.8877e-03, -1.9379e-03,  ...,  1.1292e-03,
         -2.5146e-02, -3.7384e-03],
        [ 4.0283e-03, -1.2517e-06, -6.1035e-03,  ...,  4.5471e-03,
         -2.0386e-02,  3.6774e-03],
        [-1.3199e-03,  5.5542e-03, -1.0254e-02,  ..., -1.0925e-02,
          8.1177e-03, -3.5400e-03],
        ...,
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00]], device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.4.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0044, -0.0135,  0.0017,  ...,  0.0082,  0.0096, -0.0095],
        [ 0.0105, -0.0017,  0.0104,  ..., -0.0099, -0.0048, -0.0126],
        [ 0.0037, -0.0076,  0.0103,  ...,  0.0087, -0.0074, -0.0074],
        ...,
        [ 0.0118, -0.0045, -0.0020,  ..., -0.0071, -0.0095,  0.0007],
        [ 0.0042, -0.0003,  0.0008,  ...,  0.0145, -0.0016, -0.0133],
        [-0.0006,  0.0038, -0.0010,  ...,  0.0014, -0.0053, -0.0004]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.4.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.5.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[-0.0064, -0.0048, -0.0155,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0242,  0.0145,  0.0396,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0098,  0.0084, -0.0110,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0109,  0.0062, -0.0049,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0369,  0.0152,  0.0189,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0087,  0.0220,  0.0322,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.5.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[-9.0332e-03,  1.3672e-02, -1.0986e-02,  ...,  8.4839e-03,
         -1.1597e-02, -4.3945e-03],
        [-7.5378e-03, -5.2795e-03,  5.2185e-03,  ...,  6.9427e-04,
          9.2773e-03, -8.4839e-03],
        [-1.1719e-02, -1.4038e-02, -1.1230e-02,  ..., -1.1139e-03,
          1.0559e-02,  4.0588e-03],
        ...,
        [ 1.5137e-02,  1.2024e-02, -1.2970e-03,  ...,  7.3853e-03,
         -1.0437e-02, -8.6060e-03],
        [-8.0566e-03, -1.5442e-02,  3.1128e-03,  ..., -6.2943e-04,
          1.1597e-02, -6.7139e-03],
        [ 1.4404e-02,  1.3062e-02,  6.1035e-03,  ...,  1.4954e-02,
          4.3631e-05,  1.9073e-04]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.5.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.5.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[-0.0145,  0.0295,  0.0062,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0015, -0.0009, -0.0298,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0082, -0.0168,  0.0007,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0073, -0.0007, -0.0234,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0542,  0.0209, -0.0181,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0282, -0.0242,  0.0405,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.5.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[-0.0131,  0.0034,  0.0007,  ...,  0.0062,  0.0015,  0.0143],
        [-0.0042,  0.0112, -0.0079,  ...,  0.0126, -0.0075, -0.0132],
        [-0.0134,  0.0125, -0.0082,  ...,  0.0118,  0.0080,  0.0013],
        ...,
        [ 0.0081, -0.0117, -0.0098,  ...,  0.0034, -0.0125,  0.0029],
        [-0.0120, -0.0153, -0.0154,  ..., -0.0013,  0.0110, -0.0134],
        [-0.0146,  0.0082,  0.0056,  ..., -0.0088,  0.0064,  0.0108]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.5.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.5.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[-0.0359,  0.0062,  0.0143,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0062,  0.0146, -0.0262,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0177,  0.0038,  0.0007,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0243,  0.0126,  0.0060,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0036,  0.0008, -0.0052,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0076, -0.0090,  0.0210,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.5.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0039,  0.0032, -0.0070,  ..., -0.0048,  0.0014,  0.0027],
        [ 0.0096, -0.0143, -0.0032,  ...,  0.0113,  0.0132,  0.0059],
        [ 0.0038,  0.0098,  0.0134,  ...,  0.0121, -0.0098,  0.0128],
        ...,
        [ 0.0151,  0.0039, -0.0070,  ...,  0.0123, -0.0073,  0.0132],
        [-0.0044, -0.0038, -0.0093,  ..., -0.0131,  0.0132, -0.0066],
        [-0.0025,  0.0031,  0.0052,  ...,  0.0104, -0.0115,  0.0022]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.5.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.5.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[-0.0133, -0.0060,  0.0039,  ..., -0.0151, -0.0109,  0.0244],
        [ 0.0022,  0.0017, -0.0126,  ..., -0.0177,  0.0021, -0.0007],
        [-0.0084, -0.0069,  0.0143,  ...,  0.0078,  0.0040,  0.0222],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.5.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[-0.0071,  0.0033,  0.0128,  ...,  0.0102, -0.0153,  0.0060],
        [ 0.0082, -0.0002, -0.0135,  ..., -0.0069, -0.0118, -0.0143],
        [-0.0011, -0.0078, -0.0038,  ..., -0.0103,  0.0124, -0.0048],
        ...,
        [-0.0064, -0.0064,  0.0030,  ..., -0.0071,  0.0033, -0.0053],
        [ 0.0017,  0.0122,  0.0074,  ..., -0.0057, -0.0145, -0.0114],
        [-0.0154,  0.0117,  0.0011,  ..., -0.0059, -0.0023,  0.0131]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.5.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.6.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[-0.0058, -0.0055,  0.0051,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0003, -0.0167,  0.0098,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0167,  0.0204, -0.0337,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0157,  0.0098, -0.0349,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0325,  0.0374,  0.0060,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0620, -0.0537, -0.0002,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.6.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[-0.0055,  0.0078, -0.0006,  ...,  0.0151,  0.0007, -0.0060],
        [ 0.0066, -0.0109,  0.0138,  ..., -0.0007, -0.0089, -0.0071],
        [ 0.0121, -0.0003, -0.0084,  ..., -0.0061,  0.0074,  0.0036],
        ...,
        [-0.0090, -0.0045, -0.0093,  ...,  0.0150,  0.0067, -0.0123],
        [ 0.0003,  0.0146, -0.0051,  ..., -0.0021, -0.0137, -0.0127],
        [-0.0056, -0.0093,  0.0008,  ...,  0.0055, -0.0108,  0.0057]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.6.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.6.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[ 0.0148, -0.0134,  0.0195,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0164, -0.0053,  0.0457,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0100, -0.0133, -0.0008,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0176, -0.0383, -0.0160,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0199,  0.0047, -0.0292,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0483, -0.0258, -0.0352,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.6.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[ 1.0071e-02,  1.5259e-04,  6.8359e-03,  ..., -7.4463e-03,
         -1.3672e-02, -9.3994e-03],
        [ 7.6904e-03, -4.8256e-04, -7.4158e-03,  ...,  1.5198e-02,
          5.8594e-03,  8.4839e-03],
        [ 1.5503e-02,  1.1841e-02,  2.4261e-03,  ...,  1.5503e-02,
         -1.6242e-06, -2.1667e-03],
        ...,
        [ 3.9062e-03,  1.3245e-02,  1.2024e-02,  ..., -9.9487e-03,
          6.1646e-03, -3.9978e-03],
        [ 7.8201e-04, -6.2866e-03, -1.2390e-02,  ...,  1.4893e-02,
          6.0120e-03,  7.9346e-03],
        [ 1.6785e-04, -4.6997e-03, -2.5940e-03,  ...,  3.9062e-03,
          1.0803e-02, -6.5613e-04]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.6.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.6.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[ 0.0125,  0.0347,  0.0060,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0361, -0.0007, -0.0070,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0087, -0.0056,  0.0109,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0223, -0.0058,  0.0106,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0212, -0.0095,  0.0041,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0354,  0.0063,  0.0184,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.6.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[-0.0115,  0.0153, -0.0013,  ...,  0.0123, -0.0127,  0.0068],
        [-0.0113,  0.0050, -0.0145,  ...,  0.0017,  0.0098,  0.0146],
        [-0.0038,  0.0004,  0.0099,  ...,  0.0037, -0.0030,  0.0075],
        ...,
        [-0.0019,  0.0027, -0.0065,  ..., -0.0091, -0.0088, -0.0011],
        [ 0.0081, -0.0156,  0.0006,  ..., -0.0114, -0.0051,  0.0153],
        [ 0.0015, -0.0051, -0.0007,  ..., -0.0006,  0.0131, -0.0024]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.6.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.6.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[ 0.0138, -0.0155, -0.0381,  ..., -0.0105,  0.0033,  0.0093],
        [-0.0209,  0.0057,  0.0029,  ..., -0.0187,  0.0038, -0.0036],
        [ 0.0018, -0.0040, -0.0040,  ...,  0.0115,  0.0025,  0.0098],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.6.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[ 5.3406e-03, -1.1414e-02, -1.2329e-02,  ..., -1.8311e-03,
          1.2329e-02, -1.5259e-03],
        [ 1.4221e-02,  5.9509e-03,  1.0315e-02,  ...,  1.4404e-02,
          1.2390e-02, -1.3113e-05],
        [-4.7302e-03, -9.9945e-04,  1.6022e-03,  ..., -6.3171e-03,
          1.0559e-02, -5.8899e-03],
        ...,
        [ 8.2397e-03, -5.4932e-04, -1.6861e-03,  ..., -5.9509e-03,
         -1.3245e-02,  6.8359e-03],
        [ 1.3489e-02, -3.6240e-04, -4.4250e-03,  ..., -1.4221e-02,
          1.8597e-04,  1.4465e-02],
        [-6.4087e-03, -8.5831e-06,  7.4387e-04,  ...,  6.6833e-03,
          6.2866e-03,  2.9907e-03]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.6.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.7.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[ 0.0069,  0.0024,  0.0034,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0054,  0.0061,  0.0024,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0011,  0.0079, -0.0195,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0014, -0.0374, -0.0298,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0106,  0.0549, -0.0415,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0801, -0.0239, -0.0131,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.7.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0028,  0.0078,  0.0134,  ..., -0.0010,  0.0028, -0.0038],
        [-0.0027,  0.0138,  0.0032,  ...,  0.0058, -0.0108, -0.0044],
        [-0.0134,  0.0006, -0.0007,  ...,  0.0067, -0.0112, -0.0009],
        ...,
        [ 0.0056,  0.0102,  0.0140,  ...,  0.0038, -0.0126, -0.0055],
        [ 0.0153,  0.0117, -0.0073,  ..., -0.0115, -0.0087,  0.0108],
        [-0.0141,  0.0090,  0.0014,  ...,  0.0020, -0.0124, -0.0028]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.7.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.7.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[-0.0029, -0.0063, -0.0132,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0042, -0.0155,  0.0042,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0055, -0.0066,  0.0183,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0430,  0.0104,  0.0215,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0425,  0.0540, -0.0020,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0334,  0.0084,  0.0109,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.7.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[-0.0036,  0.0086,  0.0014,  ...,  0.0026,  0.0126, -0.0110],
        [-0.0068, -0.0107,  0.0052,  ...,  0.0092,  0.0129,  0.0035],
        [ 0.0099,  0.0133, -0.0112,  ...,  0.0067, -0.0047, -0.0102],
        ...,
        [-0.0014,  0.0009, -0.0138,  ..., -0.0079, -0.0002, -0.0123],
        [-0.0041,  0.0105, -0.0013,  ...,  0.0082, -0.0074, -0.0115],
        [-0.0077, -0.0027,  0.0009,  ...,  0.0094,  0.0099, -0.0029]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.7.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.7.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[-0.0150,  0.0020,  0.0162,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0289, -0.0306, -0.0019,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0076, -0.0059, -0.0049,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0011, -0.0383,  0.0034,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0325,  0.0164, -0.0270,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0101, -0.0010, -0.0070,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.7.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[-0.0012,  0.0153,  0.0140,  ...,  0.0089,  0.0015, -0.0137],
        [ 0.0047, -0.0128, -0.0152,  ...,  0.0154, -0.0050,  0.0134],
        [ 0.0070, -0.0002, -0.0148,  ..., -0.0015,  0.0132, -0.0005],
        ...,
        [ 0.0098,  0.0070, -0.0129,  ..., -0.0068,  0.0152,  0.0129],
        [-0.0084, -0.0041, -0.0085,  ...,  0.0021,  0.0142,  0.0114],
        [ 0.0027, -0.0064,  0.0095,  ..., -0.0039, -0.0097, -0.0049]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.7.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.7.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[ 0.0148, -0.0164, -0.0050,  ...,  0.0072, -0.0253,  0.0022],
        [-0.0020,  0.0242,  0.0110,  ..., -0.0425, -0.0121, -0.0072],
        [ 0.0050,  0.0052, -0.0206,  ..., -0.0102, -0.0111, -0.0081],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.7.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0134,  0.0056,  0.0060,  ..., -0.0107, -0.0078, -0.0075],
        [-0.0010,  0.0049,  0.0027,  ..., -0.0007, -0.0020, -0.0110],
        [ 0.0040, -0.0040, -0.0057,  ...,  0.0044, -0.0054, -0.0126],
        ...,
        [-0.0039,  0.0112,  0.0021,  ..., -0.0131,  0.0078,  0.0092],
        [-0.0003, -0.0060,  0.0043,  ..., -0.0037,  0.0001, -0.0150],
        [ 0.0137,  0.0071,  0.0131,  ..., -0.0140, -0.0053, -0.0048]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.7.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.8.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[ 7.9956e-03, -1.0925e-02, -2.5024e-02,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [-1.9775e-02, -1.4954e-02, -3.4180e-02,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [-3.1128e-02, -2.3956e-03, -1.0376e-02,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        ...,
        [-6.8359e-03,  8.3984e-02,  4.2480e-02,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [ 1.5378e-05, -6.2988e-02, -1.6479e-02,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [-6.2988e-02, -5.7617e-02, -1.3916e-02,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00]], device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.8.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[-1.1230e-02, -9.9487e-03,  1.4404e-02,  ...,  2.9755e-03,
         -1.3245e-02, -1.3123e-02],
        [ 3.7537e-03,  2.3651e-03, -1.4343e-02,  ...,  1.9989e-03,
         -1.3245e-02, -9.9945e-04],
        [ 5.4321e-03, -1.2329e-02,  1.5564e-02,  ..., -5.9814e-03,
          7.1411e-03, -3.4790e-03],
        ...,
        [-9.2163e-03, -3.4637e-03, -1.8997e-03,  ..., -1.6403e-03,
          1.4465e-02,  1.3672e-02],
        [ 9.5215e-03, -2.2888e-03,  1.3428e-02,  ..., -1.2878e-02,
         -1.5503e-02,  1.1780e-02],
        [ 7.8735e-03,  9.7046e-03, -1.2573e-02,  ...,  5.0306e-05,
          9.5215e-03, -4.6082e-03]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.8.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.8.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[-0.0064,  0.0045, -0.0060,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0068,  0.0056, -0.0070,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0113,  0.0023,  0.0052,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0027, -0.0378, -0.0076,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0035,  0.0249,  0.0256,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0154, -0.0047, -0.0330,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.8.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0033,  0.0042, -0.0031,  ...,  0.0022, -0.0061,  0.0143],
        [ 0.0089, -0.0049,  0.0015,  ..., -0.0107,  0.0087,  0.0022],
        [-0.0056,  0.0131, -0.0032,  ..., -0.0021, -0.0038,  0.0070],
        ...,
        [ 0.0103,  0.0121, -0.0142,  ...,  0.0049,  0.0027,  0.0031],
        [ 0.0063,  0.0085, -0.0011,  ...,  0.0125, -0.0057, -0.0089],
        [-0.0114, -0.0019,  0.0143,  ...,  0.0098, -0.0152, -0.0087]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.8.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.8.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[-0.0260, -0.0129,  0.0044,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0226, -0.0135,  0.0071,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0052,  0.0162,  0.0125,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0295, -0.0159,  0.0030,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0261, -0.0156,  0.0023,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0165,  0.0001, -0.0055,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.8.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0075,  0.0029, -0.0059,  ...,  0.0059,  0.0049,  0.0009],
        [ 0.0069,  0.0074, -0.0150,  ...,  0.0151,  0.0118,  0.0020],
        [ 0.0051,  0.0041,  0.0028,  ..., -0.0133, -0.0089,  0.0049],
        ...,
        [ 0.0146,  0.0078,  0.0084,  ..., -0.0151,  0.0113,  0.0055],
        [-0.0107, -0.0068,  0.0014,  ..., -0.0142, -0.0027, -0.0049],
        [-0.0091, -0.0101,  0.0120,  ...,  0.0156, -0.0152, -0.0150]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.8.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.8.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[ 0.0136,  0.0132, -0.0133,  ..., -0.0105,  0.0013,  0.0157],
        [ 0.0101,  0.0043, -0.0325,  ..., -0.0066, -0.0095, -0.0139],
        [ 0.0015,  0.0107, -0.0039,  ...,  0.0011,  0.0148,  0.0211],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.8.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[-0.0060,  0.0024, -0.0057,  ..., -0.0147,  0.0052, -0.0065],
        [-0.0051,  0.0136, -0.0063,  ..., -0.0116,  0.0081, -0.0058],
        [-0.0009, -0.0008, -0.0103,  ...,  0.0100, -0.0128, -0.0140],
        ...,
        [-0.0127, -0.0109, -0.0079,  ...,  0.0010,  0.0067, -0.0149],
        [ 0.0086, -0.0091,  0.0125,  ...,  0.0064,  0.0072,  0.0015],
        [-0.0115,  0.0143,  0.0054,  ...,  0.0031,  0.0114, -0.0132]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.8.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.9.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[-0.0194,  0.0025,  0.0168,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0022, -0.0349,  0.0134,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0023, -0.0017, -0.0148,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0037, -0.0452,  0.0189,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0146, -0.0212,  0.0110,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0010,  0.0547,  0.0197,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.9.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[-0.0143,  0.0105,  0.0080,  ..., -0.0004,  0.0145,  0.0147],
        [-0.0140,  0.0071,  0.0121,  ..., -0.0066, -0.0039, -0.0125],
        [ 0.0078, -0.0119, -0.0060,  ...,  0.0123, -0.0128,  0.0045],
        ...,
        [ 0.0018,  0.0139,  0.0146,  ..., -0.0117,  0.0150, -0.0039],
        [-0.0125, -0.0154,  0.0036,  ...,  0.0133, -0.0001, -0.0085],
        [ 0.0128, -0.0072, -0.0086,  ..., -0.0046, -0.0106,  0.0022]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.9.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.9.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[-0.0347,  0.0101, -0.0167,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0057,  0.0325,  0.0094,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0118, -0.0110,  0.0189,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0114, -0.0173, -0.0068,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0325, -0.0101,  0.0215,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0061, -0.0264,  0.0302,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.9.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0061,  0.0076,  0.0086,  ..., -0.0020,  0.0022,  0.0021],
        [-0.0129, -0.0089, -0.0126,  ..., -0.0011, -0.0100, -0.0026],
        [-0.0085,  0.0070, -0.0073,  ...,  0.0150, -0.0015, -0.0024],
        ...,
        [-0.0057,  0.0096, -0.0059,  ...,  0.0146, -0.0026, -0.0152],
        [-0.0014, -0.0051, -0.0033,  ..., -0.0054,  0.0136,  0.0013],
        [ 0.0123,  0.0016,  0.0051,  ...,  0.0151, -0.0040, -0.0066]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.9.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.9.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[-0.0032,  0.0104, -0.0009,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0164, -0.0175, -0.0008,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0023, -0.0259,  0.0205,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0069, -0.0131,  0.0044,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0149,  0.0032, -0.0104,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0236,  0.0206, -0.0018,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.9.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[-0.0092, -0.0013, -0.0118,  ..., -0.0096,  0.0093, -0.0053],
        [-0.0132, -0.0101,  0.0035,  ..., -0.0117,  0.0033, -0.0145],
        [-0.0041, -0.0087, -0.0153,  ...,  0.0122, -0.0112,  0.0017],
        ...,
        [ 0.0102, -0.0097,  0.0118,  ..., -0.0112, -0.0141, -0.0148],
        [-0.0148, -0.0032,  0.0118,  ...,  0.0074,  0.0154,  0.0031],
        [-0.0012, -0.0021,  0.0104,  ...,  0.0154, -0.0079, -0.0020]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.9.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.9.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[ 0.0096,  0.0106,  0.0052,  ..., -0.0104,  0.0010, -0.0020],
        [-0.0051, -0.0149, -0.0023,  ..., -0.0002,  0.0073,  0.0145],
        [-0.0337, -0.0315,  0.0118,  ...,  0.0264,  0.0014,  0.0126],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.9.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[-0.0117,  0.0019, -0.0039,  ...,  0.0120, -0.0027, -0.0051],
        [ 0.0078,  0.0018, -0.0140,  ...,  0.0063,  0.0104, -0.0099],
        [ 0.0091,  0.0079, -0.0148,  ..., -0.0128,  0.0007,  0.0145],
        ...,
        [-0.0015, -0.0098,  0.0027,  ..., -0.0013,  0.0094, -0.0071],
        [ 0.0101,  0.0128, -0.0102,  ..., -0.0034, -0.0090, -0.0033],
        [-0.0112,  0.0052,  0.0029,  ...,  0.0139, -0.0059, -0.0018]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.9.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.10.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[-0.0014,  0.0048, -0.0016,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0041, -0.0074,  0.0166,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0099, -0.0086, -0.0054,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0330,  0.0640, -0.0152,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0591,  0.0576, -0.0002,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0498, -0.0216, -0.0050,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.10.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0033,  0.0018, -0.0084,  ...,  0.0054,  0.0045, -0.0125],
        [-0.0115,  0.0062,  0.0126,  ...,  0.0133,  0.0088,  0.0025],
        [-0.0044,  0.0121, -0.0137,  ..., -0.0110,  0.0129,  0.0095],
        ...,
        [-0.0008,  0.0016, -0.0082,  ...,  0.0073,  0.0150,  0.0130],
        [-0.0114,  0.0049,  0.0044,  ...,  0.0047, -0.0027,  0.0025],
        [ 0.0037,  0.0067,  0.0033,  ..., -0.0016, -0.0064, -0.0029]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.10.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.10.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[-0.0223,  0.0074, -0.0056,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0108,  0.0073, -0.0060,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0051,  0.0009,  0.0065,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0449, -0.0593,  0.0078,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0371, -0.0073,  0.0295,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0286,  0.0267, -0.0212,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.10.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0040, -0.0106,  0.0116,  ...,  0.0077, -0.0030,  0.0148],
        [ 0.0120,  0.0119, -0.0044,  ..., -0.0126,  0.0096,  0.0138],
        [-0.0017, -0.0102,  0.0111,  ..., -0.0087, -0.0111, -0.0139],
        ...,
        [ 0.0029,  0.0140, -0.0065,  ..., -0.0035,  0.0051, -0.0124],
        [ 0.0036,  0.0075,  0.0123,  ...,  0.0056,  0.0088,  0.0005],
        [-0.0098,  0.0127,  0.0069,  ...,  0.0045, -0.0026, -0.0090]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.10.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.10.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[-0.0400,  0.0035, -0.0065,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0080,  0.0132,  0.0123,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0149, -0.0125,  0.0152,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0231,  0.0291,  0.0002,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0025,  0.0078,  0.0007,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0143,  0.0087, -0.0153,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.10.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[-0.0110,  0.0110,  0.0136,  ...,  0.0017, -0.0106,  0.0075],
        [-0.0066, -0.0098, -0.0043,  ..., -0.0148, -0.0026,  0.0102],
        [ 0.0096,  0.0044,  0.0099,  ...,  0.0144, -0.0152,  0.0156],
        ...,
        [-0.0110,  0.0010,  0.0050,  ...,  0.0070, -0.0030, -0.0020],
        [ 0.0082, -0.0151,  0.0104,  ...,  0.0153, -0.0065, -0.0145],
        [-0.0096,  0.0063,  0.0115,  ...,  0.0013, -0.0020,  0.0145]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.10.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.10.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[-0.0030,  0.0015, -0.0315,  ..., -0.0005,  0.0104,  0.0082],
        [ 0.0010, -0.0121,  0.0126,  ...,  0.0035,  0.0138,  0.0047],
        [ 0.0132, -0.0361, -0.0200,  ...,  0.0085, -0.0094, -0.0055],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.10.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[-3.9482e-04, -1.7548e-03, -7.9346e-03,  ..., -1.3123e-02,
         -1.3123e-02,  1.2085e-02],
        [ 3.3112e-03, -1.5198e-02,  9.9945e-04,  ...,  7.8735e-03,
          3.0994e-05,  1.3916e-02],
        [ 1.0864e-02, -8.5449e-03,  1.4771e-02,  ...,  1.3855e-02,
          1.4282e-02, -1.3123e-03],
        ...,
        [-9.5215e-03, -2.3041e-03,  2.1362e-03,  ..., -1.2329e-02,
          7.4768e-03,  7.2098e-04],
        [ 3.0823e-03,  7.4463e-03, -7.9346e-03,  ..., -5.6458e-03,
          1.3855e-02, -1.5198e-02],
        [ 5.8594e-03, -4.1809e-03,  1.0803e-02,  ...,  3.3722e-03,
          9.4604e-03,  5.3787e-04]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.10.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.11.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[ 0.0153,  0.0165, -0.0070,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0132,  0.0091,  0.0111,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0009,  0.0063, -0.0240,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0006, -0.0048,  0.0325,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0447,  0.0179, -0.0120,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0718, -0.0206, -0.0214,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.11.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[-0.0020,  0.0069, -0.0154,  ...,  0.0078, -0.0098, -0.0120],
        [-0.0040, -0.0009,  0.0020,  ...,  0.0142,  0.0005,  0.0036],
        [ 0.0054,  0.0074,  0.0095,  ...,  0.0052,  0.0061,  0.0107],
        ...,
        [-0.0083,  0.0111, -0.0014,  ...,  0.0065,  0.0038,  0.0147],
        [ 0.0041,  0.0038, -0.0101,  ..., -0.0134,  0.0082,  0.0146],
        [ 0.0043, -0.0063,  0.0032,  ..., -0.0107,  0.0028,  0.0023]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.11.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.11.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[ 0.0050,  0.0187, -0.0009,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0003, -0.0034, -0.0254,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0147, -0.0254,  0.0073,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0085,  0.0211,  0.0025,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0645, -0.0107, -0.0118,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0060,  0.0056,  0.0181,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.11.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[-0.0045,  0.0046,  0.0060,  ...,  0.0096, -0.0098,  0.0074],
        [-0.0086, -0.0067,  0.0070,  ..., -0.0039,  0.0058, -0.0068],
        [ 0.0052,  0.0116,  0.0127,  ..., -0.0021, -0.0018, -0.0017],
        ...,
        [ 0.0030, -0.0154, -0.0016,  ...,  0.0126,  0.0129, -0.0104],
        [-0.0143, -0.0013, -0.0078,  ..., -0.0060, -0.0055,  0.0075],
        [ 0.0125, -0.0081, -0.0128,  ..., -0.0123,  0.0088,  0.0012]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.11.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.11.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[ 0.0050,  0.0046, -0.0101,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0023, -0.0065, -0.0121,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0025,  0.0069, -0.0069,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0154,  0.0063, -0.0141,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0083,  0.0150, -0.0103,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0131, -0.0228,  0.0027,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.11.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[-0.0029,  0.0096, -0.0128,  ...,  0.0012, -0.0063, -0.0105],
        [-0.0067, -0.0003,  0.0096,  ...,  0.0013, -0.0124,  0.0128],
        [ 0.0098,  0.0128, -0.0066,  ...,  0.0079,  0.0154, -0.0128],
        ...,
        [ 0.0020, -0.0068,  0.0145,  ...,  0.0104, -0.0003,  0.0122],
        [ 0.0004,  0.0026,  0.0012,  ...,  0.0137,  0.0005, -0.0023],
        [-0.0054, -0.0063, -0.0015,  ..., -0.0045, -0.0049,  0.0049]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.11.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.11.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[-0.0183, -0.0008,  0.0052,  ...,  0.0092,  0.0112,  0.0015],
        [ 0.0016, -0.0261,  0.0173,  ..., -0.0120,  0.0018, -0.0162],
        [-0.0077, -0.0050,  0.0193,  ...,  0.0010, -0.0002,  0.0190],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.11.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[-0.0122,  0.0001,  0.0050,  ..., -0.0027, -0.0056, -0.0114],
        [ 0.0028, -0.0107, -0.0010,  ...,  0.0125,  0.0041,  0.0059],
        [-0.0134, -0.0096,  0.0135,  ...,  0.0063, -0.0070,  0.0085],
        ...,
        [ 0.0116, -0.0083, -0.0092,  ...,  0.0035, -0.0152,  0.0027],
        [-0.0104, -0.0133,  0.0076,  ..., -0.0004, -0.0122, -0.0063],
        [-0.0120,  0.0140,  0.0065,  ...,  0.0139, -0.0066,  0.0074]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.11.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.12.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[-0.0045, -0.0197,  0.0095,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0046,  0.0053,  0.0007,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0084,  0.0356, -0.0292,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0041,  0.0386, -0.0206,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0347, -0.0062,  0.0109,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0339,  0.0135, -0.0410,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.12.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0120, -0.0101, -0.0030,  ..., -0.0011,  0.0129, -0.0146],
        [-0.0109, -0.0126, -0.0090,  ...,  0.0092, -0.0070,  0.0100],
        [-0.0072, -0.0138,  0.0081,  ..., -0.0017, -0.0075, -0.0053],
        ...,
        [-0.0008,  0.0153, -0.0002,  ...,  0.0096, -0.0034, -0.0088],
        [ 0.0058, -0.0031,  0.0068,  ...,  0.0058,  0.0156, -0.0055],
        [ 0.0115,  0.0118, -0.0154,  ..., -0.0149, -0.0128, -0.0151]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.12.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.12.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[ 0.0117, -0.0044, -0.0020,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0039,  0.0166, -0.0081,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0077, -0.0195,  0.0070,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0153,  0.0280, -0.0177,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0369,  0.0057,  0.0270,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0007,  0.0352,  0.0095,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.12.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[-0.0036,  0.0019,  0.0119,  ..., -0.0029,  0.0045, -0.0075],
        [-0.0131, -0.0134, -0.0093,  ..., -0.0083,  0.0150, -0.0062],
        [-0.0004,  0.0032,  0.0019,  ...,  0.0066, -0.0112,  0.0099],
        ...,
        [-0.0089,  0.0059, -0.0029,  ...,  0.0128, -0.0002, -0.0128],
        [-0.0050, -0.0141,  0.0022,  ..., -0.0029,  0.0076,  0.0060],
        [-0.0007,  0.0039,  0.0025,  ..., -0.0023, -0.0032,  0.0048]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.12.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.12.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[ 0.0053,  0.0054,  0.0045,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0265,  0.0166, -0.0017,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0121, -0.0116,  0.0045,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0002, -0.0023, -0.0175,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0007, -0.0120, -0.0060,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0215,  0.0017,  0.0021,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.12.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0039,  0.0047,  0.0009,  ...,  0.0066, -0.0052,  0.0062],
        [ 0.0121, -0.0122,  0.0083,  ..., -0.0133,  0.0010, -0.0035],
        [ 0.0151, -0.0022,  0.0081,  ...,  0.0074,  0.0124, -0.0110],
        ...,
        [-0.0117, -0.0074, -0.0012,  ...,  0.0069,  0.0084, -0.0100],
        [ 0.0127,  0.0030,  0.0104,  ...,  0.0078, -0.0017, -0.0134],
        [ 0.0087, -0.0154,  0.0078,  ...,  0.0092,  0.0083,  0.0151]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.12.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.12.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[ 0.0272,  0.0067, -0.0007,  ...,  0.0018,  0.0079, -0.0037],
        [-0.0058, -0.0107,  0.0014,  ..., -0.0136,  0.0049, -0.0188],
        [ 0.0031, -0.0052, -0.0048,  ...,  0.0056, -0.0267,  0.0126],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.12.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[-0.0016,  0.0143, -0.0023,  ..., -0.0115, -0.0034, -0.0090],
        [-0.0093,  0.0128, -0.0029,  ...,  0.0055,  0.0065,  0.0026],
        [ 0.0018,  0.0065,  0.0105,  ..., -0.0121,  0.0066, -0.0076],
        ...,
        [ 0.0107, -0.0014,  0.0156,  ...,  0.0094, -0.0047,  0.0028],
        [-0.0122, -0.0057, -0.0134,  ..., -0.0029, -0.0134,  0.0094],
        [-0.0153,  0.0129,  0.0120,  ...,  0.0085, -0.0074,  0.0039]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.12.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.13.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[-0.0030, -0.0080, -0.0132,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0156, -0.0122,  0.0009,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0159,  0.0099,  0.0072,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0427,  0.0049,  0.0014,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0049,  0.0113, -0.0217,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0073, -0.0220, -0.0153,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.13.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0088,  0.0048,  0.0084,  ...,  0.0092,  0.0019,  0.0113],
        [-0.0063,  0.0084, -0.0033,  ..., -0.0053, -0.0081, -0.0034],
        [ 0.0021,  0.0127, -0.0086,  ..., -0.0140,  0.0129,  0.0026],
        ...,
        [ 0.0052,  0.0041,  0.0001,  ...,  0.0008,  0.0144, -0.0087],
        [ 0.0055, -0.0035, -0.0153,  ...,  0.0053, -0.0132, -0.0003],
        [ 0.0018, -0.0031, -0.0117,  ..., -0.0004,  0.0080,  0.0129]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.13.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.13.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[ 0.0123,  0.0064, -0.0034,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0099,  0.0187, -0.0039,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0035, -0.0089,  0.0337,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0339,  0.0320, -0.0018,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0042,  0.0070,  0.0278,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0237, -0.0483, -0.0120,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.13.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[-1.1597e-02, -1.1230e-02, -6.0558e-05,  ...,  1.2970e-03,
         -8.8501e-03, -4.4861e-03],
        [-1.0498e-02,  1.2695e-02, -1.3794e-02,  ..., -7.5684e-03,
          1.2436e-03,  6.3705e-04],
        [ 3.9368e-03,  4.0894e-03, -2.4567e-03,  ...,  5.0659e-03,
          6.8359e-03,  1.1047e-02],
        ...,
        [-1.5503e-02,  9.0332e-03,  1.3794e-02,  ...,  9.9487e-03,
         -1.0559e-02,  8.9111e-03],
        [-1.0193e-02, -4.1199e-03, -1.4160e-02,  ..., -1.5381e-02,
         -1.3855e-02,  7.1411e-03],
        [-1.2512e-03,  4.6387e-03, -3.7231e-03,  ..., -1.3550e-02,
          1.4954e-02, -4.5166e-03]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.13.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.13.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[ 0.0154,  0.0031, -0.0125,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0085,  0.0141, -0.0044,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0183,  0.0013, -0.0016,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0023,  0.0214, -0.0066,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0129,  0.0115,  0.0137,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0051, -0.0096,  0.0251,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.13.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0154,  0.0101, -0.0040,  ...,  0.0053, -0.0082,  0.0077],
        [-0.0104, -0.0037,  0.0012,  ..., -0.0005, -0.0107,  0.0007],
        [ 0.0120,  0.0070,  0.0078,  ..., -0.0054, -0.0013,  0.0042],
        ...,
        [-0.0109, -0.0057, -0.0030,  ..., -0.0142,  0.0109, -0.0118],
        [-0.0047,  0.0030,  0.0088,  ..., -0.0127, -0.0140, -0.0027],
        [-0.0003,  0.0102,  0.0108,  ..., -0.0150,  0.0082, -0.0075]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.13.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.13.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[-3.8757e-03,  3.7766e-04, -5.8289e-03,  ...,  2.7466e-04,
         -4.9744e-03, -3.4790e-03],
        [ 2.4170e-02,  1.0681e-02,  4.2725e-03,  ..., -1.2268e-02,
         -8.9722e-03,  2.3499e-03],
        [ 6.5613e-03,  9.2773e-03, -6.0558e-05,  ...,  5.7373e-03,
         -2.5757e-02, -1.0376e-02],
        ...,
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00]], device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.13.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0085,  0.0110, -0.0055,  ...,  0.0051,  0.0115, -0.0154],
        [-0.0030, -0.0151, -0.0098,  ..., -0.0104,  0.0012, -0.0149],
        [ 0.0049,  0.0084,  0.0125,  ...,  0.0075, -0.0039, -0.0067],
        ...,
        [ 0.0074,  0.0094, -0.0065,  ..., -0.0141, -0.0124,  0.0146],
        [-0.0139,  0.0050, -0.0049,  ...,  0.0128,  0.0081,  0.0074],
        [ 0.0081, -0.0105,  0.0115,  ..., -0.0155, -0.0098,  0.0139]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.13.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.14.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[-0.0062,  0.0010,  0.0200,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0043,  0.0126, -0.0088,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0052, -0.0064,  0.0092,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0141, -0.0011, -0.0287,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0234, -0.0339,  0.0540,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0125,  0.0028, -0.0117,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.14.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[ 1.8082e-03, -1.1169e-02, -1.4221e-02,  ..., -2.2888e-03,
         -1.3245e-02,  7.4463e-03],
        [ 3.2043e-03,  9.4604e-03, -9.3384e-03,  ...,  1.2024e-02,
          1.2817e-02, -1.1719e-02],
        [-1.2512e-02,  2.8992e-03, -5.0964e-03,  ...,  1.4465e-02,
         -1.0742e-02,  1.1658e-02],
        ...,
        [-8.4839e-03, -1.4526e-02, -9.7656e-03,  ...,  1.2085e-02,
          6.6528e-03,  1.5564e-02],
        [-1.5572e-06,  9.9487e-03, -1.3550e-02,  ...,  6.3782e-03,
          8.5449e-03,  7.9346e-03],
        [-1.5320e-02,  1.0803e-02, -8.9722e-03,  ..., -7.6294e-03,
         -6.8918e-08, -9.8877e-03]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.14.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.14.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[-0.0108, -0.0028,  0.0087,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0281,  0.0261, -0.0221,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0203, -0.0003,  0.0209,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0315,  0.0094,  0.0275,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0289,  0.0042,  0.0141,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0150,  0.0339,  0.0250,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.14.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[-4.0588e-03, -5.6458e-03,  1.0071e-02,  ..., -1.4832e-02,
         -9.0332e-03,  1.0498e-02],
        [-7.6904e-03,  1.1475e-02,  9.5367e-04,  ..., -1.1230e-02,
          3.5553e-03,  8.3542e-04],
        [ 1.1230e-02,  1.5442e-02, -7.5989e-03,  ...,  5.8289e-03,
          1.0498e-02, -6.6223e-03],
        ...,
        [-1.0376e-02,  2.2125e-03, -2.5940e-03,  ..., -1.0437e-02,
         -8.6060e-03, -1.3062e-02],
        [-1.1108e-02,  1.3000e-02, -1.4343e-02,  ...,  4.3030e-03,
          5.9128e-05, -1.1658e-02],
        [ 7.0190e-03,  7.0190e-03,  1.3428e-02,  ...,  8.5449e-03,
          8.2397e-03, -6.9580e-03]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.14.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.14.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[ 8.5449e-03,  5.8594e-03, -4.0283e-02,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [-2.0142e-03, -1.9653e-02,  1.8787e-04,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [ 2.5146e-02, -1.8188e-02, -1.3828e-05,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        ...,
        [ 1.5442e-02, -1.2756e-02,  1.7212e-02,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [-1.5259e-02,  2.6978e-02, -3.3203e-02,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [ 7.2327e-03,  2.2705e-02, -2.3315e-02,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00]], device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.14.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[-0.0121, -0.0112,  0.0151,  ...,  0.0109, -0.0003, -0.0116],
        [ 0.0002,  0.0139, -0.0140,  ...,  0.0048, -0.0020, -0.0155],
        [-0.0087, -0.0074, -0.0134,  ...,  0.0049, -0.0104, -0.0053],
        ...,
        [-0.0153, -0.0059,  0.0099,  ..., -0.0063,  0.0023,  0.0107],
        [ 0.0043,  0.0063,  0.0034,  ...,  0.0132, -0.0056,  0.0102],
        [-0.0036,  0.0131, -0.0015,  ...,  0.0141, -0.0071,  0.0045]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.14.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.14.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[-0.0109,  0.0061, -0.0222,  ..., -0.0040,  0.0033, -0.0121],
        [ 0.0014,  0.0287,  0.0080,  ...,  0.0212, -0.0089,  0.0004],
        [ 0.0048,  0.0057, -0.0033,  ...,  0.0020,  0.0221,  0.0104],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.14.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[-1.2207e-02,  1.0803e-02, -1.4343e-03,  ..., -5.4016e-03,
          8.6060e-03, -5.8746e-04],
        [ 7.9956e-03, -1.0315e-02,  3.3112e-03,  ...,  1.4404e-02,
         -1.1108e-02,  6.5308e-03],
        [ 3.3875e-03,  5.6763e-03,  4.2114e-03,  ...,  1.4221e-02,
         -4.1485e-05, -1.4954e-02],
        ...,
        [-3.6316e-03,  1.4404e-02,  3.8338e-04,  ...,  9.8267e-03,
         -5.3711e-03,  7.0190e-03],
        [-3.2501e-03, -1.1658e-02, -9.5215e-03,  ..., -7.5684e-03,
          1.5564e-02,  6.7520e-04],
        [-5.3101e-03, -1.3794e-02,  1.5198e-02,  ...,  1.5137e-02,
          9.8877e-03, -1.0620e-02]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.14.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.15.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[-0.0063, -0.0195, -0.0057,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0250, -0.0228,  0.0089,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0061, -0.0075,  0.0029,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0444, -0.0342,  0.0118,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0173, -0.0554,  0.0186,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0184,  0.0193,  0.0280,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.15.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[-0.0115, -0.0127,  0.0095,  ...,  0.0086, -0.0065, -0.0081],
        [ 0.0031, -0.0070, -0.0092,  ..., -0.0047, -0.0025, -0.0034],
        [ 0.0110, -0.0095, -0.0103,  ...,  0.0134,  0.0077, -0.0134],
        ...,
        [-0.0019, -0.0032,  0.0010,  ..., -0.0049,  0.0005,  0.0059],
        [-0.0107, -0.0049,  0.0035,  ..., -0.0017, -0.0084,  0.0148],
        [-0.0043, -0.0010,  0.0084,  ...,  0.0124, -0.0099, -0.0092]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.15.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.15.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[ 0.0211, -0.0065, -0.0112,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0120, -0.0095, -0.0076,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0048,  0.0055, -0.0051,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0442, -0.0479,  0.0245,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0092,  0.0050,  0.0114,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0242, -0.0087,  0.0325,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.15.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0142,  0.0095,  0.0042,  ..., -0.0099,  0.0053,  0.0121],
        [-0.0003, -0.0038, -0.0151,  ..., -0.0052, -0.0104,  0.0009],
        [ 0.0098,  0.0021,  0.0059,  ...,  0.0140,  0.0142,  0.0128],
        ...,
        [-0.0075,  0.0109, -0.0055,  ...,  0.0128,  0.0073, -0.0045],
        [ 0.0095, -0.0087, -0.0037,  ..., -0.0021,  0.0019, -0.0092],
        [-0.0107,  0.0057, -0.0052,  ..., -0.0045,  0.0145,  0.0132]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.15.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.15.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[ 0.0087, -0.0004,  0.0042,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0110, -0.0089,  0.0148,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0054, -0.0354, -0.0071,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0048, -0.0034,  0.0136,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0055,  0.0038, -0.0273,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0232,  0.0034,  0.0193,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.15.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0129,  0.0045, -0.0067,  ...,  0.0047,  0.0080,  0.0082],
        [-0.0092, -0.0097, -0.0146,  ..., -0.0098, -0.0143,  0.0040],
        [ 0.0138,  0.0011,  0.0005,  ..., -0.0119,  0.0048,  0.0122],
        ...,
        [-0.0011, -0.0125, -0.0037,  ..., -0.0119,  0.0037,  0.0137],
        [-0.0044, -0.0148, -0.0019,  ..., -0.0102,  0.0119,  0.0115],
        [ 0.0006,  0.0050,  0.0072,  ..., -0.0119, -0.0060,  0.0015]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.15.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.15.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[ 0.0019,  0.0072, -0.0153,  ...,  0.0143, -0.0089,  0.0239],
        [-0.0056, -0.0047, -0.0244,  ...,  0.0137, -0.0325,  0.0181],
        [ 0.0059, -0.0078, -0.0042,  ...,  0.0038, -0.0029,  0.0067],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.15.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[ 6.1951e-03, -1.4099e-02, -1.1292e-02,  ..., -7.5989e-03,
          1.4099e-02, -1.4709e-02],
        [-7.5378e-03, -3.7537e-03, -1.1597e-02,  ...,  1.0742e-02,
          2.1667e-03, -3.4180e-03],
        [ 1.1475e-02, -7.4005e-04, -4.7913e-03,  ...,  9.0942e-03,
         -4.7607e-03,  8.9111e-03],
        ...,
        [ 4.1246e-05,  1.7242e-03, -4.3640e-03,  ...,  7.5684e-03,
         -1.3733e-04, -1.2146e-02],
        [-1.1780e-02,  1.2451e-02, -1.4587e-02,  ...,  1.3611e-02,
         -1.5137e-02, -2.5635e-03],
        [-7.9346e-03,  8.1787e-03, -6.6223e-03,  ..., -5.5542e-03,
          1.2329e-02,  3.8757e-03]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.15.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.16.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[ 0.0108,  0.0079, -0.0220,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0194, -0.0425,  0.0175,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0112, -0.0018,  0.0055,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0229, -0.0308, -0.0017,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0146, -0.0098,  0.0262,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0067,  0.0157,  0.0179,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.16.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[-0.0020, -0.0131,  0.0074,  ..., -0.0108,  0.0114, -0.0146],
        [-0.0066, -0.0103, -0.0113,  ...,  0.0132, -0.0063, -0.0136],
        [ 0.0031, -0.0005,  0.0125,  ..., -0.0024,  0.0061,  0.0099],
        ...,
        [-0.0049,  0.0071,  0.0107,  ..., -0.0124, -0.0050,  0.0106],
        [ 0.0044,  0.0027, -0.0112,  ...,  0.0125, -0.0058, -0.0108],
        [-0.0023,  0.0135,  0.0049,  ..., -0.0020,  0.0015, -0.0073]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.16.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.16.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[-0.0178,  0.0197,  0.0078,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0226, -0.0349,  0.0091,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0287, -0.0033, -0.0139,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0162,  0.0121, -0.0136,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0177,  0.0102,  0.0121,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0374,  0.0532,  0.0623,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.16.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[-0.0029, -0.0132,  0.0061,  ..., -0.0034,  0.0006, -0.0092],
        [-0.0109,  0.0043,  0.0103,  ...,  0.0118, -0.0146, -0.0067],
        [-0.0061,  0.0034, -0.0065,  ...,  0.0122, -0.0109,  0.0084],
        ...,
        [ 0.0089, -0.0098, -0.0068,  ..., -0.0129,  0.0116, -0.0143],
        [ 0.0004, -0.0031,  0.0056,  ...,  0.0014, -0.0101,  0.0077],
        [-0.0085, -0.0124,  0.0090,  ...,  0.0064, -0.0108, -0.0151]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.16.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.16.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[-0.0095,  0.0023, -0.0059,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0562, -0.0061, -0.0056,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0033, -0.0048, -0.0063,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0085, -0.0014,  0.0141,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0208,  0.0002,  0.0045,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0130,  0.0107,  0.0084,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.16.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[-0.0056, -0.0089,  0.0121,  ...,  0.0084,  0.0112, -0.0109],
        [ 0.0033,  0.0103,  0.0066,  ...,  0.0013, -0.0120, -0.0026],
        [-0.0055,  0.0046,  0.0147,  ...,  0.0079, -0.0121,  0.0140],
        ...,
        [-0.0033,  0.0054,  0.0095,  ...,  0.0092, -0.0137, -0.0122],
        [-0.0098, -0.0065,  0.0145,  ...,  0.0079,  0.0065, -0.0055],
        [-0.0032,  0.0008,  0.0129,  ...,  0.0143,  0.0104,  0.0051]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.16.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.16.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[ 4.3457e-02, -1.1292e-03, -1.9165e-02,  ...,  7.4463e-03,
         -1.5076e-02,  1.1749e-03],
        [-2.5513e-02,  1.1292e-02, -2.3315e-02,  ..., -5.4016e-03,
         -1.0729e-06, -1.2665e-03],
        [-2.8564e-02, -1.0498e-02, -5.7983e-03,  ...,  2.6398e-03,
          2.8076e-02,  1.3245e-02],
        ...,
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00]], device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.16.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[-0.0154,  0.0081,  0.0102,  ...,  0.0026, -0.0134,  0.0107],
        [ 0.0018, -0.0034, -0.0087,  ...,  0.0123, -0.0054,  0.0043],
        [ 0.0028,  0.0013, -0.0119,  ..., -0.0013,  0.0058,  0.0129],
        ...,
        [ 0.0005, -0.0035,  0.0030,  ...,  0.0048,  0.0012,  0.0033],
        [-0.0149,  0.0059, -0.0093,  ...,  0.0089, -0.0076,  0.0077],
        [ 0.0001,  0.0145, -0.0058,  ..., -0.0132,  0.0107,  0.0084]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.16.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.17.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[-0.0165, -0.0136,  0.0143,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0114, -0.0089,  0.0195,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0137, -0.0018, -0.0114,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0376, -0.0283,  0.0559,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0223, -0.0226,  0.0459,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0189, -0.0096,  0.0142,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.17.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[-0.0103, -0.0125,  0.0073,  ..., -0.0036,  0.0125,  0.0132],
        [-0.0114,  0.0127, -0.0151,  ..., -0.0088, -0.0073, -0.0089],
        [ 0.0111,  0.0057, -0.0072,  ...,  0.0117,  0.0129, -0.0045],
        ...,
        [-0.0044,  0.0118,  0.0110,  ..., -0.0022, -0.0047, -0.0070],
        [-0.0044,  0.0059, -0.0003,  ...,  0.0020,  0.0070, -0.0086],
        [ 0.0010, -0.0042,  0.0134,  ...,  0.0002, -0.0116, -0.0129]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.17.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.17.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[-0.0148,  0.0040,  0.0059,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0059,  0.0067,  0.0073,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0027, -0.0106,  0.0031,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0008, -0.0615, -0.0119,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0254,  0.0217,  0.0138,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0400, -0.0454, -0.0031,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.17.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0117,  0.0052,  0.0081,  ..., -0.0084, -0.0044, -0.0016],
        [-0.0053,  0.0081,  0.0084,  ...,  0.0049, -0.0070, -0.0103],
        [ 0.0141,  0.0027,  0.0065,  ..., -0.0036, -0.0114, -0.0063],
        ...,
        [ 0.0127, -0.0093, -0.0115,  ..., -0.0153, -0.0151, -0.0104],
        [-0.0029, -0.0148, -0.0022,  ..., -0.0071, -0.0138,  0.0061],
        [ 0.0098,  0.0128,  0.0117,  ..., -0.0056,  0.0026, -0.0114]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.17.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.17.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[-0.0143,  0.0087,  0.0063,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0024, -0.0125, -0.0090,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0130,  0.0090, -0.0019,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0093, -0.0044,  0.0146,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0173,  0.0033, -0.0403,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0087, -0.0079, -0.0013,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.17.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0074,  0.0050, -0.0063,  ..., -0.0135,  0.0133, -0.0034],
        [-0.0150, -0.0028,  0.0152,  ...,  0.0103,  0.0040,  0.0152],
        [ 0.0020,  0.0087,  0.0038,  ..., -0.0046, -0.0081, -0.0073],
        ...,
        [ 0.0003, -0.0079,  0.0067,  ..., -0.0148, -0.0073, -0.0082],
        [ 0.0119, -0.0047, -0.0015,  ..., -0.0022, -0.0103, -0.0094],
        [-0.0033, -0.0004, -0.0010,  ..., -0.0146, -0.0061, -0.0123]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.17.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.17.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[-0.0043, -0.0055,  0.0069,  ..., -0.0193,  0.0032, -0.0109],
        [-0.0146, -0.0232,  0.0240,  ...,  0.0265, -0.0069,  0.0203],
        [ 0.0118, -0.0135, -0.0056,  ..., -0.0153, -0.0030, -0.0168],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.17.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0137,  0.0116,  0.0064,  ...,  0.0075,  0.0122, -0.0142],
        [ 0.0039, -0.0052, -0.0008,  ..., -0.0082, -0.0149,  0.0067],
        [ 0.0060, -0.0123, -0.0052,  ..., -0.0011, -0.0056,  0.0073],
        ...,
        [-0.0042, -0.0125, -0.0063,  ..., -0.0056,  0.0035, -0.0087],
        [-0.0061,  0.0106, -0.0022,  ...,  0.0123,  0.0132,  0.0061],
        [ 0.0097, -0.0101, -0.0013,  ...,  0.0135, -0.0020, -0.0125]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.17.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.18.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[ 0.0089,  0.0049,  0.0028,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0022, -0.0073,  0.0120,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0036, -0.0190, -0.0162,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0310,  0.0128, -0.0073,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0649, -0.0167, -0.0271,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0260,  0.0195, -0.0006,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.18.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0003, -0.0064, -0.0045,  ...,  0.0017,  0.0063, -0.0111],
        [-0.0065,  0.0012,  0.0099,  ..., -0.0066, -0.0086, -0.0008],
        [-0.0120,  0.0152, -0.0074,  ...,  0.0149, -0.0066,  0.0002],
        ...,
        [-0.0104, -0.0061, -0.0141,  ...,  0.0130, -0.0042,  0.0061],
        [-0.0044,  0.0078, -0.0064,  ...,  0.0009, -0.0030,  0.0073],
        [-0.0014,  0.0091, -0.0012,  ...,  0.0140,  0.0150, -0.0049]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.18.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.18.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[-0.0126,  0.0264, -0.0093,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0078, -0.0172,  0.0096,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0106, -0.0015,  0.0042,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.1099, -0.0198, -0.0669,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0461,  0.0400, -0.0106,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0452, -0.0435, -0.0452,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.18.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[-0.0096, -0.0035,  0.0103,  ..., -0.0071, -0.0046, -0.0138],
        [-0.0135, -0.0094, -0.0028,  ...,  0.0130, -0.0115, -0.0032],
        [ 0.0067,  0.0155,  0.0082,  ..., -0.0110, -0.0042,  0.0131],
        ...,
        [ 0.0107,  0.0125, -0.0021,  ...,  0.0010,  0.0136,  0.0092],
        [-0.0050, -0.0030, -0.0048,  ..., -0.0065, -0.0092, -0.0093],
        [ 0.0019, -0.0148,  0.0032,  ...,  0.0068,  0.0008,  0.0128]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.18.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.18.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[-0.0266,  0.0175, -0.0197,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0115,  0.0041, -0.0059,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0024,  0.0115, -0.0053,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0135,  0.0145, -0.0042,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0029, -0.0040,  0.0017,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0015,  0.0074,  0.0070,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.18.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[-0.0019,  0.0001,  0.0149,  ...,  0.0048, -0.0070, -0.0135],
        [-0.0114, -0.0104, -0.0024,  ...,  0.0037,  0.0008,  0.0118],
        [ 0.0148,  0.0089,  0.0088,  ..., -0.0154,  0.0139, -0.0122],
        ...,
        [-0.0018,  0.0117,  0.0144,  ...,  0.0124, -0.0109,  0.0078],
        [ 0.0096, -0.0118,  0.0074,  ...,  0.0139,  0.0067, -0.0093],
        [-0.0063, -0.0111, -0.0103,  ..., -0.0135, -0.0008,  0.0146]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.18.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.18.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[-0.0168,  0.0122, -0.0137,  ...,  0.0096,  0.0278,  0.0172],
        [-0.0039, -0.0361,  0.0087,  ...,  0.0033, -0.0072, -0.0280],
        [-0.0040, -0.0165,  0.0152,  ...,  0.0258,  0.0179,  0.0304],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.18.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0056,  0.0104,  0.0041,  ..., -0.0129,  0.0093, -0.0117],
        [ 0.0151, -0.0131, -0.0007,  ..., -0.0060,  0.0055, -0.0112],
        [-0.0154,  0.0049, -0.0006,  ..., -0.0125, -0.0042,  0.0049],
        ...,
        [ 0.0077, -0.0012, -0.0131,  ...,  0.0066, -0.0023,  0.0080],
        [-0.0009, -0.0112,  0.0128,  ...,  0.0025, -0.0002,  0.0036],
        [-0.0059,  0.0052, -0.0027,  ..., -0.0091, -0.0023,  0.0035]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.18.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.19.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[ 0.0026,  0.0068, -0.0078,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0049, -0.0099,  0.0014,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0089,  0.0244,  0.0070,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0254, -0.0187, -0.0038,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0066,  0.0359,  0.0266,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0129,  0.0275,  0.0527,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.19.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0081, -0.0107,  0.0062,  ..., -0.0045,  0.0154,  0.0140],
        [-0.0004, -0.0063,  0.0150,  ...,  0.0131,  0.0056, -0.0137],
        [-0.0098,  0.0150,  0.0060,  ...,  0.0050,  0.0011,  0.0055],
        ...,
        [-0.0060, -0.0023,  0.0109,  ...,  0.0082,  0.0096, -0.0155],
        [ 0.0034, -0.0121, -0.0153,  ...,  0.0096, -0.0099,  0.0108],
        [-0.0145,  0.0090,  0.0073,  ...,  0.0009, -0.0014,  0.0139]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.19.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.19.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[ 0.0184,  0.0120,  0.0015,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0190,  0.0215, -0.0190,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0208,  0.0070, -0.0182,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0011,  0.0430, -0.0135,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0708, -0.0332, -0.0026,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0186,  0.0088, -0.0212,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.19.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[-1.2695e-02,  1.5503e-02,  1.3733e-03,  ...,  9.2163e-03,
          4.1504e-03,  9.2316e-04],
        [ 1.0925e-02,  4.8828e-03,  2.6855e-03,  ..., -1.2360e-03,
         -1.6937e-03,  1.2695e-02],
        [-6.6833e-03, -1.0742e-02,  1.1902e-03,  ..., -2.6703e-03,
          1.3245e-02,  5.6458e-03],
        ...,
        [-1.1047e-02, -1.0586e-04,  7.2098e-04,  ..., -3.5400e-03,
          1.4038e-02, -1.2146e-02],
        [-6.5918e-03,  1.3855e-02,  8.5449e-03,  ..., -1.5198e-02,
         -5.3101e-03,  1.3794e-02],
        [ 1.0071e-02, -7.2479e-05,  5.7373e-03,  ...,  6.7139e-03,
         -1.4465e-02,  3.2654e-03]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.19.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.19.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[ 3.5248e-03, -5.3711e-03, -3.8147e-03,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [ 2.5330e-03,  2.2827e-02, -1.8978e-04,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [-4.6875e-02, -2.7222e-02, -5.2185e-03,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        ...,
        [ 3.6001e-05, -3.8086e-02, -3.4790e-03,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [ 1.4099e-02, -3.9978e-03, -2.3926e-02,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [-1.5488e-03,  2.0996e-02, -1.6174e-03,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00]], device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.19.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[-0.0150, -0.0108,  0.0107,  ...,  0.0093, -0.0075, -0.0014],
        [ 0.0052,  0.0049, -0.0100,  ..., -0.0112, -0.0139,  0.0006],
        [-0.0034,  0.0095,  0.0014,  ...,  0.0025,  0.0069,  0.0135],
        ...,
        [ 0.0003,  0.0146, -0.0010,  ..., -0.0044, -0.0027,  0.0045],
        [-0.0145, -0.0134, -0.0039,  ..., -0.0087,  0.0029, -0.0021],
        [-0.0080, -0.0004,  0.0016,  ..., -0.0072, -0.0015, -0.0128]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.19.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.19.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[ 0.0219, -0.0559,  0.0238,  ...,  0.0005,  0.0007, -0.0244],
        [ 0.0255, -0.0066,  0.0034,  ..., -0.0126,  0.0071, -0.0055],
        [-0.0036, -0.0134, -0.0242,  ...,  0.0041,  0.0096, -0.0021],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.19.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[-4.7922e-05, -5.6763e-03,  6.6757e-05,  ..., -1.2451e-02,
         -1.5137e-02,  1.4832e-02],
        [-9.4604e-03,  1.0193e-02,  4.1504e-03,  ..., -6.8054e-03,
         -1.2817e-02, -1.2146e-02],
        [ 6.6833e-03, -5.9814e-03,  1.7776e-03,  ..., -6.1035e-03,
          8.9111e-03, -4.6692e-03],
        ...,
        [-3.9978e-03, -1.3550e-02,  1.2634e-02,  ...,  1.0315e-02,
         -4.4250e-03, -1.3000e-02],
        [ 3.8910e-03,  1.1108e-02,  4.4556e-03,  ..., -1.8845e-03,
          1.5442e-02,  5.3101e-03],
        [-4.8256e-04,  6.9885e-03, -6.1340e-03,  ...,  7.1411e-03,
          3.7537e-03, -1.3855e-02]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.19.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.20.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[ 0.0013, -0.0057,  0.0166,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0060, -0.0201,  0.0096,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0035, -0.0061, -0.0084,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0222, -0.0430,  0.0110,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0703, -0.0173,  0.0114,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0179, -0.0014, -0.0059,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.20.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[-0.0056, -0.0135,  0.0018,  ..., -0.0016, -0.0086, -0.0124],
        [ 0.0085,  0.0104, -0.0066,  ..., -0.0059, -0.0147, -0.0034],
        [-0.0079,  0.0053,  0.0001,  ...,  0.0015, -0.0026,  0.0067],
        ...,
        [-0.0084,  0.0012,  0.0124,  ...,  0.0066, -0.0111, -0.0005],
        [ 0.0042, -0.0034,  0.0114,  ..., -0.0145, -0.0053,  0.0103],
        [-0.0051,  0.0054,  0.0051,  ..., -0.0011, -0.0107,  0.0061]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.20.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.20.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[-0.0032,  0.0048,  0.0081,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0029,  0.0090,  0.0036,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0041,  0.0056, -0.0062,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0002, -0.0031, -0.0229,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0175,  0.0195, -0.0024,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0164, -0.0269, -0.0116,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.20.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0022, -0.0148, -0.0123,  ..., -0.0012,  0.0059, -0.0032],
        [ 0.0138,  0.0096,  0.0124,  ...,  0.0093,  0.0090,  0.0036],
        [-0.0047,  0.0095, -0.0080,  ..., -0.0090,  0.0156, -0.0144],
        ...,
        [-0.0047, -0.0040, -0.0047,  ...,  0.0070, -0.0023, -0.0152],
        [-0.0073, -0.0011,  0.0102,  ..., -0.0038, -0.0145,  0.0128],
        [-0.0033, -0.0081,  0.0056,  ...,  0.0138,  0.0073,  0.0142]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.20.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.20.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[-0.0041, -0.0126,  0.0147,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0070, -0.0413, -0.0003,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0078, -0.0134,  0.0005,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0066, -0.0083, -0.0044,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0173,  0.0101, -0.0096,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0025, -0.0056,  0.0258,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.20.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[-7.1106e-03,  9.0332e-03, -5.8899e-03,  ..., -1.2451e-02,
         -1.2878e-02, -4.2725e-03],
        [ 8.3618e-03, -6.5613e-03,  4.0894e-03,  ..., -2.9449e-03,
         -1.0742e-02, -1.1963e-02],
        [ 8.3008e-03, -1.1108e-02, -5.0049e-03,  ..., -3.0518e-05,
         -1.2268e-02, -1.8158e-03],
        ...,
        [ 1.2390e-02, -1.0559e-02,  1.3000e-02,  ...,  7.9956e-03,
          6.5308e-03,  7.0190e-03],
        [-8.4839e-03, -6.6223e-03, -6.1035e-04,  ..., -3.8757e-03,
          1.2329e-02, -4.1504e-03],
        [ 5.6152e-03,  3.1738e-03,  9.9487e-03,  ...,  1.0681e-02,
          7.3547e-03, -8.4229e-03]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.20.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.20.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[ 0.0066,  0.0041,  0.0305,  ..., -0.0042,  0.0164, -0.0100],
        [ 0.0102, -0.0131, -0.0087,  ...,  0.0014, -0.0116,  0.0215],
        [ 0.0128, -0.0036, -0.0118,  ..., -0.0269,  0.0015,  0.0092],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.20.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0095, -0.0080, -0.0028,  ...,  0.0046, -0.0140,  0.0060],
        [-0.0112, -0.0120,  0.0008,  ..., -0.0137,  0.0076,  0.0143],
        [ 0.0006,  0.0122, -0.0036,  ...,  0.0130, -0.0121, -0.0033],
        ...,
        [ 0.0143,  0.0058,  0.0078,  ..., -0.0040,  0.0094,  0.0147],
        [ 0.0064,  0.0129,  0.0073,  ..., -0.0145,  0.0043, -0.0098],
        [ 0.0066,  0.0058, -0.0048,  ..., -0.0057, -0.0149, -0.0087]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.20.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.21.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[-0.0181, -0.0054,  0.0166,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0154, -0.0021, -0.0043,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0095, -0.0070,  0.0008,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0024, -0.0082,  0.0295,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0126,  0.0168,  0.0197,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0747, -0.0110, -0.0092,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.21.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0006, -0.0014, -0.0110,  ...,  0.0078, -0.0133, -0.0012],
        [-0.0040,  0.0011, -0.0120,  ...,  0.0155, -0.0053,  0.0146],
        [-0.0126,  0.0076,  0.0121,  ..., -0.0047, -0.0028,  0.0079],
        ...,
        [-0.0120,  0.0004, -0.0137,  ...,  0.0040,  0.0079, -0.0060],
        [ 0.0121,  0.0083,  0.0095,  ..., -0.0114,  0.0148,  0.0010],
        [-0.0095, -0.0074, -0.0093,  ..., -0.0156,  0.0069, -0.0078]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.21.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.21.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[ 0.0041,  0.0107,  0.0046,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0022,  0.0109,  0.0141,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0093,  0.0031,  0.0312,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0183,  0.0200,  0.0057,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0005,  0.0038, -0.0138,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0232,  0.0054, -0.0214,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.21.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[-0.0116, -0.0131, -0.0103,  ...,  0.0031,  0.0068, -0.0025],
        [ 0.0069, -0.0084, -0.0045,  ...,  0.0068, -0.0107,  0.0009],
        [ 0.0018, -0.0045,  0.0104,  ..., -0.0020, -0.0115,  0.0052],
        ...,
        [-0.0040,  0.0040, -0.0046,  ..., -0.0148, -0.0017,  0.0072],
        [ 0.0036, -0.0092,  0.0085,  ...,  0.0018, -0.0151,  0.0055],
        [ 0.0046,  0.0129,  0.0016,  ..., -0.0148, -0.0136,  0.0045]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.21.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.21.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[ 0.0011,  0.0074, -0.0023,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0276,  0.0260, -0.0068,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0223,  0.0024, -0.0042,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0056, -0.0248,  0.0337,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0182, -0.0118,  0.0134,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0091,  0.0187, -0.0131,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.21.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[ 1.2878e-02, -1.2878e-02, -1.1597e-02,  ..., -9.1553e-03,
          2.5787e-03, -2.9297e-03],
        [-6.8665e-03,  1.4343e-02, -1.4771e-02,  ..., -7.4463e-03,
          7.0801e-03,  5.4598e-05],
        [-1.3504e-03,  1.4893e-02, -1.3306e-02,  ..., -4.1199e-03,
         -1.2634e-02, -8.0566e-03],
        ...,
        [-3.2196e-03, -3.6240e-05, -7.0190e-03,  ...,  8.1635e-04,
         -8.6670e-03,  3.8757e-03],
        [-5.5847e-03, -1.5259e-02,  1.1047e-02,  ...,  1.3184e-02,
         -1.4648e-02,  1.0925e-02],
        [-3.4637e-03,  1.4221e-02, -1.9360e-04,  ..., -1.5106e-03,
          1.1520e-03, -6.1340e-03]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.21.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.21.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[-0.0128, -0.0006,  0.0127,  ..., -0.0088,  0.0266,  0.0012],
        [-0.0044,  0.0231, -0.0125,  ..., -0.0302,  0.0093, -0.0074],
        [-0.0183, -0.0121,  0.0055,  ...,  0.0266,  0.0309,  0.0156],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.21.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0135, -0.0151,  0.0123,  ...,  0.0003,  0.0024, -0.0146],
        [ 0.0131,  0.0019, -0.0117,  ...,  0.0112,  0.0141, -0.0019],
        [-0.0152,  0.0117,  0.0132,  ...,  0.0152, -0.0050, -0.0154],
        ...,
        [ 0.0117, -0.0003,  0.0139,  ...,  0.0061, -0.0128, -0.0154],
        [-0.0026, -0.0110,  0.0094,  ...,  0.0069,  0.0132, -0.0143],
        [-0.0153, -0.0150,  0.0096,  ..., -0.0132,  0.0039, -0.0124]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.21.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.22.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[-0.0236, -0.0150, -0.0093,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0413, -0.0171, -0.0032,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0315,  0.0245, -0.0159,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0136,  0.0160,  0.0110,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0277, -0.0037, -0.0259,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0277, -0.0121,  0.0116,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.22.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[-0.0045, -0.0007, -0.0110,  ..., -0.0022, -0.0090, -0.0084],
        [-0.0153,  0.0083,  0.0100,  ...,  0.0144,  0.0090, -0.0003],
        [ 0.0109, -0.0053, -0.0026,  ...,  0.0072,  0.0076, -0.0080],
        ...,
        [ 0.0009, -0.0132, -0.0026,  ..., -0.0129,  0.0125, -0.0136],
        [ 0.0109,  0.0085,  0.0095,  ...,  0.0105, -0.0001,  0.0063],
        [ 0.0115, -0.0118,  0.0065,  ...,  0.0035,  0.0023,  0.0094]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.22.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.22.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[-0.0193, -0.0015, -0.0067,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0036, -0.0147,  0.0298,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0183,  0.0332, -0.0449,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0219,  0.0078, -0.0272,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0128, -0.0461, -0.0068,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0277,  0.0071,  0.0332,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.22.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0023, -0.0045,  0.0050,  ..., -0.0045, -0.0035,  0.0014],
        [ 0.0023,  0.0042,  0.0070,  ...,  0.0107,  0.0049, -0.0041],
        [-0.0092,  0.0043,  0.0129,  ..., -0.0114,  0.0057,  0.0111],
        ...,
        [ 0.0084,  0.0127,  0.0143,  ..., -0.0057, -0.0076,  0.0050],
        [ 0.0133,  0.0104, -0.0072,  ..., -0.0138,  0.0152, -0.0114],
        [ 0.0154,  0.0022, -0.0140,  ...,  0.0096, -0.0041,  0.0004]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.22.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.22.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[-0.0188,  0.0055, -0.0025,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0317, -0.0131,  0.0078,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0267, -0.0250, -0.0085,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0107, -0.0454, -0.0253,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0369, -0.0127,  0.0233,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0030,  0.0208, -0.0393,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.22.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0089, -0.0037, -0.0045,  ...,  0.0110, -0.0044,  0.0092],
        [-0.0063,  0.0042, -0.0152,  ..., -0.0054,  0.0150, -0.0104],
        [-0.0059,  0.0117,  0.0063,  ..., -0.0155, -0.0017, -0.0007],
        ...,
        [ 0.0095, -0.0011, -0.0072,  ...,  0.0134,  0.0101,  0.0048],
        [-0.0135, -0.0001, -0.0045,  ..., -0.0128,  0.0103,  0.0035],
        [-0.0143, -0.0137,  0.0031,  ...,  0.0145, -0.0086,  0.0010]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.22.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.22.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[ 0.0237,  0.0162, -0.0214,  ..., -0.0029, -0.0264,  0.0089],
        [ 0.0334,  0.0133,  0.0033,  ...,  0.0050, -0.0131, -0.0189],
        [ 0.0137,  0.0061, -0.0092,  ..., -0.0275,  0.0062, -0.0029],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.22.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[-1.8387e-03, -1.0315e-02, -1.2634e-02,  ...,  1.0910e-03,
         -3.1281e-03, -9.4604e-03],
        [-1.3245e-02, -3.7670e-05,  5.0659e-03,  ...,  2.2583e-03,
         -1.0986e-02, -1.0437e-02],
        [-3.2806e-03,  5.4016e-03,  1.4465e-02,  ...,  1.2634e-02,
          1.2268e-02, -6.7749e-03],
        ...,
        [ 8.2397e-03,  9.4604e-03, -1.2878e-02,  ...,  6.2256e-03,
         -3.7079e-03, -1.3184e-02],
        [ 8.1177e-03, -5.2490e-03,  4.7607e-03,  ...,  1.0376e-02,
         -7.7820e-04, -8.5449e-03],
        [-1.5503e-02,  3.4943e-03,  5.1575e-03,  ...,  1.6708e-03,
          2.3193e-03, -4.5166e-03]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.22.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.23.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[ 0.0038, -0.0150,  0.0008,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0101, -0.0030,  0.0108,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0055,  0.0031,  0.0189,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0209,  0.0625,  0.0107,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0369, -0.0325,  0.0227,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0104, -0.0588, -0.0013,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.23.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[ 7.5531e-04, -4.4556e-03, -5.7373e-03,  ...,  7.1716e-04,
          7.4768e-04,  5.2795e-03],
        [ 8.1787e-03, -1.0071e-02, -8.0566e-03,  ..., -3.2654e-03,
         -1.5442e-02,  6.9141e-05],
        [-3.6011e-03, -1.2390e-02,  6.6528e-03,  ..., -5.4626e-03,
         -4.8523e-03,  8.7280e-03],
        ...,
        [ 1.5564e-02, -7.9956e-03,  1.5137e-02,  ...,  9.3384e-03,
          1.5564e-02, -8.4229e-03],
        [-6.5613e-03, -1.3916e-02, -1.1230e-02,  ..., -3.6316e-03,
          8.7891e-03,  4.6997e-03],
        [ 1.5106e-03,  5.6763e-03,  1.0010e-02,  ...,  1.2283e-03,
         -8.4229e-03,  4.1809e-03]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.23.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.23.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[-0.0019,  0.0060,  0.0042,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0159, -0.0014, -0.0143,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0094,  0.0040, -0.0129,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0059,  0.0295,  0.0537,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0007, -0.0315, -0.0118,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0145, -0.0076, -0.0221,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.23.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0056,  0.0096, -0.0074,  ..., -0.0090, -0.0039,  0.0099],
        [-0.0139,  0.0144, -0.0128,  ...,  0.0103,  0.0042, -0.0115],
        [-0.0129, -0.0009, -0.0151,  ...,  0.0050,  0.0018, -0.0101],
        ...,
        [ 0.0109, -0.0003,  0.0019,  ..., -0.0037,  0.0019,  0.0092],
        [-0.0089, -0.0028,  0.0057,  ...,  0.0126, -0.0087,  0.0063],
        [-0.0042,  0.0051,  0.0141,  ...,  0.0007, -0.0093,  0.0062]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.23.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.23.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[ 0.0038,  0.0334, -0.0049,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0002, -0.0126, -0.0200,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0096, -0.0162, -0.0265,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0097,  0.0065,  0.0050,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0198,  0.0273,  0.0070,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0179, -0.0132, -0.0566,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.23.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[-0.0126,  0.0060,  0.0005,  ..., -0.0021, -0.0076, -0.0105],
        [-0.0149, -0.0114, -0.0030,  ..., -0.0074, -0.0070, -0.0147],
        [ 0.0052, -0.0042,  0.0098,  ..., -0.0141,  0.0017, -0.0156],
        ...,
        [ 0.0070,  0.0129,  0.0140,  ...,  0.0013, -0.0145,  0.0022],
        [ 0.0019, -0.0030,  0.0100,  ..., -0.0049, -0.0027, -0.0125],
        [ 0.0142, -0.0046, -0.0070,  ...,  0.0122,  0.0062,  0.0102]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.23.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.23.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[-0.0066, -0.0047, -0.0164,  ..., -0.0171,  0.0137, -0.0072],
        [-0.0102, -0.0179, -0.0043,  ..., -0.0029,  0.0126, -0.0031],
        [ 0.0175,  0.0047,  0.0309,  ..., -0.0140,  0.0050, -0.0217],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.23.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0110,  0.0063,  0.0091,  ...,  0.0053, -0.0039, -0.0107],
        [-0.0041, -0.0108,  0.0101,  ...,  0.0146, -0.0134, -0.0040],
        [-0.0101,  0.0044, -0.0116,  ..., -0.0115,  0.0129,  0.0142],
        ...,
        [-0.0072,  0.0153, -0.0089,  ..., -0.0123, -0.0063, -0.0031],
        [-0.0045, -0.0041,  0.0080,  ...,  0.0142,  0.0018,  0.0124],
        [ 0.0019,  0.0144, -0.0088,  ..., -0.0024, -0.0069, -0.0057]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.23.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.24.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[ 0.0209, -0.0162,  0.0126,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0245,  0.0129,  0.0054,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0041, -0.0082, -0.0214,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0056,  0.0010, -0.0205,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0203,  0.0104,  0.0067,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0229, -0.0361, -0.0197,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.24.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0044,  0.0113,  0.0033,  ..., -0.0113,  0.0006, -0.0103],
        [ 0.0034, -0.0150, -0.0155,  ..., -0.0119, -0.0148,  0.0006],
        [-0.0150, -0.0060, -0.0013,  ...,  0.0032,  0.0058,  0.0123],
        ...,
        [-0.0059,  0.0089,  0.0035,  ...,  0.0053,  0.0088, -0.0065],
        [ 0.0023, -0.0109, -0.0156,  ..., -0.0062, -0.0039, -0.0107],
        [-0.0034, -0.0132,  0.0059,  ...,  0.0026, -0.0041, -0.0038]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.24.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.24.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[ 0.0229,  0.0013, -0.0093,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0130, -0.0182,  0.0087,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0092,  0.0041, -0.0442,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0295,  0.0065,  0.0403,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0046,  0.0215,  0.0181,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0231, -0.0188, -0.0061,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.24.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0010, -0.0148, -0.0091,  ...,  0.0004,  0.0012, -0.0140],
        [ 0.0079, -0.0126,  0.0140,  ..., -0.0047,  0.0100,  0.0047],
        [ 0.0087, -0.0009,  0.0063,  ..., -0.0073,  0.0101,  0.0064],
        ...,
        [-0.0026,  0.0095,  0.0146,  ...,  0.0042, -0.0061,  0.0062],
        [ 0.0072,  0.0153, -0.0027,  ..., -0.0125, -0.0036, -0.0149],
        [-0.0137, -0.0109,  0.0057,  ..., -0.0084, -0.0046, -0.0039]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.24.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.24.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[ 0.0271, -0.0201, -0.0048,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0315, -0.0227, -0.0099,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0522, -0.0167, -0.0164,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0011, -0.0179,  0.0154,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0024, -0.0122, -0.0154,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0142,  0.0192,  0.0229,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.24.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0062, -0.0073, -0.0003,  ..., -0.0022,  0.0058, -0.0117],
        [-0.0038,  0.0017,  0.0043,  ...,  0.0130, -0.0125, -0.0063],
        [ 0.0013,  0.0070,  0.0121,  ...,  0.0089,  0.0115,  0.0081],
        ...,
        [ 0.0139,  0.0077,  0.0144,  ..., -0.0077,  0.0082, -0.0090],
        [ 0.0132, -0.0064,  0.0035,  ..., -0.0060, -0.0005, -0.0017],
        [ 0.0142,  0.0101,  0.0107,  ...,  0.0089, -0.0107, -0.0106]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.24.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.24.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[-0.0115,  0.0172,  0.0261,  ...,  0.0067,  0.0014, -0.0165],
        [-0.0102,  0.0286,  0.0135,  ...,  0.0189,  0.0273, -0.0264],
        [-0.0085, -0.0019,  0.0107,  ..., -0.0234,  0.0022, -0.0068],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.24.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[ 5.0354e-03, -2.1667e-03, -1.1780e-02,  ...,  1.2512e-02,
          1.2512e-02, -1.3184e-02],
        [-1.4099e-02, -1.1658e-02,  3.4027e-03,  ..., -4.5166e-03,
          1.1108e-02, -3.9368e-03],
        [ 7.3853e-03, -5.6152e-03, -1.1963e-02,  ..., -1.1719e-02,
          1.0498e-02,  9.5215e-03],
        ...,
        [ 7.9956e-03, -4.1809e-03, -7.3853e-03,  ...,  6.2943e-04,
          6.5918e-03,  2.1905e-06],
        [-2.6398e-03,  1.2573e-02,  1.1719e-02,  ..., -3.0823e-03,
          7.6904e-03, -5.9204e-03],
        [-1.0559e-02, -1.4038e-02,  1.0498e-02,  ..., -1.3428e-02,
         -7.2632e-03,  4.1199e-03]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.24.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.25.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[ 0.0033, -0.0132, -0.0197,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0126,  0.0050,  0.0128,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0013,  0.0056,  0.0026,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0447,  0.0466,  0.0055,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0564, -0.0574, -0.0081,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0403, -0.0422, -0.0243,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.25.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[-5.1575e-03, -7.2327e-03,  2.0752e-03,  ..., -3.9978e-03,
          4.5471e-03, -2.0218e-04],
        [-1.1108e-02,  9.5825e-03, -1.3855e-02,  ...,  1.4648e-02,
         -6.2866e-03, -1.3489e-02],
        [ 1.3611e-02, -1.3672e-02, -1.3046e-03,  ..., -1.0132e-02,
         -5.5847e-03,  6.2561e-04],
        ...,
        [-7.8201e-04,  3.1948e-05, -3.8605e-03,  ...,  1.2268e-02,
          1.5076e-02,  1.4484e-05],
        [-8.2397e-03,  6.5002e-03, -1.1063e-03,  ..., -1.0864e-02,
          1.4099e-02,  1.1047e-02],
        [-1.3550e-02,  6.5613e-03,  4.1504e-03,  ..., -8.4229e-03,
          1.1047e-02, -1.7700e-03]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.25.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.25.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[-0.0102, -0.0134, -0.0058,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0187, -0.0085,  0.0071,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0087,  0.0019, -0.0017,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0048,  0.0525, -0.0117,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0371, -0.0386,  0.0012,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0107, -0.0232,  0.0024,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.25.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0153, -0.0031, -0.0145,  ...,  0.0025,  0.0064, -0.0115],
        [ 0.0148,  0.0130,  0.0069,  ..., -0.0064, -0.0101,  0.0128],
        [ 0.0024, -0.0042, -0.0144,  ...,  0.0092,  0.0123, -0.0046],
        ...,
        [-0.0068, -0.0059,  0.0045,  ..., -0.0084,  0.0031,  0.0020],
        [ 0.0147,  0.0050, -0.0014,  ...,  0.0060,  0.0025,  0.0132],
        [ 0.0154,  0.0044, -0.0154,  ...,  0.0139, -0.0038,  0.0156]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.25.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.25.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[ 0.0085,  0.0110,  0.0059,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0056, -0.0072,  0.0046,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0155,  0.0088, -0.0167,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0179, -0.0311, -0.0070,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0135, -0.0003,  0.0200,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0334, -0.0140, -0.0216,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.25.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0046, -0.0131,  0.0008,  ..., -0.0128,  0.0111, -0.0078],
        [ 0.0090, -0.0120, -0.0124,  ..., -0.0069, -0.0100,  0.0036],
        [-0.0003,  0.0075, -0.0087,  ..., -0.0114,  0.0071, -0.0013],
        ...,
        [ 0.0130, -0.0060,  0.0060,  ...,  0.0132, -0.0090, -0.0156],
        [ 0.0055, -0.0003, -0.0042,  ...,  0.0111, -0.0078, -0.0083],
        [ 0.0038,  0.0054,  0.0066,  ..., -0.0148,  0.0014, -0.0017]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.25.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.25.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[ 0.0182,  0.0139,  0.0011,  ...,  0.0137,  0.0024,  0.0126],
        [-0.0245, -0.0183, -0.0175,  ...,  0.0248,  0.0022, -0.0046],
        [-0.0054, -0.0107,  0.0182,  ..., -0.0105, -0.0157,  0.0025],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.25.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0078,  0.0142, -0.0046,  ..., -0.0037, -0.0099, -0.0092],
        [ 0.0140,  0.0124, -0.0067,  ...,  0.0099, -0.0110, -0.0070],
        [ 0.0049,  0.0078, -0.0140,  ..., -0.0105,  0.0020,  0.0120],
        ...,
        [ 0.0112, -0.0117, -0.0109,  ...,  0.0055,  0.0096,  0.0016],
        [-0.0011, -0.0132,  0.0050,  ..., -0.0149,  0.0062, -0.0103],
        [-0.0151, -0.0028,  0.0122,  ..., -0.0052,  0.0070,  0.0051]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.25.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.26.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[ 0.0332, -0.0053, -0.0087,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0084, -0.0168,  0.0106,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0004,  0.0044,  0.0103,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0012, -0.0110, -0.0078,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0039, -0.0232, -0.0276,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0156,  0.0525,  0.0278,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.26.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[-0.0059, -0.0146, -0.0021,  ...,  0.0040, -0.0003,  0.0077],
        [-0.0111, -0.0060, -0.0022,  ...,  0.0149,  0.0024,  0.0087],
        [-0.0036, -0.0107, -0.0123,  ..., -0.0071, -0.0020, -0.0003],
        ...,
        [-0.0106, -0.0072,  0.0064,  ..., -0.0087, -0.0109,  0.0013],
        [ 0.0154,  0.0116, -0.0085,  ..., -0.0003,  0.0042, -0.0091],
        [ 0.0123,  0.0117, -0.0037,  ...,  0.0006,  0.0137, -0.0053]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.26.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.26.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[ 0.0251,  0.0190, -0.0135,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0267,  0.0085,  0.0142,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0109,  0.0278, -0.0044,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0255,  0.0212,  0.0018,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0312, -0.0347, -0.0092,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0151,  0.0237, -0.0139,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.26.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0114,  0.0029, -0.0070,  ..., -0.0113, -0.0067,  0.0001],
        [-0.0155, -0.0098,  0.0040,  ..., -0.0087,  0.0115,  0.0036],
        [ 0.0002,  0.0008,  0.0058,  ..., -0.0029, -0.0025, -0.0071],
        ...,
        [-0.0081, -0.0012,  0.0058,  ...,  0.0082, -0.0125,  0.0064],
        [ 0.0142, -0.0025, -0.0088,  ...,  0.0078,  0.0128, -0.0155],
        [ 0.0074,  0.0106,  0.0105,  ..., -0.0074, -0.0106,  0.0095]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.26.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.26.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[ 0.0574, -0.0261, -0.0253,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0547, -0.0214,  0.0258,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0021,  0.0018, -0.0278,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0228, -0.0145,  0.0035,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0081, -0.0325, -0.0052,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0061,  0.0223, -0.0110,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.26.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0140, -0.0001, -0.0051,  ..., -0.0113, -0.0036, -0.0016],
        [ 0.0115, -0.0118,  0.0131,  ...,  0.0114, -0.0036, -0.0070],
        [-0.0052,  0.0027, -0.0139,  ..., -0.0045,  0.0031,  0.0096],
        ...,
        [ 0.0051, -0.0090, -0.0016,  ..., -0.0044,  0.0085, -0.0140],
        [-0.0029, -0.0038,  0.0108,  ..., -0.0147, -0.0107,  0.0028],
        [-0.0031,  0.0036, -0.0140,  ..., -0.0084, -0.0044,  0.0077]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.26.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.26.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[ 0.0117,  0.0023, -0.0294,  ..., -0.0017,  0.0150,  0.0302],
        [ 0.0080,  0.0491,  0.0084,  ..., -0.0040,  0.0486, -0.0066],
        [-0.0078,  0.0150, -0.0208,  ..., -0.0160,  0.0038, -0.0042],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.26.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0024,  0.0131,  0.0150,  ...,  0.0036,  0.0084, -0.0079],
        [ 0.0101, -0.0057,  0.0112,  ...,  0.0098,  0.0071,  0.0050],
        [-0.0094,  0.0117,  0.0056,  ..., -0.0132, -0.0003, -0.0029],
        ...,
        [-0.0118, -0.0075,  0.0026,  ...,  0.0096,  0.0146, -0.0058],
        [-0.0081, -0.0011,  0.0150,  ...,  0.0039, -0.0021, -0.0135],
        [-0.0087, -0.0049, -0.0017,  ..., -0.0042,  0.0140, -0.0110]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.26.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.27.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[-0.0310,  0.0189,  0.0073,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0190,  0.0408,  0.0251,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0148, -0.0249,  0.0076,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0530,  0.0292, -0.0069,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0026,  0.0317,  0.0159,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0243, -0.0134,  0.0077,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.27.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0126,  0.0078, -0.0029,  ...,  0.0024, -0.0123, -0.0026],
        [-0.0059,  0.0063, -0.0068,  ...,  0.0012,  0.0102, -0.0019],
        [-0.0093,  0.0006,  0.0034,  ..., -0.0142, -0.0110,  0.0036],
        ...,
        [-0.0093, -0.0054,  0.0076,  ..., -0.0035,  0.0056, -0.0139],
        [-0.0102,  0.0149,  0.0031,  ...,  0.0131,  0.0003,  0.0095],
        [-0.0144, -0.0148,  0.0024,  ..., -0.0145,  0.0118, -0.0087]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.27.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.27.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[-0.0067, -0.0066,  0.0090,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0007,  0.0055,  0.0173,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0093, -0.0166,  0.0051,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0060, -0.0146,  0.0339,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0262, -0.0187, -0.0442,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0300,  0.0145, -0.0045,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.27.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0154, -0.0139,  0.0083,  ...,  0.0109,  0.0045, -0.0145],
        [-0.0102, -0.0139, -0.0143,  ..., -0.0099,  0.0082, -0.0058],
        [ 0.0124, -0.0081, -0.0153,  ...,  0.0007,  0.0101, -0.0087],
        ...,
        [-0.0058,  0.0035, -0.0052,  ..., -0.0049,  0.0002, -0.0062],
        [ 0.0112, -0.0073, -0.0045,  ..., -0.0031, -0.0120,  0.0058],
        [-0.0088, -0.0072,  0.0104,  ...,  0.0133,  0.0065, -0.0019]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.27.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.27.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[ 0.0211, -0.0057, -0.0374,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0194,  0.0173,  0.0092,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0277,  0.0038,  0.0231,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0082, -0.0199, -0.0210,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0008, -0.0227, -0.0303,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0208, -0.0237, -0.0189,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.27.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[ 1.1414e-02,  1.3855e-02, -5.4321e-03,  ..., -2.4872e-03,
         -3.4790e-03,  1.2329e-02],
        [ 4.4250e-03,  1.5259e-02,  3.5400e-03,  ..., -3.6774e-03,
          2.7771e-03, -1.4038e-03],
        [-4.7493e-04, -1.7319e-03,  8.5449e-03,  ...,  1.4709e-02,
         -9.5215e-03,  1.0925e-02],
        ...,
        [ 1.3855e-02,  4.2114e-03, -1.3275e-03,  ..., -1.4343e-02,
         -2.1338e-05,  2.5635e-03],
        [-1.8954e-05, -6.1340e-03, -9.9487e-03,  ..., -1.3123e-03,
          6.8359e-03, -1.3733e-03],
        [-6.1951e-03,  1.2268e-02, -1.4771e-02,  ..., -1.0437e-02,
         -3.5858e-03, -2.3956e-03]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.27.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.27.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[ 0.0039, -0.0092,  0.0057,  ...,  0.0317,  0.0099, -0.0283],
        [ 0.0036,  0.0315, -0.0222,  ..., -0.0125, -0.0101,  0.0011],
        [-0.0013, -0.0139, -0.0164,  ...,  0.0165,  0.0072, -0.0056],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.27.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[-0.0053,  0.0126, -0.0019,  ..., -0.0025,  0.0112,  0.0080],
        [ 0.0105,  0.0045, -0.0061,  ..., -0.0073,  0.0154,  0.0020],
        [ 0.0123, -0.0018,  0.0101,  ..., -0.0060, -0.0010, -0.0089],
        ...,
        [ 0.0060,  0.0038, -0.0084,  ..., -0.0082,  0.0115, -0.0057],
        [-0.0010,  0.0079, -0.0034,  ..., -0.0135, -0.0114,  0.0082],
        [ 0.0010,  0.0027,  0.0018,  ..., -0.0098, -0.0090, -0.0092]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.27.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.28.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[-0.0011, -0.0182,  0.0071,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0124, -0.0061, -0.0055,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0131,  0.0055, -0.0079,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0049,  0.0332, -0.0522,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0179, -0.0315, -0.0143,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0108,  0.0154, -0.0493,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.28.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[-0.0040,  0.0129,  0.0063,  ..., -0.0136,  0.0064, -0.0142],
        [ 0.0086,  0.0091, -0.0115,  ..., -0.0051, -0.0064, -0.0125],
        [ 0.0017,  0.0133,  0.0051,  ...,  0.0112,  0.0030,  0.0098],
        ...,
        [-0.0086, -0.0132,  0.0137,  ...,  0.0042,  0.0042,  0.0103],
        [-0.0104,  0.0050,  0.0008,  ...,  0.0109, -0.0088, -0.0121],
        [-0.0002, -0.0121, -0.0044,  ...,  0.0129, -0.0074,  0.0018]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.28.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.28.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[-0.0008, -0.0014,  0.0147,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0145,  0.0022, -0.0209,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0072,  0.0085, -0.0178,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0330, -0.0109,  0.0023,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0435, -0.0223,  0.0131,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0193, -0.0123,  0.0041,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.28.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0133, -0.0153,  0.0141,  ...,  0.0009, -0.0084,  0.0081],
        [-0.0022,  0.0120,  0.0088,  ..., -0.0045,  0.0042, -0.0043],
        [-0.0092, -0.0108, -0.0075,  ...,  0.0058, -0.0066,  0.0056],
        ...,
        [-0.0117,  0.0066, -0.0143,  ...,  0.0125, -0.0131,  0.0037],
        [-0.0092,  0.0054, -0.0107,  ..., -0.0114,  0.0045,  0.0152],
        [-0.0067,  0.0098,  0.0102,  ...,  0.0117,  0.0074,  0.0120]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.28.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.28.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[-0.0256,  0.0081,  0.0265,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0206, -0.0146, -0.0082,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0064,  0.0178, -0.0040,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0276,  0.0053, -0.0042,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0439,  0.0076,  0.0408,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0147,  0.0177, -0.0471,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.28.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[ 1.2329e-02, -1.1353e-02,  1.0193e-02,  ...,  8.5354e-05,
          2.3041e-03,  8.3618e-03],
        [ 5.6152e-03, -1.1841e-02, -1.3550e-02,  ..., -1.3428e-03,
          4.4861e-03, -7.3242e-03],
        [-3.5858e-03,  1.2756e-02, -6.3477e-03,  ...,  1.2634e-02,
          4.1809e-03,  1.2207e-02],
        ...,
        [-1.5442e-02,  2.8687e-03,  3.0975e-03,  ...,  2.3499e-03,
         -6.3705e-04, -1.4709e-02],
        [ 4.9744e-03, -8.2397e-03,  1.1414e-02,  ..., -1.2573e-02,
          2.3346e-03,  1.1719e-02],
        [ 3.7537e-03,  1.4282e-02,  1.1169e-02,  ..., -1.2939e-02,
         -8.1787e-03,  2.9907e-03]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.28.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.28.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[ 0.0101,  0.0243, -0.0454,  ..., -0.0469,  0.0193,  0.0017],
        [-0.0157, -0.0215,  0.0021,  ..., -0.0131,  0.0204,  0.0300],
        [ 0.0284,  0.0131, -0.0542,  ...,  0.0260,  0.0238, -0.0192],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.28.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[ 1.4404e-02, -1.4648e-02, -1.3184e-02,  ...,  1.1475e-02,
         -4.5586e-04, -2.9449e-03],
        [ 7.1716e-03, -1.1658e-02, -2.2888e-03,  ..., -9.0942e-03,
          7.5989e-03, -2.3499e-03],
        [-1.8692e-03,  5.5847e-03, -8.7261e-05,  ..., -1.4343e-02,
          5.7983e-04, -1.5564e-02],
        ...,
        [-1.1414e-02,  9.1171e-04,  3.7994e-03,  ...,  1.4160e-02,
         -1.3245e-02,  1.4771e-02],
        [ 4.4250e-03,  1.2207e-02, -1.3977e-02,  ...,  1.1902e-02,
         -9.3384e-03,  1.4771e-02],
        [-5.0964e-03,  9.7656e-03, -1.1902e-02,  ...,  9.7046e-03,
          1.4771e-02, -1.1475e-02]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.28.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.29.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[ 0.0028,  0.0068, -0.0055,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0242, -0.0211, -0.0256,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0189,  0.0339, -0.0013,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0077,  0.0601, -0.0342,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0236, -0.0234,  0.0198,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0005, -0.0413, -0.0110,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.29.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[-0.0125, -0.0043,  0.0101,  ...,  0.0004,  0.0090,  0.0025],
        [-0.0131, -0.0077, -0.0117,  ..., -0.0148, -0.0105, -0.0089],
        [ 0.0051,  0.0036, -0.0145,  ..., -0.0041, -0.0101, -0.0109],
        ...,
        [-0.0123, -0.0009, -0.0113,  ..., -0.0129, -0.0035, -0.0052],
        [-0.0114,  0.0024,  0.0062,  ...,  0.0087,  0.0046, -0.0091],
        [-0.0098, -0.0079,  0.0069,  ..., -0.0013, -0.0131, -0.0022]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.29.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.29.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[ 0.0381, -0.0104, -0.0168,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0161,  0.0216, -0.0023,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0183,  0.0089,  0.0188,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0347,  0.0378,  0.0057,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0031, -0.0171, -0.0688,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0160, -0.0095, -0.0562,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.29.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[-0.0095,  0.0150, -0.0030,  ..., -0.0142,  0.0012, -0.0022],
        [-0.0101,  0.0020, -0.0046,  ..., -0.0036,  0.0084,  0.0149],
        [-0.0049,  0.0084, -0.0140,  ...,  0.0074,  0.0156,  0.0002],
        ...,
        [ 0.0104,  0.0115, -0.0087,  ...,  0.0115, -0.0109,  0.0077],
        [-0.0098,  0.0032,  0.0070,  ...,  0.0130,  0.0146,  0.0003],
        [ 0.0019, -0.0059, -0.0036,  ..., -0.0063,  0.0075,  0.0065]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.29.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.29.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[ 1.1841e-02,  5.0354e-03,  2.9907e-02,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [-3.0365e-03,  2.4170e-02,  4.0771e-02,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [-1.4404e-02,  5.7373e-03, -8.3618e-03,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        ...,
        [ 8.7280e-03,  2.2217e-02,  1.2146e-02,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [ 4.3631e-05,  3.4668e-02, -2.4292e-02,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [ 2.1076e-04, -1.6968e-02,  6.9580e-03,  ...,  0.0000e+00,
          0.0000e+00,  0.0000e+00]], device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.29.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[-3.4027e-03, -1.3855e-02, -1.3733e-02,  ...,  9.7275e-04,
         -3.2501e-03, -8.6670e-03],
        [ 4.8828e-03, -4.5776e-03, -7.5912e-04,  ...,  1.0437e-02,
          3.0823e-03,  9.2773e-03],
        [ 6.3477e-03, -1.0315e-02, -8.0109e-04,  ...,  1.3000e-02,
          9.8877e-03,  1.3123e-02],
        ...,
        [-7.1411e-03, -8.2970e-05,  2.2430e-03,  ...,  1.0376e-02,
         -1.0620e-02, -9.3384e-03],
        [-8.7280e-03, -6.0120e-03, -4.7302e-03,  ...,  6.4392e-03,
         -5.3406e-03, -3.4637e-03],
        [ 3.2043e-03, -7.6599e-03,  2.5635e-03,  ...,  4.2725e-03,
         -9.3384e-03,  8.1787e-03]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.29.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.29.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[-0.0046,  0.0344,  0.0322,  ...,  0.0020, -0.0175, -0.0074],
        [-0.0203, -0.0217, -0.0094,  ...,  0.0117, -0.0267,  0.0142],
        [ 0.0229,  0.0262, -0.0123,  ..., -0.0371,  0.0162, -0.0183],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.29.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0069,  0.0101, -0.0033,  ...,  0.0143, -0.0039, -0.0070],
        [ 0.0081, -0.0071, -0.0149,  ..., -0.0024, -0.0102, -0.0029],
        [ 0.0114,  0.0118,  0.0154,  ...,  0.0154, -0.0023,  0.0049],
        ...,
        [ 0.0153, -0.0150,  0.0117,  ..., -0.0077,  0.0029,  0.0146],
        [-0.0033, -0.0075, -0.0103,  ...,  0.0149,  0.0016,  0.0049],
        [ 0.0005, -0.0128, -0.0068,  ..., -0.0147, -0.0151,  0.0119]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.29.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.30.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[-0.0004, -0.0140, -0.0013,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0164,  0.0116,  0.0110,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0061, -0.0220,  0.0461,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0086, -0.0041, -0.0034,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0001, -0.0198, -0.0161,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0693, -0.0135,  0.0398,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.30.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[-0.0145, -0.0046,  0.0083,  ...,  0.0064, -0.0132,  0.0089],
        [ 0.0118,  0.0093, -0.0046,  ..., -0.0137,  0.0112,  0.0098],
        [-0.0029,  0.0076, -0.0136,  ...,  0.0156,  0.0079, -0.0120],
        ...,
        [-0.0107,  0.0112, -0.0091,  ..., -0.0064, -0.0140,  0.0137],
        [ 0.0123,  0.0087, -0.0100,  ..., -0.0028, -0.0099,  0.0023],
        [ 0.0099,  0.0091, -0.0035,  ..., -0.0128,  0.0022, -0.0122]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.30.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.30.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[ 0.0264, -0.0223,  0.0004,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0381,  0.0157,  0.0045,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0054, -0.0117,  0.0249,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0074,  0.0261, -0.0003,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0112, -0.0654,  0.0366,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0391, -0.0134,  0.0183,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.30.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0008, -0.0035,  0.0062,  ..., -0.0143,  0.0015, -0.0034],
        [-0.0123, -0.0088, -0.0095,  ..., -0.0139,  0.0025, -0.0076],
        [ 0.0115, -0.0099,  0.0051,  ...,  0.0106,  0.0086,  0.0013],
        ...,
        [-0.0014,  0.0118, -0.0110,  ...,  0.0092,  0.0032, -0.0056],
        [ 0.0044, -0.0016,  0.0039,  ..., -0.0056,  0.0049, -0.0093],
        [-0.0023, -0.0081,  0.0124,  ..., -0.0060, -0.0054,  0.0060]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.30.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.30.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[-0.0024, -0.0354,  0.0256,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0148,  0.0251,  0.0302,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0184,  0.0133, -0.0171,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0187, -0.0058,  0.0007,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0018, -0.0469,  0.0219,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0011,  0.0054,  0.0161,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.30.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[-1.1230e-02,  1.2024e-02, -8.4229e-03,  ...,  3.3617e-05,
         -2.3804e-03,  2.5024e-03],
        [-1.6022e-03,  1.4893e-02, -1.3306e-02,  ..., -1.0071e-02,
          1.3184e-02, -2.8534e-03],
        [-4.4556e-03,  1.1230e-02,  1.3245e-02,  ...,  5.5542e-03,
         -1.1292e-03,  5.4626e-03],
        ...,
        [ 5.6152e-03,  1.6174e-03,  1.2024e-02,  ..., -1.2939e-02,
         -1.4099e-02,  9.7656e-03],
        [-6.0425e-03, -4.5776e-03,  1.3306e-02,  ...,  1.2390e-02,
         -3.1128e-03,  8.0566e-03],
        [ 4.8828e-03, -1.1780e-02,  3.6316e-03,  ..., -1.3367e-02,
          1.5503e-02, -9.3994e-03]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.30.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.30.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[-0.0067,  0.0102, -0.0112,  ...,  0.0078, -0.0172, -0.0222],
        [-0.0203, -0.0199, -0.0007,  ...,  0.0168, -0.0211, -0.0209],
        [-0.0188, -0.0032,  0.0178,  ..., -0.0518, -0.0151, -0.0117],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.30.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0114,  0.0093, -0.0101,  ...,  0.0102, -0.0130, -0.0079],
        [-0.0011,  0.0045, -0.0046,  ..., -0.0151, -0.0016, -0.0024],
        [ 0.0127, -0.0002,  0.0020,  ...,  0.0093, -0.0146,  0.0015],
        ...,
        [ 0.0090,  0.0090, -0.0026,  ..., -0.0125, -0.0104, -0.0110],
        [ 0.0139,  0.0082,  0.0103,  ...,  0.0065,  0.0149, -0.0007],
        [ 0.0107, -0.0134, -0.0047,  ..., -0.0032, -0.0008,  0.0039]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.30.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.31.self_attn.q_proj.base_layer.weightParameter containing:
tensor([[-0.0332,  0.0133,  0.0212,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0082, -0.0215, -0.0337,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0126,  0.0199,  0.0170,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0161, -0.0076, -0.0208,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0194,  0.0040, -0.0098,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0549, -0.0082,  0.0303,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.31.self_attn.q_proj.lora_A.default.weightParameter containing:
tensor([[-0.0079,  0.0146, -0.0079,  ...,  0.0085, -0.0072,  0.0110],
        [ 0.0104,  0.0037, -0.0109,  ..., -0.0040, -0.0047, -0.0076],
        [-0.0150, -0.0133,  0.0002,  ..., -0.0052,  0.0007,  0.0007],
        ...,
        [-0.0031,  0.0020, -0.0142,  ...,  0.0128, -0.0107, -0.0042],
        [-0.0093,  0.0129, -0.0132,  ...,  0.0070,  0.0155,  0.0042],
        [ 0.0098, -0.0048, -0.0032,  ..., -0.0023, -0.0097,  0.0107]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.31.self_attn.q_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.31.self_attn.k_proj.base_layer.weightParameter containing:
tensor([[-0.0126, -0.0219,  0.0136,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0250, -0.0125,  0.0127,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0091,  0.0018, -0.0088,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [ 0.0237, -0.0330, -0.0233,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0159,  0.0172,  0.0170,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0771, -0.0320,  0.0109,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.31.self_attn.k_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0111,  0.0077, -0.0026,  ..., -0.0148, -0.0110, -0.0151],
        [ 0.0148,  0.0129, -0.0050,  ..., -0.0150,  0.0117, -0.0148],
        [ 0.0081,  0.0044, -0.0146,  ..., -0.0090, -0.0103,  0.0057],
        ...,
        [ 0.0146, -0.0077,  0.0106,  ..., -0.0082, -0.0019, -0.0139],
        [-0.0077,  0.0143, -0.0115,  ..., -0.0115,  0.0014, -0.0063],
        [ 0.0078, -0.0030, -0.0021,  ..., -0.0058,  0.0033, -0.0117]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.31.self_attn.k_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.31.self_attn.v_proj.base_layer.weightParameter containing:
tensor([[ 0.0130, -0.0278, -0.0049,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0214,  0.0077,  0.0175,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0121, -0.0034, -0.0160,  ...,  0.0000,  0.0000,  0.0000],
        ...,
        [-0.0028, -0.0085, -0.0140,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0109, -0.0242,  0.0165,  ...,  0.0000,  0.0000,  0.0000],
        [-0.0118, -0.0255,  0.0275,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.31.self_attn.v_proj.lora_A.default.weightParameter containing:
tensor([[ 0.0145,  0.0034,  0.0114,  ..., -0.0117,  0.0095,  0.0061],
        [-0.0036, -0.0137, -0.0009,  ...,  0.0098,  0.0105,  0.0024],
        [-0.0125, -0.0110, -0.0074,  ...,  0.0131,  0.0140,  0.0100],
        ...,
        [ 0.0043,  0.0047,  0.0015,  ..., -0.0053,  0.0156, -0.0063],
        [-0.0072,  0.0035, -0.0031,  ...,  0.0051,  0.0019,  0.0103],
        [ 0.0101,  0.0094, -0.0117,  ..., -0.0144,  0.0151,  0.0034]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.31.self_attn.v_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.31.self_attn.o_proj.base_layer.weightParameter containing:
tensor([[ 0.0067,  0.0277, -0.0039,  ..., -0.0045, -0.0116, -0.0491],
        [-0.0078,  0.0040, -0.0267,  ...,  0.0262, -0.0190, -0.0151],
        [ 0.0113, -0.0002,  0.0024,  ..., -0.0153,  0.0254,  0.0088],
        ...,
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],
       device='cuda:0', dtype=torch.bfloat16)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.31.self_attn.o_proj.lora_A.default.weightParameter containing:
tensor([[-0.0154,  0.0132,  0.0045,  ...,  0.0134,  0.0155, -0.0028],
        [-0.0140, -0.0055, -0.0017,  ..., -0.0118, -0.0014,  0.0053],
        [ 0.0002,  0.0033, -0.0103,  ..., -0.0042,  0.0096,  0.0129],
        ...,
        [ 0.0095, -0.0128, -0.0084,  ..., -0.0036, -0.0011, -0.0118],
        [-0.0038,  0.0134, -0.0074,  ..., -0.0026, -0.0123,  0.0048],
        [ 0.0129, -0.0115, -0.0075,  ...,  0.0020, -0.0095,  0.0145]],
       device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] loader.py:180 >> name: base_model.model.model.layers.31.self_attn.o_proj.lora_B.default.weightParameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0', requires_grad=True)

[INFO|2025-01-18 10:39:36] logging.py:157 >> trainable params: 8,388,608 || all params: 6,758,404,096 || trainable%: 0.1241

[INFO|2025-01-18 10:39:36] trainer.py:698 >> Using auto half precision backend

[INFO|2025-01-18 10:39:37] trainer.py:2313 >> ***** Running training *****

[INFO|2025-01-18 10:39:37] trainer.py:2314 >>   Num examples = 91

[INFO|2025-01-18 10:39:37] trainer.py:2315 >>   Num Epochs = 3

[INFO|2025-01-18 10:39:37] trainer.py:2316 >>   Instantaneous batch size per device = 2

[INFO|2025-01-18 10:39:37] trainer.py:2319 >>   Total train batch size (w. parallel, distributed & accumulation) = 16

[INFO|2025-01-18 10:39:37] trainer.py:2320 >>   Gradient Accumulation steps = 8

[INFO|2025-01-18 10:39:37] trainer.py:2321 >>   Total optimization steps = 15

[INFO|2025-01-18 10:39:37] trainer.py:2322 >>   Number of trainable parameters = 8,388,608

[INFO|2025-01-18 10:40:45] logging.py:157 >> {'loss': 12.9414, 'learning_rate': 3.7500e-05, 'epoch': 0.87}

[INFO|2025-01-18 10:41:54] logging.py:157 >> {'loss': 14.2968, 'learning_rate': 1.2500e-05, 'epoch': 1.80}

[INFO|2025-01-18 10:43:04] logging.py:157 >> {'loss': 14.3239, 'learning_rate': 0.0000e+00, 'epoch': 2.74}

[INFO|2025-01-18 10:43:04] trainer.py:3801 >> Saving model checkpoint to saves/Llama-2-7B/lora/train_2025-01-18-10-37-18/checkpoint-15

[INFO|2025-01-18 10:43:04] configuration_utils.py:677 >> loading configuration file /home/yandong/Documents/um-data/sunxh/PycharmProjects/models/Llama-2-7b-hf-P10/config.json

[INFO|2025-01-18 10:43:04] configuration_utils.py:746 >> Model config LlamaConfig {
  "_name_or_path": "../../models/Llama-2-7b-hf",
  "architectures": [
    "LlamaModel"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "sparsity": 0.1,
  "tie_word_embeddings": false,
  "torch_dtype": "float32",
  "transformers_version": "4.46.1",
  "use_cache": true,
  "vocab_size": 32000
}


[INFO|2025-01-18 10:43:04] tokenization_utils_base.py:2646 >> tokenizer config file saved in saves/Llama-2-7B/lora/train_2025-01-18-10-37-18/checkpoint-15/tokenizer_config.json

[INFO|2025-01-18 10:43:04] tokenization_utils_base.py:2655 >> Special tokens file saved in saves/Llama-2-7B/lora/train_2025-01-18-10-37-18/checkpoint-15/special_tokens_map.json

[INFO|2025-01-18 10:43:05] trainer.py:2584 >> 

Training completed. Do not forget to share your model on huggingface.co/models =)



[INFO|2025-01-18 10:43:05] trainer.py:3801 >> Saving model checkpoint to saves/Llama-2-7B/lora/train_2025-01-18-10-37-18

[INFO|2025-01-18 10:43:05] configuration_utils.py:677 >> loading configuration file /home/yandong/Documents/um-data/sunxh/PycharmProjects/models/Llama-2-7b-hf-P10/config.json

[INFO|2025-01-18 10:43:05] configuration_utils.py:746 >> Model config LlamaConfig {
  "_name_or_path": "../../models/Llama-2-7b-hf",
  "architectures": [
    "LlamaModel"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "sparsity": 0.1,
  "tie_word_embeddings": false,
  "torch_dtype": "float32",
  "transformers_version": "4.46.1",
  "use_cache": true,
  "vocab_size": 32000
}


[INFO|2025-01-18 10:43:05] tokenization_utils_base.py:2646 >> tokenizer config file saved in saves/Llama-2-7B/lora/train_2025-01-18-10-37-18/tokenizer_config.json

[INFO|2025-01-18 10:43:05] tokenization_utils_base.py:2655 >> Special tokens file saved in saves/Llama-2-7B/lora/train_2025-01-18-10-37-18/special_tokens_map.json

[WARNING|2025-01-18 10:43:05] logging.py:162 >> No metric eval_loss to plot.

[WARNING|2025-01-18 10:43:05] logging.py:162 >> No metric eval_accuracy to plot.

[INFO|2025-01-18 10:43:05] modelcard.py:449 >> Dropping the following result as it does not have all the necessary fields:
{'task': {'name': 'Causal Language Modeling', 'type': 'text-generation'}}

"""


