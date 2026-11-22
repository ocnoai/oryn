"""Small complete native graphs with generated weights and a byte tokenizer."""

import sys
from pathlib import Path


def create(root: Path, architecture: str, native_source: Path):
    sys.path.insert(0, str(native_source / "gguf-py"))
    import gguf
    import torch
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import (
        Qwen3_5ForCausalLM,
        Qwen3_5MoeForCausalLM,
        Qwen3_5MoeTextConfig,
        Qwen3_5TextConfig,
    )

    tokens = sorted(ByteLevel.alphabet()) + [
        "<unk>",
        "<s>",
        "</s>",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
    ]
    tokens.append("aa")
    moe = architecture == "qwen35moe"
    cls = Qwen3_5MoeTextConfig if moe else Qwen3_5TextConfig
    config = cls(
        vocab_size=len(tokens),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        max_position_embeddings=1024,
        shared_expert_intermediate_size=64,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        full_attention_interval=2,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [6, 5, 5],
            "mrope_interleaved": True,
        },
        bos_token_id=257,
        eos_token_id=258,
    )
    torch.manual_seed(6142)
    torch.set_num_threads(2)
    model = (Qwen3_5MoeForCausalLM if moe else Qwen3_5ForCausalLM)(config).eval()
    with torch.no_grad():
        for i, token in enumerate(tokens):
            if token not in "abcdefghijklmnopqrstuvwxyz":
                model.lm_head.weight[i].zero_()
    kind = gguf.MODEL_ARCH.QWEN35MOE if moe else gguf.MODEL_ARCH.QWEN35
    mapping = gguf.get_tensor_name_map(kind, 2)
    path = root / f"{architecture}.Q8_0.gguf"
    writer = gguf.GGUFWriter(path, architecture)
    writer.add_name("Oryn generated translation graph")
    writer.add_file_type(7)
    writer.add_context_length(1024)
    writer.add_embedding_length(64)
    writer.add_block_count(2)
    writer.add_feed_forward_length(128)
    if moe:
        writer.add_expert_feed_forward_length(64)
        writer.add_expert_count(4)
        writer.add_expert_shared_feed_forward_length(64)
        writer.add_expert_used_count(2)
    writer.add_ssm_conv_kernel(4)
    writer.add_ssm_state_size(32)
    writer.add_ssm_group_count(2)
    writer.add_ssm_time_step_rank(4)
    writer.add_ssm_inner_size(128)
    writer.add_full_attention_interval(2)
    writer.add_rope_dimension_sections([6, 5, 5, 0])
    writer.add_head_count(2)
    writer.add_head_count_kv(1)
    writer.add_key_length(32)
    writer.add_value_length(32)
    writer.add_rope_dimension_count(32)
    writer.add_rope_freq_base(10000.0)
    writer.add_layer_norm_rms_eps(config.rms_norm_eps)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen2")
    writer.add_token_list(tokens)
    writer.add_token_types([1] * 256 + [2, 3, 3, 3, 3, 3, 3] + [1])
    writer.add_token_merges(["a a"])
    writer.add_unk_token_id(256)
    writer.add_bos_token_id(257)
    writer.add_eos_token_id(258)
    writer.add_add_bos_token(False)
    writer.add_chat_template("<|im_start|>user\n{{ message }}<|im_end|>")

    def reorder(value, dim, head):
        shape = list(value.shape)
        split = shape[:dim] + [2, 2, head] + shape[dim + 1 :]
        order = list(range(len(split)))
        order[dim], order[dim + 1] = order[dim + 1], order[dim]
        return value.reshape(split).permute(order).contiguous().reshape(shape)

    for name, value in model.state_dict().items():
        tensors = [(name, value)]
        if name.endswith("experts.gate_up_proj"):
            tensors = [
                (name.replace("gate_up_proj", key + ".weight"), part)
                for key, part in zip(("gate_proj", "up_proj"), value.chunk(2, dim=-2), strict=True)
            ]
        elif name.endswith("experts.down_proj"):
            tensors = [(name + ".weight", value)]
        for key, value in tensors:
            if "linear_attn." in key:
                if ".in_proj_qkv." in key or ".conv1d." in key:
                    if ".conv1d." in key:
                        value = value.squeeze(1)
                    value = torch.cat([value[:128], reorder(value[128:], 0, 32)], dim=0)
                elif ".in_proj_z." in key:
                    value = reorder(value, 0, 32)
                elif any(s in key for s in (".in_proj_a.", ".in_proj_b.", ".A_log", ".dt_bias")):
                    value = reorder(value, 0, 1)
                elif ".out_proj." in key:
                    value = reorder(value, 1, 32)
            if key.endswith(".A_log"):
                value = -torch.exp(value)
            elif key.endswith(".dt_bias"):
                key = key.removesuffix(".dt_bias") + ".dt_proj.bias"
            elif "conv1d" in key:
                value = value.squeeze()
            elif key.endswith("norm.weight") and not key.endswith("linear_attn.norm.weight"):
                value = value + 1
            mapped = mapping.get_name(key, try_suffixes=(".weight", ".bias"))
            assert mapped, key
            values = value.detach().float().numpy()
            quant = (
                gguf.GGMLQuantizationType.Q8_0
                if values.ndim >= 2 and values.shape[-1] % 32 == 0
                else gguf.GGMLQuantizationType.F32
            )
            writer.add_tensor(mapped, gguf.quantize(values, quant), raw_dtype=quant)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path
