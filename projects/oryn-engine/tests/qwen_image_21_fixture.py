"""Small random native checkpoint; no downloaded weights or model quality claims."""

import json


def write_checkpoint(root):
    import torch
    from diffusers import (
        AutoencoderKLQwenImage21,
        FlowMatchEulerDiscreteScheduler,
        QwenImage21Pipeline,
        QwenImage21Transformer2DModel,
    )
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import Qwen2TokenizerFast, Qwen3VLConfig, Qwen3VLForConditionalGeneration

    from oryn_engine.qwen_image_21 import sealed_processor

    torch.manual_seed(0)
    special = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    vocabulary = dict.fromkeys(special + sorted(pre_tokenizers.ByteLevel.alphabet()))
    vocabulary = {token: index for index, token in enumerate(vocabulary)}
    native = Tokenizer(models.BPE(vocabulary, [], unk_token="<|endoftext|>"))
    native.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    native.decoder = decoders.ByteLevel()
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=native,
        pad_token=special[0],
        eos_token=special[0],
        additional_special_tokens=special[1:],
        model_max_length=2048,
    )
    processor = sealed_processor(
        tokenizer,
        {
            "patch_size": 16,
            "merge_size": 2,
            "temporal_patch_size": 2,
            "size": {"shortest_edge": 32**2, "longest_edge": 1024**2},
        },
    )
    text = Qwen3VLForConditionalGeneration(
        Qwen3VLConfig(
            text_config={
                "vocab_size": len(tokenizer),
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 1000000.0,
                    "mrope_section": [1, 1, 2],
                },
            },
            vision_config={
                "depth": 2,
                "hidden_size": 16,
                "intermediate_size": 16,
                "num_heads": 2,
                "out_hidden_size": 16,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "num_position_embeddings": 64,
                "deepstack_visual_indexes": [0],
            },
            image_token_id=vocabulary["<|image_pad|>"],
            video_token_id=vocabulary["<|video_pad|>"],
            vision_start_token_id=vocabulary["<|vision_start|>"],
            vision_end_token_id=vocabulary["<|vision_end|>"],
        )
    ).eval()
    transformer = QwenImage21Transformer2DModel(
        patch_size=1,
        in_channels=8,
        out_channels=8,
        num_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        context_in_dim=16,
        mlp_ratio=2,
        axes_dims_rope=(4, 6, 6),
    )
    vae = AutoencoderKLQwenImage21(
        base_dim=4,
        decoder_base_dim=4,
        z_dim=8,
        dim_mult=[1, 1, 1, 1, 1],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False, True, True, True],
        latents_mean=[0.0] * 8,
        latents_std=[1.0] * 8,
    )
    pipe = QwenImage21Pipeline(
        transformer=transformer,
        vae=vae,
        scheduler=FlowMatchEulerDiscreteScheduler(),
        text_encoder=text,
        processor=processor,
    )
    pipe.save_pretrained(root, safe_serialization=True)
    processor.image_processor.save_pretrained(root / "processor")
    tokenizer.save_pretrained(root / "processor")
    # Preserve the real checkpoint's legacy image-processor layout.
    (root / "processor/preprocessor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "Qwen3VLProcessor",
                "image_processor_type": "Qwen2VLImageProcessor",
                "patch_size": 16,
                "merge_size": 2,
                "temporal_patch_size": 2,
                "size": {"shortest_edge": 65536, "longest_edge": 16777216},
            }
        )
    )
    (root / "processor/chat_template.jinja").write_text(
        "{{ raise_exception('Repository template executed') }}"
    )
    return root
