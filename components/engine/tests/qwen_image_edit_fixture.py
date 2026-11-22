"""Complete miniature 2511 pipeline built from random arrays, without downloads."""


def write_checkpoint(root, tokenizer_root):
    import torch
    from diffusers import (
        AutoencoderKLQwenImage,
        FlowMatchEulerDiscreteScheduler,
        QwenImageEditPlusPipeline,
        QwenImageTransformer2DModel,
    )
    from transformers import (
        Qwen2_5_VLConfig,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2TokenizerFast,
        Qwen2VLImageProcessor,
        Qwen2VLProcessor,
        Qwen3VLVideoProcessor,
    )

    torch.manual_seed(31)
    tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_root, local_files_only=True)
    vision = {
        "depth": 2,
        "hidden_size": 16,
        "intermediate_size": 16,
        "num_heads": 2,
        "out_hidden_size": 16,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "fullatt_block_indexes": [0],
        "window_size": 28,
    }
    text = Qwen2_5_VLForConditionalGeneration(
        Qwen2_5_VLConfig(
            text_config={
                "vocab_size": len(tokenizer),
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 1000000.0,
                    "mrope_section": [1, 1, 2],
                },
            },
            vision_config=vision,
            image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
            video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
            vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        )
    ).eval()
    processor = Qwen2VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(
            patch_size=14,
            merge_size=2,
            temporal_patch_size=2,
            size={"shortest_edge": 28**2, "longest_edge": 512**2},
        ),
        video_processor=Qwen3VLVideoProcessor(),
    )
    pipe = QwenImageEditPlusPipeline(
        transformer=QwenImageTransformer2DModel(
            patch_size=2,
            in_channels=16,
            out_channels=4,
            num_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=16,
            axes_dims_rope=(4, 6, 6),
            zero_cond_t=True,
        ),
        vae=AutoencoderKLQwenImage(
            base_dim=4,
            z_dim=4,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            attn_scales=[],
            latents_mean=[0.0] * 4,
            latents_std=[1.0] * 4,
        ),
        text_encoder=text,
        tokenizer=tokenizer,
        processor=processor,
        scheduler=FlowMatchEulerDiscreteScheduler(),
    )
    pipe.save_pretrained(root, safe_serialization=True)
    processor.image_processor.save_pretrained(root / "processor")
    return root
