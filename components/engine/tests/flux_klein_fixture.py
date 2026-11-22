"""Complete miniature native Klein graph with all 27 text layers; random arrays only."""


def write_checkpoint(root, tokenizer_root):
    import torch
    from diffusers import (
        AutoencoderKLFlux2,
        FlowMatchEulerDiscreteScheduler,
        Flux2KleinPipeline,
        Flux2Transformer2DModel,
    )
    from transformers import Qwen2TokenizerFast, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(37)
    tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_root, local_files_only=True)
    text = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=27,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=1024,
        )
    )
    pipe = Flux2KleinPipeline(
        transformer=Flux2Transformer2DModel(
            in_channels=16,
            out_channels=16,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=48,
            timestep_guidance_channels=16,
            mlp_ratio=2,
            axes_dims_rope=(4, 4, 4, 4),
            guidance_embeds=False,
        ),
        vae=AutoencoderKLFlux2(
            block_out_channels=(8, 8, 8, 8),
            latent_channels=4,
            norm_num_groups=4,
            layers_per_block=1,
            sample_size=64,
        ),
        text_encoder=text,
        tokenizer=tokenizer,
        scheduler=FlowMatchEulerDiscreteScheduler(),
        is_distilled=False,
    )
    pipe.save_pretrained(root, safe_serialization=True)
    return root
