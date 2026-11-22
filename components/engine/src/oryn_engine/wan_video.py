"""Complete native Wan TI2V graph validation for bounded text-to-video."""

import math

from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

COMPONENTS = ("transformer", "vae", "text_encoder")


def expanded(index):
    return index.get("_class_name") == "WanPipeline" and index.get("expand_timesteps") is True


def validate_configs(configs):
    transformer, vae, text = (configs[k] for k in COMPONENTS)
    for value in configs.values():
        validate_model_config(value)
        if value.get("quantization_config") or value.get("quantization"):
            raise ValueError("WAN_VIDEO_QUANTIZATION_UNSUPPORTED")
    if (
        transformer.get("_class_name") != "WanTransformer3DModel"
        or vae.get("_class_name") != "AutoencoderKLWan"
        or text.get("model_type") != "umt5"
        or text.get("architectures") != ["UMT5EncoderModel"]
    ):
        raise ValueError("WAN_VIDEO_COMPONENT_IDENTITY")
    for value, dimensions in (
        (
            transformer,
            {
                "num_attention_heads": (1, 64),
                "attention_head_dim": (8, 256),
                "ffn_dim": (16, 32768),
                "num_layers": (1, 80),
                "freq_dim": (8, 1024),
                "rope_max_seq_len": (32, 4096),
            },
        ),
        (
            vae,
            {
                "base_dim": (4, 512),
                "decoder_base_dim": (4, 512),
                "z_dim": (4, 64),
                "num_res_blocks": (1, 4),
            },
        ),
        (
            text,
            {
                "d_model": (16, 8192),
                "d_ff": (16, 32768),
                "d_kv": (4, 256),
                "num_heads": (1, 128),
                "num_layers": (1, 80),
                "vocab_size": (32, 300000),
            },
        ),
    ):
        for key, (low, high) in dimensions.items():
            number = value.get(key)
            if type(number) is not int or not low <= number <= high:
                raise ValueError("WAN_VIDEO_DIMENSIONS")
    if (
        transformer.get("patch_size") != [1, 2, 2]
        or transformer.get("in_channels") != vae["z_dim"]
        or transformer.get("out_channels") != vae["z_dim"]
        or transformer.get("text_dim") != text["d_model"]
        or transformer.get("image_dim") is not None
        or transformer.get("added_kv_proj_dim") is not None
        or transformer.get("pos_embed_seq_len") is not None
        or transformer.get("qk_norm") != "rms_norm_across_heads"
        or transformer.get("cross_attn_norm") is not True
        or transformer["attention_head_dim"] % 8
        or transformer["freq_dim"] % 2
        or text["num_heads"] * text["d_kv"] != text["d_model"]
        or text.get("feed_forward_proj") != "gated-gelu"
        or text.get("is_gated_act") is not True
        or text.get("pad_token_id") != 0
        or text.get("eos_token_id") != 1
    ):
        raise ValueError("WAN_VIDEO_GRAPH")
    if (
        vae.get("dim_mult") != [1, 2, 4, 4]
        or vae.get("temperal_downsample") != [False, True, True]
        or vae.get("scale_factor_spatial") != 16
        or vae.get("scale_factor_temporal") != 4
        or vae.get("patch_size") != 2
        or vae.get("is_residual") is not True
        or vae.get("in_channels") != 12
        or vae.get("out_channels") != 12
        or vae.get("attn_scales") != []
        or vae.get("clip_output", False) is not False
    ):
        raise ValueError("WAN_VIDEO_VAE_GEOMETRY")
    for key in ("latents_mean", "latents_std"):
        values = vae.get(key)
        if (
            not isinstance(values, list)
            or len(values) != vae["z_dim"]
            or any(
                type(v) not in {int, float}
                or not math.isfinite(v)
                or abs(v) > 100
                or key == "latents_std"
                and v <= 0
                for v in values
            )
        ):
            raise ValueError("WAN_VIDEO_LATENT_NORMALIZATION")


def read_configs(root):
    configs = {name: metadata_file(root / name / "config.json") for name in COMPONENTS}
    validate_configs(configs)
    return configs


def native_component(name, config):
    from diffusers import AutoencoderKLWan, WanTransformer3DModel
    from transformers import UMT5Config, UMT5EncoderModel

    if name == "text_encoder":
        native = UMT5Config.from_dict(config)
        native._attn_implementation = "eager"
        return UMT5EncoderModel(native)
    return {"transformer": WanTransformer3DModel, "vae": AutoencoderKLWan}[name].from_config(config)


def validate_weight_shapes(configs, graphs):
    import torch

    validate_configs(configs)
    for name in COMPONENTS:
        with torch.device("meta"):
            model = native_component(name, configs[name])
        expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        shapes = graphs[name]
        missing = expected.keys() - shapes.keys()
        if name == "text_encoder" and "shared.weight" in shapes:
            # UMT5's native encoder embedding is always tied to its shared vocabulary.
            missing -= {"encoder.embed_tokens.weight"}
        if (
            missing
            or shapes.keys() - expected.keys()
            or any(tuple(v) != expected[k] for k, v in shapes.items())
        ):
            raise ValueError("WAN_VIDEO_WEIGHTS_INCOMPLETE")


def preflight(root):
    validate_weight_shapes(
        read_configs(root), {name: preflight_safetensors(root / name) for name in COMPONENTS}
    )
