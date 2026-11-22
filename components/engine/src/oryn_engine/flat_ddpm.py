"""Owned loading of reviewed flat UNet/DDPM exports with legacy attention names."""

from __future__ import annotations

import math
import re

from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

MARKER = "_oryn_flat_ddpm"
COMPONENTS = {"unet": ["diffusers", "UNet2DModel"], "scheduler": ["diffusers", "DDPMScheduler"]}
UNET_FIELDS = {
    "_class_name",
    "_diffusers_version",
    "act_fn",
    "attention_head_dim",
    "block_out_channels",
    "center_input_sample",
    "down_block_types",
    "downsample_padding",
    "flip_sin_to_cos",
    "freq_shift",
    "in_channels",
    "layers_per_block",
    "mid_block_scale_factor",
    "norm_eps",
    "norm_num_groups",
    "out_channels",
    "sample_size",
    "time_embedding_type",
    "up_block_types",
}
SCHEDULER_FIELDS = {
    "_class_name",
    "_diffusers_version",
    "beta_end",
    "beta_schedule",
    "beta_start",
    "clip_sample",
    "num_train_timesteps",
    "trained_betas",
    "variance_type",
}


def normalized_index(index):
    if index.get("_class_name") != ["modeling_ddpm", "CustomPipeline"]:
        return index
    if set(index) != {"_class_name", *COMPONENTS} or any(
        index.get(k) != v for k, v in COMPONENTS.items()
    ):
        raise ValueError("FLAT_DDPM_PIPELINE")
    return {"_class_name": "DDPMPipeline", MARKER: True, **COMPONENTS}


def supports(index):
    return index.get("_class_name") == "DDPMPipeline" and index.get(MARKER) is True


def validate_configs(unet, scheduler):
    validate_model_config(unet)
    validate_model_config(scheduler)
    if set(unet) != UNET_FIELDS or set(scheduler) != SCHEDULER_FIELDS:
        raise ValueError("FLAT_DDPM_CONFIGURATION")
    fixed = dict(
        _class_name="UNet2DModel",
        act_fn="silu",
        attention_head_dim=None,
        center_input_sample=False,
        downsample_padding=0,
        flip_sin_to_cos=False,
        freq_shift=1,
        in_channels=3,
        out_channels=3,
        mid_block_scale_factor=1,
        norm_eps=1e-6,
        time_embedding_type="positional",
    )
    if any(type(unet[k]) is not type(v) or unet[k] != v for k, v in fixed.items()):
        raise ValueError("FLAT_DDPM_UNET")
    if type(unet["center_input_sample"]) is not bool or type(unet["flip_sin_to_cos"]) is not bool:
        raise ValueError("FLAT_DDPM_UNET")
    for k, lo, hi in [
        ("sample_size", 8, 256),
        ("layers_per_block", 1, 3),
        ("norm_num_groups", 1, 32),
    ]:
        if type(unet[k]) is not int or not lo <= unet[k] <= hi:
            raise ValueError("FLAT_DDPM_GEOMETRY")
    channels = unet["block_out_channels"]
    if (
        not isinstance(channels, list)
        or not 1 <= len(channels) <= 4
        or any(
            type(c) is not int or not 8 <= c <= 512 or c % unet["norm_num_groups"] for c in channels
        )
        or unet["sample_size"] % max(8, 2 ** (len(channels) - 1))
    ):
        raise ValueError("FLAT_DDPM_GEOMETRY")
    for k, allowed in [
        ("down_block_types", {"DownBlock2D", "AttnDownBlock2D"}),
        ("up_block_types", {"UpBlock2D", "AttnUpBlock2D"}),
    ]:
        if (
            not isinstance(unet[k], list)
            or len(unet[k]) != len(channels)
            or any(not isinstance(v, str) or v not in allowed for v in unet[k])
        ):
            raise ValueError("FLAT_DDPM_BLOCKS")
    if (
        scheduler["_class_name"] != "DDPMScheduler"
        or scheduler["beta_schedule"] != "linear"
        or scheduler["clip_sample"] is not True
        or scheduler["trained_betas"] is not None
        or scheduler["variance_type"] != "fixed_large"
        or type(scheduler["num_train_timesteps"]) is not int
        or not 2 <= scheduler["num_train_timesteps"] <= 1000
        or any(type(scheduler[k]) not in {int, float} for k in ["beta_start", "beta_end"])
        or not 0 < scheduler["beta_start"] < scheduler["beta_end"] < 1
    ):
        raise ValueError("FLAT_DDPM_SCHEDULER")
    steps = scheduler["num_train_timesteps"]
    if (
        math.prod(
            1
            - scheduler["beta_start"]
            - (scheduler["beta_end"] - scheduler["beta_start"]) * i / (steps - 1)
            for i in range(steps)
        )
        < 1e-30
    ):
        raise ValueError("FLAT_DDPM_SCHEDULER_PRECISION")


def read_configs(root):
    unet = metadata_file(root / "config.json")
    scheduler = metadata_file(root / "scheduler_config.json")
    validate_configs(unet, scheduler)
    return unet, scheduler


def native_name(name):
    match = re.fullmatch(
        r"(.+\.attentions\.[0-9]+)\.(query|key|value|proj_attn)\.(weight|bias)", name
    )
    if match:
        prefix, layer, suffix = match.groups()
        return (
            prefix
            + "."
            + {"query": "to_q", "key": "to_k", "value": "to_v", "proj_attn": "to_out.0"}[layer]
            + "."
            + suffix
        )
    return name


def native_unet(config):
    from diffusers import UNet2DModel

    return UNet2DModel(**{k: v for k, v in config.items() if not k.startswith("_")}).eval()


def validate_weight_shapes(config, shapes):
    import torch

    with torch.device("meta"):
        model = native_unet(config)
    mapped = {native_name(k): tuple(v) for k, v in shapes.items()}
    if len(mapped) != len(shapes) or mapped != {
        k: tuple(v.shape) for k, v in model.state_dict().items()
    }:
        raise ValueError("FLAT_DDPM_WEIGHTS_INCOMPLETE")
    return model


def load_pipeline(root):
    import torch
    from diffusers import DDPMPipeline, DDPMScheduler
    from safetensors.torch import load_file

    unet, scheduler = read_configs(root)
    if {p.name for p in root.glob("*.safetensors")} != {
        "diffusion_pytorch_model.safetensors"
    } or any(p.parent != root for p in root.rglob("*.safetensors")):
        raise ValueError("FLAT_DDPM_WEIGHT_LAYOUT")
    model = validate_weight_shapes(unet, preflight_safetensors(root))
    state = {}
    for name, value in load_file(root / "diffusion_pytorch_model.safetensors").items():
        target = native_name(name)
        if (
            target in state
            or not value.is_floating_point()
            or not bool(torch.isfinite(value.float()).all())
        ):
            raise ValueError("FLAT_DDPM_WEIGHT_VALUES")
        state[target] = value.float()
    model.load_state_dict(state, strict=True, assign=True)

    def checked_output(_module, inputs, output):
        sample = getattr(output, "sample", None)
        if (
            not isinstance(sample, torch.Tensor)
            or sample.shape != inputs[0].shape
            or not bool(torch.isfinite(sample).all())
        ):
            raise ValueError("FLAT_DDPM_DENOISER_OUTPUT")

    model.register_forward_hook(checked_output)
    schedule = DDPMScheduler(**{k: v for k, v in scheduler.items() if not k.startswith("_")})
    native_step = schedule.step

    def checked_step(model_output, timestep, sample, **kwargs):
        output = native_step(model_output, timestep, sample, **kwargs)
        if output.prev_sample.shape != sample.shape or not bool(
            torch.isfinite(output.prev_sample).all()
        ):
            raise ValueError("FLAT_DDPM_SCHEDULER_OUTPUT")
        return output

    schedule.step = checked_step
    return DDPMPipeline(unet=model.eval(), scheduler=schedule)
