"""Complete native unconditional UNet2D graphs over the closed numeric archive reader.

Early Diffusers DDPM and NCSN++ exports publish only ``diffusion_pytorch_model.bin``.
Inspection derives every learned tensor from the reviewed configuration without
importing an execution framework; loading reads plain numeric buffers through
numeric_torch_archive and never calls torch.load.
"""

from __future__ import annotations

import re

from oryn_engine.model_metadata import metadata_file, validate_model_config

WEIGHTS = "diffusion_pytorch_model.bin"
PLAIN_DOWN = {"DownBlock2D", "AttnDownBlock2D"}
PLAIN_UP = {"UpBlock2D", "AttnUpBlock2D"}
SKIP_DOWN = {"SkipDownBlock2D", "AttnSkipDownBlock2D"}
SKIP_UP = {"SkipUpBlock2D", "AttnSkipUpBlock2D"}
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
SCHEDULERS = {
    "DDPMPipeline": (
        "DDPMScheduler",
        {
            "_class_name",
            "_diffusers_version",
            "beta_end",
            "beta_schedule",
            "beta_start",
            "clip_sample",
            "num_train_timesteps",
            "trained_betas",
            "variance_type",
        },
    ),
    "ScoreSdeVePipeline": (
        "ScoreSdeVeScheduler",
        {
            "_class_name",
            "_diffusers_version",
            "correct_steps",
            "num_train_timesteps",
            "sampling_eps",
            "sigma_max",
            "sigma_min",
            "snr",
        },
    ),
}
LEGACY_ATTENTION = {"query": "to_q", "key": "to_k", "value": "to_v", "proj_attn": "to_out.0"}
FOURIER_ALIAS = "time_proj.W"


def selected(index, files):
    names = set(files)
    return (
        isinstance(index, dict)
        and index.get("_class_name") in SCHEDULERS
        and WEIGHTS in names
        and {"config.json", "scheduler_config.json", "model_index.json"} <= names
        and not any(
            n.endswith((".safetensors", ".bin", ".pt", ".pth", ".ckpt")) and n != WEIGHTS
            for n in names
        )
    )


def _number(value):
    return type(value) in {int, float}


def validate_configs(index, unet, scheduler):
    from oryn_engine.diffusion_tasks import validate_pipeline_index

    validate_model_config(unet)
    validate_model_config(scheduler)
    family = index.get("_class_name") if isinstance(index, dict) else None
    if family not in SCHEDULERS or set(unet) != UNET_FIELDS:
        raise ValueError("NUMERIC_UNET2D_CONFIGURATION")
    # Exact bundled component identities; repository classes never select code.
    validate_pipeline_index(index)
    scheduler_class, scheduler_fields = SCHEDULERS[family]
    if scheduler.get("_class_name") != scheduler_class or set(scheduler) != scheduler_fields:
        raise ValueError("NUMERIC_UNET2D_SCHEDULER")
    skip = family == "ScoreSdeVePipeline"
    fixed = dict(
        _class_name="UNet2DModel",
        act_fn="silu",
        attention_head_dim=None,
        in_channels=3,
        out_channels=3,
        norm_eps=1e-6,
        time_embedding_type="fourier" if skip else "positional",
    )
    if any(type(unet[k]) is not type(v) or unet[k] != v for k, v in fixed.items()):
        raise ValueError("NUMERIC_UNET2D_CONFIGURATION")
    for key in ("center_input_sample", "flip_sin_to_cos"):
        if type(unet[key]) is not bool:
            raise ValueError("NUMERIC_UNET2D_CONFIGURATION")
    if (
        type(unet["freq_shift"]) is not int
        or unet["freq_shift"] not in {0, 1}
        or type(unet["downsample_padding"]) is not int
        or unet["downsample_padding"] not in {0, 1}
        or not _number(unet["mid_block_scale_factor"])
        or not 0.5 <= unet["mid_block_scale_factor"] <= 2
        or type(unet["layers_per_block"]) is not int
        or not 1 <= unet["layers_per_block"] <= 4
        or type(unet["sample_size"]) is not int
        or not 8 <= unet["sample_size"] <= 512
    ):
        raise ValueError("NUMERIC_UNET2D_GEOMETRY")
    groups = unet["norm_num_groups"]
    channels = unet["block_out_channels"]
    if (
        not isinstance(channels, list)
        or not 1 <= len(channels) <= 8
        or any(type(c) is not int or not 8 <= c <= 1024 for c in channels)
        or unet["sample_size"] % 2 ** (len(channels) - 1)
    ):
        raise ValueError("NUMERIC_UNET2D_GEOMETRY")
    if skip:
        # NCSN++ derives every group count from its channels; channels must
        # also satisfy the fixed 32-group attention normalization.
        if groups is not None or any(c % 32 or c % min(c // 4, 32) for c in channels):
            raise ValueError("NUMERIC_UNET2D_GEOMETRY")
    elif type(groups) is not int or not 1 <= groups <= 32 or any(c % groups for c in channels):
        raise ValueError("NUMERIC_UNET2D_GEOMETRY")
    down, up = (SKIP_DOWN, SKIP_UP) if skip else (PLAIN_DOWN, PLAIN_UP)
    for key, allowed in (("down_block_types", down), ("up_block_types", up)):
        if (
            not isinstance(unet[key], list)
            or len(unet[key]) != len(channels)
            or any(not isinstance(v, str) or v not in allowed for v in unet[key])
        ):
            raise ValueError("NUMERIC_UNET2D_BLOCKS")
    if skip:
        if (
            type(scheduler["num_train_timesteps"]) is not int
            or not 2 <= scheduler["num_train_timesteps"] <= 4000
            or type(scheduler["correct_steps"]) is not int
            or not 0 <= scheduler["correct_steps"] <= 4
            or not all(
                _number(scheduler[k]) for k in ("sampling_eps", "sigma_max", "sigma_min", "snr")
            )
            or not 0 < scheduler["sampling_eps"] < 1
            or not 0 < scheduler["sigma_min"] < scheduler["sigma_max"] <= 10000
            or not 0 < scheduler["snr"] <= 1
        ):
            raise ValueError("NUMERIC_UNET2D_SCHEDULER")
        return
    if (
        scheduler["beta_schedule"] != "linear"
        or type(scheduler["clip_sample"]) is not bool
        or scheduler["trained_betas"] is not None
        or scheduler["variance_type"] not in {"fixed_small", "fixed_large"}
        or type(scheduler["num_train_timesteps"]) is not int
        or not 2 <= scheduler["num_train_timesteps"] <= 1000
        or not all(_number(scheduler[k]) for k in ("beta_start", "beta_end"))
        or not 0 < scheduler["beta_start"] < scheduler["beta_end"] < 1
    ):
        raise ValueError("NUMERIC_UNET2D_SCHEDULER")


def expected_shapes(unet):
    """Every learned tensor of the native UNet2DModel for a validated configuration."""
    out = {}
    channels = unet["block_out_channels"]
    first, layers = channels[0], unet["layers_per_block"]
    skip = unet["time_embedding_type"] == "fourier"
    temb = first * 4

    def bias(name, rows):
        out[name + ".bias"] = [rows]

    def conv(name, rows, columns, kernel):
        out[name + ".weight"] = [rows, columns, kernel, kernel]
        bias(name, rows)

    def linear(name, rows, columns):
        out[name + ".weight"] = [rows, columns]
        bias(name, rows)

    def norm(name, width):
        out[name + ".weight"] = [width]
        bias(name, width)

    def resnet(name, source, target, shortcut=None):
        norm(name + ".norm1", source)
        conv(name + ".conv1", target, source, 3)
        linear(name + ".time_emb_proj", target, temb)
        norm(name + ".norm2", target)
        conv(name + ".conv2", target, target, 3)
        if source != target if shortcut is None else shortcut:
            conv(name + ".conv_shortcut", target, source, 1)

    def attention(name, width):
        norm(name + ".group_norm", width)
        for part in ("to_q", "to_k", "to_v", "to_out.0"):
            linear(name + "." + part, width, width)

    conv("conv_in", first, unet["in_channels"], 3)
    if skip:
        out["time_proj.weight"] = [first]
    linear("time_embedding.linear_1", temb, 2 * first if skip else first)
    linear("time_embedding.linear_2", temb, temb)
    output = first
    for i, kind in enumerate(unet["down_block_types"]):
        source, output = output, channels[i]
        final = i == len(channels) - 1
        prefix = f"down_blocks.{i}"
        for j in range(layers):
            resnet(f"{prefix}.resnets.{j}", source if j == 0 else output, output)
            if kind.startswith("Attn"):
                attention(f"{prefix}.attentions.{j}", output)
        if not final and kind in PLAIN_DOWN:
            conv(prefix + ".downsamplers.0.conv", output, output, 3)
        if not final and kind in SKIP_DOWN:
            resnet(prefix + ".resnet_down", output, output, shortcut=True)
            conv(prefix + ".skip_conv", output, 3, 1)
    last = channels[-1]
    resnet("mid_block.resnets.0", last, last)
    attention("mid_block.attentions.0", last)
    resnet("mid_block.resnets.1", last, last)
    reverse = list(reversed(channels))
    output = reverse[0]
    for i, kind in enumerate(unet["up_block_types"]):
        previous, output = output, reverse[i]
        source = reverse[min(i + 1, len(channels) - 1)]
        final = i == len(channels) - 1
        prefix = f"up_blocks.{i}"
        count = layers + 1
        for j in range(count):
            residual = source if j == count - 1 else output
            resnet(f"{prefix}.resnets.{j}", (previous if j == 0 else output) + residual, output)
            if kind in PLAIN_UP and kind.startswith("Attn"):
                attention(f"{prefix}.attentions.{j}", output)
        if kind == "AttnSkipUpBlock2D":
            attention(prefix + ".attentions.0", output)
        if not final and kind in PLAIN_UP:
            conv(prefix + ".upsamplers.0.conv", output, output, 3)
        if not final and kind in SKIP_UP:
            resnet(prefix + ".resnet_up", output, output, shortcut=True)
            conv(prefix + ".skip_conv", 3, output, 3)
            norm(prefix + ".skip_norm", output)
    norm("conv_norm_out", first)
    conv("conv_out", unet["out_channels"], first, 3)
    return out


def native_name(name):
    match = re.fullmatch(
        r"(.+\.attentions\.[0-9]+)\.(query|key|value|proj_attn)\.(weight|bias)", name
    )
    if not match:
        return name
    prefix, layer, suffix = match.groups()
    return f"{prefix}.{LEGACY_ATTENTION[layer]}.{suffix}"


def validate_archive(unet, metadata):
    """Map exact archive headers onto the complete native graph."""
    tensors = metadata.get("tensors") if isinstance(metadata, dict) else None
    if (
        not isinstance(tensors, dict)
        or metadata.get("format") != "torch-numeric-zip-v1"
        or metadata.get("byteOrder") != "little"
    ):
        raise ValueError("NUMERIC_ARCHIVE_METADATA")
    shapes = expected_shapes(unet)
    archive = {}
    native = {}
    alias = unet["time_embedding_type"] == "fourier" and FOURIER_ALIAS in tensors
    for name, value in tensors.items():
        if name == FOURIER_ALIAS and alias:
            continue
        target = native_name(name)
        if target in native:
            raise ValueError("NUMERIC_UNET2D_DUPLICATE_TENSOR")
        native[target] = name
        archive[name] = value
    if alias:
        archive[FOURIER_ALIAS] = tensors[FOURIER_ALIAS]
    if set(native) != set(shapes):
        raise ValueError("NUMERIC_ARCHIVE_GRAPH_INCOMPLETE")
    expected = {source: shapes[target] for target, source in native.items()}
    if alias:
        expected[FOURIER_ALIAS] = shapes["time_proj.weight"]
    for name, value in archive.items():
        if (
            not isinstance(value, dict)
            or value.get("shape") != expected[name]
            or value.get("dtype") not in {"F32", "F16", "BF16"}
        ):
            raise ValueError("NUMERIC_ARCHIVE_GRAPH_INCOMPLETE")
    return expected, native


def read_configs(root):
    index = metadata_file(root / "model_index.json")
    unet = metadata_file(root / "config.json")
    scheduler = metadata_file(root / "scheduler_config.json")
    validate_configs(index, unet, scheduler)
    return index, unet, scheduler


def load_pipeline(root):
    import diffusers
    import torch

    from oryn_engine.numeric_torch_archive import inspect_archive, read_weights

    index, unet, scheduler = read_configs(root)
    path = root / WEIGHTS
    if (
        path.is_symlink()
        or not path.is_file()
        or not selected(index, [p.name for p in root.iterdir()])
    ):
        raise ValueError("NUMERIC_UNET2D_WEIGHT_LAYOUT")
    with path.open("rb") as stream:

        def read_range(start, length):
            stream.seek(start)
            return stream.read(length)

        metadata = inspect_archive(read_range, path.stat().st_size)
    expected, native = validate_archive(unet, metadata)
    model = diffusers.UNet2DModel(**{k: v for k, v in unet.items() if not k.startswith("_")})
    if {k: list(v.shape) for k, v in model.state_dict().items()} != expected_shapes(unet):
        raise ValueError("NUMERIC_ARCHIVE_NATIVE_GRAPH_CHANGED")
    values = read_weights(path, expected_shapes=expected, maximum_bytes=path.stat().st_size)
    if FOURIER_ALIAS in values and not torch.equal(
        values.pop(FOURIER_ALIAS), values[native["time_proj.weight"]]
    ):
        raise ValueError("NUMERIC_UNET2D_FOURIER_ALIAS")
    state = {}
    for target, source in native.items():
        value = values[source].float().contiguous()
        if not bool(torch.isfinite(value).all()):
            raise ValueError("NUMERIC_ARCHIVE_NONFINITE")
        state[target] = value
    model.load_state_dict(state, strict=True, assign=True)

    def checked_output(_module, inputs, output):
        sample = getattr(output, "sample", None)
        if (
            not isinstance(sample, torch.Tensor)
            or sample.shape != inputs[0].shape
            or not bool(torch.isfinite(sample).all())
        ):
            raise ValueError("NUMERIC_UNET2D_DENOISER_OUTPUT")

    model.register_forward_hook(checked_output)
    scheduler_class, _ = SCHEDULERS[index["_class_name"]]
    schedule = getattr(diffusers, scheduler_class)(
        **{k: v for k, v in scheduler.items() if not k.startswith("_")}
    )
    pipeline = getattr(diffusers, index["_class_name"])
    return pipeline(unet=model.eval(), scheduler=schedule)
