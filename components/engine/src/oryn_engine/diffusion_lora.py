"""Strict low-rank linear adapters for the bundled Qwen image transformers.

Adapters stay unmerged: keep the original base precision and evaluate the small
update in float32. ComfyUI's fused MLP uses [gate; up] rows; splitting B preserves
that exact update in the native Diffusers gate_layer and proj modules.
"""

from __future__ import annotations

import json
import math
import re

FAMILIES = {"QwenImage21Transformer2DModel", "QwenImageTransformer2DModel"}


def target_shapes(config):
    if not isinstance(config, dict) or config.get("_class_name") not in FAMILIES:
        raise ValueError("LoRA transformer family is unsupported")
    from oryn_engine.model_metadata import validate_model_config

    validate_model_config(config)
    values = [config.get(k) for k in ("num_layers", "num_attention_heads", "attention_head_dim")]
    if any(type(v) is not int or not 1 <= v <= 256 for v in values):
        raise ValueError("LoRA transformer dimensions are invalid")
    layers, heads, head_dim = values
    hidden = heads * head_dim
    if hidden > 16384:
        raise ValueError("LoRA hidden dimension is oversized")
    qwen21 = config["_class_name"] == "QwenImage21Transformer2DModel"
    ratio = config.get("mlp_ratio", 3) if qwen21 else 4
    if type(ratio) is not int or not 1 <= ratio <= 8:
        raise ValueError("LoRA feed-forward dimensions are invalid")
    middle = hidden * ratio
    shapes = {}
    for i in range(layers):
        prefix = f"transformer_blocks.{i}."
        for name in ("to_q", "to_k", "to_v", "to_out.0"):
            shapes[prefix + "attn." + name] = (hidden, hidden)
        if qwen21:
            for name in ("proj", "gate_layer"):
                shapes[prefix + "img_mlp." + name] = (middle, hidden)
            shapes[prefix + "img_mlp.out"] = (hidden, middle)
        else:
            for name in ("add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"):
                shapes[prefix + "attn." + name] = (hidden, hidden)
            for name in ("img_mlp", "txt_mlp"):
                shapes[prefix + name + ".net.0.proj"] = (middle, hidden)
                shapes[prefix + name + ".net.2"] = (hidden, middle)
    if qwen21:
        shapes.update(
            {
                "time_text_embed.timestep_embedder.linear_1": (hidden, 256),
                "time_text_embed.timestep_embedder.linear_2": (hidden, hidden),
                "modulation.1": (hidden * 4, hidden),
            }
        )
    return shapes


def _metadata_alpha(metadata):
    raw = metadata.get("lora_adapter_metadata")
    if raw is None:
        return None
    if not isinstance(raw, str) or len(raw) > 16384:
        raise ValueError("LoRA scaling metadata is invalid")
    value = json.loads(raw)
    if not isinstance(value, dict) or any(not k.startswith("transformer.") for k in value):
        raise ValueError("LoRA component metadata is invalid")
    for key, expected in {
        "alpha_pattern": {},
        "rank_pattern": {},
        "use_rslora": False,
        "use_dora": False,
        "fan_in_fan_out": False,
        "lora_bias": False,
        "lora_dropout": 0.0,
        "bias": "none",
        "peft_type": "LORA",
    }.items():
        if value.get("transformer." + key, expected) != expected:
            raise ValueError("LoRA scaling variant is unsupported")
    rank, alpha = value.get("transformer.r"), value.get("transformer.lora_alpha")
    if (
        type(rank) is not int
        or not 1 <= rank <= 1024
        or type(alpha) not in {int, float}
        or not math.isfinite(alpha)
        or not 0 < alpha <= 65536
    ):
        raise ValueError("LoRA rank/alpha metadata is invalid")
    return rank, float(alpha)


def validate_structure(structure, config):
    """Validate every key/pair and return native target plans without tensor reads."""
    shapes = target_shapes(config)
    tensors = structure.get("tensors") if isinstance(structure, dict) else None
    metadata = structure.get("metadata", {}) if isinstance(structure, dict) else None
    if (
        not isinstance(tensors, dict)
        or not 2 <= len(tensors) <= 20000
        or not isinstance(metadata, dict)
    ):
        raise ValueError("LoRA tensor directory is invalid")
    alpha_metadata = _metadata_alpha(metadata)
    pairs, alphas = {}, {}
    for original, tensor in tensors.items():
        if (
            not isinstance(original, str)
            or not isinstance(tensor, dict)
            or tensor.get("dtype") not in {"F32", "F16", "BF16"}
        ):
            raise ValueError("LoRA tensors require floating numeric arrays")
        name = original
        for prefix in ("transformer.", "diffusion_model."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        shape = tensor.get("shape")
        if name.endswith(".alpha"):
            target = name[:-6]
            if shape != [] or target in alphas:
                raise ValueError("LoRA alpha must be one scalar per target")
            alphas[target] = original
            continue
        match = re.fullmatch(r"(.+)\.(lora_A|lora_B|lora_down|lora_up)(?:\.default)?\.weight", name)
        if (
            match is None
            or not isinstance(shape, list)
            or len(shape) != 2
            or any(type(n) is not int or not 1 <= n <= 131072 for n in shape)
        ):
            raise ValueError("LoRA tensor name or shape is unsupported")
        target, part = match.groups()
        part = "A" if part in {"lora_A", "lora_down"} else "B"
        entry = pairs.setdefault(target, {})
        if part in entry:
            raise ValueError("LoRA target has duplicate aliases")
        entry[part] = original
    if set(alphas) - set(pairs) or (alpha_metadata is not None and alphas):
        raise ValueError("LoRA alpha binding is ambiguous")
    plans, occupied = [], set()
    for target, parts in sorted(pairs.items()):
        if set(parts) != {"A", "B"}:
            raise ValueError("LoRA target is missing A or B")
        rank, inputs = tensors[parts["A"]]["shape"]
        outputs, b_rank = tensors[parts["B"]]["shape"]
        if (
            rank != b_rank
            or not 1 <= rank <= 1024
            or (alpha_metadata and alpha_metadata[0] != rank)
        ):
            raise ValueError("LoRA ranks disagree")
        destinations = [(target, None)]
        if (
            target.endswith(".img_mlp.gate_up")
            and config["_class_name"] == "QwenImage21Transformer2DModel"
        ):
            prefix = target.removesuffix("gate_up")
            if outputs % 2:
                raise ValueError("Fused LoRA gate/up row count is invalid")
            destinations = [
                (prefix + "gate_layer", (0, outputs // 2)),
                (prefix + "proj", (outputs // 2, outputs)),
            ]
        for destination, rows in destinations:
            count = outputs if rows is None else rows[1] - rows[0]
            if shapes.get(destination) != (count, inputs) or destination in occupied:
                raise ValueError("LoRA target is absent, duplicated, or has incompatible geometry")
            occupied.add(destination)
            plans.append(
                {
                    "target": destination,
                    "A": parts["A"],
                    "B": parts["B"],
                    "rows": rows,
                    "rank": rank,
                    "alphaTensor": alphas.get(target),
                    "alpha": alpha_metadata[1] if alpha_metadata else float(rank),
                }
            )
    if not plans:
        raise ValueError("LoRA has no complete targets")
    return plans


def attach(transformer, path, *, scale=1.0):
    """Validate the complete adapter before replacing any native linear module."""
    import torch
    from safetensors import safe_open

    from oryn_engine.model_metadata import preflight_safetensors

    if type(scale) not in {int, float} or not math.isfinite(scale) or not 0 < scale <= 4:
        raise ValueError("LoRA scale is invalid")
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024**3:
        raise ValueError("LoRA file is unavailable or oversized")
    preflight_safetensors(path.parent)
    with safe_open(path, framework="pt", device="cpu") as source:
        names = source.keys()
        tensors = {name: source.get_tensor(name) for name in names}
        structure = {
            "metadata": source.metadata() or {},
            "tensors": {
                name: {
                    "shape": list(t.shape),
                    "dtype": {
                        torch.float32: "F32",
                        torch.float16: "F16",
                        torch.bfloat16: "BF16",
                    }.get(t.dtype),
                }
                for name, t in tensors.items()
            },
        }
    plans = validate_structure(structure, json.loads(transformer.to_json_string()))
    if any(not torch.isfinite(t).all().item() for t in tensors.values()):
        raise ValueError("LoRA contains non-finite arrays")
    prepared = []
    for plan in plans:
        base = transformer.get_submodule(plan["target"])
        a, b = tensors[plan["A"]], tensors[plan["B"]]
        if plan["rows"] is not None:
            start, end = plan["rows"]
            b = b[start:end]
        if type(base) is not torch.nn.Linear or tuple(base.weight.shape) != (
            b.shape[0],
            a.shape[1],
        ):
            raise ValueError("LoRA requires the exact native linear target")
        alpha = float(tensors[plan["alphaTensor"]]) if plan["alphaTensor"] else plan["alpha"]
        if not math.isfinite(alpha) or not 0 < alpha <= 65536:
            raise ValueError("LoRA alpha is invalid")
        prepared.append((plan["target"], base, a, b, scale * alpha / plan["rank"]))

    class AdaptedLinear(torch.nn.Module):
        def __init__(self, base, a, b, multiplier):
            super().__init__()
            self.base = base
            self.lora_a = torch.nn.Parameter(
                a.to(device=base.weight.device, dtype=torch.float32), requires_grad=False
            )
            self.lora_b = torch.nn.Parameter(
                b.to(device=base.weight.device, dtype=torch.float32), requires_grad=False
            )
            self.multiplier = multiplier
            self.enabled = True

        @property
        def weight(self):
            return self.base.weight

        @property
        def bias(self):
            return self.base.bias

        def forward(self, x):
            result = self.base(x)
            if not self.enabled:
                return result
            update = torch.nn.functional.linear(
                torch.nn.functional.linear(x.float(), self.lora_a), self.lora_b
            )
            output = (result.float() + update * self.multiplier).to(result.dtype)
            if not torch.isfinite(output).all().item():
                raise ValueError("LoRA linear output is non-finite")
            return output

    modules = []
    for target, base, a, b, multiplier in prepared:
        module = AdaptedLinear(base, a, b, multiplier)
        parent, name = target.rsplit(".", 1)
        setattr(transformer.get_submodule(parent), name, module)
        modules.append(module)
    return tuple(modules)


def configure_pipeline(pipe, root, composition):
    from oryn_engine.model_composition import adapter_name, validate_binding

    value = validate_binding(composition)
    if type(pipe).__name__ != value["family"]:
        raise ValueError("LoRA pipeline family differs from its pinned base")
    verify_base_components(pipe, root)
    pipe._oryn_lora_modules = attach(
        pipe.transformer, root / adapter_name(value), scale=value["scale"]
    )
    pipe._oryn_lora_sampling = value.get("sampling", {})
    pipe._oryn_composition = value
    if "sigmas" in pipe._oryn_lora_sampling:
        pipe._oryn_sample_sigmas = pipe._oryn_lora_sampling["sigmas"]


def validate_sampling(pipe, value):
    for key, expected in getattr(pipe, "_oryn_lora_sampling", {}).items():
        if key != "sigmas" and value.get(key) != expected:
            raise ValueError(f"This LoRA requires {key}={expected!r}")


def verify_base_components(pipe, root):
    """A native loader may warn and initialize missing arrays; never admit that."""
    import torch

    from oryn_engine.model_metadata import preflight_safetensors

    for name in ("transformer", "vae", "text_encoder"):
        module = getattr(pipe, name)
        shapes = preflight_safetensors(root / name)
        if name == "text_encoder" and type(module).__name__ == "Qwen2_5_VLForConditionalGeneration":
            # Transformers 5 retains this declared v4 checkpoint serialization.
            # Only these exact native namespaces are renamed; no arrays are dropped.
            renamed = {}
            for key, shape in shapes.items():
                target = (
                    "model." + key
                    if key.startswith("visual.")
                    else re.sub(
                        r"^model\.(embed_tokens|layers|norm)\.", r"model.language_model.\1.", key
                    )
                )
                if target in renamed:
                    raise ValueError("Pinned base text encoder has duplicate aliases")
                renamed[target] = shape
            shapes = renamed
        state = module.state_dict()
        expected = {key: tuple(tensor.shape) for key, tensor in state.items()}
        missing = set(expected) - set(shapes)
        # Only declared, native parameter aliases may reuse an original array.
        if missing and bool(getattr(module.config, "tie_word_embeddings", False)):
            parameters = dict(module.named_parameters(remove_duplicate=False))
            for key in list(missing):
                if key in parameters and any(
                    parameters.get(other) is parameters[key] for other in shapes
                ):
                    shapes[key] = expected[key]
        if shapes != expected:
            raise ValueError(f"Pinned base {name} has missing, extra or incompatible arrays")
        for tensor in state.values():
            if tensor.device.type == "meta":
                raise ValueError("Pinned base contains uninitialized arrays")
            if tensor.is_floating_point():
                flat = tensor.reshape(-1)
                if any(not torch.isfinite(chunk).all().item() for chunk in flat.split(1_048_576)):
                    raise ValueError("Pinned base contains non-finite arrays")
