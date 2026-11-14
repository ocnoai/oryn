"""Owned MiniMax Music3 component graph and bounded native music contract."""

from __future__ import annotations

import math
import re

from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

FAMILY = "MiniMaxMusic3ModularPipeline"
INDEX = "modular_model_index.json"
CONTRACT = "text-to-music-v1"
CLASSES = {
    "language_model": ("transformers", "Qwen3ForCausalLM"),
    "rvq_depth_decoder": ("diffusers", "MiniMaxMusic3RVQDepthDecoder"),
    "condition_encoder": ("diffusers", "MiniMaxMusic3ConditionEncoder"),
    "transformer": ("diffusers", "MiniMaxMusic3Transformer1DModel"),
    "vocoder": ("diffusers", "MiniMaxMusic3Vocoder"),
    "scheduler": ("diffusers", "FlowMatchEulerDiscreteScheduler"),
    "tokenizer": ("transformers", "Qwen2Tokenizer"),
}
WEIGHT_COMPONENTS = tuple(k for k in CLASSES if k not in {"scheduler", "tokenizer"})
CONFIGS = {k: f"{k}/config.json" for k in WEIGHT_COMPONENTS} | {
    "scheduler": "scheduler/scheduler_config.json",
    "tokenizer": "tokenizer/tokenizer_config.json",
}
MAX_SECONDS = 30
SAMPLE_RATE = 44100
CONTROL_IDS = {
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|endoftext|>": 151643,
    "<|audio_end|>": 151670,
    "<|audio_cfg|>": 151654,
    "<|audio_start|>": 151669,
    "<|caption_start|>": 151671,
    "<|caption_end|>": 151672,
    "<|lyrics_start|>": 151673,
    "<|lyrics_end|>": 151674,
}


def supports(index):
    return index.get("_class_name") == FAMILY


def validate_index(index):
    validate_model_config(index)
    if (
        set(index) != {*CLASSES, "_class_name", "_blocks_class_name", "_diffusers_version"}
        or not supports(index)
        or index["_blocks_class_name"] != "MiniMaxMusic3Blocks"
    ):
        raise ValueError("MUSIC3_PIPELINE")
    for name, identity in CLASSES.items():
        value = index[name]
        if not isinstance(value, list) or len(value) != 3 or value[:2] != list(identity):
            raise ValueError("MUSIC3_COMPONENT_IDENTITY")
        hint = value[2]
        # These are recorded provenance hints only. Loading always uses the
        # selected local component; neither a repository nor a revision here
        # can cause a download or choose a Python class.
        if (
            not isinstance(hint, dict)
            or set(hint)
            != {"pretrained_model_name_or_path", "revision", "subfolder", "type_hint", "variant"}
            or not isinstance(hint["pretrained_model_name_or_path"], str)
            or not re.fullmatch(
                r"[A-Za-z0-9._-]{1,96}/[A-Za-z0-9._-]{1,96}", hint["pretrained_model_name_or_path"]
            )
            or hint["revision"] is not None
            and not re.fullmatch(r"[0-9a-f]{40}", str(hint["revision"]))
            or hint["subfolder"] != name
            or hint["type_hint"] != list(identity)
            or hint["variant"] is not None
        ):
            raise ValueError("MUSIC3_COMPONENT_BINDING")


def selected_files(files, index):
    validate_index(index)
    names = set(files)
    required = {INDEX, *CONFIGS.values(), "tokenizer/tokenizer.json"}
    if not required <= names:
        raise ValueError("MUSIC3_METADATA_MISSING")
    selected = set(required)
    for component in WEIGHT_COMPONENTS:
        prefix = component + "/"
        stem = "model" if component == "language_model" else "diffusion_pytorch_model"
        shards = {n for n in names if n.startswith(prefix) and n.endswith(".safetensors")}
        if shards == {prefix + stem + ".safetensors"}:
            if prefix + stem + ".safetensors.index.json" in names:
                raise ValueError("MUSIC3_AMBIGUOUS_INDEX")
        elif (
            not 1 <= len(shards) <= 32
            or shards
            != {
                f"{prefix}{stem}-{i:05d}-of-{len(shards):05d}.safetensors"
                for i in range(1, len(shards) + 1)
            }
            or prefix + stem + ".safetensors.index.json" not in names
        ):
            raise ValueError("MUSIC3_SHARDS")
        else:
            selected.add(prefix + stem + ".safetensors.index.json")
        selected.update(shards)
    return tuple(sorted(selected))


def _integer(config, name, lo, hi, multiple=1):
    value = config.get(name)
    if type(value) is not int or not lo <= value <= hi or value % multiple:
        raise ValueError("MUSIC3_GEOMETRY:" + name)
    return value


def validate_configs(configs):
    if set(configs) != set(CONFIGS):
        raise ValueError("MUSIC3_COMPONENTS")
    for value in configs.values():
        validate_model_config(value)
    lm, depth, cond, tr, voc = (configs[k] for k in WEIGHT_COMPONENTS)
    lm_keys = {
        "architectures",
        "attention_bias",
        "attention_dropout",
        "bos_token_id",
        "dtype",
        "eos_token_id",
        "head_dim",
        "hidden_act",
        "hidden_size",
        "initializer_range",
        "intermediate_size",
        "layer_types",
        "max_position_embeddings",
        "max_window_layers",
        "model_type",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "pad_token_id",
        "rms_norm_eps",
        "rope_parameters",
        "sliding_window",
        "tie_word_embeddings",
        "transformers_version",
        "use_cache",
        "use_sliding_window",
        "vocab_size",
    }
    if set(lm) != lm_keys:
        raise ValueError("MUSIC3_LANGUAGE_OPTIONS")
    h = _integer(lm, "hidden_size", 16, 8192, 8)
    layers = _integer(lm, "num_hidden_layers", 1, 80)
    heads = _integer(lm, "num_attention_heads", 1, 64)
    kv = _integer(lm, "num_key_value_heads", 1, 64)
    hd = _integer(lm, "head_dim", 4, 256, 2)
    _integer(lm, "intermediate_size", 16, 32768, 8)
    _integer(lm, "max_position_embeddings", 1024, 32768)
    fixed = dict(
        architectures=["Qwen3ForCausalLM"],
        model_type="qwen3",
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        dtype="bfloat16",
        hidden_act="silu",
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        sliding_window=None,
        use_sliding_window=False,
        use_cache=True,
        vocab_size=200000,
        layer_types=["full_attention"] * layers,
        rope_parameters={"rope_theta": 1000000, "rope_type": "default"},
    )
    if heads % kv or heads * hd != h or any(lm.get(k) != v for k, v in fixed.items()):
        raise ValueError("MUSIC3_LANGUAGE_GRAPH")
    fields = {
        "rvq_depth_decoder": {
            "hidden_size",
            "num_layers",
            "num_attention_heads",
            "intermediate_size",
            "audio_vocab_size",
            "num_codebooks",
            "max_position_embeddings",
        },
        "condition_encoder": {
            "condition_hidden_dim",
            "num_condition_layers",
            "out_dim",
            "input_sampling_rate",
            "input_hop_length",
            "output_sampling_rate",
            "output_hop_length",
        },
        "transformer": {
            "attention_head_dim",
            "condition_dim",
            "ff_inner_dim",
            "fourier_embedding_dim",
            "in_channels",
            "num_attention_heads",
            "num_layers",
            "rotary_dim",
        },
        "vocoder": {
            "decoder_hidden_dim",
            "decoder_input_dim",
            "latent_channels",
            "sampling_rate",
            "upsampling_ratios",
        },
    }
    for name, keys in fields.items():
        c = configs[name]
        if (
            set(c) != keys | {"_class_name", "_diffusers_version"}
            or c["_class_name"] != CLASSES[name][1]
        ):
            raise ValueError("MUSIC3_COMPONENT_OPTIONS")
    _integer(depth, "num_layers", 1, 16)
    dh = _integer(depth, "num_attention_heads", 1, 64)
    _integer(depth, "intermediate_size", 16, 32768, 8)
    if (
        depth["hidden_size"] != h
        or h % dh
        or depth["audio_vocab_size"] != 1024
        or depth["num_codebooks"] != 8
        or depth["max_position_embeddings"] != 16
    ):
        raise ValueError("MUSIC3_DEPTH_GRAPH")
    condition_dim = _integer(cond, "out_dim", 16, 4096, 8)
    if any(
        cond[k] != v
        for k, v in dict(
            condition_hidden_dim=h,
            num_condition_layers=8,
            input_sampling_rate=24000,
            input_hop_length=960,
            output_sampling_rate=SAMPLE_RATE,
            output_hop_length=512,
        ).items()
    ):
        raise ValueError("MUSIC3_CONDITION_GRAPH")
    channels = _integer(tr, "in_channels", 4, 256, 2)
    ah = _integer(tr, "attention_head_dim", 4, 128, 2)
    _integer(tr, "num_attention_heads", 1, 64)
    _integer(tr, "ff_inner_dim", 16, 32768, 8)
    _integer(tr, "num_layers", 1, 80)
    _integer(tr, "fourier_embedding_dim", 8, 512, 2)
    rotary = _integer(tr, "rotary_dim", 2, 128, 2)
    if tr["condition_dim"] != condition_dim or rotary > ah:
        raise ValueError("MUSIC3_FLOW_GRAPH")
    _integer(voc, "decoder_hidden_dim", 32, 4096, 16)
    _integer(voc, "decoder_input_dim", 8, 4096, 8)
    if (
        voc["latent_channels"] != channels
        or voc["sampling_rate"] != SAMPLE_RATE
        or voc["upsampling_ratios"] != [8, 8, 4, 2]
    ):
        raise ValueError("MUSIC3_VOCODER_GRAPH")
    scheduler = configs["scheduler"]
    expected_scheduler = dict(
        _class_name="FlowMatchEulerDiscreteScheduler",
        base_image_seq_len=256,
        base_shift=0.5,
        invert_sigmas=True,
        max_image_seq_len=4096,
        max_shift=1.15,
        num_train_timesteps=1,
        shift=1.0,
        shift_terminal=None,
        stochastic_sampling=False,
        time_shift_type="exponential",
        use_beta_sigmas=False,
        use_dynamic_shifting=False,
        use_exponential_sigmas=False,
        use_karras_sigmas=False,
    )
    if set(scheduler) != {*expected_scheduler, "_diffusers_version"} or any(
        scheduler[k] != v for k, v in expected_scheduler.items()
    ):
        raise ValueError("MUSIC3_SCHEDULER")
    tok = configs["tokenizer"]
    if (
        tok.get("tokenizer_class") != "Qwen2Tokenizer"
        or tok.get("add_prefix_space") is not False
        or tok.get("eos_token") != "<|im_end|>"
        or tok.get("pad_token") != "<|endoftext|>"
    ):
        raise ValueError("MUSIC3_TOKENIZER")
    return configs


def parameter_specs(configs):
    validate_configs(configs)
    lm, depth, cond, tr, voc = (configs[k] for k in WEIGHT_COMPONENTS)
    result = {name: {} for name in WEIGHT_COMPONENTS}
    s = result["language_model"]
    h, d, n, kv, f = (
        lm[k]
        for k in (
            "hidden_size",
            "head_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
        )
    )
    s.update(
        {
            "model.embed_tokens.weight": (200000, h),
            "lm_head.weight": (200000, h),
            "model.norm.weight": (h,),
        }
    )
    for i in range(lm["num_hidden_layers"]):
        p = f"model.layers.{i}."
        s.update(
            {
                p + k: v
                for k, v in {
                    "input_layernorm.weight": (h,),
                    "post_attention_layernorm.weight": (h,),
                    "mlp.down_proj.weight": (h, f),
                    "mlp.gate_proj.weight": (f, h),
                    "mlp.up_proj.weight": (f, h),
                    "self_attn.q_norm.weight": (d,),
                    "self_attn.k_norm.weight": (d,),
                    "self_attn.q_proj.weight": (n * d, h),
                    "self_attn.k_proj.weight": (kv * d, h),
                    "self_attn.v_proj.weight": (kv * d, h),
                    "self_attn.o_proj.weight": (h, n * d),
                }.items()
            }
        )
    s = result["rvq_depth_decoder"]
    s.update(
        {
            "audio_embeddings.weight": (7168, h),
            "projection.weight": (h, h),
            "pos_embedding.weight": (16, h),
            "norm.weight": (h,),
        }
    )
    s.update({f"audio_heads.{i}.weight": (1024, h) for i in range(7)})
    for i in range(depth["num_layers"]):
        p = f"layers.{i}."
        f = depth["intermediate_size"]
        s.update(
            {
                p + k + ".weight": v
                for k, v in {
                    "attn.to_q": (h, h),
                    "attn.to_k": (h, h),
                    "attn.to_v": (h, h),
                    "attn.to_out": (h, h),
                    "input_layernorm": (h,),
                    "post_attention_layernorm": (h,),
                    "gate_proj": (f, h),
                    "up_proj": (f, h),
                    "down_proj": (h, f),
                }.items()
            }
        )
    out = cond["out_dim"]
    result["condition_encoder"] = {
        "layer_scale": (1,),
        "layer_weight_logits": (8,),
        "proj.bias": (out,),
        "proj.weight": (out, h, 3),
    }
    s = result["transformer"]
    h = tr["attention_head_dim"] * tr["num_attention_heads"]
    c = tr["in_channels"]
    input_dim = 2 * c + out
    fourier = tr["fourier_embedding_dim"]
    f = tr["ff_inner_dim"]
    s.update(
        {
            "preprocess_conv.weight": (input_dim, input_dim, 1),
            "proj_in.weight": (h, input_dim),
            "time_embed.linear_1.bias": (h,),
            "time_embed.linear_1.weight": (h, fourier),
            "time_embed.linear_2.bias": (h,),
            "time_embed.linear_2.weight": (h, h),
            "time_proj.weight": (fourier // 2, 1),
            "postprocess_conv.weight": (c, c, 1),
            "proj_out.weight": (c, h),
        }
    )
    for i in range(tr["num_layers"]):
        p = f"transformer_blocks.{i}."
        s.update(
            {p + "attn." + k + ".weight": (h, h) for k in ("to_q", "to_k", "to_v", "to_out.0")}
        )
        s.update(
            {
                p + k: v
                for k, v in {
                    "ff_in.bias": (2 * f,),
                    "ff_in.weight": (2 * f, h),
                    "ff_out.bias": (h,),
                    "ff_out.weight": (h, f),
                    "norm1.bias": (h,),
                    "norm1.weight": (h,),
                    "norm2.bias": (h,),
                    "norm2.weight": (h,),
                }.items()
            }
        )
    s = result["vocoder"]
    h = voc["decoder_hidden_dim"]
    d = voc["decoder_input_dim"]

    def conv(name, inputs, outputs, width, transpose=False):
        leading = inputs if transpose else outputs
        s[name + ".weight_g"] = (leading, 1, 1)
        s[name + ".weight_v"] = (inputs, outputs, width) if transpose else (outputs, inputs, width)
        s[name + ".bias"] = (outputs,)

    s.update({"dec_in_proj.weight": (d, c // 2, 1), "dec_in_proj.bias": (d,)})
    conv("conv_in", d, h, 7)
    for i, stride in enumerate(voc["upsampling_ratios"]):
        p = f"blocks.{i}."
        inputs = h // 2**i
        outputs = inputs // 2
        s[p + "snake1.alpha"] = (1, inputs, 1)
        conv(p + "conv_t1", inputs, outputs, 2 * stride, True)
        for unit in range(1, 4):
            n = p + f"res_unit{unit}."
            for j, width in ((1, 7), (2, 1)):
                conv(n + f"conv{j}", outputs, outputs, width)
                s[n + f"snake{j}.alpha"] = (1, outputs, 1)
    s["snake_out.alpha"] = (1, outputs, 1)
    conv("conv_out", outputs, 1, 7)
    return result


def inspect_metadata(index, read_json, read_header, files):
    selected = selected_files(files, index)
    configs = validate_configs({k: read_json(n) for k, n in CONFIGS.items()})
    specs = parameter_specs(configs)
    for component in WEIGHT_COMPONENTS:
        actual = {}
        owners = {}
        for filename in selected:
            if not filename.startswith(component + "/") or not filename.endswith(".safetensors"):
                continue
            header = read_header(filename)
            if not isinstance(header, dict) or not isinstance(header.get("tensors"), dict):
                raise ValueError("MUSIC3_HEADER")
            for name, record in header["tensors"].items():
                if (
                    name in actual
                    or not isinstance(record, dict)
                    or record.get("dtype") not in {"F32", "F16", "BF16"}
                ):
                    raise ValueError("MUSIC3_TENSOR")
                actual[name] = tuple(record.get("shape", []))
                owners[name] = filename.split("/", 1)[1]
        if actual != specs[component]:
            raise ValueError("MUSIC3_GRAPH_INCOMPLETE")
        indexes = [
            n for n in selected if n.startswith(component + "/") and n.endswith(".index.json")
        ]
        if indexes and (len(indexes) != 1 or read_json(indexes[0]).get("weight_map") != owners):
            raise ValueError("MUSIC3_INDEX_BINDING")
    return configs


def validate_input(value):
    from oryn_engine.shared_tasks import _schemas

    if not _schemas().is_valid("taskInput." + CONTRACT, value):
        raise ValueError("MUSIC3_INPUT")
    result = dict(durationSeconds=10, lyrics="[Instrumental]", seed=0) | value
    if (
        not 1 <= result["durationSeconds"] <= MAX_SECONDS
        or not math.isfinite(result["durationSeconds"])
        or type(result["seed"]) is not int
        or not 0 <= result["seed"] <= 2**32 - 1
        or not result["prompt"].strip()
        or not result["lyrics"].strip()
        or any("<|" in result[k] or "|>" in result[k] for k in ("prompt", "lyrics"))
    ):
        raise ValueError("MUSIC3_INPUT")
    return result


def native_components(configs):
    """Build owned native classes; callers choose the allocation context."""
    import copy

    import diffusers
    from transformers import Qwen3Config, Qwen3ForCausalLM

    validate_configs(configs)
    return {
        name: Qwen3ForCausalLM(Qwen3Config.from_dict(copy.deepcopy(configs[name])))
        if name == "language_model"
        else getattr(diffusers, CLASSES[name][1]).from_config(copy.deepcopy(configs[name]))
        for name in WEIGHT_COMPONENTS
    }


def load_tokenizer(root):
    from tokenizers import Tokenizer
    from transformers import Qwen2Tokenizer

    path = root / "tokenizer/tokenizer.json"
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 32 * 1024**2:
        raise ValueError("MUSIC3_TOKENIZER_GRAPH")
    try:
        vocab = Tokenizer.from_file(str(path)).get_vocab()
    except Exception as exc:
        raise ValueError("MUSIC3_TOKENIZER_GRAPH") from exc
    ids = set(vocab.values())
    if (
        len(vocab) != 151675
        or len(ids) != len(vocab)
        or ids != set(range(151675))
        or any(vocab.get(name) != token for name, token in CONTROL_IDS.items())
    ):
        raise ValueError("MUSIC3_TOKENIZER_IDS")
    tok = Qwen2Tokenizer(
        tokenizer_file=str(path),
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        add_prefix_space=False,
        clean_up_tokenization_spaces=False,
    )
    tok.chat_template = None
    if tok.get_vocab() != vocab or any(
        tok.encode(name, add_special_tokens=False) != [token] for name, token in CONTROL_IDS.items()
    ):
        raise ValueError("MUSIC3_TOKENIZER_CONTROLS")
    return tok


def load_pipeline(root, backend):
    import torch
    from accelerate import init_empty_weights
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.modular_pipelines.minimax_music3.modular_blocks_minimax_music3 import (
        MiniMaxMusic3Blocks,
    )
    from diffusers.modular_pipelines.minimax_music3.modular_pipeline import (
        MiniMaxMusic3ModularPipeline,
    )
    from safetensors.torch import load_file

    if backend not in {"cpu", "mps", "cuda"}:
        raise ValueError("MUSIC3_BACKEND")
    index = metadata_file(root / INDEX)
    selected = selected_files(
        [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()], index
    )
    configs = validate_configs({k: metadata_file(root / n) for k, n in CONFIGS.items()})
    specs = parameter_specs(configs)
    for name in WEIGHT_COMPONENTS:
        if preflight_safetensors(root / name) != specs[name]:
            raise ValueError("MUSIC3_GRAPH_INCOMPLETE")
    with init_empty_weights():
        components = native_components(configs)
    for name, model in components.items():
        if {n: tuple(t.shape) for n, t in model.state_dict().items()} != specs[name]:
            raise ValueError("MUSIC3_NATIVE_GRAPH_CHANGED")
        weights = {}
        for filename in selected:
            if filename.startswith(name + "/") and filename.endswith(".safetensors"):
                for key, tensor in load_file(root / filename, device="cpu").items():
                    if (
                        key in weights
                        or tensor.dtype not in {torch.float16, torch.bfloat16, torch.float32}
                        or not torch.isfinite(tensor).all().item()
                    ):
                        raise ValueError("MUSIC3_WEIGHT_INVALID")
                    weights[key] = tensor
        model.load_state_dict(weights, strict=True, assign=True)
        model.to(device=backend, dtype=torch.float32).eval()
    # No model index or remote component hints are passed to native loading.
    pipe = MiniMaxMusic3ModularPipeline(blocks=MiniMaxMusic3Blocks())
    pipe.update_components(
        **components,
        tokenizer=load_tokenizer(root),
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(configs["scheduler"]),
    )
    pipe.to(backend)
    return pipe


def generate(root, output_path, backend, value, event):
    import numpy as np
    import soundfile as sf
    import torch

    value = validate_input(value)
    event("loading-music-components", 0.02)
    pipe = load_pipeline(root, backend)
    calls = 0

    def progress(module, inputs):
        nonlocal calls
        calls += 1
        event(
            "generating-music", min(0.88, 0.05 + 0.8 * calls / (value["durationSeconds"] * 25 + 62))
        )

    hooks = [
        pipe.language_model.model.register_forward_pre_hook(progress),
        pipe.transformer.register_forward_pre_hook(progress),
    ]
    try:
        with torch.inference_mode():
            audio = pipe(
                prompt=value["prompt"],
                lyrics=value["lyrics"],
                audio_duration=float(value["durationSeconds"]),
                num_inference_steps=30,
                generator=torch.Generator("cpu").manual_seed(value["seed"]),
                output_type="np",
                output="audios",
            )
    finally:
        for hook in hooks:
            hook.remove()
    audio = np.asarray(audio)
    maximum = int(value["durationSeconds"] * SAMPLE_RATE)
    if (
        audio.ndim != 3
        or audio.shape[:2] != (1, 2)
        or not 1 <= audio.shape[-1] <= maximum + 1024
        or not np.isfinite(audio).all()
        or np.max(np.abs(audio)) > 1.00001
    ):
        raise ValueError("MUSIC3_AUDIO_INVALID")
    audio = audio[0, :, :maximum].T
    event("encoding-audio", 0.92)
    sf.write(output_path, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return dict(
        relativeOutput=output_path.name,
        mediaType="audio/wav",
        sampleRate=SAMPLE_RATE,
        channels=2,
        samples=len(audio),
        backend=backend,
    )


def validate_audio_artifact(path, result, seconds):
    import wave

    if (
        result.get("relativeOutput") != path.name
        or result.get("mediaType") != "audio/wav"
        or result.get("sampleRate") != SAMPLE_RATE
        or result.get("channels") != 2
        or type(result.get("samples")) is not int
        or not 1 <= result["samples"] <= int(seconds * SAMPLE_RATE)
        or path.is_symlink()
        or not path.is_file()
        or not 44 <= path.stat().st_size <= MAX_SECONDS * SAMPLE_RATE * 4 + 1024**2
    ):
        raise ValueError("MUSIC3_AUDIO_ARTIFACT")
    try:
        with wave.open(str(path), "rb") as f:
            if (
                f.getnchannels(),
                f.getsampwidth(),
                f.getframerate(),
                f.getnframes(),
                f.getcomptype(),
            ) != (2, 2, SAMPLE_RATE, result["samples"], "NONE"):
                raise ValueError("MUSIC3_AUDIO_ARTIFACT")
            if len(f.readframes(result["samples"])) != result["samples"] * 4:
                raise ValueError("MUSIC3_AUDIO_TRUNCATED")
    except (wave.Error, EOFError) as exc:
        raise ValueError("MUSIC3_AUDIO_ARTIFACT") from exc
