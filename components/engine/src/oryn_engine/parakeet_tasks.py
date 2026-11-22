"""Closed native Parakeet TDT graphs with lossless base-three weight expansion."""

from __future__ import annotations

import copy
import math

from oryn_engine import rnnt_tasks
from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

FAMILY = "parakeet_tdt"
ARCHITECTURE = "ParakeetForTDT"
METADATA_NAMES = ("ternary.json",)
PACKING = {
    "base": 3,
    "code_offset": 1,
    "elements_per_byte": 5,
    "row_bytes": "ceil(in/5)",
    "digit_order": (
        "element i of a row is base-3 digit i%5 of byte i//5, "
        "least significant digit first; pad digits 0"
    ),
    "row_major": True,
}


def supports(config, task):
    return task == "automatic-speech-recognition" and config.get("model_type") == FAMILY


def modules(config):
    encoder = config["encoder_config"]
    hidden, intermediate = encoder["hidden_size"], encoder["intermediate_size"]
    dimensions = {
        "feed_forward1.linear1": (intermediate, hidden, False),
        "feed_forward1.linear2": (hidden, intermediate, False),
        **{
            f"self_attn.{p}_proj": (hidden, hidden, False)
            for p in ("q", "k", "v", "o", "relative_k")
        },
        "conv.pointwise_conv1": (2 * hidden, hidden, True),
        "conv.pointwise_conv2": (hidden, hidden, True),
        "feed_forward2.linear1": (intermediate, hidden, False),
        "feed_forward2.linear2": (hidden, intermediate, False),
    }
    return {
        f"encoder.layers.{i}.{k}": v
        for i in range(encoder["num_hidden_layers"])
        for k, v in dimensions.items()
    }


def validate_config(config):
    validate_model_config(config)
    if (
        config.get("model_type") != FAMILY
        or config.get("architectures", [ARCHITECTURE]) != [ARCHITECTURE]
        or config.get("durations") != [0, 1, 2, 3, 4]
        or config.get("ternary_group_size") != 128
        or not isinstance(config.get("encoder_config"), dict)
        or config["encoder_config"].get("model_type") != "parakeet_encoder"
    ):
        raise ValueError("PARAKEET_CONFIGURATION")
    # The native transducers share bounded Conformer/LSTM geometry. This copy
    # validates those dimensions only; the original Parakeet graph is unchanged.
    geometry = copy.deepcopy(config)
    geometry.update(
        model_type=rnnt_tasks.FAMILY,
        architectures=[rnnt_tasks.ARCHITECTURE],
        durations=[],
        num_prompts=128,
        default_prompt_id=101,
        prompt_intermediate_size=64,
    )
    geometry["encoder_config"].update(
        model_type="nemotron_asr_streaming_encoder",
        sliding_window=57,
        supported_num_lookahead_tokens=[3, 0, 6, 13],
        default_num_lookahead_tokens=3,
    )
    rnnt_tasks.validate_config(geometry)
    names = config.get("ternary_modules")
    if (
        not isinstance(names, list)
        or any(not isinstance(n, str) for n in names)
        or len(names) != len(set(names))
        or set(names) != set(modules(config))
        or config["encoder_config"]["hidden_size"] % 128
        or config["encoder_config"]["intermediate_size"] % 128
        or any(
            k in config["encoder_config"]
            for k in (
                "sliding_window",
                "default_num_lookahead_tokens",
                "supported_num_lookahead_tokens",
            )
        )
    ):
        raise ValueError("PARAKEET_TERNARY_GRAPH")
    return config


def vad_shapes(config):
    hidden = config["encoder_config"]["hidden_size"]
    return {
        "vad_head.proj.weight": (128, hidden, 1),
        "vad_head.proj.bias": (128,),
        "vad_head.ctx.weight": (128, 128, 5),
        "vad_head.ctx.bias": (128,),
        "vad_head.out.weight": (1, 128, 1),
        "vad_head.out.bias": (1,),
    }


def validate_metadata(name, metadata, config):
    validate_model_config(metadata)
    vad = metadata.get("vad_head", {})
    expected_vad = {
        "tensors": sorted(vad_shapes(config)),
        "d_model": config["encoder_config"]["hidden_size"],
        "hidden": 128,
        "kernel": 5,
        "frame_seconds": 0.08,
        "dtype": "float16",
        "params": sum(math.prod(s) for s in vad_shapes(config).values()),
    }
    if (
        name != "ternary.json"
        or set(metadata)
        != {
            "format",
            "names",
            "packing",
            "weight_rule",
            "quant",
            "source",
            "vad_head",
            "quantized_modules",
            "n_quantized_params",
            "n_dense_params",
            "size_mb",
        }
        or metadata["format"] != "thrush-ternary-v2"
        or metadata["names"] != "hf"
        or metadata["packing"] != PACKING
        or metadata["weight_rule"]
        != "w = scales[row, col // group_size] * (code - 1) with code in {0,1,2}"
        or metadata["quant"] != {"mode": "ternary", "group_size": 128}
        or metadata["source"] != {"teacher": "nvidia/parakeet-tdt-0.6b-v3"}
        or vad != expected_vad
        or type(metadata["n_quantized_params"]) is not int
        or metadata["n_quantized_params"] != sum(o * i for o, i, _ in modules(config).values())
        or type(metadata["n_dense_params"]) is not int
        or not 1 <= metadata["n_dense_params"] <= 2**31
        or type(metadata["size_mb"]) not in {int, float}
        or not 0 < metadata["size_mb"] <= 32768
    ):
        raise ValueError("PARAKEET_TERNARY_METADATA")
    entries = metadata["quantized_modules"]
    expected = modules(config)
    if not isinstance(entries, list) or len(entries) != len(expected):
        raise ValueError("PARAKEET_TERNARY_MODULES")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "name",
            "out_features",
            "in_features",
            "group_size",
            "as_conv1d",
            "has_bias",
            "zero_fraction",
        }:
            raise ValueError("PARAKEET_TERNARY_MODULES")
        key = entry["name"]
        if not isinstance(key, str) or key in seen or key not in expected:
            raise ValueError("PARAKEET_TERNARY_MODULES")
        out, ins, conv = expected[key]
        if (
            entry["out_features"] != out
            or entry["in_features"] != ins
            or entry["group_size"] != 128
            or entry["as_conv1d"] is not conv
            or entry["has_bias"] is not False
            or type(entry["zero_fraction"]) not in {int, float}
            or not 0 <= entry["zero_fraction"] <= 1
        ):
            raise ValueError("PARAKEET_TERNARY_MODULES")
        seen.add(key)


def estimated_memory(config, file_bytes):
    # Host float32 expansion and its device copy coexist during strict loading.
    expanded = sum(out * ins * 4 for out, ins, _ in modules(config).values())
    return 2 * (file_bytes * 2 + expanded) + 2048 * 1024**2


def native_model(config):
    import torch
    from transformers import ParakeetForTDT, ParakeetTDTConfig

    native = ParakeetTDTConfig.from_dict(copy.deepcopy(config))
    native._attn_implementation = native.encoder_config._attn_implementation = "sdpa"
    model = ParakeetForTDT(native)
    # Retain every auxiliary learned tensor. Bounded transcription does not
    # segment long recordings, so this head is not used by the public contract.
    model.vad_head = torch.nn.Module()
    model.vad_head.proj = torch.nn.Conv1d(config["encoder_config"]["hidden_size"], 128, 1)
    model.vad_head.ctx = torch.nn.Conv1d(128, 128, 5, padding=2)
    model.vad_head.out = torch.nn.Conv1d(128, 1, 1)
    return model


def packed_shapes(expected, config):
    shapes = {k: tuple(v.shape) for k, v in expected.items()}
    for name, (out, ins, conv) in modules(config).items():
        if shapes.pop(name + ".weight") != (out, ins) + ((1,) if conv else ()):
            raise ValueError("PARAKEET_NATIVE_GRAPH")
        shapes[name + ".qweight"] = (out, (ins + 4) // 5)
        shapes[name + ".scales"] = (out, ins // 128)
    return shapes


def unpack(qweight, scales, ins, conv):
    import torch

    if (
        qweight.dtype != torch.uint8
        or qweight.ndim != 2
        or qweight.shape[1] != (ins + 4) // 5
        or not scales.is_floating_point()
        or tuple(scales.shape) != (qweight.shape[0], ins // 128)
        or not bool(torch.isfinite(scales).all())
        or bool((scales < 0).any())
        or bool((qweight >= 243).any())
        or ins % 5
        and bool((qweight[:, -1] >= 3 ** (ins % 5)).any())
    ):
        raise ValueError("PARAKEET_TERNARY_NUMERIC")
    raw = qweight.to(torch.int16)
    digits = torch.stack([(raw // power) % 3 for power in (1, 3, 9, 27, 81)], dim=-1)
    weights = (
        digits.reshape(qweight.shape[0], -1)[:, :ins].float() - 1
    ) * scales.float().repeat_interleave(128, dim=1)
    return weights.unsqueeze(-1) if conv else weights


def components(root, config):
    from transformers import ParakeetFeatureExtractor, ParakeetTokenizer

    if any(
        (root / name).exists()
        for name in ("processor_config.json", "preprocessor_config.json", "tokenizer_config.json")
    ):
        raise ValueError("PARAKEET_UNBOUND_PROCESSOR")
    validate_metadata("ternary.json", metadata_file(root / "ternary.json"), config)
    path = root / "tokenizer.json"
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 4 * 1024**2:
        raise ValueError("PARAKEET_TOKENIZER_FILE")
    tokenizer = ParakeetTokenizer(
        tokenizer_file=str(path),
        pad_token="<pad>",
        unk_token="<unk>",
        clean_up_tokenization_spaces=False,
    )
    tokenizer.chat_template = None
    vocab = tokenizer.get_vocab()
    if (
        len(vocab) != config["vocab_size"]
        or set(vocab.values()) != set(range(len(vocab)))
        or tokenizer.pad_token_id != config["pad_token_id"]
        or tokenizer.unk_token_id != 0
        or vocab.get("<blank>") != config["blank_token_id"]
        or tokenizer.encode("<blank>", add_special_tokens=False) != [config["blank_token_id"]]
    ):
        raise ValueError("PARAKEET_VOCABULARY")
    # The closed v3 graph uses the native Parakeet frontend: 128 Slaney filters
    # and sample-standardized log-mels. Nemotron's unnormalized frontend differs.
    return ParakeetFeatureExtractor(
        **{k: v for k, v in rnnt_tasks.FEATURES.items() if k != "feature_extractor_type"}
    ), tokenizer


def load_components(root, config, backend):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    validate_config(config)
    processor = components(root, config)
    with init_empty_weights(include_buffers=False):
        model = native_model(config)
    expected = model.state_dict()
    if preflight_safetensors(root) != packed_shapes(expected, config):
        raise ValueError("PARAKEET_WEIGHTS_INCOMPLETE")
    values = {}
    for path in sorted(root.glob("*.safetensors")):
        part = load_file(path, device="cpu")
        if values.keys() & part.keys():
            raise ValueError("PARAKEET_WEIGHTS_DUPLICATE")
        values.update(part)
    for name, (_, ins, conv) in modules(config).items():
        values[name + ".weight"] = unpack(
            values.pop(name + ".qweight"), values.pop(name + ".scales"), ins, conv
        )
    for name, value in values.items():
        if expected[name].is_floating_point():
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError("PARAKEET_WEIGHTS_NUMERIC")
            values[name] = value.float()
        elif value.dtype != expected[name].dtype or bool((value < 0).any()):
            raise ValueError("PARAKEET_WEIGHTS_NUMERIC")
    model.load_state_dict(values, assign=True, strict=True)
    model.eval().to(backend)
    return model, processor, config


def decode(model, inputs, config):
    import torch
    from transformers.models.parakeet.generation_parakeet import ParakeetRNNTDecoderCache

    with torch.inference_mode():
        encoded = model.get_audio_features(**inputs, output_attention_mask=True)
        features = encoded.pooler_output
        frames = int(encoded.attention_mask.sum().item())
        if not 1 <= frames <= 375 or not bool(torch.isfinite(features).all()):
            raise ValueError("PARAKEET_ENCODER_OUTPUT")
        cache = ParakeetRNNTDecoderCache(model.config)
        token = config["blank_token_id"]
        frame, symbols, tokens, steps = 0, 0, [], []
        while frame < frames:
            prediction = model.decoder(torch.tensor([[token]], device=features.device), cache=cache)
            logits = model.joint(prediction, features[:, frame : frame + 1])[0, -1]
            if logits.shape != (config["vocab_size"] + 5,) or not bool(
                torch.isfinite(logits).all()
            ):
                raise ValueError("PARAKEET_LOGITS")
            token = int(logits[: config["vocab_size"]].argmax())
            duration = int(logits[config["vocab_size"] :].argmax())
            symbols += 1
            if duration == 0 and (
                token == config["blank_token_id"] or symbols >= config["max_symbols_per_step"]
            ):
                duration = 1
            steps.append((frame, token, duration))
            if token != config["blank_token_id"]:
                tokens.append(token)
            frame += duration
            if duration:
                symbols = 0
            if len(steps) > frames * config["max_symbols_per_step"]:
                raise ValueError("PARAKEET_DECODING_BOUND")
        return tokens, steps


def transcribe(model, processor, config, audio, backend):
    feature, tokenizer = processor
    # Reuse input bounds/finite checks, without Nemotron's prompt or lookahead.
    inputs = rnnt_tasks.prepare_inputs(feature, audio, config | {"default_prompt_id": 101}, backend)
    del inputs["prompt_ids"], inputs["num_lookahead_tokens"]
    tokens, _ = decode(model, inputs, config)
    text = tokenizer.decode(
        tokens, group_tokens=False, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    if len(text.encode()) > 16384:
        raise ValueError("PARAKEET_OUTPUT_BOUND")
    return {"text": text, "language": None}


def execute(model, processor, config, value, backend, input_path, emit):
    from oryn_engine.vad_tasks import read_audio

    if value.get("timestamps", False) or value.get("language", "auto") != "auto":
        raise ValueError("PARAKEET_PARAMETERS")
    audio, _ = read_audio(input_path)
    emit("job.progress", {"stage": "transcribing", "progress": 0.3})
    return transcribe(model, processor, config, audio, backend)
