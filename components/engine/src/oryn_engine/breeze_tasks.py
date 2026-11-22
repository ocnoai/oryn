"""Closed original Breeze TTS 2 export in the shared PyTorch environment."""

from __future__ import annotations

import json
import re
from pathlib import Path

from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

FAMILY = "breeze"
ARCHITECTURE = "BreezeForConditionalGeneration"
TASK = "text-to-speech"
MAX_TEXT = 300
MAX_TOKENS = 512
MAX_FRAMES = 375
DATA = Path(__file__).parent / "native_breeze"
METADATA_NAMES = (
    "special_tokens_map.json",
    "generation_config.json",
    "audio_tokenizer/config.json",
    "audio_tokenizer/preprocessor_config.json",
)
REQUIRED_FILES = (
    "config.json",
    "tokenizer.json",
    *METADATA_NAMES,
    "audio_tokenizer/model.safetensors",
)
AUDIO_ALIAS = "backbone_model.embed_tokens.embed_audio_tokens.weight"
AUDIO_WEIGHT = "depth_decoder.model.embed_tokens.weight"
SPEAKER_TOKENS = tuple(f"[S{i}]" for i in range(10))
SPECIAL_TOKENS = (*SPEAKER_TOKENS, "<ins_bos>", "<ins_eos>")


def supports(config, task):
    return task == TASK and config.get("model_type") == FAMILY


def _template(name):
    return json.loads((DATA / name).read_text())


def _closed(value, template, flexible=()):
    if not isinstance(value, dict) or set(value) != set(template):
        raise ValueError("BREEZE_CONFIGURATION")
    for key, expected in template.items():
        if key not in flexible and (
            type(value[key]) is not type(expected) or value[key] != expected
        ):
            raise ValueError("BREEZE_CONFIGURATION:" + key)


def _integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("BREEZE_GEOMETRY")


def _ranges(value, ranges):
    for key, bounds in ranges.items():
        _integer(value[key], *bounds)


def _attention(c):
    if (
        c["num_attention_heads"] % c["num_key_value_heads"]
        or c["head_dim"] % 2
        or c["intermediate_size"] < c["hidden_size"]
    ):
        raise ValueError("BREEZE_ATTENTION_GEOMETRY")


def _mimi(c, template):
    ranges = {
        "hidden_size": (16, 512),
        "intermediate_size": (16, 2048),
        "num_hidden_layers": (1, 8),
        "num_attention_heads": (1, 8),
        "num_key_value_heads": (1, 8),
        "head_dim": (2, 64),
        "num_filters": (2, 64),
        "codebook_dim": (8, 256),
        "vector_quantization_hidden_dimension": (8, 256),
        "upsample_groups": (16, 512),
    }
    _closed(c, template, ranges)
    _ranges(c, ranges)
    _attention(c)
    if (
        c["hidden_size"] != c["num_attention_heads"] * c["head_dim"]
        or c["upsample_groups"] != c["hidden_size"]
        or c["codebook_dim"] != c["vector_quantization_hidden_dimension"]
    ):
        raise ValueError("BREEZE_MIMI_GEOMETRY")


def validate_config(c):
    validate_model_config(c)
    template = _template("main_config.json")
    main = {
        "hidden_size": (16, 2048),
        "intermediate_size": (16, 6144),
        "num_hidden_layers": (1, 28),
        "num_attention_heads": (1, 16),
        "num_key_value_heads": (1, 8),
        "head_dim": (2, 128),
        "audio_embed_size": (16, 2048),
        "text_vocab_size": (64, 262158),
        "audio_token_id": (4, 262144),
        "audio_eos_token_id": (5, 262145),
    }
    sections = {
        "backbone_config",
        "codec_config",
        "depth_decoder_config",
        "text_encoder_config",
        "text_encoder_special_tokens_config",
    }
    _closed(c, template, {*main, *sections})
    _ranges(c, main)
    _attention(c)
    if (
        c["audio_embed_size"] != c["hidden_size"]
        or c["text_vocab_size"] != c["audio_token_id"] + 14
        or c["audio_eos_token_id"] != c["audio_token_id"] + 1
    ):
        raise ValueError("BREEZE_MAIN_GEOMETRY")
    q = c["backbone_config"]
    dims = {
        key: value
        for key, value in main.items()
        if key
        not in {"audio_embed_size", "text_vocab_size", "audio_token_id", "audio_eos_token_id"}
    }
    _closed(
        q,
        template["backbone_config"],
        {*dims, "max_window_layers", "vocab_size", "bos_token_id", "eos_token_id"},
    )
    _ranges(q, dims)
    _integer(q["vocab_size"], 64, 151936)
    _integer(q["bos_token_id"], 0, q["vocab_size"] - 1)
    _integer(q["eos_token_id"], 0, q["vocab_size"] - 1)
    if (
        any(q[key] != c[key] for key in dims)
        or q["max_window_layers"] != q["num_hidden_layers"]
        or type(q["max_window_layers"]) is not int
    ):
        raise ValueError("BREEZE_BACKBONE_GEOMETRY")
    d = c["depth_decoder_config"]
    depth = {
        "hidden_size": (16, 1024),
        "intermediate_size": (16, 8192),
        "num_hidden_layers": (1, 12),
        "num_attention_heads": (1, 8),
        "num_key_value_heads": (1, 2),
        "head_dim": (2, 128),
        "audio_embed_size": (16, 2048),
        "backbone_hidden_size": (16, 2048),
    }
    _closed(d, template["depth_decoder_config"], depth)
    _ranges(d, depth)
    _attention(d)
    if d["audio_embed_size"] != c["hidden_size"] or d["backbone_hidden_size"] != c["hidden_size"]:
        raise ValueError("BREEZE_DEPTH_GEOMETRY")
    t = c["text_encoder_config"]
    text = {
        "hidden_size": (16, 1152),
        "intermediate_size": (16, 6912),
        "num_hidden_layers": (1, 26),
        "num_attention_heads": (1, 4),
        "num_key_value_heads": (1, 1),
        "head_dim": (2, 256),
        "query_pre_attn_scalar": (2, 256),
        "vocab_size": (64, 262158),
        "eoi_token_index": (4, 256000),
        "sliding_window": (4, 512),
    }
    _closed(t, template["text_encoder_config"], {*text, "layer_types"})
    _ranges(t, text)
    _attention(t)
    if (
        t["vocab_size"] != c["text_vocab_size"]
        or t["eoi_token_index"] >= c["audio_token_id"]
        or t["query_pre_attn_scalar"] != t["head_dim"]
        or t["layer_types"]
        != [
            "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
            for i in range(t["num_hidden_layers"])
        ]
    ):
        raise ValueError("BREEZE_TEXT_GEOMETRY")
    specials = c["text_encoder_special_tokens_config"]
    _closed(specials, template["text_encoder_special_tokens_config"], {"token_ids"})
    expected = {name: c["audio_token_id"] + 2 + i for i, name in enumerate(SPECIAL_TOKENS)}
    if specials["token_ids"] != expected or any(
        type(x) is not int for x in specials["token_ids"].values()
    ):
        raise ValueError("BREEZE_SPECIAL_TOKENS")
    _mimi(c["codec_config"], template["codec_config"])
    return c


def validate_codec(c):
    validate_model_config(c)
    template = _template("codec_config.json")
    _closed(c, template, {"encoder_config", "decoder_config"})
    _mimi(c["encoder_config"], template["encoder_config"])
    d = c["decoder_config"]
    ranges = {
        "latent_dim": (16, 1024),
        "codebook_dim": (16, 512),
        "decoder_dim": (32, 1536),
        "hidden_size": (16, 512),
        "intermediate_size": (16, 1024),
        "head_dim": (2, 64),
        "num_attention_heads": (1, 16),
        "num_hidden_layers": (1, 8),
        "num_key_value_heads": (1, 16),
        "vector_quantization_hidden_dimension": (16, 512),
    }
    _closed(d, template["decoder_config"], ranges)
    _ranges(d, ranges)
    _attention(d)
    if (
        d["codebook_dim"] % 2
        or d["decoder_dim"] % 16
        or d["vector_quantization_hidden_dimension"] != d["codebook_dim"]
    ):
        raise ValueError("BREEZE_CODEC_GEOMETRY")
    return c


def validate_metadata(name, value, config=None):
    if name == "audio_tokenizer/config.json":
        validate_codec(value)
    elif name in {"special_tokens_map.json", "generation_config.json"}:
        _closed(value, _template(name))
    elif name == "audio_tokenizer/preprocessor_config.json":
        _closed(value, _template("codec_preprocessor.json"))
    else:
        raise ValueError("BREEZE_METADATA")


def native_components(config, codec_config, *, meta=False):
    import copy

    import torch

    from oryn_engine.native_breeze.codec import Qwen3TTSTokenizerV2Model
    from oryn_engine.native_breeze.codec_config import Qwen3TTSTokenizerV2Config
    from oryn_engine.native_breeze.model import Breeze

    validate_config(config)
    validate_codec(codec_config)
    codec = Qwen3TTSTokenizerV2Config(**copy.deepcopy(codec_config))
    for c in (codec, codec.encoder_config, codec.decoder_config):
        c._attn_implementation = "eager"
    with torch.device("meta" if meta else "cpu"):
        return {
            "model": Breeze(config).eval(),
            "audio_tokenizer/model": Qwen3TTSTokenizerV2Model(codec).eval(),
        }


def validate_weight_shapes(config, codec_config, shapes):
    models = native_components(config, codec_config, meta=True)
    if set(shapes) != set(models):
        raise ValueError("BREEZE_COMPONENTS_INCOMPLETE")
    for name, model in models.items():
        expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        if name == "model" and AUDIO_ALIAS not in shapes[name]:
            del expected[AUDIO_ALIAS]
        if expected != {k: tuple(v) for k, v in shapes[name].items()}:
            raise ValueError("BREEZE_COMPONENT_WEIGHTS_INCOMPLETE:" + name)


def load_tokenizer(root, config):
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    path = root / "tokenizer.json"
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 40 * 1024**2:
        raise ValueError("BREEZE_TOKENIZER")
    tok = Tokenizer.from_file(str(path))
    controls = {
        "<pad>": 0,
        "<eos>": 1,
        "<bos>": 2,
        "<unk>": 3,
        "<end_of_image>": config["text_encoder_config"]["eoi_token_index"],
        "<|AUDIO|>": config["audio_token_id"],
        "<|audio_eos|>": config["audio_eos_token_id"],
        **config["text_encoder_special_tokens_config"]["token_ids"],
    }
    if tok.get_vocab_size() != config["text_vocab_size"] or set(tok.get_vocab().values()) != set(
        range(config["text_vocab_size"])
    ):
        raise ValueError("BREEZE_TOKENIZER")
    tok.no_padding()
    tok.no_truncation()
    for token, ident in controls.items():
        if tok.token_to_id(token) != ident or tok.encode(token, add_special_tokens=False).ids != [
            ident
        ]:
            raise ValueError("BREEZE_TOKENIZER_CONTROL")
    if tok.encode("", add_special_tokens=True).ids != [2]:
        raise ValueError("BREEZE_TOKENIZER_BOS")
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        clean_up_tokenization_spaces=False,
        model_max_length=2048,
    )


def text_ids(tok, text, config):
    if (
        not isinstance(text, str)
        or not text.strip()
        or len(text) > MAX_TEXT
        or re.search(r"<|\[S[0-9]\]", text)
    ):
        raise ValueError("BREEZE_TEXT_INPUT")
    first = tok("[S0]" + text, add_special_tokens=True).input_ids
    rendered = tok.decode(first, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    ids = tok(rendered, add_special_tokens=False).input_ids
    if (
        not 3 <= len(ids) <= MAX_TOKENS
        or ids[:2] != [2, config["audio_token_id"] + 2]
        or any(not 0 <= i < config["text_vocab_size"] for i in ids)
    ):
        raise ValueError("BREEZE_TEXT_TOKENS")
    return ids


def _finite_hook(module, args, output):
    from oryn_engine.native_breeze.model import finite

    finite(output)


def load_components(root, config, backend):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    if (
        backend not in {"cpu", "mps"}
        or root.is_symlink()
        or (root / "audio_tokenizer").is_symlink()
    ):
        raise ValueError("BREEZE_SCOPE")
    validate_config(config)
    for name in REQUIRED_FILES:
        if (root / name).is_symlink() or not (root / name).is_file():
            raise ValueError("BREEZE_COMPONENTS_INCOMPLETE")
    codec = metadata_file(root / "audio_tokenizer/config.json")
    for name in METADATA_NAMES:
        validate_metadata(name, metadata_file(root / name), config)
    codec_root = root / "audio_tokenizer"
    if {p.name for p in codec_root.glob("*.safetensors")} != {"model.safetensors"} or any(
        codec_root.glob("*.safetensors.index.json")
    ):
        raise ValueError("BREEZE_COMPONENTS_INCOMPLETE")
    shapes = {
        "model": preflight_safetensors(root),
        "audio_tokenizer/model": preflight_safetensors(codec_root),
    }
    validate_weight_shapes(config, codec, shapes)
    tokenizer = load_tokenizer(root, config)
    with init_empty_weights(include_buffers=False):
        models = native_components(config, codec)
    for name, model in models.items():
        paths = (
            sorted(root.glob("*.safetensors"))
            if name == "model"
            else [codec_root / "model.safetensors"]
        )
        state = {}
        for path in paths:
            values = load_file(path)
            if state.keys() & values.keys():
                raise ValueError("BREEZE_DUPLICATE_WEIGHT")
            for key, tensor in values.items():
                if tensor.dtype not in {torch.float16, torch.bfloat16, torch.float32} or not bool(
                    torch.isfinite(tensor).all()
                ):
                    raise ValueError("BREEZE_NUMERIC_WEIGHT")
                state[key] = tensor.float()
        if name == "model":
            if AUDIO_ALIAS in state and not torch.equal(state[AUDIO_ALIAS], state[AUDIO_WEIGHT]):
                raise ValueError("BREEZE_SHARED_EMBEDDING")
            state[AUDIO_ALIAS] = state[AUDIO_WEIGHT]
        model.load_state_dict(state, strict=True, assign=True)
        model.to(backend).eval()
    # Catch nonfinite raw samples before the native decoder's final [-1,1] clamp.
    models["audio_tokenizer/model"].decoder.decoder[-1].register_forward_hook(_finite_hook)
    return models, tokenizer, config


def synthesize(models, tok, config, text):
    import torch

    from oryn_engine.native_breeze.model import finite, generate_codes

    main, codec = models["model"], models["audio_tokenizer/model"]
    ids = torch.tensor([text_ids(tok, text, config)], dtype=torch.long, device=main.device)
    codes = generate_codes(main, ids, MAX_FRAMES)
    if (
        codes.ndim != 3
        or codes.shape[0] != 1
        or codes.shape[2] != 16
        or not 1 <= codes.shape[1] <= MAX_FRAMES
        or not bool(((codes >= 0) & (codes < 2048)).all())
    ):
        raise ValueError("BREEZE_AUDIO_CODES")
    audio = finite(codec.decode(codes).audio_values[0])
    if (
        audio.ndim != 1
        or audio.numel() != codes.shape[1] * 1920
        or float(audio.abs().max()) == 0
        or float(audio.abs().max()) > 1
    ):
        raise ValueError("BREEZE_WAVEFORM")
    return audio


def execute(models, tok, config, value, backend, input_path, output_path):
    import wave

    import numpy as np

    if (
        value.get("voice", "auto") != "auto"
        or value.get("language", "auto") != "auto"
        or value.get("speed", 1) != 1
        or "audioArtifactId" in value
        or input_path is not None
        or output_path is None
    ):
        raise ValueError("BREEZE_SPEECH_CONTROLS")
    audio = synthesize(models, tok, config, value["text"]).float().cpu().numpy()
    with wave.open(str(output_path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes(np.round(audio * 32767).astype("<i2").tobytes())
    return {
        "relativeOutput": "result.wav",
        "mediaType": "audio/wav",
        "sampleRate": 24000,
        "channels": 1,
        "frames": len(audio),
        "durationSeconds": len(audio) / 24000,
        "backend": backend,
    }
