"""Closed OmniVoice speech generation with the original bundled Higgs codec.

Only data is read from model repositories. The reviewed upstream iterative
algorithm runs in the shared PyTorch worker; no custom code, ASR or hub fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config
from oryn_engine.native_omnivoice.lang_map import LANG_IDS

FAMILY = "omnivoice"
ARCHITECTURE = "OmniVoice"
TASK = "text-to-speech"
MAX_TEXT = 300
MAX_TOKENS = 512
MAX_FRAMES = 745  # 29.8s plus the native 100ms padding at each edge = 30s.
LANGUAGES = tuple(sorted(LANG_IDS))
METADATA_NAMES = (
    "tokenizer_config.json",
    "audio_tokenizer/config.json",
    "audio_tokenizer/preprocessor_config.json",
)
FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    *METADATA_NAMES,
    "audio_tokenizer/model.safetensors",
)
DATA = Path(__file__).parent / "native_omnivoice"
CONTROLS = (
    "<|denoise|>",
    "<|lang_start|>",
    "<|lang_end|>",
    "<|instruct_start|>",
    "<|instruct_end|>",
    "<|text_start|>",
    "<|text_end|>",
)


def supports(config, task):
    return task == TASK and config.get("model_type") == FAMILY


def _closed(value, template, flexible=()):
    if not isinstance(value, dict) or set(value) != set(template):
        raise ValueError("OMNIVOICE_CONFIGURATION")
    for k, expected in template.items():
        if k not in flexible and (type(value[k]) is not type(expected) or value[k] != expected):
            raise ValueError("OMNIVOICE_CONFIGURATION:" + k)


def _integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("OMNIVOICE_GEOMETRY")


def validate_config(config):
    validate_model_config(config)
    template = json.loads((DATA / "main_config.json").read_text())
    _closed(config, template, {"llm_config", "pad_token_id", "eos_token_id"})
    q = config["llm_config"]
    ranges = {
        "hidden_size": (16, 1024),
        "intermediate_size": (16, 3072),
        "num_hidden_layers": (1, 28),
        "num_attention_heads": (1, 16),
        "num_key_value_heads": (1, 8),
        "head_dim": (2, 128),
        "vocab_size": (1035, 151676),
        "max_window_layers": (1, 28),
        "bos_token_id": (0, 151675),
        "eos_token_id": (0, 151675),
    }
    _closed(q, template["llm_config"], {*ranges, "layer_types"})
    for key, bounds in ranges.items():
        _integer(q[key], *bounds)
    if (
        q["num_attention_heads"] % q["num_key_value_heads"]
        or q["head_dim"] % 2
        or q["intermediate_size"] < q["hidden_size"]
        or q["layer_types"] != ["full_attention"] * q["num_hidden_layers"]
        or q["max_window_layers"] != q["num_hidden_layers"]
    ):
        raise ValueError("OMNIVOICE_GEOMETRY")
    for key, qkey in [("pad_token_id", "bos_token_id"), ("eos_token_id", "eos_token_id")]:
        _integer(config[key], 0, q["vocab_size"] - 1)
        if config[key] != q[qkey]:
            raise ValueError("OMNIVOICE_TOKENIZER_IDS")
    if config["pad_token_id"] == config["eos_token_id"]:
        raise ValueError("OMNIVOICE_TOKENIZER_IDS")
    return config


def validate_codec(config):
    validate_model_config(config)
    template = json.loads((DATA / "codec_config.json").read_text())
    _closed(config, template, {"acoustic_model_config", "semantic_model_config"})
    a, h = config["acoustic_model_config"], config["semantic_model_config"]
    aranges = {
        "encoder_hidden_size": (2, 64),
        "decoder_hidden_size": (32, 1024),
        "hidden_size": (8, 256),
    }
    hranges = {
        "hidden_size": (16, 768),
        "intermediate_size": (16, 3072),
        "num_hidden_layers": (1, 12),
        "num_attention_heads": (1, 12),
        "num_conv_pos_embeddings": (4, 128),
        "num_conv_pos_embedding_groups": (1, 16),
    }
    _closed(a, template["acoustic_model_config"], aranges)
    _closed(h, template["semantic_model_config"], {*hranges, "conv_dim"})
    for values, ranges in [(a, aranges), (h, hranges)]:
        for key, bounds in ranges.items():
            _integer(values[key], *bounds)
    if not isinstance(h["conv_dim"], list) or len(h["conv_dim"]) != 7:
        raise ValueError("OMNIVOICE_CODEC_GEOMETRY")
    for dim in h["conv_dim"]:
        _integer(dim, 2, 512)
    if (
        h["hidden_size"] % h["num_attention_heads"]
        or h["hidden_size"] % h["num_conv_pos_embedding_groups"]
        or h["intermediate_size"] < h["hidden_size"]
        or a["decoder_hidden_size"] % 32
    ):
        raise ValueError("OMNIVOICE_CODEC_GEOMETRY")
    return config


def validate_metadata(name, value, config=None):
    if name == "audio_tokenizer/config.json":
        validate_codec(value)
    elif name == "audio_tokenizer/preprocessor_config.json":
        _closed(value, json.loads((DATA / "codec_preprocessor.json").read_text()))
    elif name == "tokenizer_config.json":
        _closed(value, json.loads((DATA / "tokenizer_config.json").read_text()))
    else:
        raise ValueError("OMNIVOICE_METADATA")


def native_components(config, codec_config, *, meta=False, tokenizer=None):
    import torch
    from transformers import HiggsAudioV2TokenizerConfig

    from oryn_engine.native_omnivoice.codec import HiggsAudioV2TokenizerModel
    from oryn_engine.native_omnivoice.model import OmniVoice

    validate_config(config)
    validate_codec(codec_config)
    with torch.device("meta" if meta else "cpu"):
        main = OmniVoice(config, tokenizer).eval()
        codec = HiggsAudioV2TokenizerModel(HiggsAudioV2TokenizerConfig(**codec_config)).eval()
    return {"model": main, "audio_tokenizer/model": codec}


def validate_weight_shapes(config, codec_config, shapes):
    models = native_components(config, codec_config, meta=True)
    if set(shapes) != set(models):
        raise ValueError("OMNIVOICE_COMPONENTS_INCOMPLETE")
    for name, model in models.items():
        if {k: tuple(v) for k, v in shapes[name].items()} != {
            k: tuple(v.shape) for k, v in model.state_dict().items()
        }:
            raise ValueError("OMNIVOICE_COMPONENT_WEIGHTS_INCOMPLETE:" + name)


def load_tokenizer(root, config):
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    validate_metadata("tokenizer_config.json", metadata_file(root / "tokenizer_config.json"))
    path = root / "tokenizer.json"
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 16 * 1024**2:
        raise ValueError("OMNIVOICE_TOKENIZER")
    tok = Tokenizer.from_file(str(path))
    if (
        tok.get_vocab_size() != config["llm_config"]["vocab_size"]
        or set(tok.get_vocab().values()) != set(range(tok.get_vocab_size()))
        or tok.token_to_id("<|endoftext|>") != config["pad_token_id"]
        or tok.token_to_id("<|im_end|>") != config["eos_token_id"]
    ):
        raise ValueError("OMNIVOICE_TOKENIZER")
    tok.no_padding()
    tok.no_truncation()
    for control in CONTROLS:
        ident = tok.token_to_id(control)
        if ident is None or tok.encode(control, add_special_tokens=False).ids != [ident]:
            raise ValueError("OMNIVOICE_TOKENIZER_CONTROL")
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        model_max_length=40960,
    )


def _finite(value):
    import torch

    if (
        isinstance(value, torch.Tensor)
        and value.is_floating_point()
        and not bool(torch.isfinite(value).all())
    ):
        raise ValueError("OMNIVOICE_NONFINITE_OUTPUT")
    if isinstance(value, (list, tuple)):
        for v in value:
            _finite(v)
    if isinstance(value, dict):
        for v in value.values():
            _finite(v)
    return value


def _finite_hook(module, args, output):
    _finite(output)


def load_components(root, config, backend):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import HiggsAudioV2TokenizerConfig

    from oryn_engine.native_omnivoice.codec import HiggsAudioV2TokenizerModel
    from oryn_engine.native_omnivoice.model import OmniVoice

    if (
        backend not in {"cpu", "mps"}
        or root.is_symlink()
        or (root / "audio_tokenizer").is_symlink()
    ):
        raise ValueError("OMNIVOICE_BACKEND_OR_SCOPE")
    validate_config(config)
    for name in FILES:
        if (root / name).is_symlink() or not (root / name).is_file():
            raise ValueError("OMNIVOICE_COMPONENTS_INCOMPLETE")
    codec_config = metadata_file(root / "audio_tokenizer/config.json")
    for name in METADATA_NAMES:
        validate_metadata(name, metadata_file(root / name), config)
    shapes = {}
    for component in (root, root / "audio_tokenizer"):
        if {p.name for p in component.glob("*.safetensors")} != {"model.safetensors"} or any(
            component.glob("*.safetensors.index.json")
        ):
            raise ValueError("OMNIVOICE_COMPONENTS_INCOMPLETE")
        key = "model" if component == root else "audio_tokenizer/model"
        shapes[key] = preflight_safetensors(component, weight_names=("model.safetensors",))
    validate_weight_shapes(config, codec_config, shapes)
    tok = load_tokenizer(root, config)
    # Keep nonpersistent RoPE frequencies and native codec buffers materialized.
    # Learned arrays are still allocated only by the strict safe-tensor loader.
    with init_empty_weights(include_buffers=False):
        models = {
            "model": OmniVoice(config, tok),
            "audio_tokenizer/model": HiggsAudioV2TokenizerModel(
                HiggsAudioV2TokenizerConfig(**codec_config)
            ),
        }
    for name, model in models.items():
        state = load_file(root / (name + ".safetensors"))
        for key, tensor in state.items():
            if name == "model" and key == "codebook_layer_offsets":
                if tensor.dtype != torch.int64 or not torch.equal(tensor, torch.arange(8) * 1025):
                    raise ValueError("OMNIVOICE_CODEBOOK_OFFSETS")
            elif tensor.dtype not in {torch.float16, torch.bfloat16, torch.float32} or not bool(
                torch.isfinite(tensor).all()
            ):
                raise ValueError("OMNIVOICE_NUMERIC_WEIGHT")
            else:
                if key.endswith(".inited") and not torch.equal(tensor.float(), torch.ones(1)):
                    raise ValueError("OMNIVOICE_CODEBOOK_STATE")
                state[key] = tensor.float()
        model.load_state_dict(state, strict=True, assign=True)
        # The upstream codec also stays on CPU when the main graph is on MPS.
        model.to(backend if name == "model" else "cpu").eval()
        for module in model.modules():
            if not list(module.children()):
                module.register_forward_hook(_finite_hook)
    return models, tok, config


def generation_task(model, tok, text, language):
    from oryn_engine.native_omnivoice.model import (
        GenerationTask,
        _combine_text,
        _tokenize_with_nonverbal_tags,
    )

    if (
        not isinstance(text, str)
        or not text.strip()
        or len(text) > MAX_TEXT
        or "<|" in text
        or "|>" in text
        or language not in LANGUAGES
    ):
        raise ValueError("OMNIVOICE_TEXT_INPUT")
    ids = _tokenize_with_nonverbal_tags(f"<|text_start|>{_combine_text(text)}<|text_end|>", tok)
    if ids.shape[-1] > MAX_TOKENS or not bool(
        ((ids >= 0) & (ids < model.config.llm_config["vocab_size"])).all()
    ):
        raise ValueError("OMNIVOICE_TEXT_TOKEN_LIMIT")
    frames = model._estimate_target_tokens(text, None, None)
    if not 1 <= frames <= MAX_FRAMES:
        raise ValueError("OMNIVOICE_DURATION_LIMIT")
    return GenerationTask(1, [text], [frames], [language], [None], [None], [None], [None])


def synthesize(models, tok, text, language):
    import numpy as np

    from oryn_engine.native_omnivoice.audio import fade_and_pad_audio
    from oryn_engine.native_omnivoice.model import OmniVoiceGenerationConfig

    model, codec = models["model"], models["audio_tokenizer/model"]
    task = generation_task(model, tok, text, language)
    codes = model._generate_iterative(task, OmniVoiceGenerationConfig(postprocess_output=False))[0]
    if tuple(codes.shape) != (8, task.target_lens[0]) or not bool(
        ((codes >= 0) & (codes < 1024)).all()
    ):
        raise ValueError("OMNIVOICE_AUDIO_CODES")
    waveform = _finite(codec.decode(codes.cpu().unsqueeze(0)).audio_values)
    if tuple(waveform.shape) != (1, 1, task.target_lens[0] * 960):
        raise ValueError("OMNIVOICE_WAVEFORM_SHAPE")
    audio = waveform[0].float().cpu().numpy()
    peak = np.abs(audio).max()
    if peak == 0:
        raise ValueError("OMNIVOICE_EMPTY_OUTPUT")
    if peak > 1e-6:
        audio = audio / peak * 0.5
    audio = fade_and_pad_audio(audio, sample_rate=24000)
    if not np.isfinite(audio).all() or not 1 <= audio.size <= 30 * 24000 or np.abs(audio).max() > 1:
        raise ValueError("OMNIVOICE_AUDIO_OUTPUT")
    return audio[0]


def execute(models, tok, config, value, backend, input_path, output_path):
    import wave

    if (
        value.get("voice", "auto") != "auto"
        or value.get("speed", 1) != 1
        or "audioArtifactId" in value
        or input_path is not None
        or output_path is None
    ):
        raise ValueError("OMNIVOICE_SPEECH_CONTROLS")
    audio = synthesize(models, tok, value["text"], value.get("language", "en"))
    with wave.open(str(output_path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes((audio * 32767).round().astype("<i2").tobytes())
    return {
        "relativeOutput": "result.wav",
        "mediaType": "audio/wav",
        "sampleRate": 24000,
        "channels": 1,
        "frames": int(audio.size),
        "durationSeconds": audio.size / 24000,
        "backend": backend,
    }
