"""Closed four-component Sopro speech graph with user-supplied reference audio.

The original repository is data only. Graphs are the reviewed Sopro 2.2.0 subset;
TorchAudio's pure tensor math is included without its incompatible binary package.
"""

from __future__ import annotations

import math
from dataclasses import asdict

from oryn_engine.model_metadata import preflight_safetensors, validate_model_config
from oryn_engine.native_sopro.config import SoproConfig

FAMILY = "oryn_sopro"
ARCHITECTURE = "OrynSoproForSpeechSynthesis"
TASK = "text-to-speech"
COMPONENTS = ("model", "semantic_encoder", "speaker_encoder", "vocoder")
WEIGHTS = tuple(name + ".safetensors" for name in COMPONENTS)
LANGUAGES = ("en", "pt", "fr", "de")
MAX_TEXT = 300
MAX_TOKENS = 512
MAX_REFERENCE = 10
MAX_SECONDS = 30


def supports(config, task):
    return task == TASK and config.get("model_type") == FAMILY


def normalized_config(config, task):
    if (
        task == TASK
        and {"sample_rate", *COMPONENTS, "generation"} <= config.keys()
        and "model_type" not in config
    ):
        config = dict(config, model_type=FAMILY, architectures=[ARCHITECTURE])
    if supports(config, task):
        validate_config(config)
    return config


def validate_config(config):
    validate_model_config(config)
    defaults = asdict(SoproConfig())
    if (
        set(config) != {*defaults, "model_type", "architectures"}
        or config.get("model_type") != FAMILY
        or config.get("architectures") != [ARCHITECTURE]
        or type(config.get("sample_rate")) is not int
        or config["sample_rate"] != 24000
    ):
        raise ValueError("SOPRO_CONFIGURATION")
    ranges = {
        "model": {
            "latent_dim": (16, 1280),
            "cond_hidden_dim": (16, 512),
            "ar_model_dim": (16, 512),
            "ar_blocks": (1, 12),
            "ar_heads": (1, 8),
            "ar_kv_heads": (1, 8),
            "acoustic_dit_dim": (16, 512),
            "acoustic_dit_depth": (1, 8),
            "acoustic_dit_heads": (1, 8),
            "acoustic_dit_dim_head": (2, 64),
            "acoustic_time_embed_dim": (2, 256),
        },
        "semantic_encoder": {
            "d_model": (16, 512),
            "layers": (1, 6),
            "heads": (1, 8),
            "ffn_dim": (16, 2048),
        },
        "speaker_encoder": {
            "stem_channels": (8, 128),
            "id_head_hidden": (8, 256),
            "style_head_hidden": (8, 256),
            "attn_hidden": (8, 128),
        },
        "vocoder": {"dim": (16, 512), "intermediate_dim": (16, 1536), "num_layers": (1, 8)},
    }
    flexible = {
        "model": {"acoustic_mel_mean", "acoustic_mel_std"},
        "speaker_encoder": {"stage_channels", "blocks_per_stage"},
        "vocoder": {"block_lookaheads"},
    }
    defaults["model"]["acoustic_sway_sampling_coef"] = 1.0
    defaults["generation"]["ref_seconds"] = 10.0
    for section in (*COMPONENTS, "generation"):
        values = config.get(section)
        if not isinstance(values, dict) or set(values) - set(defaults[section]):
            raise ValueError("SOPRO_CONFIGURATION")
        for key, value in values.items():
            expected = defaults[section][key]
            if key in ranges.get(section, {}):
                low, high = ranges[section][key]
                if type(value) is not int or not low <= value <= high:
                    raise ValueError("SOPRO_GEOMETRY")
            elif key not in flexible.get(section, set()) and (
                value != expected or type(value) is not type(expected)
            ):
                raise ValueError("SOPRO_CONFIGURATION")
    c = SoproConfig.from_dict(config)
    m, s, v = c.model, c.semantic_encoder, c.vocoder
    if m.acoustic_sway_sampling_coef != 1.0 or c.generation.ref_seconds != 10.0:
        raise ValueError("SOPRO_CONFIGURATION")
    if (
        m.ar_model_dim % m.ar_heads
        or m.ar_heads % m.ar_kv_heads
        or (m.ar_model_dim // m.ar_heads) % 2
        or m.acoustic_dit_dim % 16
        or m.acoustic_dit_dim_head % 2
        or m.acoustic_time_embed_dim % 2
        or s.d_model % s.heads
        or s.ffn_dim < s.d_model
        or v.intermediate_dim < v.dim
    ):
        raise ValueError("SOPRO_GEOMETRY")
    for name, positive in [("acoustic_mel_mean", False), ("acoustic_mel_std", True)]:
        values = getattr(m, name)
        if (
            not isinstance(values, list)
            or len(values) != 100
            or any(
                type(x) not in {int, float}
                or not math.isfinite(x)
                or not (1e-6 <= x <= 100 if positive else -100 <= x <= 100)
                for x in values
            )
        ):
            raise ValueError("SOPRO_MEL_STATISTICS")
    for key, limits in [("stage_channels", (160, 192, 224)), ("blocks_per_stage", (4, 4, 4))]:
        values = getattr(c.speaker_encoder, key)
        if (
            not isinstance(values, list)
            or len(values) != 3
            or any(
                type(x) is not int or not 1 <= x <= high
                for x, high in zip(values, limits, strict=True)
            )
        ):
            raise ValueError("SOPRO_SPEAKER_GEOMETRY")
    if v.block_lookaheads != [3] * v.num_layers or any(
        type(x) is not int for x in v.block_lookaheads
    ):
        raise ValueError("SOPRO_VOCODER_GEOMETRY")
    return c


def native_components(config, *, meta=False):
    import torch

    from oryn_engine.native_sopro.encoders.semantic import SemanticEncoder
    from oryn_engine.native_sopro.encoders.speaker import SpeakerEncoder
    from oryn_engine.native_sopro.nn.model import SoproModel
    from oryn_engine.native_sopro.vocoder import Vocoder

    c = validate_config(config)
    with torch.device("meta" if meta else "cpu"):
        result = {
            name: cls(getattr(c, name)).eval()
            for name, cls in zip(
                COMPONENTS, (SoproModel, SemanticEncoder, SpeakerEncoder, Vocoder), strict=True
            )
        }
    return result


def validate_weight_shapes(config, shapes):
    models = native_components(config, meta=True)
    if set(shapes) != set(COMPONENTS):
        raise ValueError("SOPRO_COMPONENTS_INCOMPLETE")
    for name, model in models.items():
        if {k: tuple(v) for k, v in shapes[name].items()} != {
            k: tuple(v.shape) for k, v in model.state_dict().items()
        }:
            raise ValueError("SOPRO_COMPONENT_WEIGHTS_INCOMPLETE:" + name)
    return models


def tokenizer(root, config):
    from oryn_engine.native_sopro.text import LANGUAGE_TAGS, TextTokenizer

    path = root / "tokenizer.model"
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 1024**2:
        raise ValueError("SOPRO_TOKENIZER")
    tok = TextTokenizer(str(path))
    if (
        tok.vocab_size != validate_config(config).model.text_vocab_size
        or (tok._sp.unk_id(), tok._sp.bos_id(), tok._sp.eos_id()) != (0, 1, 2)
        or any(
            tok._sp.piece_to_id(tag) in {0, 1, 2}
            or tok._sp.id_to_piece(tok._sp.piece_to_id(tag)) != tag
            for tag in LANGUAGE_TAGS.values()
        )
    ):
        raise ValueError("SOPRO_TOKENIZER")
    return tok


def text_ids(tok, text, language):
    from oryn_engine.native_sopro.text import language_tag, normalize_text

    if (
        not isinstance(text, str)
        or not text.strip()
        or len(text) > MAX_TEXT
        or "<|" in text
        or "|>" in text
        or language not in LANGUAGES
    ):
        raise ValueError("SOPRO_TEXT_INPUT")
    normalized = normalize_text(language_tag(language) + " " + text)
    ids = [tok.bos_id, *tok._sp.encode(normalized, out_type=int), tok.eos_id]
    if len(ids) > MAX_TOKENS or any(not 0 <= t < tok.vocab_size for t in ids):
        raise ValueError("SOPRO_TEXT_TOKEN_LIMIT")
    return ids


def load_components(root, config, backend):
    import torch
    from safetensors.torch import load_file

    if backend not in {"cpu", "mps"}:
        raise ValueError("SOPRO_BACKEND")
    validate_config(config)
    if {p.name for p in root.glob("*.safetensors")} != set(WEIGHTS) or any(
        root.glob("*.safetensors.index.json")
    ):
        raise ValueError("SOPRO_COMPONENTS_INCOMPLETE")
    shapes = {
        name: preflight_safetensors(root, weight_names=(name + ".safetensors",))
        for name in COMPONENTS
    }
    models = validate_weight_shapes(config, shapes)
    tok = tokenizer(root, config)
    for name, model in models.items():
        state = load_file(root / (name + ".safetensors"))
        if any(
            v.dtype not in {torch.float16, torch.bfloat16, torch.float32}
            or not torch.isfinite(v.float()).all()
            for v in state.values()
        ):
            raise ValueError("SOPRO_WEIGHT_VALUES")
        model.load_state_dict({k: v.float() for k, v in state.items()}, strict=True, assign=True)
        # Non-persistent FSQ radix values are graph constants, not checkpoint data.
        if name == "semantic_encoder":
            bases, base = [], 1
            for level in model.levels:
                bases.append(base)
                base *= level
            model._bases = torch.tensor(bases, dtype=torch.long)
        model.requires_grad_(False).to(backend)
        for module in model.modules():
            if not list(module.children()):
                module.register_forward_hook(_check_module_output)
    return models, tok, config


def finite(value):
    import torch

    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
        raise ValueError("SOPRO_NONFINITE_OUTPUT")
    return value


def semantic_tokens(model, ids, reference, generation):
    import torch

    from oryn_engine.native_sopro.nn.decode import EagerDecoder
    from oryn_engine.native_sopro.sampling import sample_next_token

    prefix = finite(
        model.build_prefix(
            ids, reference[:, : generation.style_tokens], reference[:, : generation.prompt_tokens]
        )
    )
    maximum = math.ceil(MAX_SECONDS * 24000 / 1024)
    minimum = math.ceil(generation.min_seconds * 24000 / 1024)
    decoder = EagerDecoder(model, prefix.shape[1] + maximum)
    logits = decoder.prefill(prefix)
    tokens = []
    for step in range(maximum):
        finite(logits)
        if logits.shape != (1, model.cfg.semantic_vocab_size + 2):
            raise ValueError("SOPRO_LOGITS_SHAPE")
        allow_eos = step + 1 >= minimum
        tok = sample_next_token(
            logits,
            generation.temperature,
            generation.top_p,
            generation.top_k,
            model.cfg.semantic_bos_id,
            model.cfg.semantic_eos_id,
            allow_eos,
        )
        value = int(tok.item())
        if allow_eos and value == model.cfg.semantic_eos_id:
            if not tokens:
                raise ValueError("SOPRO_EMPTY_OUTPUT")
            return torch.cat(tokens, dim=1)
        if not 0 <= value < model.cfg.semantic_vocab_size:
            raise ValueError("SOPRO_SEMANTIC_TOKEN")
        tokens.append(tok.view(1, 1))
        logits = decoder.step(tok)
    raise ValueError("SOPRO_GENERATION_LIMIT_WITHOUT_EOS")


def synthesize(models, tok, config, text, language, reference, sample_rate, backend):
    import torch

    from oryn_engine.native_sopro import audio, audio_math

    c = validate_config(config)
    model, semantic, speaker, vocoder = (models[k] for k in COMPONENTS)
    ids = torch.tensor([text_ids(tok, text, language)], dtype=torch.long, device=backend)
    if (
        reference.ndim != 1
        or not 0.4 * sample_rate <= reference.numel() <= MAX_REFERENCE * sample_rate
    ):
        raise ValueError("SOPRO_REFERENCE_DURATION")
    finite(reference)
    if float(reference.abs().max()) > 1 or float(reference.square().mean()) < 1e-10:
        raise ValueError("SOPRO_REFERENCE_SILENT_OR_INVALID")
    wav = audio.to_mono_resampled(reference, sample_rate, 24000).to(backend)
    wav = audio.crop_on_pause(wav, MAX_REFERENCE, 24000).unsqueeze(0)
    wav, level = audio.normalize_reference(wav, 24000)
    # Crop-on-pause may append the publisher's 250 ms room tone.
    if wav.shape[-1] > int((MAX_REFERENCE + 0.25) * 24000):
        raise ValueError("SOPRO_REFERENCE_DURATION")
    embedding = speaker(audio_math.resample(wav, 24000, 16000))
    condition = finite(
        model.build_condition(
            *(finite(embedding[k]) for k in ("id_emb", "style_emb", "style_ctrl"))
        )
    )
    ref_tokens = semantic.encode(wav)
    if (
        ref_tokens.ndim != 2
        or ref_tokens.shape[0] != 1
        or not bool(((ref_tokens >= 0) & (ref_tokens < c.model.semantic_vocab_size)).all())
    ):
        raise ValueError("SOPRO_REFERENCE_TOKENS")
    mean = torch.tensor(c.model.acoustic_mel_mean, device=backend).view(1, 100, 1)
    std = torch.tensor(c.model.acoustic_mel_std, device=backend).view(1, 100, 1)
    ref_mel = finite((vocoder.mel(wav) - mean) / std)
    tokens = semantic_tokens(model, ids, ref_tokens, c.generation)
    prefix = ref_mel.shape[-1]
    frames = prefix + tokens.shape[-1] * 4
    cond_mel = ref_mel.new_zeros(1, 100, frames)
    cond_mask = ref_mel.new_zeros(1, 1, frames)
    cond_mel[..., :prefix] = ref_mel
    cond_mask[..., :prefix] = 1
    x0 = torch.randn(1, 100, frames, device=backend)
    mel = finite(
        model.acoustic_head.solve(
            x0,
            torch.cat((ref_tokens, tokens), dim=1),
            condition,
            cond_mel,
            cond_mask,
            c.generation.steps,
        )
    )
    context = min(32, prefix)
    waveform = finite(vocoder.decode(mel[..., prefix - context :] * std + mean))[0]
    count = tokens.shape[-1] * 1024
    waveform = waveform[context * 256 : context * 256 + count]
    # Native centered ISTFT emits (frames - 1) hops; preserve its final
    # 256-sample shortening instead of padding an invented final frame.
    if waveform.numel() != count - 256 or count > MAX_SECONDS * 24000:
        raise ValueError("SOPRO_WAVEFORM_SHAPE")
    waveform = audio.trim_trail(audio.trim_lead(waveform * audio.output_gain(level), 24000), 24000)
    waveform = finite(audio.fade_edges(audio.soft_limit(waveform), 24000, False, True, 0.08))
    if not waveform.numel() or float(waveform.abs().max()) == 0:
        raise ValueError("SOPRO_EMPTY_OUTPUT")
    return waveform


def read_reference(path):
    import wave

    import numpy as np
    import torch

    if (
        path is None
        or path.is_symlink()
        or not path.is_file()
        or not 44 <= path.stat().st_size <= 2 * 1024**2
    ):
        raise ValueError("SOPRO_REFERENCE_PATH")
    with wave.open(str(path), "rb") as f:
        rate, channels, count = f.getframerate(), f.getnchannels(), f.getnframes()
        if (
            rate not in {8000, 16000, 24000, 32000, 44100, 48000}
            or channels not in {1, 2}
            or f.getsampwidth() != 2
            or f.getcomptype() != "NONE"
            or not 0.4 <= count / rate <= MAX_REFERENCE
        ):
            raise ValueError("SOPRO_REFERENCE_FORMAT")
        raw = f.readframes(count)
        if len(raw) != count * channels * 2:
            raise ValueError("SOPRO_REFERENCE_TRUNCATED")
    signal = (
        np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, channels).mean(axis=1)
        / 32768
    )
    return torch.from_numpy(signal), rate


def execute(models, tok, config, value, backend, input_path, output_path):
    import re
    import wave

    import numpy as np

    if (
        set(value) - {"text", "voice", "language", "speed", "audioArtifactId"}
        or not re.fullmatch(r"artifact_[A-Za-z0-9_-]{20,80}", value.get("audioArtifactId", ""))
        or value.get("voice", "reference") != "reference"
        or type(value.get("speed", 1)) not in {int, float}
        or value.get("speed", 1) != 1
        or value.get("language", "en") not in LANGUAGES
    ):
        raise ValueError("SOPRO_INPUT")
    if (
        output_path is None
        or output_path.name != "result.wav"
        or output_path.is_symlink()
        or not output_path.is_absolute()
        or not output_path.parent.is_dir()
    ):
        raise ValueError("SOPRO_OUTPUT_PATH")
    reference, rate = read_reference(input_path)
    waveform = synthesize(
        models, tok, config, value["text"], value.get("language", "en"), reference, rate, backend
    )
    samples = waveform.float().cpu().numpy()
    pcm = np.round(samples.clip(-1, 1) * 32767).astype("<i2")
    with wave.open(str(output_path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes(pcm.tobytes())
    return dict(
        relativeOutput="result.wav",
        mediaType="audio/wav",
        sampleRate=24000,
        channels=1,
        frames=len(pcm),
        durationSeconds=len(pcm) / 24000,
        backend=backend,
    )


def _check_module_output(module, inputs, output):
    import torch

    values = [output]
    while values:
        value = values.pop()
        if isinstance(value, torch.Tensor):
            finite(value)
        elif isinstance(value, dict):
            values.extend(value.values())
        elif isinstance(value, (tuple, list)):
            values.extend(value)
