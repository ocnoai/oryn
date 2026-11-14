"""A closed Flax ndarray MessagePack grammar, without object deserializers.

Only nested string-key maps and extension-1 fixed-width real arrays are allowed.
No object restoration, import dispatch, plugins, pickle, external references,
chunk reconstruction, or repository code is involved.
"""

from __future__ import annotations

import math
import re

TASK_FAMILIES = {
    "summarization": {"bart", "t5", "mt5", "pegasus"},
    "text-classification": {"bert"},
    "zero-shot-image-classification": {"clip"},
    "automatic-speech-recognition": {"wav2vec2"},
}
FAMILIES = frozenset().union(*TASK_FAMILIES.values())
TEXT_TASKS = frozenset({"summarization", "text-classification"})
TASK_CONTRACTS = {
    "summarization": "summarization-v1",
    "text-classification": "text-classification-v1",
    "zero-shot-image-classification": "zero-shot-image-classification-v1",
}
# These tasks use the shared executor only for Flax. CTC already has a shared
# Safetensors executor and retains its existing transcription contract.


def supports(config, task):
    return config.get("model_type") in TASK_FAMILIES.get(task, set())


def decoding_options(config):
    """Only bounded native generation settings and literal task text are consumed."""
    options = {
        key: config[key]
        for key in (
            "num_beams",
            "no_repeat_ngram_size",
            "min_length",
            "length_penalty",
            "early_stopping",
        )
        if key in config
    }
    tasks = config.get("task_specific_params") or {}
    if not isinstance(tasks, dict):
        raise ValueError("FLAX_CONFIGURATION_UNSUPPORTED")
    summary = tasks.get("summarization") or {}
    if not isinstance(summary, dict) or set(summary) - {
        "num_beams",
        "no_repeat_ngram_size",
        "min_length",
        "max_length",
        "length_penalty",
        "early_stopping",
        "prefix",
    }:
        raise ValueError("FLAX_CONFIGURATION_UNSUPPORTED")
    options.update(summary)
    for key, low, high in (
        ("num_beams", 1, 8),
        ("no_repeat_ngram_size", 0, 8),
        ("min_length", 0, 128),
        ("max_length", 2, 1024),
    ):
        if key in options and (type(options[key]) is not int or not low <= options[key] <= high):
            raise ValueError("FLAX_CONFIGURATION_LIMIT")
    prefix = options.get("prefix", "")
    if not isinstance(prefix, str) or len(prefix) > 512 or any(ord(c) < 32 for c in prefix):
        raise ValueError("FLAX_CONFIGURATION_UNSUPPORTED")
    penalty = options.get("length_penalty", 1.0)
    if (
        type(options.get("early_stopping", False)) is not bool
        or type(penalty) not in {int, float}
        or not math.isfinite(penalty)
        or not 0 <= penalty <= 2
    ):
        raise ValueError("FLAX_CONFIGURATION_UNSUPPORTED")
    return options


def validate_config(config, task):
    from oryn_engine.model_metadata import validate_model_config

    validate_model_config(config)
    if not supports(config, task) or config.get("auto_map") or config.get("quantization_config"):
        raise ValueError("FLAX_CONFIGURATION_UNSUPPORTED")
    if task == "text-classification":
        validate_classifier_config(config)
        return
    if task == "zero-shot-image-classification":
        validate_clip_config(config)
        return
    if task == "automatic-speech-recognition":
        validate_ctc_config(config)
        return
    if config.get("architectures") == ["T5Model"] and (
        config.get("model_type") != "t5" or config.get("tie_word_embeddings", True) is not True
    ):
        raise ValueError("FLAX_CONFIGURATION_UNSUPPORTED")
    decoding_options(config)
    bounds = {
        "vocab_size": (1, 262144),
        "d_model": (32, 4096),
        "max_position_embeddings": (1, 4096),
        "encoder_layers": (1, 48),
        "decoder_layers": (1, 48),
        "encoder_attention_heads": (1, 64),
        "decoder_attention_heads": (1, 64),
        "encoder_ffn_dim": (32, 16384),
        "decoder_ffn_dim": (32, 16384),
        "num_layers": (1, 48),
        "num_decoder_layers": (1, 48),
        "num_heads": (1, 64),
        "d_kv": (1, 256),
        "d_ff": (32, 16384),
        "relative_attention_num_buckets": (1, 256),
        "relative_attention_max_distance": (1, 4096),
        "num_beams": (1, 8),
        "no_repeat_ngram_size": (0, 8),
        "min_length": (0, 128),
    }
    for name, (low, high) in bounds.items():
        if name == "num_decoder_layers" and config.get(name) is None:
            continue
        if name in config and (type(config[name]) is not int or not low <= config[name] <= high):
            raise ValueError("FLAX_CONFIGURATION_LIMIT")
    if not all(k in config for k in ("vocab_size", "d_model")):
        raise ValueError("FLAX_CONFIGURATION_UNSUPPORTED")
    for key in (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "decoder_start_token_id",
        "forced_bos_token_id",
        "forced_eos_token_id",
    ):
        value = config.get(key)
        if value is not None and (type(value) is not int or not 0 <= value < config["vocab_size"]):
            raise ValueError("FLAX_CONFIGURATION_UNSUPPORTED")
    if type(config.get("tie_word_embeddings", True)) is not bool:
        raise ValueError("FLAX_CONFIGURATION_UNSUPPORTED")
    if config["model_type"] == "mt5" and config.get("tie_word_embeddings", False):
        raise ValueError("FLAX_CONFIGURATION_UNSUPPORTED")
    if type(config.get("early_stopping", False)) is not bool:
        raise ValueError("FLAX_CONFIGURATION_UNSUPPORTED")
    penalty = config.get("length_penalty", 1.0)
    if type(penalty) not in {int, float} or not math.isfinite(penalty) or not 0 <= penalty <= 2:
        raise ValueError("FLAX_CONFIGURATION_UNSUPPORTED")


def validate_classifier_config(config):
    if config.get("architectures") != ["BertForSequenceClassification"]:
        raise ValueError("FLAX_CONFIGURATION_UNSUPPORTED")
    for name, low, high in (
        ("vocab_size", 1, 262144),
        ("hidden_size", 8, 4096),
        ("num_hidden_layers", 1, 48),
        ("num_attention_heads", 1, 64),
        ("intermediate_size", 8, 16384),
        ("max_position_embeddings", 1, 4096),
        ("type_vocab_size", 1, 16),
    ):
        value = config.get(name)
        if type(value) is not int or not low <= value <= high:
            raise ValueError("FLAX_CONFIGURATION_LIMIT")
    labels = config.get("id2label")
    if (
        not isinstance(labels, dict)
        or not 1 <= len(labels) <= 4096
        or set(labels) != {str(i) for i in range(len(labels))}
        or any(not isinstance(v, str) or not 1 <= len(v.strip()) <= 1024 for v in labels.values())
        or config["hidden_size"] % config["num_attention_heads"]
        or config.get("problem_type")
        not in {None, "single_label_classification", "multi_label_classification"}
        or config.get("position_embedding_type", "absolute") != "absolute"
        or config.get("is_decoder", False) is not False
        or config.get("add_cross_attention", False) is not False
        or (
            "num_labels" in config
            and (type(config["num_labels"]) is not int or config["num_labels"] != len(labels))
        )
    ):
        raise ValueError("FLAX_CONFIGURATION_UNSUPPORTED")
    pad = config.get("pad_token_id")
    if pad is not None and (type(pad) is not int or not 0 <= pad < config["vocab_size"]):
        raise ValueError("FLAX_CONFIGURATION_UNSUPPORTED")


# Inference defaults from the pinned native CLIP configuration, also checked
# against that class in regression tests. Metadata inspection imports no backend.
CLIP_DEFAULTS = {
    "text_config": {
        "hidden_size": 512,
        "intermediate_size": 2048,
        "num_hidden_layers": 12,
        "num_attention_heads": 8,
        "hidden_act": "quick_gelu",
        "layer_norm_eps": 1e-5,
        "attention_dropout": 0.0,
        "vocab_size": 49408,
        "max_position_embeddings": 77,
        "eos_token_id": 49407,
    },
    "vision_config": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "hidden_act": "quick_gelu",
        "layer_norm_eps": 1e-5,
        "attention_dropout": 0.0,
        "image_size": 224,
        "patch_size": 32,
        "num_channels": 3,
    },
}


def clip_legacy_eos_equivalent(config):
    text = config.get("text_config", {})
    legacy = config.get("text_config_dict")
    return (
        isinstance(legacy, dict)
        and text.get("eos_token_id") == 2
        and text.get("vocab_size") == 49408
        and legacy.get("vocab_size", 49408) == 49408
        and legacy.get("eos_token_id", 49407) == 49407
    )


def validate_clip_tokenizer(tokenizer, config):
    if clip_legacy_eos_equivalent(config) and (
        len(tokenizer) != 49408 or tokenizer.eos_token_id != 49407
    ):
        raise ValueError("CLIP_LEGACY_TOKENIZER_INCOMPATIBLE")


def validate_clip_config(config):

    if (
        config.get("architectures") != ["CLIPModel"]
        or type(config.get("projection_dim")) is not int
        or not 1 <= config["projection_dim"] <= 4096
    ):
        raise ValueError("FLAX_CONFIGURATION_UNSUPPORTED")
    for name, family in (
        ("text_config", "clip_text_model"),
        ("vision_config", "clip_vision_model"),
    ):
        part = config.get(name)
        if not isinstance(part, dict) or part.get("model_type") != family:
            raise ValueError("FLAX_CONFIGURATION_UNSUPPORTED")
        for key, low, high in (
            ("hidden_size", 8, 4096),
            ("intermediate_size", 8, 16384),
            ("num_hidden_layers", 1, 48),
            ("num_attention_heads", 1, 64),
        ):
            value = part.get(key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError("FLAX_CONFIGURATION_LIMIT")
        if (
            part["hidden_size"] % part["num_attention_heads"]
            or part.get("hidden_act", "quick_gelu") not in {"quick_gelu", "gelu"}
            or part.get("is_decoder", False) is not False
            or part.get("add_cross_attention", False) is not False
        ):
            raise ValueError("FLAX_CONFIGURATION_UNSUPPORTED")
        legacy = config.get(f"{name}_dict")
        if legacy is not None:
            if not isinstance(legacy, dict):
                raise ValueError("FLAX_CONFIGURATION_UNSUPPORTED")
            # Native CLIP expands the old alias with defaults before applying it.
            # Accept it only when every inference-relevant setting agrees, so a
            # short legacy dict cannot silently replace the validated full graph.
            for key, default in CLIP_DEFAULTS[name].items():
                if part.get(key, default) != legacy.get(key, default):
                    # Native CLIP's historical eos=2 means highest-token pooling.
                    # It equals explicit EOS 49407 only for the original fixed
                    # vocabulary, enforced again against the loaded tokenizer.
                    if (
                        name == "text_config"
                        and key == "eos_token_id"
                        and clip_legacy_eos_equivalent(config)
                    ):
                        continue
                    raise ValueError("FLAX_CLIP_LEGACY_CONFIGURATION_CONFLICT")
    text, vision = config["text_config"], config["vision_config"]
    for part, key, low, high in (
        (text, "vocab_size", 3, 262144),
        (text, "max_position_embeddings", 2, 512),
        (vision, "image_size", 4, 1024),
        (vision, "patch_size", 1, 64),
    ):
        value = part.get(key)
        if type(value) is not int or not low <= value <= high:
            raise ValueError("FLAX_CONFIGURATION_LIMIT")
    if (
        vision.get("num_channels", 3) != 3
        or vision["image_size"] % vision["patch_size"]
        or (vision["image_size"] // vision["patch_size"]) ** 2 > 4096
    ):
        raise ValueError("FLAX_CONFIGURATION_UNSUPPORTED")
    for key, default in (("bos_token_id", 49406), ("eos_token_id", 49407), ("pad_token_id", 1)):
        value = text.get(key, default)
        if type(value) is not int or not 0 <= value < text["vocab_size"]:
            raise ValueError("FLAX_CONFIGURATION_UNSUPPORTED")


def validate_ctc_config(config):
    from oryn_engine.onnx_asr import validate_config as validate_ctc

    validate_ctc(config)
    if (
        config.get("architectures") != ["Wav2Vec2ForCTC"]
        or config.get("do_stable_layer_norm") is not True
        or config.get("feat_extract_norm") != "layer"
        or config.get("add_adapter", False) is not False
        or config.get("adapter_attn_dim") is not None
        or config.get("hidden_act", "gelu") != "gelu"
        or config.get("feat_extract_activation", "gelu") != "gelu"
        or type(config.get("conv_bias", False)) is not bool
        or type(config.get("apply_spec_augment", True)) is not bool
    ):
        raise ValueError("FLAX_CTC_CONFIGURATION_UNSUPPORTED")
    for name, low, high in (
        ("hidden_size", 8, 4096),
        ("intermediate_size", 8, 16384),
        ("num_hidden_layers", 1, 48),
        ("num_attention_heads", 1, 64),
        ("num_conv_pos_embeddings", 2, 512),
        ("num_conv_pos_embedding_groups", 1, 64),
    ):
        value = config.get(name)
        if type(value) is not int or not low <= value <= high:
            raise ValueError("FLAX_CTC_CONFIGURATION_LIMIT")
    if (
        config["hidden_size"] % config["num_attention_heads"]
        or config["hidden_size"] % config["num_conv_pos_embedding_groups"]
    ):
        raise ValueError("FLAX_CTC_CONFIGURATION_UNSUPPORTED")
    for name, high in (("conv_dim", 1024), ("conv_kernel", 32), ("conv_stride", 16)):
        values = config.get(name)
        if (
            not isinstance(values, list)
            or not 1 <= len(values) <= 8
            or any(type(v) is not int or not 1 <= v <= high for v in values)
        ):
            raise ValueError("FLAX_CTC_CONFIGURATION_LIMIT")
    if len({len(config[name]) for name in ("conv_dim", "conv_kernel", "conv_stride")}) != 1:
        raise ValueError("FLAX_CTC_CONFIGURATION_UNSUPPORTED")
    probabilities = [config.get("mask_time_prob", 0.05), config.get("mask_feature_prob", 0.0)]
    if (
        any(
            type(v) not in {int, float} or not math.isfinite(v) or not 0 <= v <= 1
            for v in probabilities
        )
        or max(probabilities) <= 0
    ):
        # Flax always serializes the learned masking vector. This reviewed graph
        # requires the corresponding native parameter, even though eval uses no masks.
        raise ValueError("FLAX_CTC_CONFIGURATION_UNSUPPORTED")


def clip_processor_options(metadata, image_size=None):
    """Only native CLIP image operations, with bounded geometry and constants."""
    from oryn_engine.model_metadata import validate_model_config

    validate_model_config(metadata)
    if metadata.get("image_processor_type", "CLIPImageProcessor") != "CLIPImageProcessor" or (
        metadata.get("feature_extractor_type", "CLIPFeatureExtractor") != "CLIPFeatureExtractor"
    ):
        raise ValueError("FLAX_CLIP_PROCESSOR_UNSUPPORTED")
    options = {}
    for name in ("do_resize", "do_center_crop", "do_normalize", "do_rescale", "do_convert_rgb"):
        if metadata.get(name, True) is not True:
            raise ValueError("FLAX_CLIP_PROCESSOR_UNSUPPORTED")
        options[name] = True
    for name, axes in (("size", {"shortest_edge"}), ("crop_size", {"height", "width"})):
        value = metadata.get(name, 224)
        if type(value) is int:
            value = dict.fromkeys(axes, value)
        if (
            not isinstance(value, dict)
            or set(value) != axes
            or any(type(v) is not int or not 1 <= v <= 1024 for v in value.values())
            or (
                name == "crop_size"
                and image_size is not None
                and set(value.values()) != {image_size}
            )
        ):
            raise ValueError("FLAX_CLIP_PROCESSOR_GEOMETRY")
        options[name] = value
    for name, default in (
        ("image_mean", [0.48145466, 0.4578275, 0.40821073]),
        ("image_std", [0.26862954, 0.26130258, 0.27577711]),
    ):
        value = metadata.get(name, default)
        if (
            not isinstance(value, list)
            or len(value) != 3
            or any(
                type(v) not in {int, float} or not math.isfinite(v) or abs(v) > 100 for v in value
            )
            or (name == "image_std" and any(v <= 0 for v in value))
        ):
            raise ValueError("FLAX_CLIP_PROCESSOR_UNSUPPORTED")
        options[name] = value
    scale = metadata.get("rescale_factor", 1 / 255)
    if type(scale) not in {int, float} or not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("FLAX_CLIP_PROCESSOR_UNSUPPORTED")
    if type(metadata.get("resample", 3)) is not int or metadata.get("resample", 3) != 3:
        raise ValueError("FLAX_CLIP_PROCESSOR_UNSUPPORTED")
    options.update(resample=3, rescale_factor=scale)
    return options


class Reader:
    def __init__(self, source, size):
        if not 1 <= size <= 4 * 1024**3:
            raise ValueError("FLAX_FILE_LIMIT")
        self.source, self.size, self.offset = source, size, 0
        self.metadata_bytes, self.nodes = 0, 0
        self.tensors = {}

    def read(self, length):
        if not 0 <= length <= 512 or self.offset + length > self.size:
            raise ValueError("FLAX_TRUNCATED")
        self.metadata_bytes += length
        if self.metadata_bytes > 16 * 1024**2:
            raise ValueError("FLAX_METADATA_LIMIT")
        self.source.seek(self.offset)
        data = self.source.read(length)
        if len(data) != length:
            raise ValueError("FLAX_TRUNCATED")
        self.offset += length
        return data

    def number(self, width):
        return int.from_bytes(self.read(width), "big")

    def count(self, tag, family):
        base, wide = {
            "map": (0x80, {0xDE: 2, 0xDF: 4}),
            "array": (0x90, {0xDC: 2, 0xDD: 4}),
            "str": (0xA0, {0xD9: 1, 0xDA: 2, 0xDB: 4}),
        }[family]
        maximum = 32 if family == "str" else 16
        if base <= tag < base + maximum:
            return tag - base
        if tag in wide:
            return self.number(wide[tag])
        raise ValueError("FLAX_TYPE_UNSUPPORTED")

    def string(self):
        length = self.count(self.number(1), "str")
        if not 1 <= length <= 128:
            raise ValueError("FLAX_STRING_LIMIT")
        value = self.read(length).decode("ascii")
        if not re.fullmatch(r"[A-Za-z0-9_]+", value):
            raise ValueError("FLAX_KEY_INVALID")
        return value

    def integer(self):
        tag = self.number(1)
        if tag < 0x80:
            return tag
        if tag in {0xCC, 0xCD, 0xCE, 0xCF}:
            return self.number({0xCC: 1, 0xCD: 2, 0xCE: 4, 0xCF: 8}[tag])
        raise ValueError("FLAX_SHAPE_INVALID")

    def value(self, path=()):
        self.nodes += 1
        if self.nodes > 50000 or len(path) > 24:
            raise ValueError("FLAX_TREE_LIMIT")
        tag = self.number(1)
        if 0x80 <= tag <= 0x8F or tag in {0xDE, 0xDF}:
            count = self.count(tag, "map")
            if not 1 <= count <= 10000:
                raise ValueError("FLAX_MAP_LIMIT")
            keys = set()
            for _ in range(count):
                key = self.string()
                if key in keys:
                    raise ValueError("FLAX_DUPLICATE_KEY")
                keys.add(key)
                self.value((*path, key))
            return
        fixed_extensions = {0xD4: 1, 0xD5: 2, 0xD6: 4, 0xD7: 8, 0xD8: 16}
        if tag not in {0xC7, 0xC8, 0xC9, *fixed_extensions} or not path:
            raise ValueError("FLAX_TYPE_UNSUPPORTED")
        length = (
            fixed_extensions[tag]
            if tag in fixed_extensions
            else self.number({0xC7: 1, 0xC8: 2, 0xC9: 4}[tag])
        )
        if self.number(1) != 1:
            raise ValueError("FLAX_EXTENSION_UNSUPPORTED")
        end = self.offset + length
        if end > self.size or self.count(self.number(1), "array") != 3:
            raise ValueError("FLAX_ARRAY_INVALID")
        rank = self.count(self.number(1), "array")
        if not 0 <= rank <= 6 or (rank == 0 and path != ("logit_scale",)):
            raise ValueError("FLAX_SHAPE_INVALID")
        shape = tuple(self.integer() for _ in range(rank))
        if any(not 1 <= d <= 1048576 for d in shape) or math.prod(shape) > 2**31:
            raise ValueError("FLAX_SHAPE_INVALID")
        dtype = self.string()
        if dtype not in {"float16", "float32"}:
            raise ValueError("FLAX_DTYPE_UNSUPPORTED")
        tag = self.number(1)
        if tag not in {0xC4, 0xC5, 0xC6}:
            raise ValueError("FLAX_ARRAY_INVALID")
        size = self.number({0xC4: 1, 0xC5: 2, 0xC6: 4}[tag])
        if (
            size != math.prod(shape) * {"float16": 2, "float32": 4}[dtype]
            or self.offset + size != end
        ):
            raise ValueError("FLAX_ARRAY_INVALID")
        if len(self.tensors) >= 20000:
            raise ValueError("FLAX_TENSOR_LIMIT")
        self.tensors[".".join(path)] = {
            "dtype": dtype,
            "shape": shape,
            "offset": self.offset,
            "length": size,
        }
        self.offset = end


def inspect(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("FLAX_FILE_SCOPE")
    with path.open("rb") as source:
        reader = Reader(source, path.stat().st_size)
        reader.value()
        if reader.offset != reader.size or not reader.tensors:
            raise ValueError("FLAX_TRAILING_BYTES")
    return reader.tensors


def load_native(root, task):
    import numpy as np
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCTC,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForZeroShotImageClassification,
        GenerationConfig,
    )

    from oryn_engine.model_metadata import metadata_file

    raw = metadata_file(root / "config.json")
    validate_config(raw, task)
    path = root / "flax_model.msgpack"
    tensors = inspect(path)
    config = AutoConfig.for_model(
        raw["model_type"], **{k: v for k, v in raw.items() if k != "model_type"}
    )
    # Pinned Transformers forces MT5 tying even when the original config says
    # false. Preserve its independently serialized head before constructing the
    # native graph; strict key checks below prohibit borrowing missing weights.
    config.tie_word_embeddings = raw.get("tie_word_embeddings", raw["model_type"] != "mt5")
    loader = {
        "text-classification": AutoModelForSequenceClassification,
        "zero-shot-image-classification": AutoModelForZeroShotImageClassification,
        "summarization": AutoModelForSeq2SeqLM,
        "automatic-speech-recognition": AutoModelForCTC,
    }[task]
    with torch.device("meta"):
        model = loader.from_config(config, trust_remote_code=False, attn_implementation="eager")
    expected = model.state_dict()
    if sum(p.numel() for p in model.parameters()) > 1_500_000_000:
        raise ValueError("FLAX_PARAMETER_LIMIT")
    mapped = {}
    for name, tensor in tensors.items():
        parts = name.split(".")
        kernel = parts[-1] == "kernel"
        if kernel or parts[-1] in {"embedding", "scale"}:
            parts[-1] = "weight"
        key = ".".join(parts)
        if raw["model_type"] == "wav2vec2" and name in {
            "wav2vec2.encoder.pos_conv_embed.conv.weight_g",
            "wav2vec2.encoder.pos_conv_embed.conv.weight_v",
        }:
            key = "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight." + (
                "original0" if name.endswith("weight_g") else "original1"
            )
        shape = tuple(tensor["shape"])
        permutation = None
        if kernel:
            if len(shape) == 2:
                permutation = (1, 0)
            elif (
                raw["model_type"] == "wav2vec2"
                and len(shape) == 3
                and re.fullmatch(r"wav2vec2.feature_extractor.conv_layers\.\d+\.conv.kernel", name)
            ):
                permutation = (2, 1, 0)
            elif (
                raw["model_type"] == "clip"
                and name == "vision_model.embeddings.patch_embedding.kernel"
                and len(shape) == 4
            ):
                permutation = (3, 2, 0, 1)
            else:
                raise ValueError("FLAX_KERNEL_UNSUPPORTED")
            shape = tuple(shape[i] for i in permutation)
        if key in mapped or key not in expected or tuple(expected[key].shape) != shape:
            raise ValueError("FLAX_WEIGHT_GRAPH_MISMATCH")
        mapped[key] = (tensor, permutation)
    prefix = "model." if raw["model_type"] in {"bart", "pegasus"} else ""
    shared = prefix + "shared.weight"
    aliases = (
        {}
        if task != "summarization"
        else {
            prefix + "encoder.embed_tokens.weight": shared,
            prefix + "decoder.embed_tokens.weight": shared,
        }
    )
    if task == "summarization" and config.tie_word_embeddings:
        aliases["lm_head.weight"] = shared
    # Upstream Flax Pegasus computes these fixed sinusoids rather than storing
    # them as parameters. Recreate only those two native constants; every learned
    # tensor must still be present in the original numeric checkpoint.
    constants = {}
    if raw["model_type"] == "pegasus":
        constants = {
            "model.encoder.embed_positions.weight": model.model.encoder.embed_positions,
            "model.decoder.embed_positions.weight": model.model.decoder.embed_positions,
        }
    if (
        (task == "summarization" and shared not in mapped)
        or set(mapped) | set(aliases) | set(constants) != set(expected)
        or set(mapped) & (set(aliases) | set(constants))
    ):
        raise ValueError("FLAX_WEIGHT_GRAPH_MISMATCH")
    # Inspect the complete graph before allocating any native parameter storage.
    state = {}
    for key, (tensor, permutation) in mapped.items():
        array = np.memmap(
            path,
            mode="r",
            dtype={"float16": "<f2", "float32": "<f4"}[tensor["dtype"]],
            offset=tensor["offset"],
            shape=tuple(tensor["shape"]),
        )
        flat = array.reshape(-1)
        if any(
            not np.isfinite(flat[start : start + 1024**2]).all()
            for start in range(0, len(flat), 1024**2)
        ):
            raise ValueError("FLAX_WEIGHT_NONFINITE")
        state[key] = torch.from_numpy(
            np.array(
                array.transpose(permutation) if permutation else array, dtype=np.float32, copy=True
            )
        )
    state.update({name: state[target] for name, target in aliases.items()})
    state.update({name: module.create_weight() for name, module in constants.items()})
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    model.eval()
    if task == "text-classification":
        # BERT registers these deterministic, non-persistent input defaults on
        # the construction device. They are not learned checkpoint parameters.
        embeddings = model.bert.embeddings
        embeddings.position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))
        embeddings.token_type_ids = torch.zeros_like(embeddings.position_ids)
    if task == "zero-shot-image-classification":
        for embeddings in (model.text_model.embeddings, model.vision_model.embeddings):
            embeddings.position_ids = torch.arange(
                embeddings.position_embedding.num_embeddings
            ).expand((1, -1))
    if any(buffer.is_meta for buffer in model.buffers()):
        raise ValueError("FLAX_WEIGHT_GRAPH_MISMATCH")
    if task != "summarization":
        return model
    # Only these bounded native decoding options are read from legacy config.
    # Repository generation classes, imports, and extra options are never loaded.
    decoding = decoding_options(raw)
    model.generation_config = GenerationConfig(
        bos_token_id=getattr(config, "bos_token_id", None),
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
        decoder_start_token_id=config.decoder_start_token_id,
        forced_bos_token_id=raw.get("forced_bos_token_id"),
        forced_eos_token_id=raw.get(
            "forced_eos_token_id", getattr(config, "forced_eos_token_id", None)
        ),
        num_beams=decoding.get("num_beams", 1),
        no_repeat_ngram_size=decoding.get("no_repeat_ngram_size", 0),
        min_length=decoding.get("min_length", 0),
        length_penalty=decoding.get("length_penalty", 1.0),
        early_stopping=decoding.get("early_stopping", False),
        do_sample=False,
    )
    return model


def load_seq2seq(root, task):
    if task != "summarization":
        raise ValueError("FLAX_CONFIGURATION_UNSUPPORTED")
    return load_native(root, task)


def load_components(root, task, backend):
    from transformers import AutoTokenizer

    from oryn_engine.model_metadata import metadata_file, validate_model_config

    if backend not in {"cpu", "mps", "cuda"}:
        raise ValueError("FLAX_BACKEND_UNSUPPORTED")
    if (root / "tokenizer_config.json").exists():
        validate_model_config(metadata_file(root / "tokenizer_config.json"))
    raw = metadata_file(root / "config.json")
    if task == "automatic-speech-recognition":
        from oryn_engine.onnx_asr import load_components as load_ctc_components

        processor = load_ctc_components(root, raw)
        return load_native(root, task).to(backend), processor, raw
    if task == "zero-shot-image-classification":
        from transformers import CLIPImageProcessor, CLIPProcessor, CLIPTokenizer

        options = clip_processor_options(
            metadata_file(root / "preprocessor_config.json"), raw["vision_config"]["image_size"]
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            root, local_files_only=True, trust_remote_code=False
        )
        validate_clip_tokenizer(tokenizer, raw)
        tokenizer.chat_template = None
        processor = CLIPProcessor(
            tokenizer=tokenizer, image_processor=CLIPImageProcessor(**options)
        )
        return load_native(root, task).to(backend), processor, raw
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    tokenizer.chat_template = None
    model = load_native(root, task).to(backend)
    return model, tokenizer, metadata_file(root / "config.json")


def execute(model, tokenizer, value, backend, config, image=None):
    if config.get("model_type") == "clip":
        from oryn_engine.runners import _normalize_transformers_task_output

        if image is None:
            raise ValueError("FLAX_CLIP_IMAGE_REQUIRED")
        inputs = tokenizer(
            text=value["labels"], images=image, padding=True, truncation=False, return_tensors="pt"
        )
        if (
            not 1
            <= inputs["input_ids"].shape[1]
            <= config["text_config"]["max_position_embeddings"]
        ):
            raise ValueError("FLAX_INPUT_TOKEN_LIMIT")
        inputs = {k: v.to(backend) for k, v in inputs.items()}
        return _normalize_transformers_task_output(
            "zero-shot-image-classification",
            value,
            model(**inputs),
            tokenizer,
            model,
            inputs,
            image,
        )
    if config.get("model_type") == "bert":
        from oryn_engine.runners import _normalize_transformers_task_output

        inputs = tokenizer(value["text"], return_tensors="pt", truncation=False)
        if not 1 <= inputs["input_ids"].shape[1] <= min(512, config["max_position_embeddings"]):
            raise ValueError("FLAX_INPUT_TOKEN_LIMIT")
        inputs = {k: v.to(backend) for k, v in inputs.items()}
        return _normalize_transformers_task_output(
            "text-classification", value, model(**inputs), tokenizer, model, inputs, None
        )
    decoding = decoding_options(config)
    inputs = tokenizer(
        decoding.get("prefix", "") + value["text"], return_tensors="pt", truncation=False
    )
    if not 1 <= inputs["input_ids"].shape[1] <= 512:
        raise ValueError("FLAX_INPUT_TOKEN_LIMIT")
    generated = model.generate(
        **{k: v.to(backend) for k, v in inputs.items()},
        max_new_tokens=value.get("maxTokens", min(128, decoding.get("max_length", 129) - 1)),
        do_sample=False,
    )
    text = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    if not text or len(text.encode()) > 16384:
        raise ValueError("FLAX_OUTPUT_LIMIT")
    return {"text": text}
