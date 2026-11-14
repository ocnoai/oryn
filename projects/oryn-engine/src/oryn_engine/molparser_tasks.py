"""Closed MolParser Mobile V2 geometry and E-SMILES decoding on native timm/BART.

Reviewed original: UniParser/MolParser-Mobile-V2 at
11d114da4b66d1a98843f94dd4e9b208ea10645d. Repository Python is never imported.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from oryn_engine.model_metadata import (
    MAX_JSON_BYTES,
    metadata_file,
    parse_metadata,
    preflight_safetensors,
    validate_model_config,
)

FAMILY = "molparser_vision_encoder_decoder"
ARCHITECTURE = "MolParserVisionEncoderDecoderModel"
TASK = "image-to-text"
DATA = Path(__file__).parent / "molparser_assets"
MAX_TOKENS = 256  # Shared image-to-text bound, within native 384 including CLS.
METADATA_NAMES = (
    "preprocessor_config.json",
    "processor_config.json",
    "generation_config.json",
    "tokenizer_config.json",
)
FILES = ("config.json", "model.safetensors", "vocab.txt", *METADATA_NAMES)
EMBEDDING = "decoder.model.decoder.embed_tokens.weight"
ALIAS = "decoder.lm_head.weight"


def template(name):
    return json.loads((DATA / name).read_text())


def supports(config, task):
    return task == TASK and config.get("model_type") == FAMILY


def _closed(value, expected, flexible=()):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("MOLPARSER_CONFIGURATION")
    for k, v in expected.items():
        if k not in flexible and (type(value[k]) is not type(v) or value[k] != v):
            raise ValueError("MOLPARSER_CONFIGURATION:" + k)


def normalized_config(config, task):
    if not supports(config, task):
        return config
    c = copy.deepcopy(config)
    original = template("config.json")
    if c.get("architectures") != [ARCHITECTURE] or not isinstance(c.get("encoder"), dict):
        raise ValueError("MOLPARSER_CONFIGURATION")
    for part, expected in [(c, original), (c["encoder"], original["encoder"])]:
        if "auto_map" in part:
            if part["auto_map"] != expected["auto_map"]:
                raise ValueError("MOLPARSER_REMOTE_MAPPING")
            del part["auto_map"]
    validate_config(c)
    return c


def validate_config(c):
    validate_model_config(c)
    original = template("config.json")
    del original["auto_map"], original["encoder"]["auto_map"]
    _closed(c, original, {"encoder", "decoder"})
    e, d = c["encoder"], c["decoder"]
    _closed(e, original["encoder"], {"hidden_size", "timm_output_dim", "model_input_size"})
    bounds = {
        "d_model": (24, 192),
        "encoder_hidden_size": (24, 192),
        "decoder_layers": (1, 6),
        "decoder_ffn_dim": (48, 768),
        "max_position_embeddings": (16, 1024),
    }
    _closed(d, original["decoder"], bounds)
    for k, (low, high) in bounds.items():
        if type(d[k]) is not int or not low <= d[k] <= high:
            raise ValueError("MOLPARSER_DECODER_GEOMETRY")
    if (
        type(e["hidden_size"]) is not int
        or type(e["timm_output_dim"]) is not int
        or e["hidden_size"] != d["d_model"]
        or e["timm_output_dim"] != e["hidden_size"]
        or d["encoder_hidden_size"] != d["d_model"]
        or d["d_model"] % 12
        or d["decoder_ffn_dim"] < d["d_model"] * 2
        or type(e["model_input_size"]) is not int
        or not 64 <= e["model_input_size"] <= 384
        or e["model_input_size"] % 32
    ):
        raise ValueError("MOLPARSER_GEOMETRY")
    return c


def token_limit(config):
    return min(MAX_TOKENS, config["decoder"]["max_position_embeddings"] - 1)


def validate_metadata(name, value, config):
    expected = template(name)
    if name == "preprocessor_config.json":
        expected["image_size"] = config["encoder"]["model_input_size"]
    _closed(value, expected)


def finite(value):
    import torch

    if not bool(torch.isfinite(value).all()):
        raise ValueError("MOLPARSER_NONFINITE_OUTPUT")
    return value


def native_model(config):
    import timm
    import torch
    from torch import nn
    from transformers import BartConfig, BartForCausalLM

    validate_config(config)

    class MolParser(nn.Module):
        def __init__(self):
            super().__init__()
            # This small hybrid encoder probes its convolutional stem during
            # construction; allocate its bounded native graph on CPU, not meta.
            self.encoder = nn.Module()
            self.encoder.timm_model = timm.create_model(
                "vit_tiny_r_s16_p8_384.augreg_in21k_ft_in1k",
                pretrained=False,
                features_only=True,
                num_classes=0,
                global_pool="",
                img_size=config["encoder"]["model_input_size"],
                embed_dim=config["encoder"]["hidden_size"],
            )
            c = BartConfig(**copy.deepcopy(config["decoder"]))
            c._attn_implementation = "eager"
            self.decoder = BartForCausalLM(c)
            self.decoder.lm_head.register_forward_hook(lambda module, args, output: finite(output))

        def encode(self, pixels):
            features = finite(self.encoder.timm_model(pixels)[-1])
            width = config["encoder"]["hidden_size"]
            if features.ndim != 4:
                raise ValueError("MOLPARSER_FEATURE_GEOMETRY")
            if features.shape[1] != width:
                if features.shape[-1] != width:
                    raise ValueError("MOLPARSER_FEATURE_GEOMETRY")
                features = features.permute(0, 3, 1, 2).contiguous()
            return features.flatten(2).transpose(1, 2)

    with torch.device("cpu"):
        return MolParser().eval()


def validate_weight_shapes(config, shapes):
    model = native_model(config)
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    if ALIAS not in shapes:
        del expected[ALIAS]
    if expected != {k: tuple(v) for k, v in shapes.items()}:
        raise ValueError("MOLPARSER_WEIGHTS_INCOMPLETE")
    return model


def vocabulary(metadata):
    _closed(metadata, template("tokenizer_config.json"))
    words = list(
        dict.fromkeys(
            [
                *metadata["special_tokens"].values(),
                *metadata["vocab_list"],
                *metadata["additional_special_tokens"],
            ]
        )
    )
    if len(words) != 414 or words[:4] != ["[CLS]", "[PAD]", "[SEP]", "[UNK]"]:
        raise ValueError("MOLPARSER_VOCABULARY")
    return words


def decode(ids, words):
    if any(type(i) is not int or not 0 <= i < len(words) for i in ids):
        raise ValueError("MOLPARSER_OUTPUT_TOKENS")
    # Preserve chemical markup declared as additional special tokens; only the
    # four framing/unknown tokens are skipped by the original tokenizer.
    return "".join(words[i] for i in ids if i >= 4)


def load_components(root, config, backend):
    import torch
    from safetensors.torch import load_file

    if backend not in {"cpu", "mps"} or root.is_symlink():
        raise ValueError("MOLPARSER_SCOPE")
    validate_config(config)
    for name in FILES:
        if (root / name).is_symlink() or not (root / name).is_file():
            raise ValueError("MOLPARSER_EXPORT_INCOMPLETE")
    if {p.name for p in root.glob("*.safetensors")} != {"model.safetensors"} or any(
        root.glob("*.safetensors.index.json")
    ):
        raise ValueError("MOLPARSER_EXPORT_INCOMPLETE")
    # This is a closed vocabulary description, not tokenizer constructor options.
    # Its original auto_map must match the reviewed template and is never loaded.
    with (root / "tokenizer_config.json").open("rb") as stream:
        tokenizer_metadata = parse_metadata(stream.read(MAX_JSON_BYTES + 1), redundant_keys=True)
    for name in METADATA_NAMES:
        value = (
            tokenizer_metadata if name == "tokenizer_config.json" else metadata_file(root / name)
        )
        validate_metadata(name, value, config)
    if (root / "vocab.txt").read_bytes() != (DATA / "vocab.txt").read_bytes():
        raise ValueError("MOLPARSER_VOCABULARY")
    words = vocabulary(tokenizer_metadata)
    model = validate_weight_shapes(config, preflight_safetensors(root))
    state = load_file(root / "model.safetensors")
    for value in state.values():
        if value.dtype not in {torch.float16, torch.float32, torch.bfloat16} or not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError("MOLPARSER_NUMERIC_WEIGHT")
    if ALIAS in state and not torch.equal(state[ALIAS], state[EMBEDDING]):
        raise ValueError("MOLPARSER_SHARED_EMBEDDING")
    state[ALIAS] = state[EMBEDDING]
    model.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
    return model.to(backend).eval(), words, config


def pixels(image, config):
    import cv2
    import numpy as np
    import torch

    size = config["encoder"]["model_input_size"]
    values = cv2.resize(
        np.asarray(image.convert("RGB")), (size, size), interpolation=cv2.INTER_LINEAR
    )
    values = values.astype(np.float32) / 255.0
    values = (values - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(values.transpose(2, 0, 1).copy()).unsqueeze(0)


def execute(model, words, config, image, value, backend):
    import torch

    maximum = value["maxTokens"]
    if type(maximum) is not int or not 1 <= maximum <= token_limit(config):
        raise ValueError("MOLPARSER_OUTPUT_LIMIT")
    encoded = model.encode(pixels(image, config).to(backend))
    ids = model.decoder.generate(
        input_ids=torch.zeros((1, 1), dtype=torch.long, device=backend),
        encoder_hidden_states=encoded,
        encoder_attention_mask=torch.ones(encoded.shape[:2], dtype=torch.long, device=backend),
        max_new_tokens=maximum,
        do_sample=False,
        num_beams=1,
        use_cache=False,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
        forced_eos_token_id=2,
    )
    if (
        ids.ndim != 2
        or ids.shape[0] != 1
        or ids.shape[1] > maximum + 1
        or int(ids[0, 0]) != 0
        or int(ids[0, -1]) != 2
    ):
        raise ValueError("MOLPARSER_GENERATION")
    text = decode(ids[0].tolist(), words)
    if not text or len(text.encode()) > 16384:
        raise ValueError("MOLPARSER_EMPTY_OUTPUT")
    return {"text": text}
