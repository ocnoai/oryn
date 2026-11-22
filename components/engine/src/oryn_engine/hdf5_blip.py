"""Closed BLIP numeric TensorFlow exports bound to complete native task graphs."""

import math
import re

from oryn_engine.model_metadata import metadata_file, validate_model_config


def validate_config(config):
    for name, family in (
        ("text_config", "blip_text_model"),
        ("vision_config", "blip_vision_model"),
    ):
        part = config.get(name)
        if not isinstance(part, dict) or part.get("model_type") != family:
            raise ValueError("HDF5_BLIP_CONFIGURATION")
        for key, low, high in (
            ("hidden_size", 8, 4096),
            ("intermediate_size", 8, 16384),
            ("num_hidden_layers", 1, 48),
            ("num_attention_heads", 1, 64),
        ):
            v = part.get(key)
            if type(v) is not int or not low <= v <= high:
                raise ValueError("HDF5_BLIP_CONFIGURATION")
        if (
            part["hidden_size"] % part["num_attention_heads"]
            or part.get("hidden_act", "gelu") != "gelu"
        ):
            raise ValueError("HDF5_BLIP_CONFIGURATION")
    text, vision = config["text_config"], config["vision_config"]
    for part, key, low, high in (
        (text, "vocab_size", 8, 131072),
        (text, "max_position_embeddings", 16, 4096),
        (vision, "image_size", 4, 1024),
        (vision, "patch_size", 1, 64),
    ):
        v = part.get(key)
        if type(v) is not int or not low <= v <= high:
            raise ValueError("HDF5_BLIP_CONFIGURATION")
    if (
        vision["image_size"] % vision["patch_size"]
        or (vision["image_size"] // vision["patch_size"]) ** 2 > 4096
        or vision.get("num_channels", 3) != 3
        or text.get("is_decoder", True) is not True
    ):
        raise ValueError("HDF5_BLIP_CONFIGURATION")
    special = token_ids(config)
    for v in special.values():
        if type(v) is not int or not 0 <= v < text["vocab_size"]:
            raise ValueError("HDF5_BLIP_SPECIAL_TOKENS")
    if len(set(special.values())) != 3:
        raise ValueError("HDF5_BLIP_SPECIAL_TOKENS")


def native_name(source, shape):
    match = re.fullmatch(
        r"(vision_model|text_encoder|text_decoder)/tf_blip_for_"
        r"(?:conditional_generation|question_answering)(?:_\d+)?/\1/(.+):0",
        source,
    )
    if not match:
        raise ValueError("HDF5_WEIGHT_NAME_UNSUPPORTED")
    prefix, leaf = match.groups()
    leaf = re.sub(r"(layers?)_\._(\d+)", r"\1/\2", leaf)
    parts = leaf.split("/")
    kernel = parts[-1] == "kernel"
    parts[-1] = {"kernel": "weight", "embeddings": "weight", "gamma": "weight", "beta": "bias"}.get(
        parts[-1], parts[-1]
    )
    name = prefix + "." + ".".join(parts)
    permutation = None
    if kernel:
        if name == "vision_model.embeddings.patch_embedding.weight" and len(shape) == 4:
            permutation = (3, 2, 0, 1)
        elif len(shape) == 2:
            permutation = (1, 0)
        else:
            raise ValueError("HDF5_WEIGHT_SHAPE_UNSUPPORTED")
    return name, permutation


def token_ids(config):
    text = config["text_config"]
    return {
        key: text.get(key, default)
        for key, default in (("bos_token_id", 30522), ("sep_token_id", 102), ("pad_token_id", 0))
    }


def prepare_native(model, config):
    import torch
    from transformers import GenerationConfig

    text = config["text_config"]
    embeddings = [model.text_decoder.bert.embeddings]
    if hasattr(model, "text_encoder"):
        embeddings.append(model.text_encoder.embeddings)
    for module in embeddings:
        module.position_ids = torch.arange(text["max_position_embeddings"]).expand((1, -1))
    tokens = token_ids(config)
    model.text_decoder.generation_config = GenerationConfig(
        bos_token_id=tokens["bos_token_id"],
        eos_token_id=tokens["sep_token_id"],
        pad_token_id=tokens["pad_token_id"],
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )


def processor_options(metadata, config):
    validate_model_config(metadata)
    if metadata.get("image_processor_type") != "BlipImageProcessor":
        raise ValueError("HDF5_BLIP_PROCESSOR")
    size = metadata.get("size")
    expected = dict.fromkeys(("height", "width"), config["vision_config"]["image_size"])
    if type(size) is int:
        size = dict.fromkeys(("height", "width"), size)
    if size != expected or metadata.get("resample", 3) != 3:
        raise ValueError("HDF5_BLIP_PROCESSOR")
    options = {"size": expected, "resample": 3}
    for name in ("do_resize", "do_rescale", "do_normalize", "do_convert_rgb"):
        if metadata.get(name, True) is not True:
            raise ValueError("HDF5_BLIP_PROCESSOR")
        options[name] = True
    for name in ("image_mean", "image_std"):
        v = metadata.get(name)
        if (
            not isinstance(v, list)
            or len(v) != 3
            or any(type(x) not in {int, float} or not math.isfinite(x) or abs(x) > 100 for x in v)
            or name == "image_std"
            and any(x <= 0 for x in v)
        ):
            raise ValueError("HDF5_BLIP_PROCESSOR")
        options[name] = v
    scale = metadata.get("rescale_factor", 1 / 255)
    if type(scale) not in {int, float} or not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("HDF5_BLIP_PROCESSOR")
    options["rescale_factor"] = scale
    return options


def load_processor(root, config):
    from transformers import BertTokenizer, BlipImageProcessor, BlipProcessor

    tokenizer = BertTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    text = config["text_config"]
    tokens = token_ids(config)
    # Original BLIP reserves two learned vocabulary rows outside BERT's file.
    # Bind only those exact existing rows, following Salesforce's tokenizer.
    if len(tokenizer) == text["vocab_size"] - 2 and tokens["bos_token_id"] == len(tokenizer):
        tokenizer.add_special_tokens({"bos_token": "[DEC]", "additional_special_tokens": ["[ENC]"]})
    if (
        len(tokenizer) != text["vocab_size"]
        or tokenizer.convert_tokens_to_ids("[DEC]") != tokens["bos_token_id"]
        or tokenizer.sep_token_id != tokens["sep_token_id"]
        or tokenizer.pad_token_id != tokens["pad_token_id"]
    ):
        raise ValueError("HDF5_BLIP_TOKENIZER")
    tokenizer.chat_template = None
    return BlipProcessor(
        tokenizer=tokenizer,
        image_processor=BlipImageProcessor(
            **processor_options(metadata_file(root / "preprocessor_config.json"), config)
        ),
    )


def token_limit(config):
    maximum = config.get("text_config", {}).get("max_position_embeddings", 512)
    if type(maximum) is not int or not 16 <= maximum <= 4096:
        raise ValueError("HDF5_BLIP_CONFIGURATION")
    return min(512, maximum - 1)
