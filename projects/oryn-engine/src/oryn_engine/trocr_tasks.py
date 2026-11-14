"""Closed ViT/TrOCR image-to-text binding over the pinned native graph."""

from __future__ import annotations

import math

from oryn_engine.model_metadata import metadata_file, validate_model_config


def validate_config(config):
    validate_model_config(config)
    encoder, decoder = config.get("encoder", {}), config.get("decoder", {})
    if not isinstance(encoder, dict) or not isinstance(decoder, dict):
        raise ValueError("TROCR_CONFIGURATION_UNSUPPORTED")
    cross_attention = decoder.get("cross_attention_hidden_size")
    if (
        config.get("model_type") != "vision-encoder-decoder"
        or config.get("architectures") != ["VisionEncoderDecoderModel"]
        or config.get("is_encoder_decoder") is not True
        or encoder.get("model_type") != "vit"
        or decoder.get("model_type") != "trocr"
        or decoder.get("is_decoder") is not True
        or decoder.get("add_cross_attention") is not True
        or encoder.get("num_channels", 3) != 3
        or encoder.get("is_decoder", False) is not False
        or (
            cross_attention is not None
            and (type(cross_attention) is not int or cross_attention != encoder.get("hidden_size"))
        )
        or decoder.get("use_learned_position_embeddings", True) is not True
    ):
        raise ValueError("TROCR_CONFIGURATION_UNSUPPORTED")
    for part, bounds in (
        (
            encoder,
            {
                "hidden_size": (8, 4096),
                "intermediate_size": (8, 16384),
                "num_hidden_layers": (1, 48),
                "num_attention_heads": (1, 64),
                "image_size": (8, 1024),
                "patch_size": (1, 64),
            },
        ),
        (
            decoder,
            {
                "d_model": (8, 4096),
                "decoder_ffn_dim": (8, 16384),
                "decoder_layers": (1, 48),
                "decoder_attention_heads": (1, 64),
                "vocab_size": (8, 262144),
                "max_position_embeddings": (4, 4096),
            },
        ),
    ):
        for name, (low, high) in bounds.items():
            value = part.get(name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError("TROCR_CONFIGURATION_LIMIT")
    if (
        encoder["hidden_size"] % encoder["num_attention_heads"]
        or decoder["d_model"] % decoder["decoder_attention_heads"]
        or encoder["image_size"] % encoder["patch_size"]
        or (encoder["image_size"] // encoder["patch_size"]) ** 2 > 4096
    ):
        raise ValueError("TROCR_CONFIGURATION_UNSUPPORTED")
    generation_tokens(config)


def generation_tokens(config):
    decoder = config["decoder"]
    result = {}
    for name, default in (
        ("bos_token_id", 0),
        ("eos_token_id", 2),
        ("pad_token_id", 1),
        ("decoder_start_token_id", 2),
    ):
        value = config.get(name)
        if value is None:
            value = decoder.get(name, default)
        if type(value) is not int or not 0 <= value < decoder["vocab_size"]:
            raise ValueError("TROCR_TOKEN_CONFIGURATION_UNSUPPORTED")
        result[name] = value
    return result


def processor_options(metadata, config):
    validate_model_config(metadata)
    if metadata.get("image_processor_type", metadata.get("feature_extractor_type")) not in {
        "ViTImageProcessor",
        "ViTFeatureExtractor",
    }:
        raise ValueError("TROCR_PROCESSOR_UNSUPPORTED")
    size = metadata.get("size")
    expected = dict.fromkeys(("height", "width"), config["encoder"]["image_size"])
    if type(size) is int:
        size = dict.fromkeys(("height", "width"), size)
    if size != expected:
        raise ValueError("TROCR_PROCESSOR_GEOMETRY")
    options = {"size": expected, "resample": 2}
    for name in ("do_resize", "do_normalize", "do_rescale"):
        if metadata.get(name, True) is not True:
            raise ValueError("TROCR_PROCESSOR_UNSUPPORTED")
        options[name] = True
    if metadata.get("resample", 2) != 2:
        raise ValueError("TROCR_PROCESSOR_UNSUPPORTED")
    for name in ("image_mean", "image_std"):
        value = metadata.get(name)
        if (
            not isinstance(value, list)
            or len(value) != 3
            or any(
                type(v) not in {int, float} or not math.isfinite(v) or abs(v) > 100 for v in value
            )
            or name == "image_std"
            and any(v <= 0 for v in value)
        ):
            raise ValueError("TROCR_PROCESSOR_UNSUPPORTED")
        options[name] = value
    scale = metadata.get("rescale_factor", 1 / 255)
    if type(scale) not in {int, float} or not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("TROCR_PROCESSOR_UNSUPPORTED")
    options["rescale_factor"] = scale
    return options


def load_components(root, config, common):
    from transformers import (
        RobertaTokenizer,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        ViTImageProcessor,
    )

    metadata = metadata_file(root / "tokenizer_config.json")
    validate_model_config(metadata)
    if metadata.get("tokenizer_class") not in {"RobertaTokenizer", "RobertaTokenizerFast"}:
        raise ValueError("TROCR_TOKENIZER_UNSUPPORTED")
    tokenizer = RobertaTokenizer.from_pretrained(root, **common)
    tokenizer.chat_template = None
    if len(tokenizer) != config["decoder"]["vocab_size"]:
        raise ValueError("TROCR_TOKENIZER_VOCABULARY")
    tokens = generation_tokens(config)
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if getattr(tokenizer, name) != tokens[name]:
            raise ValueError("TROCR_TOKENIZER_SPECIAL_TOKENS")
    processor = TrOCRProcessor(
        tokenizer=tokenizer,
        image_processor=ViTImageProcessor(
            **processor_options(metadata_file(root / "preprocessor_config.json"), config)
        ),
    )
    return processor, VisionEncoderDecoderModel


def configure_generation(model, config):
    from transformers import GenerationConfig

    model.generation_config = GenerationConfig(
        **generation_tokens(config), do_sample=False, num_beams=1, use_cache=True
    )


def execute(model, processor, config, image, value, backend):
    inputs = processor(images=image, return_tensors="pt")
    size = config["encoder"]["image_size"]
    if set(inputs) != {"pixel_values"} or tuple(inputs["pixel_values"].shape) != (1, 3, size, size):
        raise ValueError("TROCR_IMAGE_GEOMETRY")
    limit = min(512, config["decoder"]["max_position_embeddings"] - 1)
    if not 1 <= value["maxTokens"] <= limit:
        raise ValueError("TROCR_OUTPUT_TOKEN_LIMIT")
    generated = model.generate(
        pixel_values=inputs["pixel_values"].to(backend),
        max_new_tokens=value["maxTokens"],
        do_sample=False,
    )
    answer = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    if not answer or len(answer.encode()) > 16384:
        raise ValueError("TROCR_OUTPUT_INVALID")
    return {"text": answer}
