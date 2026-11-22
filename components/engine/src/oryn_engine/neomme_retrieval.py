"""Native NeoMME page/query retrieval with fixed expansion and MeanMaxSim."""

from __future__ import annotations

import math

from oryn_engine.model_metadata import metadata_file, validate_model_config

METADATA_NAMES = ("processor_config.json", "tokenizer_config.json")
TOKENS = {
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "eos_token": "<eos>",
    "mask_token": "<mask>",
    "query_token": "<query>",
    "document_token": "<doc>",
    "image_token": "<img>",
    "row_token": "<row>",
}
QUERY_EXPANSION = 10


def validate_config(config):
    validate_model_config(config)
    if (
        config.get("model_type") != "neomme"
        or config.get("architectures") != ["NeoMMEForRetrieval"]
        or config.get("embedding_dim", 128) != 128
        or config.get("hidden_act", "relu2") != "relu2"
        or config.get("quantization_config")
        or config.get("quantization")
    ):
        raise ValueError("NEOMME_CONFIGURATION_UNSUPPORTED")
    for name, low, high in (
        ("hidden_size", 16, 4096),
        ("embedding_rank", 8, 1024),
        ("intermediate_size", 16, 16384),
        ("num_hidden_layers", 1, 64),
        ("num_attention_heads", 1, 128),
        ("num_key_value_heads", 1, 128),
        ("head_dim", 4, 256),
        ("vocab_size", 16, 262144),
        ("patch_size", 4, 64),
        ("max_position_embeddings", 256, 65536),
    ):
        value = config.get(name)
        if type(value) is not int or not low <= value <= high:
            raise ValueError("NEOMME_CONFIGURATION_LIMIT")
    if config["num_attention_heads"] % config["num_key_value_heads"]:
        raise ValueError("NEOMME_HEAD_GEOMETRY")
    layers = config.get("layer_types")
    if (
        not isinstance(layers, list)
        or len(layers) != config["num_hidden_layers"]
        or any(v not in {"full_attention", "sliding_attention"} for v in layers)
        or "full_attention" not in layers
    ):
        raise ValueError("NEOMME_LAYER_CONFIGURATION")
    rope = config.get("rope_parameters", {})
    if not isinstance(rope, dict):
        raise ValueError("NEOMME_ROTARY_CONFIGURATION")
    for kind in set(layers):
        part = rope.get(kind, {})
        if not isinstance(part, dict):
            raise ValueError("NEOMME_ROTARY_CONFIGURATION")
        factor = part.get("partial_rotary_factor", 0.25 if kind == "full_attention" else 1.0)
        theta = part.get("rope_theta", 1000000.0 if kind == "full_attention" else 10000.0)
        if (
            not isinstance(part, dict)
            or set(part) - {"rope_type", "rope_theta", "partial_rotary_factor"}
            or part.get("rope_type", "default") != "default"
            or type(factor) not in {int, float}
            or not 0 < factor <= 1
            or config["head_dim"] * factor % 4
            or type(theta) not in {int, float}
            or not math.isfinite(theta)
            or not 0 < theta <= 1e9
        ):
            raise ValueError("NEOMME_ROTARY_CONFIGURATION")
    overrides = config.get("per_layer_config", {})
    if not isinstance(overrides, dict):
        raise ValueError("NEOMME_LAYER_CONFIGURATION")
    seen = set()
    for key, part in overrides.items():
        if (
            not key.isdigit()
            or not 0 <= int(key) < len(layers)
            or int(key) in seen
            or not isinstance(part, dict)
            or set(part) != {"sliding_window"}
        ):
            raise ValueError("NEOMME_LAYER_CONFIGURATION")
        seen.add(int(key))
    for window in [
        config.get("sliding_window", 256),
        *(p["sliding_window"] for p in overrides.values()),
    ]:
        if window is not None and (type(window) is not int or not 1 <= window <= 4096):
            raise ValueError("NEOMME_LAYER_CONFIGURATION")
    for name, default in (("pad_token_id", 0), ("document_token_id", 5), ("image_token_id", 6)):
        token = config.get(name, default)
        if type(token) is not int or not 0 <= token < config["vocab_size"]:
            raise ValueError("NEOMME_SPECIAL_TOKENS")


def validate_metadata(name, metadata, config):
    validate_model_config(metadata)
    if name == "processor_config.json":
        image = metadata.get("image_processor", {})
        if (
            metadata.get("processor_class") != "NeoMMEProcessor"
            or not isinstance(image, dict)
            or image.get("image_processor_type") != "NeoMMEImageProcessor"
            or image.get("patch_size", 32) != config["patch_size"]
            or image.get("size") is not None
        ):
            raise ValueError("NEOMME_PROCESSOR_UNSUPPORTED")
        for key, expected in {
            "do_convert_rgb": True,
            "do_resize": True,
            "do_rescale": True,
            "do_normalize": True,
            "rescale_factor": 1 / 255,
            "resample": 2,
            "image_mean": [0.5] * 3,
            "image_std": [0.5] * 3,
        }.items():
            if image.get(key, expected) != expected:
                raise ValueError("NEOMME_PROCESSOR_UNSUPPORTED")
        side = image.get("max_side", 2048)
        if type(side) is not int or not 1 <= side <= 2048:
            raise ValueError("NEOMME_PROCESSOR_LIMIT")
    elif name == "tokenizer_config.json":
        if metadata.get("tokenizer_class") != "TokenizersBackend" or any(
            metadata.get(key) != token for key, token in TOKENS.items()
        ):
            raise ValueError("NEOMME_TOKENIZER_UNSUPPORTED")
    else:
        raise ValueError("NEOMME_METADATA_UNSUPPORTED")


def load_components(root, config, common):
    import transformers as tf

    metadata = {name: metadata_file(root / name) for name in METADATA_NAMES}
    for name, value in metadata.items():
        validate_metadata(name, value, config)
    path = root / "tokenizer.json"
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 32 * 1024**2:
        raise ValueError("NEOMME_TOKENIZER_UNAVAILABLE")
    tokenizer = tf.TokenizersBackend(
        tokenizer_file=str(path),
        model_max_length=256,
        chat_template=None,
        **{
            k: v
            for k, v in TOKENS.items()
            if k in {"pad_token", "unk_token", "eos_token", "mask_token"}
        },
        extra_special_tokens={
            k: v
            for k, v in TOKENS.items()
            if k in {"query_token", "document_token", "image_token", "row_token"}
        },
    )
    ids = [tokenizer.encode(token, add_special_tokens=False) for token in TOKENS.values()]
    if (
        len(tokenizer) != config["vocab_size"]
        or any(len(v) != 1 for v in ids)
        or len({v[0] for v in ids}) != len(ids)
    ):
        raise ValueError("NEOMME_TOKENIZER_VOCABULARY")
    for name, default in (("pad_token_id", 0), ("document_token_id", 5), ("image_token_id", 6)):
        if getattr(tokenizer, name) != config.get(name, default):
            raise ValueError("NEOMME_SPECIAL_TOKENS")
    # Bound patch tokens plus row markers to at most 212 of the contract's 256.
    side = min(
        metadata["processor_config.json"]["image_processor"].get("max_side", 2048),
        14 * config["patch_size"],
    )
    processor = tf.NeoMMEProcessor(
        tokenizer=tokenizer,
        chat_template=None,
        image_processor=tf.NeoMMEImageProcessor(
            patch_size=config["patch_size"],
            max_side=side,
            do_convert_rgb=True,
            do_resize=True,
            do_rescale=True,
            do_normalize=True,
            rescale_factor=1 / 255,
            resample=2,
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
        ),
    )
    return processor, tf.NeoMMEForRetrieval


def prepare_inputs(processor, config, image, queries):
    if any(any(token in query for token in TOKENS.values()) for query in queries):
        raise ValueError("NEOMME_RESERVED_TOKEN")
    page = dict(processor(images=[image], return_tensors="pt"))
    query_inputs = [
        dict(
            processor(
                text="<query>" + query + "<mask>" * QUERY_EXPANSION,
                return_tensors="pt",
                truncation=False,
            )
        )
        for query in queries
    ]
    if (
        page["input_ids"].shape[0] != 1
        or page["input_ids"].shape[1] > 256
        or tuple(page["position_ids"].shape) != (2, 1, page["input_ids"].shape[1])
        or page["pixel_values"].ndim != 2
        or page["pixel_values"].shape[0] > 196
        or page["pixel_values"].shape[1] != 3 * config["patch_size"] ** 2
    ):
        raise ValueError("NEOMME_PAGE_GEOMETRY")
    for query in query_inputs:
        ids = query["input_ids"]
        if ids.shape[0] != 1 or ids.shape[1] > 64:
            raise ValueError("NEOMME_QUERY_TOKEN_LIMIT")
        if (
            int(ids[0, 0]) != processor.tokenizer.query_token_id
            or ids[0, -QUERY_EXPANSION:].tolist()
            != [processor.tokenizer.mask_token_id] * QUERY_EXPANSION
        ):
            raise ValueError("NEOMME_QUERY_EXPANSION")
    return page, query_inputs


def retrieve(model, processor, config, image, value, backend):
    from oryn_engine.shared_tasks import _move
    from oryn_engine.visual_retrieval import late_interaction, vectors

    page, queries = prepare_inputs(processor, config, image, value["queries"])
    page = _move(page, backend)
    document = vectors(model(**page, output_dense=False), page["attention_mask"], 256)
    results = []
    for inputs in queries:
        inputs = _move(inputs, backend)
        query = vectors(model(**inputs, output_dense=False), inputs["attention_mask"], 64)
        results.append({"vectors": query, "score": late_interaction(query, document) / len(query)})
    return {
        "dimensions": 128,
        "normalized": True,
        "scoring": "mean-max-cosine",
        "documentVectors": document,
        "queries": results,
    }
