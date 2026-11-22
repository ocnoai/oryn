"""Native Qwen3-VL text retrieval, with owned prompts and complete graph loading.

The embedding checkpoints publish a Qwen3VLModel inside a `model` wrapper.
Reranking uses the original language head's yes/no rows, never a new classifier.
Repository Python, sentence-transformers modules and Jinja are not executed.
"""

from __future__ import annotations

import math

from oryn_engine.model_metadata import metadata_file, validate_model_config

TASK_CONTRACTS = {"feature-extraction": "embeddings-v1", "text-ranking": "reranking-v1"}
TASK_FAMILIES = {task: {"qwen3_vl"} for task in TASK_CONTRACTS}
MAX_BATCH = 4
MAX_TOKENS = 4096
EMBEDDING_INSTRUCTION = "Represent the user's input."
RANKING_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".'
)


def supports(task, config):
    return task in TASK_CONTRACTS and config.get("model_type") == "qwen3_vl"


def normalized_config(config, task):
    # RenderRank explicitly supports ordinary text documents with this unchanged
    # Qwen scorer. Its optional image rendering is outside the text-only contract.
    if (
        task == "text-ranking"
        and config.get("model_type") == "qwen3_vl"
        and config.get("auto_map") == {"AutoModel": "modeling_renderrank.RenderRank"}
    ):
        config = {key: value for key, value in config.items() if key != "auto_map"}
        validate_config(config)
    return config


def validate_config(config):
    validate_model_config(config)
    text, vision = config.get("text_config"), config.get("vision_config")
    if (
        config.get("model_type") != "qwen3_vl"
        or config.get("architectures") != ["Qwen3VLForConditionalGeneration"]
        or not isinstance(text, dict)
        or not isinstance(vision, dict)
        or text.get("model_type") != "qwen3_vl_text"
        or vision.get("model_type") not in {"qwen3_vl", "qwen3_vl_vision"}
        or config.get("quantization_config")
        or config.get("quantization")
    ):
        raise ValueError("QWEN_RETRIEVAL_CONFIGURATION_UNSUPPORTED")
    validate_geometry(config)


def validate_geometry(config):
    """Native Qwen3-VL tensor geometry shared by text and page retrieval."""
    text, vision = config["text_config"], config["vision_config"]
    for part, bounds in (
        (
            text,
            {
                "hidden_size": (8, 8192),
                "intermediate_size": (8, 65536),
                "num_hidden_layers": (1, 80),
                "num_attention_heads": (1, 64),
                "num_key_value_heads": (1, 64),
                "head_dim": (8, 256),
                "vocab_size": (32, 262144),
                "max_position_embeddings": (64, 1048576),
            },
        ),
        (
            vision,
            {
                "hidden_size": (8, 2048),
                "intermediate_size": (8, 16384),
                "depth": (1, 48),
                "num_heads": (1, 64),
                "out_hidden_size": (8, 8192),
                "patch_size": (1, 32),
                "temporal_patch_size": (1, 4),
                "spatial_merge_size": (1, 4),
                "num_position_embeddings": (4, 16384),
            },
        ),
    ):
        for name, (low, high) in bounds.items():
            if type(part.get(name)) is not int or not low <= part[name] <= high:
                raise ValueError("QWEN_RETRIEVAL_CONFIGURATION_LIMIT")
    if (
        text["num_attention_heads"] % text["num_key_value_heads"]
        or vision["hidden_size"] % vision["num_heads"]
        or vision["out_hidden_size"] != text["hidden_size"]
        or vision.get("in_channels", 3) != 3
    ):
        raise ValueError("QWEN_RETRIEVAL_CONFIGURATION_UNSUPPORTED")
    rope = text.get("rope_parameters", text.get("rope_scaling", {}))
    sections = rope.get("mrope_section") if isinstance(rope, dict) else None
    if (
        not isinstance(rope, dict)
        or rope.get("rope_type") != "default"
        or rope.get("mrope_interleaved") is not True
        or not isinstance(sections, list)
        or len(sections) != 3
        or any(type(s) is not int or s <= 0 for s in sections)
        or sum(sections) != text["head_dim"] // 2
        or text["head_dim"] % 2
        or not set(rope) <= {"rope_type", "rope_theta", "mrope_section", "mrope_interleaved"}
        or "rope_parameters" in text
        and "rope_scaling" in text
    ):
        raise ValueError("QWEN_RETRIEVAL_ROPE_UNSUPPORTED")
    theta = rope.get("rope_theta", text.get("rope_theta", 5000000))
    if type(theta) not in {int, float} or not math.isfinite(theta) or not 1 <= theta <= 1e9:
        raise ValueError("QWEN_RETRIEVAL_ROPE_UNSUPPORTED")
    for name in (
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    ):
        if type(config.get(name)) is not int or not 0 <= config[name] < text["vocab_size"]:
            raise ValueError("QWEN_RETRIEVAL_SPECIAL_TOKENS")
    indexes = vision.get("deepstack_visual_indexes")
    if (
        not isinstance(indexes, list)
        or not 1 <= len(indexes) <= min(8, text["num_hidden_layers"])
        or any(type(i) is not int or not 0 <= i < vision["depth"] for i in indexes)
        or indexes != sorted(set(indexes))
    ):
        raise ValueError("QWEN_RETRIEVAL_VISION_GRAPH")


def metadata_names(task):
    return ("tokenizer_config.json",) + (
        ("1_Pooling/config.json", "config_sentence_transformers.json")
        if task == "feature-extraction"
        else ()
    )


def validate_metadata(name, value, config):
    validate_model_config(value)
    if name == "tokenizer_config.json":
        if value.get("tokenizer_class") not in {"Qwen2Tokenizer", "Qwen2TokenizerFast"}:
            raise ValueError("QWEN_RETRIEVAL_TOKENIZER_UNSUPPORTED")
        if (
            value.get("add_bos_token", False) is not False
            or value.get("add_prefix_space") is not None
            and value["add_prefix_space"] is not False
        ):
            raise ValueError("QWEN_RETRIEVAL_TOKENIZER_UNSUPPORTED")
    elif name == "1_Pooling/config.json":
        if (
            value.get("include_prompt") is not True
            or type(value.get("embedding_dimension")) is not int
            or value
            != {
                "embedding_dimension": config["text_config"]["hidden_size"],
                "pooling_mode": "lasttoken",
                "include_prompt": True,
            }
        ):
            raise ValueError("QWEN_RETRIEVAL_POOLING_UNSUPPORTED")
    elif name == "config_sentence_transformers.json":
        if (
            value.get("default_prompt_name") != "default"
            or value.get("prompts") != {"default": EMBEDDING_INSTRUCTION}
            or value.get("similarity_fn_name") != "cosine"
        ):
            raise ValueError("QWEN_RETRIEVAL_PROMPT_UNSUPPORTED")


def load_components(root, task, config, backend):
    import torch
    from transformers import Qwen2Tokenizer, Qwen3VLForConditionalGeneration, Qwen3VLModel

    for name in metadata_names(task):
        validate_metadata(name, metadata_file(root / name), config)
    common = {"local_files_only": True, "trust_remote_code": False}
    tokenizer = Qwen2Tokenizer.from_pretrained(root, **common)
    tokenizer.chat_template = None
    tokenizer.padding_side = "left"
    size = config["text_config"]["vocab_size"]
    if not 32 <= len(tokenizer) <= size or tokenizer.pad_token_id is None:
        raise ValueError("QWEN_RETRIEVAL_TOKENIZER_VOCABULARY")
    special_ids = []
    for token, field in (
        ("<|im_start|>", None),
        ("<|im_end|>", None),
        ("<|image_pad|>", "image_token_id"),
        ("<|video_pad|>", "video_token_id"),
        ("<|vision_start|>", "vision_start_token_id"),
        ("<|vision_end|>", "vision_end_token_id"),
    ):
        token_id = tokenizer.get_vocab().get(token)
        if (
            type(token_id) is not int
            or not 0 <= token_id < size
            or tokenizer.encode(token, add_special_tokens=False) != [token_id]
            or field
            and token_id != config[field]
        ):
            raise ValueError("QWEN_RETRIEVAL_SPECIAL_TOKENS")
        special_ids.append(token_id)
    if len(set(special_ids)) != len(special_ids):
        raise ValueError("QWEN_RETRIEVAL_SPECIAL_TOKENS")
    if task == "text-ranking":
        yes_no_ids(tokenizer)
    loader = Qwen3VLModel if task == "feature-extraction" else Qwen3VLForConditionalGeneration
    model, loading = loader.from_pretrained(
        root,
        **common,
        use_safetensors=True,
        dtype=torch.float32,
        attn_implementation="eager",
        output_loading_info=True,
    )
    if any(
        loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError("QWEN_RETRIEVAL_WEIGHTS_INCOMPLETE")
    return model.to(backend).eval(), tokenizer, config


def yes_no_ids(tokenizer):
    result = []
    for token in ("yes", "no"):
        token_id = tokenizer.get_vocab().get(token)
        if type(token_id) is not int or tokenizer.encode(token, add_special_tokens=False) != [
            token_id
        ]:
            raise ValueError("QWEN_RETRIEVAL_SCORE_TOKENS")
        result.append(token_id)
    if result[0] == result[1]:
        raise ValueError("QWEN_RETRIEVAL_SCORE_TOKENS")
    return result


def prompt(system, content):
    # Literal renderer for the reviewed text-only upstream conversation format.
    # Block special-token syntax so input text cannot request vision placeholders.
    if any("<|" in text or "|>" in text for text in (system, content)):
        raise ValueError("QWEN_RETRIEVAL_RESERVED_TOKEN")
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
    )


def tokenize(tokenizer, prompts, config, backend):
    tokens = tokenizer(
        prompts, padding=True, truncation=False, add_special_tokens=False, return_tensors="pt"
    )
    if tokens["input_ids"].shape[1] > min(
        MAX_TOKENS, config["text_config"]["max_position_embeddings"]
    ):
        raise ValueError("QWEN_RETRIEVAL_TOKEN_LIMIT")
    return {name: tokens[name].to(backend) for name in ("input_ids", "attention_mask")}


def execute(model, tokenizer, config, task, value, backend):
    import torch
    import torch.nn.functional as functional

    if task == "feature-extraction":
        if not 1 <= len(value["texts"]) <= MAX_BATCH:
            raise ValueError("QWEN_RETRIEVAL_BATCH_LIMIT")
        inputs = tokenize(
            tokenizer,
            [prompt(EMBEDDING_INSTRUCTION, text) for text in value["texts"]],
            config,
            backend,
        )
        hidden = model(**inputs, use_cache=False).last_hidden_state[:, -1].float()
        if not torch.isfinite(hidden).all() or torch.any(hidden.norm(dim=-1) <= 1e-12):
            raise ValueError("QWEN_RETRIEVAL_EMBEDDING_INVALID")
        if value["normalize"]:
            hidden = functional.normalize(hidden, dim=-1)
        return {
            "embeddings": hidden.cpu().tolist(),
            "dimensions": hidden.shape[1],
            "normalized": value["normalize"],
        }
    yes, no = yes_no_ids(tokenizer)
    weights = model.lm_head.weight[yes] - model.lm_head.weight[no]
    results = []
    for start in range(0, len(value["documents"]), MAX_BATCH):
        prompts = [
            prompt(
                RANKING_SYSTEM,
                "<Instruct>: "
                + value["instruction"]
                + "<Query>:"
                + value["query"]
                + "\n<Document>:"
                + document,
            )
            for document in value["documents"][start : start + MAX_BATCH]
        ]
        inputs = tokenize(tokenizer, prompts, config, backend)
        hidden = model.model(**inputs, use_cache=False).last_hidden_state[:, -1]
        scores = (hidden.float() @ weights.float()).sigmoid()
        if not torch.isfinite(scores).all():
            raise ValueError("QWEN_RETRIEVAL_SCORE_INVALID")
        results.extend(
            {"index": start + i, "score": score} for i, score in enumerate(scores.cpu().tolist())
        )
    results.sort(key=lambda item: (-item["score"], item["index"]))
    return {"results": results[: value.get("topK", len(results))]}
