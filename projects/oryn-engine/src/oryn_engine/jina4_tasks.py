"""Owned Jina v4 retrieval over native Qwen2.5-VL and all three original LoRA sets.

Reviewed publisher: jinaai/jina-embeddings-v4 at
853c867b65b749f3c3c72a06868140d842e04f06. Repository code is never imported.
Only the retrieval adapter's normalized 128-dimensional multi-vectors are exposed.
"""

from __future__ import annotations

import math
import re

from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

FAMILY = "jina_embeddings_v4"
TASK = "visual-document-retrieval"
ARCHITECTURE = "JinaEmbeddingsV4Model"
DIMENSIONS, PAGE_TOKENS, QUERY_TOKENS, VISUAL_TOKENS = 128, 256, 64, 196
TASKS = ("retrieval", "text-matching", "code")
TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
METADATA_NAMES = (
    "tokenizer_config.json",
    "preprocessor_config.json",
    "adapters/adapter_config.json",
)
ADAPTER_WEIGHT = "adapters/adapter_model.safetensors"
ALIASES = {
    "AutoConfig": "configuration_jina_embeddings_v4.JinaEmbeddingsV4Config",
    "AutoModel": "modeling_jina_embeddings_v4.JinaEmbeddingsV4Model",
}
PAGE_PROMPT = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|>\n"
)
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]
TEXT_FIELDS = {
    "attention_dropout",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "hidden_act",
    "hidden_size",
    "image_token_id",
    "initializer_range",
    "intermediate_size",
    "max_position_embeddings",
    "max_window_layers",
    "model_type",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "rms_norm_eps",
    "rope_scaling",
    "rope_theta",
    "sliding_window",
    "tie_word_embeddings",
    "torch_dtype",
    "use_cache",
    "use_sliding_window",
    "vocab_size",
}
VISION_FIELDS = {
    "depth",
    "fullatt_block_indexes",
    "hidden_act",
    "hidden_size",
    "in_channels",
    "in_chans",
    "initializer_range",
    "intermediate_size",
    "model_type",
    "num_heads",
    "out_hidden_size",
    "patch_size",
    "spatial_merge_size",
    "spatial_patch_size",
    "temporal_patch_size",
    "tokens_per_second",
    "torch_dtype",
    "window_size",
}
ROOT_FIELDS = TEXT_FIELDS | {
    "_name_or_path",
    "architectures",
    "text_config",
    "vision_config",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
    "vision_token_id",
    "task_names",
    "single_vector_pool_strategy",
    "multi_vector_projector_dim",
    "matryoshka_dims",
    "truncate_dim",
    "transformers_version",
}


def supports(config, task):
    return task == TASK and (
        config.get("model_type") == FAMILY or config.get("architectures") == [ARCHITECTURE]
    )


def normalized_config(config, task):
    if not supports(config, task):
        return config
    if (
        config.get("auto_map") not in (None, ALIASES)
        or config.get("_attn_implementation") not in (None, "flash_attention_2")
        or config.get("model_type", FAMILY) != FAMILY
    ):
        raise ValueError("JINA4_REMOTE_MAPPING")
    # The publisher explicitly falls back from this exact FA2 hint to SDPA.
    # Execution always selects the owned SDPA implementation.
    cleaned = {k: v for k, v in config.items() if k not in {"auto_map", "_attn_implementation"}}
    cleaned["model_type"] = FAMILY
    validate_config(cleaned)
    return cleaned


def validate_config(config):
    validate_model_config(config)
    text, vision = config.get("text_config"), config.get("vision_config")
    if (
        config.get("model_type") != FAMILY
        or config.get("architectures") != [ARCHITECTURE]
        or not set(config) <= ROOT_FIELDS
        or not isinstance(text, dict)
        or not isinstance(vision, dict)
        or not set(text) <= TEXT_FIELDS
        or not set(vision) <= VISION_FIELDS
        or text.get("model_type") != "qwen2_5_vl_text"
        or vision.get("model_type") != "qwen2_5_vl"
        or config.get("task_names") != list(TASKS)
        or config.get("single_vector_pool_strategy") != "mean"
        or type(config.get("multi_vector_projector_dim")) is not int
        or config["multi_vector_projector_dim"] != DIMENSIONS
        or config.get("truncate_dim") is not None
    ):
        raise ValueError("JINA4_CONFIGURATION")
    for part, fields in (
        (
            text,
            {
                "hidden_size": (8, 8192),
                "intermediate_size": (8, 65536),
                "num_hidden_layers": (1, 80),
                "num_attention_heads": (1, 64),
                "num_key_value_heads": (1, 64),
                "vocab_size": (32, 262144),
                "max_position_embeddings": (256, 1048576),
                "max_window_layers": (1, 80),
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
                "window_size": (4, 256),
            },
        ),
    ):
        for name, (low, high) in fields.items():
            if type(part.get(name)) is not int or not low <= part[name] <= high:
                raise ValueError("JINA4_GEOMETRY")
    if (
        text["hidden_size"] % text["num_attention_heads"]
        or text["num_attention_heads"] % text["num_key_value_heads"]
        or vision["hidden_size"] % vision["num_heads"]
        or vision["hidden_size"] // vision["num_heads"] % 4
        or vision["out_hidden_size"] != text["hidden_size"]
        or vision.get("in_channels") != 3
        or vision.get("in_chans") != 3
        or vision.get("spatial_merge_size") != 2
        or vision.get("temporal_patch_size") != 2
        or vision.get("spatial_patch_size") != vision["patch_size"]
        or vision["window_size"] % (vision["patch_size"] * 2)
        or vision.get("tokens_per_second") != 2
        or vision.get("hidden_act") != "silu"
        or text.get("hidden_act") != "silu"
        or text.get("attention_dropout") != 0.0
        or text.get("tie_word_embeddings") is not True
        or text.get("use_sliding_window") is not False
        or text.get("sliding_window") is not None
        or text.get("image_token_id") is not None
        or config.get("tie_word_embeddings") is not True
        or config.get("use_sliding_window") is not False
    ):
        raise ValueError("JINA4_LAYER_OPTIONS")
    # Older Qwen exports duplicate text options at the root. Conflicting options
    # must never choose different graphs in inspection and execution.
    for key in TEXT_FIELDS - {"model_type", "image_token_id", "sliding_window"}:
        if key in config and (key not in text or config[key] != text[key]):
            raise ValueError("JINA4_TEXT_CONFIGURATION_CONFLICT")
    for key in ("rope_theta", "rms_norm_eps", "initializer_range"):
        low, high = {
            "rope_theta": (1, 1e9),
            "rms_norm_eps": (1e-8, 1e-3),
            "initializer_range": (0, 1),
        }[key]
        if type(text.get(key)) not in {int, float} or not low <= text[key] <= high:
            raise ValueError("JINA4_TEXT_OPTIONS")
    if (
        type(config.get("sliding_window", 32768)) is not int
        or not 1 <= config.get("sliding_window", 32768) <= 1048576
    ):
        raise ValueError("JINA4_TEXT_OPTIONS")
    rope = text.get("rope_scaling")
    sections = rope.get("mrope_section") if isinstance(rope, dict) else None
    head = text["hidden_size"] // text["num_attention_heads"]
    if (
        not isinstance(rope, dict)
        or not set(rope) <= {"mrope_section", "rope_type", "type"}
        or rope.get("rope_type") != "default"
        or rope.get("type", "default") != "default"
        or not isinstance(sections, list)
        or len(sections) != 3
        or any(type(s) is not int or s <= 0 for s in sections)
        or head % 2
        or sum(sections) != head // 2
    ):
        raise ValueError("JINA4_ROPE")
    blocks = vision.get("fullatt_block_indexes")
    if (
        not isinstance(blocks, list)
        or not blocks
        or any(type(n) is not int or not 0 <= n < vision["depth"] for n in blocks)
        or blocks != sorted(set(blocks))
    ):
        raise ValueError("JINA4_VISION_BLOCKS")
    dims = config.get("matryoshka_dims")
    if (
        not isinstance(dims, list)
        or not dims
        or any(type(n) is not int or not 8 <= n <= text["hidden_size"] for n in dims)
        or dims != sorted(set(dims))
        or dims[-1] != text["hidden_size"]
    ):
        raise ValueError("JINA4_MATRYOSHKA")
    ids = [
        config.get(k)
        for k in (
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
            "vision_token_id",
        )
    ]
    if any(type(n) is not int or not 0 <= n < text["vocab_size"] for n in ids) or len(
        set(ids)
    ) != len(ids):
        raise ValueError("JINA4_TOKEN_IDS")
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if key in text and (type(text[key]) is not int or not 0 <= text[key] < text["vocab_size"]):
            raise ValueError("JINA4_TOKEN_IDS")


def validate_adapter(value):
    expected = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": "jinaai/jina-embeddings-v4",
        "bias": "none",
        "corda_config": None,
        "eva_config": None,
        "exclude_modules": ".*visual.*",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": "gaussian",
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": 32,
        "lora_bias": False,
        "lora_dropout": 0.1,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": 32,
        "rank_pattern": {},
        "revision": None,
        "target_modules": (
            "(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|"
            ".*(single_vector_projector|multi_vector_projector).*$)"
        ),
        "task_type": "FEATURE_EXTRACTION",
        "trainable_token_indices": None,
        "use_dora": False,
        "use_rslora": False,
    }
    # This expression is compared as inert data, never executed as a regex.
    if value != expected or any(type(value[k]) is not type(v) for k, v in expected.items()):
        raise ValueError("JINA4_ADAPTER_OPTIONS")


def validate_metadata(name, value, config):
    if name == "adapters/adapter_config.json":
        validate_adapter(value)
        return
    if name == "tokenizer_config.json":
        validate_model_config(value)
        allowed = {
            "tokenizer_class",
            "processor_class",
            "added_tokens_decoder",
            "additional_special_tokens",
            "chat_template",
            "model_max_length",
            "add_bos_token",
            "add_prefix_space",
            "bos_token",
            "eos_token",
            "pad_token",
            "unk_token",
            "clean_up_tokenization_spaces",
            "errors",
            "extra_special_tokens",
            "split_special_tokens",
        }
        expected = {
            "tokenizer_class": "Qwen2Tokenizer",
            "processor_class": "JinaEmbeddingsV4Processor",
            "pad_token": "<|endoftext|>",
            "eos_token": "<|im_end|>",
            "bos_token": None,
            "unk_token": None,
            "add_bos_token": False,
            "add_prefix_space": False,
            "clean_up_tokenization_spaces": False,
            "split_special_tokens": False,
            "errors": "replace",
            "extra_special_tokens": {},
        }
        if not set(value) <= allowed or any(value.get(k) != v for k, v in expected.items()):
            raise ValueError("JINA4_TOKENIZER_OPTIONS")
        return
    if name != "preprocessor_config.json":
        raise ValueError("JINA4_METADATA")
    vision = config["vision_config"]
    tile = vision["patch_size"] * 2
    minimum, maximum = value.get("min_pixels"), value.get("max_pixels")
    if (
        type(minimum) is not int
        or type(maximum) is not int
        or not tile**2 <= minimum <= VISUAL_TOKENS * tile**2 <= maximum <= 4096 * tile**2
    ):
        raise ValueError("JINA4_IMAGE_GEOMETRY")
    expected = {
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": MEAN,
        "image_std": STD,
        "image_processor_type": "Qwen2VLImageProcessor",
        "max_pixels": maximum,
        "min_pixels": minimum,
        "merge_size": 2,
        "patch_size": vision["patch_size"],
        "processor_class": "JinaEmbeddingsV4Processor",
        "resample": 3,
        "rescale_factor": 1 / 255,
        "video_processor_type": "Qwen2VLVideoProcessor",
        "size": {"longest_edge": maximum, "shortest_edge": minimum},
        "temporal_patch_size": 2,
        "auto_map": {"AutoProcessor": "modeling_jina_embeddings_v4.JinaEmbeddingsV4Processor"},
    }
    if value != expected or any(type(value[k]) is not type(v) for k, v in expected.items()):
        raise ValueError("JINA4_IMAGE_PROCESSOR")


def native_model(config):
    import copy

    import torch
    from torch import nn
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    validate_config(config)

    class TaskLinear(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base_layer = base
            self.lora_A = nn.ModuleDict(
                {k: nn.Linear(base.in_features, 32, bias=False) for k in TASKS}
            )
            self.lora_B = nn.ModuleDict(
                {k: nn.Linear(32, base.out_features, bias=False) for k in TASKS}
            )

        def forward(self, value):
            # All three sets remain required; this task binds the retrieval set.
            # Original alpha/r = 1 and evaluation disables dropout.
            return self.base_layer(value) + self.lora_B["retrieval"](
                self.lora_A["retrieval"](value)
            )

    class Jina4(nn.Module):
        def __init__(self):
            super().__init__()
            options = Qwen2_5_VLConfig(
                text_config=copy.deepcopy(config["text_config"]) | {"torch_dtype": "float32"},
                vision_config=copy.deepcopy(config["vision_config"]) | {"torch_dtype": "float32"},
                tie_word_embeddings=True,
                **{
                    k: config[k]
                    for k in (
                        "image_token_id",
                        "video_token_id",
                        "vision_start_token_id",
                        "vision_end_token_id",
                    )
                },
            )
            for part in (options, options.text_config, options.vision_config):
                part._attn_implementation = "sdpa"
            self.vlm = Qwen2_5_VLForConditionalGeneration(options)
            for layer in self.vlm.model.language_model.layers:
                for path in TARGETS:
                    parent, leaf = path.split(".")
                    owner = getattr(layer, parent)
                    setattr(owner, leaf, TaskLinear(getattr(owner, leaf)))
            self.multi_vector_projector = TaskLinear(
                nn.Linear(config["text_config"]["hidden_size"], DIMENSIONS)
            )

        def forward(self, **inputs):
            self.vlm.model.rope_deltas = None
            if "image_grid_thw" in inputs:
                inputs["mm_token_type_ids"] = (
                    inputs["input_ids"] == config["image_token_id"]
                ).long()
            else:
                positions = inputs["attention_mask"].long().cumsum(-1) - 1
                positions.masked_fill_(inputs["attention_mask"] == 0, 1)
                inputs["position_ids"] = positions.unsqueeze(0).expand(3, -1, -1)
            hidden = self.vlm.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
            projected = self.multi_vector_projector(hidden)
            norms = projected.norm(dim=-1, keepdim=True)
            if not bool(torch.isfinite(projected).all()) or bool((norms <= 1e-12).any()):
                raise ValueError("JINA4_EMBEDDINGS_INVALID")
            return projected / norms * inputs["attention_mask"].unsqueeze(-1)

    return Jina4()


def tensor_name(name, *, adapter=False):
    if adapter:
        if not name.startswith("base_model.model."):
            raise ValueError("JINA4_WEIGHTS_NAME")
        name = name.removeprefix("base_model.model.")
        if name.startswith("model.language_model."):
            return "vlm." + name
        if name.startswith("multi_vector_projector."):
            return name
        raise ValueError("JINA4_WEIGHTS_NAME")
    if name.startswith("model."):
        name = "vlm.model.language_model." + name.removeprefix("model.")
    elif name.startswith("visual."):
        name = "vlm.model.visual." + name.removeprefix("visual.")
    elif not name.startswith("multi_vector_projector."):
        raise ValueError("JINA4_WEIGHTS_NAME")
    if (
        name.startswith("vlm.model.language_model.")
        and re.search(r"\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.(weight|bias)$", name)
        or name.startswith("multi_vector_projector.")
    ):
        stem, leaf = name.rsplit(".", 1)
        name = stem + ".base_layer." + leaf
    return name


def bind_tensors(base, adapters):
    values = {}
    for tensors, adapter in ((base, False), (adapters, True)):
        for name, value in tensors.items():
            key = tensor_name(name, adapter=adapter)
            if key in values:
                raise ValueError("JINA4_WEIGHTS_DUPLICATE")
            values[key] = value
    source = "vlm.model.language_model.embed_tokens.weight"
    if source in values:
        values["vlm.lm_head.weight"] = values[source]
    return values


def validate_weight_shapes(config, base, adapters):
    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        model = native_model(config)
    if bind_tensors(base, adapters) != {k: tuple(v.shape) for k, v in model.state_dict().items()}:
        raise ValueError("JINA4_WEIGHTS_INCOMPLETE")
    return model


def processor(root, config):
    from types import SimpleNamespace

    from transformers import Qwen2Tokenizer, Qwen2VLImageProcessor

    tokenizer = Qwen2Tokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    tokenizer.chat_template = None
    tokenizer.padding_side = "left"
    vocab = tokenizer.get_vocab()
    gap = config["text_config"]["vocab_size"] - len(vocab)
    if not 0 <= gap < 512 or set(vocab.values()) != set(range(len(vocab))):
        raise ValueError("JINA4_VOCABULARY")
    for token, field in (
        ("<|image_pad|>", "image_token_id"),
        ("<|video_pad|>", "video_token_id"),
        ("<|vision_start|>", "vision_start_token_id"),
        ("<|vision_end|>", "vision_end_token_id"),
        ("<|vision_pad|>", "vision_token_id"),
        ("<|im_start|>", None),
        ("<|im_end|>", None),
        ("<|endoftext|>", None),
    ):
        token_id = vocab.get(token)
        if (
            type(token_id) is not int
            or tokenizer.encode(token, add_special_tokens=False) != [token_id]
            or field
            and token_id != config[field]
        ):
            raise ValueError("JINA4_TOKENIZER_FRAMING")
    image = metadata_file(root / "preprocessor_config.json")
    patch = config["vision_config"]["patch_size"]
    return SimpleNamespace(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(
            patch_size=patch,
            merge_size=2,
            temporal_patch_size=2,
            size={
                "shortest_edge": image["min_pixels"],
                "longest_edge": VISUAL_TOKENS * (patch * 2) ** 2,
            },
            image_mean=MEAN,
            image_std=STD,
        ),
    )


def load_components(root, config, backend):
    import torch
    from safetensors.torch import load_file

    config = normalized_config(config, TASK)
    adapter_root = root / "adapters"
    if adapter_root.is_symlink() or not adapter_root.is_dir():
        raise ValueError("JINA4_ADAPTER_PATH")
    for name in METADATA_NAMES:
        validate_metadata(name, metadata_file(root / name), config)
    if {p.name for p in adapter_root.glob("*.safetensors")} != {"adapter_model.safetensors"}:
        raise ValueError("JINA4_ADAPTER_WEIGHT_SET")
    model = validate_weight_shapes(
        config, preflight_safetensors(root), preflight_safetensors(adapter_root)
    )
    native_processor = processor(root, config)

    def weights(directory):
        result = {}
        for path in sorted(directory.glob("*.safetensors")):
            values = load_file(path)
            if result.keys() & values.keys():
                raise ValueError("JINA4_WEIGHTS_DUPLICATE")
            for key, value in values.items():
                if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                    raise ValueError("JINA4_WEIGHTS_VALUES")
                result[key] = value.float()
                if not bool(torch.isfinite(result[key]).all()):
                    raise ValueError("JINA4_WEIGHTS_VALUES")
        return result

    model.load_state_dict(
        bind_tensors(weights(root), weights(adapter_root)), strict=True, assign=True
    )
    return model.to(backend).eval(), native_processor, config


def retrieval_inputs(processor, config, image, queries):
    if not 1 <= len(queries) <= 2 or any("<|" in q or "|>" in q for q in queries):
        raise ValueError("JINA4_QUERY_FRAMING")
    pixels = dict(processor.image_processor(images=[image.convert("RGB")], return_tensors="pt"))
    count = math.prod(pixels["image_grid_thw"][0].tolist()) // 4
    if not 1 <= count <= VISUAL_TOKENS:
        raise ValueError("JINA4_IMAGE_TOKEN_LIMIT")
    prompt = PAGE_PROMPT.replace("<|image_pad|>", "<|image_pad|>" * count)
    page = (
        dict(processor.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")) | pixels
    )
    texts = [
        dict(processor.tokenizer(["Query: " + q], add_special_tokens=False, return_tensors="pt"))
        for q in queries
    ]
    if page["input_ids"].shape[1] > PAGE_TOKENS or any(
        x["input_ids"].shape[1] > QUERY_TOKENS for x in texts
    ):
        raise ValueError("JINA4_TOKEN_LIMIT")
    return page, texts


def retrieve(model, processor, config, image, value, backend):
    from types import SimpleNamespace

    from oryn_engine.shared_tasks import _move
    from oryn_engine.visual_retrieval import late_interaction, vectors

    page, texts = retrieval_inputs(processor, config, image, value["queries"])

    def encode(inputs, limit):
        inputs = _move(inputs, backend)
        return vectors(
            SimpleNamespace(embeddings=model(**inputs)),
            inputs["attention_mask"],
            limit,
            dimensions=DIMENSIONS,
        )

    document = encode(page, PAGE_TOKENS)
    queries = []
    for inputs in texts:
        query = encode(inputs, QUERY_TOKENS)
        queries.append({"vectors": query, "score": late_interaction(query, document)})
    return {
        "dimensions": DIMENSIONS,
        "normalized": True,
        "scoring": "sum-max-cosine",
        "documentVectors": document,
        "queries": queries,
    }
