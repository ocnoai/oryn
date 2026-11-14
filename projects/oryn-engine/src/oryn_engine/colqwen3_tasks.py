"""Owned ColQwen3 projection and page/query binding over the complete Qwen3-VL graph.

Reference: TomoroAI/tomoro-colqwen3-embed-4b at
13517a29e8c5e408f7f2684337ed407df3acb212 (Apache-2.0). Repository Python is inert.
"""

from __future__ import annotations

from oryn_engine.model_metadata import metadata_file, validate_model_config

ARCHITECTURE = "ColQwen3"
DIMENSIONS = 320
PAGE_TOKENS = 144
QUERY_TOKENS = 48
VISUAL_TOKENS = 128
METADATA_NAMES = ("tokenizer_config.json", "preprocessor_config.json", "processor_config.json")
PAGE_PROMPT = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "Describe the image.<|im_end|><|endoftext|>"
)
ALIASES = {
    "AutoConfig": "configuration_colqwen3.ColQwen3Config",
    "AutoModel": "modeling_colqwen3.ColQwen3",
}


def supports(config, task):
    return task == "visual-document-retrieval" and config.get("model_type") == "colqwen3"


def native_config(config):
    from oryn_engine.qwen_retrieval import validate_geometry

    if config.get("auto_map") not in (None, ALIASES):
        raise ValueError("COLQWEN3_REMOTE_MAPPING")
    cleaned = {k: v for k, v in config.items() if k != "auto_map"}
    validate_model_config(cleaned)
    root_fields = {
        "model_type",
        "architectures",
        "embed_dim",
        "padding_side",
        "initializer_range",
        "dtype",
        "torch_dtype",
        "transformers_version",
        "max_num_visual_tokens",
        "text_config",
        "vision_config",
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
    }
    text, vision = cleaned.get("text_config"), cleaned.get("vision_config")
    if (
        cleaned.get("model_type") != "colqwen3"
        or not set(cleaned) <= root_fields
        or cleaned.get("architectures") != [ARCHITECTURE]
        or cleaned.get("embed_dim") != DIMENSIONS
        or cleaned.get("padding_side", "left") != "left"
        or cleaned.get("quantization_config")
        or cleaned.get("quantization")
        or not isinstance(text, dict)
        or not isinstance(vision, dict)
        or text.get("model_type") != "qwen3_vl_text"
        or vision.get("model_type") not in {"qwen3_vl", "qwen3_vl_vision"}
        or text.get("is_causal", True) is not True
        or text.get("tie_word_embeddings") is not True
        or text.get("attention_bias", False) is not False
        or text.get("attention_dropout", 0.0) != 0.0
        or text.get("hidden_act", "silu") != "silu"
        or vision.get("hidden_act", "gelu_pytorch_tanh") != "gelu_pytorch_tanh"
        or type(cleaned.get("max_num_visual_tokens", 1280)) is not int
        or not VISUAL_TOKENS <= cleaned.get("max_num_visual_tokens", 1280) <= 4096
    ):
        raise ValueError("COLQWEN3_CONFIGURATION_UNSUPPORTED")
    text_fields = {
        "attention_bias",
        "attention_dropout",
        "bos_token_id",
        "dtype",
        "eos_token_id",
        "head_dim",
        "hidden_act",
        "hidden_size",
        "initializer_range",
        "intermediate_size",
        "max_position_embeddings",
        "model_type",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_scaling",
        "rope_parameters",
        "rope_theta",
        "tie_word_embeddings",
        "use_cache",
        "vocab_size",
        "is_causal",
        "pad_token_id",
        "torch_dtype",
    }
    vision_fields = {
        "deepstack_visual_indexes",
        "depth",
        "dtype",
        "hidden_act",
        "hidden_size",
        "in_channels",
        "initializer_range",
        "intermediate_size",
        "model_type",
        "num_heads",
        "num_position_embeddings",
        "out_hidden_size",
        "patch_size",
        "spatial_merge_size",
        "temporal_patch_size",
        "torch_dtype",
    }
    if not set(text) <= text_fields or not set(vision) <= vision_fields:
        raise ValueError("COLQWEN3_LAYER_OPTIONS_UNSUPPORTED")
    epsilon = text.get("rms_norm_eps", 1e-6)
    if type(epsilon) not in {int, float} or not 1e-8 <= epsilon <= 1e-3:
        raise ValueError("COLQWEN3_NORMALIZATION_UNSUPPORTED")
    validate_geometry(cleaned)
    return cleaned


def normalized_config(config, task):
    return native_config(config) if supports(config, task) else config


def validate_metadata(name, value, config, *, maximum_declared_tokens=4096):
    """Review the exact processor alias without importing or evaluating it."""
    allowed_alias = {"AutoProcessor": "processing_colqwen3.ColQwen3Processor"}
    if value.get("auto_map") not in (None, allowed_alias):
        raise ValueError("COLQWEN3_PROCESSOR_MAPPING")
    value = {k: v for k, v in value.items() if k != "auto_map"}
    if name == "tokenizer_config.json":
        validate_model_config(value)
        if value.get("tokenizer_class") not in {"Qwen2Tokenizer", "Qwen2TokenizerFast"}:
            raise ValueError("COLQWEN3_TOKENIZER_CLASS")
        if value.get("pad_token") != "<|endoftext|>" or value.get("add_bos_token", False):
            raise ValueError("COLQWEN3_TOKENIZER_FRAMING")
    elif name == "processor_config.json":
        validate_model_config(value)
        prefix, suffix = PAGE_PROMPT.split("<|im_end|>")
        expected = {
            "processor_class": "ColQwen3Processor",
            "query_prefix": "",
            "visual_prompt_prefix": prefix,
            "visual_prompt_suffix": "<|im_end|>" + suffix,
            "video_prompt_prefix": (
                "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>Describe the video."
            ),
            "video_prompt_suffix": "<|im_end|><|endoftext|>",
        }
        if value != expected:
            raise ValueError("COLQWEN3_PROCESSOR_FRAMING")
    elif name == "preprocessor_config.json":
        # Published pixel caps can exceed the worker's deliberately smaller
        # window. Validate their geometry, then use an owned bounded window.
        vision = config["vision_config"]
        tile = vision["patch_size"] * vision["spatial_merge_size"]
        size = value.get("size")
        if (
            value.get("image_processor_type")
            not in {"Qwen2VLImageProcessor", "Qwen2VLImageProcessorFast"}
            or value.get("processor_class") != "ColQwen3Processor"
            or not isinstance(size, dict)
            or set(size) != {"shortest_edge", "longest_edge"}
            or any(
                type(v) is not int or not tile**2 <= v <= maximum_declared_tokens * tile**2
                for v in size.values()
            )
            or size["shortest_edge"] > VISUAL_TOKENS * tile**2
            or size["longest_edge"] < VISUAL_TOKENS * tile**2
            or value.get("max_pixels", size["longest_edge"]) != size["longest_edge"]
            or value.get("min_pixels") not in {None, size["shortest_edge"]}
        ):
            raise ValueError("COLQWEN3_IMAGE_GEOMETRY")
        for key, expected in {
            "patch_size": vision["patch_size"],
            "merge_size": vision["spatial_merge_size"],
            "temporal_patch_size": vision["temporal_patch_size"],
            "do_convert_rgb": True,
            "do_normalize": True,
            "do_resize": True,
            "do_rescale": True,
            "image_mean": [0.5] * 3,
            "image_std": [0.5] * 3,
            "rescale_factor": 1 / 255,
            "resample": 3,
        }.items():
            if value.get(key) != expected:
                raise ValueError("COLQWEN3_IMAGE_PREPROCESSING")
        for key in ("do_pad", "do_center_crop", "crop_size", "pad_size", "device"):
            if value.get(key) is not None:
                raise ValueError("COLQWEN3_IMAGE_PREPROCESSING")
        if value.get("data_format", "channels_first") != "channels_first" or any(
            value.get(k) is not None for k in ("input_data_format", "return_tensors")
        ):
            raise ValueError("COLQWEN3_IMAGE_PREPROCESSING")
        validate_model_config(
            {k: v for k, v in value.items() if k not in {"size", "max_pixels", "min_pixels"}}
        )
    else:
        raise ValueError("COLQWEN3_METADATA_UNSUPPORTED")


def native_model(config):
    from torch import nn
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = native_config(config)
    options = Qwen3VLConfig(
        text_config=config["text_config"],
        vision_config=config["vision_config"],
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
    options._attn_implementation = "eager"
    options.text_config._attn_implementation = "eager"
    options.vision_config._attn_implementation = "eager"

    class ColQwen3(nn.Module):
        def __init__(self):
            super().__init__()
            self.vlm = Qwen3VLForConditionalGeneration(options)
            self.embedding_proj_layer = nn.Linear(options.text_config.hidden_size, DIMENSIONS)

        def forward(self, **inputs):
            import torch

            hidden = self.vlm.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
            embeddings = self.embedding_proj_layer(hidden)
            norms = embeddings.norm(dim=-1, keepdim=True)
            if not bool(torch.isfinite(embeddings).all()) or bool((norms <= 1e-12).any()):
                raise ValueError("COLQWEN3_EMBEDDINGS_INVALID")
            return embeddings / norms

    return ColQwen3()


def validate_weight_shapes(config, shapes):
    import torch

    with torch.device("meta"):
        model = native_model(config)
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    alias, source = "vlm.lm_head.weight", "vlm.model.language_model.embed_tokens.weight"
    shapes = dict(shapes)
    if alias not in shapes and source in shapes:
        shapes[alias] = shapes[source]
    if shapes != expected:
        raise ValueError("COLQWEN3_WEIGHTS_INCOMPLETE")


def load_components(root, config, backend):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import (
        Qwen2Tokenizer,
        Qwen2VLImageProcessor,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    from oryn_engine.model_metadata import preflight_safetensors

    config = native_config(config)
    for name in METADATA_NAMES:
        validate_metadata(name, metadata_file(root / name), config)
    validate_weight_shapes(config, preflight_safetensors(root))
    with init_empty_weights(include_buffers=False):
        model = native_model(config)
    weights = {}
    for path in sorted(root.glob("*.safetensors")):
        state = load_file(path)
        if weights.keys() & state.keys() or any(
            not v.is_floating_point() or not bool(torch.isfinite(v).all()) for v in state.values()
        ):
            raise ValueError("COLQWEN3_WEIGHTS_INVALID")
        weights.update({k: v.float() for k, v in state.items()})
    alias, source = "vlm.lm_head.weight", "vlm.model.language_model.embed_tokens.weight"
    if alias in weights and not torch.equal(weights[alias], weights[source]):
        raise ValueError("COLQWEN3_TIED_WEIGHTS_INVALID")
    weights[alias] = weights[source]
    model.load_state_dict(weights, strict=True, assign=True)
    tokenizer = Qwen2Tokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    tokenizer.chat_template = None
    tokenizer.padding_side = "left"
    size = config["text_config"]["vocab_size"]
    vocab = tokenizer.get_vocab()
    if not 32 <= len(vocab) <= size or set(vocab.values()) != set(range(len(vocab))):
        raise ValueError("COLQWEN3_TOKENIZER_VOCABULARY")
    for token, field in (
        ("<|image_pad|>", "image_token_id"),
        ("<|video_pad|>", "video_token_id"),
        ("<|vision_start|>", "vision_start_token_id"),
        ("<|vision_end|>", "vision_end_token_id"),
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
            raise ValueError("COLQWEN3_TOKENIZER_FRAMING")
    image_config = metadata_file(root / "preprocessor_config.json")
    vision = config["vision_config"]
    tile = vision["patch_size"] * vision["spatial_merge_size"]
    image_processor = Qwen2VLImageProcessor(
        patch_size=vision["patch_size"],
        merge_size=vision["spatial_merge_size"],
        temporal_patch_size=vision["temporal_patch_size"],
        size={
            "shortest_edge": image_config["size"]["shortest_edge"],
            "longest_edge": VISUAL_TOKENS * tile**2,
        },
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    )
    # The native processor constructor requires this inert companion even for
    # image-only requests. The owned invocation never accepts a video input.
    processor = Qwen3VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=Qwen3VLVideoProcessor(),
    )
    return model.to(backend).eval(), processor, config


def retrieve(
    model,
    processor,
    config,
    image,
    value,
    backend,
    *,
    dimensions=DIMENSIONS,
    preserve_query_whitespace=False,
):
    from types import SimpleNamespace

    from oryn_engine.shared_tasks import _move
    from oryn_engine.visual_retrieval import late_interaction, vectors

    queries = value["queries"]
    if not 1 <= len(queries) <= 2 or any("<|" in q or "|>" in q for q in queries):
        raise ValueError("COLQWEN3_QUERY_FRAMING")
    page = dict(
        processor(
            images=[image.convert("RGB")],
            text=[PAGE_PROMPT],
            return_tensors="pt",
            return_mm_token_type_ids=True,
        )
    )
    text_inputs = [
        dict(
            processor.tokenizer(
                [(q if preserve_query_whitespace else q.strip()) + "<|endoftext|>" * 10],
                return_tensors="pt",
                add_special_tokens=False,
            )
        )
        for q in queries
    ]
    if page["input_ids"].shape[1] > PAGE_TOKENS or any(
        inputs["input_ids"].shape[1] > QUERY_TOKENS for inputs in text_inputs
    ):
        raise ValueError("COLQWEN3_TOKEN_LIMIT")
    page = _move(page, backend)
    document = vectors(
        SimpleNamespace(embeddings=model(**page)),
        page["attention_mask"],
        PAGE_TOKENS,
        dimensions=dimensions,
    )
    results = []
    for inputs in text_inputs:
        inputs = _move(inputs, backend)
        query = vectors(
            SimpleNamespace(embeddings=model(**inputs)),
            inputs["attention_mask"],
            QUERY_TOKENS,
            dimensions=dimensions,
        )
        results.append({"vectors": query, "score": late_interaction(query, document)})
    return {
        "dimensions": dimensions,
        "normalized": True,
        "scoring": "sum-max-cosine",
        "documentVectors": document,
        "queries": results,
    }
