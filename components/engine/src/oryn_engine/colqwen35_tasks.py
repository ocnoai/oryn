"""Complete hybrid ColQwen3.5 graphs with bounded native late interaction.

Reviewed ColPali graph/processor: tencent/EVIE-4.5B at
b732dca38a4a6f89696a5822a72b01c20295dd8a (Apache-2.0).
Single-head configuration: athrael-soju/colqwen3.5-4.5B-v3 at
c4ea4b92ff3322d68bfd6c65e492c69d0176f910. Repository code is never imported.
"""

from __future__ import annotations

from oryn_engine import colqwen3_tasks, qwen_translation
from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

METADATA_NAMES = ("tokenizer_config.json", "processor_config.json")
PAGE_TOKENS = colqwen3_tasks.PAGE_TOKENS
QUERY_TOKENS = colqwen3_tasks.QUERY_TOKENS
VISUAL_TOKENS = colqwen3_tasks.VISUAL_TOKENS


def supports(config, task):
    return (
        task == "visual-document-retrieval"
        and config.get("model_type") == "qwen3_5"
        and config.get("architectures") == ["ColQwen3_5"]
    )


def dimensions(config):
    # Prefix-MRL checkpoints explicitly support this bounded width. The entire
    # wider learned projection remains mandatory and is computed before slicing.
    return 128 if config.get("mrl_prefix") is True else config["dim"]


def graph_config(config):
    return config | {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "is_causal": False,
        "text_config": config["text_config"] | {"is_causal": False, "use_cache": False},
    }


def validate_config(config):
    validate_model_config(config)
    if (
        not supports(config, "visual-document-retrieval")
        or config.get("retrieval_attention_contract", "bidirectional") != "bidirectional"
        or config.get("is_causal", False) is not False
        or not isinstance(config.get("text_config"), dict)
        or config["text_config"].get("is_causal", False) is not False
        or config.get("mask_non_image_embeddings", False) is not False
        or config.get("padding_side", "left") != "left"
    ):
        raise ValueError("COLQWEN35_CONFIGURATION")
    qwen_translation.validate_config(graph_config(config))
    width = config.get("dim")
    heads = config.get("head_dims")
    if type(width) is not int or width > config["text_config"]["hidden_size"]:
        raise ValueError("COLQWEN35_PROJECTION")
    if heads is None:
        if (
            width not in {320, 4096}
            or config.get("embed_dim", width) != width
            or config.get("mrl_prefix", False) is not False
        ):
            raise ValueError("COLQWEN35_PROJECTION")
    elif (
        config.get("mrl_prefix") is not True
        or not isinstance(heads, list)
        or not heads
        or any(
            type(n) is not int or n not in {16, 32, 64, 128, 256, 512, 1024, 2048} for n in heads
        )
        or heads != sorted(set(heads))
        or 128 not in heads
        or max(heads) != width
        or config.get("anchor_dim") != 128
        or config.get("embed_dim") != 128
    ):
        raise ValueError("COLQWEN35_PROJECTION")


def validate_metadata(name, value, config):
    from oryn_engine.shared_tasks import validate_processor_config

    validate_processor_config(value, "qwen3_5")
    if name == "tokenizer_config.json":
        if (
            value.get("tokenizer_class") not in {"Qwen2Tokenizer", "Qwen2TokenizerFast"}
            or value.get("pad_token") != "<|endoftext|>"
            or value.get("eos_token") != "<|im_end|>"
            or value.get("add_bos_token", False)
        ):
            raise ValueError("COLQWEN35_TOKENIZER")
    elif name == "processor_config.json":
        if (
            value.get("processor_class") != "Qwen3VLProcessor"
            or not set(value) <= {"processor_class", "image_processor", "video_processor"}
            or not isinstance(value.get("image_processor"), dict)
            or not set(value["image_processor"])
            <= {
                "data_format",
                "do_convert_rgb",
                "do_normalize",
                "do_rescale",
                "do_resize",
                "image_mean",
                "image_processor_type",
                "image_std",
                "merge_size",
                "patch_size",
                "resample",
                "rescale_factor",
                "size",
                "temporal_patch_size",
            }
        ):
            raise ValueError("COLQWEN35_PROCESSOR")
        # The reviewed image pipeline is identical to native ColQwen3. Video is
        # never accepted by this contract and its configuration is not instantiated.
        image = value["image_processor"] | {"processor_class": "ColQwen3Processor"}
        # The wide native export declares a 16 MiPixel source ceiling. The
        # actual processor below still uses the profile's 128 visual-token cap.
        colqwen3_tasks.validate_metadata(
            "preprocessor_config.json",
            image,
            config,
            maximum_declared_tokens=16384 if dimensions(config) == 4096 else 4096,
        )
    else:
        raise ValueError("COLQWEN35_METADATA")


def native_model(config):
    import torch
    from transformers import Qwen3_5Model

    validate_config(config)

    class ColQwen35(Qwen3_5Model):
        def __init__(self):
            super().__init__(qwen_translation.native_config(graph_config(config)))
            self.custom_text_proj = torch.nn.Linear(
                config["text_config"]["hidden_size"], config["dim"]
            )
            for layer in self.language_model.layers:
                if hasattr(layer, "self_attn"):
                    layer.self_attn.is_causal = False

        def forward(self, **inputs):
            self.rope_deltas = None
            hidden = super().forward(**inputs, use_cache=False, return_dict=True).last_hidden_state
            projected = self.custom_text_proj(hidden)[..., : dimensions(config)]
            norm = projected.norm(dim=-1, keepdim=True)
            if not bool(torch.isfinite(projected).all()) or bool((norm <= 1e-12).any()):
                raise ValueError("COLQWEN35_EMBEDDINGS_INVALID")
            return projected / norm * inputs["attention_mask"].unsqueeze(-1)

    return ColQwen35()


def validate_weight_shapes(config, shapes):
    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        model = native_model(config)
    if shapes != {k: tuple(v.shape) for k, v in model.state_dict().items()}:
        raise ValueError("COLQWEN35_WEIGHTS_INCOMPLETE")
    return model


def processor(root, config):
    from transformers import (
        Qwen2VLImageProcessor,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
        TokenizersBackend,
    )

    for name in METADATA_NAMES:
        validate_metadata(name, metadata_file(root / name), config)
    tokenizer = TokenizersBackend(tokenizer_file=str(root / "tokenizer.json"))
    vocab = tokenizer.get_vocab()
    gap = config["text_config"]["vocab_size"] - len(vocab)
    if (
        not 0 <= gap < 256
        or gap
        and config["text_config"]["vocab_size"] % 256
        or set(vocab.values()) != set(range(len(vocab)))
    ):
        raise ValueError("COLQWEN35_VOCABULARY")
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
            raise ValueError("COLQWEN35_TOKENIZER_FRAMING")
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.padding_side = "left"
    tokenizer.chat_template = None
    image = metadata_file(root / "processor_config.json")["image_processor"]
    vision = config["vision_config"]
    image_processor = Qwen2VLImageProcessor(
        patch_size=vision["patch_size"],
        merge_size=vision["spatial_merge_size"],
        temporal_patch_size=vision["temporal_patch_size"],
        size={
            "shortest_edge": image["size"]["shortest_edge"],
            "longest_edge": VISUAL_TOKENS
            * (vision["patch_size"] * vision["spatial_merge_size"]) ** 2,
        },
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    )
    return Qwen3VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=Qwen3VLVideoProcessor(),
        chat_template=None,
    )


def load_components(root, config, backend):
    import torch
    from safetensors.torch import load_file

    model = validate_weight_shapes(config, preflight_safetensors(root))
    native_processor = processor(root, config)
    weights = {}
    for path in sorted(root.glob("*.safetensors")):
        values = load_file(path)
        if weights.keys() & values.keys() or any(
            not v.is_floating_point() or not bool(torch.isfinite(v).all()) for v in values.values()
        ):
            raise ValueError("COLQWEN35_WEIGHT_VALUES")
        weights.update({k: v.float() for k, v in values.items()})
    model.load_state_dict(weights, strict=True, assign=True)
    return model.to(backend).eval(), native_processor, config


def retrieve(model, processor, config, image, value, backend):
    return colqwen3_tasks.retrieve(
        model,
        processor,
        config,
        image,
        value,
        backend,
        dimensions=dimensions(config),
        preserve_query_whitespace=True,
    )
