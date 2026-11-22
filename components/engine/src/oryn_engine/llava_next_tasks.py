"""Native Granite/SigLIP LLaVA-NeXT images, fixed framing and complete numeric graphs."""

from __future__ import annotations

from copy import deepcopy

from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

METADATA_NAMES = ("preprocessor_config.json", "processor_config.json", "tokenizer_config.json")
MAX_CONTEXT = 16384
MAX_OUTPUT = 256
SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
MARKERS = ("<|system|>", "<|user|>", "<|assistant|>", "<image>", "<|end_of_text|>")


def normalized_config(config, task):
    if (
        task not in {"image-to-text", "visual-question-answering"}
        or config.get("model_type") != "llava_next"
        or not isinstance(config.get("text_config"), dict)
        or config["text_config"].get("model_type") != "granite"
        or not isinstance(config.get("vision_config"), dict)
        or config["vision_config"].get("model_type") != "siglip_vision_model"
    ):
        return config
    result = deepcopy(config)
    # Transformers 4 wrote this boolean bookkeeping field. It does not select an
    # attention implementation; no active implementation option is normalized.
    for part in (result, result.get("text_config", {}), result.get("vision_config", {})):
        if isinstance(part, dict) and type(part.get("_attn_implementation_autoset")) is bool:
            del part["_attn_implementation_autoset"]
    validate_config(result)
    return result


def validate_config(config):
    validate_model_config(config)
    text, vision = config.get("text_config"), config.get("vision_config")
    if (
        config.get("model_type") != "llava_next"
        or config.get("architectures") != ["LlavaNextForConditionalGeneration"]
        or not isinstance(text, dict)
        or not isinstance(vision, dict)
        or text.get("model_type") != "granite"
        or vision.get("model_type") != "siglip_vision_model"
        or config.get("vision_feature_select_strategy") != "full"
        or config.get("projector_hidden_act", "gelu") != "gelu"
        or config.get("multimodal_projector_bias", True) is not True
        or config.get("use_image_newline_parameter", True) is not True
        or vision.get("num_channels", 3) != 3
        or any(
            p.get("quantization_config") or p.get("quantization") for p in (config, text, vision)
        )
    ):
        raise ValueError("LLAVA_NEXT_CONFIGURATION_UNSUPPORTED")
    for part, bounds in (
        (
            text,
            {
                "hidden_size": (8, 8192),
                "intermediate_size": (8, 65536),
                "num_hidden_layers": (1, 80),
                "num_attention_heads": (1, 64),
                "num_key_value_heads": (1, 64),
                "vocab_size": (32, 262144),
                "max_position_embeddings": (128, 262144),
            },
        ),
        (
            vision,
            {
                "hidden_size": (8, 2048),
                "intermediate_size": (8, 16384),
                "num_hidden_layers": (1, 48),
                "num_attention_heads": (1, 64),
                "image_size": (14, 512),
                "patch_size": (1, 32),
            },
        ),
    ):
        for key, (low, high) in bounds.items():
            if type(part.get(key)) is not int or not low <= part[key] <= high:
                raise ValueError("LLAVA_NEXT_GEOMETRY")
        if part["hidden_size"] % part["num_attention_heads"]:
            raise ValueError("LLAVA_NEXT_GEOMETRY")
    if text["num_attention_heads"] % text["num_key_value_heads"]:
        raise ValueError("LLAVA_NEXT_GEOMETRY")
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if type(text.get(key)) is not int or not 0 <= text[key] < text["vocab_size"]:
            raise ValueError("LLAVA_NEXT_SPECIAL_TOKENS")
    if len({text[k] for k in ("bos_token_id", "eos_token_id", "pad_token_id")}) != 1:
        raise ValueError("LLAVA_NEXT_SPECIAL_TOKENS")
    if (
        type(config.get("image_token_index")) is not int
        or not 0 <= config["image_token_index"] < text["vocab_size"]
        or config["image_token_index"] == text["eos_token_id"]
    ):
        raise ValueError("LLAVA_NEXT_SPECIAL_TOKENS")
    selected = config.get("vision_feature_layer")
    if (
        not isinstance(selected, list)
        or not 1 <= len(selected) <= 4
        or any(
            type(i) is not int
            or not -vision["num_hidden_layers"] - 1 <= i <= vision["num_hidden_layers"]
            for i in selected
        )
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("LLAVA_NEXT_FEATURE_LAYERS")
    grids = config.get("image_grid_pinpoints")
    size, patch = vision["image_size"], vision["patch_size"]
    if (
        not isinstance(grids, list)
        or not 1 <= len(grids) <= 32
        or any(
            not isinstance(g, list)
            or len(g) != 2
            or any(type(a) is not int or not size <= a <= size * 10 or a % size for a in g)
            or g[0] * g[1] // size**2 > 10
            for g in grids
        )
        or (size // patch) ** 2 > 1024
    ):
        raise ValueError("LLAVA_NEXT_IMAGE_GRID")


def validate_metadata(name, metadata, config):
    validate_model_config(metadata)
    size, patch = config["vision_config"]["image_size"], config["vision_config"]["patch_size"]
    if name == "preprocessor_config.json":
        if metadata.get("image_processor_type") not in {
            "LlavaNextImageProcessor",
            "LlavaNextImageProcessorFast",
        }:
            raise ValueError("LLAVA_NEXT_PROCESSOR_IDENTITY")
        expected = dict(
            image_grid_pinpoints=config["image_grid_pinpoints"],
            size={"height": size, "width": size},
            crop_size={"height": size, "width": size},
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
            do_resize=True,
            do_center_crop=True,
            do_normalize=True,
            do_rescale=True,
            do_pad=True,
            resample=3,
            rescale_factor=1 / 255,
        )
        if (
            not {"image_grid_pinpoints", "size", "crop_size", "image_mean", "image_std"}
            <= metadata.keys()
        ):
            raise ValueError("LLAVA_NEXT_PROCESSOR_INCOMPLETE")
        if (
            metadata.get("do_convert_rgb") not in {None, True}
            or metadata.get("device") is not None
            or metadata.get("data_format", "channels_first") != "channels_first"
        ):
            raise ValueError("LLAVA_NEXT_PROCESSOR_OPTIONS")
    elif name == "processor_config.json":
        expected = dict(
            processor_class="LlavaNextProcessor",
            patch_size=patch,
            num_additional_image_tokens=0,
            vision_feature_select_strategy="full",
            image_token="<image>",
        )
        if not expected.keys() <= metadata.keys():
            raise ValueError("LLAVA_NEXT_PROCESSOR_INCOMPLETE")
    elif name == "tokenizer_config.json":
        if metadata.get("tokenizer_class") not in {"GPT2Tokenizer", "GPT2TokenizerFast"}:
            raise ValueError("LLAVA_NEXT_TOKENIZER_CLASS")
        expected = dict(add_bos_token=False, add_eos_token=False)
        for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
            token = metadata.get(key)
            if isinstance(token, dict):
                token = token.get("content")
            if token != "<|end_of_text|>":
                raise ValueError("LLAVA_NEXT_SPECIAL_TOKENS")
    else:
        raise ValueError("LLAVA_NEXT_METADATA_UNSUPPORTED")
    if any(metadata.get(key, default) != default for key, default in expected.items()):
        raise ValueError("LLAVA_NEXT_PROCESSOR_OPTIONS")


def processor(root, config):
    from transformers import LlavaNextImageProcessor, LlavaNextProcessor, TokenizersBackend

    for name in METADATA_NAMES:
        validate_metadata(name, metadata_file(root / name), config)
    tokenizer = TokenizersBackend(
        tokenizer_file=str(root / "tokenizer.json"),
        bos_token="<|end_of_text|>",
        eos_token="<|end_of_text|>",
        pad_token="<|end_of_text|>",
        unk_token="<|end_of_text|>",
    )
    tokenizer.chat_template = None
    tokenizer.padding_side = "left"
    if len(tokenizer) != config["text_config"]["vocab_size"]:
        raise ValueError("LLAVA_NEXT_TOKENIZER_VOCABULARY")
    ids = []
    for marker in MARKERS[-2:]:
        encoded = tokenizer.encode(marker, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.convert_ids_to_tokens(encoded[0]) != marker:
            raise ValueError("LLAVA_NEXT_SPECIAL_TOKENS")
        ids.append(encoded[0])
    if (
        len(set(ids)) != len(ids)
        or ids[-1] != config["text_config"]["eos_token_id"]
        or ids[-2] != config["image_token_index"]
    ):
        raise ValueError("LLAVA_NEXT_SPECIAL_TOKENS")
    size = config["vision_config"]["image_size"]
    image = LlavaNextImageProcessor(
        size={"height": size, "width": size},
        crop_size={"height": size, "width": size},
        image_grid_pinpoints=deepcopy(config["image_grid_pinpoints"]),
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
        do_convert_rgb=True,
        resample=3,
        do_resize=True,
        do_center_crop=True,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        do_pad=True,
    )
    return LlavaNextProcessor(
        image_processor=image,
        tokenizer=tokenizer,
        patch_size=config["vision_config"]["patch_size"],
        num_additional_image_tokens=0,
        vision_feature_select_strategy="full",
        image_token="<image>",
        chat_template=None,
    )


def _native(config):
    from accelerate import init_empty_weights
    from transformers import LlavaNextConfig, LlavaNextForConditionalGeneration

    validate_config(config)
    native = LlavaNextConfig.from_dict(deepcopy(config))
    native._attn_implementation = "eager"
    with init_empty_weights(include_buffers=False):
        model = LlavaNextForConditionalGeneration(native)
    model.tie_weights()
    return model


def _layout(config, shapes):
    model = _native(config)
    expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    names = {}
    for key, shape in shapes.items():
        target = key
        if key.startswith("language_model.model."):
            target = "model.language_model." + key.removeprefix("language_model.model.")
        elif key.startswith("language_model.lm_head."):
            target = key.removeprefix("language_model.")
        elif key.startswith(("vision_tower.", "multi_modal_projector.")) or key == "image_newline":
            target = "model." + key
        if target.startswith("model.vision_tower.vision_model."):
            target = "model.vision_tower." + target.removeprefix("model.vision_tower.vision_model.")
        if target in names or target not in expected or shape != expected[target]:
            raise ValueError("LLAVA_NEXT_WEIGHT_GEOMETRY")
        names[target] = key
    aliases = {}
    if config.get("tie_word_embeddings") is True:
        first, second = "lm_head.weight", "model.language_model.embed_tokens.weight"
        if first not in names and second in names:
            aliases[first] = second
        if second not in names and first in names:
            aliases[second] = first
    if names.keys() | aliases.keys() != expected.keys():
        raise ValueError("LLAVA_NEXT_WEIGHTS_INCOMPLETE")
    return model, names, aliases


def validate_weight_shapes(config, shapes):
    _layout(config, shapes)


def load_components(root, config, backend):
    import torch
    from safetensors.torch import load_file

    model, names, aliases = _layout(config, preflight_safetensors(root))
    native_processor = processor(root, config)
    raw = {}
    for path in sorted(root.glob("*.safetensors")):
        weights = load_file(path, device="cpu")
        if raw.keys() & weights.keys() or any(
            not t.is_floating_point() or not bool(torch.isfinite(t).all()) for t in weights.values()
        ):
            raise ValueError("LLAVA_NEXT_WEIGHT_VALUES")
        raw.update(weights)
    state = {key: raw[source].float() for key, source in names.items()}
    state.update({key: state[source] for key, source in aliases.items()})
    if config.get("tie_word_embeddings") is True and not torch.equal(
        state["lm_head.weight"], state["model.language_model.embed_tokens.weight"]
    ):
        raise ValueError("LLAVA_NEXT_TIED_WEIGHT_CONFLICT")
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(backend).eval(), native_processor, config


def prompt(question):
    if any(marker in question for marker in MARKERS) or any(
        marker in question for marker in ("<|start_of_role|>", "<|end_of_role|>", "<|tool_call|>")
    ):
        raise ValueError("LLAVA_NEXT_RESERVED_TOKEN")
    return f"<|system|>\n{SYSTEM}\n<|user|>\n <image>\n{question}\n<|assistant|>\n"


def prepare_inputs(native_processor, config, image, question, maximum):
    import torch

    if type(maximum) is not int or not 1 <= maximum <= MAX_OUTPUT:
        raise ValueError("LLAVA_NEXT_TOKEN_LIMIT")
    values = dict(native_processor(images=image, text=prompt(question), return_tensors="pt"))
    size = config["vision_config"]["image_size"]
    pixels = values.get("pixel_values")
    if (
        set(values) != {"input_ids", "attention_mask", "pixel_values", "image_sizes"}
        or pixels.ndim != 5
        or pixels.shape[0] != 1
        or not 1 <= pixels.shape[1] <= 11
        or pixels.shape[2:] != (3, size, size)
        or not bool(torch.isfinite(pixels).all())
        or values["input_ids"].shape[0] != 1
        or values["input_ids"].shape[-1] + maximum
        > min(MAX_CONTEXT, config["text_config"]["max_position_embeddings"])
    ):
        raise ValueError("LLAVA_NEXT_INPUT_GEOMETRY")
    return values


def execute(model, native_processor, config, image, value, backend):
    from transformers import GenerationConfig

    from oryn_engine.shared_tasks import _move

    inputs = _move(
        prepare_inputs(
            native_processor,
            config,
            image,
            value.get("question", "Describe this image."),
            value["maxTokens"],
        ),
        backend,
    )
    generated = model.generate(
        **inputs,
        max_new_tokens=value["maxTokens"],
        generation_config=GenerationConfig(
            do_sample=False,
            num_beams=1,
            use_cache=True,
            **{
                key: config["text_config"][key]
                for key in ("bos_token_id", "eos_token_id", "pad_token_id")
            },
        ),
    )
    answer = native_processor.batch_decode(
        generated[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )[0].strip()
    if not answer or len(answer.encode()) > 16384:
        raise ValueError("LLAVA_NEXT_OUTPUT_INVALID")
    return {"text": answer}
