"""Native LLaVA-NeXT-Video binding with owned framing and bounded clip inputs."""

from __future__ import annotations

from oryn_engine.model_metadata import metadata_file, validate_model_config

METADATA_NAMES = ("preprocessor_config.json", "processor_config.json", "tokenizer_config.json")
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]


def text_config(config):
    # Pinned LlamaConfig defaults: the original Vicuna export omits these fields.
    return {
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000,
        "max_position_embeddings": 2048,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    } | config["text_config"]


def validate_config(config):
    validate_model_config(config)
    vision, text = config.get("vision_config"), config.get("text_config")
    if (
        config.get("model_type") != "llava_next_video"
        or config.get("architectures") != ["LlavaNextVideoForConditionalGeneration"]
        or not isinstance(vision, dict)
        or not isinstance(text, dict)
        or vision.get("model_type") != "clip_vision_model"
        or text.get("model_type") != "llama"
        or config.get("spatial_pool_mode", "average") != "average"
        or config.get("spatial_pool_stride", 2) != 2
        or config.get("vision_feature_select_strategy", "default") != "default"
        or config.get("projector_hidden_act", "gelu") != "gelu"
        or vision.get("num_channels", 3) != 3
        or any(
            c.get("quantization_config") or c.get("quantization") for c in (config, vision, text)
        )
    ):
        raise ValueError("LLAVA_VIDEO_CONFIGURATION_UNSUPPORTED")
    text = text_config(config)
    for part, bounds in (
        (
            vision,
            {
                "hidden_size": (8, 4096),
                "intermediate_size": (8, 16384),
                "num_hidden_layers": (1, 48),
                "num_attention_heads": (1, 64),
                "image_size": (8, 512),
                "patch_size": (1, 64),
            },
        ),
        (
            text,
            {
                "hidden_size": (8, 16384),
                "intermediate_size": (8, 65536),
                "num_hidden_layers": (1, 128),
                "num_attention_heads": (1, 128),
                "vocab_size": (8, 262144),
                "max_position_embeddings": (128, 1048576),
            },
        ),
    ):
        for name, (low, high) in bounds.items():
            value = part.get(name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError("LLAVA_VIDEO_CONFIGURATION_LIMIT")
        if part["hidden_size"] % part["num_attention_heads"]:
            raise ValueError("LLAVA_VIDEO_HEAD_GEOMETRY")
    kv = text.get("num_key_value_heads", text["num_attention_heads"])
    if (
        type(kv) is not int
        or not 1 <= kv <= text["num_attention_heads"]
        or text["num_attention_heads"] % kv
    ):
        raise ValueError("LLAVA_VIDEO_HEAD_GEOMETRY")
    size, patch = vision["image_size"], vision["patch_size"]
    layer = config.get("vision_feature_layer", -2)
    if (
        size % (2 * patch)
        or 8 * (size // patch // 2) ** 2 > 2048
        or type(layer) is not int
        or not -vision["num_hidden_layers"] - 1 <= layer <= vision["num_hidden_layers"]
    ):
        raise ValueError("LLAVA_VIDEO_FEATURE_GEOMETRY")
    for name in ("image_token_index", "video_token_index"):
        token = config.get(name)
        if type(token) is not int or not 0 <= token < text["vocab_size"]:
            raise ValueError("LLAVA_VIDEO_TOKEN_CONFIGURATION")
    if config.get("image_token_index", 32001) == config.get("video_token_index", 32000):
        raise ValueError("LLAVA_VIDEO_TOKEN_CONFIGURATION")
    generation_tokens(config)


def generation_tokens(config):
    text = text_config(config)
    tokens = {name: text[name] for name in ("bos_token_id", "eos_token_id", "pad_token_id")}
    if any(type(v) is not int or not 0 <= v < text["vocab_size"] for v in tokens.values()):
        raise ValueError("LLAVA_VIDEO_TOKEN_CONFIGURATION")
    return tokens


def validate_metadata(name, metadata, config):
    validate_model_config(metadata)
    size, patch = config["vision_config"]["image_size"], config["vision_config"]["patch_size"]
    if name == "preprocessor_config.json":
        if not {"image_processor_type", "size", "crop_size"} <= metadata.keys():
            raise ValueError("LLAVA_VIDEO_PROCESSOR_INCOMPLETE")
        expected = {
            "image_processor_type": "LlavaNextImageProcessor",
            "processor_class": "LlavaNextVideoProcessor",
            "size": {"shortest_edge": size},
            "crop_size": {"height": size, "width": size},
            "do_resize": True,
            "do_center_crop": True,
            "do_convert_rgb": True,
            "do_rescale": True,
            "do_normalize": True,
            "resample": 3,
            "rescale_factor": 1 / 255,
            "image_mean": MEAN,
            "image_std": STD,
        }
    elif name == "processor_config.json":
        if (
            not {"processor_class", "patch_size", "vision_feature_select_strategy"}
            <= metadata.keys()
        ):
            raise ValueError("LLAVA_VIDEO_PROCESSOR_INCOMPLETE")
        expected = {
            "processor_class": "LlavaNextVideoProcessor",
            "patch_size": patch,
            "vision_feature_select_strategy": "default",
            "num_additional_image_tokens": 1,
            "image_token": "<image>",
            "video_token": "<video>",
        }
    elif name == "tokenizer_config.json":
        expected = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<unk>",
            "unk_token": "<unk>",
            "add_bos_token": True,
            "add_eos_token": False,
            "use_default_system_prompt": False,
        }
        if metadata.get("tokenizer_class") not in {"LlamaTokenizer", "LlamaTokenizerFast"}:
            raise ValueError("LLAVA_VIDEO_TOKENIZER_UNSUPPORTED")
    else:
        raise ValueError("LLAVA_VIDEO_METADATA_UNSUPPORTED")
    if any(metadata.get(key, default) != default for key, default in expected.items()):
        raise ValueError("LLAVA_VIDEO_PROCESSOR_UNSUPPORTED")


def load_components(root, config, common):
    import transformers as tf

    for name in METADATA_NAMES:
        validate_metadata(name, metadata_file(root / name), config)
    tokenizer = tf.LlamaTokenizer.from_pretrained(root, **common)
    tokenizer.chat_template = None
    if len(tokenizer) > text_config(config)["vocab_size"]:
        raise ValueError("LLAVA_VIDEO_TOKENIZER_VOCABULARY")
    for name, expected in generation_tokens(config).items():
        if getattr(tokenizer, name) != expected:
            raise ValueError("LLAVA_VIDEO_TOKENIZER_SPECIAL_TOKENS")
    for text, index in (("<image>", "image_token_index"), ("<video>", "video_token_index")):
        if tokenizer.encode(text, add_special_tokens=False) != [config[index]]:
            raise ValueError("LLAVA_VIDEO_TOKENIZER_SPECIAL_TOKENS")
    size = config["vision_config"]["image_size"]
    options = {
        "size": {"shortest_edge": size},
        "crop_size": {"height": size, "width": size},
        "image_mean": MEAN,
        "image_std": STD,
        "resample": 3,
        "do_resize": True,
        "do_center_crop": True,
        "do_rescale": True,
        "do_normalize": True,
        "do_convert_rgb": True,
        "rescale_factor": 1 / 255,
    }
    # Image processor is required by the native processor constructor but this
    # binding accepts only video. Its image settings cannot enter an invocation.
    processor = tf.LlavaNextVideoProcessor(
        tokenizer=tokenizer,
        image_processor=tf.LlavaNextImageProcessor(**options),
        video_processor=tf.LlavaNextVideoVideoProcessor(**options, do_sample_frames=False),
        patch_size=config["vision_config"]["patch_size"],
        vision_feature_select_strategy="default",
        num_additional_image_tokens=1,
        image_token="<image>",
        video_token="<video>",
        chat_template=None,
    )
    return processor, tf.LlavaNextVideoForConditionalGeneration


def prepare_inputs(processor, config, frames, question, max_tokens):
    # Matches the reviewed single user turn in the native Vicuna video template.
    # Repository Jinja is never rendered, and user text cannot add media tokens.
    if any(token in question for token in ("<video>", "<image>", "<s>", "</s>", "<unk>")):
        raise ValueError("LLAVA_VIDEO_RESERVED_TOKEN")
    inputs = dict(
        processor(
            text=f"USER: <video>\n{question} ASSISTANT:",
            videos=[frames],
            do_sample_frames=False,
            return_tensors="pt",
        )
    )
    size, patch = config["vision_config"]["image_size"], config["vision_config"]["patch_size"]
    expected = 8 * (size // patch // 2) ** 2
    if (
        set(inputs) != {"input_ids", "attention_mask", "pixel_values_videos"}
        or tuple(inputs["pixel_values_videos"].shape) != (1, 8, 3, size, size)
        or inputs["input_ids"].shape[0] != 1
        or int((inputs["input_ids"] == config["video_token_index"]).sum()) != expected
        or bool((inputs["input_ids"] == config["image_token_index"]).any())
    ):
        raise ValueError("LLAVA_VIDEO_INPUT_GEOMETRY")
    if not 1 <= max_tokens <= 512 or inputs["input_ids"].shape[-1] + max_tokens > min(
        4096, text_config(config)["max_position_embeddings"]
    ):
        raise ValueError("LLAVA_VIDEO_TOKEN_LIMIT")
    return inputs


def execute(model, processor, config, value, backend, path):
    from transformers import GenerationConfig

    from oryn_engine.coverage_tasks import sample_video
    from oryn_engine.shared_tasks import _move

    frames, times, duration = sample_video(
        path, 8, value["startSeconds"], value.get("durationSeconds")
    )
    inputs = _move(
        prepare_inputs(processor, config, frames, value["question"], value["maxTokens"]), backend
    )
    generated = model.generate(
        **inputs,
        max_new_tokens=value["maxTokens"],
        generation_config=GenerationConfig(
            **generation_tokens(config), do_sample=False, num_beams=1, use_cache=True
        ),
    )
    answer = processor.batch_decode(
        generated[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )[0].strip()
    if not answer or len(answer.encode()) > 16384:
        raise ValueError("LLAVA_VIDEO_OUTPUT_INVALID")
    return {"text": answer, "sampledTimestamps": times, "durationSeconds": duration}
