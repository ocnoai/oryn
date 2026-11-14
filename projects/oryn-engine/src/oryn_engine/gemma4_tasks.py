"""Bounded image/audio grounding using the packaged native Gemma 4 graph."""

from __future__ import annotations

TASK_CONTRACTS = {"any-to-any": "multimodal-generation-v1"}
TASK_FAMILIES = {"any-to-any": {"gemma4"}}
IMAGE_SETTINGS = {
    "do_convert_rgb": True,
    "do_normalize": False,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.0, 0.0, 0.0],
    "image_std": [1.0, 1.0, 1.0],
    "image_processor_type": "Gemma4ImageProcessor",
    "image_seq_length": 280,
    "max_soft_tokens": 280,
    "patch_size": 16,
    "pooling_kernel_size": 3,
    "resample": 3,
    "rescale_factor": 1 / 255,
}
AUDIO_SETTINGS = {
    "dither": 0.0,
    "feature_extractor_type": "Gemma4AudioFeatureExtractor",
    "feature_size": 128,
    "fft_length": 512,
    "fft_overdrive": False,
    "frame_length": 320,
    "hop_length": 160,
    "input_scale_factor": 1.0,
    "max_frequency": 8000.0,
    "mel_floor": 0.001,
    "min_frequency": 0.0,
    "padding_side": "right",
    "padding_value": 0.0,
    "per_bin_mean": None,
    "per_bin_stddev": None,
    "preemphasis": 0.0,
    "preemphasis_htk_flavor": True,
    "return_attention_mask": True,
    "sampling_rate": 16000,
}
TOKENS = {
    "image_token": "<|image|>",
    "boi_token": "<|image>",
    "eoi_token": "<image|>",
    "audio_token": "<|audio|>",
    "boa_token": "<|audio>",
    "eoa_token": "<audio|>",
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "sot_token": "<|turn>",
    "eot_token": "<turn|>",
}


def input_modes(config):
    return [
        name
        for name, key in (("image", "vision_config"), ("audio", "audio_config"))
        if isinstance(config.get(key), dict)
    ]


def precision(config, backend):
    text = config.get("text_config", {})
    declared = config.get("dtype") or text.get("dtype") or "float32"
    if declared not in {"float32", "float16", "bfloat16"}:
        raise ValueError("Native Gemma 4 precision is unsupported")
    return (
        "fp32"
        if backend == "cpu"
        else {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}[declared]
    )


def validate_config(config):
    if (
        config.get("model_type") != "gemma4"
        or config.get("architectures") != ["Gemma4ForConditionalGeneration"]
        or not isinstance(config.get("text_config"), dict)
        or not input_modes(config)
    ):
        raise ValueError("Native Gemma 4 multimodal graph is unavailable")
    precision(config, "mps")
    vision = config.get("vision_config")
    if vision is not None and (
        not isinstance(vision, dict)
        or vision.get("model_type") != "gemma4_vision"
        or vision.get("patch_size", 16) != 16
        or vision.get("pooling_kernel_size", 3) != 3
        or config.get("vision_soft_tokens_per_image", 280) != 280
    ):
        raise ValueError("Native Gemma 4 vision geometry is unsupported")
    audio = config.get("audio_config")
    if audio is not None and (
        not isinstance(audio, dict)
        or audio.get("model_type") != "gemma4_audio"
        or not isinstance(audio.get("subsampling_conv_channels"), list)
        or len(audio["subsampling_conv_channels"]) != 2
        or audio["subsampling_conv_channels"][0] != 128
        or type(audio["subsampling_conv_channels"][1]) is not int
        or not 1 <= audio["subsampling_conv_channels"][1] <= 128
    ):
        raise ValueError("Native Gemma 4 audio geometry is unsupported")


def validate_processor(metadata):
    from oryn_engine.model_metadata import validate_model_config

    validate_model_config(metadata)
    if (
        set(metadata)
        - {
            "processor_class",
            "image_processor",
            "feature_extractor",
            "video_processor",
            "audio_ms_per_token",
            "audio_seq_length",
            "image_seq_length",
        }
        or metadata.get("processor_class") != "Gemma4Processor"
        or metadata.get("image_processor") != IMAGE_SETTINGS
        or metadata.get("feature_extractor") != AUDIO_SETTINGS
        or metadata.get("image_seq_length") != 280
        or metadata.get("audio_seq_length") != 750
        or metadata.get("audio_ms_per_token") != 40
        or not isinstance(metadata.get("video_processor"), dict)
        or metadata["video_processor"].get("video_processor_type") != "Gemma4VideoProcessor"
    ):
        raise ValueError("Native Gemma 4 preprocessing is unsupported")


def validate_tokenizer_metadata(metadata):
    from oryn_engine.model_metadata import validate_model_config

    validate_model_config(metadata)
    if metadata.get("tokenizer_class") != "GemmaTokenizer" or any(
        metadata.get(key) != value for key, value in TOKENS.items()
    ):
        raise ValueError("Native Gemma 4 token framing is unsupported")


def load_components(root, config, common):
    import transformers as tf

    from oryn_engine.model_metadata import metadata_file

    validate_config(config)
    validate_processor(metadata_file(root / "processor_config.json"))
    validate_tokenizer_metadata(metadata_file(root / "tokenizer_config.json"))
    tokenizer = tf.AutoTokenizer.from_pretrained(root, **common)
    for key, token in TOKENS.items():
        identifier = tokenizer.convert_tokens_to_ids(token)
        declared = config.get(key + "_id", identifier)
        if key == "eos_token" and isinstance(declared, list):
            # Original instruction checkpoints stop at EOS or end-of-turn.
            declared = identifier if identifier in declared else None
        if (
            identifier is None
            or identifier == tokenizer.unk_token_id
            or type(identifier) is not int
            or not 0 <= identifier < config["text_config"]["vocab_size"]
            or declared != identifier
        ):
            raise ValueError("Native Gemma 4 graph and tokenizer disagree")
    processor = tf.Gemma4Processor(
        tokenizer=tokenizer,
        image_processor=tf.Gemma4ImageProcessor(),
        feature_extractor=tf.Gemma4AudioFeatureExtractor(),
        video_processor=tf.Gemma4VideoProcessor(num_frames=1, do_sample_frames=False),
        chat_template="",
        image_seq_length=280,
        audio_seq_length=750,
        audio_ms_per_token=40,
    )
    return processor, tf.Gemma4ForConditionalGeneration


def validate_input(value):
    if value["inputMode"] not in {"image", "audio"} or value.get("outputMode", "text") != "text":
        raise ValueError("Native Gemma 4 requires one declared input modality and text output")


def execute(model, processor, config, value, backend, input_path):
    import torch

    from oryn_engine.runners import _decode_wave
    from oryn_engine.shared_tasks import _move, open_task_image

    validate_input(value)
    mode = value["inputMode"]
    if mode not in input_modes(config) or input_path is None:
        raise ValueError("Selected native Gemma 4 modality is unavailable")
    instruction = value["instruction"]
    if any(token and token in instruction for token in processor.tokenizer.all_special_tokens):
        raise ValueError("Instruction contains reserved native framing")
    media = {}
    if mode == "image":
        image, _ = open_task_image(input_path)
        media["images"] = [[image]]
    else:
        if (
            input_path.is_symlink()
            or not input_path.is_file()
            or not 1 <= input_path.stat().st_size <= 12 * 1024**2
        ):
            raise ValueError("Owned audio input is unavailable or oversized")
        audio = _decode_wave(input_path, input_path.stat().st_size)
        if not 1 <= audio.size <= 30 * 16000:
            raise ValueError("Native Gemma 4 audio exceeds the declared duration")
        media.update(audio=[audio], sampling_rate=16000)
    # The canonical one-user-turn, no-thinking frame is owned by this adapter.
    # Repository Jinja and response parsers are never evaluated.
    prompt = (
        "<bos><|turn>user\n" + TOKENS[mode + "_token"] + instruction + "<turn|>\n<|turn>model\n"
    )
    inputs = _move(dict(processor(text=[prompt], **media, return_tensors="pt")), backend)
    if inputs["input_ids"].shape[-1] > 4096 or any(
        tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())
        for tensor in inputs.values()
        if isinstance(tensor, torch.Tensor)
    ):
        raise ValueError("Native Gemma 4 input is nonfinite or oversized")
    generated = model.generate(
        **inputs, do_sample=False, max_new_tokens=value.get("maxTokens", 128)
    )
    text = processor.batch_decode(
        generated[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )[0].strip()
    if not text or len(text.encode()) > 16384:
        raise ValueError("Native Gemma 4 answer is empty or oversized")
    return {"text": text}
