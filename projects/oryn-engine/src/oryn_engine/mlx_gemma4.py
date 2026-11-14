"""Owned Gemma 4 image/audio inputs and complete native MLX multimodal graphs."""

from __future__ import annotations

import copy
from pathlib import Path

from oryn_engine import gemma4_tasks, gemma_decision, mlx_gemma
from oryn_engine.model_metadata import metadata_file, preflight_safetensors

MAX_CONTEXT, MAX_OUTPUT, MAX_AUDIO_SECONDS = 4096, 512, 8
IMAGE_SETTINGS = gemma4_tasks.IMAGE_SETTINGS | {"size": {"height": 224, "width": 224}}
AUDIO_SETTINGS = {
    "feature_extractor_type": "Gemma4AudioFeatureExtractor",
    "sampling_rate": 16000,
    "num_mel_filters": 128,
    "fft_length": 512,
    "hop_length": 160,
    "chunk_duration": 8.0,
    "overlap_duration": 1.0,
}
PROCESSOR = {
    "processor_class": "Gemma4Processor",
    "image_processor": IMAGE_SETTINGS,
    "feature_extractor": AUDIO_SETTINGS,
    "image_seq_length": 280,
    "audio_seq_length": 750,
    "audio_ms_per_token": 40,
}
TOKENS = gemma4_tasks.TOKENS | {"pad_token": "<pad>", "video_token": "<|video|>"}


def validate_config(config):
    from oryn_engine.runtime_packs.mlx_lm_worker import validate_config as validate_mlx

    validate_mlx(config, {"gemma4"})
    gemma4_tasks.validate_config(config)
    text, vision, audio = (config.get(k) for k in ("text_config", "vision_config", "audio_config"))
    if not isinstance(vision, dict) or text.get("model_type") != "gemma4_text":
        raise ValueError("MLX_GEMMA4_GRAPH")
    for part, limits in (
        (
            text,
            {
                "hidden_size": (16, 8192),
                "intermediate_size": (32, 65536),
                "num_hidden_layers": (2, 80),
                "num_attention_heads": (1, 128),
                "num_key_value_heads": (1, 128),
                "head_dim": (8, 512),
                "global_head_dim": (8, 512),
                "vocab_size": (32, 262144),
                "max_position_embeddings": (4096, 1048576),
                "sliding_window": (8, 4096),
                "hidden_size_per_layer_input": (0, 1024),
                "num_kv_shared_layers": (0, 79),
            },
        ),
        (
            vision,
            {
                "hidden_size": (16, 4096),
                "intermediate_size": (32, 16384),
                "num_hidden_layers": (1, 64),
                "num_attention_heads": (1, 64),
                "num_key_value_heads": (1, 64),
                "head_dim": (4, 256),
                "position_embedding_size": (64, 16384),
            },
        ),
    ):
        for key, (low, high) in limits.items():
            if type(part.get(key)) is not int or not low <= part[key] <= high:
                raise ValueError("MLX_GEMMA4_GEOMETRY")
        if (
            part.get("attention_bias") is not False
            or part.get("attention_dropout") != 0
            or part.get("hidden_activation") != "gelu_pytorch_tanh"
            or part.get("rms_norm_eps") != 1e-6
            or any(
                part.get(k) is not None
                for k in ("quantization", "quantization_config", "per_layer_config")
            )
        ):
            raise ValueError("MLX_GEMMA4_GRAPH")
    global_heads = text.get("num_global_key_value_heads")
    if global_heads is not None and (type(global_heads) is not int or not 1 <= global_heads <= 128):
        raise ValueError("MLX_GEMMA4_GEOMETRY")
    layers = text.get("layer_types")
    shared = text["num_kv_shared_layers"]
    if (
        not isinstance(layers, list)
        or len(layers) != text["num_hidden_layers"]
        or set(layers) != {"sliding_attention", "full_attention"}
        or layers[-1] != "full_attention"
        or shared >= len(layers)
        or not set(layers[-shared:] if shared else []) <= set(layers[: len(layers) - shared])
        or text["num_attention_heads"] % text["num_key_value_heads"]
        or text["num_attention_heads"]
        % (text.get("num_global_key_value_heads") or text["num_key_value_heads"])
        or text["head_dim"] % 4
        or text["global_head_dim"] % 4
        or text.get("vocab_size_per_layer_input") != text["vocab_size"]
        or text.get("tie_word_embeddings") is not True
        or config.get("tie_word_embeddings") is not True
        or text.get("final_logit_softcapping") != 30.0
        or text.get("use_double_wide_mlp") is not False
        or text.get("use_second_mlp_block", False) is not False
        or text.get("rope_parameters")
        != {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        }
        or vision.get("rope_parameters") != {"rope_theta": 100.0, "rope_type": "default"}
        or vision["num_attention_heads"] != vision["num_key_value_heads"]
        or vision["hidden_size"] != vision["num_attention_heads"] * vision["head_dim"]
        or vision.get("default_output_length") != 280
        or type(vision.get("use_clipped_linears")) is not bool
        or type(vision.get("standardize")) is not bool
        or vision.get("layer_types") not in (None, ["full_attention"] * vision["num_hidden_layers"])
    ):
        raise ValueError("MLX_GEMMA4_ATTENTION")
    if text.get("enable_moe_block") is True:
        for key, maximum in (
            ("num_experts", 256),
            ("top_k_experts", 32),
            ("moe_intermediate_size", 16384),
        ):
            if type(text.get(key)) is not int or not 1 <= text[key] <= maximum:
                raise ValueError("MLX_GEMMA4_EXPERTS")
        if (
            text["top_k_experts"] > text["num_experts"]
            or shared
            or text["hidden_size_per_layer_input"]
            or text.get("attention_k_eq_v") is not True
            or text.get("use_bidirectional_attention") != "vision"
        ):
            raise ValueError("MLX_GEMMA4_EXPERTS")
    elif (
        text.get("enable_moe_block") is not False
        or text.get("attention_k_eq_v") is not False
        or text.get("use_bidirectional_attention") is not None
        or not text["hidden_size_per_layer_input"]
    ):
        raise ValueError("MLX_GEMMA4_DENSE")
    if audio is not None:
        for key, maximum in (
            ("hidden_size", 2048),
            ("num_hidden_layers", 32),
            ("num_attention_heads", 32),
            ("output_proj_dims", 8192),
        ):
            if type(audio.get(key)) is not int or not 1 <= audio[key] <= maximum:
                raise ValueError("MLX_GEMMA4_AUDIO")
        fixed = {
            "hidden_act": "silu",
            "conv_kernel_size": 5,
            "residual_weight": 0.5,
            "attention_chunk_size": 12,
            "attention_context_left": 13,
            "attention_context_right": 0,
            "attention_logit_cap": 50.0,
            "attention_invalid_logits_value": -1e9,
            "use_clipped_linears": True,
            "rms_norm_eps": 1e-6,
            "gradient_clipping": 1e10,
        }
        if (
            any(audio.get(k) != v for k, v in fixed.items())
            or audio["hidden_size"] % audio["num_attention_heads"]
            or audio["hidden_size"] % 2
        ):
            raise ValueError("MLX_GEMMA4_AUDIO")
    return config


def validate_processor(metadata):
    if metadata != PROCESSOR:
        raise ValueError("MLX_GEMMA4_PROCESSOR")


def validate_tokenizer_metadata(metadata):
    gemma4_tasks.validate_tokenizer_metadata(metadata)
    if any(metadata.get(k) != v for k, v in TOKENS.items() if k != "video_token") or metadata.get(
        "video_token"
    ) not in (None, TOKENS["video_token"]):
        raise ValueError("MLX_GEMMA4_TOKENIZER")


def native_model(config):
    from mlx_vlm.models.gemma4.config import AudioConfig, ModelConfig, TextConfig, VisionConfig
    from mlx_vlm.models.gemma4.gemma4 import Model
    from mlx_vlm.models.gemma4.language import Attention

    args = ModelConfig.from_dict(copy.deepcopy(config))
    for field, cls in (
        ("text_config", TextConfig),
        ("vision_config", VisionConfig),
        ("audio_config", AudioConfig),
    ):
        part = getattr(args, field)
        if isinstance(part, dict):
            setattr(args, field, cls.from_dict(part))
    model = Model(args)
    # Older exports retain the unused K/V projections in shared-attention layers.
    # Keep and validate every declared weight. Native shared-K/V forward semantics
    # are identical: these projections are bypassed when the shared cache is supplied.
    for i in range(
        args.text_config.num_hidden_layers - args.text_config.num_kv_shared_layers,
        args.text_config.num_hidden_layers,
    ):
        model.language_model.model.layers[i].self_attn = Attention(
            args.text_config, i, kv_shared_only=False
        )
    return model


def validate_weight_shapes(config, shapes):
    from mlx.utils import tree_flatten

    from oryn_engine.runtime_packs.mlx_lm_worker import validate_quantized_layers

    validate_config(config)
    model = native_model(config)
    expected = {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}
    validate_quantized_layers(model, config, shapes, language_only=False)
    for key in list(expected):
        prefix = key.removesuffix(".weight")
        if key.endswith(".weight") and prefix + ".scales" in shapes:
            if not config.get("quantization"):
                raise ValueError("MLX_GEMMA4_QUANTIZATION")
            for suffix in (".weight", ".scales", ".biases"):
                expected[prefix + suffix] = shapes[prefix + suffix]
    if expected != shapes:
        raise ValueError("MLX_GEMMA4_WEIGHTS_INCOMPLETE")
    return model


def load_model(root, config):
    import mlx.core as mx
    import mlx.nn as nn

    from oryn_engine.mlx_vision import verify_vision_build

    verify_vision_build()
    model = validate_weight_shapes(config, preflight_safetensors(root))
    weights = {}
    for path in sorted(root.glob("*.safetensors")):
        part = mx.load(str(path))
        if weights.keys() & part.keys():
            raise ValueError("MLX_GEMMA4_WEIGHT_COLLISION")
        weights.update(part)
    for key, value in weights.items():
        if key.endswith(".weight") and key.removesuffix(".weight") + ".scales" in weights:
            if value.dtype != mx.uint32:
                raise ValueError("MLX_GEMMA4_PACKED_DTYPE")
        elif not mx.issubdtype(value.dtype, mx.floating):
            raise ValueError("MLX_GEMMA4_FLOAT_DTYPE")
        elif key.endswith((".input_min", ".input_max", ".output_min", ".output_max")):
            # Infinity is a native disabled clipping bound, never a learned matrix value.
            number = value.item()
            if (
                number != number
                or key.endswith("_min")
                and number == float("inf")
                or key.endswith("_max")
                and number == float("-inf")
            ):
                raise ValueError("MLX_GEMMA4_CLIPPING")
        elif not bool(mx.all(mx.isfinite(value)).item()):
            raise ValueError("MLX_GEMMA4_NONFINITE")
    for key, value in weights.items():
        if (
            key.endswith((".input_min", ".output_min"))
            and value.item() > weights[key[:-3] + "max"].item()
        ):
            raise ValueError("MLX_GEMMA4_CLIPPING")
    quantization = config.get("quantization")
    if quantization:

        def predicate(name, module):
            return (
                quantization.get(
                    name,
                    {
                        k: quantization[k]
                        for k in ("bits", "group_size", "mode")
                        if k in quantization
                    },
                )
                if name + ".scales" in weights
                else False
            )

        nn.quantize(
            model,
            bits=quantization["bits"],
            group_size=quantization["group_size"],
            mode=quantization.get("mode", "affine"),
            class_predicate=predicate,
        )
    # The native audio attention caches this projection under a private helper.
    # Quantization replaces the public module; reconnect its alias so inference
    # cannot use newly initialized, unquantized relative-position weights.
    if model.audio_tower is not None:
        for layer in model.audio_tower.layers:
            layer.self_attn._rel_pos.pos_proj = layer.self_attn.relative_k_proj
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()
    return model


def load_tokenizer(root, config):
    validate_tokenizer_metadata(metadata_file(root / "tokenizer_config.json"))
    tokenizer = gemma_decision.load_tokenizer(root, config["text_config"])
    vocab = tokenizer.get_vocab()
    for key, token in TOKENS.items():
        identifier = vocab.get(token)
        declared = config.get(key + "_id", identifier)
        if key == "eos_token" and isinstance(declared, list):
            declared = identifier if identifier in declared else None
        if (
            identifier is None
            or tokenizer.encode(token, add_special_tokens=False) != [identifier]
            or declared != identifier
        ):
            raise ValueError("MLX_GEMMA4_FRAMING")
    mlx_gemma.stop_tokens(config, tokenizer)
    return tokenizer


def prepare_inputs(config, tokenizer, messages, mode, media, max_tokens):
    import mlx.core as mx
    import numpy as np
    from mlx_vlm.models.gemma4.audio_feature_extractor import Gemma4AudioFeatureExtractor
    from mlx_vlm.models.gemma4.processing_gemma4 import Gemma4ImageProcessor

    if (
        mode not in gemma4_tasks.input_modes(config)
        or type(max_tokens) is not int
        or not 1 <= max_tokens <= MAX_OUTPUT
    ):
        raise ValueError("MLX_GEMMA4_INPUT")
    reserved = set(TOKENS.values()) | {
        t for t in tokenizer.get_vocab() if t.startswith("<|") or t.endswith("|>")
    }
    if (
        len(messages) != 1
        or messages[-1]["role"] != "user"
        or any(t in m["content"] for m in messages for t in reserved)
    ):
        raise ValueError("MLX_GEMMA4_RESERVED_FRAMING")
    prepared = {}
    if mode == "image":
        pixels, counts = Gemma4ImageProcessor()(images=[media])
        if len(counts) != 1 or not 1 <= counts[0] <= 280:
            raise ValueError("MLX_GEMMA4_IMAGE_BUDGET")
        count = counts[0]
        prepared["pixel_values"] = mx.array(pixels["pixel_values"])
    else:
        wave = np.asarray(media)
        if (
            wave.ndim != 1
            or not 320 <= wave.size <= MAX_AUDIO_SECONDS * 16000
            or not np.isfinite(wave).all()
        ):
            raise ValueError("MLX_GEMMA4_AUDIO_BUDGET")
        features = Gemma4AudioFeatureExtractor()([wave], truncation=False, max_length=None)
        matrix = np.asarray(features["input_features"][0])
        mask = np.asarray(features["input_features_mask"][0])
        # The two stride-two convolutions produce ceil(T/4) features. Use that
        # exact count, not duration rounding that can silently repeat a feature.
        count = (matrix.shape[0] + 3) // 4
        if not 1 <= count <= 200 or matrix.shape[1] != 128:
            raise ValueError("MLX_GEMMA4_AUDIO_BUDGET")
        prepared.update(
            input_features=mx.array(matrix[None]), input_features_mask=mx.array(mask[None])
        )
    special = (
        TOKENS["boi_token" if mode == "image" else "boa_token"]
        + TOKENS[mode + "_token"] * count
        + TOKENS["eoi_token" if mode == "image" else "eoa_token"]
    )
    prompt = "<bos>"
    for i, message in enumerate(messages):
        role = "model" if message["role"] == "assistant" else message["role"]
        if role not in {"user", "model", "system"}:
            raise ValueError("MLX_GEMMA4_ROLE")
        prompt += (
            "<|turn>"
            + role
            + "\n"
            + (special if i == len(messages) - 1 else "")
            + message["content"].strip()
            + "<turn|>\n"
        )
    prompt += "<|turn>model\n"
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) + max_tokens > min(MAX_CONTEXT, config["text_config"]["max_position_embeddings"]):
        raise ValueError("MLX_GEMMA4_CONTEXT")
    marker = tokenizer.convert_tokens_to_ids(TOKENS[mode + "_token"])
    if ids.count(marker) != count:
        raise ValueError("MLX_GEMMA4_MEDIA_ALIGNMENT")
    prepared["input_ids"] = mx.array([ids])
    prepared["mm_token_type_ids"] = mx.where(
        prepared["input_ids"] == marker, 1 if mode == "image" else 3, 0
    )
    return prepared, prompt


def generate_tokens(model, prepared, tokenizer, config, max_tokens, temperature=0.0, top_p=1.0):
    import mlx.core as mx
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temperature, top_p=top_p)
    stops = mlx_gemma.stop_tokens(config, tokenizer)
    vocab = tokenizer.get_vocab()
    reserved = {
        i
        for t, i in vocab.items()
        if t in TOKENS.values() or t.startswith("<|") or t.endswith("|>")
    }
    cache = model.make_cache()
    tokens = []
    for _ in range(max_tokens):
        logits = model(cache=cache, **prepared).logits[0, -1, :].astype(mx.float32)
        if not bool(mx.all(mx.isfinite(logits)).item()):
            raise ValueError("MLX_GEMMA4_LOGITS")
        token = int(sampler(logits - mx.logsumexp(logits)).item())
        if token in stops:
            break
        if token in reserved or token not in vocab.values():
            raise ValueError("MLX_GEMMA4_OUTPUT_FRAMING")
        tokens.append(token)
        prepared = {"input_ids": mx.array([[token]])}
    text = tokenizer.decode(tokens, skip_special_tokens=False).strip()
    if not text or len(text.encode()) > 16384:
        raise ValueError("MLX_GEMMA4_OUTPUT")
    return text


def generate(root, raw, input_path, event, *, task="image-text-to-text"):
    import mlx.core as mx

    from oryn_engine.runners import _chat_input, _decode_wave
    from oryn_engine.shared_tasks import open_task_image, validate_task_input

    config = validate_config(metadata_file(root / "config.json"))
    validate_processor(metadata_file(root / "processor_config.json"))
    tokenizer = load_tokenizer(root, config)
    if task == "any-to-any":
        value = validate_task_input(task, raw)
        gemma4_tasks.validate_input(value)
        mode = value["inputMode"]
        messages = [{"role": "user", "content": value["instruction"]}]
        maximum = value.get("maxTokens", 128)
        temperature, top_p = 0.0, 1.0
    else:
        value = _chat_input(raw, allow_image=True)
        if "imageArtifactId" not in value:
            raise ValueError("MLX_GEMMA4_IMAGE_REQUIRED")
        mode = "image"
        messages = value["messages"]
        maximum = value["maxTokens"]
        temperature, top_p = value["temperature"], value["topP"]
    if input_path.is_symlink() or not input_path.is_absolute() or not input_path.is_file():
        raise ValueError("MLX_GEMMA4_MEDIA_PATH")
    if mode == "image":
        media, _ = open_task_image(input_path)
    else:
        if not 1 <= input_path.stat().st_size <= 12 * 1024**2:
            raise ValueError("MLX_GEMMA4_AUDIO_BUDGET")
        media = _decode_wave(input_path, input_path.stat().st_size)
    prepared, _ = prepare_inputs(config, tokenizer, messages, mode, media, maximum)
    event("job.progress", {"stage": "loading-model", "progress": 0.05})
    model = load_model(root, config)
    try:
        event("job.state", {"state": "running"})
        text = generate_tokens(model, prepared, tokenizer, config, maximum, temperature, top_p)
        if task == "image-text-to-text":
            event("job.output.delta", {"channel": "text", "delta": text})
        return {"text": text} if task == "any-to-any" else {"text": text, "backend": "mlx"}
    finally:
        del model
        mx.clear_cache()


def execute(payload, event):
    if (
        set(payload) != {"modelRoot", "task", "backend", "input", "inputPath", "outputPath"}
        or payload["task"] != "any-to-any"
        or payload["backend"] != "mlx"
        or payload["outputPath"] is not None
    ):
        raise ValueError("MLX_GEMMA4_REQUEST")
    root = Path(payload["modelRoot"])
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ValueError("MLX_GEMMA4_MODEL_PATH")
    return generate(root, payload["input"], Path(payload["inputPath"]), event, task="any-to-any")
