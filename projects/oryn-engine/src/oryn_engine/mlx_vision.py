"""Native joint image/text inference through the pinned MLX vision build.

Only the reviewed Qwen image families are bound. Neither repository loader code,
chat templates, quantizer plugins nor the library's URL/file dispatch are used.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
from pathlib import Path

from oryn_engine.model_metadata import metadata_file, preflight_safetensors
from oryn_engine.runtime_packs.mlx_lm_worker import validate_config, validate_quantized_layers

# The mlx-vlm release whose internals the MLX adapters were reviewed and tested against.
REVIEWED_MLX_VLM = "0.7.11"
REVIEWED_FAMILIES = frozenset({"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_5", "qwen3_5_moe"})


def validate_vision_config(config):
    from oryn_engine.generated_capabilities import architectures_for

    validate_config(config, set(architectures_for("image-text-to-text", source="mlx")))
    vision = config.get("vision_config")
    if (
        config.get("model_type") not in REVIEWED_FAMILIES
        or not isinstance(vision, dict)
        or not vision
        or ("skip_vision" in vision and type(vision["skip_vision"]) is not bool)
        or any(config.get(k) is not None for k in ("speculators_model_type", "dflash_config"))
        or any(config.get(k) is not None for k in ("audio_config", "ple_storage"))
        or config.get("skip_vision", False) is not False
    ):
        raise ValueError("MLX vision graph is not admitted.")
    for name, maximum in {"depth": 128, "num_heads": 256, "embed_dim": 16384}.items():
        if name in vision and (type(vision[name]) is not int or not 1 <= vision[name] <= maximum):
            raise ValueError("MLX vision dimensions exceed the admitted bound.")
    if config["model_type"] in {"qwen3_vl", "qwen3_5", "qwen3_5_moe"}:
        text = config.get("text_config")
        required = {
            "model_type",
            "num_hidden_layers",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "rms_norm_eps",
            "vocab_size",
            "max_position_embeddings",
        }
        required |= (
            {
                "num_experts",
                "num_experts_per_tok",
                "shared_expert_intermediate_size",
                "moe_intermediate_size",
            }
            if config["model_type"] == "qwen3_5_moe"
            else {"intermediate_size"}
        )
        required |= (
            {"head_dim", "rope_theta"}
            if config["model_type"] == "qwen3_vl"
            else {
                "linear_num_value_heads",
                "linear_num_key_heads",
                "linear_key_head_dim",
                "linear_value_head_dim",
                "linear_conv_kernel_dim",
            }
        )
        if not isinstance(text, dict) or not required <= text.keys():
            raise ValueError("MLX vision text configuration is incomplete.")
        if config["model_type"] in {"qwen3_5", "qwen3_5_moe"}:
            for key in ("linear_key_head_dim", "linear_value_head_dim"):
                value = text[key]
                if type(value) is not int or not 32 <= value <= 256 or value % 32:
                    raise ValueError("MLX recurrent attention geometry is unsupported.")
            for key in ("linear_num_key_heads", "linear_num_value_heads"):
                if type(text[key]) is not int or not 1 <= text[key] <= 256:
                    raise ValueError("MLX recurrent attention geometry is unsupported.")
            if text["linear_num_value_heads"] % text["linear_num_key_heads"]:
                raise ValueError("MLX recurrent attention head grouping is unsupported.")
        if config["model_type"] == "qwen3_5_moe":
            for key, maximum in {
                "num_experts": 1024,
                "num_experts_per_tok": 64,
                "shared_expert_intermediate_size": 262144,
                "moe_intermediate_size": 262144,
            }.items():
                if type(text[key]) is not int or not 1 <= text[key] <= maximum:
                    raise ValueError("MLX vision expert geometry is unsupported.")
            if text["num_experts_per_tok"] > text["num_experts"]:
                raise ValueError("MLX vision expert selection exceeds the graph.")
    return config


def verify_vision_build():
    from importlib.metadata import version

    from oryn_engine.generated_capabilities import generated_artifact

    inputs = generated_artifact()["buildInputs"]
    if version("mlx-vlm") != inputs["mlxVlm"]:
        raise ValueError("MLX vision build changed.")
    root = Path(importlib.util.find_spec("mlx_vlm").origin).parent
    for name, digest in inputs["mlxVlmSources"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("MLX vision source changed.")


def load_model(root: Path, config):
    import mlx.core as mx
    import mlx.nn as nn

    verify_vision_build()
    validate_vision_config(config)
    preflight_safetensors(root)
    module = importlib.import_module(f"mlx_vlm.models.{config['model_type']}")
    args = module.ModelConfig.from_dict(copy.deepcopy(config))
    # Native Qwen2-VL stores text settings at the root; Qwen2.5 also supports
    # the nested form. Use the native config transform, with no repository imports.
    if isinstance(args.text_config, dict):
        args.text_config = module.TextConfig.from_dict(args.text_config)
    if isinstance(args.vision_config, dict):
        args.vision_config = module.VisionConfig.from_dict(args.vision_config)
    model = module.Model(args)
    raw_weights = {}
    for path in sorted(root.glob("*.safetensors")):
        for name, tensor in mx.load(str(path)).items():
            if name in raw_weights or "mtp." in name or name.endswith("weight_scale_inv"):
                raise ValueError("MLX vision tensor mapping is ambiguous.")
            raw_weights[name] = tensor
    weights = sanitize_weights(model, raw_weights)
    weights = model.vision_tower.sanitize(weights)
    if hasattr(model.language_model, "sanitize"):
        weights = model.language_model.sanitize(weights)
    if not any(k.startswith("vision_tower.") for k in weights):
        raise ValueError("MLX vision weights are missing.")
    # Legacy skip_vision means leave vision layers unquantized, not omit the
    # vision graph. Quantize only layers with validated scales and require all
    # remaining floating vision tensors through strict native loading.
    quantization = getattr(args, "quantization", config.get("quantization"))
    if quantization and len(quantization) != len(config["quantization"]):
        raise ValueError("MLX vision quantization mapping is ambiguous.")
    validate_quantized_layers(
        model,
        config | {"quantization": quantization},
        {k: tuple(v.shape) for k, v in weights.items()},
        language_only=False,
    )
    if quantization:

        def predicate(name, _module):
            if name + ".scales" not in weights:
                return False
            return quantization.get(
                name,
                {k: quantization[k] for k in ("bits", "group_size", "mode") if k in quantization},
            )

        nn.quantize(
            model,
            bits=quantization["bits"],
            group_size=quantization["group_size"],
            mode=quantization.get("mode", "affine"),
            class_predicate=predicate,
        )
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()
    return model


def sanitize_weights(model, weights):
    # Check key collisions independently, then sanitize the complete graph.
    # Qwen 3.5 selects its RMSNorm convention from other tensors in the graph;
    # sanitizing one tensor at a time changes the mathematical model.
    names = set()
    for name, tensor in weights.items():
        mapped = model.sanitize({name: tensor})
        if len(mapped) != 1 or names & mapped.keys():
            raise ValueError("MLX vision tensor mapping is ambiguous.")
        names.update(mapped)
    result = model.sanitize(dict(weights))
    if set(result) != names:
        raise ValueError("MLX vision tensor mapping is ambiguous.")
    return result


def generate(root: Path, raw, image_path: Path, event):
    import mlx.core as mx
    import transformers as tf
    from mlx_vlm import stream_generate

    from oryn_engine.runners import _chat_input
    from oryn_engine.shared_tasks import open_task_image, qwen_image_options

    value = _chat_input(raw, allow_image=True)
    if "imageArtifactId" not in value or not image_path.is_absolute():
        raise ValueError("MLX image input is missing.")
    config = validate_vision_config(metadata_file(root / "config.json"))
    image, _ = open_task_image(image_path)
    options_path = root / "preprocessor_config.json"
    if not options_path.exists():
        options_path = root / "processor_config.json"
    options = qwen_image_options(metadata_file(options_path), config["vision_config"])
    options.update(min_pixels=56 * 56, max_pixels=1024 * 28 * 28)
    tokenizer = tf.AutoTokenizer.from_pretrained(
        root, local_files_only=True, trust_remote_code=False
    )
    tokenizer.chat_template = None
    klass = {
        "qwen2_vl": tf.Qwen2VLProcessor,
        "qwen2_5_vl": tf.Qwen2_5_VLProcessor,
        "qwen3_vl": tf.Qwen3VLProcessor,
        "qwen3_5": tf.Qwen3VLProcessor,
        "qwen3_5_moe": tf.Qwen3VLProcessor,
    }[config["model_type"]]
    video_class = (
        tf.Qwen3VLVideoProcessor
        if config["model_type"] in {"qwen3_vl", "qwen3_5", "qwen3_5_moe"}
        else tf.Qwen2VLVideoProcessor
    )
    processor = klass(
        tokenizer=tokenizer,
        image_processor=tf.Qwen2VLImageProcessor(**options),
        video_processor=video_class(min_frames=2, max_frames=2),
    )
    from mlx_vlm.tokenizer_utils import NaiveStreamingDetokenizer
    from mlx_vlm.utils import StoppingCriteria

    processor.detokenizer = NaiveStreamingDetokenizer(tokenizer)
    tokenizer.stopping_criteria = StoppingCriteria(
        config.get("eos_token_id", tokenizer.eos_token_id), tokenizer
    )
    prompt = image_prompt(value["messages"], config["model_type"])
    inputs = processor(text=[prompt], images=[image], return_tensors="np", padding=False)
    if inputs["input_ids"].shape[1] + value["maxTokens"] > 8192:
        raise ValueError("MLX vision context exceeds its bound.")
    prepared = {k: mx.array(v) for k, v in inputs.items()}
    prepared["mask"] = prepared.pop("attention_mask", None)
    mx.reset_peak_memory()
    event("job.progress", {"stage": "loading-model", "progress": 0.05})
    model = load_model(root, config)
    event("job.progress", {"stage": "running-model", "progress": 0.2})
    pieces, count = [], 0
    try:
        for response in stream_generate(
            model,
            processor,
            prompt,
            **prepared,
            max_tokens=value["maxTokens"],
            temperature=value["temperature"],
            top_p=value["topP"],
            skip_special_tokens=True,
        ):
            text = response.text
            count += len(text.encode())
            if count > 512 * 1024:
                raise ValueError("MLX vision output exceeds its bound.")
            if text:
                pieces.append(text)
                event("job.output.delta", {"channel": "text", "delta": text})
        return {"text": "".join(pieces), "backend": "mlx", "peakMetalBytes": mx.get_peak_memory()}
    finally:
        del model
        mx.clear_cache()


def image_prompt(messages, family):
    from oryn_engine.chat_formats import render

    last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
    # The public chat result contains the answer. Use the native non-thinking
    # framing for Qwen 3.5, without rendering the repository's Jinja template.
    return render(
        "chatml",
        messages,
        image_index=last_user,
        thinking_off=family in {"qwen3_5", "qwen3_5_moe"},
    )
