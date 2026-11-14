"""Native Gemma 4 Unified text execution with complete declared tensor layouts."""

from __future__ import annotations

import copy

from oryn_engine import gemma_decision
from oryn_engine.model_metadata import preflight_safetensors

MODEL_TYPE = "gemma4_unified"
MAX_CONTEXT, MAX_OUTPUT = 8192, 512
AUXILIARY_PREFIXES = ("vision_embedder.", "embed_vision.", "embed_audio.")


def validate_config(config):
    from oryn_engine.runtime_packs.mlx_lm_worker import validate_config as validate_mlx_config

    validate_mlx_config(config, {MODEL_TYPE})
    text, vision, audio = (config.get(k) for k in ("text_config", "vision_config", "audio_config"))
    if (
        config.get("model_type") != MODEL_TYPE
        or config.get("architectures") != ["Gemma4UnifiedForConditionalGeneration"]
        or not all(isinstance(part, dict) for part in (text, audio))
        or text.get("model_type") != "gemma4_unified_text"
        or vision is not None
        and (not isinstance(vision, dict) or vision.get("model_type") != "gemma4_unified_vision")
        or audio.get("model_type") != "gemma4_unified_audio"
    ):
        raise ValueError("MLX_GEMMA_CONFIGURATION")
    for key, low, high in (
        ("hidden_size", 16, 8192),
        ("intermediate_size", 16, 65536),
        ("num_hidden_layers", 2, 80),
        ("num_attention_heads", 1, 128),
        ("num_key_value_heads", 1, 128),
        ("num_global_key_value_heads", 1, 128),
        ("head_dim", 8, 512),
        ("global_head_dim", 16, 512),
        ("vocab_size", 32, 262144),
        ("max_position_embeddings", 8192, 1048576),
        ("sliding_window", 8, 4096),
    ):
        if type(text.get(key)) is not int or not low <= text[key] <= high:
            raise ValueError("MLX_GEMMA_GEOMETRY")
    if (
        text.get("enable_moe_block") is not False
        or text.get("attention_k_eq_v") is not True
        or text.get("attention_bias") is not False
        or text.get("attention_dropout") != 0
        or text.get("hidden_activation") != "gelu_pytorch_tanh"
        or text.get("hidden_size_per_layer_input") != 0
        or text.get("num_kv_shared_layers") != 0
        or text.get("use_double_wide_mlp") is not False
        or text.get("use_bidirectional_attention") != "vision"
        or text.get("per_layer_config") is not None
        or text.get("tie_word_embeddings") is not True
        or config.get("tie_word_embeddings") is not True
        or text["num_attention_heads"] % text["num_key_value_heads"]
        or text["num_attention_heads"] % text["num_global_key_value_heads"]
        or text["head_dim"] % 4
        or text["global_head_dim"] % 4
        or text.get("rope_parameters")
        != {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        }
        or any(
            isinstance(part, dict) and part.get(k) is not None
            for part in (text, vision, audio)
            for k in ("quantization", "quantization_config")
        )
    ):
        raise ValueError("MLX_GEMMA_GRAPH")
    layers = text.get("layer_types")
    if (
        not isinstance(layers, list)
        or len(layers) != text["num_hidden_layers"]
        or set(layers) != {"sliding_attention", "full_attention"}
        or layers[-1] != "full_attention"
    ):
        raise ValueError("MLX_GEMMA_ATTENTION_LAYOUT")
    if (
        vision is not None
        and (
            vision.get("patch_size") != 16
            or vision.get("pooling_kernel_size") != 3
            or vision.get("model_patch_size", 48) != 48
            or vision.get("mm_embed_dim") != text["hidden_size"]
            or vision.get("output_proj_dims") != text["hidden_size"]
            or type(vision.get("mm_posemb_size")) is not int
            or not 1 <= vision["mm_posemb_size"] <= 4096
        )
        or audio.get("audio_embed_dim") != 640
        or audio.get("output_proj_dims", 640) != 640
        or audio.get("hidden_size", 640) != 640
    ):
        raise ValueError("MLX_GEMMA_AUXILIARY_GEOMETRY")
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if type(text.get(key)) is not int or not 0 <= text[key] < text["vocab_size"]:
            raise ValueError("MLX_GEMMA_TOKEN_IDS")


def _native_model(config, *, auxiliary):
    from mlx_vlm.models.gemma4_unified import Model, ModelConfig

    selected = copy.deepcopy(config)
    if not auxiliary:
        # The published text export omits all auxiliary tensors. Bind only that
        # complete task graph; partially present auxiliary graphs still fail.
        selected.update(vision_config=None, audio_config=None)
    return Model(ModelConfig.from_dict(selected))


def _mapped_names(shapes):
    result = {}
    for name, shape in shapes.items():
        name = name.removeprefix("model.")
        if name.startswith("language_model.") and not name.startswith("language_model.model."):
            name = "language_model.model." + name.removeprefix("language_model.")
        if name in result:
            raise ValueError("MLX_GEMMA_WEIGHT_COLLISION")
        result[name] = shape
    return result


def validate_weight_shapes(config, shapes):
    from mlx.utils import tree_flatten

    from oryn_engine.runtime_packs.mlx_lm_worker import validate_quantized_layers

    validate_config(config)
    mapped = _mapped_names(shapes)
    if not config.get("quantization") and any(
        name.endswith((".scales", ".biases")) for name in mapped
    ):
        raise ValueError("MLX_GEMMA_QUANTIZATION_MISSING")
    auxiliary = any(k.startswith(AUXILIARY_PREFIXES) for k in mapped)
    if auxiliary and not isinstance(config.get("vision_config"), dict):
        raise ValueError("MLX_GEMMA_AUXILIARY_CONFIGURATION")
    model = _native_model(config, auxiliary=auxiliary)
    expected = {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}
    validate_quantized_layers(model, config, mapped, language_only=False)
    for name in list(expected):
        if name.endswith(".weight") and name.removesuffix(".weight") + ".scales" in mapped:
            prefix = name.removesuffix(".weight")
            expected[name] = mapped[name]
            expected[prefix + ".scales"] = mapped[prefix + ".scales"]
            expected[prefix + ".biases"] = mapped[prefix + ".biases"]
    if expected != mapped:
        raise ValueError("MLX_GEMMA_WEIGHTS_INCOMPLETE")
    return model, auxiliary


def load_model(root, config):
    import mlx.core as mx
    import mlx.nn as nn

    from oryn_engine.mlx_vision import sanitize_weights, verify_vision_build

    verify_vision_build()
    model, auxiliary = validate_weight_shapes(config, preflight_safetensors(root))
    weights = {}
    for path in sorted(root.glob("*.safetensors")):
        part = mx.load(str(path))
        if weights.keys() & part.keys():
            raise ValueError("MLX_GEMMA_WEIGHT_COLLISION")
        weights.update(part)
    weights = sanitize_weights(model, weights)
    for name, value in weights.items():
        packed = name.endswith(".weight") and name.removesuffix(".weight") + ".scales" in weights
        if packed:
            if value.dtype != mx.uint32:
                raise ValueError("MLX_GEMMA_PACKED_DTYPE")
        elif not mx.issubdtype(value.dtype, mx.floating) or not bool(mx.all(mx.isfinite(value))):
            raise ValueError("MLX_GEMMA_WEIGHT_VALUES")
    quant = config.get("quantization")
    if quant:

        def predicate(name, _module):
            if name + ".scales" not in weights:
                return False
            return quant.get(
                name, {k: quant[k] for k in ("bits", "group_size", "mode") if k in quant}
            )

        nn.quantize(
            model,
            bits=quant["bits"],
            group_size=quant["group_size"],
            mode=quant.get("mode", "affine"),
            class_predicate=predicate,
        )
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()
    return model, auxiliary


def prompt(messages):
    reserved = (
        "<bos>",
        "<eos>",
        "<pad>",
        "<|turn>",
        "<turn|>",
        "<|channel>",
        "<channel|>",
        "<|think|>",
        "<|image|>",
        "<|audio|>",
        "<|video|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|tool_response>",
        "<tool_response|>",
    )
    if any(token in message["content"] for message in messages for token in reserved):
        raise ValueError("MLX_GEMMA_RESERVED_FRAMING")
    result = "<bos>"
    for index, message in enumerate(messages):
        role = "model" if message["role"] == "assistant" else message["role"]
        continuation = role == "model" and index > 0 and messages[index - 1]["role"] == "assistant"
        if not continuation:
            result += "<|turn>" + role + "\n"
        result += message["content"].strip()
        result += "<turn|>\n"
    # Close the thought channel explicitly, matching the reviewed non-thinking
    # template branch while leaving repository templates inert.
    return result + "<|turn>model\n<|channel>thought\n<channel|>"


def stop_tokens(config, tokenizer):
    vocabulary = tokenizer.get_vocab()
    for token in ("<|channel>", "<channel|>"):
        if token not in vocabulary or tokenizer.encode(token, add_special_tokens=False) != [
            vocabulary[token]
        ]:
            raise ValueError("MLX_GEMMA_FRAMING")
    stops = {
        vocabulary[t]
        for t in ("<eos>", "<turn|>", "<|tool_call>", "<tool_call|>", "<|tool_response>")
        if t in vocabulary
    }
    declared = config.get("eos_token_id", sorted(stops))
    declared = [declared] if type(declared) is int else declared
    if (
        not isinstance(declared, list)
        or not declared
        or any(type(token) is not int or token not in stops for token in declared)
    ):
        raise ValueError("MLX_GEMMA_STOP_TOKENS")
    return stops


def generate(root, config, value, event):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    validate_config(config)
    tokenizer = gemma_decision.load_tokenizer(root, config["text_config"])
    stops = stop_tokens(config, tokenizer)
    tokens = tokenizer.encode(prompt(value["messages"]), add_special_tokens=False)
    if not 1 <= value["maxTokens"] <= MAX_OUTPUT or len(tokens) + value["maxTokens"] > min(
        MAX_CONTEXT, config["text_config"]["max_position_embeddings"]
    ):
        raise ValueError("MLX_GEMMA_CONTEXT_LIMIT")
    event("job.progress", {"stage": "loading-model", "progress": 0.05})
    model, _ = load_model(root, config)

    class TextGraph(nn.Module):
        def __init__(self, graph):
            super().__init__()
            self.graph = graph

        def __call__(self, ids, cache=None):
            logits = self.graph(ids, cache=cache).logits
            if not bool(mx.all(mx.isfinite(logits))):
                raise ValueError("MLX_GEMMA_LOGITS_NONFINITE")
            return logits

        def make_cache(self):
            return self.graph.make_cache()

        @property
        def layers(self):
            return self.graph.layers

    wrapped = TextGraph(model)
    tokenizer = TokenizerWrapper(tokenizer, eos_token_ids=stops)
    chunks, byte_count = [], 0
    event("job.state", {"state": "running"})
    try:
        for response in stream_generate(
            wrapped,
            tokenizer,
            prompt=tokens,
            max_tokens=value["maxTokens"],
            sampler=make_sampler(temp=value["temperature"], top_p=value["topP"]),
        ):
            byte_count += len(response.text.encode("utf-8"))
            if byte_count > 65536:
                raise ValueError("MLX_GEMMA_OUTPUT_LIMIT")
            if response.text:
                chunks.append(response.text)
                event("job.output.delta", {"channel": "text", "delta": response.text})
        return {"text": "".join(chunks), "backend": "mlx"}
    finally:
        del wrapped, model
        mx.clear_cache()
