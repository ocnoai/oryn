"""Complete native GPT-OSS MLX graphs and owned text-only Harmony framing."""

from datetime import UTC, datetime

from oryn_engine.model_metadata import metadata_file, preflight_safetensors

MAX_CONTEXT = 8192
MARKERS = ("<|start|>", "<|end|>", "<|message|>", "<|channel|>", "<|return|>", "<|call|>")


def validate_config(config):
    from oryn_engine.runtime_packs.mlx_lm_worker import validate_config as validate_mlx

    validate_mlx(config, {"gpt_oss"})
    if config.get("architectures") != ["GptOssForCausalLM"]:
        raise ValueError("MLX_GPT_OSS_ARCHITECTURE")
    for name, low, high in (
        ("hidden_size", 32, 8192),
        ("intermediate_size", 32, 32768),
        ("vocab_size", 32, 262144),
        ("num_hidden_layers", 2, 128),
        ("num_local_experts", 2, 256),
        ("num_experts_per_tok", 1, 32),
        ("num_attention_heads", 1, 128),
        ("num_key_value_heads", 1, 128),
        ("head_dim", 8, 512),
        ("sliding_window", 2, 8192),
        ("max_position_embeddings", 128, 1048576),
    ):
        if type(config.get(name)) is not int or not low <= config[name] <= high:
            raise ValueError("MLX_GPT_OSS_GEOMETRY")
    if (
        config["num_attention_heads"] % config["num_key_value_heads"]
        or config["head_dim"] % 2
        or config["num_experts_per_tok"] > config["num_local_experts"]
        or config.get("experts_per_token", config["num_experts_per_tok"])
        != config["num_experts_per_tok"]
        or config.get("attention_bias", True) is not True
        or config.get("attention_dropout", 0) != 0
        or config.get("hidden_act", "silu") != "silu"
        or config.get("swiglu_limit", 7.0) != 7.0
        or config.get("tie_word_embeddings", False) is not False
    ):
        raise ValueError("MLX_GPT_OSS_GRAPH")
    layers = config.get("layer_types")
    if (
        not isinstance(layers, list)
        or len(layers) != config["num_hidden_layers"]
        or set(layers) != {"sliding_attention", "full_attention"}
    ):
        raise ValueError("MLX_GPT_OSS_ATTENTION")
    rope = config.get("rope_scaling")
    if rope is not None and (
        not isinstance(rope, dict)
        or rope.get("rope_type") != "yarn"
        or not set(rope)
        <= {
            "rope_type",
            "factor",
            "original_max_position_embeddings",
            "beta_fast",
            "beta_slow",
            "truncate",
        }
        or type(rope.get("truncate", False)) is not bool
    ):
        raise ValueError("MLX_GPT_OSS_ROPE")


def validate_weight_shapes(config, shapes):
    from mlx.utils import tree_flatten
    from mlx_lm.models.gpt_oss import Model, ModelArgs

    from oryn_engine.runtime_packs.mlx_lm_worker import validate_quantized_layers

    validate_config(config)
    model = Model(ModelArgs.from_dict(config))
    expected = {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}
    validate_quantized_layers(model, config, shapes)
    quantization = config.get("quantization") or {}
    for name in list(expected):
        prefix = name.removesuffix(".weight")
        if name.endswith(".weight") and prefix + ".scales" in shapes:
            if not quantization:
                raise ValueError("MLX_GPT_OSS_QUANTIZATION_MISSING")
            expected[name] = shapes[name]
            expected[prefix + ".scales"] = shapes[prefix + ".scales"]
            if quantization.get(prefix, quantization).get("mode", "affine") == "affine":
                expected[prefix + ".biases"] = shapes[prefix + ".biases"]
    if shapes != expected:
        raise ValueError("MLX_GPT_OSS_WEIGHTS_INCOMPLETE")
    return model


def load_model(root, config):
    import mlx.core as mx
    import mlx.nn as nn

    model = validate_weight_shapes(config, preflight_safetensors(root))
    weights = {}
    for path in sorted(root.glob("*.safetensors")):
        part = mx.load(str(path))
        if part.keys() & weights.keys():
            raise ValueError("MLX_GPT_OSS_WEIGHT_COLLISION")
        weights.update(part)
    quant = config.get("quantization") or {}
    for name, value in weights.items():
        prefix = name.rsplit(".", 1)[0]
        settings = quant.get(prefix, quant)
        mode = settings.get("mode", "affine") if isinstance(settings, dict) else "affine"
        if name.endswith(".weight") and prefix + ".scales" in weights:
            valid = value.dtype == mx.uint32
        elif name.endswith(".scales") and mode == "mxfp4":
            # MX scales are exponent bytes; they are not affine float scales.
            valid = value.dtype == mx.uint8 and bool(mx.all(value != 255))
        else:
            valid = mx.issubdtype(value.dtype, mx.floating) and bool(mx.all(mx.isfinite(value)))
        if not valid:
            raise ValueError("MLX_GPT_OSS_WEIGHT_VALUES")
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
    return model


def load_tokenizer(root, config):
    from transformers import TokenizersBackend

    metadata = metadata_file(root / "tokenizer_config.json")
    if any(metadata.get(k) for k in ("auto_map", "custom_pipelines")):
        raise ValueError("MLX_GPT_OSS_TOKENIZER_CODE")
    for name, expected in (
        ("eos_token", "<|return|>"),
        ("pad_token", "<|endoftext|>"),
        ("bos_token", "<|startoftext|>"),
    ):
        token = metadata.get(name)
        if isinstance(token, dict):
            token = token.get("content")
        if token != expected:
            raise ValueError("MLX_GPT_OSS_TOKENIZER_METADATA")
    path = root / "tokenizer.json"
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 32 * 1024**2:
        raise ValueError("MLX_GPT_OSS_TOKENIZER_GRAPH")
    tokenizer = TokenizersBackend(
        tokenizer_file=str(path),
        eos_token="<|return|>",
        pad_token="<|endoftext|>",
        bos_token="<|startoftext|>",
    )
    tokenizer.chat_template = None
    vocab = tokenizer.get_vocab()
    # The native output matrix has padded vocabulary rows. Retain those learned
    # rows, but never turn an out-of-vocabulary generated ID into silent text.
    if not 0 <= config["vocab_size"] - len(vocab) <= 2048 or set(vocab.values()) != set(
        range(len(vocab))
    ):
        raise ValueError("MLX_GPT_OSS_VOCABULARY")
    if any(len(token.encode()) > 4096 for token in vocab):
        raise ValueError("MLX_GPT_OSS_TOKEN_SIZE")
    for token in (*MARKERS, "<|startoftext|>", "<|endoftext|>"):
        if token not in vocab or tokenizer.encode(token, add_special_tokens=False) != [
            vocab[token]
        ]:
            raise ValueError("MLX_GPT_OSS_FRAMING")
    if (
        config.get("eos_token_id") != vocab["<|return|>"]
        or config.get("pad_token_id") != vocab["<|endoftext|>"]
    ):
        raise ValueError("MLX_GPT_OSS_STOP_TOKENS")
    return tokenizer


def prompt(messages, *, date=None):
    date = date or datetime.now(UTC).strftime("%Y-%m-%d")
    if not messages or any("<|" in m["content"] or "|>" in m["content"] for m in messages):
        raise ValueError("MLX_GPT_OSS_RESERVED_TOKEN")
    result = (
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
        f"Knowledge cutoff: 2024-06\nCurrent date: {date}\n\nReasoning: medium\n\n"
        "# Valid channels: analysis, commentary, final. Channel must be included for every message."
        "<|end|>"
    )
    if messages[0]["role"] == "system":
        result += (
            "<|start|>developer<|message|># Instructions\n\n"
            + messages[0]["content"]
            + "\n\n<|end|>"
        )
        messages = messages[1:]
    for message in messages:
        if message["role"] not in {"user", "assistant"}:
            raise ValueError("MLX_GPT_OSS_MESSAGE_ROLE")
        role = message["role"] + ("<|channel|>final" if message["role"] == "assistant" else "")
        result += f"<|start|>{role}<|message|>{message['content']}<|end|>"
    return result + "<|start|>assistant"


def final_text(tokenizer, tokens):
    """Decode only the declared final channel; tool/unknown channels fail closed."""
    vocab = tokenizer.get_vocab()
    if not tokens or any(type(t) is not int or not 0 <= t < len(vocab) for t in tokens):
        raise ValueError("MLX_GPT_OSS_OUTPUT_TOKENS")
    text = "<|start|>assistant" + tokenizer.decode(tokens, skip_special_tokens=False)
    for _ in range(8):
        prefix = "<|start|>assistant<|channel|>"
        if not text.startswith(prefix) or "<|message|>" not in text:
            break
        header, body = text[len(prefix) :].split("<|message|>", 1)
        if header == "analysis" and "<|end|>" in body:
            analysis, text = body.split("<|end|>", 1)
            if "<|" in analysis:
                break
            continue
        if header == "final":
            body = body.removesuffix("<|return|>")
            if body.strip() and "<|" not in body and len(body.encode()) <= 65536:
                return body
        break
    raise ValueError("MLX_GPT_OSS_FINAL_CHANNEL_MISSING")


def generate(root, config, value, event):
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    validate_config(config)
    tokenizer = load_tokenizer(root, config)
    tokens = tokenizer.encode(prompt(value["messages"]), add_special_tokens=False)
    if len(tokens) + value["maxTokens"] > min(MAX_CONTEXT, config["max_position_embeddings"]):
        raise ValueError("MLX_GPT_OSS_CONTEXT_LIMIT")
    event("job.progress", {"stage": "loading-model", "progress": 0.05})
    model = load_model(root, config)
    wrapper = TokenizerWrapper(tokenizer, eos_token_ids={tokenizer.eos_token_id})
    output = []
    event("job.state", {"state": "running"})
    try:
        for response in stream_generate(
            model,
            wrapper,
            prompt=tokens,
            max_tokens=value["maxTokens"],
            sampler=make_sampler(temp=value["temperature"], top_p=value["topP"]),
        ):
            output.append(int(response.token))
        text = final_text(tokenizer, output)
        event("job.output.delta", {"channel": "text", "delta": text})
        return {"text": text, "backend": "mlx"}
    finally:
        del model
        mx.clear_cache()
