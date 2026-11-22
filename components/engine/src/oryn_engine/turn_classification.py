"""Owned text end-of-utterance scores from complete native causal Llama graphs.

The frozen publisher graph uses special user/assistant tokens and ASCII punctuation
normalization. Newer Qwen turn-detector documentation does not describe this graph.
"""

from __future__ import annotations

import string

from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

FAMILY = "oryn_llama_turn"
ARCHITECTURE = "OrynLlamaTurnForClassification"
MAX_TOKENS = 512
LABELS = ("continue", "end-of-utterance")
TEMPLATE = (
    "{% for message in messages %}{{'<|im_start|>' + '<|' + message['role'] + '|>'"
    " + message['content'] + '<|im_end|>'}}{% endfor %}"
)
CONTROLS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|user|>", "<|assistant|>")
FIELDS = set(
    [
        "_attn_implementation_autoset",
        "_name_or_path",
        "architectures",
        "attention_bias",
        "attention_dropout",
        "bos_token_id",
        "eos_token_id",
        "head_dim",
        "hidden_act",
        "hidden_size",
        "initializer_range",
        "intermediate_size",
        "is_llama_config",
        "max_position_embeddings",
        "mlp_bias",
        "model_type",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "pretraining_tp",
        "rms_norm_eps",
        "rope_interleaved",
        "rope_scaling",
        "rope_theta",
        "tie_word_embeddings",
        "transformers_version",
        "use_cache",
        "vocab_size",
    ]
)


def supports(config, task):
    return (
        task == "text-classification"
        and config.get("model_type") == FAMILY
        and config.get("architectures") == [ARCHITECTURE]
    )


def normalized_config(config, task):
    if (
        task == "text-classification"
        and config.get("model_type") == "llama"
        and config.get("architectures") == ["LlamaForCausalLM"]
        and config.get("is_llama_config") is True
    ):
        config = dict(config)
        marker = config.pop("_attn_implementation_autoset", None)
        if marker is not True:
            raise ValueError("TURN_NATIVE_IDENTITY")
        config.update(model_type=FAMILY, architectures=[ARCHITECTURE])
        validate_config(config)
    return config


def validate_config(config):
    validate_model_config(config)
    if not supports(config, "text-classification") or set(config) - FIELDS:
        raise ValueError("TURN_CONFIGURATION")
    fixed = dict(
        attention_bias=False,
        attention_dropout=0,
        bos_token_id=0,
        eos_token_id=0,
        hidden_act="silu",
        is_llama_config=True,
        mlp_bias=False,
        pretraining_tp=1,
        rms_norm_eps=1e-5,
        rope_interleaved=False,
        rope_scaling=None,
        rope_theta=100000,
        tie_word_embeddings=True,
        use_cache=True,
    )
    if any(config.get(k) != v for k, v in fixed.items()):
        raise ValueError("TURN_NATIVE_GRAPH")
    for key, low, high in [
        ("hidden_size", 16, 4096),
        ("intermediate_size", 16, 16384),
        ("num_hidden_layers", 1, 64),
        ("num_attention_heads", 1, 64),
        ("num_key_value_heads", 1, 64),
        ("head_dim", 8, 256),
        ("vocab_size", 261, 65536),
        ("max_position_embeddings", 512, 8192),
    ]:
        if type(config.get(key)) is not int or not low <= config[key] <= high:
            raise ValueError("TURN_GEOMETRY")
    if (
        config["hidden_size"] != config["num_attention_heads"] * config["head_dim"]
        or config["num_attention_heads"] % config["num_key_value_heads"]
        or config["head_dim"] % 2
    ):
        raise ValueError("TURN_ATTENTION_GEOMETRY")


def native_model(config):
    from transformers import LlamaConfig, LlamaForCausalLM

    validate_config(config)
    clean = {
        k: v
        for k, v in config.items()
        if k
        not in {
            "architectures",
            "model_type",
            "is_llama_config",
            "_name_or_path",
            "_attn_implementation_autoset",
        }
    }
    native = LlamaConfig(**clean)
    native._attn_implementation = "eager"
    return LlamaForCausalLM(native)


def validate_weight_shapes(config, shapes):
    import torch

    with torch.device("meta"):
        expected = {k: tuple(t.shape) for k, t in native_model(config).state_dict().items()}
    if "lm_head.weight" not in shapes:
        expected.pop("lm_head.weight")
    if shapes != expected:
        raise ValueError("TURN_WEIGHTS_INCOMPLETE")


def validate_metadata(name, metadata, config):
    validate_model_config(metadata)
    if (
        name != "tokenizer_config.json"
        or metadata.get("tokenizer_class") != "GPT2Tokenizer"
        or metadata.get("chat_template") != TEMPLATE
        or metadata.get("add_prefix_space") is not False
        or any(
            metadata.get(k) != CONTROLS[0]
            for k in ("bos_token", "eos_token", "pad_token", "unk_token")
        )
        or metadata.get("model_max_length") != config["max_position_embeddings"]
    ):
        raise ValueError("TURN_TOKENIZER_METADATA")
    expected = dict(
        zip((0, 1, 2, config["vocab_size"] - 2, config["vocab_size"] - 1), CONTROLS, strict=True)
    )
    added = metadata.get("added_tokens_decoder")
    if not isinstance(added, dict) or any(
        not isinstance(added.get(str(i)), dict)
        or added[str(i)].get("content") != token
        or added[str(i)].get("special") is not True
        or added[str(i)].get("normalized") is not False
        for i, token in expected.items()
    ):
        raise ValueError("TURN_TOKENIZER_CONTROLS")


def load_tokenizer(root, config):
    from tokenizers import Tokenizer
    from transformers import TokenizersBackend

    validate_metadata(
        "tokenizer_config.json", metadata_file(root / "tokenizer_config.json"), config
    )
    path = root / "tokenizer.json"
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 8 * 1024**2:
        raise ValueError("TURN_TOKENIZER_GRAPH")
    vocab = Tokenizer.from_file(str(path)).get_vocab()
    if (
        len(vocab) != config["vocab_size"]
        or set(vocab.values()) != set(range(config["vocab_size"]))
        or any(len(t.encode()) > 4096 for t in vocab)
    ):
        raise ValueError("TURN_VOCABULARY")
    tokenizer = TokenizersBackend(
        tokenizer_file=str(path),
        bos_token=CONTROLS[0],
        eos_token=CONTROLS[0],
        pad_token=CONTROLS[0],
        unk_token=CONTROLS[0],
    )
    tokenizer.chat_template = None
    if tokenizer.get_vocab() != vocab or any(
        tokenizer.encode(token, add_special_tokens=False) != [i]
        for i, token in zip((0, 1, 2, len(vocab) - 2, len(vocab) - 1), CONTROLS, strict=True)
    ):
        raise ValueError("TURN_TOKENIZER_CONTROLS")
    return tokenizer


def load_components(root, config, backend):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    validate_weight_shapes(config, preflight_safetensors(root))
    tokenizer = load_tokenizer(root, config)
    state = {}
    for path in sorted(root.glob("*.safetensors")):
        part = load_file(path)
        if state.keys() & part.keys():
            raise ValueError("TURN_DUPLICATE_WEIGHT")
        for name, tensor in part.items():
            if not tensor.is_floating_point() or not bool(torch.isfinite(tensor.float()).all()):
                raise ValueError("TURN_NUMERIC_WEIGHT")
            state[name] = tensor.float()
    if "lm_head.weight" in state and not torch.equal(
        state["lm_head.weight"], state["model.embed_tokens.weight"]
    ):
        raise ValueError("TURN_TIED_WEIGHT")
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    with init_empty_weights(include_buffers=False):
        model = native_model(config)
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    return model.to(backend).eval(), tokenizer, config


def prompt(text):
    if any(marker in text for marker in ("<|", "|>")):
        raise ValueError("TURN_RESERVED_CONTROL")
    normalized = " ".join(
        text.translate(str.maketrans("", "", string.punctuation.replace("'", ""))).lower().split()
    )
    if not normalized:
        raise ValueError("TURN_EMPTY_TEXT")
    return "<|im_start|><|user|>" + normalized


def execute(model, tokenizer, config, value, backend):
    import torch

    ids = tokenizer.encode(prompt(value["text"]), add_special_tokens=False)
    if not 3 <= len(ids) <= MAX_TOKENS:
        raise ValueError("TURN_TOKEN_LIMIT")
    inputs = torch.tensor([ids], dtype=torch.long, device=backend)
    logits = model(input_ids=inputs, use_cache=False, logits_to_keep=1).logits.float()
    if tuple(logits.shape) != (1, 1, config["vocab_size"]) or not bool(
        torch.isfinite(logits).all()
    ):
        raise ValueError("TURN_OUTPUT_INVALID")
    probability = float(logits[0, 0].softmax(-1)[2].cpu())
    scores = (1 - probability, probability)
    order = sorted(range(2), key=lambda i: (-scores[i], i))[: min(value["topK"], 2)]
    return {"predictions": [{"label": LABELS[i], "labelId": i, "score": scores[i]} for i in order]}
