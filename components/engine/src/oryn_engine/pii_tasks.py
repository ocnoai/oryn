"""Native bidirectional Qwen3 PII token labels with complete heads and decoder."""

from __future__ import annotations

from oryn_engine import bioes
from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

TYPES = (
    "private_person",
    "private_email",
    "private_phone",
    "private_address",
    "private_url",
    "private_date",
    "account_number",
    "secret",
    "other_pii",
)
LABELS = ["O"] + [f"{tag}-{kind}" for kind in TYPES for tag in "BIES"]


def supports(config, task):
    return (
        task == "token-classification"
        and config.get("model_type") == "pii_masking"
        and config.get("architectures") == ["PiiMaskingModel"]
    )


def normalized_config(config, task):
    if supports(config, task) and config.get("auto_map") == {
        "AutoConfig": "modeling_pii_masking.PiiMaskingConfig",
        "AutoModel": "modeling_pii_masking.PiiMaskingModel",
    }:
        config = {k: v for k, v in config.items() if k != "auto_map"}
        validate_config(config)
    return config


def validate_config(config):
    validate_model_config(config)
    backbone = config.get("backbone")
    if (
        not supports(config, "token-classification")
        or not isinstance(backbone, dict)
        or backbone.get("model_type") != "qwen3"
        or backbone.get("architectures") != ["Qwen3Model"]
        or backbone.get("is_causal") is not False
        or backbone.get("use_cache") is not False
        or config.get("num_token_labels") != 37
        or config.get("hidden_size") != backbone.get("hidden_size")
        or type(config.get("max_seq_len")) is not int
        or not 1 <= config["max_seq_len"] <= 4096
        or config.get("id2label", dict(enumerate(LABELS)))
        not in (dict(enumerate(LABELS)), {str(i): label for i, label in enumerate(LABELS)})
        or config.get("label2id", {v: i for i, v in enumerate(LABELS)})
        != {v: i for i, v in enumerate(LABELS)}
        or any(
            part.get(key) is not None
            for part in (config, backbone)
            for key in ("quantization", "quantization_config")
        )
    ):
        raise ValueError("PII_CONFIGURATION_UNSUPPORTED")
    for key, low, high in (
        ("hidden_size", 8, 8192),
        ("intermediate_size", 8, 65536),
        ("num_hidden_layers", 1, 80),
        ("num_attention_heads", 1, 128),
        ("num_key_value_heads", 1, 128),
        ("head_dim", 4, 256),
        ("vocab_size", 32, 262144),
        ("max_position_embeddings", 4096, 1048576),
    ):
        if type(backbone.get(key)) is not int or not low <= backbone[key] <= high:
            raise ValueError("PII_CONFIGURATION_GEOMETRY")
    if (
        backbone["num_attention_heads"] % backbone["num_key_value_heads"]
        or backbone["head_dim"] % 2
        or backbone.get("attention_bias", False) is not False
        or backbone.get("attention_dropout", 0) != 0
        or backbone.get("hidden_act", "silu") != "silu"
        or backbone.get("sliding_window") is not None
        or backbone.get("use_sliding_window", False) is not False
        or backbone.get("rope_scaling") is not None
        or backbone.get("layer_types") != ["full_attention"] * backbone["num_hidden_layers"]
    ):
        raise ValueError("PII_CONFIGURATION_GRAPH")
    rope = backbone.get("rope_parameters")
    if (
        not isinstance(rope, dict)
        or set(rope) != {"rope_type", "rope_theta"}
        or rope["rope_type"] != "default"
    ):
        raise ValueError("PII_CONFIGURATION_POSITIONS")
    for key in ("viterbi_b_bias", "viterbi_e_bias"):
        if type(config.get(key)) not in (int, float) or abs(config[key]) > 1e6:
            raise ValueError("PII_CONFIGURATION_DECODER")


def native_model(config):
    import torch
    from transformers import Qwen3Config, Qwen3Model

    native = Qwen3Config.from_dict(dict(config["backbone"]))
    native._attn_implementation = "eager"
    model = torch.nn.Module()
    model.add_module("backbone", Qwen3Model(native))
    for layer in model.backbone.layers:
        layer.self_attn.is_causal = False
    model.add_module("token_cls_head", torch.nn.Linear(config["hidden_size"], len(LABELS)))
    model.add_module("sensitivity_head", torch.nn.Linear(config["hidden_size"], 1))
    decoder = torch.nn.Module()
    decoder.register_buffer("b_bias", torch.tensor([float(config["viterbi_b_bias"])]))
    decoder.register_buffer("e_bias", torch.tensor([float(config["viterbi_e_bias"])]))
    decoder.register_buffer("transition_mask", bioes.transition_mask(TYPES))
    model.add_module("viterbi", decoder)
    return model


def validate_weight_shapes(config, shapes):
    import torch

    validate_config(config)
    with torch.device("meta"):
        expected = {k: tuple(v.shape) for k, v in native_model(config).state_dict().items()}
    if shapes != expected:
        raise ValueError("PII_WEIGHTS_INCOMPLETE")


def validate_metadata(name, metadata, config):
    validate_model_config(metadata)
    if (
        name != "tokenizer_config.json"
        or metadata.get("tokenizer_class") not in {"PreTrainedTokenizerFast", "TokenizersBackend"}
        or metadata.get("add_prefix_space", False) is not False
        or metadata.get("add_bos_token", False) is not False
        or metadata.get("bos_token") is not None
        or metadata.get("eos_token") != metadata.get("pad_token")
        or not isinstance(metadata.get("eos_token"), str)
    ):
        raise ValueError("PII_TOKENIZER_METADATA")


def load_tokenizer(root, config):
    from transformers import TokenizersBackend

    metadata = metadata_file(root / "tokenizer_config.json")
    validate_metadata("tokenizer_config.json", metadata, config)
    path = root / "tokenizer.json"
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 32 * 1024**2:
        raise ValueError("PII_TOKENIZER_GRAPH")
    tokenizer = TokenizersBackend(
        tokenizer_file=str(path), eos_token=metadata["eos_token"], pad_token=metadata["pad_token"]
    )
    tokenizer.chat_template = None
    vocab = tokenizer.get_vocab()
    if (
        not 32 <= len(vocab) <= config["backbone"]["vocab_size"]
        or set(vocab.values()) != set(range(len(vocab)))
        or tokenizer.eos_token_id != config["backbone"]["eos_token_id"]
    ):
        raise ValueError("PII_TOKENIZER_VOCABULARY")
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
            raise ValueError("PII_WEIGHTS_DUPLICATE")
        for name, tensor in part.items():
            if name == "viterbi.transition_mask":
                if tensor.dtype != torch.bool or not torch.equal(
                    tensor, bioes.transition_mask(TYPES)
                ):
                    raise ValueError("PII_DECODER_TRANSITIONS")
                state[name] = tensor
            else:
                if not tensor.is_floating_point() or not bool(torch.isfinite(tensor.float()).all()):
                    raise ValueError("PII_WEIGHTS_INVALID")
                if name in {"viterbi.b_bias", "viterbi.e_bias"} and abs(float(tensor)) > 1e6:
                    raise ValueError("PII_DECODER_BIAS")
                state[name] = tensor.float()
    with init_empty_weights(include_buffers=False):
        model = native_model(config)
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(backend).eval(), tokenizer, config


def execute(model, tokenizer, config, value, backend):
    import torch

    from oryn_engine.giga_embeddings import attention_mask

    text = value["text"]
    batch = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    ids = batch["input_ids"]
    if not 1 <= ids.shape[1] <= config["max_seq_len"]:
        raise ValueError("PII_TOKEN_LIMIT")
    offsets = batch["offset_mapping"][0].tolist()
    if any(not 0 <= start <= end <= len(text) for start, end in offsets):
        raise ValueError("PII_TOKENIZER_OFFSETS")
    mask = batch["attention_mask"].to(backend)
    hidden = model.backbone(
        input_ids=ids.to(backend), attention_mask=attention_mask(mask), use_cache=False
    ).last_hidden_state.float()
    logits = model.token_cls_head(hidden)[0].cpu()
    sensitivity = model.sensitivity_head(
        (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
    )
    if not bool(torch.isfinite(logits).all() and torch.isfinite(sensitivity).all()):
        raise ValueError("PII_OUTPUT_INVALID")
    path = bioes.decode(
        logits, model.viterbi.transition_mask, model.viterbi.b_bias, model.viterbi.e_bias
    )
    tokens = tokenizer.convert_ids_to_tokens(ids[0].tolist())
    return {
        "tokens": [
            {
                "token": token,
                "labelId": label,
                "label": LABELS[label],
                "score": float(logits[i, label].sigmoid()),
                "start": start,
                "end": end,
            }
            for i, (token, label, (start, end)) in enumerate(
                zip(tokens, path, offsets, strict=True)
            )
            if 0 <= start < end <= len(text)
        ]
    }
