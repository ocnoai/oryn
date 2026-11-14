"""Complete native contextual embedding graphs with explicit query/document semantics."""

from __future__ import annotations

from oryn_engine import qwen_translation
from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

FAMILY = "pplx_contextual_qwen3_5"
MAX_BATCH = 4
MAX_CONTEXT = 4096
METADATA_NAMES = ("tokenizer_config.json",)
ALIASES = {
    "AutoConfig": "configuration_pplx_contextual.PplxContextualConfig",
    "AutoModel": "modeling_pplx_contextual.PplxContextualModel",
}


def supports(config, task):
    return task == "feature-extraction" and config.get("model_type") == FAMILY


def normalized_config(config, task):
    if supports(config, task):
        if config.get("auto_map") == ALIASES:
            config = {key: value for key, value in config.items() if key != "auto_map"}
        validate_config(config)
    return config


def graph_config(config):
    return config | {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}


def validate_config(config):
    validate_model_config(config)
    if (
        config.get("model_type") != FAMILY
        or config.get("architectures") != ["PplxContextualModel"]
        or config.get("query_prefix") != "[Q] "
        or config.get("document_prefix") != "[D] "
        or config.get("boundary_marker") != "<|chunk_sep|>"
        or type(config.get("embedding_dim")) is not int
        or not 16 <= config["embedding_dim"] <= 4096
        or any(
            type(config.get(k)) is not int or not 128 <= config[k] <= 262144
            for k in ("query_length", "document_length")
        )
        or not isinstance(config.get("text_config"), dict)
        or config["text_config"].get("is_causal") is not False
        or config["text_config"].get("use_cache") is not False
    ):
        raise ValueError("CONTEXTUAL_EMBEDDING_CONFIGURATION")
    qwen_translation.validate_config(graph_config(config))


def validate_metadata(name, value, config):
    validate_model_config(value)
    if (
        name != "tokenizer_config.json"
        or value.get("tokenizer_class")
        not in {"Qwen2Tokenizer", "Qwen2TokenizerFast", "TokenizersBackend"}
        or value.get("pad_token") != "<|endoftext|>"
        or value.get("eos_token") != "<|im_end|>"
    ):
        raise ValueError("CONTEXTUAL_EMBEDDING_TOKENIZER")


def tokenizer(root, config):
    from transformers import TokenizersBackend

    validate_metadata(
        "tokenizer_config.json", metadata_file(root / "tokenizer_config.json"), config
    )
    result = TokenizersBackend(tokenizer_file=str(root / "tokenizer.json"))
    if not {"<|endoftext|>", "<|im_end|>"} <= result.get_vocab().keys():
        raise ValueError("CONTEXTUAL_EMBEDDING_TOKENIZER")
    result.pad_token = "<|endoftext|>"
    result.eos_token = "<|im_end|>"
    result.padding_side = "right"
    result.chat_template = None
    vocabulary = config["text_config"]["vocab_size"]
    gap = vocabulary - len(result)
    # The original vocabulary pads its embedding matrix to a 256-row boundary.
    # All rows remain required weights. Generation EOS settings are inert for
    # this encoder; tokenization uses the serialized tokenizer's own markers.
    if (
        not 0 <= gap < 256
        or gap
        and vocabulary % 256
        or set(result.get_vocab().values()) != set(range(len(result)))
        or result.eos_token_id is None
        or result.pad_token_id is None
    ):
        raise ValueError("CONTEXTUAL_EMBEDDING_VOCABULARY")
    query_id = result.get_vocab().get(config["query_prefix"])
    if (
        type(query_id) is not int
        or result.convert_ids_to_tokens(query_id) != config["query_prefix"]
    ):
        raise ValueError("CONTEXTUAL_EMBEDDING_QUERY_PREFIX")
    return result


def _native(config):
    import torch
    from accelerate import init_empty_weights
    from transformers import Qwen3_5Model

    validate_config(config)

    class NativeContextual(Qwen3_5Model):
        def __init__(self, native):
            super().__init__(native)
            self.contextual_projection = torch.nn.Linear(
                native.text_config.hidden_size, config["embedding_dim"], bias=False
            )

    with init_empty_weights(include_buffers=False):
        model = NativeContextual(qwen_translation.native_config(graph_config(config)))
    return model


def validate_weight_shapes(config, shapes):
    model = _native(config)
    if shapes != {key: tuple(value.shape) for key, value in model.state_dict().items()}:
        raise ValueError("CONTEXTUAL_EMBEDDING_WEIGHTS_INCOMPLETE")
    return model


def load_components(root, config, backend):
    import torch
    from safetensors.torch import load_file

    model = validate_weight_shapes(config, preflight_safetensors(root))
    native_tokenizer = tokenizer(root, config)
    state = {}
    for path in sorted(root.glob("*.safetensors")):
        values = load_file(path, device="cpu")
        if state.keys() & values.keys() or any(
            not value.is_floating_point() or not bool(torch.isfinite(value).all())
            for value in values.values()
        ):
            raise ValueError("CONTEXTUAL_EMBEDDING_WEIGHT_VALUES")
        state.update({key: value.float() for key, value in values.items()})
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(backend).eval(), native_tokenizer, config


def prepare_inputs(native_tokenizer, config, value):
    import torch

    kind, contextual = value.get("inputType", "document"), value.get("contextual", False)
    texts = value["texts"]
    if (
        kind not in {"query", "document"}
        or type(contextual) is not bool
        or kind == "query"
        and contextual
        or not 1 <= len(texts) <= MAX_BATCH
    ):
        raise ValueError("CONTEXTUAL_EMBEDDING_INPUT_MODE")
    documents = [texts] if contextual else [[text] for text in texts]
    options = dict(
        padding=True,
        truncation=False,
        split_special_tokens=True,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    if kind == "query":
        batch = dict(native_tokenizer(texts, **options))
        prefix = torch.full(
            (len(texts), 1),
            native_tokenizer.convert_tokens_to_ids(config["query_prefix"]),
            dtype=torch.long,
        )
        batch["input_ids"] = torch.cat((prefix, batch["input_ids"]), dim=1)
        batch["attention_mask"] = torch.cat(
            (torch.ones_like(prefix), batch["attention_mask"]), dim=1
        )
    else:
        joined, spans = [], []
        for chunks in documents:
            text, intervals = config["document_prefix"], []
            for index, chunk in enumerate(chunks):
                if index:
                    text += config["boundary_marker"]
                intervals.append((len(text), len(text) + len(chunk)))
                text += chunk
            joined.append(text)
            spans.append(intervals)
        batch = dict(native_tokenizer(joined, return_offsets_mapping=True, **options))
        offsets = batch.pop("offset_mapping")
        token_spans = torch.zeros((len(documents), max(map(len, documents)), 2), dtype=torch.long)
        for row, intervals in enumerate(spans):
            starts, ends = offsets[row].unbind(-1)
            valid = batch["attention_mask"][row].bool() & (ends > starts)
            for column, (start, end) in enumerate(intervals):
                indexes = (valid & (starts < end) & (ends > start)).nonzero().flatten()
                if not indexes.numel():
                    raise ValueError("CONTEXTUAL_EMBEDDING_EMPTY_CHUNK")
                token_spans[row, column] = torch.tensor([indexes[0], indexes[-1] + 1])
        batch["chunk_token_spans"] = token_spans
    if batch["input_ids"].shape[-1] > min(
        MAX_CONTEXT, config[kind + "_length"], config["text_config"]["max_position_embeddings"]
    ):
        raise ValueError("CONTEXTUAL_EMBEDDING_TOKEN_LIMIT")
    return batch


def execute(model, native_tokenizer, config, value, backend):
    import torch
    import torch.nn.functional as functional

    values = {
        key: tensor.to(backend)
        for key, tensor in prepare_inputs(native_tokenizer, config, value).items()
    }
    with torch.inference_mode():
        hidden = model(
            input_ids=values["input_ids"], attention_mask=values["attention_mask"], use_cache=False
        ).last_hidden_state.float()
        if value.get("inputType", "document") == "query":
            mask = values["attention_mask"].unsqueeze(-1).float()
            pooled = ((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).unsqueeze(1)
        else:
            prefix = torch.cat(
                (hidden.new_zeros(hidden.shape[0], 1, hidden.shape[-1]), hidden.cumsum(1)), dim=1
            )
            starts, ends = values["chunk_token_spans"].unbind(-1)
            rows = torch.arange(hidden.shape[0], device=hidden.device).unsqueeze(1)
            pooled = (prefix[rows, ends] - prefix[rows, starts]) / (ends - starts).clamp_min(
                1
            ).unsqueeze(-1)
        projected = functional.linear(pooled, model.contextual_projection.weight.float())
        if not bool(torch.isfinite(projected).all()):
            raise ValueError("CONTEXTUAL_EMBEDDING_OUTPUT_INVALID")
        vectors = torch.round(torch.tanh(projected) * 127).clamp(-128, 127)
        if value["normalize"]:
            vectors = functional.normalize(vectors, p=2, dim=-1)
        vectors = vectors.reshape(len(value["texts"]), config["embedding_dim"])
    return {
        "embeddings": vectors.cpu().tolist(),
        "dimensions": config["embedding_dim"],
        "normalized": value["normalize"],
    }
