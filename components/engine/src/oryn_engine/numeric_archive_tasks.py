"""Complete native encoder heads over the closed numeric archive reader."""

from __future__ import annotations

from oryn_engine import (
    numeric_coverage,
    numeric_dpt,
    numeric_layoutlmv2,
    numeric_mobilebert,
    numeric_owlvit,
    numeric_seq2seq,
    numeric_tapas,
)
from oryn_engine.model_metadata import metadata_file, validate_model_config

TASK_CONTRACTS = {
    task: task + "-v1"
    for task in (
        "fill-mask",
        "token-classification",
        "depth-estimation",
        "question-answering",
        "summarization",
        "translation",
        "zero-shot-object-detection",
        "table-question-answering",
        "document-question-answering",
        "video-classification",
        "visual-question-answering",
    )
}
ARCHITECTURES = {
    **numeric_coverage.ARCHITECTURES,
    **numeric_layoutlmv2.ARCHITECTURES,
    **numeric_seq2seq.ARCHITECTURES,
    **numeric_dpt.ARCHITECTURES,
    **numeric_owlvit.ARCHITECTURES,
    **numeric_tapas.ARCHITECTURES,
    ("question-answering", "bert"): "BertForQuestionAnswering",
    ("fill-mask", "bert"): "BertForMaskedLM",
    ("fill-mask", "distilbert"): "DistilBertForMaskedLM",
    ("fill-mask", "modernbert"): "ModernBertForMaskedLM",
    ("fill-mask", "mobilebert"): "MobileBertForMaskedLM",
    ("token-classification", "bert"): "BertForTokenClassification",
    ("token-classification", "xlm-roberta"): "XLMRobertaForTokenClassification",
}
TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "tokenizer.model",
    "spiece.model",
}


def supports(config, task):
    return (task, config.get("model_type")) in ARCHITECTURES


def selected(config, task, names):
    return (
        supports(config, task)
        and "pytorch_model.bin" in names
        and not any(n.endswith(".safetensors") for n in names)
        and not {"flax_model.msgpack", "tf_model.h5"} & set(names)
    )


def validate_config(config, task):
    if numeric_layoutlmv2.supports(config, task):
        numeric_layoutlmv2.validate_config(config)
        return
    if numeric_coverage.supports(config, task):
        numeric_coverage.validate_config(config, task)
        return
    for binding in [numeric_owlvit, numeric_tapas]:
        if binding.supports(config, task):
            binding.validate_config(config)
            return
    if numeric_dpt.supports(config, task):
        numeric_dpt.validate_config(config)
        return
    if numeric_seq2seq.supports(config, task):
        numeric_seq2seq.validate_config(config, task)
        return
    validate_model_config(config)
    if (
        not supports(config, task)
        or config.get("architectures") != [ARCHITECTURES[(task, config.get("model_type"))]]
        or config.get("auto_map")
        or config.get("quantization_config")
        or config.get("quantization")
        or config.get("is_decoder", False)
        or config.get("add_cross_attention", False)
    ):
        raise ValueError("NUMERIC_ARCHIVE_CONFIGURATION")
    distil = config["model_type"] == "distilbert"
    keys = (
        ("dim", "hidden_dim", "n_layers", "n_heads")
        if distil
        else ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads")
    )
    limits = ((8, 4096), (8, 32768), (1, 64), (1, 64))
    for key, (low, high) in zip(keys, limits, strict=True):
        if type(config.get(key)) is not int or not low <= config[key] <= high:
            raise ValueError("NUMERIC_ARCHIVE_CONFIGURATION_LIMIT")
    for key, low, high in (("vocab_size", 8, 300000), ("max_position_embeddings", 2, 8192)):
        if type(config.get(key)) is not int or not low <= config[key] <= high:
            raise ValueError("NUMERIC_ARCHIVE_CONFIGURATION_LIMIT")
    if config[keys[0]] % config[keys[3]] or config.get(
        "hidden_act", config.get("activation", "gelu")
    ) not in {"gelu", "gelu_new", "relu", "silu"}:
        raise ValueError("NUMERIC_ARCHIVE_CONFIGURATION")
    if config["model_type"] in {"bert", "xlm-roberta"} and (
        type(config.get("type_vocab_size", 2)) is not int
        or not 1 <= config.get("type_vocab_size", 2) <= 16
        or config.get("position_embedding_type", "absolute") != "absolute"
    ):
        raise ValueError("NUMERIC_ARCHIVE_CONFIGURATION")
    if distil and config.get("sinusoidal_pos_embds", False):
        raise ValueError("NUMERIC_ARCHIVE_CONFIGURATION")
    if config["model_type"] == "mobilebert":
        numeric_mobilebert.validate_config(config)
    if config["model_type"] == "modernbert":
        if config.get("hidden_activation", "gelu") not in {"gelu", "gelu_new", "relu", "silu"}:
            raise ValueError("NUMERIC_ARCHIVE_CONFIGURATION")
        for key in ("attention_bias", "mlp_bias", "norm_bias", "classifier_bias", "decoder_bias"):
            if key in config and type(config[key]) is not bool:
                raise ValueError("NUMERIC_ARCHIVE_CONFIGURATION")
        for key, default, low, high in (
            ("local_attention", 128, 2, 8192),
            ("global_attn_every_n_layers", 3, 1, 64),
        ):
            value = config.get(key, default)
            if type(value) is not int or not low <= value <= high:
                raise ValueError("NUMERIC_ARCHIVE_CONFIGURATION")
    if task == "token-classification":
        labels = config.get("id2label")
        if (
            not isinstance(labels, dict)
            or not 2 <= len(labels) <= 4096
            or set(labels) != {str(i) for i in range(len(labels))}
            or any(not isinstance(v, str) or not 1 <= len(v) <= 512 for v in labels.values())
            or config.get("num_labels", len(labels)) != len(labels)
        ):
            raise ValueError("NUMERIC_ARCHIVE_LABELS")
    if task == "question-answering" and (
        config.get("num_labels", 2) != 2
        or len(config.get("id2label", {"0": "start", "1": "end"})) != 2
    ):
        raise ValueError("NUMERIC_ARCHIVE_QA_HEAD")
    pad = config.get("pad_token_id", 0)
    if type(pad) is not int or not 0 <= pad < config["vocab_size"]:
        raise ValueError("NUMERIC_ARCHIVE_CONFIGURATION")


def expected_shapes(config, task):
    """Complete head/backbone shapes without importing an execution framework."""
    validate_config(config, task)
    if numeric_layoutlmv2.supports(config, task):
        return numeric_layoutlmv2.expected_shapes(config)
    if numeric_coverage.supports(config, task):
        return numeric_coverage.expected_shapes(config)
    for binding in [numeric_owlvit, numeric_tapas]:
        if binding.supports(config, task):
            return binding.expected_shapes(config)
    if numeric_seq2seq.supports(config, task):
        return numeric_seq2seq.expected_shapes(config)
    if numeric_dpt.supports(config, task):
        return numeric_dpt.expected_shapes(config)
    if config["model_type"] == "mobilebert":
        return numeric_mobilebert.expected_shapes(config)
    family = config["model_type"]
    hidden = config["dim" if family == "distilbert" else "hidden_size"]
    intermediate = config["hidden_dim" if family == "distilbert" else "intermediate_size"]
    layers = config["n_layers" if family == "distilbert" else "num_hidden_layers"]
    vocab, positions = config["vocab_size"], config["max_position_embeddings"]
    result = {}

    def linear(name, rows, columns, bias=True):
        result[name + ".weight"] = [rows, columns]
        if bias:
            result[name + ".bias"] = [rows]

    def norm(name, bias=True):
        result[name + ".weight"] = [hidden]
        if bias:
            result[name + ".bias"] = [hidden]

    if family == "modernbert":
        norm_bias = config.get("norm_bias", False)
        result["model.embeddings.tok_embeddings.weight"] = [vocab, hidden]
        for name in ("model.embeddings.norm", "model.final_norm", "head.norm"):
            norm(name, norm_bias)
        linear("head.dense", hidden, hidden, config.get("classifier_bias", False))
        linear("decoder", vocab, hidden, config.get("decoder_bias", True))
        for i in range(layers):
            prefix = f"model.layers.{i}"
            if i:
                norm(prefix + ".attn_norm", norm_bias)
            norm(prefix + ".mlp_norm", norm_bias)
            linear(prefix + ".attn.Wqkv", 3 * hidden, hidden, config.get("attention_bias", False))
            linear(prefix + ".attn.Wo", hidden, hidden, config.get("attention_bias", False))
            linear(prefix + ".mlp.Wi", 2 * intermediate, hidden, config.get("mlp_bias", False))
            linear(prefix + ".mlp.Wo", hidden, intermediate, config.get("mlp_bias", False))
        return result
    prefix = "roberta" if family == "xlm-roberta" else family
    result[prefix + ".embeddings.word_embeddings.weight"] = [vocab, hidden]
    result[prefix + ".embeddings.position_embeddings.weight"] = [positions, hidden]
    norm(prefix + ".embeddings.LayerNorm")
    if family == "distilbert":
        for i in range(layers):
            prefix = f"distilbert.transformer.layer.{i}"
            for name in ("q_lin", "k_lin", "v_lin", "out_lin"):
                linear(prefix + ".attention." + name, hidden, hidden)
            for name in ("sa_layer_norm", "output_layer_norm"):
                norm(prefix + "." + name)
            linear(prefix + ".ffn.lin1", intermediate, hidden)
            linear(prefix + ".ffn.lin2", hidden, intermediate)
        linear("vocab_transform", hidden, hidden)
        norm("vocab_layer_norm")
        linear("vocab_projector", vocab, hidden)
    else:
        result[prefix + ".embeddings.token_type_embeddings.weight"] = [
            config.get("type_vocab_size", 2),
            hidden,
        ]
        for i in range(layers):
            layer = f"{prefix}.encoder.layer.{i}"
            for name in ("query", "key", "value"):
                linear(layer + ".attention.self." + name, hidden, hidden)
            linear(layer + ".attention.output.dense", hidden, hidden)
            norm(layer + ".attention.output.LayerNorm")
            linear(layer + ".intermediate.dense", intermediate, hidden)
            linear(layer + ".output.dense", hidden, intermediate)
            norm(layer + ".output.LayerNorm")
        if task == "fill-mask":
            linear("cls.predictions.transform.dense", hidden, hidden)
            norm("cls.predictions.transform.LayerNorm")
            linear("cls.predictions.decoder", vocab, hidden)
            result["cls.predictions.bias"] = [vocab]
        elif task == "question-answering":
            linear("qa_outputs", 2, hidden)
        else:
            linear("classifier", len(config["id2label"]), hidden)
    return result


def validate_archive(config, task, metadata):
    if numeric_layoutlmv2.supports(config, task):
        return numeric_layoutlmv2.validate_archive(config, metadata), None
    shapes = expected_shapes(config, task)
    tensors = metadata.get("tensors", {})
    if (
        metadata.get("format") != "torch-numeric-zip-v1"
        or metadata.get("byteOrder") != "little"
        or not isinstance(tensors, dict)
    ):
        raise ValueError("NUMERIC_ARCHIVE_METADATA")
    position_name = {"bert": "bert", "xlm-roberta": "roberta", "mobilebert": "mobilebert"}.get(
        config["model_type"]
    )
    positions = (
        {position_name + ".embeddings.position_ids": config["max_position_embeddings"]}
        if position_name
        else {}
    )
    if config["model_type"] == "owlvit":
        positions = {
            "owlvit.text_model.embeddings.position_ids": config["text_config"][
                "max_position_embeddings"
            ],
            "owlvit.vision_model.embeddings.position_ids": (
                config["vision_config"]["image_size"] // config["vision_config"]["patch_size"]
            )
            ** 2
            + 1,
        }
    if config["model_type"] == "tapas":
        positions = {"tapas.embeddings.position_ids": config["max_position_embeddings"]}
    if numeric_coverage.supports(config, task):
        buffer = numeric_coverage.position_buffer(config)
        positions = dict([buffer]) if buffer else {}
        # Exact historical layer names map one-to-one onto the native graph.
        renamed = {}
        for name in tensors:
            target = numeric_coverage.native_name(config, name)
            if target in renamed.values() or target != name and target in tensors:
                raise ValueError("NUMERIC_ARCHIVE_GRAPH_INCOMPLETE")
            renamed[name] = target
        if set(renamed.values()) - set(positions) != set(shapes):
            raise ValueError("NUMERIC_ARCHIVE_GRAPH_INCOMPLETE")
        shapes = {source: shapes[target] for source, target in renamed.items() if target in shapes}
    positions = {name: length for name, length in positions.items() if name in tensors}
    for name, length in positions.items():
        shapes[name] = [1, length]
    if set(tensors) != set(shapes):
        raise ValueError("NUMERIC_ARCHIVE_GRAPH_INCOMPLETE")
    for name, value in tensors.items():
        if (
            not isinstance(value, dict)
            or value.get("shape") != shapes[name]
            or any(type(x) is not int for x in value["shape"])
            or value.get("dtype")
            not in ({"I64"} if name in positions else {"F32", "F16", "BF16", "F64"})
        ):
            raise ValueError("NUMERIC_ARCHIVE_GRAPH_INCOMPLETE")
    return shapes, positions or None


def _load_native(root, task):
    import torch
    import transformers as tf

    from oryn_engine.generated_capabilities import normalized_transformers_config
    from oryn_engine.numeric_torch_archive import inspect_archive, read_weights

    config = normalized_transformers_config(metadata_file(root / "config.json"), task)
    validate_config(config, task)
    if numeric_layoutlmv2.supports(config, task):
        return numeric_layoutlmv2.load_native(root, config), config
    path = root / "pytorch_model.bin"
    if path.is_symlink() or not path.is_file():
        raise ValueError("NUMERIC_ARCHIVE_SCOPE")
    with path.open("rb") as stream:

        def read_range(start, length):
            stream.seek(start)
            return stream.read(length)

        metadata = inspect_archive(read_range, path.stat().st_size)
    shapes, legacy_position = validate_archive(config, task, metadata)
    native_config = tf.AutoConfig.for_model(**config)
    native_config._attn_implementation = "eager"
    if config["model_type"] == "modernbert":
        native_config.reference_compile = False
    # Native constructors preserve deterministic non-persistent position/rotary
    # buffers. Header geometry is checked before constructing this complete graph.
    cls = getattr(tf, ARCHITECTURES[(task, config["model_type"])])
    model = cls(native_config).float()
    native_shapes = {k: list(v.shape) for k, v in model.state_dict().items()}
    if native_shapes != expected_shapes(config, task):
        raise ValueError("NUMERIC_ARCHIVE_NATIVE_GRAPH_CHANGED")
    state = read_weights(path, expected_shapes=shapes, maximum_bytes=path.stat().st_size)
    for name, length in (legacy_position or {}).items():
        expected = torch.arange(length).expand(1, -1)
        if not torch.equal(state.pop(name), expected):
            raise ValueError("NUMERIC_ARCHIVE_POSITION_BUFFER")
    if numeric_coverage.supports(config, task):
        state = {numeric_coverage.native_name(config, k): v for k, v in state.items()}
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    for names in aliases.values():
        if any(not torch.equal(state[names[0]], state[name]) for name in names[1:]):
            raise ValueError("NUMERIC_ARCHIVE_TIED_WEIGHTS")
    for name, value in state.items():
        # Native device kernels must see canonical dense parameters, including
        # archives whose physical storage order is transposed.
        state[name] = value.float().contiguous()
        if not bool(torch.isfinite(state[name]).all()):
            raise ValueError("NUMERIC_ARCHIVE_NONFINITE")
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    return model, config


def load_components(root, task, backend):
    tokenizer_config = root / "tokenizer_config.json"
    if tokenizer_config.exists() or tokenizer_config.is_symlink():
        # Recheck before native construction, including already-installed models.
        metadata_file(tokenizer_config)
    import transformers as tf

    if backend not in {"cpu", "mps", "cuda"}:
        raise ValueError("NUMERIC_ARCHIVE_BACKEND")
    model, config = _load_native(root, task)
    if config["model_type"] == "mobilebert":
        numeric_mobilebert.prepare_device(model, backend)
    if numeric_dpt.supports(config, task):
        options = numeric_dpt.processor_options(
            config, metadata_file(root / "preprocessor_config.json")
        )
        return model.eval().to(backend), tf.DPTImageProcessor(**options), config
    if numeric_owlvit.supports(config, task):
        options = numeric_owlvit.processor_options(
            config, metadata_file(root / "preprocessor_config.json")
        )
        tokenizer = tf.CLIPTokenizer.from_pretrained(root, local_files_only=True)
        numeric_owlvit.prepare_tokenizer(config, tokenizer)
        processor = tf.OwlViTProcessor(tf.OwlViTImageProcessor(**options), tokenizer)
        return model.eval().to(backend), processor, config
    if numeric_layoutlmv2.supports(config, task):
        processor = numeric_layoutlmv2.load_processor(root, config)
        return model.eval().to(backend), processor, config
    if numeric_coverage.supports(config, task):
        processor = numeric_coverage.load_processor(root, config, task)
        numeric_coverage.prepare(model, processor, config)
        return model.eval().to(backend), processor, config
    if numeric_tapas.supports(config, task):
        tokenizer = tf.TapasTokenizer.from_pretrained(root, local_files_only=True)
        if len(tokenizer) > config["vocab_size"] or tokenizer.pad_token_id != config.get(
            "pad_token_id", 0
        ):
            raise ValueError("NUMERIC_TAPAS_TOKENIZER")
        return model.eval().to(backend), tokenizer, config
    tokenizer = tf.AutoTokenizer.from_pretrained(
        root, local_files_only=True, trust_remote_code=False
    )
    tokenizer.chat_template = None
    if not tokenizer.is_fast:
        raise ValueError("NUMERIC_ARCHIVE_TOKENIZER")
    if numeric_seq2seq.supports(config, task):
        if config["model_type"] == "m2m_100":
            numeric_seq2seq.validate_tokenizer_metadata(
                config, metadata_file(root / "tokenizer_config.json")
            )
        numeric_seq2seq.prepare(model, tokenizer, config)
    return model.eval().to(backend), tokenizer, config


def execute(
    model, tokenizer, config, task, value, backend, image=None, input_path=None, original=None
):
    import torch

    from oryn_engine.runners import _normalize_transformers_task_output

    if numeric_layoutlmv2.supports(config, task):
        from oryn_engine.coverage_tasks import document_answer

        return document_answer(model, tokenizer, config, image, value, backend)
    if numeric_coverage.supports(config, task):
        return numeric_coverage.execute(
            model, tokenizer, config, task, value, backend, image, input_path, original
        )
    if numeric_dpt.supports(config, task):
        return numeric_dpt.execute(model, tokenizer, config, image, backend)
    if numeric_owlvit.supports(config, task):
        return numeric_owlvit.execute(model, tokenizer, config, image, value, backend)
    if numeric_tapas.supports(config, task):
        from oryn_engine.coverage_tasks import table_answer

        return table_answer(model, tokenizer, config, value, backend)

    if numeric_seq2seq.supports(config, task):
        return numeric_seq2seq.execute(model, tokenizer, config, task, value, backend)
    if task == "question-answering":
        if not value["question"].strip() or not value["context"].strip():
            raise ValueError("NUMERIC_ARCHIVE_INPUT_EMPTY")
        encoded = tokenizer(
            value["question"],
            value["context"],
            return_tensors="pt",
            truncation=False,
            return_offsets_mapping=True,
        )
    else:
        encoded = tokenizer(
            value["text"],
            return_tensors="pt",
            truncation=False,
            **({"return_offsets_mapping": True} if task == "token-classification" else {}),
        )
    limit = min(
        512,
        config["max_position_embeddings"]
        - (config["pad_token_id"] + 1 if config["model_type"] == "xlm-roberta" else 0),
    )
    if not 1 <= encoded["input_ids"].shape[1] <= limit:
        raise ValueError("NUMERIC_ARCHIVE_INPUT_LIMIT")
    token_metadata = None
    if task == "question-answering":
        offsets = encoded.pop("offset_mapping")[0].tolist()
        sequences = encoded.sequence_ids(0)
        if any(
            seq == 1 and not 0 <= a <= b <= len(value["context"])
            for (a, b), seq in zip(offsets, sequences, strict=True)
        ):
            raise ValueError("NUMERIC_ARCHIVE_TOKEN_OFFSETS")
        token_metadata = {"offsets": offsets, "sequence_ids": sequences}
    elif task == "token-classification":
        offsets = encoded.pop("offset_mapping")[0].tolist()
        if any(not 0 <= a <= b <= len(value["text"]) for a, b in offsets):
            raise ValueError("NUMERIC_ARCHIVE_TOKEN_OFFSETS")
        token_metadata = {"offsets": offsets}
    elif (
        tokenizer.mask_token_id is None
        or int((encoded["input_ids"] == tokenizer.mask_token_id).sum()) != 1
    ):
        raise ValueError("NUMERIC_ARCHIVE_MASK_TOKEN")
    allowed = {"input_ids", "attention_mask"}
    if config["model_type"] in {"bert", "xlm-roberta", "mobilebert"}:
        allowed.add("token_type_ids")
    encoded = {k: v.to(backend) for k, v in encoded.items() if k in allowed}
    output = model(**encoded)
    logits = (
        [output.start_logits, output.end_logits]
        if task == "question-answering"
        else [output.logits]
    )
    if not all(bool(torch.isfinite(part).all()) for part in logits):
        raise ValueError("NUMERIC_ARCHIVE_NONFINITE")
    return _normalize_transformers_task_output(
        task,
        {"topK": 5, "maxAnswerLength": 64, **value},
        output,
        tokenizer,
        model,
        encoded,
        None,
        token_metadata,
    )
