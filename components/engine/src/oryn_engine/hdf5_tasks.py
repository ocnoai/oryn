"""Reviewed TensorFlow weight names bound to complete native PyTorch task graphs."""

from __future__ import annotations

import re

from oryn_engine.generated_capabilities import normalized_transformers_config
from oryn_engine.model_metadata import metadata_file, validate_model_config

ARCHITECTURES = {
    ("table-question-answering", "tapas"): "TapasForQuestionAnswering",
    ("zero-shot-image-classification", "clip"): "CLIPModel",
    ("image-feature-extraction", "vit"): "ViTModel",
    ("text-classification", "roberta"): "RobertaForSequenceClassification",
    ("text-classification", "bert"): "BertForSequenceClassification",
    ("image-to-text", "blip"): "BlipForConditionalGeneration",
    ("visual-question-answering", "blip"): "BlipForQuestionAnswering",
    ("image-segmentation", "segformer"): "SegformerForSemanticSegmentation",
    ("question-answering", "electra"): "ElectraForQuestionAnswering",
    ("translation", "marian"): "MarianMTModel",
}
FORMAT_ONLY_TASKS = frozenset({"question-answering", "translation"})


def supports(config, task):
    config = normalized_transformers_config(config, task)
    return (task, config.get("model_type")) in ARCHITECTURES


def validate_config(config, task):
    config = normalized_transformers_config(config, task)
    validate_model_config(config)
    if (
        not supports(config, task)
        or config.get("architectures") != [ARCHITECTURES.get((task, config.get("model_type")))]
        or config.get("quantization_config")
        or config.get("quantization")
    ):
        raise ValueError("HDF5_CONFIGURATION_UNSUPPORTED")
    family = config["model_type"]
    if family == "bert":
        from oryn_engine.flax_weights import validate_classifier_config

        validate_classifier_config(config)
        if type(config.get("pad_token_id", 0)) is not int:
            raise ValueError("HDF5_CONFIGURATION_UNSUPPORTED")
        return
    if family == "marian":
        from oryn_engine.hdf5_marian import validate_config as validate_marian

        validate_marian(config)
        return
    if family == "electra":
        from oryn_engine.flax_weights import validate_classifier_config

        validate_classifier_config(
            config
            | {
                "architectures": ["BertForSequenceClassification"],
                "id2label": {"0": "start", "1": "end"},
            }
        )
        width = config.get("embedding_size")
        if (
            type(width) is not int
            or not 8 <= width <= 4096
            or config.get("hidden_act", "gelu") != "gelu"
            or config.get("num_labels", len(config.get("id2label", {"0": "start", "1": "end"})))
            != 2
        ):
            raise ValueError("HDF5_CONFIGURATION_UNSUPPORTED")
        return
    if family == "segformer":
        from oryn_engine.hdf5_segformer import validate_config as validate_segformer

        validate_segformer(config)
        return
    if family == "blip":
        from oryn_engine.hdf5_blip import validate_config as validate_blip

        validate_blip(config)
        return
    if family == "clip":
        from oryn_engine.flax_weights import validate_clip_config

        validate_clip_config(config)
        return
    if family == "roberta":
        from oryn_engine.flax_weights import validate_classifier_config

        # The shared encoder geometry/label rules apply to both native graphs.
        validate_classifier_config(config | {"architectures": ["BertForSequenceClassification"]})
        if (
            len(config["id2label"]) < 2
            or config.get("hidden_act", "gelu") != "gelu"
            or type(config.get("pad_token_id")) is not int
            or config["max_position_embeddings"] <= config["pad_token_id"] + 2
        ):
            raise ValueError("HDF5_CONFIGURATION_UNSUPPORTED")
        return
    if family == "vit":
        for name, low, high in (
            ("hidden_size", 8, 4096),
            ("intermediate_size", 8, 16384),
            ("num_hidden_layers", 1, 48),
            ("num_attention_heads", 1, 64),
            ("image_size", 4, 1024),
            ("patch_size", 1, 64),
        ):
            value = config.get(name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError("HDF5_CONFIGURATION_LIMIT")
        if (
            config["hidden_size"] % config["num_attention_heads"]
            or config["image_size"] % config["patch_size"]
            or (config["image_size"] // config["patch_size"]) ** 2 > 4096
            or config.get("num_channels", 3) != 3
            or config.get("hidden_act", "gelu") != "gelu"
        ):
            raise ValueError("HDF5_CONFIGURATION_UNSUPPORTED")
        return
    for name, low, high in (
        ("hidden_size", 16, 2048),
        ("intermediate_size", 16, 8192),
        ("num_hidden_layers", 1, 48),
        ("num_attention_heads", 1, 32),
        ("vocab_size", 8, 131072),
        ("max_position_embeddings", 128, 2048),
        ("max_num_rows", 2, 256),
        ("max_num_columns", 2, 256),
        ("num_aggregation_labels", 0, 4),
    ):
        value = config.get(name)
        if type(value) is not int or not low <= value <= high:
            raise ValueError("HDF5_CONFIGURATION_LIMIT")
    if config["hidden_size"] % config["num_attention_heads"]:
        raise ValueError("HDF5_CONFIGURATION_UNSUPPORTED")
    types = config.get("type_vocab_sizes")
    if (
        not isinstance(types, list)
        or len(types) != 7
        or any(type(v) is not int or not 2 <= v <= 512 for v in types)
        or config.get("type_vocab_size", types) != types
    ):
        raise ValueError("HDF5_CONFIGURATION_UNSUPPORTED")
    # These affect inference, not only training losses.
    for name in ("temperature", "aggregation_temperature"):
        value = config.get(name, 1.0)
        if type(value) not in {int, float} or not 0 < value <= 100:
            raise ValueError("HDF5_CONFIGURATION_UNSUPPORTED")


def tapas_name(name):
    heads = {
        "compute_column_logits/column_output/column_output_bias:0": "column_output_bias",
        "compute_column_logits/column_output/column_output_weights:0": "column_output_weights",
        "compute_token_logits/output/output_bias:0": "output_bias",
        "compute_token_logits/output/output_weights:0": "output_weights",
    }
    if name in heads:
        return heads[name], False
    match = re.fullmatch(
        r"(tapas|aggregation_classifier)/tf_tapas_for_question_answering(?:_\d+)?/\1/(.+):0", name
    )
    if not match:
        raise ValueError("HDF5_WEIGHT_NAME_UNSUPPORTED")
    prefix, leaf = match.groups()
    leaf = re.sub(r"layer_\._(\d+)", r"layer/\1", leaf)
    parts = leaf.split("/")
    transpose = parts[-1] == "kernel"
    parts[-1] = {"kernel": "weight", "embeddings": "weight", "gamma": "weight", "beta": "bias"}.get(
        parts[-1], parts[-1]
    )
    return prefix + "." + ".".join(parts), transpose


def native_name(source, family, shape):
    if family == "marian":
        from oryn_engine.hdf5_marian import native_name as marian_name

        return marian_name(source, shape)
    if family == "segformer":
        from oryn_engine.hdf5_segformer import native_name as segformer_name

        return segformer_name(source, shape)
    if family == "blip":
        from oryn_engine.hdf5_blip import native_name as blip_name

        return blip_name(source, shape)
    if family == "tapas":
        name, transpose = tapas_name(source)
        if transpose and len(shape) != 2:
            raise ValueError("HDF5_WEIGHT_SHAPE_UNSUPPORTED")
        return name, (1, 0) if transpose else None
    patterns = {
        "clip": r"clip/tfclip_model(?:_\d+)?/clip/(.+):0",
        "vit": r"vit/tf_vi_t_model(?:_\d+)?/vit/(.+):0",
        "roberta": (
            r"(roberta|classifier)/tf_roberta_for_sequence_classification(?:_\d+)?/\1/(.+):0"
        ),
        "electra": r"(electra|qa_outputs)/tf_electra_for_question_answering(?:_\d+)?/\1/(.+):0",
        "bert": r"(bert|classifier)/tf_bert_for_sequence_classification(?:_\d+)?/\1/(.+):0",
    }
    match = re.fullmatch(patterns[family], source)
    if not match:
        raise ValueError("HDF5_WEIGHT_NAME_UNSUPPORTED")
    leaf = match.group(match.lastindex)
    leaf = re.sub(r"(layers?)_\._(\d+)", r"\1/\2", leaf)
    parts = leaf.split("/")
    kernel = parts[-1] == "kernel"
    parts[-1] = {"kernel": "weight", "embeddings": "weight", "gamma": "weight", "beta": "bias"}.get(
        parts[-1], parts[-1]
    )
    name = ".".join(parts)
    if family == "vit":
        name = re.sub(r"^encoder\.layer\.(\d+)\.", r"layers.\1.", name)
        for old, new in (
            ("attention.attention.query.", "attention.q_proj."),
            ("attention.attention.key.", "attention.k_proj."),
            ("attention.attention.value.", "attention.v_proj."),
            ("attention.output.dense.", "attention.o_proj."),
            ("intermediate.dense.", "mlp.fc1."),
            ("output.dense.", "mlp.fc2."),
        ):
            name = name.replace(old, new)
    if family in {"roberta", "electra", "bert"}:
        name = match.group(1) + "." + name
    permutation = None
    if kernel:
        convolution = {
            "clip": "vision_model.embeddings.patch_embedding.weight",
            "vit": "embeddings.patch_embeddings.projection.weight",
        }.get(family)
        if name == convolution and len(shape) == 4:
            permutation = (3, 2, 0, 1)
        elif len(shape) == 2:
            permutation = (1, 0)
        else:
            raise ValueError("HDF5_WEIGHT_SHAPE_UNSUPPORTED")
    return name, permutation


def load_native(root, task):
    import torch
    from transformers import (
        BertConfig,
        BertForSequenceClassification,
        BlipConfig,
        BlipForConditionalGeneration,
        BlipForQuestionAnswering,
        CLIPConfig,
        CLIPModel,
        ElectraConfig,
        ElectraForQuestionAnswering,
        MarianConfig,
        MarianMTModel,
        RobertaConfig,
        RobertaForSequenceClassification,
        SegformerConfig,
        SegformerForSemanticSegmentation,
        TapasConfig,
        TapasForQuestionAnswering,
        ViTConfig,
        ViTModel,
    )

    from oryn_engine.numeric_hdf5 import read_weights

    config = normalized_transformers_config(metadata_file(root / "config.json"), task)
    validate_config(config, task)
    family = config["model_type"]
    config_cls, model_cls = {
        "tapas": (TapasConfig, TapasForQuestionAnswering),
        "clip": (CLIPConfig, CLIPModel),
        "vit": (ViTConfig, ViTModel),
        "roberta": (RobertaConfig, RobertaForSequenceClassification),
        "bert": (BertConfig, BertForSequenceClassification),
        "electra": (ElectraConfig, ElectraForQuestionAnswering),
        "marian": (MarianConfig, MarianMTModel),
        "segformer": (SegformerConfig, SegformerForSemanticSegmentation),
        "blip": (
            BlipConfig,
            BlipForConditionalGeneration if task == "image-to-text" else BlipForQuestionAnswering,
        ),
    }[family]
    native = config_cls.from_dict(config)
    native._attn_implementation = "eager"
    arrays = read_weights(root / "tf_model.h5")
    mapped = {}
    for source, array in arrays.items():
        name, permutation = native_name(source, family, array.shape)
        if name in mapped:
            raise ValueError("HDF5_WEIGHT_GRAPH_MISMATCH")
        if permutation:
            array = array.transpose(permutation)
        # TF stores its scalar CLIP temperature as a one-element variable.
        if family == "clip" and name == "logit_scale" and array.shape == (1,):
            array = array.reshape(())
        mapped[name] = array
    options = {}
    if family == "vit":
        # CLS embeddings need no pooler. Admit either complete native layout;
        # a partially serialized pooler still fails the exact graph check.
        options["add_pooling_layer"] = any(k.startswith("pooler.") for k in mapped)
    with torch.device("meta"):
        model = model_cls(native, **options)
    expected = model.state_dict()
    state = {}
    for name, array in mapped.items():
        if (
            name in state
            or name not in expected
            or tuple(array.shape) != tuple(expected[name].shape)
        ):
            raise ValueError("HDF5_WEIGHT_GRAPH_MISMATCH")
        state[name] = torch.from_numpy(array.copy())
    if family == "marian":
        source = "model.shared.weight"
        aliases = {
            "model.encoder.embed_tokens.weight",
            "model.decoder.embed_tokens.weight",
            "lm_head.weight",
        }
        if source not in state or aliases & set(state):
            raise ValueError("HDF5_WEIGHT_GRAPH_INCOMPLETE")
        state.update({name: state[source] for name in aliases})
    if family == "blip":
        # Native BLIP exposes the one serialized prediction bias under two keys.
        alias = "text_decoder.cls.predictions.decoder.bias"
        source = "text_decoder.cls.predictions.bias"
        if alias in state or source not in state:
            raise ValueError("HDF5_WEIGHT_GRAPH_INCOMPLETE")
        state[alias] = state[source]
    if family == "segformer":
        counter = "decode_head.batch_norm.num_batches_tracked"
        if counter in state or "decode_head.batch_norm.running_var" not in state:
            raise ValueError("HDF5_WEIGHT_GRAPH_INCOMPLETE")
        if not bool((state["decode_head.batch_norm.running_var"] >= 0).all()):
            raise ValueError("HDF5_WEIGHT_VARIANCE_INVALID")
        # This native training-only counter has no TF counterpart; learned
        # normalization statistics remain mandatory and are never synthesized.
        state[counter] = torch.tensor(0, dtype=torch.long)
    if set(state) != set(expected):
        raise ValueError("HDF5_WEIGHT_GRAPH_INCOMPLETE")
    model.load_state_dict(state, strict=True, assign=True)
    if family == "marian":
        from oryn_engine.hdf5_marian import prepare_native

        prepare_native(model, config)
    if family == "blip":
        from oryn_engine.hdf5_blip import prepare_native

        prepare_native(model, config)
    if family in {"roberta", "electra", "bert"}:
        embeddings = getattr(model, family).embeddings
        embeddings.position_ids = torch.arange(config["max_position_embeddings"]).expand((1, -1))
        embeddings.token_type_ids = torch.zeros_like(embeddings.position_ids)
    if family == "clip":
        for embeddings in (model.text_model.embeddings, model.vision_model.embeddings):
            embeddings.position_ids = torch.arange(
                embeddings.position_embedding.num_embeddings
            ).expand((1, -1))
    if any(b.is_meta for b in model.buffers()):
        raise ValueError("HDF5_WEIGHT_GRAPH_INCOMPLETE")
    return model.eval()


def vit_processor_options(metadata, config):
    from oryn_engine.trocr_tasks import processor_options

    return processor_options(metadata, {"encoder": config})


def load_components(root, task, backend):
    from transformers import (
        BertTokenizer,
        CLIPImageProcessor,
        CLIPProcessor,
        CLIPTokenizer,
        ElectraTokenizer,
        RobertaTokenizer,
        SegformerImageProcessor,
        TapasTokenizer,
        ViTImageProcessor,
    )

    config = normalized_transformers_config(metadata_file(root / "config.json"), task)
    validate_config(config, task)
    if backend not in {"cpu", "mps", "cuda"}:
        raise ValueError("HDF5_BACKEND_UNSUPPORTED")
    family = config["model_type"]
    if (root / "tokenizer_config.json").is_file():
        validate_model_config(metadata_file(root / "tokenizer_config.json"))
    common = {"local_files_only": True, "trust_remote_code": False}
    if family == "marian":
        from oryn_engine.hdf5_marian import load_tokenizer

        tokenizer = load_tokenizer(root, config)
    elif family == "segformer":
        from oryn_engine.hdf5_segformer import processor_options

        tokenizer = SegformerImageProcessor(
            **processor_options(metadata_file(root / "preprocessor_config.json"), config)
        )
    elif family == "blip":
        from oryn_engine.hdf5_blip import load_processor

        tokenizer = load_processor(root, config)
    elif family == "vit":
        tokenizer = ViTImageProcessor(
            **vit_processor_options(metadata_file(root / "preprocessor_config.json"), config)
        )
    else:
        cls = {
            "tapas": TapasTokenizer,
            "clip": CLIPTokenizer,
            "roberta": RobertaTokenizer,
            "electra": ElectraTokenizer,
            "bert": BertTokenizer,
        }[family]
        tokenizer = cls.from_pretrained(root, **common)
        text = config["text_config"] if family == "clip" else config
        if len(tokenizer) != text["vocab_size"]:
            raise ValueError("HDF5_TOKENIZER_VOCABULARY")
        if family == "bert" and tokenizer.pad_token_id != config.get("pad_token_id", 0):
            raise ValueError("HDF5_TOKENIZER_SPECIAL_TOKENS")
        tokenizer.chat_template = None
        if family == "roberta":
            for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
                if getattr(tokenizer, key) != text.get(
                    key, {"bos_token_id": 0, "eos_token_id": 2, "pad_token_id": 1}[key]
                ):
                    raise ValueError("HDF5_TOKENIZER_SPECIAL_TOKENS")
        if family == "clip":
            from oryn_engine.flax_weights import clip_processor_options, validate_clip_tokenizer

            validate_clip_tokenizer(tokenizer, config)
            options = clip_processor_options(
                metadata_file(root / "preprocessor_config.json"),
                config["vision_config"]["image_size"],
            )
            tokenizer = CLIPProcessor(
                tokenizer=tokenizer, image_processor=CLIPImageProcessor(**options)
            )
    model = load_native(root, task).to(backend)
    return model, tokenizer, config


def classify(model, tokenizer, config, value, backend):
    from oryn_engine.runners import _normalize_transformers_task_output

    inputs = tokenizer(value["text"], return_tensors="pt", truncation=False)
    # RoBERTa positions start after its padding index, including special tokens.
    limit = min(512, config["max_position_embeddings"] - config["pad_token_id"] - 1)
    if not 1 <= inputs["input_ids"].shape[1] <= limit:
        raise ValueError("HDF5_INPUT_TOKEN_LIMIT")
    inputs = {k: v.to(backend) for k, v in inputs.items()}
    return _normalize_transformers_task_output(
        "text-classification", value, model(**inputs), tokenizer, model, inputs, None
    )


def answer_question(model, tokenizer, config, value, backend):
    from oryn_engine.runners import _normalize_transformers_task_output

    if not value["question"].strip() or not value["context"].strip():
        raise ValueError("HDF5_QA_INPUT_EMPTY")
    processed = tokenizer(
        value["question"],
        value["context"],
        truncation=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    if not 1 <= processed["input_ids"].shape[1] <= min(512, config["max_position_embeddings"]):
        raise ValueError("HDF5_INPUT_TOKEN_LIMIT")
    metadata = {
        "offsets": processed.pop("offset_mapping")[0].tolist(),
        "sequence_ids": processed.sequence_ids(),
    }
    inputs = {k: v.to(backend) for k, v in processed.items()}
    return _normalize_transformers_task_output(
        "question-answering", value, model(**inputs), tokenizer, model, inputs, None, metadata
    )
