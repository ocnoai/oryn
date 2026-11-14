"""Complete native table, document, video and VQA graphs over numeric archives.

These families already have reviewed safetensors task behavior. This binding adds
only exact numeric-archive graphs: closed configurations, every learned tensor
derived without an execution framework, and exact historical layer names.
"""

from __future__ import annotations

import re

from oryn_engine import coverage_tasks, numeric_seq2seq
from oryn_engine.model_metadata import validate_model_config

ARCHITECTURES = {
    ("table-question-answering", "bart"): "BartForConditionalGeneration",
    ("document-question-answering", "layoutlmv3"): "LayoutLMv3ForQuestionAnswering",
    ("video-classification", "vivit"): "VivitForVideoClassification",
    ("visual-question-answering", "vilt"): "ViltForQuestionAnswering",
}
PROCESSOR_TASKS = {
    "document-question-answering",
    "video-classification",
    "visual-question-answering",
}
TOKENIZER_FREE_TASKS = {"video-classification"}
POSITIONS = {
    "layoutlmv3": "layoutlmv3.embeddings.position_ids",
    "vilt": "vilt.embeddings.text_embeddings.position_ids",
}
# Pre-Transformers-5 ViT layer names; the native graph is unchanged.
LEGACY_VIT = (
    (r"encoder\.layer\.(\d+)\.attention\.attention\.query\.", r"layers.\1.attention.q_proj."),
    (r"encoder\.layer\.(\d+)\.attention\.attention\.key\.", r"layers.\1.attention.k_proj."),
    (r"encoder\.layer\.(\d+)\.attention\.attention\.value\.", r"layers.\1.attention.v_proj."),
    (r"encoder\.layer\.(\d+)\.attention\.output\.dense\.", r"layers.\1.attention.o_proj."),
    (r"encoder\.layer\.(\d+)\.intermediate\.dense\.", r"layers.\1.mlp.fc1."),
    (r"encoder\.layer\.(\d+)\.output\.dense\.", r"layers.\1.mlp.fc2."),
    (r"encoder\.layer\.(\d+)\.layernorm_(before|after)\.", r"layers.\1.layernorm_\2."),
)


def supports(config, task):
    return (task, config.get("model_type")) in ARCHITECTURES


def normalized_config(config, task):
    """Map ViViT's historical video_size field onto the native geometry fields."""
    if (
        task != "video-classification"
        or not isinstance(config, dict)
        or config.get("model_type") != "vivit"
        or "video_size" not in config
    ):
        return config
    size = config["video_size"]
    if (
        "num_frames" in config
        or "image_size" in config
        or not isinstance(size, list)
        or len(size) != 3
        or any(type(v) is not int for v in size)
        or size[1] != size[2]
    ):
        raise ValueError("NUMERIC_COVERAGE_VIDEO_GEOMETRY")
    result = {k: v for k, v in config.items() if k != "video_size"}
    return result | {"num_frames": size[0], "image_size": size[1]}


def _bounded(config, limits, code):
    for key, (low, high) in limits.items():
        if type(config.get(key)) is not int or not low <= config[key] <= high:
            raise ValueError(code)


def _common(config, task, architecture=None):
    validate_model_config(config)
    if (
        not supports(config, task)
        or config.get("architectures")
        != [architecture or ARCHITECTURES[(task, config["model_type"])]]
        or config.get("auto_map")
        or config.get("quantization")
        or config.get("quantization_config")
        or config.get("is_decoder", False)
        or config.get("add_cross_attention", False)
    ):
        raise ValueError("NUMERIC_COVERAGE_CONFIGURATION")


def _labels(config, low=2):
    labels = config.get("id2label")
    if (
        not isinstance(labels, dict)
        or not low <= len(labels) <= 65536
        or set(labels) != {str(i) for i in range(len(labels))}
        or any(not isinstance(v, str) or not v.strip() or len(v) > 1024 for v in labels.values())
        or config.get("num_labels", len(labels)) != len(labels)
    ):
        raise ValueError("NUMERIC_COVERAGE_LABELS")
    return len(labels)


def validate_config(config, task):
    family = config.get("model_type")
    if family == "bart":
        _common(config, task)
        # Identical native class and decoding bounds as numeric BART summarization.
        numeric_seq2seq.validate_config(config, "summarization")
    elif family == "layoutlmv3":
        _common(config, task)
        _bounded(
            config,
            {
                "hidden_size": (8, 2048),
                "intermediate_size": (8, 8192),
                "num_hidden_layers": (1, 48),
                "num_attention_heads": (1, 32),
                "vocab_size": (8, 300000),
                "max_position_embeddings": (8, 4096),
                "max_2d_position_embeddings": (8, 4096),
                "coordinate_size": (1, 1024),
                "shape_size": (1, 1024),
                "rel_pos_bins": (2, 256),
                "rel_2d_pos_bins": (2, 256),
                "max_rel_pos": (2, 4096),
                "max_rel_2d_pos": (2, 4096),
                "input_size": (16, 1024),
                "patch_size": (1, 64),
                "type_vocab_size": (1, 16),
            },
            "NUMERIC_COVERAGE_CONFIGURATION_LIMIT",
        )
        if (
            config["hidden_size"] % config["num_attention_heads"]
            or config["input_size"] % config["patch_size"]
            # Left/upper/right/lower coordinates plus height/width form the layout embedding.
            or 4 * config["coordinate_size"] + 2 * config["shape_size"] != config["hidden_size"]
            or config.get("num_channels", 3) != 3
            or config.get("hidden_act", "gelu") != "gelu"
            or any(
                config.get(k, True) is not True
                for k in (
                    "visual_embed",
                    "text_embed",
                    "has_relative_attention_bias",
                    "has_spatial_attention_bias",
                )
            )
            or config.get("num_labels", 2) != 2
            or type(config.get("pad_token_id")) is not int
            or not 0 <= config["pad_token_id"] < config["vocab_size"]
        ):
            raise ValueError("NUMERIC_COVERAGE_CONFIGURATION")
    elif family == "vivit":
        _common(config, task)
        _bounded(
            config,
            {
                "hidden_size": (8, 2048),
                "intermediate_size": (8, 8192),
                "num_hidden_layers": (1, 48),
                "num_attention_heads": (1, 32),
                "image_size": (16, 1024),
                "num_frames": (2, 32),
            },
            "NUMERIC_COVERAGE_CONFIGURATION_LIMIT",
        )
        tubelet = config.get("tubelet_size")
        if (
            not isinstance(tubelet, list)
            or len(tubelet) != 3
            or any(type(v) is not int or not 1 <= v <= 64 for v in tubelet)
            or config["num_frames"] % tubelet[0]
            or config["image_size"] % tubelet[1]
            or config["image_size"] % tubelet[2]
            or config["hidden_size"] % config["num_attention_heads"]
            or config.get("num_channels", 3) != 3
            or config.get("qkv_bias", True) is not True
            or config.get("hidden_act", "gelu_fast") not in {"gelu", "gelu_fast", "gelu_new"}
        ):
            raise ValueError("NUMERIC_COVERAGE_CONFIGURATION")
        _labels(config)
    elif family == "vilt":
        _common(config, task)
        _bounded(
            config,
            {
                "hidden_size": (8, 2048),
                "intermediate_size": (8, 8192),
                "num_hidden_layers": (1, 48),
                "num_attention_heads": (1, 32),
                "vocab_size": (8, 300000),
                "max_position_embeddings": (2, 4096),
                "type_vocab_size": (1, 16),
                "image_size": (16, 1024),
                "patch_size": (1, 64),
            },
            "NUMERIC_COVERAGE_CONFIGURATION_LIMIT",
        )
        if (
            config["hidden_size"] % config["num_attention_heads"]
            or config["image_size"] % config["patch_size"]
            or config.get("num_channels", 3) != 3
            or config.get("qkv_bias", True) is not True
            or config.get("modality_type_vocab_size", 2) != 2
            or config.get("num_images", -1) != -1
            or config.get("hidden_act", "gelu") != "gelu"
        ):
            raise ValueError("NUMERIC_COVERAGE_CONFIGURATION")
        _labels(config, low=1)
    else:
        raise ValueError("NUMERIC_COVERAGE_CONFIGURATION")
    if task in coverage_tasks.TASK_CONTRACTS:
        coverage_tasks.validate_config(task, config)


def expected_shapes(config):
    family = config["model_type"]
    if family == "bart":
        return numeric_seq2seq.expected_shapes(config)
    out = {}
    h = config["hidden_size"]

    def linear(name, rows, columns, bias=True):
        out[name + ".weight"] = [rows, columns]
        if bias:
            out[name + ".bias"] = [rows]

    def norm(name, width=h):
        out[name + ".weight"] = [width]
        out[name + ".bias"] = [width]

    def bert_layer(prefix, inner="self"):
        for part in ("query", "key", "value"):
            linear(f"{prefix}.attention.{inner}.{part}", h, h)
        linear(prefix + ".attention.output.dense", h, h)
        linear(prefix + ".intermediate.dense", config["intermediate_size"], h)
        linear(prefix + ".output.dense", h, config["intermediate_size"])

    if family == "layoutlmv3":
        patches = (config["input_size"] // config["patch_size"]) ** 2
        heads = config["num_attention_heads"]
        p = "layoutlmv3"
        out[p + ".cls_token"] = [1, 1, h]
        out[p + ".pos_embed"] = [1, patches + 1, h]
        out[p + ".embeddings.word_embeddings.weight"] = [config["vocab_size"], h]
        out[p + ".embeddings.token_type_embeddings.weight"] = [config["type_vocab_size"], h]
        norm(p + ".embeddings.LayerNorm")
        out[p + ".embeddings.position_embeddings.weight"] = [config["max_position_embeddings"], h]
        for axis, width in (
            ("x", "coordinate_size"),
            ("y", "coordinate_size"),
            ("h", "shape_size"),
            ("w", "shape_size"),
        ):
            out[f"{p}.embeddings.{axis}_position_embeddings.weight"] = [
                config["max_2d_position_embeddings"],
                config[width],
            ]
        size = config["patch_size"]
        out[p + ".patch_embed.proj.weight"] = [h, 3, size, size]
        out[p + ".patch_embed.proj.bias"] = [h]
        norm(p + ".LayerNorm")
        norm(p + ".norm")
        for i in range(config["num_hidden_layers"]):
            prefix = f"{p}.encoder.layer.{i}"
            bert_layer(prefix)
            norm(prefix + ".attention.output.LayerNorm")
            norm(prefix + ".output.LayerNorm")
        out[p + ".encoder.rel_pos_bias.weight"] = [heads, config["rel_pos_bins"]]
        out[p + ".encoder.rel_pos_x_bias.weight"] = [heads, config["rel_2d_pos_bins"]]
        out[p + ".encoder.rel_pos_y_bias.weight"] = [heads, config["rel_2d_pos_bins"]]
        linear("qa_outputs.dense", h, h)
        linear("qa_outputs.out_proj", 2, h)
        return out
    if family == "vivit":
        t, a, b = config["tubelet_size"]
        size = config["image_size"]
        patches = (config["num_frames"] // t) * (size // a) * (size // b)
        out["vivit.embeddings.cls_token"] = [1, 1, h]
        out["vivit.embeddings.position_embeddings"] = [1, patches + 1, h]
        out["vivit.embeddings.patch_embeddings.projection.weight"] = [h, 3, t, a, b]
        out["vivit.embeddings.patch_embeddings.projection.bias"] = [h]
        for i in range(config["num_hidden_layers"]):
            prefix = f"vivit.layers.{i}"
            for part in ("q_proj", "k_proj", "v_proj", "o_proj"):
                linear(f"{prefix}.attention.{part}", h, h)
            norm(prefix + ".layernorm_before")
            norm(prefix + ".layernorm_after")
            linear(prefix + ".mlp.fc1", config["intermediate_size"], h)
            linear(prefix + ".mlp.fc2", h, config["intermediate_size"])
        norm("vivit.layernorm")
        linear("classifier", len(config["id2label"]), h)
        return out
    assert family == "vilt"
    patches = (config["image_size"] // config["patch_size"]) ** 2
    p = "vilt.embeddings"
    out[p + ".cls_token"] = [1, 1, h]
    out[p + ".position_embeddings"] = [1, patches + 1, h]
    out[p + ".text_embeddings.word_embeddings.weight"] = [config["vocab_size"], h]
    out[p + ".text_embeddings.position_embeddings.weight"] = [config["max_position_embeddings"], h]
    out[p + ".text_embeddings.token_type_embeddings.weight"] = [config["type_vocab_size"], h]
    norm(p + ".text_embeddings.LayerNorm")
    size = config["patch_size"]
    out[p + ".patch_embeddings.projection.weight"] = [h, 3, size, size]
    out[p + ".patch_embeddings.projection.bias"] = [h]
    out[p + ".token_type_embeddings.weight"] = [2, h]
    for i in range(config["num_hidden_layers"]):
        prefix = f"vilt.encoder.layer.{i}"
        bert_layer(prefix, "attention")
        norm(prefix + ".layernorm_before")
        norm(prefix + ".layernorm_after")
    norm("vilt.layernorm")
    linear("vilt.pooler.dense", h, h)
    linear("classifier.0", 2 * h, h)
    norm("classifier.1", 2 * h)
    linear("classifier.3", len(config["id2label"]), 2 * h)
    return out


def native_name(config, name):
    if config["model_type"] != "vivit" or not name.startswith("vivit.encoder.layer."):
        return name
    for pattern, replacement in LEGACY_VIT:
        target, count = re.subn("^vivit\\." + pattern, "vivit." + replacement, name)
        if count:
            return target
    return name


def position_buffer(config):
    name = POSITIONS.get(config["model_type"])
    return (name, config["max_position_embeddings"]) if name else None


def load_processor(root, config, task):
    import transformers as tf

    common = {"local_files_only": True, "trust_remote_code": False}
    if task in coverage_tasks.TASK_CONTRACTS:
        processor, _ = coverage_tasks.load_components(root, task, config, common)
        if config["model_type"] == "bart":
            numeric_seq2seq_tokenizer_check(processor, config)
        return processor
    processor = tf.ViltProcessor.from_pretrained(root, **common)
    if len(processor.tokenizer) > config["vocab_size"] or not processor.tokenizer.is_fast:
        raise ValueError("NUMERIC_COVERAGE_TOKENIZER")
    return processor


def numeric_seq2seq_tokenizer_check(tokenizer, config):
    if (
        len(tokenizer) > config["vocab_size"]
        or tokenizer.pad_token_id != config["pad_token_id"]
        or tokenizer.eos_token_id != config["eos_token_id"]
    ):
        raise ValueError("NUMERIC_COVERAGE_TOKENIZER")


def prepare(model, processor, config):
    if config["model_type"] == "bart":
        numeric_seq2seq.prepare(model, processor, config)


def execute(
    model, processor, config, task, value, backend, image=None, input_path=None, original=None
):
    if task in coverage_tasks.TASK_CONTRACTS:
        return coverage_tasks.execute(
            task, model, processor, config, value, backend, input_path, image, original
        )
    from oryn_engine.shared_tasks import _vision_text

    return _vision_text(model, processor, config, task, image, value, backend)
