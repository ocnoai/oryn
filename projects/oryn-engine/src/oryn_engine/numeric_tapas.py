"""Owned full TAPAS graph and explicit table-task contract."""

from oryn_engine import coverage_tasks
from oryn_engine.model_metadata import validate_model_config


def validate_config(c):
    validate_model_config(c)
    if (
        c.get("model_type") != "tapas"
        or c.get("architectures") != ["TapasForQuestionAnswering"]
        or c.get("auto_map")
        or c.get("quantization")
        or c.get("quantization_config")
        or c.get("span_prediction", "none") != "none"
        or c.get("is_decoder", False)
        or c.get("add_cross_attention", False)
    ):
        raise ValueError("NUMERIC_TAPAS_CONFIGURATION")
    for k, lo, hi in [
        ("hidden_size", 8, 2048),
        ("intermediate_size", 8, 8192),
        ("num_hidden_layers", 1, 48),
        ("num_attention_heads", 1, 32),
        ("vocab_size", 8, 131072),
        ("max_position_embeddings", 2, 2048),
        ("num_aggregation_labels", 0, 4),
    ]:
        if type(c.get(k)) is not int or not lo <= c[k] <= hi:
            raise ValueError("NUMERIC_TAPAS_CONFIGURATION_LIMIT")
    if (
        c["hidden_size"] % c["num_attention_heads"]
        or c.get("hidden_act", "gelu") not in ["gelu", "relu"]
        or type(c.get("pad_token_id", 0)) is not int
        or not 0 <= c.get("pad_token_id", 0) < c["vocab_size"]
    ):
        raise ValueError("NUMERIC_TAPAS_CONFIGURATION")
    types = c.get("type_vocab_sizes")
    if (
        not isinstance(types, list)
        or len(types) != 7
        or any(type(x) is not int or not 2 <= x <= 4096 for x in types)
        or c.get("type_vocab_size", types) != types
    ):
        raise ValueError("NUMERIC_TAPAS_CONFIGURATION")
    for k, default in [
        ("select_one_column", True),
        ("allow_empty_column_selection", False),
        ("average_logits_per_cell", False),
        ("reset_position_index_per_cell", True),
    ]:
        if type(c.get(k, default)) is not bool:
            raise ValueError("NUMERIC_TAPAS_CONFIGURATION")
    for k in ["use_gumbel_for_cells", "use_gumbel_for_aggregation"]:
        if c.get(k, False) is not False:
            raise ValueError("NUMERIC_TAPAS_CONFIGURATION")
    for k in ["temperature", "aggregation_temperature"]:
        value = c.get(k, 1.0)
        if type(value) not in [int, float] or not 0 < value <= 100:
            raise ValueError("NUMERIC_TAPAS_CONFIGURATION")
    coverage_tasks.validate_config("table-question-answering", c)
    if c["num_aggregation_labels"] and (
        not c.get("aggregation_labels") or c.get("no_aggregation_label_index", 0) != 0
    ):
        raise ValueError("NUMERIC_TAPAS_AGGREGATION")


def expected_shapes(c):
    validate_config(c)
    h = c["hidden_size"]
    out = {}

    def linear(name, rows, columns):
        out[name + ".weight"] = [rows, columns]
        out[name + ".bias"] = [rows]

    def norm(name):
        out[name + ".weight"] = [h]
        out[name + ".bias"] = [h]

    prefix = "tapas.embeddings"
    out[prefix + ".word_embeddings.weight"] = [c["vocab_size"], h]
    out[prefix + ".position_embeddings.weight"] = [c["max_position_embeddings"], h]
    norm(prefix + ".LayerNorm")
    for i, count in enumerate(c["type_vocab_sizes"]):
        out[prefix + f".token_type_embeddings_{i}.weight"] = [count, h]
    for i in range(c["num_hidden_layers"]):
        p = f"tapas.encoder.layer.{i}"
        for key in ["query", "key", "value"]:
            linear(p + ".attention.self." + key, h, h)
        linear(p + ".attention.output.dense", h, h)
        norm(p + ".attention.output.LayerNorm")
        linear(p + ".intermediate.dense", c["intermediate_size"], h)
        linear(p + ".output.dense", h, c["intermediate_size"])
        norm(p + ".output.LayerNorm")
    linear("tapas.pooler.dense", h, h)
    for name in ["output", "column_output"]:
        out[name + "_weights"] = [h]
        out[name + "_bias"] = []
    if c["num_aggregation_labels"]:
        linear("aggregation_classifier", c["num_aggregation_labels"], h)
    return out


ARCHITECTURES = {("table-question-answering", "tapas"): "TapasForQuestionAnswering"}


def supports(config, task):
    return (task, config.get("model_type")) in ARCHITECTURES
