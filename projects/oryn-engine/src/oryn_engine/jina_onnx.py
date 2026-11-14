"""Reviewed Jina ONNX task metadata; all encoder operations remain in the graph."""

EMBEDDING_ALIASES = {
    key: "jinaai/xlm-roberta-flash-implementation--" + value
    for key, value in {
        "AutoConfig": "configuration_xlm_roberta.XLMRobertaFlashConfig",
        "AutoModel": "modeling_lora.XLMRobertaLoRA",
        "AutoModelForMaskedLM": "modeling_xlm_roberta.XLMRobertaForMaskedLM",
        "AutoModelForPreTraining": "modeling_xlm_roberta.XLMRobertaForPreTraining",
    }.items()
}
RANKING_ALIASES = {
    "AutoConfig": "configuration_xlm_roberta.XLMRobertaFlashConfig",
    "AutoModel": "modeling_xlm_roberta.XLMRobertaModel",
    "AutoModelForSequenceClassification": (
        "modeling_xlm_roberta.XLMRobertaForSequenceClassification"
    ),
}
TASKS = ["retrieval.query", "retrieval.passage", "separation", "classification", "text-matching"]


def normalized_config(config, task):
    architecture, aliases = (
        ("XLMRobertaModel", EMBEDDING_ALIASES)
        if task == "feature-extraction"
        else ("XLMRobertaForSequenceClassification", RANKING_ALIASES)
        if task == "text-ranking"
        else (None, None)
    )
    if (
        aliases is not None
        and config.get("auto_map") == aliases
        and config.get("architectures") == [architecture]
        and config.get("model_type") in {None, "xlm-roberta"}
    ):
        config = {k: v for k, v in config.items() if k != "auto_map"} | {
            "model_type": "xlm-roberta"
        }
    return config


def uses_task_selector(config, task):
    if "lora_adaptations" not in config:
        return False
    if (
        task != "feature-extraction"
        or config.get("model_type") != "xlm-roberta"
        or config.get("architectures") != ["XLMRobertaModel"]
        or config.get("lora_adaptations") != TASKS
        or config.get("load_trained_adapters") is not True
        or config.get("lora_main_params_trainable") is not False
        or type(config.get("lora_rank")) is not int
        or config["lora_rank"] != 4
        or config.get("lora_alpha") != 1
        or config.get("lora_dropout_p") != 0.0
        or config.get("truncate_dim") is not None
        or config.get("task_instructions")
        != {
            "classification": "",
            "retrieval.passage": "Represent the document for retrieval: ",
            "retrieval.query": "Represent the query for retrieving evidence documents: ",
            "separation": "",
            "text-matching": "",
        }
    ):
        raise ValueError("ONNX_TASK_SELECTOR_UNSUPPORTED")
    return True
