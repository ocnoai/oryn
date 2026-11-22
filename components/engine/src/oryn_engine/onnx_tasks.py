"""Sealed ONNX encoder task bindings over the common ONNX foundation."""

from __future__ import annotations

import json

from oryn_engine import jina_onnx, onnx_classifier, onnx_tdt
from oryn_engine.model_metadata import metadata_file, validate_model_config
from oryn_engine.onnx_graph import scan_graph

TASK_CONTRACTS = {
    "image-classification": "image-classification-v1",
    "feature-extraction": "embeddings-v1",
    "text-classification": "text-classification-v1",
    "token-classification": "token-classification-v1",
    "text-ranking": "reranking-v1",
    "voice-activity-detection": "voice-activity-detection-v1",
    "automatic-speech-recognition": "transcription-v1",
    "audio-to-audio": "audio-transformation-v1",
}
# Reviewed input/pooling behaviors; architecture identities come from the pinned
# Transformers Auto mappings, and operators come from the pinned ONNX Runtime.
ENCODER_FAMILIES = {"bert", "distilbert", "mpnet", "nomic_bert", "roberta", "xlm-roberta"}


def variant_id(names, task):
    from oryn_engine.runtime_routing import canonical_digest

    if len(names) == 1:
        identity = names[0]
    elif len(names) == 2 and task == "automatic-speech-recognition":
        from oryn_engine.onnx_whisper import graph_pairs

        pairs = graph_pairs(names)
        if len(pairs) != 1 or set(pairs[0]) != set(names):
            raise ValueError("ONNX_GRAPH_AMBIGUOUS")
        identity = pairs[0]
    elif (
        len(names) == 3 and task == "automatic-speech-recognition" and onnx_tdt.graph_groups(names)
    ):
        identity = onnx_tdt.GRAPH_NAMES
    else:
        raise ValueError("ONNX_GRAPH_AMBIGUOUS")
    return "oryn.onnx." + canonical_digest(identity)[:24]


def native_config(config, task):
    """Normalize reviewed Nomic/Jina aliases only for their ONNX task bindings.

    The graph supplies the encoder operations; the native BERT tokenizer and
    declared pooling supply the task binding. No repository Python is loaded.
    Other custom mappings, including nested mappings, still fail validation.
    """
    config = jina_onnx.normalized_config(config, task)
    if config.get("auto_map"):
        prefix = "nomic-ai/nomic-bert-2048--"
        aliases = {
            "AutoConfig": "configuration_hf_nomic_bert.NomicBertConfig",
            "AutoModel": "modeling_hf_nomic_bert.NomicBertModel",
            "AutoModelForMaskedLM": "modeling_hf_nomic_bert.NomicBertForPreTraining",
            "AutoModelForMultipleChoice": "modeling_hf_nomic_bert.NomicBertForMultipleChoice",
            "AutoModelForQuestionAnswering": "modeling_hf_nomic_bert.NomicBertForQuestionAnswering",
            "AutoModelForSequenceClassification": (
                "modeling_hf_nomic_bert.NomicBertForSequenceClassification"
            ),
            "AutoModelForTokenClassification": (
                "modeling_hf_nomic_bert.NomicBertForTokenClassification"
            ),
        }
        mapping = config["auto_map"]
        if (
            task == "feature-extraction"
            and config.get("model_type") == "nomic_bert"
            and config.get("architectures") == ["NomicBertModel"]
            and isinstance(mapping, dict)
            and all(k in aliases and v == prefix + aliases[k] for k, v in mapping.items())
        ):
            config = {k: v for k, v in config.items() if k != "auto_map"}
    validate_model_config(config)
    return config


def validate_pooling(options, config):
    """Bind a single declared Sentence Transformers pooling operation."""
    if options is None:
        return "mean"
    modes = {
        "pooling_mode_cls_token": "cls",
        "pooling_mode_mean_tokens": "mean",
        "pooling_mode_max_tokens": None,
        "pooling_mode_mean_sqrt_len_tokens": None,
        "pooling_mode_weightedmean_tokens": None,
        "pooling_mode_lasttoken": None,
    }
    if (
        not isinstance(options, dict)
        or not set(options) <= set(modes) | {"word_embedding_dimension", "include_prompt"}
        or any(type(options[k]) is not bool for k in set(options) & modes.keys())
        or options.get("include_prompt", True) is not True
    ):
        raise ValueError("ONNX_POOLING_UNSUPPORTED")
    active = [modes[k] for k in modes if options.get(k) is True]
    dimension = options.get("word_embedding_dimension", config.get("hidden_size"))
    if (
        len(active) != 1
        or active[0] is None
        or type(dimension) is not int
        or not 1 <= dimension <= 8192
        or dimension != config.get("hidden_size")
    ):
        raise ValueError("ONNX_POOLING_UNSUPPORTED")
    if jina_onnx.uses_task_selector(config, "feature-extraction") and active != ["mean"]:
        raise ValueError("ONNX_POOLING_UNSUPPORTED")
    return active[0]


def validate_config(config, task):
    from oryn_engine.generated_capabilities import architectures_for

    config = native_config(config, task)
    if task == "image-classification":
        if onnx_classifier.supports(config):
            onnx_classifier.configuration(config)
            return config
        from oryn_engine.onnx_vision import configuration

        configuration(config)
        return config
    jina_onnx.uses_task_selector(config, task)
    if task == "audio-to-audio":
        if config:
            raise ValueError("ONNX_CONFIGURATION_UNSUPPORTED")
        return config
    if task == "voice-activity-detection":
        if config.get("auto_map"):
            raise ValueError("ONNX_CONFIGURATION_UNSUPPORTED")
        return config
    if task not in TASK_CONTRACTS or config.get("model_type") not in architectures_for(
        task, source="onnx"
    ):
        raise ValueError("ONNX_TASK_UNSUPPORTED")
    if config.get("auto_map") or config.get("quantization_config"):
        raise ValueError("ONNX_CONFIGURATION_UNSUPPORTED")
    if task == "token-classification" and (
        not isinstance(config.get("architectures"), list)
        or len(config["architectures"]) != 1
        or not set(config["architectures"]) <= architectures_for(task)
    ):
        raise ValueError("ONNX_TOKEN_CLASSIFIER_UNSUPPORTED")
    if task == "automatic-speech-recognition":
        if onnx_tdt.supports(config, task):
            onnx_tdt.validate_config(config)
            return config
        if config.get("model_type") == "whisper":
            from oryn_engine.onnx_whisper import validate_config as validate_asr_config
        else:
            from oryn_engine.onnx_asr import validate_config as validate_asr_config

        validate_asr_config(config)
        return config
    if task != "feature-extraction":
        labels = config.get("id2label", {})
        if (
            not isinstance(labels, dict)
            or not 1 <= len(labels) <= 4096
            or set(labels) != {str(i) for i in range(len(labels))}
        ):
            raise ValueError("ONNX_LABELS_INVALID")
        if any(not isinstance(v, str) or not 1 <= len(v) <= 512 for v in labels.values()):
            raise ValueError("ONNX_LABELS_INVALID")
        if task == "text-ranking" and len(labels) != 1:
            raise ValueError("ONNX_RANKING_HEAD_UNSUPPORTED")
        if config.get("problem_type") not in {None, "single_label_classification", "regression"}:
            raise ValueError("ONNX_CLASSIFICATION_UNSUPPORTED")
        if task == "text-classification" and len(labels) < 2:
            raise ValueError("ONNX_CLASSIFICATION_UNSUPPORTED")
    return config


def validate_graph(graph, task, config, *, vision=None):
    from oryn_engine.generated_capabilities import generated_artifact

    config = validate_config(config, task)
    known = generated_artifact()["buildInputs"]["onnxOperators"]
    if onnx_tdt.supports(config, task):
        for role in onnx_tdt.ROLES:
            member = graph[role]
            if any(
                op not in known or not any(v <= member["opset"] for v in known[op])
                for op in member["operators"]
            ):
                raise ValueError("ONNX_OPERATOR_UNSUPPORTED")
        onnx_tdt.validate_graphs(graph, config)
        return "outputs"
    if task == "automatic-speech-recognition" and config.get("model_type") == "whisper":
        from oryn_engine.onnx_whisper import validate_graphs

        for member in (graph["encoder"], graph["decoder"]):
            if any(
                op not in known or not any(v <= member["opset"] for v in known[op])
                for op in member["operators"]
            ):
                raise ValueError("ONNX_OPERATOR_UNSUPPORTED")
        validate_graphs(graph, config)
        return "logits"
    if any(
        op not in known
        or not any(
            v
            <= (
                graph.get("operatorDomains", {}).get("com.microsoft", 0)
                if op.startswith("com.microsoft::")
                else graph["opset"]
            )
            for v in known[op]
        )
        for op in graph["operators"]
    ):
        raise ValueError("ONNX_OPERATOR_UNSUPPORTED")
    if task == "image-classification":
        if onnx_classifier.supports(config):
            return onnx_classifier.validate_graph(graph)
        from oryn_engine.onnx_vision import validate_graph as validate_vision_graph

        return validate_vision_graph(graph, vision)
    if task == "voice-activity-detection":
        from oryn_engine.vad_tasks import validate_graph as validate_vad_graph

        validate_vad_graph(graph)
        return "output"
    if task == "audio-to-audio":
        from oryn_engine.onnx_audio import validate_graph as validate_audio_graph

        validate_audio_graph(graph)
        return "spec_enhanced"
    if task == "automatic-speech-recognition":
        from oryn_engine.onnx_asr import validate_graph as validate_asr_graph

        validate_asr_graph(graph, config)
        return "logits"
    inputs = {x["name"]: x for x in graph["inputs"]}
    task_selector = jina_onnx.uses_task_selector(config, task)
    if task_selector:
        selector = inputs.pop("task_id", None)
        if selector is None or selector["dtype"] != 7 or selector["shape"] != []:
            raise ValueError("ONNX_TASK_SELECTOR_UNSUPPORTED")
    if not {"input_ids", "attention_mask"} <= inputs.keys() or not inputs.keys() <= {
        "input_ids",
        "attention_mask",
        "token_type_ids",
    }:
        raise ValueError("ONNX_INPUT_BINDING_UNSUPPORTED")
    if any(x["dtype"] != 7 or len(x["shape"]) != 2 for x in inputs.values()):
        raise ValueError("ONNX_INPUT_BINDING_UNSUPPORTED")
    # A separate batch invocation is used for each item; static batches larger
    # than one and fixed token padding are not silently reinterpreted.
    if any(isinstance(x["shape"][0], int) and x["shape"][0] != 1 for x in inputs.values()):
        raise ValueError("ONNX_BATCH_UNSUPPORTED")
    if any(isinstance(x["shape"][1], int) for x in inputs.values()):
        raise ValueError("ONNX_SEQUENCE_UNSUPPORTED")
    name = "last_hidden_state" if task == "feature-extraction" else "logits"
    outputs = {x["name"]: x for x in graph["outputs"]}
    if task == "feature-extraction":
        token_outputs = (
            {"text_embeds"} if task_selector else {"last_hidden_state", "token_embeddings"}
        ) & outputs.keys()
        if len(token_outputs) != 1:
            raise ValueError("ONNX_OUTPUT_BINDING_UNSUPPORTED")
        name = token_outputs.pop()
    if (
        name not in outputs
        or outputs[name]["dtype"] != 1
        or len(outputs[name]["shape"])
        != (3 if task in {"feature-extraction", "token-classification"} else 2)
        or task_selector
        and outputs[name]["shape"][-1] != config.get("hidden_size")
    ):
        raise ValueError("ONNX_OUTPUT_BINDING_UNSUPPORTED")
    if task == "token-classification" and (
        set(outputs) != {"logits"}
        or outputs[name]["shape"][-1] != len(config["id2label"])
        or isinstance(outputs[name]["shape"][1], int)
        or isinstance(outputs[name]["shape"][0], int)
        and outputs[name]["shape"][0] != 1
    ):
        raise ValueError("ONNX_TOKEN_GEOMETRY_UNSUPPORTED")
    return name


def input_value(task, value):
    from oryn_protocol import ProtocolSchemaStore

    if task not in TASK_CONTRACTS or not ProtocolSchemaStore().is_valid(
        "taskInput." + TASK_CONTRACTS[task], value
    ):
        raise ValueError("ONNX_INPUT_INVALID")
    if len(json.dumps(value, allow_nan=False).encode()) > 128 * 1024:
        raise ValueError("ONNX_INPUT_LIMIT")
    value = dict(value)
    if task == "audio-to-audio":
        from oryn_engine.onnx_audio import input_value as audio_input

        return audio_input(value)
    if task == "text-ranking" and "instruction" in value:
        raise ValueError("ONNX_INSTRUCTION_UNSUPPORTED")
    return value


def output_value(task, value):
    from oryn_protocol.task_contracts import make_profile, validate_result

    if len(json.dumps(value, allow_nan=False).encode()) > 900 * 1024:
        raise ValueError("ONNX_OUTPUT_LIMIT")
    validate_result(make_profile(TASK_CONTRACTS[task]), {"kind": "structured", "value": value})
    return value


def verify_runtime():
    import hashlib
    from pathlib import Path

    import onnxruntime as ort

    from oryn_engine.generated_capabilities import generated_artifact

    if ort.__version__ != generated_artifact()["buildInputs"]["onnxruntime"]:
        raise ValueError("ONNX_RUNTIME_CHANGED")
    if (
        hashlib.sha256(Path(ort.capi.onnxruntime_pybind11_state.__file__).read_bytes()).hexdigest()
        != generated_artifact()["buildInputs"]["onnxNativeSha256"]
    ):
        raise ValueError("ONNX_RUNTIME_CHANGED")


def prepare(root, task, backend):
    import onnxruntime as ort

    verify_runtime()
    if backend not in {"cpu", "coreml"}:
        raise ValueError("ONNX_BACKEND_UNSUPPORTED")
    providers = ["CPUExecutionProvider"] if backend == "cpu" else ["CoreMLExecutionProvider"]
    if not set(providers) <= set(ort.get_available_providers()):
        raise ValueError("ONNX_PROVIDER_UNAVAILABLE")
    config = metadata_file(root / "config.json") if (root / "config.json").exists() else {}
    files = [p for p in root.rglob("*") if p.is_file()]
    if any(p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) for p in files):
        raise ValueError("ONNX_FILE_SCOPE_INVALID")
    if task == "image-classification":
        config = onnx_classifier.resolve_local(config, root, files)
    config = validate_config(config, task)
    graphs = [p for p in files if p.suffix == ".onnx"]
    sizes = {p.relative_to(root).as_posix(): p.stat().st_size for p in files}
    whisper = task == "automatic-speech-recognition" and config.get("model_type") == "whisper"
    tdt = onnx_tdt.supports(config, task)
    if tdt:
        if backend != "cpu":
            raise ValueError("ONNX_TDT_BACKEND")
        if len(graphs) != 3 or not onnx_tdt.graph_groups(
            [p.relative_to(root).as_posix() for p in graphs]
        ):
            raise ValueError("ONNX_TDT_GRAPH_INCOMPLETE")
        graphs = [root / name for name in onnx_tdt.GRAPH_NAMES]
    elif whisper:
        from oryn_engine.onnx_whisper import graph_pairs

        pairs = graph_pairs([p.relative_to(root).as_posix() for p in graphs])
        if len(graphs) != 2 or len(pairs) != 1:
            raise ValueError("ONNX_GRAPH_AMBIGUOUS")
        graphs = [root / n for n in pairs[0]]
    elif len(graphs) != 1:
        raise ValueError("ONNX_GRAPH_AMBIGUOUS")
    members = []
    for path in graphs:
        with path.open("rb") as source:

            def read_range(start, length):
                source.seek(start)
                return source.read(length)

            members.append(
                scan_graph(
                    read_range, path.stat().st_size, path.relative_to(root).as_posix(), sizes
                )
            )
    graph = (
        onnx_tdt.compose(members)
        if tdt
        else {"encoder": members[0], "decoder": members[1]}
        if whisper
        else members[0]
    )
    vision = None
    if task == "image-classification":
        if onnx_classifier.supports(config):
            if graphs[0].relative_to(root).as_posix() != onnx_classifier.configuration(config):
                raise ValueError("ONNX_CLASSIFIER_GRAPH_AMBIGUOUS")
            onnx_classifier.labels()
            onnx_classifier.verify_preprocessor()
        else:
            from oryn_engine.onnx_vision import configuration, local_labels

            if graphs[0].relative_to(root).as_posix() != configuration(config)[0]:
                raise ValueError("ONNX_VISION_GRAPH_AMBIGUOUS")
            vision = local_labels(root, config)
    output = validate_graph(graph, task, config, vision=vision)
    if tdt:
        geometry = onnx_tdt.validate_graphs(graph, config)
        onnx_tdt.vocabulary(root, geometry["vocab"])
    pooling = root / "1_Pooling/config.json"
    if task == "feature-extraction":
        validate_pooling(metadata_file(pooling) if pooling.exists() else None, config)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.enable_mem_pattern = False
    options.log_severity_level = 3
    if backend == "coreml":
        # Do not label an automatic CPU fallback as CoreML execution evidence.
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    sessions = [
        ort.InferenceSession(str(path), sess_options=options, providers=providers)
        for path in graphs
    ]
    for session in sessions:
        session.disable_fallback()
    if vision is not None:
        config = config | {"_orynVision": vision}
    if tdt:
        config = config | {"_orynTdtGeometry": geometry}
    return (tuple(sessions) if whisper or tdt else sessions[0]), config, graph, output


def execute(root, task, raw, backend, event, input_path=None, output_path=None):
    import numpy as np
    from transformers import AutoTokenizer

    value = input_value(task, raw)
    event("job.progress", {"stage": "loading-model", "progress": 0.1})
    session, config, graph, output = prepare(root, task, backend)
    if task == "image-classification":
        if onnx_classifier.supports(config):
            return output_value(task, onnx_classifier.execute(session, value, input_path, event))
        from oryn_engine.onnx_vision import execute as execute_vision

        return output_value(
            task, execute_vision(session, config["_orynVision"], value, input_path, event)
        )
    if task == "audio-to-audio":
        from oryn_engine.onnx_audio import execute as execute_audio

        return execute_audio(session, value, input_path, output_path, event)
    pooling_path = root / "1_Pooling/config.json"
    pooling = (
        validate_pooling(metadata_file(pooling_path) if pooling_path.exists() else None, config)
        if task == "feature-extraction"
        else None
    )
    if task == "voice-activity-detection":
        from oryn_engine.vad_tasks import execute as execute_vad

        return output_value(task, execute_vad(session, value, input_path, event))
    if task == "automatic-speech-recognition":
        if onnx_tdt.supports(config, task):
            return output_value(
                task, onnx_tdt.execute(session, root, config, value, input_path, event)
            )
        if config.get("model_type") == "whisper":
            from oryn_engine.onnx_whisper import execute as execute_asr
        else:
            from oryn_engine.onnx_asr import execute as execute_asr

        return output_value(task, execute_asr(session, root, config, value, input_path, event))
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    tokenizer.chat_template = None
    limit = min(
        512, int(config.get("max_position_embeddings", 512)), int(tokenizer.model_max_length)
    )
    names = {x["name"] for x in graph["inputs"]}
    pairs = (
        [(t, None) for t in value["texts"]]
        if task == "feature-extraction"
        else [(value["query"], t) for t in value["documents"]]
        if task == "text-ranking"
        else [(value["text"], None)]
    )
    vectors, logits = [], []
    for index, (text, pair) in enumerate(pairs):
        encoded = tokenizer(
            text,
            text_pair=pair,
            return_tensors="np",
            truncation=False,
            **(
                {"return_offsets_mapping": True, "return_special_tokens_mask": True}
                if task == "token-classification"
                else {}
            ),
        )
        if not 1 <= encoded["input_ids"].shape[1] <= limit:
            raise ValueError("ONNX_TOKEN_LIMIT")
        if "token_type_ids" in names and "token_type_ids" not in encoded:
            encoded["token_type_ids"] = np.zeros_like(encoded["input_ids"])
        if jina_onnx.uses_task_selector(config, task):
            if pooling != "mean":
                raise ValueError("ONNX_POOLING_UNSUPPORTED")
            encoded["task_id"] = np.array(jina_onnx.TASKS.index("text-matching"), dtype=np.int64)
        event(
            "job.progress", {"stage": "running-model", "progress": 0.2 + index / len(pairs) * 0.7}
        )
        result = session.run([output], {k: np.asarray(encoded[k], dtype=np.int64) for k in names})[
            0
        ]
        if not np.isfinite(result).all():
            raise ValueError("ONNX_OUTPUT_NONFINITE")
        if task == "token-classification":
            if result.shape != (*encoded["input_ids"].shape, len(config["id2label"])):
                raise ValueError("ONNX_TOKEN_GEOMETRY_UNSUPPORTED")
            scores = result[0].astype(np.float64)
            scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
            scores /= scores.sum(axis=-1, keepdims=True)
            labels = scores.argmax(axis=-1)
            tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].tolist())
            items = []
            for i, (start, end) in enumerate(encoded["offset_mapping"][0]):
                if not 0 <= start <= end <= len(text):
                    raise ValueError("ONNX_TOKEN_OFFSETS_INVALID")
                if encoded["special_tokens_mask"][0, i] or end == start:
                    continue
                label = int(labels[i])
                items.append(
                    {
                        "token": tokens[i],
                        "labelId": label,
                        "label": config["id2label"][str(label)],
                        "score": float(scores[i, label]),
                        "start": int(start),
                        "end": int(end),
                    }
                )
            return output_value(task, {"tokens": items})
        if task == "feature-extraction":
            if (
                result.ndim != 3
                or result.shape[:2] != encoded["input_ids"].shape
                or not 1 <= result.shape[2] <= 8192
                or result.shape[2] != config.get("hidden_size", result.shape[2])
            ):
                raise ValueError("ONNX_EMBEDDING_GEOMETRY")
            mask = encoded["attention_mask"][..., None]
            vector = (
                result[:, 0] if pooling == "cls" else (result * mask).sum(axis=1) / mask.sum(axis=1)
            ).astype(np.float32)[0]
            if not np.isfinite(vector).all():
                raise ValueError("ONNX_OUTPUT_NONFINITE")
            if value.get("normalize", True):
                norm = np.linalg.norm(vector)
                if not np.isfinite(norm) or norm <= 1e-12:
                    raise ValueError("ONNX_EMBEDDING_ZERO")
                vector /= norm
            vectors.append(vector.tolist())
        else:
            if result.shape != (1, len(config["id2label"])):
                raise ValueError("ONNX_LABEL_GEOMETRY")
            logits.append(result[0].astype(np.float64))
    if vectors:
        result = {
            "embeddings": vectors,
            "dimensions": len(vectors[0]),
            "normalized": value.get("normalize", True),
        }
    elif task == "text-ranking":
        scores = [float(1 / (1 + np.exp(-np.clip(x[0], -80, 80)))) for x in logits]
        result = {
            "results": sorted(
                [{"index": i, "score": s} for i, s in enumerate(scores)], key=lambda x: -x["score"]
            )[: value.get("topK", len(scores))]
        }
    else:
        probabilities = np.exp(logits[0] - logits[0].max())
        probabilities /= probabilities.sum()
        result = {
            "predictions": [
                {
                    "label": config["id2label"][str(i)],
                    "labelId": int(i),
                    "score": float(probabilities[i]),
                }
                for i in np.argsort(-probabilities)[: value.get("topK", 5)]
            ]
        }
    return output_value(task, result)
