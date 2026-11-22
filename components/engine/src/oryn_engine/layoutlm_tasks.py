"""Complete RoBERTa-tokenized LayoutLM QA, including the invoice selection head."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from oryn_engine.model_metadata import metadata_file, preflight_safetensors, validate_model_config

DATA = Path(__file__).parent / "layoutlm_assets"
TASK = "document-question-answering"
FILES = (
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "model.safetensors",
)
POSITION = "layoutlm.embeddings.position_ids"


def supports(config, task):
    return (
        task == TASK
        and config.get("model_type") == "layoutlm"
        and (
            config.get("tokenizer_class") in {"RobertaTokenizer", "RobertaTokenizerFast"}
            or "token_classification" in config
        )
    )


def template(name):
    return json.loads((DATA / name).read_text())


def validate_config(config):
    validate_model_config(config)
    invoice = config.get("token_classification", False)
    if type(invoice) is not bool:
        raise ValueError("LAYOUTLM_QA_CONFIGURATION")
    expected = template("invoices.json" if invoice else "document.json")
    dimensions = {
        "hidden_size": (24, 768),
        "intermediate_size": (48, 3072),
        "num_hidden_layers": (1, 12),
        "num_attention_heads": (1, 12),
        "vocab_size": (261, 50265),
    }
    if set(config) != set(expected):
        raise ValueError("LAYOUTLM_QA_CONFIGURATION")
    for key, val in expected.items():
        if (
            key not in dimensions
            and key not in {"_name_or_path", "tokenizer_class"}
            and (type(config[key]) is not type(val) or config[key] != val)
        ):
            raise ValueError("LAYOUTLM_QA_CONFIGURATION:" + key)
    if config["tokenizer_class"] not in {"RobertaTokenizer", "RobertaTokenizerFast"}:
        raise ValueError("LAYOUTLM_QA_TOKENIZER")
    for key, (low, high) in dimensions.items():
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError("LAYOUTLM_QA_GEOMETRY")
    if (
        config["hidden_size"] % config["num_attention_heads"]
        or config["intermediate_size"] < config["hidden_size"] * 2
    ):
        raise ValueError("LAYOUTLM_QA_GEOMETRY")
    return config


def validate_metadata(name, value):
    expected = template(name)
    if json.dumps(value, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("LAYOUTLM_QA_TOKENIZER_METADATA")


def native_model(config, *, meta=False):
    import torch
    from transformers import LayoutLMConfig, LayoutLMModel

    validate_config(config)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = LayoutLMConfig(**config)
            self.config._attn_implementation = "eager"
            self.layoutlm = LayoutLMModel(self.config)
            self.layoutlm.embeddings.register_buffer(
                "position_ids",
                torch.arange(config["max_position_embeddings"]).expand(1, -1),
                persistent=True,
            )
            self.qa_outputs = torch.nn.Linear(config["hidden_size"], 2)
            self.token_classifier_head = (
                torch.nn.Linear(config["hidden_size"], 2)
                if config.get("token_classification")
                else None
            )

        def forward(self, **inputs):
            hidden = self.layoutlm(**inputs).last_hidden_state
            logits = self.qa_outputs(hidden)
            tokens = (
                self.token_classifier_head(hidden)
                if self.token_classifier_head is not None
                else None
            )
            return SimpleNamespace(
                start_logits=logits[..., 0], end_logits=logits[..., 1], token_logits=tokens
            )

    with torch.device("meta" if meta else "cpu"):
        return Model().eval()


def validate_weight_shapes(config, shapes):
    expected = {k: tuple(v.shape) for k, v in native_model(config, meta=True).state_dict().items()}
    if expected != {k: tuple(v) for k, v in shapes.items()}:
        raise ValueError("LAYOUTLM_QA_WEIGHTS_INCOMPLETE")


def load_components(root, config, backend):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    if backend not in {"cpu", "mps"} or root.is_symlink():
        raise ValueError("LAYOUTLM_QA_SCOPE")
    if any(not (root / name).is_file() or (root / name).is_symlink() for name in FILES):
        raise ValueError("LAYOUTLM_QA_EXPORT_INCOMPLETE")
    if {p.name for p in root.glob("*.safetensors")} != {"model.safetensors"} or any(
        root.glob("*.safetensors.index.json")
    ):
        raise ValueError("LAYOUTLM_QA_EXPORT_INCOMPLETE")
    validate_config(config)
    if (root / "tokenizer.json").stat().st_size > 2_000_000:
        raise ValueError("LAYOUTLM_QA_TOKENIZER_SIZE")
    for name in ("special_tokens_map.json",):
        validate_metadata(name, metadata_file(root / name))
    validate_weight_shapes(config, preflight_safetensors(root))
    state = load_file(root / "model.safetensors")
    for key, value in state.items():
        if key == POSITION:
            if value.dtype != torch.int64 or not torch.equal(
                value, torch.arange(config["max_position_embeddings"]).expand(1, -1)
            ):
                raise ValueError("LAYOUTLM_QA_POSITIONS")
        elif value.dtype not in {torch.float32, torch.float16, torch.bfloat16} or not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError("LAYOUTLM_QA_NUMERIC_WEIGHT")
    with init_empty_weights(include_buffers=False):
        model = native_model(config)
    model.load_state_dict(
        {k: v if k == POSITION else v.float() for k, v in state.items()}, strict=True, assign=True
    )
    return model.to(backend).eval(), (load_tokenizer(root, config), None), config


def load_tokenizer(root, config):
    from transformers import RobertaTokenizer

    # Original tokenizer_config.json repeats add_prefix_space with conflicting
    # values. It is deliberately not selected or read. This owned family binding
    # supplies the publisher's documented prefix and framing semantics directly.
    tokenizer = RobertaTokenizer(
        tokenizer_file=str(root / "tokenizer.json"), add_prefix_space=True, model_max_length=512
    )

    if (
        not tokenizer.is_fast
        or len(tokenizer) != config["vocab_size"]
        or (
            tokenizer.bos_token_id,
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            tokenizer.unk_token_id,
        )
        != (0, 1, 2, 3)
        or tokenizer.sep_token_id != 2
        or tokenizer.model_max_length != 512
    ):
        raise ValueError("LAYOUTLM_QA_TOKENIZER")
    from tokenizers.pre_tokenizers import ByteLevel

    spec = json.loads(tokenizer.backend_tokenizer.to_str())
    prefix = (
        spec["pre_tokenizer"].get("add_prefix_space")
        if isinstance(spec["pre_tokenizer"], dict)
        else None
    )
    if type(prefix) is not bool or spec["pre_tokenizer"] != {
        "type": "ByteLevel",
        "add_prefix_space": prefix,
        "trim_offsets": True,
        "use_regex": True,
    }:
        raise ValueError("LAYOUTLM_QA_TOKENIZER")
    # Loading tokenizer_file restores the serialized pre-tokenizer after the
    # wrapper constructor. Apply the owned prefix setting to that native object.
    tokenizer.backend_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    spec = json.loads(tokenizer.backend_tokenizer.to_str())
    if (
        spec["normalizer"] is not None
        or spec["model"]["type"] != "BPE"
        or spec["pre_tokenizer"]
        != {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True}
        or spec["post_processor"]
        != {
            "type": "RobertaProcessing",
            "sep": ["</s>", 2],
            "cls": ["<s>", 0],
            "trim_offsets": True,
            "add_prefix_space": True,
        }
        or spec["decoder"]
        != {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True}
    ):
        raise ValueError("LAYOUTLM_QA_TOKENIZER")
    return tokenizer


def prepare(tokenizer, value):
    import torch

    inputs = tokenizer(
        value["question"].split(),
        value["words"],
        is_split_into_words=True,
        return_token_type_ids=True,
        return_tensors="pt",
        truncation=False,
    )
    words, sequence = inputs.word_ids(0), inputs.sequence_ids(0)
    if len(words) > 512:
        raise ValueError("LAYOUTLM_QA_TOKEN_LIMIT")
    boxes = [[round(c * 1000) for c in box] for box in value["boxes"]]
    inputs["bbox"] = torch.tensor(
        [
            [
                boxes[w]
                if s == 1 and w is not None
                else [1000] * 4
                if int(i) == tokenizer.sep_token_id
                else [0] * 4
                for i, s, w in zip(inputs["input_ids"][0], sequence, words, strict=True)
            ]
        ]
    )
    return inputs, words, sequence


def answer(model, processor, config, value, backend):
    import torch

    tokenizer, _ = processor
    inputs, words, sequence = prepare(tokenizer, value)
    valid = torch.tensor([s == 1 and w is not None for s, w in zip(sequence, words, strict=True)])
    if not bool(valid.any()):
        raise ValueError("LAYOUTLM_QA_GROUNDING")
    output = model(**{k: v.to(backend) for k, v in inputs.items()})
    start, end = output.start_logits[0].float().cpu(), output.end_logits[0].float().cpu()
    token = output.token_logits
    if (
        start.shape != valid.shape
        or end.shape != valid.shape
        or not bool(torch.isfinite(start).all() and torch.isfinite(end).all())
    ):
        raise ValueError("LAYOUTLM_QA_NONFINITE")
    invoice = config.get("token_classification", False)
    if invoice and (
        token is None or token.shape != (1, len(words), 2) or not bool(torch.isfinite(token).all())
    ):
        raise ValueError("LAYOUTLM_QA_TOKEN_LOGITS")
    # Publisher normalizes over document tokens and CLS, then excludes CLS from
    # candidate spans. Keep its null mass in the score without returning null.
    normalized = valid.clone()
    normalized[inputs["input_ids"][0] == tokenizer.cls_token_id] = True
    start = start.masked_fill(~normalized, -10000.0).softmax(0)
    end = end.masked_fill(~normalized, -10000.0).softmax(0)
    start[0] = end[0] = 0
    scores = start[:, None] * end[None, :]
    positions = torch.arange(len(words))
    allowed = (
        (positions[:, None] <= positions[None, :])
        & (positions[None, :] - positions[:, None] < (512 if invoice else 15))
        & valid[:, None]
        & valid[None, :]
    )
    scores = scores.masked_fill(~allowed, -1)
    a, b = divmod(int(scores.argmax()), len(words))
    selected_tokens = range(a, b + 1)
    if invoice:
        selected_tokens = [i for i in selected_tokens if int(token[0, i].argmax()) == 1]
    selected = list(dict.fromkeys(words[i] for i in selected_tokens if valid[i]))
    if not selected:
        raise ValueError("LAYOUTLM_QA_NO_SELECTED_WORDS")
    boxes = [value["boxes"][i] for i in selected]
    return dict(
        answer=" ".join(value["words"][i] for i in selected),
        score=float(scores[a, b]),
        wordStart=selected[0],
        wordEnd=selected[-1],
        box=[
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        ],
    )
