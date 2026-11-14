"""Complete tiny document and invoice graphs plus original metadata regressions."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from oryn_engine import layoutlm_tasks as l


def fixture(root, invoice=False, precision="float32"):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors

    root.mkdir(exist_ok=True)
    c = l.template("invoices.json" if invoice else "document.json")
    c.update(
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=261,
    )
    (root / "config.json").write_text(json.dumps(c))
    for name in ("special_tokens_map.json",):
        (root / name).write_bytes((l.DATA / name).read_bytes())
    vocab = {
        t: i
        for i, t in enumerate(
            [
                "<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "<mask>",
                *sorted(pre_tokenizers.ByteLevel.alphabet()),
            ]
        )
    }
    tokenizer = Tokenizer(models.BPE(vocab, [], unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.RobertaProcessing(
        ("</s>", 2), ("<s>", 0), add_prefix_space=False
    )
    tokenizer.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(
        '{"add_prefix_space":false,"add_prefix_space":true,"auto_map":{"AutoTokenizer":"untrusted.Tokenizer"}}'
    )
    torch.manual_seed(6180)
    m = l.native_model(c)
    if invoice:
        with torch.no_grad():
            m.token_classifier_head.weight.zero_()
            m.token_classifier_head.bias.copy_(torch.tensor([-1.0, 1.0]))
    state = {
        k: v.contiguous() if k == l.POSITION else v.to(getattr(torch, precision)).contiguous()
        for k, v in m.state_dict().items()
    }
    save_file(state, root / "model.safetensors")
    m.load_state_dict({k: v if k == l.POSITION else v.float() for k, v in state.items()})
    return c, m


def question():
    return dict(
        imageArtifactId="artifact_" + "a" * 24,
        question="Number?",
        words=["Invoice", "123", "Total", "400"],
        boxes=[
            [0.1, 0.1, 0.2, 0.2],
            [0.3, 0.1, 0.4, 0.2],
            [0.1, 0.3, 0.2, 0.4],
            [0.3, 0.3, 0.4, 0.4],
        ],
    )


@pytest.mark.parametrize("invoice", [False, True])
def test_original_complete_geometry(invoice):
    name = "invoices" if invoice else "document"
    c = l.template(name + ".json")
    h = json.loads(
        (Path(__file__).parent / "fixtures/layoutlm" / f"{name}.header.json").read_text()
    )
    l.validate_weight_shapes(c, {k: v["shape"] for k, v in h.items() if k != "__metadata__"})
    assert len(h) - ("__metadata__" in h) == (208 if invoice else 206)


@pytest.mark.parametrize("invoice", [False, True])
@pytest.mark.parametrize("precision", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_complete_loaded_graph_parity(tmp_path, invoice, precision, backend):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    c, reference = fixture(tmp_path, invoice, precision)
    model, processor, _ = l.load_components(tmp_path, c, backend)
    reference.to(backend)
    inputs, _, _ = l.prepare(processor[0], question())
    inputs = {k: v.to(backend) for k, v in inputs.items()}
    with torch.inference_mode():
        actual = model(**inputs)
        expected = reference(**inputs)
        torch.testing.assert_close(actual.start_logits, expected.start_logits, atol=0, rtol=0)
        torch.testing.assert_close(actual.end_logits, expected.end_logits, atol=0, rtol=0)
        if invoice:
            torch.testing.assert_close(actual.token_logits, expected.token_logits, atol=0, rtol=0)
        result = l.answer(model, processor, c, question(), backend)
    assert result["answer"] and 0 <= result["score"] <= 1
    from oryn_engine.shared_tasks import validate_task_output

    validate_task_output(l.TASK, result)


@pytest.mark.parametrize("invoice", [False, True])
def test_metadata_audit_resolves_original_without_code_or_weights(tmp_path, invoice):
    from test_metadata_audit import row, snapshot

    from oryn_engine.metadata_audit import inspect_row

    c = l.template("invoices.json" if invoice else "document.json")
    hub = snapshot(
        tmp_path, config=c, files=[(n, 100, "b" * 64) for n in l.FILES], pipeline_tag=l.TASK
    )
    hub.client.read_json.side_effect = lambda repository, revision, name: (
        c if name == "config.json" else l.template(name)
    )
    result = inspect_row(row(task=l.TASK), hub)
    assert result["status"] == "metadata-compatible", result
    assert {f["name"] for f in result["selectedFiles"]} == set(l.FILES)
    assert not result["executionVerified"]
    hub.client.download.assert_not_called()


@pytest.mark.parametrize("missing", [None, *l.FILES])
def test_discovery_requires_complete_original_export(missing):
    from oryn_engine.model_layouts import assess_layout

    names = {*l.FILES, "pytorch_model.bin", "model.py", "vocab.json", "merges.txt"}
    if missing:
        names.remove(missing)
    result = assess_layout(sorted(names), l.template("invoices.json"), l.TASK)
    if missing:
        assert result.artifact_reasons
    else:
        assert set(result.selected_files) == set(l.FILES) and not result.runtime_reasons


@pytest.mark.parametrize(
    "key,value",
    [
        ("num_hidden_layers", 13),
        ("hidden_size", 25),
        ("max_2d_position_embeddings", 512),
        ("type_vocab_size", 2),
        ("token_classification", 1),
        ("token_classifier_reduction", "mean"),
        ("token_classifier_constant", 1.0),
        ("auto_map", {"AutoModel": "bad.Model"}),
    ],
)
def test_unknown_or_inconsistent_geometry_rejected(key, value):
    c = l.template("invoices.json")
    c[key] = value
    with pytest.raises(ValueError):
        l.validate_config(c)


@pytest.mark.parametrize(
    "change",
    ["missing", "auxiliary", "pooler", "extra", "shape", "integer", "nonfinite", "positions"],
)
def test_all_arrays_and_position_buffer_validated(tmp_path, change):
    from safetensors.torch import load_file

    c, _ = fixture(tmp_path, True)
    p = tmp_path / "model.safetensors"
    state = load_file(p)
    key = "qa_outputs.weight"
    if change == "missing":
        del state[key]
    elif change == "auxiliary":
        del state["token_classifier_head.weight"]
    elif change == "pooler":
        del state["layoutlm.pooler.dense.weight"]
    elif change == "extra":
        state["extra.weight"] = torch.ones(1)
    elif change == "shape":
        state[key] = state[key][:-1]
    elif change == "integer":
        state[key] = state[key].to(torch.int32)
    elif change == "nonfinite":
        state[key][0, 0] = float("nan")
    else:
        state[l.POSITION][0, 10] = 12
    save_file(state, p)
    with pytest.raises(ValueError, match="LAYOUTLM_QA"):
        l.load_components(tmp_path, c, "cpu")


@pytest.mark.parametrize("name", l.FILES)
def test_symlinks_rejected(tmp_path, name):
    c, _ = fixture(tmp_path)
    p = tmp_path / name
    other = tmp_path / "outside"
    p.rename(other)
    p.symlink_to(other)
    with pytest.raises(ValueError, match="LAYOUTLM_QA"):
        l.load_components(tmp_path, c, "cpu")


def test_exact_publisher_separator_boxes_and_word_grounding(tmp_path):
    c, _ = fixture(tmp_path)
    _, (tokenizer, _), _ = l.load_components(tmp_path, c, "cpu")
    value = question()
    inputs, words, seq = l.prepare(tokenizer, value)
    for i, s, w, box in zip(inputs["input_ids"][0], seq, words, inputs["bbox"][0], strict=True):
        expected = (
            [round(x * 1000) for x in value["boxes"][w]]
            if s == 1
            else [1000] * 4
            if i == 2
            else [0] * 4
        )
        assert box.tolist() == expected
    assert not bool(inputs["token_type_ids"].any())
    with pytest.raises(ValueError, match="TOKEN_LIMIT"):
        l.prepare(tokenizer, {**value, "question": "word " * 512})


@pytest.mark.parametrize("mode", ["nonconsecutive", "empty", "nonfinite", "span_nonfinite"])
@pytest.mark.parametrize("long_span", [False, True])
def test_invoice_token_head_controls_grounded_answer(tmp_path, mode, long_span):
    c, _ = fixture(tmp_path, True)
    _, processor, _ = l.load_components(tmp_path, c, "cpu")
    value = question()
    if long_span:
        value["words"] = ["Invoice", *["ignored"] * 10, "400"]
        value["boxes"] = [[0.1, 0.1, 0.2, 0.2], *[[0.2, 0.2, 0.3, 0.3]] * 10, [0.3, 0.3, 0.4, 0.4]]
    inputs, words, seq = l.prepare(processor[0], value)
    n = len(words)
    start, end = torch.full((1, n), -10.0), torch.full((1, n), -10.0)
    token = torch.zeros(1, n, 2)
    token[..., 0] = 1
    positions = [i for i, (w, s) in enumerate(zip(words, seq, strict=True)) if s == 1]
    start[0, positions[0]] = 10
    end[0, positions[-1]] = 10
    if mode != "empty":
        for i in positions:
            if words[i] in (0, len(value["words"]) - 1):
                token[0, i] = torch.tensor([0.0, 1.0])
    if mode == "nonfinite":
        token[0, 0, 0] = float("nan")
    if mode == "span_nonfinite":
        start[0, 0] = float("inf")

    def model(**kwargs):
        return SimpleNamespace(start_logits=start, end_logits=end, token_logits=token)

    if mode != "nonconsecutive":
        with pytest.raises(ValueError, match="LAYOUTLM_QA"):
            l.answer(model, processor, c, value, "cpu")
    else:
        r = l.answer(model, processor, c, value, "cpu")
        assert (
            r["answer"] == "Invoice 400"
            and (r["wordStart"], r["wordEnd"]) == (0, len(value["words"]) - 1)
            and r["box"] == [0.1, 0.1, 0.4, 0.4]
        )


@pytest.mark.parametrize("invoice", [False, True])
@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_offline_worker_uses_complete_owned_graph(tmp_path, invoice, backend):
    import os
    import subprocess
    import sys

    from PIL import Image

    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    root = tmp_path / "model"
    fixture(root, invoice)
    image = tmp_path / "page.png"
    Image.new("RGB", (32, 32), "white").save(image)
    payload = {
        "modelRoot": str(root),
        "task": l.TASK,
        "backend": backend,
        "input": question(),
        "inputPath": str(image),
        "outputPath": None,
    }
    request = dict(
        abi="1",
        packId="oryn.transformers-pytorch",
        version=builtin_registry().active_pack("oryn.transformers-pytorch").version,
        command="generate",
        payload=payload,
    )
    run = subprocess.run(
        [sys.executable, "-m", "oryn_engine.runtime_packs.transformers_worker"],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        timeout=60,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout.splitlines()[-1])
    assert result["ok"], (result, run.stderr)
    assert result["result"]["answer"]


@pytest.mark.parametrize("change", ["special_tokens", "normalizer", "prefix_algorithm"])
def test_unknown_tokenization_is_rejected(tmp_path, change):
    c, _ = fixture(tmp_path)
    if change == "special_tokens":
        p = tmp_path / "special_tokens_map.json"
        value = json.loads(p.read_text())
        value["bos_token"] = "wrong"
    else:
        p = tmp_path / "tokenizer.json"
        value = json.loads(p.read_text())
        value["normalizer" if change == "normalizer" else "pre_tokenizer"] = (
            {"type": "Lowercase"} if change == "normalizer" else {"type": "Whitespace"}
        )
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="LAYOUTLM_QA"):
        l.load_components(tmp_path, c, "cpu")
