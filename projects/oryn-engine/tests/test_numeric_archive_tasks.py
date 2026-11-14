"""Closed numeric archives bind complete native graphs; originals remain metadata-only."""

import json
import pickle

import pytest
import torch
import transformers as tf

from oryn_engine import numeric_archive_tasks as family

CASES = [case for case in family.ARCHITECTURES if case[0] in {"fill-mask", "token-classification"}]


def fixture(root, task, kind, dtype=torch.float32, legacy=False):
    torch.manual_seed(6127)
    options = dict(
        vocab_size=32,
        max_position_embeddings=64,
        pad_token_id=0,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        type_vocab_size=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    if kind == "distilbert":
        options.update(
            dim=32, hidden_dim=48, n_layers=2, n_heads=4, dropout=0.0, attention_dropout=0.0
        )
    if kind == "modernbert":
        options.update(local_attention=16, global_attn_every_n_layers=2, reference_compile=False)
    if kind == "mobilebert":
        options.update(embedding_size=8, intra_bottleneck_size=16)
    if task == "token-classification":
        options.update(
            id2label={0: "O", 1: "B-PER", 2: "I-PER"}, label2id={"O": 0, "B-PER": 1, "I-PER": 2}
        )
    config = tf.AutoConfig.for_model(kind, **options)
    config._attn_implementation = "eager"
    cls = getattr(tf, family.ARCHITECTURES[(task, kind)])
    native = cls(config).eval().to(dtype).float()
    config.architectures = [cls.__name__]
    config.save_pretrained(root)
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "élise", "world"]
    words += [f"word{i}" for i in range(8, 32)]
    tokenizer = tf.BertTokenizer(vocab={w: i for i, w in enumerate(words)}, do_lower_case=True)
    tokenizer.save_pretrained(root)
    state = native.to(dtype).state_dict()
    if legacy:
        prefix = "roberta" if kind == "xlm-roberta" else "bert"
        state[prefix + ".embeddings.position_ids"] = torch.arange(64).expand(1, -1)
    torch.save(state, root / "pytorch_model.bin")
    native.float()
    return native, tokenizer, json.loads((root / "config.json").read_text())


@pytest.mark.parametrize("task,kind", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_complete_native_graph_values_and_heads(tmp_path, monkeypatch, task, kind, dtype, backend):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    native, tokenizer, config = fixture(tmp_path, task, kind, dtype)

    def forbidden(*args, **kwargs):
        raise AssertionError("Object deserialization is forbidden")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(pickle, "load", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)
    model, actual_tokenizer, actual_config = family.load_components(tmp_path, task, backend)
    assert family.expected_shapes(config, task) == {
        k: list(v.shape) for k, v in native.state_dict().items()
    }
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), native.state_dict()[name], rtol=0, atol=0)
    text = "hello [MASK] world" if task == "fill-mask" else "  hello Élise  "
    inputs = tokenizer(text, return_tensors="pt")
    if kind in {"distilbert", "modernbert"}:
        inputs.pop("token_type_ids", None)
    with torch.inference_mode():
        expected = native(**inputs).logits
        actual = model(**{k: v.to(backend) for k, v in inputs.items()}).logits.cpu()
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-5)
        result = family.execute(
            model, actual_tokenizer, actual_config, task, {"text": text}, backend
        )
    if task == "fill-mask":
        index = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero().item()
        scores, ids = expected[0, index].softmax(-1).topk(5)
        assert [x["token"] for x in result["predictions"]] == [
            tokenizer.decode([i]) for i in ids.tolist()
        ]
        assert [x["score"] for x in result["predictions"]] == pytest.approx(
            scores.tolist(), abs=2e-5
        )
    else:
        assert result["tokens"]
        for item in result["tokens"]:
            assert text[item["start"] : item["end"]].strip()
            assert item["label"] == config["id2label"][str(item["labelId"])]


@pytest.mark.parametrize(
    "task,kind", [("fill-mask", "bert"), ("token-classification", "xlm-roberta")]
)
@pytest.mark.parametrize("corrupt", [False, True])
def test_legacy_position_buffer_is_validated(tmp_path, task, kind, corrupt):
    native, _, _ = fixture(tmp_path, task, kind, legacy=True)
    if corrupt:
        state = native.state_dict()
        prefix = "bert" if kind == "bert" else "roberta"
        state[prefix + ".embeddings.position_ids"] = torch.ones(1, 64, dtype=torch.int64)
        torch.save(state, tmp_path / "pytorch_model.bin")
        with pytest.raises(ValueError, match="POSITION_BUFFER"):
            family.load_components(tmp_path, task, "cpu")
    else:
        family.load_components(tmp_path, task, "cpu")


@pytest.mark.parametrize("change", ["missing", "extra", "shape", "tied", "nonfinite"])
def test_incomplete_or_conflicting_graph_never_loads(tmp_path, change):
    native, _, _ = fixture(tmp_path, "fill-mask", "bert")
    state = native.state_dict()
    if change == "missing":
        del state["bert.encoder.layer.0.attention.self.query.weight"]
    if change == "extra":
        state["custom.head"] = torch.zeros(1)
    if change == "shape":
        state["cls.predictions.transform.dense.weight"] = torch.zeros(2, 2)
    if change == "tied":
        state["cls.predictions.decoder.weight"] = (
            state["cls.predictions.decoder.weight"].clone() + 1
        )
    if change == "nonfinite":
        state["cls.predictions.transform.dense.weight"][0, 0] = float("nan")
    torch.save(state, tmp_path / "pytorch_model.bin")
    with pytest.raises(ValueError):
        family.load_components(tmp_path, "fill-mask", "cpu")


@pytest.mark.parametrize("text", ["hello world", "[MASK] [MASK]", "hello " * 70 + "[MASK]"])
def test_mask_count_and_context_budget(tmp_path, text):
    fixture(tmp_path, "fill-mask", "distilbert")
    model, tokenizer, config = family.load_components(tmp_path, "fill-mask", "cpu")
    with torch.inference_mode(), pytest.raises(ValueError):
        family.execute(model, tokenizer, config, "fill-mask", {"text": text}, "cpu")


@pytest.mark.parametrize(
    "name,task",
    [
        ("modernbert", "fill-mask"),
        ("clinicalbert", "fill-mask"),
        ("punctuate", "token-classification"),
    ],
)
def test_original_metadata_has_complete_native_shapes(name, task):
    from pathlib import Path

    from oryn_engine.numeric_state_dict import parse

    root = Path(__file__).parent / "fixtures/numeric_archive"
    config = json.loads((root / (name + ".config.json")).read_text())
    tensors = parse((root / (name + ".metadata")).read_bytes())
    metadata = {
        "format": "torch-numeric-zip-v1",
        "byteOrder": "little",
        "tensors": {
            k: {"shape": list(v.shape), "dtype": v.storage.dtype} for k, v in tensors.items()
        },
    }
    shapes, legacy = family.validate_archive(config, task, metadata)
    assert len(shapes) == len(tensors)
    assert bool(legacy) == (name == "punctuate")


@pytest.mark.parametrize("task,kind", CASES)
@pytest.mark.parametrize("damage", [None, "missing", "extra", "dtype", "shape", "custom-code"])
def test_discovery_requires_complete_numeric_graph_without_execution(
    tmp_path, task, kind, damage, monkeypatch
):
    from pathlib import Path
    from types import SimpleNamespace

    from test_models import FakeHub

    from oryn_engine.models import ModelStore
    from oryn_engine.runtime_packs.execution import PackRuntime

    _, _, config = fixture(tmp_path, task, kind)
    metadata = {
        "format": "torch-numeric-zip-v1",
        "byteOrder": "little",
        "tensors": {
            k: {"shape": v, "dtype": "F32"} for k, v in family.expected_shapes(config, task).items()
        },
    }
    key = next(iter(metadata["tensors"]))
    if damage == "missing":
        del metadata["tensors"][key]
    if damage == "extra":
        metadata["tensors"]["custom.head"] = {"shape": [1], "dtype": "F32"}
    if damage == "dtype":
        metadata["tensors"][key]["dtype"] = "I64"
    if damage == "shape":
        metadata["tensors"][key]["shape"] = [1]
    if damage == "custom-code":
        config["auto_map"] = {"AutoModel": "repository.Model"}

    class Hub(FakeHub):
        def read_numeric_archive(self, repo, revision, filename, size):
            assert filename == "pytorch_model.bin" and size == 100000
            return metadata

        def download(self, *args, **kwargs):
            raise AssertionError("Original weight download is forbidden")

    hub = Hub(
        config=config,
        pipeline_tag=task,
        files=[
            ("config.json", 1000, "a" * 64),
            ("pytorch_model.bin", 100000, "b" * 64),
            ("tokenizer.json", 1000, "c" * 64),
            ("repository.py", 100, "d" * 64),
        ],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Native model construction during metadata inspection")

    monkeypatch.setattr(tf.AutoConfig, "for_model", forbidden)
    store = ModelStore(tmp_path / "store", hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    result = store.inspect(
        {"provider": "huggingface", "repository": "test/numeric-encoder", "revision": "main"}
    )
    assert result["executable"] == (damage is None), result.get("blockingReasons")
    if damage is None:
        assert {x["name"] for x in result["selectedFiles"]} == {
            "config.json",
            "pytorch_model.bin",
            "tokenizer.json",
        }
        assert result["runtimePack"]["runnerId"] == "transformers.shared-tasks"
        assert result["taskProfile"]["contract"]["id"] == family.TASK_CONTRACTS[task]


@pytest.mark.parametrize("task,kind", CASES)
def test_repeated_offline_worker_uses_shared_pack(tmp_path, task, kind):
    import os
    import subprocess
    import sys

    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry

    fixture(tmp_path, task, kind)
    pack = builtin_registry().active_pack("oryn.transformers-pytorch")
    request = dict(
        abi=pack.abi,
        packId=pack.pack_id,
        version=pack.version,
        command="generate",
        payload=dict(
            modelRoot=str(tmp_path),
            task=task,
            backend="cpu",
            input={"text": "hello [MASK] world" if task == "fill-mask" else "hello Élise"},
            inputPath=None,
            outputPath=None,
        ),
    )
    results = []
    for _ in range(2):
        child = subprocess.run(
            [sys.executable, "-m", "oryn_engine.runtime_packs.transformers_worker"],
            input=json.dumps(request) + "\n",
            text=True,
            capture_output=True,
            timeout=60,
            env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        assert child.returncode == 0, child.stderr
        result = next(
            json.loads(line)
            for line in child.stdout.splitlines()
            if json.loads(line)["type"] == "result"
        )
        assert result["ok"], (result, child.stderr)
        results.append(result["result"])
    assert results[0] == results[1]


@pytest.mark.parametrize("task,kind", CASES)
def test_safe_exports_and_other_archives_keep_separate_routes(tmp_path, task, kind):
    from oryn_engine.model_layouts import assess_layout

    _, _, config = fixture(tmp_path, task, kind)
    names = [p.name for p in tmp_path.iterdir()]
    layout = assess_layout(names, config, task)
    assert not layout.artifact_reasons and not layout.runtime_reasons
    assert "pytorch_model.bin" in layout.selected_files
    for name in ["model.safetensors", "flax_model.msgpack", "tf_model.h5"]:
        assert not family.selected(config, task, names + [name])
    assert not family.selected(
        config, task, [n for n in names if n != "pytorch_model.bin"] + ["model.pt"]
    )
    assert not family.selected(config | {"model_type": "custom"}, task, names)
