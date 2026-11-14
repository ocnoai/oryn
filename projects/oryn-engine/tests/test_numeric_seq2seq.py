"""Complete numeric QA/encoder-decoder graphs, deterministic task semantics."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import transformers as tf

from oryn_engine import numeric_archive_tasks as family
from oryn_engine import numeric_seq2seq as seq
from oryn_engine import shared_tasks

CASES = [("question-answering", "bert"), ("summarization", "bart"), ("translation", "m2m_100")]


def fixture(root, task, kind, dtype=torch.float32):
    torch.manual_seed(6128)
    if kind == "bert":
        from test_numeric_archive_tasks import fixture as encoder_fixture

        native, tokenizer, config = encoder_fixture(root, task, kind, dtype)
        return native, tokenizer, config
    vocab = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
        **{chr(97 + i): i + 5 for i in range(26)},
    }
    tokenizer = (
        tf.BartTokenizer(vocab=vocab, merges=[])
        if kind == "bart"
        else tf.NllbTokenizer(
            vocab=vocab,
            merges=[],
            src_lang="eng_Latn",
            additional_special_tokens=["eng_Latn", "fra_Latn", "deu_Latn"],
        )
    )
    c = tf.AutoConfig.for_model(
        kind,
        vocab_size=len(tokenizer),
        d_model=32,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=48,
        decoder_ffn_dim=48,
        max_position_embeddings=64,
        bos_token_id=0,
        pad_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=2,
        tie_word_embeddings=False,
        tokenizer_class=tokenizer.__class__.__name__,
    )
    c._attn_implementation = "eager"
    c.architectures = [family.ARCHITECTURES[task, kind]]
    native = tf.AutoModelForSeq2SeqLM.from_config(c).eval()
    if kind == "bart":
        native.final_logits_bias[0, 5] = 3
    # Only serialized numeric tensors have archive precision. Native positional
    # buffers are non-persistent and retain their deterministic float32 values.
    state = {name: tensor.to(dtype) for name, tensor in native.state_dict().items()}
    torch.save(state, root / "pytorch_model.bin")
    native.load_state_dict(state, strict=True)
    c.save_pretrained(root)
    tokenizer.save_pretrained(root)
    config = json.loads((root / "config.json").read_text())
    seq.prepare(native, tokenizer, config)
    return native, tokenizer, config


def value(task):
    return (
        {"question": "who", "context": "  hello Élise world  ", "maxAnswerLength": 2}
        if task == "question-answering"
        else {
            "text": "hello world",
            "maxTokens": 4,
            **({"targetLanguage": "fr"} if task == "translation" else {}),
        }
    )


@pytest.mark.parametrize("task,kind", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_complete_native_logits_and_generation(tmp_path, task, kind, dtype, backend):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    native, tokenizer, config = fixture(tmp_path, task, kind, dtype)
    model, actual_tokenizer, actual_config = shared_tasks.load_task(tmp_path, task, backend)
    assert family.expected_shapes(config, task) == {
        k: list(v.shape) for k, v in native.state_dict().items()
    }
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor.cpu(), native.state_dict()[name], atol=0, rtol=0)
    request = value(task)
    inputs = (
        tokenizer(request["question"], request["context"], return_tensors="pt")
        if kind == "bert"
        else tokenizer(request["text"], return_tensors="pt")
    )
    if kind != "bert":
        inputs["decoder_input_ids"] = torch.tensor([[2, 5, 6]])
    with torch.inference_mode():
        expected = native(**inputs)
        actual = model(**{k: v.to(backend) for k, v in inputs.items()})
        for name in ["start_logits", "end_logits"] if kind == "bert" else ["logits"]:
            torch.testing.assert_close(
                getattr(actual, name).cpu(), getattr(expected, name), atol=3e-5, rtol=3e-4
            )
        result = family.execute(model, actual_tokenizer, actual_config, task, request, backend)
        reference = family.execute(native, tokenizer, config, task, request, "cpu")
    assert result == reference
    shared_tasks.validate_task_output(task, result)
    if kind == "bert":
        assert result["answer"] == request["context"][result["start"] : result["end"]]
    else:
        assert result["text"]


@pytest.mark.parametrize("task,kind", CASES)
@pytest.mark.parametrize("part", ["backbone", "head", "shape", "extra", "nonfinite"])
def test_reject_incomplete_or_invalid_complete_graphs(tmp_path, task, kind, part):
    native, _, _ = fixture(tmp_path, task, kind)
    state = native.state_dict()
    head = "qa_outputs.weight" if kind == "bert" else "lm_head.weight"
    key = head if part in {"head", "shape", "nonfinite"} else next(iter(state))
    if part in {"backbone", "head"}:
        del state[key]
    elif part == "shape":
        state[key] = torch.zeros(2, 3)
    elif part == "extra":
        state["repository.custom_head"] = torch.zeros(1)
    else:
        state[key][0, 0] = float("inf")
    torch.save(state, tmp_path / "pytorch_model.bin")
    with pytest.raises(ValueError):
        family.load_components(tmp_path, task, "cpu")


@pytest.mark.parametrize("task,kind", CASES)
def test_repeated_isolated_worker(tmp_path, task, kind):
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
            input=value(task),
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


@pytest.mark.parametrize("language", [None, "unknown", "fr-FR", "fra_Latn", "../fr"])
def test_nllb_requires_explicit_supported_target_language(tmp_path, language):
    model, tokenizer, config = fixture(tmp_path, "translation", "m2m_100")
    with pytest.raises(ValueError, match="TARGET_LANGUAGE"):
        seq.execute(
            model,
            tokenizer,
            config,
            "translation",
            {"text": "hello", "maxTokens": 4, "targetLanguage": language},
            "cpu",
        )


@pytest.mark.parametrize(
    "name,task",
    [("chinese-qa", "question-answering"), ("bart", "summarization"), ("nllb", "translation")],
)
def test_complete_original_archive_metadata(name, task):
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
    assert bool(legacy) == (name == "chinese-qa")


@pytest.mark.parametrize("task,kind", CASES)
@pytest.mark.parametrize("damage", [None, "missing-head", "extra-head", "shape"])
def test_metadata_discovery_requires_complete_task_graph(tmp_path, monkeypatch, task, kind, damage):
    from types import SimpleNamespace

    from test_models import FakeHub

    from oryn_engine.models import ModelStore
    from oryn_engine.runtime_packs.execution import PackRuntime

    _, _, config = fixture(tmp_path, task, kind)
    # Original NLLB 1.3B keeps its tokenizer class only in tokenizer_config.json.
    if kind == "m2m_100":
        config.pop("tokenizer_class", None)
    metadata = {
        "format": "torch-numeric-zip-v1",
        "byteOrder": "little",
        "tensors": {
            k: {"shape": v, "dtype": "F32"} for k, v in family.expected_shapes(config, task).items()
        },
    }
    head = "qa_outputs.weight" if kind == "bert" else "lm_head.weight"
    if damage == "missing-head":
        del metadata["tensors"][head]
    if damage == "extra-head":
        metadata["tensors"]["custom.head"] = {"shape": [1], "dtype": "F32"}
    if damage == "shape":
        metadata["tensors"][head]["shape"] = [1, 2]

    class Hub(FakeHub):
        def read_numeric_archive(self, *args):
            return metadata

        def read_json(self, repo, revision, name):
            return json.loads((tmp_path / name).read_text())

        def download(self, *args, **kwargs):
            raise AssertionError("Original checkpoint download forbidden")

    files = [
        ("config.json", 1000, "a" * 64),
        ("pytorch_model.bin", 100000, "b" * 64),
        ("tokenizer.json", 1000, "c" * 64),
        ("tokenizer_config.json", 1000, "d" * 64),
    ]
    hub = Hub(config=config, pipeline_tag=task, files=files)

    def forbidden(*args, **kwargs):
        raise AssertionError("Native construction forbidden during inspection")

    monkeypatch.setattr(tf.AutoConfig, "for_model", forbidden)
    store = ModelStore(tmp_path / "store", hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    result = store.inspect(
        {"provider": "huggingface", "repository": "test/numeric-seq2seq", "revision": "main"}
    )
    assert result["executable"] == (damage is None), result.get("blockingReasons")
    if damage is None:
        assert result["runtimePack"]["runnerId"] == "transformers.shared-tasks"
        assert result["taskProfile"]["contract"]["id"] == family.TASK_CONTRACTS[task]


@pytest.mark.parametrize("damage", ["unknown-class", "custom-code", "conflict", "invalid-source"])
def test_nllb_tokenizer_metadata_rejects_wrong_semantics(damage):
    config = {"tokenizer_class": "NllbTokenizer"}
    metadata = {"tokenizer_class": "NllbTokenizer", "src_lang": "eng_Latn"}
    if damage == "unknown-class":
        metadata["tokenizer_class"] = "M2M100Tokenizer"
    if damage == "custom-code":
        metadata["auto_map"] = {"AutoTokenizer": "repo.Custom"}
    if damage == "conflict":
        config["tokenizer_class"] = "M2M100Tokenizer"
    if damage == "invalid-source":
        metadata["src_lang"] = ["eng_Latn"]
    with pytest.raises(ValueError):
        seq.validate_tokenizer_metadata(config, metadata)


@pytest.mark.parametrize("task,kind", CASES)
def test_context_and_output_generation_budgets_never_silently_truncate(tmp_path, task, kind):
    native, tokenizer, config = fixture(tmp_path, task, kind)
    request = value(task)
    request["context" if kind == "bert" else "text"] = "hello " * 80
    with torch.inference_mode(), pytest.raises(ValueError, match="INPUT_LIMIT|TOKEN_LIMIT"):
        family.execute(native, tokenizer, config, task, request, "cpu")
    if kind != "bert":
        with torch.inference_mode(), pytest.raises(ValueError, match="TOKEN_LIMIT"):
            family.execute(native, tokenizer, config, task, value(task) | {"maxTokens": 64}, "cpu")
