"""Complete T5 and MobileBERT numeric graphs; only generated weights execute."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import transformers as tf

from oryn_engine import numeric_archive_tasks as family
from oryn_engine import numeric_seq2seq as seq
from oryn_engine import shared_tasks
from oryn_engine.numeric_torch_archive import inspect_archive

CASES = ["t5-relu", "t5-gated", "mobile-no-norm", "mobile-layer-norm"]


def fixture(root, case, dtype=torch.float32):
    torch.manual_seed(6130)
    if case.startswith("t5"):
        task = "summarization"
        tokenizer = tf.T5Tokenizer(
            vocab=[("<pad>", 0.0), ("</s>", 0.0), ("<unk>", 0.0), ("▁", -1.0)]
            + [(x, -2.0) for x in "abcdefghijklmnopqrstuvwxyz:"],
            extra_ids=0,
        )
        c = tf.T5Config(
            vocab_size=len(tokenizer),
            d_model=32,
            d_ff=48,
            d_kv=8,
            num_layers=2,
            num_decoder_layers=3 if case.endswith("relu") else 2,
            num_heads=4,
            feed_forward_proj="gated-gelu" if case.endswith("gated") else "relu",
            dropout_rate=0.0,
            pad_token_id=0,
            eos_token_id=1,
            decoder_start_token_id=0,
            forced_bos_token_id=5,
            tie_word_embeddings=False,
            task_specific_params={"summarization": {"prefix": "summarize: ", "num_beams": 2}},
        )
        native = tf.T5ForConditionalGeneration(c)
    else:
        task = "fill-mask"
        words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world"] + [
            f"word{i}" for i in range(7, 32)
        ]
        tokenizer = tf.BertTokenizer(vocab={v: i for i, v in enumerate(words)})
        c = tf.MobileBertConfig(
            vocab_size=32,
            hidden_size=32,
            embedding_size=8,
            intra_bottleneck_size=16,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=64,
            num_feedforward_networks=3 if case.endswith("no-norm") else 1,
            normalization_type="no_norm" if case.endswith("no-norm") else "layer_norm",
            trigram_input=case.endswith("no-norm"),
            use_bottleneck_attention=case.endswith("layer-norm"),
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        native = tf.MobileBertForMaskedLM(c)
    c._attn_implementation = "eager"
    c.architectures = [type(native).__name__]
    state = {k: v.to(dtype) for k, v in native.state_dict().items()}
    # Preserve values while physically transposing every matrix, including
    # rectangular projections. This exercises dense views throughout loading.
    state = {k: v.T.contiguous().T if v.ndim == 2 else v for k, v in state.items()}
    native.load_state_dict(state, strict=True)
    if task == "fill-mask":
        state["mobilebert.embeddings.position_ids"] = torch.arange(64).expand(1, -1)
    torch.save(state, root / "pytorch_model.bin")
    c.save_pretrained(root)
    tokenizer.save_pretrained(root)
    config = json.loads((root / "config.json").read_text())
    if task == "summarization":
        # Current Transformers moves this field out of config when saving.
        # Keep an explicit deterministic output in this generated fixture.
        config["forced_bos_token_id"] = 5
        (root / "config.json").write_text(json.dumps(config))
        seq.prepare(native, tokenizer, config)
    return native.eval(), tokenizer, config, task


def request(task):
    return (
        {"text": "hello world", "maxTokens": 5}
        if task == "summarization"
        else {"text": "hello [MASK] world", "topK": 5}
    )


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_complete_values_logits_and_task_semantics(tmp_path, case, dtype, backend):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    native, tokenizer, config, task = fixture(tmp_path, case, dtype)
    model, actual_tokenizer, actual_config = shared_tasks.load_task(tmp_path, task, backend)
    assert family.expected_shapes(config, task) == {
        k: list(v.shape) for k, v in native.state_dict().items()
    }
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor.cpu(), native.state_dict()[key], atol=0, rtol=0)
    value = request(task)
    inputs = tokenizer(
        ("summarize: " if task == "summarization" else "") + value["text"], return_tensors="pt"
    )
    if task == "summarization":
        inputs["decoder_input_ids"] = torch.tensor([[0, 5, 6]])
    with torch.inference_mode():
        expected = native(**inputs).logits
        actual = model(**{k: v.to(backend) for k, v in inputs.items()}).logits.cpu()
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)
        result = family.execute(model, actual_tokenizer, actual_config, task, value, backend)
        if task == "summarization":
            inputs.pop("decoder_input_ids")
            reference = native.generate(**inputs, max_new_tokens=5, do_sample=False)
            assert (
                result["text"] == tokenizer.decode(reference[0], skip_special_tokens=True).strip()
            )
        else:
            index = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero().item()
            scores, ids = expected[0, index].softmax(-1).topk(5)
            assert [p["token"] for p in result["predictions"]] == [
                tokenizer.decode([i]) for i in ids.tolist()
            ]
            assert [p["score"] for p in result["predictions"]] == pytest.approx(
                scores.tolist(), abs=2e-5
            )
    shared_tasks.validate_task_output(task, result)


@pytest.mark.parametrize("case", ["t5-relu", "mobile-no-norm"])
@pytest.mark.parametrize(
    "damage", ["head", "backbone", "extra", "shape", "nonfinite", "legacy-position"]
)
def test_incomplete_graphs_fail_closed(tmp_path, case, damage):
    native, _, config, task = fixture(tmp_path, case)
    state = native.state_dict()
    key = "lm_head.weight" if task == "summarization" else "cls.predictions.dense.weight"
    if damage == "head":
        del state[key]
    elif damage == "backbone":
        del state[next(iter(state))]
    elif damage == "extra":
        state["custom.head"] = torch.zeros(1)
    elif damage == "shape":
        state[key] = torch.zeros(2, 3)
    elif damage == "nonfinite":
        state[key][0, 0] = float("inf")
    else:
        state["mobilebert.embeddings.position_ids"] = torch.ones(
            1, config.get("max_position_embeddings", 64), dtype=torch.long
        )
    torch.save(state, tmp_path / "pytorch_model.bin")
    with pytest.raises(ValueError):
        family.load_components(tmp_path, task, "cpu")


@pytest.mark.parametrize("case", ["t5-relu", "mobile-no-norm"])
@pytest.mark.parametrize("damage", [None, "head", "shape", "extra"])
def test_metadata_admission_without_model_construction(tmp_path, monkeypatch, case, damage):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore
    from oryn_engine.runtime_packs.execution import PackRuntime

    _, _, config, task = fixture(tmp_path, case)
    raw = (tmp_path / "pytorch_model.bin").read_bytes()
    metadata = inspect_archive(lambda a, n: raw[a : a + n], len(raw))
    key = "lm_head.weight" if task == "summarization" else "cls.predictions.dense.weight"
    if damage == "head":
        del metadata["tensors"][key]
    if damage == "shape":
        metadata["tensors"][key]["shape"] = [1, 2]
    if damage == "extra":
        metadata["tensors"]["custom.head"] = {"shape": [1], "dtype": "F32"}

    class Hub(FakeHub):
        def read_numeric_archive(self, *args):
            return metadata

        def read_json(self, repo, revision, name):
            return json.loads((tmp_path / name).read_text())

        def download(self, *args, **kwargs):
            raise AssertionError("No original download")

    hub = Hub(
        config=config,
        pipeline_tag=task,
        files=[(p.name, p.stat().st_size, "a" * 64) for p in tmp_path.iterdir()],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("No native model construction during metadata inspection")

    monkeypatch.setattr(tf.AutoConfig, "for_model", forbidden)
    store = ModelStore(tmp_path / "store", hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    result = store.inspect(
        {"provider": "huggingface", "repository": "test/numeric-text", "revision": "main"}
    )
    assert result["executable"] == (damage is None), result.get("blockingReasons")
    if damage is None:
        assert result["runtimePack"]["runnerId"] == "transformers.shared-tasks"


@pytest.mark.parametrize("case", ["t5-relu", "mobile-no-norm"])
def test_repeated_offline_worker(tmp_path, case):
    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry

    _, _, _, task = fixture(tmp_path, case)
    pack = builtin_registry().active_pack("oryn.transformers-pytorch")
    message = dict(
        abi=pack.abi,
        packId=pack.pack_id,
        version=pack.version,
        command="generate",
        payload=dict(
            modelRoot=str(tmp_path),
            task=task,
            backend="cpu",
            input=request(task),
            inputPath=None,
            outputPath=None,
        ),
    )
    results = []
    for _ in range(2):
        child = subprocess.run(
            [sys.executable, "-m", "oryn_engine.runtime_packs.transformers_worker"],
            input=json.dumps(message) + "\n",
            text=True,
            capture_output=True,
            timeout=60,
            env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        assert child.returncode == 0, child.stderr
        result = next(
            json.loads(s) for s in child.stdout.splitlines() if json.loads(s)["type"] == "result"
        )
        assert result["ok"], (result, child.stderr)
        results.append(result["result"])
    assert results[0] == results[1]


@pytest.mark.parametrize("name,task", [("t5", "summarization"), ("mobilebert", "fill-mask")])
def test_complete_original_metadata(name, task):
    p = Path(__file__).parent / "fixtures/numeric_archive"
    config = json.loads((p / (name + ".config.json")).read_text())
    metadata = json.loads((p / (name + ".tensor-structure.json")).read_text())
    shapes, legacy = family.validate_archive(config, task, metadata)
    assert len(shapes) == (134 if name == "t5" else 1120)
    assert bool(legacy) == (name == "mobilebert")


@pytest.mark.parametrize("scale", [0.0, 1e-7, 1e-4, 1.0, 100.0])
@pytest.mark.parametrize("epsilon", [1e-12, 1e-5])
def test_mobilebert_mps_normalization_against_float64_reference(scale, epsilon):
    from oryn_engine.numeric_mobilebert import stable_layer_norm

    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    generator = torch.Generator().manual_seed(6130)
    value = torch.randn(2, 7, 32, generator=generator) * scale + 0.25
    weight = torch.linspace(0.5, 1.5, 32)
    bias = torch.linspace(-0.5, 0.5, 32)
    expected = torch.nn.functional.layer_norm(
        value.double(), (32,), weight.double(), bias.double(), epsilon
    )
    actual = stable_layer_norm(value.to("mps"), weight.to("mps"), bias.to("mps"), epsilon).cpu()
    torch.testing.assert_close(actual.double(), expected, atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize(
    "key,value",
    [
        ("d_model", True),
        ("num_layers", 0),
        ("num_decoder_layers", 49),
        ("d_kv", 0),
        ("relative_attention_num_buckets", 31),
        ("relative_attention_max_distance", 16),
        ("feed_forward_proj", "custom"),
        ("decoder_start_token_id", -1),
        ("tie_encoder_decoder", True),
        ("auto_map", {"AutoModel": "custom.Model"}),
    ],
)
def test_t5_metadata_rejects_unowned_geometry(tmp_path, key, value):
    _, _, config, task = fixture(tmp_path, "t5-relu")
    config[key] = value
    with pytest.raises(ValueError):
        family.validate_config(config, task)


@pytest.mark.parametrize(
    "key,value",
    [
        ("intra_bottleneck_size", 12),
        ("embedding_size", 32),
        ("use_bottleneck", False),
        ("key_query_shared_bottleneck", False),
        ("normalization_type", "custom"),
        ("true_hidden_size", 32),
        ("trigram_input", 1),
        ("num_feedforward_networks", 9),
    ],
)
def test_mobilebert_metadata_rejects_unowned_geometry(tmp_path, key, value):
    _, _, config, task = fixture(tmp_path, "mobile-no-norm")
    config[key] = value
    with pytest.raises(ValueError):
        family.validate_config(config, task)
