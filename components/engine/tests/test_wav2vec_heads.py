"""Original graph metadata and complete generated CPU/Metal classifier regressions."""

import json
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from oryn_engine import shared_tasks
from oryn_engine import wav2vec_heads as binding
from oryn_engine.generated_capabilities import normalized_transformers_config

FIXTURES = Path(__file__).parent / "fixtures/wav2vec_heads"


def document(family, name):
    return json.loads((FIXTURES / family / name).read_text())


def configuration(family, name="tiny-config.json"):
    return binding.normalized_config(document(family, name), binding.TASK)


@pytest.fixture(params=["age", "dual"])
def checkpoint(request, tmp_path):
    family = request.param
    config = configuration(family)
    model = binding.native_model(config)
    state = model.state_dict()
    generator = torch.Generator().manual_seed(61283)
    for name, tensor in sorted(state.items()):
        tensor.copy_(torch.randn(tensor.shape, generator=generator) * 0.15)
        if name.endswith("layer_norm.weight"):
            tensor.add_(1)
    if family == "age":
        prefix = "wav2vec2.encoder.pos_conv_embed.conv."
        state = {
            k.replace(prefix + "parametrizations.weight.original0", prefix + "weight_g").replace(
                prefix + "parametrizations.weight.original1", prefix + "weight_v"
            ): v
            for k, v in state.items()
        }
    save_file(state, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(document(family, "tiny-config.json")))
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(document(family, "preprocessor_config.json"))
    )
    return tmp_path, family, config


@pytest.mark.parametrize("family,count", [("age", 430), ("dual", 215)])
def test_original_complete_tensor_graph_and_processor(family, count):
    config = configuration(family, "config.json")
    shapes = document(family, "original-shapes.json")
    model = binding.validate_weight_shapes(config, shapes)
    assert len(shapes) == len(model.state_dict()) == count
    binding.validate_processor(document(family, "preprocessor_config.json"), config["model_type"])
    assert binding.minimum_samples(config) == 400
    assert binding.duration_limit(config) == (30 if family == "age" else 12)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("case", range(4))
def test_publisher_full_graph_probabilities_preprocessing_and_padding(checkpoint, backend, case):
    root, family, config = checkpoint
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    model, processor, loaded = binding.load_components(root, config, backend)
    assert loaded == config
    # Publisher vectors are CPU results. Metal kernels round differently between
    # PyTorch releases (2.13 drifted by ~1e-4; 2.14 matches CPU), so GPU runs are
    # held to the CPU reference with a bounded Metal tolerance.
    prefix = str(case)
    scale = 100 if backend == "mps" else 1
    with np.load(FIXTURES / family / "reference-vectors.npz") as gold:
        raw = [gold[f"{case}_raw_{i}"] for i in range(len(gold[f"{case}_input"]))]
        processed = processor(raw, sampling_rate=16000, padding=True, return_tensors="pt")
        np.testing.assert_allclose(
            processed["input_values"].numpy(), gold[f"{case}_input"], atol=2e-6
        )
        np.testing.assert_array_equal(processed["attention_mask"].numpy(), gold[f"{case}_mask"])
        with torch.inference_mode():
            output = model(**{k: v.to(backend) for k, v in processed.items()})
        np.testing.assert_allclose(
            output["logits"].softmax(-1).cpu(),
            gold[f"{prefix}_probabilities"],
            atol=2e-6 * scale,
            rtol=2e-5 * scale,
        )
        if family == "age":
            np.testing.assert_allclose(
                output["pooled"].cpu(),
                gold[f"{prefix}_pooled"],
                atol=2e-5 * scale,
                rtol=2e-4 * scale,
            )
            np.testing.assert_allclose(
                output["age"].cpu(), gold[f"{prefix}_age"], atol=2e-6 * scale, rtol=2e-5 * scale
            )


@pytest.mark.parametrize(
    "part", ["backbone", "head", "position", "extra", "shape", "alias-collision"]
)
def test_every_weight_branch_is_mandatory(checkpoint, part):
    root, family, config = checkpoint
    path = root / "model.safetensors"
    state = load_file(path)
    key = next(k for k in state if k.startswith("wav2vec2.encoder.layers.1."))
    if part == "head":
        key = "age.out_proj.weight" if family == "age" else "projector.weight"
    elif part in {"position", "alias-collision"}:
        key = next(
            k for k in state if "pos_conv_embed" in k and k.endswith(("weight_v", "original1"))
        )
    if part == "extra":
        state["unreviewed.weight"] = torch.zeros(1)
    elif part == "shape":
        state[key] = torch.zeros(1)
    elif part == "alias-collision":
        alternative = (
            binding.native_name(key)
            if key.endswith("weight_v")
            else key.replace("parametrizations.weight.original1", "weight_v")
        )
        state[alternative] = state[key].clone()
    else:
        state.pop(key)
    save_file(state, path)
    with pytest.raises(ValueError, match="WAV2VEC_HEAD_WEIGHTS_INCOMPLETE"):
        binding.load_components(root, config, "cpu")


@pytest.mark.parametrize(
    "damage", ["nan", "infinite", "integer", "overflow", "zero-position-direction"]
)
def test_numeric_weight_rejection_precedes_execution(checkpoint, damage):
    root, family, config = checkpoint
    path = root / "model.safetensors"
    state = load_file(path)
    key = "age.out_proj.weight" if family == "age" else "classifier.weight"
    if damage == "zero-position-direction":
        key = next(
            k for k in state if "pos_conv_embed" in k and k.endswith(("weight_v", "original1"))
        )
        state[key] = torch.zeros_like(state[key])
    elif damage == "integer":
        state[key] = state[key].to(torch.int32)
    elif damage == "overflow":
        state[key] = torch.full(state[key].shape, 1e40, dtype=torch.float64)
    else:
        state[key].flatten()[0] = float("nan" if damage == "nan" else "inf")
    save_file(state, path)
    with pytest.raises(ValueError, match="WAV2VEC_HEAD_(WEIGHT_VALUES|POSITION_NORM)"):
        binding.load_components(root, config, "cpu")


@pytest.mark.parametrize(
    "field,value",
    [
        ("add_adapter", True),
        ("adapter_attn_dim", 4),
        ("use_weighted_layer_sum", True),
        ("hidden_size", True),
        ("num_attention_heads", 3),
        ("num_hidden_layers", 99),
        ("conv_kernel", [0] * 7),
        ("conv_stride", [2]),
        ("num_conv_pos_embeddings", 7),
        ("problem_type", "regression"),
        ("id2label", {"0": "invented"}),
        ("label2id", {"wrong": 0}),
        ("hidden_act", "relu"),
        ("hidden_dropout", True),
        ("layer_norm_eps", 0.0),
        ("auto_map", {"AutoModel": "unsafe.Custom"}),
        ("implementation", "repository.callback"),
    ],
)
@pytest.mark.parametrize("family", ["age", "dual"])
def test_unreviewed_configuration_is_rejected(family, field, value):
    with pytest.raises(ValueError):
        binding.validate_config(configuration(family) | {field: value})


@pytest.mark.parametrize("family", ["age", "dual"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("do_normalize", False),
        ("sampling_rate", 8000),
        ("padding_side", "left"),
        ("return_attention_mask", False),
        ("auto_map", {"AutoProcessor": "remote.Code"}),
    ],
)
def test_processor_metadata_is_closed(family, field, value):
    with pytest.raises(ValueError):
        binding.validate_processor(
            document(family, "preprocessor_config.json") | {field: value},
            configuration(family)["model_type"],
        )


def write_audio(path, values, rate=16000, channels=1):
    pcm = np.round(np.clip(values, -1, 1) * 16000).astype("<i2")
    if channels == 2:
        pcm = np.stack((pcm, pcm), axis=1)
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(channels)
        destination.setsampwidth(2)
        destination.setframerate(rate)
        destination.writeframes(pcm.tobytes())


@pytest.mark.parametrize("rate,channels", [(16000, 1), (8000, 2), (48000, 1)])
def test_public_audio_contract_uses_original_labels_and_stable_topk(checkpoint, rate, channels):
    root, family, config = checkpoint
    path = root / "input.wav"
    write_audio(path, np.sin(np.arange(rate // 5) * (2 * np.pi * 220 / rate)), rate, channels)
    events = []
    value = {"audioArtifactId": "artifact_" + "a" * 24, "topK": 2}
    result = shared_tasks.execute_task(
        root, binding.TASK, value, "cpu", path, None, lambda *a: events.append(a)
    )
    shared_tasks.validate_task_output(binding.TASK, result)
    predictions = result["predictions"]
    assert len(predictions) == 2
    assert predictions[0]["score"] >= predictions[1]["score"]
    assert all(row["label"] == config["id2label"][str(row["labelId"])] for row in predictions)
    assert sum(row["score"] for row in predictions) <= 1.000001
    assert events


@pytest.mark.parametrize("family", ["age", "dual"])
@pytest.mark.parametrize("damage", [None, "processor", "missing-processor", "remote-code"])
def test_public_metadata_inspection_routes_complete_family_without_downloads(
    tmp_path, family, damage
):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore, ModelStoreError
    from oryn_engine.runtime_packs.execution import PackRuntime

    data = {name: document(family, name) for name in ("config.json", "preprocessor_config.json")}
    names = {"config.json", "preprocessor_config.json", "model.safetensors", "modeling.py"}
    if damage == "processor":
        data["preprocessor_config.json"]["do_normalize"] = False
    elif damage == "missing-processor":
        names.remove("preprocessor_config.json")
    elif damage == "remote-code":
        data["config.json"]["auto_map"] = {"AutoModel": "unsafe.Custom"}

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            return data[filename]

    hub = Hub(
        config=data["config.json"],
        pipeline_tag=binding.TASK,
        files=[(n, 100, "b" * 64) for n in names],
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    request = {"provider": "huggingface", "repository": "someone/renamed", "revision": "main"}
    if damage == "remote-code":
        with pytest.raises(ModelStoreError):
            store.inspect(request)
    else:
        result = store.inspect(request)
        assert result["executable"] is (damage is None), result
        if damage is None:
            assert result["runtimePack"]["runnerId"] == "transformers.shared-tasks"
            assert result["taskProfile"]["limits"]["audioDurationSeconds"] == (
                30 if family == "age" else 12
            )
            assert "modeling.py" not in {f["name"] for f in result["selectedFiles"]}
    assert not hub.downloaded


@pytest.mark.parametrize("damage", ["short", "long", "missing", "nonfinite-output"])
def test_audio_and_output_bounds(checkpoint, damage):
    root, family, config = checkpoint
    model, processor, _ = binding.load_components(root, config, "cpu")
    path = root / "input.wav"
    count = (
        200
        if damage == "short"
        else 16000 * (binding.duration_limit(config) + 1)
        if damage == "long"
        else 1600
    )
    write_audio(path, np.zeros(count))
    if damage == "nonfinite-output":

        def model(**_):
            return {"logits": torch.full((1, len(config["id2label"])), float("nan"))}

    with pytest.raises(ValueError):
        binding.execute(model, processor, config, {}, "cpu", None if damage == "missing" else path)


def test_unrelated_wav2vec_models_and_regression_aliases_are_not_rewritten():
    original = document("age", "config.json")
    for task, config in [
        ("feature-extraction", original),
        (
            binding.TASK,
            original
            | {
                "finetuning_task": "wav2vec2_reg",
                "architectures": ["Wav2Vec2ForSpeechClassification"],
            },
        ),
        (binding.TASK, original | {"architectures": ["Wav2Vec2ForSequenceClassification"]}),
    ]:
        assert binding.normalized_config(config, task) == config
    assert normalized_transformers_config(original, binding.TASK)["model_type"] == binding.AGE
