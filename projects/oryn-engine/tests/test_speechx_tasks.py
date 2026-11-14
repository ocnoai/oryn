"""Original metadata and complete small native multi-head audio graphs."""

import copy
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from oryn_engine import shared_tasks
from oryn_engine import speechx_tasks as adapter
from oryn_engine.runtime_packs.builtin_catalog import builtin_registry

FIXTURES = Path(__file__).parent / "fixtures/speechx"
RAW = json.loads((FIXTURES / "config.json").read_text())
PROCESSOR = json.loads((FIXTURES / "preprocessor_config.json").read_text())


def fixture(root, precision="float32"):
    from safetensors.torch import save_file
    from transformers import WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    root.mkdir(exist_ok=True, parents=True)
    config = copy.deepcopy(adapter.normalized_config(RAW, adapter.TASK))
    config.update(
        d_model=32,
        dtype=precision,
        num_langs=3,
        id2lang={"0": "en", "1": "de", "2": "fr"},
        lang2id={"en": 0, "de": 1, "fr": 2},
    )
    config["encoder_config"].update(
        d_model=32,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        max_source_positions=50,
    )
    metadata = PROCESSOR | dict(chunk_length=1, n_samples=16000, nb_max_frames=100)
    (root / "config.json").write_text(json.dumps(config))
    (root / "preprocessor_config.json").write_text(json.dumps(metadata))
    # Independent native reference: no adapter constructor or forward is used.
    native = WhisperConfig(
        d_model=32,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        max_source_positions=50,
        num_mel_bins=80,
        activation_function="gelu",
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        encoder_layerdrop=0.0,
        scale_embedding=False,
    )
    native._attn_implementation = "eager"
    torch.manual_seed(6171)
    reference = torch.nn.ModuleDict(
        {
            "encoder": WhisperEncoder(native),
            "gender_head": torch.nn.Linear(32, 2),
            "lang_head": torch.nn.Linear(32, 3),
        }
    ).eval()
    weights = {
        k: v.detach().to(getattr(torch, precision)).clone()
        for k, v in reference.state_dict().items()
    }
    save_file(weights, str(root / "model.safetensors"))
    reference.load_state_dict({k: v.float() for k, v in weights.items()})
    return config, reference, weights


def audio(path, count=3200, rate=16000, channels=1):
    signal = (0.25 * np.sin(np.arange(count) * 2 * np.pi * 220 / rate)).astype(np.float32)
    pcm = np.round(signal * 32767).astype("<i2")
    if channels == 2:
        pcm = np.stack([pcm, pcm], axis=1)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.tobytes())


def test_original_complete_encoder_and_both_heads_match_without_weights():
    config = adapter.normalized_config(RAW, adapter.TASK)
    header = json.loads((FIXTURES / "model.safetensors.header.json").read_text())
    shapes = {k: tuple(v["shape"]) for k, v in header.items() if k != "__metadata__"}
    assert len(shapes) == 101
    model = adapter.validate_weight_shapes(config, shapes)
    assert model.encoder.layers[0].self_attn.num_heads == 8
    assert model.gender_head.out_features == 2 and model.lang_head.out_features == 109
    adapter.validate_processor(PROCESSOR, config)


@pytest.mark.parametrize("length", [320, 1601, 480000])
def test_original_whisper_frontend_matches_independent_numpy(length):
    from transformers import WhisperFeatureExtractor

    processor = WhisperFeatureExtractor(
        feature_size=80, sampling_rate=16000, hop_length=160, n_fft=400, chunk_length=30, dither=0.0
    )
    samples = (0.2 * np.sin(np.arange(length) * 0.13)).astype(np.float32)
    actual = processor(
        samples, sampling_rate=16000, return_tensors="np", truncation=False
    ).input_features[0]
    padded = np.pad(np.pad(samples, (0, 480000 - length)), (200, 200), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, 400)[::160][:-1]
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)).astype(np.float32)
    power = np.abs(np.fft.rfft(frames * window, axis=-1)) ** 2
    mel = power @ processor.mel_filters
    log = np.log10(np.maximum(mel, 1e-10))
    expected = (np.maximum(log, log.max() - 8) + 4) / 4
    assert actual.shape == (80, 3000)
    np.testing.assert_allclose(actual, expected.T, atol=1e-4, rtol=0)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("precision", ["float16", "float32"])
def test_complete_both_head_logits_and_independent_probabilities(tmp_path, backend, precision):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    config, reference, _ = fixture(tmp_path, precision)
    model, processor, _ = adapter.load_components(tmp_path, config, backend)
    path = tmp_path / "clip.wav"
    audio(path)
    from oryn_engine.vad_tasks import read_audio

    samples, _ = read_audio(path)
    features = processor(samples, sampling_rate=16000, return_tensors="pt").input_features
    reference.to(backend)
    with torch.inference_mode():
        pooled = reference["encoder"](features.to(backend)).last_hidden_state.mean(dim=1)
        output = model(features.to(backend))
        for head, layer in [("language", "lang_head"), ("gender", "gender_head")]:
            logits = reference[layer](pooled)
            torch.testing.assert_close(output[head], logits, atol=2e-5, rtol=2e-5)
            raw = logits[0].float().cpu().numpy().astype(np.float64)
            scores = np.exp(raw - raw.max())
            scores /= scores.sum()
            result = adapter.execute(
                model, processor, config, {"classificationHead": head, "topK": 20}, backend, path
            )
            rows = result["predictions"]
            assert len(rows) == len(scores)
            for row in rows:
                assert row["score"] == pytest.approx(scores[row["labelId"]], abs=2e-7)
            assert sum(row["score"] for row in rows) == pytest.approx(1, abs=2e-7)


@pytest.mark.parametrize(
    "change",
    [
        {"auto_map": {"AutoModel": "unreviewed.Code"}},
        {"base_model": "other/whisper"},
        {"d_model": 384},
        {"encoder_config": adapter.BASE_ENCODER},
        {"num_langs": True},
        {"num_genders": 3},
        {"lang2id": {"en": True}},
        {"id2lang": {"0": "en"}},
        {"dtype": "int8"},
        {"dropout": 1},
        {"quantization_config": {"bits": 4}},
        {"attention_implementation": "custom"},
        {"trust_remote_code": True},
    ],
)
def test_unreviewed_original_graphs_and_labels_rejected(change):
    with pytest.raises(ValueError):
        adapter.normalized_config(RAW | change, adapter.TASK)


@pytest.mark.parametrize(
    "key,value",
    [
        ("d_model", 31),
        ("encoder_layers", True),
        ("encoder_attention_heads", 3),
        ("num_mel_bins", 128),
        ("max_source_positions", 51),
        ("rotary", True),
    ],
)
def test_closed_native_geometry(tmp_path, key, value):
    config, _, _ = fixture(tmp_path)
    config["encoder_config"][key] = value
    with pytest.raises(ValueError):
        adapter.validate_config(config)


@pytest.mark.parametrize(
    "change",
    [
        {"dither": 0.1},
        {"sampling_rate": 8000},
        {"padding_side": "left"},
        {"chunk_length": True},
        {"nb_max_frames": 2999},
        {"do_normalize": True},
    ],
)
def test_original_preprocessor_is_closed(change):
    with pytest.raises(ValueError):
        adapter.validate_processor(PROCESSOR | change, adapter.normalized_config(RAW, adapter.TASK))


@pytest.mark.parametrize(
    "damage", ["missing-gender", "extra", "shape", "integer", "nan", "symlink", "duplicate"]
)
def test_complete_numeric_weights_required(tmp_path, damage):
    from safetensors.torch import save_file

    config, _, state = fixture(tmp_path)
    key = "gender_head.weight"
    if damage == "missing-gender":
        del state[key]
    elif damage == "extra":
        state["decoder.weight"] = torch.ones(2)
    elif damage == "shape":
        state[key] = state[key][:1]
    elif damage == "integer":
        state[key] = state[key].to(torch.int32)
    elif damage == "nan":
        state[key][0, 0] = float("nan")
    save_file(state, str(tmp_path / "model.safetensors"))
    if damage == "symlink":
        (tmp_path / "model.safetensors").rename(tmp_path / "outside.bin")
        (tmp_path / "model.safetensors").symlink_to(tmp_path / "outside.bin")
    elif damage == "duplicate":
        save_file({key: state[key]}, str(tmp_path / "other.safetensors"))
    with pytest.raises(ValueError):
        adapter.load_components(tmp_path, config, "cpu")


@pytest.mark.parametrize("rate,channels", [(16000, 1), (8000, 2), (48000, 1)])
def test_shared_contract_head_selection_and_audio_resampling(tmp_path, rate, channels):
    config, _, _ = fixture(tmp_path)
    path = tmp_path / "clip.wav"
    audio(path, rate // 5, rate, channels)
    for head in [None, "language", "gender"]:
        value = {"audioArtifactId": "artifact_" + "a" * 24, "topK": 20}
        if head:
            value["classificationHead"] = head
        result = shared_tasks.execute_task(
            tmp_path, adapter.TASK, value, "cpu", path, None, lambda *_: None
        )
        labels = config["id2gender" if head == "gender" else "id2lang"]
        assert len(result["predictions"]) == len(labels)
        assert all(row["label"] == labels[str(row["labelId"])] for row in result["predictions"])


@pytest.mark.parametrize("damage", [None, "processor", "missing", "remote-code"])
def test_original_metadata_discovery_is_offline_and_requires_full_processor(tmp_path, damage):
    from test_models import FakeHub, model

    from oryn_engine.models import ModelStore, ModelStoreError

    data = {"config.json": copy.deepcopy(RAW), "preprocessor_config.json": copy.deepcopy(PROCESSOR)}
    names = {"config.json", "preprocessor_config.json", "model.safetensors", "modeling_speechx.py"}
    if damage == "processor":
        data["preprocessor_config.json"]["chunk_length"] = 15
    elif damage == "missing":
        names.remove("preprocessor_config.json")
    elif damage == "remote-code":
        data["config.json"]["auto_map"]["AutoModel"] = "unsafe.Code"

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            return data[filename]

    hub = Hub(
        config=data["config.json"],
        pipeline_tag=adapter.TASK,
        files=[(n, 100, "b" * 64) for n in names],
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda p: None if p.provider else object()
    )
    if damage == "remote-code":
        with pytest.raises(ModelStoreError):
            store.inspect(model("independent/renamed"))
    else:
        result = store.inspect(model("independent/renamed"))
        assert result["executable"] is (damage is None), result
        if damage is None:
            assert result["runtimePack"]["runnerId"] == "transformers.shared-tasks"
            assert (
                result["taskProfile"]["parameters"]["classificationHead"]["default"] == "language"
            )
            assert result["taskProfile"]["limits"]["audioDurationSeconds"] == 30
            assert "modeling_speechx.py" not in {f["name"] for f in result["selectedFiles"]}
    assert not hub.downloaded


@pytest.mark.parametrize(
    "damage", ["short", "long", "missing", "gender-nan", "language-shape", "head"]
)
def test_audio_bounds_and_both_output_heads_are_checked(tmp_path, damage):
    config, _, _ = fixture(tmp_path)
    model, processor, _ = adapter.load_components(tmp_path, config, "cpu")
    path = tmp_path / "clip.wav"
    audio(path, 319 if damage == "short" else 16001 if damage == "long" else 3200)
    if damage in {"gender-nan", "language-shape"}:

        def model(_):
            return {
                "gender": torch.full((1, 2), float("nan") if damage == "gender-nan" else 0),
                "language": torch.zeros(1, 2 if damage == "language-shape" else 3),
            }

    with torch.inference_mode(), pytest.raises(ValueError):
        adapter.execute(
            model,
            processor,
            config,
            {"classificationHead": "other" if damage == "head" else "language"},
            "cpu",
            None if damage == "missing" else path,
        )


def test_other_audio_models_cannot_silently_ignore_head_selection(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "wav2vec2"}))
    with pytest.raises(ValueError, match="AUDIO_CLASSIFICATION_HEAD_UNSUPPORTED"):
        shared_tasks.execute_task(
            tmp_path,
            adapter.TASK,
            {"audioArtifactId": "artifact_" + "a" * 24, "classificationHead": "gender"},
            "cpu",
            None,
            None,
            lambda *_: None,
        )


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_offline_worker_preserves_each_head(tmp_path, backend):
    import os
    import subprocess
    import sys

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    fixture(tmp_path)
    path = tmp_path / "clip.wav"
    audio(path)
    for head in ["language", "gender"]:
        value = {"audioArtifactId": "artifact_" + "a" * 24, "classificationHead": head}
        expected = shared_tasks.execute_task(
            tmp_path, adapter.TASK, value, backend, path, None, lambda *_: None
        )
        request = dict(
            abi="1",
            packId="oryn.transformers-pytorch",
            version=builtin_registry().active_pack("oryn.transformers-pytorch").version,
            command="generate",
            payload=dict(
                modelRoot=str(tmp_path),
                task=adapter.TASK,
                backend=backend,
                input=value,
                inputPath=str(path),
                outputPath=None,
            ),
        )
        for _ in range(2):
            run = subprocess.run(
                [sys.executable, "-m", "oryn_engine.runtime_packs.transformers_worker"],
                input=json.dumps(request) + "\n",
                text=True,
                capture_output=True,
                timeout=60,
                env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            )
            assert run.returncode == 0, run.stderr
            result = [json.loads(line) for line in run.stdout.splitlines()][-1]
            assert result["ok"], (result, run.stderr)
            assert result["result"] == expected
