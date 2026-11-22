"""Native CTranslate2 graphs, original numeric directories and strict rejection."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from oryn_engine import ct2_tasks as adapter
from oryn_engine import ct2_weights as weights

FIXTURE = Path(__file__).parent / "fixtures/ct2"


def native_fixture(root, precision="float16", silence=False, mels=80):
    from ctranslate2.converters.transformers import WhisperLoader
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    config = json.loads((FIXTURE / "config.json").read_text())
    config["alignment_heads"] = [[0, 0]]
    native = WhisperConfig(
        vocab_size=51865,
        d_model=32,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=128,
        decoder_ffn_dim=128,
        num_mel_bins=mels,
        max_source_positions=1500,
        max_target_positions=448,
        pad_token_id=50257,
        eos_token_id=50257,
        decoder_start_token_id=50258,
    )
    native._attn_implementation = "sdpa"
    torch.manual_seed(61042)
    model = WhisperForConditionalGeneration(native).eval()
    if silence:
        with torch.no_grad():
            model.model.decoder.layer_norm.weight.zero_()
            model.model.decoder.layer_norm.bias.fill_(1)
            model.proj_out.weight[50257].fill_(1)
    spec = WhisperLoader().get_model_spec(model)
    spec.register_vocabulary((FIXTURE / "vocabulary.txt").read_text().splitlines())
    spec.validate()
    spec.optimize(precision)
    spec._serialize(str(root / "model.bin"))
    (root / "config.json").write_text(json.dumps(config))
    for name in ("tokenizer.json", "vocabulary.txt"):
        shutil.copyfile(FIXTURE / name, root / name)
    # The independent native reference uses exactly the stored precision.
    if precision == "float16":
        model.half().float()
    return model, config


def test_original_numeric_directory_binds_every_tensor():
    directory = json.loads((FIXTURE / "original-directory.json").read_text())
    geometry = weights.validate_whisper(directory)
    assert len(directory["tensors"]) == 404
    assert directory["aliases"] == {"decoder/projection/weight": "decoder/embeddings/weight"}
    assert geometry == {
        "vocab_size": 51865,
        "hidden_size": 768,
        "num_mel_bins": 80,
        "encoder_layers": 12,
        "decoder_layers": 12,
        "encoder_heads": 12,
        "decoder_heads": 12,
    }
    config = json.loads((FIXTURE / "config.json").read_text())
    feature, tokenizer = adapter.components(FIXTURE, config, geometry)
    assert feature.feature_size == 80
    assert len(tokenizer.get_vocab()) == 50364  # original timestamp-only vocabulary tail


@pytest.mark.parametrize(
    "damage", ["missing", "extra", "shape", "dtype", "heads", "alibi", "version", "spec", "layers"]
)
def test_incomplete_or_changed_original_graph_rejected(damage):
    d = json.loads((FIXTURE / "original-directory.json").read_text())
    key = "decoder/layer_0/attention/linear_0/weight"
    if damage == "missing":
        del d["tensors"][key]
    elif damage == "extra":
        d["tensors"]["other/weight"] = d["tensors"][key]
    elif damage == "shape":
        d["tensors"][key]["shape"][0] -= 1
    elif damage == "dtype":
        d["tensors"][key]["dtype"] = 1
    elif damage == "heads":
        d["tensors"]["decoder/num_heads"]["controlValue"] = 7
    elif damage == "alibi":
        d["tensors"]["decoder/alibi"]["controlValue"] = 1
    elif damage == "version":
        d["binaryVersion"] = 5
    elif damage == "spec":
        d["spec"] = "TransformerSpec"
    else:
        d["tensors"][key.replace("layer_0", "layer_99")] = d["tensors"].pop(key)
    with pytest.raises(ValueError):
        weights.validate_whisper(d)


@pytest.mark.parametrize("precision", ["float16", "float32"])
@pytest.mark.parametrize("mels", [80, 128])
def test_complete_native_encode_and_decoder_logits_match_pytorch(tmp_path, precision, mels):
    import ctranslate2

    reference, config = native_fixture(tmp_path, precision, mels=mels)
    directory, geometry = weights.inspect_file(tmp_path / "model.bin", numeric=True)
    assert geometry["hidden_size"] == 32
    model, _, _ = adapter.load_components(tmp_path, config, "cpu")
    features = np.random.default_rng(61042).normal(0, 0.1, (1, mels, 3000)).astype(np.float32)
    encoded = model.encode(ctranslate2.StorageView.from_array(features))
    with torch.inference_mode():
        expected = reference.model.encoder(torch.from_numpy(features)).last_hidden_state
    np.testing.assert_allclose(np.asarray(encoded), expected, atol=2e-4, rtol=2e-4)
    prompt = [50258, 50259, 50359, 50363]
    result = model.generate(
        encoded,
        [prompt],
        max_length=8,
        beam_size=1,
        return_logits_vocab=True,
        return_scores=True,
        suppress_tokens=[],
        suppress_blank=False,
    )[0]
    with torch.inference_mode():
        output = reference(
            encoder_outputs=(expected,), decoder_input_ids=torch.tensor([prompt])
        ).logits[:, -1]
        expected_logits = output
    actual = np.asarray(result.logits)
    assert actual.shape[-1] == 51865
    np.testing.assert_allclose(
        actual.reshape(-1, 51865)[0], expected_logits[0], atol=3e-4, rtol=3e-4
    )
    assert result.sequences_ids[0][0] == int(expected_logits.argmax())
    model.unload_model()


def test_native_auto_language_and_silent_audio(tmp_path):
    _, config = native_fixture(tmp_path, silence=True)
    model, processor, _ = adapter.load_components(tmp_path, config, "cpu")
    assert model.is_multilingual
    result = adapter.transcribe(model, processor, config, np.zeros(1600, dtype=np.float32))
    assert result["text"] == "affiliated"  # EOS is suppressed only on the first step.
    assert result["language"] in {processor[1].id_to_token(i)[2:-2] for i in config["lang_ids"]}
    model.unload_model()


@pytest.mark.parametrize("damage", ["pickle", "truncated", "tail", "nonfinite", "symlink", "range"])
def test_invalid_numeric_container_rejected_before_native_loading(tmp_path, damage):
    _, _ = native_fixture(tmp_path)
    path = tmp_path / "model.bin"
    d, _ = weights.inspect_file(path)
    if damage == "pickle":
        path.write_bytes(b"\x80\x04" + b"X" * 100)
    elif damage == "truncated":
        path.write_bytes(path.read_bytes()[:-4])
    elif damage == "tail":
        with path.open("ab") as f:
            f.write(b"\0")
    elif damage == "nonfinite":
        tensor = d["tensors"]["decoder/embeddings/weight"]
        with path.open("r+b") as f:
            f.seek(tensor["dataOffset"])
            f.write(np.array([np.nan], dtype="<f2").tobytes())
    elif damage == "symlink":
        path.rename(tmp_path / "data.bin")
        path.symlink_to(tmp_path / "data.bin")
    else:
        with pytest.raises(ValueError, match="CT2_RANGE_INVALID"):
            weights.scan_directory(lambda *_: b"bad", path.stat().st_size)
        return
    with pytest.raises(ValueError):
        weights.inspect_file(path, numeric=True)


@pytest.mark.parametrize(
    "damage", ["timestamp-tail", "vocabulary", "override", "language", "alignment"]
)
def test_unbound_vocabularies_and_controls_rejected(tmp_path, damage):
    _, config = native_fixture(tmp_path)
    _, geometry = weights.inspect_file(tmp_path / "model.bin")
    if damage in {"timestamp-tail", "vocabulary"}:
        p = tmp_path / "vocabulary.txt"
        words = p.read_text().splitlines()
        words[-1 if damage == "timestamp-tail" else 1] = "changed"
        p.write_text("\n".join(words) + "\n")
    elif damage == "override":
        (tmp_path / "vocabulary.json").write_text("[]")
    elif damage == "language":
        config["lang_ids"] = [1]
    else:
        config["alignment_heads"] = [[31, 31]]
    with pytest.raises(ValueError):
        adapter.components(tmp_path, config, geometry)


@pytest.mark.parametrize("damage", [None, "missing", "control", "extra", "override", "alignment"])
@pytest.mark.parametrize("available_bytes", [1, 64 * 1024**3], ids=["low-memory", "ample-memory"])
def test_metadata_discovery_uses_numeric_directory_and_cpu_without_weights(
    tmp_path, damage, available_bytes
):
    from test_models import FakeHub, model

    from oryn_engine.models import ModelStore

    config = json.loads((FIXTURE / "config.json").read_text())
    if damage == "alignment":
        config["alignment_heads"] = [[31, 31]]

    class Hub(FakeHub):
        def read_ctranslate2_metadata(self, repo, revision, name, size):
            assert name == "model.bin"
            d = json.loads((FIXTURE / "original-directory.json").read_text())
            if damage == "control":
                d["tensors"]["decoder/alibi"]["controlValue"] = 1
            if damage == "extra":
                d["tensors"]["extra/weight"] = d["tensors"]["encoder/conv1/weight"]
            return d

    names = sorted(adapter.FILES)
    if damage == "missing":
        names.remove("model.bin")
    if damage == "override":
        names.append("vocabulary.json")
    hub = Hub(
        config=config,
        pipeline_tag="automatic-speech-recognition",
        files=[(n, 483546902 if n == "model.bin" else 100, "a" * 64) for n in names],
    )
    store = ModelStore(
        tmp_path,
        hub=hub,
        runtime_available=True,
        runtime_planning_state_provider=lambda: ("balanced", available_bytes),
    )
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda p: object() if p.pack_id == "oryn.transformers-pytorch" else None
    )
    result = store.inspect(model("independent/numeric-whisper"))
    assert result["executable"] is (damage is None and available_bytes > 1), json.dumps(
        result, indent=2
    )
    if damage is None:
        assert result["executableBackends"] == (["cpu"] if available_bytes > 1 else [])
        assert result["runtimePack"]["runnerId"] == "transformers.shared-tasks"
        assert result["taskProfile"]["features"]["wordTimestamps"] == "unsupported"
        assert result["taskProfile"]["features"]["languageDetection"] == "supported"
        assert {v["name"] for v in result["selectedFiles"]} == adapter.FILES
        if available_bytes == 1:
            assert "MODEL_MEMORY_INSUFFICIENT" in result["blockingReasons"]
    assert not hub.downloaded


@pytest.mark.parametrize("damage", [None, "status", "range", "short", "identity"])
def test_production_bounded_range_reader_and_cache(tmp_path, monkeypatch, damage):
    import httpx

    from oryn_engine.models import HuggingFaceClient, ModelStoreError

    native_fixture(tmp_path)
    data = (tmp_path / "model.bin").read_bytes()
    calls = []

    def response(request):
        first, last = map(int, request.headers["range"].removeprefix("bytes=").split("-"))
        calls.append((first, last))
        chunk = data[first : last + 1]
        if damage == "short":
            chunk = chunk[:-1]
        return httpx.Response(
            200 if damage == "status" else 206,
            content=chunk,
            headers={
                "Content-Range": "bad" if damage == "range" else f"bytes {first}-{last}/{len(data)}"
            },
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(response), **kw)
    )
    hub = HuggingFaceClient()
    args = ("org/renamed", "main" if damage == "identity" else "a" * 40, "model.bin", len(data))
    if damage:
        with pytest.raises((ValueError, ModelStoreError)):
            hub.read_ctranslate2_metadata(*args)
    else:
        d = hub.read_ctranslate2_metadata(*args)
        n = len(calls)
        assert d["transferredBytes"] == sum(last - first + 1 for first, last in calls)
        assert d["transferredBytes"] < len(data) // 10
        assert weights.validate_whisper(d)["hidden_size"] == 32
        d["tensors"].clear()
        assert hub.read_ctranslate2_metadata(*args)["tensors"]
        assert len(calls) == n


@pytest.mark.parametrize(
    "damage", ["language", "timestamps", "backend", "empty", "long", "nan", "limit"]
)
def test_public_runtime_bounds(tmp_path, damage):
    import wave

    _, config = native_fixture(tmp_path, silence=damage != "limit")
    model, processor, _ = adapter.load_components(tmp_path, config, "cpu")
    if damage in {"empty", "long", "nan", "limit"}:
        audio = np.zeros(
            0 if damage == "empty" else 480001 if damage == "long" else 1600, np.float32
        )
        if damage == "nan":
            audio[0] = np.nan
        with pytest.raises(ValueError):
            adapter.transcribe(model, processor, config, audio)
        model.unload_model()
    else:
        path = tmp_path / "input.wav"
        with wave.open(str(path), "wb") as f:
            f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            f.writeframes(bytes(3200))
        value = {
            "language": "en" if damage == "language" else "auto",
            "timestamps": damage == "timestamps",
        }
        with pytest.raises(ValueError, match="CT2_PARAMETERS"):
            adapter.execute(
                model,
                processor,
                config,
                value,
                "mps" if damage == "backend" else "cpu",
                path,
                lambda *_: None,
            )
        model.unload_model()


def test_offline_worker_uses_numeric_graph_and_owned_generation(tmp_path):
    import os
    import subprocess
    import sys
    import wave

    from oryn_engine.runtime_packs import transformers_worker
    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry

    native_fixture(tmp_path, silence=True)
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"do_sample": True, "max_new_tokens": 1, "eos_token_id": 1})
    )
    audio = tmp_path / "input.wav"
    with wave.open(str(audio), "wb") as f:
        f.setparams((2, 2, 8000, 0, "NONE", "not compressed"))
        f.writeframes(bytes(3200))
    payload = {
        "modelRoot": str(tmp_path),
        "task": "automatic-speech-recognition",
        "backend": "cpu",
        "input": {
            "audioArtifactId": "artifact_" + "b" * 24,
            "language": "auto",
            "timestamps": False,
        },
        "inputPath": str(audio),
        "outputPath": None,
    }
    expected = transformers_worker.generate(payload)
    assert expected["text"] == "affiliated"
    request = {
        "abi": "1",
        "packId": "oryn.transformers-pytorch",
        "version": builtin_registry().active_pack("oryn.transformers-pytorch").version,
        "command": "generate",
        "payload": payload,
    }
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
        result = json.loads(run.stdout.splitlines()[-1])
        assert result["ok"], (result, run.stderr)
        assert result["result"] == expected


@pytest.mark.parametrize("mels", [80, 128])
@pytest.mark.parametrize("length", [1, 1601, 480000])
def test_frontend_matches_independent_numpy_stft(mels, length):
    config = json.loads((FIXTURE / "config.json").read_text())
    d = json.loads((FIXTURE / "original-directory.json").read_text())
    geometry = weights.validate_whisper(d) | {"num_mel_bins": mels}
    feature, _ = adapter.components(FIXTURE, config, geometry)
    audio = (np.sin(np.arange(length) * 0.13) * 0.2).astype(np.float32)
    actual = feature(
        audio,
        sampling_rate=16000,
        return_tensors="np",
        padding="max_length",
        max_length=480000,
        truncation=False,
    ).input_features[0]
    padded = np.pad(np.pad(audio, (0, 480000 - length)), (200, 200), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 400)[::160][:-1]
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)).astype(np.float32)
    power = np.abs(np.fft.rfft(windows * window, axis=-1)) ** 2
    mel = power @ feature.mel_filters
    log = np.log10(np.maximum(mel, 1e-10))
    expected = (np.maximum(log, log.max() - 8) + 4) / 4
    # Float32 Torch FFT and NumPy FFT differ at the log floor for pure tones.
    # The 1e-4 log-mel bound is far below frontend scale/crop errors.
    np.testing.assert_allclose(actual, expected.T, atol=1e-4, rtol=0)


@pytest.mark.parametrize(
    "change",
    [
        {"trust_remote_code": True},
        {"lang_ids": [True]},
        {"lang_ids": [50259, 50259]},
        {"lang_ids": []},
        {"suppress_ids": [51866]},
        {"suppress_ids_begin": [-1]},
        {"alignment_heads": [[0, True]]},
        {"alignment_heads": [[32, 0]]},
    ],
)
def test_closed_configuration_rejects_unreviewed_changes(change):
    config = json.loads((FIXTURE / "config.json").read_text()) | change
    with pytest.raises(ValueError):
        adapter.validate_config(config)
