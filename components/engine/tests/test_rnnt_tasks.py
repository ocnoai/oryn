"""Native transducer graph, greedy frame advancement and metadata-only routing."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from oryn_engine import rnnt_tasks as adapter
from oryn_engine.runtime_packs.builtin_catalog import builtin_registry

FIXTURES = Path(__file__).parent / "fixtures/rnnt"


def original():
    return json.loads((FIXTURES / "config.json").read_text())


@pytest.mark.parametrize(
    "section,change",
    [
        (None, {"architectures": ["RepositoryModel"]}),
        (None, {"quantization": {"bits": 4}}),
        (None, {"vocab_size": 0}),
        (None, {"blank_token_id": 0}),
        (None, {"num_prompts": 127}),
        (None, {"default_prompt_id": 0}),
        (None, {"max_symbols_per_step": 100}),
        (None, {"durations": [0, 1]}),
        (None, {"hidden_act": "silu"}),
        ("encoder_config", {"num_mel_bins": 80}),
        ("encoder_config", {"hidden_size": 31}),
        ("encoder_config", {"num_attention_heads": 7}),
        ("encoder_config", {"subsampling_factor": 4}),
        ("encoder_config", {"subsampling_conv_stride": 1}),
        ("encoder_config", {"default_num_lookahead_tokens": 13}),
        ("encoder_config", {"sliding_window": 99999}),
        ("encoder_config", {"scale_input": True}),
        ("encoder_config", {"attention_bias": True}),
        ("encoder_config", {"hidden_act": "relu"}),
    ],
)
def test_unbound_graphs_are_rejected(section, change):
    c = original()
    (c if section is None else c[section]).update(change)
    with pytest.raises(ValueError):
        adapter.validate_config(c)


@pytest.mark.parametrize(
    "name,change",
    [
        ("processor_config.json", {"num_prompts": 256}),
        ("processor_config.json", {"default_num_lookahead_tokens": 0}),
        ("processor_config.json", {"prompt_dictionary": {"auto": 0}}),
        ("processor_config.json", {"feature_extractor": {}}),
        ("processor_config.json", {"unsupported_active_setting": True}),
        ("tokenizer_config.json", {"tokenizer_class": "CustomTokenizer"}),
        ("tokenizer_config.json", {"pad_token": "unknown"}),
        ("tokenizer_config.json", {"extra_special_tokens": ["x" * 100]}),
    ],
)
def test_processor_and_tokenizer_changes_rejected(name, change):
    m = json.loads((FIXTURES / name).read_text())
    m.update(change)
    with pytest.raises(ValueError):
        adapter.validate_metadata(name, m, original())


def test_complete_original_native_graph_without_payload():
    from accelerate import init_empty_weights

    config = original()
    adapter.validate_config(config)
    with init_empty_weights(include_buffers=False):
        model = adapter.native_model(config)
    expected = json.loads((FIXTURES / "original-shapes.json").read_text())
    assert {k: list(v.shape) for k, v in model.state_dict().items()} == {
        k: v["shape"] for k, v in expected.items()
    }
    assert len(expected) == 655


@pytest.mark.parametrize("damage", [None, "missing", "processor", "tokenizer"])
def test_discovery_profile_and_required_metadata(tmp_path, damage):
    from test_models import FakeHub, model

    from oryn_engine.models import ModelStore

    config = original()

    class Hub(FakeHub):
        def read_json(self, repo, revision, name):
            v = config if name == "config.json" else json.loads((FIXTURES / name).read_text())
            if damage == "processor" and name == "processor_config.json":
                v["default_num_lookahead_tokens"] = 0
            if damage == "tokenizer" and name == "tokenizer_config.json":
                v["pad_token"] = "x"
            return v

    names = ["config.json", "model.safetensors", "tokenizer.json", *adapter.METADATA_NAMES]
    if damage == "missing":
        names.remove("processor_config.json")
    hub = Hub(
        config=config,
        pipeline_tag="automatic-speech-recognition",
        files=[(n, 100, "a" * 64) for n in names],
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda p: object() if p.pack_id == "oryn.transformers-pytorch" else None
    )
    result = store.inspect(model("independent/native-transducer"))
    if damage:
        assert not result["executable"], result
    else:
        assert result["executable"], result
        profile = result["taskProfile"]
        assert profile["contract"]["id"] == "transcription-v1"
        assert profile["features"]["wordTimestamps"] == "unsupported"
        assert profile["limits"]["audioDurationSeconds"] == 30
    assert not hub.downloaded


def native_fixture(root, *, terminal=False):
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, decoders, models

    config = original()
    config.update(
        vocab_size=32,
        blank_token_id=31,
        pad_token_id=0,
        decoder_hidden_size=32,
        prompt_intermediate_size=64,
    )
    config["encoder_config"].update(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        subsampling_conv_channels=8,
        max_position_embeddings=512,
    )
    words = ["<unk>" if terminal else "<pad>", "a", "b", "c", "▁hello", "▁world", "x"]
    while len(words) < 31:
        words.append("unused_" + str(len(words)))
    words += ["<pad>", "<blank>"] if terminal else ["<blank>"]
    t = Tokenizer(models.BPE(vocab={w: i for i, w in enumerate(words)}, merges=[]))
    t.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    t.add_special_tokens(["<pad>", "<blank>"] + (["<unk>"] if terminal else []))
    t.save(str(root / "tokenizer.json"))
    tm = json.loads((FIXTURES / "tokenizer_config.json").read_text())
    tm["extra_special_tokens"] = []
    # Aligned fixtures also need an existing unknown token; reuse a reserved row.
    if not terminal:
        raw = json.loads((root / "tokenizer.json").read_text())
        raw["model"]["vocab"]["<unk>"] = raw["model"]["vocab"].pop("unused_7")
        (root / "tokenizer.json").write_text(json.dumps(raw))
    (root / "tokenizer_config.json").write_text(json.dumps(tm))
    (root / "processor_config.json").write_bytes((FIXTURES / "processor_config.json").read_bytes())
    (root / "config.json").write_text(json.dumps(config))
    torch.manual_seed(61631)
    model = adapter.native_model(config).eval()
    weights = model.state_dict()
    save_file(weights, str(root / "model.safetensors"))
    return model, config, weights


def manual_rnnt(model, inputs, config):
    with torch.inference_mode():
        features = model.get_audio_features(**inputs).pooler_output
        history = [config["blank_token_id"]]
        result = []
        for frame in range(features.shape[1]):
            for _ in range(config["max_symbols_per_step"]):
                decoded = model.decoder(torch.tensor([history], device=features.device))[:, -1:]
                logits = model.joint(decoded, features[:, frame : frame + 1])
                token = int(logits[0, -1].argmax())
                result.append(token)
                if token == config["blank_token_id"]:
                    break
                history.append(token)
        return features, result


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("mode", ["seeded", "blank", "repeat"])
@pytest.mark.parametrize("terminal", [False, True])
def test_full_graph_native_decode_matches_independent_frame_loop(tmp_path, backend, mode, terminal):
    from safetensors.torch import save_file

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS device unavailable")
    reference, config, weights = native_fixture(tmp_path, terminal=terminal)
    if mode != "seeded":
        with torch.no_grad():
            reference.joint.head.weight.zero_()
            reference.joint.head.bias.fill_(-10)
            reference.joint.head.bias[config["blank_token_id"] if mode == "blank" else 1] = 10
        save_file(reference.state_dict(), str(tmp_path / "model.safetensors"))
    reference.to(backend)
    loaded, processor, _ = adapter.load_components(tmp_path, config, backend)
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in reference.state_dict().items())
    audio = np.sin(np.arange(1600, dtype=np.float32) * 0.08)
    prepared = adapter.prepare_inputs(processor[0], audio, config, backend)
    expected_features, tokens = manual_rnnt(reference, prepared, config)
    with torch.inference_mode():
        actual_features = loaded.get_audio_features(**prepared).pooler_output
    torch.testing.assert_close(actual_features, expected_features, rtol=2e-5, atol=2e-5)
    expected = (
        processor[1]
        .decode(
            [i for i in tokens if i not in {config["pad_token_id"], config["blank_token_id"]}],
            group_tokens=False,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        .strip()
    )
    result = adapter.transcribe(loaded, processor, config, audio, backend)
    assert result == {"text": expected, "language": None}
    if mode == "repeat":
        assert result["text"] == "a" * (10 * expected_features.shape[1])
    if mode == "blank":
        assert result["text"] == ""


@pytest.mark.parametrize("damage", ["missing", "extra", "shape", "nan", "integer", "duplicate"])
def test_incomplete_or_non_numeric_graph_rejected(tmp_path, damage):
    from safetensors.torch import save_file

    _, config, weights = native_fixture(tmp_path)
    key = "joint.head.weight"
    if damage == "missing":
        del weights[key]
    elif damage == "extra":
        weights["unknown.weight"] = torch.ones((2, 2))
    elif damage == "shape":
        weights[key] = weights[key][:-1]
    elif damage == "nan":
        weights[key] = torch.full_like(weights[key], float("nan"))
    elif damage == "integer":
        weights[key] = weights[key].to(torch.int32)
    elif damage == "duplicate":
        save_file({key: weights[key]}, str(tmp_path / "extra.safetensors"))
    save_file(weights, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError):
        adapter.load_components(tmp_path, config, "cpu")


@pytest.mark.parametrize(
    "damage", ["vocabulary", "blank", "pad", "nonfinite", "too-short", "too-long"]
)
def test_tokenizer_and_audio_bounds(tmp_path, damage):
    _, config, _ = native_fixture(tmp_path, terminal=True)
    if damage in {"vocabulary", "blank", "pad"}:
        p = tmp_path / "tokenizer.json"
        raw = json.loads(p.read_text())
        if damage == "vocabulary":
            raw["model"]["vocab"]["unexpected"] = len(raw["model"]["vocab"])
        else:
            token = "<blank>" if damage == "blank" else "<pad>"
            raw["model"]["vocab"][token], raw["model"]["vocab"]["a"] = (
                raw["model"]["vocab"]["a"],
                raw["model"]["vocab"][token],
            )
            raw["added_tokens"] = [x for x in raw["added_tokens"] if x["content"] != token]
        p.write_text(json.dumps(raw))
        with pytest.raises(ValueError):
            adapter.components(tmp_path, config)
    else:
        feature, _ = adapter.components(tmp_path, config)
        audio = (
            np.full(1000, np.nan)
            if damage == "nonfinite"
            else np.zeros(100 if damage == "too-short" else 480001, dtype=np.float32)
        )
        with pytest.raises(ValueError):
            adapter.prepare_inputs(feature, audio, config, "cpu")


@pytest.mark.parametrize("length", [512, 1599, 1600, 1601, 480000])
def test_native_preprocessing_matches_independent_stft(tmp_path, length):
    import librosa

    _, config, _ = native_fixture(tmp_path)
    feature, _ = adapter.components(tmp_path, config)
    audio = np.sin(np.arange(length, dtype=np.float32) * 0.08)
    actual = adapter.prepare_inputs(feature, audio, config, "cpu")
    emphasized = np.concatenate((audio[:1], audio[1:] - np.float32(0.97) * audio[:-1]))
    padded = np.pad(emphasized, (256, 256))
    windows = np.lib.stride_tricks.sliding_window_view(padded, 512)[::160]
    window = np.pad(np.hanning(400), (56, 56))
    power = np.abs(np.fft.rfft(windows * window, axis=-1)) ** 2
    filters = librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, norm="slaney")
    expected = np.log(power @ filters.T + 2**-24).astype(np.float32)
    mask = np.arange(expected.shape[0]) < length // 160
    expected[~mask] = 0
    np.testing.assert_allclose(actual["input_features"][0], expected, rtol=2e-5, atol=3e-5)
    np.testing.assert_array_equal(actual["attention_mask"][0], mask)
    assert actual["prompt_ids"].tolist() == [101]


@pytest.mark.parametrize(
    "damage",
    ["extra-cpu", "two-extra", "missing", "unfinished", "wrong-step", "dtype", "start", "token"],
)
def test_duration_reconciliation_does_not_accept_partial_decoding(tmp_path, damage):
    _, config, _ = native_fixture(tmp_path)
    _, tokenizer = adapter.components(tmp_path, config)
    ids, durations, backend = [31, 31, 31], [0, 1, 1], "mps"
    if damage == "extra-cpu":
        durations += [1]
        backend = "cpu"
    elif damage == "two-extra":
        durations += [1, 1]
    elif damage == "missing":
        durations.pop()
    elif damage == "unfinished":
        ids[-1], durations[-1] = 1, 0
    elif damage == "wrong-step":
        durations[0], durations[1] = 1, 0
    elif damage == "start":
        ids[0] = 1
    elif damage == "token":
        ids[1] = 32
    output = SimpleNamespace(sequences=torch.tensor([ids]), durations=torch.tensor([durations]))
    if damage == "dtype":
        output.durations = output.durations.float()
    with pytest.raises(ValueError, match="RNNT_DECODING_INCOMPLETE"):
        adapter.decode_output(output, tokenizer, config, 2, backend)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_original_style_tokenizer_repeated_worker_requests_and_owned_generation(tmp_path, backend):
    import os
    import subprocess
    import sys
    import wave

    from safetensors.torch import save_file

    from oryn_engine.runtime_packs import transformers_worker

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS device unavailable")
    model, config, _ = native_fixture(tmp_path, terminal=True)
    with torch.no_grad():
        model.joint.head.weight.zero_()
        model.joint.head.bias.fill_(-10)
        model.joint.head.bias[1] = 10
    save_file(model.state_dict(), str(tmp_path / "model.safetensors"))
    # Repository defaults must not impose a premature EOS or sampling.
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": 1, "do_sample": True, "max_new_tokens": 1})
    )
    audio = tmp_path / "input.wav"
    with wave.open(str(audio), "wb") as f:
        f.setparams((2, 2, 32000, 0, "NONE", "not compressed"))
        values = (np.sin(np.arange(3200) * 0.04) * 16000).astype("<i2")
        f.writeframes(np.repeat(values[:, None], 2, axis=1).tobytes())
    payload = {
        "modelRoot": str(tmp_path),
        "task": "automatic-speech-recognition",
        "backend": backend,
        "input": {
            "audioArtifactId": "artifact_" + "a" * 24,
            "language": "auto",
            "timestamps": False,
        },
        "inputPath": str(audio),
        "outputPath": None,
    }
    assert transformers_worker.generate(payload) == {"text": "a" * 30, "language": None}
    envelope = {
        "abi": "1",
        "packId": "oryn.transformers-pytorch",
        "version": builtin_registry().active_pack("oryn.transformers-pytorch").version,
    }
    for command in ("probe", "generate", "generate"):
        run = subprocess.run(
            [sys.executable, "-m", "oryn_engine.runtime_packs.transformers_worker"],
            input=json.dumps(
                envelope | {"command": command, "payload": payload if command == "generate" else {}}
            )
            + "\n",
            text=True,
            capture_output=True,
            timeout=60,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        assert run.returncode == 0, run.stderr
        result = json.loads(run.stdout.splitlines()[-1])
        assert result["ok"], (result, run.stderr)
        if command == "generate":
            assert result["result"] == {"text": "a" * 30, "language": None}
        else:
            assert result["result"]["networkPolicy"] == "deny"


def test_nonfinite_native_logits_fail_and_remove_hook(tmp_path):
    model, config, _ = native_fixture(tmp_path)
    processor = adapter.components(tmp_path, config)
    with torch.no_grad():
        model.joint.head.bias.fill_(float("inf"))
    with pytest.raises(ValueError, match="RNNT_LOGITS_NONFINITE"):
        adapter.transcribe(model, processor, config, np.zeros(1600, dtype=np.float32), "cpu")
    assert not model._forward_hooks


@pytest.mark.parametrize("value", [{"language": "en"}, {"timestamps": True}])
def test_unbound_asr_options_rejected(value):
    with pytest.raises(ValueError, match="RNNT_PARAMETERS"):
        adapter.execute(None, None, original(), value, "cpu", None, lambda *_: None)
