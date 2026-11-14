"""Complete ternary transducer graphs and independent numeric decoding references."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from oryn_engine import parakeet_tasks as adapter

FIXTURE = Path(__file__).parent / "fixtures/parakeet"


def original():
    return json.loads((FIXTURE / "config.json").read_text())


def test_original_993_headers_match_complete_native_graph():
    from accelerate import init_empty_weights

    config = original()
    adapter.validate_config(config)
    adapter.validate_metadata(
        "ternary.json", json.loads((FIXTURE / "ternary.json").read_text()), config
    )
    with init_empty_weights(include_buffers=False):
        model = adapter.native_model(config)
    expected = adapter.packed_shapes(model.state_dict(), config)
    headers = json.loads((FIXTURE / "model.safetensors.header.json").read_text())
    assert expected == {k: tuple(v["shape"]) for k, v in headers.items()}
    assert len(expected) == 993
    assert len(adapter.modules(config)) == 264
    assert adapter.estimated_memory(config, 177774490) > 6 * 1024**3


@pytest.mark.parametrize(
    "section,change",
    [
        (None, {"model_type": "parakeet_rnnt"}),
        (None, {"architectures": ["CustomModel"]}),
        (None, {"durations": [1, 2, 3]}),
        (None, {"ternary_group_size": 64}),
        (None, {"ternary_modules": []}),
        (None, {"blank_token_id": 0}),
        (None, {"max_symbols_per_step": 20}),
        (None, {"quantization_config": {"bits": 2}}),
        ("encoder_config", {"hidden_size": 129}),
        ("encoder_config", {"intermediate_size": 300}),
        ("encoder_config", {"num_hidden_layers": 1000}),
        ("encoder_config", {"subsampling_factor": 4}),
        ("encoder_config", {"num_mel_bins": 80}),
        ("encoder_config", {"scale_input": True}),
        ("encoder_config", {"attention_bias": True}),
        ("encoder_config", {"sliding_window": 57}),
    ],
)
def test_unsupported_configuration_rejected(section, change):
    c = original()
    (c if section is None else c[section]).update(change)
    with pytest.raises(ValueError):
        adapter.validate_config(c)


@pytest.mark.parametrize(
    "damage",
    [
        "packing",
        "missing",
        "duplicate",
        "dimensions",
        "conv",
        "bias",
        "scale",
        "source",
        "vad",
        "format",
    ],
)
def test_closed_ternary_metadata_rejects_changes(damage):
    c = original()
    m = json.loads((FIXTURE / "ternary.json").read_text())
    if damage == "packing":
        m["packing"]["base"] = 4
    elif damage == "missing":
        m["quantized_modules"].pop()
    elif damage == "duplicate":
        m["quantized_modules"][1] = m["quantized_modules"][0]
    elif damage == "dimensions":
        m["quantized_modules"][0]["in_features"] += 1
    elif damage == "conv":
        m["quantized_modules"][0]["as_conv1d"] = True
    elif damage == "bias":
        m["quantized_modules"][0]["has_bias"] = True
    elif damage == "scale":
        m["quantized_modules"][0]["zero_fraction"] = float("nan")
    elif damage == "source":
        m["source"]["teacher"] = "other"
    elif damage == "vad":
        m["vad_head"]["tensors"].pop()
    else:
        m["format"] = "thrush-ternary-v3"
    with pytest.raises(ValueError):
        adapter.validate_metadata("ternary.json", m, c)


def pack_digits(digits):
    padded = np.pad(digits, ((0, 0), (0, -digits.shape[1] % 5)))
    return torch.from_numpy(
        (padded.reshape(digits.shape[0], -1, 5) @ np.array([1, 3, 9, 27, 81])).astype(np.uint8)
    )


@pytest.mark.parametrize("ins", [128, 256, 640])
@pytest.mark.parametrize("conv", [False, True])
def test_ternary_unpack_matches_scalar_base_three(ins, conv):
    digits = np.random.default_rng(61027).integers(0, 3, (7, ins), dtype=np.int16)
    scales = torch.linspace(0, 0.2, 7 * (ins // 128)).reshape(7, ins // 128).half()
    actual = (
        adapter.unpack(pack_digits(digits), scales, ins, conv).squeeze(-1)
        if conv
        else adapter.unpack(pack_digits(digits), scales, ins, conv)
    )
    expected = np.array(
        [
            [float(scales[row, col // 128]) * (int(digits[row, col]) - 1) for col in range(ins)]
            for row in range(7)
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("damage", ["overflow", "padding", "dtype", "shape", "negative", "nan"])
def test_corrupt_ternary_numeric_values_rejected(damage):
    q = pack_digits(np.ones((2, 128), dtype=np.int16))
    scales = torch.ones((2, 1), dtype=torch.float16)
    if damage == "overflow":
        q[0, 0] = 243
    elif damage == "padding":
        q[0, -1] = 27
    elif damage == "dtype":
        q = q.long()
    elif damage == "shape":
        scales = torch.ones((2, 2))
    elif damage == "negative":
        scales[0, 0] = -1
    else:
        scales[0, 0] = float("nan")
    with pytest.raises(ValueError):
        adapter.unpack(q, scales, 128, False)


def native_fixture(root):
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, decoders, models

    c = original()
    c.update(vocab_size=32, blank_token_id=31, decoder_hidden_size=32)
    c["encoder_config"].update(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        subsampling_conv_channels=8,
        max_position_embeddings=512,
    )
    c["ternary_modules"] = list(adapter.modules(c))
    metadata = json.loads((FIXTURE / "ternary.json").read_text())
    metadata["n_quantized_params"] = sum(o * i for o, i, _ in adapter.modules(c).values())
    metadata["vad_head"].update(
        d_model=128, params=sum(np.prod(s).item() for s in adapter.vad_shapes(c).values())
    )
    metadata["quantized_modules"] = [
        dict(
            name=k,
            out_features=o,
            in_features=i,
            group_size=128,
            as_conv1d=conv,
            has_bias=False,
            zero_fraction=0.33,
        )
        for k, (o, i, conv) in adapter.modules(c).items()
    ]
    words = (
        ["<unk>", "a", "<pad>", "b", "▁hello", "▁world"]
        + [f"unused_{i}" for i in range(6, 31)]
        + ["<blank>"]
    )
    t = Tokenizer(models.BPE(vocab={w: i for i, w in enumerate(words)}, merges=[]))
    t.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    t.add_special_tokens(["<unk>", "<pad>", "<blank>"])
    t.save(str(root / "tokenizer.json"))
    (root / "config.json").write_text(json.dumps(c))
    (root / "ternary.json").write_text(json.dumps(metadata))
    torch.manual_seed(61637)
    model = adapter.native_model(c).eval()
    state = model.state_dict()
    packed = dict(state)
    rng = np.random.default_rng(61637)
    for name, (out, ins, conv) in adapter.modules(c).items():
        digits = rng.integers(0, 3, (out, ins), dtype=np.int16)
        scales = torch.full((out, ins // 128), 0.01, dtype=torch.float16)
        floats = torch.from_numpy((digits.astype(np.float32) - 1) * float(scales[0, 0]))
        state[name + ".weight"] = floats.unsqueeze(-1) if conv else floats
        del packed[name + ".weight"]
        packed[name + ".qweight"] = pack_digits(digits)
        packed[name + ".scales"] = scales
    model.load_state_dict(state, strict=True)
    save_file(packed, str(root / "model.safetensors"))
    return model, c, packed


def reference_decode(model, inputs, config):
    with torch.inference_mode():
        encoded = model.get_audio_features(**inputs, output_attention_mask=True)
        frames = int(encoded.attention_mask.sum())
        history, steps, frame = [config["blank_token_id"]], [], 0
        while frame < frames:
            for index in range(10):
                prediction = model.decoder(
                    torch.tensor([history], device=encoded.pooler_output.device)
                )[:, -1:]
                logits = model.joint(prediction, encoded.pooler_output[:, frame : frame + 1])[0, 0]
                token = int(torch.softmax(logits[: config["vocab_size"]], -1).argmax())
                duration = int(torch.softmax(logits[config["vocab_size"] :], -1).argmax())
                if token != config["blank_token_id"]:
                    history.append(token)
                if duration == 0 and (token == config["blank_token_id"] or index == 9):
                    duration = 1
                steps.append((frame, token, duration))
                frame += duration
                if duration:
                    break
        return history[1:], steps


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("mode", ["seeded", "blank", "repeat-zero", "repeat-two", "overshoot"])
def test_whole_native_graph_matches_independent_decode(tmp_path, backend, mode):
    from safetensors.torch import save_file

    reference, config, packed = native_fixture(tmp_path)
    if mode != "seeded":
        with torch.no_grad():
            reference.joint.head.weight.zero_()
            reference.joint.head.bias.fill_(-10)
            reference.joint.head.bias[31 if mode == "blank" else 1] = 10
            reference.joint.head.bias[
                32 + (2 if mode == "repeat-two" else 4 if mode == "overshoot" else 0)
            ] = 10
        for key in ("joint.head.weight", "joint.head.bias"):
            packed[key] = reference.state_dict()[key]
        save_file(packed, str(tmp_path / "model.safetensors"))
    loaded, processor, _ = adapter.load_components(tmp_path, config, backend)
    reference.to(backend)
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in reference.state_dict().items())
    audio = np.sin(np.arange(4800, dtype=np.float32) * 0.08)
    inputs = dict(processor[0]([audio], sampling_rate=16000, return_tensors="pt"))
    inputs = {k: v.to(backend) for k, v in inputs.items()}
    expected = reference_decode(reference, inputs, config)
    actual = adapter.decode(loaded, inputs, config)
    assert actual == expected
    result = adapter.transcribe(loaded, processor, config, audio, backend)
    assert result == {
        "text": processor[1]
        .decode(
            expected[0],
            group_tokens=False,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        .strip(),
        "language": None,
    }
    if mode == "repeat-zero":
        assert result["text"] == "a" * 40
    if mode == "blank":
        assert result["text"] == ""


@pytest.mark.parametrize("damage", ["missing", "vad", "shape", "extra", "nan", "duplicate"])
def test_incomplete_graphs_rejected(tmp_path, damage):
    from safetensors.torch import save_file

    _, c, packed = native_fixture(tmp_path)
    key = "encoder.layers.0.self_attn.q_proj.qweight"
    if damage == "missing":
        del packed[key]
    elif damage == "vad":
        del packed["vad_head.proj.weight"]
    elif damage == "shape":
        packed[key] = packed[key][:, :-1].contiguous()
    elif damage == "extra":
        packed["unknown.weight"] = torch.ones((2, 2))
    elif damage == "nan":
        packed["joint.head.weight"].fill_(float("nan"))
    else:
        save_file({key: packed[key]}, str(tmp_path / "extra.safetensors"))
    save_file(packed, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError):
        adapter.load_components(tmp_path, c, "cpu")


@pytest.mark.parametrize("damage", [None, "missing", "metadata", "unbound"])
def test_metadata_discovery_without_original_weights(tmp_path, damage):
    from test_models import FakeHub, model

    from oryn_engine.models import ModelStore

    config = original()

    class Hub(FakeHub):
        def read_json(self, repo, revision, name):
            v = config if name == "config.json" else json.loads((FIXTURE / name).read_text())
            if name == "ternary.json" and damage == "metadata":
                v["quant"]["group_size"] = 64
            return v

    names = ["config.json", "model.safetensors", "tokenizer.json", "ternary.json"]
    if damage == "missing":
        names.remove("ternary.json")
    if damage == "unbound":
        names.append("processor_config.json")
    hub = Hub(
        config=config,
        pipeline_tag="automatic-speech-recognition",
        files=[(n, 100, "a" * 64) for n in names],
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda p: object() if p.pack_id == "oryn.transformers-pytorch" else None
    )
    result = store.inspect(model("independent/ternary-transducer"))
    assert result["executable"] is (damage is None), result
    if damage is None:
        assert result["estimatedMemoryBytes"] > 6 * 1024**3
        assert result["taskProfile"]["features"]["wordTimestamps"] == "unsupported"
    assert not hub.downloaded


@pytest.mark.parametrize("length", [512, 1599, 1600, 1601, 480000])
def test_standardized_features_match_independent_numpy(tmp_path, length):
    import librosa

    _, c, _ = native_fixture(tmp_path)
    feature, _ = adapter.components(tmp_path, c)
    audio = np.random.default_rng(61637).uniform(-0.9, 0.9, length).astype(np.float32)
    actual = feature([audio], sampling_rate=16000, return_tensors="pt")
    emphasized = np.concatenate((audio[:1], audio[1:] - np.float32(0.97) * audio[:-1]))
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(emphasized, (256, 256)), 512)[::160]
    power = np.abs(np.fft.rfft(windows * np.pad(np.hanning(400), (56, 56)), axis=-1)) ** 2
    filters = librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, norm="slaney")
    mels = np.log(power @ filters.T + 2**-24)
    valid = length // 160
    expected = (mels - mels[:valid].mean(0)) / (mels[:valid].std(0, ddof=1) + 1e-5)
    expected[valid:] = 0
    np.testing.assert_allclose(actual["input_features"][0], expected, atol=3e-4, rtol=3e-4)
    np.testing.assert_array_equal(actual["attention_mask"][0], np.arange(len(mels)) < valid)


def test_worker_uses_original_export_and_ignores_repository_generation(tmp_path):
    import os
    import subprocess
    import sys
    import wave

    from safetensors.torch import save_file

    from oryn_engine.runtime_packs import transformers_worker
    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry

    model, c, packed = native_fixture(tmp_path)
    with torch.no_grad():
        model.joint.head.weight.zero_()
        model.joint.head.bias.fill_(-10)
        model.joint.head.bias[1] = 10
        model.joint.head.bias[33] = 10
    for key in ("joint.head.weight", "joint.head.bias"):
        packed[key] = model.state_dict()[key]
    save_file(packed, str(tmp_path / "model.safetensors"))
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": 1, "do_sample": True, "max_new_tokens": 1})
    )
    audio = tmp_path / "input.wav"
    with wave.open(str(audio), "wb") as f:
        f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        f.writeframes((np.sin(np.arange(4800) * 0.08) * 16000).astype("<i2").tobytes())
    payload = {
        "modelRoot": str(tmp_path),
        "task": "automatic-speech-recognition",
        "backend": "mps",
        "input": {
            "audioArtifactId": "artifact_" + "b" * 24,
            "language": "auto",
            "timestamps": False,
        },
        "inputPath": str(audio),
        "outputPath": None,
    }
    assert transformers_worker.generate(payload) == {"text": "aaaa", "language": None}
    pack = builtin_registry().active_pack("oryn.transformers-pytorch")
    for _ in range(2):
        request = {
            "abi": pack.abi,
            "packId": pack.pack_id,
            "version": pack.version,
            "command": "generate",
            "payload": payload,
        }
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
        assert result["result"] == {"text": "aaaa", "language": None}


@pytest.mark.parametrize(
    "damage", ["vocabulary", "blank", "processor", "nonfinite", "short", "long", "logits"]
)
def test_invalid_components_and_audio_rejected(tmp_path, damage):
    model, config, _ = native_fixture(tmp_path)
    if damage in {"vocabulary", "blank"}:
        p = tmp_path / "tokenizer.json"
        v = json.loads(p.read_text())
        if damage == "vocabulary":
            v["model"]["vocab"]["other"] = 32
        else:
            v["model"]["vocab"]["<blank>"], v["model"]["vocab"]["a"] = 1, 31
            v["added_tokens"] = [t for t in v["added_tokens"] if t["content"] != "<blank>"]
        p.write_text(json.dumps(v))
        with pytest.raises(ValueError):
            adapter.components(tmp_path, config)
    elif damage == "processor":
        (tmp_path / "processor_config.json").write_text("{}")
        with pytest.raises(ValueError, match="PARAKEET_UNBOUND_PROCESSOR"):
            adapter.components(tmp_path, config)
    else:
        processor = adapter.components(tmp_path, config)
        audio = np.zeros(
            100 if damage == "short" else 480001 if damage == "long" else 1600, dtype=np.float32
        )
        if damage == "nonfinite":
            audio[0] = np.nan
        if damage == "logits":
            with torch.no_grad():
                model.joint.head.bias.fill_(float("inf"))
        with pytest.raises(ValueError):
            adapter.transcribe(model, processor, config, audio, "cpu")
