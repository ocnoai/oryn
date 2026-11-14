"""Standalone original codec uses the same full native graph as Breeze."""

import json
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from test_breeze_tasks import CODEC, small_configs

from oryn_engine import qwen_codec_tasks as q


def fixture(root, precision="float32"):
    root.mkdir(exist_ok=True)
    _, config = small_configs()
    (root / "config.json").write_text(json.dumps(config))
    (root / "preprocessor_config.json").write_bytes(
        (q.DATA / "codec_preprocessor.json").read_bytes()
    )
    torch.manual_seed(6176)
    model = q.native_model(config)
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if name.endswith(("embedding_sum", "embed_sum")):
                value.normal_(std=0.05)
        for module in model.decoder.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                module.weight.mul_(0.5)
    state = {k: v.to(getattr(torch, precision)).contiguous() for k, v in model.state_dict().items()}
    save_file(state, root / "model.safetensors")
    model.load_state_dict({k: v.float() for k, v in state.items()})
    return config, model


def wav(path, samples=7 * 1920 + 13, rate=24000, channels=1):
    pcm = (np.sin(np.arange(samples) * 0.13) * 5000).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.tobytes())


def test_original_complete_codec_geometry():
    header = json.loads(
        (Path(__file__).parent / "fixtures/qwen_codec/model.safetensors.header.json").read_text()
    )
    shapes = {k: v["shape"] for k, v in header.items() if k != "__metadata__"}
    q.validate_weight_shapes(CODEC, shapes)
    assert len(shapes) == 496


def test_original_metadata_resolves_complete_audio_runtime_without_text_tokenizer(tmp_path):
    from test_metadata_audit import row, snapshot

    from oryn_engine.metadata_audit import inspect_row

    hub = snapshot(
        tmp_path,
        config=CODEC,
        files=[(name, 100, "b" * 64) for name in q.FILES],
        pipeline_tag=q.TASK,
    )
    processor = json.loads((q.DATA / "codec_preprocessor.json").read_text())
    hub.client.read_json.side_effect = lambda repository, revision, filename: {
        "config.json": CODEC,
        "preprocessor_config.json": processor,
    }[filename]
    result = inspect_row(row(task=q.TASK), hub)
    assert result["status"] == "metadata-compatible", result
    assert result["taskProfile"]["contract"]["id"] == "audio-transformation-v1"
    assert result["executionVerified"] is False
    assert {item["name"] for item in result["selectedFiles"]} == set(q.FILES)
    hub.client.download.assert_not_called()


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("precision", ["float32", "float16", "bfloat16"])
def test_complete_encode_decode_parity(tmp_path, backend, precision):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    config, reference = fixture(tmp_path, precision)
    model, _, _ = q.load_components(tmp_path, config, backend)
    reference.to(backend)
    source = tmp_path / "input.wav"
    wav(source)
    value = {"operation": "encode", "audioArtifactId": "artifact_" + "a" * 24}
    with torch.inference_mode():
        actual = q.execute(model, value, backend, source, None)
        expected = q.execute(reference, value, backend, source, None)
        assert (
            actual == expected and len(actual["codes"]) == 8 and actual["durationSeconds"] == 0.64
        )
        q.execute(
            model,
            {"operation": "decode", "codes": actual["codes"]},
            backend,
            None,
            tmp_path / "result.wav",
        )
        q.execute(
            reference,
            {"operation": "decode", "codes": expected["codes"]},
            backend,
            None,
            tmp_path / "expected.wav",
        )
    with wave.open(str(tmp_path / "result.wav"), "rb") as f:
        assert (f.getframerate(), f.getnchannels(), f.getnframes()) == (24000, 1, 8 * 1920)
        raw = f.readframes(f.getnframes())
        assert any(raw)
    assert (tmp_path / "result.wav").read_bytes() == (tmp_path / "expected.wav").read_bytes()


@pytest.mark.parametrize("missing", [None, *q.FILES])
def test_discovery_requires_original_complete_codec(missing):
    from oryn_engine.model_layouts import assess_layout

    files = {*q.FILES, "configuration.json", "README.md", "modeling_custom.py", "pytorch_model.bin"}
    if missing:
        files.remove(missing)
    result = assess_layout(sorted(files), CODEC, q.TASK)
    if missing:
        assert result.artifact_reasons
    else:
        assert not result.artifact_reasons and not result.runtime_reasons
        assert set(result.selected_files) == set(q.FILES)


@pytest.mark.parametrize(
    "change", ["encoder", "decoder", "shape", "extra", "integer", "nonfinite", "initialized"]
)
def test_full_weight_validation(tmp_path, change):
    from safetensors.torch import load_file

    config, _ = fixture(tmp_path)
    path = tmp_path / "model.safetensors"
    state = load_file(path)
    key = next(k for k in state if k.startswith("decoder.") and state[k].ndim == 2)
    if change in {"encoder", "decoder"}:
        del state[next(k for k in state if k.startswith(change + "."))]
    elif change == "shape":
        state[key] = state[key][:-1]
    elif change == "extra":
        state["extra.weight"] = torch.ones(1)
    elif change == "integer":
        state[key] = state[key].to(torch.int32)
    elif change == "nonfinite":
        state[key][0, 0] = float("inf")
    else:
        state[next(k for k in state if k.endswith(".initialized"))].zero_()
    save_file(state, path)
    with pytest.raises(ValueError, match="QWEN_CODEC"):
        q.load_components(tmp_path, config, "cpu")


@pytest.mark.parametrize(
    "codes",
    [
        [],
        [[0] * 15],
        [[0] * 17],
        [[2048] * 16],
        [[-1] * 16],
        [[True] * 16],
        [[1.0] * 16],
        [[0] * 16] * 376,
    ],
)
def test_codes_are_bounded_and_have_all_sixteen_books(codes):
    with pytest.raises(ValueError, match="QWEN_CODEC_CODES"):
        q.validate_codes(codes)


@pytest.mark.parametrize(
    "samples,rate,channels",
    [(1919, 24000, 1), (1920, 16000, 1), (3840, 24000, 2), (720001, 24000, 1)],
)
def test_source_audio_has_native_geometry(tmp_path, samples, rate, channels):
    path = tmp_path / "input.wav"
    wav(path, samples, rate, channels)
    with pytest.raises(ValueError, match="QWEN_CODEC"):
        q.read_audio(path)


@pytest.mark.parametrize("name", q.FILES)
def test_symlink_export_is_rejected(tmp_path, name):
    config, _ = fixture(tmp_path)
    path = tmp_path / name
    moved = tmp_path / "outside"
    path.rename(moved)
    path.symlink_to(moved)
    with pytest.raises(ValueError):
        q.load_components(tmp_path, config, "cpu")


@pytest.mark.parametrize("component", ["encoder", "decoder"])
def test_nonfinite_intermediates_cannot_become_valid_codes_or_clipped_audio(tmp_path, component):
    config, _ = fixture(tmp_path)
    model, _, _ = q.load_components(tmp_path, config, "cpu")
    if component == "encoder":
        from transformers.models.mimi.modeling_mimi import MimiEuclideanCodebook

        book = next(m for m in model.encoder.modules() if isinstance(m, MimiEuclideanCodebook))
        book.embed_sum.fill_(torch.finfo(torch.float32).max)
        source = tmp_path / "input.wav"
        wav(source)
        value = {"operation": "encode", "audioArtifactId": "artifact_" + "a" * 24}
        target = None
    else:
        model.decoder.decoder[-1].register_forward_pre_hook(
            lambda module, args: (torch.full_like(args[0], float("inf")),)
        )
        source = None
        target = tmp_path / "result.wav"
        value = {"operation": "decode", "codes": [[1] * 16]}
    with torch.inference_mode(), pytest.raises(ValueError, match="NONFINITE"):
        q.execute(model, value, "cpu", source, target)


@pytest.mark.parametrize("frames", [1, 7, 304])
def test_chunked_decode_and_native_frame_clock(tmp_path, frames):
    config, model = fixture(tmp_path)
    codes = torch.arange(frames * 16).view(1, frames, 16) % 2048
    with torch.inference_mode():
        native = model.decode(codes).audio_values[0]
        result = q.execute(
            model,
            {"operation": "decode", "codes": codes[0].tolist()},
            "cpu",
            None,
            tmp_path / "result.wav",
        )
    assert native.shape == (frames * 1920,) and result["frames"] == frames * 1920
    with wave.open(str(tmp_path / "result.wav"), "rb") as f:
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    np.testing.assert_array_equal(pcm, np.round(native.numpy() * 32767).astype("<i2"))


def test_profile_preserves_encode_decode_and_rejects_unbound_operations():
    from oryn_protocol.task_contracts import effective_input, validate_invocation

    from oryn_engine.task_profiles import profile_for_execution

    profile = profile_for_execution(
        {
            "runtimePack": {
                "packId": "oryn.transformers-pytorch",
                "taskContract": "audio-transformation-v1",
            }
        },
        CODEC,
    )
    encoded = effective_input(
        profile, {"operation": "encode", "audioArtifactId": "artifact_" + "a" * 24}
    )
    assert encoded["sampleRate"] == 24000
    validate_invocation(profile, encoded)
    validate_invocation(profile, {"operation": "decode", "codes": [[1] * 16], "sampleRate": 24000})
    for value in [
        {**encoded, "operation": "enhance"},
        {**encoded, "sampleRate": 16000},
        {**encoded, "codes": [[1] * 16]},
        {"operation": "decode", "audioArtifactId": "artifact_" + "a" * 24},
    ]:
        with pytest.raises(ValueError):
            validate_invocation(profile, value)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_offline_worker_roundtrip(tmp_path, backend):
    import os
    import subprocess
    import sys

    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    fixture(tmp_path)
    source = tmp_path / "input.wav"
    wav(source)
    request = dict(
        abi="1",
        packId="oryn.transformers-pytorch",
        version=builtin_registry().active_pack("oryn.transformers-pytorch").version,
        command="generate",
    )

    def invoke(value, input_path, output_path):
        request["payload"] = dict(
            modelRoot=str(tmp_path),
            task=q.TASK,
            backend=backend,
            input=value,
            inputPath=str(input_path) if input_path else None,
            outputPath=str(output_path) if output_path else None,
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
        return result["result"]

    encoded = invoke(
        {"operation": "encode", "audioArtifactId": "artifact_" + "a" * 24}, source, None
    )
    assert len(encoded["codes"]) == 8
    result = invoke(
        {"operation": "decode", "codes": encoded["codes"]}, None, tmp_path / "result.wav"
    )
    assert result["frames"] == 8 * 1920
    with wave.open(str(tmp_path / "result.wav"), "rb") as f:
        assert any(f.readframes(f.getnframes()))


def test_engine_checks_decoded_waveform_against_requested_codes(tmp_path):
    from oryn_engine.runners import JobExecutionError, _validate_codec_audio

    _, model = fixture(tmp_path)
    path = tmp_path / "result.wav"
    value = {"operation": "decode", "codes": [[1] * 16] * 4}
    with torch.inference_mode():
        result = q.execute(model, value, "cpu", None, path)
    _validate_codec_audio(path, result, value, "cpu")
    for changed in [
        {**result, "frames": 1920},
        {**result, "durationSeconds": 0.1},
        {**result, "channels": 2},
        {**result, "sampleRate": 16000},
        {**result, "extra": True},
    ]:
        with pytest.raises(JobExecutionError):
            _validate_codec_audio(path, changed, value, "cpu")
    with pytest.raises(JobExecutionError):
        _validate_codec_audio(path, result, {"codes": [[1] * 16] * 3}, "cpu")


def test_encoded_result_obeys_structured_contract(tmp_path):
    from oryn_engine.shared_tasks import validate_task_output

    _, model = fixture(tmp_path)
    source = tmp_path / "input.wav"
    wav(source)
    with torch.inference_mode():
        result = q.execute(
            model,
            {"operation": "encode", "audioArtifactId": "artifact_" + "a" * 24},
            "cpu",
            source,
            None,
        )
    assert validate_task_output(q.TASK, result) == result
    with pytest.raises(ValueError):
        validate_task_output(q.TASK, {**result, "codes": [[70000] * 16]})


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_checked_encoder_and_decoder_match_unmodified_native_codec(tmp_path, backend):
    from oryn_engine.native_breeze.codec import Qwen3TTSTokenizerV2Model
    from oryn_engine.native_breeze.codec_config import Qwen3TTSTokenizerV2Config

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    config, owned = fixture(tmp_path)
    c = Qwen3TTSTokenizerV2Config(**config)
    for part in [c, c.encoder_config, c.decoder_config]:
        part._attn_implementation = "eager"
    reference = Qwen3TTSTokenizerV2Model(c).eval()
    reference.load_state_dict(owned.state_dict(), strict=True)
    owned.to(backend)
    reference.to(backend)
    signal = (torch.sin(torch.arange(7 * 1920 + 13, device=backend) * 0.13) * 0.2).unsqueeze(0)
    with torch.inference_mode():
        actual = owned.encode(
            signal, padding_mask=torch.ones_like(signal, dtype=torch.bool)
        ).audio_codes[0]
        expected = reference.encode(
            signal, padding_mask=torch.ones_like(signal, dtype=torch.bool)
        ).audio_codes[0]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        a = owned.decode(actual.unsqueeze(0)).audio_values[0]
        z = reference.decode(expected.unsqueeze(0)).audio_values[0]
    torch.testing.assert_close(a, z, atol=0, rtol=0)
