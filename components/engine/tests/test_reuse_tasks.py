"""Complete RE-USE graphs, original metadata, native spectral math and worker I/O."""

import json
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from oryn_engine import reuse_tasks as m


def config(causal=False, tiny=True):
    raw = json.loads((m.DATA / ("causal.json" if causal else "offline.json")).read_text())
    if tiny:
        raw["model_cfg"].update(hid_feature=8, num_tfmamba=2)
        if causal:
            raw["model_cfg"]["maximum_layer"] = 2
    return m.normalized_config(raw, m.TASK)


def fixture(root, causal=False, precision="float32"):
    root.mkdir(exist_ok=True)
    c = config(causal)
    (root / "config.json").write_text(
        json.dumps({k: v for k, v in c.items() if k not in {"model_type", "architectures"}})
    )
    torch.manual_seed(6185)
    model = m.native_model(c)
    state = {k: v.to(getattr(torch, precision)).contiguous() for k, v in model.state_dict().items()}
    save_file(state, root / "model.safetensors")
    model.load_state_dict({k: v.float() for k, v in state.items()})
    return c, model


def wav(path, samples=643, rate=8000, channels=1):
    pcm = (np.sin(np.arange(samples) * 0.07) * 5000).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.tobytes())


@pytest.mark.parametrize("causal,count", [(False, 1416), (True, 1318)])
def test_original_complete_tensor_geometry(causal, count):
    h = json.loads(
        (
            Path(__file__).parent
            / "fixtures/reuse"
            / ("causal" if causal else "offline")
            / "model.safetensors.header.json"
        ).read_text()
    )
    shapes = {k: v["shape"] for k, v in h.items() if k != "__metadata__"}
    m.validate_weight_shapes(config(causal, tiny=False), shapes)
    assert len(shapes) == count


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("precision", ["float32", "float16", "bfloat16"])
def test_complete_stored_graph_and_waveform(tmp_path, causal, backend, precision):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    c, reference = fixture(tmp_path, causal, precision)
    model, _, _ = m.load_components(tmp_path, c, backend)
    source = tmp_path / "input.wav"
    wav(source)
    audio = m.read_audio(source)
    with torch.inference_mode():
        expected = m.enhance(reference, audio, c, "cpu")
        actual = m.enhance(model, audio, c, backend)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        result = m.execute(
            model,
            c,
            {"operation": "enhance", "audioArtifactId": "artifact_" + "a" * 24},
            backend,
            source,
            tmp_path / "result.wav",
        )
    assert actual.shape == (643,) and torch.isfinite(actual).all()
    m.validate_output(tmp_path / "result.wav", result, source, backend)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "key,value",
    [
        ("hid_feature", 7),
        ("num_tfmamba", 0),
        ("d_state", 8),
        ("expand", 2),
        ("mapping", False),
        ("compress_factor", "log1p"),
        ("norm_epsilon", True),
    ],
)
def test_closed_configuration_rejects_unreviewed_geometry(causal, key, value):
    c = config(causal)
    c["model_cfg"][key] = value
    with pytest.raises(ValueError, match="REUSE"):
        m.validate_config(c)


def test_normalization_preserves_explicit_architecture_and_validates_causal_depth():
    c = config(True)
    c["architectures"] = ["OtherGraph"]
    with pytest.raises(ValueError, match="REUSE"):
        m.normalized_config(c, m.TASK)
    c = config(True)
    c["model_cfg"]["maximum_layer"] = 1
    with pytest.raises(ValueError, match="REUSE"):
        m.validate_config(c)
    with pytest.raises(ValueError, match="REUSE"):
        m.normalized_config({"model_type": m.FAMILY}, m.TASK)


@pytest.mark.parametrize("causal", [False, True])
def test_original_metadata_without_processor_or_tokenizer(tmp_path, causal):
    from test_metadata_audit import row, snapshot

    from oryn_engine.metadata_audit import inspect_row

    raw = json.loads((m.DATA / ("causal.json" if causal else "offline.json")).read_text())
    hub = snapshot(
        tmp_path, config=raw, files=[(name, 100, "b" * 64) for name in m.FILES], pipeline_tag=m.TASK
    )
    result = inspect_row(row(task=m.TASK), hub)
    assert result["status"] == "metadata-compatible", result
    assert {f["name"] for f in result["selectedFiles"]} == set(m.FILES) and not result[
        "executionVerified"
    ]
    hub.client.download.assert_not_called()


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("missing", [None, *m.FILES])
def test_original_complete_subset(causal, missing):
    from oryn_engine.model_layouts import assess_layout

    files = {*m.FILES, "model.py", "generator.pt"}
    if missing:
        files.remove(missing)
    result = assess_layout(sorted(files), config(causal), m.TASK)
    if missing:
        assert result.artifact_reasons
    else:
        assert (
            set(result.selected_files) == set(m.FILES)
            and not result.runtime_reasons
            and not result.artifact_reasons
        )


@pytest.mark.parametrize(
    "change",
    [
        "encoder",
        "lookahead",
        "mamba",
        "unused_base",
        "early_magnitude",
        "early_phase",
        "extra",
        "shape",
        "integer",
        "nonfinite",
        "decay_overflow",
    ],
)
def test_every_original_array_required_including_unused_heads(tmp_path, change):
    c, _ = fixture(tmp_path, True)
    p = tmp_path / "model.safetensors"
    state = load_file(p)
    key = next(k for k in state if k.endswith(".A_log"))
    prefixes = {
        "encoder": "dense_encoder.dense_conv_1_1.",
        "lookahead": "dense_encoder.dense_conv_1_3.",
        "mamba": "TSMamba.0.freq_mamba.backward_blocks.",
        "unused_base": "mask_decoder.",
        "early_magnitude": "mask_decoder_list.0.",
        "early_phase": "phase_decoder_list.0.",
    }
    if change in prefixes:
        del state[next(k for k in state if k.startswith(prefixes[change]))]
    elif change == "extra":
        state["extra.weight"] = torch.ones(1)
    elif change == "shape":
        state[key] = state[key][:-1]
    elif change == "integer":
        state[key] = state[key].to(torch.int32)
    elif change == "nonfinite":
        state[key][0, 0] = float("nan")
    else:
        state[key][0, 0] = 1000
    save_file(state, p)
    with pytest.raises(ValueError, match="REUSE"):
        m.load_components(tmp_path, c, "cpu")


@pytest.mark.parametrize(
    "samples,rate,channels", [(319, 8000, 1), (643, 16000, 1), (640, 8000, 2), (16001, 8000, 1)]
)
def test_native_audio_bounds(tmp_path, samples, rate, channels):
    p = tmp_path / "input.wav"
    wav(p, samples, rate, channels)
    with pytest.raises(ValueError, match="REUSE"):
        m.read_audio(p)


@pytest.mark.parametrize("name", m.FILES)
def test_symlink_checkpoint_rejected(tmp_path, name):
    c, _ = fixture(tmp_path)
    p = tmp_path / name
    other = tmp_path / "outside"
    p.rename(other)
    p.symlink_to(other)
    with pytest.raises(ValueError, match="REUSE"):
        m.load_components(tmp_path, c, "cpu")


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("samples", [320, 479, 16000])
def test_original_clock_and_silence_remain_finite(causal, samples):
    c = config(causal)
    model = m.native_model(c)
    with torch.inference_mode():
        result = m.enhance(model, torch.zeros(samples), c, "cpu")
    assert result.shape == (samples,) and torch.isfinite(result).all()
    tail = samples % c["stft_cfg"]["hop_size"]
    if tail:
        torch.testing.assert_close(result[-tail:], torch.full((tail,), 1e-8), atol=0, rtol=0)


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("ahead", [0, 1, 2])
def test_all_causal_decoder_and_lookahead_branches_are_native(depth, ahead):
    c = config(True)
    model = m.native_model(c)
    with torch.inference_mode():
        result = model(torch.rand(1, 161, 7), torch.rand(1, 161, 7), depth, ahead)
    assert [list(x.shape) for x in result] == [[1, 161, 7], [1, 161, 7], [1, 161, 7, 2]]
    assert all(torch.isfinite(x).all() for x in result)


@pytest.mark.parametrize("failure", ["phase", "magnitude"])
def test_nonfinite_values_rejected_before_bounded_math(failure):
    c = config()
    model = m.native_model(c)
    layer = (
        model.phase_decoder.phase_conv_r if failure == "phase" else model.mask_decoder.final_conv
    )
    layer.register_forward_pre_hook(lambda module, args: (torch.full_like(args[0], float("inf")),))
    with torch.inference_mode(), pytest.raises(ValueError, match="NONFINITE"):
        m.enhance(model, torch.rand(643), c, "cpu")


def test_spectral_suppression_threshold_is_strict_and_padding_is_exact():
    c = config()
    phase = torch.zeros(1, 161, 9)
    amp = torch.ones_like(phase)
    amp[:, :81] = 0
    output = m.spectral_output(amp, phase, 323, c)
    torch.testing.assert_close(output[:320], torch.zeros(320), atol=0, rtol=0)
    torch.testing.assert_close(output[320:], torch.full((3,), 1e-8), atol=0, rtol=0)
    amp[:, 80] = 1
    assert bool(m.spectral_output(amp, phase, 320, c).abs().max() > 0)


@pytest.mark.parametrize("causal", [False, True])
def test_profile_and_host_validation_preserve_input_clock(tmp_path, causal):
    from oryn_protocol.task_contracts import effective_input, validate_invocation

    from oryn_engine.runners import JobExecutionError, _validate_reuse_audio
    from oryn_engine.task_profiles import profile_for_execution

    c, model = fixture(tmp_path, causal)
    profile = profile_for_execution(
        {
            "runtimePack": {
                "packId": "oryn.transformers-pytorch",
                "taskContract": "audio-transformation-v1",
            }
        },
        c,
    )
    value = effective_input(profile, {"audioArtifactId": "artifact_" + "a" * 24})
    assert (
        value["sampleRate"] == 8000
        and value["operation"] == "enhance"
        and profile["limits"]["audioDurationSeconds"] == 2
    )
    validate_invocation(profile, value)
    for change in [{"operation": "encode"}, {"sampleRate": 16000}]:
        with pytest.raises(ValueError):
            validate_invocation(profile, {**value, **change})
    source = tmp_path / "input.wav"
    wav(source)
    output = tmp_path / "result.wav"
    with torch.inference_mode():
        result = m.execute(model, c, value, "cpu", source, output)
    _validate_reuse_audio(output, result, source, "cpu")
    for change in [
        {"frames": 640},
        {"channels": True},
        {"sampleRate": 16000},
        {"durationSeconds": 1},
        {"backend": "mps"},
        {"extra": True},
    ]:
        with pytest.raises(JobExecutionError):
            _validate_reuse_audio(output, {**result, **change}, source, "cpu")


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_offline_shared_worker_complete_native_graph(tmp_path, causal, backend):
    import os
    import subprocess
    import sys

    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    fixture(tmp_path, causal)
    source = tmp_path / "input.wav"
    wav(source)
    output = tmp_path / "result.wav"
    request = dict(
        abi="1",
        packId="oryn.transformers-pytorch",
        version=builtin_registry().active_pack("oryn.transformers-pytorch").version,
        command="generate",
        payload=dict(
            modelRoot=str(tmp_path),
            task=m.TASK,
            backend=backend,
            input={
                "operation": "enhance",
                "sampleRate": 8000,
                "audioArtifactId": "artifact_" + "a" * 24,
            },
            inputPath=str(source),
            outputPath=str(output),
        ),
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
    m.validate_output(output, result["result"], source, backend)


def test_truncated_and_symlink_audio_rejected(tmp_path):
    p = tmp_path / "input.wav"
    wav(p)
    p.write_bytes(p.read_bytes()[:-2])
    with pytest.raises(ValueError, match="TRUNCATED"):
        m.read_audio(p)
    other = tmp_path / "outside.wav"
    p.rename(other)
    p.symlink_to(other)
    with pytest.raises(ValueError, match="SCOPE"):
        m.read_audio(p)


@pytest.mark.parametrize(
    "change", [{"operation": "encode"}, {"codes": [[1] * 16]}, {"sampleRate": 24000}]
)
def test_only_native_enhancement_inputs_are_admitted(tmp_path, change):
    c, model = fixture(tmp_path)
    source = tmp_path / "input.wav"
    wav(source)
    with torch.inference_mode(), pytest.raises(ValueError, match="REUSE_INPUT"):
        m.execute(
            model,
            c,
            {"operation": "enhance", "audioArtifactId": "artifact_" + "a" * 24, **change},
            "cpu",
            source,
            tmp_path / "result.wav",
        )
