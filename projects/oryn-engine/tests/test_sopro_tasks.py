"""Complete synthetic speech graphs and original metadata; no original weights."""

import copy
import json
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from oryn_engine import sopro_tasks as s

FIXTURES = Path(__file__).parent / "fixtures/sopro"
RAW = json.loads((FIXTURES / "config.json").read_text())


def small_config():
    c = copy.deepcopy(s.normalized_config(RAW, s.TASK))
    c["model"].update(
        latent_dim=16,
        cond_hidden_dim=32,
        ar_model_dim=32,
        ar_blocks=2,
        ar_heads=2,
        ar_kv_heads=2,
        acoustic_dit_dim=32,
        acoustic_dit_depth=2,
        acoustic_dit_heads=2,
        acoustic_dit_dim_head=16,
        acoustic_time_embed_dim=16,
    )
    c["semantic_encoder"].update(d_model=32, layers=2, heads=2, ffn_dim=64)
    c["speaker_encoder"].update(
        stem_channels=8,
        stage_channels=[8, 12, 16],
        blocks_per_stage=[1] * 3,
        id_head_hidden=16,
        style_head_hidden=16,
        attn_hidden=8,
    )
    c["vocoder"].update(dim=32, intermediate_dim=64, num_layers=2, block_lookaheads=[3] * 2)
    return c


def fixture(root, precision="float32"):
    root.mkdir(exist_ok=True)
    c = small_config()
    torch.manual_seed(6172)
    models = s.native_components(c)
    # Complete graph still runs every layer; a strongly biased EOS head bounds its
    # random generation. The minimum duration must be reached before EOS is legal.
    with torch.no_grad():
        models["model"].ar_prior.token_head.weight.mul_(0.01)
        models["model"].ar_prior.token_head.bias.zero_()
        models["model"].ar_prior.token_head.bias[-1] = 100
    for name, model in models.items():
        state = {
            k: v.to(getattr(torch, precision)).contiguous() for k, v in model.state_dict().items()
        }
        save_file(state, root / (name + ".safetensors"))
        model.load_state_dict({k: v.float() for k, v in state.items()})
    (root / "config.json").write_text(json.dumps(c))
    shutil.copyfile(FIXTURES / "tokenizer.model", root / "tokenizer.model")
    return c, models


def test_original_headers_match_all_four_graphs():
    c = s.normalized_config(RAW, s.TASK)
    shapes = {
        name: {
            k: v["shape"]
            for k, v in json.loads(
                (FIXTURES / (name + ".safetensors.header.json")).read_text()
            ).items()
            if k != "__metadata__"
        }
        for name in s.COMPONENTS
    }
    models = s.validate_weight_shapes(c, shapes)
    assert [len(shapes[n]) for n in s.COMPONENTS] == [316, 103, 206, 83]
    assert len(models["model"].ar_prior.temporal.layers) == 12
    assert len(models["model"].acoustic_head.blocks) == 8
    tok = s.tokenizer(FIXTURES, c)
    assert s.text_ids(tok, "hello", "en") == [1, 276, 4, 6393, 265, 2]


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("precision", ["float32", "float16"])
def test_complete_graph_synthesis_and_loaded_parity(tmp_path, backend, precision):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    c, reference = fixture(tmp_path, precision)
    loaded, tok, _ = s.load_components(tmp_path, c, backend)
    for m in reference.values():
        m.to(backend)
    wav = torch.sin(torch.arange(16000) * 0.11) * 0.2
    with torch.inference_mode():
        torch.manual_seed(6172)
        expected = s.synthesize(reference, tok, c, "hello", "en", wav, 16000, backend)
        torch.manual_seed(6172)
        actual = s.synthesize(loaded, tok, c, "hello", "en", wav, 16000, backend)
    assert 1000 <= actual.numel() <= 24000
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert float(actual.abs().max()) > 0


@pytest.mark.parametrize("kind", ["semantic", "speaker", "vocoder"])
def test_mel_frontends_against_numpy(kind):
    from oryn_engine.native_sopro.audio_math import MelSpectrogram

    sample_rate, nfft, win, hop, mels, power, scale, norm, lo, hi = {
        "semantic": (16000, 400, 400, 160, 80, 2, "slaney", "slaney", 0, 8000),
        "speaker": (16000, 1024, 400, 160, 80, 2, "slaney", "slaney", 20, 7600),
        "vocoder": (24000, 1024, 1024, 256, 100, 1, "htk", None, 0, 12000),
    }[kind]
    signal = (0.21 * np.sin(np.arange(4807) * 0.17) + 0.03 * np.cos(np.arange(4807) * 0.31)).astype(
        np.float32
    )
    module = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=nfft,
        win_length=win,
        hop_length=hop,
        n_mels=mels,
        power=power,
        norm=norm,
        mel_scale=scale,
        f_min=lo,
        f_max=hi,
    )
    actual = module(torch.tensor(signal)).numpy()

    def hz_to_mel(x):
        if scale == "htk":
            return 2595 * np.log10(1 + x / 700)
        return x / (200 / 3) if x < 1000 else 15 + np.log(x / 1000) / (np.log(6.4) / 27)

    points = np.linspace(hz_to_mel(lo), hz_to_mel(hi), mels + 2, dtype=np.float64)
    hz = (
        700 * (10 ** (points / 2595) - 1)
        if scale == "htk"
        else np.where(
            points < 15, points * (200 / 3), 1000 * np.exp((points - 15) * (np.log(6.4) / 27))
        )
    )
    freq = np.linspace(0, sample_rate / 2, nfft // 2 + 1)
    bank = np.maximum(
        0,
        np.minimum(
            (freq[:, None] - hz[:-2]) / (hz[1:-1] - hz[:-2]),
            (hz[2:] - freq[:, None]) / (hz[2:] - hz[1:-1]),
        ),
    )
    if norm == "slaney":
        bank *= 2 / (hz[2:] - hz[:-2])
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / win)
    window = np.pad(window, ((nfft - win) // 2, (nfft - win) // 2))
    padded = np.pad(signal, (nfft // 2, nfft // 2), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, nfft)[::hop]
    expected = (np.abs(np.fft.rfft(frames * window)) ** power @ bank).T
    np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize(
    "source,target", [(16000, 24000), (24000, 16000), (44100, 24000), (48000, 24000)]
)
def test_resampling_against_independent_sinc_sum(source, target):
    from oryn_engine.native_sopro.audio_math import Resample, resample

    # Direct continuous-time convolution, independent of TorchAudio's polyphase
    # kernel construction, padding and conv1d output interleaving.
    x = (0.2 * np.sin(np.arange(317) * 0.13)).astype(np.float32)
    times = np.arange(int(np.ceil(len(x) * target / source))) * source / target
    delta = (np.arange(len(x))[None, :] - times[:, None]) * min(source, target) * 0.99 / source
    clipped = np.clip(delta, -6, 6)
    kernel = (
        np.sinc(clipped) * np.cos(clipped * np.pi / 12) ** 2 * min(source, target) * 0.99 / source
    )
    expected = kernel @ x
    actual = resample(torch.tensor(x), source, target).numpy()
    cached = Resample(source, target)(torch.tensor(x)).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(cached, expected, atol=1e-6, rtol=1e-5)


def test_autoregressive_cache_matches_full_causal_prefix():
    from oryn_engine.native_sopro.nn.decode import EagerDecoder

    model = s.native_components(small_config())["model"]
    prefix = torch.randn(1, 17, 32)
    decoder = EagerDecoder(model, 32)
    with torch.inference_mode():
        decoder.prefill(prefix)
        for value in [13, 200, 6]:
            token = torch.tensor([value])
            actual = decoder.step(token)
            prefix = torch.cat([prefix, model.embed_semantic(token.view(1, 1))], dim=1)
            expected = EagerDecoder(model, 32).prefill(prefix)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_native_istft_matches_independent_overlap_add():
    from oryn_engine.native_sopro.vocoder import ISTFT

    nfft, hop, frames = 1024, 256, 17
    rng = np.random.default_rng(61)
    spec = (rng.normal(size=(513, frames)) + 1j * rng.normal(size=(513, frames))).astype(
        np.complex64
    )
    actual = ISTFT(nfft, hop)(torch.tensor(spec)[None])[0].numpy()
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nfft) / nfft)
    samples = np.fft.irfft(spec, nfft, axis=0) * window[:, None]
    out = np.zeros((frames - 1) * hop + nfft)
    den = np.zeros_like(out)
    for i in range(frames):
        out[i * hop : i * hop + nfft] += samples[:, i]
        den[i * hop : i * hop + nfft] += window**2
    expected = out[512:-512] / den[512:-512]
    assert len(actual) == (frames - 1) * hop
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-5)


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.update(trust_remote_code=True),
        lambda c: c["model"].update(ar_blocks=1000),
        lambda c: c["model"].update(ar_heads=3),
        lambda c: c["model"].update(acoustic_dit_dim=33),
        lambda c: c["model"].update(acoustic_dit_dim_head=3),
        lambda c: c["model"].update(acoustic_time_embed_dim=True),
        lambda c: c["model"].update(acoustic_mel_std=[0] * 100),
        lambda c: c["model"].update(acoustic_mel_mean=[float("nan")] * 100),
        lambda c: c["model"].update(acoustic_sway_sampling_coef=-1.0),
        lambda c: c["semantic_encoder"].update(fsq_levels=[7, 5]),
        lambda c: c["speaker_encoder"].update(stage_channels=[8, 12]),
        lambda c: c["speaker_encoder"].update(blocks_per_stage=[True] * 3),
        lambda c: c["vocoder"].update(block_lookaheads=[3]),
        lambda c: c["generation"].update(max_seconds=999),
    ],
)
def test_closed_metadata_rejects_unreviewed_behavior(change):
    c = small_config()
    change(c)
    with pytest.raises(ValueError):
        s.validate_config(c)


@pytest.mark.parametrize("component", s.COMPONENTS)
def test_each_component_is_mandatory(tmp_path, component):
    c, _ = fixture(tmp_path)
    (tmp_path / (component + ".safetensors")).unlink()
    with pytest.raises(ValueError, match="COMPONENTS_INCOMPLETE"):
        s.load_components(tmp_path, c, "cpu")


@pytest.mark.parametrize(
    "corruption", ["missing", "extra", "shape", "nan", "integer", "symlink", "index", "extra-file"]
)
def test_weight_rejections(tmp_path, corruption):
    from safetensors.torch import load_file

    c, _ = fixture(tmp_path)
    path = tmp_path / "vocoder.safetensors"
    state = load_file(path)
    key = next(iter(state))
    if corruption == "missing":
        del state[key]
    elif corruption == "extra":
        state["unexpected"] = torch.ones(1)
    elif corruption == "shape":
        state[key] = torch.ones(3)
    elif corruption == "nan":
        state[key].flatten()[0] = float("nan")
    elif corruption == "integer":
        state[key] = state[key].long()
    elif corruption == "symlink":
        path.rename(tmp_path / "hidden.weights")
        path.symlink_to(tmp_path / "hidden.weights")
    elif corruption == "index":
        (tmp_path / "model.safetensors.index.json").write_text("{}")
    elif corruption == "extra-file":
        save_file({"extra": torch.ones(1)}, tmp_path / "extra.safetensors")
    if corruption in {"missing", "extra", "shape", "nan", "integer"}:
        save_file(state, path)
    with pytest.raises(ValueError):
        s.load_components(tmp_path, c, "cpu")


@pytest.mark.parametrize(
    "text,language", [("", "en"), ("x" * 301, "en"), ("<|lang_fr|> Hi", "en"), ("hello", "zz")]
)
def test_text_controls_and_bounds(text, language):
    tok = s.tokenizer(FIXTURES, s.normalized_config(RAW, s.TASK))
    with pytest.raises(ValueError):
        s.text_ids(tok, text, language)


def write_audio(path, rate=24000, seconds=1, channels=1):
    pcm = np.round(np.sin(np.arange(int(rate * seconds)) * 0.11) * 6000).astype("<i2")
    if channels == 2:
        pcm = np.stack([pcm, pcm], axis=1)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.tobytes())


@pytest.mark.parametrize(
    "rate,channels", [(8000, 1), (16000, 2), (24000, 1), (32000, 1), (44100, 2), (48000, 1)]
)
def test_reference_preserves_rate_for_native_resampler(tmp_path, rate, channels):
    path = tmp_path / "ref.wav"
    write_audio(path, rate, channels=channels)
    wav, actual = s.read_reference(path)
    assert actual == rate and wav.shape == (rate,)


@pytest.mark.parametrize("seconds", [0.1, 10.001])
def test_reference_duration_rejected(tmp_path, seconds):
    path = tmp_path / "ref.wav"
    write_audio(path, seconds=seconds)
    with pytest.raises(ValueError, match="REFERENCE_FORMAT"):
        s.read_reference(path)


def test_generation_requires_eos(tmp_path, monkeypatch):
    import oryn_engine.native_sopro.sampling as sampling

    c, _ = fixture(tmp_path)
    models, tok, _ = s.load_components(tmp_path, c, "cpu")
    monkeypatch.setattr(sampling, "sample_next_token", lambda *a: torch.tensor([0]))
    # Bounded shortened loop retains the production termination rule.
    monkeypatch.setattr(s, "MAX_SECONDS", 0.1)
    with torch.inference_mode(), pytest.raises(ValueError, match="WITHOUT_EOS"):
        s.semantic_tokens(
            models["model"],
            torch.tensor([[1, 2]]),
            torch.tensor([[3]]),
            s.validate_config(c).generation,
        )


def test_invalid_logits_and_hidden_values_rejected(tmp_path):
    c, _ = fixture(tmp_path)
    models, tok, _ = s.load_components(tmp_path, c, "cpu")
    with torch.no_grad():
        models["semantic_encoder"].digit_head.weight.fill_(float("nan"))
    with torch.inference_mode(), pytest.raises(ValueError, match="NONFINITE"):
        s.synthesize(
            models, tok, c, "hello", "en", torch.sin(torch.arange(16000) * 0.1) * 0.2, 16000, "cpu"
        )


@pytest.mark.parametrize("missing", [None, *s.WEIGHTS, "tokenizer.model"])
def test_discovery_selects_complete_original_subset(missing):
    from oryn_engine.model_layouts import assess_layout

    names = [
        *s.WEIGHTS,
        "config.json",
        "tokenizer.model",
        "modeling_sopro.py",
        "README.md",
        "alternate.bin",
    ]
    if missing:
        names.remove(missing)
    result = assess_layout(names, RAW, s.TASK)
    if missing:
        assert "MODEL_SUBSET_INCOMPLETE" in result.artifact_reasons
    else:
        assert not result.artifact_reasons and not result.runtime_reasons
        assert set(result.selected_files) == {*s.WEIGHTS, "config.json", "tokenizer.model"}


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_offline_worker_emits_bounded_native_wav(tmp_path, backend):
    import os
    import subprocess
    import sys

    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    fixture(tmp_path)
    path = tmp_path / "reference.wav"
    write_audio(path)
    output = tmp_path / "result.wav"
    value = {
        "text": "Hello.",
        "voice": "reference",
        "language": "en",
        "speed": 1,
        "audioArtifactId": "artifact_" + "a" * 24,
    }
    request = dict(
        abi="1",
        packId="oryn.transformers-pytorch",
        version=builtin_registry().active_pack("oryn.transformers-pytorch").version,
        command="generate",
        payload=dict(
            modelRoot=str(tmp_path),
            task=s.TASK,
            backend=backend,
            input=value,
            inputPath=str(path),
            outputPath=str(output),
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
        result = json.loads(run.stdout.splitlines()[-1])
        assert result["ok"], (result, run.stderr)
        metadata = result["result"]
        assert metadata["frames"] == 9 * 1024 - 256
        with wave.open(str(output), "rb") as f:
            assert (f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getnframes()) == (
                24000,
                1,
                2,
                metadata["frames"],
            )
            assert any(f.readframes(f.getnframes()))
    request["payload"]["input"].pop("audioArtifactId")
    run = subprocess.run(
        [sys.executable, "-m", "oryn_engine.runtime_packs.transformers_worker"],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert not json.loads(run.stdout.splitlines()[-1])["ok"]


@pytest.mark.parametrize("reference", [False, True])
def test_metadata_requirements_preserve_fixed_voice_models(reference):
    from oryn_engine.metadata_audit import requirements_for

    requirements = requirements_for(s.TASK, RAW if reference else {"model_type": "qwen3_tts"})
    assert requirements["inputModalities"] == (["text", "audio"] if reference else ["text"])
    assert requirements.get("requiredFeatures", []) == (["referenceAudio"] if reference else [])
