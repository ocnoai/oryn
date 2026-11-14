"""Original header geometry and complete tiny graphs; no original inference."""

import copy
import json
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from oryn_engine import omnivoice_tasks as o

FIXTURES = Path(__file__).parent / "fixtures/omnivoice"
RAW = json.loads((o.DATA / "main_config.json").read_text())
CODEC = json.loads((o.DATA / "codec_config.json").read_text())


def small_configs():
    c, a = copy.deepcopy(RAW), copy.deepcopy(CODEC)
    c.update(pad_token_id=0, eos_token_id=1)
    c["llm_config"].update(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=1035,
        max_window_layers=2,
        layer_types=["full_attention"] * 2,
        bos_token_id=0,
        eos_token_id=1,
    )
    a["acoustic_model_config"].update(encoder_hidden_size=2, decoder_hidden_size=32, hidden_size=8)
    a["semantic_model_config"].update(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_conv_pos_embeddings=4,
        num_conv_pos_embedding_groups=2,
        conv_dim=[8] * 7,
    )
    return c, a


def fixture(root, precision="float32"):
    from tokenizers import Tokenizer, models, pre_tokenizers

    root.mkdir(exist_ok=True)
    (root / "audio_tokenizer").mkdir(exist_ok=True)
    c, a = small_configs()
    (root / "config.json").write_text(json.dumps(c))
    (root / "audio_tokenizer/config.json").write_text(json.dumps(a))
    shutil.copyfile(
        o.DATA / "codec_preprocessor.json", root / "audio_tokenizer/preprocessor_config.json"
    )
    shutil.copyfile(o.DATA / "tokenizer_config.json", root / "tokenizer_config.json")
    words = [
        "<|endoftext|>",
        "<|im_end|>",
        "[UNK]",
        *o.CONTROLS,
        "hello",
        "en",
        "None",
        "Nice",
        "to",
        "meet",
        "you",
        ".",
    ]
    words += [f"item{x}" for x in range(1035 - len(words))]
    tok = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.add_special_tokens(list(o.CONTROLS))
    tok.save(str(root / "tokenizer.json"))
    torch.manual_seed(6173)
    result = o.native_components(c, a, tokenizer=o.load_tokenizer(root, c))
    # Real random learned codebooks, all 8 residual quantizers, both full graphs.
    # Native initialization makes codebook buffers zero, which is unsuitable for
    # a synthetic acoustic signal; populate every learned embedding independently.
    with torch.no_grad():
        for q in result["audio_tokenizer/model"].quantizer.quantizers:
            q.codebook.embed.normal_(std=0.05)
            q.codebook.embed_avg.copy_(q.codebook.embed)
    for name, model in result.items():
        state = {
            k: (v.to(getattr(torch, precision)) if v.is_floating_point() else v).contiguous()
            for k, v in model.state_dict().items()
        }
        save_file(state, root / (name + ".safetensors"))
        model.load_state_dict(
            {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
        )
    return c, a, result


def test_original_complete_geometry():
    shapes = {
        name: {
            k: v["shape"]
            for k, v in json.loads(
                (FIXTURES / (name + ".safetensors.header.json")).read_text()
            ).items()
            if k != "__metadata__"
        }
        for name in ("model", "audio_tokenizer/model")
    }
    o.validate_weight_shapes(RAW, CODEC, shapes)
    assert [len(v) for v in shapes.values()] == [313, 527]
    assert len(o.LANGUAGES) >= 600


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("precision", ["float32", "float16"])
def test_complete_speech_graph_and_loaded_parity(tmp_path, backend, precision):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    c, a, reference = fixture(tmp_path, precision)
    loaded, tok, _ = o.load_components(tmp_path, c, backend)
    reference["model"].to(backend)
    with torch.inference_mode():
        torch.manual_seed(6173)
        expected = o.synthesize(reference, reference["model"].text_tokenizer, "hello", "en")
        torch.manual_seed(6173)
        actual = o.synthesize(loaded, tok, "hello", "en")
    assert actual.shape == (
        o.generation_task(loaded["model"], tok, "hello", "en").target_lens[0] * 960 + 4800,
    )
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    assert np.max(np.abs(actual)) > 0.01
    assert not actual[:2400].any() and not actual[-2400:].any()
    assert next(loaded["audio_tokenizer/model"].parameters()).device.type == "cpu"


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_native_sdpa_matches_bidirectional_eager_and_future_context(tmp_path, backend):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    _, _, models = fixture(tmp_path)
    model = models["model"].to(backend)
    eager = copy.deepcopy(model)
    eager.llm.set_attn_implementation("eager")
    ids = torch.full((2, 8, 7), 1024, dtype=torch.long, device=backend)
    ids[0, :, :3] = torch.tensor([10, 11, 12], device=backend)
    audio = torch.zeros(2, 7, dtype=torch.bool, device=backend)
    audio[0, 3:] = True
    audio[1, :4] = True
    mask = torch.zeros(2, 1, 7, 7, dtype=torch.bool, device=backend)
    mask[0] = True
    mask[1, :, :4, :4] = True
    mask[1, :, torch.arange(4, 7), torch.arange(4, 7)] = True
    with torch.inference_mode():
        actual = model(ids, audio, mask).logits
        expected = eager(ids, audio, mask).logits
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        ids[0, :, -1] = 17
        changed = model(ids, audio, mask).logits
    assert float((actual[0, :, 0] - changed[0, :, 0]).abs().max()) > 1e-4
    torch.testing.assert_close(actual[1], changed[1])


def test_guidance_scores_match_numpy_and_exclude_mask(tmp_path):
    from oryn_engine.native_omnivoice.model import OmniVoiceGenerationConfig

    _, _, models = fixture(tmp_path)
    rng = np.random.default_rng(6173)
    c = rng.normal(size=(1, 8, 3, 1025)).astype(np.float32)
    u = rng.normal(size=c.shape).astype(np.float32)
    c[..., 1024] = 9

    def logsoftmax(x):
        shifted = x.astype(np.float64) - x.max(-1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))

    cg, ug = logsoftmax(c), logsoftmax(u)
    expected = logsoftmax(cg + 2 * (cg - ug))
    expected[..., 1024] = -np.inf
    tokens, scores = models["model"]._predict_tokens_with_scoring(
        torch.tensor(c), torch.tensor(u), OmniVoiceGenerationConfig()
    )
    np.testing.assert_array_equal(tokens.numpy(), expected.argmax(-1))
    np.testing.assert_allclose(scores.numpy(), expected.max(-1), atol=5e-6, rtol=5e-6)
    assert tokens.max() < 1024


def test_all_eight_codebooks_decode_by_independent_lookup(tmp_path):
    _, _, models = fixture(tmp_path)
    codec = models["audio_tokenizer/model"]
    codes = torch.arange(32).reshape(8, 1, 4)
    with torch.inference_mode():
        quantized = codec.quantizer.decode(codes)
        expected = sum(
            torch.tensor(
                q.codebook.embed.detach().numpy()[codes[i].numpy()]
                @ q.project_out.weight.detach().numpy().T
                + q.project_out.bias.detach().numpy()
            ).transpose(1, 2)
            for i, q in enumerate(codec.quantizer.quantizers)
        )
        torch.testing.assert_close(quantized, expected, atol=2e-6, rtol=2e-6)
        direct = codec.acoustic_decoder(codec.fc2(expected.transpose(1, 2)).transpose(1, 2))
        native = codec.decode(codes.transpose(0, 1)).audio_values
    torch.testing.assert_close(native, direct, atol=2e-6, rtol=2e-6)
    assert native.shape == (1, 1, 4 * 960)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_type", "qwen3"),
        ("architectures", ["AutoModel"]),
        ("num_audio_codebook", 7),
        ("audio_mask_id", 0),
        ("audio_vocab_size", 2049),
        ("auto_map", {"AutoModel": "evil.Model"}),
        ("quantization_config", {"quant_method": "gptq"}),
        ("pad_token_id", 151645),
    ],
)
def test_closed_main_config(field, value):
    c = copy.deepcopy(RAW)
    c[field] = value
    with pytest.raises(ValueError):
        o.validate_config(c)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rope_parameters", {"rope_type": "linear", "factor": 2}),
        ("hidden_size", True),
        ("head_dim", 9),
        ("num_key_value_heads", 3),
        ("layer_types", ["sliding_attention"] * 28),
        ("vocab_size", 1024),
        ("attention_bias", True),
        ("use_sliding_window", True),
    ],
)
def test_closed_qwen_geometry(field, value):
    c = copy.deepcopy(RAW)
    c["llm_config"][field] = value
    with pytest.raises(ValueError):
        o.validate_config(c)


@pytest.mark.parametrize(
    "section,field,value",
    [
        (None, "sample_rate", 16000),
        (None, "codebook_size", 512),
        (None, "target_bandwidths", [0.5, 1, 2, 4]),
        ("acoustic_model_config", "model_type", "encodec"),
        ("acoustic_model_config", "upsampling_ratios", [8, 8, 4, 2]),
        ("acoustic_model_config", "decoder_hidden_size", 33),
        ("semantic_model_config", "conv_dim", [8] * 6),
        ("semantic_model_config", "do_stable_layer_norm", True),
        ("semantic_model_config", "num_attention_heads", 5),
    ],
)
def test_closed_codec_metadata(section, field, value):
    c = copy.deepcopy(CODEC)
    (c[section] if section else c)[field] = value
    with pytest.raises(ValueError):
        o.validate_codec(c)


@pytest.mark.parametrize("missing", [None, *o.FILES])
def test_discovery_requires_complete_original_export(missing):
    from oryn_engine.model_layouts import assess_layout

    files = [*o.FILES, "modeling_omnivoice.py", "pytorch_model.bin", "README.md"]
    if missing:
        files.remove(missing)
    result = assess_layout(files, RAW, o.TASK)
    if missing:
        assert result.artifact_reasons
    else:
        assert not result.artifact_reasons and not result.runtime_reasons
        assert set(result.selected_files) == set(o.FILES)


@pytest.mark.parametrize(
    "corruption",
    [
        "missing",
        "extra",
        "shape",
        "integer",
        "nonfinite",
        "offsets",
        "offset_dtype",
        "uninitialized",
    ],
)
def test_complete_weights_fail_closed(tmp_path, corruption):
    from safetensors.torch import load_file

    c, _, _ = fixture(tmp_path)
    path = tmp_path / (
        "model.safetensors"
        if corruption.startswith("offset")
        else "audio_tokenizer/model.safetensors"
    )
    state = load_file(path)
    key = "fc2.weight"
    if corruption == "missing":
        del state[key]
    elif corruption == "extra":
        state["invented.weight"] = torch.ones(1)
    elif corruption == "shape":
        state[key] = state[key][:-1]
    elif corruption == "integer":
        state[key] = state[key].to(torch.int32)
    elif corruption == "nonfinite":
        state[key][0, 0] = float("nan")
    elif corruption == "offsets":
        state["codebook_layer_offsets"][2] += 1
    elif corruption == "offset_dtype":
        state["codebook_layer_offsets"] = state["codebook_layer_offsets"].to(torch.int32)
    else:
        state["quantizer.quantizers.7.codebook.inited"].zero_()
    save_file(state, path)
    with pytest.raises(ValueError, match="OMNIVOICE"):
        o.load_components(tmp_path, c, "cpu")


@pytest.mark.parametrize(
    "text,language",
    [
        ("", "en"),
        ("x" * 301, "en"),
        ("<|text_end|>", "en"),
        ("hello", "not-a-language"),
        ("界" * 300, "zh"),
    ],
)
def test_text_controls_and_duration_are_bounded(tmp_path, text, language):
    _, _, models = fixture(tmp_path)
    with pytest.raises(ValueError, match="OMNIVOICE"):
        o.generation_task(models["model"], models["model"].text_tokenizer, text, language)


def test_profile_uses_auto_voice_and_preserves_old_fixed_voice_requirements():
    from oryn_protocol.task_contracts import effective_input, validate_invocation

    from oryn_engine.metadata_audit import requirements_for
    from oryn_engine.task_profiles import profile_for_execution

    profile = profile_for_execution(
        {
            "runtimePack": {
                "packId": "oryn.transformers-pytorch",
                "taskContract": "speech-synthesis-v1",
            }
        },
        RAW,
    )
    assert profile["inputModalities"] == ["text"]
    assert profile["features"]["referenceAudio"] == "unsupported"
    value = effective_input(profile, {"text": "hello"})
    assert (value["voice"], value["language"], value["speed"]) == ("auto", "en", 1)
    validate_invocation(profile, value)
    for changed in [
        {"voice": "reference"},
        {"speed": 1.1},
        {"language": "fake"},
        {"audioArtifactId": "artifact_" + "a" * 24},
        {"text": "x" * 301},
    ]:
        with pytest.raises(ValueError):
            validate_invocation(profile, {**value, **changed})
    assert requirements_for(o.TASK, RAW)["inputModalities"] == ["text"]
    assert requirements_for(o.TASK, {"model_type": "qwen3_tts"})["inputModalities"] == ["text"]


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_offline_worker_emits_speech_without_reference(tmp_path, backend):
    import os
    import subprocess
    import sys

    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    fixture(tmp_path)
    output = tmp_path / "result.wav"
    request = dict(
        abi="1",
        packId="oryn.transformers-pytorch",
        version=builtin_registry().active_pack("oryn.transformers-pytorch").version,
        command="generate",
        payload=dict(
            modelRoot=str(tmp_path),
            task=o.TASK,
            backend=backend,
            input={"text": "hello", "voice": "auto", "language": "en", "speed": 1},
            inputPath=None,
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
        with wave.open(str(output), "rb") as f:
            assert (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (24000, 1, 2)
            assert 4800 < f.getnframes() <= 720000
            assert f.getnframes() == result["result"]["frames"]
            assert any(f.readframes(f.getnframes()))
    request["payload"]["input"]["audioArtifactId"] = "artifact_" + "a" * 24
    run = subprocess.run(
        [sys.executable, "-m", "oryn_engine.runtime_packs.transformers_worker"],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert not json.loads(run.stdout.splitlines()[-1])["ok"]


@pytest.mark.parametrize("frames", [4, 7])
def test_complete_codec_encoder_and_decoder_preserve_native_clock(tmp_path, frames):
    _, _, models = fixture(tmp_path)
    codec = models["audio_tokenizer/model"]
    signal = torch.sin(torch.arange(frames * 960) * 0.13).view(1, 1, -1) * 0.2
    with torch.inference_mode():
        codes = codec.encode(signal).audio_codes
        audio = codec.decode(codes).audio_values
    assert codes.shape == (1, 8, frames)
    assert bool(((codes >= 0) & (codes < 1024)).all())
    assert audio.shape == signal.shape and torch.isfinite(audio).all()


@pytest.mark.parametrize("component", ["audio_embeddings", "audio_heads", "codec"])
def test_nonfinite_intermediate_values_rejected(tmp_path, component):
    c, _, _ = fixture(tmp_path)
    models, tok, _ = o.load_components(tmp_path, c, "cpu")
    with torch.no_grad():
        if component == "codec":
            models["audio_tokenizer/model"].fc2.weight.fill_(float("nan"))
        else:
            getattr(models["model"], component).weight.fill_(float("nan"))
    with torch.inference_mode(), pytest.raises(ValueError, match="NONFINITE"):
        o.synthesize(models, tok, "hello", "en")


@pytest.mark.parametrize(
    "name",
    ["model.safetensors", "audio_tokenizer/model.safetensors", "tokenizer.json", "audio_tokenizer"],
)
def test_component_symlinks_rejected(tmp_path, name):
    c, _, _ = fixture(tmp_path)
    path = tmp_path / name
    moved = tmp_path / "outside"
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=moved.is_dir())
    with pytest.raises(ValueError, match="OMNIVOICE"):
        o.load_components(tmp_path, c, "cpu")
