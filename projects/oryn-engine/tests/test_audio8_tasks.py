"""Complete native graphs, original tensor metadata and bounded streaming ASR."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from oryn_engine import audio8_tasks as adapter

FIXTURES = Path(__file__).parent / "fixtures/audio8"


def original():
    return adapter.normalized_config(
        json.loads((FIXTURES / "config.json").read_text()), adapter.TASK
    )


def native_fixture(root):
    from tokenizers import Tokenizer, decoders, models

    c = original()
    c.update(
        hidden_size=32,
        vocab_size=32,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        projection_size=256,
    )
    c["audio_config"].update(
        hidden_size=32,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=16,
    )
    c["text_config"].update(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=32,
        projection_size=256,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        layer_types=["full_attention"] * 2,
        max_position_embeddings=512,
    )
    words = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", *adapter.SPECIAL, "hello", "world"]
    words += [f"word{i}" for i in range(32 - len(words))]
    tok = Tokenizer(models.BPE(vocab={s: i for i, s in enumerate(words)}, merges=[]))
    tok.decoder = decoders.Fuse()
    tok.add_special_tokens(words[:7])
    tok.save(str(root / "tokenizer.json"))
    for name in adapter.METADATA_NAMES:
        (root / name).write_bytes((FIXTURES / name).read_bytes())
    (root / "config.json").write_text(json.dumps(c))
    torch.manual_seed(61808)
    model = adapter.native_model(c).eval()
    weights = {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if k != "language_model.lm_head.weight"
    }
    write_weights(root, weights)
    return model, c, weights


def write_weights(root, weights):
    from safetensors.torch import save_file

    main = {k: v for k, v in weights.items() if not k.startswith("semantic_vad_heads.")}
    aux = {k: v for k, v in weights.items() if k.startswith("semantic_vad_heads.")}
    save_file(main, str(root / "model.safetensors"))
    save_file(aux, str(root / "semantic_vad_heads.safetensors"))
    index = {
        "metadata": {"total_size": sum(v.numel() * v.element_size() for v in weights.values())},
        "weight_map": {
            k: "semantic_vad_heads.safetensors" if k in aux else "model.safetensors"
            for k in weights
        },
    }
    (root / "model.safetensors.index.json").write_text(json.dumps(index))


def test_original_complete_tensor_graph_without_payload():
    from accelerate import init_empty_weights

    c = original()
    with init_empty_weights(include_buffers=False):
        model = adapter.native_model(c)
    shapes = {
        k: list(v.shape)
        for k, v in model.state_dict().items()
        if k != "language_model.lm_head.weight"
    }
    expected = {
        k: v["shape"]
        for name in ["model.safetensors.header.json", "semantic_vad_heads.safetensors.header.json"]
        for k, v in json.loads((FIXTURES / name).read_text()).items()
        if k != "__metadata__"
    }
    assert len(shapes) == 938
    assert shapes == expected
    assert not model.time_embedding.inv_freq.is_meta
    assert torch.isfinite(model.time_embedding.inv_freq).all()


@pytest.mark.parametrize(
    "section,change",
    [
        (None, {"auto_map": {"AutoModel": "untrusted.Model"}}),
        (None, {"max_frame_len": 4}),
        (None, {"frame_lens": [8]}),
        (None, {"use_frame_len_embedding": False}),
        (None, {"weight_format_version": 1}),
        (None, {"semantic_vad_horizons_seconds": []}),
        (None, {"projection_size": 1}),
        (None, {"quantization_config": {"bits": 4}}),
        (None, {"custom_flag": True}),
        (None, {"bos_token_id": 0}),
        (None, {"default_num_delay_tokens": 3}),
        ("audio_config", {"num_mel_bins": 80}),
        ("audio_config", {"sliding_window": 0}),
        ("audio_config", {"head_dim": 15}),
        ("audio_config", {"num_hidden_layers": 0}),
        ("text_config", {"num_key_value_heads": 3}),
        ("text_config", {"tie_word_embeddings": False}),
        ("text_config", {"layer_types": ["sliding_attention"]}),
        ("text_config", {"model_type": "qwen3"}),
        ("text_config", {"rope_parameters": {"rope_type": "dynamic", "rope_theta": 10000}}),
    ],
)
def test_closed_configuration(section, change):
    c = original()
    (c if section is None else c[section]).update(change)
    with pytest.raises(ValueError):
        adapter.normalized_config(c, adapter.TASK)


@pytest.mark.parametrize(
    "name,change",
    [
        ("preprocessor_config.json", {"global_log_mel_max": None}),
        ("preprocessor_config.json", {"sampling_rate": 8000}),
        ("preprocessor_config.json", {"n_fft": 512}),
        ("tokenizer_config.json", {"extra_special_tokens": []}),
        ("tokenizer_config.json", {"tokenizer_class": "CustomTokenizer"}),
        ("tokenizer_config.json", {"bos_token": "unbound"}),
        ("tokenizer_config.json", {"chat_template": "ignored?"}),
    ],
)
def test_processor_contract(name, change):
    m = json.loads((FIXTURES / name).read_text())
    m.update(change)
    with pytest.raises(ValueError):
        adapter.validate_metadata(name, m, original())


def independent_reference(model, processor, c, audio, language):
    """Native full-prefix towers; adapt native Qwen2 with post-norm modulation hooks.

    No production streaming/cache loop or custom decoder layer is used. Window
    boundaries follow the pinned publisher definition; each layer/head is loaded.
    """
    from torch.nn import functional as F
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from transformers.models.voxtral_realtime.configuration_voxtral_realtime import (
        VoxtralRealtimeEncoderConfig,
    )
    from transformers.models.voxtral_realtime.modeling_voxtral_realtime import (
        VoxtralRealtimeEncoder,
    )

    ac = VoxtralRealtimeEncoderConfig(**c["audio_config"])
    ac._attn_implementation = "sdpa"
    tc = Qwen2Config(**c["text_config"])
    tc._attn_implementation = "sdpa"
    encoder = VoxtralRealtimeEncoder(ac).eval()
    text = Qwen2ForCausalLM(tc).eval()
    encoder.load_state_dict(model.audio_tower.state_dict(), strict=True)
    text.load_state_dict(
        {k: v for k, v in model.language_model.state_dict().items() if ".ada_rms_norm." not in k},
        strict=True,
    )
    cond = (
        torch.cat(
            (
                torch.cos(6 * torch.exp(-np.log(10000) * torch.arange(16) / 16)),
                torch.sin(6 * torch.exp(-np.log(10000) * torch.arange(16) / 16)),
            )
        )
        + model.frame_len_embedding.weight[0]
    )
    hooks = []
    for native, owned in zip(text.model.layers, model.language_model.model.layers, strict=True):
        modulation = owned.ada_rms_norm(cond[None, None, :]).detach()
        hooks.append(
            native.post_attention_layernorm.register_forward_hook(
                lambda _m, _args, out, mod=modulation: out * (1 + mod)
            )
        )
    feature, tok = processor
    vocab = tok.get_vocab()
    generated = []
    traces = []
    ids = [c["bos_token_id"], vocab["[LANGUAGE_" + language.upper() + "]"]] + [
        vocab["[STREAMING_PAD]"]
    ] * 23
    stream = np.pad(audio, (23040, 21760))
    conv_chunks = []
    projected_chunks = []
    all_ids = []
    try:
        with torch.inference_mode():
            stop = 32000
            start = 0
            while stop + 40 <= len(stream):
                count = (stop - start) // 1280
                window = stream[max(0, start - 840) : stop + 40].astype(np.float32)
                features = feature._torch_extract_fbank_features(
                    torch.from_numpy(window).unsqueeze(0)
                )
                conv_chunks.append(encoder.embedder(features)[:, -count * 4 :])
                full = encoder(
                    inputs_embeds=torch.cat(conv_chunks, dim=1), use_cache=False
                ).last_hidden_state
                group = F.pad(full[:, -count * 4 :].reshape(1, count, 4, 32), (0, 0, 0, 4)).flatten(
                    2
                )
                p = F.linear(
                    F.gelu(F.linear(group, model.multi_modal_projector.linear_1.weight)),
                    model.multi_modal_projector.linear_2.weight,
                )
                projected_chunks.append(p)
                all_ids.extend(ids)
                embedding = text.model.embed_tokens(torch.tensor([all_ids])) + torch.cat(
                    projected_chunks, dim=1
                )
                logits = (
                    text(inputs_embeds=embedding, t_cond=None, use_cache=False)
                    .logits[0, -1]
                    .clone()
                )
                traces.append((full[:, -count * 4 :].clone(), p.clone(), logits.clone()))
                logits[c["eos_token_id"]] = -torch.inf
                token = int(logits.argmax())
                generated.append(token)
                ids = [token]
                start, stop = stop, stop + 1280
    finally:
        for hook in hooks:
            hook.remove()
    return (
        {
            "text": tok.decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip(),
            "language": language,
        },
        generated,
        traces,
    )


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("length,language", [(320, "en"), (1280, "zh"), (4800, "en")])
def test_complete_native_cached_and_independent_prefix_parity(tmp_path, backend, length, language):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    reference, c, weights = native_fixture(tmp_path)
    # Original tied head has no separate persisted tensor.
    reference.language_model.lm_head.weight = reference.language_model.model.embed_tokens.weight
    processor = adapter.components(tmp_path, c)
    audio = np.sin(np.arange(length, dtype=np.float32) * 0.071) * 0.2
    expected, ids, traces = independent_reference(reference, processor, c, audio, language)
    model, processor, _ = adapter.load_components(tmp_path, c, backend)
    assert model.language_model.lm_head.weight is model.language_model.model.embed_tokens.weight
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    actual_encoder = []
    actual_projector = []
    actual_logits = []
    hooks = [
        model.audio_tower.register_forward_hook(
            lambda _m, _a, out: actual_encoder.append(out.last_hidden_state.detach().cpu())
        ),
        model.multi_modal_projector.register_forward_hook(
            lambda _m, _a, out: actual_projector.append(out.detach().cpu())
        ),
        model.language_model.register_forward_hook(
            lambda _m, _a, out: actual_logits.append(out.logits[0, -1].detach().cpu())
        ),
    ]
    try:
        assert adapter.transcribe(model, processor, c, audio, backend, language) == expected
    finally:
        for hook in hooks:
            hook.remove()
    assert len(traces) == len(actual_encoder) == len(actual_projector) == len(actual_logits)
    for (enc, proj, logits), ae, ap, al in zip(
        traces, actual_encoder, actual_projector, actual_logits, strict=True
    ):
        torch.testing.assert_close(ae, enc, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(ap, proj, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(al, logits, atol=3e-5, rtol=3e-5)
    assert len(ids) == (length - 40) // 1280 + 11
    # Delay and frame embeddings must both affect the complete language graph.
    assert torch.isfinite(model.condition()).all()
    assert len(model.semantic_vad_heads) == 4


@pytest.mark.parametrize("length", [320, 1280, 1320, 4800, 480000])
def test_clock_windows_and_full_thirty_second_bound(length):
    wave = np.arange(length, dtype=np.float32)
    windows = list(adapter.audio_windows(wave))
    assert len(windows) == (length - 40) // 1280 + 11
    assert windows[0][1] == 25 and len(windows[0][0]) == 32040
    assert all(count == 1 and len(window) == 2160 for window, count in windows[1:])
    stream = np.pad(wave, (18 * 1280, 17 * 1280))
    for i, (window, _count) in enumerate(windows):
        start = 0 if i == 0 else (24 + i) * 1280
        end = (25 + i) * 1280
        np.testing.assert_array_equal(window, stream[max(0, start - 840) : end + 40])
    assert sum(n for _, n in windows) <= 410


@pytest.mark.parametrize("length", [2160, 32040, 1320])
def test_frontend_independent_numpy_formula(tmp_path, length):
    import librosa

    _, c, _ = native_fixture(tmp_path)
    feature, _ = adapter.components(tmp_path, c)
    rng = np.random.default_rng(618)
    wave = rng.normal(0, 0.1, length).astype(np.float32)
    actual = feature._torch_extract_fbank_features(torch.from_numpy(wave).unsqueeze(0))[0].numpy()
    padded = np.pad(wave, (200, 200), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 400)[::160][:-1]
    hann = np.hanning(401)[:-1]
    power = np.abs(np.fft.rfft(windows * hann, axis=-1)) ** 2
    mel = librosa.filters.mel(sr=16000, n_fft=400, n_mels=128, norm="slaney")
    expected = (np.maximum(np.log10(np.maximum(power @ mel.T, 1e-10)), -6.5) + 4) / 4
    np.testing.assert_allclose(actual.T, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "damage", ["missing", "extra", "shape", "nan", "int", "index", "auxiliary"]
)
def test_complete_numeric_loading_rejects_damage(tmp_path, damage):
    _, c, w = native_fixture(tmp_path)
    key = "audio_tower.embedder.conv1.weight"
    if damage == "missing":
        w.pop(key)
    elif damage == "extra":
        w["extra.weight"] = torch.ones(1)
    elif damage == "shape":
        w[key] = w[key][:1]
    elif damage == "nan":
        w[key].flatten()[0] = torch.nan
    elif damage == "int":
        w[key] = w[key].to(torch.int32)
    elif damage == "auxiliary":
        w.pop("semantic_vad_heads.0.weight")
    write_weights(tmp_path, w)
    if damage == "index":
        p = tmp_path / "model.safetensors.index.json"
        d = json.loads(p.read_text())
        d["weight_map"][key] = "missing.safetensors"
        p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        adapter.load_components(tmp_path, c, "cpu")


@pytest.mark.parametrize("value", [{"language": "auto"}, {"language": "de"}, {"timestamps": True}])
def test_unbound_controls(value):
    with pytest.raises(ValueError, match="AUDIO8_PARAMETERS"):
        adapter.execute(None, None, original(), value, "cpu", None, lambda *_: None)


@pytest.mark.parametrize(
    "audio", [np.zeros(319), np.zeros(480001), np.zeros((2, 320)), np.array([np.nan] * 320)]
)
def test_audio_bound(audio):
    with pytest.raises(ValueError, match="AUDIO8_AUDIO_BOUND"):
        list(adapter.audio_windows(audio))


@pytest.mark.parametrize(
    "damage", [None, "frontend", "tokenizer", "missing-vad", "missing-index", "mapping"]
)
def test_discovery_and_profile(tmp_path, damage):
    from test_models import FakeHub, model

    from oryn_engine.models import ModelStore, ModelStoreError

    config = json.loads((FIXTURES / "config.json").read_text())
    if damage == "mapping":
        config["auto_map"]["AutoConfig"] = "foreign.Config"

    class Hub(FakeHub):
        def read_json(self, repo, revision, name):
            v = config if name == "config.json" else json.loads((FIXTURES / name).read_text())
            if damage == "frontend" and name == "preprocessor_config.json":
                v["global_log_mel_max"] = None
            if damage == "tokenizer" and name == "tokenizer_config.json":
                v["eos_token"] = "unbound"
            return v

    names = [
        "config.json",
        "model.safetensors",
        "semantic_vad_heads.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        *adapter.METADATA_NAMES,
    ]
    if damage == "missing-vad":
        names.remove("semantic_vad_heads.safetensors")
    if damage == "missing-index":
        names.remove("model.safetensors.index.json")
    hub = Hub(config=config, pipeline_tag=adapter.TASK, files=[(n, 100, "a" * 64) for n in names])
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda p: object() if p.pack_id == "oryn.transformers-pytorch" else None
    )
    if damage == "mapping":
        with pytest.raises(ModelStoreError) as failure:
            store.inspect(model("independent/audio-clock"))
        assert failure.value.code == "MODEL_METADATA_INVALID"
        assert not hub.downloaded
        return
    result = store.inspect(model("independent/audio-clock"))
    if damage:
        assert not result["executable"], result
    else:
        assert result["executable"], result
        profile = result["taskProfile"]
        assert profile["contract"]["id"] == "transcription-v1"
        assert profile["features"]["languageDetection"] == "unsupported"
        assert profile["features"]["wordTimestamps"] == "unsupported"
        assert profile["parameters"]["language"]["default"] == "en"
        assert profile["parameters"]["language"]["valueSchema"]["allOf"][-1] == {
            "enum": ["en", "zh"]
        }
        assert profile["limits"]["audioDurationSeconds"] == 30
    assert not hub.downloaded


@pytest.mark.parametrize("damage", ["missing-control", "sparse", "identity", "padded"])
def test_tokenizer_identities(tmp_path, damage):
    _, c, _ = native_fixture(tmp_path)
    p = tmp_path / "tokenizer.json"
    d = json.loads(p.read_text())
    if damage == "missing-control":
        d["model"]["vocab"]["unbound"] = d["model"]["vocab"].pop("[STREAMING_PAD]")
        d["added_tokens"] = [x for x in d["added_tokens"] if x["content"] != "[STREAMING_PAD]"]
    elif damage == "sparse":
        d["model"]["vocab"]["hello"] = 100
    elif damage == "identity":
        c["bos_token_id"] = 4
    elif damage == "padded":
        c["vocab_size"] = 1000
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        adapter.components(tmp_path, c)


def test_repeated_offline_worker_preserves_owned_clock(tmp_path):
    import os
    import subprocess
    import sys
    import wave

    from oryn_engine.runtime_packs import transformers_worker
    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry

    _, c, _ = native_fixture(tmp_path)
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"do_sample": True, "max_new_tokens": 1, "eos_token_id": 7})
    )
    audio = tmp_path / "input.wav"
    with wave.open(str(audio), "wb") as f:
        f.setparams((2, 2, 32000, 0, "NONE", "not compressed"))
        signal = (np.sin(np.arange(3200) * 0.04) * 8000).astype("<i2")
        f.writeframes(np.repeat(signal[:, None], 2, axis=1).tobytes())
    payload = {
        "modelRoot": str(tmp_path),
        "task": adapter.TASK,
        "backend": "cpu",
        "input": {"audioArtifactId": "artifact_" + "a" * 24, "language": "zh", "timestamps": False},
        "inputPath": str(audio),
        "outputPath": None,
    }
    expected = transformers_worker.generate(payload)
    assert expected["language"] == "zh"
    envelope = {
        "abi": "1",
        "packId": "oryn.transformers-pytorch",
        "version": builtin_registry().active_pack("oryn.transformers-pytorch").version,
    }
    for command in ("probe", "generate", "generate"):
        run = subprocess.run(
            [sys.executable, "-m", "oryn_engine.runtime_packs.transformers_worker"],
            input=json.dumps({**envelope, "command": command, "payload": payload}) + "\n",
            text=True,
            capture_output=True,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            timeout=60,
        )
        assert run.returncode == 0, run.stderr + run.stdout
        response = json.loads(run.stdout.strip().splitlines()[-1])
        assert response["ok"], response
        if command == "generate":
            assert response["result"] == expected


def test_complete_thirty_second_clock_execution(tmp_path):
    _, c, _ = native_fixture(tmp_path)
    model, processor, _ = adapter.load_components(tmp_path, c, "cpu")
    calls = []
    hook = model.language_model.register_forward_hook(
        lambda _m, _a, out: calls.append(out.past_key_values.get_seq_length())
    )
    try:
        result = adapter.transcribe(model, processor, c, np.zeros(480000, np.float32), "cpu", "en")
    finally:
        hook.remove()
    assert len(calls) == 385 and calls == list(range(25, 410))
    assert result["language"] == "en" and len(result["text"].encode()) <= 16384


@pytest.mark.parametrize("stage", ["frontend", "encoder", "logits", "padded-id"])
def test_invalid_native_runtime_outputs_fail(tmp_path, stage):
    _, c, _ = native_fixture(tmp_path)
    model, processor, _ = adapter.load_components(tmp_path, c, "cpu")

    def damage(_module, _args, out):
        if stage == "frontend":
            out[0, 0, 0] = torch.nan
        elif stage == "encoder":
            out.last_hidden_state[0, 0, 0] = torch.nan
        elif stage == "logits":
            out.logits.flatten()[0] = torch.nan
        else:
            out.logits.fill_(-10)
            out.logits[0, -1, -1] = 10
        return out

    if stage == "padded-id":
        # A native output row can exist without a tokenizer identity. It must
        # cause rejection, not a remapped/suppressed alternative greedy token.
        raw = tmp_path / "tokenizer.json"
        d = json.loads(raw.read_text())
        last = max(d["model"]["vocab"], key=d["model"]["vocab"].get)
        d["model"]["vocab"].pop(last)
        raw.write_text(json.dumps(d))
        processor = adapter.components(tmp_path, c)
    module = (
        model.audio_tower.embedder
        if stage == "frontend"
        else model.audio_tower
        if stage == "encoder"
        else model.language_model
    )
    hook = module.register_forward_hook(damage)
    try:
        with pytest.raises(ValueError, match="AUDIO8_"):
            adapter.transcribe(model, processor, c, np.zeros(320, np.float32), "cpu", "en")
    finally:
        hook.remove()
