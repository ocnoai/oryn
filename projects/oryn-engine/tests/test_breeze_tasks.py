"""Complete tiny Breeze graphs, original metadata, and closed inference bounds."""

import copy
import json
import shutil
import wave
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from oryn_engine import breeze_tasks as b

RAW = json.loads((b.DATA / "main_config.json").read_text())
CODEC = json.loads((b.DATA / "codec_config.json").read_text())
FIXTURES = Path(__file__).parent / "fixtures/breeze"


def small_configs():
    c, codec = copy.deepcopy(RAW), copy.deepcopy(CODEC)
    dims = dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    c.update(
        **dims, audio_embed_size=32, text_vocab_size=64, audio_token_id=50, audio_eos_token_id=51
    )
    c["backbone_config"].update(
        **dims, max_window_layers=2, vocab_size=64, bos_token_id=2, eos_token_id=1
    )
    c["depth_decoder_config"].update(**dims, audio_embed_size=32, backbone_hidden_size=32)
    c["text_encoder_config"].update(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        query_pre_attn_scalar=8,
        vocab_size=64,
        eoi_token_index=30,
        sliding_window=4,
        layer_types=["sliding_attention"] * 5 + ["full_attention"],
    )
    c["text_encoder_special_tokens_config"]["token_ids"] = {
        token: 52 + i for i, token in enumerate(b.SPECIAL_TOKENS)
    }
    mimi = dict(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=2,
        num_filters=2,
        codebook_dim=8,
        vector_quantization_hidden_dimension=8,
        upsample_groups=16,
    )
    c["codec_config"].update(**mimi)
    codec["encoder_config"].update(**mimi)
    codec["decoder_config"].update(
        latent_dim=32,
        codebook_dim=16,
        decoder_dim=32,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=2,
        vector_quantization_hidden_dimension=16,
    )
    return c, codec


def fixture(root, precision="float32"):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors

    root.mkdir(exist_ok=True)
    (root / "audio_tokenizer").mkdir(exist_ok=True)
    c, codec = small_configs()
    (root / "config.json").write_text(json.dumps(c))
    (root / "audio_tokenizer/config.json").write_text(json.dumps(codec))
    for src, dest in [
        ("codec_preprocessor.json", "audio_tokenizer/preprocessor_config.json"),
        ("generation_config.json", "generation_config.json"),
        ("special_tokens_map.json", "special_tokens_map.json"),
    ]:
        shutil.copyfile(b.DATA / src, root / dest)
    words = ["<pad>", "<eos>", "<bos>", "<unk>", "hello", "world", "."]
    words += [f"word{i}" for i in range(64 - len(words))]
    words[30] = "<end_of_image>"
    words[50:64] = ["<|AUDIO|>", "<|audio_eos|>", *b.SPECIAL_TOKENS]
    tok = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.add_special_tokens([*words[:4], words[30], *words[50:]])
    tok.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", 2)]
    )
    tok.decoder = decoders.Fuse()
    tok.save(str(root / "tokenizer.json"))
    tokenizer = b.load_tokenizer(root, c)
    torch.manual_seed(6174)
    models = b.native_components(c, codec)
    main = models["model"]
    with torch.no_grad():
        for name, param in models["audio_tokenizer/model"].named_parameters():
            if name.endswith("embedding_sum"):
                param.normal_(std=0.05)
        # Initializer variance must account for the miniature decoder widths;
        # the original .02 scale vanishes through its complete convolution stack
        # below PCM16 resolution. Keep the entire native signal path active.
        for module in models["audio_tokenizer/model"].decoder.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                module.weight.mul_(0.5)
        # Shape-preserving learned weights make every depth head predict code 1,
        # then distinguish the first text step from the next acoustic step to
        # terminate with native EOS. No graph or sampling operation is mocked.
        ids = torch.tensor([b.text_ids(tokenizer, "hello", c)])
        text = main.text_embeddings(ids)
        _, h0, cache = main.backbone_step(text)
        depth = main.depth_decoder
        depth.codebooks_head.weight.zero_()
        for i in range(15):
            prefix = torch.ones(1, i + 2, dtype=torch.long)
            prefix[:, 0] = 0
            positions = torch.arange(i + 2)
            embed = depth.model.embed_tokens(prefix + (positions - 1).clamp(min=0) * 2051)
            embed[:, 0] = h0
            hs = depth.model(
                inputs_embeds=depth.model.inputs_embeds_projector(embed), use_cache=False
            ).last_hidden_state[:, -1]
            depth.codebooks_head.weight[i, :, 1] = hs[0] * 100
        frame = torch.ones(1, 1, 16, dtype=torch.long)
        _, h1, _ = main.backbone_step(main.audio_embeddings(frame), cache)
        direction = h1[0] - h0[0] * (h1[0] @ h0[0]) / (h0[0] @ h0[0])
        assert float(direction.square().sum()) > 0.01
        main.lm_head.weight.zero_()
        main.lm_head.weight[1] = h0[0] * 100
        main.lm_head.weight[2051] = direction * 10000
    for name, model in models.items():
        state = {
            k: v.to(getattr(torch, precision)).contiguous()
            for k, v in model.state_dict().items()
            if name != "model" or k != b.AUDIO_ALIAS
        }
        save_file(state, root / (name + ".safetensors"))
        if name == "model":
            state[b.AUDIO_ALIAS] = state[b.AUDIO_WEIGHT]
        model.load_state_dict({k: v.float() for k, v in state.items()})
    return c, codec, models, tokenizer


def test_original_complete_geometry():
    shapes = {"model": {}, "audio_tokenizer/model": {}}
    for name in ("model-00001-of-00002", "model-00002-of-00002", "audio_tokenizer/model"):
        header = json.loads((FIXTURES / (name + ".safetensors.header.json")).read_text())
        shapes["audio_tokenizer/model" if name.startswith("audio_tokenizer") else "model"].update(
            {k: v["shape"] for k, v in header.items() if k != "__metadata__"}
        )
    b.validate_weight_shapes(RAW, CODEC, shapes)
    assert [len(v) for v in shapes.values()] == [1115, 496]


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("precision", ["float32", "float16", "bfloat16"])
def test_complete_generation_and_loaded_parity(tmp_path, backend, precision):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    c, _, reference, tok = fixture(tmp_path, precision)
    models, tokenizer, _ = b.load_components(tmp_path, c, backend)
    for model in reference.values():
        model.to(backend)
    with torch.inference_mode():
        torch.manual_seed(6174)
        actual = b.synthesize(models, tokenizer, c, "hello")
        torch.manual_seed(6174)
        expected = b.synthesize(reference, tok, c, "hello")
    assert actual.shape == (1920,)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert actual.abs().max() > 0
    assert (
        models["model"].backbone_model.embed_tokens.embed_audio_tokens
        is models["model"].depth_decoder.model.embed_tokens
    )


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_cached_backbone_and_all_depth_heads_match_full_prefix(tmp_path, backend):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    c, _, models, tok = fixture(tmp_path)
    main = models["model"].to(backend)
    with torch.no_grad():
        main.lm_head.weight.normal_(std=0.02)
        main.depth_decoder.codebooks_head.weight.normal_(std=0.02)
    with torch.inference_mode():
        text = main.text_embeddings(torch.tensor([b.text_ids(tok, "hello", c)], device=backend))
        _, hidden, cache = main.backbone_step(text)
        frames = torch.randint(0, 2048, (1, 3, 16), device=backend)
        all_embeds = text
        for i in range(3):
            embeds = main.audio_embeddings(frames[:, i : i + 1])
            all_embeds = torch.cat([all_embeds, embeds], dim=1)
            cached, _, cache = main.backbone_step(embeds, cache)
            full, _, _ = main.backbone_step(all_embeds, use_cache=False)
            torch.testing.assert_close(cached, full, atol=2e-5, rtol=2e-5)
        prefix = torch.cat(
            [torch.zeros(1, 1, dtype=torch.long, device=backend), frames[:, 0]], dim=1
        )
        cache = None
        for length in range(2, 17):
            full, _ = main.depth_decoder.step(prefix[:, :length], hidden, use_cache=False)
            incremental = prefix[:, :length] if cache is None else prefix[:, length - 1 : length]
            cached, cache = main.depth_decoder.step(incremental, hidden, cache)
            torch.testing.assert_close(cached, full, atol=2e-5, rtol=2e-5)


def test_native_sampling_matches_independent_distribution(monkeypatch):
    import numpy as np

    from oryn_engine.native_breeze.model import sample

    logits = torch.linspace(-2, 3, 2052).reshape(1, -1)
    history = [2047, 2047, 2001]
    expected = logits.numpy().copy()
    for ident in set(history):
        expected[:, ident] = np.where(
            expected[:, ident] < 0, expected[:, ident] * 1.1, expected[:, ident] / 1.1
        )
    expected /= 0.9
    expected[:, 2048:2051] = -np.inf
    expected[expected < np.sort(expected, axis=-1)[:, -50:-49]] = -np.inf
    expected = np.exp(expected - np.max(expected, axis=-1, keepdims=True))
    expected /= expected.sum(axis=-1, keepdims=True)
    captured = []

    def choose(probabilities, count):
        captured.append(probabilities.numpy())
        return probabilities.argmax(-1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", choose)
    assert sample(logits, history, eos=True).item() == 2051
    np.testing.assert_allclose(captured[0], expected, atol=1e-7, rtol=1e-6)
    assert not captured[0][0, 2048:2051].any()


@pytest.mark.parametrize(
    "section,field,value",
    [
        (None, "num_codebooks", 8),
        (None, "vocab_size", 2048),
        (None, "audio_eos_token_id", 53),
        (None, "audio_embed_size", 64),
        ("backbone_config", "num_hidden_layers", 1),
        ("backbone_config", "rope_theta", 10000),
        ("depth_decoder_config", "audio_embed_size", 64),
        ("depth_decoder_config", "rope_scaling", None),
        ("text_encoder_config", "layer_types", ["full_attention"] * 6),
        ("text_encoder_config", "eoi_token_index", 51),
        ("text_encoder_config", "query_pre_attn_scalar", 16),
        ("codec_config", "upsample_groups", 32),
        ("text_encoder_special_tokens_config", "token_ids", {}),
    ],
)
def test_closed_main_metadata(section, field, value):
    c, _ = small_configs()
    (c[section] if section else c)[field] = value
    with pytest.raises(ValueError):
        b.validate_config(c)


@pytest.mark.parametrize(
    "section,field,value",
    [
        (None, "output_sample_rate", 16000),
        ("decoder_config", "upsample_rates", [8, 5, 4, 2]),
        ("decoder_config", "num_quantizers", 8),
        ("decoder_config", "decoder_dim", 33),
        ("encoder_config", "upsample_groups", 32),
        ("decoder_config", "rope_theta", 100),
    ],
)
def test_closed_codec_metadata(section, field, value):
    _, c = small_configs()
    (c[section] if section else c)[field] = value
    with pytest.raises(ValueError):
        b.validate_codec(c)


@pytest.mark.parametrize("missing", [None, "model.safetensors", *b.REQUIRED_FILES])
def test_discovery_requires_complete_export(missing):
    from oryn_engine.model_layouts import assess_layout

    required = {*b.REQUIRED_FILES, "model.safetensors"}
    files = required | {"modeling_breeze.py", "pytorch_model.bin", "tokenizer_config.json"}
    if missing:
        files.remove(missing)
    result = assess_layout(sorted(files), RAW, b.TASK)
    if missing:
        assert result.artifact_reasons
    else:
        assert not result.artifact_reasons and not result.runtime_reasons
        assert set(result.selected_files) == required


@pytest.mark.parametrize(
    "corruption",
    ["missing_unused", "missing_encoder", "shape", "integer", "nonfinite", "alias", "extra"],
)
def test_complete_weights_fail_closed(tmp_path, corruption):
    from safetensors.torch import load_file

    c, _, _, _ = fixture(tmp_path)
    path = tmp_path / (
        "audio_tokenizer/model.safetensors"
        if corruption == "missing_encoder"
        else "model.safetensors"
    )
    state = load_file(path)
    key = "embed_text_tokens.weight"
    if corruption == "missing_unused":
        del state[key]
    elif corruption == "missing_encoder":
        del state[next(k for k in state if k.startswith("encoder."))]
    elif corruption == "shape":
        state[key] = state[key][:-1]
    elif corruption == "integer":
        state[key] = state[key].to(torch.int32)
    elif corruption == "nonfinite":
        state[key][0, 0] = float("nan")
    elif corruption == "alias":
        state[b.AUDIO_ALIAS] = state[b.AUDIO_WEIGHT].clone() + 1
    else:
        state["invented.weight"] = torch.ones(1)
    save_file(state, path)
    with pytest.raises(ValueError, match="BREEZE"):
        b.load_components(tmp_path, c, "cpu")


def test_original_sharded_layout_loads_every_component(tmp_path):
    from safetensors.torch import load_file

    c, _, _, _ = fixture(tmp_path)
    original = tmp_path / "model.safetensors"
    state = load_file(original)
    groups = [dict(list(state.items())[::2]), dict(list(state.items())[1::2])]
    mapping = {}
    for index, group in enumerate(groups, 1):
        name = f"model-{index:05}-of-00002.safetensors"
        save_file(group, tmp_path / name)
        mapping.update({key: name for key in group})
    original.unlink()
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": mapping})
    )
    models, tok, _ = b.load_components(tmp_path, c, "cpu")
    with torch.inference_mode():
        assert b.synthesize(models, tok, c, "hello").shape == (1920,)


@pytest.mark.parametrize("text", ["", " ", "x" * 301, "<ins_bos>", "[S1]hello"])
def test_text_bounds(tmp_path, text):
    c, _, _, tok = fixture(tmp_path)
    with pytest.raises(ValueError, match="BREEZE_TEXT"):
        b.text_ids(tok, text, c)


@pytest.mark.parametrize("mode", ["first_eos", "never_eos"])
def test_generation_requires_nonempty_bounded_eos(tmp_path, monkeypatch, mode):
    from oryn_engine.native_breeze import model as native

    c, _, models, tok = fixture(tmp_path)

    def sample(logits, history=None, *, eos=False):
        ident = 2051 if eos and mode == "first_eos" else 1
        return torch.tensor([[ident]], device=logits.device)

    monkeypatch.setattr(native, "sample", sample)
    ids = torch.tensor([b.text_ids(tok, "hello", c)])
    with torch.inference_mode(), pytest.raises(ValueError, match="EMPTY_GENERATION|EOS_LIMIT"):
        native.generate_codes(models["model"], ids, max_frames=1)


@pytest.mark.parametrize("component", ["text", "coarse", "depth", "codec"])
def test_nonfinite_intermediate_is_rejected(tmp_path, component):
    c, _, _, _ = fixture(tmp_path)
    models, tok, _ = b.load_components(tmp_path, c, "cpu")
    main = models["model"]
    with torch.no_grad():
        if component == "text":
            main.text_encoder_proj.weight.fill_(float("nan"))
        elif component == "coarse":
            main.lm_head.weight.fill_(float("nan"))
        elif component == "depth":
            main.depth_decoder.codebooks_head.weight.fill_(float("nan"))
        else:
            models["audio_tokenizer/model"].decoder.decoder[-1].register_forward_pre_hook(
                lambda module, args: (torch.full_like(args[0], float("inf")),)
            )
    with torch.inference_mode(), pytest.raises(ValueError, match="NONFINITE"):
        b.synthesize(models, tok, c, "hello")


@pytest.mark.parametrize(
    "name",
    ["model.safetensors", "audio_tokenizer/model.safetensors", "tokenizer.json", "audio_tokenizer"],
)
def test_symlink_components_rejected(tmp_path, name):
    c, _, _, _ = fixture(tmp_path)
    path = tmp_path / name
    moved = tmp_path / "outside"
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=moved.is_dir())
    with pytest.raises(ValueError):
        b.load_components(tmp_path, c, "cpu")


def test_profile_negotiates_automatic_voice_and_language():
    from oryn_protocol.task_contracts import effective_input, validate_invocation

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
    value = effective_input(profile, {"text": "hello"})
    assert (value["voice"], value["language"], value["speed"]) == ("auto", "auto", 1)
    assert profile["inputModalities"] == ["text"]
    validate_invocation(profile, value)
    for changed in [
        {"voice": "reference"},
        {"language": "en"},
        {"speed": 1.1},
        {"audioArtifactId": "artifact_" + "a" * 24},
    ]:
        with pytest.raises(ValueError):
            validate_invocation(profile, {**value, **changed})


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
            task=b.TASK,
            backend=backend,
            input={"text": "hello", "voice": "auto", "language": "auto", "speed": 1},
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
            assert f.getnframes() == 1920
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
def test_complete_codec_encoder_and_retained_main_codec(tmp_path, frames):
    _, _, models, _ = fixture(tmp_path)
    codec = models["audio_tokenizer/model"]
    signal = torch.sin(torch.arange(frames * 1920) * 0.13).view(1, -1) * 0.2
    with torch.inference_mode():
        codes = codec.encode(
            signal, padding_mask=torch.ones_like(signal, dtype=torch.bool)
        ).audio_codes[0]
        audio = codec.decode(codes.unsqueeze(0)).audio_values[0]
        retained = models["model"].codec_model(signal.unsqueeze(1))
    assert codes.shape == (frames, 16)
    assert bool(((codes >= 0) & (codes < 2048)).all())
    assert audio.shape == signal[0].shape and torch.isfinite(audio).all()
    assert retained.audio_values.shape == signal.unsqueeze(1).shape
    assert torch.isfinite(retained.audio_values).all()


@pytest.mark.parametrize("change", ["bos", "control"])
def test_tokenizer_controls_fail_closed(tmp_path, change):
    from tokenizers import Tokenizer, processors

    c, _, _, _ = fixture(tmp_path)
    path = tmp_path / "tokenizer.json"
    if change == "bos":
        tok = Tokenizer.from_file(str(path))
        tok.post_processor = processors.TemplateProcessing(single="$A")
        tok.save(str(path))
    else:
        data = json.loads(path.read_text())
        data["model"]["vocab"]["[S0]"], data["model"]["vocab"]["[S1]"] = 53, 52
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="BREEZE_TOKENIZER"):
        b.load_tokenizer(tmp_path, c)
