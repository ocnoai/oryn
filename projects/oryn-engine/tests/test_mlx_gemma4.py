"""Header-only admission and complete synthetic Gemma 4 multimodal regressions."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from oryn_engine import mlx_gemma4 as adapter

FIXTURES = Path(__file__).parent / "fixtures/mlx_gemma4"
ORIGINALS = json.loads((FIXTURES / "originals.json").read_text())
DENSE = "lmstudio-community/gemma-4-E4B-it-MLX-4bit"
MOE = "lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit"


@pytest.mark.parametrize("repository", list(ORIGINALS))
def test_original_metadata_configurations(repository):
    adapter.validate_config(ORIGINALS[repository]["config"])
    adapter.validate_processor(json.loads((FIXTURES / "processor_config.json").read_text()))
    adapter.validate_tokenizer_metadata(
        json.loads((FIXTURES / "tokenizer_config.json").read_text())
    )


@pytest.mark.parametrize(
    "section,change",
    [
        (None, {"architectures": ["CustomModel"]}),
        (None, {"quantization_config": {"quant_method": "custom"}}),
        ("text_config", {"num_kv_shared_layers": 42}),
        ("text_config", {"head_dim": 15}),
        ("text_config", {"num_global_key_value_heads": 0}),
        ("text_config", {"num_global_key_value_heads": True}),
        ("text_config", {"num_global_key_value_heads": 129}),
        ("text_config", {"num_attention_heads": 7}),
        ("text_config", {"enable_moe_block": True}),
        ("text_config", {"use_bidirectional_attention": "all"}),
        ("text_config", {"per_layer_config": {"0": {}}}),
        ("text_config", {"layer_types": ["full_attention"] * 42}),
        ("text_config", {"vocab_size_per_layer_input": 42}),
        ("text_config", {"rope_parameters": {}}),
        ("text_config", {"attention_bias": True}),
        ("text_config", {"tie_word_embeddings": False}),
        ("text_config", {"use_double_wide_mlp": True}),
        ("vision_config", {"use_clipped_linears": 1}),
        ("vision_config", {"standardize": "yes"}),
        ("vision_config", {"pooling_kernel_size": 2}),
        ("vision_config", {"num_attention_heads": 7}),
        ("audio_config", {"attention_context_right": 4}),
        ("audio_config", {"hidden_act": "relu"}),
        ("audio_config", {"num_attention_heads": 7}),
        ("audio_config", {"subsampling_conv_channels": [2, 2]}),
    ],
)
def test_unbound_geometry_is_rejected(section, change):
    config = copy.deepcopy(ORIGINALS[DENSE]["config"])
    (config if section is None else config[section]).update(change)
    with pytest.raises((ValueError, TypeError)):
        adapter.validate_config(config)


@pytest.mark.parametrize(
    "change",
    [
        {"audio_ms_per_token": 20},
        {"audio_seq_length": 1000},
        {"chat_template": "evil"},
        {"feature_extractor": {}},
        {"image_processor": {}},
    ],
)
def test_processor_overrides_rejected(change):
    with pytest.raises(ValueError):
        adapter.validate_processor(adapter.PROCESSOR | change)


@pytest.mark.parametrize("repository", [DENSE, MOE])
@pytest.mark.parametrize("damage", [None, "processor", "tokenizer", "missing"])
def test_metadata_route_and_profile_without_weights(tmp_path, repository, damage):
    from test_models import FakeHub, model

    from oryn_engine.models import ModelStore

    config = copy.deepcopy(ORIGINALS[repository]["config"])
    task = "any-to-any" if repository == DENSE else "image-text-to-text"

    class Hub(FakeHub):
        def read_json(self, repo, revision, name):
            value = (
                config
                if name == "config.json"
                else copy.deepcopy(adapter.PROCESSOR)
                if name == "processor_config.json"
                else json.loads((FIXTURES / "tokenizer_config.json").read_text())
            )
            if name == "processor_config.json" and damage == "processor":
                value["audio_ms_per_token"] = 20
            if name == "tokenizer_config.json" and damage == "tokenizer":
                value["audio_token"] = "wrong"
            return value

    names = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
    ]
    if damage == "missing":
        names.remove("processor_config.json")
    hub = Hub(config=config, pipeline_tag=task, files=[(n, 100, "a" * 64) for n in names])
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda p: None if p.provider else object()
    )
    result = store.inspect(model("independent/renamed-gemma"))
    if damage:
        assert not result["executable"], result
    else:
        assert result["executable"], result
        assert result["technicalCapabilities"]["task"] == task
        if task == "image-text-to-text":
            assert result["taskProfile"]["limits"]["contextTokens"] == 4096
        if task == "any-to-any":
            assert result["taskProfile"]["limits"]["audioDurationSeconds"] == 8
            assert result["taskProfile"]["features"]["audioOutput"] == "unsupported"
    assert not hub.downloaded


def native_fixture(root, kind="dense", bits=None):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_vlm")
    pytest.importorskip("mlx_lm.sample_utils")
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    config = copy.deepcopy(ORIGINALS[DENSE if kind == "dense" else MOE]["config"])
    config.pop("quantization", None)
    config.pop("quantization_config", None)
    text = config["text_config"]
    text.update(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        head_dim=32,
        global_head_dim=32,
        vocab_size=512,
        vocab_size_per_layer_input=512,
        hidden_size_per_layer_input=8 if kind == "dense" else 0,
        num_kv_shared_layers=2 if kind == "dense" else 0,
        sliding_window=512,
        layer_types=["sliding_attention", "full_attention"] * 2,
        max_position_embeddings=4096,
    )
    if kind == "moe":
        text.update(num_experts=4, top_k_experts=2, moe_intermediate_size=64)
    config["vision_config"].update(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        global_head_dim=8,
        num_hidden_layers=1,
        position_embedding_size=10240,
    )
    if kind == "dense":
        config["audio_config"].update(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            output_proj_dims=64,
            subsampling_conv_channels=[128, 2],
        )
    markers = list(
        dict.fromkeys(
            [
                "<pad>",
                "<eos>",
                "<bos>",
                *adapter.TOKENS.values(),
                "<|channel>",
                "<channel|>",
                "<|tool_call>",
                "<tool_call|>",
                "<|tool_response>",
            ]
        )
    )
    vocab = {t: i for i, t in enumerate(markers + sorted(pre_tokenizers.ByteLevel.alphabet()))}
    while len(vocab) < 512:
        vocab[f"unused_{len(vocab)}"] = len(vocab)
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(markers)
    tokenizer.save(str(root / "tokenizer.json"))
    for key, token in adapter.TOKENS.items():
        config[key + "_id"] = vocab[token]
    config["eos_token_id"] = [vocab["<eos>"], vocab["<turn|>"], vocab["<|tool_response>"]]
    text.update(
        bos_token_id=vocab["<bos>"], eos_token_id=vocab["<eos>"], pad_token_id=vocab["<pad>"]
    )
    metadata = json.loads((FIXTURES / "tokenizer_config.json").read_text())
    metadata.pop("added_tokens_decoder", None)
    metadata["chat_template"] = "{{ raise_exception('never run') }}"
    (root / "tokenizer_config.json").write_text(json.dumps(metadata))
    (root / "processor_config.json").write_text(json.dumps(adapter.PROCESSOR))
    mx.random.seed(61624)
    reference = adapter.native_model(config)
    # Native initial RMS/clip constants remain intact; deterministic seeded weights.
    if bits:
        config["quantization"] = {"bits": bits, "group_size": 32}

        def predicate(name, module):
            w = getattr(module, "weight", None)
            return bool(
                callable(getattr(module, "to_quantized", None))
                and w is not None
                and len(w.shape) in {2, 3}
                and w.shape[-1] % 32 == 0
            )

        nn.quantize(reference, bits=bits, group_size=32, class_predicate=predicate)
    reference.eval()
    weights = dict(tree_flatten(reference.parameters()))
    mx.eval(weights)
    mx.save_safetensors(str(root / "model.safetensors"), weights)
    (root / "config.json").write_text(json.dumps(config))
    return reference, adapter.load_tokenizer(root, config), config, weights, mx


@pytest.mark.parametrize("repository", list(ORIGINALS))
def test_full_original_header_graphs_without_payload(repository):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_vlm")
    v = ORIGINALS[repository]
    model = adapter.validate_weight_shapes(
        v["config"], {k: tuple(s) for k, s in v["shapes"].items()}
    )
    assert mx.get_active_memory() < 128 * 1024**2
    del model
    mx.clear_cache()


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize(
    "kind,bits",
    [("dense", None), ("dense", 4), ("dense", 5), ("dense", 6), ("dense", 8), ("moe", 4)],
)
def test_complete_native_graph_modalities_and_incremental_cache(tmp_path, backend, kind, bits):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_vlm")
    pytest.importorskip("mlx_lm.sample_utils")
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_vlm.models.gemma4.audio_feature_extractor import Gemma4AudioFeatureExtractor
    from mlx_vlm.models.gemma4.processing_gemma4 import Gemma4ImageProcessor
    from PIL import Image

    with mx.stream(getattr(mx, backend)):
        reference, tokenizer, config, weights, mx = native_fixture(tmp_path, kind, bits)
        loaded = adapter.load_model(tmp_path, config)
        actual = dict(tree_flatten(loaded.parameters()))
        assert actual.keys() == weights.keys()
        assert all(bool(mx.array_equal(actual[k], v)) for k, v in weights.items())
        image = Image.fromarray(
            np.random.default_rng(61).integers(0, 255, (60, 96, 3), dtype=np.uint8)
        )
        wave = np.sin(np.arange(1997, dtype=np.float32) * 0.08)
        for mode, media in [("image", image)] + ([("audio", wave)] if kind == "dense" else []):
            prepared, prompt = adapter.prepare_inputs(
                config, tokenizer, [{"role": "user", "content": "Čau 🌍"}], mode, media, 3
            )
            ids = prepared["input_ids"]
            mask = ids == config[mode + "_token_id"]
            if mode == "image":
                pixels, _ = Gemma4ImageProcessor()(images=[media])
                np.testing.assert_array_equal(
                    np.array(prepared["pixel_values"]), pixels["pixel_values"]
                )
                encoded = reference.embed_vision(
                    reference.vision_tower(mx.array(pixels["pixel_values"]))
                )
            else:
                features = Gemma4AudioFeatureExtractor()([media], truncation=False, max_length=None)
                np.testing.assert_array_equal(
                    np.array(prepared["input_features"])[0], features["input_features"][0]
                )
                encoded, _ = reference.audio_tower(
                    mx.array(np.array(features["input_features"])),
                    ~mx.array(np.array(features["input_features_mask"])),
                )
                encoded = reference.embed_audio(encoded)
            count = int(mask.sum().item())
            assert count == encoded.shape[1]
            # Independent direct splice, no native modulo-scatter helper.
            start = int(np.where(np.array(mask)[0])[0][0])
            embeds = reference.language_model.model.embed_tokens(ids) * 64**0.5
            embeds = mx.concatenate(
                [embeds[:, :start], encoded, embeds[:, start + count :]], axis=1
            )
            pli = (
                reference.language_model.model.get_per_layer_inputs(mx.where(mask, 0, ids))
                if kind == "dense"
                else None
            )
            expected = reference.language_model(
                inputs_embeds=embeds,
                per_layer_inputs=pli,
                mm_token_type_ids=prepared["mm_token_type_ids"],
            ).logits
            result = loaded(**prepared).logits
            np.testing.assert_allclose(np.array(result), np.array(expected), rtol=2e-4, atol=2e-4)
            cache = loaded.make_cache()
            loaded(cache=cache, **prepared)
            token = tokenizer.encode("X", add_special_tokens=False)[0]
            incremental = loaded(mx.array([[token]]), cache=cache).logits
            expanded = prepared | {
                "input_ids": mx.concatenate([ids, mx.array([[token]])], axis=1),
                "mm_token_type_ids": mx.concatenate(
                    [prepared["mm_token_type_ids"], mx.array([[0]])], axis=1
                ),
            }
            full = reference(**expanded).logits[:, -1:]
            np.testing.assert_allclose(np.array(incremental), np.array(full), rtol=3e-4, atol=3e-4)
            assert prompt.startswith("<bos><|turn>user\n") and prompt.endswith(
                "<turn|>\n<|turn>model\n"
            )
        del loaded, reference
        mx.clear_cache()


@pytest.mark.parametrize(
    "damage",
    [
        "missing-text",
        "missing-vision",
        "missing-audio",
        "missing-shared-kv",
        "extra",
        "shape",
        "scale",
        "packed-dtype",
        "float-dtype",
        "nonfinite",
        "clip-nan",
        "clip-inverted",
    ],
)
def test_damaged_complete_graph_rejected(tmp_path, damage):
    _, _, config, weights, mx = native_fixture(tmp_path, bits=4)
    key = "language_model.model.embed_tokens.weight"
    if damage.startswith("missing"):
        key = {
            "missing-text": key,
            "missing-vision": "embed_vision.embedding_projection.weight",
            "missing-audio": "embed_audio.embedding_projection.weight",
            "missing-shared-kv": "language_model.model.layers.2.self_attn.k_proj.weight",
        }[damage]
        del weights[key]
    elif damage == "extra":
        weights["unknown.weight"] = mx.ones((2, 2))
    elif damage == "shape":
        weights[key] = weights[key][:-1]
    elif damage == "scale":
        del weights[key.removesuffix(".weight") + ".scales"]
    elif damage == "packed-dtype":
        weights[key] = weights[key].astype(mx.float32)
    elif damage == "float-dtype":
        weights["language_model.model.norm.weight"] = mx.ones((64,), dtype=mx.uint32)
    elif damage == "nonfinite":
        weights["language_model.model.norm.weight"] = mx.full((64,), float("nan"))
    else:
        key = next(k for k in weights if k.endswith(".input_min"))
        weights[key] = mx.array(float("nan") if damage == "clip-nan" else float("inf"))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    with pytest.raises(ValueError):
        adapter.load_model(tmp_path, config)


@pytest.mark.parametrize("samples", [320, 639, 640, 641, 1997, 16000, 128000])
def test_audio_features_exact_count_even_for_ragged_durations(tmp_path, samples):
    _, tokenizer, config, _, mx = native_fixture(tmp_path)
    import numpy as np

    prepared, _ = adapter.prepare_inputs(
        config,
        tokenizer,
        [{"role": "user", "content": "Describe."}],
        "audio",
        np.zeros(samples, dtype=np.float32),
        1,
    )
    assert (
        int((prepared["input_ids"] == config["audio_token_id"]).sum())
        == (prepared["input_features"].shape[1] + 3) // 4
    )


def test_reserved_inputs_and_output_limits(tmp_path):
    _, tokenizer, config, _, mx = native_fixture(tmp_path)
    import numpy as np

    for text, wave, maximum in [
        ("hello", np.zeros(128001), 1),
        ("<|audio|>", np.zeros(1000), 1),
        ("hello", np.full(1000, np.nan), 1),
        ("hello", np.zeros(1000), 513),
        ("x" * 20000, np.zeros(1000), 1),
    ]:
        with pytest.raises(ValueError):
            adapter.prepare_inputs(
                config, tokenizer, [{"role": "user", "content": text}], "audio", wave, maximum
            )


def test_worker_native_audio_and_image_generation(tmp_path, monkeypatch):
    _, tokenizer, config, _, mx = native_fixture(tmp_path)
    import wave

    import numpy as np
    from PIL import Image

    from oryn_engine.runtime_packs import mlx_lm_worker

    # Constrain only sampling; every generated step still runs the native full graph.
    calls = []
    sequence = iter(tokenizer.encode("Hi", add_special_tokens=False) + [config["eos_token_id"][0]])

    def factory(**kwargs):
        def sample(logits):
            calls.append(logits.shape)
            return mx.array(next(sequence))

        return sample

    monkeypatch.setattr("mlx_lm.sample_utils.make_sampler", factory)
    events = []
    monkeypatch.setattr(mlx_lm_worker, "_event", lambda n, d: events.append((n, d)))
    audio = tmp_path / "input.wav"
    with wave.open(str(audio), "wb") as f:
        f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        f.writeframes((np.sin(np.arange(1997) * 0.1) * 16000).astype("<i2").tobytes())
    result = mlx_lm_worker.generate(
        {
            "modelRoot": str(tmp_path),
            "task": "any-to-any",
            "backend": "mlx",
            "input": {
                "instruction": "Describe",
                "inputMode": "audio",
                "outputMode": "text",
                "audioArtifactId": "artifact_" + "a" * 24,
                "maxTokens": 8,
            },
            "inputPath": str(audio),
            "outputPath": None,
        }
    )
    assert result == {"text": "Hi"} and len(calls) == 3
    assert not any(n == "job.output.delta" for n, d in events)
    sequence = iter(tokenizer.encode("Hi", add_special_tokens=False) + [config["eos_token_id"][0]])
    image = tmp_path / "input.png"
    Image.new("RGB", (48, 48), "red").save(image)
    result = mlx_lm_worker.generate(
        {
            "modelRoot": str(tmp_path),
            "inputPath": str(image),
            "input": {
                "imageArtifactId": "artifact_" + "b" * 24,
                "messages": [{"role": "user", "content": "Describe"}],
                "maxTokens": 8,
                "temperature": 0,
                "topP": 1,
            },
        }
    )
    assert result == {"text": "Hi", "backend": "mlx"} and len(calls) == 6
    assert [d["delta"] for n, d in events if n == "job.output.delta"] == ["Hi"]


def test_isolated_worker_probe_and_repeatable_native_audio(tmp_path):
    import os
    import subprocess
    import sys
    import wave

    import numpy as np

    from oryn_engine.runtime_packs import mlx_lm_worker

    _, tokenizer, config, _, mx = native_fixture(tmp_path, bits=4)
    audio = tmp_path / "input.wav"
    with wave.open(str(audio), "wb") as f:
        f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        f.writeframes((np.sin(np.arange(1997) * 0.1) * 16000).astype("<i2").tobytes())
    payload = {
        "modelRoot": str(tmp_path),
        "task": "any-to-any",
        "backend": "mlx",
        "input": {
            "instruction": "Describe.",
            "inputMode": "audio",
            "audioArtifactId": "artifact_" + "c" * 24,
            "maxTokens": 2,
        },
        "inputPath": str(audio),
        "outputPath": None,
    }
    expected = mlx_lm_worker.generate(payload)
    envelope = {"abi": "1", "packId": "oryn.mlx-lm", "version": "0.10.0"}
    for command, body in [("probe", {}), ("generate", payload), ("generate", payload)]:
        run = subprocess.run(
            [sys.executable, "-m", "oryn_engine.runtime_packs.mlx_lm_worker"],
            input=json.dumps(envelope | {"command": command, "payload": body}) + "\n",
            text=True,
            capture_output=True,
            timeout=60,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        assert run.returncode == 0, run.stderr
        messages = [json.loads(line) for line in run.stdout.splitlines()]
        result = messages[-1]
        assert result["ok"], (result, run.stderr)
        assert result["result"] == (
            {"backend": "mlx", "available": True} if command == "probe" else expected
        )


def test_existing_native_text_route_does_not_require_multimodal_processors(tmp_path):
    from test_models import FakeHub, model

    from oryn_engine.models import ModelStore

    config = copy.deepcopy(ORIGINALS[MOE]["config"])
    # This pre-existing text binding has no authority to advertise image/audio.
    # It retains the native MLX-LM metadata path even when no processor is present.
    hub = FakeHub(
        config=config,
        pipeline_tag="text-generation",
        files=[
            (name, 100, "a" * 64) for name in ("config.json", "model.safetensors", "tokenizer.json")
        ],
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda p: None if p.provider else object()
    )
    result = store.inspect(model("independent/text-gemma"))
    assert result["executable"], result
    assert result["technicalCapabilities"]["task"] == "text-generation"
    assert result["taskProfile"]["features"]["multimodalConversation"] == "unsupported"
    assert not hub.downloaded
