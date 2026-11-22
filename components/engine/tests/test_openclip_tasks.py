import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from oryn_engine import openclip_tasks as binding
from oryn_engine import shared_tasks
from oryn_engine.model_metadata import root_config_name

FIXTURES = Path(__file__).parent / "fixtures/openclip"


def original():
    return json.loads((FIXTURES / "config.json").read_text())


@pytest.fixture
def tiny(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    vectors = json.loads((FIXTURES / "reference.json").read_text())
    config = binding.normalized_config(vectors["config"], binding.TASK)
    torch.manual_seed(vectors["seed"])
    reference = binding.native_model(config).eval()
    state = reference.state_dict()
    assert {
        k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in state.items()
    } == vectors["stateSha256"]
    save_file(state, tmp_path / binding.WEIGHTS)
    (tmp_path / binding.CONFIG).write_text(json.dumps(vectors["config"]))
    (tmp_path / "model.py").write_text("raise AssertionError('repository code is forbidden')")
    yield tmp_path, config, reference, vectors
    torch.set_num_threads(previous)


def fixture_image(case):
    channels = {"RGB": 3, "RGBA": 4, "L": 1}[case["mode"]]
    w, h = case["width"], case["height"]
    data = ((np.arange(w * h * channels) * 17 + 29) % 256).astype(np.uint8)
    return Image.fromarray(
        data.reshape((h, w, channels) if channels != 1 else (h, w)), mode=case["mode"]
    )


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("case", range(4))
def test_native_factory_parity_and_full_shared_result(tiny, backend, case):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    root, config, reference, fixture = tiny
    model, processor, selected = shared_tasks.load_task(root, binding.TASK, backend)
    assert selected == config
    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name].cpu(), expected, rtol=0, atol=0)
    data = fixture["cases"][case]
    image = fixture_image(data)
    transform, tokenizer = processor
    pixels = transform(image)
    assert hashlib.sha256(pixels.numpy().tobytes()).hexdigest() == data["pixelSha256"]
    assert tokenizer(data["labels"]).tolist() == data["tokens"]
    with torch.inference_mode():
        visual = model.encode_image(pixels[None].to(backend), normalize=True)
        text = model.encode_text(tokenizer(data["labels"]).to(backend), normalize=True)
        result = binding.execute(model, processor, image, {"labels": data["labels"]}, backend)
    np.testing.assert_allclose(visual.cpu(), data["imageFeatures"], atol=2e-5, rtol=2e-4)
    np.testing.assert_allclose(text.cpu(), data["textFeatures"], atol=2e-5, rtol=2e-4)
    expected = sorted(zip(data["labels"], data["scores"], strict=True), key=lambda p: -p[1])
    assert [p["label"] for p in result["predictions"]] == [p[0] for p in expected]
    np.testing.assert_allclose(
        [p["score"] for p in result["predictions"]], [p[1] for p in expected], atol=2e-5, rtol=2e-4
    )


@pytest.mark.parametrize(
    "key",
    [
        "visual.conv1.weight",
        "visual.proj",
        "visual.transformer.resblocks.1.attn.in_proj_weight",
        "token_embedding.weight",
        "text_projection",
        "transformer.resblocks.1.mlp.c_proj.weight",
        "positional_embedding",
        "logit_scale",
    ],
)
def test_complete_vision_text_and_projection_required(tiny, key):
    root, config, _, _ = tiny
    state = load_file(root / binding.WEIGHTS)
    del state[key]
    save_file(state, root / binding.WEIGHTS)
    with pytest.raises(ValueError, match="OPENCLIP_WEIGHTS_INCOMPLETE"):
        binding.load_components(root, config, "cpu")


@pytest.mark.parametrize(
    "damage", ["extra", "shape", "nan", "integer", "overflow", "scale", "alternate"]
)
def test_invalid_numeric_graph_rejected(tiny, damage):
    root, config, _, _ = tiny
    state = load_file(root / binding.WEIGHTS)
    name = "text_projection"
    if damage == "extra":
        state["extra"] = torch.zeros(1)
    elif damage == "shape":
        state[name] = state[name][:-1]
    elif damage == "nan":
        state[name][0, 0] = float("nan")
    elif damage == "integer":
        state[name] = state[name].long()
    elif damage == "overflow":
        state[name] = state[name].double()
        state[name][0, 0] = 1e300
    elif damage == "scale":
        state["logit_scale"].fill_(100)
    else:
        save_file({"extra": torch.zeros(1)}, root / "extra.safetensors")
    save_file(state, root / binding.WEIGHTS)
    with pytest.raises(ValueError, match="OPENCLIP_"):
        binding.load_components(root, config, "cpu")


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (["auto_map"], {"AutoModel": "remote.Model"}),
        (["architectures"], ["CLIPModel"]),
        (["model_type"], "clip"),
        (["model_cfg", "quick_gelu"], True),
        (["model_cfg", "embed_dim"], True),
        (["model_cfg", "vision_cfg", "timm_model_name"], "remote"),
        (["model_cfg", "vision_cfg", "layers"], [1, 1, 1, 1]),
        (["model_cfg", "vision_cfg", "width"], 1279),
        (["model_cfg", "vision_cfg", "image_size"], 225),
        (["model_cfg", "vision_cfg", "patch_size"], 2),
        (["model_cfg", "text_cfg", "hf_model_name"], "remote"),
        (["model_cfg", "text_cfg", "hf_tokenizer_name"], "remote"),
        (["model_cfg", "text_cfg", "tokenizer_kwargs"], {"clean": "canonicalize"}),
        (["model_cfg", "text_cfg", "vocab_size"], 50000),
        (["model_cfg", "text_cfg", "context_length"], 512),
        (["model_cfg", "text_cfg", "heads"], 15),
        (["preprocess_cfg", "interpolation"], "bilinear"),
        (["preprocess_cfg", "resize_mode"], "squash"),
        (["preprocess_cfg", "std"], [0, 1, 1]),
        (["preprocess_cfg", "mean"], [float("nan"), 0, 0]),
        (["preprocess_cfg", "size"], 224),
    ],
)
def test_closed_native_configuration(path, value):
    config = binding.normalized_config(original(), binding.TASK)
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        binding.validate_config(config)


def test_metadata_only_normalization_and_layout_do_not_import_runtime():
    code = f"""
import json,sys
from pathlib import Path
from oryn_engine.generated_capabilities import normalized_transformers_config
from oryn_engine.model_layouts import assess_layout
raw=json.loads(Path({str(FIXTURES / "config.json")!r}).read_text())
config=normalized_transformers_config(raw,{binding.TASK!r})
layout=assess_layout(['open_clip_config.json','open_clip_model.safetensors','model.py','open_clip_pytorch_model.bin','tokenizer.json'],config,{binding.TASK!r})
assert layout.selected_files==('open_clip_config.json','open_clip_model.safetensors')
assert not layout.artifact_reasons and not layout.runtime_reasons
assert 'torch' not in sys.modules and 'open_clip' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("damage", [None, "missing", "pickle", "remote", "alternate"])
def test_original_metadata_inspection_selects_shared_runtime(tmp_path, damage):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore, ModelStoreError
    from oryn_engine.runtime_packs.execution import PackRuntime

    config = original()
    if damage == "remote":
        config["model_cfg"]["text_cfg"]["hf_tokenizer_name"] = "remote"
    files = [(binding.CONFIG, 100, "a" * 64)]
    if damage != "missing":
        files.append(
            (
                "open_clip_pytorch_model.bin" if damage == "pickle" else binding.WEIGHTS,
                100,
                "b" * 64,
            )
        )
    if damage == "alternate":
        files.append(("model.safetensors", 100, "c" * 64))
    hub = FakeHub(config=config, files=files, pipeline_tag=binding.TASK)
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    request = {"provider": "huggingface", "repository": "test/openclip", "revision": "main"}
    if damage == "remote":
        with pytest.raises(ModelStoreError):
            store.inspect(request)
    else:
        result = store.inspect(request)
        assert result["executable"] == (damage in {None, "alternate"}), result
        if result["executable"]:
            assert result["runtimePack"]["runnerId"] == "transformers.shared-tasks"
            assert [f["name"] for f in result["selectedFiles"]] == [binding.CONFIG, binding.WEIGHTS]
    assert not hub.downloaded


@pytest.mark.parametrize(
    "names,expected",
    [
        (["open_clip_config.json"], "open_clip_config.json"),
        (["config.json", "open_clip_config.json"], "config.json"),
        (["model_index.json", "config.json", "open_clip_config.json"], "model_index.json"),
    ],
)
def test_hub_root_priority_matches_execution(monkeypatch, names, expected):
    from oryn_engine.models import HuggingFaceClient

    assert root_config_name(names) == expected
    client = HuggingFaceClient()
    info = SimpleNamespace(
        sha="a" * 40,
        gated=False,
        private=False,
        siblings=[SimpleNamespace(rfilename=n) for n in names],
        config={},
    )
    monkeypatch.setattr(client._api, "model_info", lambda **_: info)

    def read(repo, revision, name):
        assert name == expected
        return {"root": name}

    monkeypatch.setattr(client, "read_json", read)
    assert client.model_info("test/openclip", "main").config == {"root": expected}


def test_public_image_contract_runs_native_preprocessing(tiny):
    root, _, _, fixture = tiny
    case = fixture["cases"][0]
    image = root / "input.png"
    fixture_image(case).save(image)
    result = shared_tasks.execute_task(
        root,
        binding.TASK,
        {"imageArtifactId": "artifact_aaaaaaaaaaaaaaaaaaaaaaaa", "labels": case["labels"]},
        "cpu",
        image,
        None,
        lambda *_: None,
    )
    assert len(result["predictions"]) == 3
    assert sum(p["score"] for p in result["predictions"]) == pytest.approx(1)


def test_native_default_tokenizer_is_owned_by_runtime(tiny):
    import open_clip

    root, config, _, _ = tiny
    (root / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "RemoteTokenizer"}))
    (root / "vocab.json").write_text("{}")
    _, (_, tokenizer), _ = binding.load_components(root, config, "cpu")
    native = open_clip.get_tokenizer(f"local-dir:{root}")
    assert tokenizer.encoder == native.encoder
    assert tokenizer.bpe_ranks == native.bpe_ranks
    assert tokenizer.vocab_size == 49408
    assert tokenizer.all_special_ids == [49406, 49407]


@pytest.mark.parametrize("name", ["config.json", "model_index.json"])
def test_competing_root_configs_cannot_change_selected_graph(name):
    from oryn_engine.model_layouts import assess_layout

    config = binding.normalized_config(original(), binding.TASK)
    layout = assess_layout([binding.CONFIG, binding.WEIGHTS, name], config, binding.TASK)
    assert "MODEL_CONFIG_AMBIGUOUS" in layout.artifact_reasons
