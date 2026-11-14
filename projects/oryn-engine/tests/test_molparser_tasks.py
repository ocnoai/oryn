"""Original MolParser geometry and complete synthetic native image-to-text graphs."""

import copy
import json
import shutil
from pathlib import Path

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from oryn_engine import molparser_tasks as m

RAW = m.template("config.json")


def small_config():
    c = copy.deepcopy(RAW)
    c["encoder"].update(hidden_size=24, timm_output_dim=24, model_input_size=64)
    c["decoder"].update(
        d_model=24,
        encoder_hidden_size=24,
        decoder_layers=2,
        decoder_ffn_dim=48,
        max_position_embeddings=32,
    )
    return m.normalized_config(c, m.TASK)


def fixture(root, precision="float32"):
    root.mkdir(exist_ok=True)
    c = small_config()
    (root / "config.json").write_text(json.dumps(c))
    for name in [*m.METADATA_NAMES, "vocab.txt"]:
        shutil.copyfile(m.DATA / name, root / name)
    pre = m.template("preprocessor_config.json")
    pre["image_size"] = 64
    (root / "preprocessor_config.json").write_text(json.dumps(pre))
    torch.manual_seed(6175)
    model = m.native_model(c)
    image = Image.new("RGB", (83, 61), (52, 120, 210))
    words = m.vocabulary(m.template("tokenizer_config.json"))
    with torch.inference_mode():
        model.decoder.model.decoder.embed_tokens.weight.zero_()
        hidden = model.encode(m.pixels(image, c))
        output = model.decoder.model.decoder(
            input_ids=torch.zeros(1, 1, dtype=torch.long),
            encoder_hidden_states=hidden,
            use_cache=False,
        )
        # One learned shared embedding makes the first greedy token a real
        # chemical symbol; all transformer and vision layers remain active.
        model.decoder.model.decoder.embed_tokens.weight[words.index("C")] = (
            output.last_hidden_state[0, -1] * 10
        )
    state = {
        k: v.to(getattr(torch, precision)).contiguous()
        for k, v in model.state_dict().items()
        if k != m.ALIAS
    }
    save_file(state, root / "model.safetensors")
    state[m.ALIAS] = state[m.EMBEDDING]
    model.load_state_dict({k: v.float() for k, v in state.items()})
    return c, model, words, image


def test_original_complete_geometry():
    header = json.loads(
        (Path(__file__).parent / "fixtures/molparser/model.safetensors.header.json").read_text()
    )
    shapes = {k: v["shape"] for k, v in header.items() if k != "__metadata__"}
    model = m.validate_weight_shapes(m.normalized_config(RAW, m.TASK), shapes)
    assert len(shapes) == 311 and len(model.state_dict()) == 312
    assert model.decoder.lm_head.weight is model.decoder.model.decoder.embed_tokens.weight


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("precision", ["float32", "float16", "bfloat16"])
def test_complete_native_generation_and_loaded_parity(tmp_path, backend, precision):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    c, reference, words, image = fixture(tmp_path, precision)
    actual, vocabulary, _ = m.load_components(tmp_path, c, backend)
    reference.to(backend)
    with torch.inference_mode():
        expected = m.execute(reference, words, c, image, {"maxTokens": 4}, backend)
        result = m.execute(actual, vocabulary, c, image, {"maxTokens": 4}, backend)
    assert result == expected and "C" in result["text"]
    assert actual.decoder.lm_head.weight is actual.decoder.model.decoder.embed_tokens.weight


@pytest.mark.parametrize(
    "section,field,value",
    [
        (None, "decoder_start_token_id", 2),
        (None, "vocab_size", 415),
        ("encoder", "timm_pretrained", True),
        ("encoder", "timm_model_name", "vit_base_patch16_224"),
        ("encoder", "pixel_unshuffle", 2),
        ("encoder", "model_input_size", 65),
        ("encoder", "hidden_size", 25),
        ("encoder", "timm_output_dim", 48),
        ("decoder", "tie_word_embeddings", False),
        ("decoder", "add_cross_attention", False),
        ("decoder", "decoder_layers", 7),
        ("decoder", "d_model", 25),
        ("decoder", "encoder_hidden_size", 48),
        ("decoder", "forced_eos_token_id", 3),
    ],
)
def test_closed_geometry_and_semantics(section, field, value):
    c = small_config()
    (c[section] if section else c)[field] = value
    with pytest.raises(ValueError, match="MOLPARSER"):
        m.validate_config(c)


@pytest.mark.parametrize("section", [None, "encoder", "decoder"])
def test_only_reviewed_declarative_aliases_are_accepted(section):
    c = copy.deepcopy(RAW)
    (c[section] if section else c)["auto_map"] = {"AutoModel": "unknown.Model"}
    with pytest.raises(ValueError):
        m.normalized_config(c, m.TASK)


@pytest.mark.parametrize("name", m.METADATA_NAMES)
def test_closed_metadata(name):
    c = m.normalized_config(RAW, m.TASK)
    value = m.template(name)
    m.validate_metadata(name, value, c)
    value["invented"] = True
    with pytest.raises(ValueError):
        m.validate_metadata(name, value, c)


@pytest.mark.parametrize("missing", [None, *m.FILES])
def test_discovery_selects_only_complete_declarative_original(missing):
    from oryn_engine.model_layouts import assess_layout

    c = m.normalized_config(RAW, m.TASK)
    files = {
        *m.FILES,
        "modeling_molparser_mobile.py",
        "tokenization_molparser_mobile.py",
        "pytorch_model.bin",
    }
    if missing:
        files.remove(missing)
    result = assess_layout(sorted(files), c, m.TASK)
    if missing:
        assert result.artifact_reasons
    else:
        assert not result.artifact_reasons and not result.runtime_reasons
        assert set(result.selected_files) == set(m.FILES)


@pytest.mark.parametrize(
    "corruption", ["encoder", "decoder", "extra", "shape", "integer", "nonfinite", "alias"]
)
def test_weights_fail_closed(tmp_path, corruption):
    from safetensors.torch import load_file

    c, _, _, _ = fixture(tmp_path)
    path = tmp_path / "model.safetensors"
    state = load_file(path)
    key = m.EMBEDDING
    if corruption in {"encoder", "decoder"}:
        del state[next(k for k in state if k.startswith(corruption + "."))]
    elif corruption == "extra":
        state["invented.weight"] = torch.ones(1)
    elif corruption == "shape":
        state[key] = state[key][:-1]
    elif corruption == "integer":
        state[key] = state[key].to(torch.int32)
    elif corruption == "nonfinite":
        state[key][0, 0] = float("nan")
    else:
        state[m.ALIAS] = state[key].clone() + 1
    save_file(state, path)
    with pytest.raises(ValueError, match="MOLPARSER"):
        m.load_components(tmp_path, c, "cpu")


def test_chemical_markup_is_preserved():
    words = m.vocabulary(m.template("tokenizer_config.json"))
    names = ["C", "<sep>", "<a>", "1", ":", "O", "</a>", "<r>", "2", "</r>"]
    ids = [0, *(words.index(token) for token in names), 2, 1, 3]
    assert m.decode(ids, words) == "".join(names)
    with pytest.raises(ValueError):
        m.decode([414], words)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_complete_native_graph_preserves_image_conditioning(tmp_path, backend):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    c, reference, _, _ = fixture(tmp_path)
    actual, _, _ = m.load_components(tmp_path, c, backend)
    inputs = torch.cat(
        [m.pixels(Image.new("RGB", (83, 61), color), c) for color in ["black", "white"]]
    )
    ids = torch.tensor([[0, 4, 5], [0, 4, 5]])
    with torch.inference_mode():
        expected = reference.encode(inputs)
        encoded = actual.encode(inputs.to(backend))
        a = actual.decoder(
            input_ids=ids.to(backend), encoder_hidden_states=encoded, use_cache=False
        ).logits.cpu()
        z = reference.decoder(input_ids=ids, encoder_hidden_states=expected, use_cache=False).logits
    torch.testing.assert_close(encoded.cpu(), expected, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(a, z, atol=2e-3, rtol=2e-4)
    assert not torch.allclose(encoded[0], encoded[1], atol=1e-7, rtol=1e-7)
    assert not torch.allclose(a[0], a[1], atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("component", ["encoder", "decoder"])
def test_nonfinite_intermediate_is_rejected(tmp_path, component):
    c, model, words, image = fixture(tmp_path)
    with torch.no_grad():
        if component == "encoder":
            model.encoder.timm_model.model.patch_embed.proj.weight.fill_(float("nan"))
        else:
            model.decoder.model.decoder.layers[0].fc1.weight.fill_(float("nan"))
    with torch.inference_mode(), pytest.raises(ValueError, match="NONFINITE"):
        m.execute(model, words, c, image, {"maxTokens": 4}, "cpu")


@pytest.mark.parametrize("name", ["model.safetensors", "vocab.txt", "tokenizer_config.json"])
def test_symlink_components_rejected(tmp_path, name):
    c, _, _, _ = fixture(tmp_path)
    path = tmp_path / name
    moved = tmp_path / "outside"
    path.rename(moved)
    path.symlink_to(moved)
    with pytest.raises(ValueError):
        m.load_components(tmp_path, c, "cpu")


@pytest.mark.parametrize("limit", [0, 384, True])
def test_invalid_generation_limit_is_rejected(tmp_path, limit):
    c, model, words, image = fixture(tmp_path)
    with torch.inference_mode(), pytest.raises(ValueError, match="OUTPUT_LIMIT"):
        m.execute(model, words, c, image, {"maxTokens": limit}, "cpu")


def test_profile_uses_native_total_token_bound():
    from oryn_protocol.task_contracts import effective_input, validate_invocation

    from oryn_engine.task_profiles import profile_for_execution

    c = m.normalized_config(RAW, m.TASK)
    profile = profile_for_execution(
        {
            "runtimePack": {
                "packId": "oryn.transformers-pytorch",
                "taskContract": "image-to-text-v1",
            }
        },
        c,
    )
    value = effective_input(profile, {"imageArtifactId": "artifact_" + "a" * 24})
    assert value["maxTokens"] == 128
    validate_invocation(profile, {**value, "maxTokens": 256})
    with pytest.raises(ValueError):
        validate_invocation(profile, {**value, "maxTokens": 257})


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_offline_worker_emits_chemical_text(tmp_path, backend):
    import os
    import subprocess
    import sys

    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    _, _, _, image = fixture(tmp_path)
    path = tmp_path / "input.png"
    image.save(path)
    request = dict(
        abi="1",
        packId="oryn.transformers-pytorch",
        version=builtin_registry().active_pack("oryn.transformers-pytorch").version,
        command="generate",
        payload=dict(
            modelRoot=str(tmp_path),
            task=m.TASK,
            backend=backend,
            input={"imageArtifactId": "artifact_" + "a" * 24, "maxTokens": 4},
            inputPath=str(path),
            outputPath=None,
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
        assert "C" in result["result"]["text"]


def test_shared_image_path_keeps_original_pixels(tmp_path, monkeypatch):
    from oryn_engine import shared_tasks

    fixture(tmp_path)
    path = tmp_path / "input.png"
    Image.new("RGBA", (1121, 73), (52, 120, 210, 128)).save(path)
    sizes = []
    original = m.pixels

    def capture(image, config):
        sizes.append(image.size)
        return original(image, config)

    monkeypatch.setattr(m, "pixels", capture)
    result = shared_tasks.execute_task(
        tmp_path,
        m.TASK,
        {"imageArtifactId": "artifact_" + "a" * 24, "maxTokens": 4},
        "cpu",
        input_path=path,
        output_path=None,
        emit=lambda *args: None,
    )
    assert sizes == [(1121, 73)] and "C" in result["text"]
