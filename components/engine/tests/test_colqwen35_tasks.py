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

from oryn_engine import colqwen35_tasks as binding
from oryn_engine import shared_tasks

FIXTURES = Path(__file__).parent / "fixtures/colqwen35"


def original(kind="single", filename="config.json"):
    return json.loads((FIXTURES / kind / filename).read_text())


@pytest.fixture(params=["single", "mrl", "wide"])
def tiny(tmp_path, request):
    from test_qwen_translation import fixture
    from transformers import Qwen3_5Model

    old = torch.get_num_threads()
    torch.set_num_threads(1)
    _, tokenizer, config, _ = fixture(tmp_path, mtp=0)
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.save_pretrained(tmp_path)
    kind = request.param
    config.update(
        architectures=["ColQwen3_5"],
        dim=320,
        embed_dim=320,
        retrieval_attention_contract="bidirectional",
    )
    hidden = 4096 if kind == "wide" else 320
    config["text_config"].update(
        hidden_size=hidden,
        intermediate_size=64 if kind == "wide" else 640,
        vocab_size=512,
        is_causal=False,
    )
    config["vision_config"]["out_hidden_size"] = hidden
    if kind == "wide":
        config.update(dim=4096, embed_dim=4096)
    if kind == "mrl":
        config.update(
            head_dims=[64, 128, 256], dim=256, embed_dim=128, anchor_dim=128, mrl_prefix=True
        )
        del config["text_config"]["is_causal"]  # Original EVIE enables this in its owned loader.
    native = binding.qwen_translation.native_config(binding.graph_config(config))
    reference = Qwen3_5Model(native).float().eval()
    for layer in reference.language_model.layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn.is_causal = False
    projection = torch.nn.Linear(hidden, config["dim"])
    weights = {k: v.detach().clone() for k, v in reference.state_dict().items()}
    weights.update(
        {"custom_text_proj." + k: v.detach().clone() for k, v in projection.state_dict().items()}
    )
    save_file(weights, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(config))
    data = original(kind, "processor_config.json")
    data["image_processor"].update(patch_size=2, size={"shortest_edge": 1024, "longest_edge": 2048})
    (tmp_path / "processor_config.json").write_text(json.dumps(data))
    (tmp_path / "modeling_colqwen3_5.py").write_text(
        "raise RuntimeError('never import repository code')"
    )
    yield tmp_path, config, reference, projection, tokenizer
    torch.set_num_threads(old)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("size", [(24, 24), (36, 20)])
def test_complete_native_hybrid_page_query_projection_and_maxsim(tiny, backend, size):
    from transformers import Qwen2VLImageProcessor

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    root, config, reference, projection, tokenizer = tiny
    model, processor, selected = shared_tasks.load_task(root, "visual-document-retrieval", backend)
    assert selected == config and processor.tokenizer.chat_template is None
    for name, tensor in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name].cpu(), tensor, rtol=0, atol=0)
    torch.testing.assert_close(
        model.custom_text_proj.weight.cpu(), projection.weight, rtol=0, atol=0
    )
    assert model.custom_text_proj.out_features == config["dim"]
    image = Image.fromarray(
        np.arange(size[0] * size[1] * 3, dtype=np.uint8).reshape(size[1], size[0], 3)
    )
    native_image = Qwen2VLImageProcessor(
        patch_size=2,
        temporal_patch_size=2,
        merge_size=2,
        size={"shortest_edge": 1024, "longest_edge": 2048},
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    )(images=[image], return_tensors="pt")
    count = int(native_image["image_grid_thw"][0].prod()) // 4
    prompt = (
        "<|im_start|>user\n<|vision_start|>"
        + "<|image_pad|>" * count
        + "<|vision_end|>Describe the image.<|im_end|><|endoftext|>"
    )
    page = dict(tokenizer([prompt], return_tensors="pt", add_special_tokens=False)) | dict(
        native_image
    )
    page["mm_token_type_ids"] = (page["input_ids"] == config["image_token_id"]).long()
    queries = ["  blue page  ", "Žltý svet"]
    expected = []
    with torch.inference_mode():
        for inputs in [
            page,
            *[
                tokenizer([q + "<|endoftext|>" * 10], return_tensors="pt", add_special_tokens=False)
                for q in queries
            ],
        ]:
            reference.rope_deltas = None
            hidden = reference(**inputs, use_cache=False, return_dict=True).last_hidden_state
            projected = projection(hidden)[0, :, : binding.dimensions(config)]
            expected.append(projected / projected.norm(dim=-1, keepdim=True))
        result = binding.retrieve(model, processor, config, image, {"queries": queries}, backend)
    assert result["dimensions"] == binding.dimensions(config)
    np.testing.assert_allclose(result["documentVectors"], expected[0].numpy(), rtol=2e-3, atol=3e-5)
    for actual, vector in zip(result["queries"], expected[1:], strict=True):
        np.testing.assert_allclose(actual["vectors"], vector.numpy(), rtol=2e-3, atol=3e-5)
        score = (vector @ expected[0].T).max(-1).values.sum().item()
        assert actual["score"] == pytest.approx(score, abs=3e-4)
    shared_tasks.validate_task_output("visual-document-retrieval", result)
    # Images never leave position state behind for an independent query.
    input_ids = tokenizer(["same query" + "<|endoftext|>" * 10], return_tensors="pt")
    with torch.inference_mode():
        before = model(**{k: v.to(backend) for k, v in input_ids.items()}).cpu()
        binding.retrieve(model, processor, config, image, {"queries": ["other"]}, backend)
        after = model(**{k: v.to(backend) for k, v in input_ids.items()}).cpu()
    torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.parametrize(
    "damage",
    [
        "vision",
        "linear",
        "attention",
        "head",
        "shape",
        "extra",
        "nonfinite",
        "integer",
        "unused-prefix-tail",
    ],
)
def test_all_learned_tensors_including_unused_mrl_tail_required(tiny, damage):
    root, config, _, _, _ = tiny
    state = load_file(root / "model.safetensors")
    fragment = {
        "vision": "visual.patch_embed",
        "linear": "linear_attn.conv1d",
        "attention": "self_attn.q_proj",
        "head": "custom_text_proj.bias",
    }.get(damage, "custom_text_proj.weight")
    key = next(k for k in state if fragment in k)
    if damage in {"vision", "linear", "attention", "head"}:
        del state[key]
    elif damage == "shape":
        state[key] = torch.zeros(3, 3)
    elif damage == "extra":
        state["unreviewed.weight"] = torch.zeros(2)
    elif damage == "integer":
        state[key] = state[key].to(torch.int32)
    elif damage == "unused-prefix-tail":
        state[key][-1, 0] = float("nan")
    else:
        state[key][0, 0] = float("inf")
    save_file(state, root / "model.safetensors")
    with pytest.raises(ValueError):
        binding.load_components(root, config, "cpu")


@pytest.mark.parametrize(
    "path,value",
    [
        ("auto_map", {"AutoModel": "remote.Model"}),
        ("dim", 4096),
        ("embed_dim", 128),
        ("text_config.is_causal", True),
        ("text_config.num_attention_heads", 3),
        ("retrieval_attention_contract", "causal"),
        ("mask_non_image_embeddings", True),
        ("padding_side", "right"),
        ("vision_config.out_hidden_size", 17),
    ],
)
def test_closed_graph_and_scoring_contract(path, value):
    config = original()
    target = config
    keys = path.split(".")
    for k in keys[:-1]:
        target = target[k]
    target[keys[-1]] = value
    with pytest.raises(ValueError):
        binding.validate_config(config)


@pytest.mark.parametrize(
    "patch",
    [
        {"head_dims": [64, 256, 2048]},
        {"head_dims": [128, 64, 2048]},
        {"head_dims": [64, 128, True, 2048]},
        {"mrl_prefix": False},
        {"anchor_dim": 64},
        {"embed_dim": 320},
        {"dim": 1024},
    ],
)
def test_mrl_head_selection_must_be_declared(patch):
    with pytest.raises(ValueError):
        binding.validate_config(original("mrl") | patch)


def test_wide_original_head_is_preserved_without_mrl_truncation():
    config = original("wide")
    binding.validate_config(config)
    assert binding.dimensions(config) == config["dim"] == 4096


def test_metadata_validation_without_native_imports():
    script = f"""
import json,sys
from pathlib import Path
from oryn_engine import colqwen35_tasks as binding
root=Path({str(FIXTURES)!r})
for kind in ('single','mrl','wide'):
 config=json.loads((root/kind/'config.json').read_text())
 binding.validate_config(config)
 for name in binding.METADATA_NAMES:
  binding.validate_metadata(name,json.loads((root/kind/name).read_text()),config)
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize(
    "damage",
    [None, "missing-tokenizer", "missing-processor", "normalization", "remote", "projection"],
)
@pytest.mark.parametrize("kind", ["mrl", "wide"])
def test_original_discovery_is_metadata_only_and_closed(tmp_path, damage, kind):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore
    from oryn_engine.runtime_packs.execution import PackRuntime

    data = {name: original(kind, name) for name in ["config.json", *binding.METADATA_NAMES]}
    if damage == "remote":
        data["config.json"]["auto_map"] = {"AutoModel": "unreviewed.Model"}
    if damage == "projection":
        data["config.json"]["dim"] = 4097
    if damage == "normalization":
        data["processor_config.json"]["image_processor"]["image_mean"] = [0.0] * 3
    files = set(data) | {"model.safetensors", "tokenizer.json"}
    if damage == "missing-tokenizer":
        files.remove("tokenizer.json")
    if damage == "missing-processor":
        files.remove("processor_config.json")

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            return data[filename]

    hub = Hub(
        config=data["config.json"],
        files=[(n, 100, "a" * 64) for n in files],
        pipeline_tag="visual-document-retrieval",
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    request = {"provider": "huggingface", "repository": "test/colqwen35", "revision": "main"}
    result = store.inspect(request)
    assert not hub.downloaded
    assert result["executable"] == (damage is None), result
    if damage is None:
        assert result["taskProfile"]["limits"]["embeddingDimensions"] == binding.dimensions(
            data["config.json"]
        )
        assert result["runtimePack"]["runnerId"] == "transformers.shared-tasks"


def test_profile_bounds_and_shared_execution(tiny):
    from oryn_protocol.task_contracts import validate_invocation, validate_result

    from oryn_engine.task_profiles import profile_for_execution

    root, config, _, _, _ = tiny
    profile = profile_for_execution(
        {
            "runtimePack": {
                "runnerId": "transformers.shared-tasks",
                "taskContract": "visual-document-retrieval-v1",
            }
        },
        config,
    )
    assert profile["limits"]["embeddingDimensions"] == binding.dimensions(config)
    value = {"imageArtifactId": "artifact_aaaaaaaaaaaaaaaaaaaaaaaa", "queries": ["hello"]}
    validate_invocation(profile, value)
    with pytest.raises(ValueError):
        validate_invocation(profile, value | {"queries": ["x"] * 3})
    image = Image.fromarray(np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3))
    path = root / "input.png"
    image.save(path)
    result = shared_tasks.execute_task(
        root, "visual-document-retrieval", value, "cpu", path, None, lambda *_: None
    )
    validate_result(profile, {"kind": "structured", "value": result}, value)
    model, processor, _ = binding.load_components(root, config, "cpu")
    with pytest.raises(ValueError, match="TOKEN_LIMIT"):
        binding.retrieve(model, processor, config, image, {"queries": ["z" * 100]}, "cpu")
    with pytest.raises(ValueError, match="FRAMING"):
        binding.retrieve(model, processor, config, image, {"queries": ["<|image_pad|>"]}, "cpu")


@pytest.mark.parametrize(
    "damage", ["missing-marker", "pixel-budget", "remote-video", "unknown-image-transform"]
)
def test_closed_tokenizer_and_processor(tiny, damage):
    root, config, _, _, _ = tiny
    if damage == "missing-marker":
        p = root / "tokenizer.json"
        p.write_text(p.read_text().replace("<|im_end|>", "<|renamed|>"))
    else:
        p = root / "processor_config.json"
        value = json.loads(p.read_text())
        if damage == "pixel-budget":
            value["image_processor"]["size"]["shortest_edge"] = 1024**2
        elif damage == "remote-video":
            value["video_processor"]["auto_map"] = {"AutoProcessor": "remote.Video"}
        else:
            value["image_processor"]["do_unreviewed_transform"] = True
        p.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        binding.processor(root, config)


@pytest.mark.parametrize("tiny", ["wide"], indirect=True)
def test_wide_projection_two_offline_workers_use_small_file_descriptors(tiny):
    import os

    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry
    from oryn_engine.structured_outputs import read_result

    root, config, _, _, _ = tiny
    image_path = root / "page.png"
    Image.new("RGB", (24, 24), "blue").save(image_path)
    output_path = root / "result.json"
    pack = builtin_registry().active_pack("oryn.transformers-pytorch")
    payload = dict(
        modelRoot=str(root),
        task="visual-document-retrieval",
        backend="cpu",
        input=dict(imageArtifactId="artifact_" + "a" * 24, queries=["hello"]),
        inputPath=str(image_path),
        outputPath=str(output_path),
    )
    request = dict(
        abi=pack.abi, packId=pack.pack_id, version=pack.version, command="generate", payload=payload
    )
    previous = None
    for _ in range(2):
        run = subprocess.run(
            [sys.executable, "-m", "oryn_engine.runtime_packs.transformers_worker"],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            timeout=90,
            env=os.environ
            | dict(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1"),
        )
        assert run.returncode == 0, run.stderr
        lines = run.stdout.splitlines()
        assert lines and all(len(line.encode()) < 1024 * 1024 for line in lines)
        message = json.loads(lines[-1])
        assert message["ok"], run.stderr
        value = read_result(output_path, message["result"], "cpu")
        assert value["dimensions"] == 4096
        assert len(value["documentVectors"][0]) == 4096
        assert message["result"]["sizeBytes"] > 1024 * 1024
        shared_tasks.validate_task_output("visual-document-retrieval", value)
        raw = output_path.read_bytes()
        assert previous is None or previous == raw
        previous = raw
        output_path.unlink()


def test_original_wide_header_matches_complete_native_meta_graph():
    header = original("wide", "model.safetensors.structure.json")
    config = original("wide")
    shapes = {name: tuple(v["shape"]) for name, v in header["tensors"].items()}
    model = binding.validate_weight_shapes(config, shapes)
    assert len(shapes) == 761
    assert tuple(model.custom_text_proj.weight.shape) == (4096, 4096)
    assert tuple(model.custom_text_proj.bias.shape) == (4096,)
    assert binding.dimensions(config) == 4096
