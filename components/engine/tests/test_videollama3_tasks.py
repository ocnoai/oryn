"""Original metadata, complete synthetic graphs and bounded offline video workers."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from oryn_engine import shared_tasks
from oryn_engine import videollama3_tasks as m

FIXTURES = Path(__file__).parent / "fixtures/videollama3"


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config(tiny=True, tied=False):
    raw = m.template("config.json")
    if tiny:
        raw.update(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=288,
            bos_token_id=256,
            eos_token_id=258,
            image_token_index=278,
            max_position_embeddings=1024,
            tie_word_embeddings=tied,
        )
        raw["vision_encoder_config"].update(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4
        )
    return m.normalized_config(raw, m.TASK)


def original_key(k):
    return (
        k.replace("model.vision_model.", "model.vision_encoder.")
        .replace("model.projector.", "model.mm_projector.")
        .replace("model.language_model.", "model.")
    )


def fixture(root, precision="float32", tied=False, deterministic=False, sharded=False):
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import VideoLlama3ForConditionalGeneration

    root.mkdir(exist_ok=True)
    c = config(tied=tied)
    for name in m.METADATA_NAMES:
        value = m.template(name)
        if name == "added_tokens.json":
            value = {k: v - 151643 + 256 for k, v in value.items()}
        if name == "tokenizer_config.json":
            value["added_tokens_decoder"] = {
                str(int(k) - 151643 + 256): v for k, v in value["added_tokens_decoder"].items()
            }
        (root / name).write_text(json.dumps(value))
    vocab = {k: i for i, k in enumerate(sorted(ByteLevel.alphabet()))}
    (root / "vocab.json").write_text(json.dumps(vocab))
    (root / "merges.txt").write_text("#version: 0.2\n")
    (root / "config.json").write_text(
        json.dumps(c | {"auto_map": m.template("config.json")["auto_map"]})
    )
    torch.manual_seed(6192)
    model = VideoLlama3ForConditionalGeneration(m.native_config(c)).float().eval()
    if deterministic:
        # A complete, deliberately simple graph emits A then EOS. No mocked
        # forward/generate path; every original vision/projector/text array loads.
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.model.language_model.embed_tokens.weight.fill_(1)
            model.model.language_model.embed_tokens.weight[vocab["A"]].fill_(-1)
            model.model.language_model.norm.weight.fill_(1)
            model.lm_head.weight[vocab["A"]].fill_(1)
            model.lm_head.weight[c["eos_token_id"]].fill_(-1)
    state = {
        original_key(k): v.to(getattr(torch, precision)).clone().contiguous()
        for k, v in model.state_dict().items()
    }
    if tied:
        del state["lm_head.weight"]
    if sharded:
        groups = [dict(list(state.items())[i::2]) for i in range(2)]
        index = {}
        for i, group in enumerate(groups, 1):
            name = f"model-{i:05d}-of-00002.safetensors"
            save_file(group, root / name)
            index.update({k: name for k in group})
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    else:
        save_file(state, root / "model.safetensors")
    native = {m.native_key(k): v.float() for k, v in state.items()}
    if tied:
        native["lm_head.weight"] = native[m.EMBEDDING]
    model.load_state_dict(native, strict=True)
    return c, model, m.tokenizer(root, c)


def frames():
    result = [
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (77, 91, 3), dtype=np.uint8))
        for i in range(8)
    ]
    result[1] = result[0].copy()
    return result


@pytest.mark.parametrize("size,arrays", [("2B", 778), ("7B", 779)])
def test_original_complete_header_geometry(size, arrays):
    c = m.normalized_config(json.loads((FIXTURES / f"{size}-config.json").read_text()), m.TASK)
    shapes = {}
    for path in FIXTURES.glob(size + "-*.safetensors.header.json"):
        shapes.update(
            {k: v["shape"] for k, v in json.loads(path.read_text()).items() if k != "__metadata__"}
        )
    assert len(shapes) == arrays
    m.validate_weight_shapes(c, shapes)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("precision", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("tied", [True, False])
def test_complete_native_graph_all_storage_precisions(tmp_path, backend, precision, tied):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    c, reference, tokenizer = fixture(tmp_path, precision, tied, sharded=not tied)
    loaded, processor, selected = shared_tasks.load_task(tmp_path, m.TASK, backend)
    assert selected == c and processor.chat_template is None
    inputs = m.prepare_inputs(tokenizer, c, frames(), [i / 4 for i in range(8)], "What moves?", 4)
    assert int(inputs["video_compression_mask"].reshape(8, -1)[1].sum()) == 1
    with torch.inference_mode():
        expected = reference(**inputs, use_cache=False).logits
        actual = loaded(**shared_tasks._move(inputs, backend), use_cache=False).logits.cpu()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    "change",
    [
        {"mm_projector_type": "linear"},
        {"use_token_compression": False},
        {"vision_encoder": "outside/model"},
        {"use_sliding_window": True},
        {"rope_scaling": {"rope_type": "linear", "factor": 2}},
        {"num_attention_heads": 3},
        {"num_key_value_heads": True},
        {"image_token_index": 277},
        {"tie_word_embeddings": 1},
        {"architectures": ["Qwen2ForCausalLM"]},
        {"unexpected": True},
        {"auto_map": {"AutoConfig": "other.Code"}},
    ],
)
def test_closed_config_rejects_other_graphs(change):
    with pytest.raises(ValueError):
        m.normalized_config(config() | change, m.TASK)


@pytest.mark.parametrize("name", m.METADATA_NAMES)
def test_processor_dispatch_and_active_overrides_are_closed(name):
    c = config(False)
    value = m.template(name)
    m.validate_metadata(name, value, c)
    value["unexpected"] = True
    with pytest.raises(ValueError):
        m.validate_metadata(name, value, c)


@pytest.mark.parametrize("missing", [None, *m.REQUIRED_FILES])
def test_subset_requires_complete_original_processors(missing):
    from oryn_engine.model_layouts import assess_layout

    files = {
        *m.REQUIRED_FILES,
        "model.safetensors",
        "modeling_videollama3.py",
        "chat_template.json",
    }
    if missing:
        files.remove(missing)
    result = assess_layout(sorted(files), config(), m.TASK)
    if missing:
        assert result.artifact_reasons
    else:
        assert not result.artifact_reasons and not result.runtime_reasons
        assert set(result.selected_files) == {*m.REQUIRED_FILES, "model.safetensors"}


@pytest.mark.parametrize(
    "override",
    [
        {},
        {"auto_map": {"AutoProcessor": "unreviewed.Code"}},
        {"vocab_file": "../unapproved.json"},
        {"trust_remote_code": True},
        {"unexpected": True},
    ],
)
def test_original_metadata_discovery_without_weights(tmp_path, override):
    from test_metadata_audit import row, snapshot

    from oryn_engine.metadata_audit import inspect_row
    from oryn_engine.models import HuggingFaceClient

    raw = m.template("config.json")
    hub = snapshot(
        tmp_path,
        config=raw,
        files=[(n, 100, "b" * 64) for n in (*m.REQUIRED_FILES, "model.safetensors")],
        pipeline_tag=m.TASK,
    )
    reader = object.__new__(HuggingFaceClient)
    reader._bounded_bytes = lambda repository, revision, name, limit: json.dumps(
        raw
        if name == "config.json"
        else m.template(name) | (override if name == "tokenizer_config.json" else {})
    ).encode()
    hub.client.read_json.side_effect = reader.read_json
    result = inspect_row(row(task=m.TASK), hub)
    assert result["status"] == ("metadata-incompatible" if override else "metadata-compatible"), (
        result
    )
    if override:
        assert "MODEL_METADATA_INVALID" in result["reasons"]
    assert not result["executionVerified"]
    hub.client.download.assert_not_called()


@pytest.mark.parametrize(
    "change",
    ["vision", "projector", "language", "head", "extra", "shape", "integer", "nonfinite", "tie"],
)
def test_every_original_tensor_is_required(tmp_path, change):
    c, _, _ = fixture(tmp_path, tied=change == "tie")
    path = tmp_path / "model.safetensors"
    state = load_file(path)
    prefixes = {
        "vision": "model.vision_encoder.",
        "projector": "model.mm_projector.",
        "language": "model.layers.",
        "head": "lm_head.",
    }
    key = next(iter(state))
    if change in prefixes:
        del state[next(k for k in state if k.startswith(prefixes[change]))]
    elif change == "extra":
        state["unexpected.weight"] = torch.ones(2)
    elif change == "shape":
        state[key] = state[key].flatten()[:1]
    elif change == "integer":
        state[key] = state[key].to(torch.int16)
    elif change == "nonfinite":
        state[key].flatten()[0] = float("nan")
    else:
        state["lm_head.weight"] = state["model.embed_tokens.weight"].clone() + 1
    save_file(state, path)
    with pytest.raises(ValueError):
        m.load_components(tmp_path, c, "cpu")


@pytest.mark.parametrize(
    "question,times,limit",
    [
        ("<image>", list(range(8)), 4),
        ("<|im_end|>", list(range(8)), 4),
        ("x", [0] * 7 + [float("nan")], 4),
        ("x", list(range(8))[::-1], 4),
        ("x", list(range(8)), 257),
        ("x", list(range(8)), True),
    ],
)
def test_prompt_rejects_unowned_media_and_invalid_limits(tmp_path, question, times, limit):
    c, _, tokenizer = fixture(tmp_path)
    with pytest.raises(ValueError):
        m.prepare_inputs(tokenizer, c, frames(), times, question, limit)


def test_exact_normalization_patch_order_and_compression():
    selected = [Image.new("RGB", (112, 112), (i * 16, 0, 255)) for i in range(8)]
    selected[1] = selected[0].copy()
    pixels, grid, merges = m.video_pixels(selected)
    assert grid.tolist() == [[8, 8, 8]] and merges.tolist() == [2]
    assert pixels.shape == (512, 588)
    torch.testing.assert_close(pixels[0, :196], torch.full((196,), -1.0))
    torch.testing.assert_close(pixels[0, -196:], torch.ones(196))
    mask = m.compression_mask(pixels, grid).reshape(8, 16)
    assert mask.sum(1).tolist() == [16, 1, 16, 16, 16, 16, 16, 16]


@pytest.mark.parametrize("change", ["size", "count", "mixed", "mode", "overflow"])
def test_bounded_pixels_and_arithmetic(change):
    images = frames()
    if change == "size":
        images = [Image.new("RGB", (513, 513))] * 8
    elif change == "count":
        images.pop()
    elif change == "mixed":
        images[0] = Image.new("RGB", (2, 3))
    elif change == "mode":
        images[0] = images[0].convert("L")
    else:
        with pytest.raises(ValueError):
            m.finite(torch.tensor([float("inf")]))
        return
    with pytest.raises(ValueError):
        m.video_pixels(images)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_offline_worker_complete_native_video_answer(tmp_path, backend):
    import os
    import subprocess
    import sys

    from test_coverage_tasks import video

    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    fixture(tmp_path, deterministic=True)
    path = tmp_path / "clip.mp4"
    video(path)
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
                "videoArtifactId": "artifact_" + "a" * 24,
                "question": "What moves?",
                "maxTokens": 4,
            },
            inputPath=str(path),
            outputPath=None,
        ),
    )
    run = subprocess.run(
        [sys.executable, "-m", "oryn_engine.runtime_packs.transformers_worker"],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        timeout=90,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout.splitlines()[-1])
    assert result["ok"], (result, run.stderr)
    assert result["result"]["text"] == "A"
    assert len(result["result"]["sampledTimestamps"]) == 8


def test_generation_requires_eos_and_ignores_repository_defaults(tmp_path):
    from test_coverage_tasks import video

    c, _, _ = fixture(tmp_path, deterministic=True)
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"do_sample": True, "num_beams": 10})
    )
    path = tmp_path / "clip.mp4"
    video(path)
    loaded, processor, _ = m.load_components(tmp_path, c, "cpu")
    with torch.inference_mode(), pytest.raises(ValueError, match="INCOMPLETE"):
        m.execute(
            loaded,
            processor,
            c,
            {"startSeconds": 0, "question": "What moves?", "maxTokens": 1},
            "cpu",
            path,
        )


def test_profile_reports_owned_video_and_token_bounds():
    from oryn_engine.task_profiles import profile_for_execution

    profile = profile_for_execution(
        {
            "technicalCapabilities": {"task": m.TASK},
            "runtimePack": {
                "id": "oryn.transformers-pytorch",
                "taskContract": "video-question-answering-v1",
            },
        },
        config(),
    )
    assert profile["limits"]["videoFrames"] == 8
    assert profile["parameters"]["maxTokens"]["valueSchema"]["allOf"][1]["maximum"] == 256
