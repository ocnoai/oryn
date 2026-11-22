"""Original LLaVA-Video metadata and complete synthetic SigLIP/Qwen2 graphs."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from oryn_engine import llava_video_qwen as m
from oryn_engine import shared_tasks
from oryn_engine.native_qwen_bpe import load_qwen_bpe


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config(tiny=True):
    c = m.normalized_config(m.template("config.json"), m.TASK)
    if tiny:
        c.update(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=272,
            bos_token_id=256,
            eos_token_id=258,
            image_token_index=259,
            max_position_embeddings=1024,
            tokenizer_model_max_length=1024,
            mm_hidden_size=32,
        )
        c["vision_config"].update(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=56,
        )
    m.validate_config(c)
    return c


def fixture(root, precision="float32", sharded=False, deterministic=False):
    from tokenizers.pre_tokenizers import ByteLevel

    root.mkdir(exist_ok=True)
    c = config()
    for name in m.METADATA_NAMES:
        value = m.template(name)
        if name == "added_tokens.json":
            value = {k: v - 151643 + 256 for k, v in value.items()}
        if name == "tokenizer_config.json":
            value["added_tokens_decoder"] = {
                str(int(k) - 151643 + 256): v for k, v in value["added_tokens_decoder"].items()
            }
        if name == "preprocessor_config.json":
            value["size"] = dict(height=56, width=56)
        (root / name).write_text(json.dumps(value))
    vocab = {k: i for i, k in enumerate(sorted(ByteLevel.alphabet()))}
    (root / "vocab.json").write_text(json.dumps(vocab))
    (root / "merges.txt").write_text("#version: 0.2\n")
    (root / "config.json").write_text(json.dumps(c))
    torch.manual_seed(6194)
    model = m.native_model(c).float().eval()
    if deterministic:
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.model.embed_tokens.weight.fill_(1)
            model.model.embed_tokens.weight[vocab["A"]].fill_(-1)
            model.model.norm.weight.fill_(1)
            model.lm_head.weight[vocab["A"]].fill_(1)
            model.lm_head.weight[c["eos_token_id"]].fill_(-1)
    state = {
        k.replace(
            "model.vision_tower.vision_tower.", "model.vision_tower.vision_tower.vision_model."
        ): v.to(getattr(torch, precision)).clone().contiguous()
        for k, v in model.state_dict().items()
    }
    if sharded:
        index = {}
        for i in range(2):
            part = dict(list(state.items())[i::2])
            name = f"model-{i + 1:05d}-of-00002.safetensors"
            save_file(part, root / name)
            index.update({k: name for k in part})
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    else:
        save_file(state, root / "model.safetensors")
    model.load_state_dict({m.native_key(k): v.float() for k, v in state.items()}, strict=True)
    processor = load_qwen_bpe(
        root,
        256,
        272,
        json.loads((root / "tokenizer_config.json").read_text())["added_tokens_decoder"],
    )
    return c, model, processor


def frames():
    return [
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (73, 91, 3), dtype=np.uint8))
        for i in range(8)
    ]


def test_original_all_exported_vision_and_language_arrays():
    h = {}
    for p in (Path(__file__).parent / "fixtures/llava_video_qwen").glob("*.json"):
        h.update(
            {k: v["shape"] for k, v in json.loads(p.read_text()).items() if k != "__metadata__"}
        )
    assert len(h) == 765 and not any(".head." in k for k in h)
    assert "model.vision_tower.vision_tower.vision_model.post_layernorm.weight" in h
    assert "model.image_newline" in h
    m.validate_weight_shapes(config(False), h)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("precision", ["float32", "float16", "bfloat16"])
def test_complete_original_graph_storage_and_row_features(tmp_path, backend, precision):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    c, reference, _ = fixture(tmp_path, precision, sharded=True)
    loaded, processor, selected = shared_tasks.load_task(tmp_path, m.TASK, backend)
    assert selected == c and processor.chat_template is None
    pixels = m.video_pixels(c, frames())
    with torch.inference_mode():
        expected = m.visual_features(reference, c, pixels)
        actual = m.visual_features(loaded, c, pixels.to(backend)).cpu()
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        assert actual.shape == (48, 32)
        torch.testing.assert_close(
            actual.reshape(8, 2, 3, 32)[:, :, -1],
            loaded.model.image_newline.cpu().expand(8, 2, 32),
            atol=0,
            rtol=0,
        )
        prefix = torch.tensor([[1, 4]])
        x = torch.cat([reference.model.embed_tokens(prefix), expected[None]], dim=1)
        y = torch.cat(
            [loaded.model.embed_tokens(prefix.to(backend)), actual.to(backend)[None]], dim=1
        )
        torch.testing.assert_close(
            loaded(inputs_embeds=y, use_cache=False).logits.cpu(),
            reference(inputs_embeds=x, use_cache=False).logits,
            atol=2e-5,
            rtol=2e-5,
        )


@pytest.mark.parametrize(
    "change",
    [
        {"mm_spatial_pool_mode": "average"},
        {"mm_newline_position": "frame"},
        {"add_faster_video": True},
        {"mm_patch_merge_type": "flat"},
        {"mm_vision_select_layer": -1},
        {"mm_projector_type": "linear"},
        {"vision_tower_pretrained": "remote/model"},
        {"use_pos_skipping": True},
        {"mm_hidden_size": 64},
        {"num_attention_heads": 3},
        {"image_token_index": 260},
        {"auto_map": {"AutoModel": "remote.Code"}},
    ],
)
def test_graph_changes_are_not_silently_ignored(change):
    with pytest.raises(ValueError):
        m.normalized_config(config() | change, m.TASK)


@pytest.mark.parametrize("name", m.METADATA_NAMES)
def test_original_processors_are_closed(name):
    value = m.template(name)
    m.validate_metadata(name, value, config(False))
    value["unsafe_override"] = True
    with pytest.raises(ValueError):
        m.validate_metadata(name, value, config(False))


@pytest.mark.parametrize("missing", [None, *m.REQUIRED_FILES])
def test_original_subset_is_complete(missing):
    from oryn_engine.model_layouts import assess_layout

    files = {*m.REQUIRED_FILES, "model.safetensors", "remote.py", "chat_template.json"}
    if missing:
        files.remove(missing)
    result = assess_layout(sorted(files), config(), m.TASK)
    if missing:
        assert result.artifact_reasons
    else:
        assert not result.artifact_reasons and not result.runtime_reasons
        assert set(result.selected_files) == {*m.REQUIRED_FILES, "model.safetensors"}


def test_original_metadata_discovery_uses_no_weights(tmp_path):
    from test_metadata_audit import row, snapshot

    from oryn_engine.metadata_audit import inspect_row

    raw = m.template("config.json")
    hub = snapshot(
        tmp_path,
        config=raw,
        files=[(n, 100, "b" * 64) for n in (*m.REQUIRED_FILES, "model.safetensors")],
        pipeline_tag=m.TASK,
    )
    hub.client.read_json.side_effect = lambda repository, revision, name: (
        raw if name == "config.json" else m.template(name)
    )
    result = inspect_row(row(task=m.TASK), hub)
    assert result["status"] == "metadata-compatible", result
    assert not result["executionVerified"]
    hub.client.download.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        "vision",
        "postnorm",
        "projector",
        "newline",
        "language",
        "head",
        "extra",
        "integer",
        "nonfinite",
    ],
)
def test_all_original_arrays_including_unused_postnorm_are_required(tmp_path, change):
    c, _, _ = fixture(tmp_path)
    path = tmp_path / "model.safetensors"
    state = load_file(path)
    prefixes = {
        "vision": "model.vision_tower.vision_tower.vision_model.encoder.",
        "postnorm": "model.vision_tower.vision_tower.vision_model.post_layernorm.",
        "projector": "model.mm_projector.",
        "newline": "model.image_newline",
        "language": "model.layers.",
        "head": "lm_head.",
    }
    key = next(iter(state))
    if change in prefixes:
        del state[next(k for k in state if k.startswith(prefixes[change]))]
    elif change == "extra":
        state["wrong.weight"] = torch.ones(2)
    elif change == "integer":
        state[key] = state[key].to(torch.int16)
    else:
        state[key].flatten()[0] = float("nan")
    save_file(state, path)
    with pytest.raises(ValueError):
        m.load_components(tmp_path, c, "cpu")


@pytest.mark.parametrize("change", ["symlink", "duplicate", "merges", "added_id"])
def test_shared_qwen_bpe_rejects_malformed_data(tmp_path, change):
    c, _, _ = fixture(tmp_path)
    p = tmp_path / "vocab.json"
    if change == "symlink":
        p.rename(tmp_path / "outside.json")
        p.symlink_to(tmp_path / "outside.json")
    elif change == "duplicate":
        p.write_text('{"a":0,"a":1}')
    elif change == "merges":
        (tmp_path / "merges.txt").write_text("#version: 0.2\na wrong_token\n")
    else:
        p = tmp_path / "tokenizer_config.json"
        v = json.loads(p.read_text())
        v["added_tokens_decoder"]["999"] = v["added_tokens_decoder"].pop("256")
        p.write_text(json.dumps(v))
    with pytest.raises(ValueError):
        m.load_components(tmp_path, c, "cpu")


@pytest.mark.parametrize(
    "question,times,limit",
    [
        ("<image>", list(range(8)), 4),
        ("<|im_end|>", list(range(8)), 4),
        ("x", [0] * 7 + [float("nan")], 4),
        ("x", list(range(8)), 257),
    ],
)
def test_prompt_scope(tmp_path, question, times, limit):
    c, _, processor = fixture(tmp_path)
    with pytest.raises(ValueError):
        m.prompt_tokens(processor, c, times, 8.0, question, limit)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_offline_complete_video_worker(tmp_path, backend):
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
    assert result["result"]["text"] == "A" and len(result["result"]["sampledTimestamps"]) == 8


def test_answer_requires_eos_and_config_cannot_override_generation(tmp_path):
    from test_coverage_tasks import video

    c, _, _ = fixture(tmp_path, deterministic=True)
    path = tmp_path / "clip.mp4"
    video(path)
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"do_sample": True, "num_beams": 10})
    )
    model, tokenizer, _ = m.load_components(tmp_path, c, "cpu")
    with torch.inference_mode(), pytest.raises(ValueError, match="OUTPUT_INCOMPLETE"):
        m.execute(
            model,
            tokenizer,
            c,
            {"startSeconds": 0, "question": "What moves?", "maxTokens": 1},
            "cpu",
            path,
        )


def test_eight_frame_profile():
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
