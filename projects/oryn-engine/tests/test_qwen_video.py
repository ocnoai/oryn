"""Original metadata plus complete synthetic native video graphs; no original weights."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from oryn_engine import coverage_tasks, shared_tasks

FIXTURES = Path(__file__).parent / "fixtures/qwen_video"


def original(name):
    return json.loads((FIXTURES / name).read_text())


def test_original_complete_graph_and_processors():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = original("config.json")
    shared_tasks.validate_task_config("video-text-to-text", config)
    with torch.device("meta"):
        native = Qwen3VLForConditionalGeneration(Qwen3VLConfig.from_dict(config))
    shapes = {k: list(v.shape) for k, v in native.state_dict().items()}
    assert shapes == {k: v["shape"] for k, v in original("original-shapes.json").items()}
    image = original("preprocessor_config.json")
    with pytest.raises(ValueError):
        shared_tasks.qwen_image_options(image, config["vision_config"])
    assert (
        shared_tasks.qwen_image_options(image, config["vision_config"], video_only=True)[
            "do_resize"
        ]
        is False
    )
    options = shared_tasks.qwen_video_options(
        original("video_preprocessor_config.json"), config["vision_config"], "qwen3_vl"
    )
    assert options["image_mean"] == options["image_std"] == [0.5] * 3
    assert options["patch_size"] == 16 and options["do_resize"] is True
    assert options["max_frames"] == options["min_frames"] == 8
    assert options["do_sample_frames"] is False
    assert options["size"]["longest_edge"] == 256 * 28**2


@pytest.mark.parametrize(
    "change",
    [
        {"video_processor_type": "Qwen2VLVideoProcessor"},
        {"do_resize": 0},
        {"patch_size": 14},
        {"temporal_patch_size": 1},
        {"merge_size": 4},
        {"image_std": [0, 0.5, 0.5]},
        {"image_mean": [0.5]},
        {"do_normalize": False},
        {"do_center_crop": True},
        {"do_rescale": False},
        {"resample": 2},
        {"fps": float("nan")},
        {"fps": True},
        {"min_frames": 0},
        {"max_frames": 10000},
        {"min_frames": 8, "max_frames": 4},
        {"num_frames": 1.5},
        {"do_sample_frames": 1},
        {"max_video_tokens": 100000},
        {"video_metadata": {"fps": 1}},
        {"auto_map": {"Processor": "repo.Custom"}},
        {"size": {"shortest_edge": 1, "longest_edge": 2**40}},
        {"unknown_override": True},
    ],
)
def test_active_video_overrides_are_rejected(change):
    config = original("video_preprocessor_config.json") | change
    with pytest.raises((ValueError, TypeError)):
        shared_tasks.qwen_video_options(
            config, original("config.json")["vision_config"], "qwen3_vl"
        )


@pytest.fixture
def tiny(tmp_path):
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import Qwen2Tokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration

    old = torch.get_num_threads()
    torch.set_num_threads(1)
    vocab = {token: i for i, token in enumerate(sorted(ByteLevel.alphabet()))}
    markers = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
    ]
    for token in markers:
        vocab[token] = len(vocab)
    tokenizer = Qwen2Tokenizer(vocab=vocab, merges=[])
    tokenizer.add_special_tokens({"additional_special_tokens": markers})
    tokenizer.save_pretrained(tmp_path)
    config = original("config.json")
    config["text_config"].update(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=len(vocab),
        max_position_embeddings=1024,
        bos_token_id=vocab["<|endoftext|>"],
        eos_token_id=vocab["<|im_end|>"],
    )
    config["text_config"]["rope_scaling"]["mrope_section"] = [1, 1, 2]
    config["vision_config"].update(
        hidden_size=16,
        intermediate_size=32,
        depth=3,
        num_heads=2,
        out_hidden_size=16,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0, 1, 2],
    )
    for token, key in [
        ("<|image_pad|>", "image_token_id"),
        ("<|video_pad|>", "video_token_id"),
        ("<|vision_start|>", "vision_start_token_id"),
        ("<|vision_end|>", "vision_end_token_id"),
        ("<|im_end|>", "eos_token_id"),
        ("<|endoftext|>", "pad_token_id"),
    ]:
        config[key] = vocab[token]
    native_config = Qwen3VLConfig.from_dict(copy.deepcopy(config))
    for part in [native_config, native_config.text_config, native_config.vision_config]:
        part._attn_implementation = "eager"
    torch.manual_seed(6150)
    native = Qwen3VLForConditionalGeneration(native_config).float().eval()
    native.save_pretrained(tmp_path, max_shard_size="100KB")
    (tmp_path / "config.json").write_text(json.dumps(config))
    for name in ["preprocessor_config.json", "video_preprocessor_config.json"]:
        (tmp_path / name).write_text(json.dumps(original(name)))
    (tmp_path / "chat_template.jinja").write_text("{{ raise_exception('never render') }}")
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"do_sample": True, "num_beams": 5, "repetition_penalty": 10.0})
    )
    yield tmp_path, config, native, tokenizer
    torch.set_num_threads(old)


def frames():
    return [
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (65, 81, 3), dtype=np.uint8))
        for i in range(8)
    ]


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("resize", [False, True])
def test_native_complete_graph_pixels_prompt_logits_and_generation(tiny, backend, resize):
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import functional as tv
    from transformers import GenerationConfig, Qwen3VLVideoProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    root, config, native, tokenizer = tiny
    metadata = original("video_preprocessor_config.json") | {"do_resize": resize}
    (root / "video_preprocessor_config.json").write_text(json.dumps(metadata))
    loaded, processor, selected = shared_tasks.load_task(root, "video-text-to-text", backend)
    assert selected == config
    assert processor.chat_template is None and processor.tokenizer.chat_template is None
    for name, tensor in native.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name].cpu(), tensor, rtol=0, atol=0)
    images = frames()
    times = [3 + i * 0.25 for i in range(8)]
    question = "Čo sa mení?"
    actual = coverage_tasks.qwen3_video_inputs(processor, config, images, times, question, 4)
    # Independent publisher-style external resize followed by do_resize=False.
    raw = torch.stack([torch.from_numpy(np.array(f)).permute(2, 0, 1) for f in images])
    height, width = smart_resize(
        8, 65, 81, temporal_factor=2, factor=32, min_pixels=56**2, max_pixels=256 * 28**2
    )
    resized = tv.resize(
        raw, [height, width], interpolation=InterpolationMode.BICUBIC, antialias=True
    )
    reference_pixels = dict(
        Qwen3VLVideoProcessor(
            do_resize=False,
            do_sample_frames=False,
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
            patch_size=16,
            merge_size=2,
            temporal_patch_size=2,
            cap_pixels_per_frame=True,
        )(videos=[resized], return_tensors="pt", return_metadata=False)
    )
    for key in reference_pixels:
        torch.testing.assert_close(actual[key], reference_pixels[key], rtol=0, atol=0)
    grid = reference_pixels["video_grid_thw"][0]
    blocks = [
        f"<{3.125 + i * 0.5:.1f} seconds><|vision_start|>"
        + "<|video_pad|>" * (int(grid[1] * grid[2]) // 4)
        + "<|vision_end|>"
        for i in range(4)
    ]
    prompt = (
        "<|im_start|>user\n" + "".join(blocks) + question + "<|im_end|>\n<|im_start|>assistant\n"
    )
    reference_tokens = dict(
        tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt", return_token_type_ids=False
        )
    )
    for key in reference_tokens:
        torch.testing.assert_close(actual[key], reference_tokens[key])
    assert processor.tokenizer.decode(actual["input_ids"][0]).count("<|vision_start|>") == 4
    expected = reference_tokens | reference_pixels
    expected["mm_token_type_ids"] = (expected["input_ids"] == config["video_token_id"]).long() * 2
    generation = GenerationConfig(
        do_sample=False,
        num_beams=1,
        use_cache=True,
        eos_token_id=config["eos_token_id"],
        pad_token_id=config["pad_token_id"],
    )
    with torch.inference_mode():
        logits = native(**expected, use_cache=False).logits
        actual_logits = loaded(
            **{k: v.to(backend) for k, v in actual.items()}, use_cache=False
        ).logits.cpu()
        torch.testing.assert_close(actual_logits, logits, rtol=4e-4, atol=4e-5)
        reverse = coverage_tasks.qwen3_video_inputs(
            processor, config, images[::-1], times, question, 4
        )
        reverse_logits = native(**reverse, use_cache=False).logits
        assert not torch.allclose(reverse_logits[:, -1], logits[:, -1], rtol=1e-7, atol=1e-7)
        wanted = native.generate(**expected, max_new_tokens=4, generation_config=generation)
        got = loaded.generate(
            **{k: v.to(backend) for k, v in actual.items()},
            max_new_tokens=4,
            generation_config=generation,
        ).cpu()
        torch.testing.assert_close(got, wanted, rtol=0, atol=0)


@pytest.mark.parametrize(
    "damage", ["missing-vision", "missing-deepstack", "missing-language", "extra", "shape"]
)
def test_complete_native_sharded_graph_is_required(tiny, damage):
    root, _, _, _ = tiny
    index = json.loads((root / "model.safetensors.index.json").read_text())
    mapping = index["weight_map"]
    match = {
        "missing-vision": "patch_embed",
        "missing-deepstack": "deepstack_merger_list",
        "missing-language": "embed_tokens",
        "shape": "patch_embed",
    }
    key = next(k for k in mapping if match.get(damage, "patch_embed") in k)
    path = root / mapping[key]
    state = load_file(path)
    if damage.startswith("missing"):
        del state[key]
        del mapping[key]
    elif damage == "extra":
        state["unreviewed_head.weight"] = torch.zeros(2, 2)
        mapping["unreviewed_head.weight"] = path.name
    else:
        state[key] = torch.zeros(2, 2)
    save_file(state, path)
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises((ValueError, RuntimeError)):
        shared_tasks.load_task(root, "video-text-to-text", "cpu")


@pytest.mark.parametrize(
    "damage", ["framing", "empty", "times", "count", "tokens", "budget", "marker"]
)
def test_temporal_and_token_bounds(tiny, damage):
    root, config, _, _ = tiny
    _, processor, _ = shared_tasks.load_task(root, "video-text-to-text", "cpu")
    images, times, question, maximum = frames(), [i / 4 for i in range(8)], "question", 4
    if damage == "framing":
        question = "<|video_pad|>"
    elif damage == "empty":
        question = " "
    elif damage == "times":
        times[2] = times[1]
    elif damage == "count":
        images.pop()
    elif damage == "tokens":
        maximum = 513
    elif damage == "budget":
        config["text_config"]["max_position_embeddings"] = 32
    else:
        config["video_token_id"] = 0
    with pytest.raises(ValueError):
        coverage_tasks.qwen3_video_inputs(processor, config, images, times, question, maximum)


def test_metadata_discovery_uses_configuration_and_reads_video_settings(tmp_path):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            self.reads.append(filename)
            return copy.deepcopy(self.metadata[filename])

    names = [
        "config.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ]
    hub = Hub(
        config=original("config.json"),
        pipeline_tag="video-text-to-text",
        files=[(n, 1000, "a" * 64) for n in names + ["model.safetensors", "tokenizer.json"]],
    )
    hub.metadata = {n: original(n) for n in names}
    hub.reads = []
    store = ModelStore(
        tmp_path,
        hub=hub,
        runtime_available=True,
        runtime_planning_state_provider=lambda: ("balanced", 64_000_000_000),
    )
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda pack: None if pack.provider else object()
    )
    selected = {"provider": "huggingface", "repository": "renamed/native-video", "revision": "main"}
    result = store.inspect(selected)
    assert result["executable"], result
    assert result["runtimePack"]["taskContract"] == "video-question-answering-v1"
    assert "video_preprocessor_config.json" in hub.reads and not hub.downloaded
    hub.metadata["video_preprocessor_config.json"]["patch_size"] = 14
    assert not store.inspect(selected)["executable"]


@pytest.mark.parametrize("damage", ["tied-alias", "nonfinite", "integer"])
def test_original_numeric_values_and_tied_aliases_are_checked(tiny, damage):
    root, _, _, _ = tiny
    index = json.loads((root / "model.safetensors.index.json").read_text())
    key = "model.language_model.embed_tokens.weight"
    path = root / index["weight_map"][key]
    state = load_file(path)
    if damage == "tied-alias":
        state["lm_head.weight"] = state[key].clone() + 1
        index["weight_map"]["lm_head.weight"] = path.name
    elif damage == "nonfinite":
        state[key][0, 0] = float("nan")
    else:
        state[key] = state[key].to(torch.int32)
    save_file(state, path)
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError):
        shared_tasks.load_task(root, "video-text-to-text", "cpu")


def test_optional_video_metadata_keeps_native_defaults_and_model_patch():
    from transformers import Qwen3VLVideoProcessor

    options = shared_tasks.qwen_video_options(
        {}, original("config.json")["vision_config"], "qwen3_vl"
    )
    native = Qwen3VLVideoProcessor(**options)
    assert native.patch_size == 16
    assert list(native.image_mean) == list(native.image_std) == [0.5] * 3
    assert native.do_sample_frames is False and native.max_frames == 8
    with pytest.raises(ValueError):
        shared_tasks.qwen_video_options(
            original("video_preprocessor_config.json") | {"processor_class": "Unknown"},
            original("config.json")["vision_config"],
            "qwen3_vl",
        )


@pytest.mark.parametrize("text", ["A clear answer.", "", "x" * 16385])
def test_public_video_result_and_generation_policy(tiny, monkeypatch, text):
    root, config, _, tokenizer = tiny
    model, processor, _ = shared_tasks.load_task(root, "video-text-to-text", "cpu")
    assert model.generation_config.do_sample is False
    assert model.generation_config.num_beams == 1
    assert model.generation_config.repetition_penalty in {None, 1.0}
    times = [3 + i / 4 for i in range(8)]
    monkeypatch.setattr(coverage_tasks, "sample_video", lambda *args: (frames(), times, 6.0))
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        assert kwargs["max_new_tokens"] == 128
        assert kwargs["generation_config"].do_sample is False
        assert kwargs["generation_config"].num_beams == 1
        decoded = tokenizer.decode(kwargs["input_ids"][0])
        assert "<3.1 seconds>" in decoded and "<0.1 seconds>" not in decoded
        return torch.cat(
            [
                kwargs["input_ids"],
                torch.tensor([tokenizer.encode(text, add_special_tokens=False)], dtype=torch.long),
            ],
            dim=1,
        )

    monkeypatch.setattr(model, "generate", generate)
    value = shared_tasks.validate_task_input(
        "video-text-to-text",
        {
            "videoArtifactId": "artifact_aaaaaaaaaaaaaaaaaaaaaaaa",
            "question": "Čo sa mení?",
            "startSeconds": 3,
        },
    )
    if not text or len(text) > 16384:
        with pytest.raises(ValueError):
            coverage_tasks.video_answer(model, processor, config, value, "cpu", root / "video.mp4")
    else:
        result = coverage_tasks.video_answer(
            model, processor, config, value, "cpu", root / "video.mp4"
        )
        shared_tasks.validate_task_output("video-text-to-text", result)
        assert result == {"text": text, "sampledTimestamps": times, "durationSeconds": 6.0}
    assert len(calls) == 1
