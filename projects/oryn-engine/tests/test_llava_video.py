import copy
import json

import numpy as np
import pytest
import torch
from PIL import Image

from oryn_engine import llava_video as binding
from oryn_engine import shared_tasks


def tiny_model(root):
    from tokenizers import processors
    from transformers import (
        CLIPVisionConfig,
        LlamaConfig,
        LlamaTokenizer,
        LlavaNextVideoConfig,
        LlavaNextVideoForConditionalGeneration,
    )

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<video>": 3, "<image>": 4}
    for char in "▁abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:?.\n":
        vocab.setdefault(char, len(vocab))
    tokenizer = LlamaTokenizer(
        vocab=vocab, pad_token="<unk>", add_bos_token=True, add_eos_token=False
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["<video>", "<image>"]})
    tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", 1)]
    )
    tokenizer.save_pretrained(root)
    config = LlavaNextVideoConfig(
        vision_config=CLIPVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=32,
            patch_size=8,
        ),
        text_config=LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=len(tokenizer),
            max_position_embeddings=256,
            pad_token_id=0,
        ),
        image_token_index=4,
        video_token_index=3,
        vision_feature_layer=-2,
        image_grid_pinpoints=[[32, 32]],
    )
    torch.manual_seed(61)
    native = LlavaNextVideoForConditionalGeneration(config).eval()
    native.save_pretrained(root)
    (root / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "image_processor_type": "LlavaNextImageProcessor",
                "processor_class": "LlavaNextVideoProcessor",
                "size": {"shortest_edge": 32},
                "crop_size": {"height": 32, "width": 32},
                "image_mean": binding.MEAN,
                "image_std": binding.STD,
            }
        )
    )
    (root / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "LlavaNextVideoProcessor",
                "patch_size": 8,
                "vision_feature_select_strategy": "default",
                "num_additional_image_tokens": 1,
            }
        )
    )
    (root / "chat_template.jinja").write_text("{{ raise_exception('must not render') }}")
    # Repository generation settings cannot override the owned generation policy.
    (root / "generation_config.json").write_text(json.dumps({"do_sample": True, "num_beams": 3}))
    return native, tokenizer


def frames():
    return [
        Image.fromarray(np.full((32, 32, 3), [20 + i * 20, 40, 60], np.uint8)) for i in range(8)
    ]


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_native_joint_video_logits_and_temporal_order(tmp_path, backend):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    native, tokenizer = tiny_model(tmp_path)
    model, processor, config = shared_tasks.load_task(tmp_path, "video-text-to-text", backend)
    inputs = binding.prepare_inputs(processor, config, frames(), "What changes?", 4)
    # Construct the reference prompt expansion and pixel normalization independently.
    ids = tokenizer("USER: " + "<video>" * 32 + "\nWhat changes? ASSISTANT:", return_tensors="pt")
    torch.testing.assert_close(inputs["input_ids"], ids["input_ids"])
    assert int(inputs["input_ids"][0, 0]) == 1
    pixels = (
        torch.stack([torch.tensor(np.array(f)).permute(2, 0, 1) for f in frames()]).float() / 255
    )
    pixels = (pixels - torch.tensor(binding.MEAN)[:, None, None]) / torch.tensor(binding.STD)[
        :, None, None
    ]
    torch.testing.assert_close(inputs["pixel_values_videos"], pixels[None], atol=1e-6, rtol=1e-6)
    with torch.inference_mode():
        expected = native(**ids, pixel_values_videos=pixels[None], use_cache=False).logits
        actual = model(
            **{k: v.to(backend) for k, v in inputs.items()}, use_cache=False
        ).logits.cpu()
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        reversed_inputs = inputs | {"pixel_values_videos": inputs["pixel_values_videos"].flip(1)}
        reversed_logits = native(**reversed_inputs, use_cache=False).logits
        assert not torch.allclose(expected[:, -1], reversed_logits[:, -1], atol=1e-7, rtol=1e-7)
    assert processor.chat_template is None and processor.tokenizer.chat_template is None


@pytest.mark.parametrize("part", ["vision", "projector", "language", "head", "extra", "shape"])
def test_complete_native_graph_is_required(tmp_path, part):
    from safetensors.torch import load_file, save_file

    tiny_model(tmp_path)
    path = tmp_path / "model.safetensors"
    state = load_file(path)
    match = {
        "vision": "vision_tower",
        "projector": "multi_modal_projector",
        "language": "embed_tokens",
        "head": "lm_head",
    }
    if part in match:
        del state[next(key for key in state if match[part] in key)]
    elif part == "extra":
        state["custom_head.weight"] = torch.zeros(2, 2)
    else:
        key = next(
            key for key in state if "multi_modal_projector" in key and key.endswith("weight")
        )
        state[key] = torch.zeros(2, 2)
    save_file(state, path)
    with pytest.raises((ValueError, RuntimeError)):
        shared_tasks.load_task(tmp_path, "video-text-to-text", "cpu")


@pytest.mark.parametrize(
    "change",
    [
        {"spatial_pool_mode": "conv"},
        {"spatial_pool_stride": 1},
        {"vision_feature_select_strategy": "full"},
        {"vision_feature_layer": [0, 1]},
        {"auto_map": {"AutoModel": "custom.Model"}},
        {"video_token_index": 4},
    ],
)
def test_unreviewed_video_graph_is_rejected(tmp_path, change):
    tiny_model(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text()) | change
    with pytest.raises(ValueError):
        shared_tasks.validate_task_config("video-text-to-text", config)


@pytest.mark.parametrize(
    "name,change",
    [
        ("preprocessor_config.json", {"do_center_crop": False}),
        ("preprocessor_config.json", {"size": {"shortest_edge": 224}}),
        ("processor_config.json", {"patch_size": 14}),
        ("tokenizer_config.json", {"add_bos_token": False}),
        ("tokenizer_config.json", {"tokenizer_class": "CustomTokenizer"}),
    ],
)
def test_metadata_and_runtime_reject_incompatible_processing(tmp_path, name, change):
    tiny_model(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    path = tmp_path / name
    metadata = json.loads(path.read_text()) | change
    with pytest.raises(ValueError):
        binding.validate_metadata(name, metadata, config)
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        shared_tasks.load_task(tmp_path, "video-text-to-text", "cpu")


def test_bounded_input_layout_and_original_metadata(tmp_path):
    from oryn_engine.model_layouts import assess_layout

    tiny_model(tmp_path)
    _, processor, config = shared_tasks.load_task(tmp_path, "video-text-to-text", "cpu")
    files = [p.name for p in tmp_path.iterdir()]
    layout = assess_layout(files, config, "video-text-to-text")
    assert not layout.runtime_reasons and not layout.artifact_reasons
    for name in binding.METADATA_NAMES:
        assert assess_layout(
            [f for f in files if f != name], config, "video-text-to-text"
        ).artifact_reasons
    with pytest.raises(ValueError, match="TOKEN_LIMIT"):
        binding.prepare_inputs(processor, config, frames(), "What " * 80, 128)
    with pytest.raises(ValueError, match="RESERVED_TOKEN"):
        binding.prepare_inputs(processor, config, frames(), "<video>", 4)
    with pytest.raises(ValueError, match="INPUT_GEOMETRY"):
        binding.prepare_inputs(processor, config, frames()[:4], "What changes?", 4)
    # Pinned legacy Vicuna configuration omits native Llama dimensions.
    legacy = copy.deepcopy(config)
    for key in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
    ):
        legacy["text_config"].pop(key, None)
    binding.validate_config(legacy)
    assert binding.text_config(legacy)["hidden_size"] == 4096


def test_full_shared_video_result_and_owned_generation(tmp_path):
    from oryn_protocol.task_contracts import validate_result
    from test_coverage_tasks import video

    from oryn_engine.task_profiles import profile_for_execution

    tiny_model(tmp_path)
    path = tmp_path / "clip.mp4"
    video(path)
    model, processor, config = shared_tasks.load_task(tmp_path, "video-text-to-text", "cpu")
    value = {"question": "What changes?", "maxTokens": 4, "startSeconds": 0}
    with torch.inference_mode():
        first = binding.execute(model, processor, config, value, "cpu", path)
        second = binding.execute(model, processor, config, value, "cpu", path)
    assert first == second
    assert len(first["sampledTimestamps"]) == 8 and first["durationSeconds"] == 2
    shared_tasks.validate_task_output("video-text-to-text", first)
    value["videoArtifactId"] = "artifact_aaaaaaaaaaaaaaaaaaaaaaaa"
    result = shared_tasks.execute_task(
        tmp_path, "video-text-to-text", value, "cpu", path, None, lambda *_: None
    )
    assert result == first
    profile = profile_for_execution(
        {
            "technicalCapabilities": {"task": "video-text-to-text"},
            "runtimePack": {"runnerId": "transformers.shared-tasks"},
        },
        config,
    )
    validate_result(profile, {"kind": "structured", "value": result}, value)
