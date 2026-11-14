"""Admission regressions; native paired-image evidence is in the sprint harness."""

import copy
from types import SimpleNamespace

import pytest
from test_models import FakeHub, model

from oryn_engine.mlx_vision import validate_vision_config
from oryn_engine.models import ModelStore

CONFIG = {
    "model_type": "qwen2_vl",
    "architectures": ["Qwen2VLForConditionalGeneration"],
    "vision_config": {"depth": 32, "embed_dim": 1280, "num_heads": 16},
    "quantization": {"bits": 4, "group_size": 64},
}


@pytest.mark.parametrize(
    "change",
    [
        {"vision_config": None},
        {"vision_config": {"skip_vision": "yes"}},
        {"model_type": "qwen3_5"},
        {"model_file": "implementation.py"},
        {"quantization": {"bits": 1, "group_size": 64}},
        {"vision_config": {"depth": 10000}},
        {"text_config": {"auto_map": {"AutoModel": "repo.Model"}}},
    ],
)
def test_vision_admission_rejects_incomplete_or_unreviewed_graphs(change):
    with pytest.raises(ValueError):
        validate_vision_config(copy.deepcopy(CONFIG) | change)


@pytest.mark.parametrize("family", ["qwen2_vl", "qwen3_5_moe"])
def test_quantized_vision_discovers_without_a_catalog_row(tmp_path, family):
    config = copy.deepcopy(CONFIG) if family == "qwen2_vl" else tiny_vision_config(family)
    config["quantization"] = {"bits": 4, "group_size": 64}
    config["vision_config"]["skip_vision"] = True
    files = [
        (name, 1024, "a" * 64)
        for name in (
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "preprocessor_config.json",
        )
    ]
    store = ModelStore(
        tmp_path,
        hub=FakeHub(config=config, files=files, pipeline_tag="image-text-to-text"),
        runtime_available=True,
        runtime_planning_state_provider=lambda: ("balanced", 128 * 1024**3),
    )
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda p: None if p.provider else object()
    )
    result = store.inspect(
        model("independent/vision"),
        {
            "task": "image-text-to-text",
            "contract": {"id": "chat-v1", "revision": 1},
            "requiredFeatures": ["multimodalConversation"],
        },
    )
    assert result["executable"]
    assert result["runtimePack"]["packId"] == "oryn.mlx-lm"
    assert result["technicalCapabilities"]["inputModalities"] == ["text", "image"]
    assert result["taskProfile"]["features"]["multimodalConversation"] == "supported"
    assert set(f["name"] for f in result["selectedFiles"]) == set(f[0] for f in files)


@pytest.mark.parametrize(
    "change",
    [
        {"num_experts_per_tok": 5},
        {"num_experts": True},
        {"moe_intermediate_size": 0},
        {"shared_expert_intermediate_size": 2**30},
    ],
)
def test_vision_moe_requires_bounded_complete_expert_geometry(change):
    config = tiny_vision_config("qwen3_5_moe")
    config["text_config"].update(change)
    with pytest.raises(ValueError):
        validate_vision_config(config)


def test_joint_qwen_quantization_checks_the_vision_layers():
    from oryn_engine.runtime_packs.mlx_lm_worker import validate_quantized_layers

    name = "vision_tower.blocks.0.attn.qkv"
    module = SimpleNamespace(weight=SimpleNamespace(shape=(192, 64)), to_quantized=lambda: None)
    model = SimpleNamespace(named_modules=lambda: [(name, module)])
    config = {"model_type": "qwen3_5", "quantization": {"bits": 4, "group_size": 64}}
    tensors = {name + ".weight": (192, 8), name + ".scales": (192, 1), name + ".biases": (192, 1)}
    validate_quantized_layers(model, config, tensors, language_only=False)
    with pytest.raises(ValueError, match="does not match"):
        validate_quantized_layers(
            model, config, tensors | {name + ".weight": (192, 9)}, language_only=False
        )
    with pytest.raises(ValueError, match="unknown native layer"):
        validate_quantized_layers(
            model, config, tensors | {"vision_tower.unknown.scales": (192, 1)}, language_only=False
        )


def tiny_vision_config(family):
    text = {
        "model_type": family + "_text",
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "vocab_size": 64,
        "head_dim": 32,
        "rope_theta": 10000,
        "max_position_embeddings": 1024,
        "rope_scaling": {"type": "default", "mrope_section": [6, 5, 5]},
    }
    if family in {"qwen3_5", "qwen3_5_moe"}:
        text.update(
            linear_num_value_heads=2,
            linear_num_key_heads=2,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_conv_kernel_dim=4,
            full_attention_interval=2,
            rope_parameters={
                "type": "default",
                "mrope_section": [6, 5, 5],
                "rope_theta": 10000,
                "partial_rotary_factor": 1.0,
            },
        )
    if family == "qwen3_5_moe":
        text.pop("intermediate_size")
        text.update(
            num_experts=4,
            num_experts_per_tok=2,
            shared_expert_intermediate_size=128,
            moe_intermediate_size=128,
        )
    return {
        "model_type": family,
        "text_config": text,
        "vision_config": {
            "depth": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "out_hidden_size": 64,
            "num_heads": 2,
            "patch_size": 16,
            "spatial_patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16,
        },
        "image_token_id": 61,
        "video_token_id": 62,
        "vision_start_token_id": 60,
        "vision_end_token_id": 63,
    }


@pytest.mark.parametrize("family", ["qwen3_vl", "qwen3_5", "qwen3_5_moe"])
@pytest.mark.parametrize("quantized", [False, True])
def test_native_joint_vision_safe_load_preserves_image_sensitive_logits(
    tmp_path, family, quantized
):
    import importlib
    import json

    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    pytest.importorskip("mlx_vlm")
    from mlx.utils import tree_flatten

    from oryn_engine.mlx_vision import load_model

    config = tiny_vision_config(family)
    native = importlib.import_module("mlx_vlm.models." + family)
    mx.random.seed(61)
    model = native.Model(native.ModelConfig.from_dict(copy.deepcopy(config)))
    if quantized:
        config["quantization"] = {"bits": 4, "group_size": 64}
        nn.quantize(model, bits=4, group_size=64)
    model.eval()
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(json.dumps(config))
    loaded = load_model(tmp_path, config)
    inputs = mx.array([[60, 61, 63, 4]])
    options = {"image_grid_thw": mx.array([[1, 2, 2]]), "mask": mx.ones((1, 4), dtype=mx.int32)}
    outputs = []
    for fill in [0.0, 1.0]:
        pixels = mx.full((4, 3 * 2 * 16 * 16), fill)
        reference = model(inputs, pixel_values=pixels, **options).logits
        actual = loaded(inputs, pixel_values=pixels, **options).logits
        mx.eval(reference, actual)
        assert bool(mx.allclose(actual, reference, atol=1e-5, rtol=1e-5))
        assert bool(mx.all(mx.isfinite(actual)))
        outputs.append(actual)
    assert not bool(mx.allclose(outputs[0], outputs[1], atol=1e-5, rtol=1e-5))
    missing = dict(weights)
    del missing[next(k for k in weights if k.startswith("vision_tower."))]
    mx.save_safetensors(str(tmp_path / "model.safetensors"), missing)
    with pytest.raises(ValueError):
        load_model(tmp_path, config)


def test_native_qwen35_sanitizes_norms_with_the_complete_graph():
    mx = pytest.importorskip("mlx.core")
    native = pytest.importorskip("mlx_vlm.models.qwen3_5")
    from oryn_engine.mlx_vision import sanitize_weights

    model = native.Model(native.ModelConfig.from_dict(tiny_vision_config("qwen3_5")))
    norm = "language_model.model.norm.weight"
    conv = "language_model.model.layers.0.linear_attn.conv1d.weight"
    weights = {norm: mx.zeros((64,)), conv: mx.zeros((64, 1, 4))}
    result = sanitize_weights(model, weights)
    assert bool(mx.all(result[norm] == 1.0))
    assert result[conv].shape == (64, 4, 1)
    with pytest.raises(ValueError, match="ambiguous"):
        sanitize_weights(
            model,
            {
                "model.visual.proj.weight": mx.zeros((2, 2)),
                "vision_tower.proj.weight": mx.zeros((2, 2)),
            },
        )


@pytest.mark.parametrize(
    "change",
    [
        {"linear_key_head_dim": 16},
        {"linear_value_head_dim": 31},
        {"linear_num_key_heads": 0},
        {"linear_num_key_heads": 3},
    ],
)
def test_qwen35_recurrent_geometry_is_rejected_before_native_kernels(change):
    config = tiny_vision_config("qwen3_5")
    config["text_config"].update(change)
    with pytest.raises(ValueError, match="recurrent"):
        validate_vision_config(config)


def test_qwen35_chat_uses_one_current_image_and_answer_framing():
    from oryn_engine.mlx_vision import image_prompt

    messages = [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "What color?"},
    ]
    prompt = image_prompt(messages, "qwen3_5")
    assert prompt.count("<|image_pad|>") == 1
    assert "<|vision_end|>What color?" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert image_prompt(messages, "qwen2_5_vl").endswith("<|im_start|>assistant\n")
