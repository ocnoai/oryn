"""Complete native LayoutLMv2 document QA with the owned ResNeXt-FPN backbone."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import transformers as tf
from PIL import Image

from oryn_engine import numeric_archive_tasks as archive
from oryn_engine import numeric_layoutlmv2 as lv2
from oryn_engine import shared_tasks
from oryn_engine.numeric_torch_archive import inspect_archive

FIXTURES = Path(__file__).parent / "fixtures/numeric_layoutlmv2"
ORIGINAL = json.loads((FIXTURES / "docvqa.config.json").read_text())
TASK = "document-question-answering"
ARTIFACT = "artifact_" + "a" * 24
WORDS = ["invoice", "number", "a12", "total", "due", "42"]
BOXES = [
    [0.1, 0.1, 0.2, 0.15],
    [0.22, 0.1, 0.32, 0.15],
    [0.35, 0.1, 0.45, 0.15],
    [0.1, 0.3, 0.2, 0.35],
    [0.22, 0.3, 0.3, 0.35],
    [0.33, 0.3, 0.4, 0.35],
]


def tiny_config():
    config = copy.deepcopy(ORIGINAL)
    config["detectron2_config_args"].update(
        {
            "MODEL.RESNETS.DEPTH": 50,
            "MODEL.RESNETS.NUM_GROUPS": 1,
            "MODEL.RESNETS.WIDTH_PER_GROUP": 4,
        }
    )
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "what", "is", "the", "?", *WORDS]
    config.update(
        hidden_size=24,
        coordinate_size=4,
        shape_size=4,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=64,
        vocab_size=len(words),
    )
    return config, words


def pixels(width=96, height=80):
    return Image.fromarray(
        ((np.arange(width * height * 3) * 31 + 7) % 256).astype(np.uint8).reshape(height, width, 3)
    )


def fixture(root, legacy=True):
    config, words = tiny_config()
    root.mkdir(parents=True, exist_ok=True)
    (root / "vocab.txt").write_text("\n".join(words) + "\n")
    tf.LayoutLMv2Tokenizer(vocab=str(root / "vocab.txt")).save_pretrained(root)
    tf.LayoutLMv2ImageProcessor(apply_ocr=False).save_pretrained(root)
    (root / "config.json").write_text(json.dumps(config))
    torch.manual_seed(6171)
    model = lv2.native_model(config).eval()
    with torch.no_grad():
        for module in model.modules():
            if type(module).__name__ == "FrozenBatchNorm":
                module.weight.uniform_(0.5, 1.5)
                module.bias.uniform_(-0.1, 0.1)
                module.running_mean.uniform_(-0.1, 0.1)
                module.running_var.uniform_(0.5, 1.5)
    state = dict(model.state_dict())
    if legacy:
        # Historical exports persisted these fixed buffers.
        state[lv2.POSITIONS] = torch.arange(64).expand(1, -1)
        state[lv2.PIXELS[0]] = model.layoutlmv2.visual.pixel_mean.clone()
        state[lv2.PIXELS[1]] = model.layoutlmv2.visual.pixel_std.clone()
    torch.save(state, root / "pytorch_model.bin")
    return model, config


def test_original_header_maps_onto_complete_owned_graph():
    lv2.validate_config(ORIGINAL)
    with torch.device("meta"):
        native = lv2.native_model(ORIGINAL)
    expected = lv2.expected_shapes(ORIGINAL)
    assert expected == {k: list(v.shape) for k, v in native.state_dict().items()}
    header = json.loads((FIXTURES / "docvqa.header.json").read_text())
    metadata = {
        "format": "torch-numeric-zip-v1",
        "byteOrder": "little",
        "tensors": {
            k: {"shape": v, "dtype": "I64" if k.endswith("position_ids") else "F32"}
            for k, v in header.items()
        },
    }
    assert set(lv2.validate_archive(ORIGINAL, metadata)) == set(header)
    assert len(header) == 716 and sum(".bottom_up.res4." in k for k in header) == 23 * 15 + 5


def test_owned_bottom_up_matches_independent_torchvision_resnet():
    from torchvision.models.resnet import Bottleneck, ResNet

    config, _ = tiny_config()
    model = lv2.native_model(config).eval()
    bottom_up = model.layoutlmv2.visual.backbone.bottom_up
    with torch.no_grad():
        for module in bottom_up.modules():
            if type(module).__name__ == "FrozenBatchNorm":
                module.weight.uniform_(0.5, 1.5)
                module.bias.uniform_(-0.1, 0.1)
                module.running_mean.uniform_(-0.1, 0.1)
                module.running_var.uniform_(0.5, 1.5)
    reference = ResNet(Bottleneck, [3, 4, 6, 3], groups=1, width_per_group=4).eval()
    source = bottom_up.state_dict()
    mapped = {}

    def norm(target, prefix):
        for part in ("weight", "bias", "running_mean", "running_var"):
            mapped[f"{target}.{part}"] = source[f"{prefix}.norm.{part}"]

    mapped["conv1.weight"] = source["stem.conv1.weight"]
    norm("bn1", "stem.conv1")
    for stage, (layer, count) in enumerate(zip(range(1, 5), (3, 4, 6, 3), strict=True), 2):
        for index in range(count):
            prefix, target = f"res{stage}.{index}", f"layer{layer}.{index}"
            for conv in (1, 2, 3):
                mapped[f"{target}.conv{conv}.weight"] = source[f"{prefix}.conv{conv}.weight"]
                norm(f"{target}.bn{conv}", f"{prefix}.conv{conv}")
            if f"{prefix}.shortcut.weight" in source:
                mapped[f"{target}.downsample.0.weight"] = source[f"{prefix}.shortcut.weight"]
                norm(f"{target}.downsample.1", f"{prefix}.shortcut")
    for name, value in reference.state_dict().items():
        if name.startswith("fc.") or name.endswith("num_batches_tracked"):
            mapped[name] = value
    reference.load_state_dict(mapped, strict=True)
    x = torch.randn(1, 3, 224, 224, generator=torch.Generator().manual_seed(3)) * 50
    with torch.inference_mode():
        actual = bottom_up(x)
        y = reference.maxpool(reference.relu(reference.bn1(reference.conv1(x))))
        for features, layer in zip(
            actual,
            (reference.layer1, reference.layer2, reference.layer3, reference.layer4),
            strict=True,
        ):
            y = layer(y)
            torch.testing.assert_close(features, y, atol=1e-4, rtol=1e-4)


def test_fpn_top_down_pathway_and_pooled_visual_tokens():
    config, _ = tiny_config()
    model = lv2.native_model(config).eval()
    backbone = model.layoutlmv2.visual.backbone
    x = torch.randn(1, 3, 224, 224, generator=torch.Generator().manual_seed(4))
    with torch.inference_mode():
        res2, res3, res4, res5 = backbone.bottom_up(x)
        top = backbone.fpn_lateral5(res5)
        for lateral, feature in (
            (backbone.fpn_lateral4, res4),
            (backbone.fpn_lateral3, res3),
            (backbone.fpn_lateral2, res2),
        ):
            top = lateral(feature) + top.repeat_interleave(2, -1).repeat_interleave(2, -2)
        torch.testing.assert_close(backbone(x), backbone.fpn_output2(top))
        tokens = model.layoutlmv2.visual(x * 0 + 128)
    assert tokens.shape == (1, 49, 256)


@pytest.mark.parametrize("legacy", [True, False])
def test_native_values_and_grounded_document_answer(tmp_path, legacy):
    from oryn_engine.coverage_tasks import validate_relations

    reference, _ = fixture(tmp_path, legacy)
    model, processor, config = shared_tasks.load_task(tmp_path, TASK, "cpu")
    expected = reference.state_dict()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
    value = dict(imageArtifactId=ARTIFACT, question="what is the total ?", words=WORDS, boxes=BOXES)
    validate_relations(TASK, value)
    with torch.inference_mode():
        first = archive.execute(
            model, processor, config, TASK, value, "cpu", pixels(), None, (96, 80)
        )
        second = archive.execute(
            model, processor, config, TASK, value, "cpu", pixels(), None, (96, 80)
        )
    assert first == second
    shared_tasks.validate_task_output(TASK, first)
    assert first["answer"] == " ".join(WORDS[first["wordStart"] : first["wordEnd"] + 1])


@pytest.mark.parametrize(
    "damage", ["backbone", "fpn", "head", "pixel-mean", "position", "extra", "nonfinite"]
)
def test_incomplete_or_altered_graphs_are_rejected(tmp_path, damage):
    fixture(tmp_path)
    state = torch.load(tmp_path / "pytorch_model.bin", weights_only=True)
    if damage == "backbone":
        del state["layoutlmv2.visual.backbone.bottom_up.res3.0.shortcut.norm.running_var"]
    elif damage == "fpn":
        del state["layoutlmv2.visual.backbone.fpn_output5.weight"]
    elif damage == "head":
        del state["qa_outputs.weight"]
    elif damage == "pixel-mean":
        state[lv2.PIXELS[0]] = state[lv2.PIXELS[0]] + 1
    elif damage == "position":
        state[lv2.POSITIONS] = torch.ones_like(state[lv2.POSITIONS])
    elif damage == "extra":
        state["layoutlmv2.visual.backbone.fpn_output6.weight"] = torch.zeros(1)
    else:
        state["qa_outputs.weight"][0, 0] = float("nan")
    torch.save(state, tmp_path / "pytorch_model.bin")
    with pytest.raises(ValueError):
        archive.load_components(tmp_path, TASK, "cpu")


@pytest.mark.parametrize(
    "change",
    [
        {"MODEL.RESNETS.DEPTH": 152},
        {"MODEL.RESNETS.NUM_GROUPS": True},
        {"MODEL.RESNETS.STRIDE_IN_1X1": True},
        {"MODEL.BACKBONE.NAME": "build_resnet_backbone"},
        {"MODEL.FPN.IN_FEATURES": ["res3", "res4", "res5"]},
        {"MODEL.PIXEL_MEAN": [0.0, 0.0, 0.0]},
        {"MODEL.RESNETS.NORM": "BN"},
    ],
)
def test_only_the_reviewed_detectron2_backbone_is_admitted(change):
    config = copy.deepcopy(ORIGINAL)
    config["detectron2_config_args"].update(change)
    with pytest.raises(ValueError):
        lv2.validate_config(config)


@pytest.mark.parametrize(
    "change",
    [
        {"image_feature_pool_shape": [14, 14, 256]},
        {"has_visual_segment_embedding": False},
        {"coordinate_size": 64},
        {"num_labels": 3},
        {"auto_map": {"AutoModel": "modeling.Custom"}},
        {"fast_qkv": "yes"},
    ],
)
def test_closed_text_and_layout_configuration(change):
    with pytest.raises(ValueError):
        lv2.validate_config(ORIGINAL | change)


def test_unfused_attention_projection_variant_is_complete():
    config = ORIGINAL | {"fast_qkv": False}
    with torch.device("meta"):
        native = lv2.native_model(config)
    assert lv2.expected_shapes(config) == {k: list(v.shape) for k, v in native.state_dict().items()}


@pytest.mark.parametrize("damage", [None, "head", "processor", "tokenizer", "custom"])
def test_metadata_admission_requires_full_graph_and_assets(tmp_path, monkeypatch, damage):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore
    from oryn_engine.runtime_packs.execution import PackRuntime

    _, config = fixture(tmp_path)
    data = (tmp_path / "pytorch_model.bin").read_bytes()
    metadata = inspect_archive(lambda a, n: data[a : a + n], len(data))
    if damage == "head":
        del metadata["tensors"]["qa_outputs.bias"]
    if damage == "custom":
        config = config | {"auto_map": {"AutoModel": "modeling.Custom"}}
    excluded = {
        "processor": {"preprocessor_config.json"},
        "tokenizer": {"vocab.txt", "tokenizer.json"},
    }.get(damage, set())

    class Hub(FakeHub):
        def read_numeric_archive(self, *args):
            return metadata

        def read_json(self, repo, revision, name):
            return config if name == "config.json" else json.loads((tmp_path / name).read_text())

        def download(self, *args, **kwargs):
            raise AssertionError("No checkpoint download")

    hub = Hub(
        config=config,
        pipeline_tag=TASK,
        files=[
            (p.name, p.stat().st_size, "a" * 64)
            for p in tmp_path.iterdir()
            if p.is_file() and p.name not in excluded
        ],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("No native construction during metadata inspection")

    monkeypatch.setattr(lv2, "native_model", forbidden)
    store = ModelStore(tmp_path / "store", hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    result = store.inspect(
        dict(provider="huggingface", repository="test/layoutlmv2", revision="main")
    )
    assert result["executable"] == (damage is None), result.get("blockingReasons")
    if damage is None:
        assert result["runtimePack"]["taskContract"] == "document-question-answering-v1"
        assert "pytorch_model.bin" in {f["name"] for f in result["selectedFiles"]}
        # A numeric-only family still advertises the contract's page-image input.
        assert set(result["taskProfile"]["inputModalities"]) >= {"image", "text"}
