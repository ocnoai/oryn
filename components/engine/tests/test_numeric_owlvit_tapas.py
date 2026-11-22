"""Complete detection/table graphs over bounded numeric archives."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import transformers as tf
from PIL import Image

from oryn_engine import (
    numeric_archive_tasks as archive,
)
from oryn_engine import (
    numeric_owlvit as owl,
)
from oryn_engine import numeric_tapas as tapas
from oryn_engine import (
    shared_tasks,
)
from oryn_engine.numeric_torch_archive import inspect_archive

CASES = ["owl-patch8", "owl-patch16", "tapas-sqa", "tapas-wtq"]
ARTIFACT = "artifact_" + "a" * 24


def pixels(width=43, height=29):
    return Image.fromarray(
        ((np.arange(width * height * 3) * 37 + 19) % 256).astype(np.uint8).reshape(height, width, 3)
    )


def fixture(root, case, dtype=torch.float32):
    torch.manual_seed(6131)
    if case.startswith("owl"):
        task = "zero-shot-object-detection"
        size, patch = (32, 8) if case.endswith("8") else (48, 16)
        words = (
            ["<unk>"]
            + list("abcdefghijklmnopqrstuvwxyz")
            + ["cat</w>", "dog</w>", "<|startoftext|>", "<|endoftext|>"]
        )
        tokenizer = tf.CLIPTokenizer(
            vocab={w: i for i, w in enumerate(words)}, merges=[], unk_token="<unk>"
        )
        cfg = tf.OwlViTConfig(
            text_config=dict(
                vocab_size=len(tokenizer),
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                max_position_embeddings=16,
                bos_token_id=0,
                eos_token_id=2,
                pad_token_id=1,
            ),
            vision_config=dict(
                hidden_size=24,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                image_size=size,
                patch_size=patch,
            ),
            projection_dim=16,
        )
        cfg._attn_implementation = "eager"
        native = tf.OwlViTForObjectDetection(cfg)
        p = dict(
            image_processor_type="OwlViTImageProcessor",
            processor_class="OwlViTProcessor",
            size=[size, size],
            do_center_crop=False,
            do_convert_rgb=True,
            do_normalize=True,
            do_resize=True,
            resample=3,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
        )
        (root / "preprocessor_config.json").write_text(json.dumps(p))
        positions = {
            "owlvit.text_model.embeddings.position_ids": 16,
            "owlvit.vision_model.embeddings.position_ids": (size // patch) ** 2 + 1,
        }
    else:
        task = "table-question-answering"
        aggs = 4 if case.endswith("wtq") else 0
        words = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "[EMPTY]",
            "name",
            "score",
            "alice",
            "bob",
            "1",
            "2",
            "what",
            "is",
            "the",
            "?",
        ] + [f"word{i}" for i in range(16, 32)]
        (root / "vocab.txt").write_text("\n".join(words) + "\n")
        tokenizer = tf.TapasTokenizer(str(root / "vocab.txt"))
        cfg = tf.TapasConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=64,
            num_aggregation_labels=aggs,
            no_aggregation_label_index=0 if aggs else None,
            aggregation_labels={0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"} if aggs else None,
        )
        cfg._attn_implementation = "eager"
        native = tf.TapasForQuestionAnswering(cfg)
        with torch.no_grad():
            native.output_bias.fill_(2.0)
            if aggs:
                native.aggregation_classifier.bias[3] = 20.0
        positions = {"tapas.embeddings.position_ids": 64}
    cfg.architectures = [type(native).__name__]
    state = {k: v.to(dtype) for k, v in native.state_dict().items()}
    state = {k: v.T.contiguous().T if v.ndim == 2 else v for k, v in state.items()}
    native.load_state_dict(state, strict=True)
    state.update({k: torch.arange(length).expand(1, -1) for k, length in positions.items()})
    torch.save(state, root / "pytorch_model.bin")
    cfg.save_pretrained(root)
    tokenizer.save_pretrained(root)
    c = json.loads((root / "config.json").read_text())
    if case.startswith("owl"):
        owl.prepare_tokenizer(c, tokenizer)
        processor = tf.OwlViTProcessor(
            tf.OwlViTImageProcessor(**owl.processor_options(c, p)), tokenizer
        )
    else:
        processor = tokenizer
    return native.eval(), processor, c, task


def request(task):
    if task == "zero-shot-object-detection":
        return dict(imageArtifactId=ARTIFACT, labels=["cat", "dog"], threshold=0.0)
    return dict(
        question="what is the score ?",
        columns=["name", "score"],
        rows=[["alice", "1"], ["bob", "2"]],
    )


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_complete_native_values_logits_and_task_results(tmp_path, case, dtype, backend):
    import pandas as pd

    from oryn_engine.coverage_tasks import table_answer

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    native, processor, c, task = fixture(tmp_path, case, dtype)
    model, loaded, actual_c = shared_tasks.load_task(tmp_path, task, backend)
    assert archive.expected_shapes(c, task) == {
        k: list(v.shape) for k, v in native.state_dict().items()
    }
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor.cpu(), native.state_dict()[key], atol=0, rtol=0)
    value = request(task)
    image = pixels()
    inputs = (
        processor(
            images=image,
            text=[value["labels"]],
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        if case.startswith("owl")
        else processor(
            table=pd.DataFrame(value["rows"], columns=value["columns"]),
            queries=value["question"],
            return_tensors="pt",
            truncation=False,
        )
    )
    with torch.inference_mode():
        expected = native(**inputs)
        actual = model(**{k: v.to(backend) for k, v in inputs.items()})
        fields = (
            ["logits", "pred_boxes"]
            if case.startswith("owl")
            else ["logits"] + (["logits_aggregation"] if c["num_aggregation_labels"] else [])
        )
        for key in fields:
            torch.testing.assert_close(
                getattr(actual, key).cpu(), getattr(expected, key), atol=3e-5, rtol=3e-4
            )
        result = archive.execute(
            model, loaded, actual_c, task, value, backend, image if case.startswith("owl") else None
        )
        if case.startswith("owl"):
            reference = processor.post_process_grounded_object_detection(
                expected,
                threshold=0.0,
                target_sizes=[(image.height, image.width)],
                text_labels=[value["labels"]],
            )[0]
            assert len(result["detections"]) == len(reference["scores"])
            for row, score, label, box in zip(
                result["detections"],
                reference["scores"],
                reference["text_labels"],
                reference["boxes"],
                strict=True,
            ):
                assert row["label"] == label and row["score"] == pytest.approx(
                    float(score), abs=3e-5
                )
                clamped = [
                    max(0, min(limit, float(x)))
                    for limit, x in zip([image.width, image.height] * 2, box, strict=True)
                ]
                assert row["box"] == pytest.approx(clamped, abs=3e-5, rel=3e-4)
        else:
            assert result == table_answer(native, processor, c, value, "cpu")
    shared_tasks.validate_task_output(task, result)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("damage", ["backbone", "head", "shape", "extra", "nonfinite", "position"])
def test_reject_incomplete_graph_or_bad_legacy_positions(tmp_path, case, damage):
    native, _, c, task = fixture(tmp_path, case)
    state = native.state_dict()
    head = "class_head.dense0.weight" if case.startswith("owl") else "output_weights"
    if damage == "backbone":
        del state[next(k for k in state if ".embeddings." in k)]
    elif damage == "head":
        del state[head]
    elif damage == "shape":
        state[head] = torch.zeros(2, 3)
    elif damage == "extra":
        state["custom.head"] = torch.zeros(1)
    elif damage == "nonfinite":
        state[head].view(-1)[0] = float("inf")
    else:
        key = (
            "owlvit.vision_model.embeddings.position_ids"
            if case.startswith("owl")
            else "tapas.embeddings.position_ids"
        )
        length = (
            (c["vision_config"]["image_size"] // c["vision_config"]["patch_size"]) ** 2 + 1
            if case.startswith("owl")
            else c["max_position_embeddings"]
        )
        state[key] = torch.ones(1, length, dtype=torch.long)
    torch.save(state, tmp_path / "pytorch_model.bin")
    with pytest.raises(ValueError):
        archive.load_components(tmp_path, task, "cpu")


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("damage", [None, "head", "shape", "asset"])
def test_metadata_admission_requires_full_graph_and_assets(tmp_path, monkeypatch, case, damage):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore
    from oryn_engine.runtime_packs.execution import PackRuntime

    _, _, c, task = fixture(tmp_path, case)
    raw = (tmp_path / "pytorch_model.bin").read_bytes()
    m = inspect_archive(lambda a, n: raw[a : a + n], len(raw))
    head = "class_head.dense0.weight" if case.startswith("owl") else "output_weights"
    if damage == "head":
        del m["tensors"][head]
    if damage == "shape":
        m["tensors"][head]["shape"] = [1, 2]
    excluded = (
        ("preprocessor_config.json" if case.startswith("owl") else "vocab.txt")
        if damage == "asset"
        else None
    )

    class Hub(FakeHub):
        def read_numeric_archive(self, *args):
            return m

        def read_json(self, repo, revision, name):
            return json.loads((tmp_path / name).read_text())

        def download(self, *args, **kwargs):
            raise AssertionError("No checkpoint download")

    hub = Hub(
        config=c,
        pipeline_tag=task,
        files=[
            (p.name, p.stat().st_size, "a" * 64) for p in tmp_path.iterdir() if p.name != excluded
        ],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("No native construction during metadata inspection")

    monkeypatch.setattr(tf.AutoConfig, "for_model", forbidden)
    store = ModelStore(tmp_path / "store", hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    result = store.inspect(
        dict(provider="huggingface", repository="test/numeric-task", revision="main")
    )
    assert result["executable"] == (damage is None), result.get("blockingReasons")
    if damage is None:
        assert result["runtimePack"]["runnerId"] == "transformers.shared-tasks"


@pytest.mark.parametrize("case", CASES)
def test_repeated_offline_worker(tmp_path, case):
    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry

    _, _, _, task = fixture(tmp_path, case)
    path = tmp_path / "image.png"
    pixels().save(path)
    pack = builtin_registry().active_pack("oryn.transformers-pytorch")
    message = dict(
        abi=pack.abi,
        packId=pack.pack_id,
        version=pack.version,
        command="generate",
        payload=dict(
            modelRoot=str(tmp_path),
            task=task,
            backend="cpu",
            input=request(task),
            inputPath=str(path) if case.startswith("owl") else None,
            outputPath=None,
        ),
    )
    results = []
    for _ in range(2):
        child = subprocess.run(
            [sys.executable, "-m", "oryn_engine.runtime_packs.transformers_worker"],
            input=json.dumps(message) + "\n",
            text=True,
            capture_output=True,
            timeout=60,
            env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        assert child.returncode == 0, child.stderr
        result = next(
            json.loads(s) for s in child.stdout.splitlines() if json.loads(s)["type"] == "result"
        )
        assert result["ok"], (result, child.stderr)
        results.append(result["result"])
    assert results[0] == results[1]


@pytest.mark.parametrize("width,height", [(43, 29), (29, 43)])
def test_bicubic_pixels_match_independent_scalar_normalization(tmp_path, width, height):
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import functional as tvf

    _, processor, _, _ = fixture(tmp_path, "owl-patch8")
    image = pixels(width, height)
    actual = processor.image_processor(images=image, return_tensors="pt")["pixel_values"]
    resized = tvf.resize(
        torch.from_numpy(np.array(image)).permute(2, 0, 1),
        [32, 32],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float64)[:, None, None]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float64)[:, None, None]
    expected = ((resized.double() / 255 - mean) / std).float().unsqueeze(0)
    torch.testing.assert_close(actual, expected, atol=5e-7, rtol=0)


@pytest.mark.parametrize(
    "name,count",
    [
        ("owl-base", 414),
        ("owl-large", 606),
        ("owl-finetune", 414),
        ("tapas-wtq", 35),
        ("tapas-sqa", 33),
        ("tapas-finetune", 211),
        ("tapas-no-tokenizer", 401),
    ],
)
def test_complete_original_graph_metadata(name, count):
    p = Path(__file__).parent / "fixtures/numeric_detection_table"
    c = json.loads((p / (name + ".config.json")).read_text())
    m = json.loads((p / (name + ".metadata.json")).read_text())
    task = "zero-shot-object-detection" if name.startswith("owl") else "table-question-answering"
    shapes, legacy = archive.validate_archive(c, task, m)
    assert len(shapes) == count
    assert len(legacy or {}) == (2 if name.startswith("owl") else 0)
    if name.startswith("owl"):
        owl.processor_options(c, json.loads((p / (name + ".preprocessor_config.json")).read_text()))


def test_original_tiny_owlvit_invalid_token_ids_remain_rejected():
    p = Path(__file__).parent / "fixtures/numeric_detection_table/owl-invalid-token-ids.config.json"
    c = json.loads(p.read_text())
    with pytest.raises(ValueError, match="TOKENIZER"):
        owl.validate_config(c)


@pytest.mark.parametrize(
    "key,value",
    [
        ("size", [32.0, 32]),
        ("size", {"height": 32, "width": 32.0}),
        ("resample", 2),
        ("resample", 3.0),
        ("do_center_crop", True),
        ("do_pad", True),
        ("image_mean", [0.5] * 3),
        ("image_std", [0.0] * 3),
        ("image_processor_type", "CustomProcessor"),
        ("auto_map", {"AutoProcessor": "custom.Processor"}),
    ],
)
def test_owlvit_rejects_unowned_preprocessing(tmp_path, key, value):
    _, _, c, _ = fixture(tmp_path, "owl-patch8")
    p = json.loads((tmp_path / "preprocessor_config.json").read_text())
    p[key] = value
    with pytest.raises(ValueError):
        owl.processor_options(c, p)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("text_config", "eos_token_id", 49407),
        ("text_config", "num_attention_heads", 3),
        ("vision_config", "image_size", 32.0),
        ("vision_config", "patch_size", 7),
        (None, "projection_dim", 32),
        (None, "quantization", {"bits": 4}),
    ],
)
def test_owlvit_rejects_inconsistent_geometry(tmp_path, section, key, value):
    _, _, c, _ = fixture(tmp_path, "owl-patch8")
    (c[section] if section else c)[key] = value
    with pytest.raises(ValueError):
        owl.validate_config(c)


@pytest.mark.parametrize(
    "key,value",
    [
        ("num_aggregation_labels", 5),
        ("aggregation_labels", None),
        ("no_aggregation_label_index", 1),
        ("type_vocab_sizes", [3, 256]),
        ("temperature", 0.0),
        ("span_prediction", "span"),
        ("use_gumbel_for_cells", True),
        ("select_one_column", 1),
    ],
)
def test_tapas_rejects_unowned_table_semantics(tmp_path, key, value):
    _, _, c, _ = fixture(tmp_path, "tapas-wtq")
    c[key] = value
    with pytest.raises(ValueError):
        tapas.validate_config(c)


@pytest.mark.parametrize("case", ["owl-patch8", "tapas-sqa"])
def test_inputs_never_silently_truncate(tmp_path, case):
    model, processor, c, task = fixture(tmp_path, case)
    value = request(task)
    if case.startswith("owl"):
        value["labels"] = ["cat " * 30]
    else:
        value["question"] = "what " * 80
    with torch.inference_mode(), pytest.raises(ValueError, match="INPUT_LIMIT|token limit"):
        archive.execute(
            model, processor, c, task, value, "cpu", pixels() if case.startswith("owl") else None
        )
