"""Complete table, document, video and VQA graphs over bounded numeric archives."""

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

from oryn_engine import coverage_tasks, shared_tasks
from oryn_engine import numeric_archive_tasks as archive
from oryn_engine import numeric_coverage as coverage
from oryn_engine.generated_capabilities import normalized_transformers_config
from oryn_engine.numeric_torch_archive import inspect_archive

FIXTURES = Path(__file__).parent / "fixtures/numeric_coverage"
CASES = ["tapex", "layoutlmv3", "vivit", "vilt"]
TASKS = {
    "tapex": "table-question-answering",
    "layoutlmv3": "document-question-answering",
    "vivit": "video-classification",
    "vilt": "visual-question-answering",
}
ARTIFACT = "artifact_" + "a" * 24


def pixels(width=43, height=29):
    return Image.fromarray(
        ((np.arange(width * height * 3) * 37 + 19) % 256).astype(np.uint8).reshape(height, width, 3)
    )


def byte_vocab():
    # GPT-2 byte alphabet: every byte has one printable token, so no merges are needed.
    printable = [*range(33, 127), *range(161, 173), *range(174, 256)]
    alphabet, extra = list(printable), 0
    for b in range(256):
        if b not in printable:
            alphabet.append(256 + extra)
            extra += 1
    chars = [chr(c) for c in alphabet]
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
    for char in chars:
        vocab.setdefault(char, len(vocab))
    vocab["<mask>"] = len(vocab)
    return vocab


def video(path, frames=16):
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=8)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        for i in range(frames):
            value = np.full((32, 32, 3), (i * 13) % 256, dtype=np.uint8)
            value[..., 1] = 255 - value[..., 0]
            for packet in stream.encode(av.VideoFrame.from_ndarray(value, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def legacy_vivit(name):
    for new, old in [
        (".attention.q_proj.", ".attention.attention.query."),
        (".attention.k_proj.", ".attention.attention.key."),
        (".attention.v_proj.", ".attention.attention.value."),
        (".attention.o_proj.", ".attention.output.dense."),
        (".mlp.fc1.", ".intermediate.dense."),
        (".mlp.fc2.", ".output.dense."),
    ]:
        name = name.replace(new, old)
    return name.replace("vivit.layers.", "vivit.encoder.layer.")


def fixture(root, case, dtype=torch.float32):
    torch.manual_seed(6161)
    root.mkdir(parents=True, exist_ok=True)
    positions = {}
    rename = None
    raw = None
    if case == "tapex":
        vocab = byte_vocab()
        (root / "vocab.json").write_text(json.dumps(vocab))
        (root / "merges.txt").write_text("#version: 0.2\n")
        tokenizer = tf.RobertaTokenizer(str(root / "vocab.json"), str(root / "merges.txt"))
        cfg = tf.BartConfig(
            vocab_size=len(vocab),
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_position_embeddings=256,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            decoder_start_token_id=2,
            forced_eos_token_id=2,
        )
        native = tf.BartForConditionalGeneration(cfg)
        tokenizer.save_pretrained(root)
        metadata = json.loads((root / "tokenizer_config.json").read_text())
        metadata.update(tokenizer_class="TapexTokenizer", do_lower_case=True, max_cell_length=15)
        (root / "tokenizer_config.json").write_text(json.dumps(metadata))
    elif case == "layoutlmv3":
        vocab = byte_vocab()
        (root / "vocab.json").write_text(json.dumps(vocab))
        (root / "merges.txt").write_text("#version: 0.2\n")
        tokenizer = tf.LayoutLMv3Tokenizer(
            vocab=vocab, merges=[], add_prefix_space=True, unk_token="<unk>"
        )
        cfg = tf.LayoutLMv3Config(
            vocab_size=len(vocab),
            hidden_size=24,
            coordinate_size=4,
            shape_size=4,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            input_size=32,
            patch_size=8,
            max_position_embeddings=130,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
        )
        native = tf.LayoutLMv3ForQuestionAnswering(cfg)
        tokenizer.save_pretrained(root)
        tf.LayoutLMv3ImageProcessor(
            size={"height": 32, "width": 32}, apply_ocr=False
        ).save_pretrained(root)
        positions = {"layoutlmv3.embeddings.position_ids": 130}
    elif case == "vivit":
        cfg = tf.VivitConfig(
            image_size=16,
            num_frames=4,
            tubelet_size=[2, 8, 8],
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            id2label={0: "first", 1: "second"},
        )
        native = tf.VivitForVideoClassification(cfg)
        tf.VivitImageProcessor(
            size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}
        ).save_pretrained(root)
        rename = legacy_vivit
    else:
        words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "what", "color", "is", "it", "?"]
        (root / "vocab.txt").write_text("\n".join(words) + "\n")
        tokenizer = tf.BertTokenizerFast(str(root / "vocab.txt"))
        cfg = tf.ViltConfig(
            vocab_size=len(words),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            image_size=32,
            patch_size=8,
            max_position_embeddings=16,
            id2label={0: "red", 1: "green", 2: "blue"},
        )
        native = tf.ViltForQuestionAnswering(cfg)
        tokenizer.save_pretrained(root)
        tf.ViltImageProcessor(size={"shortest_edge": 32}, size_divisor=8).save_pretrained(root)
        positions = {"vilt.embeddings.text_embeddings.position_ids": 16}
    cfg._attn_implementation = "eager"
    cfg.architectures = [type(native).__name__]
    state = {k: v.to(dtype) for k, v in native.state_dict().items()}
    native.load_state_dict(state, strict=True)
    native.tie_weights()
    saved = {(rename(k) if rename else k): v for k, v in state.items()}
    saved.update({k: torch.arange(n).expand(1, -1) for k, n in positions.items()})
    torch.save(saved, root / "pytorch_model.bin")
    cfg.save_pretrained(root)
    raw = json.loads((root / "config.json").read_text())
    if case == "vivit":
        # Original exports record the historical class spelling and video_size.
        raw.pop("image_size")
        raw.pop("num_frames")
        raw.update(video_size=[4, 16, 16], architectures=["ViViTForVideoClassification"])
    if case == "vilt":
        raw["architectures"] = ["ViltForVisualQuestionAnswering"]
    (root / "config.json").write_text(json.dumps(raw))
    return native.float().eval(), raw, TASKS[case]


def request(case, root):
    if case == "tapex":
        return dict(
            question="what is the score?",
            columns=["name", "score"],
            rows=[["alice", "1"], ["bob", "2"]],
        ), None
    if case == "layoutlmv3":
        return dict(
            imageArtifactId=ARTIFACT,
            question="what is the total?",
            words=["total", "42", "date", "today"],
            boxes=[
                [0.1, 0.1, 0.2, 0.2],
                [0.3, 0.1, 0.4, 0.2],
                [0.1, 0.5, 0.2, 0.6],
                [0.3, 0.5, 0.5, 0.6],
            ],
        ), "image"
    if case == "vivit":
        path = root / "clip.mp4"
        video(path)
        return dict(videoArtifactId=ARTIFACT, topK=2), path
    return dict(imageArtifactId=ARTIFACT, question="what color is it?", maxTokens=8), "image"


@pytest.mark.parametrize("case", CASES)
def test_original_headers_match_complete_native_graph(case):
    task = TASKS[case]
    config = normalized_transformers_config(
        json.loads((FIXTURES / f"{case}.config.json").read_text()), task
    )
    archive.validate_config(config, task)
    expected = archive.expected_shapes(config, task)
    with torch.device("meta"):
        native = getattr(tf, archive.ARCHITECTURES[(task, config["model_type"])])(
            tf.AutoConfig.for_model(**config)
        )
    assert expected == {k: list(v.shape) for k, v in native.state_dict().items()}
    header = json.loads((FIXTURES / f"{case}.header.json").read_text())
    metadata = {
        "format": "torch-numeric-zip-v1",
        "byteOrder": "little",
        "tensors": {
            k: {"shape": v, "dtype": "I64" if k.endswith("position_ids") else "F32"}
            for k, v in header.items()
        },
    }
    shapes, positions = archive.validate_archive(config, task, metadata)
    assert set(shapes) == set(header)
    assert bool(positions) is (case in {"layoutlmv3", "vilt"})


def test_historical_vivit_geometry_is_normalized_exactly():
    raw = json.loads((FIXTURES / "vivit.config.json").read_text())
    config = normalized_transformers_config(raw, "video-classification")
    assert config["architectures"] == ["VivitForVideoClassification"]
    assert (config["num_frames"], config["image_size"]) == (32, 224) and "video_size" not in config
    for damage in [{"video_size": [32, 224, 112]}, {"video_size": [32, 224]}, {"num_frames": 8}]:
        with pytest.raises(ValueError):
            coverage.normalized_config(raw | damage, "video-classification")
    assert coverage.normalized_config(raw, "image-classification") is raw


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_native_values_and_shared_task_results(tmp_path, case, dtype):
    native, raw, task = fixture(tmp_path, case, dtype)
    model, processor, config = shared_tasks.load_task(tmp_path, task, "cpu")
    reference = native.state_dict()
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, reference[key], atol=0, rtol=0)
    value, extra = request(case, tmp_path)
    if task in coverage_tasks.TASK_CONTRACTS:
        coverage_tasks.validate_relations(task, value)
    image = pixels() if extra == "image" else None
    input_path = extra if case == "vivit" else None
    with torch.inference_mode():
        result = archive.execute(
            model, processor, config, task, value, "cpu", image, input_path, (43, 29)
        )
        again = archive.execute(
            model, processor, config, task, value, "cpu", image, input_path, (43, 29)
        )
    assert result == again
    shared_tasks.validate_task_output(task, result)
    if case == "vilt":
        inputs = processor(images=image, text=value["question"], return_tensors="pt")
        with torch.inference_mode():
            logits = native(**inputs).logits[0]
        assert result["text"] == config["id2label"][str(int(logits.argmax()))]
    if case == "tapex":
        assert result["aggregation"] == "generated" and result["cells"] == []
    if case == "vivit":
        assert {p["label"] for p in result["predictions"]} <= {"first", "second"}


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize(
    "damage", ["backbone", "head", "shape", "extra", "nonfinite", "position", "duplicate"]
)
def test_incomplete_or_ambiguous_graphs_are_rejected(tmp_path, case, damage):
    fixture(tmp_path, case)
    state = torch.load(tmp_path / "pytorch_model.bin", weights_only=True)
    head = {
        "tapex": "final_logits_bias",
        "layoutlmv3": "qa_outputs.out_proj.weight",
        "vivit": "classifier.weight",
        "vilt": "classifier.3.weight",
    }[case]
    if damage == "backbone":
        del state[next(k for k in state if ".embeddings." in k or "embed_tokens" in k)]
    elif damage == "head":
        del state[head]
    elif damage == "shape":
        state[head] = torch.zeros(2, 3)
    elif damage == "extra":
        state["custom.head"] = torch.zeros(1)
    elif damage == "nonfinite":
        state[head].view(-1)[0] = float("inf")
    elif damage == "position":
        key = next((k for k in state if k.endswith("position_ids")), None)
        if key is None:
            pytest.skip("No historical position buffer")
        state[key] = torch.ones_like(state[key])
    else:
        source = next(k for k in state if "encoder.layer.0" in k or "layers.0" in k)
        if case != "vivit":
            pytest.skip("Only historical ViT names are renamed")
        state[coverage.native_name({"model_type": "vivit"}, source)] = state[source].clone()
    torch.save(state, tmp_path / "pytorch_model.bin")
    with pytest.raises(ValueError):
        archive.load_components(tmp_path, TASKS[case], "cpu")


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("damage", [None, "head", "shape", "asset", "custom"])
def test_metadata_admission_requires_full_graph_and_assets(tmp_path, monkeypatch, case, damage):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore
    from oryn_engine.runtime_packs.execution import PackRuntime

    _, raw, task = fixture(tmp_path, case)
    data = (tmp_path / "pytorch_model.bin").read_bytes()
    metadata = inspect_archive(lambda a, n: data[a : a + n], len(data))
    head = {
        "tapex": "final_logits_bias",
        "layoutlmv3": "qa_outputs.out_proj.weight",
        "vivit": "classifier.weight",
        "vilt": "classifier.3.weight",
    }[case]
    if damage == "head":
        del metadata["tensors"][head]
    if damage == "shape":
        metadata["tensors"][head]["shape"] = [1, 2]
    if damage == "custom":
        raw = raw | {"auto_map": {"AutoModel": "modeling.Custom"}}
    excluded = (
        ("tokenizer_config.json" if case == "tapex" else "preprocessor_config.json")
        if damage == "asset"
        else None
    )

    class Hub(FakeHub):
        def read_numeric_archive(self, *args):
            return metadata

        def read_json(self, repo, revision, name):
            return raw if name == "config.json" else json.loads((tmp_path / name).read_text())

        def download(self, *args, **kwargs):
            raise AssertionError("No checkpoint download")

    hub = Hub(
        config=raw,
        pipeline_tag=task,
        files=[
            (p.name, p.stat().st_size, "a" * 64)
            for p in tmp_path.iterdir()
            if p.is_file() and p.name != excluded and p.suffix != ".mp4"
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
        dict(provider="huggingface", repository="test/numeric-coverage", revision="main")
    )
    assert result["executable"] == (damage is None), result.get("blockingReasons")
    if damage is None:
        assert result["runtimePack"]["runnerId"] == "transformers.shared-tasks"
        assert "pytorch_model.bin" in {f["name"] for f in result["selectedFiles"]}


@pytest.mark.parametrize("case", ["vilt", "vivit"])
def test_repeated_offline_worker(tmp_path, case):
    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry

    _, _, task = fixture(tmp_path, case)
    value, extra = request(case, tmp_path)
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
            input=value,
            inputPath=str(extra if case == "vivit" else path),
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
            timeout=120,
            env=os.environ | {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        assert child.returncode == 0, child.stderr
        result = next(
            json.loads(s) for s in child.stdout.splitlines() if json.loads(s)["type"] == "result"
        )
        assert result["ok"], (result, child.stderr)
        results.append(result["result"])
    assert results[0] == results[1]
