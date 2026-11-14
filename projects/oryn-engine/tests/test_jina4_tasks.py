import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from oryn_engine import jina4_tasks as binding
from oryn_engine import shared_tasks

FIXTURES = Path(__file__).parent / "fixtures/jina4"


def document(name):
    return json.loads((FIXTURES / name).read_text())


def normalized(name="tiny-config.json"):
    return binding.normalized_config(document(name), binding.TASK)


@pytest.fixture
def checkpoint(tmp_path):
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import Qwen2Tokenizer

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    vocab = {token: i for i, token in enumerate(sorted(ByteLevel.alphabet()))}
    special = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
    ]
    for token in special:
        vocab[token] = len(vocab)
    tokenizer = Qwen2Tokenizer(
        vocab=vocab,
        merges=[],
        bos_token=None,
        unk_token=None,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
    )
    tokenizer.add_special_tokens({"additional_special_tokens": special})
    tokenizer.save_pretrained(tmp_path)
    metadata = document("tokenizer_config.json") | {
        "added_tokens_decoder": {
            str(vocab[token]): {
                "content": token,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "single_word": False,
                "special": True,
            }
            for token in special
        },
        "additional_special_tokens": special,
        "chat_template": "{{ raise_exception('inert') }}",
    }
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(metadata))
    config = document("tiny-config.json")
    (tmp_path / "config.json").write_text(json.dumps(config))
    processor = document("preprocessor_config.json") | {
        "patch_size": 2,
        "min_pixels": 64,
        "max_pixels": 12288,
        "size": {"shortest_edge": 64, "longest_edge": 12288},
    }
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(processor))
    (tmp_path / "adapters").mkdir()
    (tmp_path / "adapters/adapter_config.json").write_text(
        json.dumps(document("adapters/adapter_config.json"))
    )
    for name in ("modeling_jina_embeddings_v4.py", "qwen2_5_vl.py", "custom_lora_module.py"):
        (tmp_path / name).write_text("raise AssertionError('repository code must remain inert')")
    state = binding.native_model(normalized()).state_dict()
    generator = torch.Generator().manual_seed(61281)
    for name, value in sorted(state.items()):
        value.copy_(torch.randn(value.shape, generator=generator) * 0.05)
        if name.endswith(("norm.weight", "norm1.weight", "norm2.weight")):
            value.add_(1)
    base, adapters = {}, {}
    for name, value in state.items():
        if ".lora_" in name:
            key = name.removeprefix("vlm.")
            adapters["base_model.model." + key] = value.clone()
        elif name != "vlm.lm_head.weight":
            key = name.replace("vlm.model.language_model.", "model.").replace(
                "vlm.model.visual.", "visual."
            )
            base[key.replace(".base_layer.", ".")] = value.clone()
    save_file(base, tmp_path / "model.safetensors")
    save_file(adapters, tmp_path / binding.ADAPTER_WEIGHT)
    yield tmp_path
    for path in [*tmp_path.glob("*.safetensors"), *tmp_path.glob("adapters/*.safetensors")]:
        if not path.is_symlink():
            path.unlink()
    torch.set_num_threads(threads)


def image(case):
    width, height = case["width"], case["height"]
    pixels = (
        np.arange(width * height * 3, dtype=np.uint32).reshape(height, width, 3) + case["offset"]
    ).astype(np.uint8)
    return Image.fromarray(pixels)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("index", range(4))
def test_full_publisher_graph_lora_pixels_prompt_vectors_and_maxsim(checkpoint, backend, index):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    model, processor, config = shared_tasks.load_task(checkpoint, binding.TASK, backend)
    case = document("reference.json")["cases"][index]
    page, queries = binding.retrieval_inputs(processor, config, image(case), case["queries"])
    assert page["input_ids"].tolist() == case["inputIds"]
    assert page["image_grid_thw"].tolist() == case["grid"]
    assert hashlib.sha256(page["pixel_values"].numpy().tobytes()).hexdigest() == case["pixelHash"]
    assert processor.tokenizer.decode(queries[0]["input_ids"][0]) == "Query:   blue page  "
    assert processor.tokenizer.chat_template is None
    with (
        torch.inference_mode(),
        np.load(FIXTURES / "reference-vectors.npz", allow_pickle=False) as gold,
    ):
        result = binding.retrieve(
            model, processor, config, image(case), {"queries": case["queries"]}, backend
        )
        np.testing.assert_allclose(
            result["documentVectors"], gold[f"page_{index}"], atol=3e-6, rtol=5e-5
        )
        for j, item in enumerate(result["queries"]):
            np.testing.assert_allclose(
                item["vectors"], gold[f"query_{index}_{j}"], atol=3e-6, rtol=5e-5
            )
            assert item["score"] == pytest.approx(case["scores"][j], abs=2e-5)
    shared_tasks.validate_task_output(binding.TASK, result)
    assert result["dimensions"] == 128 and result["scoring"] == "sum-max-cosine"
    assert len(model.state_dict()) == 149


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_padded_queries_and_repeated_images_do_not_reuse_rope_cache(checkpoint, backend):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    model, processor, config = binding.load_components(checkpoint, normalized(), backend)
    inputs = {
        k: torch.tensor(v, device=backend)
        for k, v in document("reference.json")["paddedQueryInputs"].items()
    }
    case = document("reference.json")["cases"][0]
    page, _ = binding.retrieval_inputs(processor, config, image(case), case["queries"])
    page = {k: v.to(backend) for k, v in page.items()}
    with (
        torch.inference_mode(),
        np.load(FIXTURES / "reference-vectors.npz", allow_pickle=False) as gold,
    ):
        before = model(**inputs)
        first = model(**page)
        after = model(**inputs)
        second = model(**page)
        np.testing.assert_allclose(before.cpu(), gold["padded_query"], atol=3e-6, rtol=5e-5)
        torch.testing.assert_close(after, before, atol=0, rtol=0)
        torch.testing.assert_close(second, first, atol=0, rtol=0)
        assert torch.count_nonzero(before[1, :2]) == 0


def test_original_all_2344_tensors_and_metadata_are_bound():
    config = normalized("config.json")
    unchanged = copy.deepcopy(config)
    for name in binding.METADATA_NAMES:
        binding.validate_metadata(name, document(name), config)
    shapes = document("original-shapes.json")
    assert len(shapes["base"]) == 826 and len(shapes["adapters"]) == 1518
    model = binding.validate_weight_shapes(
        config, *({k: tuple(v) for k, v in shapes[group].items()} for group in ("base", "adapters"))
    )
    assert len(model.state_dict()) == 2345
    assert all(v.device.type == "meta" for v in model.parameters())
    assert config == unchanged


def test_sharded_base_keeps_separate_complete_adapters(checkpoint):
    path = checkpoint / "model.safetensors"
    weights = load_file(path)
    names = sorted(weights)
    midpoint = len(names) // 2
    index = {}
    for i, group in enumerate((names[:midpoint], names[midpoint:]), 1):
        name = f"model-{i:05d}-of-00002.safetensors"
        save_file({k: weights[k] for k in group}, checkpoint / name)
        index.update({k: name for k in group})
    path.unlink()
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    model, _, _ = binding.load_components(checkpoint, normalized(), "cpu")
    adapters = load_file(checkpoint / binding.ADAPTER_WEIGHT)
    bound = binding.bind_tensors(weights, adapters)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, bound[name], atol=0, rtol=0)


def test_task_selection_retains_inactive_adapters_and_uses_retrieval(checkpoint):
    model, _, _ = binding.load_components(checkpoint, normalized(), "cpu")
    inputs = {
        k: torch.tensor(v) for k, v in document("reference.json")["paddedQueryInputs"].items()
    }
    with torch.inference_mode():
        before = model(**inputs)
        model.multi_vector_projector.lora_B["code"].weight.add_(0.5)
        model.multi_vector_projector.lora_B["text-matching"].weight.add_(0.5)
        torch.testing.assert_close(model(**inputs), before, atol=0, rtol=0)
        model.multi_vector_projector.lora_B["retrieval"].weight.add_(0.5)
        assert (model(**inputs) - before).abs().max().item() > 0.01


@pytest.mark.parametrize(
    "damage",
    [
        "missing-head",
        "missing-vision",
        "missing-text",
        "missing-retrieval",
        "missing-code",
        "missing-matching",
        "extra",
        "shape",
        "nonfinite-base",
        "nonfinite-adapter",
        "integer",
        "overflow-fp64",
        "wrong-prefix",
        "adapter-extra-file",
        "missing-adapter",
        "adapter-symlink",
    ],
)
def test_original_graph_and_every_adapter_are_mandatory(checkpoint, damage, tmp_path):
    base_path, adapter_path = checkpoint / "model.safetensors", checkpoint / binding.ADAPTER_WEIGHT
    base, adapters = load_file(base_path), load_file(adapter_path)
    if damage.startswith("missing-"):
        if damage == "missing-adapter":
            adapter_path.unlink()
        elif damage in {"missing-retrieval", "missing-code", "missing-matching"}:
            task = {
                "missing-retrieval": "retrieval",
                "missing-code": "code",
                "missing-matching": "text-matching",
            }[damage]
            del adapters[next(k for k in adapters if "." + task + "." in k)]
        else:
            prefix = {
                "missing-head": "multi_vector_projector",
                "missing-vision": "visual.",
                "missing-text": "model.layers",
            }[damage]
            del base[next(k for k in base if k.startswith(prefix))]
    elif damage == "extra":
        base["unreviewed.weight"] = torch.ones(1)
    elif damage == "shape":
        adapters[next(iter(adapters))] = torch.ones(1)
    elif damage == "nonfinite-base":
        base["multi_vector_projector.bias"][0] = float("nan")
    elif damage == "nonfinite-adapter":
        adapters[next(iter(adapters))].flatten()[0] = float("inf")
    elif damage == "integer":
        adapters[next(iter(adapters))] = adapters[next(iter(adapters))].long()
    elif damage == "overflow-fp64":
        base["multi_vector_projector.bias"] = torch.full_like(
            base["multi_vector_projector.bias"], 1e100, dtype=torch.float64
        )
    elif damage == "wrong-prefix":
        key = next(iter(adapters))
        adapters["wrong." + key] = adapters.pop(key)
    elif damage == "adapter-extra-file":
        save_file({"extra": torch.ones(1)}, checkpoint / "adapters/extra.safetensors")
    elif damage == "adapter-symlink":
        folder = checkpoint / "adapters"
        moved = checkpoint / "outside"
        folder.rename(moved)
        folder.symlink_to(moved, target_is_directory=True)
    save_file(base, base_path)
    if damage not in {"missing-adapter", "adapter-symlink"}:
        save_file(adapters, adapter_path)
    with pytest.raises(ValueError, match="JINA4_|Unsafe|Metadata"):
        binding.load_components(checkpoint, normalized(), "cpu")


@pytest.mark.parametrize(
    "path,value",
    [
        ("auto_map", {"AutoModel": "evil.Code"}),
        ("_attn_implementation", "custom"),
        ("hidden_size", 1024),
        ("text_config.num_attention_heads", 3),
        ("text_config.unknown_layer", True),
        ("text_config.rope_scaling.type", "linear"),
        ("text_config.rope_scaling.mrope_section", [16, 24, 25]),
        ("text_config.use_sliding_window", True),
        ("text_config.rms_norm_eps", -1e-6),
        ("text_config.tie_word_embeddings", False),
        ("text_config.image_token_id", 151655),
        ("vision_config.out_hidden_size", 1024),
        ("vision_config.fullatt_block_indexes", [31, 7]),
        ("vision_config.window_size", 111),
        ("vision_config.spatial_merge_size", 1),
        ("vision_config.temporal_patch_size", 1),
        ("vision_config.hidden_act", "gelu"),
        ("vision_config.unknown_layer", True),
        ("vision_config.depth", True),
        ("task_names", ["retrieval"]),
        ("multi_vector_projector_dim", 2048),
        ("single_vector_pool_strategy", "last"),
        ("truncate_dim", 128),
        ("matryoshka_dims", [128]),
        ("vision_token_id", 151655),
        ("trust_remote_code", True),
    ],
)
def test_metadata_cannot_change_graph(path, value):
    config = document("config.json")
    target = config
    keys = path.split(".")
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    with pytest.raises(ValueError):
        binding.normalized_config(config, binding.TASK)


@pytest.mark.parametrize(
    "field,value",
    [
        ("r", 16),
        ("r", True),
        ("lora_alpha", 16),
        ("lora_dropout", 0),
        ("use_dora", True),
        ("use_rslora", True),
        ("bias", "all"),
        ("exclude_modules", None),
        ("target_modules", ".*"),
        ("modules_to_save", ["visual"]),
        ("rank_pattern", {"q_proj": 8}),
    ],
)
def test_lora_options_are_closed(field, value):
    with pytest.raises(ValueError, match="ADAPTER_OPTIONS"):
        binding.validate_metadata(
            "adapters/adapter_config.json",
            document("adapters/adapter_config.json") | {field: value},
            normalized(),
        )


@pytest.mark.parametrize(
    "name,field,value",
    [
        ("preprocessor_config.json", "image_mean", [0.5] * 3),
        ("preprocessor_config.json", "do_normalize", False),
        ("preprocessor_config.json", "do_resize", False),
        ("preprocessor_config.json", "min_pixels", 1),
        ("preprocessor_config.json", "max_pixels", 1024),
        ("preprocessor_config.json", "auto_map", {"AutoProcessor": "evil.Code"}),
        ("tokenizer_config.json", "add_bos_token", True),
        ("tokenizer_config.json", "split_special_tokens", True),
        ("tokenizer_config.json", "unk_token", "<|endoftext|>"),
        ("tokenizer_config.json", "auto_map", {"AutoTokenizer": "evil.Code"}),
    ],
)
def test_processor_and_tokenizer_semantics_are_closed(name, field, value):
    with pytest.raises(ValueError):
        binding.validate_metadata(name, document(name) | {field: value}, normalized("config.json"))


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "missing-base",
        "missing-adapter",
        "missing-config",
        "missing-tokenizer",
        "lora",
        "processor",
        "remote-code",
    ],
)
def test_metadata_inspection_selects_complete_original_and_shared_route(tmp_path, damage):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore, ModelStoreError
    from oryn_engine.runtime_packs.execution import PackRuntime

    data = {name: document(name) for name in ["config.json", *binding.METADATA_NAMES]}
    if damage == "lora":
        data["adapters/adapter_config.json"]["r"] = 16
    if damage == "processor":
        data["preprocessor_config.json"]["image_std"] = [1.0] * 3
    if damage == "remote-code":
        data["config.json"]["auto_map"]["AutoModel"] = "evil.Code"
    names = set(data) | {
        "model.safetensors",
        binding.ADAPTER_WEIGHT,
        "tokenizer.json",
        "modeling_jina_embeddings_v4.py",
    }
    missing = {
        "missing-base": "model.safetensors",
        "missing-adapter": binding.ADAPTER_WEIGHT,
        "missing-config": "adapters/adapter_config.json",
        "missing-tokenizer": "tokenizer.json",
    }.get(damage)
    if missing:
        names.remove(missing)

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            return data[filename]

    hub = Hub(
        config=data["config.json"],
        pipeline_tag=binding.TASK,
        files=[(n, 100, "b" * 64) for n in names],
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    request = {"provider": "huggingface", "repository": "test/jina4", "revision": "main"}
    if damage == "remote-code":
        with pytest.raises(ModelStoreError):
            store.inspect(request)
    else:
        result = store.inspect(request)
        if damage:
            assert not result["executable"], result
        else:
            assert result["executable"], result
            assert result["runtimePack"]["runnerId"] == "transformers.shared-tasks"
            assert result["taskProfile"]["limits"]["embeddingDimensions"] == 128
            assert result["taskProfile"]["limits"]["queryTokens"] == 64
            assert {x["name"] for x in result["selectedFiles"]} == names - {
                "modeling_jina_embeddings_v4.py"
            }
    assert not hub.downloaded


@pytest.mark.parametrize("queries", [[], ["x"] * 3, ["<|image_pad|>"], ["bad |>"], ["x" * 65]])
def test_bounded_owned_queries(checkpoint, queries):
    processor = binding.processor(checkpoint, normalized())
    with pytest.raises(ValueError, match="JINA4_"):
        binding.retrieval_inputs(processor, normalized(), Image.new("RGB", (16, 8)), queries)


def test_public_task_preserves_image_and_original_prompt(checkpoint):
    case = document("reference.json")["cases"][3]
    path = checkpoint / "input.png"
    image(case).save(path)
    result = shared_tasks.execute_task(
        checkpoint,
        binding.TASK,
        {"imageArtifactId": "artifact_" + "a" * 24, "queries": case["queries"]},
        "cpu",
        path,
        None,
        lambda *_: None,
    )
    with np.load(FIXTURES / "reference-vectors.npz", allow_pickle=False) as gold:
        np.testing.assert_allclose(result["documentVectors"], gold["page_3"], atol=3e-6, rtol=5e-5)


def test_source_and_native_graph_are_in_signed_build():
    from oryn_engine.generated_capabilities import generated_artifact

    inputs = generated_artifact()["buildInputs"]
    path = Path(binding.__file__)
    assert (
        inputs["orynNativeSources"]["jina4_tasks.py"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    for name in ("modeling_qwen2_5_vl.py", "configuration_qwen2_5_vl.py"):
        assert "models/qwen2_5_vl/" + name in inputs["transformersVisionSources"]
