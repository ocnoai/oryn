import copy
import json

import numpy as np
import pytest
import torch
from PIL import Image

from oryn_engine import neomme_retrieval as binding
from oryn_engine import shared_tasks


def tiny_model(root):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import NeoMMEConfig, NeoMMEForRetrieval, TokenizersBackend

    vocab = {token: i for i, token in enumerate(binding.TOKENS.values())}
    vocab.update({"what": 8, "color": 9, "page": 10, "blue": 11})
    vocab.update({f"extra_{i}": i for i in range(12, 32)})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    native_tokenizer = TokenizersBackend(
        tokenizer_object=tokenizer,
        **{
            k: v
            for k, v in binding.TOKENS.items()
            if k in {"pad_token", "unk_token", "eos_token", "mask_token"}
        },
        extra_special_tokens={
            k: v
            for k, v in binding.TOKENS.items()
            if k in {"query_token", "document_token", "image_token", "row_token"}
        },
    )
    native_tokenizer.save_pretrained(root)
    metadata = json.loads((root / "tokenizer_config.json").read_text()) | binding.TOKENS
    (root / "tokenizer_config.json").write_text(json.dumps(metadata))
    config = NeoMMEConfig(
        hidden_size=32,
        intermediate_size=64,
        embedding_rank=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        vocab_size=32,
        patch_size=8,
        max_position_embeddings=256,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=16,
        embedding_dim=128,
    )
    torch.manual_seed(61)
    model = NeoMMEForRetrieval(config).eval()
    model.save_pretrained(root)
    (root / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "NeoMMEProcessor",
                "image_processor": {
                    "image_processor_type": "NeoMMEImageProcessor",
                    "patch_size": 8,
                    "max_side": 2048,
                    "image_mean": [0.5] * 3,
                    "image_std": [0.5] * 3,
                },
            }
        )
    )
    (root / "chat_template.jinja").write_text("{{ raise_exception('never render') }}")
    return model, native_tokenizer


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_native_image_and_query_vectors_with_independent_patch_grid(tmp_path, backend):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    native, tokenizer = tiny_model(tmp_path)
    model, processor, config = shared_tasks.load_task(
        tmp_path, "visual-document-retrieval", backend
    )
    pixels = np.arange(17 * 25 * 3, dtype=np.uint8).reshape(17, 25, 3)
    page, queries = binding.prepare_inputs(
        processor, config, Image.fromarray(pixels), ["what color", "blue"]
    )
    # Independent row-major RGB patches, bottom/right black padding, and 2D positions.
    canvas = np.zeros((24, 32, 3), np.float32)
    canvas[:17, :25] = pixels
    canvas = canvas / 127.5 - 1
    patches = np.stack(
        [canvas[y : y + 8, x : x + 8].reshape(-1) for y in range(0, 24, 8) for x in range(0, 32, 8)]
    )
    ids = torch.tensor([[5, 6, *([6] * 4 + [7]) * 3]])
    positions = torch.tensor(
        [[0, 0], [1, 1], *[[y + 2, x + 2] for y in range(3) for x in range(5)]]
    ).T[:, None]
    expected_page = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "position_ids": positions,
        "pixel_values": torch.tensor(patches),
    }
    for key in expected_page:
        torch.testing.assert_close(page[key], expected_page[key], atol=1e-6, rtol=1e-6)
    expected_ids = tokenizer("<query>what color" + "<mask>" * 10, return_tensors="pt")["input_ids"]
    torch.testing.assert_close(queries[0]["input_ids"], expected_ids)
    assert expected_ids.tolist() == [[4, 8, 9] + [3] * 10]
    with torch.inference_mode():
        for inputs in [expected_page, *queries]:
            expected = native(**inputs, output_dense=False).embeddings
            actual = model(
                **{k: v.to(backend) for k, v in inputs.items()}, output_dense=False
            ).embeddings.cpu()
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        result = binding.retrieve(
            model,
            processor,
            config,
            Image.fromarray(pixels),
            {"queries": ["what color", "blue"]},
            backend,
        )
    assert result["scoring"] == "mean-max-cosine"
    shared_tasks.validate_task_output("visual-document-retrieval", result)
    assert processor.chat_template is None and processor.tokenizer.chat_template is None
    assert len(result["queries"][0]["vectors"]) == 13


@pytest.mark.parametrize("part", ["patch", "embedding", "attention", "head", "extra", "shape"])
def test_complete_native_encoder_and_projection_are_required(tmp_path, part):
    from safetensors.torch import load_file, save_file

    tiny_model(tmp_path)
    path = tmp_path / "model.safetensors"
    state = load_file(path)
    match = {
        "patch": "patch_embeddings",
        "embedding": "word_embeddings",
        "attention": "q_proj",
        "head": "multi_vector_head",
    }
    if part in match:
        del state[next(k for k in state if match[part] in k)]
    elif part == "extra":
        state["unreviewed.weight"] = torch.zeros(2, 2)
    else:
        state[next(k for k in state if "multi_vector_head" in k)] = torch.zeros(2, 2)
    save_file(state, path)
    with pytest.raises((ValueError, RuntimeError)):
        shared_tasks.load_task(tmp_path, "visual-document-retrieval", "cpu")


@pytest.mark.parametrize(
    "change",
    [
        {"embedding_dim": 64},
        {"num_key_value_heads": 3},
        {"layer_types": ["causal_attention"] * 2},
        {"rope_parameters": {"full_attention": {"partial_rotary_factor": 0.1}}},
        {"per_layer_config": {"1": {"sliding_window": 0}}},
        {"auto_map": {"AutoModel": "untrusted.Model"}},
    ],
)
def test_unreviewed_config_is_rejected(tmp_path, change):
    tiny_model(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text()) | change
    with pytest.raises(ValueError):
        shared_tasks.validate_task_config("visual-document-retrieval", config)


def test_query_expansion_page_bounds_layout_and_shared_result(tmp_path):
    from oryn_protocol.task_contracts import validate_result

    from oryn_engine.model_layouts import assess_layout
    from oryn_engine.task_profiles import profile_for_execution

    tiny_model(tmp_path)
    _, processor, config = shared_tasks.load_task(tmp_path, "visual-document-retrieval", "cpu")
    image = Image.new("RGB", (1000, 2000), (100, 80, 60))
    page, _ = binding.prepare_inputs(processor, config, image, ["what color"])
    assert page["pixel_values"].shape[0] <= 196 and page["input_ids"].shape[1] <= 212
    with pytest.raises(ValueError, match="QUERY_TOKEN_LIMIT"):
        binding.prepare_inputs(processor, config, image, ["what " * 54])
    with pytest.raises(ValueError, match="RESERVED_TOKEN"):
        binding.prepare_inputs(processor, config, image, ["what <img>"])
    files = [p.name for p in tmp_path.iterdir()]
    layout = assess_layout(files, config, "visual-document-retrieval")
    assert not layout.runtime_reasons and not layout.artifact_reasons
    assert assess_layout(
        [f for f in files if f != "processor_config.json"], config, "visual-document-retrieval"
    ).artifact_reasons
    path = tmp_path / "page.png"
    image.save(path)
    value = {
        "imageArtifactId": "artifact_aaaaaaaaaaaaaaaaaaaaaaaa",
        "queries": ["what color", "blue"],
    }
    result = shared_tasks.execute_task(
        tmp_path, "visual-document-retrieval", value, "cpu", path, None, lambda *_: None
    )
    profile = profile_for_execution(
        {
            "technicalCapabilities": {"task": "visual-document-retrieval"},
            "runtimePack": {"runnerId": "transformers.shared-tasks"},
        },
        config,
    )
    validate_result(profile, {"kind": "structured", "value": result}, value)


@pytest.mark.parametrize("part", ["processor", "tokenizer", "vocabulary"])
def test_processor_or_tokenizer_mismatch_cannot_reach_execution(tmp_path, part):
    tiny_model(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    if part == "processor":
        name = "processor_config.json"
        metadata = json.loads((tmp_path / name).read_text())
        metadata["image_processor"]["image_std"] = [1, 1, 1]
    elif part == "tokenizer":
        name = "tokenizer_config.json"
        metadata = json.loads((tmp_path / name).read_text())
        metadata["query_token"] = "<doc>"
    else:
        path = tmp_path / "tokenizer.json"
        data = json.loads(path.read_text())
        data["model"]["vocab"]["rogue"] = 32
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            shared_tasks.load_task(tmp_path, "visual-document-retrieval", "cpu")
        return
    with pytest.raises(ValueError):
        binding.validate_metadata(name, copy.deepcopy(metadata), config)
    (tmp_path / name).write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        shared_tasks.load_task(tmp_path, "visual-document-retrieval", "cpu")
