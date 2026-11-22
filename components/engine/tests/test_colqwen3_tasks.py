import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from oryn_engine import colqwen3_tasks as binding
from oryn_engine import shared_tasks


def originals():
    root = Path(__file__).parent / "fixtures/colqwen3"
    return {
        name: json.loads((root / name).read_text())
        for name in ("config.json", "preprocessor_config.json", "processor_config.json")
    }


@pytest.fixture
def tiny(tmp_path):
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import Qwen2Tokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    vocab = {token: i for i, token in enumerate(sorted(ByteLevel.alphabet()))}
    special = (
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
    )
    for token in special:
        vocab[token] = len(vocab)
    tokenizer = Qwen2Tokenizer(vocab=vocab, merges=[])
    tokenizer.add_special_tokens({"additional_special_tokens": list(special)})
    tokenizer.save_pretrained(tmp_path)
    data = originals()
    config = data["config.json"]
    config["text_config"].update(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=len(vocab),
        max_position_embeddings=256,
    )
    config["text_config"]["rope_scaling"]["mrope_section"] = [1, 1, 2]
    config["text_config"].update(
        bos_token_id=vocab["<|im_start|>"], eos_token_id=vocab["<|im_end|>"]
    )
    config["vision_config"].update(
        hidden_size=16,
        intermediate_size=32,
        depth=1,
        num_heads=2,
        out_hidden_size=16,
        patch_size=2,
        deepstack_visual_indexes=[0],
        num_position_embeddings=16,
    )
    for token, key in (
        ("<|image_pad|>", "image_token_id"),
        ("<|video_pad|>", "video_token_id"),
        ("<|vision_start|>", "vision_start_token_id"),
        ("<|vision_end|>", "vision_end_token_id"),
    ):
        config[key] = vocab[token]
    native_config = Qwen3VLConfig(
        text_config=copy.deepcopy(config["text_config"]),
        vision_config=copy.deepcopy(config["vision_config"]),
        tie_word_embeddings=True,
        **{
            k: config[k]
            for k in (
                "image_token_id",
                "video_token_id",
                "vision_start_token_id",
                "vision_end_token_id",
            )
        },
    )
    for part in (native_config, native_config.text_config, native_config.vision_config):
        part._attn_implementation = "eager"
    torch.manual_seed(61612)
    reference = Qwen3VLForConditionalGeneration(native_config).eval()
    projection = torch.nn.Linear(16, 320).eval()
    weights = {
        "vlm." + k: v.detach().clone()
        for k, v in reference.state_dict().items()
        if k != "lm_head.weight"
    }
    weights.update(
        {
            "embedding_proj_layer." + k: v.detach().clone()
            for k, v in projection.state_dict().items()
        }
    )
    save_file(weights, tmp_path / "model.safetensors")
    data["preprocessor_config.json"].update(
        patch_size=2, size={"shortest_edge": 1024, "longest_edge": 20480}, max_pixels=20480
    )
    for name, value in data.items():
        (tmp_path / name).write_text(json.dumps(value))
    (tmp_path / "modeling_colqwen3.py").write_text(
        "raise RuntimeError('repository code must stay inert')"
    )
    (tmp_path / "chat_template.jinja").write_text("{{ raise_exception('must not render') }}")
    yield tmp_path, config, reference, projection, tokenizer
    torch.set_num_threads(threads)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_complete_native_page_query_projection_and_maxsim(tiny, backend):
    from transformers import Qwen2VLImageProcessor

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    root, config, reference, projection, tokenizer = tiny
    model, processor, selected = shared_tasks.load_task(root, "visual-document-retrieval", backend)
    for key, tensor in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()["vlm." + key].cpu(), tensor, rtol=0, atol=0)
    for key, tensor in projection.state_dict().items():
        torch.testing.assert_close(
            model.state_dict()["embedding_proj_layer." + key].cpu(), tensor, rtol=0, atol=0
        )
    image = Image.fromarray(np.arange(17 * 25 * 3, dtype=np.uint8).reshape(17, 25, 3))
    queries = ["  blue page  ", "river"]
    native_image = Qwen2VLImageProcessor(
        patch_size=2,
        temporal_patch_size=2,
        merge_size=2,
        size={"shortest_edge": 1024, "longest_edge": 2048},
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    )(images=[image], return_tensors="pt")
    count = int(native_image["image_grid_thw"][0].prod()) // 4
    text = (
        "<|im_start|>user\n<|vision_start|>"
        + "<|image_pad|>" * count
        + "<|vision_end|>Describe the image.<|im_end|><|endoftext|>"
    )
    page = dict(tokenizer([text], return_tensors="pt", add_special_tokens=False)) | dict(
        native_image
    )
    page["mm_token_type_ids"] = (page["input_ids"] == config["image_token_id"]).long()
    expected = []
    with torch.inference_mode():
        for inputs in [
            page,
            *(
                tokenizer(
                    [q.strip() + "<|endoftext|>" * 10],
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                for q in queries
            ),
        ]:
            hidden = reference.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
            vectors = projection(hidden)[0]
            expected.append(vectors / vectors.norm(dim=-1, keepdim=True))
        result = binding.retrieve(model, processor, selected, image, {"queries": queries}, backend)
    assert result["dimensions"] == 320
    np.testing.assert_allclose(result["documentVectors"], expected[0].numpy(), rtol=1e-3, atol=2e-5)
    for item, query in zip(result["queries"], expected[1:], strict=True):
        np.testing.assert_allclose(item["vectors"], query.numpy(), rtol=1e-3, atol=2e-5)
        score = (query @ expected[0].T).max(dim=-1).values.sum().item()
        assert item["score"] == pytest.approx(score, abs=2e-4)
    shared_tasks.validate_task_output("visual-document-retrieval", result)
    with pytest.raises(ValueError, match="TOKEN_LIMIT"):
        binding.retrieve(model, processor, selected, image, {"queries": ["z" * 100]}, backend)
    with pytest.raises(ValueError, match="FRAMING"):
        binding.retrieve(model, processor, selected, image, {"queries": ["<|image_pad|>"]}, backend)


@pytest.mark.parametrize(
    "damage", ["missing-head", "missing-vision", "extra", "shape", "nonfinite", "tied-alias"]
)
def test_complete_graph_and_projection_are_mandatory(tiny, damage):
    root, config, _, _, _ = tiny
    path = root / "model.safetensors"
    weights = load_file(path)
    if damage == "missing-head":
        del weights["embedding_proj_layer.bias"]
    elif damage == "missing-vision":
        del weights[next(k for k in weights if k.startswith("vlm.model.visual."))]
    elif damage == "extra":
        weights["unknown.head"] = torch.ones(1)
    elif damage == "shape":
        weights["embedding_proj_layer.weight"] = weights["embedding_proj_layer.weight"][:-1]
    elif damage == "nonfinite":
        weights["embedding_proj_layer.bias"][0] = float("inf")
    else:
        weights["vlm.lm_head.weight"] = weights["vlm.model.language_model.embed_tokens.weight"] + 1
    save_file(weights, path)
    with pytest.raises(ValueError, match="WEIGHTS"):
        binding.load_components(root, config, "cpu")


@pytest.mark.parametrize(
    "path,value",
    [
        ("auto_map", {"AutoModel": "unreviewed.Code"}),
        ("embed_dim", 128),
        ("padding_side", "right"),
        ("text_config.is_causal", False),
        ("text_config.tie_word_embeddings", False),
        ("text_config.rms_norm_eps", -1e-6),
        ("text_config.unreviewed_attention", True),
        ("text_config.num_attention_heads", 3),
        ("vision_config.out_hidden_size", 2561),
        ("vision_config.deepstack_visual_indexes", [999]),
    ],
)
def test_metadata_cannot_select_another_graph(path, value):
    config = originals()["config.json"]
    target = config
    names = path.split(".")
    for key in names[:-1]:
        target = target[key]
    target[names[-1]] = value
    with pytest.raises(ValueError):
        binding.native_config(config)


@pytest.mark.parametrize(
    "damage", [None, "missing-processor", "changed-prompt", "changed-normalization", "remote-code"]
)
def test_full_metadata_admission_is_closed_and_uses_shared_planner(tmp_path, damage):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore, ModelStoreError
    from oryn_engine.runtime_packs.execution import PackRuntime

    data = originals()
    data["tokenizer_config.json"] = {
        "tokenizer_class": "Qwen2Tokenizer",
        "pad_token": "<|endoftext|>",
    }
    if damage == "changed-prompt":
        data["processor_config.json"]["query_prefix"] = "unreviewed"
    elif damage == "changed-normalization":
        data["preprocessor_config.json"]["image_std"] = [1.0] * 3
    elif damage == "remote-code":
        data["config.json"]["auto_map"]["AutoModel"] = "unreviewed.Code"
    files = set(data) | {"model.safetensors", "vocab.json", "merges.txt"}
    if damage == "missing-processor":
        files.remove("processor_config.json")

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            return data[filename]

    hub = Hub(
        config=data["config.json"],
        pipeline_tag="visual-document-retrieval",
        files=[(name, 100, "b" * 64) for name in files],
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    request = {"provider": "huggingface", "repository": "test/colqwen3", "revision": "main"}
    if damage == "remote-code":
        with pytest.raises(ModelStoreError):
            store.inspect(request)
        return
    result = store.inspect(request)
    assert not hub.downloaded
    if damage:
        assert not result["executable"]
    else:
        assert result["executable"], result
        assert result["runtimePack"]["packId"] == "oryn.transformers-pytorch"
        assert result["taskProfile"]["limits"]["embeddingDimensions"] == 320
