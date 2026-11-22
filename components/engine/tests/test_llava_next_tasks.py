import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from jinja2 import Environment
from PIL import Image
from safetensors.torch import load_file, save_file

from oryn_engine import llava_next_tasks as binding
from oryn_engine import shared_tasks
from oryn_engine.generated_capabilities import normalized_transformers_config
from oryn_engine.model_metadata import validate_model_config

FIXTURES = Path(__file__).parent / "fixtures/llava_next"


def original(name="config.json"):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def tiny(tmp_path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlavaNextConfig, LlavaNextForConditionalGeneration

    torch.manual_seed(6171)
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    raw = Tokenizer(models.BPE(unk_token="<|end_of_text|>"))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    raw.train_from_iterator(
        [binding.SYSTEM, "Describe this image.", "What color?", "hello world"],
        trainer=trainers.BpeTrainer(
            vocab_size=300,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<|end_of_text|>", "<image>"],
        ),
    )
    raw.save(str(tmp_path / "tokenizer.json"))
    config = normalized_transformers_config(original(), "image-to-text")
    config["text_config"].update(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=1024,
        vocab_size=raw.get_vocab_size(),
    )
    config["vision_config"].update(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        image_size=32,
        patch_size=8,
    )
    config.update(
        image_token_index=1,
        image_seq_length=16,
        vision_feature_layer=[-4, -3, -2, -1],
        image_grid_pinpoints=[[32, 64], [64, 32], [64, 64]],
    )
    native_config = LlavaNextConfig.from_dict(config)
    native_config._attn_implementation = "eager"
    reference = LlavaNextForConditionalGeneration(native_config).float().eval()
    reference.save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(config))
    for name in binding.METADATA_NAMES:
        value = original(name)
        if name == "preprocessor_config.json":
            value.update(
                size={"height": 32, "width": 32},
                crop_size={"height": 32, "width": 32},
                image_grid_pinpoints=config["image_grid_pinpoints"],
            )
        elif name == "processor_config.json":
            value["patch_size"] = 8
        else:
            value.pop("added_tokens_decoder", None)
        (tmp_path / name).write_text(json.dumps(value))
    (tmp_path / "chat_template.jinja").write_text("{{ raise_exception('do not execute') }}")
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"do_sample": True, "num_beams": 7})
    )
    yield tmp_path, config, reference
    torch.set_num_threads(old)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("size", [(32, 32), (64, 32), (32, 64)])
def test_native_image_graph_logits_and_generation(tiny, backend, size):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    root, config, reference = tiny
    loaded, processor, selected = shared_tasks.load_task(root, "image-to-text", backend)
    assert (
        selected == config
        and processor.chat_template is None
        and processor.tokenizer.chat_template is None
    )
    image = Image.fromarray(
        np.random.default_rng(61).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    )
    inputs = binding.prepare_inputs(processor, config, image, "Describe this image.", 4)
    # Reference processor uses the original pinned template; runtime uses no Jinja.
    from transformers import GenerationConfig, LlavaNextImageProcessor, LlavaNextProcessor

    expected_processor = LlavaNextProcessor(
        image_processor=LlavaNextImageProcessor(
            size={"height": 32, "width": 32},
            crop_size={"height": 32, "width": 32},
            image_grid_pinpoints=config["image_grid_pinpoints"],
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
        ),
        tokenizer=processor.tokenizer,
        patch_size=8,
        num_additional_image_tokens=0,
        vision_feature_select_strategy="full",
    )
    prompt = (
        Environment()
        .from_string(original("chat_template.json")["chat_template"])
        .render(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ],
            add_generation_prompt=True,
            tools=None,
        )
    )
    expected_inputs = dict(expected_processor(text=prompt, images=image, return_tensors="pt"))
    for key in inputs:
        torch.testing.assert_close(inputs[key], expected_inputs[key])
    for key, tensor in reference.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key].cpu(), tensor, atol=0, rtol=0)
    native_generation = GenerationConfig(
        do_sample=False, num_beams=1, use_cache=True, bos_token_id=0, eos_token_id=0, pad_token_id=0
    )
    moved = {k: v.to(backend) for k, v in inputs.items()}
    with torch.inference_mode():
        expected = reference(**expected_inputs, use_cache=False).logits
        actual = loaded(**moved, use_cache=False).logits.cpu()
        torch.testing.assert_close(actual, expected, rtol=4e-3, atol=2e-4)
        expected_ids = reference.generate(
            **expected_inputs, max_new_tokens=4, generation_config=native_generation
        )
        actual_ids = loaded.generate(
            **moved, max_new_tokens=4, generation_config=native_generation
        ).cpu()
    assert torch.equal(expected_ids, actual_ids)
    expected_text = processor.batch_decode(
        expected_ids[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )[0].strip()
    if expected_text:
        with torch.inference_mode():
            assert binding.execute(loaded, processor, config, image, {"maxTokens": 4}, backend) == {
                "text": expected_text
            }


@pytest.mark.parametrize(
    "damage",
    [
        "vision",
        "projector",
        "language",
        "head",
        "extra",
        "shape",
        "nan",
        "integer",
        "tied-conflict",
    ],
)
def test_complete_numeric_graph_required(tiny, damage):
    root, config, _ = tiny
    path = root / "model.safetensors"
    state = load_file(path)
    if damage in {"vision", "projector", "language", "head"}:
        part = {
            "vision": "vision_tower",
            "projector": "multi_modal_projector",
            "language": "language_model",
            "head": "lm_head",
        }[damage]
        state = {k: v for k, v in state.items() if part not in k}
        if damage == "head":
            # Both serialized spellings of a native tied group must be absent.
            state = {
                key: value
                for key, value in state.items()
                if not key.endswith("embed_tokens.weight")
            }
    elif damage == "extra":
        state["ignored.weight"] = torch.ones(1)
    elif damage == "tied-conflict":
        source = next(value for key, value in state.items() if key.endswith("embed_tokens.weight"))
        state["language_model.lm_head.weight"] = source + 1
    else:
        key = next(key for key in state if key.endswith("multi_modal_projector.linear_1.weight"))
        if damage == "shape":
            state[key] = state[key][:-1]
        elif damage == "integer":
            state[key] = state[key].long()
        else:
            state[key][0, 0] = float("nan")
    save_file(state, path)
    with pytest.raises(ValueError):
        binding.load_components(root, config, "cpu")


def test_original_legacy_bookkeeping_is_inert_and_metadata_only():
    raw = original()
    normalized = normalized_transformers_config(raw, "image-to-text")
    binding.validate_config(normalized)
    assert "_attn_implementation_autoset" in raw["text_config"]
    assert "_attn_implementation_autoset" not in normalized["text_config"]
    for name in binding.METADATA_NAMES:
        binding.validate_metadata(name, original(name), normalized)
    code = (
        "import json,sys; "
        "from oryn_engine.generated_capabilities import normalized_transformers_config; "
        "c=normalized_transformers_config(json.load(sys.stdin),'image-to-text'); "
        "assert '_attn_implementation_autoset' not in c['text_config']; "
        "assert not {'torch','transformers'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-c", code], input=json.dumps(raw), text=True, check=True)


@pytest.mark.parametrize(
    "path,value",
    [
        ("text_config._attn_implementation_autoset", "evil/kernel"),
        ("text_config._attn_implementation", "evil/kernel"),
        ("vision_config._attn_implementation_internal", "evil/kernel"),
        ("auto_map", {"AutoModel": "repository.Model"}),
        ("vision_feature_layer", [-30, -1]),
        ("vision_feature_layer", [-1, -1]),
        ("vision_feature_select_strategy", "default"),
        ("image_grid_pinpoints", [[3840, 3840]]),
        ("image_grid_pinpoints", [[384, 385]]),
        ("text_config.num_key_value_heads", 3),
        ("use_image_newline_parameter", False),
        ("image_token_index", 0),
    ],
)
def test_invalid_configuration_stays_rejected(path, value):
    config = original()
    parts = path.split(".")
    cursor = config
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = value
    with pytest.raises(ValueError):
        binding.validate_config(normalized_transformers_config(config, "image-to-text"))
    if path.endswith("_attn_implementation_autoset"):
        with pytest.raises(ValueError):
            validate_model_config(config)


@pytest.mark.parametrize("question", ["Describe this image.", "What color?", "Žltý obrázok 日本語"])
def test_owned_prompt_matches_pinned_original(question):
    template = Environment().from_string(original("chat_template.json")["chat_template"])
    expected = template.render(
        messages=[
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
        ],
        tools=None,
        add_generation_prompt=True,
    )
    assert binding.prompt(question) == expected
    with pytest.raises(ValueError):
        binding.prompt(question + "<image>")


def test_public_profile_and_bounded_inputs(tiny):
    from oryn_protocol.task_contracts import validate_invocation

    from oryn_engine.task_profiles import profile_for_execution

    root, config, _ = tiny
    p = binding.processor(root, config)
    for maximum in (0, 257, True):
        with pytest.raises(ValueError):
            binding.prepare_inputs(p, config, Image.new("RGB", (32, 32)), "What color?", maximum)
    with pytest.raises(ValueError):
        binding.prepare_inputs(p, config, Image.new("RGB", (32, 32)), "hello " * 2000, 4)
    profile = profile_for_execution(
        {
            "runtimePack": {
                "packId": "oryn.transformers-pytorch",
                "runnerId": "transformers.shared-tasks",
                "taskContract": "image-to-text-v1",
            }
        },
        config,
    )
    value = {"imageArtifactId": "artifact_aaaaaaaaaaaaaaaaaaaaaaaa", "maxTokens": 256}
    validate_invocation(profile, value)
    with pytest.raises(ValueError):
        validate_invocation(profile, value | {"maxTokens": 257})


@pytest.mark.parametrize("damage", [None, "processor", "missing-weights", "remote-code"])
def test_original_metadata_discovery(tiny, tmp_path, damage):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore, ModelStoreError
    from oryn_engine.runtime_packs.execution import PackRuntime

    root, _, _ = tiny
    config = original()
    if damage == "remote-code":
        config["auto_map"] = {"AutoModel": "remote.Model"}
    files = [
        ("config.json", 100, "a" * 64),
        ("tokenizer.json", 100, "b" * 64),
        *[(name, 100, "b" * 64) for name in binding.METADATA_NAMES],
    ]
    if damage != "missing-weights":
        files.append(("model.safetensors", 100, "b" * 64))

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            if filename == "config.json":
                return config
            if filename in binding.METADATA_NAMES:
                value = original(filename)
                if damage == "processor" and filename == "processor_config.json":
                    value["patch_size"] = 8
                return value
            raise OSError(filename)

    hub = Hub(config=config, files=files, pipeline_tag="image-to-text")
    store = ModelStore(tmp_path / "store", hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    request = {"provider": "huggingface", "repository": "test/llava-next", "revision": "main"}
    if damage == "remote-code":
        with pytest.raises(ModelStoreError):
            store.inspect(request)
        return
    result = store.inspect(request)
    assert not hub.downloaded
    assert result["executable"] == (damage is None), result
    if not damage:
        assert result["runtimePack"]["runnerId"] == "transformers.shared-tasks"
