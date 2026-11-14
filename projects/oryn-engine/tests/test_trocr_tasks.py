"""Generated native TrOCR graph, processor and task-contract regressions."""

import json

import pytest
import torch
from PIL import Image

from oryn_engine import shared_tasks, trocr_tasks


def tiny_model(root, *, cross_attention=16, tied=False):
    from transformers import (
        RobertaTokenizer,
        TrOCRConfig,
        TrOCRForCausalLM,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        ViTConfig,
        ViTImageProcessor,
        ViTModel,
    )

    torch.manual_seed(61)
    encoder = ViTModel(
        ViTConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=16,
            patch_size=8,
        )
    )
    decoder = TrOCRForCausalLM(
        TrOCRConfig(
            vocab_size=16,
            d_model=24,
            decoder_layers=1,
            decoder_attention_heads=3,
            decoder_ffn_dim=32,
            max_position_embeddings=32,
            is_decoder=True,
            add_cross_attention=True,
            cross_attention_hidden_size=cross_attention,
            tie_word_embeddings=tied,
        )
    )
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder).eval()
    model.save_pretrained(root)
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4}
    vocab.update({chr(97 + i): i + 5 for i in range(11)})
    tokenizer = RobertaTokenizer(vocab=vocab, merges=[])
    image_processor = ViTImageProcessor(size={"height": 16, "width": 16})
    TrOCRProcessor(tokenizer=tokenizer, image_processor=image_processor).save_pretrained(root)
    image_processor.save_pretrained(root)
    return model


@pytest.mark.parametrize("backend", ["cpu", "mps"])
@pytest.mark.parametrize("cross_attention,tied", [(16, False), (None, True)])
def test_native_trocr_loading_preserves_logits_and_image_conditioning(
    tmp_path, backend, cross_attention, tied
):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    reference = tiny_model(tmp_path, cross_attention=cross_attention, tied=tied)
    model, processor, config = shared_tasks.load_task(tmp_path, "image-to-text", backend)
    pixels = torch.stack([torch.zeros(3, 16, 16), torch.ones(3, 16, 16)])
    tokens = torch.tensor([[2, 5, 6], [2, 5, 6]])
    with torch.inference_mode():
        expected = reference(pixel_values=pixels, decoder_input_ids=tokens).logits
        actual = model(
            pixel_values=pixels.to(backend), decoder_input_ids=tokens.to(backend)
        ).logits.cpu()
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    assert not torch.allclose(actual[0], actual[1], atol=1e-7, rtol=1e-7)
    assert processor.tokenizer.chat_template is None
    assert model.generation_config.decoder_start_token_id == 2
    if tied:
        return
    with torch.inference_mode():
        result = trocr_tasks.execute(
            model, processor, config, Image.new("RGB", (24, 20), "white"), {"maxTokens": 3}, backend
        )
    shared_tasks.validate_task_output("image-to-text", result)
    with pytest.raises(ValueError, match="TOKEN_LIMIT"):
        trocr_tasks.execute(
            model, processor, config, Image.new("RGB", (24, 20)), {"maxTokens": 32}, backend
        )


def test_trocr_runs_the_shared_artifact_path_and_reports_its_token_limit(tmp_path):
    from oryn_engine.task_profiles import profile_for_execution

    tiny_model(tmp_path)
    path = tmp_path / "input.png"
    Image.new("RGB", (24, 20), "white").save(path)
    value = {"imageArtifactId": "artifact_aaaaaaaaaaaaaaaaaaaaaaaa", "maxTokens": 3}
    result = shared_tasks.execute_task(
        tmp_path, "image-to-text", value, "cpu", path, None, lambda *_: None
    )
    assert result["text"]
    config = json.loads((tmp_path / "config.json").read_text())
    profile = profile_for_execution(
        {
            "technicalCapabilities": {"task": "image-to-text"},
            "runtimePack": {"taskContract": "image-to-text-v1"},
        },
        config,
    )
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(profile["parameters"]["maxTokens"]["valueSchema"])
    assert validator.is_valid(31) and not validator.is_valid(32)
    assert profile["parameters"]["maxTokens"]["default"] == 31
    # Other composite decoders retain a generic task profile and their actual
    # adapter rejection; they must not look like missing TrOCR metadata.
    config["decoder"]["model_type"] = "bert"
    unsupported = profile_for_execution(
        {"technicalCapabilities": {"task": "image-to-text"}}, config
    )
    assert unsupported["contract"]["id"] == "image-to-text-v1"


@pytest.mark.parametrize("mutation", ["encoder", "decoder", "bridge", "extra"])
def test_trocr_missing_components_are_never_initialized(tmp_path, mutation):
    from safetensors.torch import load_file, save_file

    tiny_model(tmp_path, cross_attention=None if mutation == "bridge" else 16)
    path = tmp_path / "model.safetensors"
    state = load_file(path)
    prefix = "encoder." if mutation == "encoder" else "decoder."
    key = next(k for k in state if k.startswith(prefix))
    if mutation == "extra":
        # An unexpected graph is also disallowed, even when all original keys exist.
        state["unreviewed_bridge.weight"] = torch.zeros(2, 2)
    else:
        if mutation == "bridge":
            key = "enc_to_dec_proj.weight"
        del state[key]
    save_file(state, path)
    with pytest.raises(ValueError, match="incomplete"):
        shared_tasks.load_task(tmp_path, "image-to-text", "cpu")


@pytest.mark.parametrize(
    "part,field,value",
    [
        ("decoder", "model_type", "mbart"),
        ("encoder", "model_type", "deit"),
        ("decoder", "add_cross_attention", False),
        ("decoder", "vocab_size", True),
        ("decoder", "decoder_start_token_id", 100),
        ("encoder", "image_size", 17),
        ("encoder", "num_channels", 1),
        ("decoder", "cross_attention_hidden_size", 12),
    ],
)
def test_trocr_rejects_unreviewed_graphs(tmp_path, part, field, value):
    tiny_model(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    config[part][field] = value
    with pytest.raises(ValueError):
        trocr_tasks.validate_config(config)


@pytest.mark.parametrize(
    "change",
    [
        {"size": 32},
        {"do_resize": False},
        {"do_normalize": False},
        {"image_std": [0, 1, 1]},
        {"image_processor_type": "RepositoryProcessor"},
    ],
)
def test_trocr_metadata_and_loading_share_processor_bounds(tmp_path, change):
    tiny_model(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    path = tmp_path / "preprocessor_config.json"
    metadata = json.loads(path.read_text()) | change
    with pytest.raises(ValueError):
        trocr_tasks.processor_options(metadata, config)
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        shared_tasks.load_task(tmp_path, "image-to-text", "cpu")
