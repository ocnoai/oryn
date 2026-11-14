from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from oryn_engine.model_metadata import parse_metadata, preflight_safetensors, validate_model_config
from oryn_engine.shared_tasks import (
    TASK_CONTRACTS,
    image_vector,
    nli_label_ids,
    nli_scores,
    open_task_image,
    validate_task_input,
    validate_task_output,
)

CASES = json.loads(
    (
        Path(__file__).resolve().parents[2] / "oryn-protocol/fixtures/shared-tasks/inputs.json"
    ).read_text()
)
CASES["zero-shot-image-classification-v1"] = {
    "imageArtifactId": "artifact_aaaaaaaaaaaaaaaaaaaaaaaa",
    "labels": ["cat", "dog"],
}
CASES["embeddings-v1"] = {"texts": ["A bounded example."]}
CASES["reranking-v1"] = {"query": "color", "documents": ["blue"]}
CASES["question-answering-v1"] = {"question": "What color?", "context": "The sky is blue."}
CASES["translation-v1"] = {"text": "A bounded example."}
CASES["depth-estimation-v1"] = {"imageArtifactId": "artifact_aaaaaaaaaaaaaaaaaaaaaaaa"}
CASES["audio-classification-v1"] = {"audioArtifactId": "artifact_aaaaaaaaaaaaaaaaaaaaaaaa"}
CASES["speech-synthesis-v1"] = {
    "text": "Hello.",
    "audioArtifactId": "artifact_aaaaaaaaaaaaaaaaaaaaaaaa",
    "voice": "reference",
}


@pytest.mark.parametrize("task,contract", TASK_CONTRACTS.items())
def test_closed_input_contracts(task, contract):
    value = (
        {"text": "A bounded example."}
        if task in {"summarization", "text-classification", "token-classification", "fill-mask"}
        else CASES[contract]
    )
    validate_task_input(task, value)
    for field in ("backend", "modelRoot", "runtimeOptions", "path", "trust_remote_code"):
        with pytest.raises(ValueError):
            validate_task_input(task, {**value, field: "forbidden"})


@pytest.mark.parametrize(
    "value",
    [
        b'{"x":NaN}',
        b'{"x":1,"x":2}',
        b'{"a":Infinity}',
        b"[]",
        b'{"x":' + b"[" * 40 + b"0" + b"]" * 40 + b"}",
    ],
)
def test_hostile_metadata(value):
    with pytest.raises(ValueError):
        parse_metadata(value)


def test_tokenizer_redundant_settings_are_unambiguous_locally_and_remotely(tmp_path):
    from oryn_engine.model_metadata import metadata_file

    raw = b'{"model_input_names":["input_ids"],"model_input_names":["input_ids"]}'
    assert parse_metadata(raw, redundant_keys=True) == {"model_input_names": ["input_ids"]}
    tokenizer = tmp_path / "tokenizer_config.json"
    tokenizer.write_bytes(raw)
    assert metadata_file(tokenizer) == parse_metadata(raw, redundant_keys=True)
    config = tmp_path / "config.json"
    config.write_bytes(raw)
    with pytest.raises(ValueError, match="Duplicate"):
        metadata_file(config)
    for value in (
        b'{"add_prefix_space":false,"add_prefix_space":true}',
        b'{"x":true,"x":1}',
        b'{"x":1,"x":1.0}',
        b'{"nested":{"x":false,"x":true}}',
    ):
        with pytest.raises(ValueError, match="Duplicate"):
            parse_metadata(value, redundant_keys=True)


@pytest.mark.parametrize(
    "value",
    [
        {"auto_map": {"A": "custom.A"}},
        {"vision_config": {"custom_pipelines": {"evil": 1}}},
        {"hidden_size": 1000000},
        {"model_file": "/signed/package/arbitrary.py"},
        {"text_config": {"model_file": "repo.py"}},
        {"_attn_implementation_internal": "attacker/kernel"},
        {"vision_config": {"trust_remote_code": True}},
    ],
)
def test_hostile_configuration(value):
    with pytest.raises(ValueError):
        validate_model_config(value)


def test_tensor_headers_are_checked_before_loading(tmp_path):
    header = {"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}

    def write(value, data=b"0000"):
        raw = json.dumps(value).encode()
        (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + data)

    write(header)
    assert preflight_safetensors(tmp_path) == {"x": (1,)}
    header["x"]["shape"] = [1000000000]
    write(header)
    with pytest.raises(ValueError):
        preflight_safetensors(tmp_path)

    header["x"]["shape"] = [1]
    header["y"] = header["x"].copy()
    write(header)
    with pytest.raises(ValueError):
        preflight_safetensors(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", 2**40))
    with pytest.raises(ValueError):
        preflight_safetensors(tmp_path)


def test_weight_indexes_cannot_redirect_or_mislabel_selected_tensors(tmp_path):
    header = {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    raw = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + b"0000")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": "model.safetensors"}}))
    assert preflight_safetensors(tmp_path) == {"weight": (1,)}
    for mapping in (
        {"weight": "../model.safetensors"},
        {"weight": "/model.safetensors"},
        {"weight": "other.safetensors"},
        {"another_weight": "model.safetensors"},
    ):
        index.write_text(json.dumps({"weight_map": mapping}))
        with pytest.raises(ValueError):
            preflight_safetensors(tmp_path)


def test_ocr_processor_uses_bounded_native_geometry_and_sealed_prompt():
    from oryn_engine.shared_tasks import (
        _vision_prompt,
        qwen_image_options,
        validate_processor_config,
        validate_task_config,
    )

    metadata = {
        "image_processor_type": "Glm46VImageProcessor",
        "size": {"shortest_edge": 12544, "longest_edge": 9633792},
        "patch_size": 14,
        "merge_size": 2,
        "temporal_patch_size": 2,
    }
    validate_processor_config(metadata, "glm_ocr")
    assert qwen_image_options(metadata, {"patch_size": 14})["size"]["longest_edge"] == 1024**2
    assert "Text Recognition:" in _vision_prompt("glm_ocr", "ignored")
    assert "ignored" not in _vision_prompt("glm_ocr", "ignored")
    validate_task_config("image-to-text", {"model_type": "glm_ocr"})
    with pytest.raises(ValueError):
        validate_task_config("visual-question-answering", {"model_type": "glm_ocr"})
    with pytest.raises(ValueError):
        validate_processor_config({**metadata, "auto_map": {"Processor": "repo.Code"}}, "glm_ocr")
    with pytest.raises(ValueError):
        validate_processor_config({**metadata, "image_processor_type": "Unknown"}, "glm_ocr")


def test_nli_semantics_are_explicit_and_probabilities_correct():
    assert nli_label_ids(
        {"id2label": {"0": "contradiction", "1": "neutral", "2": "entailment"}}
    ) == (2, 0)
    with pytest.raises(ValueError):
        nli_label_ids({"id2label": {"0": "LABEL_0", "1": "LABEL_1"}})
    logits = torch.tensor([[0.0, 0.0, 4.0], [4.0, 0.0, 0.0]])
    scores = nli_scores(logits, 2, 0, multi_label=False)
    assert float(scores[0]) > 0.98 and float(scores.sum()) == pytest.approx(1)
    multi = nli_scores(torch.tensor([[0.0, 0.0, 4.0], [0.0, 0.0, 4.0]]), 2, 0, multi_label=True)
    assert all(float(s) > 0.98 for s in multi)
    with pytest.raises(ValueError):
        nli_scores(torch.tensor([[float("nan"), 0, 0]]), 2, 0, multi_label=False)


@pytest.mark.parametrize("entailment", [0, 1])
@pytest.mark.parametrize("multi_label", [False, True])
def test_binary_nli_matches_native_pipeline_and_preserves_label_permutation(
    entailment, multi_label
):
    from transformers.pipelines import ZeroShotClassificationPipeline

    labels = {str(entailment): "entailment", str(1 - entailment): "not_entailment"}
    ids = nli_label_ids({"id2label": labels})
    assert ids == (entailment, 1 - entailment)
    logits = torch.tensor([[3.0, -2.0], [-1.0, 2.0], [0.5, 1.0]])
    pipeline = object.__new__(ZeroShotClassificationPipeline)
    pipeline.model = SimpleNamespace(
        config=SimpleNamespace(label2id={name: int(i) for i, name in labels.items()})
    )
    outputs = [
        {"candidate_label": label, "sequence": "example", "logits": logits[i : i + 1]}
        for i, label in enumerate(["sports", "politics", "science"])
    ]
    expected = pipeline.postprocess(outputs, multi_label=multi_label)
    scores = nli_scores(logits, *ids, multi_label=multi_label)
    by_label = dict(zip(["sports", "politics", "science"], scores.tolist(), strict=True))
    assert [by_label[label] for label in expected["labels"]] == pytest.approx(expected["scores"])
    permutation = [2, 0, 1]
    assert nli_scores(logits[permutation], *ids, multi_label=multi_label).tolist() == pytest.approx(
        scores[permutation].tolist()
    )


@pytest.mark.parametrize(
    "labels",
    [
        {"0": "entailment", "1": "not_entailment", "2": "neutral"},
        {"0": "entailment", "1": "contradiction", "2": "not_entailment"},
        {"0": "entailment", "1": "LABEL_1"},
        {"0": "entailment", "1": "ENTAILMENT"},
        {"0": "entailment", "2": "not_entailment"},
    ],
)
def test_binary_nli_rejects_ambiguous_or_incomplete_label_bindings(labels):
    with pytest.raises(ValueError):
        nli_label_ids({"id2label": labels})


@pytest.mark.parametrize(
    "logits,entailment,negative",
    [
        (torch.empty(0, 2), 0, 1),
        (torch.zeros(2, 1), 0, 1),
        (torch.zeros(2, 4), 0, 1),
        (torch.zeros(2, 2), -1, 0),
        (torch.zeros(2, 2), 0, 2),
        (torch.zeros(2, 2), 0, 0),
        (torch.zeros(2, 2), False, 1),
        (torch.full((2, 2), 1e100, dtype=torch.float64), 0, 1),
    ],
)
def test_nli_rejects_unbound_or_nonfinite_logit_heads(logits, entailment, negative):
    with pytest.raises(ValueError):
        nli_scores(logits, entailment, negative, multi_label=True)


def test_cls_pooling_and_output_constraints():
    value = image_vector(
        SimpleNamespace(last_hidden_state=torch.tensor([[[3.0, 4.0], [99.0, 99.0]]])),
        normalize=True,
    )
    assert value["vector"] == pytest.approx([0.6, 0.8])
    validate_task_output("image-feature-extraction", value)
    for change in (
        {"dimensions": 3},
        {"vector": [float("nan"), 0]},
        {"backend": "mps"},
        {"vector": [0.0, 0.0]},
    ):
        with pytest.raises(ValueError):
            validate_task_output("image-feature-extraction", {**value, **change})


def test_image_header_bounds_before_pixel_decode(tmp_path):
    from PIL import Image

    path = tmp_path / "image.png"
    Image.new("RGB", (4097, 1)).save(path)
    with pytest.raises(ValueError):
        open_task_image(path)
    Image.new("RGB", (2000, 1000)).save(path)
    image, original = open_task_image(path)
    assert original == (2000, 1000) and image.size == (1024, 512)


def test_contract_negotiation_is_required_before_load(tmp_path):
    from oryn_engine.models import ModelStoreError
    from oryn_engine.protocol.dispatcher import ProtocolDispatcher

    dispatcher = ProtocolDispatcher()
    dispatcher.models = SimpleNamespace(
        installed_record=lambda model: {
            "technicalCapabilities": {"task": "image-feature-extraction"}
        }
    )
    with pytest.raises(ModelStoreError, match="Browser does not support"):
        dispatcher._require_task_contract({})
    dispatcher.dispatch(
        {
            "type": "request",
            "id": "hello",
            "method": "protocol.handshake",
            "params": {
                "supportedVersions": [1],
                "browserVersion": "0.1.0",
                "maxMessageBytes": 16777216,
            },
        }
    )
    response, _ = dispatcher.dispatch(
        {
            "v": 1,
            "type": "request",
            "id": "negotiate",
            "method": "task.contracts.negotiate",
            "params": {"contracts": ["image-embeddings-v1", "future-task-v9"]},
        }
    )
    assert response["result"] == {"contracts": ["image-embeddings-v1"]}
    dispatcher._require_task_contract({})


@pytest.mark.parametrize(
    "task,contract",
    [
        ("automatic-speech-recognition", "transcription-v1"),
        ("image-classification", "image-classification-v1"),
        ("text-classification", "text-classification-v1"),
        ("summarization", "summarization-v1"),
        ("zero-shot-image-classification", "zero-shot-image-classification-v1"),
        ("feature-extraction", "embeddings-v1"),
        ("text-ranking", "reranking-v1"),
    ],
)
def test_base_contract_stays_available_when_shared_executor_is_added(task, contract):
    from oryn_engine.protocol.dispatcher import ProtocolDispatcher

    dispatcher = ProtocolDispatcher()
    dispatcher.models = SimpleNamespace(
        installed_record=lambda model: {
            "technicalCapabilities": {"task": task},
            "runtimePack": {"taskContract": contract},
        }
    )
    assert not dispatcher.negotiated_task_contracts
    dispatcher._require_task_contract({})


@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_mlx_affine_and_layer_layout_validation(bits):
    from oryn_engine.runtime_packs.mlx_lm_worker import validate_config, validate_quantized_layers

    config = {
        "model_type": "qwen3",
        "quantization": {
            "bits": bits,
            "group_size": 64,
            "model.layers.0.self_attn.q_proj": {"bits": bits, "group_size": 64},
        },
    }
    validate_config(config, {"qwen3"})
    name = "model.layers.0.self_attn.q_proj"
    module = SimpleNamespace(weight=SimpleNamespace(shape=(64, 64)), to_quantized=lambda: None)
    model = SimpleNamespace(named_modules=lambda: [(name, module)])
    tensors = {
        name + ".weight": (64, 2 * bits),
        name + ".scales": (64, 1),
        name + ".biases": (64, 1),
    }
    validate_quantized_layers(model, config, tensors)
    with pytest.raises(ValueError):
        validate_quantized_layers(model, config, {**tensors, name + ".weight": (64, 999)})
    config["quantization"]["__import__.os"] = False
    with pytest.raises(ValueError):
        validate_config(config, {"qwen3"})


def test_native_registry_retains_withheld_families_and_complete_constraints():
    from oryn_engine.native_capability_sources import native_registry, validate_gguf_policy

    registry = native_registry()
    supported = next(
        v for v in registry.tuples if v.architecture == "lfm2" and v.source_kind == "gguf"
    )
    assert supported.constraints["templates"] == ["chatml"]
    withheld = next(
        v for v in registry.tuples if v.architecture == "gpt-oss" and v.source_kind == "gguf"
    )
    assert withheld.policy_state == "withheld" and withheld.policy_reasons
    value = SimpleNamespace(
        architecture="lfm2",
        version=3,
        quantization="Q4_K_M",
        tokenizer_model="gpt2",
        chat_template="<|im_start|>x<|im_end|>",
    )
    validate_gguf_policy(value)
    for field, invalid in [
        ("tokenizer_model", "custom"),
        ("quantization", "MXFP4"),
        ("chat_template", "{{ messages }}"),
    ]:
        values = vars(value).copy()
        values[field] = invalid
        with pytest.raises(ValueError):
            validate_gguf_policy(SimpleNamespace(**values))


def test_diffusion_contract_never_discards_requested_semantics():
    from oryn_engine.diffusion_tasks import image_options, seal_tokenizer, validate_pipeline_index

    value = {"width": 512, "height": 512, "negativePrompt": ""}
    options = {"negative_prompt": "", "prompt": "a bird"}
    assert image_options("Flux2KleinPipeline", "text-to-image-v1", value, options) == {
        "prompt": "a bird",
        "max_sequence_length": 512,
    }
    with pytest.raises(ValueError):
        image_options(
            "Flux2KleinPipeline", "text-to-image-v1", {**value, "negativePrompt": "blur"}, options
        )
    with pytest.raises(ValueError):
        image_options("ZImagePipeline", "image-to-image-v1", value, options)
    with pytest.raises(ValueError):
        validate_pipeline_index(
            {"_class_name": "ZImagePipeline", "transformer": ["diffusers", "ArbitraryNativeClass"]}
        )
    tokenizer = SimpleNamespace(
        get_vocab=lambda: {"<|im_start|>": 1, "<|im_end|>": 2},
        chat_template="{{ raise_exception('should not render') }}",
    )
    seal_tokenizer(tokenizer)
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": "bird"}])
    assert "bird" in prompt and "should not render" not in prompt


@pytest.mark.parametrize(
    "config",
    [
        {"size": {"height": 999999, "width": 1}},
        {"image_size": [1, 2**31]},
        {"max_pixels": 10**9},
        {"max_patches": 10**9},
    ],
)
def test_processor_allocation_bounds(config):
    with pytest.raises(ValueError):
        validate_model_config(config)


def test_valid_multi_stage_vision_config_and_invalid_class_identity():
    validate_model_config(
        {"model_type": "segformer", "num_attention_heads": [1, 2, 5, 8], "image_size": 224}
    )
    for config in (
        {"architectures": [["unsafe"]]},
        {"_class_name": ["custom", "Pipeline"]},
        {"num_attention_heads": [1, 10**9]},
    ):
        with pytest.raises(ValueError):
            validate_model_config(config)


def test_metadata_redirect_cannot_forward_credentials_to_another_host(monkeypatch):
    import contextlib

    from oryn_engine.models import HuggingFaceClient, ModelStoreError

    calls = []

    @contextlib.contextmanager
    def stream(method, url, **kwargs):
        calls.append((method, url, kwargs))
        yield SimpleNamespace(
            status_code=302, headers={"location": "https://attacker.invalid/config.json"}
        )

    monkeypatch.setattr("oryn_engine.models.httpx.stream", stream)
    with pytest.raises(ModelStoreError, match="outside its host"):
        HuggingFaceClient(lambda: "fixture-token").read_json("org/model", "a" * 40, "config.json")
    assert len(calls) == 1 and calls[0][1].startswith("https://huggingface.co/")
    for name in ("../config.json", "/config.json", "weights.safetensors", "a\\config.json"):
        with pytest.raises(ModelStoreError):
            HuggingFaceClient().read_json("org/model", "a" * 40, name)
    assert len(calls) == 1


def test_audit_and_live_inspection_share_new_task_admission(tmp_path):
    from test_metadata_audit import MetadataClient
    from test_models import FakeHub, model

    from oryn_engine.audit_metadata import MetadataSnapshot
    from oryn_engine.metadata_audit import inspect_row
    from oryn_engine.models import ModelStore, ModelStoreError

    config = {"model_type": "dinov2", "architectures": ["Dinov2Model"]}
    files = [
        ("config.json", 1, "a" * 64),
        ("model.safetensors", 100, "b" * 64),
        ("preprocessor_config.json", 1, "c" * 64),
    ]
    store = ModelStore(
        tmp_path,
        hub=FakeHub(config=config, files=files, pipeline_tag="image-feature-extraction"),
        runtime_available=True,
        runtime_planning_state_provider=lambda: ("balanced", 16 * 1024**3),
    )
    store._pack_runtime_locator = SimpleNamespace(resolve=lambda pack: object())
    store._hub.read_json = lambda *args: {"image_processor_type": "BitImageProcessor"}
    live = store.inspect(model())

    def audit(name, root_config, *, invalid_processor=False):
        client = MetadataClient(
            config=root_config, files=files, pipeline_tag="image-feature-extraction"
        )

        def metadata(repository, revision, filename):
            if filename == "config.json":
                return root_config
            if invalid_processor:
                try:
                    return parse_metadata(b'{"size":224,"size":384}')
                except ValueError as exc:
                    raise ModelStoreError(
                        "MODEL_METADATA_INVALID", "Invalid processor JSON."
                    ) from exc
            return {"image_processor_type": "BitImageProcessor"}

        client.read_json.side_effect = metadata
        snapshot = MetadataSnapshot(tmp_path / f"{name}.json", "test/model", "main", client=client)
        result = inspect_row(
            {"row": 1, "input": "test/model", "task": "image-feature-extraction"}, snapshot
        )
        client.download.assert_not_called()
        return result

    decision = audit("valid", config)
    assert decision["status"] == "metadata-compatible" and live["executable"] is True
    assert decision["executionVerified"] is False
    for axis in ("access", "artifacts", "runtime", "task", "conformance"):
        assert decision["inspectorProfile"]["eligibility"][axis] == live["eligibility"][axis]

    store._hub.read_json = lambda *args: parse_metadata(b'{"size":224,"size":384}')
    blocked = store.inspect(model())
    assert blocked["executable"] is False
    assert "MODEL_METADATA_INVALID" in blocked["blockingReasons"]
    invalid = audit("invalid", config, invalid_processor=True)
    assert invalid["status"] == "metadata-incompatible"
    assert "MODEL_METADATA_INVALID" in invalid["reasons"]

    custom_config = {**config, "auto_map": {"AutoModel": "custom.Model"}}
    custom = audit("custom", custom_config)
    assert custom["status"] == "metadata-incompatible"
    assert "MODEL_METADATA_INVALID" in custom["reasons"]


def test_native_point_coordinates_use_float32_without_changing_token_ids():
    from oryn_engine.shared_tasks import _move

    value = _move(
        {
            "input_points": torch.tensor([[[0.5, 0.5]]], dtype=torch.float64),
            "input_labels": torch.tensor([[1]], dtype=torch.int64),
        },
        "cpu",
    )
    assert value["input_points"].dtype == torch.float32
    assert value["input_labels"].dtype == torch.int64


def test_untrusted_identity_errors_become_an_inspection_rejection():
    from oryn_engine.capability_sources import framework_registry
    from oryn_engine.model_inspection import inspect_descriptor

    assessment = inspect_descriptor(
        repository="test/model",
        revision="a" * 40,
        files=["model_index.json"],
        config={"_class_name": ["custom", "Pipeline"]},
        task="text-to-image",
        registry=framework_registry(),
        access="available",
        source_kind="diffusers",
    )
    assert "MODEL_METADATA_INVALID" in assessment.eligibility.axes["runtime"].reasons
    assert assessment.eligibility.axes["runtime"].state == "unsupported"
    assert assessment.eligibility.disposition is None  # static inspection has no device admission


def test_qwen_area_metadata_uses_sealed_image_budget():
    from oryn_engine.shared_tasks import qwen_image_options, validate_processor_config

    config = {
        "image_processor": {
            "image_processor_type": "Qwen2VLImageProcessorFast",
            "size": {"shortest_edge": 65536, "longest_edge": 16777216},
            "patch_size": 16,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        },
        "processor_class": "Qwen3VLProcessor",
    }
    validate_processor_config(config, "qwen3_5")
    options = qwen_image_options(config, {"patch_size": 16})
    assert options["size"] == {"shortest_edge": 1024, "longest_edge": 1024**2}
    assert config["image_processor"]["size"]["longest_edge"] == 16777216
    config["image_processor"].update(
        min_pixels=None,
        max_pixels=None,
        crop_size=None,
        do_center_crop=None,
        device=None,
        disable_grouping=None,
        do_pad=None,
        input_data_format=None,
        pad_size=None,
        return_tensors=None,
        default_to_square=True,
    )
    validate_processor_config(config, "qwen3_vl")
    assert qwen_image_options(config, {"patch_size": 16}) == options
    for change in ({"device": "cuda"}, {"do_center_crop": True}, {"do_resize": False}):
        altered = {"image_processor": config["image_processor"] | change}
        with pytest.raises(ValueError):
            qwen_image_options(altered, {"patch_size": 16})
    with pytest.raises(ValueError):
        validate_processor_config(config, "blip")
    with pytest.raises(ValueError):
        validate_processor_config({**config, "auto_map": {"AutoProcessor": "custom.P"}}, "qwen3_5")
    with pytest.raises(ValueError):
        qwen_image_options(config, {"patch_size": 1})
    config["image_processor"]["image_std"] = [0, 0, 0]
    with pytest.raises(ValueError):
        qwen_image_options(config, {"patch_size": 16})


def test_signed_native_policy_selects_unambiguous_sealed_template():
    from oryn_engine.native_capability_sources import llama_cpp_chat_template, validate_gguf_policy

    chatml = "<|im_start|>user<|im_end|>"
    llama3 = "<|start_header_id|>user<|eot_id|>"
    assert llama_cpp_chat_template("llama", chatml) == "chatml"
    assert llama_cpp_chat_template("llama", llama3) == "llama3"
    assert llama_cpp_chat_template("llama", chatml + llama3) is None
    assert llama_cpp_chat_template("qwen2", llama3) is None
    metadata = SimpleNamespace(
        architecture="llama",
        version=3,
        quantization="IQ4_XS",
        tokenizer_model="gpt2",
        chat_template=chatml,
    )
    validate_gguf_policy(metadata, "chatml")
    with pytest.raises(ValueError):
        validate_gguf_policy(metadata, "llama3")


def test_sealed_qwen_processor_classes_exist_in_pinned_transformers():
    import transformers as tf

    assert tf.Qwen2VLProcessor
    assert tf.Qwen2_5_VLProcessor
    assert tf.Qwen3VLProcessor
    assert tf.Qwen2VLImageProcessor
    assert tf.Qwen3VLVideoProcessor


def test_ace_legacy_mapping_is_removed_only_for_the_sealed_native_component(tmp_path):
    from oryn_engine.runtime_packs.ace_step_worker import _component_config

    root = tmp_path / "acestep-v15-turbo"
    root.mkdir()
    config = {
        "model_type": "ace_step",
        "auto_map": {
            "AutoConfig": "configuration_acestep_v15.AceStepConfig",
            "AutoModel": "modeling_acestep_v15_turbo.AceStepConditionGenerationModel",
        },
    }
    (root / "config.json").write_text(json.dumps(config))
    assert _component_config(root) == {"model_type": "ace_step"}
    config["auto_map"]["AutoModel"] = "repository.Code"
    (root / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        _component_config(root)


def test_mlx_language_only_projection_does_not_grant_vision_input():
    from oryn_engine.model_inspection import effective_task

    config = {"model_type": "qwen3_5_moe"}
    assert effective_task("mlx", "image-text-to-text", config) == "text-generation"
    assert (
        effective_task("mlx", "image-text-to-text", config | {"vision_config": {"depth": 1}})
        == "image-text-to-text"
    )
    assert effective_task("transformers", "image-text-to-text", config) == "image-text-to-text"
    assert (
        effective_task("mlx", "image-text-to-text", {"model_type": "qwen3_5"})
        == "image-text-to-text"
    )
    assert (
        effective_task("mlx", "image-text-to-text", {"model_type": "unknown"})
        == "image-text-to-text"
    )
