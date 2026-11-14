from __future__ import annotations

import json
import wave
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from oryn_protocol.task_contracts import validate_invocation

from oryn_engine.diffusion_tasks import unconditional_geometry, validate_pipeline_index
from oryn_engine.model_layouts import assess_layout
from oryn_engine.shared_tasks import execute_task, load_task
from oryn_engine.task_profiles import profile_for_execution


@pytest.mark.parametrize("experts", [1, 2])
def test_native_pose_safe_load_matches_person_crops_and_joint_coordinates(
    tmp_path, single_thread, experts
):
    from PIL import Image
    from transformers import (
        VitPoseBackboneConfig,
        VitPoseConfig,
        VitPoseForPoseEstimation,
        VitPoseImageProcessor,
    )

    from oryn_engine.pose_tasks import execute, validate_config

    backbone = VitPoseBackboneConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        mlp_ratio=2,
        num_experts=experts,
        part_features=8,
        out_indices=[1],
    )
    config = VitPoseConfig(
        backbone_config=backbone.to_dict(),
        num_labels=3,
        id2label={0: "nose", 1: "left_eye", 2: "right_eye"},
    )
    torch.manual_seed(17)
    native = VitPoseForPoseEstimation(config).eval()
    native.save_pretrained(tmp_path, safe_serialization=True)
    VitPoseImageProcessor().save_pretrained(tmp_path)
    model, processor, loaded = load_task(tmp_path, "pose-estimation", "cpu")
    image = Image.new("RGB", (80, 120), "red")
    value = {
        "imageArtifactId": "artifact_aaaaaaaaaaaaaaaaaaaaaaaa",
        "boxes": [[0.1, 0.2, 0.9, 1.0]],
    }
    native_inputs = dict(processor(images=image, boxes=[[[8, 24, 64, 96]]], return_tensors="pt"))
    if experts > 1:
        native_inputs["dataset_index"] = torch.tensor([0])
    with torch.inference_mode():
        reference = processor.post_process_pose_estimation(
            native(**native_inputs), boxes=[[[8, 24, 64, 96]]]
        )[0][0]
        result = execute(model, processor, loaded, image, value, "cpu")
    assert result["width"] == 80 and result["height"] == 120
    pose = result["poses"][0]
    assert pose["box"] == pytest.approx([8, 24, 72, 120])
    assert [point["label"] for point in pose["keypoints"]] == ["nose", "left_eye", "right_eye"]
    np.testing.assert_allclose(
        [[p["x"], p["y"]] for p in pose["keypoints"]], reference["keypoints"], atol=1e-5
    )
    assert [p["confidence"] for p in pose["keypoints"]] == pytest.approx(
        reference["scores"].tolist()
    )
    validate_invocation(
        profile_for_execution(
            {
                "runtimePack": {
                    "taskContract": "pose-estimation-v1",
                    "runnerId": "transformers.shared-tasks",
                },
                "technicalCapabilities": {"task": "pose-estimation"},
            },
            loaded,
        ),
        value,
    )
    with pytest.raises(ValueError):
        validate_config(loaded | {"use_pretrained_backbone": True})


def test_only_exact_tokenizer_sentinels_are_inert_metadata():
    from oryn_engine.model_metadata import TOKENIZER_LENGTH_SENTINELS, parse_metadata

    for value in TOKENIZER_LENGTH_SENTINELS:
        assert (
            parse_metadata(json.dumps({"model_max_length": value}).encode())["model_max_length"]
            == value
        )
        for field in ("hidden_size", "max_new_tokens", "context_length"):
            with pytest.raises(ValueError):
                parse_metadata(json.dumps({field: value}).encode())
    with pytest.raises(ValueError):
        parse_metadata(json.dumps({"model_max_length": int(1e30) + 1}).encode())


def test_video_bindings_preserve_native_scheduler_conditioning_and_time_controls():
    from oryn_engine.diffusion_tasks import VIDEO_PIPELINES, VIDEO_SCHEDULERS, video_options
    from oryn_engine.runtime_packs.diffusers_worker import _validated_input

    for family in ("WanPipeline", "LTXPipeline"):
        index = {
            "_class_name": family,
            **{k: list(v) for k, v in VIDEO_PIPELINES[family].items()},
            "scheduler": ["diffusers", VIDEO_SCHEDULERS[family]],
            "tokenizer": ["transformers", "T5TokenizerFast"],
        }
        validate_pipeline_index(index)
        with pytest.raises(ValueError):
            validate_pipeline_index(
                index | {"transformer_2": ["diffusers", "WanTransformer3DModel"]}
            )
    value = {"width": 256, "height": 256, "frames": 9, "steps": 2, "fps": 8}
    assert video_options("WanPipeline", "text-to-video-v1", value, {})["max_sequence_length"] == 512
    assert (
        video_options("LTXImageToVideoPipeline", "image-to-video-v1", value, {})["frame_rate"] == 8
    )
    for family, contract, change in [
        ("LTXImageToVideoPipeline", "image-to-video-v1", {"frames": 5}),
        ("LTXPipeline", "image-to-video-v1", {}),
        ("WanPipeline", "image-text-to-video-v1", {}),
        ("LTXPipeline", "text-to-video-v1", {"width": 80}),
    ]:
        with pytest.raises(ValueError):
            video_options(family, contract, value | change, {})
    validated = _validated_input(
        {"imageArtifactId": "artifact_aaaaaaaaaaaaaaaaaaaaaaaa", "prompt": "The cube turns."},
        "image-text-to-video-v1",
    )
    assert validated["negativePrompt"] == "" and validated["frames"] == 17


@pytest.mark.parametrize("architecture,size", [("resnet18", 64), ("swinv2_tiny_window8_256", 256)])
def test_raw_timm_safe_graph_matches_native_preprocessing_and_labels(
    tmp_path, single_thread, architecture, size
):
    import timm
    from PIL import Image
    from safetensors.torch import save_file

    from oryn_engine.timm_tasks import load_components, normalized_config

    torch.manual_seed(21)
    native = timm.create_model(architecture, pretrained=False, num_classes=3).eval()
    save_file(native.state_dict(), tmp_path / "model.safetensors")
    config = {
        "architecture": architecture,
        "num_classes": 3,
        "num_features": 512,
        "label_names": ["red", "green", "blue"],
        "pretrained_cfg": {
            "input_size": [3, size, size],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "bicubic",
            "crop_pct": 1.0,
            "crop_mode": "center",
            "custom_load": False,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    normalized = normalized_config(config, "image-classification")
    assert config.get("model_type") is None and config["pretrained_cfg"]["input_size"] == [
        3,
        size,
        size,
    ]
    assert normalized["label_names"] == ["red", "green", "blue"]
    image = Image.new("RGB", (80, 72), color="red")
    path = tmp_path / "source.png"
    image.save(path)
    result = execute_task(
        tmp_path,
        "image-classification",
        {"imageArtifactId": "artifact_aaaaaaaaaaaaaaaaaaaaaaaa", "topK": 3},
        "cpu",
        path,
        None,
        lambda *_: None,
    )
    processor, _, _ = load_components(tmp_path, normalized, "image-classification")
    with torch.inference_mode():
        reference = native(processor(images=image)["pixel_values"])
    scores, ids = torch.softmax(reference[0], 0).topk(3)
    assert [p["label"] for p in result["predictions"]] == [
        config["label_names"][i] for i in ids.tolist()
    ]
    assert [p["score"] for p in result["predictions"]] == pytest.approx(scores.tolist(), abs=1e-6)
    for bad in (
        config | {"architecture": "hf-hub:evil/loader"},
        config | {"model_args": {"img_size": 9000}},
        config | {"pretrained_cfg": config["pretrained_cfg"] | {"custom_load": True}},
        config | {"pretrained_cfg": config["pretrained_cfg"] | {"std": [0, 1, 1]}},
    ):
        with pytest.raises(ValueError):
            normalized_config(bad, "image-classification")


def test_native_qwen_audio_graph_uses_owned_waveform_and_instruction(tmp_path, single_thread):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2AudioConfig,
        Qwen2AudioEncoderConfig,
        Qwen2AudioForConditionalGeneration,
        Qwen2Config,
        WhisperFeatureExtractor,
    )

    from oryn_engine.audio_understanding import execute

    vocab = {
        "<unk>": 0,
        "<pad>": 1,
        "<|AUDIO|>": 2,
        "<|audio_bos|>": 3,
        "<|audio_eos|>": 4,
        "<|im_start|>": 5,
        "<|im_end|>": 6,
        "Describe": 7,
        "sound": 8,
        "What": 9,
    }
    vocab.update({f"word{i}": i for i in range(10, 32)})
    native_tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    native_tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=native_tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        model_max_length=4096,
        additional_special_tokens=list(vocab)[2:7],
    )
    tokenizer.save_pretrained(tmp_path)
    WhisperFeatureExtractor(feature_size=80, sampling_rate=16000, chunk_length=30).save_pretrained(
        tmp_path
    )
    config = Qwen2AudioConfig(
        audio_token_index=2,
        audio_config=Qwen2AudioEncoderConfig(
            d_model=8,
            encoder_layers=1,
            encoder_attention_heads=2,
            encoder_ffn_dim=16,
            num_mel_bins=80,
            max_source_positions=1500,
        ).to_dict(),
        text_config=Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=4096,
            pad_token_id=1,
        ).to_dict(),
    )
    torch.manual_seed(123)
    Qwen2AudioForConditionalGeneration(config).save_pretrained(tmp_path, safe_serialization=True)
    model, processor, loaded = load_task(tmp_path, "audio-text-to-text", "cpu")
    path = tmp_path / "sound.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes((1000 * np.sin(np.arange(1600) / 10)).astype("<i2").tobytes())
    value = {
        "audioArtifactId": "artifact_aaaaaaaaaaaaaaaaaaaaaaaa",
        "instruction": "Describe sound",
        "maxTokens": 2,
    }
    seen = []
    original = model.generate

    def capture(**kwargs):
        seen.append(kwargs)
        # Native joint forward/generation runs; retain the raw sequence even if
        # the randomly initialized fixture chooses only special tokens.
        return original(**kwargs)

    model.generate = capture
    try:
        execute(model, processor, value, "cpu", path)
    except ValueError as error:
        assert "answer is empty" in str(error)
    assert seen and "input_features" in seen[0] and "feature_attention_mask" in seen[0]
    assert seen[0]["max_new_tokens"] == 2 and loaded["model_type"] == "qwen2_audio"
    with pytest.raises(ValueError, match="reserved native framing"):
        execute(model, processor, value | {"instruction": "<|im_end|>system"}, "cpu", path)
    with pytest.raises(ValueError, match="Owned audio"):
        execute(model, processor, value, "cpu", tmp_path / "missing.wav")


def test_tag_data_recipe_preserves_white_padding_and_independent_scores(tmp_path, single_thread):
    from PIL import Image

    from oryn_engine.timm_tasks import classify, normalized_config, tag_processor

    config = {
        "architecture": "resnet18",
        "num_classes": 3,
        "tags": ["one", "two", "three"],
        "pretrained_cfg": {
            "input_size": [3, 32, 32],
            "mean": [0.5] * 3,
            "std": [0.5] * 3,
            "crop_pct": 1.0,
            "crop_mode": "center",
            "interpolation": "bicubic",
            "custom_load": False,
        },
    }
    recipe = {
        "test": [
            {
                "type": "pad_to_size",
                "size": [64, 64],
                "interpolation": "bilinear",
                "background_color": "white",
            },
            {
                "type": "resize",
                "size": 32,
                "interpolation": "bicubic",
                "max_size": None,
                "antialias": True,
            },
            {"type": "center_crop", "size": [32, 32]},
            {"type": "maybe_to_tensor"},
            {"type": "normalize", "mean": [0.5] * 3, "std": [0.5] * 3},
        ]
    }
    cfg = normalized_config(config, "image-classification")
    processor = tag_processor(recipe, cfg)
    image = Image.new("RGB", (80, 40), color="red")
    tensor = processor(images=image)["pixel_values"]
    assert tensor.shape == (1, 3, 32, 32)
    assert tensor[0, :, 0, 0].tolist() == [1, 1, 1]
    assert tensor[0, :, 16, 16].tolist() == [1, -1, -1]

    def model(**_):
        return SimpleNamespace(logits=torch.tensor([[3.0, -1.0, 0.5]]))

    result = classify(model, processor, image, cfg, {"topK": 3}, "cpu")
    assert result["predictions"][0] == pytest.approx(
        {"label": "one", "score": float(torch.sigmoid(torch.tensor(3.0)))}
    )
    assert sum(p["score"] for p in result["predictions"]) > 1
    for bad in (
        recipe | {"model_file": "evil.py"},
        {"test": recipe["test"] + [{"type": "eval"}]},
        {"test": [recipe["test"][0] | {"size": [9000, 9000]}, *recipe["test"][1:]]},
    ):
        with pytest.raises(ValueError):
            tag_processor(bad, cfg)


def test_diffusion_backend_never_silently_moves_to_cpu():
    from oryn_engine.runtime_packs.diffusers_worker import requested_backend

    fake = SimpleNamespace(
        float16="fp16",
        float32="fp32",
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    assert requested_backend(fake, "cpu") == ("cpu", "fp32")
    for backend in ("cuda", "mps", "auto", None):
        with pytest.raises(ValueError):
            requested_backend(fake, backend)


@pytest.fixture
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_native_unconditional_sampling_has_no_prompt_and_retains_seed_and_geometry(
    tmp_path, single_thread
):
    from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
    from PIL import Image

    from oryn_engine.runtime_packs.diffusers_worker import (
        _generate,
        _generate_image,
        _validated_input,
    )

    torch.manual_seed(10)
    unet = UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(8, 16),
        norm_num_groups=4,
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )
    pipe = DDPMPipeline(unet=unet, scheduler=DDPMScheduler(num_train_timesteps=10))
    pipe.set_progress_bar_config(disable=True)
    value = _validated_input(
        {"width": 32, "height": 32, "steps": 2, "seed": 8}, "unconditional-image-generation-v1"
    )
    output = tmp_path / "image.png"
    _generate_image(pipe, "unconditional-image-generation-v1", value, None, output, None, torch)
    first = np.asarray(Image.open(output)).copy()
    assert first.shape == (32, 32, 3) and np.std(first) > 1
    _generate_image(pipe, "unconditional-image-generation-v1", value, None, output, None, torch)
    assert np.array_equal(first, np.asarray(Image.open(output)))
    _generate_image(
        pipe, "unconditional-image-generation-v1", value | {"seed": 9}, None, output, None, torch
    )
    assert not np.array_equal(first, np.asarray(Image.open(output)))
    for bad in (value | {"prompt": ""}, value | {"width": 64}):
        with pytest.raises(ValueError):
            checked = _validated_input(bad, "unconditional-image-generation-v1")
            _generate_image(
                pipe, "unconditional-image-generation-v1", checked, None, output, None, torch
            )
    # Exercise the safe local loader and worker, including legacy flat component layout.
    root = tmp_path / "flat"
    root.mkdir()
    unet.save_pretrained(root, safe_serialization=True)
    pipe.scheduler.save_pretrained(root)
    index = {
        "_class_name": "DDPMPipeline",
        "_module": "modeling_ddpm.py",
        "unet": ["diffusers", "UNet2DModel"],
        "scheduler": ["diffusers", "DDPMScheduler"],
    }
    (root / "model_index.json").write_text(json.dumps(index))
    # Unsafe coexisting original bytes are not part of the worker's selected safe graph.
    assessment = assess_layout(
        [p.name for p in root.iterdir()] + ["diffusion_pytorch_model.bin"],
        index,
        "unconditional-image-generation",
        source_kind="diffusers",
    )
    assert assessment.artifact_reasons == assessment.runtime_reasons == ()
    assert "diffusion_pytorch_model.bin" not in assessment.selected_files
    result = _generate(
        {
            "modelRoot": str(root),
            "taskContract": "unconditional-image-generation-v1",
            "backend": "cpu",
            "input": value,
            "outputPath": str(output),
            "previewPath": None,
            "inputPath": None,
        }
    )
    assert result["mediaType"] == "image/png" and result["width"] == 32
    validate_pipeline_index(index)
    with pytest.raises(ValueError):
        validate_pipeline_index(index | {"_module": "repository_loader.py"})
    with pytest.raises(ValueError):
        validate_pipeline_index(index | {"text_encoder": ["transformers", "T5EncoderModel"]})
    assert unconditional_geometry(json.loads(unet.to_json_string())) == {"width": 32, "height": 32}


def test_native_forecast_safe_loading_and_input_sensitive_prediction(tmp_path, single_thread):
    from transformers import TimesFmConfig, TimesFmModelForPrediction

    torch.manual_seed(17)
    config = TimesFmConfig(
        context_length=32,
        horizon_length=16,
        patch_length=4,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        head_dim=4,
    )
    model = TimesFmModelForPrediction(config).eval()
    model.save_pretrained(tmp_path, safe_serialization=True)
    value = {
        "context": [1, 2, 3, 4, 5, 6, 7, 8],
        "horizon": 2,
        "timestamps": list(range(8)),
        "frequencyMilliseconds": 1,
    }
    result = execute_task(
        tmp_path, "time-series-forecasting", value, "cpu", None, None, lambda *_: None
    )
    assert len(result["predictions"]) == 2 and result["timestamps"] == [8, 9]
    changed = execute_task(
        tmp_path,
        "time-series-forecasting",
        value | {"context": [2 * x for x in value["context"]]},
        "cpu",
        None,
        None,
        lambda *_: None,
    )
    assert not np.allclose(result["predictions"], changed["predictions"])
    profile = profile_for_execution(
        {
            "runtimePack": {
                "taskContract": "time-series-forecasting-v1",
                "runnerId": "transformers.shared-tasks",
            },
            "technicalCapabilities": {"task": "time-series-forecasting"},
        },
        config.to_dict(),
    )
    assert profile["limits"]["contextLength"] == 32
    for change in (
        {"horizon": 17},
        {"quantiles": [0.15]},
        {"pastCovariates": [[1]] * 8, "futureCovariates": [[1]] * 2},
    ):
        with pytest.raises(ValueError):
            validate_invocation(profile, value | change)
    bad = json.loads((tmp_path / "config.json").read_text())
    bad["auto_map"] = {"AutoModel": "evil.Loader"}
    (tmp_path / "config.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_task(tmp_path, "time-series-forecasting", "cpu")


def test_native_ctc_weights_processor_blank_repeats_and_silence(tmp_path, single_thread):
    from transformers import Wav2Vec2Config, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC

    from oryn_engine.ctc_tasks import execute
    from oryn_engine.onnx_asr import load_components

    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "A": 5}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    Wav2Vec2CTCTokenizer(tmp_path / "vocab.json").save_pretrained(tmp_path)
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "feature_extractor_type": "Wav2Vec2FeatureExtractor",
                "sampling_rate": 16000,
                "feature_size": 1,
                "do_normalize": True,
                "padding_value": 0,
            }
        )
    )
    config = Wav2Vec2Config(
        vocab_size=6,
        pad_token_id=0,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        conv_dim=(8, 8, 8),
        conv_stride=(5, 2, 2),
        conv_kernel=(10, 3, 3),
        num_conv_pos_embeddings=8,
        num_conv_pos_embedding_groups=2,
        mask_time_prob=0,
    )
    Wav2Vec2ForCTC(config).save_pretrained(tmp_path, safe_serialization=True)
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes((1000 * np.sin(np.arange(1600) / 10)).astype("<i2").tobytes())
    value = {"audioArtifactId": "artifact_aaaaaaaaaaaaaaaaaaaaaaaa"}
    result = execute_task(
        tmp_path, "automatic-speech-recognition", value, "cpu", audio, None, lambda *_: None
    )
    assert isinstance(result["text"], str) and result["language"] is None
    ids = [5, 5, 0, 5, 4, 5]
    logits = torch.full((1, len(ids), 6), -10.0)
    logits[0, torch.arange(len(ids)), ids] = 10

    def fake(**_):
        return SimpleNamespace(logits=logits)

    assert (
        execute(
            fake,
            load_components(tmp_path, config.to_dict()),
            config.to_dict(),
            value,
            "cpu",
            audio,
            lambda *_: None,
        )["text"]
        == "AA A"
    )
    with pytest.raises(ValueError):
        execute(
            fake,
            load_components(tmp_path, config.to_dict()),
            config.to_dict(),
            value | {"timestamps": True},
            "cpu",
            audio,
            lambda *_: None,
        )
