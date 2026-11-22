"""Read the reviewed build artifact; production never reflects over repository code."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from oryn_engine.runtime_packs.manifests import SignedManifestVerifier, TrustedPublisherKey


@lru_cache(maxsize=1)
def generated_artifact() -> dict[str, Any]:
    from oryn_engine.generated_trust import DEVELOPMENT_KEY_ID, DEVELOPMENT_MODULUS

    root = Path(__file__).parent / "data"
    raw = (root / "capabilities.json").read_bytes()
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("Capability build artifact is oversized.")
    envelope = json.loads((root / "capabilities.signature.json").read_bytes())
    key = TrustedPublisherKey(DEVELOPMENT_KEY_ID, DEVELOPMENT_MODULUS)
    payload, _ = SignedManifestVerifier({key.key_id: key}).verify(envelope)
    if payload != {
        "kind": "oryn-capability-build",
        "schemaVersion": 1,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }:
        raise ValueError("Capability build identity mismatch.")
    result = json.loads(raw)
    if (
        set(result) != {"schemaVersion", "sources", "mlxModelTypes", "buildInputs"}
        or result["schemaVersion"] != 1
    ):
        raise ValueError("Capability build schema is invalid.")
    return result


def generated_sources() -> tuple[dict[str, Any], ...]:
    return tuple(generated_artifact()["sources"])


def mlx_model_types() -> set[str]:
    return set(generated_artifact()["mlxModelTypes"])


def verify_audio_build(mlx_audio_package: Path) -> None:
    """Verify the isolated audio pack against the reviewed generated source inventory."""
    inputs = generated_artifact()["buildInputs"]["mlxAudio"]
    actual = {
        str(path.relative_to(mlx_audio_package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in mlx_audio_package.rglob("*.py")
    }
    if actual != inputs["sources"]:
        raise ValueError("Packaged MLX-Audio sources differ from capability build inputs.")
    root = Path(__file__).parent
    for name, expected in inputs["adapterSources"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("MLX-Audio adapter differs from capability build inputs.")


def normalized_transformers_config(config: dict, task: str | None) -> dict:
    """Resolve omitted family/class identities from unambiguous signed native mappings."""
    from oryn_engine.audio8_tasks import normalized_config as normalized_audio8
    from oryn_engine.aurora_tasks import normalized_config as normalized_aurora
    from oryn_engine.birefnet_tasks import normalized_config as normalized_birefnet
    from oryn_engine.colqwen3_tasks import normalized_config as normalized_retrieval_config
    from oryn_engine.contextual_embeddings import normalized_config as normalized_contextual
    from oryn_engine.ct2_tasks import normalized_config as normalized_ct2
    from oryn_engine.depth3_tasks import normalized_config as normalized_depth_config
    from oryn_engine.giga_embeddings import normalized_config as normalized_embedding_config
    from oryn_engine.isnet_tasks import normalized_config as normalized_isnet
    from oryn_engine.jina4_tasks import normalized_config as normalized_jina4
    from oryn_engine.jina_ranking import normalized_config as normalized_jina_ranking
    from oryn_engine.llava_next_tasks import normalized_config as normalized_llava_next
    from oryn_engine.llava_video_qwen import normalized_config as normalized_llava_video_qwen
    from oryn_engine.mlx_bolding import normalized_config as normalized_bolding
    from oryn_engine.molparser_tasks import normalized_config as normalized_molparser
    from oryn_engine.mpsenet_tasks import normalized_config as normalized_mpsenet
    from oryn_engine.numeric_coverage import normalized_config as normalized_vivit
    from oryn_engine.openclip_tasks import normalized_config as normalized_openclip
    from oryn_engine.pii_tasks import normalized_config as normalized_pii
    from oryn_engine.pixai_tasks import normalized_config as normalized_pixai
    from oryn_engine.qwen_retrieval import normalized_config as normalized_ranking_config
    from oryn_engine.reuse_tasks import normalized_config as normalized_reuse
    from oryn_engine.rtmw_tasks import normalized_config as normalized_rtmw
    from oryn_engine.sopro_tasks import normalized_config as normalized_sopro
    from oryn_engine.speechx_tasks import normalized_config as normalized_speechx
    from oryn_engine.tiger_tasks import normalized_config as normalized_tiger
    from oryn_engine.timesfm3_tasks import normalized_config
    from oryn_engine.tips_tasks import normalized_config as normalized_tips
    from oryn_engine.turn_classification import normalized_config as normalized_turn
    from oryn_engine.videollama3_tasks import normalized_config as normalized_videollama3
    from oryn_engine.wav2vec_heads import normalized_config as normalized_audio_heads

    config = normalized_bolding(config, task)
    config = normalized_llava_video_qwen(config, task)
    config = normalized_videollama3(config, task)
    config = normalized_tiger(config, task)
    config = normalized_reuse(config, task)
    config = normalized_mpsenet(config, task)
    config = normalized_molparser(config, task)
    config = normalized_sopro(config, task)
    config = normalized_speechx(config, task)
    config = normalized_audio8(config, task)
    config = normalized_turn(config, task)
    config = normalized_ct2(config, task)
    task = "feature-extraction" if task == "sentence-similarity" else task
    config = normalized_embedding_config(
        normalized_retrieval_config(normalized_config(config, task), task), task
    )
    config = normalized_ranking_config(config, task)
    config = normalized_jina_ranking(config, task)
    config = normalized_pii(config, task)
    config = normalized_depth_config(config, task)
    config = normalized_llava_next(config, task)
    config = normalized_contextual(config, task)
    config = normalized_aurora(config, task)
    config = normalized_openclip(config, task)
    config = normalized_birefnet(config, task)
    config = normalized_isnet(config, task)
    config = normalized_pixai(config, task)
    config = normalized_tips(config, task)
    config = normalized_rtmw(config, task)
    config = normalized_jina4(config, task)
    config = normalized_audio_heads(config, task)
    config = normalized_vivit(config, task)
    mapping = (
        generated_artifact()["buildInputs"].get("transformersTaskArchitectures", {}).get(task, {})
    )
    if "model_type" not in config and not (
        config.get("auto_map")
        or config.get("custom_pipelines")
        or config.get("model_file") is not None
        or config.get("trust_remote_code") is True
    ):
        architectures = config.get("architectures")
        if isinstance(architectures, list) and len(architectures) == 1:
            families = [family for family, names in mapping.items() if names == architectures]
            if len(families) == 1:
                config = config | {"model_type": families[0]}
    names = mapping.get(config.get("model_type"), [])
    if "architectures" in config:
        # Pinned Transformers renamed these native public classes. Model loading
        # still selects the same graph from model_type; require that exact signed
        # Auto mapping before accepting the historical spelling.
        legacy = {
            ("object-detection", "rt_detr"): (
                "RtDetrForObjectDetection",
                "RTDetrForObjectDetection",
            ),
            ("object-detection", "rt_detr_v2"): (
                "RtDetrV2ForObjectDetection",
                "RTDetrV2ForObjectDetection",
            ),
            ("video-classification", "vivit"): (
                "ViViTForVideoClassification",
                "VivitForVideoClassification",
            ),
            ("visual-question-answering", "vilt"): (
                "ViltForVisualQuestionAnswering",
                "ViltForQuestionAnswering",
            ),
        }.get((task, config.get("model_type")))
        if legacy and config["architectures"] == [legacy[0]] and names == [legacy[1]]:
            return config | {"architectures": list(names)}
        return config
    return config | {"architectures": list(names)} if len(names) == 1 else config


def verify_build_inputs(*, mlx_package: Path, diffusers_package: Path, llama_binary: Path) -> None:
    """Release gate: capability declarations must match the exact installed library bytes."""
    from importlib import import_module
    from importlib.metadata import version

    import autogluon.tabular.models.mitra as mitra
    import autogluon.tabular.models.tabpfnmix as tabpfnmix
    import chronos
    import ctranslate2
    import cv2
    import depth_anything_3
    import librosa
    import tabpfn
    import timesfm3
    import timm
    from tiktoken import _tiktoken
    from transformers.models.auto import modeling_auto

    from oryn_engine.runtime_packs.llama_cpp_worker import VISION_LIBRARY

    inputs = generated_artifact()["buildInputs"]
    from oryn_engine.reviewed_implementations import installed_index

    installed_index()  # Release gate: signed source profiles match installed code/runtime.
    for dependency, expected in inputs["rnntDependencies"].items():
        if version(dependency) != expected:
            raise ValueError("RNN-T audio runtime version differs from capability data.")
    for dependency, expected in inputs["openclipDependencies"].items():
        if version(dependency) != expected:
            raise ValueError("OpenCLIP runtime version differs from capability data.")
    for dependency in ("h5py", "sacremoses", "tiktoken", "ctranslate2"):
        if version(dependency) != inputs[dependency]:
            raise ValueError(f"{dependency} build version differs from capability data.")
    if version("autogluon.tabular") != inputs["autogluonTabular"]:
        raise ValueError("Mitra build version differs from capability data.")
    if version("transformers") != inputs["transformers"]:
        raise ValueError("Transformers build version differs from capability data.")
    if version("chronos-forecasting") != inputs["chronosForecasting"]:
        raise ValueError("Chronos build version differs from capability data.")
    if version("timesfm") != inputs["timesfm"]:
        raise ValueError("TimesFM build version differs from capability data.")
    for dependency in ("oryn-depth-native", "opencv-python-headless", "addict", "omegaconf"):
        if version(dependency) != inputs["depthDependencies"][dependency]:
            raise ValueError("Depth runtime version differs from capability data.")
    if version("tabpfn") != inputs["tabpfn"]:
        raise ValueError("TabPFN build version differs from capability data.")
    if (
        version("timm") != inputs["timm"]
        or sorted(timm.list_models()) != inputs["timmArchitectures"]
    ):
        raise ValueError("Native vision build differs from capability data.")
    paths = {
        **{
            Path(ctranslate2.__file__).parent / name: digest
            for name, digest in inputs["ctranslate2Sources"].items()
        },
        **{
            Path(librosa.__file__).parent / name: digest
            for name, digest in inputs["librosaSources"].items()
        },
        **{
            Path(import_module(module).__file__).parent / name: digest
            for module, sources in inputs["openclipSources"].items()
            for name, digest in sources.items()
        },
        Path(cv2.__file__).parent / inputs["opencvBinaryName"]: inputs["opencvBinarySha256"],
        **{
            Path(depth_anything_3.__file__).parent / name: digest
            for name, digest in inputs["depthNativeSources"].items()
        },
        Path(_tiktoken.__file__): inputs["tiktokenBinarySha256"],
        **{
            Path(__file__).parent / name: digest
            for name, digest in inputs.get("orynNativeSources", {}).items()
        },
        llama_binary: inputs["llamaBinarySha256"],
        llama_binary.with_name("llama-mtmd-cli"): inputs["llamaVisionBinarySha256"],
        llama_binary.with_name(VISION_LIBRARY): inputs["llamaVisionLibrarySha256"],
        Path(modeling_auto.__file__): inputs["transformersMappingSha256"],
        **{
            Path(modeling_auto.__file__).parents[2] / name: digest
            for name, digest in inputs["transformersVisionSources"].items()
        },
        **{
            Path(tabpfnmix.__file__).parent / name: digest
            for name, digest in inputs["tabpfnMixSources"].items()
        },
        **{
            Path(mitra.__file__).parent / name: digest
            for name, digest in inputs["mitraSources"].items()
        },
        **{
            Path(tabpfn.__file__).parent / name: digest
            for name, digest in inputs["tabpfnSources"].items()
        },
        **{
            Path(chronos.__file__).parent / name: digest
            for name, digest in inputs["chronosSources"].items()
        },
        **{
            Path(timesfm3.__file__).parent / name: digest
            for name, digest in inputs["timesfm3Sources"].items()
        },
        **{
            Path(timm.__file__).parent / name: digest
            for name, digest in inputs["timmSources"].items()
        },
        **{
            Path(__file__).parent / name: digest
            for name, digest in inputs.get("diffusionEngineSources", {}).items()
        },
        diffusers_package / "pipelines/auto_pipeline.py": inputs["diffusersMappingSha256"],
        **{
            diffusers_package / name: digest
            for name, digest in inputs.get("diffusersSchedulerSources", {}).items()
        },
        **{
            diffusers_package / "pipelines" / name: digest
            for name, digest in inputs.get("diffusersPipelineSources", {}).items()
        },
        **{
            diffusers_package / "models" / name: digest
            for name, digest in inputs.get("diffusersModelSources", {}).items()
        },
        **{
            mlx_package / "models" / name: digest
            for name, digest in inputs["mlxModelSources"].items()
        },
        **{
            mlx_package.parent / "mlx_vlm" / name: digest
            for name, digest in inputs["mlxVlmSources"].items()
        },
    }
    for path, expected in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Packaged library bytes differ from capability build inputs: {path}")


def architectures_for(
    task: str, *, source: str = "transformers", family_adapter: str | None = None
) -> frozenset[str]:
    return frozenset(
        item["architecture"]
        for provider in generated_sources()
        for item in provider["capabilities"]
        if item["task"] == task
        and item["sourceKind"] == source
        and (family_adapter is None or item["familyAdapter"]["id"] == family_adapter)
    )
