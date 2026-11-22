"""Offline MLX-Audio task worker using the bundled library's native model loaders."""

from __future__ import annotations

import contextlib
import importlib
import inspect
import json
import math
import os
import pickle
import re
import sys
import wave
import zipfile
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

from oryn_engine.model_metadata import parse_metadata

PACK_ID = "oryn.mlx-audio"
MLX_AUDIO_VERSION = "0.6.6"
TASK_CONTRACTS = {
    "text-to-speech": "speech-synthesis-v1",
    "automatic-speech-recognition": "transcription-v1",
}
MAX_AUDIO_SECONDS = 30
MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESULT_BYTES = 900 * 1024
MAX_NUMERIC_BYTES = 128 * 1024 * 1024
ARTIFACT_ID = re.compile(r"artifact_[A-Za-z0-9_-]{20,80}")


def message(value):
    print(json.dumps(value, separators=(",", ":"), allow_nan=False), flush=True)


def event(name, data):
    # Framework stdout is redirected while inference runs. Protocol events retain
    # their own stream, including events produced between generator yields.
    sys.__stdout__.write(
        json.dumps({"type": "event", "event": name, "data": data}, separators=(",", ":")) + "\n"
    )
    sys.__stdout__.flush()


def input_value(task: str, value: object) -> dict:
    """Validate the closed public task input without importing native libraries."""
    if not isinstance(value, dict) or task not in TASK_CONTRACTS:
        raise ValueError("MODEL_INPUT_INVALID")
    language = value.get("language", "auto")
    if not isinstance(language, str) or not 1 <= len(language) <= 32:
        raise ValueError("MODEL_INPUT_INVALID")
    if task == "automatic-speech-recognition":
        artifact = value.get("audioArtifactId")
        if (
            set(value) - {"audioArtifactId", "language", "timestamps"}
            or not isinstance(artifact, str)
            or ARTIFACT_ID.fullmatch(artifact) is None
            or value.get("timestamps", False) is not False
        ):
            raise ValueError("MODEL_INPUT_INVALID")
        return {"audioArtifactId": artifact, "language": language, "timestamps": False}
    text = value.get("text")
    voice = value.get("voice", "default")
    speed = value.get("speed", 1)
    if (
        set(value) - {"text", "voice", "language", "speed", "audioArtifactId", "referenceText"}
        or not isinstance(text, str)
        or not text.strip()
        or len(text) > 350
        or not isinstance(voice, str)
        or not 1 <= len(voice) <= 64
        or type(speed) not in {int, float}
        or speed != 1
    ):
        raise ValueError("MODEL_INPUT_INVALID")
    result = {"text": text, "voice": voice, "language": language, "speed": 1.0}
    artifact = value.get("audioArtifactId")
    if artifact is not None:
        if (
            not isinstance(artifact, str)
            or ARTIFACT_ID.fullmatch(artifact) is None
            or voice != "reference"
        ):
            raise ValueError("MODEL_INPUT_INVALID")
        result["audioArtifactId"] = artifact
    elif voice == "reference":
        raise ValueError("MODEL_INPUT_INVALID")
    if "referenceText" in value:
        reference = value["referenceText"]
        if (
            artifact is None
            or not isinstance(reference, str)
            or not reference.strip()
            or len(reference) > 1000
        ):
            raise ValueError("MODEL_INPUT_INVALID")
        result["referenceText"] = reference
    return result


def verify_runtime() -> str:
    from oryn_engine.generated_capabilities import verify_audio_build

    installed = version("mlx-audio")
    if installed != MLX_AUDIO_VERSION:
        raise ValueError("RUNTIME_DEPENDENCY_UNAVAILABLE")
    package = importlib.import_module("mlx_audio")
    verify_audio_build(Path(package.__file__).parent)
    return installed


def output_value(task: str, value: object) -> dict:
    """Validate a structured transcription before the shared runner publishes it."""
    from oryn_protocol.task_contracts import make_profile, validate_result

    if task != "automatic-speech-recognition" or not isinstance(value, dict):
        raise ValueError("MODEL_GENERATION_FAILED")
    if len(json.dumps(value, allow_nan=False).encode()) > MAX_RESULT_BYTES:
        raise ValueError("MODEL_GENERATION_FAILED")
    validate_result(make_profile(TASK_CONTRACTS[task]), {"kind": "structured", "value": value})
    return value


def _numeric_arrays(root):
    """Preflight numeric voice/codec arrays with bounds before NumPy allocates."""
    import numpy as np

    total = 0
    arrays = 0

    def header(stream, size):
        nonlocal total, arrays
        format_version = np.lib.format.read_magic(stream)
        if format_version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif format_version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("MODEL_RUNTIME_INCOMPATIBLE")
        count = math.prod(shape) * dtype.itemsize
        arrays += 1
        total += count
        if (
            dtype.hasobject
            or dtype.kind not in "biufc"
            or len(shape) > 8
            or arrays > 2048
            or not 0 <= count <= size - stream.tell()
            or total > MAX_NUMERIC_BYTES
        ):
            raise ValueError("MODEL_RUNTIME_INCOMPATIBLE")

    for path in root.rglob("*"):
        if path.suffix not in {".npy", ".npz"}:
            continue
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_NUMERIC_BYTES:
            raise ValueError("MODEL_RUNTIME_INCOMPATIBLE")
        if path.suffix == ".npy":
            with path.open("rb") as stream:
                header(stream, path.stat().st_size)
        else:
            with zipfile.ZipFile(path) as archive:
                entries = archive.infolist()
                if len(entries) > 2048 or len({entry.filename for entry in entries}) != len(
                    entries
                ):
                    raise ValueError("MODEL_RUNTIME_INCOMPATIBLE")
                for entry in entries:
                    if (
                        entry.is_dir()
                        or not entry.filename.endswith(".npy")
                        or entry.file_size > MAX_NUMERIC_BYTES
                    ):
                        raise ValueError("MODEL_RUNTIME_INCOMPATIBLE")
                    with archive.open(entry) as stream:
                        header(stream, entry.file_size)
        loaded = np.load(path, allow_pickle=False)
        try:
            values = (
                (loaded[name] for name in loaded.files)
                if isinstance(loaded, np.lib.npyio.NpzFile)
                else (loaded,)
            )
            if any(not np.isfinite(value).all() for value in values):
                raise ValueError("MODEL_RUNTIME_INCOMPATIBLE")
        finally:
            if isinstance(loaded, np.lib.npyio.NpzFile):
                loaded.close()


def _local_pretrained_path(value, root):
    if not isinstance(value, (str, Path)):
        raise ValueError("MODEL_RUNTIME_INCOMPATIBLE")
    path = Path(value)
    if (
        not path.is_absolute()
        or not path.exists()
        or path.is_symlink()
        or not path.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError("MODEL_RUNTIME_INCOMPATIBLE")
    return str(path)


def _preflight(root, config, task):
    from oryn_engine import mlx_audio_tasks
    from oryn_engine.model_metadata import (
        metadata_file,
        preflight_safetensors,
        validate_model_config,
    )

    paths = list(root.rglob("*"))
    if len(paths) > 20_000 or any(path.is_symlink() for path in paths):
        raise ValueError("MODEL_RUNTIME_INCOMPATIBLE")
    files = tuple(str(path.relative_to(root)) for path in paths if path.is_file())
    if set(mlx_audio_tasks.selected_files(files, config, task)) != set(files):
        raise ValueError("MODEL_RUNTIME_INCOMPATIBLE")
    codec_path = root / mlx_audio_tasks.QWEN3_TTS_CODEC_CONFIG
    qwen3_tts = mlx_audio_tasks.model_family(config, task) == "qwen3_tts"
    for path in paths:
        if path.is_file() and (path.name == "config.json" or path.name.endswith("_config.json")):
            # Native tokenizer/processor post-load hooks receive local paths, but
            # their constructor metadata also cannot redirect file authority.
            # The potentially large tokenizer.json remains native Rust data.
            metadata = metadata_file(path)
            if qwen3_tts and path == codec_path:
                mlx_audio_tasks.validate_qwen3_tts_codec(metadata)
            else:
                validate_model_config(metadata)
    weights = [path for path in paths if path.suffix == ".safetensors" and path.is_file()]
    if sum(path.stat().st_size for path in weights) > 128 * 1024**3:
        raise ValueError("MODEL_RUNTIME_INCOMPATIBLE")
    components = {}
    for directory in {path.parent for path in weights}:
        names = {path.name for path in weights if path.parent == directory}
        indexes = list(directory.glob("*.safetensors.index.json"))
        companions = frozenset()
        if len(indexes) == 1:
            with indexes[0].open("rb") as stream:
                index = parse_metadata(stream.read(8 * 1024**2 + 1), limit=8 * 1024**2)
            mapping = index.get("weight_map")
            if not isinstance(mapping, dict) or any(
                not isinstance(v, str) for v in mapping.values()
            ):
                raise ValueError("MODEL_RUNTIME_INCOMPATIBLE")
            companions = frozenset(names - set(mapping.values()))
        components[str(directory.relative_to(root))] = preflight_safetensors(
            directory, companion_files=companions
        )
    _numeric_arrays(root)
    return components


def _soprano_decoder_config(config, tensors):
    """Recover the native decoder graph from tensors, independently of a path name."""
    normalized = {}
    for name, shape in tensors.items():
        name = name.removeprefix("model.")
        if name in normalized:
            raise ValueError("MODEL_RUNTIME_INCOMPATIBLE")
        normalized[name] = shape
    embed = normalized.get("decoder.decoder.embed.weight", ())
    projection = normalized.get("decoder.decoder.convnext.0.pwconv1.weight", ())
    depthwise = normalized.get("decoder.decoder.convnext.0.dwconv.weight", ())
    head = normalized.get("decoder.head.out.weight", ())
    indices = {
        int(match[1])
        for name in normalized
        if (match := re.fullmatch(r"decoder\.decoder\.convnext\.(\d+)\.pwconv1\.weight", name))
    }
    if (
        len(embed) != 3
        or len(projection) != 2
        or len(depthwise) != 3
        or len(head) != 2
        or not 1 <= len(indices) <= 32
        or indices != set(range(len(indices)))
    ):
        raise ValueError("MODEL_RUNTIME_INCOMPATIBLE")
    dim, kernel, hidden = embed
    intermediate, projected_dim = projection
    fft = head[0] - 2
    if (
        hidden != config.get("hidden_size")
        or not 8 <= dim <= 4096
        or not dim <= intermediate <= 16384
        or projected_dim != dim
        or head[1] != dim
        or depthwise[0] != dim
        or depthwise[2] != 1
        or kernel not in {1, 3, 5, 7}
        or depthwise[1] not in {1, 3, 5, 7}
        or fft != 2048
    ):
        raise ValueError("MODEL_RUNTIME_INCOMPATIBLE")
    for index in indices:
        prefix = f"decoder.decoder.convnext.{index}"
        if (
            normalized.get(f"{prefix}.pwconv1.weight") != (intermediate, dim)
            or normalized.get(f"{prefix}.pwconv2.weight") != (dim, intermediate)
            or normalized.get(f"{prefix}.dwconv.weight") != depthwise
        ):
            raise ValueError("MODEL_RUNTIME_INCOMPATIBLE")
    declared = config.get("decoder_config")
    if declared is not None and (
        not isinstance(declared, dict)
        or any(
            declared.get(key, default) != default
            for key, default in {
                "hop_length": 512,
                "upscale": 4,
                "token_size": 2048,
                "receptive_field": 4,
            }.items()
        )
    ):
        raise ValueError("MODEL_RUNTIME_INCOMPATIBLE")
    inferred = {
        "decoder_num_layers": len(indices),
        "decoder_dim": dim,
        "decoder_intermediate_dim": intermediate,
        "input_kernel": kernel,
        "dw_kernel": depthwise[1],
        "n_fft": fft,
    }
    if isinstance(declared, dict) and any(
        declared.get(key, expected) != expected for key, expected in inferred.items()
    ):
        raise ValueError("MODEL_RUNTIME_INCOMPATIBLE")
    return inferred


@contextlib.contextmanager
def _native_model_config(capability, config, components):
    if capability["family"] != "soprano":
        yield
        return
    # Pinned Soprano selects its 1.1 acoustic dimensions from a directory-name
    # substring. Installed selected-file views have opaque names. Keep its native
    # architecture/loader and bind that constructor to the actual tensor graph.
    decoder = _soprano_decoder_config(config, components.get(".", {}))
    module = importlib.import_module("mlx_audio.tts.models.soprano")
    original = module.ModelConfig.from_dict.__func__

    def from_dict(cls, values):
        model_config = original(cls, values | {"decoder_config": None})
        model_config.decoder_config = module.DecoderConfig(**decoder)
        return model_config

    with patch.object(module.ModelConfig, "from_dict", classmethod(from_dict)):
        yield


@contextlib.contextmanager
def _local_loaders(root):
    """Keep bundled post-load hooks inside the installed data-only checkpoint."""
    import huggingface_hub
    import numpy as np
    import torch
    from transformers.configuration_utils import PretrainedConfig
    from transformers.feature_extraction_utils import FeatureExtractionMixin
    from transformers.models.auto.auto_factory import _BaseAutoModelClass
    from transformers.models.auto.configuration_auto import AutoConfig
    from transformers.models.auto.feature_extraction_auto import AutoFeatureExtractor
    from transformers.models.auto.processing_auto import AutoProcessor
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    from transformers.processing_utils import ProcessorMixin
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    def unavailable(*args, **kwargs):
        raise ValueError("MODEL_RUNTIME_INCOMPATIBLE")

    numeric_load = np.load

    def numeric_only(*args, **kwargs):
        kwargs["allow_pickle"] = False
        return numeric_load(*args, **kwargs)

    def local_pretrained(original):
        def load(cls, pretrained_model_name_or_path, *args, **kwargs):
            local_path = _local_pretrained_path(pretrained_model_name_or_path, root)
            kwargs.update(trust_remote_code=False, local_files_only=True)
            return original(cls, local_path, *args, **kwargs)

        return classmethod(load)

    with contextlib.ExitStack() as stack:
        for owner in (
            AutoTokenizer,
            AutoProcessor,
            AutoFeatureExtractor,
            AutoConfig,
            _BaseAutoModelClass,
            PreTrainedTokenizerBase,
            ProcessorMixin,
            FeatureExtractionMixin,
            PretrainedConfig,
        ):
            method = owner.__dict__.get("from_pretrained")
            if not isinstance(method, classmethod):
                raise ValueError("MODEL_RUNTIME_INCOMPATIBLE")
            stack.enter_context(
                patch.object(owner, "from_pretrained", local_pretrained(method.__func__))
            )
        # No model-local archive is unpickled and no hook resolves an external
        # tokenizer/codec from a machine-wide cache. Missing companion data fails.
        stack.enter_context(patch.object(torch, "load", unavailable))
        stack.enter_context(patch.object(pickle, "load", unavailable))
        stack.enter_context(patch.object(np, "load", numeric_only))
        stack.enter_context(patch.object(huggingface_hub, "hf_hub_download", unavailable))
        stack.enter_context(patch.object(huggingface_hub, "snapshot_download", unavailable))
        yield


def _paths(payload):
    if not isinstance(payload, dict) or set(payload) != {
        "modelRoot",
        "task",
        "backend",
        "input",
        "inputPath",
        "outputPath",
    }:
        raise ValueError("MODEL_INPUT_INVALID")
    if payload["backend"] != "mlx" or payload["task"] not in TASK_CONTRACTS:
        raise ValueError("MODEL_RUNTIME_INCOMPATIBLE")
    root = Path(payload["modelRoot"])
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("MODEL_INPUT_INVALID")
    paths = []
    for key in ("inputPath", "outputPath"):
        value = payload[key]
        if value is not None and not isinstance(value, str):
            raise ValueError("MODEL_INPUT_INVALID")
        path = Path(value) if value is not None else None
        if path is not None and (not path.is_absolute() or path.is_symlink()):
            raise ValueError("MODEL_INPUT_INVALID")
        paths.append(path)
    return root, *paths


def _load(root: Path, task: str):
    from oryn_engine import mlx_audio_tasks

    config_path = root / "config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("MODEL_RUNTIME_INCOMPATIBLE")
    with config_path.open("rb") as config_file:
        config = parse_metadata(config_file.read(MAX_REQUEST_BYTES + 1))
    config = mlx_audio_tasks.validate_config(config, task)
    components = _preflight(root, config, task)
    capability = mlx_audio_tasks.capability(config, task)
    if capability is None or capability["category"] not in {"tts", "stt"}:
        raise ValueError("MODEL_RUNTIME_INCOMPATIBLE")
    # Only a reviewed family from the signed build table reaches the bundled
    # loader. Local directory names and repository code never select imports.
    utils = importlib.import_module(f"mlx_audio.{capability['category']}.utils")
    with _native_model_config(capability, config, components):
        model = utils.load_model(
            root,
            model_type=capability["family"],
            model_name_parts=[],
            strict=True,
            trust_remote_code=False,
            local_files_only=True,
        )
    return model, capability


def _parameters(model, capability):
    parameters = inspect.signature(model.generate).parameters
    # **kwargs is deliberately excluded: several upstream models silently ignore
    # controls accepted by the library's generic CLI.
    return set(capability["parameters"]) & {
        name
        for name, parameter in parameters.items()
        if parameter.kind in {parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY}
    }


def _tts_kwargs(model, capability, value, input_path):
    parameters = _parameters(model, capability)
    if "text" not in parameters:
        raise ValueError("MODEL_RUNTIME_INCOMPATIBLE")
    kwargs = {"text": value["text"]}
    if value["voice"] not in {"default", "reference"}:
        if "voice" not in parameters:
            raise ValueError("MODEL_INPUT_INVALID")
        kwargs["voice"] = value["voice"]
    if value["language"] != "auto":
        language_key = next((key for key in ("lang_code", "language") if key in parameters), None)
        if language_key is None:
            raise ValueError("MODEL_INPUT_INVALID")
        kwargs[language_key] = value["language"]
    if input_path is not None:
        if "ref_audio" not in parameters:
            raise ValueError("MODEL_INPUT_INVALID")
        kwargs["ref_audio"] = str(input_path)
    if "referenceText" in value:
        if "ref_text" not in parameters:
            raise ValueError("MODEL_INPUT_INVALID")
        kwargs["ref_text"] = value["referenceText"]
    if "stream" in parameters:
        kwargs["stream"] = False
    if "verbose" in parameters:
        kwargs["verbose"] = False
    return kwargs


def _speech(model, capability, value, input_path, output_path):
    import numpy as np

    if (
        output_path is None
        or output_path.name != "result.wav"
        or output_path.exists()
        or not output_path.parent.is_dir()
    ):
        raise ValueError("MODEL_INPUT_INVALID")
    kwargs = _tts_kwargs(model, capability, value, input_path)
    count, rate = 0, None
    try:
        with output_path.open("xb") as target, wave.open(target, "wb") as audio_file:
            # A provisional empty header also keeps close() from obscuring an
            # invalid first chunk or an empty generator with a wave.Error.
            audio_file.setnchannels(1)
            audio_file.setsampwidth(2)
            audio_file.setframerate(8000)
            for chunk in model.generate(**kwargs):
                samples = np.asarray(chunk.audio)
                chunk_rate = chunk.sample_rate
                if type(chunk_rate) is not int or not 8000 <= chunk_rate <= 192000:
                    raise ValueError("MODEL_GENERATION_FAILED")
                if samples.ndim == 2 and 1 in samples.shape:
                    samples = samples.reshape(-1)
                if (
                    samples.ndim != 1
                    or samples.dtype.kind != "f"
                    or not np.isfinite(samples).all()
                    or (rate is not None and chunk_rate != rate)
                    or count + len(samples) > MAX_AUDIO_SECONDS * chunk_rate
                ):
                    raise ValueError("MODEL_GENERATION_FAILED")
                if rate is None:
                    rate = chunk_rate
                    audio_file.setnchannels(1)
                    audio_file.setsampwidth(2)
                    audio_file.setframerate(rate)
                pcm = np.rint(np.clip(samples, -1, 1) * 32767).astype("<i2")
                audio_file.writeframes(pcm.tobytes())
                count += len(samples)
                event("job.progress", {"stage": "synthesizing", "progress": 0.8})
            if not count:
                raise ValueError("MODEL_GENERATION_FAILED")
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    return {
        "relativeOutput": "result.wav",
        "mediaType": "audio/wav",
        "sampleRate": rate,
        "channels": 1,
        "frames": count,
        "durationSeconds": count / rate,
        "backend": "mlx",
    }


def _transcription(model, capability, value, input_path):
    parameters = _parameters(model, capability)
    kwargs = {}
    if value["language"] != "auto":
        if "language" in parameters:
            kwargs["language"] = value["language"]
        elif {"source_lang", "target_lang"} <= parameters:
            kwargs.update(source_lang=value["language"], target_lang=value["language"])
        else:
            raise ValueError("MODEL_INPUT_INVALID")
    for name in ("stream", "verbose", "return_timestamps", "word_timestamps"):
        if name in parameters:
            kwargs[name] = False
    result = model.generate(str(input_path), **kwargs)
    text = result.get("text") if isinstance(result, dict) else getattr(result, "text", None)
    language = (
        result.get("language") if isinstance(result, dict) else getattr(result, "language", None)
    )
    if (
        not isinstance(text, str)
        or len(text) > MAX_RESULT_BYTES
        or language is not None
        and (not isinstance(language, str) or len(language) > 32)
    ):
        raise ValueError("MODEL_GENERATION_FAILED")
    result = {"text": text, "language": language}
    if len(json.dumps(result, allow_nan=False).encode()) > MAX_RESULT_BYTES:
        raise ValueError("MODEL_GENERATION_FAILED")
    return output_value("automatic-speech-recognition", result)


def generate(payload):
    root, input_path, output_path = _paths(payload)
    task = payload["task"]
    value = input_value(task, payload["input"])
    needs_input = "audioArtifactId" in value
    if needs_input != (input_path is not None) or (
        task == "automatic-speech-recognition" and output_path is not None
    ):
        raise ValueError("MODEL_INPUT_INVALID")
    if input_path is not None:
        # Validate the authorized WAV and duration before a library decoder sees it.
        from oryn_engine.vad_tasks import read_audio

        _, duration = read_audio(input_path)
        if task == "text-to-speech" and duration > 10:
            raise ValueError("MODEL_INPUT_INVALID")
    event("job.progress", {"stage": "loading-model", "progress": 0.1})
    with contextlib.redirect_stdout(sys.stderr), _local_loaders(root):
        model, capability = _load(root, task)
        event("job.state", {"state": "running"})
        event("job.progress", {"stage": "running-model", "progress": 0.3})
        if task == "text-to-speech":
            return _speech(model, capability, value, input_path, output_path)
        return _transcription(model, capability, value, input_path)


def main():
    # The launcher also denies networking. Offline flags prevent hidden library
    # fetch attempts and make missing local tokenizer/codec data fail visibly.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    try:
        from oryn_engine.runtime_packs.builtin_catalog import builtin_registry

        request = parse_metadata(sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1))
        pack = builtin_registry().active_pack(PACK_ID)
        if (
            pack is None
            or set(request) != {"abi", "packId", "version", "command", "payload"}
            or (request["abi"], request["packId"], request["version"])
            != (pack.abi, pack.pack_id, pack.version)
            or not isinstance(request["payload"], dict)
        ):
            raise ValueError("MODEL_INPUT_INVALID")
        installed = verify_runtime()
        if request["command"] == "probe":
            # Import the two public backends before reporting availability; a
            # version marker alone cannot prove that the dependency closure loads.
            with contextlib.redirect_stdout(sys.stderr):
                importlib.import_module("mlx_audio.tts.utils")
                importlib.import_module("mlx_audio.stt.utils")
            if set(TASK_CONTRACTS.values()) != set(
                pack.runner("mlx-audio.shared-tasks").task_contracts
            ):
                raise ValueError("MODEL_RUNTIME_INCOMPATIBLE")
            result = {
                "mlx-audio": installed,
                "taskContracts": sorted(TASK_CONTRACTS.values()),
                "backend": "mlx",
                "available": True,
            }
        elif request["command"] == "generate":
            result = generate(request["payload"])
        elif request["command"] == "prepare":
            root, _, _ = _paths(request["payload"])
            with contextlib.redirect_stdout(sys.stderr), _local_loaders(root):
                _load(root, request["payload"]["task"])
            result = {"prepared": True}
        else:
            raise ValueError("MODEL_INPUT_INVALID")
        message({"type": "result", "ok": True, "result": result})
    except Exception as exc:
        code = str(exc)
        if code not in {
            "MODEL_INPUT_INVALID",
            "MODEL_RUNTIME_INCOMPATIBLE",
            "MODEL_GENERATION_FAILED",
            "RUNTIME_DEPENDENCY_UNAVAILABLE",
        }:
            code = "MODEL_GENERATION_FAILED"
        print(f"MLX-Audio worker failed: {type(exc).__name__}: {code}", file=sys.stderr)
        message({"type": "result", "ok": False, "error": {"code": code}})


if __name__ == "__main__":
    main()
