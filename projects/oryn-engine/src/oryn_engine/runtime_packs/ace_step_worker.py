from __future__ import annotations

import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

MAX_REQUEST_BYTES = 1024 * 1024
MAX_PROMPT_CHARS = 2_000
MAX_LYRICS_CHARS = 12_000


def _send(value: dict[str, Any]) -> None:
    sys.__stdout__.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.__stdout__.flush()


def _event(event: str, data: dict[str, Any]) -> None:
    _send({"type": "event", "event": event, "data": data})


def _request() -> dict[str, Any]:
    line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
    if not line or len(line) > MAX_REQUEST_BYTES or sys.stdin.buffer.read(1):
        raise ValueError("invalid request")
    value = json.loads(line)
    if not isinstance(value, dict) or set(value) != {
        "abi",
        "packId",
        "version",
        "command",
        "payload",
    }:
        raise ValueError("invalid request")
    if value["abi"] != "1" or value["packId"] != "oryn.ace-step-mlx":
        raise ValueError("invalid runtime identity")
    if not isinstance(value["payload"], dict):
        raise ValueError("invalid payload")
    return value


def _validated_payload(payload: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    if set(payload) != {"modelRoot", "outputPath", "input"}:
        raise ValueError("invalid payload")
    model_root = Path(payload["modelRoot"])
    output_path = Path(payload["outputPath"])
    input_value = payload["input"]
    if (
        not model_root.is_absolute()
        or model_root.is_symlink()
        or not model_root.is_dir()
        or not output_path.is_absolute()
        or output_path.is_symlink()
        or output_path.exists()
        or not output_path.parent.is_dir()
        or not isinstance(input_value, dict)
        or set(input_value) - {"prompt", "lyrics", "durationSeconds", "seed"}
    ):
        raise ValueError("invalid payload")
    prompt = input_value.get("prompt")
    lyrics = input_value.get("lyrics", "[Instrumental]")
    duration = input_value.get("durationSeconds", 10)
    seed = input_value.get("seed", -1)
    if (
        not isinstance(prompt, str)
        or not 1 <= len(prompt.strip()) <= MAX_PROMPT_CHARS
        or not isinstance(lyrics, str)
        or len(lyrics) > MAX_LYRICS_CHARS
        or not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not math.isfinite(float(duration))
        or not 10 <= float(duration) <= 180
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or not -1 <= seed <= 2**32 - 1
    ):
        raise ValueError("invalid music input")
    required = (
        "acestep-v15-turbo/config.json",
        "acestep-v15-turbo/model.safetensors",
        "acestep-v15-turbo/silence_latent.pt",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "Qwen3-Embedding-0.6B/config.json",
        "Qwen3-Embedding-0.6B/model.safetensors",
        "Qwen3-Embedding-0.6B/tokenizer.json",
    )
    if any(not (model_root / name).is_file() for name in required):
        raise ValueError("model subset is incomplete")
    return (
        model_root.resolve(),
        output_path.resolve(strict=False),
        {
            "prompt": prompt.strip(),
            "lyrics": lyrics,
            "durationSeconds": float(duration),
            "seed": seed,
        },
    )


def _component_config(directory: Path) -> dict[str, Any]:
    from oryn_engine.model_metadata import metadata_file, validate_model_config

    config = metadata_file(directory / "config.json")
    if directory.name == "acestep-v15-turbo" and config.get("auto_map") == {
        "AutoConfig": "configuration_acestep_v15.AceStepConfig",
        "AutoModel": "modeling_acestep_v15_turbo.AceStepConditionGenerationModel",
    }:
        # Legacy metadata names the classes already sealed into this signed adapter.
        # Remove this inert mapping before constructing that explicit native class.
        config = {key: value for key, value in config.items() if key != "auto_map"}
    validate_model_config(config)
    return config


def _generate(payload: dict[str, Any]) -> dict[str, Any]:
    model_root, output_path, input_value = _validated_payload(payload)
    from oryn_engine.model_metadata import preflight_safetensors

    for component in ("acestep-v15-turbo", "vae", "Qwen3-Embedding-0.6B"):
        directory = model_root / component
        _component_config(directory)
        preflight_safetensors(directory)
    _event("job.state", {"state": "loading"})
    _event("job.progress", {"stage": "loading-runtime", "progress": 0.01})

    # Upstream logs and dependency notices are never part of the pack protocol stream.
    sys.stdout = sys.stderr
    os.environ["ACESTEP_DISABLE_TQDM"] = "1"
    os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(model_root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.models.turbo.configuration_acestep_v15 import AceStepConfig
    from acestep.models.turbo.modeling_acestep_v15_turbo import (
        AceStepConditionGenerationModel,
    )

    class OrynAceStepHandler(AceStepHandler):
        def _ensure_models_present(self, **_kwargs: Any) -> None:
            # Downloads are Browser-owned and complete before this process starts.
            return None

        @staticmethod
        def _sync_model_code_if_needed(_config_path: str, _checkpoint_path: Path) -> None:
            # Repository Python is never copied or imported.
            return None

        def _load_main_model_from_checkpoint(
            self,
            *,
            model_checkpoint_path: str,
            device: str,
            use_flash_attention: bool,
            compile_model: bool,
            quantization: str | None,
        ) -> str:
            del use_flash_attention, compile_model, quantization
            config = AceStepConfig.from_dict(_component_config(Path(model_checkpoint_path)))
            self.model = AceStepConditionGenerationModel.from_pretrained(
                model_checkpoint_path,
                config=config,
                local_files_only=True,
                dtype=torch.float32,
            )
            self.config = self.model.config
            self._sync_alignment_config()
            self.model = self.model.to("cpu").to(torch.float32).eval()
            silence_path = Path(model_checkpoint_path) / "silence_latent.pt"
            self.silence_latent = torch.load(
                silence_path, map_location="cpu", weights_only=True
            ).transpose(1, 2)
            self.silence_latent = self.silence_latent.to("cpu").to(torch.float32)
            self.offload_to_cpu = True
            self.offload_dit_to_cpu = True
            return "eager"

    handler = OrynAceStepHandler()
    status, ok = handler.initialize_service(
        project_root=str(model_root.parent),
        config_path="acestep-v15-turbo",
        device="cpu",
        use_flash_attention=False,
        # torch.compile wraps VAE parameters in an `_orig_mod` namespace before
        # ACE-Step converts them to MLX. The native MLX loader expects the
        # original checkpoint names, so compilation must stay disabled here.
        compile_model=False,
        offload_to_cpu=True,
        offload_dit_to_cpu=True,
        use_mlx_dit=True,
    )
    if not ok or not handler.use_mlx_dit or not handler.use_mlx_vae:
        raise RuntimeError(f"ACE-Step MLX initialization failed: {status}")

    _event("job.state", {"state": "running"})

    def progress(ratio: float, desc: str = "Generating music") -> None:
        bounded = max(0.05, min(0.94, float(ratio)))
        _event("job.progress", {"stage": desc[:128], "progress": bounded})

    result = generate_music(
        handler,
        None,
        params=GenerationParams(
            caption=input_value["prompt"],
            lyrics=input_value["lyrics"],
            instrumental=input_value["lyrics"].strip() in {"", "[Instrumental]"},
            duration=input_value["durationSeconds"],
            seed=input_value["seed"],
            thinking=False,
            use_cot_metas=False,
            use_cot_caption=False,
            use_cot_language=False,
            inference_steps=8,
            guidance_scale=1.0,
            dcw_enabled=True,
        ),
        config=GenerationConfig(
            batch_size=1,
            use_random_seed=input_value["seed"] < 0,
            seeds=None if input_value["seed"] < 0 else [input_value["seed"]],
            audio_format="wav32",
        ),
        save_dir=str(output_path.parent),
        progress=progress,
    )
    if not result.success or not result.audios:
        raise RuntimeError(result.error or result.status_message or "generation failed")
    source = Path(result.audios[0].get("path", ""))
    if not source.is_file() or source.stat().st_size > 1024 * 1024 * 1024:
        raise RuntimeError("ACE-Step returned an invalid audio file")
    _event("job.state", {"state": "saving"})
    shutil.copyfile(source, output_path)
    return {
        "relativeOutput": output_path.name,
        "mediaType": "audio/wav",
        "sampleRate": int(handler.sample_rate),
        "backend": "mlx",
    }


def main() -> int:
    try:
        request = _request()
        command = request["command"]
        if command == "probe":
            result = {
                "packId": "oryn.ace-step-mlx",
                "abi": "1",
                "networkPolicy": os.environ.get("ORYN_PACK_NETWORK_POLICY"),
            }
        elif command == "generate":
            result = _generate(request["payload"])
        else:
            raise ValueError("unsupported command")
        _send({"type": "result", "ok": True, "result": result})
        return 0
    except Exception:
        _send({"type": "result", "ok": False, "error": {"code": "PACK_EXECUTION_FAILED"}})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
