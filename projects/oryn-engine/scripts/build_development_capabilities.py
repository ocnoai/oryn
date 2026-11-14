"""Refresh local development signatures. Production releases use protected release signing."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx-package", type=Path, required=True)
    parser.add_argument("--diffusers-package", type=Path, required=True)
    parser.add_argument("--llama-binary", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "src/oryn_engine/data"
    key_id = "oryn-sprint47-development-2026"
    with tempfile.TemporaryDirectory(prefix="oryn-build-signing-") as temporary:
        key = Path(temporary) / "development.pem"
        subprocess.run(
            ["openssl", "genrsa", "-out", str(key), "3072"], check=True, capture_output=True
        )
        from oryn_engine.runtime_packs.builtin_foundations import FOUNDATION_ENVELOPES

        foundation = json.loads(
            json.dumps(
                next(
                    item
                    for item in FOUNDATION_ENVELOPES
                    if item["payload"]["foundationId"] == "oryn.runtime.pytorch"
                )
            )
        )
        foundation["payload"].update(
            version="1.2.0", dependencyClosure="oryn-engine-py312-torch2.15-ct2-4.8.4"
        )
        foundation["payload"]["formats"] = sorted(
            set(foundation["payload"]["formats"]) | {"ctranslate2", "torch-numeric-zip"}
        )
        (output / "pytorch-foundation.json").write_text(json.dumps(foundation, indent=2) + "\n")
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts/build_capabilities.py"),
                "--mlx-package",
                str(args.mlx_package),
                "--diffusers-package",
                str(args.diffusers_package),
                "--llama-binary",
                str(args.llama_binary),
                "--signing-key",
                str(key),
                "--output",
                str(output),
            ],
            check=True,
        )
        from qwen_tts_catalog import write_manifests

        write_manifests(root, output)
        for name in (
            "pytorch-foundation.json",
            "mlx-foundation.json",
            "mlx-pack.json",
            "mlx-adapter.json",
            "transformers-pack.json",
            "transformers-adapter.json",
            "onnx-pack.json",
            "onnx-adapter.json",
            "llama-pack.json",
            "llama-adapter.json",
            "diffusers-pack.json",
            "diffusers-adapter.json",
            "mflux-pack.json",
            "mflux-adapter.json",
            "kokoro-pack.json",
            "kokoro-adapter.json",
            "qwen-tts-pack.json",
            "qwen-tts-foundation.json",
            "qwen-tts-adapter.json",
            "pack-catalog.json",
        ):
            path = output / name
            if not path.exists():
                continue
            payload = json.loads(path.read_text())["payload"]
            if name in {"onnx-pack.json", "onnx-adapter.json"}:
                from oryn_engine.onnx_tasks import TASK_CONTRACTS as ONNX_TASKS

                if name == "onnx-pack.json":
                    specification = json.loads(
                        (root.parent / "oryn-browser/packaging/runtime-spec.json").read_text()
                    )["runtimePacks"]["oryn.onnx-runtime"]
                    payload.update(
                        version=specification["version"],
                        dependencyClosure=specification["dependencyClosure"],
                    )
                    next(r for r in payload["runners"] if r["runnerId"] == "onnx.encoder")[
                        "taskContracts"
                    ] = sorted(ONNX_TASKS.values())
                else:
                    payload["version"] = "1.3.0"
                    from oryn_engine.onnx_classifier import ARCHITECTURE as CLASSIFIER_ARCHITECTURE
                    from oryn_engine.onnx_vision import ARCHITECTURE as VISION_ARCHITECTURE

                    payload["taskContracts"] = sorted(ONNX_TASKS.values())
                    payload["architectures"] = sorted(
                        set(payload["architectures"])
                        | {VISION_ARCHITECTURE, CLASSIFIER_ARCHITECTURE, "nemo-conformer-tdt"}
                    )
                    # Individual bindings validate their config/tokenizer/label
                    # dependencies. The shared audio/vision pack needs no tokenizer.
                    payload["requiredFiles"] = ["*.onnx"]
            if name in {"kokoro-pack.json", "mlx-foundation.json"}:
                specification = json.loads(
                    (root.parent / "oryn-browser/packaging/runtime-spec.json").read_text()
                )
                specification = (
                    specification["runtimePacks"]["oryn.kokoro-onnx"]
                    if name == "kokoro-pack.json"
                    else specification["runtimeFoundations"]["oryn.runtime.mlx"]
                )
                payload.update(
                    version=specification["version"],
                    dependencyClosure=specification["dependencyClosure"],
                )
            if name == "llama-pack.json":
                specification = json.loads(
                    (root.parent / "oryn-browser/packaging/runtime-spec.json").read_text()
                )["runtimePacks"]["oryn.llama-cpp"]
                payload.update(
                    version=specification["version"],
                    dependencyClosure=specification["dependencyClosure"],
                )
                payload["runners"] = [
                    r for r in payload["runners"] if r["runnerId"] != "llama.cpp.translation"
                ] + [
                    {
                        "runnerId": "llama.cpp.translation",
                        "taskContracts": ["translation-v1"],
                        "backends": ["metal", "cuda", "cpu"],
                    }
                ]
            if name == "llama-adapter.json":
                payload["version"] = "1.2.0"
                payload["taskContracts"] = sorted(
                    set(payload["taskContracts"]) | {"translation-v1"}
                )
            if name == "mlx-pack.json":
                specification = json.loads(
                    (root.parent / "oryn-browser/packaging/runtime-spec.json").read_text()
                )["runtimePacks"]["oryn.mlx-lm"]
                payload.update(
                    version=specification["version"],
                    dependencyClosure=specification["dependencyClosure"],
                )
                payload["runners"] = [
                    r
                    for r in payload["runners"]
                    if r["runnerId"] not in {"mlx.vad", "mlx.multimodal"}
                ] + [
                    {
                        "runnerId": "mlx.multimodal",
                        "taskContracts": ["multimodal-generation-v1"],
                        "backends": ["mlx"],
                    },
                    {
                        "runnerId": "mlx.vad",
                        "taskContracts": ["voice-activity-detection-v1"],
                        "backends": ["mlx"],
                    },
                ]
            if name == "mlx-adapter.json":
                payload["version"] = "1.4.0"
                payload["taskContracts"] = sorted(
                    set(payload["taskContracts"])
                    | {"voice-activity-detection-v1", "multimodal-generation-v1"}
                )
                # Per-task capability formats and native layout validation enforce
                # tokenizer requirements for chat. Audio has no tokenizer.
                payload["requiredFiles"] = ["config.json", "*.safetensors"]
            if name in {"transformers-pack.json", "transformers-adapter.json"}:
                from oryn_engine.shared_tasks import TASK_CONTRACTS

                if name == "transformers-pack.json":
                    specification = json.loads(
                        (root.parent / "oryn-browser/packaging/runtime-spec.json").read_text()
                    )["runtimePacks"]["oryn.transformers-pytorch"]
                    payload.update(
                        version=specification["version"],
                        dependencyClosure=specification["dependencyClosure"],
                    )
                    next(
                        r
                        for r in payload["runners"]
                        if r["runnerId"] == "transformers.shared-tasks"
                    )["taskContracts"] = sorted(TASK_CONTRACTS.values())
                else:
                    payload["version"] = "1.24.0"
                    payload["foundation"]["minimumVersion"] = "1.2.0"
                    payload["formats"] = sorted(
                        set(payload["formats"]) | {"ctranslate2", "torch-numeric-zip"}
                    )
                    payload["taskContracts"] = sorted(
                        set(payload["taskContracts"]) | set(TASK_CONTRACTS.values())
                    )
            if name in {"diffusers-pack.json", "diffusers-adapter.json"}:
                if name == "diffusers-pack.json":
                    specification = json.loads(
                        (root.parent / "oryn-browser/packaging/runtime-spec.json").read_text()
                    )["runtimePacks"]["oryn.diffusers-pytorch"]
                    payload.update(
                        version=specification["version"],
                        dependencyClosure=specification["dependencyClosure"],
                    )
                    contracts = next(
                        r for r in payload["runners"] if r["runnerId"] == "diffusers.pipeline"
                    )["taskContracts"]
                else:
                    payload["version"] = "1.7.0"
                    contracts = payload["taskContracts"]
                if "unconditional-image-generation-v1" not in contracts:
                    contracts.append("unconditional-image-generation-v1")
                if "image-text-to-video-v1" not in contracts:
                    contracts.append("image-text-to-video-v1")
                if "text-to-music-v1" not in contracts:
                    contracts.append("text-to-music-v1")
            if name == "mflux-pack.json":
                specification = json.loads(
                    (root.parent / "oryn-browser/packaging/runtime-spec.json").read_text()
                )["runtimePacks"]["oryn.mflux-mlx"]
                payload.update(
                    version=specification["version"],
                    dependencyClosure=specification["dependencyClosure"],
                )
                catalog = (
                    Path(__file__).resolve().parents[1]
                    / "providers/mflux/src/oryn_mflux_provider/catalog.json"
                )
                payload["provider"]["catalogSha256"] = hashlib.sha256(
                    catalog.read_bytes()
                ).hexdigest()
            if name == "pack-catalog.json":
                payload["catalogVersion"] = 7
                for item in payload["packs"]:
                    if item["packId"] in {"oryn.llama-cpp", "oryn.kokoro-onnx"}:
                        replacement = json.loads(
                            (
                                output
                                / (
                                    "llama-pack.json"
                                    if item["packId"] == "oryn.llama-cpp"
                                    else "kokoro-pack.json"
                                )
                            ).read_text()
                        )
                        item.update(
                            version=replacement["payload"]["version"],
                            manifestDigest=replacement["integrity"]["digest"],
                            minimumSafeVersion=replacement["payload"]["version"],
                            dependencyClosure=replacement["payload"]["dependencyClosure"],
                        )
                    if item["packId"] == "oryn.onnx-runtime":
                        replacement = json.loads((output / "onnx-pack.json").read_text())
                        item.update(
                            version=replacement["payload"]["version"],
                            manifestDigest=replacement["integrity"]["digest"],
                            minimumSafeVersion=replacement["payload"]["version"],
                            dependencyClosure=replacement["payload"]["dependencyClosure"],
                        )
                    if item["packId"] == "oryn.qwen-tts-pytorch":
                        replacement = json.loads((output / "qwen-tts-pack.json").read_text())
                        item.update(
                            version=replacement["payload"]["version"],
                            manifestDigest=replacement["integrity"]["digest"],
                            minimumSafeVersion=replacement["payload"]["version"],
                            dependencyClosure=replacement["payload"]["dependencyClosure"],
                        )
                    if item["packId"] == "oryn.mflux-mlx":
                        replacement = json.loads((output / "mflux-pack.json").read_text())
                        item.update(
                            version=replacement["payload"]["version"],
                            manifestDigest=replacement["integrity"]["digest"],
                            minimumSafeVersion=replacement["payload"]["version"],
                            dependencyClosure=replacement["payload"]["dependencyClosure"],
                        )
                    if item["packId"] == "oryn.diffusers-pytorch":
                        replacement = json.loads((output / "diffusers-pack.json").read_text())
                        item.update(
                            version=replacement["payload"]["version"],
                            manifestDigest=replacement["integrity"]["digest"],
                            minimumSafeVersion=replacement["payload"]["version"],
                            dependencyClosure=replacement["payload"]["dependencyClosure"],
                        )
                    if item["packId"] == "oryn.mlx-lm":
                        replacement = json.loads((output / "mlx-pack.json").read_text())
                        item.update(
                            version=replacement["payload"]["version"],
                            manifestDigest=replacement["integrity"]["digest"],
                            minimumSafeVersion=replacement["payload"]["version"],
                            dependencyClosure=replacement["payload"]["dependencyClosure"],
                        )

                    if (
                        item["packId"] == "oryn.transformers-pytorch"
                        and (output / "transformers-pack.json").exists()
                    ):
                        replacement = json.loads((output / "transformers-pack.json").read_text())
                        item.update(
                            version=replacement["payload"]["version"],
                            manifestDigest=replacement["integrity"]["digest"],
                            minimumSafeVersion=replacement["payload"]["version"],
                            dependencyClosure=replacement["payload"]["dependencyClosure"],
                        )
            raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            signature = subprocess.run(
                ["openssl", "dgst", "-sha256", "-sign", str(key)],
                input=raw,
                capture_output=True,
                check=True,
            ).stdout
            path.write_text(
                json.dumps(
                    {
                        "payload": payload,
                        "integrity": {
                            "algorithm": "rsa-sha256",
                            "keyId": key_id,
                            "digest": hashlib.sha256(raw).hexdigest(),
                            "signature": base64.b64encode(signature).decode(),
                        },
                    },
                    indent=2,
                )
                + "\n"
            )
        for compatibility_name in (
            "transformers-compatibility.json",
            "diffusers-compatibility.json",
            "mflux-compatibility.json",
            "llama-compatibility.json",
            "kokoro-compatibility.json",
        ):
            compatibility_path = output / compatibility_name
            if compatibility_path.exists():
                envelopes = json.loads(compatibility_path.read_text())
                for envelope in envelopes:
                    pack_name = compatibility_name.replace("-compatibility.json", "-pack.json")
                    pack = json.loads((output / pack_name).read_text())["payload"]
                    runtime = envelope["payload"]["runtime"]
                    if runtime["packId"] != pack["packId"]:
                        raise ValueError("Compatibility file references a different runtime pack")
                    runtime["packVersion"] = pack["version"]
                    raw = json.dumps(
                        envelope["payload"], sort_keys=True, separators=(",", ":")
                    ).encode()
                    signature = subprocess.run(
                        ["openssl", "dgst", "-sha256", "-sign", str(key)],
                        input=raw,
                        capture_output=True,
                        check=True,
                    ).stdout
                    envelope["integrity"] = {
                        "algorithm": "rsa-sha256",
                        "keyId": key_id,
                        "digest": hashlib.sha256(raw).hexdigest(),
                        "signature": base64.b64encode(signature).decode(),
                    }
                compatibility_path.write_text(json.dumps(envelopes, indent=2) + "\n")
        modulus = (
            subprocess.run(
                ["openssl", "rsa", "-in", str(key), "-noout", "-modulus"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .split("=", 1)[1]
        )
        (root / "src/oryn_engine/generated_trust.py").write_text(
            "# ruff: noqa: E501\n# Development only. No private key is distributed.\n"
            f'DEVELOPMENT_KEY_ID = "{key_id}"\nDEVELOPMENT_MODULUS = {int(modulus, 16)}\n'
        )


if __name__ == "__main__":
    main()
