"""Build the separate speech dependency closure from reviewed runtime inputs."""

import json


def write_manifests(root, output):
    from oryn_engine import qwen_tts
    from oryn_engine.runtime_packs.builtin_foundations import _adapter, _foundation

    spec = json.loads((root.parent / "browser/packaging/runtime-spec.json").read_text())[
        "runtimePacks"
    ][qwen_tts.PACK_ID]
    foundation_id = "oryn.runtime.pytorch-speech"
    formats, backends = ["safetensors", "json", "tokenizer"], ["mps", "cpu"]
    pack = {
        "kind": "runtime-pack",
        "schemaVersion": 1,
        "packId": qwen_tts.PACK_ID,
        "version": spec["version"],
        "publisher": "oryn.first-party",
        "abi": "1",
        "runtime": {"kind": "python", "entrypoint": "oryn_engine.runtime_packs.qwen_tts_worker"},
        "dependencyClosure": spec["dependencyClosure"],
        "runners": [
            {
                "runnerId": "qwen-tts.speech-synthesis",
                "taskContracts": ["speech-synthesis-v1"],
                "backends": backends,
            }
        ],
        "policy": {"network": "deny", "filesystemRoots": ["model", "artifact-output", "scratch"]},
    }
    foundation = _foundation(
        foundation_id,
        spec["dependencyClosure"],
        backends,
        formats,
        ["fp32"],
        [],
        "Apache-2.0 Qwen TTS and BSD-3-Clause PyTorch with composite dependencies",
    )
    adapter = _adapter(
        "oryn.adapter.qwen-tts",
        foundation_id,
        [qwen_tts.ARCHITECTURE],
        formats,
        ["speech-synthesis-v1"],
        backends,
        ["fp32"],
        [],
        sorted(qwen_tts.REQUIRED_FILES),
        "qwen-tts-v1",
        qwen_tts.PACK_ID,
    )
    for filename, payload in (
        ("qwen-tts-pack.json", pack),
        ("qwen-tts-foundation.json", foundation),
        ("qwen-tts-adapter.json", adapter),
    ):
        (output / filename).write_text(json.dumps({"payload": payload}, indent=2) + "\n")
    catalog_path = output / "pack-catalog.json"
    catalog = json.loads(catalog_path.read_text())
    entries = catalog["payload"]["packs"]
    if not any(e["packId"] == qwen_tts.PACK_ID for e in entries):
        entries.append(
            {
                "packId": qwen_tts.PACK_ID,
                "version": spec["version"],
                "publisher": "oryn.first-party",
                "manifestDigest": "0" * 64,
                "abi": "1",
                "dependencyClosure": spec["dependencyClosure"],
                "platforms": ["macos-arm64"],
                "distribution": "bundled",
                "artifact": None,
                "minimumSafeVersion": spec["version"],
                "revokedVersions": [],
                "license": {
                    "summary": "Apache-2.0 Qwen TTS with composite speech dependencies",
                    "notice": "Review packaged upstream dependency notices before distribution.",
                    "access": "public",
                    "acceptanceRequired": False,
                },
            }
        )
    catalog["payload"]["catalogVersion"] = 7
    entries.sort(key=lambda item: item["packId"])
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
