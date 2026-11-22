"""Build the reusable audio adapter on Oryn's shared MLX foundation."""

import json


def write_manifests(root, output):
    from oryn_engine.mlx_audio_tasks import TASK_CONTRACTS
    from oryn_engine.runtime_packs.builtin_foundations import _adapter

    pack_id = "oryn.mlx-audio"
    spec = json.loads((root.parent / "browser/packaging/runtime-spec.json").read_text())[
        "runtimePacks"
    ][pack_id]
    formats, backends = ["safetensors", "json", "tokenizer"], ["mlx"]
    contracts = sorted(TASK_CONTRACTS.values())
    pack = {
        "kind": "runtime-pack",
        "schemaVersion": 1,
        "packId": pack_id,
        "version": spec["version"],
        "publisher": "oryn.first-party",
        "abi": "1",
        "runtime": {"kind": "python", "entrypoint": "oryn_engine.runtime_packs.mlx_audio_worker"},
        "dependencyClosure": spec["dependencyClosure"],
        "runners": [
            {"runnerId": "mlx-audio.shared-tasks", "taskContracts": contracts, "backends": backends}
        ],
        "policy": {
            "network": "deny",
            "filesystemRoots": ["model", "artifact-input", "artifact-output", "scratch"],
        },
    }
    adapter = _adapter(
        "oryn.adapter.mlx-audio",
        "oryn.runtime.mlx",
        [],
        formats,
        contracts,
        backends,
        ["fp32", "fp16", "bf16"],
        ["int3", "int4", "int5", "int6", "int8"],
        ["config.json", "*.safetensors"],
        "mlx-audio-v1",
        pack_id,
    )
    adapter["foundation"]["minimumVersion"] = spec["foundationVersion"]
    for filename, payload in (
        ("mlx-audio-pack.json", pack),
        ("mlx-audio-adapter.json", adapter),
    ):
        (output / filename).write_text(json.dumps({"payload": payload}, indent=2) + "\n")

    catalog_path = output / "pack-catalog.json"
    catalog = json.loads(catalog_path.read_text())
    entries = catalog["payload"]["packs"]
    if not any(entry["packId"] == pack_id for entry in entries):
        entries.append(
            {
                "packId": pack_id,
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
                    "summary": "MIT MLX-Audio with composite audio runtime dependencies",
                    "notice": "Review packaged MLX-Audio, Mistral Common and dependency notices.",
                    "access": "public",
                    "acceptanceRequired": False,
                },
            }
        )
    catalog["payload"]["catalogVersion"] = 8
    entries.sort(key=lambda item: item["packId"])
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
