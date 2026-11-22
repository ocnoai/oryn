from __future__ import annotations

import importlib.util
import sys
from typing import Any

from oryn_engine import __version__
from oryn_engine.runtime.hardware import HardwareManager


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def runtime_capabilities(
    engine_instance_id: str, max_message_bytes: int, *, chat_v2=False
) -> dict[str, Any]:
    from oryn_protocol.task_contracts import TASK_CONTRACTS

    transformers_available = _module_available("transformers") and _module_available("torch")
    hardware = HardwareManager().detect()
    return {
        "engineVersion": __version__,
        "engineInstanceId": engine_instance_id,
        "protocolVersion": 1,
        **(
            {
                "taskContracts": [
                    {"id": name, "revision": contract["revision"]}
                    for name, contract in sorted(TASK_CONTRACTS.items())
                ]
            }
            if chat_v2
            else {}
        ),
        "providers": ["huggingface"],
        "backends": hardware.backends,
        "runners": (
            [{"id": "transformers", "framework": "transformers"}] if transformers_available else []
        ),
        "scheduler": {
            "lanes": [{"id": "generation", "capacity": 1, "global": True}],
        },
        "features": {
            "reviewedSourceInspection": sys.platform == "darwin"
            and sys.version_info[:3] == (3, 12, 16),
            "taskProfiles": True,
            "streaming": False,
            "artifacts": True,
            "modelInspection": True,
            "memoryPressureMonitoring": True,
            "multipleResidentModels": True,
        },
        "limits": {
            "maxMessageBytes": max_message_bytes,
            "maxArtifactBindingsPerJob": 32,
        },
    }
