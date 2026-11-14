from __future__ import annotations

import importlib.util
import platform
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HardwareInfo:
    platform: str
    architecture: str
    backends: list[dict[str, str]]
    preferred_backend: str
    unified_memory: bool


class HardwareManager:
    def detect(self) -> HardwareInfo:
        system = platform.system().lower()
        architecture = platform.machine().lower()
        backends = [{"id": "system-cpu", "deviceClass": "cpu"}]
        preferred = "cpu"
        torch = self._torch()
        if torch is not None:
            if torch.backends.mps.is_available():
                backends.append({"id": "pytorch-mps", "deviceClass": "mps"})
                preferred = "mps"
            if torch.cuda.is_available():
                backends.append({"id": "pytorch-cuda", "deviceClass": "cuda"})
                preferred = "cuda"
        return HardwareInfo(
            platform=system,
            architecture=architecture,
            backends=backends,
            preferred_backend=preferred,
            unified_memory=system == "darwin" and architecture in {"arm64", "aarch64"},
        )

    def accelerator_metrics(self) -> dict[str, Any]:
        info = self.detect()
        result: dict[str, Any] = {"backend": info.preferred_backend}
        torch = self._torch()
        if torch is None:
            return result
        try:
            if info.preferred_backend == "mps":
                result["allocatedBytes"] = int(torch.mps.current_allocated_memory())
                driver = getattr(torch.mps, "driver_allocated_memory", None)
                if driver is not None:
                    result["reservedBytes"] = int(driver())
            elif info.preferred_backend == "cuda":
                result["allocatedBytes"] = int(torch.cuda.memory_allocated())
                result["reservedBytes"] = int(torch.cuda.memory_reserved())
        except (RuntimeError, AttributeError):
            pass
        return result

    @staticmethod
    def _torch() -> Any | None:
        if importlib.util.find_spec("torch") is None:
            return None
        import torch

        return torch
