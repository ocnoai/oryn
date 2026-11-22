"""Fixed metadata projections; these never declare installed or qualified devices."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from oryn_engine.runtime.hardware import HardwareInfo
from oryn_engine.runtime_routing import canonical_digest


@dataclass(frozen=True)
class AuditTarget:
    id: str
    platform: str
    system: str
    architecture: str
    backends: tuple[str, ...]
    preferred_backend: str

    def hardware(self):
        return HardwareInfo(
            self.system,
            self.architecture,
            [{"id": f"assumed-{b}", "deviceClass": b} for b in self.backends],
            self.preferred_backend,
            self.platform == "macos-arm64",
        )

    def eligible_backends(self, pack, adapter, foundation):
        if foundation is None or adapter is None or self.platform not in foundation.platforms:
            return ()
        declared = {backend for runner in pack.runners for backend in runner.backends}
        return tuple(
            b
            for b in self.backends
            if b in declared and b in adapter.backends and b in foundation.backends
        )


TARGETS = {
    "macos-arm64": AuditTarget(
        "macos-arm64",
        "macos-arm64",
        "darwin",
        "arm64",
        ("mps", "metal", "mlx", "coreml", "cpu"),
        "mps",
    ),
    "linux-x64-cuda": AuditTarget(
        "linux-x64-cuda",
        "linux-x64",
        "linux",
        "x86_64",
        ("cuda", "cpu"),
        "cuda",
    ),
    "windows-x64-cuda": AuditTarget(
        "windows-x64-cuda",
        "windows-x64",
        "windows",
        "x86_64",
        ("cuda", "cpu"),
        "cuda",
    ),
}
TARGET_CHOICES = (*TARGETS, "mac-nvidia-union")


def targets_for(selection):
    return tuple(TARGETS.values()) if selection == "mac-nvidia-union" else (TARGETS[selection],)


def profile_provenance(selection):
    value = {
        "selection": selection,
        "profiles": [{**asdict(t), "backends": list(t.backends)} for t in targets_for(selection)],
    }
    return {**value, "digest": canonical_digest(value)}
