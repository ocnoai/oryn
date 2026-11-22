from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from oryn_engine.runtime_packs.manifests import (
    CompatibilityManifest,
    ManifestError,
    PackManifest,
    SignedManifestVerifier,
    parse_compatibility_manifest,
    parse_pack_manifest,
)


class RuntimePackRegistry:
    """Verified registry whose executable entrypoints come only from trusted pack manifests."""

    def __init__(self, verifier: SignedManifestVerifier, *, abi: str = "1") -> None:
        self._verifier = verifier
        self._abi = abi
        self._packs: dict[tuple[str, str], PackManifest] = {}
        self._active: dict[str, str] = {}
        self._previous: dict[str, str] = {}
        self._compatibility: dict[str, CompatibilityManifest] = {}
        self._catalog_compatibility: set[str] = set()

    @property
    def packs(self) -> tuple[PackManifest, ...]:
        return tuple(self._packs[key] for key in sorted(self._packs))

    @property
    def compatibility(self) -> tuple[CompatibilityManifest, ...]:
        return tuple(self._compatibility[key] for key in sorted(self._compatibility))

    def register_pack(self, envelope: object, *, activate: bool = True) -> PackManifest:
        payload, digest = self._verifier.verify(envelope)
        pack = parse_pack_manifest(payload, digest)
        if pack.abi != self._abi:
            raise ManifestError("PACK_ABI_INCOMPATIBLE", "Runtime-pack ABI is incompatible.")
        key = (pack.pack_id, pack.version)
        existing = self._packs.get(key)
        if existing is not None and existing.digest != pack.digest:
            raise ManifestError("PACK_IDENTITY_CONFLICT", "Runtime-pack identity is ambiguous.")
        self._packs[key] = pack
        if activate:
            current = self._active.get(pack.pack_id)
            if current is not None and current != pack.version:
                self._previous[pack.pack_id] = current
            self._active[pack.pack_id] = pack.version
        return pack

    def register_compatibility(
        self, envelope: object, *, publish_to_catalog: bool = True
    ) -> CompatibilityManifest:
        payload, digest = self._verifier.verify(envelope)
        compatibility = parse_compatibility_manifest(payload, digest)
        pack = self._packs.get((compatibility.pack_id, compatibility.pack_version))
        if pack is None:
            raise ManifestError("PACK_NOT_INSTALLED", "Compatibility manifest references no pack.")
        runner = pack.runner(compatibility.runner_id)
        if runner is None or compatibility.task_contract not in runner.task_contracts:
            raise ManifestError(
                "PACK_RUNNER_INCOMPATIBLE",
                "Compatibility manifest references an unsupported runner contract.",
            )
        existing = self._compatibility.get(compatibility.manifest_id)
        if existing is not None and existing.digest != compatibility.digest:
            raise ManifestError(
                "PACK_IDENTITY_CONFLICT", "Compatibility manifest identity is ambiguous."
            )
        self._compatibility[compatibility.manifest_id] = compatibility
        if publish_to_catalog:
            self._catalog_compatibility.add(compatibility.manifest_id)
        else:
            self._catalog_compatibility.discard(compatibility.manifest_id)
        return compatibility

    def load_directory(self, directory: Path) -> None:
        root = directory.resolve()
        for path in sorted(root.glob("pack-*.json")):
            self.register_pack(_read_json(path, root))
        for path in sorted(root.glob("compat-*.json")):
            self.register_compatibility(_read_json(path, root))

    def active_pack(self, pack_id: str) -> PackManifest:
        version = self._active.get(pack_id)
        if version is None:
            raise ManifestError("PACK_NOT_INSTALLED", "Runtime pack is not active.")
        return self._packs[(pack_id, version)]

    def rollback(self, pack_id: str) -> PackManifest:
        previous = self._previous.get(pack_id)
        if previous is None:
            raise ManifestError(
                "PACK_ROLLBACK_UNAVAILABLE", "No runtime-pack rollback is available."
            )
        current = self._active[pack_id]
        self._active[pack_id] = previous
        self._previous[pack_id] = current
        return self._packs[(pack_id, previous)]

    def match(
        self,
        *,
        repository: str,
        revision: str,
        architectures: Iterable[str],
        task: str | None,
    ) -> tuple[CompatibilityManifest, ...]:
        architecture_set = set(architectures)
        matches = [
            item
            for item in self._compatibility.values()
            if self._active.get(item.pack_id) == item.pack_version
            and item.matches(repository, revision, architecture_set, task)
        ]
        exact_first = sorted(
            matches,
            key=lambda item: (
                item.repository is None,
                item.level != "native-declarative",
                item.manifest_id,
            ),
        )
        return tuple(exact_first)

    def catalog(self, *, task: str, output_modalities: Iterable[str]) -> tuple[dict[str, Any], ...]:
        expected_outputs = set(output_modalities)
        result: list[dict[str, Any]] = []
        for item in self._compatibility.values():
            outputs = set(item.capabilities.get("outputModalities", []))
            if (
                item.repository is not None
                and item.manifest_id in self._catalog_compatibility
                and self._active.get(item.pack_id) == item.pack_version
                and task in item.tasks
                and expected_outputs.issubset(outputs)
            ):
                result.append(
                    {
                        "provider": "huggingface",
                        "repository": item.repository,
                        "revision": item.revision,
                    }
                )
        return tuple(result[:64])


def _read_json(path: Path, root: Path) -> object:
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise ManifestError("PACK_PATH_INVALID", "Runtime-pack manifest path is unsafe.")
    if path.stat().st_size > 512 * 1024:
        raise ManifestError("PACK_MANIFEST_INVALID", "Runtime-pack manifest file is too large.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(
            "PACK_MANIFEST_INVALID", "Runtime-pack manifest cannot be read."
        ) from exc
