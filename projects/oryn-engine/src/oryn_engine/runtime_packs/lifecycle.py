from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from oryn_engine.runtime_packs.catalog import CatalogPack, PackCatalog
from oryn_engine.runtime_packs.manifests import ManifestError
from oryn_engine.runtime_packs.registry import RuntimePackRegistry

MAX_ARCHIVE_FILES = 10_000
MAX_EXPANDED_BYTES = 64 * 1024**3
CONTENT_INTEGRITY_FILE = ".oryn-content.sha256"


class PackLifecycleManager:
    """Browser-owned first-party pack storage with atomic activation and rollback.

    This class intentionally accepts an already downloaded local artifact. Network fetches,
    credentials, third-party keys, and website-triggered installation are outside its authority.
    """

    def __init__(
        self,
        root: Path,
        catalog: PackCatalog,
        registry: RuntimePackRegistry,
        *,
        platform: str,
        abi: str = "1",
    ) -> None:
        if not root.is_absolute() or root.is_symlink():
            raise ManifestError("PACK_STORAGE_INVALID", "Pack storage root is invalid.")
        self.root = root
        self.catalog = catalog
        self.registry = registry
        self.platform = platform
        self.abi = abi
        self.packs_root = root / "packs"
        self.staging_root = root / ".staging"
        self.state_path = root / "lifecycle-state.json"
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.packs_root.mkdir(exist_ok=True, mode=0o700)
        self.staging_root.mkdir(exist_ok=True, mode=0o700)
        self._state = self._read_state()
        self._reconcile_catalog()

    def inventory(self) -> dict[str, Any]:
        installed = self._downloaded_versions()
        active = self._state["active"]
        accepted = self._state["acceptedLicenses"]
        records: list[dict[str, Any]] = []
        for pack in self.catalog.packs:
            bundled = pack.distribution == "bundled"
            key = _key(pack.pack_id, pack.version)
            path = self._version_path(pack.pack_id, pack.version)
            downloaded = key in installed
            records.append(
                {
                    "packId": pack.pack_id,
                    "version": pack.version,
                    "publisher": pack.publisher,
                    "abi": pack.abi,
                    "dependencyClosure": pack.dependency_closure,
                    "platforms": list(pack.platforms),
                    "distribution": pack.distribution,
                    "state": (
                        "revoked"
                        if pack.revoked
                        else "active"
                        if bundled or active.get(pack.pack_id) == pack.version
                        else "staged"
                        if downloaded
                        else "available"
                    ),
                    "bundled": bundled,
                    "removable": downloaded and active.get(pack.pack_id) != pack.version,
                    "sizeBytes": _directory_size(path) if downloaded else 0,
                    "minimumSafeVersion": pack.minimum_safe_version,
                    "revoked": pack.revoked,
                    "license": {
                        "summary": pack.license.summary,
                        "notice": pack.license.notice,
                        "access": pack.license.access,
                        "acceptanceRequired": pack.license.acceptance_required,
                        "accepted": (
                            not pack.license.acceptance_required or accepted.get(key) is True
                        ),
                    },
                }
            )
        return {
            "catalogVersion": self.catalog.catalog_version,
            "catalogDigest": self.catalog.digest,
            "thirdPartyPublishersEnabled": False,
            "automaticExecutableUpdatesEnabled": False,
            "packs": records,
        }

    def stage_file(self, pack_id: str, version: str, archive: Path) -> dict[str, Any]:
        pack = self.catalog.pack(pack_id, version)
        self._validate_installable(pack)
        if pack.distribution != "download" or pack.artifact is None:
            raise ManifestError("PACK_BUNDLED", "Bundled packs are managed by application updates.")
        if (
            not archive.is_absolute()
            or archive.is_symlink()
            or not archive.is_file()
            or archive.stat().st_size != pack.artifact.size_bytes
            or _sha256(archive) != pack.artifact.sha256
        ):
            raise ManifestError("PACK_ARTIFACT_INVALID", "Pack artifact integrity is invalid.")
        destination = self._version_path(pack.pack_id, pack.version)
        if destination.exists():
            self._validate_marker(destination, pack)
            return self._record(pack)
        temporary = Path(tempfile.mkdtemp(prefix="pack-", dir=self.staging_root))
        try:
            self._extract(archive, temporary, pack)
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self._record(pack)

    def activate(self, pack_id: str, version: str) -> dict[str, Any]:
        pack = self.catalog.pack(pack_id, version)
        self._validate_installable(pack)
        if pack.distribution == "bundled":
            registered = self.registry.active_pack(pack.pack_id)
            if registered.version != pack.version or registered.digest != pack.manifest_digest:
                raise ManifestError("PACK_IDENTITY_CONFLICT", "Bundled pack identity is invalid.")
            return self._record(pack)
        destination = self._version_path(pack.pack_id, pack.version)
        self._validate_marker(destination, pack)
        current = self._state["active"].get(pack.pack_id)
        if current and current != pack.version:
            self._state["previous"][pack.pack_id] = current
        self._state["active"][pack.pack_id] = pack.version
        self._write_state()
        return self._record(pack)

    def rollback(self, pack_id: str) -> dict[str, Any]:
        previous = self._state["previous"].get(pack_id)
        if not isinstance(previous, str):
            raise ManifestError("PACK_ROLLBACK_UNAVAILABLE", "No safe pack rollback is available.")
        pack = self.catalog.pack(pack_id, previous)
        self._validate_installable(pack)
        self._validate_marker(self._version_path(pack_id, previous), pack)
        current = self._state["active"].get(pack_id)
        self._state["active"][pack_id] = previous
        if isinstance(current, str):
            self._state["previous"][pack_id] = current
        self._write_state()
        return self._record(pack)

    def accept_license(self, pack_id: str, version: str, *, accepted: bool) -> dict[str, Any]:
        pack = self.catalog.pack(pack_id, version)
        key = _key(pack_id, version)
        if not pack.license.acceptance_required and not accepted:
            raise ManifestError("PACK_LICENSE_INVALID", "Public pack acceptance cannot be revoked.")
        if accepted:
            self._state["acceptedLicenses"][key] = True
        else:
            self._state["acceptedLicenses"].pop(key, None)
        self._write_state()
        return self._record(pack)

    def remove(self, pack_id: str, version: str) -> dict[str, Any]:
        pack = self.catalog.pack(pack_id, version)
        if pack.distribution == "bundled":
            raise ManifestError("PACK_BUNDLED", "Bundled packs are removed by application updates.")
        if self._state["active"].get(pack_id) == version:
            raise ManifestError("PACK_ACTIVE", "An active runtime pack cannot be removed.")
        path = self._version_path(pack_id, version)
        reclaimed = _directory_size(path)
        self._safe_remove(path)
        if self._state["previous"].get(pack_id) == version:
            self._state["previous"].pop(pack_id, None)
        self._state["acceptedLicenses"].pop(_key(pack_id, version), None)
        self._write_state()
        return {"packId": pack_id, "version": version, "reclaimedBytes": reclaimed}

    def cleanup(self) -> dict[str, Any]:
        active = {_key(pack_id, version) for pack_id, version in self._state["active"].items()}
        removed = 0
        reclaimed = 0
        for key, path in self._downloaded_versions().items():
            if key in active:
                continue
            reclaimed += _directory_size(path)
            self._safe_remove(path)
            self._state["acceptedLicenses"].pop(key, None)
            removed += 1
        self._write_state()
        return {"removedPacks": removed, "reclaimedBytes": reclaimed}

    def refresh_catalog(self, catalog: PackCatalog) -> None:
        if catalog.catalog_version < self.catalog.catalog_version:
            raise ManifestError("PACK_CATALOG_ROLLBACK", "Pack catalog rollback is forbidden.")
        if (
            catalog.catalog_version == self.catalog.catalog_version
            and catalog.digest != self.catalog.digest
        ):
            raise ManifestError(
                "PACK_CATALOG_CONFLICT", "Pack catalog version has conflicting content."
            )
        self.catalog = catalog
        self._reconcile_catalog()

    def _reconcile_catalog(self) -> None:
        bundled_ids = {
            pack.pack_id for pack in self.catalog.packs if pack.distribution == "bundled"
        }
        for pack_id in bundled_ids:
            registered = self.registry.active_pack(pack_id)
            expected = next(
                (
                    pack
                    for pack in self.catalog.packs
                    if pack.pack_id == pack_id
                    and pack.version == registered.version
                    and pack.distribution == "bundled"
                ),
                None,
            )
            if (
                expected is None
                or expected.manifest_digest != registered.digest
                or expected.abi != registered.abi
                or expected.dependency_closure != registered.dependency_closure
            ):
                raise ManifestError("PACK_IDENTITY_CONFLICT", "Bundled pack identity is invalid.")
        for pack_id, version in tuple(self._state["active"].items()):
            try:
                pack = self.catalog.pack(pack_id, version)
            except ManifestError:
                self._state["active"].pop(pack_id, None)
                continue
            if pack.revoked:
                previous = self._state["previous"].get(pack_id)
                try:
                    fallback = self.catalog.pack(pack_id, previous) if previous else None
                except ManifestError:
                    fallback = None
                if fallback is not None and not fallback.revoked:
                    self._state["active"][pack_id] = fallback.version
                else:
                    self._state["active"].pop(pack_id, None)
        self._state["catalogVersion"] = self.catalog.catalog_version
        self._state["catalogDigest"] = self.catalog.digest
        self._write_state()

    def _validate_installable(self, pack: CatalogPack) -> None:
        if pack.revoked:
            raise ManifestError("PACK_REVOKED", "Runtime pack version is revoked or unsafe.")
        if self.platform not in pack.platforms:
            raise ManifestError(
                "PACK_PLATFORM_INCOMPATIBLE", "Runtime pack platform is incompatible."
            )
        if pack.abi != self.abi:
            raise ManifestError("PACK_ABI_INCOMPATIBLE", "Runtime pack ABI is incompatible.")
        if pack.license.acceptance_required and not self._state["acceptedLicenses"].get(
            _key(pack.pack_id, pack.version), False
        ):
            raise ManifestError(
                "PACK_LICENSE_REQUIRED", "Runtime pack license acceptance is required."
            )

    def _extract(self, archive: Path, destination: Path, pack: CatalogPack) -> None:
        try:
            source = zipfile.ZipFile(archive)
        except (OSError, zipfile.BadZipFile) as exc:
            raise ManifestError("PACK_ARTIFACT_INVALID", "Pack archive is invalid.") from exc
        with source:
            entries = source.infolist()
            if not 1 <= len(entries) <= MAX_ARCHIVE_FILES:
                raise ManifestError("PACK_ARTIFACT_INVALID", "Pack archive file count is invalid.")
            total = sum(item.file_size for item in entries)
            if total > MAX_EXPANDED_BYTES or total > max(512 * 1024**2, archive.stat().st_size * 8):
                raise ManifestError("PACK_ARTIFACT_INVALID", "Pack archive expansion is invalid.")
            for item in entries:
                path = PurePosixPath(item.filename)
                mode = item.external_attr >> 16
                file_type = stat.S_IFMT(mode)
                if (
                    not item.filename
                    or path.is_absolute()
                    or ".." in path.parts
                    or "\\" in item.filename
                    or item.flag_bits & 0x1
                    or (file_type and file_type not in {stat.S_IFREG, stat.S_IFDIR})
                ):
                    raise ManifestError("PACK_ARTIFACT_INVALID", "Pack archive path is unsafe.")
                target = destination.joinpath(*path.parts)
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True, mode=0o700)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with source.open(item) as input_file, target.open("xb") as output_file:
                    shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
                target.chmod(0o700 if mode & stat.S_IXUSR else 0o600)
        self._validate_marker(destination, pack, require_integrity=False)
        integrity = destination / CONTENT_INTEGRITY_FILE
        integrity.write_text(_content_digest(destination) + "\n", encoding="ascii")
        integrity.chmod(0o600)

    def _validate_marker(
        self, root: Path, pack: CatalogPack, *, require_integrity: bool = True
    ) -> None:
        marker = root / "oryn-runtime-pack.json"
        if (
            not root.is_dir()
            or root.is_symlink()
            or not marker.is_file()
            or marker.is_symlink()
            or marker.stat().st_size > 64 * 1024
        ):
            raise ManifestError("PACK_ARTIFACT_INVALID", "Pack marker is unavailable.")
        try:
            value = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ManifestError("PACK_ARTIFACT_INVALID", "Pack marker is invalid.") from exc
        expected = {
            "schemaVersion": 1,
            "packId": pack.pack_id,
            "version": pack.version,
            "manifestDigest": pack.manifest_digest,
            "abi": pack.abi,
            "dependencyClosure": pack.dependency_closure,
            "platform": self.platform,
        }
        if value != expected:
            raise ManifestError("PACK_IDENTITY_CONFLICT", "Pack marker identity is invalid.")
        if require_integrity:
            integrity = root / CONTENT_INTEGRITY_FILE
            try:
                expected_digest = integrity.read_text(encoding="ascii").strip()
            except (OSError, UnicodeDecodeError) as exc:
                raise ManifestError(
                    "PACK_ARTIFACT_INVALID", "Pack content integrity is unavailable."
                ) from exc
            if (
                integrity.is_symlink()
                or not re.fullmatch(r"[0-9a-f]{64}", expected_digest)
                or not _constant_time_equal(expected_digest, _content_digest(root))
            ):
                raise ManifestError("PACK_ARTIFACT_INVALID", "Pack content integrity is invalid.")

    def _record(self, pack: CatalogPack) -> dict[str, Any]:
        return next(
            item
            for item in self.inventory()["packs"]
            if item["packId"] == pack.pack_id and item["version"] == pack.version
        )

    def _version_path(self, pack_id: str, version: str) -> Path:
        path = self.packs_root / pack_id / version
        resolved_parent = path.parent.resolve()
        if not resolved_parent.is_relative_to(self.packs_root.resolve()):
            raise ManifestError("PACK_PATH_INVALID", "Pack storage path is invalid.")
        return path

    def _downloaded_versions(self) -> dict[str, Path]:
        result: dict[str, Path] = {}
        for pack_root in self.packs_root.iterdir():
            if not pack_root.is_dir() or pack_root.is_symlink():
                continue
            for version_root in pack_root.iterdir():
                if version_root.is_dir() and not version_root.is_symlink():
                    result[_key(pack_root.name, version_root.name)] = version_root
        return result

    def _safe_remove(self, path: Path) -> None:
        if path.is_symlink() or not path.resolve().is_relative_to(self.packs_root.resolve()):
            raise ManifestError("PACK_PATH_INVALID", "Pack removal path is invalid.")
        if path.exists():
            shutil.rmtree(path)

    def _read_state(self) -> dict[str, Any]:
        default = {
            "schemaVersion": 1,
            "catalogVersion": 0,
            "catalogDigest": "",
            "active": {},
            "previous": {},
            "acceptedLicenses": {},
        }
        if not self.state_path.exists():
            return default
        if self.state_path.is_symlink() or not self.state_path.is_file():
            raise ManifestError("PACK_STORAGE_INVALID", "Pack lifecycle state is invalid.")
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ManifestError("PACK_STORAGE_INVALID", "Pack lifecycle state is invalid.") from exc
        if (
            not isinstance(value, dict)
            or set(value) != set(default)
            or value.get("schemaVersion") != 1
            or not isinstance(value.get("catalogVersion"), int)
            or isinstance(value.get("catalogVersion"), bool)
            or value["catalogVersion"] < 0
            or not isinstance(value.get("catalogDigest"), str)
            or (
                value["catalogDigest"] != ""
                and re.fullmatch(r"[0-9a-f]{64}", value["catalogDigest"]) is None
            )
            or not all(
                isinstance(value.get(name), dict)
                for name in ("active", "previous", "acceptedLicenses")
            )
            or len(value["active"]) > 256
            or len(value["previous"]) > 256
            or len(value["acceptedLicenses"]) > 512
            or not all(
                isinstance(pack_id, str)
                and isinstance(version, str)
                and len(pack_id) <= 128
                and len(version) <= 64
                for section in (value["active"], value["previous"])
                for pack_id, version in section.items()
            )
            or not all(
                isinstance(key, str) and len(key) <= 193 and accepted is True
                for key, accepted in value["acceptedLicenses"].items()
            )
        ):
            raise ManifestError("PACK_STORAGE_INVALID", "Pack lifecycle state is invalid.")
        return value

    def _write_state(self) -> None:
        encoded = json.dumps(self._state, sort_keys=True, separators=(",", ":")) + "\n"
        descriptor, temporary = tempfile.mkstemp(prefix="lifecycle-", dir=self.root)
        temporary_path = Path(temporary)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            temporary_path.chmod(0o600)
            os.replace(temporary_path, self.state_path)
        finally:
            temporary_path.unlink(missing_ok=True)


def _key(pack_id: str, version: str) -> str:
    return f"{pack_id}@{version}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _content_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix()):
        relative = item.relative_to(root).as_posix()
        if relative == CONTENT_INTEGRITY_FILE:
            continue
        if item.is_symlink():
            raise ManifestError("PACK_ARTIFACT_INVALID", "Pack content path is unsafe.")
        if item.is_dir():
            continue
        if not item.is_file():
            raise ManifestError("PACK_ARTIFACT_INVALID", "Pack content path is unsafe.")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with item.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _constant_time_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left, right)


def validate_content_integrity(root: Path) -> bool:
    integrity = root / CONTENT_INTEGRITY_FILE
    try:
        expected = integrity.read_text(encoding="ascii").strip()
        return (
            root.is_dir()
            and not root.is_symlink()
            and integrity.is_file()
            and not integrity.is_symlink()
            and re.fullmatch(r"[0-9a-f]{64}", expected) is not None
            and hmac.compare_digest(expected, _content_digest(root))
        )
    except (OSError, UnicodeDecodeError, ManifestError):
        return False


def _directory_size(path: Path) -> int:
    if not path.is_dir() or path.is_symlink():
        return 0
    return sum(
        item.stat().st_size for item in path.rglob("*") if item.is_file() and not item.is_symlink()
    )
