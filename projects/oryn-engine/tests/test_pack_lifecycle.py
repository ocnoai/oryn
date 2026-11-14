from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from oryn_engine.runtime_packs.builtin_catalog import builtin_registry
from oryn_engine.runtime_packs.builtin_pack_catalog import (
    BUILTIN_PACK_CATALOG,
    CATALOG_PUBLISHER,
    builtin_pack_catalog,
)
from oryn_engine.runtime_packs.catalog import (
    CatalogPack,
    PackArtifact,
    PackCatalog,
    PackLicense,
    PublisherTrustPolicy,
    parse_pack_catalog,
)
from oryn_engine.runtime_packs.lifecycle import PackLifecycleManager
from oryn_engine.runtime_packs.manifests import ManifestError, SignedManifestVerifier


def _archive(path: Path, version: str, *, unsafe_name: str | None = None) -> None:
    marker = {
        "schemaVersion": 1,
        "packId": "oryn.fixture-download",
        "version": version,
        "manifestDigest": "a" * 64,
        "abi": "1",
        "dependencyClosure": f"fixture-{version}",
        "platform": "macos-arm64",
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr("oryn-runtime-pack.json", json.dumps(marker))
        output.writestr(unsafe_name or "bin/worker", b"safe executable fixture")


def _pack(path: Path, version: str, *, acceptance: bool = False) -> CatalogPack:
    return CatalogPack(
        "oryn.fixture-download",
        version,
        "oryn.first-party",
        "a" * 64,
        "1",
        f"fixture-{version}",
        ("macos-arm64",),
        "download",
        PackArtifact(
            "https://downloads.oryn.ai/packs/fixture.zip",
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_size,
        ),
        "0.1.0",
        (),
        PackLicense(
            "Fixture license", "Fixture notice", "gated" if acceptance else "public", acceptance
        ),
    )


def _catalog(*packs: CatalogPack, version: int = 1) -> PackCatalog:
    return PackCatalog(version, "oryn.first-party", 1_776_643_200_000, tuple(packs), "b" * 64)


def test_builtin_pack_catalog_is_signed_first_party_and_matches_registered_packs() -> None:
    catalog = builtin_pack_catalog()
    registry = builtin_registry()

    assert catalog.catalog_version == 7
    assert {item.pack_id for item in catalog.packs} == {item.pack_id for item in registry.packs}
    for item in catalog.packs:
        registered = registry.active_pack(item.pack_id)
        assert item.publisher == "oryn.first-party"
        assert item.distribution == "bundled"
        assert item.artifact is None
        assert item.manifest_digest == registered.digest
        assert item.dependency_closure == registered.dependency_closure
        assert item.license.summary and item.license.notice

    tampered = copy.deepcopy(BUILTIN_PACK_CATALOG)
    tampered["payload"]["catalogVersion"] = 3
    with pytest.raises(ManifestError) as failure:
        parse_pack_catalog(
            tampered,
            SignedManifestVerifier({CATALOG_PUBLISHER.key_id: CATALOG_PUBLISHER}),
            PublisherTrustPolicy(
                frozenset({CATALOG_PUBLISHER.key_id}), frozenset({"oryn.first-party"})
            ),
        )
    assert failure.value.code == "PACK_DIGEST_INVALID"

    with pytest.raises(ManifestError) as publisher_failure:
        parse_pack_catalog(
            BUILTIN_PACK_CATALOG,
            SignedManifestVerifier({CATALOG_PUBLISHER.key_id: CATALOG_PUBLISHER}),
            PublisherTrustPolicy(frozenset({CATALOG_PUBLISHER.key_id}), frozenset()),
        )
    assert publisher_failure.value.code == "PACK_PUBLISHER_UNTRUSTED"


def test_download_pack_staging_activation_rollback_cleanup_and_restart(tmp_path: Path) -> None:
    first_archive = tmp_path / "first.zip"
    second_archive = tmp_path / "second.zip"
    _archive(first_archive, "0.1.0")
    _archive(second_archive, "0.2.0")
    first = _pack(first_archive, "0.1.0")
    second = _pack(second_archive, "0.2.0")
    catalog = _catalog(first, second)
    root = tmp_path / "lifecycle"
    manager = PackLifecycleManager(
        root.resolve(), catalog, builtin_registry(), platform="macos-arm64"
    )

    assert (
        manager.stage_file(first.pack_id, first.version, first_archive.resolve())["state"]
        == "staged"
    )
    assert manager.activate(first.pack_id, first.version)["state"] == "active"
    manager.stage_file(second.pack_id, second.version, second_archive.resolve())
    assert manager.activate(second.pack_id, second.version)["state"] == "active"
    assert manager.rollback(first.pack_id)["version"] == "0.1.0"

    restarted = PackLifecycleManager(
        root.resolve(), catalog, builtin_registry(), platform="macos-arm64"
    )
    records = restarted.inventory()["packs"]
    assert next(item for item in records if item["version"] == "0.1.0")["state"] == "active"
    cleanup = restarted.cleanup()
    assert cleanup["removedPacks"] == 1 and cleanup["reclaimedBytes"] > 0
    assert not (root / "packs" / first.pack_id / "0.2.0").exists()


def test_pack_lifecycle_rejects_tampering_traversal_platform_and_unaccepted_license(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "pack.zip"
    _archive(archive, "0.1.0")
    pack = _pack(archive, "0.1.0", acceptance=True)
    manager = PackLifecycleManager(
        (tmp_path / "lifecycle").resolve(),
        _catalog(pack),
        builtin_registry(),
        platform="macos-arm64",
    )
    with pytest.raises(ManifestError) as license_failure:
        manager.stage_file(pack.pack_id, pack.version, archive.resolve())
    assert license_failure.value.code == "PACK_LICENSE_REQUIRED"

    manager.accept_license(pack.pack_id, pack.version, accepted=True)
    corrupted = tmp_path / "corrupted.zip"
    corrupted.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(ManifestError) as digest_failure:
        manager.stage_file(pack.pack_id, pack.version, corrupted.resolve())
    assert digest_failure.value.code == "PACK_ARTIFACT_INVALID"

    unsafe = tmp_path / "unsafe.zip"
    _archive(unsafe, "0.1.0", unsafe_name="../escape")
    unsafe_pack = _pack(unsafe, "0.1.0")
    unsafe_manager = PackLifecycleManager(
        (tmp_path / "unsafe-lifecycle").resolve(),
        _catalog(unsafe_pack),
        builtin_registry(),
        platform="macos-arm64",
    )
    with pytest.raises(ManifestError) as path_failure:
        unsafe_manager.stage_file(unsafe_pack.pack_id, unsafe_pack.version, unsafe.resolve())
    assert path_failure.value.code == "PACK_ARTIFACT_INVALID"
    assert not (tmp_path / "escape").exists()

    incompatible = replace(
        pack,
        platforms=("windows-amd64",),
        license=replace(pack.license, acceptance_required=False, access="public"),
    )
    incompatible_manager = PackLifecycleManager(
        (tmp_path / "platform-lifecycle").resolve(),
        _catalog(incompatible),
        builtin_registry(),
        platform="macos-arm64",
    )
    with pytest.raises(ManifestError) as platform_failure:
        incompatible_manager.stage_file(pack.pack_id, pack.version, archive.resolve())
    assert platform_failure.value.code == "PACK_PLATFORM_INCOMPATIBLE"


def test_pack_lifecycle_rejects_installed_tree_tampering(tmp_path: Path) -> None:
    archive = tmp_path / "pack.zip"
    _archive(archive, "0.1.0")
    pack = _pack(archive, "0.1.0")
    root = (tmp_path / "lifecycle").resolve()
    manager = PackLifecycleManager(root, _catalog(pack), builtin_registry(), platform="macos-arm64")
    manager.stage_file(pack.pack_id, pack.version, archive.resolve())
    worker = root / "packs" / pack.pack_id / pack.version / "bin" / "worker"
    worker.write_bytes(b"tampered after verified extraction")

    with pytest.raises(ManifestError) as failure:
        manager.activate(pack.pack_id, pack.version)
    assert failure.value.code == "PACK_ARTIFACT_INVALID"


def test_revocation_disables_unsafe_active_version_and_catalog_rollback_is_rejected(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "pack.zip"
    _archive(archive, "0.1.0")
    pack = _pack(archive, "0.1.0")
    manager = PackLifecycleManager(
        (tmp_path / "lifecycle").resolve(),
        _catalog(pack, version=2),
        builtin_registry(),
        platform="macos-arm64",
    )
    manager.stage_file(pack.pack_id, pack.version, archive.resolve())
    manager.activate(pack.pack_id, pack.version)
    manager.refresh_catalog(_catalog(replace(pack, revoked_versions=("0.1.0",)), version=3))
    assert manager.inventory()["packs"][0]["state"] == "revoked"

    with pytest.raises(ManifestError) as rollback:
        manager.refresh_catalog(_catalog(pack, version=2))
    assert rollback.value.code == "PACK_CATALOG_ROLLBACK"

    with pytest.raises(ManifestError) as conflict:
        manager.refresh_catalog(replace(manager.catalog, digest="c" * 64))
    assert conflict.value.code == "PACK_CATALOG_CONFLICT"


def test_lifecycle_has_no_network_or_package_install_authority() -> None:
    source = (Path(__file__).parents[1] / "src/oryn_engine/runtime_packs/lifecycle.py").read_text()
    assert "httpx" not in source
    assert "requests" not in source
    assert "subprocess" not in source
    assert "pip install" not in source
