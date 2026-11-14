from __future__ import annotations

import re
from dataclasses import dataclass

from oryn_engine.runtime_packs.manifests import (
    IDENTIFIER,
    SEMVER,
    ManifestError,
    SignedManifestVerifier,
)

MAX_CATALOG_PACKS = 256
SUPPORTED_PLATFORMS = {"macos-arm64", "windows-amd64", "linux-amd64"}


@dataclass(frozen=True)
class PackLicense:
    summary: str
    notice: str
    access: str
    acceptance_required: bool


@dataclass(frozen=True)
class PackArtifact:
    url: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class CatalogPack:
    pack_id: str
    version: str
    publisher: str
    manifest_digest: str
    abi: str
    dependency_closure: str
    platforms: tuple[str, ...]
    distribution: str
    artifact: PackArtifact | None
    minimum_safe_version: str
    revoked_versions: tuple[str, ...]
    license: PackLicense

    @property
    def revoked(self) -> bool:
        return self.version in self.revoked_versions or _version(self.version) < _version(
            self.minimum_safe_version
        )


@dataclass(frozen=True)
class PackCatalog:
    catalog_version: int
    publisher: str
    generated_at: int
    packs: tuple[CatalogPack, ...]
    digest: str

    def pack(self, pack_id: str, version: str | None = None) -> CatalogPack:
        matches = [
            item
            for item in self.packs
            if item.pack_id == pack_id and (version is None or item.version == version)
        ]
        if not matches:
            raise ManifestError("PACK_NOT_IN_CATALOG", "Runtime pack is absent from the catalog.")
        return max(matches, key=lambda item: _version(item.version))


@dataclass(frozen=True)
class PublisherTrustPolicy:
    catalog_key_ids: frozenset[str]
    pack_publishers: frozenset[str]

    def verify_catalog_envelope(self, envelope: object) -> None:
        if not isinstance(envelope, dict):
            raise ManifestError("PACK_CATALOG_INVALID", "Pack catalog envelope is invalid.")
        integrity = envelope.get("integrity")
        key_id = integrity.get("keyId") if isinstance(integrity, dict) else None
        if key_id not in self.catalog_key_ids:
            raise ManifestError(
                "PACK_PUBLISHER_UNTRUSTED", "Pack catalog publisher is not trusted."
            )

    def verify_pack(self, pack: CatalogPack) -> None:
        if pack.publisher not in self.pack_publishers:
            raise ManifestError("PACK_PUBLISHER_UNTRUSTED", "Pack publisher is not trusted.")


def parse_pack_catalog(
    envelope: object,
    verifier: SignedManifestVerifier,
    trust: PublisherTrustPolicy,
) -> PackCatalog:
    trust.verify_catalog_envelope(envelope)
    payload, digest = verifier.verify(envelope)
    if set(payload) != {
        "kind",
        "schemaVersion",
        "catalogVersion",
        "publisher",
        "generatedAt",
        "packs",
    }:
        raise ManifestError("PACK_CATALOG_INVALID", "Pack catalog fields are invalid.")
    if payload.get("kind") != "runtime-pack-catalog" or payload.get("schemaVersion") != 1:
        raise ManifestError("PACK_CATALOG_INVALID", "Pack catalog version is invalid.")
    catalog_version = _integer(payload.get("catalogVersion"), 1, 2**31 - 1)
    generated_at = _integer(payload.get("generatedAt"), 0, 2**63 - 1)
    publisher = _identifier(payload.get("publisher"))
    if publisher not in trust.pack_publishers:
        raise ManifestError("PACK_PUBLISHER_UNTRUSTED", "Pack catalog publisher is not trusted.")
    raw_packs = payload.get("packs")
    if not isinstance(raw_packs, list) or not 1 <= len(raw_packs) <= MAX_CATALOG_PACKS:
        raise ManifestError("PACK_CATALOG_INVALID", "Pack catalog entries are invalid.")
    packs = tuple(_pack(item) for item in raw_packs)
    identities = {(item.pack_id, item.version) for item in packs}
    if len(identities) != len(packs):
        raise ManifestError("PACK_CATALOG_INVALID", "Pack catalog identities are ambiguous.")
    for item in packs:
        trust.verify_pack(item)
    return PackCatalog(catalog_version, publisher, generated_at, packs, digest)


def _pack(raw: object) -> CatalogPack:
    if not isinstance(raw, dict) or set(raw) != {
        "packId",
        "version",
        "publisher",
        "manifestDigest",
        "abi",
        "dependencyClosure",
        "platforms",
        "distribution",
        "artifact",
        "minimumSafeVersion",
        "revokedVersions",
        "license",
    }:
        raise ManifestError("PACK_CATALOG_INVALID", "Pack catalog entry is invalid.")
    pack_id = _identifier(raw.get("packId"))
    version = _semver(raw.get("version"))
    publisher = _identifier(raw.get("publisher"))
    manifest_digest = _digest(raw.get("manifestDigest"))
    abi = _bounded_string(raw.get("abi"), 32)
    dependency_closure = _bounded_string(raw.get("dependencyClosure"), 128)
    platforms_raw = raw.get("platforms")
    if (
        not isinstance(platforms_raw, list)
        or not 1 <= len(platforms_raw) <= len(SUPPORTED_PLATFORMS)
        or len(set(platforms_raw)) != len(platforms_raw)
        or not set(platforms_raw).issubset(SUPPORTED_PLATFORMS)
    ):
        raise ManifestError("PACK_CATALOG_INVALID", "Pack platforms are invalid.")
    distribution = raw.get("distribution")
    artifact_raw = raw.get("artifact")
    artifact: PackArtifact | None = None
    if distribution == "bundled":
        if artifact_raw is not None:
            raise ManifestError("PACK_CATALOG_INVALID", "Bundled pack artifact is invalid.")
    elif distribution == "download":
        if not isinstance(artifact_raw, dict) or set(artifact_raw) != {
            "url",
            "sha256",
            "sizeBytes",
        }:
            raise ManifestError("PACK_CATALOG_INVALID", "Pack artifact is invalid.")
        url = _bounded_string(artifact_raw.get("url"), 2048)
        if not re.fullmatch(
            r"https://[A-Za-z0-9.-]+(?::[0-9]+)?/[A-Za-z0-9._~!$&'()*+,;=:@%/-]+", url
        ):
            raise ManifestError("PACK_CATALOG_INVALID", "Pack artifact URL is invalid.")
        artifact = PackArtifact(
            url,
            _digest(artifact_raw.get("sha256")),
            _integer(artifact_raw.get("sizeBytes"), 1, 32 * 1024**3),
        )
    else:
        raise ManifestError("PACK_CATALOG_INVALID", "Pack distribution is invalid.")
    minimum_safe_version = _semver(raw.get("minimumSafeVersion"))
    revoked_raw = raw.get("revokedVersions")
    if (
        not isinstance(revoked_raw, list)
        or len(revoked_raw) > 128
        or len(set(revoked_raw)) != len(revoked_raw)
    ):
        raise ManifestError("PACK_CATALOG_INVALID", "Pack revocations are invalid.")
    revoked = tuple(_semver(item) for item in revoked_raw)
    license_raw = raw.get("license")
    if not isinstance(license_raw, dict) or set(license_raw) != {
        "summary",
        "notice",
        "access",
        "acceptanceRequired",
    }:
        raise ManifestError("PACK_CATALOG_INVALID", "Pack license metadata is invalid.")
    access = license_raw.get("access")
    acceptance = license_raw.get("acceptanceRequired")
    if access not in {"public", "gated"} or not isinstance(acceptance, bool):
        raise ManifestError("PACK_CATALOG_INVALID", "Pack access metadata is invalid.")
    if access == "gated" and not acceptance:
        raise ManifestError("PACK_CATALOG_INVALID", "Gated pack acceptance is required.")
    return CatalogPack(
        pack_id,
        version,
        publisher,
        manifest_digest,
        abi,
        dependency_closure,
        tuple(platforms_raw),
        distribution,
        artifact,
        minimum_safe_version,
        revoked,
        PackLicense(
            _bounded_string(license_raw.get("summary"), 256),
            _bounded_string(license_raw.get("notice"), 512),
            access,
            acceptance,
        ),
    )


def _version(value: str) -> tuple[int, int, int, int, str]:
    match = SEMVER.fullmatch(value)
    if match is None:
        raise ManifestError("PACK_CATALOG_INVALID", "Pack version is invalid.")
    prerelease = value.partition("-")[2]
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
        1 if not prerelease else 0,
        prerelease,
    )


def _semver(value: object) -> str:
    result = _bounded_string(value, 64)
    _version(result)
    return result


def _identifier(value: object) -> str:
    result = _bounded_string(value, 128)
    if IDENTIFIER.fullmatch(result) is None:
        raise ManifestError("PACK_CATALOG_INVALID", "Pack catalog identifier is invalid.")
    return result


def _bounded_string(value: object, maximum: int) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= maximum:
        raise ManifestError("PACK_CATALOG_INVALID", "Pack catalog string is invalid.")
    return value


def _digest(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ManifestError("PACK_CATALOG_INVALID", "Pack digest is invalid.")
    return value


def _integer(value: object, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ManifestError("PACK_CATALOG_INVALID", "Pack catalog integer is invalid.")
    return value
