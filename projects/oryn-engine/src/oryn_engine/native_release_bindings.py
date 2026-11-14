"""Verify Developer ID bindings between reviewed inputs and signed release files.

The original RSA capability artifact remains the authority for reviewed inputs.
Only a separately sealed, publisher-pinned resource bundle may bind those input
digests to files changed by the macOS release build.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path, PurePosixPath

PUBLISHER_TEAM = "3AWD2NU8UF"
BUNDLE_IDENTIFIER = "ai.ocno.oryn.browser.nativebindings"
BUNDLE_NAME = "native-release-bindings.bundle"
MANIFEST_PATH = "Contents/Resources/native-hashes.json"
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_ENTRIES = 8192
PACKAGE_PATH = "runtime/lib/python3.12/site-packages/oryn_engine"
ORIGINAL_SOURCES = "Contents/Resources/original-sources"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MACHO_MAGICS = {
    b"\xfe\xed\xfa\xce",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"\xca\xfe\xba\xbf",
    b"\xbf\xba\xfe\xca",
}
SIGNING_REQUIREMENT = (
    "anchor apple generic and certificate 1[field.1.2.840.113635.100.6.2.6] exists "
    "and certificate leaf[field.1.2.840.113635.100.6.1.13] exists "
    f'and certificate leaf[subject.OU] = "{PUBLISHER_TEAM}" '
    f'and identifier "{BUNDLE_IDENTIFIER}"'
)
_SOURCE_PATCHES = {
    "generated_capabilities.py": (
        (
            "from oryn_engine.runtime_packs.manifests import "
            "SignedManifestVerifier, TrustedPublisherKey",
            "from oryn_engine.native_release_bindings import matches_build_input\n"
            "from oryn_engine.runtime_packs.manifests import "
            "SignedManifestVerifier, TrustedPublisherKey",
        ),
        (
            "if hashlib.sha256(path.read_bytes()).hexdigest() != expected:",
            "if not matches_build_input(path, expected):",
        ),
    ),
    "onnx_tasks.py": (
        (
            "from oryn_engine.model_metadata import metadata_file, validate_model_config",
            "from oryn_engine.model_metadata import metadata_file, validate_model_config\n"
            "from oryn_engine.native_release_bindings import matches_build_input",
        ),
        (
            "hashlib.sha256(Path(ort.capi.onnxruntime_pybind11_state.__file__).read_bytes()).hexdigest()\n"
            '        != generated_artifact()["buildInputs"]["onnxNativeSha256"]',
            "not matches_build_input(\n"
            "            Path(ort.capi.onnxruntime_pybind11_state.__file__),\n"
            '            generated_artifact()["buildInputs"]["onnxNativeSha256"],\n'
            "        )",
        ),
    ),
    "onnx_classifier.py": (
        (
            "from oryn_engine.onnx_graph import safe_path",
            "from oryn_engine.native_release_bindings import matches_build_input\n"
            "from oryn_engine.onnx_graph import safe_path",
        ),
        (
            '    with binary.open("rb") as stream:\n'
            '        digest = hashlib.file_digest(stream, "sha256").hexdigest()\n',
            '    digest_matches = matches_build_input(binary, inputs["opencvBinarySha256"])\n',
        ),
        ('or digest != inputs["opencvBinarySha256"]', "or not digest_matches"),
    ),
    "runtime_packs/llama_cpp_worker.py": (
        (
            "from oryn_engine.runtime_packs.gguf import (",
            "from oryn_engine.native_release_bindings import matches_build_input\n"
            "from oryn_engine.runtime_packs.gguf import (",
        ),
        (
            "path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != inputs[key]",
            "path.is_symlink() or not matches_build_input(path, inputs[key])",
        ),
    ),
}


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _bounded_bytes(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Release binding resource is oversized")
    return raw


def transformed_source(name: str, raw: bytes) -> bytes:
    """Apply only the reviewed native-digest substitutions to one original source."""
    if name not in _SOURCE_PATCHES:
        raise ValueError("Source is outside the release binding allowlist")
    value = raw.decode("utf-8")
    for original, replacement in _SOURCE_PATCHES[name]:
        if value.count(original) != 1:
            raise ValueError(f"Unexpected release source patch input: {name}")
        value = value.replace(original, replacement, 1)
    result = value.encode("utf-8")
    compile(result, name, "exec", dont_inherit=True)
    return result


def prepare_release_sources(engine_root: Path) -> list[dict[str, str]]:
    """Instrument only a fresh packaged app after its original runtime gate passes."""
    root = Path(engine_root).resolve(strict=True)
    loaded_root = _engine_root()
    if (
        loaded_root is None
        or loaded_root.resolve(strict=True) != root
        or root.name != "engine"
        or root.parent.name != "Resources"
        or root.parent.parent.name != "Contents"
        or root.parent.parent.parent.suffix != ".app"
    ):
        raise ValueError("Release source preparation requires the copied packaged app")
    from oryn_engine.generated_capabilities import generated_artifact

    inputs = generated_artifact()["buildInputs"]["orynNativeSources"]
    bundle = root / BUNDLE_NAME
    if bundle.exists() or bundle.is_symlink():
        raise ValueError("Release source preparation requires a fresh binding bundle")
    packages = [root / PACKAGE_PATH]
    packs = root / "runtime-packs"
    if packs.exists():
        for pack in packs.iterdir():
            package = pack / "site-packages/oryn_engine"
            if package.exists():
                packages.append(package)
                if len(packages) > 16:
                    raise ValueError("Too many Engine package copies")
    capability_digest = _capability_digest()
    updates = []
    for package in sorted(packages):
        if _digest(package / "data/capabilities.json") != capability_digest:
            raise ValueError("Bundled Engine copies have different capability identities")
        for name in _SOURCE_PATCHES:
            path = package / name
            relative = path.relative_to(root).as_posix()
            if _relative_file(root, path) != relative:
                raise ValueError("Unexpected original source path")
            raw = _bounded_bytes(path, 256 * 1024)
            original_digest = hashlib.sha256(raw).hexdigest()
            if original_digest != inputs.get(name):
                raise ValueError(f"Original source differs from signed capability inputs: {name}")
            updates.append((relative, raw, transformed_source(name, raw), original_digest))
    records = []
    for relative, original, updated, original_digest in updates:
        backup = bundle / ORIGINAL_SOURCES / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        with backup.open("xb") as stream:
            stream.write(original)
        (root / relative).write_bytes(updated)
        records.append(
            {
                "path": relative,
                "originalSha256": original_digest,
                "signedSha256": hashlib.sha256(updated).hexdigest(),
            }
        )
    return records


def _engine_root() -> Path | None:
    package = Path(__file__).resolve().parent
    if package.parent.name == "site-packages" and package.parents[2].name == "runtime-packs":
        return package.parents[3]
    if (
        package.parent.name != "site-packages"
        or package.parent.parent.name != "python3.12"
        or package.parents[2].name != "lib"
        or package.parents[3].name != "runtime"
    ):
        return None
    return package.parents[4]


def _source_name(relative: str) -> str | None:
    prefix = PACKAGE_PATH + "/"
    if relative.startswith(prefix):
        name = relative.removeprefix(prefix)
    else:
        parts = PurePosixPath(relative).parts
        if (
            len(parts) < 5
            or parts[0] != "runtime-packs"
            or parts[2:4] != ("site-packages", "oryn_engine")
        ):
            return None
        name = "/".join(parts[4:])
    return name if name in _SOURCE_PATCHES else None


def _capability_digest() -> str:
    from oryn_engine.generated_capabilities import generated_artifact

    generated_artifact()  # Authenticate the original RSA envelope before using its identity.
    return _digest(Path(__file__).parent / "data/capabilities.json")


def _relative_file(root: Path, path: Path) -> str:
    resolved = path.resolve(strict=True)
    relative = path.absolute().relative_to(root)
    if resolved != root / relative or not resolved.is_file():
        raise ValueError("Release input path is not a regular in-root file")
    for parent in (root / relative, *(root / relative).parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("Release input path traverses a symlink")
    return relative.as_posix()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate release binding JSON member")
        result[key] = value
    return result


def parse_bindings(raw: bytes, capability_sha256: str) -> dict[str, tuple[str, str]]:
    """Parse a closed, bounded manifest; signature verification remains mandatory."""
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError("Release binding manifest is oversized")
    value = json.loads(raw, object_pairs_hook=_object)
    if (
        not isinstance(value, dict)
        or set(value) != {"schemaVersion", "capabilitySha256", "files"}
        or type(value["schemaVersion"]) is not int
        or value["schemaVersion"] != 1
        or value["capabilitySha256"] != capability_sha256
        or not isinstance(value["files"], list)
        or not 1 <= len(value["files"]) <= MAX_ENTRIES
    ):
        raise ValueError("Invalid release binding manifest identity")
    result = {}
    for entry in value["files"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "originalSha256", "signedSha256"}:
            raise ValueError("Invalid release binding entry")
        name = entry["path"]
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or len(name) > 1024
            or "\\" in name
            or "\x00" in name
            or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            or str(PurePosixPath(name)) != name
            or name in result
            or any(
                not isinstance(entry[key], str) or not _DIGEST.fullmatch(entry[key])
                for key in ("originalSha256", "signedSha256")
            )
        ):
            raise ValueError("Invalid release binding path or digest")
        result[name] = (entry["originalSha256"], entry["signedSha256"])
    return result


def _bundle_fingerprint(bundle: Path) -> str:
    """Cache only identical bundle bytes, including every signature resource."""
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("Release binding bundle is not a regular directory")
    digest = hashlib.sha256()
    total = 0
    entries = []
    for path in bundle.rglob("*"):
        entries.append(path)
        if len(entries) > 192:
            raise ValueError("Release binding bundle has unexpected content")
    entries.sort()
    for path in entries:
        if path.is_symlink():
            raise ValueError("Release binding bundle contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Release binding bundle contains an irregular file")
        raw = _bounded_bytes(path, MAX_MANIFEST_BYTES)
        total += len(raw)
        if total > MAX_MANIFEST_BYTES + 512 * 1024:
            raise ValueError("Release binding bundle is oversized")
        name = path.relative_to(bundle).as_posix().encode()
        digest.update(len(name).to_bytes(4))
        digest.update(name)
        digest.update(len(raw).to_bytes(8))
        digest.update(raw)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _verified_bindings(bundle: Path, fingerprint: str, capability_sha256: str):
    raw = _bounded_bytes(bundle / MANIFEST_PATH, MAX_MANIFEST_BYTES)
    parsed = parse_bindings(raw, capability_sha256)
    result = subprocess.run(
        ["/usr/bin/codesign", "--verify", "--strict", "-R=" + SIGNING_REQUIREMENT, str(bundle)],
        capture_output=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0 or _bundle_fingerprint(bundle) != fingerprint:
        raise ValueError("Release binding publisher signature is invalid")
    return parsed


def matches_build_input(path: Path, expected_sha256: str) -> bool:
    """Match exact inputs or publisher-authenticated, bounded release transformations."""
    try:
        if not isinstance(expected_sha256, str) or not _DIGEST.fullmatch(expected_sha256):
            return False
        actual = _digest(path)
        if actual == expected_sha256:
            return True
        if sys.platform != "darwin":
            return False
        root = _engine_root()
        if root is None:
            return False
        root = root.resolve(strict=True)
        relative = _relative_file(root, path)
        bundle = root / BUNDLE_NAME
        fingerprint = _bundle_fingerprint(bundle)
        entries = _verified_bindings(bundle, fingerprint, _capability_digest())
        if entries.get(relative) != (expected_sha256, actual):
            return False
        source_name = _source_name(relative)
        if source_name is not None:
            original = _bounded_bytes(bundle / ORIGINAL_SOURCES / relative, 256 * 1024)
            return hashlib.sha256(original).hexdigest() == expected_sha256 and transformed_source(
                source_name, original
            ) == _bounded_bytes(path, 256 * 1024)
        with path.open("rb") as stream:
            return stream.read(4) in _MACHO_MAGICS
    except (OSError, ValueError, TypeError, RecursionError, subprocess.SubprocessError):
        return False
