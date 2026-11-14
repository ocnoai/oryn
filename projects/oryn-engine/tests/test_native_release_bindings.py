from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from oryn_engine import native_release_bindings as binding

CAPABILITY_DIGEST = hashlib.sha256(
    (Path(binding.__file__).parent / "data/capabilities.json").read_bytes()
).hexdigest()


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def release(tmp_path, monkeypatch):
    root = tmp_path / "Oryn Browser.app/Contents/Resources/engine"
    root.mkdir(parents=True)
    calls = []

    def verify(command, **options):
        calls.append(command)
        assert options["timeout"] == 30
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(binding, "_engine_root", lambda: root)
    monkeypatch.setattr(binding, "_capability_digest", lambda: CAPABILITY_DIGEST)
    monkeypatch.setattr(binding.sys, "platform", "darwin")
    monkeypatch.setattr(binding.subprocess, "run", verify)
    binding._verified_bindings.cache_clear()
    yield root, calls
    binding._verified_bindings.cache_clear()


def manifest(root, files, *, capability_digest=CAPABILITY_DIGEST):
    path = root / binding.BUNDLE_NAME / binding.MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "capabilitySha256": capability_digest,
                "files": files,
            }
        )
    )
    return path


def native(root):
    path = root / "runtime/native.dylib"
    path.parent.mkdir()
    original = b"\xcf\xfa\xed\xfeoriginal"
    signed = b"\xcf\xfa\xed\xfesigned"
    path.write_bytes(signed)
    return path, {
        "path": "runtime/native.dylib",
        "originalSha256": digest(original),
        "signedSha256": digest(signed),
    }


def copied_sources(root):
    package = Path(binding.__file__).parent
    data = root / binding.PACKAGE_PATH / "data"
    data.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(package / "data/capabilities.json", data / "capabilities.json")
    originals = {}
    for name in binding._SOURCE_PATCHES:
        destination = root / binding.PACKAGE_PATH / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / name, destination)
        originals[name] = destination.read_bytes()
    return originals


def test_exact_inputs_need_no_release_binding_or_signature(tmp_path, monkeypatch):
    path = tmp_path / "input.py"
    raw = b"original source"
    path.write_bytes(raw)
    monkeypatch.setattr(binding, "_engine_root", lambda: pytest.fail("Unexpected binding lookup"))
    assert binding.matches_build_input(path, digest(raw))
    assert not binding.matches_build_input(path, "invalid")


def test_signed_native_binding_preserves_original_identity_and_publisher(release):
    root, calls = release
    path, entry = native(root)
    manifest(root, [entry])
    assert binding.matches_build_input(path, entry["originalSha256"])
    assert binding.matches_build_input(path, entry["originalSha256"])
    assert len(calls) == 1
    requirement = calls[0][3]
    assert requirement.startswith("-R=anchor apple generic")
    assert "1.2.840.113635.100.6.2.6" in requirement
    assert "1.2.840.113635.100.6.1.13" in requirement
    assert 'subject.OU] = "3AWD2NU8UF"' in requirement
    assert f'identifier "{binding.BUNDLE_IDENTIFIER}"' in requirement


@pytest.mark.parametrize("damage", ["payload", "original", "capability", "publisher", "timeout"])
def test_unsigned_or_mismatched_native_translation_fails_closed(release, monkeypatch, damage):
    root, _ = release
    path, entry = native(root)
    capability_digest = CAPABILITY_DIGEST
    if damage == "payload":
        path.write_bytes(b"\xcf\xfa\xed\xfetampered")
    if damage == "original":
        entry["originalSha256"] = "b" * 64
    if damage == "capability":
        capability_digest = "c" * 64
    if damage == "publisher":
        monkeypatch.setattr(
            binding.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1)
        )
    if damage == "timeout":

        def timeout(*args, **options):
            raise subprocess.TimeoutExpired("codesign", 30)

        monkeypatch.setattr(binding.subprocess, "run", timeout)
    manifest(root, [entry], capability_digest=capability_digest)
    assert not binding.matches_build_input(path, digest(b"\xcf\xfa\xed\xfeoriginal"))


def test_bundle_change_invalidates_cached_signature(release, monkeypatch):
    root, calls = release
    path, entry = native(root)
    document = manifest(root, [entry])
    assert binding.matches_build_input(path, entry["originalSha256"])
    document.write_text(document.read_text() + "\n")
    monkeypatch.setattr(binding.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1))
    assert not binding.matches_build_input(path, entry["originalSha256"])
    assert len(calls) == 1


@pytest.mark.parametrize(
    "name", ["../native", "/native", "a/../native", "a//b", "./native", "a\\b", "."]
)
def test_manifest_rejects_ambiguous_or_escaping_paths(name):
    raw = json.dumps(
        {
            "schemaVersion": 1,
            "capabilitySha256": CAPABILITY_DIGEST,
            "files": [{"path": name, "originalSha256": "b" * 64, "signedSha256": "c" * 64}],
        }
    ).encode()
    with pytest.raises(ValueError):
        binding.parse_bindings(raw, CAPABILITY_DIGEST)


@pytest.mark.parametrize(
    "damage", ["duplicate-key", "duplicate-path", "unknown-field", "boolean-version"]
)
def test_manifest_rejects_schema_ambiguities(damage):
    value = {
        "schemaVersion": 1,
        "capabilitySha256": CAPABILITY_DIGEST,
        "files": [{"path": "runtime/native", "originalSha256": "b" * 64, "signedSha256": "c" * 64}],
    }
    if damage == "duplicate-path":
        value["files"] *= 2
    elif damage == "unknown-field":
        value["unexpected"] = True
    elif damage == "boolean-version":
        value["schemaVersion"] = True
    raw = json.dumps(value).encode()
    if damage == "duplicate-key":
        raw = raw.replace(b'"schemaVersion": 1', b'"schemaVersion": 1,"schemaVersion": 1')
    with pytest.raises(ValueError):
        binding.parse_bindings(raw, CAPABILITY_DIGEST)


def test_prepare_sources_preserves_originals_and_only_changes_copied_app(release):
    root, _ = release
    originals = copied_sources(root)
    records = binding.prepare_release_sources(root)
    assert len(records) == 4
    for name, original in originals.items():
        relative = f"{binding.PACKAGE_PATH}/{name}"
        backup = root / binding.BUNDLE_NAME / binding.ORIGINAL_SOURCES / relative
        assert backup.read_bytes() == original
        assert (Path(binding.__file__).parent / name).read_bytes() == original
        updated = (root / relative).read_bytes()
        assert updated != original
        assert updated == binding.transformed_source(name, original)
        assert {
            "path": relative,
            "originalSha256": digest(original),
            "signedSha256": digest(updated),
        } in records
    with pytest.raises(ValueError, match="fresh binding bundle"):
        binding.prepare_release_sources(root)


def test_prepare_sources_checks_all_originals_before_any_mutation(release):
    root, _ = release
    originals = copied_sources(root)
    damaged = "runtime_packs/llama_cpp_worker.py"
    (root / binding.PACKAGE_PATH / damaged).write_bytes(originals[damaged] + b"\n")
    with pytest.raises(ValueError, match="Original source differs"):
        binding.prepare_release_sources(root)
    assert not (root / binding.BUNDLE_NAME).exists()
    assert (root / binding.PACKAGE_PATH / "generated_capabilities.py").read_bytes() == originals[
        "generated_capabilities.py"
    ]


def test_prepare_sources_refuses_stage_or_source_checkout(release, monkeypatch):
    root, _ = release
    copied_sources(root)
    monkeypatch.setattr(binding, "_engine_root", lambda: None)
    with pytest.raises(ValueError, match="copied packaged app"):
        binding.prepare_release_sources(root)


def test_prepare_sources_instruments_worker_package_copies(release):
    root, _ = release
    copied_sources(root)
    worker = root / "runtime-packs/oryn.llama-cpp/site-packages/oryn_engine"
    shutil.copytree(root / binding.PACKAGE_PATH, worker)
    records = binding.prepare_release_sources(root)
    assert len(records) == 8
    manifest(root, records)
    for record in records:
        assert binding.matches_build_input(root / record["path"], record["originalSha256"])


def test_helper_locates_shared_root_from_worker_package(tmp_path, monkeypatch):
    root = tmp_path / "engine"
    module = root / "runtime-packs/oryn.llama-cpp/site-packages/oryn_engine/helper.py"
    module.parent.mkdir(parents=True)
    module.touch()
    monkeypatch.setattr(binding, "__file__", str(module))
    assert binding._engine_root() == root


@pytest.mark.parametrize("damage", [None, "source", "backup"])
def test_source_translation_requires_signed_original_and_exact_reviewed_patch(release, damage):
    root, _ = release
    originals = copied_sources(root)
    records = binding.prepare_release_sources(root)
    record = records[0]
    path = root / record["path"]
    if damage == "source":
        path.write_bytes(path.read_bytes() + b"\nprint('unrelated code')\n")
        record["signedSha256"] = digest(path.read_bytes())
    if damage == "backup":
        backup = root / binding.BUNDLE_NAME / binding.ORIGINAL_SOURCES / record["path"]
        backup.write_bytes(originals["generated_capabilities.py"] + b"\n")
    manifest(root, records)
    assert binding.matches_build_input(path, record["originalSha256"]) is (damage is None)


def test_binding_cannot_authorize_arbitrary_python_source(release):
    root, _ = release
    path = root / binding.PACKAGE_PATH / "arbitrary.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"print('changed')")
    entry = {
        "path": path.relative_to(root).as_posix(),
        "originalSha256": digest(b"original"),
        "signedSha256": digest(path.read_bytes()),
    }
    manifest(root, [entry])
    assert not binding.matches_build_input(path, entry["originalSha256"])


def test_binding_cannot_follow_native_or_bundle_symlinks(release):
    root, _ = release
    path, entry = native(root)
    manifest(root, [entry])
    target = path.with_name("real.dylib")
    path.rename(target)
    path.symlink_to(target.name)
    assert not binding.matches_build_input(path, entry["originalSha256"])


def test_source_patch_rejects_changed_snippet_counts():
    original = (Path(binding.__file__).parent / "generated_capabilities.py").read_bytes()
    old = binding._SOURCE_PATCHES["generated_capabilities.py"][1][0]
    with pytest.raises(ValueError, match="Unexpected release source patch input"):
        binding.transformed_source("generated_capabilities.py", original + old.encode())
