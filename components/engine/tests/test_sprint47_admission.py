from __future__ import annotations

from pathlib import Path

import pytest
from test_models import FakeHub, model

from oryn_engine.capability_sources import framework_registry
from oryn_engine.jobs import JobExecutionError
from oryn_engine.model_inspection import inspect_descriptor
from oryn_engine.model_layouts import assess_layout
from oryn_engine.models import ModelStore, ModelStoreError
from oryn_engine.runners import (
    RunnerRegistry,
    _bounded_chat_image,
    _chat_input,
    _verify_installed_files,
)
from oryn_engine.runtime_packs.mlx_lm_worker import validate_config


def test_runtime_build_change_invalidates_installed_admission(tmp_path, monkeypatch):
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    exact = store.install("op_build_identity", model())["model"]
    monkeypatch.setattr("oryn_engine.models.runtime_build_digest", lambda: "changed-build")
    restarted = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    # Inspection can obtain a fresh plan; the old installation remains invalid
    # until explicit preparation binds that plan to the new build.
    profile = restarted.inspect(exact)
    assert profile["executable"] is True
    assert profile["runtimePreparationRequired"] is True
    with pytest.raises(ModelStoreError) as error:
        restarted.installed_record(exact)
    assert error.value.code == "RUNTIME_BUILD_CHANGED"


def test_unknown_revision_remains_unverified_after_install_and_restart(tmp_path):
    first = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    exact = first.install("op_fresh", model())["model"]
    second = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    profile = second.inspect(exact)
    assert profile["loadDisposition"] == "trust-required"
    assert profile["eligibility"]["conformance"]["state"] == "untested"


def test_installed_file_changes_block_load_before_any_import(tmp_path):
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    exact = store.install("op_integrity", model())["model"]
    record = store.installed_record(exact)
    _verify_installed_files(record)
    file = record["files"][0]
    (Path(record["cacheRoot"]) / file["cache_path"]).write_bytes(b"x" * file["size"])
    with pytest.raises(JobExecutionError, match="Installed model files changed"):
        RunnerRegistry().load(record)


def test_download_hash_mismatch_blocks_install(tmp_path):
    class TamperedHub(FakeHub):
        def download(self, *args, **kwargs):
            path = Path(super().download(*args, **kwargs))
            path.write_bytes(b"x" * path.stat().st_size)
            return str(path)

    store = ModelStore(tmp_path, hub=TamperedHub(), runtime_available=True)
    with pytest.raises(ModelStoreError) as failure:
        store.install("op_tampered", model())
    assert failure.value.code == "MODEL_INTEGRITY_FAILED"
    assert store.inventory()["models"] == []


def test_static_and_live_inspection_differ_only_on_device_admission():
    facts = dict(
        repository="org/new-checkpoint",
        revision="a" * 40,
        files=["config.json", "model.safetensors", "tokenizer.json"],
        config={"architectures": ["BertModel"]},
        task="feature-extraction",
        registry=framework_registry(),
        access="available",
        estimated_memory=1024,
    )
    static = inspect_descriptor(**facts)
    live = inspect_descriptor(**facts, available_memory=512)
    for name in ("access", "artifacts", "runtime", "task", "conformance"):
        assert static.eligibility.axes[name] == live.eligibility.axes[name]
    assert static.eligibility.disposition is None
    assert live.eligibility.disposition == "not-admissible-on-this-device"


def test_mlx_quantization_is_closed_data_and_cannot_choose_code():
    good = {"model_type": "qwen2", "quantization": {"bits": 4, "group_size": 64}}
    validate_config(good, {"qwen2"})
    for config in [
        good | {"auto_map": {"AutoModel": "repo.Model"}},
        good | {"quantization": {"bits": 4, "group_size": 64, "callable": "repo.fn"}},
        good | {"model_type": "../../repo/code"},
    ]:
        with pytest.raises(ValueError):
            validate_config(config, {"qwen2"})


def test_image_reference_cannot_be_url_path_or_used_by_text_model():
    base = {"messages": [{"role": "user", "content": "Describe it."}]}
    valid = base | {"imageArtifactId": "artifact_" + "x" * 24}
    assert _chat_input(valid, allow_image=True)["imageArtifactId"] == valid["imageArtifactId"]
    for value, allow in [
        (valid, False),
        (base | {"imageArtifactId": "https://example.com/x"}, True),
        (base | {"imageArtifactId": "/etc/passwd"}, True),
    ]:
        with pytest.raises(JobExecutionError):
            _chat_input(value, allow_image=allow)


def test_image_byte_bound_is_enforced_before_decode(tmp_path):
    path = tmp_path / "input.png"
    path.write_bytes(b"invalid image")
    binding = {
        "artifactId": "artifact_" + "x" * 24,
        "internalPath": str(path),
        "mimeType": "image/png",
        "sizeBytes": 21 * 1024**2,
    }
    with pytest.raises(JobExecutionError):
        _bounded_chat_image((binding,), binding["artifactId"])


def test_diffusion_requires_every_safe_component_and_excludes_duplicate_exports():
    config = {
        "_class_name": "StableDiffusionPipeline",
        "unet": ["diffusers", "UNet2DConditionModel"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
    }
    files = [
        "model_index.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "tokenizer/vocab.json",
        "checkpoint.safetensors",
        "tools/install.py",
    ]
    layout = assess_layout(files, config, "image-to-image", source_kind="diffusers")
    assert not layout.artifact_reasons
    assert "checkpoint.safetensors" not in layout.selected_files
    assert not any(".fp16." in name or name.endswith(".py") for name in layout.selected_files)
    missing = assess_layout(
        [x for x in files if not x.startswith("unet/")],
        config,
        "image-to-image",
        source_kind="diffusers",
    )
    assert "MODEL_SUBSET_INCOMPLETE" in missing.artifact_reasons
    hostile = config | {"unet": ["repo_module", "CustomModel"]}
    assert (
        "CUSTOM_CODE_REQUIRED"
        in assess_layout(files, hostile, "image-to-image", source_kind="diffusers").runtime_reasons
    )


def test_worker_view_contains_only_selected_files_and_cleans_up(tmp_path):
    from oryn_engine.runners import _selected_model_view

    cache = tmp_path / "cache"
    root = cache / "models--org--model" / "snapshots" / ("a" * 40)
    root.mkdir(parents=True)
    blob = cache / "weight-blob"
    blob.write_bytes(b"safe weights")
    (root / "model.safetensors").symlink_to(blob)
    (root / "unused.py").write_text("raise RuntimeError()")
    record = {"cacheRoot": str(cache), "files": [{"name": "model.safetensors"}]}
    with _selected_model_view(record, root) as view:
        assert (view / "model.safetensors").samefile(blob)
        assert not (view / "model.safetensors").is_symlink()
        assert not (view / "unused.py").exists()
    assert not view.exists()
    assert blob.exists()
    outside = tmp_path / "outside"
    outside.write_bytes(b"private")
    (root / "model.safetensors").unlink()
    (root / "model.safetensors").symlink_to(outside)
    with pytest.raises(JobExecutionError), _selected_model_view(record, root):
        pytest.fail("escaped source admitted")
