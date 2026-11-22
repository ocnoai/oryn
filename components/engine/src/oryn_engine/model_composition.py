"""Pinned, data-only dependency graphs for original adapter checkpoints.

The catalog supplies reviewed identities; the Hub projection preserves the
original model identity while resolving each file against its own source pin.
No repository metadata can add a dependency or select executable code.
"""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

ADAPTER_DIRECTORY = "oryn_adapter"


@lru_cache(maxsize=1)
def _catalog():
    return json.loads((Path(__file__).parent / "data/model-compositions.json").read_text())


def binding(repository, revision):
    value = next(
        (
            x
            for x in _catalog()
            if (x["model"]["repository"], x["model"]["revision"]) == (repository, revision)
        ),
        None,
    )
    return copy.deepcopy(value)


def validate_binding(value):
    if not isinstance(value, dict) or not isinstance(value.get("model"), dict):
        raise ValueError("Composition identity is invalid")
    model = value["model"]
    expected = binding(model.get("repository"), model.get("revision"))
    if expected is None or value != expected:
        raise ValueError("Composition differs from its reviewed dependency pins")
    return expected


def adapter_name(value):
    return f"{ADAPTER_DIRECTORY}/{value['adapterFile']}"


def source_ref(value, name):
    """Map one virtual file to one original, immutable repository file."""
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or PurePosixPath(name).is_absolute()
        or any(p in {"", ".", ".."} for p in name.split("/"))
    ):
        raise ValueError("Composition file name is invalid")
    if name == adapter_name(value):
        return value["model"], value["adapterFile"]
    if name.startswith(ADAPTER_DIRECTORY + "/"):
        raise ValueError("Composition adapter file is not selected")
    if name in value.get("overrides", []):
        return value["model"], name
    return value["base"], name


def record_binding(record):
    value = record.get("runtimePack", {}).get("trustedMetadata", {}).get("composition")
    if value is None:
        if binding(
            record.get("model", {}).get("repository"), record.get("model", {}).get("revision")
        ):
            raise ValueError("Installed adapter lost its dependency binding")
        return None
    value = validate_binding(value)
    if any(record.get("model", {}).get(k) != v for k, v in value["model"].items()):
        raise ValueError("Installed adapter identity changed")
    return value


def generation_defaults(value, index):
    from oryn_engine.qwen_image_21 import generation_defaults as qwen_defaults
    from oryn_engine.runtime_packs.generation_profiles import generation_profile

    result = (
        qwen_defaults(index)
        if value["family"] == "QwenImage21Pipeline"
        else generation_profile(task_contract="image-editing-v1")
    )
    result.update(profileId="runtime.diffusers.pinned-lora", provenance="adapter-derived")
    defaults = value.get("sampling", {})
    if "steps" in defaults:
        result["parameters"]["steps"].update(
            default=defaults["steps"],
            minimum=defaults["steps"],
            maximum=defaults["steps"],
            editable=False,
        )
    if "guidanceScale" in defaults:
        result["parameters"]["guidanceScale"].update(
            default=defaults["guidanceScale"],
            minimum=defaults["guidanceScale"],
            maximum=defaults["guidanceScale"],
            editable=False,
        )
    return result


class CompositionHub:
    """Transparent source routing, with one explicit pinned base per adapter.

    Metadata snapshots provide a separate, digest-bound child snapshot. Ordinary
    Hub clients use their existing bounded readers and verified download cache.
    """

    def __init__(self, client):
        self.client = client

    def __getattr__(self, name):
        return getattr(self.client, name)

    @property
    def current_provider(self):
        return self.client.current_provider

    @current_provider.setter
    def current_provider(self, value):
        self.client.current_provider = value

    def _base_client(self, value):
        dependency = getattr(self.client, "dependency_snapshot", None)
        return dependency(value["base"]) if callable(dependency) else self.client

    def _source(self, repository, revision, filename):
        value = binding(repository, revision)
        if value is None:
            return self.client, repository, revision, filename
        ref, name = source_ref(value, filename)
        client = self.client if ref == value["model"] else self._base_client(value)
        return client, ref["repository"], ref["revision"], name

    def model_info(self, repo_id, revision, files_metadata=True):
        original = self.client.model_info(repo_id, revision, files_metadata=files_metadata)
        value = binding(repo_id, original.sha)
        if value is None:
            return original
        from oryn_engine.models import ModelStoreError

        try:
            return self._composed_info(original, value)
        except (ValueError, KeyError, TypeError) as exc:
            raise ModelStoreError(
                "MODEL_METADATA_INVALID", "Pinned adapter dependencies are invalid."
            ) from exc

    def _composed_info(self, original, value):
        base = self._base_client(value)
        ref = value["base"]
        info = base.model_info(ref["repository"], ref["revision"], files_metadata=True)
        if info.sha != ref["revision"] or info.config.get("_class_name") != value["family"]:
            raise ValueError("Composition base identity changed")
        original_files = {x.rfilename: x for x in original.siblings}
        required = [value["adapterFile"], *value.get("overrides", [])]
        if any(name not in original_files for name in required):
            raise ValueError("Composition original adapter assets are missing")
        if any(x.rfilename.split("/", 1)[0] == ADAPTER_DIRECTORY for x in info.siblings):
            raise ValueError("Composition reserved directory collision")
        siblings = [
            copy.copy(x) for x in info.siblings if x.rfilename not in value.get("overrides", [])
        ]
        for name in required:
            item = copy.copy(original_files[name])
            item.rfilename = adapter_name(value) if name == value["adapterFile"] else name
            siblings.append(item)
        for item in siblings:
            digest = getattr(getattr(item, "lfs", None), "sha256", None) or getattr(
                item, "blob_id", None
            )
            if not isinstance(digest, str) or len(digest) not in {40, 64}:
                raise ValueError("Composition files require publisher content hashes")
            item.lfs = SimpleNamespace(sha256=digest)
        # Every original adapter tensor must target the complete base transformer.
        from oryn_engine.diffusion_lora import validate_structure

        structure = self.client.read_tensor_structure(
            value["model"]["repository"], value["model"]["revision"], value["adapterFile"]
        )
        config = base.read_json(ref["repository"], ref["revision"], "transformer/config.json")
        validate_structure(structure, config)
        return SimpleNamespace(
            sha=original.sha,
            config=copy.deepcopy(info.config),
            siblings=siblings,
            pipeline_tag=original.pipeline_tag,
            library_name="diffusers",
            card_data=getattr(original, "card_data", {}),
            gated=bool(getattr(original, "gated", False) or getattr(info, "gated", False)),
            private=bool(getattr(original, "private", False) or getattr(info, "private", False)),
            gguf=None,
        )

    def has_access(self, repo_id):
        candidates = [x for x in _catalog() if x["model"]["repository"] == repo_id]
        if not candidates:
            return self.client.has_access(repo_id)
        return self.client.has_access(repo_id) and all(
            self._base_client(x).has_access(x["base"]["repository"]) for x in candidates
        )

    def read_json(self, repository, revision, filename):
        client, repo, rev, name = self._source(repository, revision, filename)
        return client.read_json(repo, rev, name)

    def read_tensor_metadata(self, repository, revision, filename):
        client, repo, rev, name = self._source(repository, revision, filename)
        return client.read_tensor_metadata(repo, rev, name)

    def read_tensor_structure(self, repository, revision, filename):
        client, repo, rev, name = self._source(repository, revision, filename)
        return client.read_tensor_structure(repo, rev, name)

    def cached_file(self, repository, filename, revision, cache_dir):
        client, repo, rev, name = self._source(repository, revision, filename)
        method = getattr(client, "cached_file", None)
        return method(repo, name, rev, cache_dir) if callable(method) else None

    def download(self, repository, filename, revision, cache_dir, progress=None):
        client, repo, rev, name = self._source(repository, revision, filename)
        return client.download(repo, name, rev, cache_dir, progress)
