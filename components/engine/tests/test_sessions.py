from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from oryn_engine.models import DownloadProgress, ModelStore, ModelStoreError
from oryn_engine.residency import ResidencyManager
from oryn_engine.runtime.resources import ResourceManager
from oryn_engine.sessions import ModelSessionRegistry


class FakeHub:
    def model_info(self, repo_id: str, revision: str, files_metadata: bool = True) -> Any:
        del revision, files_metadata
        digest = (repo_id.encode().hex() + "a" * 40)[:40]
        return SimpleNamespace(
            sha=digest,
            config={"architectures": ["Qwen2ForCausalLM"], "chat_template": "x"},
            pipeline_tag="text-generation",
            siblings=[
                SimpleNamespace(rfilename="config.json", size=0, lfs=None),
                SimpleNamespace(rfilename="tokenizer.json", size=0, lfs=None),
                SimpleNamespace(
                    rfilename="model.safetensors",
                    size=100,
                    lfs=None,
                ),
            ],
        )

    def download(
        self,
        repo_id: str,
        filename: str,
        revision: str,
        cache_dir: str,
        progress: DownloadProgress | None = None,
    ) -> str:
        target = Path(cache_dir) / revision / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        size = next(
            item.size
            for item in self.model_info(repo_id, revision).siblings
            if item.rfilename == filename
        )
        target.write_bytes(b"x" * size)
        if progress is not None:
            progress(4, 4)
        return str(target)


def install(store: ModelStore, repository: str, operation: str) -> dict[str, str]:
    return store.install(
        operation,
        {"provider": "huggingface", "repository": repository, "revision": "main"},
    )["model"]


def test_attach_is_idempotent_conflict_safe_and_release_keeps_residency(tmp_path: Path) -> None:
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    first = install(store, "org/first", "op_install_first")
    second = install(store, "org/second", "op_install_second")
    residency = ResidencyManager(store, ResourceManager())
    sessions = ModelSessionRegistry(store, residency)

    attached = sessions.attach("session_example", first, ensure_loaded=True)
    duplicate = sessions.attach("session_example", first, ensure_loaded=True)
    assert duplicate == attached
    assert attached["model"] == first
    assert attached["residencyState"] == "loaded"
    assert store.inventory()["models"][0]["activeSessions"] == 1

    with pytest.raises(ModelStoreError) as conflict:
        sessions.attach("session_example", second, ensure_loaded=True)
    assert conflict.value.code == "SESSION_ID_CONFLICT"

    assert sessions.release("session_example") == {"released": True}
    assert sessions.release("session_example") == {"released": False}
    inventory = store.inventory()["models"][0]
    assert inventory["activeSessions"] == 0
    assert inventory["residencyState"] == "loaded"


def test_attach_without_loading_keeps_session_model_pinned(tmp_path: Path) -> None:
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    model = install(store, "org/pinned", "op_install_pinned")
    residency = ResidencyManager(store, ResourceManager())
    sessions = ModelSessionRegistry(store, residency)

    attached = sessions.attach("session_pinned", model, ensure_loaded=False)
    assert attached["model"] == model
    assert attached["residencyState"] == "unloaded"
    assert store.inventory()["models"][0]["activeSessions"] == 1


def test_unload_keeps_active_session_valid_and_installation_available(tmp_path: Path) -> None:
    store = ModelStore(tmp_path, hub=FakeHub(), runtime_available=True)
    model = install(store, "org/unload-session", "op_install_unload_session")
    residency = ResidencyManager(store, ResourceManager())
    sessions = ModelSessionRegistry(store, residency)
    sessions.attach("session_unload", model, ensure_loaded=True)

    assert residency.unload("unload_active_session", model)["state"] == "unloaded"
    assert sessions.get("session_unload").model == model
    inventory = store.inventory()["models"][0]
    assert inventory["installationState"] == "installed"
    assert inventory["residencyState"] == "unloaded"
    assert inventory["activeSessions"] == 1
