import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from oryn_engine.generated_capabilities import normalized_transformers_config
from oryn_engine.model_metadata import validate_model_config


def originals():
    root = Path(__file__).parent / "fixtures/renderrank"
    return {p.name: json.loads(p.read_text()) for p in root.glob("*.json")}


@pytest.mark.parametrize(
    "task", ["feature-extraction", "image-text-to-text", "visual-document-retrieval"]
)
def test_text_alias_does_not_enable_other_modalities_or_tasks(task):
    config = normalized_transformers_config(originals()["config.json"], task)
    with pytest.raises(ValueError):
        validate_model_config(config)


@pytest.mark.parametrize(
    "mapping",
    [
        {"AutoModel": "unreviewed.RenderRank"},
        {"AutoModel": "modeling_renderrank.RenderRank", "AutoTokenizer": "unreviewed.Code"},
    ],
)
def test_only_the_exact_reviewed_alias_is_removed(mapping):
    config = originals()["config.json"] | {"auto_map": mapping}
    with pytest.raises(ValueError):
        validate_model_config(normalized_transformers_config(config, "text-ranking"))


def test_original_text_mode_uses_native_reranker_without_downloading_weights(tmp_path):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore
    from oryn_engine.runtime_packs.execution import PackRuntime

    data = originals()

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            return data[filename]

    hub = Hub(
        config=data["config.json"],
        pipeline_tag="text-ranking",
        files=[(n, 100, "b" * 64) for n in set(data) | {"model.safetensors", "tokenizer.json"}],
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    result = store.inspect(
        {"provider": "huggingface", "repository": "test/renderrank", "revision": "main"}
    )
    assert result["executable"], result
    assert result["taskProfile"]["contract"]["id"] == "reranking-v1"
    assert result["taskProfile"]["inputModalities"] == ["text"]
    assert not hub.downloaded
