"""Repository tokenizer metadata cannot supply native constructor file authority."""

import json

import pytest

from oryn_engine.model_metadata import metadata_file, validate_tokenizer_config


@pytest.mark.parametrize(
    "options",
    [
        {"vocab": "/dummy/unapproved-vocab.txt"},
        {"vocab": "../unapproved-vocab.txt"},
        {"merges": "/dummy/unapproved-merges.txt"},
        {"vocab_file": "/dummy/unapproved-vocab.txt"},
        {"tokenizer_file": "../unapproved.json"},
        {"fast_tokenizer_files": ["../unapproved.json"]},
        {"init_inputs": ["/dummy/unapproved-vocab.txt"]},
        {"tokenizer_object": {"model": "unapproved"}},
        {"nested": {"vocab": "/dummy/unapproved-vocab.txt"}},
        {"sp_model_kwargs": {"model_file": "/dummy/unapproved.model"}},
        {"auto_map": {"AutoTokenizer": "custom.Tokenizer"}},
        {"trust_remote_code": True},
    ],
)
def test_tokenizer_metadata_rejects_file_authority(tmp_path, options):
    with pytest.raises(ValueError):
        validate_tokenizer_config(options)
    path = tmp_path / "tokenizer_config.json"
    path.write_text(json.dumps(options))
    with pytest.raises(ValueError):
        metadata_file(path)


@pytest.mark.parametrize(
    "options",
    [
        {"tokenizer_class": "BertTokenizer", "do_lower_case": True, "init_inputs": []},
        {"vocab": {"[UNK]": 0, "vocab_file": 1, "ordinary": 2}},
        {"vocab": {"[UNK]": 0, "size": 3000, "model_file": 1, "auto_map": 2}},
        {"vocab": [["<unk>", 0.0], ["ordinary", -1.0]]},
        {"merges": ["a b", ["c", "d"]]},
        {"vocab_file": None, "tokenizer_file": None},
    ],
)
def test_tokenizer_metadata_preserves_literal_native_options(tmp_path, options):
    validate_tokenizer_config(options)
    path = tmp_path / "tokenizer_config.json"
    path.write_text(json.dumps(options))
    assert metadata_file(path) == options


def test_numeric_loader_rejects_path_metadata_before_native_construction(tmp_path, monkeypatch):
    from oryn_engine import numeric_archive_tasks as family

    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"vocab": "/dummy/unapproved-vocab.txt"})
    )
    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("Native model construction must not start")

    monkeypatch.setattr(family, "_load_native", forbidden)
    with pytest.raises(ValueError):
        family.load_components(tmp_path, "fill-mask", "cpu")
    assert calls == []


def test_hub_json_boundary_rejects_path_metadata_without_network(monkeypatch):
    from oryn_engine.models import HuggingFaceClient, ModelStoreError

    client = object.__new__(HuggingFaceClient)
    monkeypatch.setattr(
        client, "_bounded_bytes", lambda *_args: b'{"vocab":"/dummy/unapproved-vocab.txt"}'
    )
    with pytest.raises(ModelStoreError) as error:
        client.read_json("dummy/model", "0" * 40, "tokenizer_config.json")
    assert error.value.code == "MODEL_METADATA_INVALID"


@pytest.mark.parametrize("family", ["molparser", "videollama3", "bolding"])
@pytest.mark.parametrize(
    "override",
    [{}, {"vocab_file": "../unapproved.json"}, {"auto_map": {"AutoTokenizer": "other.Code"}}],
)
def test_hub_json_preserves_only_closed_native_tokenizer_declarations(
    monkeypatch, family, override
):
    from bolding_fixture import original

    from oryn_engine import molparser_tasks, videollama3_tasks
    from oryn_engine.models import HuggingFaceClient, ModelStoreError

    reader = {
        "molparser": molparser_tasks.template,
        "videollama3": videollama3_tasks.template,
        "bolding": original,
    }[family]
    config = reader("tokenizer_config.json") | override
    client = object.__new__(HuggingFaceClient)
    monkeypatch.setattr(client, "_bounded_bytes", lambda *_args: json.dumps(config).encode())
    if override:
        with pytest.raises(ModelStoreError) as error:
            client.read_json("dummy/model", "0" * 40, "tokenizer_config.json")
        assert error.value.code == "MODEL_METADATA_INVALID"
    else:
        assert client.read_json("dummy/model", "0" * 40, "tokenizer_config.json") == config
        # Generic constructors still reject these declarations.
        with pytest.raises(ValueError, match="Repository code is prohibited"):
            validate_tokenizer_config(config)
