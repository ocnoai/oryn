"""Adversarial data tests; none of the candidate Python fixtures is executed."""

import ast
import copy
import tokenize
from types import SimpleNamespace

import pytest

from oryn_engine import reviewed_implementations as catalog
from oryn_engine import source_inspector_worker as parser


def facts(source):
    return parser.inspect_source(source.encode() if isinstance(source, str) else source)


@pytest.mark.parametrize(
    "left,right",
    [
        ("x=1\ny=2\n", "# ordinary comment\nx = 1; y = 2 # another comment\n"),
        ("def f():\n    return 1\n", "def f():\n\treturn 1\n"),
        ("x='hello'\n", 'x="hello"\n'),
        ("x=1\n", "x=1\r\n"),
    ],
)
def test_formatting_equivalence(left, right):
    a, b = facts(left), facts(right)
    assert a["rawSha256"] != b["rawSha256"]
    assert a["structuralSha256"] == b["structuralSha256"]


@pytest.mark.parametrize(
    "left,right",
    [
        ("x=1", "x=2"),
        ("x=a+b", "x=a-b"),
        ("import os", "import sys"),
        ("from .a import B", "from .b import B"),
        ("@first\ndef f(): pass", "@second\ndef f(): pass"),
        ("x: int=1", "x: float=1"),
        ("x=1 # type: int", "x=1 # type: float"),
        ("def f():\n 'first'\n pass", "def f():\n 'second'\n pass"),
        ("if flag: a()", "if flag: b()"),
        ("x='a b'", "x='ab'"),
        ("x=0.0", "x=-0.0"),
    ],
)
def test_meaningful_changes_are_not_aliases(left, right):
    assert facts(left)["structuralSha256"] != facts(right)["structuralSha256"]


@pytest.mark.parametrize(
    "directive",
    ["type: ignore", "noqa", "pylint: disable=all", "mypy: ignore-errors", "pragma: jit"],
)
def test_source_sensitive_comments_are_exact_only(directive):
    assert facts("x=1 # " + directive)["structuralSha256"] is None


def test_introspection_example_is_recognition_not_semantic_proof():
    a = "import inspect\nx=inspect.currentframe().f_lineno\n"
    b = "# moved\n\n" + a
    assert facts(a)["structuralSha256"] == facts(b)["structuralSha256"]
    assert ast.parse(a).body[-1].lineno != ast.parse(b).body[-1].lineno
    assert catalog.POLICY == "bundled-only-no-upstream-kernels-v1"


@pytest.mark.parametrize(
    "source",
    [b"", b"\xff", b"# coding: latin-1\nx='a'", b"def f(:", b"x" * (parser.MAX_SOURCE_BYTES + 1)],
)
def test_invalid_or_excessive_source(source):
    with pytest.raises((ValueError, SyntaxError, UnicodeError, tokenize.TokenError)):
        facts(source)


def test_ast_depth_and_unknown_schema_fail_closed(monkeypatch):
    with pytest.raises(parser.InspectionError):
        facts("x=" + "[" * 110 + "1" + "]" * 110)
    monkeypatch.setitem(parser.AST_SCHEMA, "Constant", ["value"])
    with pytest.raises(parser.InspectionError):
        facts("x=1")


@pytest.fixture
def bundle():
    sources = {
        "configuration.py": "class Config: pass\n",
        "model.py": (
            "from .configuration import Config\nfrom .conv import conv\nclass Model: pass\n"
        ),
        "conv.py": "def conv(x): return x + 1\n",
    }
    modules = {name: facts(source) for name, source in sources.items()}
    profile = {
        "id": "oryn.test.v1",
        "version": 1,
        "normalizer": parser.NORMALIZER,
        "adapter": "oryn.lfm3.torch.v1",
        "state": "active",
        "sourcePolicy": catalog.POLICY,
        "representations": ["bf16-cpu"],
        "references": [],
        "modules": {
            name[:-3]: {
                "raw": f["rawSha256"],
                "structural": f["structuralSha256"],
                "exports": f["exports"],
            }
            for name, f in modules.items()
        },
        "bindings": {
            "AutoConfig": ["configuration", "Config"],
            "AutoModelForCausalLM": ["model", "Model"],
        },
    }
    auto_map = {"AutoConfig": "configuration.Config", "AutoModelForCausalLM": "model.Model"}
    return sources, modules, profile, auto_map


def test_indexes_use_contents_and_validate_roles(bundle):
    _, modules, profile, auto_map = bundle
    index = catalog.ImplementationIndex([profile], identity="f" * 64)
    assert index.match(modules, auto_map)[1] == "raw"
    renamed = {("renamed.py" if name == "model.py" else name): f for name, f in modules.items()}
    binding = auto_map | {"AutoModelForCausalLM": "renamed.Model"}
    assert index.match(renamed, binding)[1] == "raw"
    assert index.match(renamed, auto_map) is None
    alias = dict(modules) | {
        "model.py": facts(
            "# ordinary comment\nfrom .configuration import Config\n"
            "from .conv import conv\nclass Model: pass\n"
        )
    }
    assert index.match(alias, auto_map)[1] == "structural"
    assert index.match({"conv.py": modules["conv.py"]}, auto_map) is None
    changed = dict(modules) | {"conv.py": facts("def conv(x): return x + 2\n")}
    assert index.match(changed, auto_map) is None
    assert catalog.content_collection(["a" * 64, "a" * 64]) != catalog.content_collection(
        ["a" * 64]
    )


def test_unknown_profiles_revocation_and_ambiguity(bundle):
    _, modules, profile, auto_map = bundle
    for patch in ({"normalizer": "unknown"}, {"adapter": "author.module"}, {"extra": True}):
        with pytest.raises(parser.InspectionError):
            catalog.ImplementationIndex([profile | patch], identity="f" * 64)
    revoked = catalog.ImplementationIndex([profile | {"state": "revoked"}], identity="f" * 64)
    assert revoked.match(modules, auto_map) is None
    index = catalog.ImplementationIndex(
        [profile, profile | {"id": "oryn.test.alias.v1"}], identity="f" * 64
    )
    assert index.match(modules, auto_map) is None


def test_transitive_closure_and_evidence_binding(bundle, monkeypatch):
    sources, _, profile, auto_map = bundle
    index = catalog.ImplementationIndex([profile], identity="f" * 64)
    monkeypatch.setattr(catalog, "installed_index", lambda: index)
    files = [
        SimpleNamespace(name=name, size=len(s.encode()), blob_id=None)
        for name, s in sources.items()
    ]

    def read(name, limit):
        return sources[name].encode()

    def parse(raw, cancellation=None):
        return parser.inspect_source(raw)

    evidence = catalog.inspect_repository({"auto_map": auto_map}, files, read, parser=parse)
    evidence["admitted"] = True
    assert catalog.validate_evidence(evidence) == profile
    bad = copy.deepcopy(evidence)
    bad["files"][0]["rawSha256"] = "0" * 64
    with pytest.raises(parser.InspectionError):
        catalog.validate_evidence(bad)
    with pytest.raises(parser.InspectionError):
        catalog.inspect_repository(
            {"auto_map": {"AutoConfig": "other/repo--module.Config"}}, files, read, parser=parse
        )
    renamed = dict(sources)
    renamed["renamed.py"] = renamed.pop("configuration.py")
    renamed_files = [
        SimpleNamespace(name=n, size=len(s.encode()), blob_id=None) for n, s in renamed.items()
    ]
    with pytest.raises(parser.InspectionError):
        catalog.inspect_repository(
            {"auto_map": auto_map | {"AutoConfig": "renamed.Config"}},
            renamed_files,
            read,
            parser=parse,
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/A/B",
        "https://huggingface.co.evil/A/B",
        "http://huggingface.co/A/B",
        "https://user@huggingface.co/A/B",
        "https://huggingface.co/C/D",
        "https://huggingface.co/A/B?url=other",
    ],
)
def test_hf_entry_context_cannot_be_forged_from_other_sites(url):
    with pytest.raises(parser.InspectionError):
        catalog.validate_hf_context({"sourceURL": url, "tabId": "tab", "documentId": "doc"}, "A/B")


def test_exact_only_profile_has_no_structural_alias(bundle):
    _, modules, profile, auto_map = bundle
    for record in profile["modules"].values():
        record["structural"] = None
    index = catalog.ImplementationIndex([profile], identity="f" * 64)
    assert index.match(modules, auto_map)[1] == "raw"
    assert not index.structural
    alias = dict(modules) | {"conv.py": facts("# ordinary comment\ndef conv(x): return x + 1\n")}
    assert index.match(alias, auto_map) is None


@pytest.mark.parametrize("digests", [[None], ["a" * 64, None], [True], []])
def test_malformed_content_collections_fail_closed(digests):
    with pytest.raises(parser.InspectionError):
        catalog.content_collection(digests)


def test_signed_installed_catalog_and_tamper_rejection(monkeypatch):
    from oryn_engine import generated_capabilities

    catalog._cached_index.cache_clear()
    index = catalog.installed_index()
    assert len(index.profiles) == 3
    assert index.profiles["oryn.lfm3.source-b.v1"]["state"] == "recognition-only"
    artifact = copy.deepcopy(generated_capabilities.generated_artifact())
    artifact["buildInputs"]["reviewedImplementations"]["catalogSha256"] = "0" * 64
    monkeypatch.setattr(generated_capabilities, "generated_artifact", lambda: artifact)
    catalog._cached_index.cache_clear()
    with pytest.raises(parser.InspectionError):
        catalog.installed_index()
    catalog._cached_index.cache_clear()


def test_broker_has_separate_source_budget_and_no_path_authority(tmp_path):
    from oryn_engine.runtime_packs.broker import SOURCE_INSPECTOR, WorkerBroker

    broker = object.__new__(WorkerBroker)
    broker.token = "dummy"
    request = {
        "token": "dummy",
        "packId": SOURCE_INSPECTOR,
        "version": parser.NORMALIZER,
        "command": "inspect-source",
        "payload": {"source": "eD0x"},
        "timeoutSeconds": 5,
    }
    assert broker._validate(request) == request
    for patch in (
        {"command": "generate"},
        {"version": "unknown"},
        {"payload": {"source": "eD0x", "modelRoot": str(tmp_path)}},
    ):
        with pytest.raises(ValueError):
            broker._validate(request | patch)


def test_private_wire_rejects_website_proofs_and_ordinary_method_flags():
    from oryn_protocol import ProtocolSchemaStore

    schemas = ProtocolSchemaStore()
    request = {
        "v": 1,
        "type": "request",
        "id": "req_source",
        "method": "model.source.inspect",
        "params": {
            "model": {"provider": "huggingface", "repository": "A/B", "revision": "main"},
            "sourceContext": {
                "sourceURL": "https://huggingface.co/A/B",
                "tabId": "tab",
                "documentId": "doc",
            },
        },
    }
    assert schemas.validator("modelSourceInspectRequest").is_valid(request)
    assert not schemas.validator("modelInspectRequest").is_valid(
        request | {"method": "model.inspect"}
    )
    request["params"]["sourceContext"]["rawSha256"] = "a" * 64
    assert not schemas.validator("modelSourceInspectRequest").is_valid(request)
    assert not schemas.validator("sourceRecognition").is_valid(
        {"state": "recognized", "source": "arbitrary Python"}
    )


def test_only_trusted_hf_inspection_can_admit_reviewed_graph(tmp_path, monkeypatch):
    from test_models import SHA, FakeHub
    from test_native_lfm3 import config

    from oryn_engine.models import ModelStore
    from oryn_engine.runtime_routing import canonical_digest

    index = catalog.installed_index()
    profile = index.profiles["oryn.lfm3.source-a.v1"]
    names = {
        "configuration": "configuration_lfm3.py",
        "model": "modeling_lfm3.py",
        "convolution": "lfm3_conv.py",
    }
    evidence = {
        "state": "recognized",
        "match": "exact",
        "profileId": profile["id"],
        "profileVersion": 1,
        "normalizer": parser.NORMALIZER,
        "catalogDigest": index.identity,
        "adapterId": profile["adapter"],
        "sourcePolicy": catalog.POLICY,
        "sourceDigest": catalog.content_collection(m["raw"] for m in profile["modules"].values()),
        "files": [
            {"name": names[role], "rawSha256": m["raw"], "structuralSha256": m["structural"]}
            for role, m in profile["modules"].items()
        ],
    }
    raw_config = config() | {
        "auto_map": {
            key: f"{names[role][:-3]}.{cls}" for key, (role, cls) in profile["bindings"].items()
        }
    }
    hub = FakeHub(config=raw_config, pipeline_tag="text-generation")
    hub.source_model_info = hub.model_info
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._runtime_planning_state_provider = lambda: ("balanced", 32 * 1024**3)
    monkeypatch.setenv("ORYN_PACK_SANDBOX_PROFILE", "/fixed/trusted/runtime-pack.sb")
    reference = {"provider": "huggingface", "repository": "A/B", "revision": SHA}
    assert "CUSTOM_CODE_REQUIRED" in store.inspect(reference)["blockingReasons"]
    monkeypatch.setattr(catalog, "inspect_repository", lambda *a, **k: copy.deepcopy(evidence))
    context = {"sourceURL": "https://huggingface.co/A/B", "tabId": "tab", "documentId": "doc"}
    result = store.inspect(reference, source_context=context)
    assert result["sourceRecognition"]["admitted"] is True
    assert result["sourceRecognition"]["configDigest"] == canonical_digest(raw_config)
    assert result["admissionScope"] == "huggingface-page"
    assert result["executableBackends"] == ["cpu"]
    assert result["loadDisposition"] == "trust-required"
    assert not any(f["name"].endswith(".py") for f in result["selectedFiles"])
    # The new route cannot leak back into an ordinary SDK inspection.
    assert "CUSTOM_CODE_REQUIRED" in store.inspect(reference)["blockingReasons"]
    hub.config = raw_config | {
        "quantization_config": {"bits": 4, "mode": "affine", "group_size": 64}
    }
    unsupported = store.inspect(reference, source_context=context)
    assert not unsupported["executable"]
    assert "REVIEWED_REPRESENTATION_UNSUPPORTED" in unsupported["blockingReasons"]


def test_reviewed_profile_cannot_be_disabled_by_environment_flags(tmp_path, monkeypatch):
    import sys
    from pathlib import Path

    from oryn_engine.runtime_packs.execution import PackRuntime, StreamingPackProcess

    monkeypatch.setenv("ORYN_SANDBOX_REQUIRED", "0")
    monkeypatch.setenv("ORYN_PACK_SANDBOX_PROFILE", "/permissive/override.sb")
    runtime = PackRuntime(Path(sys.executable), tmp_path, inherit_base_runtime=True)
    command = StreamingPackProcess._sandboxed_arguments(
        runtime,
        {"implementationId": "oryn.lfm3.torch.v1", "modelRoot": str(tmp_path)},
        str(tmp_path),
        [str(Path(sys.executable)), "-I"],
    )
    assert command[0] == "/usr/bin/sandbox-exec"
    assert command[2].endswith("data/reviewed-inference.sb")
    assert "/permissive/override.sb" not in command


def test_shard_mutation_cannot_reuse_a_cached_active_index(tmp_path, monkeypatch):
    from pathlib import Path

    from oryn_engine.generated_capabilities import generated_artifact

    original = Path(catalog.__file__).parent
    binding = generated_artifact()["buildInputs"]["reviewedImplementations"]
    for name in (
        *binding["files"],
        "data/reviewed-implementations.json",
        "data/reviewed-implementations-lfm3.json",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((original / name).read_bytes())
    monkeypatch.setattr(catalog, "__file__", str(tmp_path / "reviewed_implementations.py"))
    catalog._cached_index.cache_clear()
    assert catalog.installed_index().profiles
    shard = tmp_path / "data/reviewed-implementations-lfm3.json"
    shard.write_bytes(shard.read_bytes() + b" ")
    with pytest.raises(parser.InspectionError):
        catalog.installed_index()
    catalog._cached_index.cache_clear()


def test_whole_transaction_deadline_and_cancel_discard_late_reader():
    import threading
    import time

    from oryn_engine.source_inspection import bounded_inspection

    release, finished, cancellation = threading.Event(), threading.Event(), threading.Event()

    def slow_read():
        release.wait(2)
        finished.set()
        return "late evidence"

    try:
        started = time.monotonic()
        with pytest.raises(parser.InspectionError, match="deadline"):
            bounded_inspection(slow_read, cancellation, seconds=0.02)
        assert time.monotonic() - started < 0.5
        assert cancellation.is_set()
        assert not finished.is_set()
    finally:
        release.set()
        assert finished.wait(1)
    with pytest.raises(parser.InspectionError, match="cancelled"):
        bounded_inspection(lambda: "must not run", cancellation)


def test_document_bound_cancel_reaches_pending_source_transaction():
    import threading

    from oryn_engine.models import ModelStore
    from oryn_engine.source_inspection import inspection_operation_id

    store = object.__new__(ModelStore)
    store._lock = threading.RLock()
    store._source_cancellations = {}
    store._operations = {}
    context = {"sourceURL": "https://huggingface.co/A/B", "tabId": "tab", "documentId": "doc"}
    entered, ended = threading.Event(), threading.Event()
    failures = []

    def inspect(*args, cancellation, **kwargs):
        entered.set()
        cancellation.wait(2)
        return {"late": "not admission authority"}

    store._inspect = inspect

    def run():
        try:
            store.inspect(
                {"provider": "huggingface", "repository": "A/B", "revision": "a" * 40},
                source_context=context,
            )
        except parser.InspectionError as exc:
            failures.append(str(exc))
        finally:
            ended.set()

    worker = threading.Thread(target=run)
    worker.start()
    assert entered.wait(1)
    assert inspection_operation_id({"tabId": "tab_hf", "documentId": "doc_hf"}) == (
        "source_5091ce246f93c9546725de079218b6035f71735251eb5e15cedf10bf1e290675"
    )
    assert store.cancel_install(inspection_operation_id(context))
    assert ended.wait(1)
    worker.join(timeout=1)
    assert failures == ["Source inspection cancelled"]
    assert not store._source_cancellations
