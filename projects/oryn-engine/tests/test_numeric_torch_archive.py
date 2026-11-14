import io
import json
import pickle
import struct
import zipfile
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from oryn_engine import numeric_state_dict as grammar
from oryn_engine.numeric_torch_archive import MetadataRanges, inspect_archive, read_weights

FIXTURES = Path(__file__).parent / "fixtures/numeric_archive"


def archive_bytes(state):
    stream = io.BytesIO()
    torch.save(state, stream)
    return stream.getvalue()


def inspect(raw):
    return inspect_archive(lambda offset, length: raw[offset : offset + length], len(raw))


def rewrite(raw, change):
    target = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as source, zipfile.ZipFile(target, "w") as output:
        for item in source.infolist():
            result = change(item, source.read(item))
            if result is not None:
                output.writestr(*result)
    return target.getvalue()


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float64,
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
        torch.bool,
    ],
)
def test_exact_dense_values_aliases_offsets_and_scalar_without_deserializers(
    tmp_path, monkeypatch, dtype
):
    base = torch.arange(12, dtype=torch.int64).to(dtype)
    state = OrderedDict(
        matrix=base.reshape(3, 4), alias=base.reshape(3, 4), slice=base[3:8], scalar=base[6].clone()
    )
    state._metadata = OrderedDict([("", {"version": 1}), ("matrix", {"version": 2})])
    raw = archive_bytes(state)
    path = tmp_path / "pytorch_model.bin"
    path.write_bytes(raw)

    def forbidden(*args, **kwargs):
        raise AssertionError("Object deserialization is forbidden")

    monkeypatch.setattr(pickle, "load", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    metadata = inspect(raw)
    assert len(metadata["tensors"]) == 4 and len(metadata["storages"]) == 2
    actual = read_weights(
        path, expected_shapes={k: list(v.shape) for k, v in state.items()}, maximum_bytes=1024
    )
    assert set(actual) == set(state)
    for name in state:
        assert actual[name].dtype == state[name].dtype
        assert torch.equal(actual[name], state[name])
    assert actual["matrix"].data_ptr() == actual["alias"].data_ptr()
    assert actual["slice"].storage_offset() == 3


def test_metadata_ranges_never_read_numeric_storage_bytes():
    raw = archive_bytes({"weight": torch.arange(100000, dtype=torch.float32).reshape(100, 1000)})
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        storage = next(x for x in archive.infolist() if "/data/" in x.filename)
    header = struct.unpack("<4s5H3L2H", raw[storage.header_offset : storage.header_offset + 30])
    start = storage.header_offset + 30 + header[-2] + header[-1]
    end = start + storage.file_size
    calls = []

    def read_range(offset, length):
        assert offset + length <= start or offset >= end
        calls.append((offset, length))
        return raw[offset : offset + length]

    result = inspect_archive(read_range, len(raw))
    assert result["tensors"]["weight"]["shape"] == [100, 1000]
    assert sum(n for _, n in calls) < 4096


@pytest.mark.parametrize(
    "name",
    [
        "modernbert",
        "punctuate",
        "clinicalbert",
        "mdeberta",
        "chinese-qa",
        "nllb",
        "bart",
        "dpt",
        "musicgen",
    ],
)
def test_original_metadata_grammar_and_complete_storage_directory(name):
    tensors = grammar.parse((FIXTURES / f"{name}.metadata").read_bytes())
    directory = json.loads((FIXTURES / f"{name}.directory.json").read_text())
    storages = {x["name"].split("/data/")[-1]: x for x in directory if "/data/" in x["name"]}
    assert {x.storage.key for x in tensors.values()} == set(storages)
    for value in tensors.values():
        assert (
            value.storage.elements * value.storage.itemsize == storages[value.storage.key]["size"]
        )
        assert value.offset >= 0


@pytest.mark.parametrize(
    "global_name",
    [
        "os\nsystem",
        "posix\nsystem",
        "builtins\neval",
        "torch\nload",
        "torch._utils\n_rebuild_parameter",
        "numpy.core.multiarray\n_reconstruct",
    ],
)
def test_global_tags_never_resolve_to_imports_or_functions(global_name):
    raw = b"\x80\x02c" + global_name.encode() + b"\n)R."
    with pytest.raises(ValueError, match="NUMERIC_ARCHIVE_GLOBAL"):
        grammar.parse(raw)


@pytest.mark.parametrize(
    "damage",
    [
        "trailing",
        "truncated",
        "protocol",
        "duplicate-key",
        "cycle",
        "non-tensor",
        "unknown-opcode",
        "memo-rewrite",
        "unknown-memo",
        "mark",
        "depth",
    ],
)
def test_closed_grammar_rejects_invalid_or_object_state(damage):
    state = {"weight": torch.arange(3, dtype=torch.float32)}
    with zipfile.ZipFile(io.BytesIO(archive_bytes(state))) as archive:
        raw = archive.read("archive/data.pkl")
    if damage == "trailing":
        raw += b"."
    elif damage == "truncated":
        raw = raw[:-1]
    elif damage == "protocol":
        raw = b"\x80\x04" + raw[2:]
    elif damage == "duplicate-key":
        raw = b"\x80\x02}X\x01\x00\x00\x00aK\x01sX\x01\x00\x00\x00aK\x02s."
    elif damage == "cycle":
        cyclic = {}
        cyclic["cycle"] = cyclic
        raw = pickle.dumps(cyclic, protocol=2)
    elif damage == "non-tensor":
        raw = pickle.dumps({"weight": 42}, protocol=2)
    elif damage == "unknown-opcode":
        raw = b"\x80\x02]."
    elif damage == "memo-rewrite":
        raw = b"\x80\x02}q\x00q\x00."
    elif damage == "unknown-memo":
        raw = b"\x80\x02h\x01."
    elif damage == "mark":
        raw = b"\x80\x02t."
    else:
        raw = b"\x80\x02" + b"(" * 17 + b"."
    with pytest.raises(ValueError):
        grammar.parse(raw)


@pytest.mark.parametrize(
    "damage",
    [
        "compression",
        "extra-code",
        "missing-storage",
        "storage-size",
        "byteorder",
        "version",
        "symlink",
        "comment",
        "metadata-crc",
        "payload-crc",
        "duplicate",
        "truncated",
    ],
)
def test_archive_container_rejects_unreviewed_or_damaged_members(tmp_path, damage):
    raw = archive_bytes({"weight": torch.arange(4, dtype=torch.float32)})

    def change(item, data):
        name = item.filename
        if damage == "compression":
            item.compress_type = zipfile.ZIP_DEFLATED
        elif damage == "extra-code" and name.endswith("/version"):
            item.filename = "archive/code/model.py"
        elif damage == "missing-storage" and "/data/" in name:
            return None
        elif damage == "storage-size" and "/data/" in name:
            data = data[:-4]
        elif damage == "byteorder" and name.endswith("/byteorder"):
            data = b"big"
        elif damage == "version" and name.endswith("/version"):
            data = b"99\n"
        elif damage == "symlink" and "/data/" in name:
            item.create_system = 3
            item.external_attr = 0o120777 << 16
        return item, data

    raw = rewrite(raw, change)
    if damage in {"comment", "duplicate"}:
        stream = io.BytesIO(raw)
        with zipfile.ZipFile(stream, "a") as archive:
            if damage == "comment":
                archive.comment = b"undeclared data"
            else:
                with pytest.warns(UserWarning, match="Duplicate"):
                    archive.writestr("archive/version", b"3\n")
        raw = stream.getvalue()
    if damage.endswith("-crc"):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            member = archive.getinfo(
                "archive/data.pkl" if damage == "metadata-crc" else "archive/data/0"
            )
        header = struct.unpack("<4s5H3L2H", raw[member.header_offset : member.header_offset + 30])
        position = member.header_offset + 30 + header[-2] + header[-1]
        raw = raw[:position] + bytes([raw[position] ^ 1]) + raw[position + 1 :]
    if damage == "truncated":
        raw = raw[:-25]
    if damage != "payload-crc":
        with pytest.raises(ValueError):
            inspect(raw)
    else:
        assert inspect(raw)["tensors"]["weight"]["shape"] == [4]
        path = tmp_path / "model.bin"
        path.write_bytes(raw)
        with pytest.raises(ValueError):
            read_weights(path, expected_shapes={"weight": [4]}, maximum_bytes=100)


@pytest.mark.parametrize(
    "damage",
    ["shape", "missing-key", "extra-key", "budget", "symlink", "nan", "inf", "gapped"],
)
def test_runtime_checks_graph_budget_and_numeric_content_before_exposing_weights(tmp_path, damage):
    weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    if damage == "nan":
        weight[0, 0] = torch.nan
    elif damage == "inf":
        weight[0, 0] = torch.inf
    elif damage == "gapped":
        weight = weight[:, ::2]
    path = tmp_path / "model.bin"
    path.write_bytes(archive_bytes({"weight": weight}))
    shapes = {"weight": list(weight.shape)}
    if damage == "shape":
        shapes["weight"] = [6]
    elif damage == "missing-key":
        shapes = {"different": [2, 3]}
    elif damage == "extra-key":
        shapes["extra"] = [1]
    elif damage == "symlink":
        link = tmp_path / "link.bin"
        link.symlink_to(path)
        path = link
    with pytest.raises(ValueError):
        read_weights(path, expected_shapes=shapes, maximum_bytes=1 if damage == "budget" else 1024)


@pytest.mark.parametrize(
    "damage", ["size", "read-budget", "total-budget", "calls", "short-read", "seek"]
)
def test_range_reader_enforces_fixed_metadata_resource_bounds(damage):
    if damage == "size":
        with pytest.raises(ValueError):
            MetadataRanges(lambda *_: b"", 2**63)
        return
    reader = MetadataRanges(
        lambda _, length: b"x" * (length - (damage == "short-read")), 32 * 1024**2
    )
    if damage == "total-budget":
        reader.bytes = 16 * 1024**2
    elif damage == "calls":
        reader.calls = 64
    with pytest.raises(ValueError):
        if damage == "seek":
            reader.seek(-1)
        else:
            reader.read(9 * 1024**2 if damage == "read-budget" else 10)


@pytest.mark.parametrize(
    "damage",
    [
        "negative-offset",
        "overrun",
        "shape-list",
        "huge-shape",
        "stride",
        "gradients",
        "hooks",
        "integer-limit",
        "bookkeeping",
        "complex",
        "sparse",
    ],
)
def test_tensor_descriptor_semantics_cannot_restore_arbitrary_tensor_objects(damage):
    if damage == "complex":
        raw = archive_bytes({"weight": torch.ones(4, dtype=torch.complex64)})
    elif damage == "sparse":
        raw = archive_bytes({"weight": torch.eye(4).to_sparse()})
    elif damage == "bookkeeping":
        state = OrderedDict(weight=torch.ones(4))
        state._metadata = {"": {"version": 1, "extra": "not permitted"}}
        raw = archive_bytes(state)
    else:
        sentinel = object()
        args = [sentinel, 0, (4,), (1,), False, OrderedDict()]
        if damage == "negative-offset":
            args[1] = -1
        elif damage == "overrun":
            args[1] = 1
        elif damage == "shape-list":
            args[2] = [4]
        elif damage == "huge-shape":
            args[2] = (2**40,)
        elif damage == "stride":
            args[3] = (2,)
        elif damage == "gradients":
            args[4] = True
        elif damage == "hooks":
            args[5] = OrderedDict(hook=1)
        else:
            args[1] = 2**63

        class Descriptor:
            def __reduce__(self):
                # Serialization writes this tag; the reader must never call it.
                return torch._utils._rebuild_tensor_v2, tuple(args)

        class Writer(pickle.Pickler):
            def persistent_id(self, value):
                if value is sentinel:
                    return "storage", torch.FloatStorage, "0", "cpu", 4
                return None

        stream = io.BytesIO()
        Writer(stream, protocol=2).dump({"weight": Descriptor()})
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("archive/data.pkl", stream.getvalue())
            archive.writestr("archive/version", b"3\n")
            archive.writestr("archive/byteorder", b"little")
            archive.writestr("archive/data/0", b"\x00" * 16)
        raw = output.getvalue()
    with pytest.raises(ValueError):
        inspect(raw)


@pytest.mark.parametrize(
    "damage",
    [None, "full-body", "wrong-range", "truncated", "oversized", "mutable-revision", "wrong-file"],
)
def test_hub_reader_requires_exact_bounded_ranges_without_full_download(monkeypatch, damage):
    import httpx

    import oryn_engine.models as models

    raw = archive_bytes({"weight": torch.arange(100000, dtype=torch.float32).reshape(100, 1000)})
    calls = []

    def response(request):
        assert request.headers["Accept-Encoding"] == "identity"
        start, end = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        calls.append((start, end))
        body = raw[start : end + 1]
        if damage == "truncated":
            body = body[:-1]
        if damage == "oversized":
            body += b"x"
        return httpx.Response(
            200 if damage == "full-body" else 206,
            headers={
                "content-range": "wrong"
                if damage == "wrong-range"
                else f"bytes {start}-{end}/{len(raw)}"
            },
            content=body,
        )

    original = httpx.Client
    monkeypatch.setattr(
        models.httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(response), **kwargs),
    )
    client = models.HuggingFaceClient()
    args = (
        "test/numeric",
        "main" if damage == "mutable-revision" else "a" * 40,
        "model.bin" if damage == "wrong-file" else "pytorch_model.bin",
        len(raw),
    )
    if damage:
        with pytest.raises((ValueError, models.ModelStoreError)):
            client.read_numeric_archive(*args)
    else:
        result = client.read_numeric_archive(*args)
        assert result["tensors"]["weight"]["shape"] == [100, 1000]
        assert sum(b - a + 1 for a, b in calls) < 4096
    if damage in {"mutable-revision", "wrong-file"}:
        assert not calls


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64])
@pytest.mark.parametrize(
    "layout", ["transpose", "channels-last", "permutation", "singleton-offset"]
)
def test_dense_permutations_preserve_exact_values_strides_and_aliases(
    tmp_path, monkeypatch, dtype, layout
):
    base = torch.arange(60).to(dtype)
    if layout == "transpose":
        weight = base.reshape(6, 10).T
    elif layout == "channels-last":
        weight = base.reshape(1, 3, 4, 5).contiguous(memory_format=torch.channels_last)
    elif layout == "permutation":
        weight = base.reshape(3, 4, 5).permute(2, 0, 1)
    else:
        weight = base[3:27].reshape(2, 1, 3, 4).permute(3, 1, 0, 2)
    state = {"weight": weight, "alias": weight}
    raw = archive_bytes(state)
    path = tmp_path / "pytorch_model.bin"
    path.write_bytes(raw)

    def forbidden(*args, **kwargs):
        raise AssertionError("No object deserialization")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)
    monkeypatch.setattr(pickle, "load", forbidden)
    metadata = inspect(raw)
    assert metadata["tensors"]["weight"]["stride"] == list(weight.stride())
    loaded = read_weights(
        path, expected_shapes={k: list(v.shape) for k, v in state.items()}, maximum_bytes=1024
    )
    assert torch.equal(loaded["weight"], weight)
    assert loaded["weight"].stride() == weight.stride()
    assert loaded["weight"].storage_offset() == weight.storage_offset()
    assert loaded["weight"].data_ptr() == loaded["alias"].data_ptr()


@pytest.mark.parametrize("layout", ["overlap", "broadcast", "gap", "nonfinite"])
def test_dense_permutation_support_does_not_admit_unsafe_views(tmp_path, layout):
    base = torch.arange(12, dtype=torch.float32)
    if layout == "nonfinite":
        base[7] = float("nan")
        value = base.reshape(3, 4).T
    else:
        strides = {"overlap": (1, 1), "broadcast": (0, 1), "gap": (6, 2)}[layout]
        value = base.as_strided((2, 3), strides)
    raw = archive_bytes({"weight": value})
    path = tmp_path / "pytorch_model.bin"
    path.write_bytes(raw)
    if layout != "nonfinite":
        with pytest.raises(ValueError, match="STRIDE"):
            inspect(raw)
    with pytest.raises(ValueError, match="STRIDE|NONFINITE"):
        read_weights(path, expected_shapes={"weight": list(value.shape)}, maximum_bytes=1024)
