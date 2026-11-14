import h5py
import numpy as np
import pytest

from oryn_engine.numeric_hdf5 import read_weights


def test_reads_only_numeric_payloads_and_ignores_object_descriptions(tmp_path):
    path = tmp_path / "weights.h5"
    with h5py.File(path, "w") as f:
        f.attrs["model_config"] = '{"class_name":"Lambda","function":"never execute"}'
        f.create_dataset("encoder/kernel:0", data=np.arange(12, dtype=">f4").reshape(3, 4))
        f.create_dataset("bias:0", data=np.float32(0.5))
    h5py.h5pl.append(str(tmp_path).encode())
    values = read_weights(path)
    assert h5py.h5pl.size() == 0
    np.testing.assert_array_equal(values["encoder/kernel:0"], np.arange(12).reshape(3, 4))
    assert values["bias:0"].shape == () and values["bias:0"] == 0.5


@pytest.mark.parametrize(
    "kind",
    [
        "soft",
        "external",
        "alias",
        "cycle",
        "virtual",
        "external-storage",
        "gzip",
        "chunked",
        "object",
        "compound",
        "complex",
        "named-type",
        "oversized",
        "unallocated",
        "nonfinite",
        "empty",
    ],
)
def test_unsupported_storage_cannot_be_interpreted_as_numeric_weights(tmp_path, kind):
    path = tmp_path / "weights.h5"
    with h5py.File(path, "w") as f:
        if kind == "soft":
            f["evil"] = h5py.SoftLink("/missing")
        elif kind == "external":
            f["evil"] = h5py.ExternalLink("must-not-open.h5", "/secret")
        elif kind == "alias":
            f["value"] = np.zeros(2, np.float32)
            f["alias"] = f["value"]
        elif kind == "cycle":
            f["cycle"] = f
        elif kind == "virtual":
            layout = h5py.VirtualLayout(shape=(2,), dtype="f4")
            layout[:] = h5py.VirtualSource("must-not-open.h5", "secret", shape=(2,))
            f.create_virtual_dataset("virtual", layout)
        elif kind == "external-storage":
            f.create_dataset(
                "external", shape=(2,), dtype="f4", external=[("must-not-open.raw", 0, 8)]
            )
        elif kind == "gzip":
            f.create_dataset("compressed", data=np.zeros(2, np.float32), compression="gzip")
        elif kind == "chunked":
            f.create_dataset("chunks", data=np.zeros(2, np.float32), chunks=(1,))
        elif kind == "object":
            f.create_dataset("string", data="not a weight", dtype=h5py.string_dtype())
        elif kind == "compound":
            f.create_dataset("compound", data=np.zeros(2, dtype=[("x", "f4")]))
        elif kind == "complex":
            f.create_dataset("complex", data=np.zeros(2, np.complex64))
        elif kind == "named-type":
            dtype = h5py.h5t.py_create(np.dtype("f4"))
            dtype.commit(f.id, b"datatype")
        elif kind == "oversized":
            f.create_dataset("oversized", shape=(1000001,), dtype="f4")
        elif kind == "unallocated":
            f.create_dataset("empty-storage", shape=(2,), dtype="f4")
        elif kind == "nonfinite":
            f.create_dataset("nonfinite", data=np.array([np.nan], np.float32))
    with pytest.raises(ValueError, match="HDF5_"):
        read_weights(path)


def test_all_descriptors_are_validated_before_any_payload(tmp_path, monkeypatch):
    path = tmp_path / "weights.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("a_good", data=np.zeros(2, np.float32))
        f.create_dataset("z_bad", data=np.zeros(2, np.float32), compression="gzip")
    monkeypatch.setattr(
        h5py.Dataset,
        "__getitem__",
        lambda *_: pytest.fail("payload read before descriptor validation"),
    )
    with pytest.raises(ValueError, match="STORAGE_UNSUPPORTED"):
        read_weights(path)


def test_symlink_and_non_hdf5_are_rejected(tmp_path):
    path = tmp_path / "weights.h5"
    path.write_bytes(b"not hdf5 at all")
    with pytest.raises(ValueError, match="SIGNATURE"):
        read_weights(path)
    link = tmp_path / "linked.h5"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="FILE_INVALID"):
        read_weights(link)
