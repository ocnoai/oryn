"""Closed numeric weight datasets, never a Keras model or object deserializer.

Attributes are inert and unread. Links, virtual/external storage, filters, chunked
storage, named datatypes and non-floating datasets are outside this grammar.
"""

from __future__ import annotations

import math
from pathlib import Path

MAX_FILE_BYTES = 4 * 1024**3
MAX_TENSOR_BYTES = 1024**3
MAX_TOTAL_BYTES = 4 * 1024**3


def read_weights(path: Path):
    import h5py
    import numpy as np

    if path.is_symlink() or not path.is_file() or not 8 <= path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError("HDF5_WEIGHT_FILE_INVALID")
    # This runs inside the owned worker. No repository can load a native HDF5
    # filter plugin; this reader also rejects every dataset filter before reading.
    while h5py.h5pl.size():
        h5py.h5pl.remove(0)
    with path.open("rb") as stream:
        if stream.read(8) != b"\x89HDF\r\n\x1a\n":
            raise ValueError("HDF5_WEIGHT_SIGNATURE_INVALID")
    with h5py.File(path, "r", driver="sec2") as source:
        seen, datasets = {h5py.h5o.get_info(source.id).addr}, {}
        nodes = total = 0

        def inspect(group, depth):
            nonlocal nodes, total
            if depth > 16 or len(group) > 50_000:
                raise ValueError("HDF5_STRUCTURE_LIMIT")
            for key in group:
                nodes += 1
                if nodes > 50_000 or not key.isascii() or not 1 <= len(key) <= 256:
                    raise ValueError("HDF5_STRUCTURE_LIMIT")
                if not isinstance(group.get(key, getlink=True), h5py.HardLink):
                    raise ValueError("HDF5_LINK_UNSUPPORTED")
                item = group[key]
                address = h5py.h5o.get_info(item.id).addr
                if address in seen:
                    raise ValueError("HDF5_ALIAS_UNSUPPORTED")
                seen.add(address)
                if isinstance(item, h5py.Group):
                    inspect(item, depth + 1)
                    continue
                if not isinstance(item, h5py.Dataset):
                    raise ValueError("HDF5_OBJECT_UNSUPPORTED")
                shape, dtype = item.shape, item.dtype
                if (
                    shape is None
                    or len(shape) > 4
                    or any(not 1 <= d <= 1_000_000 for d in shape)
                    or dtype.kind != "f"
                    or dtype.itemsize not in {2, 4, 8}
                    or dtype.fields
                    or dtype.subdtype
                    or dtype.hasobject
                ):
                    raise ValueError("HDF5_DTYPE_OR_SHAPE_UNSUPPORTED")
                size = math.prod(shape) * dtype.itemsize
                total += size
                if size > MAX_TENSOR_BYTES or total > MAX_TOTAL_BYTES:
                    raise ValueError("HDF5_WEIGHT_SIZE_LIMIT")
                properties = item.id.get_create_plist()
                if (
                    item.is_virtual
                    or properties.get_external_count()
                    or properties.get_nfilters()
                    or properties.get_layout() not in {h5py.h5d.CONTIGUOUS, h5py.h5d.COMPACT}
                    or item.id.get_storage_size() != size
                ):
                    raise ValueError("HDF5_STORAGE_UNSUPPORTED")
                datasets[item.name.lstrip("/")] = item

        inspect(source, 0)
        if not datasets:
            raise ValueError("HDF5_WEIGHTS_MISSING")
        # Every descriptor passes before any numeric weight payload is allocated.
        result = {}
        for name, dataset in datasets.items():
            array = np.asarray(dataset[()], dtype=np.float32)
            if not bool(np.isfinite(array).all()):
                raise ValueError("HDF5_NONFINITE_WEIGHT")
            result[name] = array
        return result
