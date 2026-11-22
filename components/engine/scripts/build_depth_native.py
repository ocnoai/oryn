"""Build the reviewed DA3 numerical subset; no upstream application loader is installed."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path


def build(source: Path, destination: Path) -> Path:
    manifest = json.loads((destination / "depth-native-source.json").read_text())
    files = {}
    info = "oryn_depth_native-0.1.0.dist-info"
    for name, expected in manifest["files"].items():
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Native source is missing or redirected")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Native source changed: {name}")
        target = info + "/licenses/LICENSE" if name == "LICENSE" else name.removeprefix("src/")
        files[target] = raw
    files["depth_anything_3/__init__.py"] = (
        '__version__ = "0.1.0"\n__upstream_commit__ = ' + repr(manifest["commit"]) + "\n"
    ).encode()
    files["depth_anything_3/cfg.py"] = (
        b'"""The Oryn native subset accepts engine-constructed modules only."""\n\n'
        b"def create_object(*args, **kwargs):\n"
        b'    raise ValueError("DEPTH_NATIVE_DYNAMIC_CONSTRUCTION_DISABLED")\n'
    )
    files[info + "/METADATA"] = (
        b"Metadata-Version: 2.4\nName: oryn-depth-native\nVersion: 0.1.0\n"
        b"Summary: Reviewed Depth Anything 3 numerical components for Oryn\n"
        b"Requires-Python: >=3.11\nLicense-Expression: Apache-2.0\n"
        b"Requires-Dist: torch>=2.13,<3\nRequires-Dist: numpy>=2,<3\n"
        b"Requires-Dist: einops==0.8.2\nRequires-Dist: addict==2.4.0\n"
        b"Requires-Dist: omegaconf==2.4.0\n"
    )
    files[info + "/WHEEL"] = (
        b"Wheel-Version: 1.0\nGenerator: oryn-build-depth-native\n"
        b"Root-Is-Purelib: true\nTag: py3-none-any\n"
    )
    files[info + "/upstream.json"] = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    records = io.StringIO(newline="")
    writer = csv.writer(records, lineterminator="\n")
    for name, raw in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
        writer.writerow((name, "sha256=" + digest, len(raw)))
    writer.writerow((info + "/RECORD", "", ""))
    files[info + "/RECORD"] = records.getvalue().encode()
    wheel = destination / "oryn_depth_native-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(files.items()):
            member = zipfile.ZipInfo(name, date_time=(2026, 10, 15, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o644 << 16
            archive.writestr(member, raw)
    return wheel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    args = parser.parse_args()
    print(build(args.source_root, Path(__file__).resolve().parents[1] / "vendor"))
