"""Reproduce the pinned upstream wheel used by Oryn's local image runtime."""

import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    source = json.loads((root / "packaging/diffusers-source.json").read_text())
    revision = source["revision"]
    with tempfile.TemporaryDirectory(prefix="oryn-diffusers-build-") as temporary:
        directory = Path(temporary)
        archive = directory / "source.tar.gz"
        url = f"https://codeload.github.com/huggingface/diffusers/tar.gz/{revision}"
        with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as output:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 64 * 1024 * 1024:
                    raise ValueError("Upstream source archive exceeds the build limit")
                output.write(chunk)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != source["archiveSha256"]:
            raise ValueError("Pinned upstream source archive changed")
        with tarfile.open(archive) as payload:
            payload.extractall(directory, filter="data")
        output = directory / "wheels"
        subprocess.run(
            [
                "uv",
                "build",
                "--wheel",
                "--python",
                "3.12.13",
                "--build-constraint",
                str(root / "packaging/diffusers-build-constraints.txt"),
                "--out-dir",
                str(output),
                str(directory / f"diffusers-{revision}"),
            ],
            check=True,
            env={**os.environ, "SOURCE_DATE_EPOCH": str(source["sourceDateEpoch"])},
        )
        wheel = output / source["wheel"]["file"]
        if hashlib.sha256(wheel.read_bytes()).hexdigest() != source["wheel"]["sha256"]:
            raise ValueError("Rebuilt wheel differs from the reviewed build")
        destination = root / "vendor" / wheel.name
        destination.write_bytes(wheel.read_bytes())
        print(f"Verified pinned Diffusers wheel: {destination}")


if __name__ == "__main__":
    main()
