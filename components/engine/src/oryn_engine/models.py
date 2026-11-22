from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import httpx
import psutil
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
from huggingface_hub import constants as hub_constants
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
from huggingface_hub.utils import build_hf_headers, tqdm

from oryn_engine.capability_sources import framework_sources
from oryn_engine.compatibility_plane import (
    SCHEMA_VERSION as DESCRIPTOR_SCHEMA_VERSION,
)
from oryn_engine.compatibility_plane import (
    ModelDescriptor,
    compile_registry,
    evaluate_eligibility,
)
from oryn_engine.model_inspection import inspect_descriptor
from oryn_engine.model_layouts import (
    SHARD,
    WEIGHT_INDEX,
    assess_layout,
    validate_weight_index,
    weight_index_name,
)
from oryn_engine.model_support import transformers_workload
from oryn_engine.native_capability_sources import (
    LLAMA_CPP_CHAT_QUANTIZATIONS,
    llama_cpp_chat_template,
    native_sources,
    normalized_gguf_architecture,
)
from oryn_engine.runtime.hardware import HardwareManager
from oryn_engine.runtime_packs.builtin_catalog import builtin_registry
from oryn_engine.runtime_packs.builtin_foundations import builtin_foundation_registry
from oryn_engine.runtime_packs.execution import PackRuntimeLocator
from oryn_engine.runtime_packs.generation_profiles import generation_profile
from oryn_engine.runtime_packs.gguf import GGUFError, inspect_gguf, inspect_gguf_set, split_file_set
from oryn_engine.runtime_packs.manifests import CompatibilityManifest
from oryn_engine.runtime_packs.registry import RuntimePackRegistry
from oryn_engine.runtime_planning import (
    RuntimeCandidate,
    RuntimeIdentity,
    RuntimePlan,
    RuntimePlanError,
    RuntimePolicy,
)
from oryn_engine.runtime_routing import (
    GENERATED_PACKS,
    RoutingContext,
    RoutingEvidenceStore,
    RuntimeRoutingService,
    canonical_digest,
    generated_pack_id,
    runtime_build_digest,
)

ModelEventSink = Callable[[str, dict[str, Any]], None]
RuntimeStateProvider = Callable[[dict[str, Any]], dict[str, Any]]
RuntimePlanningStateProvider = Callable[[], tuple[str, int]]
DownloadProgress = Callable[[int, int | None], None]
REPOSITORY_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"
)
REVISION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
EXACT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")
UNSAFE_WEIGHT_SUFFIXES = {".bin", ".ckpt", ".pkl", ".pickle", ".pt", ".pth"}
MAX_LOCAL_COMPONENT_CONFIG_BYTES = 256 * 1024
BITSANDBYTES_NF4 = "bitsandbytes-nf4"
# GGUF variants are chosen by their headers. File names only order the candidates,
# and each inspection reads at most this many bounded headers.
MAX_GGUF_HEADER_READS = 6
MAX_GGUF_PROJECTOR_READS = 4
# A variant that fits this share of the machine's memory is preferred when one exists.
GGUF_VARIANT_MEMORY_SHARE = 0.75


class ModelStoreError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.details = details


class InstallCancelled(ModelStoreError):
    def __init__(self) -> None:
        super().__init__("MODEL_INSTALL_CANCELLED", "Model installation was cancelled.")


class HubClient(Protocol):
    def model_info(self, repo_id: str, revision: str, files_metadata: bool = True) -> Any: ...

    def has_access(self, repo_id: str) -> bool: ...

    def download(
        self,
        repo_id: str,
        filename: str,
        revision: str,
        cache_dir: str,
        progress: DownloadProgress | None = None,
    ) -> str: ...


def _safe_metadata_name(filename: str) -> bool:
    return (
        isinstance(filename, str)
        and 1 <= len(filename) <= 512
        and filename.endswith(".json")
        and not filename.startswith("/")
        and "\\" not in filename
        and "\0" not in filename
        and all(part not in {"", ".", ".."} for part in filename.split("/"))
    )


class HuggingFaceClient:
    def __init__(self, token_provider: Callable[[], str | None] | None = None) -> None:
        # Hub header construction can otherwise probe the agent-harness registry,
        # outside the requested model and our bounded metadata request budget.
        # Set the already-imported constant too: import order must not enable it.
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        hub_constants.HF_HUB_DISABLE_TELEMETRY = True
        self._token_provider = token_provider or (lambda: None)
        self._api = HfApi(endpoint="https://huggingface.co", token=False)
        self._onnx_headers: dict[str, dict] = {}

    def _token(self) -> str | bool:
        return self._token_provider() or False

    def model_info(self, repo_id: str, revision: str, files_metadata: bool = True) -> Any:
        try:
            info = self._api.model_info(
                repo_id=repo_id,
                revision=revision,
                files_metadata=files_metadata,
                token=self._token(),
            )
            # Hub summaries omit quantization and component requirements. Read only bounded
            # JSON at the resolved revision, after access authorization; never import model code.
            if not (
                getattr(info, "gated", False) or getattr(info, "private", False)
            ) or self.has_access(repo_id):
                names = {item.rfilename for item in info.siblings or []}
                from oryn_engine.model_metadata import root_config_name

                metadata_name = root_config_name(names)
                if metadata_name in names:
                    config = self.read_json(repo_id, info.sha, metadata_name)
                    # The root file is authoritative. Hub summaries contain derived
                    # fields (for example `diffusers`) that are not pipeline options.
                    info.config = config
            return info
        except GatedRepoError as exc:
            raise ModelStoreError(
                "MODEL_ACCESS_RESTRICTED",
                "The model host requires approval or authentication for this model.",
            ) from exc
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in {401, 403}:
                raise ModelStoreError(
                    "MODEL_ACCESS_RESTRICTED",
                    "The model host requires approval or authentication for this model.",
                ) from exc
            raise

    def read_json(
        self, repository: str, revision: str, filename: str, *, _vision_labels=False
    ) -> dict[str, Any]:
        from oryn_engine.onnx_vision import MAX_LABEL_BYTES

        limit = MAX_LABEL_BYTES if _vision_labels else 1024 * 1024
        raw = self._bounded_bytes(repository, revision, filename, limit)
        from oryn_engine.model_metadata import (
            parse_metadata,
            validate_inspection_tokenizer_metadata,
        )

        try:
            if _vision_labels:
                from oryn_engine.onnx_vision import parse_labels

                return parse_labels(bytes(raw))
            value = parse_metadata(
                bytes(raw), redundant_keys=Path(filename).name == "tokenizer_config.json"
            )
            if Path(filename).name == "tokenizer_config.json":
                validate_inspection_tokenizer_metadata(value)
            return value
        except (ValueError, OverflowError) as exc:
            raise ModelStoreError(
                "MODEL_METADATA_INVALID",
                "Model JSON is invalid.",
                details={
                    "metadataRejection": {
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "sizeBytes": len(raw),
                    }
                },
            ) from exc

    def read_template(self, repository: str, revision: str, filename: str) -> str:
        """Return a repository chat template as inert text; it is never compiled."""

        if filename != "chat_template.jinja":
            raise ModelStoreError("MODEL_METADATA_INVALID", "Template filename is invalid.")
        raw = self._bounded_bytes(repository, revision, filename, 512 * 1024)
        try:
            return bytes(raw).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ModelStoreError("MODEL_METADATA_INVALID", "Template is not UTF-8.") from exc

    def _bounded_bytes(
        self, repository: str, revision: str, filename: str, limit: int
    ) -> bytearray:

        if not re.fullmatch(
            r"[A-Za-z0-9._-]{1,96}/[A-Za-z0-9._-]{1,96}", repository
        ) or not re.fullmatch(r"[0-9a-f]{40,64}", revision):
            raise ModelStoreError("MODEL_METADATA_INVALID", "Metadata identity must be immutable.")
        if not (_safe_metadata_name(filename) or filename == "chat_template.jinja"):
            raise ModelStoreError("MODEL_METADATA_INVALID", "Metadata filename is invalid.")
        url = hf_hub_url(repository, filename, revision=revision, endpoint="https://huggingface.co")
        return self._bounded_hf_url(url, limit)

    def read_source(self, repository, revision, filename, limit):
        from oryn_engine.source_inspector_worker import MAX_SOURCE_BYTES

        if (
            not REPOSITORY_PATTERN.fullmatch(repository)
            or not EXACT_REVISION_PATTERN.fullmatch(revision)
            or re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,127}\.py", filename) is None
            or type(limit) is not int
            or not 0 < limit <= MAX_SOURCE_BYTES
        ):
            raise ModelStoreError("MODEL_METADATA_INVALID", "Invalid inert source request")
        return self._bounded_hf_url(
            hf_hub_url(repository, filename, revision=revision, endpoint="https://huggingface.co"),
            limit,
        )

    def source_model_info(self, repo_id, revision, files_metadata=True):
        from urllib.parse import quote

        from huggingface_hub import ModelInfo

        from oryn_engine.model_metadata import parse_metadata

        if (
            not REPOSITORY_PATTERN.fullmatch(repo_id)
            or not isinstance(revision, str)
            or len(revision) > 128
        ):
            raise ModelStoreError("MODEL_METADATA_INVALID", "Invalid source model identity")
        raw = self._bounded_hf_url(
            f"https://huggingface.co/api/models/{repo_id}/revision/"
            f"{quote(revision, safe='')}?blobs=true",
            4 * 1024 * 1024,
        )
        metadata = parse_metadata(bytes(raw), limit=4 * 1024 * 1024)
        if (
            not isinstance(metadata.get("siblings"), list)
            or len(metadata["siblings"]) > 10000
            or not isinstance(metadata.get("sha"), str)
            or not EXACT_REVISION_PATTERN.fullmatch(metadata["sha"])
        ):
            raise ModelStoreError("MODEL_METADATA_INVALID", "Invalid bounded model inventory")
        info = ModelInfo(**metadata)
        names = {item.rfilename for item in info.siblings}
        if "config.json" in names:
            info.config = self.read_json(repo_id, info.sha, "config.json")
        return info

    def _bounded_hf_url(self, url, limit):
        from urllib.parse import urljoin, urlsplit

        deadline = time.monotonic() + 60
        raw = bytearray()
        for _ in range(4):
            parsed = urlsplit(url)
            if (
                parsed.scheme != "https"
                or parsed.netloc != "huggingface.co"
                or parsed.username
                or parsed.fragment
            ):
                raise ModelStoreError(
                    "MODEL_METADATA_INVALID", "Metadata redirect is outside its host."
                )
            with httpx.stream(
                "GET",
                url,
                headers={**build_hf_headers(token=self._token()), "Accept-Encoding": "identity"},
                timeout=15,
                follow_redirects=False,
            ) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    url = urljoin(url, response.headers.get("location", ""))
                    continue
                response.raise_for_status()
                if (
                    getattr(response, "headers", {}).get("content-encoding", "identity")
                    != "identity"
                ):
                    raise ModelStoreError(
                        "MODEL_METADATA_INVALID", "Encoded metadata is not admitted"
                    )
                for chunk in response.iter_bytes(chunk_size=64 * 1024):
                    if time.monotonic() > deadline:
                        raise ModelStoreError(
                            "MODEL_METADATA_UNAVAILABLE", "Metadata read exceeded its time budget."
                        )
                    if len(raw) + len(chunk) > limit:
                        raise ModelStoreError(
                            "MODEL_METADATA_INVALID", "Metadata exceeds time or byte limits."
                        )
                    raw.extend(chunk)
                break
        else:
            raise ModelStoreError("MODEL_METADATA_INVALID", "Metadata redirect limit exceeded.")
        return raw

    def has_access(self, repo_id: str) -> bool:
        try:
            self._api.auth_check(repo_id=repo_id, repo_type="model", token=self._token())
            return True
        except (GatedRepoError, HfHubHTTPError):
            return False

    def read_vision_labels(self, repository, revision, filename):
        return self.read_json(repository, revision, filename, _vision_labels=True)

    def read_numeric_archive(self, repository, revision, filename, size):
        import time

        from oryn_engine.numeric_torch_archive import inspect_archive

        if not EXACT_REVISION_PATTERN.fullmatch(revision) or filename not in {
            "pytorch_model.bin",
            "diffusion_pytorch_model.bin",
        }:
            raise ModelStoreError("MODEL_METADATA_INVALID", "Numeric archive identity is invalid.")
        url = hf_hub_url(repository, filename, revision=revision, endpoint="https://huggingface.co")
        token = self._token()
        deadline = time.monotonic() + 90

        def byte_range(start, length):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OSError("NUMERIC_ARCHIVE_METADATA_BUDGET")
            headers = build_hf_headers(token=token)
            headers.update(
                Range=f"bytes={start}-{start + length - 1}", **{"Accept-Encoding": "identity"}
            )
            with client.stream("GET", url, headers=headers, timeout=min(30, remaining)) as response:
                response.raise_for_status()
                expected = f"bytes {start}-{start + length - 1}/{size}"
                if response.status_code != 206 or response.headers.get("content-range") != expected:
                    raise ValueError("NUMERIC_ARCHIVE_RANGE_UNAVAILABLE")
                data = bytearray()
                for chunk in response.iter_bytes(chunk_size=32768):
                    if time.monotonic() > deadline or len(data) + len(chunk) > length:
                        raise ValueError("NUMERIC_ARCHIVE_RANGE_INVALID")
                    data.extend(chunk)
                return bytes(data)

        with httpx.Client(follow_redirects=True, timeout=30) as client:
            return inspect_archive(byte_range, size)

    def read_onnx_metadata(self, repository, revision, filename, sizes):
        from oryn_engine.onnx_graph import safe_path, scan_graph

        if not EXACT_REVISION_PATTERN.fullmatch(revision) or not safe_path(filename):
            raise ModelStoreError("MODEL_METADATA_INVALID", "ONNX metadata path is invalid.")
        url = hf_hub_url(repository, filename, revision=revision, endpoint="https://huggingface.co")
        token = self._token()
        cache_key = canonical_digest(
            [repository, revision, filename, sizes, hashlib.sha256(str(token).encode()).hexdigest()]
        )
        if cache_key in self._onnx_headers:
            return json.loads(json.dumps(self._onnx_headers[cache_key]))

        def byte_range(start, length):
            headers = build_hf_headers(token=token)
            headers["Range"] = f"bytes={start}-{start + length - 1}"
            with client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()
                expected = f"bytes {start}-{start + length - 1}/{sizes[filename]}"
                if response.status_code != 206 or response.headers.get("content-range") != expected:
                    raise ValueError("ONNX_RANGE_UNAVAILABLE")
                data = bytearray()
                for chunk in response.iter_bytes(chunk_size=32768):
                    data.extend(chunk)
                    if len(data) > length:
                        raise ValueError("ONNX_RANGE_INVALID")
                return bytes(data)

        with httpx.Client(follow_redirects=True, timeout=30) as client:
            result = scan_graph(byte_range, sizes[filename], filename, sizes)
        if len(self._onnx_headers) >= 16:
            self._onnx_headers.pop(next(iter(self._onnx_headers)))
        self._onnx_headers[cache_key] = result
        return json.loads(json.dumps(result))

    def read_ctranslate2_metadata(self, repository, revision, filename, size):
        from oryn_engine.ct2_weights import scan_directory, validate_whisper

        if not EXACT_REVISION_PATTERN.fullmatch(revision) or filename != "model.bin":
            raise ModelStoreError("MODEL_METADATA_INVALID", "CT2 metadata identity is invalid.")
        url = hf_hub_url(repository, filename, revision=revision, endpoint="https://huggingface.co")
        token = self._token()
        cache_key = canonical_digest(
            [
                "ct2",
                repository,
                revision,
                filename,
                size,
                hashlib.sha256(str(token).encode()).hexdigest(),
            ]
        )
        if cache_key in self._onnx_headers:
            return json.loads(json.dumps(self._onnx_headers[cache_key]))

        def byte_range(start, length):
            headers = build_hf_headers(token=token)
            headers["Range"] = f"bytes={start}-{start + length - 1}"
            with client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()
                expected = f"bytes {start}-{start + length - 1}/{size}"
                if response.status_code != 206 or response.headers.get("content-range") != expected:
                    raise ValueError("CT2_RANGE_UNAVAILABLE")
                data = bytearray()
                for chunk in response.iter_bytes(chunk_size=512):
                    data.extend(chunk)
                    if len(data) > length:
                        raise ValueError("CT2_RANGE_INVALID")
                return bytes(data)

        with httpx.Client(follow_redirects=True, timeout=30) as client:
            result = scan_directory(byte_range, size)
        validate_whisper(result)
        if len(self._onnx_headers) >= 16:
            self._onnx_headers.pop(next(iter(self._onnx_headers)))
        self._onnx_headers[cache_key] = result
        return json.loads(json.dumps(result))

    def read_gguf_metadata(self, repository, revision, filename, size, count=1, projector=False):
        """Read one GGUF header through bounded ranges; tensor data is never fetched."""
        from oryn_engine.runtime_packs.gguf import describe, inspect_remote_header, safe_member_name

        if (
            not EXACT_REVISION_PATTERN.fullmatch(revision)
            or not safe_member_name(filename)
            or type(size) is not int
            or type(count) is not int
            or type(projector) is not bool
        ):
            raise ModelStoreError("MODEL_METADATA_INVALID", "GGUF metadata identity is invalid.")
        url = hf_hub_url(repository, filename, revision=revision, endpoint="https://huggingface.co")
        token = self._token()
        cache_key = canonical_digest(
            [
                "gguf",
                repository,
                revision,
                filename,
                size,
                count,
                projector,
                hashlib.sha256(str(token).encode()).hexdigest(),
            ]
        )
        if cache_key in self._onnx_headers:
            return json.loads(json.dumps(self._onnx_headers[cache_key]))
        deadline = time.monotonic() + 90
        requests = 0

        def byte_range(start, length):
            nonlocal requests
            requests += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0 or requests > 32:
                raise GGUFError("GGUF header read exceeded its budget")
            headers = build_hf_headers(token=token)
            headers.update(
                Range=f"bytes={start}-{start + length - 1}", **{"Accept-Encoding": "identity"}
            )
            with client.stream("GET", url, headers=headers, timeout=min(30, remaining)) as response:
                response.raise_for_status()
                expected = f"bytes {start}-{start + length - 1}/{size}"
                if response.status_code != 206 or response.headers.get("content-range") != expected:
                    raise GGUFError("GGUF header range is unavailable")
                data = bytearray()
                for chunk in response.iter_bytes(chunk_size=65536):
                    if time.monotonic() > deadline or len(data) + len(chunk) > length:
                        raise GGUFError("GGUF header range is invalid")
                    data.extend(chunk)
                return bytes(data)

        try:
            with httpx.Client(follow_redirects=True, timeout=30) as client:
                result = describe(
                    inspect_remote_header(byte_range, size, count=count, projector=projector)
                )
        except GGUFError as exc:
            raise ModelStoreError(
                "MODEL_METADATA_INVALID", "GGUF header is invalid or outside the reader."
            ) from exc
        if len(self._onnx_headers) >= 16:
            self._onnx_headers.pop(next(iter(self._onnx_headers)))
        self._onnx_headers[cache_key] = result
        return json.loads(json.dumps(result))

    def read_tensor_metadata(self, repository: str, revision: str, filename: str) -> dict:
        return self._read_tensor_header(repository, revision, filename)

    def read_tensor_structure(self, repository: str, revision: str, filename: str) -> dict:
        return self._read_tensor_header(repository, revision, filename, structure=True)

    def _read_tensor_header(self, repository, revision, filename, *, structure=False) -> dict:
        """Read a bounded SafeTensors header, never download the tensor payload."""
        import struct

        if not EXACT_REVISION_PATTERN.fullmatch(revision) or not self._safe_header_name(filename):
            raise ModelStoreError("MODEL_METADATA_INVALID", "Tensor metadata path is invalid.")
        url = hf_hub_url(repository, filename, revision=revision, endpoint="https://huggingface.co")

        def byte_range(start: int, length: int) -> bytes:
            headers = build_hf_headers(token=self._token())
            headers["Range"] = f"bytes={start}-{start + length - 1}"
            with httpx.stream(
                "GET", url, headers=headers, follow_redirects=True, timeout=30
            ) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    if not response.headers.get("content-range", "").startswith(f"bytes {start}-"):
                        raise ValueError("Tensor header range differs from the request")
                    skip = 0
                else:
                    skip = start
                data = bytearray()
                for chunk in response.iter_bytes(chunk_size=min(65536, skip + length)):
                    if skip:
                        removed = min(skip, len(chunk))
                        skip -= removed
                        chunk = chunk[removed:]
                    data.extend(chunk[: length - len(data)])
                    if len(data) == length:
                        return bytes(data)
            raise ValueError("Tensor metadata is truncated")

        try:
            size = struct.unpack("<Q", byte_range(0, 8))[0]
            if not 2 <= size <= 4 * 1024**2:
                raise ValueError("Tensor metadata exceeds its bound")
            raw = byte_range(8, size)
            if structure:
                from oryn_engine.model_metadata import parse_metadata

                header = parse_metadata(raw, limit=4 * 1024**2)
            else:
                header = json.loads(raw)
            if not isinstance(header, dict) or len(header) > 100000:
                raise ValueError("Tensor header is invalid")
            metadata = header.get("__metadata__", {})
            # SafeTensors' optional metadata can be serialized as null. The
            # native reader treats that exactly like an absent metadata map.
            if metadata is None:
                metadata = {}
            if not isinstance(metadata, dict) or len(json.dumps(metadata)) > 16384:
                raise ValueError("Tensor metadata is invalid")
            result = {
                "metadata": metadata,
                "dtypes": sorted(
                    {
                        v["dtype"]
                        for k, v in header.items()
                        if k != "__metadata__"
                        and isinstance(v, dict)
                        and isinstance(v.get("dtype"), str)
                    }
                ),
            }
            if structure:
                import math

                widths = {"F16": 2, "BF16": 2, "F32": 4, "U32": 4}
                tensors, ranges = {}, []
                for name, tensor in header.items():
                    if name == "__metadata__":
                        continue
                    if (
                        not re.fullmatch(r"[A-Za-z0-9_.-]{1,256}", name)
                        or not isinstance(tensor, dict)
                        or set(tensor) != {"dtype", "shape", "data_offsets"}
                    ):
                        raise ValueError("Tensor structure is invalid")
                    shape, dtype = tensor.get("shape"), tensor.get("dtype")
                    offsets = tensor.get("data_offsets")
                    if (
                        not isinstance(dtype, str)
                        or dtype not in widths
                        or not isinstance(shape, list)
                        or not 0 <= len(shape) <= 6
                        or any(type(n) is not int or not 1 <= n <= 1_000_000 for n in shape)
                        or not isinstance(offsets, list)
                        or len(offsets) != 2
                        or any(type(n) is not int or not 0 <= n <= 2**50 for n in offsets)
                        or offsets[1] - offsets[0] != math.prod(shape) * widths[dtype]
                    ):
                        raise ValueError("Tensor structure is invalid")
                    tensors[name] = {"shape": shape, "dtype": dtype}
                    ranges.append(offsets)
                ranges.sort()
                if (
                    not ranges
                    or ranges[0][0]
                    or any(a[1] != b[0] for a, b in zip(ranges, ranges[1:], strict=False))
                ):
                    raise ValueError("Tensor structure ranges are invalid")
                result["tensors"] = tensors
                result["headerBytes"] = size + 8
                result["dataBytes"] = ranges[-1][1]
            return result
        except httpx.HTTPStatusError as exc:
            code = (
                "MODEL_ACCESS_RESTRICTED"
                if exc.response.status_code in {401, 403}
                else "MODEL_NOT_FOUND"
                if exc.response.status_code == 404
                else "MODEL_METADATA_UNAVAILABLE"
            )
            raise ModelStoreError(code, "The tensor metadata artifact is unavailable.") from exc
        except httpx.HTTPError as exc:
            raise ModelStoreError(
                "MODEL_METADATA_UNAVAILABLE", "The tensor metadata request failed."
            ) from exc
        except ValueError as exc:
            raise ModelStoreError(
                "MODEL_METADATA_INVALID", "Tensor metadata could not be read."
            ) from exc

    @staticmethod
    def _safe_header_name(filename: str) -> bool:
        from oryn_engine.providers import safe_relative

        return safe_relative(filename) and filename.endswith(".safetensors")

    def cached_file(self, repo_id: str, filename: str, revision: str, cache_dir: str) -> str | None:
        """Locate exact cached bytes without downloading or granting installation trust."""
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(repo_id, filename, cache_dir=cache_dir, revision=revision)
        return path if isinstance(path, str) else None

    def download(
        self,
        repo_id: str,
        filename: str,
        revision: str,
        cache_dir: str,
        progress: DownloadProgress | None = None,
    ) -> str:
        class ReportingTqdm(tqdm):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._oryn_transferred = int(kwargs.get("initial", self.n) or 0)
                if progress is not None:
                    progress(
                        self._oryn_transferred,
                        int(self.total) if self.total is not None else None,
                    )

            def update(self, n: int | float | None = 1) -> bool | None:
                displayed = super().update(n)
                self._oryn_transferred += int(n or 0)
                if progress is not None:
                    progress(
                        self._oryn_transferred,
                        int(self.total) if self.total is not None else None,
                    )
                return displayed

        try:
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                cache_dir=cache_dir,
                tqdm_class=ReportingTqdm,
                token=self._token(),
            )
        except GatedRepoError as exc:
            raise ModelStoreError(
                "MODEL_ACCESS_RESTRICTED",
                "The model host requires approval or authentication for this model.",
            ) from exc
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in {401, 403}:
                raise ModelStoreError(
                    "MODEL_ACCESS_RESTRICTED",
                    "The model host requires approval or authentication for this model.",
                ) from exc
            raise


@dataclass(frozen=True)
class ModelFile:
    name: str
    size: int
    blob_id: str


@dataclass
class InstallOperation:
    operation_id: str
    installation_id: str
    model: dict[str, str]
    cancel: threading.Event
    logical_size_bytes: int
    estimated_resident_bytes: int


class ModelStore:
    def __init__(
        self,
        root: Path,
        event_sink: ModelEventSink | None = None,
        hub: HubClient | None = None,
        runtime_available: bool | None = None,
        runtime_packs: RuntimePackRegistry | None = None,
        runtime_planning_state_provider: RuntimePlanningStateProvider | None = None,
    ) -> None:
        self.root = root.resolve()
        self.cache_root = self.root / "cache"
        self.partial_root = self.root / "partial"
        self.registry_path = self.root / "installations.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(self.partial_root, ignore_errors=True)
        self.partial_root.mkdir(parents=True, exist_ok=True)
        self._event_sink = event_sink
        self._runtime_state_provider: RuntimeStateProvider | None = None
        from oryn_engine.model_composition import CompositionHub

        self._hub = CompositionHub(hub or HuggingFaceClient())
        self._runtime_packs = runtime_packs or builtin_registry()
        self._runtime_foundations = builtin_foundation_registry()
        self._capability_registry = compile_registry((*framework_sources(), *native_sources()))
        self._runtime_routing = RuntimeRoutingService(
            RoutingEvidenceStore(self.root / "routing-evidence.json")
        )
        self._runtime_planning_state_provider = runtime_planning_state_provider or (
            lambda: (RuntimePolicy.BALANCED.value, int(psutil.virtual_memory().available))
        )
        self._pack_runtime_locator = PackRuntimeLocator()
        self._provider_plans: dict[str, dict] = {}
        self._execution_offers: dict[str, dict] = {}
        self._inspection_contexts: dict[str, tuple[dict, dict, list]] = {}
        self._provider_process = None
        self._runtime_available = (
            runtime_available
            if runtime_available is not None
            else importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("transformers") is not None
        )
        self._lock = threading.RLock()
        self._operations: dict[str, InstallOperation] = {}
        self._source_cancellations: dict[str, threading.Event] = {}
        self._records = self._load_registry()
        if self._migrate_runtime_plans():
            self._save_registry()
        for pending in [
            record
            for record in list(self._records)
            if record.get("installationState") == "removing"
        ]:
            self.complete_remove(pending["installationId"])

    @property
    def runtime_routing(self) -> RuntimeRoutingService:
        return self._runtime_routing

    def set_event_sink(self, sink: ModelEventSink) -> None:
        self._event_sink = sink

    def set_runtime_state_provider(self, provider: RuntimeStateProvider) -> None:
        self._runtime_state_provider = provider

    def installed_record(self, model: dict[str, Any]) -> dict[str, Any]:
        repository, revision = self._validate_model_ref(model)
        if not EXACT_REVISION_PATTERN.fullmatch(revision):
            raise ModelStoreError(
                "MODEL_REFERENCE_INVALID", "Residency requires an exact model revision."
            )
        key = self._model_key(model)
        with self._lock:
            record = next(
                (item for item in self._records if self._model_key(item["model"]) == key), None
            )
            if record is None:
                raise ModelStoreError("MODEL_NOT_INSTALLED", "The model is not installed.")
            if record.get("installationState") != "installed":
                raise ModelStoreError(
                    "MODEL_REMOVAL_PENDING", "The model installation is being removed."
                )
            record = dict(record)
        if self._provider_requires_reinspection(record):
            self.install(f"revalidate_{uuid.uuid4().hex}", model)
            return self.installed_record(model)
        if record.get("capabilityRegistryDigest") not in {
            None,
            self._capability_registry.digest,
        }:
            raise ModelStoreError(
                "CAPABILITY_REGISTRY_CHANGED",
                "Reinspect and reinstall this model after the capability update.",
            )
        if record.get("runtimeBuildDigest") not in {None, runtime_build_digest()}:
            raise ModelStoreError(
                "RUNTIME_BUILD_CHANGED", "Reinspect and reinstall after the runtime update."
            )
        return dict(record) | {"cacheRoot": str(self.cache_root)}

    def set_runtime_override(
        self, model: dict[str, Any], candidate_id: str | None
    ) -> dict[str, Any]:
        repository, revision = self._validate_model_ref(model)
        if not EXACT_REVISION_PATTERN.fullmatch(revision):
            raise ModelStoreError(
                "MODEL_REFERENCE_INVALID", "A runtime override requires an exact revision."
            )
        if candidate_id is not None and not re.fullmatch(r"candidate-[0-9a-f]{24}", candidate_id):
            raise ModelStoreError("RUNTIME_OVERRIDE_INVALID", "Runtime candidate ID is invalid.")
        key = self._model_key(model)
        with self._lock:
            record = next(
                (
                    item
                    for item in self._records
                    if item.get("installationState") == "installed"
                    and self._model_key(item["model"]) == key
                ),
                None,
            )
            if record is None:
                raise ModelStoreError("MODEL_NOT_INSTALLED", "The model is not installed.")
            current = record.get("runtimePlan")
            if not isinstance(current, dict):
                raise ModelStoreError(
                    "RUNTIME_OVERRIDE_UNAVAILABLE", "The installed model has no runtime plan."
                )
            automatic = record.get("automaticRuntimePlan")
            if not isinstance(automatic, dict):
                automatic = json.loads(json.dumps(current))
            if candidate_id is None:
                record["runtimePlan"] = automatic
                record.pop("runtimeOverrideCandidateId", None)
                record.pop("automaticRuntimePlan", None)
                self._apply_plan_runtime_pack(record, automatic)
            else:
                candidates = [automatic.get("selected"), *automatic.get("fallbacks", [])]
                selected = next(
                    (
                        item
                        for item in candidates
                        if isinstance(item, dict) and item.get("candidateId") == candidate_id
                    ),
                    None,
                )
                if selected is None:
                    raise ModelStoreError(
                        "RUNTIME_OVERRIDE_UNAVAILABLE",
                        "The runtime candidate is not admitted for this installed model.",
                    )
                self._assert_candidate_files_installed(record, selected)
                fallbacks: list[dict[str, Any]] = []
                plan_key = {
                    "candidates": [candidate_id] + [item["candidateId"] for item in fallbacks],
                    "logicalModel": key,
                    "override": candidate_id,
                    "policy": automatic.get("policy", RuntimePolicy.BALANCED.value),
                }
                digest = hashlib.sha256(
                    json.dumps(plan_key, separators=(",", ":"), sort_keys=True).encode()
                ).hexdigest()
                plan = {
                    "planId": f"plan-{digest[:24]}",
                    "policy": automatic.get("policy", RuntimePolicy.BALANCED.value),
                    "reasonCodes": ["USER_OVERRIDE"],
                    "selected": selected,
                    "fallbacks": fallbacks,
                    "rejected": automatic.get("rejected", []),
                }
                record["automaticRuntimePlan"] = automatic
                record["runtimeOverrideCandidateId"] = candidate_id
                record["runtimePlan"] = plan
                self._apply_plan_runtime_pack(record, plan)
            self._refresh_installed_generation_profile(record)
            self._save_registry()
            return dict(record["runtimePlan"])

    def _assert_candidate_files_installed(
        self, record: dict[str, Any], candidate: dict[str, Any]
    ) -> None:
        if candidate.get("candidateId") in record.get("executionOffers", {}):
            from oryn_engine.provider_discovery import record_for_offer

            record = record_for_offer(record, candidate)
            if candidate.get("variantId") == "oryn.builtin.transformers":
                return
        variant_id = candidate.get("variantId")
        from oryn_engine.providers import provider_variant, verify_plan

        if provider_variant(variant_id):
            bound = record.get("providerPlans", {}).get(candidate.get("candidateId"))
            if not isinstance(bound, dict):
                raise ModelStoreError(
                    "RUNTIME_OVERRIDE_UNAVAILABLE", "Provider plan is unavailable."
                )
            pack = self._runtime_packs.active_pack(bound["packId"])
            runtime = self._pack_runtime_locator.resolve(pack)
            if runtime is None:
                raise ModelStoreError(
                    "RUNTIME_OVERRIDE_UNAVAILABLE", "Provider runtime is unavailable."
                )
            verify_plan(bound, pack, runtime, record)
            return
        compatibility = next(
            (item for item in self._runtime_packs.compatibility if item.manifest_id == variant_id),
            None,
        )
        if generated_pack_id(variant_id):
            pack = self._runtime_packs.active_pack(generated_pack_id(variant_id))
            if record.get("runtimePack", {}).get("manifestDigest") != pack.digest:
                raise ModelStoreError(
                    "RUNTIME_OVERRIDE_UNAVAILABLE", "The selected runtime changed."
                )
            return
        if compatibility is None:
            raise ModelStoreError(
                "RUNTIME_OVERRIDE_UNAVAILABLE", "The runtime variant is no longer trusted."
            )
        files = [
            ModelFile(str(item.get("name")), int(item.get("size", 0)), str(item.get("blob_id", "")))
            for item in record.get("files", [])
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        ]
        if not self._required_patterns_present(files, compatibility):
            raise ModelStoreError(
                "RUNTIME_OVERRIDE_UNAVAILABLE",
                "The alternate runtime requires model files that are not installed.",
            )

    def _apply_plan_runtime_pack(self, record: dict[str, Any], plan: dict[str, Any]) -> None:
        selected = plan.get("selected")
        variant_id = selected.get("variantId") if isinstance(selected, dict) else None
        from oryn_engine.providers import provider_variant

        if isinstance(selected, dict) and selected.get("candidateId") in record.get(
            "executionOffers", {}
        ):
            from oryn_engine.provider_discovery import record_for_offer

            record.update(record_for_offer(record, selected))

        if variant_id == "oryn.builtin.transformers":
            self._assert_candidate_files_installed(record, selected)
            return

        if provider_variant(variant_id):
            self._assert_candidate_files_installed(record, selected)
            bound = record["providerPlans"][selected["candidateId"]]
            record["runtimePack"] = {
                "packId": bound["packId"],
                "packVersion": bound["packVersion"],
                "runnerId": bound["runnerId"],
                "taskContract": bound["taskContract"],
                "compatibilityLevel": "native-declarative",
                "manifestDigest": bound["packDigest"],
            }
            record["generationProfile"] = bound["generationProfile"]
            return
        if generated_pack_id(variant_id):
            self._assert_candidate_files_installed(record, selected)
            return
        compatibility = next(
            (item for item in self._runtime_packs.compatibility if item.manifest_id == variant_id),
            None,
        )
        if compatibility is None:
            raise ModelStoreError(
                "RUNTIME_OVERRIDE_UNAVAILABLE", "The runtime variant is no longer trusted."
            )
        record["runtimePack"] = {
            "packId": compatibility.pack_id,
            "packVersion": compatibility.pack_version,
            "runnerId": compatibility.runner_id,
            "taskContract": compatibility.task_contract,
            "compatibilityLevel": compatibility.level,
            "manifestDigest": compatibility.digest,
            **(
                {"trustedMetadata": self._public_trusted_metadata(compatibility.trusted_metadata)}
                if compatibility.trusted_metadata
                else {}
            ),
        }

    @classmethod
    def model_key(cls, model: dict[str, Any]) -> str:
        repository, revision = cls._validate_model_ref(model)
        return cls._model_key(model)

    @classmethod
    def validate_operation_id(cls, operation_id: str) -> None:
        cls._validate_operation_id(operation_id)

    def inspect(
        self,
        model: dict[str, Any],
        requirements: dict | None = None,
        recommendation: dict | None = None,
        *,
        source_context: dict | None = None,
        _source_cancel_id: str | None = None,
    ) -> dict[str, Any]:
        if source_context is None:
            return self._inspect(model, requirements, recommendation)
        from oryn_engine.reviewed_implementations import validate_hf_context
        from oryn_engine.source_inspection import bounded_inspection, inspection_operation_id

        repository, _ = self._validate_model_ref(model)
        validate_hf_context(source_context, repository)
        operation_id = inspection_operation_id(source_context)
        cancellation = threading.Event()
        keys = {operation_id, *([_source_cancel_id] if _source_cancel_id else [])}
        with self._lock:
            for key in keys:
                previous = self._source_cancellations.get(key)
                if previous is not None:
                    previous.set()
                self._source_cancellations[key] = cancellation
        try:
            return bounded_inspection(
                lambda: self._inspect(
                    model,
                    requirements,
                    recommendation,
                    source_context=source_context,
                    cancellation=cancellation,
                ),
                cancellation,
            )
        finally:
            with self._lock:
                for key in keys:
                    if self._source_cancellations.get(key) is cancellation:
                        self._source_cancellations.pop(key, None)

    def _inspect(
        self,
        model: dict[str, Any],
        requirements: dict | None = None,
        recommendation: dict | None = None,
        *,
        source_context: dict | None = None,
        cancellation: threading.Event | None = None,
    ) -> dict[str, Any]:
        from dataclasses import replace

        from oryn_protocol.task_contracts import (
            REGISTRY_DIGEST,
            TASK_CONTRACTS,
            requirement_failures,
            validate_requirements,
        )

        from oryn_engine.provider_discovery import BackendProvider, ExecutionProviderRegistry
        from oryn_engine.task_profiles import profile_for_execution

        requirements = requirements or {}
        try:
            validate_requirements(requirements)
        except ValueError as exc:
            raise ModelStoreError(
                "TASK_REQUIREMENTS_INVALID", "Task requirements are invalid."
            ) from exc
        repository, revision = self._validate_model_ref(model)
        if source_context is not None:
            from oryn_engine.reviewed_implementations import validate_hf_context

            try:
                validate_hf_context(source_context, repository)
            except ValueError as exc:
                raise ModelStoreError("MODEL_SECURITY_BLOCKED", str(exc)) from exc
        if EXACT_REVISION_PATTERN.fullmatch(revision):
            installed = self._installed_inspection(repository, revision, model.get("files"))
            if (
                installed is not None
                and installed.get("taskProfile")
                and not requirement_failures(installed["taskProfile"], requirements)
                and not {
                    "RUNTIME_BUILD_CHANGED",
                    "CAPABILITY_REGISTRY_CHANGED",
                    "TASK_CONTRACT_REGISTRY_CHANGED",
                }.intersection(installed.get("blockingReasons", []))
            ):
                return self._route_installed(installed, requirements, recommendation)
        try:
            info_reader = (
                self._hub.source_model_info if source_context is not None else self._hub.model_info
            )
            info = info_reader(repository, revision, files_metadata=True)
        except ModelStoreError:
            raise
        except Exception as exc:
            raise ModelStoreError(
                "MODEL_NOT_FOUND",
                "The requested Hugging Face model revision could not be inspected.",
                retryable=True,
            ) from exc
        exact_revision = getattr(info, "sha", None)
        if not isinstance(exact_revision, str) or not EXACT_REVISION_PATTERN.fullmatch(
            exact_revision
        ):
            raise ModelStoreError("MODEL_METADATA_INVALID", "Model revision metadata is invalid.")
        config = getattr(info, "config", None) or {}
        if not isinstance(config, dict):
            raise ModelStoreError(
                "MODEL_METADATA_INVALID", "Model configuration metadata is invalid."
            )
        self._validate_metadata_value(config)
        files = self._validate_files(
            getattr(info, "siblings", None), f"{repository}@{exact_revision}"
        )
        source_recognition = None
        reviewed_admission = None
        if source_context is not None and config.get("auto_map"):
            from oryn_engine.native_lfm3 import (
                ARCHITECTURE,
                CONTEXT_LIMIT,
                representation,
                validate_config,
            )
            from oryn_engine.reviewed_implementations import inspect_repository, validate_evidence

            try:
                source_recognition = inspect_repository(
                    config,
                    files,
                    lambda name, limit: self._hub.read_source(
                        repository, exact_revision, name, limit
                    ),
                    cancellation=cancellation,
                )
                source_recognition["representation"] = representation(config)
                if source_recognition["state"] == "recognized":
                    source_recognition["configDigest"] = canonical_digest(config)
                    p = validate_evidence(source_recognition, require_admission=False)
                    if (
                        source_recognition["representation"] in p["representations"]
                        and source_recognition["representation"] == "bf16-cpu"
                    ):
                        validate_config(config)
                        config = {k: v for k, v in config.items() if k != "auto_map"}
                        config["architectures"] = [ARCHITECTURE]
                        config["max_position_embeddings"] = CONTEXT_LIMIT
                        source_recognition["admitted"] = True
                        reviewed_admission = source_recognition
                        info.config = config
            except (ValueError, OSError, TypeError) as exc:
                source_recognition = source_recognition or {"state": "inspection-unavailable"}
                source_recognition["admitted"] = False
                if str(exc) == "SOURCE_SANDBOX_UNAVAILABLE":
                    source_recognition["reason"] = "SOURCE_SANDBOX_UNAVAILABLE"
        architectures = config.get("architectures", [])
        if not isinstance(architectures, list) or any(
            not isinstance(a, str) or not 0 < len(a) <= 128 for a in architectures
        ):
            raise ModelStoreError(
                "MODEL_METADATA_INVALID", "Model architectures metadata is invalid."
            )
        requested_task = requirements.get("task")
        if not requested_task and requirements.get("contract"):
            requested_task = TASK_CONTRACTS[requirements["contract"]["id"]]["task"]
        task = requested_task or getattr(info, "pipeline_tag", None)
        if task is not None and (not isinstance(task, str) or len(task) > 128):
            raise ModelStoreError("MODEL_METADATA_INVALID", "Model task metadata is invalid.")
        from oryn_engine.generated_capabilities import normalized_transformers_config
        from oryn_engine.tabular_tasks import normalized_config as normalized_tabular_config
        from oryn_engine.timm_tasks import normalized_config

        try:
            config = normalized_transformers_config(
                normalized_tabular_config(normalized_config(config, task), task), task
            )
        except (ValueError, TypeError) as exc:
            raise ModelStoreError(
                "MODEL_METADATA_INVALID", "Native metadata is outside its reviewed binding."
            ) from exc
        architectures = config.get("architectures", architectures)
        info.config = config
        from oryn_engine.model_inspection import effective_task

        task = effective_task("transformers", task, config)
        if task == "sentence-similarity":
            task = "feature-extraction"
        matches = self._runtime_packs.match(
            repository=repository, revision=exact_revision, architectures=architectures, task=task
        )
        providers = []
        from oryn_engine.model_composition import binding as composition_binding

        composition = composition_binding(repository, exact_revision)
        for declared in self._runtime_packs.packs:
            if reviewed_admission is not None and declared.pack_id != "oryn.transformers-pytorch":
                continue
            if composition is not None and declared.pack_id != "oryn.diffusers-pytorch":
                continue
            if self._runtime_packs.active_pack(declared.pack_id).version != declared.version:
                continue
            variants = [m.manifest_id for m in matches if m.pack_id == declared.pack_id]
            # Each signed export is inspected; generated/native discovery also gets a turn.
            for variant in variants or [None]:
                providers.append(
                    BackendProvider(
                        declared.pack_id,
                        lambda offers, backend=declared.pack_id, variant=variant: (
                            self._inspect_backend(
                                model,
                                backend=backend,
                                info_value=info,
                                requested_task=task,
                                offers=offers,
                                manifest_id=variant,
                                reviewed_source=reviewed_admission,
                            )
                        ),
                    )
                )
        discovered = ExecutionProviderRegistry().inspect(providers)
        # Generation metadata is data only; absent alignment heads remain unknown.
        config = dict(config)
        if task == "automatic-speech-recognition" and any(
            f.name == "generation_config.json" for f in files
        ):
            with contextlib.suppress(AttributeError, OSError, ValueError):
                config["_orynGenerationConfig"] = self._hub.read_json(
                    repository, exact_revision, "generation_config.json"
                )
        projected, candidates, bound_offers, provider_plans = {}, {}, {}, {}
        for candidate, raw_profile, provider_plan in discovered.offers:
            profile = dict(raw_profile)
            # Provider flags cannot contradict the common descriptor's hard axes.
            # In particular, a valid family config does not replace a missing
            # signed architecture/task tuple when the planner ranks all offers.
            axes = profile.get("eligibility") or {}

            def allowed(axis, state, axes=axes):
                return axis not in axes or axes[axis].get("state") == state

            candidate = replace(
                candidate,
                task_compatible=candidate.task_compatible
                and allowed("runtime", "supported")
                and allowed("task", "supported"),
                files_complete=candidate.files_complete and allowed("artifacts", "safe"),
                access_granted=candidate.access_granted and allowed("access", "available"),
                security_allowed=candidate.security_allowed
                and (axes.get("conformance") or {}).get("state") != "failed",
            )
            if model.get("files"):
                if set(model["files"]) != {f["name"] for f in profile.get("selectedFiles", [])}:
                    continue
                profile["model"] = {**profile["model"], "files": sorted(model["files"])}
                candidate = replace(candidate, logical_model_id=self._model_key(profile["model"]))
                if provider_plan is not None:
                    provider_plan = {
                        **{k: v for k, v in provider_plan.items() if k != "digest"},
                        "model": profile["model"],
                    }
                    provider_plan["digest"] = canonical_digest(provider_plan)

            try:
                task_profile = profile_for_execution(
                    profile, config, contract_id=requirements.get("contract", {}).get("id")
                )
                if task_profile is None:
                    continue
                failures = requirement_failures(task_profile, requirements)
            except (ValueError, TypeError, KeyError):
                discovered.diagnostics.append(
                    {
                        "providerId": profile.get("runtimePack", {}).get(
                            "packId", candidate.identity.adapter_id
                        ),
                        "code": "PROVIDER_INSPECTION_FAILED",
                    }
                )
                continue
            profile["taskProfile"] = task_profile
            profile.pop("_chatThinkingSupport", None)
            if (
                task_profile["contract"]["id"] == "chat-v2"
                and profile.get("runtimePack", {}).get("taskContract") == "chat-v1"
            ):
                profile["runtimePack"] = {**profile["runtimePack"], "taskContract": "chat-v2"}
            candidate = replace(
                candidate,
                task_contract=task_profile["contract"]["id"],
                task_compatible=candidate.task_compatible and not failures,
                equivalence_class="exact-model",
                integrity_digest=canonical_digest(
                    {
                        "native": candidate.integrity_digest,
                        "profile": task_profile,
                        "registry": REGISTRY_DIGEST,
                    }
                ),
            )
            key = candidate.identity.candidate_id
            if key in candidates:
                continue
            if failures:
                profile["executable"] = False
                profile["executableBackends"] = []
                profile["blockingReasons"] = list(
                    dict.fromkeys([*profile.get("blockingReasons", []), *failures])
                )
            profile["executableBackends"] = (
                [candidate.identity.backend] if profile.get("executable") else []
            )
            profile["estimatedMemoryBytes"] = candidate.estimated_memory_bytes
            projected[key] = profile
            candidates[key] = candidate
            selected_names = {f["name"] for f in profile.get("selectedFiles", [])}
            private = {
                "schemaVersion": 1,
                "model": profile["model"],
                "candidateId": key,
                "runtime": {
                    "foundation": {
                        "id": candidate.identity.foundation_id,
                        "version": candidate.identity.foundation_version,
                    },
                    "adapter": {
                        "id": candidate.identity.adapter_id,
                        "version": candidate.identity.adapter_version,
                    },
                    "variantId": candidate.identity.variant_id,
                    "backend": candidate.identity.backend,
                    "precision": candidate.identity.precision,
                    "quantization": candidate.identity.quantization,
                    "estimatedMemoryBytes": candidate.estimated_memory_bytes,
                },
                "registryDigest": REGISTRY_DIGEST,
                "runtimeBuildDigest": runtime_build_digest(),
                "taskProfile": task_profile,
                "runtimePack": profile.get("runtimePack"),
                "runner": profile.get("runner"),
                "technicalCapabilities": profile["technicalCapabilities"],
                "generationProfile": profile.get("generationProfile"),
                **(
                    {
                        "sourceRecognition": profile["sourceRecognition"],
                        "admissionScope": "huggingface-page",
                    }
                    if profile.get("sourceRecognition")
                    else {}
                ),
                "files": [
                    {"name": f.name, "sizeBytes": f.size, "blobId": f.blob_id}
                    for f in files
                    if f.name in selected_names
                ],
            }
            bound_offers[key] = private | {"digest": canonical_digest(private)}
            if provider_plan is not None:
                provider_plans[key] = provider_plan
        policy, available_memory = self._runtime_planning_state_provider()
        hardware = self._hardware_info()
        backends = tuple(
            dict.fromkeys(
                [
                    hardware.preferred_backend,
                    *(["metal", "mlx", "coreml"] if hardware.unified_memory else []),
                    *self._executable_backends(),
                ]
            )
        )
        # Offers for a different task contract cannot be fallback routes.
        contract = requirements.get("contract", {}).get("id")
        if not contract:
            admissible = [
                c
                for c in candidates.values()
                if c.task_compatible
                and c.files_complete
                and c.security_allowed
                and c.runtime_available
            ]
            contract = (
                (admissible or list(candidates.values()))[0].task_contract if candidates else None
            )
        selected_candidates = [c for c in candidates.values() if c.task_contract == contract]
        plan = None
        if selected_candidates:
            try:
                plan = self._runtime_routing.plan(
                    selected_candidates,
                    context=RoutingContext.current(
                        self._capability_registry.digest,
                        {"requirements": requirements, "registry": REGISTRY_DIGEST},
                    ),
                    available_memory_bytes=available_memory,
                    available_backends=backends,
                    policy=RuntimePolicy(policy),
                    recommendation_scores={
                        c.identity.candidate_id: len((recommendation or {}).get("engines", []))
                        - (recommendation or {})
                        .get("engines", [])
                        .index(
                            (projected[c.identity.candidate_id].get("runtimePack") or {}).get(
                                "packId", "oryn.transformers-pytorch"
                            )
                        )
                        for c in selected_candidates
                        if (projected[c.identity.candidate_id].get("runtimePack") or {}).get(
                            "packId", "oryn.transformers-pytorch"
                        )
                        in (recommendation or {}).get("engines", [])
                    },
                    recommendation_digest=(recommendation or {}).get("snapshotDigest", ""),
                )
            except RuntimePlanError as exc:
                # If memory is the only blocker for a candidate, report its
                # requirements instead of an unavailable runtime's estimates.
                reported_candidate = next(
                    (
                        evaluation.candidate
                        for evaluation in exc.evaluations
                        if evaluation.rejection_reasons == ("MEMORY_INSUFFICIENT",)
                    ),
                    selected_candidates[0],
                )
                profile = dict(projected[reported_candidate.identity.candidate_id])
                profile["executable"] = False
                profile["executableBackends"] = []
                profile["runtimePlanFailure"] = {
                    "code": exc.code,
                    "rejected": [
                        {
                            "candidateId": e.candidate.identity.candidate_id,
                            "reasonCodes": list(e.rejection_reasons),
                        }
                        for e in exc.evaluations
                    ][:16],
                }
                profile["blockingReasons"] = list(
                    dict.fromkeys([*profile.get("blockingReasons", []), exc.code])
                )
            else:
                profile = dict(projected[plan.selected.identity.candidate_id])
                profile["executable"] = True
                profile["executableBackends"] = [plan.selected.identity.backend]
                profile["runtimePlan"] = plan.trusted_summary()
                evidence_context = RoutingContext.current(
                    self._capability_registry.digest,
                    {"requirements": requirements, "registry": REGISTRY_DIGEST},
                )
                profile["recommendationContext"] = {
                    "runtime": canonical_digest(
                        [
                            plan.selected.identity.candidate_id,
                            runtime_build_digest(),
                            plan.selected.integrity_digest,
                        ]
                    ),
                    "hardware": canonical_digest(
                        [evidence_context.device, evidence_context.system]
                    ),
                    "workload": evidence_context.task_settings_digest,
                }
                evidence = self._runtime_routing.measurement(
                    plan.selected,
                    evidence_context,
                )
                if evidence:
                    profile["recommendationEvidence"] = evidence

        else:
            useful = [
                p for p in discovered.profiles if p.get("runtimePack") or p.get("eligibility")
            ]
            profile = (
                dict((useful or discovered.profiles)[0])
                if discovered.profiles
                else {
                    "model": {
                        "provider": "huggingface",
                        "repository": repository,
                        "revision": exact_revision,
                    },
                    "exactRevision": exact_revision,
                    "executable": False,
                    "executableBackends": [],
                    "requiresCustomCode": bool(config.get("auto_map")),
                    "technicalCapabilities": self._technical_capabilities(
                        task, architectures, config
                    ),
                    "blockingReasons": ["RUNNER_UNSUPPORTED"],
                }
            )
            try:
                task_profile = profile_for_execution(
                    profile, config, contract_id=requirements.get("contract", {}).get("id")
                )
            except (ValueError, TypeError, KeyError):
                task_profile = None
                profile["executable"] = False
                profile["executableBackends"] = []
                profile["blockingReasons"] = list(
                    dict.fromkeys(
                        [*profile.get("blockingReasons", []), "TASK_CONFIGURATION_UNSUPPORTED"]
                    )
                )
            if task_profile:
                profile["taskProfile"] = task_profile
                if (
                    task_profile["contract"]["id"] == "chat-v2"
                    and profile.get("runtimePack", {}).get("taskContract") == "chat-v1"
                ):
                    profile["runtimePack"] = {**profile["runtimePack"], "taskContract": "chat-v2"}
            failures = requirement_failures(task_profile, requirements)
            if failures:
                profile["executable"] = False
                profile["executableBackends"] = []
                profile["blockingReasons"] = list(
                    dict.fromkeys([*profile.get("blockingReasons", []), *failures])
                )
        if model.get("files"):
            profile["model"] = {**profile["model"], "files": sorted(model["files"])}
            if (
                set(model["files"]) != {f["name"] for f in profile.get("selectedFiles", [])}
                or not selected_candidates
            ):
                profile["executable"] = False
                profile["executableBackends"] = []
                profile["blockingReasons"] = list(
                    dict.fromkeys([*profile.get("blockingReasons", []), "MODEL_SUBSET_INCOMPLETE"])
                )
        card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
        if hasattr(card, "to_dict"):
            card = card.to_dict()
        if isinstance(card, dict):
            bases = card.get("base_model", [])
            bases = [bases] if isinstance(bases, str) else bases
            if isinstance(bases, list):
                profile["lineageDeclarations"] = [
                    v for v in bases if isinstance(v, str) and REPOSITORY_PATTERN.fullmatch(v)
                ][:16]
        if discovered.diagnostics:
            profile["providerDiagnostics"] = discovered.diagnostics[:32]
        if source_recognition is not None:
            profile["sourceRecognition"] = source_recognition
            if source_recognition.get("state") == "recognized":
                source_recognition["admitted"] = reviewed_admission is not None
                if source_recognition["admitted"]:
                    profile["admissionScope"] = "huggingface-page"
                elif not profile.get("executable"):
                    profile["executable"] = False
                    profile["executableBackends"] = []
                    profile["blockingReasons"] = [
                        reason
                        for reason in profile.get("blockingReasons", [])
                        if reason != "CUSTOM_CODE_REQUIRED"
                    ] + ["REVIEWED_REPRESENTATION_UNSUPPORTED"]
                    profile["requiresCustomCode"] = False
        profile.pop("_chatThinkingSupport", None)
        profile["routingFacts"] = self._routing_facts(bound_offers)
        model_key = self._model_key(profile["model"])
        with self._lock:
            installed_record = next(
                (r for r in self._records if self._model_key(r["model"]) == model_key), None
            )
            if installed_record is not None:
                installed_files = {f["name"] for f in installed_record.get("files", [])}
                profile["installationRequired"] = bool(
                    {f["name"] for f in profile.get("selectedFiles", [])} - installed_files
                ) or installed_record.get("taskProfile") != profile.get("taskProfile")
                profile["runtimePreparationRequired"] = (
                    installed_record.get("runtimeBuildDigest") != runtime_build_digest()
                    or installed_record.get("capabilityRegistryDigest")
                    != self._capability_registry.digest
                    or installed_record.get("runtimePlan", {}).get("selected")
                    != profile.get("runtimePlan", {}).get("selected")
                )
        with self._lock:
            for cache in (self._provider_plans, self._execution_offers):
                if len(cache) >= 128:
                    cache.pop(next(iter(cache)))
            self._provider_plans[model_key] = provider_plans
            self._execution_offers[model_key] = bound_offers
            if len(self._inspection_contexts) >= 128:
                self._inspection_contexts.pop(next(iter(self._inspection_contexts)))
            self._inspection_contexts[canonical_digest(profile)] = (
                provider_plans,
                bound_offers,
                [asdict(c) for c in selected_candidates],
            )
        return profile

    @staticmethod
    def _routing_facts(offers: dict) -> dict:
        facts = []
        for offer in offers.values():
            pack = offer.get("runtimePack") or {}
            engine = pack.get("packId") or "oryn.transformers-pytorch"
            version = pack.get("packVersion") or offer["runtime"]["foundation"]["version"]
            item = {
                "engine": engine,
                "engineVersion": version,
                "features": sorted(
                    k for k, v in offer["taskProfile"]["features"].items() if v == "supported"
                ),
            }
            if item not in facts:
                facts.append(item)
        return {"version": 1, "offers": facts[:256]}

    def _route_installed(self, profile: dict, requirements: dict, recommendation: dict | None):
        """Replan verified installed offers without Hub access or mutating active sessions."""
        from oryn_protocol.task_contracts import REGISTRY_DIGEST, requirement_failures

        from oryn_engine.provider_discovery import record_for_offer

        key = self._model_key(profile["model"])
        with self._lock:
            record = next(r for r in self._records if self._model_key(r["model"]) == key)
            offers = record.get("executionOffers", {})
            profile["routingFacts"] = self._routing_facts(offers)
            if (
                not recommendation
                or not profile["executable"]
                or not record.get("routingCandidates")
            ):
                return profile
            usage = self._runtime_state(profile["model"])
            if (
                record.get("runtimeOverrideCandidateId")
                or any(
                    usage.get(k, 0)
                    for k in ("activeSessions", "activeJobs", "runningJobs", "queuedJobs")
                )
                or usage.get("residencyState") == "loaded"
            ):
                return profile
            candidates = []
            projected = {}
            for raw in record["routingCandidates"]:
                candidate = RuntimeCandidate(
                    **{**raw, "identity": RuntimeIdentity(**raw["identity"])}
                )
                candidate_id = candidate.identity.candidate_id
                offer = offers.get(candidate_id)
                if offer is None:
                    continue
                summary = {"candidateId": candidate_id, **offer["runtime"]}
                try:
                    bound = record_for_offer(record, summary)
                    self._assert_candidate_files_installed(record, summary)
                except Exception:
                    continue
                pack_id = (offer.get("runtimePack") or {}).get(
                    "packId", "oryn.transformers-pytorch"
                )
                available = (
                    self._runtime_available
                    if pack_id == "oryn.transformers-pytorch"
                    else (
                        self._pack_runtime_locator.resolve(self._runtime_packs.active_pack(pack_id))
                        is not None
                    )
                )
                candidates.append(
                    replace(
                        candidate,
                        installed=True,
                        resident=False,
                        download_bytes=0,
                        runtime_available=available,
                        task_compatible=not requirement_failures(
                            offer["taskProfile"], requirements
                        ),
                    )
                )
                projected[candidate_id] = bound
            if not candidates:
                return profile
            policy, memory = self._runtime_planning_state_provider()
            hardware = self._hardware_info()
            context = RoutingContext.current(
                self._capability_registry.digest,
                {
                    "requirements": requirements,
                    "registry": REGISTRY_DIGEST,
                },
            )
            engines = recommendation.get("engines", [])
            try:
                plan = self._runtime_routing.plan(
                    candidates,
                    context=context,
                    available_memory_bytes=memory,
                    available_backends=tuple(
                        dict.fromkeys(
                            [
                                hardware.preferred_backend,
                                *(["metal", "mlx", "coreml"] if hardware.unified_memory else []),
                                *self._executable_backends(),
                            ]
                        )
                    ),
                    policy=RuntimePolicy(policy),
                    recommendation_scores={
                        c.identity.candidate_id: len(engines)
                        - engines.index(
                            (projected[c.identity.candidate_id].get("runtimePack") or {}).get(
                                "packId", "oryn.transformers-pytorch"
                            )
                        )
                        for c in candidates
                        if (projected[c.identity.candidate_id].get("runtimePack") or {}).get(
                            "packId", "oryn.transformers-pytorch"
                        )
                        in engines
                    },
                    recommendation_digest=recommendation.get("snapshotDigest", ""),
                )
            except RuntimePlanError:
                profile["executable"] = False
                profile["executableBackends"] = []
                profile["blockingReasons"] = [
                    *profile.get("blockingReasons", []),
                    "RUNTIME_PLAN_UNAVAILABLE",
                ]
                return profile
            selected = projected[plan.selected.identity.candidate_id]
            for field in (
                "sourceRecognition",
                "admissionScope",
                "runner",
                "runtimePack",
                "technicalCapabilities",
                "taskProfile",
                "generationProfile",
            ):
                if field in selected:
                    profile[field] = selected[field]
                else:
                    profile.pop(field, None)
            profile["estimatedMemoryBytes"] = plan.selected.estimated_memory_bytes
            profile["executableBackends"] = [plan.selected.identity.backend]
            profile["runtimePlan"] = plan.trusted_summary()
            profile["runtimePreparationRequired"] = (
                plan.selected.identity.candidate_id
                != record["runtimePlan"]["selected"]["candidateId"]
            )
            profile["installationRequired"] = False
            profile["recommendationContext"] = {
                "runtime": canonical_digest(
                    [
                        plan.selected.identity.candidate_id,
                        runtime_build_digest(),
                        plan.selected.integrity_digest,
                    ]
                ),
                "hardware": canonical_digest([context.device, context.system]),
                "workload": context.task_settings_digest,
            }
            profile.pop("recommendationEvidence", None)
            evidence = self._runtime_routing.measurement(plan.selected, context)
            if evidence:
                profile["recommendationEvidence"] = evidence
            self._inspection_contexts[canonical_digest(profile)] = (
                record.get("providerPlans", {}),
                offers,
                record["routingCandidates"],
            )
            while len(self._inspection_contexts) > 128:
                self._inspection_contexts.pop(next(iter(self._inspection_contexts)))
            return profile

    def _add_builtin_offer(self, profile, offers):
        adapter = self._runtime_foundations.adapter_for_legacy_pack("oryn.transformers-pytorch")
        if adapter is None:
            return
        from oryn_engine.task_profiles import profile_for_execution

        task_profile = profile_for_execution(profile)
        if task_profile is None:
            return
        foundation = self._runtime_foundations.active_foundation(adapter.foundation_id)
        for backend in profile["executableBackends"]:
            offers.append(
                (
                    RuntimeCandidate(
                        RuntimeIdentity(
                            foundation.foundation_id,
                            foundation.version,
                            adapter.adapter_id,
                            adapter.version,
                            "oryn.builtin.transformers",
                            backend,
                            "bf16"
                            if profile.get("sourceRecognition")
                            else "fp32"
                            if backend == "cpu"
                            else "fp16",
                            None,
                        ),
                        self._model_key(profile["model"]),
                        "exact-model",
                        task_profile["contract"]["id"],
                        profile["estimatedMemoryBytes"],
                        profile["estimatedDiskBytes"],
                        "untested",
                        False,
                        False,
                        self._runtime_available,
                        True,
                        True,
                        True,
                        bool(profile["selectedFiles"]),
                        True,
                        True,
                        not profile["requiresCustomCode"],
                        integrity_digest=canonical_digest(profile["selectedFiles"]),
                    ),
                    profile,
                    None,
                )
            )

    def _inspect_backend(
        self,
        model: dict[str, Any],
        *,
        backend,
        info_value,
        requested_task,
        offers,
        manifest_id=None,
        reviewed_source=None,
    ) -> dict[str, Any] | None:
        repository, revision = self._validate_model_ref(model)
        try:
            info = info_value
        except ModelStoreError:
            raise
        except Exception as exc:
            raise ModelStoreError(
                "MODEL_NOT_FOUND",
                "The requested Hugging Face model revision could not be inspected.",
                retryable=True,
            ) from exc
        exact_revision = getattr(info, "sha", None)
        if not isinstance(exact_revision, str) or not EXACT_REVISION_PATTERN.fullmatch(
            exact_revision
        ):
            raise ModelStoreError("MODEL_METADATA_INVALID", "Model revision metadata is invalid.")
        config = getattr(info, "config", None) or {}
        if not isinstance(config, dict):
            raise ModelStoreError(
                "MODEL_METADATA_INVALID", "Model configuration metadata is invalid."
            )
        self._validate_metadata_value(config)
        namespace = f"{repository}@{exact_revision}"
        files = self._validate_files(getattr(info, "siblings", None), namespace)
        if model.get("files"):
            files = [f for f in files if f.name in model["files"]]
            if len(files) != len(model["files"]):
                raise ModelStoreError("MODEL_SUBSET_INCOMPLETE", "Pinned files are unavailable.")

        architectures = config.get("architectures", [])
        if not isinstance(architectures, list) or not all(
            isinstance(item, str) and 0 < len(item) <= 128 for item in architectures
        ):
            raise ModelStoreError(
                "MODEL_METADATA_INVALID", "Model architectures metadata is invalid."
            )
        task = requested_task or getattr(info, "pipeline_tag", None)
        if task == "sentence-similarity":
            task = "feature-extraction"
        if task is not None and (not isinstance(task, str) or len(task) > 128):
            raise ModelStoreError("MODEL_METADATA_INVALID", "Model task metadata is invalid.")

        from oryn_engine.model_inspection import effective_task

        task = effective_task("transformers", task, config)
        # Repository code is inert unless the selected model path actually requires it. A stray
        from oryn_engine.generated_capabilities import normalized_transformers_config

        if {"model_index.json", "modular_model_index.json"} & {item.name for item in files}:
            from oryn_engine.flat_ddpm import normalized_index

            config = normalized_index(config)
        config = normalized_transformers_config(config, task)
        architectures = config.get("architectures", architectures)
        # README helper or training script is recorded but cannot select execution behavior.
        requires_custom_code = bool(config.get("auto_map")) or (
            "OrynReviewedLfm3ForCausalLM" in architectures and reviewed_source is None
        )
        repository_executable_files = tuple(
            item.name
            for item in files
            if PurePosixPath(item.name).suffix.lower()
            in {".py", ".pyc", ".dylib", ".so", ".dll", ".exe", ".sh"}
        )
        has_safe_weights = any(item.name.endswith(".safetensors") for item in files)
        has_unsafe_weights = any(
            PurePosixPath(item.name).suffix in UNSAFE_WEIGHT_SUFFIXES for item in files
        )
        technical = self._technical_capabilities(task, architectures, config)
        workload = transformers_workload(architectures, task)
        pack_matches = self._runtime_packs.match(
            repository=repository,
            revision=exact_revision,
            architectures=architectures,
            task=task,
        )
        pack_matches = tuple(
            m
            for m in pack_matches
            if m.pack_id == backend and (manifest_id is None or m.manifest_id == manifest_id)
        )
        access_managed = bool(getattr(info, "gated", False) or getattr(info, "private", False))
        access_granted = False
        if access_managed:
            access_check = getattr(self._hub, "has_access", None)
            if callable(access_check):
                try:
                    access_granted = bool(access_check(repository))
                except Exception:
                    access_granted = False
        from oryn_engine.flax_weights import TASK_CONTRACTS as FLAX_TASKS
        from oryn_engine.hdf5_tasks import FORMAT_ONLY_TASKS
        from oryn_engine.hdf5_tasks import supports as supports_hdf5
        from oryn_engine.instruction_translation import supports as supports_instruction_translation
        from oryn_engine.shared_tasks import TASK_CONTRACTS

        pack = self._runtime_packs.active_pack(backend)
        if backend == "oryn.mlx-audio":
            from oryn_engine.mlx_audio_inspection import inspect_mlx_audio

            return inspect_mlx_audio(
                self,
                repository,
                exact_revision,
                files,
                config,
                task,
                not access_managed or access_granted,
                offers,
            )
        if backend == "oryn.onnx-runtime":
            from oryn_engine.onnx_inspection import inspect_onnx

            return inspect_onnx(
                self,
                repository,
                exact_revision,
                files,
                config,
                task,
                not access_managed or access_granted,
                offers,
            )
        if backend == "oryn.qwen-tts-pytorch":
            from oryn_engine.qwen_tts_inspection import inspect_qwen_tts

            return inspect_qwen_tts(
                self,
                repository,
                exact_revision,
                files,
                config,
                task,
                not access_managed or access_granted,
                offers,
            )
        if pack.provider is not None:
            return self._inspect_provider_packs(
                info,
                files,
                config,
                task,
                not access_managed or access_granted,
                repository,
                exact_revision,
                offers=offers,
                pack_ids={backend},
            )

        from oryn_engine import numeric_archive_tasks

        if backend == "oryn.transformers-pytorch" and numeric_archive_tasks.selected(
            config, task, {item.name for item in files}
        ):
            return self._inspect_generated_pack(
                repository,
                exact_revision,
                files,
                config,
                task,
                not access_managed or access_granted,
                source_kind="transformers",
                offers=offers,
            )
        if (
            backend == "oryn.transformers-pytorch"
            and task in TASK_CONTRACTS
            and task != "fill-mask"
            and (
                task not in {*FLAX_TASKS, *FORMAT_ONLY_TASKS}
                or config.get("model_type")
                in {"oryn_open_clip", "oryn_pixai", "tipsv2", "oryn_llama_turn"}
                or supports_instruction_translation(config, task)
                or (not has_safe_weights and "flax_model.msgpack" in {item.name for item in files})
                or (
                    not has_safe_weights
                    and supports_hdf5(config, task)
                    and "tf_model.h5" in {item.name for item in files}
                )
            )
            and (
                task != "automatic-speech-recognition"
                or transformers_workload(architectures, task) == task
            )
            and (
                task != "image-classification"
                or config.get("model_type") in {"timm_wrapper", "oryn_pixai"}
            )
            and (task != "token-classification" or config.get("model_type") == "pii_masking")
            and (
                task != "audio-classification"
                or config.get("model_type")
                in {"oryn_wav2vec2_age_gender", "oryn_wav2vec2_dual_hypersphere", "oryn_speechx"}
            )
            and (task != "depth-estimation" or config.get("model_type") == "depth_anything_3")
            and (
                task not in {"feature-extraction", "text-ranking"}
                or config.get("model_type") == "qwen3_vl"
                or task == "text-ranking"
                and config.get("architectures") == ["JinaForRanking"]
                or task == "feature-extraction"
                and config.get("model_type")
                in {"ministral3", "qwen3_bidirec", "pplx_contextual_qwen3_5"}
            )
        ):
            return self._inspect_generated_pack(
                repository,
                exact_revision,
                files,
                config,
                task,
                not access_managed or access_granted,
                source_kind="transformers",
                offers=offers,
            )
        from oryn_engine.generated_capabilities import architectures_for, mlx_model_types

        if backend == "oryn.mlx-lm" and (
            getattr(info, "library_name", None) == "mlx"
            or isinstance(config.get("quantization"), dict)
            or (
                has_safe_weights
                and (
                    task in {"text-generation", "conversational"}
                    and config.get("model_type") in mlx_model_types()
                    or task in {"image-text-to-text", "any-to-any"}
                    and config.get("model_type") in architectures_for(task, source="mlx")
                )
            )
        ):
            return self._inspect_generated_pack(
                repository,
                exact_revision,
                files,
                config,
                task,
                not access_managed or access_granted,
                offers=offers,
            )
        if (
            backend == "oryn.diffusers-pytorch"
            and {"model_index.json", "modular_model_index.json"} & {item.name for item in files}
            and (
                not pack_matches
                or (
                    config.get("_class_name") in architectures_for(task, source="diffusers")
                    and all(
                        match.manifest_id == "oryn.compat.generic-safe-diffusers-image"
                        for match in pack_matches
                    )
                )
            )
        ):
            return self._inspect_generated_pack(
                repository,
                exact_revision,
                files,
                config,
                task,
                not access_managed or access_granted,
                source_kind="diffusers",
                offers=offers,
            )
        if (
            backend == "oryn.llama-cpp"
            and not pack_matches
            and any(item.name.endswith(".gguf") for item in files)
            and (not has_safe_weights or task in {"text-generation", "conversational"})
        ):
            return self._inspect_generic_gguf(
                info=info,
                config=config,
                repository=repository,
                revision=exact_revision,
                files=files,
                task=task,
                access_managed=access_managed,
                access_granted=access_granted,
                repository_executable_files=repository_executable_files,
                offers=offers,
            )
        if not pack_matches and backend != "oryn.transformers-pytorch":
            return None
        plan: RuntimePlan | None = None
        plan_failure: RuntimePlanError | None = None
        if pack_matches:
            try:
                plan = self._plan_runtime_matches(
                    pack_matches,
                    repository=repository,
                    revision=exact_revision,
                    files=files,
                    access_granted=not access_managed or access_granted,
                    requires_custom_code=requires_custom_code,
                    has_safe_weights=has_safe_weights,
                    has_unsafe_weights=has_unsafe_weights,
                )
            except RuntimePlanError as exc:
                plan_failure = exc
        selected_manifest = (
            plan.selected.identity.variant_id
            if plan is not None
            else (pack_matches[0].manifest_id if pack_matches else None)
        )
        pack_match = next(
            (item for item in pack_matches if item.manifest_id == selected_manifest),
            None,
        )
        isolated_pack = pack_match is not None and pack_match.pack_id != "oryn.transformers-pytorch"
        blocking: list[str] = []
        if access_managed and not access_granted:
            blocking.append("MODEL_ACCESS_RESTRICTED")
        audited_adapter = pack_match is not None and pack_match.level == "audited-adapter"
        if requires_custom_code and not audited_adapter:
            blocking.append("CUSTOM_CODE_REQUIRED")
        declared_data_only = (
            pack_match is not None
            and pack_match.pack_id == "oryn.kokoro-onnx"
            and pack_match.level == "converted-artifact"
        )
        if has_unsafe_weights and not has_safe_weights and not declared_data_only:
            blocking.append("UNSAFE_MODEL_FORMAT")

        if workload is None and pack_match is None:
            if plan_failure is not None:
                blocking.append(plan_failure.code)
            else:
                blocking.append("RUNNER_UNSUPPORTED")
        if isolated_pack and pack_match is not None:
            pack = self._runtime_packs.active_pack(pack_match.pack_id)
            runtime_available = self._pack_runtime_locator.resolve(pack) is not None
            runner_declaration = pack.runner(pack_match.runner_id)
            executable_backends = self._available_pack_backends(
                runner_declaration.backends if runner_declaration else ()
            )
            if not executable_backends:
                blocking.append("BACKEND_UNAVAILABLE")
        else:
            runtime_available = self._runtime_available
            executable_backends = self._executable_backends()
        if plan_failure is not None:
            blocking.append(plan_failure.code)
        if not runtime_available:
            blocking.append("RUNTIME_DEPENDENCY_UNAVAILABLE")
        selected_files = self._installation_files(files, pack_match)
        blocking.extend(self._selected_index_reasons(repository, exact_revision, selected_files))
        if not isolated_pack and workload is not None:
            layout = assess_layout([item.name for item in selected_files], config, task)
            blocking.extend(
                reason
                for reason in (*layout.artifact_reasons, *layout.runtime_reasons)
                if reason not in blocking
            )
        selected_names = {item.name for item in selected_files}
        requires_custom_code = requires_custom_code or any(
            PurePosixPath(item.name).suffix.lower()
            in {".py", ".pyc", ".dylib", ".so", ".dll", ".exe", ".sh"}
            for item in selected_files
        )
        unselected_executable_files = sorted(
            name for name in repository_executable_files if name not in selected_names
        )
        if requires_custom_code and not audited_adapter and "CUSTOM_CODE_REQUIRED" not in blocking:
            blocking.append("CUSTOM_CODE_REQUIRED")
        if pack_match is not None and not self._required_patterns_present(files, pack_match):
            blocking.append("MODEL_SUBSET_INCOMPLETE")
        logical_size = sum(item.size for item in selected_files)
        exact_model = {
            "provider": "huggingface",
            "repository": repository,
            "revision": exact_revision,
        }
        executable = not blocking
        estimated_memory = int(logical_size * 1.25)
        if pack_match is not None:
            estimated_memory = max(
                estimated_memory,
                pack_match.resources.get("estimatedResidentBytes", 0)
                + pack_match.resources.get("workingMemoryBytes", 0),
            )
        if reviewed_source is not None:
            # This profile is CPU-only; it cannot inherit generic GPU offers.
            executable_backends = ["cpu"]
            estimated_memory = max(estimated_memory, 6 * 1024**3)
            if sys.platform != "darwin":
                blocking.append("SOURCE_SANDBOX_UNAVAILABLE")
        result: dict[str, Any] = {
            "model": exact_model,
            "exactRevision": exact_revision,
            "executable": executable,
            "executableBackends": executable_backends if executable else [],
            "requiresCustomCode": requires_custom_code,
            "estimatedDiskBytes": logical_size,
            "estimatedMemoryBytes": estimated_memory,
            "selectedFiles": [
                {"name": item.name, "sizeBytes": item.size} for item in selected_files
            ],
            "technicalCapabilities": (
                dict(pack_match.capabilities) if pack_match is not None else technical
            ),
        }
        if isolated_pack:
            result["runner"] = "runtime-pack"
        elif workload is not None:
            result["runner"] = "transformers"
        if pack_match is not None:
            result["runtimePack"] = {
                "packId": pack_match.pack_id,
                "packVersion": pack_match.pack_version,
                "runnerId": pack_match.runner_id,
                "taskContract": pack_match.task_contract,
                "compatibilityLevel": pack_match.level,
                "manifestDigest": pack_match.digest,
                **(
                    {"trustedMetadata": self._public_trusted_metadata(pack_match.trusted_metadata)}
                    if pack_match.trusted_metadata
                    else {}
                ),
            }
            normalized_generation_profile = generation_profile(
                task_contract=pack_match.task_contract,
                trusted_metadata=dict(pack_match.trusted_metadata),
                declarative_metadata=self._bounded_declarative_metadata(config),
            )
            if normalized_generation_profile is not None:
                result["generationProfile"] = normalized_generation_profile
        if plan is not None:
            result["runtimePlan"] = plan.trusted_summary()
        elif plan_failure is not None:
            result["runtimePlanFailure"] = {
                "code": plan_failure.code,
                "rejected": [
                    {
                        "candidateId": item.candidate.identity.candidate_id,
                        "reasonCodes": list(item.rejection_reasons),
                    }
                    for item in plan_failure.evaluations
                ][:16],
            }
        if blocking:
            result["blockingReasons"] = blocking
        if unselected_executable_files:
            result["unselectedExecutableFiles"] = unselected_executable_files[:256]
        compatibility = (
            None
            if isolated_pack
            else self._compatibility_projection(
                repository=repository,
                revision=exact_revision,
                task=task,
                architectures=architectures,
                files=selected_files,
                requires_custom_code=requires_custom_code and not audited_adapter,
                access="authorization-required"
                if access_managed and not access_granted
                else "available",
                estimated_memory=estimated_memory,
                blocking=blocking,
                config=config,
            )
        )
        if compatibility is not None:
            result.update(compatibility)
        if reviewed_source is not None:
            result["sourceRecognition"] = reviewed_source
            result["admissionScope"] = "huggingface-page"
        if plan is not None:
            for evaluation in plan.evaluations:
                offers.append((evaluation.candidate, dict(result), None))
        elif plan_failure is not None:
            for evaluation in plan_failure.evaluations:
                offers.append((evaluation.candidate, dict(result), None))
        elif result.get("executable"):
            self._add_builtin_offer(result, offers)
        return result

    def _inspect_provider_packs(
        self, info, files, config, task, access, repository, revision, *, offers=None, pack_ids=None
    ):
        from oryn_engine.compatibility_plane import CapabilityRegistry, NormalizedCapabilityTuple
        from oryn_engine.providers import ProviderRegistry, bind_plan

        card = getattr(info, "card_data", None)
        if hasattr(card, "to_dict"):
            card = card.to_dict()
        card = card if isinstance(card, dict) else {}
        inventory = {
            "repository": repository,
            "revision": revision,
            "task": task,
            "library": getattr(info, "library_name", None),
            "config": config,
            "card": {k: card[k] for k in ("base_model", "license") if k in card},
            "files": [{"name": f.name, "sizeBytes": f.size, "blobId": f.blob_id} for f in files],
        }
        providers = ProviderRegistry(
            self._runtime_packs, self._pack_runtime_locator, self._provider_process
        )
        results = providers.inspect(inventory, self._hub, access=access, pack_ids=pack_ids)
        if not results:
            if providers.diagnostics:
                return {
                    "model": {
                        "provider": "huggingface",
                        "repository": repository,
                        "revision": revision,
                    },
                    "exactRevision": revision,
                    "executable": False,
                    "executableBackends": [],
                    "requiresCustomCode": False,
                    "estimatedDiskBytes": 0,
                    "estimatedMemoryBytes": 0,
                    "selectedFiles": [],
                    "technicalCapabilities": {
                        "task": task,
                        "architectures": [],
                        "inputModalities": [],
                        "outputModalities": [],
                        "capabilities": {},
                    },
                    "blockingReasons": sorted(set(providers.diagnostics)),
                }
            return None
        # Legacy providers remain available during migration. An incomplete optional
        # export must not shadow an existing native route.
        policy, available_memory = self._runtime_planning_state_provider()
        hardware = self._hardware_info()
        backends = set(self._executable_backends()) | (
            {"mlx", "metal"} if hardware.unified_memory else set()
        )
        projected, candidates, plans = {}, [], {}
        for result in results:
            value, pack = result.candidate, result.pack
            adapter = self._runtime_foundations.adapter_for_legacy_pack(pack.pack_id)
            if adapter is None:
                continue
            foundation = self._runtime_foundations.active_foundation(adapter.foundation_id)
            descriptor = ModelDescriptor(
                "huggingface",
                repository,
                revision,
                "runtime-provider",
                (value["architecture"],),
                value["task"],
                ("provider-artifacts",),
                canonical_digest(value["selectedFiles"]),
                bool(config.get("auto_map")),
                "available" if access else "authorization-required",
                value["estimatedMemoryBytes"],
            )
            registry = CapabilityRegistry(
                "oryn-capability-registry/1",
                (
                    NormalizedCapabilityTuple(
                        value["id"],
                        "runtime-provider",
                        value["architecture"],
                        value["task"],
                        ("provider-artifacts",),
                        foundation.foundation_id,
                        foundation.version,
                        adapter.adapter_id,
                        adapter.version,
                        value["taskContract"],
                        "1.0.0",
                        "untested",
                    ),
                ),
                pack.provider["catalogSha256"],
                (pack.digest,),
            )
            errors = list(value["diagnostics"])
            if value["backend"] not in backends:
                errors.append("BACKEND_UNAVAILABLE")
            eligibility = evaluate_eligibility(
                descriptor,
                registry,
                resource_admissible=0 < value["estimatedMemoryBytes"] <= available_memory,
                runtime_reasons=tuple(errors),
            )
            selected = [f for f in files if f.name in value["selectedFiles"]]
            disk = sum(f.size for f in selected)
            bound = bind_plan(result, inventory)
            candidate = RuntimeCandidate(
                RuntimeIdentity(
                    foundation.foundation_id,
                    foundation.version,
                    adapter.adapter_id,
                    adapter.version,
                    value["id"],
                    value["backend"],
                    value["precision"],
                    value["quantization"],
                ),
                f"huggingface:{repository}@{revision}",
                "exact-model",
                value["taskContract"],
                value["estimatedMemoryBytes"],
                disk,
                "untested",
                False,
                False,
                True,
                True,
                foundation.abi == adapter.foundation_abi,
                not errors,
                bool(selected) and not errors,
                access,
                True,
                not errors,
                integrity_digest=bound["digest"],
            )
            plans[candidate.identity.candidate_id] = bound
            candidates.append(candidate)
            projected[candidate.identity.candidate_id] = {
                "model": bound["model"],
                "exactRevision": revision,
                "executable": eligibility.disposition in {"trust-required", "verified-load"},
                "executableBackends": [value["backend"]]
                if eligibility.disposition in {"trust-required", "verified-load"}
                else [],
                "requiresCustomCode": descriptor.requires_repository_code,
                "estimatedDiskBytes": disk,
                "estimatedMemoryBytes": value["estimatedMemoryBytes"],
                "selectedFiles": [{"name": f.name, "sizeBytes": f.size} for f in selected],
                "technicalCapabilities": {
                    "task": value["task"],
                    "architectures": [value["architecture"]],
                    "inputModalities": value["inputModalities"],
                    "outputModalities": value["outputModalities"],
                    "capabilities": value["capabilities"],
                },
                "runner": "runtime-pack",
                "descriptorSchemaVersion": DESCRIPTOR_SCHEMA_VERSION,
                "capabilityRegistryDigest": self._capability_registry.digest,
                "eligibility": eligibility.protocol_value(),
                "loadDisposition": eligibility.disposition,
                "runtimePack": {
                    "packId": pack.pack_id,
                    "packVersion": pack.version,
                    "runnerId": value["runnerId"],
                    "taskContract": value["taskContract"],
                    "compatibilityLevel": "native-declarative",
                    "manifestDigest": pack.digest,
                },
                "generationProfile": value["generationProfile"],
                "blockingReasons": sorted(
                    {r for axis in eligibility.axes.values() for r in axis.reasons}
                ),
            }
        if offers is not None:
            offers.extend(
                (
                    candidate,
                    projected[candidate.identity.candidate_id],
                    plans[candidate.identity.candidate_id],
                )
                for candidate in candidates
            )
            return next(iter(projected.values()), None)
        if not candidates:
            return None
        try:
            plan = self._runtime_routing.plan(
                candidates,
                context=RoutingContext.current(self._capability_registry.digest, {"task": task}),
                available_memory_bytes=available_memory,
                available_backends=tuple(backends),
                policy=RuntimePolicy(policy),
            )
        except RuntimePlanError:
            return projected[candidates[0].identity.candidate_id]
        key = self._model_key(bound["model"])
        with self._lock:
            if len(self._provider_plans) >= 128:
                self._provider_plans.pop(next(iter(self._provider_plans)))
            self._provider_plans[key] = plans
        profile = projected[plan.selected.identity.candidate_id]
        profile["runtimePlan"] = plan.trusted_summary()
        return profile

    def _repository_template(self, repository, revision, files) -> str | None:
        """Read a repository chat template as data to choose a reviewed format."""

        from oryn_engine import chat_formats

        names = {item.name for item in files}
        try:
            if "chat_template.jinja" in names and hasattr(self._hub, "read_template"):
                return self._hub.read_template(repository, revision, "chat_template.jinja")
            if "tokenizer_config.json" in names:
                return chat_formats.template_from_config(
                    {
                        "tokenizer_config": self._hub.read_json(
                            repository, revision, "tokenizer_config.json"
                        )
                    }
                )
        except (ModelStoreError, OSError, ValueError):
            return None  # An unreadable template is not evidence of an unreviewed format.
        return None

    def _inspect_generated_pack(
        self,
        repository: str,
        revision: str,
        files: list[ModelFile],
        config: dict[str, Any],
        task: str | None,
        access: bool,
        *,
        source_kind: str = "mlx",
        offers=None,
    ) -> dict[str, Any]:
        from oryn_engine.generated_capabilities import mlx_model_types
        from oryn_engine.runtime_packs.mlx_lm_worker import validate_config
        from oryn_engine.shared_tasks import TASK_CONTRACTS, validate_task_config

        is_mlx = source_kind == "mlx"
        is_shared_task = source_kind == "transformers"
        from oryn_engine.model_inspection import effective_task

        task = effective_task(source_kind, task, config)
        is_mlx_vision = is_mlx and task == "image-text-to-text"
        is_mlx_vad = is_mlx and task == "voice-activity-detection"
        is_mlx_multimodal = is_mlx and task == "any-to-any"
        task = (task or "text-generation") if is_mlx else task
        contract = (
            TASK_CONTRACTS.get(task, "unknown-v1")
            if is_shared_task
            else "multimodal-generation-v1"
            if is_mlx_multimodal
            else "voice-activity-detection-v1"
            if is_mlx_vad
            else "chat-v1"
            if is_mlx
            else f"{task}-v1"
        )
        if source_kind == "diffusers":
            from oryn_engine.diffusion_tasks import (
                IMAGE_PIPELINES,
                VIDEO_PIPELINES,
                pipeline_contract,
                pipeline_contracts,
            )

            contract = pipeline_contract(config.get("_class_name", ""), task)
        pack_id = (
            "oryn.transformers-pytorch"
            if is_shared_task
            else "oryn.mlx-lm"
            if is_mlx
            else "oryn.diffusers-pytorch"
        )
        variant = (
            "oryn.generated.transformers-tasks"
            if is_shared_task
            else "oryn.generated.mlx-vad"
            if is_mlx_vad
            else "oryn.generated.mlx-chat"
            if is_mlx
            else "oryn.generated.diffusers"
        )
        architecture = str(config.get("model_type" if is_mlx else "_class_name", "unknown"))
        if is_shared_task:
            declared = config.get("architectures")
            architecture = (
                declared[0] if isinstance(declared, list) and len(declared) == 1 else "unknown"
            )
            if task == "keypoint-detection":
                architecture = str(config.get("model_type", "unknown"))
        reasons: list[str] = []
        thinking_hint = None
        if is_mlx:
            from oryn_engine import mlx_bolding, mlx_kolibri, mlx_prism

            if task not in {
                "text-generation",
                "conversational",
                "image-text-to-text",
                "voice-activity-detection",
                "any-to-any",
            }:
                reasons.append("TASK_ADAPTER_UNAVAILABLE")
            try:
                if mlx_kolibri.supports(config):
                    mlx_kolibri.validate_config(config)
                    if task not in {"text-generation", "conversational"}:
                        raise ValueError("KOLIBRI_TASK")
                elif mlx_bolding.supports(config):
                    mlx_bolding.normalized_config(config, task)
                elif mlx_prism.supports(config) and task in {"text-generation", "conversational"}:
                    mlx_prism.validate_config(config)
                elif is_mlx_vad:
                    from oryn_engine.mlx_vad import validate_config as validate_vad_config

                    validate_vad_config(config)
                elif is_mlx_multimodal or is_mlx_vision and config.get("model_type") == "gemma4":
                    from oryn_engine.mlx_gemma4 import validate_config as validate_gemma4_config

                    validate_gemma4_config(config)
                elif is_mlx_vision:
                    from oryn_engine.mlx_vision import validate_vision_config

                    validate_vision_config(config)
                elif config.get("model_type") == "gpt_oss":
                    from oryn_engine.mlx_gpt_oss import validate_config as validate_gpt_oss_config

                    validate_gpt_oss_config(config)
                elif config.get("model_type") == "gemma4_unified":
                    from oryn_engine.mlx_gemma import validate_config as validate_gemma_config

                    validate_gemma_config(config)
                else:
                    validate_config(config, mlx_model_types())
                    from oryn_engine import chat_formats

                    # This generic route frames prompts only from the reviewed table.
                    template = chat_formats.template_from_config(
                        config
                    ) or self._repository_template(repository, revision, files)
                    if template and chat_formats.select(template) is None:
                        reasons.append("CHAT_FORMAT_UNREVIEWED")
                    elif template and chat_formats.select(template) == "chatml":
                        from oryn_engine.chat_v2 import thinking_support

                        thinking_hint = thinking_support(config.get("model_type"), template)
            except (ValueError, TypeError):
                reasons.append("MODEL_RUNTIME_INCOMPATIBLE")
        elif is_shared_task:
            try:
                from oryn_engine.numeric_archive_tasks import selected as numeric_archive_selected

                validate_task_config(
                    task,
                    config,
                    numeric_archive=numeric_archive_selected(
                        config, task, {item.name for item in files}
                    ),
                )
            except (ValueError, TypeError):
                reasons.append("TASK_CONFIGURATION_UNSUPPORTED")
        elif (
            task not in {"text-to-image", "image-to-image", "image-text-to-image"}
            and not (
                architecture == "MiniMaxMusic3ModularPipeline"
                and task in {"text-to-audio", "text-to-music"}
                or architecture == "MiniMaxH3ModularPipeline"
                and task in {"text-to-video", "image-text-to-video"}
            )
            and not (
                task == "unconditional-image-generation"
                and contract in pipeline_contracts(architecture)
            )
            and not (
                task in {"text-to-video", "image-to-video", "image-text-to-video"}
                and architecture in VIDEO_PIPELINES
                and contract in pipeline_contracts(architecture)
            )
        ):
            reasons.append("TASK_ADAPTER_UNAVAILABLE")
        layout = assess_layout([item.name for item in files], config, task, source_kind=source_kind)
        from oryn_engine import numeric_archive_tasks

        if (
            is_shared_task
            and "pytorch_model.bin" in layout.selected_files
            and access
            and not reasons
        ):
            try:
                archive = next(item for item in files if item.name == "pytorch_model.bin")
                numeric_archive_tasks.validate_archive(
                    config,
                    task,
                    self._hub.read_numeric_archive(
                        repository, revision, archive.name, archive.size
                    ),
                )
                if numeric_archive_tasks.numeric_dpt.supports(config, task):
                    numeric_archive_tasks.numeric_dpt.processor_options(
                        config,
                        self._hub.read_json(repository, revision, "preprocessor_config.json"),
                    )
                if numeric_archive_tasks.numeric_owlvit.supports(config, task):
                    numeric_archive_tasks.numeric_owlvit.processor_options(
                        config,
                        self._hub.read_json(repository, revision, "preprocessor_config.json"),
                    )
                if config.get("model_type") == "m2m_100":
                    numeric_archive_tasks.numeric_seq2seq.validate_tokenizer_metadata(
                        config, self._hub.read_json(repository, revision, "tokenizer_config.json")
                    )
                if task == "table-question-answering" and config.get("model_type") == "bart":
                    from oryn_engine.coverage_tasks import tapex_options

                    tapex_options(
                        self._hub.read_json(repository, revision, "tokenizer_config.json")
                    )
            except ModelStoreError as exc:
                reasons.append(exc.code)
            except OSError as exc:
                reasons.append(
                    str(exc)
                    if str(exc) == "SNAPSHOT_METADATA_MISSING"
                    else "MODEL_METADATA_UNAVAILABLE"
                )
            except (ValueError, TypeError, KeyError):
                reasons.append("MODEL_METADATA_INVALID")
        mlx_metadata_adapter = (
            mlx_kolibri
            if is_mlx and mlx_kolibri.supports(config)
            else mlx_bolding
            if is_mlx and mlx_bolding.supports(config)
            else None
        )
        if mlx_metadata_adapter is not None and access and not reasons:
            try:
                mlx_metadata_adapter.inspect_metadata(
                    config,
                    lambda name: self._hub.read_json(repository, revision, name),
                    lambda name: self._hub.read_tensor_structure(repository, revision, name),
                    [item.name for item in files],
                )
            except OSError as exc:
                reasons.append(
                    str(exc)
                    if str(exc) == "SNAPSHOT_METADATA_MISSING"
                    else "MODEL_PROCESSOR_UNAVAILABLE"
                )
            except ModelStoreError as exc:
                reasons.append(exc.code)
            except (ValueError, TypeError, KeyError):
                reasons.append("TASK_CONFIGURATION_UNSUPPORTED")
        elif is_mlx and mlx_prism.supports(config) and access and not reasons:
            try:
                mlx_prism.validate_tokenizer_metadata(
                    self._hub.read_json(repository, revision, "tokenizer_config.json")
                )
                if config.get("model_type") == mlx_prism.FAMILY:
                    mlx_prism.validate_hadamard(
                        config, self._hub.read_json(repository, revision, "hadamard.json")
                    )
                    mlx_prism.validate_generation_metadata(
                        config, self._hub.read_json(repository, revision, "generation_config.json")
                    )
            except OSError as exc:
                reasons.append(
                    str(exc)
                    if str(exc) == "SNAPSHOT_METADATA_MISSING"
                    else "MODEL_PROCESSOR_UNAVAILABLE"
                )
            except ModelStoreError as exc:
                reasons.append(exc.code)
            except (ValueError, TypeError):
                reasons.append("TASK_CONFIGURATION_UNSUPPORTED")
        if (
            (is_mlx_vision or is_mlx_multimodal)
            and config.get("model_type") == "gemma4"
            and access
            and not reasons
        ):
            from oryn_engine import mlx_gemma4

            try:
                mlx_gemma4.validate_processor(
                    self._hub.read_json(repository, revision, "processor_config.json")
                )
                mlx_gemma4.validate_tokenizer_metadata(
                    self._hub.read_json(repository, revision, "tokenizer_config.json")
                )
            except OSError as exc:
                reasons.append(
                    str(exc)
                    if str(exc) == "SNAPSHOT_METADATA_MISSING"
                    else "MODEL_PROCESSOR_UNAVAILABLE"
                )
            except ModelStoreError as exc:
                reasons.append(exc.code)
            except (ValueError, TypeError):
                reasons.append("TASK_CONFIGURATION_UNSUPPORTED")
        if is_mlx and config.get("model_type") == "gemma4_unified" and access and not reasons:
            from oryn_engine.gemma_decision import validate_tokenizer_metadata

            try:
                validate_tokenizer_metadata(
                    self._hub.read_json(repository, revision, "tokenizer_config.json")
                )
            except OSError as exc:
                reasons.append(
                    str(exc)
                    if str(exc) == "SNAPSHOT_METADATA_MISSING"
                    else "MODEL_PROCESSOR_UNAVAILABLE"
                )
            except ModelStoreError as exc:
                reasons.append(exc.code)
            except (ValueError, TypeError):
                reasons.append("TASK_CONFIGURATION_UNSUPPORTED")
        if is_shared_task and config.get("_orynTimmTagClassifier") and access:
            from oryn_engine.timm_tasks import tag_processor

            try:
                tag_processor(self._hub.read_json(repository, revision, "preprocess.json"), config)
            except OSError as exc:
                reasons.append(
                    str(exc)
                    if str(exc) == "SNAPSHOT_METADATA_MISSING"
                    else "MODEL_PROCESSOR_UNAVAILABLE"
                )
            except ModelStoreError as exc:
                reasons.append(exc.code)
            except (ValueError, TypeError):
                reasons.append("TASK_CONFIGURATION_UNSUPPORTED")
        if (
            source_kind == "diffusers"
            and config.get("_class_name")
            in {"MiniMaxMusic3ModularPipeline", "MiniMaxH3ModularPipeline"}
            and access
            and not reasons
        ):
            from oryn_engine import minimax_h3, minimax_music

            adapter_module = minimax_h3 if minimax_h3.supports(config) else minimax_music
            try:
                adapter_module.inspect_metadata(
                    config,
                    lambda name: self._hub.read_json(repository, revision, name),
                    lambda name: self._hub.read_tensor_structure(repository, revision, name),
                    [item.name for item in files],
                )
            except OSError as exc:
                reasons.append(
                    str(exc)
                    if str(exc) == "SNAPSHOT_METADATA_MISSING"
                    else "MODEL_METADATA_UNAVAILABLE"
                )
            except ModelStoreError as exc:
                reasons.append(exc.code)
            except (ValueError, TypeError, KeyError):
                reasons.append("TASK_CONFIGURATION_UNSUPPORTED")
        if source_kind == "diffusers" and access and not reasons:
            from oryn_engine import wan_video

            if wan_video.expanded(config):
                try:
                    wan_video.validate_configs(
                        {
                            name: self._hub.read_json(repository, revision, f"{name}/config.json")
                            for name in wan_video.COMPONENTS
                        }
                    )
                except OSError as exc:
                    reasons.append(
                        str(exc)
                        if str(exc) == "SNAPSHOT_METADATA_MISSING"
                        else "MODEL_PROCESSOR_UNAVAILABLE"
                    )
                except ModelStoreError as exc:
                    reasons.append(exc.code)
                except (ValueError, TypeError, KeyError):
                    reasons.append("TASK_CONFIGURATION_UNSUPPORTED")
        if config.get("model_type") == "oryn_ct2_whisper" and access and not reasons:
            from oryn_engine.ct2_tasks import validate_config as validate_ct2_config
            from oryn_engine.ct2_weights import validate_whisper

            try:
                directory = self._hub.read_ctranslate2_metadata(
                    repository,
                    revision,
                    "model.bin",
                    next(item.size for item in files if item.name == "model.bin"),
                )
                validate_ct2_config(config, validate_whisper(directory))
            except ModelStoreError as exc:
                reasons.append(exc.code)
            except OSError as exc:
                reasons.append(
                    str(exc)
                    if str(exc) == "SNAPSHOT_METADATA_MISSING"
                    else "MODEL_METADATA_UNAVAILABLE"
                )
            except (ValueError, TypeError, KeyError, StopIteration):
                reasons.append("MODEL_METADATA_INVALID")
        if is_shared_task and access and not reasons:
            from oryn_engine.shared_tasks import QWEN_VISION_FAMILIES, validate_processor_config

            metadata_names = (
                "tokenizer_config.json",
                "processor_config.json",
                "preprocessor_config.json",
            ) + (
                ("video_preprocessor_config.json",)
                if config.get("model_type") in {"vjepa2", "sam2_video"}
                or task == "video-text-to-text"
                and config.get("model_type") in QWEN_VISION_FAMILIES
                else ()
            )
            from oryn_engine import (
                bidirectional_embeddings,
                colqwen35_tasks,
                contextual_embeddings,
                decision_tasks,
                instruction_translation,
                jina4_tasks,
                jina_ranking,
                layoutlm_tasks,
                pii_tasks,
                qwen_retrieval,
                turn_classification,
            )

            if decision_tasks.supports(config, task):
                metadata_names = decision_tasks.METADATA_NAMES
            elif layoutlm_tasks.supports(config, task):
                metadata_names = ("special_tokens_map.json",)
            elif config.get("model_type") == "oryn_llava_video_qwen2":
                from oryn_engine.llava_video_qwen import METADATA_NAMES

                metadata_names = METADATA_NAMES

            elif config.get("model_type") == "videollama3_qwen2":
                from oryn_engine.videollama3_tasks import METADATA_NAMES

                metadata_names = METADATA_NAMES
            elif config.get("model_type") == "molparser_vision_encoder_decoder":
                from oryn_engine.molparser_tasks import METADATA_NAMES

                metadata_names = METADATA_NAMES
            elif config.get("model_type") in {"oryn_mpsenet", "oryn_reuse", "oryn_tiger_dnr"}:
                metadata_names = ()
            elif config.get("model_type") == "qwen3_tts_tokenizer_12hz":
                metadata_names = ("preprocessor_config.json",)
            elif config.get("model_type") == "breeze":
                from oryn_engine.breeze_tasks import METADATA_NAMES

                metadata_names = METADATA_NAMES
            elif config.get("model_type") == "omnivoice":
                from oryn_engine.omnivoice_tasks import METADATA_NAMES

                metadata_names = METADATA_NAMES
            elif config.get("model_type") == "audio8_asr_infinite":
                metadata_names = ("preprocessor_config.json", "tokenizer_config.json")
            elif config.get("model_type") in {"oryn_ct2_whisper", "oryn_sopro"}:
                metadata_names = ()
            elif config.get("model_type") == "parakeet_tdt":
                metadata_names = ("ternary.json",)
            elif config.get("model_type") in {
                "oryn_wav2vec2_age_gender",
                "oryn_wav2vec2_dual_hypersphere",
                "oryn_speechx",
            }:
                metadata_names = ("preprocessor_config.json",)
            elif jina4_tasks.supports(config, task):
                metadata_names = jina4_tasks.METADATA_NAMES
            elif colqwen35_tasks.supports(config, task):
                metadata_names = colqwen35_tasks.METADATA_NAMES
                if not {*metadata_names, "tokenizer.json"} <= set(layout.selected_files):
                    reasons.append("MODEL_PROCESSOR_MISSING")
            elif (
                pii_tasks.supports(config, task)
                or turn_classification.supports(config, task)
                or contextual_embeddings.supports(config, task)
                or jina_ranking.supports(config, task)
                or instruction_translation.supports(config, task)
            ):
                metadata_names = ("tokenizer_config.json",)
            elif qwen_retrieval.supports(task, config):
                metadata_names = qwen_retrieval.metadata_names(task)
            elif bidirectional_embeddings.supports(task, config):
                metadata_names = bidirectional_embeddings.METADATA_NAMES
            elif config.get("model_type") == "neomme":
                from oryn_engine.neomme_retrieval import METADATA_NAMES

                metadata_names = METADATA_NAMES
            elif config.get("model_type") == "colqwen3":
                from oryn_engine.colqwen3_tasks import METADATA_NAMES

                metadata_names = METADATA_NAMES
                if not set(metadata_names) <= set(layout.selected_files):
                    reasons.append("MODEL_PROCESSOR_MISSING")
            for name in metadata_names:
                if name not in layout.selected_files:
                    continue
                try:
                    metadata = self._hub.read_json(repository, revision, name)
                    # These owned readers validate the complete publisher template
                    # below and consume only literal token data. Their exact auto_map
                    # declarations never reach a tokenizer constructor.
                    if name == "tokenizer_config.json" and config.get("model_type") not in {
                        "videollama3_qwen2",
                        "molparser_vision_encoder_decoder",
                    }:
                        from oryn_engine.model_metadata import validate_tokenizer_config

                        validate_tokenizer_config(metadata)
                    if layoutlm_tasks.supports(config, task):
                        layoutlm_tasks.validate_metadata(name, metadata)
                    elif config.get("model_type") == "oryn_llava_video_qwen2":
                        from oryn_engine.llava_video_qwen import validate_metadata

                        validate_metadata(name, metadata, config)

                    elif config.get("model_type") == "videollama3_qwen2":
                        from oryn_engine.videollama3_tasks import validate_metadata

                        validate_metadata(name, metadata, config)
                    elif config.get("model_type") == "molparser_vision_encoder_decoder":
                        from oryn_engine.molparser_tasks import validate_metadata

                        validate_metadata(name, metadata, config)
                    elif config.get("model_type") == "qwen3_tts_tokenizer_12hz":
                        from oryn_engine.qwen_codec_tasks import validate_processor

                        validate_processor(metadata)
                    elif config.get("model_type") == "breeze":
                        from oryn_engine.breeze_tasks import validate_metadata

                        validate_metadata(name, metadata, config)
                    elif config.get("model_type") == "omnivoice":
                        from oryn_engine.omnivoice_tasks import validate_metadata

                        validate_metadata(name, metadata, config)
                    elif config.get("model_type") == "oryn_speechx":
                        from oryn_engine.speechx_tasks import validate_processor

                        validate_processor(metadata, config)
                    elif config.get("model_type") == "audio8_asr_infinite":
                        from oryn_engine.audio8_tasks import validate_metadata

                        validate_metadata(name, metadata, config)
                    elif config.get("model_type") == "parakeet_tdt":
                        from oryn_engine.parakeet_tasks import validate_metadata

                        validate_metadata(name, metadata, config)
                    elif config.get("model_type") == "nemotron3_5_asr":
                        from oryn_engine.rnnt_tasks import validate_metadata

                        validate_metadata(name, metadata, config)
                    elif config.get("model_type") == "oryn_pixai":
                        from oryn_engine.pixai_tasks import validate_processor

                        validate_processor(metadata, config)
                    elif config.get("model_type") == "rtmw":
                        from oryn_engine.rtmw_tasks import validate_processor

                        validate_processor(metadata, config)
                    elif config.get("model_type") == "tipsv2":
                        from oryn_engine.tips_tasks import validate_metadata

                        validate_metadata(name, metadata, config)
                    elif jina4_tasks.supports(config, task):
                        jina4_tasks.validate_metadata(name, metadata, config)
                    elif colqwen35_tasks.supports(config, task):
                        colqwen35_tasks.validate_metadata(name, metadata, config)
                    elif config.get("model_type") == "colqwen3":
                        from oryn_engine.colqwen3_tasks import validate_metadata

                        validate_metadata(name, metadata, config)
                    else:
                        validate_processor_config(metadata, config.get("model_type", ""))
                    if decision_tasks.supports(config, task):
                        decision_tasks.validate_metadata(name, metadata, config)
                    if jina_ranking.supports(config, task):
                        jina_ranking.validate_metadata(name, metadata, config)
                    if pii_tasks.supports(config, task):
                        pii_tasks.validate_metadata(name, metadata, config)
                    if turn_classification.supports(config, task):
                        turn_classification.validate_metadata(name, metadata, config)
                    if contextual_embeddings.supports(config, task):
                        contextual_embeddings.validate_metadata(name, metadata, config)
                    if instruction_translation.supports(config, task):
                        instruction_translation.adapter(config).validate_metadata(
                            name, metadata, config
                        )
                    if (
                        config.get("model_type") == "marian"
                        and "tf_model.h5" in layout.selected_files
                        and name == "tokenizer_config.json"
                    ):
                        from oryn_engine.hdf5_marian import validate_tokenizer_metadata

                        validate_tokenizer_metadata(metadata)
                    if qwen_retrieval.supports(task, config):
                        qwen_retrieval.validate_metadata(name, metadata, config)
                    elif bidirectional_embeddings.supports(task, config):
                        bidirectional_embeddings.validate_metadata(name, metadata, config)
                    if config.get("model_type") == "llava_next_video":
                        from oryn_engine.llava_video import validate_metadata

                        validate_metadata(name, metadata, config)
                    if config.get("model_type") == "llava_next":
                        from oryn_engine.llava_next_tasks import validate_metadata

                        validate_metadata(name, metadata, config)
                    if config.get("model_type") == "neomme":
                        from oryn_engine.neomme_retrieval import validate_metadata

                        validate_metadata(name, metadata, config)
                    if (
                        config.get("model_type") == "sapiens2"
                        and name == "preprocessor_config.json"
                    ):
                        from oryn_engine.sapiens2_pose import processor_options

                        processor_options(metadata, config)
                    if config.get("model_type") == "vision-encoder-decoder":
                        from oryn_engine.trocr_tasks import processor_options

                        if name == "preprocessor_config.json":
                            processor_options(metadata, config)
                        if name == "tokenizer_config.json" and metadata.get(
                            "tokenizer_class"
                        ) not in {"RobertaTokenizer", "RobertaTokenizerFast"}:
                            raise ValueError("TROCR_TOKENIZER_UNSUPPORTED")
                    if (
                        config.get("model_type") == "wav2vec2"
                        and name == "preprocessor_config.json"
                    ):
                        from oryn_engine.onnx_asr import (
                            validate_processor_config as validate_ctc_processor,
                        )

                        validate_ctc_processor(metadata)
                    if config.get("model_type") == "clip" and name == "preprocessor_config.json":
                        from oryn_engine.flax_weights import clip_processor_options

                        clip_processor_options(metadata, config["vision_config"]["image_size"])
                    if (
                        config.get("model_type") == "vit"
                        and "tf_model.h5" in layout.selected_files
                        and name == "preprocessor_config.json"
                    ):
                        from oryn_engine.hdf5_tasks import vit_processor_options

                        vit_processor_options(metadata, config)
                    if (
                        config.get("model_type") == "blip"
                        and "tf_model.h5" in layout.selected_files
                        and name == "preprocessor_config.json"
                    ):
                        from oryn_engine.hdf5_blip import processor_options

                        processor_options(metadata, config)
                    if (
                        config.get("model_type") == "segformer"
                        and "tf_model.h5" in layout.selected_files
                        and name == "preprocessor_config.json"
                    ):
                        from oryn_engine.hdf5_segformer import processor_options

                        processor_options(metadata, config)
                    from oryn_engine.shared_tasks import QWEN_VISION_FAMILIES, qwen_image_options

                    if config.get("model_type") in QWEN_VISION_FAMILIES | {"glm_ocr"} and (
                        "image_processor" in metadata or "image_processor_type" in metadata
                    ):
                        qwen_image_options(
                            metadata,
                            config.get("vision_config", {}),
                            video_only=task == "video-text-to-text"
                            and config.get("model_type") in QWEN_VISION_FAMILIES,
                        )
                    if (
                        task == "video-text-to-text"
                        and config.get("model_type") in QWEN_VISION_FAMILIES
                        and name == "video_preprocessor_config.json"
                    ):
                        from oryn_engine.shared_tasks import qwen_video_options

                        qwen_video_options(
                            metadata, config.get("vision_config", {}), config["model_type"]
                        )
                    if config.get("model_type") == "gemma4" and task == "any-to-any":
                        from oryn_engine.gemma4_tasks import (
                            validate_processor,
                            validate_tokenizer_metadata,
                        )

                        if name == "processor_config.json":
                            validate_processor(metadata)
                        elif name == "tokenizer_config.json":
                            validate_tokenizer_metadata(metadata)
                except (ValueError, TypeError):
                    reasons.append("MODEL_METADATA_INVALID")
                    break
                except ModelStoreError as exc:
                    reasons.append(
                        "MODEL_METADATA_INVALID"
                        if exc.code == "MODEL_METADATA_INVALID"
                        else "PROCESSOR_METADATA_UNAVAILABLE"
                    )
                    break
                except Exception:
                    reasons.append("PROCESSOR_METADATA_UNAVAILABLE")
                    break
        selected = [item for item in files if item.name in layout.selected_files]
        from oryn_engine.model_composition import adapter_name
        from oryn_engine.model_composition import binding as composition_binding

        composition = (
            composition_binding(repository, revision) if source_kind == "diffusers" else None
        )
        if composition is not None:
            adapter_files = [item for item in files if item.name == adapter_name(composition)]
            if len(adapter_files) != 1:
                reasons.append("MODEL_SUBSET_INCOMPLETE")
            selected.extend(adapter_files)
        reasons.extend(self._selected_index_reasons(repository, revision, selected))
        unconditional_native_geometry = None
        if (
            source_kind == "diffusers"
            and task == "unconditional-image-generation"
            and access
            and not reasons
        ):
            from oryn_engine.diffusion_tasks import unconditional_geometry

            name = (
                "unet/config.json" if "unet/config.json" in layout.selected_files else "config.json"
            )
            try:
                unet_metadata = self._hub.read_json(repository, revision, name)
                unconditional_native_geometry = unconditional_geometry(unet_metadata)
                from oryn_engine import flat_ddpm

                if flat_ddpm.supports(config):
                    if name != "config.json":
                        raise ValueError("FLAT_DDPM_WEIGHT_LAYOUT")
                    flat_ddpm.validate_configs(
                        unet_metadata,
                        self._hub.read_json(repository, revision, "scheduler_config.json"),
                    )
                from oryn_engine import numeric_unet2d

                if numeric_unet2d.WEIGHTS in layout.selected_files:
                    numeric_unet2d.validate_configs(
                        config,
                        unet_metadata,
                        self._hub.read_json(repository, revision, "scheduler_config.json"),
                    )
                    archive = next(item for item in files if item.name == numeric_unet2d.WEIGHTS)
                    numeric_unet2d.validate_archive(
                        unet_metadata,
                        self._hub.read_numeric_archive(
                            repository, revision, archive.name, archive.size
                        ),
                    )
            except ModelStoreError as exc:
                reasons.append(exc.code)
            except OSError as exc:
                reasons.append(
                    str(exc)
                    if str(exc) == "SNAPSHOT_METADATA_MISSING"
                    else "MODEL_METADATA_UNAVAILABLE"
                )
            except (ValueError, TypeError):
                reasons.append("MODEL_METADATA_INVALID")
        size = sum(item.size for item in selected)
        memory = max(
            1,
            int(size * (3 if is_shared_task else 1.25 if is_mlx else 2))
            + (512 if is_mlx else 2048) * 1024**2,
        )
        if is_mlx and mlx_prism.supports(config):
            # Q1 words widen to Q2 without float expansion; account for loading,
            # the packed result, the complete floating vision graph and KV state.
            memory = max(
                memory, size * (3 if config["quantization"]["bits"] == 1 else 2) + 2048 * 1024**2
            )
        if config.get("model_type") == "audio8_asr_infinite":
            # Complete float32 host graph plus device copy from BF16 originals.
            memory = max(memory, size * 4 + 2048 * 1024**2)
        if config.get("model_type") == "parakeet_tdt" and not reasons:
            from oryn_engine.parakeet_tasks import estimated_memory

            memory = max(memory, estimated_memory(config, size))
        if set(layout.selected_files) & {
            "flax_model.msgpack",
            "tf_model.h5",
            "pytorch_model.bin",
            "diffusion_pytorch_model.bin",
        }:
            # Numeric validation, float32 host weights, device copy and decoding.
            memory = size * 6 + 1024 * 1024**2
        pack = self._runtime_packs.active_pack(pack_id)
        adapter = self._runtime_foundations.adapter_for_legacy_pack(pack.pack_id)
        if adapter is None:
            raise ModelStoreError(
                "RUNTIME_DEPENDENCY_UNAVAILABLE", "Runtime adapter is unavailable."
            )
        foundation = self._runtime_foundations.active_foundation(adapter.foundation_id)
        available = self._pack_runtime_locator.resolve(pack) is not None
        hardware = self._hardware_info()
        backend = (
            "mlx"
            if is_mlx
            else "cpu"
            if config.get("model_type") == "oryn_ct2_whisper"
            else hardware.preferred_backend
        )
        shared_precision = "fp32"
        if is_shared_task and task == "any-to-any" and config.get("model_type") == "gemma4":
            from oryn_engine.gemma4_tasks import precision

            shared_precision = precision(config, backend)
            if shared_precision in {"bf16", "fp16"}:
                # Retain original checkpoint precision on GPU; budget both host
                # and device weights during load plus bounded generation work.
                memory = int(size * 2) + 2048 * 1024**2
        if source_kind == "diffusers" and architecture in {
            "MiniMaxMusic3ModularPipeline",
            "MiniMaxH3ModularPipeline",
        }:
            memory = max(memory, size * 4 + 1024 * 1024**2)
        if source_kind == "diffusers" and architecture == "QwenImage21Pipeline":
            from oryn_engine.qwen_image_21 import estimated_memory

            memory = estimated_memory(selected, backend)
        legacy_image_input = False
        instruction_image_input = False
        if source_kind == "diffusers":
            from oryn_engine.generated_capabilities import architectures_for

            legacy_image_input = (
                task == "text-to-image"
                and architecture in architectures_for("image-to-image", source="diffusers")
                and pipeline_contract(architecture, "image-to-image") == "image-to-image-v1"
            )
            instruction_image_input = (
                task == "text-to-image"
                and architecture in architectures_for("image-to-image", source="diffusers")
                and pipeline_contract(architecture, "image-to-image") == "image-editing-v1"
            )
        backend_available = hardware.unified_memory if is_mlx else backend in adapter.backends
        policy, available_memory = self._runtime_planning_state_provider()
        runtime_reasons = tuple(
            reasons
            + ([] if available else ["RUNTIME_DEPENDENCY_UNAVAILABLE"])
            + ([] if backend_available else ["BACKEND_UNAVAILABLE"])
        )
        assessment = inspect_descriptor(
            repository=repository,
            revision=revision,
            files=[x.name for x in selected],
            config=config,
            task="text-generation"
            if is_mlx and not (is_mlx_vision or is_mlx_vad or is_mlx_multimodal)
            else task,
            registry=self._capability_registry,
            access="available" if access else "authorization-required",
            source_kind=source_kind,
            file_identities=[(x.name, x.size, x.blob_id) for x in selected],
            estimated_memory=memory,
            available_memory=available_memory,
            runtime_reasons=runtime_reasons,
        )
        descriptor, eligibility = assessment.descriptor, assessment.eligibility
        candidate = RuntimeCandidate(
            RuntimeIdentity(
                foundation.foundation_id,
                foundation.version,
                adapter.adapter_id,
                adapter.version,
                variant,
                backend,
                "quantized"
                if config.get("quantization")
                else "native"
                if is_mlx
                else (
                    "fp32"
                    if task == "unconditional-image-generation"
                    or architecture == "MiniMaxMusic3ModularPipeline"
                    else "bf16"
                    if source_kind == "diffusers"
                    and backend in {"mps", "cuda"}
                    and architecture in IMAGE_PIPELINES
                    else shared_precision
                    if is_shared_task
                    else "fp32"
                    if backend == "cpu"
                    else "fp16"
                ),
                f"int{config['quantization']['bits']}"
                if isinstance(config.get("quantization"), dict)
                and config["quantization"].get("bits") in {4, 5, 6, 8}
                else None,
            ),
            f"huggingface:{repository}@{revision}",
            "exact-model",
            contract,
            memory,
            size,
            "untested",
            False,
            False,
            available,
            True,
            foundation.abi == adapter.foundation_abi,
            not reasons,
            not layout.artifact_reasons,
            access,
            True,
            not runtime_reasons,
            integrity_digest=canonical_digest(
                {
                    "files": descriptor.selected_files_digest,
                    "foundation": foundation.digest,
                    "pack": pack.digest,
                    "capabilities": self._capability_registry.digest,
                }
            ),
        )
        result: dict[str, Any] = {
            "model": {"provider": "huggingface", "repository": repository, "revision": revision},
            "exactRevision": revision,
            "executable": eligibility.disposition == "trust-required",
            "executableBackends": [backend] if eligibility.disposition == "trust-required" else [],
            "requiresCustomCode": bool(config.get("auto_map")),
            "estimatedDiskBytes": size,
            "estimatedMemoryBytes": memory,
            "selectedFiles": [{"name": x.name, "sizeBytes": x.size} for x in selected],
            "technicalCapabilities": {
                "task": "text-generation"
                if is_mlx and not (is_mlx_vision or is_mlx_vad or is_mlx_multimodal)
                else task,
                "architectures": [architecture],
                "inputModalities": ["text", "image"]
                if is_mlx_vision
                or legacy_image_input
                or instruction_image_input
                or task
                in {
                    "image-to-image",
                    "image-text-to-image",
                    "image-to-video",
                    "image-text-to-video",
                }
                else []
                if task == "unconditional-image-generation"
                else ["text"],
                "outputModalities": ["image", "structured"]
                if contract == "image-layers-v1"
                else ["text"]
                if is_mlx
                else ["video"]
                if task in {"text-to-video", "image-to-video", "image-text-to-video"}
                else ["image"],
                "capabilities": {"chat": True, "streaming": True}
                if is_mlx
                else {
                    "imageEditing": legacy_image_input
                    or instruction_image_input
                    or task == "image-to-image"
                    or contract == "image-layers-v1",
                    "instructionEditing": instruction_image_input,
                },
            },
            "runner": "runtime-pack",
            "descriptorSchemaVersion": DESCRIPTOR_SCHEMA_VERSION,
            "capabilityRegistryDigest": self._capability_registry.digest,
            "eligibility": eligibility.protocol_value(),
            "loadDisposition": eligibility.disposition,
            "runtimePack": {
                "packId": pack.pack_id,
                "packVersion": pack.version,
                "runnerId": "transformers.shared-tasks"
                if is_shared_task
                else "mlx.multimodal"
                if is_mlx_multimodal
                else "mlx.vad"
                if is_mlx_vad
                else "mlx.chat"
                if is_mlx
                else "diffusers.pipeline",
                "taskContract": contract,
                "compatibilityLevel": "native-declarative",
                "manifestDigest": pack.digest,
            },
        }
        if is_shared_task:
            result["technicalCapabilities"] = self._technical_capabilities(
                task, [architecture], config
            )
        if architecture == "MiniMaxMusic3ModularPipeline" and source_kind == "diffusers":
            result["technicalCapabilities"] = {
                "task": task,
                "architectures": [architecture],
                "inputModalities": ["text"],
                "outputModalities": ["audio"],
                "capabilities": {
                    "audioOutput": True,
                    "lyrics": True,
                    "deterministicSeed": True,
                    "streaming": False,
                },
            }
        if is_mlx_multimodal:
            from oryn_engine.gemma4_tasks import input_modes

            result["technicalCapabilities"] = {
                "task": task,
                "architectures": [architecture],
                "inputModalities": ["text", *input_modes(config)],
                "outputModalities": ["text"],
                "capabilities": {"streaming": False},
            }
        if is_mlx_vad:
            result["technicalCapabilities"] = {
                "task": task,
                "architectures": [architecture],
                "inputModalities": ["audio"],
                "outputModalities": ["structured"],
                "capabilities": {"streaming": False},
            }
        if source_kind == "diffusers" and architecture == "QwenImage21Pipeline":
            from oryn_engine.qwen_image_21 import generation_defaults

            # Invalid pipeline options already block this candidate's layout.
            with contextlib.suppress(ValueError):
                result["generationProfile"] = generation_defaults(config)
        if source_kind == "diffusers" and architecture == "Flux2KleinPipeline":
            from oryn_engine.diffusion_tasks import klein_generation_defaults

            with contextlib.suppress(ValueError):
                result["generationProfile"] = klein_generation_defaults(config)
        if unconditional_native_geometry is not None:
            from oryn_engine.diffusion_tasks import unconditional_generation_defaults

            result["generationProfile"] = unconditional_generation_defaults(
                unconditional_native_geometry
            )
        if composition is not None:
            from oryn_engine.model_composition import generation_defaults

            result["runtimePack"]["trustedMetadata"] = {
                "composition": composition,
                "license": "See pinned base and adapter model cards",
                "conformance": {
                    "status": "fixture-only",
                    "platforms": ["cpu", "macos-arm64-metal"],
                },
            }
            result["generationProfile"] = generation_defaults(composition, config)
        if thinking_hint is not None:
            # Internal inspection evidence is consumed only by the negotiated
            # v2 task profile; never extend legacy trustedMetadata on the wire.
            result["_chatThinkingSupport"] = thinking_hint
        if offers is not None:
            offers.append((candidate, result, None))
        if result["executable"] and offers is None:
            result["runtimePlan"] = self._runtime_routing.plan(
                [candidate],
                context=RoutingContext.current(
                    self._capability_registry.digest, {"task": contract}
                ),
                available_memory_bytes=available_memory,
                available_backends=(backend,),
                policy=RuntimePolicy(policy),
            ).trusted_summary()
        result["blockingReasons"] = sorted(
            {reason for axis in eligibility.axes.values() for reason in axis.reasons}
        )
        return result

    def _inspect_generic_gguf(
        self,
        *,
        info: object,
        config: dict[str, Any],
        repository: str,
        revision: str,
        files: list[ModelFile],
        task: str | None,
        access_managed: bool,
        access_granted: bool,
        repository_executable_files: tuple[str, ...],
        offers=None,
    ) -> dict[str, Any]:
        """Project an arbitrary data-only GGUF through the packaged llama.cpp provider."""

        # Headers decide the variant, its architecture and its projector when they can be
        # read. Otherwise the Hub summary and file names give a provisional choice; either
        # way installation re-reads every downloaded header.
        choice = self._choose_gguf_by_header(repository, revision, files, task)
        header = choice.header if choice is not None else None
        architecture = (
            (header or (choice.seen if choice is not None else None) or {}).get("architecture")
            or normalized_gguf_architecture(info, config)
            or "unknown"
        )
        from oryn_engine.generated_capabilities import architectures_for

        vision = task == "image-text-to-text" and architecture in architectures_for(
            task, source="gguf"
        )
        normalized_task = (
            "text-generation"
            if task in {None, "conversational", "text-generation"}
            or task == "image-text-to-text"
            and not vision
            else task
        )
        translation = normalized_task == "translation"
        contract = "translation-v1" if translation else "chat-v1"
        runner_id = "llama.cpp.translation" if translation else "llama.cpp.chat"
        if header is not None:
            selected_file, quantization_hint = choice.file, header["quantization"]
        else:
            selected_file, quantization_hint = self._select_generic_gguf_file(files)
        selected_names = (
            split_file_set([f.name for f in files], selected_file.name) if selected_file else []
        )
        projector = None
        if vision and header is not None:
            projector = choice.projector
            if projector:
                selected_names.append(projector.name)
        elif vision and selected_file is not None:
            from oryn_engine.runtime_packs.gguf import safe_member_name

            # A projector sits next to the weights or at the repository root. F16 is
            # preferred, BF16 accepted; the installed projector header is still checked.
            folder = PurePosixPath(selected_file.name).parent
            projectors = [
                f
                for f in files
                if safe_member_name(f.name)
                and PurePosixPath(f.name).parent in {folder, PurePosixPath(".")}
                and PurePosixPath(f.name).name.lower().startswith("mmproj-")
                and PurePosixPath(f.name).name.lower().endswith(("-f16.gguf", "-bf16.gguf"))
            ]
            for preferred in (
                [f for f in projectors if PurePosixPath(f.name).parent == folder],
                projectors,
            ):
                if preferred:
                    projectors = preferred
                    break
            f16 = [f for f in projectors if not f.name.lower().endswith("-bf16.gguf")]
            projectors = f16 or projectors
            projector = projectors[0] if len(projectors) == 1 else None
            if projector:
                selected_names.append(projector.name)
        selected_files = sorted(
            (f for f in files if f.name in selected_names), key=lambda f: f.name
        )
        identity = [
            {"name": item.name, "size": item.size, "digest": item.blob_id}
            for item in selected_files
        ]
        logical_size = sum(item.size for item in selected_files)
        estimated_memory = int(logical_size * 1.25)
        try:
            _policy, available_memory = self._runtime_planning_state_provider()
        except Exception:
            available_memory = int(psutil.virtual_memory().available)
        descriptor = ModelDescriptor(
            provider="huggingface",
            repository=repository,
            revision=revision,
            source_kind="gguf",
            architectures=(architecture,),
            task=normalized_task,
            formats=("gguf",),
            selected_files_digest=hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            requires_repository_code=False,
            access=(
                "authorization-required" if access_managed and not access_granted else "available"
            ),
            estimated_memory_bytes=estimated_memory,
        )
        eligibility = evaluate_eligibility(
            descriptor,
            self._capability_registry,
            resource_admissible=estimated_memory <= max(0, available_memory),
        )
        blocking = sorted(
            {
                reason
                for name, axis in eligibility.axes.items()
                if name != "conformance"
                for reason in axis.reasons
            }
        )
        if access_managed and not access_granted:
            blocking.append("MODEL_ACCESS_RESTRICTED")
        if selected_file is None or (choice is not None and header is None):
            blocking.append("GGUF_VARIANT_UNSUPPORTED")
        if vision and projector is None:
            blocking.append("GGUF_PROJECTOR_MISSING_OR_AMBIGUOUS")

        pack = self._runtime_packs.active_pack("oryn.llama-cpp")
        runner = pack.runner(runner_id)
        runtime_available = self._pack_runtime_locator.resolve(pack) is not None
        if not runtime_available:
            blocking.append("RUNTIME_DEPENDENCY_UNAVAILABLE")
        blocking = list(dict.fromkeys(blocking))
        matched = eligibility.matched
        executable = bool(
            matched is not None
            and selected_file is not None
            and runtime_available
            and not blocking
            and eligibility.disposition in {"verified-load", "trust-required"}
        )
        exact_model = {
            "provider": "huggingface",
            "repository": repository,
            "revision": revision,
        }
        technical = {
            "architectures": [architecture],
            "task": normalized_task,
            "inputModalities": ["text", "image"] if vision else ["text"],
            "outputModalities": ["text"],
            "capabilities": {} if translation else {"chat": True, "streaming": True},
        }
        result: dict[str, Any] = {
            "model": exact_model,
            "exactRevision": revision,
            "executable": executable,
            "executableBackends": list(runner.backends) if executable and runner else [],
            "requiresCustomCode": False,
            "estimatedDiskBytes": logical_size,
            "estimatedMemoryBytes": estimated_memory,
            "selectedFiles": [
                {"name": item.name, "sizeBytes": item.size} for item in selected_files
            ],
            "technicalCapabilities": technical,
            "descriptorSchemaVersion": DESCRIPTOR_SCHEMA_VERSION,
            "capabilityRegistryDigest": self._capability_registry.digest,
            "eligibility": eligibility.protocol_value(),
            "loadDisposition": eligibility.disposition,
        }
        if matched is not None:
            from oryn_engine.chat_v2 import thinking_support

            result["_chatThinkingSupport"] = thinking_support(
                architecture, header["chatTemplate"] if header else ""
            )
            chat_template = llama_cpp_chat_template(
                architecture, header["chatTemplate"] if header is not None else None
            )
            if chat_template is None:
                blocking.append("TASK_ADAPTER_UNAVAILABLE")
                result["executable"] = False
                result["executableBackends"] = []
            summary = getattr(info, "gguf", None)
            context_length = (
                header["contextLength"]
                if header is not None
                else summary.get("context_length")
                if isinstance(summary, dict)
                else None
            )
            if (
                not isinstance(context_length, int)
                or isinstance(context_length, bool)
                or not 128 <= context_length <= 4_194_304
            ):
                context_length = 8_192
            binding_digest = pack.digest
            result.update(
                {
                    "runner": "runtime-pack",
                    "matchedCapability": matched.trusted_projection(),
                    "runtimePack": {
                        "packId": "oryn.llama-cpp",
                        "packVersion": pack.version,
                        "runnerId": runner_id,
                        "taskContract": contract,
                        "compatibilityLevel": "native-declarative",
                        "manifestDigest": binding_digest,
                        "trustedMetadata": {
                            "license": "Repository-defined",
                            "quantization": quantization_hint or "UNKNOWN",
                            "contextLength": context_length,
                            "chatTemplate": chat_template or "plain",
                            "modelFile": selected_file.name
                            if selected_file is not None
                            else "model.gguf",
                            "adapterArchitecture": architecture,
                            **({"projectorFile": projector.name} if projector else {}),
                            "conformance": {
                                "status": "fixture-only",
                                "platforms": ["macos-arm64-metal", "cpu"],
                            },
                        },
                    },
                }
            )
        if result["executable"]:
            adapter = self._runtime_foundations.adapter_for_legacy_pack(pack.pack_id)
            if adapter is None:
                raise ModelStoreError(
                    "RUNTIME_DEPENDENCY_UNAVAILABLE", "GGUF adapter is unavailable."
                )
            foundation = self._runtime_foundations.active_foundation(adapter.foundation_id)
            hardware = self._hardware_info()
            backend = "metal" if hardware.unified_memory else "cpu"
            candidate = RuntimeCandidate(
                identity=RuntimeIdentity(
                    foundation.foundation_id,
                    foundation.version,
                    adapter.adapter_id,
                    adapter.version,
                    "oryn.generated.gguf-chat",
                    backend,
                    "quantized",
                    quantization_hint,
                ),
                logical_model_id=f"huggingface:{repository}@{revision}",
                equivalence_class="exact-model",
                task_contract=contract,
                estimated_memory_bytes=max(1, estimated_memory),
                download_bytes=logical_size,
                conformance="untested",
                installed=False,
                resident=False,
                runtime_available=runtime_available,
                trusted=True,
                abi_compatible=foundation.abi == adapter.foundation_abi,
                task_compatible=True,
                files_complete=selected_file is not None,
                access_granted=not access_managed or access_granted,
                license_allowed=True,
                security_allowed=True,
                integrity_digest=canonical_digest(
                    {
                        "files": identity,
                        "pack": pack.digest,
                        "foundation": foundation.digest,
                        "adapter": adapter.digest,
                    }
                ),
            )
            if offers is not None:
                offers.append((candidate, result, None))
            else:
                result["runtimePlan"] = self._runtime_routing.plan(
                    [candidate],
                    context=RoutingContext.current(
                        self._capability_registry.digest, {"task": contract}
                    ),
                    available_memory_bytes=available_memory,
                    available_backends=(backend,),
                    policy=RuntimePolicy(_policy),
                ).trusted_summary()
            result["executableBackends"] = [backend]
        if blocking:
            result["blockingReasons"] = list(dict.fromkeys(blocking))
        if repository_executable_files:
            result["unselectedExecutableFiles"] = list(repository_executable_files[:256])
        return result

    @staticmethod
    def _gguf_candidates(files: list[ModelFile]) -> list[tuple]:
        """Complete GGUF variants, ordered by the quantization their names suggest.

        A variant is a root file or a file in one quantization folder. A name label only
        orders candidates; names without a known label come last.
        """
        from oryn_engine.runtime_packs.gguf import safe_member_name

        names = [f.name for f in files]
        sizes = {f.name: f.size for f in files}
        candidates = []
        for item in files:
            path = PurePosixPath(item.name)
            lowered = path.name.lower()
            if (
                path.suffix.lower() != ".gguf"
                or not safe_member_name(item.name)
                or "mmproj" in lowered
                or "imatrix" in lowered
                or lowered.startswith("mtp-")
            ):
                continue
            upper = path.stem.upper()
            label = next(
                (
                    value
                    for value in LLAMA_CPP_CHAT_QUANTIZATIONS
                    if re.search(rf"(?:^|[-_.]){re.escape(value)}(?:$|[-_.])", upper)
                ),
                None,
            )
            members = split_file_set(names, item.name)
            if not members:
                continue
            rank = (
                LLAMA_CPP_CHAT_QUANTIZATIONS.index(label)
                if label is not None
                else len(LLAMA_CPP_CHAT_QUANTIZATIONS)
            )
            candidates.append(
                (rank, sum(sizes[m] for m in members), item.name, item, label, members)
            )
        return sorted(candidates, key=lambda candidate: candidate[:3])

    @staticmethod
    def _select_generic_gguf_file(
        files: list[ModelFile],
    ) -> tuple[ModelFile | None, str | None]:
        """Name-only provisional choice, used when no header can be read.

        Installation re-reads every header and checks the signed policy.
        """
        labelled = [c for c in ModelStore._gguf_candidates(files) if c[4] is not None]
        if not labelled:
            return None, None
        _rank, _size, _name, selected, quantization, _members = labelled[0]
        return selected, quantization

    def _gguf_memory_budget(self) -> int:
        return int(psutil.virtual_memory().total * GGUF_VARIANT_MEMORY_SHARE)

    def _choose_gguf_by_header(self, repository, revision, files, task):
        """Choose a GGUF variant by what its header declares, not by its file name.

        Candidates that fit the memory budget are tried first, each in name order. A
        candidate is accepted when its header names an admitted architecture, a signed
        quantization and tokenizer, and an embedded template that selects exactly one
        sealed renderer. Image projectors are accepted by their header too. Returns None
        when no header could be read; the caller then keeps the name-only choice.
        """
        from types import SimpleNamespace

        from oryn_engine.generated_capabilities import architectures_for
        from oryn_engine.native_capability_sources import (
            NativeDescriptorError,
            validate_gguf_policy,
        )
        from oryn_engine.runtime_packs.gguf import check_projector, safe_member_name

        reader = getattr(self._hub, "read_gguf_metadata", None)
        if not callable(reader):
            return None
        failures = (ModelStoreError, GGUFError, httpx.HTTPError, OSError, ValueError, KeyError)
        chat = "translation" if task == "translation" else "text-generation"
        admitted = set(architectures_for(chat, source="gguf"))
        budget = self._gguf_memory_budget()
        candidates = sorted(
            self._gguf_candidates(files),
            key=lambda c: (int(c[1] * 1.25) > budget, c[0], c[1], c[2]),
        )
        seen = None
        chosen = None
        for _rank, _total, name, item, _label, members in candidates[:MAX_GGUF_HEADER_READS]:
            try:
                header = reader(repository, revision, name, item.size, len(members), False)
                seen = seen or header
                validate_gguf_policy(
                    SimpleNamespace(
                        version=header["version"],
                        architecture=header["architecture"],
                        quantization=header["quantization"],
                        tokenizer_model=header["tokenizerModel"],
                        chat_template=header["chatTemplate"],
                    )
                )
            except (*failures, NativeDescriptorError, TypeError):
                continue
            if header["architecture"] in admitted:
                chosen = (item, header)
                break
        if seen is None:
            return None
        projector = None
        if (
            chosen is not None
            and task == "image-text-to-text"
            and chosen[1]["architecture"] in architectures_for(task, source="gguf")
        ):
            folder = PurePosixPath(chosen[0].name).parent

            def order(f):
                lowered = PurePosixPath(f.name).name.lower()
                plain = 0 if "f16" in lowered and "bf16" not in lowered else 1
                return (PurePosixPath(f.name).parent != folder, plain, f.name)

            projectors = sorted(
                (
                    f
                    for f in files
                    if safe_member_name(f.name)
                    and f.name.lower().endswith(".gguf")
                    and "mmproj" in PurePosixPath(f.name).name.lower()
                    and PurePosixPath(f.name).parent in {folder, PurePosixPath(".")}
                ),
                key=order,
            )
            for item in projectors[:MAX_GGUF_PROJECTOR_READS]:
                try:
                    header = reader(repository, revision, item.name, item.size, 1, True)
                    check_projector(
                        SimpleNamespace(
                            projector_type=header["projectorType"],
                            projection_length=header["projectionLength"],
                            tensor_types=tuple(header["tensorTypes"]),
                        ),
                        chosen[1]["architecture"],
                        chosen[1]["embeddingLength"],
                    )
                except (*failures, TypeError):
                    continue
                projector = item
                break
        return SimpleNamespace(
            file=chosen[0] if chosen else None,
            header=chosen[1] if chosen else None,
            seen=seen,
            projector=projector,
        )

    def _compatibility_projection(
        self,
        *,
        repository: str,
        revision: str,
        task: str | None,
        architectures: list[str],
        files: list[ModelFile],
        requires_custom_code: bool,
        access: str,
        estimated_memory: int,
        blocking: list[str],
        config: dict[str, Any],
    ) -> dict[str, Any] | None:
        names = [item.name for item in files]
        # Exact native packs retain their signed admission until their binary descriptor is read.
        if any(name.endswith((".gguf", ".onnx")) for name in names):
            return None
        source_kind = (
            "diffusers"
            if {"model_index.json", "modular_model_index.json"} & set(names)
            else "transformers"
        )
        _policy, available = self._runtime_planning_state_provider()
        artifact_blockers = {
            "MODEL_SUBSET_INCOMPLETE",
            "MODEL_CONFIG_MISSING",
            "MODEL_TOKENIZER_MISSING",
            "MODEL_PROCESSOR_MISSING",
            "MODEL_SHARDS_INVALID",
            "MODEL_FILE_LAYOUT_INVALID",
            "UNSAFE_MODEL_FORMAT",
        }
        assessment = inspect_descriptor(
            repository=repository,
            revision=revision,
            files=names,
            config=config,
            task=task,
            registry=self._capability_registry,
            access=access,
            source_kind=source_kind,
            file_identities=[(item.name, item.size, item.blob_id) for item in files],
            estimated_memory=estimated_memory,
            available_memory=max(0, available),
            artifact_reasons=tuple(reason for reason in blocking if reason in artifact_blockers),
            runtime_reasons=tuple(
                reason
                for reason in blocking
                if reason not in artifact_blockers and reason != "MODEL_ACCESS_RESTRICTED"
            ),
        )
        eligibility = assessment.eligibility
        if eligibility.matched is None and source_kind == "diffusers":
            return None
        flattened = sorted(
            {reason for axis in eligibility.axes.values() for reason in axis.reasons}
        )
        return {
            "descriptorSchemaVersion": DESCRIPTOR_SCHEMA_VERSION,
            "capabilityRegistryDigest": self._capability_registry.digest,
            "eligibility": eligibility.protocol_value(),
            "loadDisposition": eligibility.disposition,
            **(
                {"matchedCapability": eligibility.matched.trusted_projection()}
                if eligibility.matched
                else {}
            ),
            **({"blockingReasons": flattened} if flattened else {}),
            "executable": eligibility.disposition in {"verified-load", "trust-required"},
        }

    def _plan_runtime_matches(
        self,
        matches: tuple[CompatibilityManifest, ...],
        *,
        repository: str,
        revision: str,
        files: list[ModelFile],
        access_granted: bool,
        requires_custom_code: bool,
        has_safe_weights: bool,
        has_unsafe_weights: bool,
    ) -> RuntimePlan:
        candidates: list[RuntimeCandidate] = []
        hardware = self._hardware_info()
        available = {item["deviceClass"] for item in hardware.backends} | {"cpu"}
        if hardware.platform == "darwin" and hardware.architecture in {"arm64", "aarch64"}:
            available.update({"metal", "mlx", "coreml"})
        preferred = [hardware.preferred_backend]
        if hardware.preferred_backend == "mps":
            preferred.append("metal")
        preferred.extend(["mlx", "coreml", "cpu"])
        preferred_backends = tuple(dict.fromkeys(item for item in preferred if item in available))
        logical_model_id = f"huggingface:{repository}@{revision}"

        for match in matches:
            adapter = self._runtime_foundations.adapter_for_legacy_pack(match.pack_id)
            if adapter is None:
                continue
            foundation = self._runtime_foundations.active_foundation(adapter.foundation_id)
            pack = self._runtime_packs.active_pack(match.pack_id)
            runner = pack.runner(match.runner_id)
            declared_backends = runner.backends if runner is not None else ()
            backends = tuple(item for item in declared_backends if item in available)
            if match.pack_id == "oryn.transformers-pytorch":
                runtime_available = self._runtime_available
            else:
                runtime_available = self._pack_runtime_locator.resolve(pack) is not None
            selected_files = self._installation_files(files, match)
            estimated_memory = max(
                int(sum(item.size for item in selected_files) * 1.25),
                match.resources.get("estimatedResidentBytes", 0)
                + match.resources.get("workingMemoryBytes", 0),
            )
            metadata = match.trusted_metadata
            quantization = metadata.get("quantization")
            if not isinstance(quantization, str):
                quantization = None
            conformance_value = metadata.get("conformance")
            conformance = (
                conformance_value.get("status", "developer-tested")
                if isinstance(conformance_value, dict)
                else "developer-tested"
            )
            candidate_backends = backends or declared_backends[:1] or ("cpu",)
            for backend in candidate_backends:
                candidates.append(
                    RuntimeCandidate(
                        identity=RuntimeIdentity(
                            foundation.foundation_id,
                            foundation.version,
                            adapter.adapter_id,
                            adapter.version,
                            match.manifest_id,
                            backend,
                            "quantized"
                            if quantization
                            else ("fp32" if backend == "cpu" else "fp16"),
                            quantization,
                        ),
                        logical_model_id=logical_model_id,
                        equivalence_class=adapter.equivalence_class,
                        task_contract=match.task_contract,
                        estimated_memory_bytes=estimated_memory,
                        download_bytes=sum(item.size for item in selected_files),
                        conformance=conformance,
                        installed=False,
                        resident=False,
                        runtime_available=runtime_available,
                        trusted=True,
                        abi_compatible=foundation.abi == adapter.foundation_abi,
                        task_compatible=match.task_contract in adapter.task_contracts,
                        files_complete=self._required_patterns_present(files, match),
                        access_granted=access_granted,
                        license_allowed=True,
                        security_allowed=(
                            (not requires_custom_code or match.level == "audited-adapter")
                            and not (
                                has_unsafe_weights
                                and not has_safe_weights
                                and not (
                                    match.pack_id == "oryn.kokoro-onnx"
                                    and match.level == "converted-artifact"
                                )
                            )
                        ),
                        integrity_digest=canonical_digest(
                            {
                                "foundation": foundation.digest,
                                "adapter": adapter.digest,
                                "manifest": match.digest,
                                "files": [
                                    (item.name, item.size, item.blob_id) for item in selected_files
                                ],
                            }
                        ),
                        quality_score=50,
                        performance_score=50,
                        reliability_score=(
                            95
                            if conformance == "passed" and match.repository is not None
                            else 85
                            if match.repository is not None
                            else 60
                        ),
                    )
                )
        policy_value, available_memory = self._runtime_planning_state_provider()
        try:
            policy = RuntimePolicy(policy_value)
        except ValueError:
            policy = RuntimePolicy.BALANCED
        return self._runtime_routing.plan(
            candidates,
            context=RoutingContext.current(
                self._capability_registry.digest, {"policy": policy.value}
            ),
            policy=policy,
            available_memory_bytes=available_memory,
            available_backends=preferred_backends,
        )

    def _provider_requires_reinspection(self, record):
        runtime_pack = record.get("runtimePack")
        if not isinstance(runtime_pack, dict):
            return False
        try:
            pack = self._runtime_packs.active_pack(runtime_pack["packId"])
        except Exception:
            return False
        if pack.provider is None:
            return False
        selected = record.get("runtimePlan", {}).get("selected", {})
        bound = record.get("providerPlans", {}).get(selected.get("candidateId"), {})
        return (
            bound.get("packDigest") != pack.digest
            or record.get("runtimeBuildDigest") != runtime_build_digest()
            or record.get("capabilityRegistryDigest") != self._capability_registry.digest
        )

    def _installed_inspection(
        self, repository: str, revision: str, files: list[str] | None = None
    ) -> dict[str, Any] | None:
        key = self._model_key(
            {
                "provider": "huggingface",
                "repository": repository,
                "revision": revision,
                **({"files": files} if files else {}),
            }
        )
        with self._lock:
            record = next(
                (
                    item
                    for item in self._records
                    if item.get("installationState") == "installed"
                    and self._model_key(item["model"]) == key
                ),
                None,
            )
            if record is None:
                return None
            record = dict(record)

        if record.get("sourceRecognition"):
            from oryn_engine.reviewed_implementations import validate_evidence

            try:
                validate_evidence(record["sourceRecognition"])
            except (ValueError, OSError):
                return None
            record["admissionScope"] = "huggingface-page"

        runtime_pack = record.get("runtimePack")
        if self._provider_requires_reinspection(record):
            return None
        blocking: list[str] = []
        from oryn_protocol.task_contracts import REGISTRY_DIGEST

        if any(
            o.get("registryDigest") != REGISTRY_DIGEST
            for o in record.get("executionOffers", {}).values()
        ):
            blocking.append("TASK_CONTRACT_REGISTRY_CHANGED")
        if record.get("runtimeBuildDigest") not in {None, runtime_build_digest()}:
            blocking.append("RUNTIME_BUILD_CHANGED")
        if isinstance(runtime_pack, dict):
            if runtime_pack.get("packId") == "oryn.transformers-pytorch":
                runtime_available = self._runtime_available
            else:
                try:
                    pack = self._runtime_packs.active_pack(runtime_pack["packId"])
                except Exception:
                    runtime_available = False
                else:
                    runtime_available = self._pack_runtime_locator.resolve(pack) is not None
        else:
            runtime_available = self._runtime_available
        if not runtime_available:
            blocking.append("RUNTIME_DEPENDENCY_UNAVAILABLE")

        result: dict[str, Any] = {
            "model": dict(record["model"]),
            "exactRevision": revision,
            "executable": not blocking,
            "executableBackends": (
                list(record.get("executableBackends", [])) if not blocking else []
            ),
            "requiresCustomCode": bool(
                isinstance(runtime_pack, dict)
                and runtime_pack.get("compatibilityLevel") == "audited-adapter"
            ),
            "estimatedDiskBytes": record["logicalSizeBytes"],
            "estimatedMemoryBytes": record["estimatedResidentBytes"],
            "selectedFiles": [
                {"name": item["name"], "sizeBytes": item["size"]}
                for item in record.get("files", [])
                if isinstance(item, dict)
                and isinstance(item.get("name"), str)
                and isinstance(item.get("size"), int)
            ],
            "technicalCapabilities": dict(record["technicalCapabilities"]),
            "runner": record["runner"],
        }
        if isinstance(runtime_pack, dict):
            result["runtimePack"] = dict(runtime_pack)
        plan = record.get("runtimePlan")
        if isinstance(plan, dict) and self._runtime_routing.is_quarantined(
            plan.get("planId"), plan.get("selected", {}).get("candidateId", "")
        ):
            blocking.append("RUNTIME_QUARANTINED")
        if isinstance(record.get("runtimePlan"), dict):
            result["runtimePlan"] = dict(record["runtimePlan"])
        if isinstance(record.get("taskProfile"), dict):
            result["taskProfile"] = dict(record["taskProfile"])
        if isinstance(record.get("generationProfile"), dict):
            result["generationProfile"] = dict(record["generationProfile"])
        for field in (
            "eligibility",
            "loadDisposition",
            "descriptorSchemaVersion",
            "capabilityRegistryDigest",
            "matchedCapability",
            "sourceRecognition",
            "admissionScope",
        ):
            if field in record:
                result[field] = record[field]
        if isinstance(record.get("eligibility"), dict):
            axes = {name: dict(value) for name, value in record["eligibility"].items()}
            _policy, available = self._runtime_planning_state_provider()
            if record["estimatedResidentBytes"] > available:
                blocking.append("MODEL_MEMORY_INSUFFICIENT")
                axes["resources"] = {
                    "state": "not-admissible",
                    "reasons": ["MODEL_MEMORY_INSUFFICIENT"],
                }
                result["loadDisposition"] = "not-admissible-on-this-device"
            if record.get("capabilityRegistryDigest") != self._capability_registry.digest:
                blocking.append("CAPABILITY_REGISTRY_CHANGED")
            if blocking:
                result["executable"] = False
                result["executableBackends"] = []
                if result.get("loadDisposition") != "not-admissible-on-this-device":
                    result["loadDisposition"] = "unsupported"
                    axes["runtime"] = {"state": "unsupported", "reasons": sorted(set(blocking))}
            result["eligibility"] = axes
        if blocking:
            result["blockingReasons"] = blocking
            result["executable"] = False
            result["executableBackends"] = []
        if isinstance(result.get("eligibility"), dict):
            # The legacy summary must retain every eligibility reason, including
            # conformance/trust reasons when current memory adds a new blocker.
            # Those informational reasons alone do not make a model unexecutable.
            reasons = sorted(
                set(blocking)
                | {reason for axis in result["eligibility"].values() for reason in axis["reasons"]}
            )
            if reasons:
                result["blockingReasons"] = reasons
        return result

    def source_url(self, model: dict[str, Any]) -> str:
        repository, revision = self._validate_model_ref(model)
        return f"https://huggingface.co/{repository}/tree/{revision}"

    def catalog(self, task: str, output_modalities: list[str]) -> dict[str, Any]:
        if not isinstance(task, str) or not 1 <= len(task) <= 128:
            raise ModelStoreError("MODEL_REQUEST_INVALID", "Model task is invalid.")
        allowed = {"text", "image", "audio", "video", "embedding", "structured"}
        if (
            not isinstance(output_modalities, list)
            or not output_modalities
            or len(output_modalities) > len(allowed)
            or any(not isinstance(item, str) or item not in allowed for item in output_modalities)
        ):
            raise ModelStoreError("MODEL_REQUEST_INVALID", "Output modalities are invalid.")
        return {
            "models": list(
                self._runtime_packs.catalog(task=task, output_modalities=output_modalities)
            )
        }

    def install(
        self,
        operation_id: str,
        model: dict[str, Any],
        requirements: dict | None = None,
        recommendation: dict | None = None,
        *,
        source_context: dict | None = None,
    ) -> dict[str, Any]:
        self._validate_operation_id(operation_id)
        profile = self.inspect(
            model,
            requirements,
            **({"recommendation": recommendation} if recommendation else {}),
            **(
                {"source_context": source_context, "_source_cancel_id": operation_id}
                if source_context is not None
                else {}
            ),
        )
        exact_model = profile["model"]
        key = self._model_key(exact_model)
        with self._lock:
            existing = next(
                (record for record in self._records if self._model_key(record["model"]) == key),
                None,
            )
            context = self._inspection_contexts.get(canonical_digest(profile))
            if (
                context is None
                and profile.get("taskProfile")
                and (
                    existing is None
                    or existing.get("taskProfile") != profile["taskProfile"]
                    or existing.get("runtimePlan") != profile.get("runtimePlan")
                )
            ):
                raise ModelStoreError(
                    "RUNTIME_PLAN_INVALID",
                    "The model inspection expired. Retry installation.",
                    retryable=True,
                )
            provider_plans, execution_offers, routing_candidates = (
                context
                if context is not None
                else (
                    dict((existing or {}).get("providerPlans", {})),
                    dict((existing or {}).get("executionOffers", {})),
                    list((existing or {}).get("routingCandidates", [])),
                )
            )
            if existing:
                if existing.get("installationState") != "installed":
                    raise ModelStoreError(
                        "MODEL_REMOVAL_PENDING", "The model installation is being removed."
                    )
                if (
                    existing.get("runtimeBuildDigest") in {None, runtime_build_digest()}
                    and existing.get("capabilityRegistryDigest")
                    in {None, self._capability_registry.digest}
                    and (
                        not requirements
                        or existing.get("taskProfile", {}).get("contract")
                        == profile.get("taskProfile", {}).get("contract")
                    )
                    and (
                        not execution_offers or existing.get("executionOffers") == execution_offers
                    )
                    and (
                        not provider_plans
                        or (
                            existing.get("providerPlans") == provider_plans
                            and existing.get("runtimeBuildDigest") == runtime_build_digest()
                            and existing.get("capabilityRegistryDigest")
                            == self._capability_registry.digest
                        )
                    )
                ):
                    if profile.get("runtimePreparationRequired"):
                        usage = self._runtime_state(exact_model)
                        if (
                            any(
                                usage.get(k, 0)
                                for k in (
                                    "activeSessions",
                                    "activeJobs",
                                    "runningJobs",
                                    "queuedJobs",
                                )
                            )
                            or usage.get("residencyState") == "loaded"
                        ):
                            raise ModelStoreError(
                                "TASK_SESSION_CONFLICT",
                                "Release active sessions before changing this model's runtime.",
                            )
                        existing["runtimePlan"] = profile["runtimePlan"]
                        self._apply_plan_runtime_pack(existing, profile["runtimePlan"])
                        existing["executableBackends"] = profile["executableBackends"]
                        existing["estimatedResidentBytes"] = profile["estimatedMemoryBytes"]
                        self._refresh_installed_generation_profile(existing)
                        self._save_registry()
                    return self._install_result(operation_id, existing)
            if existing and (
                existing.get("taskProfile") != profile.get("taskProfile")
                or existing.get("runtimeBuildDigest") != runtime_build_digest()
                or existing.get("runtimePlan", {}).get("selected")
                != profile.get("runtimePlan", {}).get("selected")
            ):
                usage = self._runtime_state(exact_model)
                if (
                    any(
                        usage.get(k, 0)
                        for k in ("activeSessions", "activeJobs", "runningJobs", "queuedJobs")
                    )
                    or usage.get("residencyState") == "loaded"
                ):
                    raise ModelStoreError(
                        "TASK_SESSION_CONFLICT",
                        "Release active sessions before changing this model's task or runtime.",
                    )
            active = next(
                (item for item in self._operations.values() if self._model_key(item.model) == key),
                None,
            )
            if active:
                raise ModelStoreError(
                    "MODEL_INSTALL_IN_PROGRESS",
                    "The exact model revision is already being installed.",
                    details={"operationId": active.operation_id},
                )
            if (
                profile["requiresCustomCode"]
                and profile.get("runtimePack", {}).get("compatibilityLevel") != "audited-adapter"
            ):
                raise ModelStoreError(
                    "CUSTOM_CODE_REQUIRED", "The model requires untrusted custom code."
                )
            if "UNSAFE_MODEL_FORMAT" in profile.get("blockingReasons", []):
                raise ModelStoreError(
                    "UNSAFE_MODEL_FORMAT", "The model does not provide safe weights."
                )
            if "MODEL_ACCESS_RESTRICTED" in profile.get("blockingReasons", []):
                raise ModelStoreError(
                    "MODEL_ACCESS_RESTRICTED",
                    "The model host requires approval or authentication for this model.",
                )
            if not profile["executable"]:
                raise ModelStoreError(
                    "MODEL_NOT_EXECUTABLE", "No bundled runner can execute this model."
                )
            operation = InstallOperation(
                operation_id,
                existing["installationId"] if existing else f"install_{uuid.uuid4().hex}",
                exact_model,
                threading.Event(),
                profile["estimatedDiskBytes"],
                profile["estimatedMemoryBytes"],
            )
            self._operations[operation_id] = operation

        partial = self.partial_root / operation_id
        partial.mkdir(parents=True, exist_ok=False)
        try:
            info = self._hub.model_info(
                exact_model["repository"], exact_model["revision"], files_metadata=True
            )
            if getattr(info, "sha", None) != exact_model["revision"]:
                raise ModelStoreError(
                    "MODEL_METADATA_INVALID",
                    "Model revision changed during installation.",
                )
            all_files = self._validate_files(
                getattr(info, "siblings", None),
                f"{exact_model['repository']}@{exact_model['revision']}",
            )
            config = getattr(info, "config", None) or {}
            architectures = config.get("architectures", []) if isinstance(config, dict) else []
            task = getattr(info, "pipeline_tag", None)
            matches = self._runtime_packs.match(
                repository=exact_model["repository"],
                revision=exact_model["revision"],
                architectures=architectures if isinstance(architectures, list) else [],
                task=task if isinstance(task, str) else None,
            )
            profile_runtime_pack = profile.get("runtimePack")
            generic_gguf = bool(
                isinstance(profile_runtime_pack, dict)
                and profile_runtime_pack.get("packId") == "oryn.llama-cpp"
                and profile_runtime_pack.get("compatibilityLevel") == "native-declarative"
            )
            selected_variant = profile.get("runtimePlan", {}).get("selected", {}).get("variantId")
            compatibility = next(
                (item for item in matches if item.manifest_id == selected_variant),
                matches[0] if matches else None,
            )
            if generic_gguf:
                metadata = profile_runtime_pack.get("trustedMetadata", {})
                selected_name = metadata.get("modelFile") if isinstance(metadata, dict) else None
                selected_names = split_file_set([f.name for f in all_files], selected_name or "")
                if isinstance(metadata, dict) and metadata.get("projectorFile"):
                    selected_names.append(metadata["projectorFile"])
                files = sorted(
                    (f for f in all_files if f.name in selected_names), key=lambda f: f.name
                )
                if not files or {f.name for f in files} != {
                    f["name"] for f in profile["selectedFiles"]
                }:
                    raise ModelStoreError(
                        "MODEL_SUBSET_INCOMPLETE",
                        "The selected GGUF variant is no longer available.",
                    )
            elif (
                execution_offers
                or str(selected_variant).startswith("oryn.provider.")
                or selected_variant
                in {
                    "oryn.generated.mlx-chat",
                    "oryn.generated.mlx-vad",
                    "oryn.generated.diffusers",
                    "oryn.generated.transformers-tasks",
                }
            ):
                names = {item["name"] for item in profile["selectedFiles"]}
                files = [item for item in all_files if item.name in names]
                if len(files) != len(names):
                    raise ModelStoreError("MODEL_SUBSET_INCOMPLETE", "Selected files changed.")
            else:
                files = self._installation_files(all_files, compatibility)
            total = sum(item.size for item in files)
            # Re-preparing an installed runtime does not duplicate its weights.
            # These paths receive the same size/hash verification below as fresh
            # downloads; disk credit never grants authority to cached contents.
            reusable_paths: dict[str, Path] = {}
            reusable_digests: dict[str, str] = {}
            previous_files = {item.get("name"): item for item in (existing or {}).get("files", [])}
            for item in files:
                previous = previous_files.get(item.name, {})
                relative = previous.get("cache_path")
                if (
                    not isinstance(relative, str)
                    or previous.get("blob_id") != item.blob_id
                    or not re.fullmatch(r"[0-9a-f]{64}", str(previous.get("content_sha256", "")))
                ):
                    continue
                cached = (self.cache_root / relative).absolute()
                target = cached.resolve()
                if (
                    cached.is_relative_to(self.cache_root)
                    and target.is_relative_to(self.cache_root)
                    and target.is_file()
                    and target.stat().st_size == item.size
                ):
                    reusable_paths[item.name] = cached
                    reusable_digests[item.name] = previous["content_sha256"]
            # A completed download can survive an interrupted installation. It
            # need not consume its entire size again, but an unadmitted cache
            # entry earns disk credit only after matching the publisher's hash.
            cached_file = getattr(self._hub, "cached_file", None)
            for item in files:
                if (
                    item.name in reusable_paths
                    or not callable(cached_file)
                    or not re.fullmatch(r"[0-9a-f]{64}", item.blob_id)
                ):
                    continue
                self._check_cancelled(operation)
                cached_name = cached_file(
                    exact_model["repository"],
                    item.name,
                    exact_model["revision"],
                    str(self.cache_root),
                )
                if not isinstance(cached_name, str):
                    continue
                cached = Path(cached_name).absolute()
                target = cached.resolve()
                if not cached.is_relative_to(self.cache_root) or not target.is_relative_to(
                    self.cache_root
                ):
                    raise ModelStoreError(
                        "MODEL_SECURITY_BLOCKED", "Model cache path escaped its root."
                    )
                if not target.is_file() or target.stat().st_size != item.size:
                    raise ModelStoreError("MODEL_INTEGRITY_FAILED", "Cached file size changed.")
                digest = hashlib.sha256()
                with target.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        self._check_cancelled(operation)
                        digest.update(chunk)
                if digest.hexdigest() != item.blob_id:
                    raise ModelStoreError("MODEL_INTEGRITY_FAILED", "Cached file digest changed.")
                reusable_paths[item.name] = cached
                reusable_digests[item.name] = item.blob_id
            available_bytes = shutil.disk_usage(self.root).free
            required_bytes = (
                sum(item.size for item in files if item.name not in reusable_paths)
                + 512 * 1024 * 1024
            )
            if available_bytes < required_bytes:
                raise ModelStoreError(
                    "DISK_SPACE_INSUFFICIENT",
                    "There is not enough disk space.",
                    details={
                        "requiredBytes": required_bytes,
                        "availableBytes": available_bytes,
                    },
                )
            self._state(operation, "downloading")
            self._progress(operation, "resolving", 0, total)
            downloaded = 0
            cache_paths: list[str] = []
            content_digests: list[str] = []
            for item in files:
                self._check_cancelled(operation)
                completed_before_file = downloaded
                file_progress_high_water = 0

                def report_file_progress(
                    file_downloaded: int,
                    _file_total: int | None,
                    *,
                    expected_size: int = item.size,
                    completed_offset: int = completed_before_file,
                ) -> None:
                    nonlocal file_progress_high_water
                    self._check_cancelled(operation)
                    bounded = min(expected_size, max(0, file_downloaded))
                    if bounded <= file_progress_high_water:
                        return
                    file_progress_high_water = bounded
                    self._progress(
                        operation,
                        "downloading",
                        completed_offset + bounded,
                        total,
                    )

                downloaded_path = (
                    reusable_paths.get(item.name)
                    or Path(
                        self._hub.download(
                            exact_model["repository"],
                            item.name,
                            exact_model["revision"],
                            str(self.cache_root),
                            report_file_progress,
                        )
                    ).absolute()
                )
                path = downloaded_path.resolve()
                if not downloaded_path.is_relative_to(self.cache_root) or not path.is_relative_to(
                    self.cache_root
                ):
                    raise ModelStoreError(
                        "MODEL_SECURITY_BLOCKED", "Model cache path escaped its root."
                    )
                if path.stat().st_size != item.size:
                    raise ModelStoreError("MODEL_INTEGRITY_FAILED", "Downloaded file size changed.")
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        self._check_cancelled(operation)
                        digest.update(chunk)
                content_digest = digest.hexdigest()
                if item.name in reusable_paths and content_digest != reusable_digests[item.name]:
                    raise ModelStoreError("MODEL_INTEGRITY_FAILED", "Cached file digest changed.")
                if re.fullmatch(r"[0-9a-f]{64}", item.blob_id) and content_digest != item.blob_id:
                    raise ModelStoreError(
                        "MODEL_INTEGRITY_FAILED", "Downloaded file digest changed."
                    )
                from oryn_engine.model_composition import binding as composition_binding

                if composition_binding(
                    exact_model["repository"], exact_model["revision"]
                ) and re.fullmatch(r"[0-9a-f]{40}", item.blob_id):
                    git_digest = hashlib.sha1(f"blob {item.size}\0".encode())
                    with path.open("rb") as stream:
                        while chunk := stream.read(1024 * 1024):
                            self._check_cancelled(operation)
                            git_digest.update(chunk)
                    if git_digest.hexdigest() != item.blob_id:
                        raise ModelStoreError(
                            "MODEL_INTEGRITY_FAILED", "Dependency metadata digest changed."
                        )
                content_digests.append(content_digest)
                cache_paths.append(str(downloaded_path.relative_to(self.cache_root)))
                downloaded += item.size
                self._progress(operation, "downloading", downloaded, total)
            self._check_cancelled(operation)
            self._progress(operation, "verifying", downloaded, total)
            for item, cache_path in zip(files, cache_paths, strict=True):
                if WEIGHT_INDEX.fullmatch(item.name):
                    index_path = self.cache_root / cache_path
                    try:
                        if index_path.stat().st_size > 16 * 1024**2:
                            raise ValueError("Weight index is oversized")
                        parent = str(PurePosixPath(item.name).parent)
                        selected = (
                            {x.name.removeprefix(parent + "/") for x in files}
                            if parent != "."
                            else {x.name for x in files}
                        )
                        validate_weight_index(json.loads(index_path.read_text()), selected)
                    except (OSError, ValueError) as exc:
                        raise ModelStoreError(
                            "MODEL_INTEGRITY_FAILED", "Weight index is invalid."
                        ) from exc
            installed_runtime_pack = (
                dict(profile_runtime_pack) if isinstance(profile_runtime_pack, dict) else None
            )
            installed_technical = dict(profile["technicalCapabilities"])
            if generic_gguf:
                provisional = profile_runtime_pack.get("trustedMetadata", {})
                projector_name = provisional.get("projectorFile")
                language = [
                    (f, p)
                    for f, p in zip(files, cache_paths, strict=True)
                    if f.name != projector_name
                ]
                try:
                    gguf = (
                        inspect_gguf((self.cache_root / language[0][1]).resolve())
                        if len(language) == 1
                        else inspect_gguf_set(
                            [(self.cache_root / p).resolve() for f, p in language],
                            [f.name for f, p in language],
                        )
                    )
                    if projector_name:
                        from oryn_engine.runtime_packs.gguf import inspect_projector

                        projection_path = next(
                            p
                            for f, p in zip(files, cache_paths, strict=True)
                            if f.name == projector_name
                        )
                        inspect_projector((self.cache_root / projection_path).resolve(), gguf)
                except GGUFError as exc:
                    raise ModelStoreError(
                        "MODEL_METADATA_INVALID",
                        "The downloaded GGUF failed bounded structural validation.",
                    ) from exc
                provisional = profile_runtime_pack.get("trustedMetadata", {})
                expected_architecture = (
                    provisional.get("adapterArchitecture")
                    if isinstance(provisional, dict)
                    else None
                )
                if gguf.architecture != expected_architecture:
                    raise ModelStoreError(
                        "MODEL_METADATA_INVALID",
                        "The downloaded GGUF architecture does not match its inspected descriptor.",
                    )
                if gguf.quantization not in LLAMA_CPP_CHAT_QUANTIZATIONS:
                    raise ModelStoreError(
                        "RUNTIME_CAPABILITY_UNSUPPORTED",
                        "The downloaded GGUF quantization is outside the packaged "
                        "llama.cpp policy.",
                    )
                from oryn_engine.native_capability_sources import validate_gguf_policy

                validate_gguf_policy(gguf)
                chat_template = llama_cpp_chat_template(gguf.architecture, gguf.chat_template)
                if chat_template is None:
                    raise ModelStoreError(
                        "TASK_ADAPTER_UNAVAILABLE",
                        "No sealed chat adapter supports the downloaded GGUF architecture.",
                    )
                installed_runtime_pack["trustedMetadata"] = {
                    "license": "Repository-defined",
                    "quantization": gguf.quantization,
                    "contextLength": gguf.context_length,
                    "chatTemplate": chat_template,
                    "modelFile": language[0][0].name,
                    **({"projectorFile": projector_name} if projector_name else {}),
                    "adapterArchitecture": gguf.architecture,
                    "conformance": {
                        "status": "fixture-only",
                        "platforms": ["macos-arm64-metal", "cpu"],
                    },
                }
                installed_technical["architectures"] = [gguf.architecture]
                if installed_runtime_pack.get("taskContract") == "translation-v1":
                    from oryn_engine import gguf_translation
                    from oryn_engine.task_profiles import profile_for_execution

                    gguf_translation.validate_binding(
                        gguf.architecture, chat_template, projector_name
                    )
                    if gguf.tokenizer_model != "gpt2":
                        raise ModelStoreError(
                            "MODEL_METADATA_INVALID",
                            "GGUF translation requires its native byte tokenizer.",
                        )
                    profile["taskProfile"] = profile_for_execution(
                        {
                            "runtimePack": installed_runtime_pack,
                            "technicalCapabilities": installed_technical,
                        }
                    )
            record = {
                "installationId": operation.installation_id,
                "model": exact_model,
                "installationState": "installed",
                "residencyState": "unloaded",
                "logicalSizeBytes": total,
                "estimatedResidentBytes": profile["estimatedMemoryBytes"],
                "runner": profile.get("runner"),
                "executableBackends": profile.get("executableBackends", []),
                "technicalCapabilities": installed_technical,
                **(
                    {"runtimePack": installed_runtime_pack}
                    if installed_runtime_pack is not None
                    else {}
                ),
                **(
                    {"runtimePlan": profile["runtimePlan"]}
                    if profile.get("runtimePlan") is not None
                    else {}
                ),
                **(
                    {"generationProfile": profile["generationProfile"]}
                    if profile.get("generationProfile") is not None
                    else {}
                ),
                **{
                    key: profile[key]
                    for key in (
                        "eligibility",
                        "loadDisposition",
                        "descriptorSchemaVersion",
                        "capabilityRegistryDigest",
                        "matchedCapability",
                    )
                    if key in profile
                },
                **({"taskProfile": profile["taskProfile"]} if profile.get("taskProfile") else {}),
                **({"executionOffers": execution_offers} if execution_offers else {}),
                **({"routingCandidates": routing_candidates} if routing_candidates else {}),
                "installedAt": int(time.time() * 1000),
                "runtimeBuildDigest": runtime_build_digest(),
                **(
                    {
                        "sourceRecognition": profile["sourceRecognition"],
                        "admissionScope": "huggingface-page",
                    }
                    if profile.get("sourceRecognition", {}).get("admitted")
                    else {}
                ),
                **({"providerPlans": provider_plans} if provider_plans else {}),
                "files": [
                    item.__dict__
                    | {"cache_path": cache_paths[index], "content_sha256": content_digests[index]}
                    for index, item in enumerate(files)
                ],
            }
            self._prepare_provider_installation(record, operation.cancel)
            self._refresh_installed_runtime_metadata(record)
            self._refresh_installed_generation_profile(record)
            self._progress(operation, "installing", downloaded, total)
            with self._lock:
                if existing:
                    self._records = [
                        r
                        for r in self._records
                        if r["installationId"] != existing["installationId"]
                    ]
                self._records.append(record)
                self._save_registry()
            self._progress(operation, "finalizing", downloaded, total)
            self._state(operation, "installed")
            self._emit("model.inventory.changed", {"reason": "installed"})
            return self._install_result(operation_id, record)
        except InstallCancelled:
            self._state(operation, "cancelled")
            raise
        except Exception:
            self._state(operation, "error")
            raise
        finally:
            shutil.rmtree(partial, ignore_errors=True)
            with self._lock:
                self._operations.pop(operation_id, None)

    def _prepare_provider_installation(self, record, cancellation):
        from oryn_engine.providers import provider_variant, verify_plan
        from oryn_engine.runners import _selected_model_view
        from oryn_engine.runtime_packs.execution import StreamingPackProcess

        selected = record.get("runtimePlan", {}).get("selected", {})
        if record.get("runtimePack", {}).get("packId") in {
            "oryn.onnx-runtime",
            "oryn.qwen-tts-pytorch",
        }:
            pack = self._runtime_packs.active_pack(record["runtimePack"]["packId"])
            runtime = self._pack_runtime_locator.resolve(pack)
            if runtime is None:
                raise ModelStoreError(
                    "RUNTIME_DEPENDENCY_UNAVAILABLE", "Native task runtime is unavailable."
                )
            model = record["model"]
            model_root = (
                self.cache_root
                / f"models--{model['repository'].replace('/', '--')}"
                / "snapshots"
                / model["revision"]
            )
            with _selected_model_view(
                record | {"cacheRoot": str(self.cache_root)}, model_root
            ) as root:
                prepared = (self._provider_process or StreamingPackProcess()).invoke(
                    pack,
                    runtime,
                    "prepare",
                    {
                        "modelRoot": str(root),
                        "task": record["technicalCapabilities"]["task"],
                        "backend": selected["backend"],
                        "input": {},
                        "inputPath": None,
                        "outputPath": None,
                    },
                    cancellation=cancellation,
                    event_sink=lambda *_: None,
                    timeout_seconds=120,
                )
            if prepared != {"prepared": True}:
                raise ModelStoreError(
                    "MODEL_PREPARATION_FAILED", "Native graph preparation failed."
                )
            return
        if not provider_variant(selected.get("variantId")):
            return
        bound = record.get("providerPlans", {}).get(selected.get("candidateId"))
        if not isinstance(bound, dict):
            raise ModelStoreError("RUNTIME_PLAN_INVALID", "Provider plan is unavailable.")
        pack = self._runtime_packs.active_pack(bound["packId"])
        runtime = self._pack_runtime_locator.resolve(pack)
        if runtime is None:
            raise ModelStoreError("RUNTIME_DEPENDENCY_UNAVAILABLE", "Provider pack is unavailable.")
        verify_plan(bound, pack, runtime, record)
        model = record["model"]
        model_root = (
            self.cache_root
            / f"models--{model['repository'].replace('/', '--')}"
            / "snapshots"
            / model["revision"]
        )
        view_record = record | {"cacheRoot": str(self.cache_root)}
        with _selected_model_view(view_record, model_root) as root:
            prepared = (self._provider_process or StreamingPackProcess()).invoke(
                pack,
                runtime,
                "prepare",
                {
                    "modelRoot": str(root),
                    "plan": bound["providerPlan"],
                    "input": {"prompt": "Installation validation"},
                },
                cancellation=cancellation,
                event_sink=lambda *_: None,
                timeout_seconds=120,
            )
        if (
            prepared.get("ready") is not True
            or prepared.get("catalogDigest") != bound["catalogDigest"]
            or prepared.get("estimatedMemoryBytes") != record["estimatedResidentBytes"]
        ):
            raise ModelStoreError(
                "MODEL_INTEGRITY_FAILED", "Provider preparation changed its plan."
            )

    def cancel_install(self, operation_id: str) -> bool:
        self._validate_operation_id(operation_id)
        with self._lock:
            source = self._source_cancellations.get(operation_id)
            if source is not None:
                source.set()
                return True
            operation = self._operations.get(operation_id)
            if not operation:
                return False
            operation.cancel.set()
            return True

    def inventory(self) -> dict[str, Any]:
        with self._lock:
            counts = self._blob_reference_counts()
            records = [self._public_record(record, counts) for record in self._records]
            records.extend(
                {
                    "installationId": operation.installation_id,
                    "model": operation.model,
                    "installationState": "downloading",
                    "residencyState": "unloaded",
                    "logicalSizeBytes": operation.logical_size_bytes,
                    "reclaimableSizeBytes": 0,
                    "estimatedResidentBytes": operation.estimated_resident_bytes,
                    "activeSessions": 0,
                    "runningJobs": 0,
                    "queuedJobs": 0,
                }
                for operation in self._operations.values()
            )
        return {"models": records}

    def remove_preview(self, installation_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._record_by_installation(installation_id)
            counts = self._blob_reference_counts()
            reclaimable, shared = self._reclaimable(record, counts)
            runtime = self._runtime_state(record["model"])
            blockers = [
                blocker
                for field, blocker in (
                    ("activeSessions", "active-sessions"),
                    ("runningJobs", "running-jobs"),
                    ("queuedJobs", "queued-jobs"),
                )
                if int(runtime.get(field, 0)) > 0
            ]
            if record.get("installationState") == "removing":
                blockers.append("removal-pending")
            return {
                "installationId": installation_id,
                "canRemoveNow": not blockers,
                "activeSessions": int(runtime.get("activeSessions", 0)),
                "runningJobs": int(runtime.get("runningJobs", 0)),
                "queuedJobs": int(runtime.get("queuedJobs", 0)),
                "reclaimableBytes": reclaimable,
                "sharedBytesRetained": shared,
                "blockers": blockers,
            }

    def mark_removing(self, operation_id: str, installation_id: str) -> dict[str, Any]:
        self._validate_operation_id(operation_id)
        with self._lock:
            record = self._record_by_installation(installation_id)
            record["installationState"] = "removing"
            record["removeOperationId"] = operation_id
            self._save_registry()
            _reclaimable, shared = self._reclaimable(record, self._blob_reference_counts())
        self._emit("model.inventory.changed", {"reason": "reconciled"})
        return {
            "operationId": operation_id,
            "installationId": installation_id,
            "state": "deferred",
            "reclaimedBytes": 0,
            "sharedBytesRetained": shared,
        }

    def pending_removal(self, model: dict[str, Any]) -> tuple[str, str] | None:
        key = self._model_key(model)
        with self._lock:
            record = next(
                (
                    item
                    for item in self._records
                    if self._model_key(item["model"]) == key
                    and item.get("installationState") == "removing"
                ),
                None,
            )
            if record is None:
                return None
            return str(record["removeOperationId"]), str(record["installationId"])

    def complete_remove(self, installation_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._record_by_installation(installation_id)
            counts = self._blob_reference_counts()
            reclaimable, shared = self._reclaimable(record, counts)
            remaining = [item for item in self._records if item is not record]
            operation_id = str(record.get("removeOperationId", f"remove_{uuid.uuid4().hex}"))
            if record.get("installationState") != "removing":
                record["installationState"] = "removing"
                record["removeOperationId"] = operation_id
                self._save_registry()
            self._remove_cached_files(record, remaining, counts)
            self._records.remove(record)
            self._save_registry()
        self._emit("model.inventory.changed", {"reason": "removed"})
        return {
            "operationId": operation_id,
            "installationId": installation_id,
            "state": "removed",
            "reclaimedBytes": reclaimable,
            "sharedBytesRetained": shared,
        }

    def record_by_installation(self, installation_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._record_by_installation(installation_id))

    def _record_by_installation(self, installation_id: str) -> dict[str, Any]:
        record = next(
            (item for item in self._records if item["installationId"] == installation_id), None
        )
        if record is None:
            raise ModelStoreError("MODEL_NOT_INSTALLED", "The model installation was not found.")
        return record

    def _runtime_state(self, model: dict[str, Any]) -> dict[str, Any]:
        if self._runtime_state_provider:
            return self._runtime_state_provider(model)
        return {
            "residencyState": "unloaded",
            "activeSessions": 0,
            "runningJobs": 0,
            "queuedJobs": 0,
        }

    def _remove_cached_files(
        self,
        record: dict[str, Any],
        remaining: list[dict[str, Any]],
        counts: dict[str, int],
    ) -> None:
        remaining_paths = {
            str(item.get("cache_path"))
            for other in remaining
            for item in other.get("files", [])
            if item.get("cache_path")
        }
        for item in record.get("files", []):
            relative = item.get("cache_path") or self._default_cache_path(record, item)
            if not isinstance(relative, str) or relative in remaining_paths:
                continue
            if ".." in Path(relative).parts:
                raise ModelStoreError("MODEL_SECURITY_BLOCKED", "Model cache path is invalid.")
            path = self.cache_root / relative
            if not path.absolute().is_relative_to(self.cache_root):
                raise ModelStoreError("MODEL_SECURITY_BLOCKED", "Model cache path is invalid.")
            if not path.exists() and not path.is_symlink():
                continue
            target = path.resolve()
            was_symlink = path.is_symlink()
            path.unlink()
            if (
                was_symlink
                and counts.get(str(item.get("blob_id")), 0) <= 1
                and target.is_relative_to(self.cache_root)
                and target.exists()
                and target.is_file()
            ):
                target.unlink()
            parent = path.parent
            while parent != self.cache_root:
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent

    @staticmethod
    def _default_cache_path(record: dict[str, Any], item: dict[str, Any]) -> str:
        repository = record["model"]["repository"].replace("/", "--")
        return str(
            Path(f"models--{repository}") / "snapshots" / record["model"]["revision"] / item["name"]
        )

    def _validate_files(self, siblings: Any, namespace: str) -> list[ModelFile]:
        if not isinstance(siblings, list) or len(siblings) > 10_000:
            raise ModelStoreError("MODEL_METADATA_INVALID", "Model file metadata is invalid.")
        result: list[ModelFile] = []
        for sibling in siblings:
            name = getattr(sibling, "rfilename", None)
            size = getattr(sibling, "size", None)
            if not isinstance(name, str) or not self._safe_filename(name):
                raise ModelStoreError("MODEL_METADATA_INVALID", "Model file name is invalid.")
            if size is None:
                size = 0
            if not isinstance(size, int) or isinstance(size, bool) or size < 0 or size > 2**63 - 1:
                raise ModelStoreError("MODEL_METADATA_INVALID", "Model file size is invalid.")
            lfs = getattr(sibling, "lfs", None)
            blob_id = getattr(lfs, "sha256", None) if lfs is not None else None
            if not isinstance(blob_id, str) or not re.fullmatch(r"[0-9a-f]{40,64}", blob_id):
                blob_id = f"file:{namespace}:{name}:{size}"
            result.append(ModelFile(name, size, blob_id))
        return result

    @staticmethod
    def _validate_metadata_value(value: Any) -> None:
        from oryn_engine.model_metadata import validate_metadata

        try:
            validate_metadata(value)
        except (ValueError, OverflowError) as exc:
            raise ModelStoreError("MODEL_METADATA_INVALID", "Model metadata is invalid.") from exc

    @staticmethod
    def _safe_filename(name: str) -> bool:
        if len(name) > 1024 or "\\" in name or "\x00" in name:
            return False
        path = PurePosixPath(name)
        return not path.is_absolute() and bool(path.parts) and ".." not in path.parts

    @staticmethod
    def _installation_files(
        files: list[ModelFile], compatibility: CompatibilityManifest | None = None
    ) -> list[ModelFile]:
        """Cache only declarative assets and safe Transformers weights."""
        allowed_suffixes = {
            ".json",
            ".jinja",
            ".model",
            ".safetensors",
            ".spm",
            ".tiktoken",
            ".txt",
        }
        if compatibility is not None and compatibility.pack_id != "oryn.transformers-pytorch":
            selected = [
                item
                for item in files
                if any(
                    fnmatch.fnmatchcase(item.name, pattern)
                    for pattern in compatibility.required_files
                )
            ]
            if compatibility.level == "converted-artifact":
                if compatibility.pack_id == "oryn.kokoro-onnx":
                    voice_files = {
                        f"voices/{v['id']}.bin"
                        for v in compatibility.trusted_metadata.get("voices", [])
                    }
                    return [
                        item
                        for item in selected
                        if PurePosixPath(item.name).suffix in allowed_suffixes | {".onnx"}
                        or item.name in voice_files
                    ]
                if compatibility.pack_id == "oryn.llama-cpp":
                    return [item for item in selected if PurePosixPath(item.name).suffix == ".gguf"]
                return [
                    item for item in selected if PurePosixPath(item.name).suffix in allowed_suffixes
                ]
            if compatibility.level == "audited-adapter":
                return [item for item in selected if PurePosixPath(item.name).suffix != ".py"]
            selected = [
                item for item in selected if PurePosixPath(item.name).suffix in allowed_suffixes
            ]
            if (
                compatibility.pack_id == "oryn.diffusers-pytorch"
                and compatibility.runner_id == "diffusers.pipeline"
            ):
                nested_safe_weights = any(
                    "/" in item.name and item.name.endswith(".safetensors") for item in selected
                )
                if nested_safe_weights:
                    # Prefer a complete component tree over a duplicate root checkpoint.
                    selected = [
                        item
                        for item in selected
                        if "/" in item.name or not item.name.endswith(".safetensors")
                    ]
            return selected
        selected = set(assess_layout([item.name for item in files], {}, None).selected_files)
        return [item for item in files if item.name in selected]

    @staticmethod
    def _required_patterns_present(
        files: list[ModelFile], compatibility: CompatibilityManifest
    ) -> bool:
        names = tuple(item.name for item in files)
        return all(
            any(fnmatch.fnmatchcase(name, pattern) for name in names)
            for pattern in compatibility.required_files
        )

    @staticmethod
    def _validate_model_ref(model: dict[str, Any]) -> tuple[str, str]:
        if not isinstance(model, dict) or model.get("provider") != "huggingface":
            raise ModelStoreError("MODEL_REFERENCE_INVALID", "Only Hugging Face is supported.")
        repository = model.get("repository")
        revision = model.get("revision")
        if not isinstance(repository, str) or not REPOSITORY_PATTERN.fullmatch(repository):
            raise ModelStoreError("MODEL_REFERENCE_INVALID", "Model repository is invalid.")
        if (
            not isinstance(revision, str)
            or not REVISION_PATTERN.fullmatch(revision)
            or ".." in PurePosixPath(revision).parts
        ):
            raise ModelStoreError("MODEL_REFERENCE_INVALID", "Model revision is invalid.")
        files = model.get("files")
        if files is not None and (
            not EXACT_REVISION_PATTERN.fullmatch(revision)
            or not isinstance(files, list)
            or not 1 <= len(files) <= 256
            or any(
                not isinstance(f, str)
                or not re.fullmatch(r"[A-Za-z0-9_. /-]{1,512}", f)
                or f.startswith("/")
                or any(p in {"", ".", ".."} for p in f.split("/"))
                for f in files
            )
            or len(set(files)) != len(files)
        ):
            raise ModelStoreError("MODEL_REFERENCE_INVALID", "Pinned file identity is invalid.")
        return repository, revision

    @staticmethod
    def _validate_operation_id(operation_id: str) -> None:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,127}", operation_id):
            raise ModelStoreError("MODEL_REFERENCE_INVALID", "Operation ID is invalid.")

    def _hardware_info(self):
        return HardwareManager().detect()

    def _executable_backends(self) -> list[str]:
        preferred = self._hardware_info().preferred_backend
        return [preferred, "cpu"] if preferred != "cpu" else ["cpu"]

    def _selected_index_reasons(self, repository, revision, files):
        names = {item.name for item in files}
        reasons = []
        for name in sorted(names):
            if not WEIGHT_INDEX.fullmatch(name):
                continue
            parent = str(PurePosixPath(name).parent)
            relative = (
                {item.removeprefix(parent + "/") for item in names if item.startswith(parent + "/")}
                if parent != "."
                else {item for item in names if "/" not in item}
            )
            index_name = PurePosixPath(name).name
            shards = {
                item
                for item in relative
                if (match := SHARD.fullmatch(item))
                and weight_index_name(match[1], relative) == index_name
            }
            try:
                index = self._hub.read_json(repository, revision, name)
                validate_weight_index(index, relative, expected_shards=shards or None)
            except (ValueError, TypeError):
                reasons.append("MODEL_WEIGHT_INDEX_INVALID")
            except Exception:
                reasons.append("MODEL_WEIGHT_INDEX_UNRESOLVED")
        return reasons

    def _available_pack_backends(self, declared: tuple[str, ...]) -> list[str]:
        hardware = self._hardware_info()
        available = {"cpu"}
        available.update(item["deviceClass"] for item in hardware.backends)
        if "mps" in available:
            available.add("metal")
        if hardware.platform == "darwin" and hardware.architecture in {"arm64", "aarch64"}:
            available.update({"coreml", "mlx"})
        native_preferred = (
            "metal" if hardware.preferred_backend == "mps" else hardware.preferred_backend
        )
        preferred = [hardware.preferred_backend, native_preferred, "mlx", "coreml", "cpu"]
        return list(
            dict.fromkeys(name for name in preferred if name in declared and name in available)
        )

    @staticmethod
    def _technical_capabilities(
        task: str | None,
        architectures: list[str],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        workload = transformers_workload(architectures, task)
        from oryn_engine.shared_tasks import TASK_CONTRACTS

        if task in TASK_CONTRACTS and workload == task:
            from oryn_protocol.task_contracts import TASK_CONTRACTS as contracts

            definition = contracts[TASK_CONTRACTS[task]]
            if task == "any-to-any" and config.get("model_type") == "gemma4":
                from oryn_engine.gemma4_tasks import input_modes

                modes = input_modes(config)
                return {
                    "architectures": architectures,
                    "task": task,
                    "inputModalities": ["text", *modes],
                    "outputModalities": ["text"],
                    "capabilities": {"structuredOutput": True, "audioOutput": False},
                }
            return {
                "architectures": architectures,
                "task": task,
                "inputModalities": definition["inputModalities"],
                "outputModalities": definition["outputModalities"],
                "capabilities": {"structuredOutput": True},
            }
        result: dict[str, Any] = {"architectures": architectures}
        if workload in {"causal-chat", "multimodal-chat"}:
            # Advertise the bounded operation Oryn's runner exposes, rather than
            # a broader repository pipeline tag such as image-text-to-text.
            result["task"] = "text-generation"
        elif workload == "forced-alignment":
            # The Hub labels the checkpoint with its broad AutoModel class;
            # Oryn exposes the narrower bounded operation the adapter implements.
            result["task"] = "forced-alignment"
        elif task:
            result["task"] = task
        parameters = config.get("num_parameters")
        if isinstance(parameters, int) and parameters >= 0:
            result["parameterCount"] = parameters
        context = config.get("max_position_embeddings") or config.get("n_positions")
        if isinstance(context, int) and context > 0:
            result["contextLength"] = context
        if workload == "asr":
            result.update(
                {
                    "inputModalities": ["audio"],
                    "outputModalities": ["text"],
                    "capabilities": {"speechRecognition": True, "streaming": "unknown"},
                }
            )
        elif workload == "forced-alignment":
            result.update(
                {
                    "inputModalities": ["audio", "text"],
                    "outputModalities": ["structured"],
                    "capabilities": {"structuredOutput": True},
                }
            )
        elif workload == "musicgen":
            result.update(
                {
                    "inputModalities": ["text"],
                    "outputModalities": ["audio"],
                    "capabilities": {"audioOutput": True, "streaming": False},
                }
            )
        elif workload in {"embedding", "encoder-embedding"}:
            result.update(
                {
                    "inputModalities": ["text"],
                    "outputModalities": ["embedding"],
                    "capabilities": {"structuredOutput": True},
                }
            )
        elif workload in {"reranking", "cross-encoder-reranking"}:
            result.update(
                {
                    "inputModalities": ["text"],
                    "outputModalities": ["structured"],
                    "capabilities": {"structuredOutput": True},
                }
            )
        elif workload in {
            "text-classification",
            "token-classification",
            "question-answering",
            "fill-mask",
            "seq2seq",
        }:
            result.update(
                {
                    "inputModalities": ["text"],
                    "outputModalities": ["structured" if workload != "seq2seq" else "text"],
                    "capabilities": {"structuredOutput": workload != "seq2seq"},
                }
            )
        elif workload in {
            "image-classification",
            "object-detection",
            "depth-estimation",
            "zero-shot-image-classification",
        }:
            result.update(
                {
                    "inputModalities": ["image", "text"]
                    if workload == "zero-shot-image-classification"
                    else ["image"],
                    "outputModalities": ["structured"],
                    "capabilities": {"structuredOutput": True},
                }
            )
        elif workload == "audio-classification":
            result.update(
                {
                    "inputModalities": ["audio"],
                    "outputModalities": ["structured"],
                    "capabilities": {"structuredOutput": True},
                }
            )
        else:
            tokenizer_config = config.get("tokenizer_config")
            has_chat_template = bool(config.get("chat_template")) or (
                isinstance(tokenizer_config, dict) and bool(tokenizer_config.get("chat_template"))
            )
            result.update(
                {
                    "inputModalities": ["text", "image"]
                    if workload == "multimodal-chat"
                    else ["text"],
                    "outputModalities": ["text"],
                    "capabilities": {
                        "chat": True if has_chat_template else "unknown",
                        "multimodalConversation": workload == "multimodal-chat",
                        "streaming": True,
                        "structuredOutput": "unknown",
                        "toolCalling": "unknown",
                    },
                }
            )
        return result

    def _load_registry(self) -> list[dict[str, Any]]:
        if not self.registry_path.exists():
            return []
        try:
            value = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelStoreError(
                "MODEL_METADATA_INVALID", "Installation registry is invalid."
            ) from exc
        if (
            not isinstance(value, dict)
            or value.get("version") != 1
            or not isinstance(value.get("models"), list)
        ):
            raise ModelStoreError("MODEL_METADATA_INVALID", "Installation registry is invalid.")
        records = value["models"]
        for record in records:
            # Residency is process-local. A previous engine crash must never make
            # persisted installation metadata claim that weights remain loaded.
            record["residencyState"] = "unloaded"
        return records

    def _migrate_runtime_plans(self) -> bool:
        changed = False
        for record in self._records:
            if not isinstance(record.get("runtimePlan"), dict):
                runtime_pack = record.get("runtimePack")
                model = record.get("model")
                if not isinstance(runtime_pack, dict) or not isinstance(model, dict):
                    continue
                pack_id = runtime_pack.get("packId")
                adapter = (
                    self._runtime_foundations.adapter_for_legacy_pack(pack_id)
                    if isinstance(pack_id, str)
                    else None
                )
                if adapter is None:
                    continue
                foundation = self._runtime_foundations.active_foundation(adapter.foundation_id)
                match = self._installed_compatibility(record)
                if match is None:
                    continue
                backends = record.get("executableBackends")
                backend = (
                    backends[0]
                    if isinstance(backends, list) and backends and isinstance(backends[0], str)
                    else "cpu"
                )
                quantization = self._installed_diffusers_quantization(record)
                if quantization is None:
                    quantization = match.trusted_metadata.get("quantization")
                if not isinstance(quantization, str):
                    quantization = None
                identity = RuntimeIdentity(
                    foundation.foundation_id,
                    foundation.version,
                    adapter.adapter_id,
                    adapter.version,
                    match.manifest_id,
                    backend,
                    "quantized" if quantization else ("fp32" if backend == "cpu" else "fp16"),
                    quantization,
                )
                record["runtimePlan"] = {
                    "planId": f"plan-{identity.candidate_id.removeprefix('candidate-')}",
                    "policy": RuntimePolicy.BALANCED.value,
                    "reasonCodes": ["MIGRATED_TRUSTED_RUNTIME"],
                    "selected": {
                        "candidateId": identity.candidate_id,
                        "foundation": {
                            "id": foundation.foundation_id,
                            "version": foundation.version,
                        },
                        "adapter": {"id": adapter.adapter_id, "version": adapter.version},
                        "variantId": match.manifest_id,
                        "backend": backend,
                        "precision": identity.precision,
                        **({"quantization": quantization} if quantization else {}),
                        "estimatedMemoryBytes": int(record.get("estimatedResidentBytes", 0)),
                    },
                    "fallbacks": [],
                    "rejected": [],
                }
                changed = True
            changed = self._refresh_installed_runtime_metadata(record) or changed
            changed = self._refresh_installed_generation_profile(record) or changed
        return changed

    def _installed_compatibility(self, record: dict[str, Any]) -> CompatibilityManifest | None:
        """Resolve the active signed compatibility for an existing installation.

        Older registries predate canonical runtime plans. Exact manifests can be
        recovered from their pinned model identity. Generic manifests additionally
        require the installed file subset to satisfy the signed file contract; this
        prevents a broad task tag from silently rebinding an incompatible install.
        """
        runtime_pack = record.get("runtimePack")
        model = record.get("model")
        if not isinstance(runtime_pack, dict) or not isinstance(model, dict):
            return None
        pack_id = runtime_pack.get("packId")
        repository = model.get("repository")
        revision = model.get("revision")
        if not all(isinstance(item, str) for item in (pack_id, repository, revision)):
            return None

        technical = record.get("technicalCapabilities")
        technical = technical if isinstance(technical, dict) else {}
        architectures = technical.get("architectures")
        architecture_names = (
            [item for item in architectures if isinstance(item, str)]
            if isinstance(architectures, list)
            else []
        )
        task = technical.get("task")
        task_name = task if isinstance(task, str) else None
        files = [
            ModelFile(
                item["name"],
                item.get("size", 0) if isinstance(item.get("size", 0), int) else 0,
                str(item.get("blob_id", "")),
            )
            for item in record.get("files", [])
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        ]
        matches = [
            item
            for item in self._runtime_packs.match(
                repository=repository,
                revision=revision,
                architectures=architecture_names,
                task=task_name,
            )
            if item.pack_id == pack_id
            and (item.repository is not None or self._required_patterns_present(files, item))
        ]
        if not matches:
            return None

        plan = record.get("runtimePlan")
        selected = plan.get("selected") if isinstance(plan, dict) else None
        variant_id = selected.get("variantId") if isinstance(selected, dict) else None
        if isinstance(variant_id, str):
            selected_match = next(
                (item for item in matches if item.manifest_id == variant_id), None
            )
            if selected_match is not None:
                return selected_match

        manifest_digest = runtime_pack.get("manifestDigest")
        if isinstance(manifest_digest, str):
            digest_match = next((item for item in matches if item.digest == manifest_digest), None)
            if digest_match is not None:
                return digest_match
        return matches[0]

    def _refresh_installed_runtime_metadata(self, record: dict[str, Any]) -> bool:
        if record.get("runtimePlan", {}).get("selected", {}).get("variantId") in GENERATED_PACKS:
            return False
        runtime_pack = record.get("runtimePack")
        if not isinstance(runtime_pack, dict) or runtime_pack.get("packId") != (
            "oryn.diffusers-pytorch"
        ):
            return False
        compatibility = self._installed_compatibility(record)
        if compatibility is None:
            return False
        quantization = self._installed_diffusers_quantization(record)

        changed = False
        if quantization is not None:
            plans = [record.get("runtimePlan"), record.get("automaticRuntimePlan")]
            for plan in plans:
                if not isinstance(plan, dict):
                    continue
                plan_changed = False
                candidates = [plan.get("selected"), *plan.get("fallbacks", [])]
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    adapter = candidate.get("adapter")
                    if (
                        not isinstance(adapter, dict)
                        or adapter.get("id") != "oryn.adapter.diffusers"
                    ):
                        continue
                    identity = RuntimeIdentity(
                        str(candidate.get("foundation", {}).get("id")),
                        str(candidate.get("foundation", {}).get("version")),
                        str(adapter.get("id")),
                        str(adapter.get("version")),
                        str(candidate.get("variantId")),
                        str(candidate.get("backend")),
                        "quantized",
                        quantization,
                    )
                    if (
                        candidate.get("precision") != "quantized"
                        or candidate.get("quantization") != quantization
                        or candidate.get("candidateId") != identity.candidate_id
                    ):
                        candidate["precision"] = "quantized"
                        candidate["quantization"] = quantization
                        candidate["candidateId"] = identity.candidate_id
                        plan_changed = True
                if plan_changed:
                    selected = plan.get("selected")
                    fallbacks = plan.get("fallbacks", [])
                    if isinstance(selected, dict):
                        candidate_ids = [selected.get("candidateId")]
                        candidate_ids.extend(
                            item.get("candidateId") for item in fallbacks if isinstance(item, dict)
                        )
                        model = record.get("model", {})
                        logical_model = (
                            f"huggingface:{model.get('repository')}@{model.get('revision')}"
                        )
                        override = record.get("runtimeOverrideCandidateId")
                        if override is not None:
                            override = selected.get("candidateId")
                            record["runtimeOverrideCandidateId"] = override
                        plan_key = {
                            "candidates": candidate_ids,
                            "logicalModel": logical_model,
                            "policy": plan.get("policy", RuntimePolicy.BALANCED.value),
                            "override": override,
                        }
                        digest = hashlib.sha256(
                            json.dumps(plan_key, separators=(",", ":"), sort_keys=True).encode()
                        ).hexdigest()
                        plan["planId"] = f"plan-{digest[:24]}"
                    changed = True

        canonical_runtime_pack = {
            "packId": compatibility.pack_id,
            "packVersion": compatibility.pack_version,
            "runnerId": compatibility.runner_id,
            "taskContract": compatibility.task_contract,
            "compatibilityLevel": compatibility.level,
            "manifestDigest": compatibility.digest,
            **(
                {"trustedMetadata": self._public_trusted_metadata(compatibility.trusted_metadata)}
                if compatibility.trusted_metadata
                else {}
            ),
        }
        if runtime_pack != canonical_runtime_pack:
            record["runtimePack"] = canonical_runtime_pack
            changed = True
        return changed

    def _installed_diffusers_quantization(self, record: dict[str, Any]) -> str | None:
        for item in record.get("files", []):
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            cache_path = item.get("cache_path")
            if (
                not isinstance(name, str)
                or not name.endswith("/config.json")
                or not isinstance(cache_path, str)
            ):
                continue
            path = (self.cache_root / cache_path).resolve()
            if not path.is_relative_to(self.cache_root) or not path.is_file():
                continue
            try:
                if path.stat().st_size > MAX_LOCAL_COMPONENT_CONFIG_BYTES:
                    continue
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(value, dict):
                continue
            config = value.get("quantization_config")
            if not isinstance(config, dict):
                continue
            method = config.get("quant_method")
            quant_type = config.get("bnb_4bit_quant_type")
            if (
                isinstance(method, str)
                and method.lower() == "bitsandbytes"
                and config.get("load_in_4bit") is True
                and isinstance(quant_type, str)
                and quant_type.lower() == "nf4"
            ):
                return BITSANDBYTES_NF4
        return None

    def _refresh_installed_generation_profile(self, record: dict[str, Any]) -> bool:
        from oryn_engine.providers import provider_variant

        selected = record.get("runtimePlan", {}).get("selected", {})
        if provider_variant(selected.get("variantId")):
            bound = record.get("providerPlans", {}).get(selected.get("candidateId"), {})
            profile = bound.get("generationProfile")
            if profile is not None and record.get("generationProfile") != profile:
                record["generationProfile"] = profile
                return True
            return False
        metadata = self._installed_declarative_metadata(record)
        compatibility = self._installed_compatibility(record)
        profile = (
            generation_profile(
                task_contract=compatibility.task_contract,
                trusted_metadata=dict(compatibility.trusted_metadata),
                declarative_metadata=metadata,
            )
            if compatibility is not None
            else None
        )
        if (
            record.get("runtimePlan", {}).get("selected", {}).get("variantId")
            == "oryn.generated.diffusers"
            and record.get("capabilityRegistryDigest") == self._capability_registry.digest
            and metadata.get("_class_name") in {"QwenImage21Pipeline", "Flux2KleinPipeline"}
        ):
            from oryn_engine.qwen_image_21 import generation_defaults

            try:
                if metadata.get("_class_name") == "Flux2KleinPipeline":
                    from oryn_engine.diffusion_tasks import klein_generation_defaults

                    profile = klein_generation_defaults(metadata)
                else:
                    profile = generation_defaults(metadata)
            except ValueError:
                profile = None
        current = record.get("generationProfile")
        from oryn_engine.model_composition import record_binding

        composition = record_binding(record)
        if (
            composition is not None
            and record.get("capabilityRegistryDigest") == self._capability_registry.digest
        ):
            from oryn_engine.model_composition import generation_defaults as composition_defaults

            profile = composition_defaults(composition, metadata)
        if profile is None:
            if current is not None:
                record.pop("generationProfile", None)
                return True
            return False
        if current != profile:
            record["generationProfile"] = profile
            return True
        return False

    def _installed_declarative_metadata(self, record: dict[str, Any]) -> dict[str, Any]:
        runtime_pack = record.get("runtimePack")
        if not isinstance(runtime_pack, dict) or runtime_pack.get("packId") != (
            "oryn.diffusers-pytorch"
        ):
            return {}
        for item in record.get("files", []):
            if not isinstance(item, dict) or item.get("name") not in {
                "model_index.json",
                "modular_model_index.json",
            }:
                continue
            cache_path = item.get("cache_path")
            if not isinstance(cache_path, str):
                continue
            path = (self.cache_root / cache_path).resolve()
            if not path.is_relative_to(self.cache_root) or not path.is_file():
                continue
            try:
                if path.stat().st_size > MAX_LOCAL_COMPONENT_CONFIG_BYTES:
                    continue
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                metadata = self._bounded_declarative_metadata(value)
                if value.get("_class_name") == "QwenImage21Pipeline":
                    from oryn_engine.qwen_image_21 import sample_sigmas

                    try:
                        sigmas = sample_sigmas(value)
                    except ValueError:
                        return {}
                    if sigmas is not None:
                        metadata["sample_sigmas"] = sigmas
                return metadata
        return {}

    @staticmethod
    def _bounded_declarative_metadata(value: dict[str, Any]) -> dict[str, Any]:
        """Keep only audited, non-executable model facts used by runtime adapters."""
        result: dict[str, Any] = {}
        for name, item in value.items():
            if len(result) >= 64:
                break
            if (
                not isinstance(name, str)
                or not re.fullmatch(r"_?[A-Za-z][A-Za-z0-9_]{0,127}", name)
                or not isinstance(item, (str, bool, int, float))
                or isinstance(item, str)
                and not 1 <= len(item) <= 256
            ):
                continue
            result[name] = item
        return result

    @staticmethod
    def _public_trusted_metadata(value: dict[str, Any]) -> dict[str, Any]:
        """Do not expose internal adapter matching tables across the public protocol."""
        return {name: item for name, item in value.items() if name != "generationProfiles"}

    def _save_registry(self) -> None:
        temporary = self.registry_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"version": 1, "models": self._records}, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, self.registry_path)

    def _blob_reference_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self._records:
            for item in record.get("files", []):
                counts[item["blob_id"]] = counts.get(item["blob_id"], 0) + 1
        return counts

    @staticmethod
    def _reclaimable(record: dict[str, Any], counts: dict[str, int]) -> tuple[int, int]:
        reclaimable = 0
        shared = 0
        for item in record.get("files", []):
            if counts.get(item["blob_id"], 0) <= 1:
                reclaimable += item["size"]
            else:
                shared += item["size"]
        return reclaimable, shared

    def _public_record(self, record: dict[str, Any], counts: dict[str, int]) -> dict[str, Any]:
        reclaimable, _ = self._reclaimable(record, counts)
        runtime = self._runtime_state(record["model"])
        return (
            {
                key: record[key]
                for key in (
                    "admissionScope",
                    "installationId",
                    "model",
                    "installationState",
                    "logicalSizeBytes",
                    "estimatedResidentBytes",
                    "installedAt",
                    "runner",
                    "executableBackends",
                    "technicalCapabilities",
                    "runtimePack",
                    "runtimePlan",
                )
                if key in record
            }
            | {
                "reclaimableSizeBytes": reclaimable,
            }
            | runtime
            | ({"admissionScope": "huggingface-page"} if record.get("sourceRecognition") else {})
        )

    @staticmethod
    def _model_key(model: dict[str, str]) -> str:
        return f"{model['provider']}:{model['repository']}@{model['revision']}" + (
            ":" + json.dumps(sorted(model["files"]), separators=(",", ":"))
            if model.get("files")
            else ""
        )

    @staticmethod
    def _install_result(operation_id: str, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "operationId": operation_id,
            "installationId": record["installationId"],
            "model": record["model"],
            "logicalSizeBytes": record["logicalSizeBytes"],
        }

    @staticmethod
    def _check_cancelled(operation: InstallOperation) -> None:
        if operation.cancel.is_set():
            raise InstallCancelled()

    def _progress(
        self,
        operation: InstallOperation,
        stage: str,
        downloaded: int,
        total: int,
    ) -> None:
        self._emit(
            "model.install.progress",
            {
                "operationId": operation.operation_id,
                "model": operation.model,
                "stage": stage,
                "downloadedBytes": downloaded,
                "totalBytes": total,
            },
        )

    def _state(self, operation: InstallOperation, state: str) -> None:
        self._emit(
            "model.install.state",
            {"operationId": operation.operation_id, "model": operation.model, "state": state},
        )

    def _emit(self, event: str, data: dict[str, Any]) -> None:
        if self._event_sink:
            self._event_sink(event, data)
