from __future__ import annotations

import contextlib
import json
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oryn_engine.jobs import JobExecutionError
from oryn_engine.runtime_packs.builtin_foundations import builtin_foundation_registry
from oryn_engine.runtime_packs.foundations import FoundationManifest
from oryn_engine.runtime_packs.lifecycle import validate_content_integrity
from oryn_engine.runtime_packs.manifests import PackManifest

MAX_LINE_BYTES = 1024 * 1024
SAFE_WORKER_ERROR_CODES = {
    "ARTIFACT_IO_ERROR",
    "MODEL_FORMAT_UNSUPPORTED",
    "MODEL_GENERATION_FAILED",
    "MODEL_OUTPUT_FILTERED",
    "PACK_BROKER_REJECTED",
    "MODEL_INPUT_INVALID",
    "MODEL_MEMORY_INSUFFICIENT",
    "MODEL_RUNTIME_INCOMPATIBLE",
    "PACK_EXECUTION_FAILED",
    "RUNTIME_DEPENDENCY_UNAVAILABLE",
}


@dataclass(frozen=True)
class PackRuntime:
    python: Path
    site_packages: Path | None
    inherit_base_runtime: bool = False
    foundation_site_packages: Path | None = None


class PackRuntimeLocator:
    """Resolves only Browser-owned runtime-pack dependency closures."""

    def resolve(self, pack: PackManifest) -> PackRuntime | None:
        foundation_registry = builtin_foundation_registry()
        adapter = foundation_registry.adapter_for_legacy_pack(pack.pack_id)
        foundation = (
            foundation_registry.active_foundation(adapter.foundation_id)
            if adapter is not None
            else None
        )
        token = pack.pack_id.upper().replace(".", "_").replace("-", "_")
        overrides_allowed = os.environ.get("ORYN_PACK_OVERRIDES_ALLOWED") == "1"
        python_override = os.environ.get(f"ORYN_PACK_{token}_PYTHON") if overrides_allowed else None
        site_override = (
            os.environ.get(f"ORYN_PACK_{token}_SITE_PACKAGES") if overrides_allowed else None
        )
        foundation_override: str | None = None
        if foundation is not None:
            foundation_token = foundation.foundation_id.upper().replace(".", "_").replace("-", "_")
            if overrides_allowed:
                foundation_override = os.environ.get(
                    f"ORYN_FOUNDATION_{foundation_token}_SITE_PACKAGES"
                )
        if python_override or site_override:
            if not python_override or not site_override:
                return None
            return self._validated(
                pack,
                Path(python_override),
                Path(site_override),
                foundation,
                Path(foundation_override) if foundation_override else None,
            )

        executable = Path(sys.executable).resolve()
        candidates: tuple[Path, ...] = ()
        lifecycle_root_value = os.environ.get("ORYN_RUNTIME_PACK_ROOT")
        if lifecycle_root_value:
            lifecycle_root = Path(lifecycle_root_value)
            state_path = lifecycle_root / "lifecycle-state.json"
            try:
                if (
                    lifecycle_root.is_absolute()
                    and lifecycle_root.is_dir()
                    and not lifecycle_root.is_symlink()
                    and state_path.is_file()
                    and not state_path.is_symlink()
                    and state_path.stat().st_size <= 64 * 1024
                ):
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                    active = state.get("active") if isinstance(state, dict) else None
                    version = active.get(pack.pack_id) if isinstance(active, dict) else None
                    version_root = lifecycle_root / "packs" / pack.pack_id / pack.version
                    if version == pack.version and validate_content_integrity(version_root):
                        candidates += (version_root / "site-packages",)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                pass
        candidates += (
            executable.parents[1] / "runtime-packs" / pack.pack_id / "site-packages",
            executable.parents[2] / "runtime-packs" / pack.pack_id / "site-packages",
        )
        for site_packages in candidates:
            foundation_site_packages = None
            if foundation is not None:
                if site_packages.parents[1].name == "runtime-packs":
                    # Bundled layouts place foundations beside runtime-packs.
                    foundation_site_packages = (
                        site_packages.parents[2]
                        / "runtime-foundations"
                        / foundation.foundation_id
                        / "site-packages"
                    )
                else:
                    # Lifecycle installs retain immutable foundation versions independently.
                    foundation_site_packages = (
                        site_packages.parents[3]
                        / "foundations"
                        / foundation.foundation_id
                        / foundation.version
                        / "site-packages"
                    )
            runtime = self._validated(
                pack, executable, site_packages, foundation, foundation_site_packages
            )
            if runtime is not None:
                return runtime
        return None

    @staticmethod
    def _validated(
        pack: PackManifest,
        python: Path,
        site_packages: Path,
        foundation: FoundationManifest | None = None,
        foundation_site_packages: Path | None = None,
    ) -> PackRuntime | None:
        if not python.is_absolute() or not site_packages.is_absolute():
            return None
        if python.is_symlink() or site_packages.is_symlink():
            return None
        if not python.is_file() or not os.access(python, os.X_OK) or not site_packages.is_dir():
            return None
        marker = site_packages / "oryn-runtime-pack.json"
        if not marker.is_file() or marker.is_symlink():
            return None
        try:
            marker_value = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if (
            not isinstance(marker_value, dict)
            or marker_value.get("schemaVersion") != 1
            or marker_value.get("packId") != pack.pack_id
            or marker_value.get("version") != pack.version
            or marker_value.get("dependencyClosure") != pack.dependency_closure
        ):
            return None
        inherit_base_runtime = marker_value.get("inheritBaseRuntime", False)
        if not isinstance(inherit_base_runtime, bool):
            return None
        resolved_foundation: Path | None = None
        foundation_marker_value = marker_value.get("foundation")
        if foundation_marker_value is not None:
            if (
                foundation is None
                or not isinstance(foundation_marker_value, dict)
                or foundation_marker_value
                != {
                    "id": foundation.foundation_id,
                    "version": foundation.version,
                    "dependencyClosure": foundation.dependency_closure,
                }
                or foundation_site_packages is None
                or not foundation_site_packages.is_absolute()
                or foundation_site_packages.is_symlink()
                or not foundation_site_packages.is_dir()
            ):
                return None
            foundation_marker = foundation_site_packages / "oryn-runtime-foundation.json"
            if not foundation_marker.is_file() or foundation_marker.is_symlink():
                return None
            try:
                actual_foundation_marker = json.loads(foundation_marker.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                return None
            if actual_foundation_marker != {
                "schemaVersion": 1,
                "foundationId": foundation.foundation_id,
                "version": foundation.version,
                "dependencyClosure": foundation.dependency_closure,
            }:
                return None
            resolved_foundation = foundation_site_packages.resolve()
        return PackRuntime(
            python.resolve(),
            site_packages.resolve(),
            inherit_base_runtime=inherit_base_runtime,
            foundation_site_packages=resolved_foundation,
        )


class StreamingPackProcess:
    """Runs one bounded pack command and contains cancellation by terminating the worker."""

    def invoke(
        self,
        pack: PackManifest,
        runtime: PackRuntime,
        command: str,
        payload: dict[str, Any],
        *,
        cancellation: Any,
        event_sink: Callable[[str, dict[str, Any]], None],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        from oryn_engine.providers import PROVIDER_COMMANDS

        if command not in {"probe", "prepare", "generate"} and not (
            pack.provider is not None and command in PROVIDER_COMMANDS
        ):
            raise JobExecutionError("PACK_COMMAND_INVALID", "Runtime-pack command is invalid.")
        if os.environ.get("ORYN_WORKER_BROKER_SOCKET"):
            from oryn_engine.runtime_packs.broker import invoke_broker

            return invoke_broker(
                pack,
                command,
                payload,
                cancellation=cancellation,
                event_sink=event_sink,
                timeout_seconds=timeout_seconds,
            )
        request = {
            "abi": pack.abi,
            "packId": pack.pack_id,
            "version": pack.version,
            "command": command,
            "payload": payload,
        }
        encoded = json.dumps(request, separators=(",", ":")).encode("utf-8") + b"\n"
        if len(encoded) > MAX_LINE_BYTES:
            raise JobExecutionError("PACK_MESSAGE_TOO_LARGE", "Runtime-pack request is too large.")

        package_roots = tuple(
            path
            for path in (runtime.foundation_site_packages, runtime.site_packages)
            if path is not None
        )
        if pack.provider is not None:
            from oryn_engine.providers import verify_catalog

            verify_catalog(pack, runtime)
        bootstrap = (
            "import runpy,site,sys;"
            + f"sys.path[:0]={[str(path) for path in package_roots]!r};"
            + "".join(f"site.addsitedir({str(path)!r});" for path in package_roots)
            + (
                "from oryn_engine.runtime_packs.execution import StreamingPackProcess;"
                f"StreamingPackProcess._resource_limits({timeout_seconds!r})();"
                if os.name == "posix"
                else ""
            )
            + f"runpy.run_module({pack.entrypoint!r},run_name='__main__')"
        )
        environment = {
            "HOME": tempfile.gettempdir(),
            "PATH": "/usr/bin:/bin",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "ORYN_PACK_NETWORK_POLICY": "deny",
            "ORYN_PACK_ROOT": str(runtime.site_packages.parent),
            "ORYN_FOUNDATION_ROOT": (
                str(runtime.foundation_site_packages.parent)
                if runtime.foundation_site_packages is not None
                else ""
            ),
        }
        with tempfile.TemporaryDirectory(prefix="oryn-pack-process-") as working_directory:
            environment["HOME"] = working_directory
            environment["TMPDIR"] = working_directory
            interpreter_options = ["-I", "-B"]
            if not runtime.inherit_base_runtime:
                interpreter_options.append("-S")
            arguments = [str(runtime.python), *interpreter_options, "-c", bootstrap]
            arguments = self._sandboxed_arguments(runtime, payload, working_directory, arguments)
            started = time.monotonic()
            process = subprocess.Popen(
                arguments,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=working_directory,
                env=environment,
                start_new_session=os.name == "posix",
            )
            assert process.stdin is not None
            assert process.stdout is not None
            process.stdin.write(encoded)
            process.stdin.close()
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            result: dict[str, Any] | None = None
            try:
                while True:
                    if cancellation.is_set():
                        self._terminate(process)
                        raise JobExecutionError("JOB_CANCELLED", "Runtime-pack work was cancelled.")
                    if time.monotonic() - started > timeout_seconds:
                        self._terminate(process)
                        raise JobExecutionError("PACK_TIMEOUT", "Runtime-pack process timed out.")
                    for key, _mask in selector.select(timeout=0.1):
                        line = key.fileobj.readline(MAX_LINE_BYTES + 1)
                        if len(line) > MAX_LINE_BYTES:
                            self._terminate(process)
                            raise JobExecutionError(
                                "PACK_PROTOCOL_INVALID", "Runtime-pack response is too large."
                            )
                        if not line:
                            continue
                        message = self._message(line)
                        kind = message.get("type")
                        if kind == "event":
                            event = message.get("event")
                            data = message.get("data")
                            if event not in {
                                "job.progress",
                                "job.state",
                                "job.output.delta",
                                "pack.preview",
                            } or not isinstance(data, dict):
                                raise JobExecutionError(
                                    "PACK_PROTOCOL_INVALID", "Runtime-pack event is invalid."
                                )
                            event_sink(event, data)
                        elif (
                            kind == "result"
                            and message.get("ok") is True
                            and isinstance(message.get("result"), dict)
                        ):
                            result = message["result"]
                        elif kind == "result":
                            error_code = self._worker_error_code(message.get("error"))
                            raise JobExecutionError(error_code, "Runtime-pack execution failed.")
                        else:
                            raise JobExecutionError(
                                "PACK_PROTOCOL_INVALID", "Runtime-pack response is invalid."
                            )
                    return_code = process.poll()
                    if return_code is not None:
                        if return_code != 0:
                            raise JobExecutionError(
                                "PACK_CRASHED", "Runtime-pack process exited unexpectedly."
                            )
                        if result is None:
                            raise JobExecutionError(
                                "PACK_PROTOCOL_INVALID", "Runtime-pack omitted its result."
                            )
                        return result
            finally:
                selector.close()
                if process.poll() is None:
                    self._terminate(process)

    @staticmethod
    def _sandboxed_arguments(
        runtime: PackRuntime,
        payload: dict[str, Any],
        working_directory: str,
        arguments: list[str],
    ) -> list[str]:
        profile_value = os.environ.get("ORYN_PACK_SANDBOX_PROFILE")
        required = os.environ.get("ORYN_SANDBOX_REQUIRED") == "1"
        if payload.get("implementationId") == "oryn.lfm3.torch.v1":
            # Closed installed adapter: its containment cannot be disabled by
            # resetting an environment flag or choosing a permissive profile.
            profile_value = str(Path(__file__).parents[1] / "data/reviewed-inference.sb")
            required = True
        if sys.platform != "darwin":
            if required:
                raise JobExecutionError(
                    "PACK_SANDBOX_UNAVAILABLE", "Runtime-pack sandboxing is unavailable."
                )
            return arguments
        if not profile_value:
            if required:
                raise JobExecutionError(
                    "PACK_SANDBOX_UNAVAILABLE", "Runtime-pack sandboxing is unavailable."
                )
            return arguments
        profile = Path(profile_value)
        if not profile.is_absolute() or profile.is_symlink() or not profile.is_file():
            raise JobExecutionError(
                "PACK_SANDBOX_INVALID", "Runtime-pack sandbox configuration is invalid."
            )

        empty = Path("/private/var/empty")

        def root_for(key: str, *, parent: bool = False) -> Path:
            value = payload.get(key)
            if not isinstance(value, str) or not value:
                return empty
            path = Path(value)
            if not path.is_absolute():
                raise JobExecutionError("PACK_SANDBOX_INVALID", "Runtime-pack path is invalid.")
            resolved = path.resolve()
            if parent and resolved != path:
                # The broker already supplied the canonical artifact path. Do
                # not let a new alias grant a different directory at launch.
                raise JobExecutionError(
                    "PACK_SANDBOX_INVALID", "Runtime-pack artifact path changed."
                )
            return path.parent if parent else resolved

        parameters = {
            "PYTHON": Path(runtime.python).resolve(),
            "PYTHON_ROOT": Path(sys.base_prefix).resolve(),
            "ENGINE_ROOT": Path(os.environ.get("ORYN_ENGINE_ROOT", sys.prefix)).resolve(),
            "PACK_RUNTIME_ROOT": runtime.site_packages.parent.resolve()
            if runtime.site_packages is not None
            else empty,
            "FOUNDATION_ROOT": runtime.foundation_site_packages.parent.resolve()
            if runtime.foundation_site_packages is not None
            else empty,
            "MODEL_ROOT": root_for("modelRoot"),
            "ARTIFACT_INPUT_ROOT": root_for("inputPath", parent=True),
            "ARTIFACT_OUTPUT_ROOT": root_for("outputPath", parent=True),
            "SCRATCH_ROOT": Path(working_directory).resolve(),
            # Darwin's Metal frontend ignores HOME/TMPDIR when locating its module
            # cache. Scope the compiler helper to this OS-owned cache alone.
            # _CS_DARWIN_USER_CACHE_DIR from Darwin unistd.h (not in Python's names).
            "METAL_COMPILER_CACHE": (
                empty
                if payload.get("implementationId") == "oryn.lfm3.torch.v1"
                else Path(os.confstr(65538)).resolve() / "com.apple.metalfe"
            ),
        }
        sandbox_arguments = ["/usr/bin/sandbox-exec", "-f", str(profile.resolve())]
        for name, value in parameters.items():
            sandbox_arguments.extend(("-D", f"{name}={value}"))
        return [*sandbox_arguments, *arguments]

    @staticmethod
    def _resource_limits(timeout_seconds: float) -> Callable[[], None]:
        def apply_limits() -> None:
            import resource

            def bounded(kind, soft, hard):
                inherited_soft, inherited_hard = resource.getrlimit(kind)
                if inherited_hard != resource.RLIM_INFINITY:
                    hard = min(hard, inherited_hard)
                soft = min(soft, hard)
                if inherited_soft != resource.RLIM_INFINITY:
                    soft = min(soft, inherited_soft)
                resource.setrlimit(kind, (soft, hard))

            bounded(resource.RLIMIT_CORE, 0, 0)
            # The pinned MLX-LM loader sets (2048, 4096) at import. Keep a bounded
            # ceiling that permits that library initialization in the isolated worker.
            bounded(resource.RLIMIT_NOFILE, 256, 4096)
            file_limit = 3 * 1024 * 1024 * 1024
            bounded(resource.RLIMIT_FSIZE, file_limit, file_limit)
            cpu_limit = max(1, int(timeout_seconds) + 30)
            bounded(resource.RLIMIT_CPU, cpu_limit, cpu_limit)

        return apply_limits

    @staticmethod
    def _message(line: bytes) -> dict[str, Any]:
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise JobExecutionError(
                "PACK_PROTOCOL_INVALID", "Runtime-pack response is invalid."
            ) from exc
        if not isinstance(value, dict):
            raise JobExecutionError("PACK_PROTOCOL_INVALID", "Runtime-pack response is invalid.")
        return value

    @staticmethod
    def _worker_error_code(value: object) -> str:
        # Runtime packs may report only a closed, non-sensitive category. Raw exception text,
        # paths, stack traces, and arbitrary worker-selected codes never cross this boundary.
        if value == "PACK_EXECUTION_FAILED":
            return "PACK_EXECUTION_FAILED"
        if isinstance(value, dict) and set(value) == {"code"}:
            code = value.get("code")
            if isinstance(code, str) and code in SAFE_WORKER_ERROR_CODES:
                return code
        return "PACK_EXECUTION_FAILED"

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        if os.name == "posix":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=5)
