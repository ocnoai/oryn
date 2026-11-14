from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from oryn_engine.runtime_packs.manifests import PackManifest

MAX_WORKER_MESSAGE_BYTES = 1024 * 1024


class PackProcessError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PackProcessSupervisor:
    """Runs trusted pack entrypoints in independently supervised Python processes.

    This is process/crash/dependency isolation. OS filesystem/network sandbox enforcement is a
    separate release gate and the declared policy is denied by default until that layer applies it.
    """

    def invoke(
        self,
        pack: PackManifest,
        command: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float = 10,
    ) -> dict[str, Any]:
        if command not in {"probe", "conformance", "crash"}:
            raise PackProcessError("PACK_COMMAND_INVALID", "Runtime-pack command is invalid.")
        request = {
            "abi": pack.abi,
            "packId": pack.pack_id,
            "version": pack.version,
            "dependencyClosure": pack.dependency_closure,
            "command": command,
            "payload": payload or {},
        }
        encoded = json.dumps(request, separators=(",", ":")).encode() + b"\n"
        if len(encoded) > MAX_WORKER_MESSAGE_BYTES:
            raise PackProcessError("PACK_MESSAGE_TOO_LARGE", "Runtime-pack request is too large.")
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "ORYN_PACK_NETWORK_POLICY": pack.network_policy,
        }
        with tempfile.TemporaryDirectory(prefix="oryn-pack-") as working_directory:
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", "-B", "-m", pack.entrypoint],
                    input=encoded,
                    capture_output=True,
                    cwd=working_directory,
                    env=environment,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise PackProcessError("PACK_TIMEOUT", "Runtime-pack process timed out.") from exc
        if completed.returncode != 0:
            raise PackProcessError("PACK_CRASHED", "Runtime-pack process exited unexpectedly.")
        if len(completed.stdout) > MAX_WORKER_MESSAGE_BYTES:
            raise PackProcessError("PACK_PROTOCOL_INVALID", "Runtime-pack response is too large.")
        try:
            response = json.loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PackProcessError(
                "PACK_PROTOCOL_INVALID", "Runtime-pack response is invalid."
            ) from exc
        if not isinstance(response, dict) or set(response) != {"ok", "result"}:
            raise PackProcessError("PACK_PROTOCOL_INVALID", "Runtime-pack response is invalid.")
        if response.get("ok") is not True or not isinstance(response.get("result"), dict):
            raise PackProcessError("PACK_EXECUTION_FAILED", "Runtime-pack rejected the request.")
        return response["result"]

    @staticmethod
    def validate_roots(pack: PackManifest, roots: dict[str, Path]) -> dict[str, str]:
        if set(roots) != set(pack.filesystem_roots):
            raise PackProcessError("PACK_POLICY_INVALID", "Runtime-pack roots do not match policy.")
        result: dict[str, str] = {}
        for name, path in roots.items():
            if not path.is_absolute() or path.is_symlink():
                raise PackProcessError("PACK_PATH_INVALID", "Runtime-pack root is unsafe.")
            result[name] = str(path.resolve())
        return result
