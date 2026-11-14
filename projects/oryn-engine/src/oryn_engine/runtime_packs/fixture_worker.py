from __future__ import annotations

import json
import os
import sys


def main() -> int:
    raw = sys.stdin.buffer.readline(1024 * 1024 + 1)
    if not raw or len(raw) > 1024 * 1024 or sys.stdin.buffer.read(1):
        return 2
    try:
        request = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return 2
    if not isinstance(request, dict) or request.get("abi") != "1":
        return 2
    if os.environ.get("ORYN_PACK_NETWORK_POLICY") != "deny":
        return 2
    if request.get("command") == "crash":
        return 23
    if request.get("command") not in {"probe", "conformance"}:
        return 2
    result = {
        "pid": os.getpid(),
        "packId": request.get("packId"),
        "version": request.get("version"),
        "dependencyClosure": request.get("dependencyClosure"),
        "networkPolicy": "deny",
        "payload": request.get("payload"),
    }
    sys.stdout.write(json.dumps({"ok": True, "result": result}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
