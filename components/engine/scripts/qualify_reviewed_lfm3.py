"""Offline packaged CPU smoke, multi-turn, cancellation and sandbox qualification.

Trusted orchestration only: never imports Engine or repository model Python.
Use the pinned data-only snapshot and an independently built Engine wheel.
"""

import argparse
import contextlib
import hashlib
import json
import os
import resource
import signal
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

import psutil

PINNED = {
    "model.safetensors": "135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330",
    "tokenizer.json": "7e226e34bface23ccf6e495d157bf77186c1c30a7659880e1a8994db528c3701",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args()
    work = Path(tempfile.mkdtemp(prefix="oryn-s63-qualified-")).resolve()
    package, model, scratch = (work / name for name in ("installed", "model", "scratch"))
    for path in (package, model, scratch):
        path.mkdir()
    with zipfile.ZipFile(args.wheel) as archive:
        if (
            len(archive.infolist()) > 4096
            or sum(i.file_size for i in archive.infolist()) > 128 * 1024**2
        ):
            raise ValueError("Wheel size limit")
        for item in archive.infolist():
            name = PurePosixPath(item.filename)
            if (
                name.is_absolute()
                or ".." in name.parts
                or item.file_size > 16 * 1024**2
                or (item.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError("Invalid wheel member")
            if not item.is_dir():
                target = package.joinpath(*name.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(item))
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        path = args.data_root / name
        if name in PINNED:
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != PINNED[name]:
                    raise ValueError("Pinned model-data digest mismatch")
        os.link(path, model / name)
    engine = package / "oryn_engine"
    runtime = args.runtime_root.resolve()
    python = (runtime / "bin/python3.12").resolve()
    pack = json.loads((engine / "data/transformers-pack.json").read_bytes())["payload"]
    config = json.loads((model / "config.json").read_bytes())
    config_digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    catalog_digest = hashlib.sha256(
        (engine / "data/reviewed-implementations.json").read_bytes()
    ).hexdigest()
    command = ["/usr/bin/sandbox-exec", "-f", str(engine / "data/reviewed-inference.sb")]
    for key, path in {
        "PYTHON": python,
        "PYTHON_ROOT": runtime,
        "ENGINE_ROOT": package,
        "PACK_RUNTIME_ROOT": runtime,
        "FOUNDATION_ROOT": runtime,
        "MODEL_ROOT": model,
        "SCRATCH_ROOT": scratch,
    }.items():
        command.extend(["-D", key + "=" + str(path)])
    bootstrap = (
        "import sys,runpy;sys.path.insert(0," + repr(str(package)) + ");"
        "runpy.run_module('oryn_engine.runtime_packs.transformers_worker',run_name='__main__')"
    )
    env = {
        "HOME": str(scratch),
        "TMPDIR": str(scratch),
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }

    def limits():
        for kind, cap in (
            (resource.RLIMIT_CPU, 60),
            (resource.RLIMIT_NPROC, 1),
            (resource.RLIMIT_NOFILE, 128),
            (resource.RLIMIT_FSIZE, 64 * 1024**2),
            (resource.RLIMIT_CORE, 0),
        ):
            resource.setrlimit(kind, (cap, cap))

    def run(label, messages=None, *, cancel=False, code=None):
        log = work / (label + ".jsonl")
        request = {
            "abi": pack["abi"],
            "packId": pack["packId"],
            "version": pack["version"],
            "command": "generate",
            "payload": {
                "implementationId": "oryn.lfm3.torch.v1",
                "profileId": "oryn.lfm3.source-a.v1",
                "configDigest": config_digest,
                "catalogDigest": catalog_digest,
                "modelRoot": str(model),
                "backend": "cpu",
                "input": {
                    "messages": messages,
                    "maxTokens": 128 if cancel else 24,
                    "temperature": 0,
                    "topP": 1,
                },
            },
        }
        started, peak, cancelled = time.monotonic(), 0, False
        with log.open("wb") as output:
            proc = subprocess.Popen(
                [*command, str(python), "-I", "-B", "-c", code or bootstrap],
                stdin=subprocess.PIPE,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=scratch,
                start_new_session=True,
                preexec_fn=limits,
            )
            proc.stdin.write(json.dumps(request).encode() + b"\n")
            proc.stdin.close()
            monitored = psutil.Process(proc.pid)
            while proc.poll() is None:
                with contextlib.suppress(psutil.NoSuchProcess):
                    peak = max(peak, monitored.memory_info().rss)
                if (
                    peak > 12 * 1024**3
                    or time.monotonic() - started > 60
                    or log.stat().st_size > 1024**2
                ):
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
                    raise RuntimeError("Qualification resource limit")
                if cancel and b'"event":"job.output.delta"' in log.read_bytes():
                    os.killpg(proc.pid, signal.SIGTERM)
                    cancelled = True
                    proc.wait(timeout=5)
                time.sleep(0.02)
        frames = [json.loads(line) for line in log.read_text().splitlines() if line.startswith("{")]
        result = next((f for f in frames if f.get("type") == "result"), {})
        if cancel:
            if not cancelled or proc.returncode != -signal.SIGTERM or result:
                raise RuntimeError("Cancellation qualification failed")
        elif proc.returncode or (code is None and result.get("ok") is not True):
            raise RuntimeError("Worker qualification failed: " + log.read_text()[-4000:])
        return {
            "case": label,
            "elapsedSeconds": round(time.monotonic() - started, 3),
            "peakRssMiB": round(peak / 1024**2, 1),
            "exitCode": proc.returncode,
            "cancelled": cancelled,
            "result": result.get("result"),
            "frames": len(frames),
        }

    results = [
        run(
            "greeting",
            [{"role": "user", "content": "Write one friendly sentence greeting the user."}],
        ),
        run(
            "multi-turn",
            [
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Nice to meet you, Alice."},
                {"role": "user", "content": "What is my name? Answer with just the name."},
            ],
        ),
        run("cancel", [{"role": "user", "content": "Count from one to one hundred."}], cancel=True),
    ]
    # A synthetic canary, never an actual user secret. This file is outside all
    # worker read scopes, although the trusted orchestrator owns the work tree.
    canary = work / "outside-worker-scope.txt"
    canary.write_text("synthetic sandbox canary\n")
    probe = (
        "import pathlib,socket\n"
        "def denied(action):\n"
        " try: action()\n"
        " except PermissionError: return\n"
        " raise RuntimeError('sandbox permitted forbidden access')\n"
        f"denied(lambda: pathlib.Path({str(canary)!r}).read_bytes())\n"
        f"denied(lambda: pathlib.Path({str(model / 'config.json')!r}).open('wb'))\n"
        "denied(lambda: socket.socket().connect(('127.0.0.1',9)))\n"
        f"pathlib.Path({str(scratch / 'allowed.txt')!r}).write_text('scratch only')\n"
    )
    results.append(run("sandbox-denials", code=probe))
    report = {
        "wheelSha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
        "work": str(work),
        "pinnedDataSha256": PINNED,
        "modelFiles": sorted(p.name for p in model.iterdir()),
        "sandbox": "fixed reviewed-inference.sb; offline CPU; empty allowlisted environment",
        "cases": results,
        "conformance": "unverified; smoke and numerical checks are separate",
    }
    (work / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
