"""Browser-owned launcher for separately sandboxed Engine and signed pack workers.

macOS forbids Seatbelt reinitialization. This small parent passes protocol and credential
pipes directly to Engine and brokers only bounded, authenticated signed-worker requests.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

from oryn_engine.runtime_packs.broker import WorkerBroker


def main() -> None:
    if sys.platform != "darwin" or len(sys.argv) < 2 or sys.argv[1] != "-f":
        raise SystemExit("The sandbox launcher requires an explicit macOS Engine profile.")
    scratch = Path(os.environ["TMPDIR"]).resolve(strict=True)
    models = Path(os.environ["ORYN_MODEL_ROOT"]).resolve(strict=True)
    inputs = Path(os.environ["ORYN_ARTIFACT_INPUT_ROOT"]).resolve(strict=True)
    outputs = Path(os.environ["ORYN_ARTIFACT_OUTPUT_ROOT"]).resolve(strict=True)
    broker = WorkerBroker(
        scratch,
        {
            "models": (models,),
            "inputs": (inputs, scratch),
            "outputs": (outputs, scratch),
        },
    )
    environment = dict(os.environ)
    environment.update(
        ORYN_WORKER_BROKER_SOCKET=str(broker.path),
        ORYN_WORKER_BROKER_TOKEN=broker.token,
    )
    credential_fds = tuple(
        int(os.environ[key])
        for key in ("ORYN_HF_CREDENTIAL_INPUT_FD", "ORYN_HF_CREDENTIAL_ACK_FD")
        if key in os.environ
    )
    child = None
    try:
        child = subprocess.Popen(
            ["/usr/bin/sandbox-exec", *sys.argv[1:]],
            env=environment,
            pass_fds=credential_fds,
        )
        for descriptor in credential_fds:
            os.close(descriptor)

        def stop(_signum, _frame):
            broker.stopped.set()
            if child.poll() is None:
                child.terminate()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        status = child.wait()
    finally:
        broker.close()
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=5)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
