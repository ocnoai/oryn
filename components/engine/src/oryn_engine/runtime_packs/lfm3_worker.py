"""Closed dispatch for the installed reviewed LFM3 graph, never a repo module."""

from pathlib import Path


def generate(payload, event):
    import resource

    # No child processes, compiler workers or model-written persistent files.
    for kind, cap in ((resource.RLIMIT_NPROC, 1), (resource.RLIMIT_FSIZE, 64 * 1024 * 1024)):
        inherited = resource.getrlimit(kind)
        cap = min([cap, *(limit for limit in inherited if limit != resource.RLIM_INFINITY)])
        resource.setrlimit(kind, (cap, cap))
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {
            "implementationId",
            "profileId",
            "modelRoot",
            "input",
            "backend",
            "configDigest",
            "catalogDigest",
        }
        or payload["implementationId"] != "oryn.lfm3.torch.v1"
        or payload["backend"] != "cpu"
    ):
        raise ValueError("Invalid reviewed implementation payload")
    root = Path(payload["modelRoot"])
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("Invalid LFM3 data scope")
    if any(root.glob("*.py")):
        raise ValueError("Source inspection data must not enter model scopes")
    from oryn_engine.native_lfm3 import generate as run
    from oryn_engine.reviewed_implementations import installed_index

    if payload["catalogDigest"] != installed_index().identity:
        raise ValueError("Reviewed implementation snapshot changed")

    return run(
        root,
        payload["input"],
        event=event,
        profile_id=payload["profileId"],
        config_digest=payload["configDigest"],
    )
