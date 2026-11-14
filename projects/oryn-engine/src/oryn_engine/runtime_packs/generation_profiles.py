from __future__ import annotations

from typing import Any


def generation_profile(
    *,
    task_contract: str,
    trusted_metadata: dict[str, Any] | None = None,
    declarative_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Normalize runtime-owned media controls without using model repository identity."""
    if task_contract not in {
        "text-to-image-v1",
        "image-to-image-v1",
        "image-editing-v1",
        "text-to-video-v1",
        "image-to-video-v1",
    }:
        return None

    trusted_defaults = (trusted_metadata or {}).get("inferenceDefaults")
    if isinstance(trusted_defaults, dict) and trusted_defaults:
        conformance = (trusted_metadata or {}).get("conformance")
        tested = isinstance(conformance, dict) and conformance.get("status") == "passed"
        return _profile(
            profile_id="runtime.manifest-recommended",
            provenance="runtime-tested" if tested else "runtime-declared",
            recommended=True,
            width=_bounded_integer(trusted_defaults.get("width"), 512, 64, 2048, 8),
            height=_bounded_integer(trusted_defaults.get("height"), 512, 64, 2048, 8),
            steps=_bounded_integer(trusted_defaults.get("steps"), 20, 1, 100, 1),
            guidance=_bounded_number(trusted_defaults.get("guidanceScale"), 7.5, 0, 30, 0.1),
            guidance_supported=True,
        )

    conformance = (trusted_metadata or {}).get("conformance")
    tested = isinstance(conformance, dict) and conformance.get("status") == "passed"
    for descriptor in (trusted_metadata or {}).get("generationProfiles", []):
        if not isinstance(descriptor, dict) or not _matches(
            descriptor.get("match"), declarative_metadata or {}
        ):
            continue
        profile = descriptor.get("profile")
        if isinstance(profile, dict):
            return {
                **profile,
                "provenance": "runtime-tested" if tested else "runtime-declared",
            }

    return _profile(
        profile_id="runtime.generic.fallback",
        provenance="runtime-fallback",
        recommended=False,
        width=512,
        height=512,
        steps=20,
        guidance=7.5,
        guidance_supported=True,
    )


def _profile(
    *,
    profile_id: str,
    provenance: str,
    recommended: bool,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    guidance_supported: bool,
) -> dict[str, Any]:
    return {
        "profileId": profile_id,
        "provenance": provenance,
        "recommended": recommended,
        "parameters": {
            "width": {
                "default": width,
                "minimum": 64,
                "maximum": 2048,
                "step": 8,
                "supported": True,
                "editable": True,
            },
            "height": {
                "default": height,
                "minimum": 64,
                "maximum": 2048,
                "step": 8,
                "supported": True,
                "editable": True,
            },
            "steps": {
                "default": steps,
                "minimum": 1,
                "maximum": 100,
                "step": 1,
                "supported": True,
                "editable": True,
            },
            "guidanceScale": {
                "default": float(guidance),
                "minimum": 0.0,
                "maximum": 30.0,
                "step": 0.1,
                "supported": guidance_supported,
                "editable": guidance_supported,
            },
        },
    }


def _matches(selector: object, metadata: dict[str, Any]) -> bool:
    return (
        isinstance(selector, dict)
        and bool(selector)
        and all(metadata.get(key) == expected for key, expected in selector.items())
    )


def _bounded_integer(value: object, fallback: int, minimum: int, maximum: int, step: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        return fallback
    if (value - minimum) % step:
        return fallback
    return value


def _bounded_number(
    value: object, fallback: float, minimum: float, maximum: float, _step: float
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not minimum <= float(value) <= maximum
    ):
        return fallback
    return float(value)
