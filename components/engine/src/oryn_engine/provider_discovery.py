"""Common discovery ABI, including reviewed bridges for existing native loaders.

Bridges run bounded Engine metadata code. Native providers use the isolated pack
broker. Both produce the same authenticated runtime candidate and task profile.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

MAX_OFFERS = 256


@dataclass(frozen=True)
class BackendProvider:
    provider_id: str
    inspect: Callable[[list], dict | None]


@dataclass
class DiscoveryResult:
    offers: list
    profiles: list[dict[str, Any]]
    diagnostics: list[dict[str, str]]


class ExecutionProviderRegistry:
    def inspect(self, providers: list[BackendProvider]) -> DiscoveryResult:
        result = DiscoveryResult([], [], [])
        for provider in providers:
            local = []
            try:
                profile = provider.inspect(local)
                if len(local) > 32 or len(result.offers) + len(local) > MAX_OFFERS:
                    raise ValueError("PROVIDER_OFFER_LIMIT")
                result.offers.extend(local)
                if profile is not None:
                    result.profiles.append(profile)
            except Exception:
                # Exception strings may include paths, credentials or native diagnostics.
                # A failed provider never suppresses another provider's valid offer.
                result.diagnostics.append(
                    {"providerId": provider.provider_id, "code": "PROVIDER_INSPECTION_FAILED"}
                )
        return result


def record_for_offer(record: dict, candidate: dict) -> dict:
    """Rebind only a verified, installed offer; fallback never changes the model."""
    from oryn_protocol.task_contracts import REGISTRY_DIGEST, validate_profile

    from oryn_engine.jobs import JobExecutionError
    from oryn_engine.runtime_routing import canonical_digest, runtime_build_digest

    offer = record.get("executionOffers", {}).get(candidate.get("candidateId"))
    if offer is None:
        if "executionOffers" in record:
            raise JobExecutionError(
                "RUNTIME_PLAN_INVALID", "The runtime candidate has no bound offer."
            )
        return record
    expected = canonical_digest({k: v for k, v in offer.items() if k != "digest"})
    if (
        offer.get("digest") != expected
        or offer.get("model") != record["model"]
        or offer.get("registryDigest") != REGISTRY_DIGEST
        or offer.get("runtimeBuildDigest") != runtime_build_digest()
    ):
        raise JobExecutionError("RUNTIME_PLAN_INVALID", "The task contract or runtime changed.")
    if any(candidate.get(k) != v for k, v in offer.get("runtime", {}).items()):
        raise JobExecutionError("RUNTIME_PLAN_INVALID", "The runtime offer identity changed.")
    validate_profile(offer["taskProfile"])
    installed = {f["name"]: (f["size"], f["blob_id"]) for f in record.get("files", [])}
    if any(installed.get(f["name"]) != (f["sizeBytes"], f["blobId"]) for f in offer["files"]):
        raise JobExecutionError(
            "RUNTIME_PLAN_INVALID", "The alternate runtime requires different model files."
        )
    if (
        record.get("taskProfile")
        and offer["taskProfile"]["contract"] != record["taskProfile"]["contract"]
    ):
        raise JobExecutionError(
            "RUNTIME_PLAN_INVALID", "Runtime fallback changed the task contract."
        )
    result = dict(record)
    for k in ("taskProfile", "runtimePack", "runner", "technicalCapabilities", "generationProfile"):
        if offer.get(k) is None:
            result.pop(k, None)
        else:
            result[k] = offer[k]
    result["_runtimeCandidate"] = dict(candidate)
    return result
