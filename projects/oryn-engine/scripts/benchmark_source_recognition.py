"""Offline synthetic capacity qualification; never registers execution profiles.

Run in the contained qualification harness. The million-entry measurement is
headroom only: the installed catalog deliberately caps unique contents at 100k.
"""

import gc
import hashlib
import json
import statistics
import sys
import time

from oryn_engine.reviewed_implementations import POLICY, ImplementationIndex, installed_index
from oryn_engine.source_inspector_worker import NORMALIZER, inspect_source


def milliseconds(function, repeats=5):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def retained_bytes(value, seen=None):
    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    total = sys.getsizeof(value)
    if isinstance(value, dict):
        total += sum(retained_bytes(k, seen) + retained_bytes(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple, set, frozenset)):
        total += sum(retained_bytes(v, seen) for v in value)
    elif hasattr(value, "__dict__"):
        total += retained_bytes(value.__dict__, seen)
    return total


def profile(i):
    return {
        "id": f"oryn.synthetic.{i}",
        "version": 1,
        "normalizer": NORMALIZER,
        "adapter": "oryn.lfm3.torch.v1",
        "state": "active",
        "sourcePolicy": POLICY,
        "representations": ["bf16-cpu"],
        "references": [],
        "bindings": {"AutoModelForCausalLM": ["r0", "Model"]},
        "modules": {
            f"r{j}": {
                "raw": hashlib.sha256(f"raw:{i}:{j}".encode()).hexdigest(),
                "structural": hashlib.sha256(f"structural:{i}:{j}".encode()).hexdigest(),
                "exports": ["Model"],
            }
            for j in range(5)
        },
    }


def main():
    profiles = [profile(i) for i in range(20_000)]
    shards = [
        json.dumps(profiles[i : i + 500], separators=(",", ":")).encode()
        for i in range(0, len(profiles), 500)
    ]
    assert len(shards) <= 64 and max(map(len, shards)) <= 1024**2
    start = time.perf_counter()
    decoded = [p for shard in shards for p in json.loads(shard)]
    parse_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    index = ImplementationIndex(decoded, identity="a" * 64)
    activation_ms = (time.perf_counter() - start) * 1000
    facts = {
        f"renamed{j}.py": {
            "rawSha256": m["raw"],
            "structuralSha256": m["structural"],
            "exports": m["exports"],
        }
        for j, m in enumerate(profiles[0]["modules"].values())
    }
    binding = {"AutoModelForCausalLM": "renamed0.Model"}
    assert index.match(facts, binding)[0]["id"] == "oryn.synthetic.0"
    lookup_ms = (
        milliseconds(lambda index=index: [index.match(facts, binding) for _ in range(1000)]) / 1000
    )
    report = {
        "python": sys.version.split()[0],
        "productionCapacity": {
            "uniqueContents": len(index.contents),
            "profiles": len(index.profiles),
            "shards": len(shards),
            "serializedMiB": sum(map(len, shards)) / 1024**2,
            "coldJsonParseMs": parse_ms,
            "coldIndexActivationMs": activation_ms,
            "shardHashMs": milliseconds(
                lambda shards=shards: [hashlib.sha256(s).digest() for s in shards]
            ),
            "retainedIndexMiB": retained_bytes(index) / 1024**2,
            "warmCompleteFiveModuleLookupMs": lookup_ms,
        },
    }
    del index, profiles, decoded, shards
    gc.collect()
    headroom = {hashlib.sha256(str(i).encode()).digest(): i for i in range(1_000_000)}
    key = next(iter(headroom))
    report["millionEntryHeadroomOnly"] = {
        "entries": len(headroom),
        "retainedMiB": retained_bytes(headroom) / 1024**2,
        "warmDigestLookupMs": milliseconds(
            lambda headroom=headroom: [headroom[key] for _ in range(100_000)]
        )
        / 100_000,
        "installedCatalogCapacity": False,
    }
    del headroom
    gc.collect()
    source = ("# ordinary comment\nx = 1\n" * 1000).encode()
    report["inProcessParser24KiBSyntheticMs"] = milliseconds(lambda: inspect_source(source))
    report["installedSmallSignedCatalogWarmValidationMs"] = milliseconds(installed_index)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
