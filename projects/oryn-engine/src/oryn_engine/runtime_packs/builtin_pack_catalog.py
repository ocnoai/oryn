from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oryn_engine.generated_trust import DEVELOPMENT_KEY_ID, DEVELOPMENT_MODULUS
from oryn_engine.runtime_packs.catalog import (
    PackCatalog,
    PublisherTrustPolicy,
    parse_pack_catalog,
)
from oryn_engine.runtime_packs.manifests import SignedManifestVerifier, TrustedPublisherKey

CATALOG_PUBLISHER = TrustedPublisherKey(
    "oryn-catalog-bitsandbytes-development-2026",
    26159033626337466440768045963452832694349856238271137527580765239839590698804780982918247587631510013703675763109331388107775574038097449159374159489803426985641506439184168711263361142433078349195761791057273850583513931202907615098642051368027611238718004769724150692190010665814322760339450060396262300706110859536321886005430819646038065971268706234612789680643153967153118619595710473889633890012499189340488754571140044416537169228091588990655546000521118279391964413963520955813470236267872656302419011825630883315524168043139962116780415588463060283081461690074334864947082393564640868869150707498547530006153,
)


def _entry(
    pack_id: str,
    manifest_digest: str,
    dependency_closure: str,
    license_summary: str,
    notice: str,
    version: str = "0.1.0",
) -> dict[str, Any]:
    return {
        "packId": pack_id,
        "version": version,
        "publisher": "oryn.first-party",
        "manifestDigest": manifest_digest,
        "abi": "1",
        "dependencyClosure": dependency_closure,
        "platforms": ["macos-arm64"],
        "distribution": "bundled",
        "artifact": None,
        "minimumSafeVersion": version,
        "revokedVersions": [],
        "license": {
            "summary": license_summary,
            "notice": notice,
            "access": "public",
            "acceptanceRequired": False,
        },
    }


BUILTIN_PACK_CATALOG: dict[str, Any] = {
    "payload": {
        "kind": "runtime-pack-catalog",
        "schemaVersion": 1,
        "catalogVersion": 2,
        "publisher": "oryn.first-party",
        "generatedAt": 1_776_643_200_000,
        "packs": [
            _entry(
                "oryn.ace-step-mlx",
                "6d360179ac5c9f04b65613ab06c65ca6f6c334cb92dedb64c6a7570c3907f7fd",
                "ace-step-v1.5.0-dce621408bee-py312",
                "MIT core with composite runtime dependencies",
                "Review the application and packaged upstream license notices before distribution.",
            ),
            _entry(
                "oryn.diffusers-pytorch",
                "90ae4e70406b39b15d53613429ae89faa2357cf302506c8dcc2aed9165c1393a",
                "diffusers0.39-bitsandbytes0.50.1-imageio2.37-base-torch2.13-py312",
                "Apache-2.0 core with composite runtime dependencies",
                "Review the application and packaged upstream license notices before distribution.",
                "0.2.0",
            ),
            _entry(
                "oryn.kokoro-onnx",
                "febcffdc1878aba13b2a0e457c9820a64386046e85845b10ddfd2b99d25221f6",
                "kokoro-onnx-ort1.22-espeak3.3.2-py312",
                "Composite open-source speech runtime",
                "Review ONNX Runtime, phonemizer, and eSpeak NG notices before distribution.",
            ),
            _entry(
                "oryn.llama-cpp",
                "af93dac50c7698b390b31047c7cd75a3c484940755b54585593c1c7f06bbfd0e",
                "llama.cpp-b10218-de699957-abi1",
                "MIT native runtime",
                "llama.cpp LICENSE is packaged beside the native executable closure.",
            ),
            _entry(
                "oryn.mflux-mlx",
                "f61e6555fd14d95a825aed6e3a6b473c2e7c37072b0b3610c97705e289ea2c16",
                "mflux0.18.1-adapter-py312",
                "MIT MFlux adapter with composite runtime dependencies",
                "MFlux is packaged as Oryn adapter code; review upstream dependency notices "
                "before distribution.",
            ),
            _entry(
                "oryn.transformers-pytorch",
                "3a1dd26b3441ec34e081f059a5c61bfe31ae644fddd24c28fddb1bbd95759d39",
                "oryn-engine-py312-transformers-5",
                "Apache-2.0 core with composite runtime dependencies",
                "Review the application and packaged upstream license notices before distribution.",
            ),
        ],
    },
    "integrity": {
        "algorithm": "rsa-sha256",
        "keyId": "oryn-catalog-bitsandbytes-development-2026",
        "digest": "3e7dd4db018b467b8d780c96adfaec203eb175d3c87eba8a4edaf7bbdca4b991",
        "signature": "H+Op+/ydD8PbdjSXn2LsUqHxV2JIuzjt17dgFyMXlcFd2mHW/mDtriGsQjTKHH2ZmenlK4l/cSbHUUc5WqbPLflaa27FIn03CBB11yvj0tJmrg9qjoOthieDzuFckUvf+h54zsXgQ7+8N/9f+3wKoCGJPLcQdDbnPB5fluDCuoNI9r48oBwS+6bqYqekg+pOdi8ODVcuDjFKrg+/skUNKSG325IEhDw88y9vXIwDRV283MQHVPvjwMg6NQyBLKLRV7TCnwysOEFpNQrHZXjZj9EIObBXaEyzELjbrRsI5Is7A00U0fxknygOH7F86Hc/56GUnRcXl45TDBMVImZ/zQ==",  # noqa: E501
    },
}


def builtin_pack_catalog() -> PackCatalog:
    key = TrustedPublisherKey(DEVELOPMENT_KEY_ID, DEVELOPMENT_MODULUS)
    verifier = SignedManifestVerifier({key.key_id: key})
    trust = PublisherTrustPolicy(frozenset({key.key_id}), frozenset({"oryn.first-party"}))
    envelope = json.loads((Path(__file__).parents[1] / "data/pack-catalog.json").read_text())
    return parse_pack_catalog(envelope, verifier, trust)
