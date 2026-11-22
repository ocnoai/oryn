from __future__ import annotations

import copy

import pytest

from oryn_engine.runtime_packs.builtin_foundations import (
    ADAPTER_ENVELOPES,
    BITSANDBYTES_DEVELOPMENT_PUBLISHER,
    FOUNDATION_ENVELOPES,
    FOUNDATION_PUBLISHER,
    builtin_foundation_registry,
)
from oryn_engine.runtime_packs.foundations import FoundationRegistry
from oryn_engine.runtime_packs.manifests import ManifestError, SignedManifestVerifier


def test_builtin_foundations_and_adapters_are_independently_verified() -> None:
    registry = builtin_foundation_registry()
    assert {item.foundation_id for item in registry.foundations} == {
        "oryn.runtime.pytorch",
        "oryn.runtime.pytorch-speech",
        "oryn.runtime.mlx",
        "oryn.runtime.onnx",
        "oryn.runtime.llama-cpp",
    }
    assert {item.adapter_id for item in registry.adapters} == {
        "oryn.adapter.transformers",
        "oryn.adapter.diffusers",
        "oryn.adapter.ace-step",
        "oryn.adapter.mflux-image",
        "oryn.adapter.mlx-chat",
        "oryn.adapter.mlx-audio",
        "oryn.adapter.kokoro",
        "oryn.adapter.gguf-chat",
        "oryn.adapter.onnx-encoder",
        "oryn.adapter.qwen-tts",
    }
    assert registry.adapter_for_legacy_pack("oryn.ace-step-mlx").foundation_id == "oryn.runtime.mlx"
    assert registry.adapter_for_legacy_pack("oryn.mflux-mlx").foundation_id == "oryn.runtime.mlx"
    assert registry.adapter_for_legacy_pack("oryn.mlx-audio").foundation_id == "oryn.runtime.mlx"
    pytorch = registry.active_foundation("oryn.runtime.pytorch")
    diffusers = registry.adapter_for_legacy_pack("oryn.diffusers-pytorch")
    assert "bitsandbytes-nf4" in pytorch.quantizations
    assert diffusers is not None and "bitsandbytes-nf4" in diffusers.quantizations


@pytest.mark.parametrize("count", [33, 64, 65])
def test_shared_adapter_and_runner_contract_capacity_remains_bounded(count):
    import json
    from pathlib import Path

    from oryn_engine.runtime_packs.foundations import parse_adapter_manifest
    from oryn_engine.runtime_packs.manifests import parse_pack_manifest

    root = Path(__file__).parents[1] / "src/oryn_engine/data"
    adapter = json.loads((root / "transformers-adapter.json").read_text())["payload"]
    pack = json.loads((root / "transformers-pack.json").read_text())["payload"]
    contracts = [f"bounded-task-{i}-v1" for i in range(count)]
    adapter["taskContracts"] = contracts
    pack["runners"][0]["taskContracts"] = contracts
    if count > 64:
        for parser, value in ((parse_adapter_manifest, adapter), (parse_pack_manifest, pack)):
            with pytest.raises(ManifestError, match="task contracts"):
                parser(value, "0" * 64)
    else:
        assert len(parse_adapter_manifest(adapter, "0" * 64).task_contracts) == count
        assert len(parse_pack_manifest(pack, "0" * 64).runners[0].task_contracts) == count


def test_manifest_tampering_fails_before_registration() -> None:
    tampered = copy.deepcopy(FOUNDATION_ENVELOPES[0])
    tampered["payload"]["backends"] = ["cpu"]
    registry = FoundationRegistry(
        SignedManifestVerifier(
            {
                FOUNDATION_PUBLISHER.key_id: FOUNDATION_PUBLISHER,
                BITSANDBYTES_DEVELOPMENT_PUBLISHER.key_id: (BITSANDBYTES_DEVELOPMENT_PUBLISHER),
            }
        )
    )
    with pytest.raises(ManifestError) as failure:
        registry.register_foundation(tampered)
    assert failure.value.code == "PACK_DIGEST_INVALID"


def test_adapter_cannot_claim_foundation_capability_it_does_not_have() -> None:
    registry = FoundationRegistry(
        SignedManifestVerifier({FOUNDATION_PUBLISHER.key_id: FOUNDATION_PUBLISHER})
    )
    registry.register_foundation(FOUNDATION_ENVELOPES[2])
    incompatible = copy.deepcopy(ADAPTER_ENVELOPES[4])
    incompatible["payload"]["backends"] = ["cuda"]
    # Digest failure is intentionally first: compatibility cannot be changed without publisher
    # authority. The parser's compatibility gate is separately exercised by built-in registration.
    with pytest.raises(ManifestError) as failure:
        registry.register_adapter(incompatible)
    assert failure.value.code == "PACK_DIGEST_INVALID"


def test_mlc_is_not_a_foundation() -> None:
    assert all(
        "mlc" not in item.foundation_id for item in builtin_foundation_registry().foundations
    )


def test_foundation_revocation_invalidates_adapters_and_safe_rollback_is_explicit() -> None:
    registry = builtin_foundation_registry()
    registry.revoke_foundation("oryn.runtime.mlx", "1.3.0")

    with pytest.raises(ManifestError) as inactive:
        registry.active_foundation("oryn.runtime.mlx")
    assert inactive.value.code == "FOUNDATION_NOT_INSTALLED"
    assert registry.adapter_for_legacy_pack("oryn.mflux-mlx") is None
    assert registry.adapter_for_legacy_pack("oryn.ace-step-mlx") is None

    with pytest.raises(ManifestError) as revoked:
        registry.activate_foundation("oryn.runtime.mlx", "1.3.0")
    assert revoked.value.code == "FOUNDATION_REVOKED"

    # A rollback reconstructs authority from the earlier signed catalog; revocation is never
    # locally cleared or bypassed on the affected registry instance.
    rollback = builtin_foundation_registry()
    assert rollback.adapter_for_legacy_pack("oryn.ace-step-mlx") is not None
    assert rollback.adapter_for_legacy_pack("oryn.mflux-mlx") is not None


def test_revoked_adapter_cannot_be_reactivated() -> None:
    registry = builtin_foundation_registry()
    registry.revoke_adapter("oryn.adapter.mflux-image", "1.2.0")
    with pytest.raises(ManifestError) as revoked:
        registry.activate_adapter("oryn.adapter.mflux-image", "1.2.0")
    assert revoked.value.code == "ADAPTER_REVOKED"
