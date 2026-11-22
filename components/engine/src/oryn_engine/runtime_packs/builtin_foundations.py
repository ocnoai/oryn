from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oryn_engine.generated_trust import DEVELOPMENT_KEY_ID, DEVELOPMENT_MODULUS
from oryn_engine.runtime_packs.foundations import FoundationRegistry
from oryn_engine.runtime_packs.manifests import SignedManifestVerifier, TrustedPublisherKey

FOUNDATION_PUBLISHER = TrustedPublisherKey(
    "oryn-foundations-development-2026",
    26625732807394322264147027745129656576245310948535290372879700834231347057778180547998139486213733797348502450398543449298680814008613885763516150303818521033047200950337532200436053618193727450555905053867815457277593335664217254481280127910781984261448677240308863387159008065032102719289785041351879664568288664927355055576447367943953118598070408362631245875793288088594862012775728017962674889518606900862077364879040769461516969818118538274678762665976956283382377857171042214838478409615927866686723236565443289790563515373406180856508671356705014251083722528616907429070723466193062813918557567750936477116981,
)
BITSANDBYTES_DEVELOPMENT_PUBLISHER = TrustedPublisherKey(
    "oryn-bitsandbytes-development-2026",
    20495391919518967233188423799300250913516682572102753863441375270843792231578332098908684138516252477895072816359067119243433951185308694549179908354769971623703991837264944089240173606988108797729642245352593938079265307481649812722944740298890840022910903527193898522677057493365708674047872470555312850716577699517874747542328543006609002881820902674468142059120079035088810557243325144111433703840286755784414998505565323719822691988016969996186265160757149016471954267973177141125369229449462597561436045554622509728977876237840937715348458066080045089225299515286783110565877335091956654776939190112774808611459,
)

_POLICY = {
    "network": "deny",
    "filesystemRoots": ["model", "artifact-input", "artifact-output", "scratch"],
}
_FORBIDDEN = [".py", ".pyc", ".pkl", ".pickle", ".dylib", ".so", ".dll", ".exe", ".sh"]


def _foundation(
    foundation_id: str,
    dependency_closure: str,
    backends: list[str],
    formats: list[str],
    dtypes: list[str],
    quantizations: list[str],
    license_summary: str,
) -> dict[str, Any]:
    return {
        "kind": "runtime-foundation",
        "schemaVersion": 1,
        "foundationId": foundation_id,
        "version": "1.0.0",
        "publisher": "oryn.first-party",
        "abi": "1",
        "minimumSafeVersion": "1.0.0",
        "dependencyClosure": dependency_closure,
        "platforms": ["macos-arm64"],
        "backends": backends,
        "formats": formats,
        "dtypes": dtypes,
        "quantizations": quantizations,
        "workerAbi": "1",
        "policy": _POLICY,
        "license": {
            "summary": license_summary,
            "notice": "Review packaged upstream license notices before distribution.",
        },
    }


def _adapter(
    adapter_id: str,
    foundation_id: str,
    architectures: list[str],
    formats: list[str],
    tasks: list[str],
    backends: list[str],
    dtypes: list[str],
    quantizations: list[str],
    required_files: list[str],
    resource_estimator: str,
    legacy_pack_id: str,
) -> dict[str, Any]:
    return {
        "kind": "runtime-adapter",
        "schemaVersion": 1,
        "adapterId": adapter_id,
        "version": "1.0.0",
        "publisher": "oryn.first-party",
        "foundation": {"id": foundation_id, "abi": "1", "minimumVersion": "1.0.0"},
        "architectures": architectures,
        "formats": formats,
        "taskContracts": tasks,
        "backends": backends,
        "dtypes": dtypes,
        "quantizations": quantizations,
        "requiredFiles": required_files,
        "forbiddenSuffixes": _FORBIDDEN,
        "resourceEstimator": resource_estimator,
        "equivalenceClass": "exact-model",
        "legacyPackIds": [legacy_pack_id],
    }


def _envelope(
    payload: dict[str, Any],
    digest: str,
    signature: str,
    publisher: TrustedPublisherKey = FOUNDATION_PUBLISHER,
) -> dict[str, Any]:
    return {
        "payload": payload,
        "integrity": {
            "algorithm": "rsa-sha256",
            "keyId": publisher.key_id,
            "digest": digest,
            "signature": signature,
        },
    }


FOUNDATION_ENVELOPES = (
    _envelope(
        _foundation(
            "oryn.runtime.pytorch",
            "oryn-engine-py312-torch2.13",
            ["mps", "cuda", "cpu"],
            ["safetensors", "json", "tokenizer", "diffusers"],
            ["fp32", "fp16", "bf16"],
            ["bitsandbytes-nf4"],
            "BSD-3-Clause PyTorch with composite dependencies",
        ),
        "dafdad3d78e58aa27c9a68ffe064377a70839acc32382090115bd5bd8146592b",
        "kOoCzQGby70ayvhWd/Iw1daFHlVsXNfoPY1dEkuEUF/ex2eAFvfdciLYFSTRody/8lg0GGGOzeR+jHrYYtP+LMzHm+wgCUw7Yzu8SEnKIGxywdH8VPtom+UoJWicpOe5LN7tZSBRjiMPDYO+84toJfRNxKLOn0lZS2Z0Mx9OWy50zdi7ZNGVqQuw+qMn3Luh+YnoieP4La3Y4TqO3iq2Deh1ZkX/gVVtqLnLancJ+SvJK2JaG+Z5LcTaZncMY6JVm0TtNAcM4BPc3AdxuKNnVBS6S7sHk7hqXB7mkD52FvgsFLRrG1FcX6MULgH1bWDJnoEyBFu+FnZ443GRSwef6Q==",
        BITSANDBYTES_DEVELOPMENT_PUBLISHER,
    ),
    _envelope(
        _foundation(
            "oryn.runtime.mlx",
            "oryn-mlx0.29-py312",
            ["mlx"],
            ["safetensors", "json", "tokenizer", "mflux"],
            ["fp32", "fp16", "bf16"],
            ["int8", "int4", "q4"],
            "MIT MLX with composite dependencies",
        ),
        "d33e6d91eb1226a2d4525ec8b03821e0e66fd853cd83e9d2630e3950c68acba2",
        "jEesSkB/JO3G/hclKeVRi5zh4k+L3S4HXsM6ccnOEwtgpuSBYHzUW66KdHC7Mc9yBZhUEwZSSOzpKGCLlnx4nER8colBLejiLvGugQ1/xJxVJCq6wQ+b9u7Yny8MlS99q0G/Yd7IzQSkFs9U9avAHuvy+qVxbdo514hbQgdgRvO+Edpq17qYWNCDV33CCEz7wbgpM3+us843ShfH7s8HlTQWwxQveJHSXV8+tHhYljaFqOhnLQiaXxByHDtA32BjQM+8P7TtMpOBK/cPLrQq/0pERFKRC8gXrKLP8PqQAlrMaUCdI4DC5lsGYSNsL+9qpup6IIVf1OguToj1Me4/0w==",
    ),
    _envelope(
        _foundation(
            "oryn.runtime.onnx",
            "oryn-onnxruntime1.22-py312",
            ["coreml", "cpu"],
            ["onnx", "json", "tokenizer", "npy"],
            ["fp32", "fp16", "int8"],
            ["int8"],
            "MIT ONNX Runtime with composite dependencies",
        ),
        "5caa2a1a84bdcf6b64bfa7232269c873a960541c4482d481061c46d6903035eb",
        "MjGLAM/3ZapT71v1Frmwd7MipVFa3pmrIyGngtQaFG1ESJfSj1ABkthtYh+vjXvSLHVBK8WjhuhoxzXVRdCFegc0zKsyTwTV5OH03Oll855lgnJeK/49ms29V9JKAsDsgj+C6fcLxmZHPiEKyWVff0CHQSRRKujnIPMV0owZA9N9go2q1/4hPPHX/P9h/UgXfSbnMIf3p1Xd7eM2INVjLqbYkUjCDIl19dKH/8BxmDSLGIEK5WqWCqlhDYG8lSdinJQRy70i3SCjDkWDUxCGYftzd9R6agwo/9do9n8NnHJiHaCWWp3WdVae4Q3tCQHzgvRCzhtgL+yHNMzsObcWjQ==",
    ),
    _envelope(
        _foundation(
            "oryn.runtime.llama-cpp",
            "llama.cpp-b10218-de699957-abi1",
            ["metal", "cuda", "cpu"],
            ["gguf"],
            ["fp32", "fp16", "bf16"],
            ["Q4_K_M", "Q8_0", "Q5_K_M", "IQ4_XS"],
            "MIT llama.cpp native runtime",
        ),
        "e071c71665fba3816b38e2c9511c26f011e555da9252dd4dc6263e58ca7eb02b",
        "bPhJDJtxvD6r9x1w7/H9qfF2wbzbaxIu9cRoV84GNI6nsForCiOKYbNRvNUQq9NjFI9PV2STtR+H3ZIiEDclWnl55W4qxGSQKhWhXxDQESL+OcjQ8B3fTXUv0Bz+QQ5GNaSOCY2E16gX9ZGybn8bp9JKaeNo3MV3m7STqVWbDhLp5RqRbKu+1Jp0/xZPpXDA2VNY7dusbHOvUhwxNi1OsyWKDeoI+jZf9MKvhUmFztLvgy7jj+hqVEK13+e25Ejk99sDUYW0juECmY9mf4N0nXqCw7eGN9H9yAAMB5Vs3MM1F3Lp2WxgSp9A0DqmaLeFyLjo+s0djd5++OXfKDzBJQ==",
    ),
)

ADAPTER_ENVELOPES = (
    _envelope(
        _adapter(
            "oryn.adapter.transformers",
            "oryn.runtime.pytorch",
            [
                "AutoModelForCausalLM",
                "AutoModelForSpeechSeq2Seq",
                "WhisperForConditionalGeneration",
                "MusicgenForConditionalGeneration",
                "Gemma2ForCausalLM",
                "Gemma3ForCausalLM",
                "LlamaForCausalLM",
                "MistralForCausalLM",
                "Qwen2ForCausalLM",
                "Qwen3ForCausalLM",
                "Qwen3_5ForCausalLM",
                "Qwen3_5ForConditionalGeneration",
            ],
            ["safetensors", "json", "tokenizer"],
            [
                "chat-v1",
                "transcription-v1",
                "text-to-music-v1",
                "embeddings-v1",
                "reranking-v1",
            ],
            ["mps", "cuda", "cpu"],
            ["fp32", "fp16", "bf16"],
            [],
            ["config.json", "*.safetensors", "tokenizer*"],
            "transformers-v1",
            "oryn.transformers-pytorch",
        ),
        "a6401ad03d85893fa306e14756df6d7659eb8e73efa539be13d2d6907e8c4b00",
        "v1himMlxnvMHzRh08iKMCGwqyljBXJq6GytB8Q83rRoonaolt/61HNlTmMO2zi9WVi7ulFNniDZ4Z6U6+sZ+LzqE2OdP3VYYQ5w+CkKs+Ym1/kh31deqVs4ofjjzEhctEDZIuuro352fSyq4Z6Dw+mBt1dyaVPVHhYllUYdCG1TtY/Y2YAvHaa8kYKMYfVXbpaxyYD1wNPQP9QFdLE4FFj92G/H7qUd1GpWNb1X78ZRJbz/f/N8c/8QAtXpJCgrXIgkIp9DEnwlYMZdM54aFxG34ToNZG8yYUQNtb/sqvw9sc8g0qHi5KHFSGJfBjVmSTTvTRV4dVKMyHI7Ptt/ZgQ==",
    ),
    _envelope(
        _adapter(
            "oryn.adapter.diffusers",
            "oryn.runtime.pytorch",
            ["DiffusionPipeline", "StableDiffusionXLPipeline", "WanPipeline"],
            ["safetensors", "json", "tokenizer", "diffusers"],
            ["text-to-image-v1", "image-to-image-v1", "text-to-video-v1", "image-to-video-v1"],
            ["mps", "cuda", "cpu"],
            ["fp32", "fp16", "bf16"],
            ["bitsandbytes-nf4"],
            ["model_index.json", "*/*.json", "*.safetensors"],
            "diffusers-v1",
            "oryn.diffusers-pytorch",
        ),
        "b616a39480931299cfbdbd3156458c278121d39feb976283b08dc1a91496adb2",
        "FagALqPLqwTN1ULE1RWfcZqYbjSJWyMoQWPoRdqO5Jj2zoT1aQ8vAM/DOgh3PvpSRpOYEBAXo7XlgwVGrPNyDTpiauwquoIlNMcnX34GUBy7zlei3/tNIcnMTMTwz9wg9/tL/gZOQfDSD/rRdDu8TMGt7RkBOBwKXKIkII9AZmyLtjyxqjo0JOvbDOtWNFZSDPnGVJ0FfU5RIELsYbJMAomSy0GCAGqG0h9qk/Hs5sv+i1w7I4zqfvZ43VJoaSLRPTfxtRVULsWZm93Yx8rI0AwJZiiwmDZEvpedDy8ZY0OViS2XzzCQfGDDTauuXCEIVXh6ocGdPpRX9T93xweJsA==",
        BITSANDBYTES_DEVELOPMENT_PUBLISHER,
    ),
    _envelope(
        _adapter(
            "oryn.adapter.ace-step",
            "oryn.runtime.mlx",
            ["AceStep1.5Turbo"],
            ["safetensors", "json", "tokenizer"],
            ["text-to-music-v1"],
            ["mlx"],
            ["fp32", "fp16", "bf16"],
            [],
            ["acestep-v15-turbo/config.json", "acestep-v15-turbo/model.safetensors"],
            "ace-step-v1",
            "oryn.ace-step-mlx",
        ),
        "a88c87fb3b8011e21e136ffec32c5a690048c88eed3f22de9943123412558e14",
        "HLOdMsaQ3YDhGKVStVWyzeiHh6pDrYJol5ActWkMxtYSAVCgEii7XltkchvdPJ7B3saVUn/33iur4WGhfOVjgT7PHzD2wIoXbYfrCfGGSh1H1+Ei7U1V8XhOJGfmMfxaqaA7h0oUg7EoOfgOJx9Ee8cUiVnqLMJiIXJOOvYj0ZVyi/EzAp1wDR2H0TBasfIAyQGjDV3jCHGv/CWprJgxLX93JT2rdsprjZtEpYKCCIqpsLpPdXYqBE2fSV1Y0ZVs+nz9mVTZFIPD79LJRCrJBOtmpPhwE5ed1RuhllUpTvOiLjZ/f+dgSpt3rGnlKD3oRL3EX6UQ3f221LKSrypT6A==",
    ),
    _envelope(
        _adapter(
            "oryn.adapter.mflux-image",
            "oryn.runtime.mlx",
            ["FluxTransformer2DModel", "Krea2Pipeline"],
            ["safetensors", "json", "tokenizer", "mflux"],
            ["text-to-image-v1"],
            ["mlx"],
            ["fp16", "bf16"],
            ["int8", "int4", "q4"],
            ["model_index.json", "transformer/*.safetensors", "vae/*.safetensors"],
            "mflux-v1",
            "oryn.mflux-mlx",
        ),
        "f3e396a8659d9b83921d6fb2fdfe66dd7791e75af2554f97c72d98df4ce99e0c",
        "KAZYdPNt1CKDDmZSHXC37gS//YXzTd19hDqVmrqiURwuXeZeyyNYbzBTF1EVtUPRjuPaE5U+AyjraOz1mqeQtqYEhEl9dEPxaFSz/VEUfpIqdK0t71fwSmkpvh6XtbfMYvzBMS8uOZX89yJWLL+JN1dT47VvTf/hZTAxp1oWGSSzkzgiZUwccSC4AjChrUZlwhW4gUZ9GDcsMKerxHD4iJf88bzKMFv5b60VYWwnEgIbswcFBfh+UU7zcb7gqqT2vB4lq5nmRw0pnEMn+dT3rzMCj8VeUGsyni30rmy9v7AXBR1oDvpX5h2pYUZzL4HLPnOnnHNhZeSqszqKEGWdfQ==",
    ),
    _envelope(
        _adapter(
            "oryn.adapter.kokoro",
            "oryn.runtime.onnx",
            ["KokoroForSpeechSynthesis"],
            ["onnx", "json", "tokenizer", "npy"],
            ["speech-synthesis-v1"],
            ["coreml", "cpu"],
            ["fp32", "fp16", "int8"],
            ["int8"],
            ["*.onnx", "*.json", "voices/*.bin"],
            "kokoro-v1",
            "oryn.kokoro-onnx",
        ),
        "3d7196c7b29a40b1100a808b8d417f28f21332a22acf2bf9557e8ad37e35331d",
        "kOxJSujNRJxN699JZpoLCKB7znNO2cr0+9E4ZLIX+VZ9bQ+ZqCVz+WD0U+wXxYWWRUdh0fQaaCb7JSIRHwupNdNYpZNBIxDlGD33qBaAbEXfyGCk2s7AD3ze/DTRSnYJA0bzqvxfqGMxFlpw4PLnR789rjD8Lf9/PhYz1K1nAhpsqwULKj7pIvkdzFwvIlf87ZvD4Fk3Cs8raf6qKz07Y7AaOdME+nqgA77rNAoXRlXM8qknVT+/IeRkgzC4MfMBYRfHCGpCxd8uoBkNNPQOjt3fXxKwXbSzVa/p1mw9/aTHVSpTzGhB3igLlR+vvsfhFfzyaxf2AaQNHICSTddJ8Q==",
    ),
    _envelope(
        _adapter(
            "oryn.adapter.gguf-chat",
            "oryn.runtime.llama-cpp",
            ["llama", "gemma3", "qwen2"],
            ["gguf"],
            ["chat-v1"],
            ["metal", "cuda", "cpu"],
            ["fp32", "fp16", "bf16"],
            ["Q4_K_M", "Q8_0", "Q5_K_M", "IQ4_XS"],
            ["*.gguf"],
            "gguf-v1",
            "oryn.llama-cpp",
        ),
        "18848d10bd2ee90c4a4c5b730069b1e1bb3567e560e2fbf9a2aa8acf1ff6434f",
        "YA9CIH4br0Ptb2i2Sjr4Dq6L1y1AdL4nmbm93bELiz7OLTbmMV0yIvyuyX3gVJKg5miHRDBwlrQ5tERSs2ehi0cwSTqRDbkBKWvTKHqHD+0yB73USg9w6rsFBGZXksm+aOkXmS11MDl5rbRAz7zsZWhVfonIbALktkmy4wtT6r4rffG7OnkDzw38ikctQCB2JbQKpW4h4SHoAq31lpp6kQQnaBy8Yb6FoAdIgVyzMJuhVxMuRbou053FDaGyaUux4fjOU4D+OVYjlVjH4usqrU3KPfo7YQMs9mOfQtuWzzj9A4kvi/QGHeEhU6GV/bz00mlbRBKWNzy4K62RhfqIGw==",
    ),
)


def builtin_foundation_registry() -> FoundationRegistry:
    verifier = SignedManifestVerifier(
        {
            DEVELOPMENT_KEY_ID: TrustedPublisherKey(DEVELOPMENT_KEY_ID, DEVELOPMENT_MODULUS),
            FOUNDATION_PUBLISHER.key_id: FOUNDATION_PUBLISHER,
            BITSANDBYTES_DEVELOPMENT_PUBLISHER.key_id: BITSANDBYTES_DEVELOPMENT_PUBLISHER,
        }
    )
    registry = FoundationRegistry(verifier)
    for envelope in FOUNDATION_ENVELOPES:
        if envelope["payload"]["foundationId"] not in {"oryn.runtime.mlx", "oryn.runtime.pytorch"}:
            registry.register_foundation(envelope)
    registry.register_foundation(
        json.loads((Path(__file__).parents[1] / "data/pytorch-foundation.json").read_text())
    )
    registry.register_foundation(
        json.loads((Path(__file__).parents[1] / "data/mlx-foundation.json").read_text())
    )
    registry.register_foundation(
        json.loads((Path(__file__).parents[1] / "data/qwen-tts-foundation.json").read_text())
    )
    for envelope in ADAPTER_ENVELOPES:
        if envelope["payload"]["adapterId"] not in {
            "oryn.adapter.transformers",
            "oryn.adapter.diffusers",
            "oryn.adapter.mflux-image",
            "oryn.adapter.gguf-chat",
            "oryn.adapter.kokoro",
        }:
            registry.register_adapter(envelope)
    registry.register_adapter(
        json.loads((Path(__file__).parents[1] / "data/transformers-adapter.json").read_text())
    )
    registry.register_adapter(
        json.loads((Path(__file__).parents[1] / "data/mlx-adapter.json").read_text())
    )
    registry.register_adapter(
        json.loads((Path(__file__).parents[1] / "data/diffusers-adapter.json").read_text())
    )
    registry.register_adapter(
        json.loads((Path(__file__).parents[1] / "data/mflux-adapter.json").read_text())
    )
    registry.register_adapter(
        json.loads((Path(__file__).parents[1] / "data/onnx-adapter.json").read_text())
    )
    registry.register_adapter(
        json.loads((Path(__file__).parents[1] / "data/llama-adapter.json").read_text())
    )
    registry.register_adapter(
        json.loads((Path(__file__).parents[1] / "data/kokoro-adapter.json").read_text())
    )
    registry.register_adapter(
        json.loads((Path(__file__).parents[1] / "data/qwen-tts-adapter.json").read_text())
    )
    registry.register_adapter(
        json.loads((Path(__file__).parents[1] / "data/mlx-audio-adapter.json").read_text())
    )
    return registry
