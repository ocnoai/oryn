"""Verify a packaged provider's code identity, native loading and optional real execution.

Use the packaged Python interpreter with -I. Arguments: Engine root, worker sandbox
profile, and optionally a ModelStore root containing an installed provider checkpoint.
The optional generation uses the existing exact files and never downloads weights.
A fourth argument saves the generated PNG outside the temporary working directory.
"""

import hashlib
import json
import os
import shutil
import site
import sys
import tempfile
import threading
from pathlib import Path


def main():
    engine, sandbox = (Path(a).resolve(strict=True) for a in sys.argv[1:3])
    if not Path(sys.executable).resolve().is_relative_to(engine):
        raise ValueError("Use the packaged Python interpreter")
    os.environ.update(
        ORYN_ENGINE_ROOT=str(engine),
        ORYN_PACK_SANDBOX_PROFILE=str(sandbox),
        ORYN_SANDBOX_REQUIRED="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    site.addsitedir(str(engine / "runtime-foundations/oryn.runtime.mlx/site-packages"))
    site.addsitedir(str(engine / "runtime-packs/oryn.mflux-mlx/site-packages"))
    import mlx.core as mx
    import mlx.nn as nn
    from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
    from mflux.models.common.weights.loading.weight_applier import WeightApplier
    from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
    from mflux.utils.image_util import ImageUtil
    from mlx.utils import tree_flatten
    from oryn_mflux_provider.worker import validated_decoding

    from oryn_engine.providers import verify_catalog, verify_plan
    from oryn_engine.runners import _selected_model_view
    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry
    from oryn_engine.runtime_packs.execution import PackRuntimeLocator, StreamingPackProcess

    pack = builtin_registry().active_pack("oryn.mflux-mlx")
    runtime = PackRuntimeLocator().resolve(pack)
    assert runtime is not None
    data = verify_catalog(pack, runtime)
    native = runtime.site_packages / "mflux"
    for name, digest in data["sources"].items():
        assert hashlib.sha256((native / name).read_bytes()).hexdigest() == digest, name
    probe = StreamingPackProcess().invoke(
        pack,
        runtime,
        "describe",
        {},
        cancellation=threading.Event(),
        event_sink=lambda *_: None,
        timeout_seconds=60,
    )
    assert probe["catalogDigest"] == pack.provider["catalogSha256"]

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(128, 64)

        def __call__(self, value):
            return self.projection(value)

    quantized = []
    for bits in (None, 4, 8):
        source = Tiny()
        if bits is not None:
            nn.quantize(source, bits=bits, group_size=64)
        weights = LoadedWeights({"transformer": source.parameters()}, MetaData(bits, "0.20.0"))
        loaded = Tiny()
        component = ComponentDefinition(name="transformer", hf_subdir="transformer")
        WeightApplier.apply_and_quantize_single(weights, loaded, component, None)
        value = mx.ones((1, 128))
        result = loaded(value)
        mx.eval(result)
        assert bool(mx.allclose(source(value), result).item())
        assert all(
            bool(mx.all(mx.isfinite(v)).item()) for _, v in tree_flatten(loaded.parameters())
        )
        quantized.append(bits)
    original_denormalize = ImageUtil._denormalize
    with validated_decoding():
        assert bool(mx.allclose(ImageUtil._denormalize(mx.array([0.0])), mx.array([0.5])).item())
        for invalid in (float("nan"), float("inf"), -float("inf")):
            try:
                ImageUtil._denormalize(mx.array([invalid]))
            except FloatingPointError:
                pass
            else:
                raise AssertionError("Non-finite decoded pixels were accepted")
    assert ImageUtil._denormalize is original_denormalize
    report = {
        "provider": pack.pack_id,
        "catalogDigest": probe["catalogDigest"],
        "models": probe["models"],
        "uncovered": probe["uncovered"],
        "nativeStoredPrecisions": quantized,
        "signedCodeVerified": True,
        "nonFiniteOutputRejected": True,
    }
    if len(sys.argv) > 3:
        store = Path(sys.argv[3]).resolve(strict=True)
        records = json.loads((store / "installations.json").read_text())
        records = records["models"] if isinstance(records, dict) else records
        record = next(r for r in records if r.get("providerPlans"))
        selected = record["runtimePlan"]["selected"]
        bound = record["providerPlans"][selected["candidateId"]]
        verify_plan(bound, pack, runtime, record)
        entry = next(e for e in probe["entries"] if e["id"] == bound["providerPlan"]["entryId"])
        generation_input = entry["defaults"] | {
            "prompt": "A simple blue circle on a white background",
            "seed": 42,
        }
        record["cacheRoot"] = str(store / "cache")
        model = record["model"]
        root = (
            store
            / "cache"
            / ("models--" + model["repository"].replace("/", "--"))
            / "snapshots"
            / model["revision"]
        )
        with (
            _selected_model_view(record, root) as view,
            tempfile.TemporaryDirectory(prefix="oryn-provider-result-") as output,
        ):
            path = Path(output) / "image.png"
            result = StreamingPackProcess().invoke(
                pack,
                runtime,
                "generate",
                {
                    "modelRoot": str(view),
                    "outputPath": str(path),
                    "plan": bound["providerPlan"],
                    "input": generation_input,
                },
                cancellation=threading.Event(),
                event_sink=lambda event, value: print(
                    json.dumps({"event": event, "data": value}), flush=True
                ),
                timeout_seconds=1800,
            )
            from PIL import Image, ImageStat

            with Image.open(path) as generated:
                deviation = ImageStat.Stat(generated.convert("RGB")).stddev
                assert max(deviation) > 1, "The generation probe produced a constant image"
            report["execution"] = {
                "model": model,
                "settings": generation_input,
                **result,
                "pngBytes": path.stat().st_size,
                "pixelStandardDeviation": deviation,
            }
            if len(sys.argv) > 4:
                shutil.copyfile(path, Path(sys.argv[4]).resolve())
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
