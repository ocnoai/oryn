"""Exercise the packaged Qwen 2.1 worker with small random native weights.

Run with the app's runtime/bin/python3.12 -I, passing its Resources/engine path
and Resources/security/runtime-pack.sb. This is execution evidence, not full
checkpoint conformance or image-quality evidence.
"""

import argparse
import importlib.util
import json
import os
import site
import sys
import tempfile
import threading
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument("sandbox", type=Path)
    parser.add_argument("--turbo", action="store_true")
    args = parser.parse_args()
    engine, sandbox = (path.resolve(strict=True) for path in (args.engine, args.sandbox))
    if not Path(sys.executable).resolve().is_relative_to(engine):
        raise ValueError("Use the packaged Python interpreter")
    site.addsitedir(str(engine / "runtime-packs/oryn.diffusers-pytorch/site-packages"))
    os.environ.update(
        ORYN_ENGINE_ROOT=str(engine),
        ORYN_PACK_SANDBOX_PROFILE=str(sandbox),
        ORYN_SANDBOX_REQUIRED="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    from PIL import Image

    from oryn_engine.runtime_packs.builtin_catalog import builtin_registry
    from oryn_engine.runtime_packs.execution import PackRuntimeLocator, StreamingPackProcess

    fixture_path = Path(__file__).resolve().parents[1] / "tests/qwen_image_21_fixture.py"
    spec = importlib.util.spec_from_file_location("qwen_fixture", fixture_path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    pack = builtin_registry().active_pack("oryn.diffusers-pytorch")
    runtime = PackRuntimeLocator().resolve(pack)
    assert runtime is not None
    results = []
    with tempfile.TemporaryDirectory(prefix="oryn-qwen21-packaged-") as temporary:
        root = Path(temporary).resolve()
        model = fixture.write_checkpoint(root / "model")
        steps = 2
        if args.turbo:
            metadata = json.loads(
                (fixture_path.parent / "fixtures/qwen-image-2.1-turbo.json").read_text()
            )
            index = json.loads((model / "model_index.json").read_text())
            sigmas = metadata["modelIndex"]["sample_sigmas"]
            steps = len(sigmas)
            (model / "model_index.json").write_text(json.dumps({**index, "sample_sigmas": sigmas}))
            (model / "processor/preprocessor_config.json").unlink()
            (model / "processor/processor_config.json").write_text(
                json.dumps(metadata["processor"])
            )
            (model / "scheduler/scheduler_config.json").write_text(
                json.dumps(metadata["scheduler"])
            )
        source = root / "reference.png"
        Image.new("RGBA", (64, 64), (220, 40, 10, 80)).save(source)
        for contract in ("text-to-image-v1", "image-editing-v1"):
            output = root / contract / "image.png"
            output.parent.mkdir()
            value = {
                "prompt": "A blue circle",
                "negativePrompt": "",
                "width": 64,
                "height": 64,
                "steps": steps,
                "guidanceScale": 1.0,
                "seed": 7,
            }
            payload = {
                "modelRoot": str(model),
                "taskContract": contract,
                "input": value,
                "outputPath": str(output),
            }
            if contract == "image-editing-v1":
                value["imageArtifactId"] = "artifact_" + "r" * 24
                payload["inputPath"] = str(source)
            events = []
            result = StreamingPackProcess().invoke(
                pack,
                runtime,
                "generate",
                payload,
                cancellation=threading.Event(),
                event_sink=lambda event, data, sink=events: sink.append((event, data)),
                timeout_seconds=120,
            )
            with Image.open(output) as image:
                assert image.mode == "RGBA" and image.size == (64, 64)
                assert image.getchannel("A").getextrema()[0] < 255
            denoising = [
                data
                for event, data in events
                if event == "job.progress" and data.get("stage") == "denoising"
            ]
            assert len(denoising) == steps
            results.append(
                {"contract": contract, "mode": "RGBA", "denoisingSteps": steps, **result}
            )
    print(
        json.dumps(
            {"fixture": "small-random-native-checkpoint", "turbo": args.turbo, "results": results}
        )
    )


if __name__ == "__main__":
    main()
