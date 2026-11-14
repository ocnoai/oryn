"""Oryn-owned acceptance harness; explicit prepare downloads data, execute is offline.

Run with the packaged interpreter to measure the shipped runtime. No model code, package
installation, conversion, automatic certification, or cross-model fallback is performed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
from PIL import Image, ImageDraw, ImageFont, ImageStat

from oryn_engine.artifacts import ArtifactOutputStaging
from oryn_engine.jobs import EngineJob, JobExecutionError
from oryn_engine.models import ModelStore
from oryn_engine.runners import RunnerRegistry
from oryn_engine.shared_tasks import (
    ARTIFACT_TASKS,
    IMAGE_TASKS,
    TASK_CONTRACTS,
    validate_task_output,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "execute"])
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument(
        "--models",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/fixtures/sprint47-models.json",
    )
    parser.add_argument("--groups", default="mlx,gguf,embedding,ranking,vision,editing")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-root", type=Path)
    parser.add_argument("--sandbox-profiles", type=Path)
    parser.add_argument(
        "--image-manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/fixtures/sprint47-image.json",
    )
    args = parser.parse_args()
    args.state = args.state.resolve()
    args.output = args.output.resolve()
    runtime_root = args.engine_root or next(
        (
            parent
            for parent in Path(sys.prefix).parents
            if (parent / "release-manifest.json").is_file()
        ),
        None,
    )
    if runtime_root:
        os.environ["ORYN_ENGINE_ROOT"] = str(runtime_root.resolve())
    if args.sandbox_profiles:
        raise SystemExit(sandboxed_execution(args))
    if args.phase == "execute":
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    store = ModelStore(args.state)
    if args.cache:
        store.cache_root = args.cache.resolve()
        store.cache_root.mkdir(parents=True, exist_ok=True)
    registry = RunnerRegistry(ArtifactOutputStaging(args.state / "outputs", "engine_acceptance"))
    registry.routing = store.runtime_routing
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = (
        subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "hw.model", "machdep.cpu.brand_string", "hw.memsize"],
            text=True,
        ).strip()
        if platform.system() == "Darwin"
        else platform.machine()
    )
    report = {
        "schemaVersion": 1,
        "phase": args.phase,
        "device": device,
        "os": platform.platform(),
        "harnessSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "modelManifestSha256": hashlib.sha256(args.models.read_bytes()).hexdigest(),
        "automaticCertification": False,
        "workerSandboxRequired": os.environ.get("ORYN_SANDBOX_REQUIRED") == "1",
        "parentSandbox": os.environ.get("ORYN_ACCEPTANCE_PARENT_SANDBOX", "none"),
        "results": [],
    }
    engine_root = os.environ.get("ORYN_ENGINE_ROOT")
    if engine_root:
        manifest = Path(engine_root) / "release-manifest.json"
        if manifest.is_file():
            report["releaseManifestSha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    cases = json.loads(args.models.read_text())
    if {"broad", "expansion", "completion", "existing"}.intersection(args.groups.split(",")):
        image_manifest = json.loads(args.image_manifest.read_text())
        reference_image = args.state / "reference-image.png"
        if args.phase == "prepare":
            from huggingface_hub import hf_hub_download

            image_file = hf_hub_download(
                image_manifest["repository"],
                image_manifest["file"],
                repo_type=image_manifest["repoType"],
                revision=image_manifest["revision"],
            )
            shutil.copyfile(image_file, reference_image)
        if hashlib.sha256(reference_image.read_bytes()).hexdigest() != image_manifest["sha256"]:
            raise ValueError("Acceptance image changed")
        report["referenceImage"] = image_manifest
    failures = 0
    for index, case in enumerate(cases):
        if case["group"] not in args.groups.split(","):
            continue
        result = dict(case)
        model = {
            "provider": "huggingface",
            "repository": case["repository"],
            "revision": case["revision"],
        }
        start = time.monotonic()
        handle = None
        try:
            if args.phase == "prepare":
                profile = store.inspect(model)
                result["admission"] = {
                    k: profile.get(k)
                    for k in (
                        "executable",
                        "loadDisposition",
                        "blockingReasons",
                        "estimatedMemoryBytes",
                    )
                }
                if not profile["executable"]:
                    raise ValueError(result["admission"])
                store.install(f"op_acceptance_{index}", model)
                result["status"] = "installed"
            else:
                record = store.installed_record(model)
                result["selectedFilesSha256"] = hashlib.sha256(
                    json.dumps(record["files"], sort_keys=True).encode()
                ).hexdigest()
                result["runtimePlan"] = record.get("runtimePlan")
                result["samples"] = []
                for sample in range(2):
                    # This measures reload, including workers that deliberately exit after each job.
                    load_began = time.monotonic()
                    handle = registry.load(record)
                    load_seconds = time.monotonic() - load_began
                    artifact_id = "artifact_" + "a" * 24
                    image_path = args.state / "input.png"
                    color = "red" if sample == 0 else "blue"
                    Image.new("RGB", (256, 256), color).save(image_path)
                    if (
                        case["task"]
                        in {
                            "image-classification",
                            "object-detection",
                            "depth-estimation",
                            "zero-shot-image-classification",
                        }
                        | IMAGE_TASKS
                    ):
                        shutil.copyfile(args.state / "reference-image.png", image_path)
                    if case.get("fixture") == "ocr":
                        image = Image.new("RGB", (640, 160), "white")
                        ImageDraw.Draw(image).text(
                            (20, 45),
                            "ORYN LOCAL AI 2026",
                            fill="black",
                            font=ImageFont.load_default(size=40),
                        )
                        image.save(image_path)
                    binding = {
                        "artifactId": artifact_id,
                        "internalPath": str(image_path.resolve()),
                        "mimeType": "image/png",
                        "sizeBytes": image_path.stat().st_size,
                    }
                    task = case["task"]
                    if task in {"chat", "vision"}:
                        value = {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "What is 2 + 2? Answer briefly."
                                    if task == "chat"
                                    else "What color fills this image? Answer with the color name.",
                                }
                            ],
                            "maxTokens": 32,
                            "temperature": 0,
                            "topP": 1,
                        }
                        if task == "vision":
                            value["imageArtifactId"] = artifact_id
                    elif task in TASK_CONTRACTS:
                        value = {"imageArtifactId": artifact_id} if task in IMAGE_TASKS else {}
                        if task == "zero-shot-classification":
                            value.update(
                                text="A cat sits on a sofa.",
                                labels=["animals", "financial markets"],
                                multiLabel=False,
                            )
                        if task == "image-feature-extraction":
                            value["normalize"] = True
                        if task == "mask-generation":
                            value["points"] = [{"x": 0.5, "y": 0.5, "label": 1}]
                        if task == "zero-shot-object-detection":
                            value.update(labels=["cat", "dog"], threshold=0.1)
                        if task == "visual-question-answering":
                            value.update(question="What animal is in this picture?", maxTokens=32)
                        if task == "image-to-text":
                            value["maxTokens"] = 32
                    elif task == "embedding":
                        value = {
                            "texts": [
                                "A dog is running.",
                                "A puppy runs.",
                                "The stock market closed.",
                            ],
                            "normalize": True,
                        }
                    elif task == "ranking":
                        value = {
                            "query": "What is the capital of France?",
                            "documents": ["Paris is the capital of France.", "Penguins eat fish."],
                            "topK": 2,
                        }
                    elif task == "text-classification":
                        value = {
                            "text": "I loved this excellent movie. It was wonderful!",
                            "topK": 2,
                        }
                    elif task == "question-answering":
                        value = {
                            "question": "What is the capital of France?",
                            "context": (
                                "Paris is the capital of France. Berlin is the capital of Germany."
                            ),
                        }
                    elif task == "fill-mask":
                        value = {"text": "The capital of France is [MASK].", "topK": 5}
                    elif task == "token-classification":
                        value = {"text": "John Smith lives in London."}
                    elif task == "zero-shot-image-classification":
                        value = {
                            "imageArtifactId": artifact_id,
                            "labels": ["a photo of a cat", "a photo of a car"],
                        }
                    elif task == "depth-estimation":
                        value = {"imageArtifactId": artifact_id}
                    elif task == "summarization":
                        value = {
                            "text": (
                                "summarize: Paris is the capital of France. It is known for the "
                                "Eiffel Tower and many museums. Millions of tourists visit "
                                "the city each year."
                            ),
                            "maxTokens": 32,
                        }
                    elif task in {"image-classification", "object-detection"}:
                        value = {"imageArtifactId": artifact_id}
                    elif task == "image-editing":
                        value = {
                            "prompt": "Paint a small blue bird in watercolor",
                            "imageArtifactId": artifact_id,
                            "width": 256,
                            "height": 256,
                            "steps": 4,
                            "guidanceScale": 1,
                            "seed": 7,
                        }
                    elif task == "image-layers":
                        value = {
                            "prompt": "Separate the foreground bird from the background",
                            "imageArtifactId": artifact_id,
                            "resolution": 640,
                            "layers": 2,
                            "steps": 4,
                            "guidanceScale": 1,
                            "seed": 7,
                        }
                    elif task == "music":
                        value = {
                            "prompt": "gentle acoustic guitar instrumental",
                            "lyrics": "[Instrumental]",
                            "durationSeconds": 10,
                            "seed": 7,
                        }
                    elif task == "text-to-image":
                        value = {
                            "prompt": "a small blue bird, watercolor painting",
                            "width": 256,
                            "height": 256,
                            "steps": 4,
                            "guidanceScale": 0,
                            "seed": 7,
                        }
                    else:
                        value = {
                            "prompt": "a small blue bird, watercolor painting",
                            "imageArtifactId": artifact_id,
                            "width": 256,
                            "height": 256,
                            "steps": 12,
                            "guidanceScale": 5,
                            "strength": 0.8,
                            "seed": 7,
                        }
                    value.update(case.get("input", {}))
                    job = EngineJob(
                        job_id=f"job_acceptance_{index}_{sample}",
                        session_id="session_acceptance",
                        model=model,
                        input=value,
                        input_artifacts=(binding,)
                        if task
                        in {
                            "vision",
                            "editing",
                            "image-editing",
                            "image-layers",
                            "image-classification",
                            "object-detection",
                            "depth-estimation",
                            "zero-shot-image-classification",
                        }
                        | IMAGE_TASKS
                        else (),
                        invocation_digest="oryn-acceptance",
                        state="running",
                        enqueued_at=0,
                    )
                    first = []
                    began = time.monotonic()

                    def event(name, data, first=first, began=began):
                        if name == "job.output.delta" and not first:
                            first.append(time.monotonic() - began)

                    output = registry.execute(job, handle, threading.Event(), event)
                    sample_result = {
                        "loadSeconds": load_seconds,
                        "elapsedSeconds": time.monotonic() - began,
                        "firstDeltaSeconds": first[0] if first else None,
                        "backend": getattr(handle, "backend", None),
                        "output": output,
                    }
                    if task in TASK_CONTRACTS:
                        if task in ARTIFACT_TASKS:
                            with (
                                Image.open(output["artifact"]["internalPath"]) as mask,
                                Image.open(image_path) as original,
                            ):
                                assert mask.size == original.size
                                extrema = mask.getextrema()
                                assert extrema[0] != extrema[1], (
                                    "Expected a nonconstant spatial output"
                                )
                                if task == "mask-generation":
                                    assert extrema == (0, 255)
                            sample_result["output"] = {
                                "mimeType": "image/png",
                                "dimensions": list(mask.size),
                                "extrema": extrema,
                            }
                        else:
                            result_value = validate_task_output(task, output["value"])
                            if task == "zero-shot-classification":
                                assert result_value["predictions"][0]["label"] == "animals"
                            if task == "zero-shot-object-detection":
                                assert any(
                                    "cat" in row["label"].lower()
                                    for row in result_value["detections"]
                                )
                            if task in {"image-to-text", "visual-question-answering"}:
                                expected = (
                                    "ORYN LOCAL AI 2026" if case.get("fixture") == "ocr" else "cat"
                                )
                                assert expected.lower() in " ".join(
                                    result_value["text"].lower().split()
                                )
                            if task == "image-feature-extraction":
                                sample_result["output"] = {
                                    "dimensions": result_value["dimensions"],
                                    "normalized": True,
                                    "pooling": result_value["pooling"],
                                }
                    elif task == "embedding":
                        vectors = output["value"]["embeddings"]
                        assert all(
                            abs(math.sqrt(sum(x * x for x in row)) - 1) < 1e-4 for row in vectors
                        )
                        related = sum(a * b for a, b in zip(vectors[0], vectors[1], strict=True))
                        unrelated = sum(a * b for a, b in zip(vectors[0], vectors[2], strict=True))
                        assert related > unrelated
                        sample_result["output"] = {
                            "dimensions": len(vectors[0]),
                            "normalized": True,
                            "relatedCosine": related,
                            "unrelatedCosine": unrelated,
                        }
                    elif task == "fill-mask":
                        predictions = output["value"]["predictions"]
                        assert any("paris" in item["token"].lower() for item in predictions)
                        assert all(
                            math.isfinite(item["score"]) and 0 <= item["score"] <= 1
                            for item in predictions
                        )
                    elif task == "token-classification":
                        tokens = output["value"]["tokens"]
                        assert any(
                            "PER" in token["label"] and "John" in token["token"] for token in tokens
                        )
                        assert any(
                            "LOC" in token["label"] and "London" in token["token"]
                            for token in tokens
                        )
                    elif task == "zero-shot-image-classification":
                        predictions = output["value"]["predictions"]
                        assert (
                            max(predictions, key=lambda item: item["score"])["label"]
                            == "a photo of a cat"
                        )
                    elif task == "depth-estimation":
                        depth = output["value"]
                        assert all(math.isfinite(depth[k]) for k in ("minimum", "maximum", "mean"))
                        assert depth["minimum"] < depth["mean"] < depth["maximum"]
                        assert 1 <= depth["width"] <= 2048 and 1 <= depth["height"] <= 2048
                    elif task == "ranking":
                        assert output["value"]["results"][0]["index"] == 0
                    elif task == "vision":
                        assert color in output["value"]["text"].lower()
                    elif task == "chat":
                        assert output["value"]["text"].strip()
                    elif task == "question-answering":
                        answer = output["value"]
                        assert answer["answer"] == "Paris"
                        assert value["context"][answer["start"] : answer["end"]] == "Paris"
                    elif task == "summarization":
                        assert output["value"]["text"].strip()
                    elif task in {"text-classification", "image-classification"}:
                        predictions = output["value"]["predictions"]
                        assert predictions and all(
                            math.isfinite(x["score"]) and 0 <= x["score"] <= 1 for x in predictions
                        )
                        if task == "text-classification":
                            assert "positive" in predictions[0]["label"].lower()
                        else:
                            assert any("cat" in item["label"].lower() for item in predictions)
                    elif task == "object-detection":
                        detections = output["value"]["detections"]
                        assert any(item["labelId"] == 17 for item in detections), (
                            "Expected cat detection"
                        )
                        assert all(math.isfinite(v) for item in detections for v in item["box"])
                    elif task == "music":
                        audio_path = Path(output["artifact"]["internalPath"])
                        assert audio_path.stat().st_size <= 8 * 1024**2
                        raw = audio_path.read_bytes()
                        assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
                        chunks, offset = {}, 12
                        while offset + 8 <= len(raw):
                            length = struct.unpack_from("<I", raw, offset + 4)[0]
                            assert offset + 8 + length <= len(raw)
                            chunks[raw[offset : offset + 4]] = raw[offset + 8 : offset + 8 + length]
                            offset += 8 + length + length % 2
                        encoding, channels, rate, byte_rate, alignment, bits = struct.unpack_from(
                            "<HHIIHH", chunks[b"fmt "]
                        )
                        assert encoding == 3 and bits == 32 and channels in {1, 2}
                        assert alignment == channels * 4 and byte_rate == rate * alignment
                        duration = len(chunks[b"data"]) / byte_rate
                        assert 9 <= duration <= 11
                        import numpy as np

                        samples = np.frombuffer(chunks[b"data"], dtype="<f4")
                        assert np.isfinite(samples).all() and float(np.max(np.abs(samples))) > 1e-4
                        sample_result["audioDurationSeconds"] = duration
                    elif task == "image-layers":
                        assert (
                            output["kind"] == "multi" and len(output["outputs"]) == value["layers"]
                        )
                        geometry = None
                        for layer in output["outputs"]:
                            with Image.open(layer["artifact"]["internalPath"]) as image:
                                assert (
                                    image.mode == "RGBA" and image.width * image.height <= 1024**2
                                )
                                assert geometry is None or geometry == image.size
                                geometry = image.size
                        sample_result["layerCount"] = len(output["outputs"])
                    elif task in {"editing", "text-to-image", "image-editing"}:
                        with Image.open(output["artifact"]["internalPath"]) as generated:
                            assert generated.size == (256, 256)
                            deviation = ImageStat.Stat(generated.convert("RGB")).stddev
                            assert max(deviation) > 1, "Image editing returned a constant image"
                            sample_result["imageStandardDeviation"] = deviation
                            if case.get("fixture") == "colored-background-edit":
                                corner = ImageStat.Stat(
                                    generated.convert("RGB").crop((0, 0, 32, 32))
                                ).mean
                                dominant = 0 if sample == 0 else 2
                                other = 2 if sample == 0 else 0
                                assert corner[dominant] > corner[other] + 20, (
                                    "Instruction editing did not preserve the source background"
                                )
                                sample_result["backgroundMeanRgb"] = corner
                        sample_result["imageSha256"] = hashlib.sha256(
                            Path(output["artifact"]["internalPath"]).read_bytes()
                        ).hexdigest()
                    sample_result["processRssAfterBytes"] = psutil.Process().memory_info().rss
                    result["samples"].append(sample_result)
                    registry.unload(handle)
                    handle = None
                if case["task"] in {"editing", "image-editing"}:
                    assert (
                        result["samples"][0]["imageSha256"] != result["samples"][1]["imageSha256"]
                    ), "Changing the source image did not affect the edit"
                if case["task"] in {
                    "chat",
                    "vision",
                    "editing",
                    "text-to-image",
                    "image-editing",
                    "image-layers",
                } | set(TASK_CONTRACTS):
                    handle = registry.load(record)
                    cancel = threading.Event()
                    cancel.set()
                    try:
                        registry.execute(job, handle, cancel, lambda _name, _data: None)
                        raise AssertionError("Cancelled request executed")
                    except JobExecutionError as exc:
                        assert exc.code == "JOB_CANCELLED"
                    result["preCancelled"] = True
                    cancel.clear()
                    cancelled_after_delta = []

                    def cancel_after_delta(
                        name,
                        data,
                        cancelled_after_delta=cancelled_after_delta,
                        cancel=cancel,
                        task=case["task"],
                    ):
                        if (
                            (
                                task in TASK_CONTRACTS
                                and name == "job.state"
                                and data.get("state") == "running"
                            )
                            or name == "job.output.delta"
                            or (
                                name == "job.progress"
                                and (
                                    task
                                    in {"editing", "text-to-image", "image-editing", "image-layers"}
                                    and data.get("stage") == "denoising"
                                )
                            )
                        ):
                            cancelled_after_delta.append(True)
                            cancel.set()

                    try:
                        registry.execute(job, handle, cancel, cancel_after_delta)
                        raise AssertionError("Generation continued after cancellation")
                    except JobExecutionError as exc:
                        assert exc.code == "JOB_CANCELLED"
                    assert cancelled_after_delta
                    result["cancelledDuringGeneration"] = True
                    if case["task"] in {"chat", "vision"}:
                        result["cancelledAfterFirstDelta"] = True
                result["status"] = "passed"
        except Exception as exc:
            result.update(
                status="failed",
                error=type(exc).__name__,
                reason=str(exc),
                code=getattr(exc, "code", None),
            )
            failures += 1
        finally:
            if handle is not None:
                registry.unload(handle)
        result["totalSeconds"] = time.monotonic() - start
        report["results"].append(result)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{case['repository']}: {result['status']} ({result.get('reason', '')})", flush=True)
    raise SystemExit(1 if failures else 0)


def sandboxed_execution(args) -> int:
    """Run the packaged interpreter under Engine + worker profiles, without network access."""
    if args.phase != "execute" or sys.platform != "darwin" or not args.engine_root:
        raise ValueError("Sandbox execution requires macOS, execute, and --engine-root")
    engine = args.engine_root.resolve()
    profiles = args.sandbox_profiles.resolve()
    state = args.state
    state.mkdir(parents=True, exist_ok=True)
    scratch = state / "scratch"
    scratch.mkdir(exist_ok=True)
    harness = state / "acceptance-harness.py"
    manifest = state / "acceptance-models.json"
    shutil.copyfile(Path(__file__), harness)
    shutil.copyfile(args.models, manifest)
    image_manifest_path = state / "acceptance-image.json"
    shutil.copyfile(args.image_manifest, image_manifest_path)
    profile = state / "acceptance-offline.sb"
    profile.write_text(
        (profiles / "engine.sb").read_text() + '\n(deny network-outbound (remote ip "*:*"))\n'
    )
    result = state / "acceptance-result.json"
    # A child that fails before writing its report must not reuse a previous cohort's evidence.
    result.unlink(missing_ok=True)
    args.output.unlink(missing_ok=True)
    parameters = {
        "ENGINE_ROOT": engine,
        "PYTHON_ROOT": engine / "runtime",
        "SECURITY_ROOT": profiles,
        "MODEL_ROOT": (args.cache or state / "cache").resolve(),
        "PACK_ROOT": engine / "runtime-packs",
        "ARTIFACT_INPUT_ROOT": state,
        "ARTIFACT_OUTPUT_ROOT": state,
        "SCRATCH_ROOT": scratch,
    }
    command = [
        str(engine / "runtime/bin/python3.12"),
        "-I",
        "-B",
        "-m",
        "oryn_engine.sandbox_launcher",
        "-f",
        str(profile),
    ]
    for key, value in parameters.items():
        command.extend(["-D", f"{key}={value}"])
    command.extend(
        [
            str(engine / "runtime/bin/python3.12"),
            "-I",
            "-B",
            str(harness),
            "execute",
            "--state",
            str(state),
            "--cache",
            str(parameters["MODEL_ROOT"]),
            "--models",
            str(manifest),
            "--image-manifest",
            str(image_manifest_path),
            "--groups",
            args.groups,
            "--output",
            str(result),
        ]
    )
    env = dict(os.environ)
    env.update(
        ORYN_ENGINE_ROOT=str(engine),
        ORYN_PACK_SANDBOX_PROFILE=str(profiles / "runtime-pack.sb"),
        ORYN_SANDBOX_REQUIRED="1",
        ORYN_ACCEPTANCE_PARENT_SANDBOX=(
            "engine.sb with IP networking denied; sibling sandboxed workers"
        ),
        ORYN_MODEL_ROOT=str(parameters["MODEL_ROOT"]),
        ORYN_ARTIFACT_INPUT_ROOT=str(state),
        ORYN_ARTIFACT_OUTPUT_ROOT=str(state),
        HOME=str(state),
        TMPDIR=str(scratch),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    completed = psutil.Popen(command, env=env, cwd=state)
    peak_rss = 0
    while completed.poll() is None:
        resident = 0
        try:
            processes = [completed, *completed.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            processes = []
        for process in processes:
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                resident += process.memory_info().rss
        peak_rss = max(peak_rss, resident)
        time.sleep(0.05)
    if result.is_file():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = json.loads(result.read_text())
        report["peakSummedProcessRssBytes"] = peak_rss
        report["memoryMeasurement"] = (
            "50 ms process-tree RSS sampling; shared pages may be counted more than once; "
            "not peak unified/GPU memory"
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return completed.returncode


if __name__ == "__main__":
    main()
