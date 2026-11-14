"""Original header validation and complete synthetic graph tests; no model downloads."""

import copy
import hashlib
import json
import shutil
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from oryn_engine.diffusion_lora import attach, target_shapes, validate_structure
from oryn_engine.model_composition import CompositionHub, adapter_name, binding, validate_binding

ORIGINALS = json.loads((Path(__file__).parent / "fixtures/qwen-lora-headers.json").read_text())
NATURAL = binding(ORIGINALS[2]["repository"], ORIGINALS[2]["revision"])
ANGLE = binding(ORIGINALS[1]["repository"], ORIGINALS[1]["revision"])
VIGGLE = binding(ORIGINALS[0]["repository"], ORIGINALS[0]["revision"])


def tiny_transformer(family="QwenImage21Transformer2DModel"):
    import diffusers
    import torch

    torch.manual_seed(11)
    config = dict(
        patch_size=1,
        in_channels=8,
        out_channels=8,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        axes_dims_rope=(4, 6, 6),
    )
    config.update(context_in_dim=16, mlp_ratio=2) if family.endswith(
        "21Transformer2DModel"
    ) else config.update(joint_attention_dim=16)
    return getattr(diffusers, family)(**config).eval()


def arrays(model, *, fused=False, alpha=False):
    import torch

    torch.manual_seed(29)
    shapes = target_shapes(json.loads(model.to_json_string()))
    if fused:
        for name in list(shapes):
            if name.endswith("img_mlp.gate_layer"):
                shape = shapes.pop(name)
                shapes.pop(name.removesuffix("gate_layer") + "proj")
                shapes[name.removesuffix("gate_layer") + "gate_up"] = (2 * shape[0], shape[1])
    tensors = {}
    for target, (outputs, inputs) in shapes.items():
        prefix = "diffusion_model." if fused else "transformer."
        a, b = ("lora_down", "lora_up") if alpha else ("lora_A", "lora_B")
        tensors[f"{prefix}{target}.{a}.weight"] = torch.randn(2, inputs) * 0.08
        tensors[f"{prefix}{target}.{b}.weight"] = torch.randn(outputs, 2) * 0.08
        if alpha:
            tensors[f"{prefix}{target}.alpha"] = torch.tensor(1.0)
    return tensors


def header(tensors):
    import torch

    return {
        "metadata": {},
        "tensors": {
            k: {
                "shape": list(v.shape),
                "dtype": {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16"}[
                    v.dtype
                ],
            }
            for k, v in tensors.items()
        },
    }


@pytest.mark.parametrize("original", ORIGINALS, ids=[x["repository"] for x in ORIGINALS])
def test_all_original_adapter_headers_match_complete_native_transformers(original):
    import diffusers
    import torch

    config = original["config"]
    with torch.device("meta"):
        native = getattr(diffusers, config["_class_name"]).from_config(config)
    plans = validate_structure(original["structure"], config)
    for plan in plans:
        a = original["structure"]["tensors"][plan["A"]]["shape"]
        b = original["structure"]["tensors"][plan["B"]]["shape"]
        outputs = b[0] if plan["rows"] is None else plan["rows"][1] - plan["rows"][0]
        assert tuple(native.get_submodule(plan["target"]).weight.shape) == (outputs, a[1])
    assert {p[key] for p in plans for key in ("A", "B", "alphaTensor") if p[key]} == set(
        original["structure"]["tensors"]
    )


@pytest.mark.parametrize("family", ["QwenImage21Transformer2DModel", "QwenImageTransformer2DModel"])
@pytest.mark.parametrize("precision", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_every_native_linear_update_matches_unmerged_reference(
    tmp_path, family, precision, backend
):
    import torch
    from safetensors.torch import save_file

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Apple Silicon")
    model = tiny_transformer(family).to(dtype=getattr(torch, precision), device=backend)
    tensors = {k: v.to(getattr(torch, precision)) for k, v in arrays(model, alpha=True).items()}
    path = tmp_path / "adapter.safetensors"
    save_file(tensors, path)
    original = copy.deepcopy(model)
    plans = validate_structure(header(tensors), json.loads(model.to_json_string()))
    modules = attach(model, path)
    for plan, module in zip(plans, modules, strict=True):
        x = torch.randn(1, 3, module.weight.shape[1], device=backend, dtype=module.weight.dtype)
        base = original.get_submodule(plan["target"])(x)
        a, b = tensors[plan["A"]].float().to(backend), tensors[plan["B"]].float().to(backend)
        expected = (
            base.float()
            + torch.nn.functional.linear(torch.nn.functional.linear(x.float(), a), b) * 0.5
        ).to(base.dtype)
        torch.testing.assert_close(module(x), expected, rtol=0, atol=0)
        module.enabled = False
        torch.testing.assert_close(module(x), base, rtol=0, atol=0)
        assert torch.equal(module.weight, original.get_submodule(plan["target"]).weight)


def test_comfy_fused_gate_up_preserves_both_nonzero_row_updates(tmp_path):
    import torch
    from safetensors.torch import save_file

    model = tiny_transformer()
    original = copy.deepcopy(model)
    tensors = arrays(model, fused=True)
    path = tmp_path / "adapter.safetensors"
    save_file(tensors, path)
    attach(model, path)
    for index in range(2):
        native = model.transformer_blocks[index].img_mlp
        reference = original.transformer_blocks[index].img_mlp
        x = torch.randn(2, 5, 32)
        prefix = f"diffusion_model.transformer_blocks.{index}.img_mlp."
        projected = torch.nn.functional.linear(
            torch.nn.functional.linear(x, tensors[prefix + "gate_up.lora_A.weight"]),
            tensors[prefix + "gate_up.lora_B.weight"],
        )
        gate, up = projected.chunk(2, dim=-1)
        hidden = torch.nn.functional.silu(reference.gate_layer(x) + gate) * (reference.proj(x) + up)
        expected = reference.out(hidden) + torch.nn.functional.linear(
            torch.nn.functional.linear(hidden, tensors[prefix + "out.lora_A.weight"]),
            tensors[prefix + "out.lora_B.weight"],
        )
        torch.testing.assert_close(native(x), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "problem",
    [
        "missing_b",
        "rank",
        "unknown",
        "collision",
        "geometry",
        "dtype",
        "alpha_shape",
        "orphan_alpha",
        "dora",
        "rslora",
        "alpha_pattern",
    ],
)
def test_malformed_adapter_metadata_is_rejected(problem):
    data = copy.deepcopy(ORIGINALS[0]["structure"])
    tensors = data["tensors"]
    a = next(k for k in tensors if ".lora_A." in k)
    b = a.replace(".lora_A.", ".lora_B.")
    if problem == "missing_b":
        del tensors[b]
    elif problem == "rank":
        tensors[b]["shape"][1] += 1
    elif problem == "unknown":
        tensors["evil.weight"] = tensors[a]
    elif problem == "collision":
        tensors[a.removeprefix("transformer.")] = tensors[a]
    elif problem == "geometry":
        tensors[a]["shape"][1] += 1
    elif problem == "dtype":
        tensors[a]["dtype"] = "U32"
    elif problem in {"alpha_shape", "orphan_alpha"}:
        tensors["absent.alpha"] = {"shape": [1] if problem == "alpha_shape" else [], "dtype": "F32"}
    else:
        metadata = json.loads(data["metadata"]["lora_adapter_metadata"])
        metadata[
            "transformer."
            + {"dora": "use_dora", "rslora": "use_rslora", "alpha_pattern": "alpha_pattern"}[
                problem
            ]
        ] = {"x": 5} if problem == "alpha_pattern" else True
        data["metadata"]["lora_adapter_metadata"] = json.dumps(metadata)
    with pytest.raises(ValueError):
        validate_structure(data, ORIGINALS[0]["config"])


@pytest.mark.parametrize(
    "problem", ["nan", "inf", "alpha_zero", "alpha_negative", "truncated", "extra", "symlink"]
)
def test_invalid_numeric_adapter_never_partially_changes_native_graph(tmp_path, problem):
    import torch
    from safetensors.torch import save_file

    model = tiny_transformer()
    tensors = arrays(model, alpha=True)
    if problem in {"nan", "inf"}:
        tensors[next(iter(tensors))][0, 0] = float(problem)
    if problem.startswith("alpha_"):
        tensors[next(k for k in tensors if k.endswith(".alpha"))] = torch.tensor(
            0.0 if problem == "alpha_zero" else -1.0
        )
    if problem == "extra":
        tensors["unrecognized.weight"] = torch.ones(2, 2)
    path = tmp_path / "adapter.safetensors"
    save_file(tensors, path)
    if problem == "truncated":
        path.write_bytes(path.read_bytes()[:-4])
    if problem == "symlink":
        real = tmp_path / "real.safetensors"
        path.rename(real)
        path.symlink_to(real)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises((ValueError, RuntimeError)):
        attach(model, path)
    assert set(before) == set(model.state_dict())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())


class LocalHub:
    """Two exact source repositories backed only by complete generated fixtures."""

    def __init__(self, base, originals):
        self.base, self.originals = base, originals
        self.downloads = []
        self.reads = []
        self.current_provider = "common"
        self._token = lambda: False
        self._api = SimpleNamespace(model_info=self.model_info)

    def root(self, repository, revision):
        if repository == NATURAL["base"]["repository"] and revision == NATURAL["base"]["revision"]:
            return self.base
        value = binding(repository, revision)
        assert value is not None and repository in self.originals
        return self.originals[repository]

    def model_info(self, repository, revision, **kwargs):
        root = self.root(repository, revision)
        files = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            data = path.read_bytes()
            digest = (
                hashlib.sha256(data).hexdigest()
                if path.suffix == ".safetensors"
                else hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            )
            files.append(
                SimpleNamespace(
                    rfilename=path.relative_to(root).as_posix(),
                    size=len(data),
                    blob_id=digest,
                    lfs=SimpleNamespace(sha256=digest) if path.suffix == ".safetensors" else None,
                )
            )
        return SimpleNamespace(
            sha=revision,
            config=json.loads((root / "model_index.json").read_text()) if root == self.base else {},
            siblings=files,
            pipeline_tag="image-to-image",
            library_name="diffusers",
            gated=False,
            private=False,
            card_data={},
        )

    def read_json(self, repository, revision, filename):
        self.reads.append((repository, revision, filename))
        return json.loads((self.root(repository, revision) / filename).read_text())

    def read_tensor_structure(self, repository, revision, filename):
        path = self.root(repository, revision) / filename
        with path.open("rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            data = json.loads(f.read(length))
        return {
            "metadata": data.get("__metadata__", {}),
            "tensors": {
                k: {"shape": v["shape"], "dtype": v["dtype"]}
                for k, v in data.items()
                if k != "__metadata__"
            },
        }

    def has_access(self, repository):
        return True

    def download(self, repository, filename, revision, cache_dir, progress=None):
        self.downloads.append((repository, revision, filename))
        target = (
            Path(cache_dir)
            / ("models--" + repository.replace("/", "--"))
            / "snapshots"
            / revision
            / filename
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.root(repository, revision) / filename, target)
        return str(target)


@pytest.fixture(autouse=True)
def available_test_disk(tmp_path, monkeypatch):
    # These local-archive tests exercise provenance and installation. Disk
    # admission has separate rejection tests and must not depend on host load.
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda _: usage._replace(free=100 * 1024**3))


@pytest.fixture(scope="module")
def local_hub(tmp_path_factory):
    from diffusers import QwenImage21Transformer2DModel
    from qwen_image_21_fixture import write_checkpoint
    from safetensors.torch import save_file

    root = tmp_path_factory.mktemp("lora-complete")
    base = write_checkpoint(root / "base")
    model = QwenImage21Transformer2DModel.from_pretrained(
        base / "transformer", local_files_only=True
    )
    originals = {}
    for value in (NATURAL, ANGLE, VIGGLE):
        path = root / value["model"]["repository"].split("/")[0]
        path.mkdir()
        save_file(arrays(model, fused=value == ANGLE), path / value["adapterFile"])
        if value == VIGGLE:
            (path / "scheduler").mkdir()
            config = json.loads((base / "scheduler/scheduler_config.json").read_text())
            (path / "scheduler/scheduler_config.json").write_text(
                json.dumps({**config, "shift_terminal": None})
            )
        originals[value["model"]["repository"]] = path
    return LocalHub(base, originals)


def model_ref(value=NATURAL):
    return {"provider": "huggingface", **value["model"]}


def store_for(tmp_path, hub):
    from oryn_engine.models import ModelStore

    store = ModelStore(
        tmp_path,
        hub=hub,
        runtime_available=True,
        runtime_planning_state_provider=lambda: ("balanced", 256 * 1024**3),
    )
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda pack: None if pack.provider else object()
    )
    return store


def test_discovery_installs_exact_closure_and_preserves_offline_identity(
    tmp_path, local_hub, monkeypatch
):
    from oryn_protocol import ProtocolSchemaStore

    from oryn_engine.models import ModelStoreError
    from oryn_engine.runners import _selected_model_view, _verify_installed_files

    hub = copy.copy(local_hub)
    hub.downloads = []
    store = store_for(tmp_path, hub)
    profile = store.inspect(model_ref())
    assert profile["executable"], profile
    assert ProtocolSchemaStore().is_valid("candidateExecutionProfile", profile)
    assert profile["model"] == model_ref()
    assert profile["runtimePack"]["trustedMetadata"]["composition"] == NATURAL
    names = {x["name"] for x in profile["selectedFiles"]}
    assert (
        adapter_name(NATURAL) in names
        and "transformer/diffusion_pytorch_model.safetensors" in names
    )
    assert profile["estimatedDiskBytes"] == sum(x["sizeBytes"] for x in profile["selectedFiles"])
    assert not hub.downloads
    subset = store.inspect({**model_ref(), "files": sorted(names - {adapter_name(NATURAL)})})
    assert not subset["executable"]
    store.install("install_lora", model_ref())
    assert (
        NATURAL["model"]["repository"],
        NATURAL["model"]["revision"],
        NATURAL["adapterFile"],
    ) in hub.downloads
    assert all(
        (repo, rev)
        in {
            (NATURAL["model"]["repository"], NATURAL["model"]["revision"]),
            (NATURAL["base"]["repository"], NATURAL["base"]["revision"]),
        }
        for repo, rev, _ in hub.downloads
    )
    record = store.installed_record(model_ref())
    _verify_installed_files(record)
    model_root = store.cache_root / "models--primary" / "snapshots" / NATURAL["model"]["revision"]
    with _selected_model_view(record, model_root) as view:
        assert {p.relative_to(view).as_posix() for p in view.rglob("*") if p.is_file()} == names
        assert (view / "model_index.json").read_bytes() == (
            hub.base / "model_index.json"
        ).read_bytes()
    monkeypatch.setattr(
        hub, "model_info", lambda *a, **k: pytest.fail("offline inspection accessed Hub")
    )
    replay = store_for(tmp_path, hub)
    assert replay.inspect(model_ref())["executable"]
    assert replay.inspect(model_ref())["runtimePack"]["trustedMetadata"]["composition"] == NATURAL
    with pytest.raises((ValueError, ModelStoreError)):
        validate_binding({**NATURAL, "base": {**NATURAL["base"], "revision": "main"}})


def test_snapshot_embeds_exact_dependency_and_replays_with_no_network(tmp_path, local_hub):
    from unittest.mock import Mock

    from oryn_engine.audit_metadata import MetadataSnapshot

    snap = MetadataSnapshot(
        tmp_path / "snapshot.json",
        NATURAL["model"]["repository"],
        NATURAL["model"]["revision"],
        client=local_hub,
    )
    client = CompositionHub(snap)
    info = client.model_info(
        **{"repo_id": NATURAL["model"]["repository"], "revision": NATURAL["model"]["revision"]}
    )
    config = client.read_json(
        NATURAL["model"]["repository"], NATURAL["model"]["revision"], "transformer/config.json"
    )
    assert info.sha == NATURAL["model"]["revision"]
    assert len(snap.data["dependencies"]) == 1
    frozen_digest = snap.digest
    bomb = Mock()
    bomb._api.model_info.side_effect = AssertionError("network forbidden")
    bomb.read_json.side_effect = AssertionError("network forbidden")
    replay = MetadataSnapshot(
        snap.path, snap.repository, snap.requested_revision, offline=True, client=bomb
    )
    wrapped = CompositionHub(replay)
    wrapped.model_info(snap.repository, snap.requested_revision)
    assert (
        wrapped.read_json(snap.repository, snap.requested_revision, "transformer/config.json")
        == config
    )
    assert replay.digest == frozen_digest
    with pytest.raises(ValueError):
        snap.dependency_snapshot({**NATURAL["base"], "revision": "main"})
    with pytest.raises(ValueError):
        snap.model_info(NATURAL["base"]["repository"], NATURAL["base"]["revision"])
    with pytest.raises(RuntimeError, match="WEIGHT_DOWNLOAD_FORBIDDEN"):
        wrapped.download(
            snap.repository, adapter_name(NATURAL), snap.requested_revision, str(tmp_path)
        )


@pytest.mark.parametrize("value", [NATURAL, ANGLE])
@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_complete_offline_pipeline_adapter_changes_generation(
    local_hub, tmp_path, monkeypatch, value, backend
):
    import numpy as np
    import torch
    from PIL import Image

    from oryn_engine.runtime_packs import diffusers_worker

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Apple Silicon")
    root = tmp_path / "view"
    shutil.copytree(local_hub.base, root)
    (root / "oryn_adapter").mkdir()
    shutil.copyfile(
        local_hub.originals[value["model"]["repository"]] / value["adapterFile"],
        root / adapter_name(value),
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    diffusers_worker._validate_model(root)
    pipe, selected = diffusers_worker._pipeline(
        root, "image-editing-v1", torch, backend_request=backend, composition=value
    )
    assert selected == backend
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64), (150, 30, 100)).save(source)
    request = {
        "prompt": "Make it blue",
        "negativePrompt": "",
        "width": 64,
        "height": 64,
        "steps": 2,
        "guidanceScale": 1.0,
        "seed": 7,
    }
    enabled, disabled = tmp_path / "enabled.png", tmp_path / "disabled.png"
    diffusers_worker._generate_image(
        pipe, "image-editing-v1", request, source, enabled, None, torch
    )
    for module in pipe._oryn_lora_modules:
        module.enabled = False
    diffusers_worker._generate_image(
        pipe, "image-editing-v1", request, source, disabled, None, torch
    )
    assert Image.open(enabled).size == (64, 64)
    assert np.any(np.asarray(Image.open(enabled)) != np.asarray(Image.open(disabled)))


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_turbo_composition_crosses_offline_worker_boundary(local_hub, tmp_path, backend):
    import os
    import subprocess
    import sys

    import torch
    from PIL import Image

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Apple Silicon")
    root = tmp_path / "view"
    shutil.copytree(local_hub.base, root)
    (root / "oryn_adapter").mkdir()
    primary = local_hub.originals[VIGGLE["model"]["repository"]]
    shutil.copyfile(primary / VIGGLE["adapterFile"], root / adapter_name(VIGGLE))
    shutil.copyfile(
        primary / "scheduler/scheduler_config.json", root / "scheduler/scheduler_config.json"
    )
    output = tmp_path / "image.png"
    request = {
        "abi": "1",
        "packId": "oryn.diffusers-pytorch",
        "version": "0.6.0",
        "command": "generate",
        "payload": {
            "modelRoot": str(root),
            "outputPath": str(output),
            "previewPath": None,
            "inputPath": None,
            "taskContract": "text-to-image-v1",
            "backend": backend,
            "composition": VIGGLE,
            "input": {
                "prompt": "A blue circle",
                "negativePrompt": "",
                "width": 64,
                "height": 64,
                "steps": 6,
                "guidanceScale": 1.0,
                "seed": 7,
            },
        },
    }
    run = subprocess.run(
        [sys.executable, "-m", "oryn_engine.runtime_packs.diffusers_worker"],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        timeout=90,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "ORYN_PACK_NETWORK_POLICY": "deny",
            "OMP_NUM_THREADS": "1",
        },
    )
    assert run.returncode == 0, run.stderr
    response = json.loads(run.stdout.splitlines()[-1])
    assert response.get("ok") is True, (response, run.stderr)
    assert response["result"]["backend"] == backend
    assert response["result"]["relativeOutput"] == "image.png"
    assert Image.open(output).mode == "RGBA" and Image.open(output).size == (64, 64)


def test_dependency_removal_retains_shared_base_and_cleans_final_owner(tmp_path, local_hub):
    store = store_for(tmp_path, local_hub)
    store.install("install_one", model_ref(NATURAL))
    store.install("install_two", model_ref(ANGLE))
    first, second = (
        store.installed_record(model_ref(NATURAL)),
        store.installed_record(model_ref(ANGLE)),
    )
    base_paths = {
        x["cache_path"] for x in first["files"] if not x["name"].startswith("oryn_adapter/")
    }
    assert store.remove_preview(first["installationId"])["sharedBytesRetained"] > 0
    store.complete_remove(first["installationId"])
    assert all((store.cache_root / p).is_file() for p in base_paths)
    store.complete_remove(second["installationId"])
    assert not any((store.cache_root / p).exists() for p in base_paths)


@pytest.mark.parametrize("precision", ["float16", "bfloat16"])
@pytest.mark.parametrize("temporal", [1, 2, 4])
def test_mps_native_average_shortcut_matches_cpu_without_nonfinite_pixels(precision, temporal):
    import torch
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import QwenImage21AvgDown3D

    from oryn_engine.qwen_image_21 import stable_mps_downsampling

    if not torch.backends.mps.is_available():
        pytest.skip("requires Apple Silicon")
    torch.manual_seed(3)
    original = QwenImage21AvgDown3D(4, 4, temporal, 2)
    x = torch.randn(1, 4, 1, 256, 256).to(getattr(torch, precision))
    expected = original(x)
    stable_mps_downsampling(original)
    result = original(x.to("mps")).cpu()
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("component", ["transformer", "vae", "text_encoder"])
@pytest.mark.parametrize("problem", ["missing", "nonfinite"])
def test_complete_base_cannot_silently_initialize_missing_arrays(
    local_hub, tmp_path, component, problem
):
    import torch
    from safetensors.torch import load_file, save_file

    from oryn_engine.runtime_packs.diffusers_worker import _pipeline

    root = tmp_path / "view"
    shutil.copytree(local_hub.base, root)
    (root / "oryn_adapter").mkdir()
    shutil.copyfile(
        local_hub.originals[NATURAL["model"]["repository"]] / NATURAL["adapterFile"],
        root / adapter_name(NATURAL),
    )
    path = next((root / component).glob("*.safetensors"))
    tensors = load_file(path)
    key = next(k for k, v in tensors.items() if v.is_floating_point())
    if problem == "missing":
        del tensors[key]
    else:
        tensors[key].view(-1)[0] = float("nan")
    save_file(tensors, path)
    with pytest.raises((ValueError, RuntimeError), match="missing|arrays|weight|meta"):
        _pipeline(root, "image-editing-v1", torch, backend_request="cpu", composition=NATURAL)


def test_original_scheduler_override_and_turbo_profile_survive_install(tmp_path, local_hub):
    store = store_for(tmp_path, local_hub)
    profile = store.inspect(model_ref(VIGGLE))
    assert profile["generationProfile"]["parameters"]["steps"]["default"] == 6
    assert not profile["generationProfile"]["parameters"]["steps"]["editable"]
    store.install("install_turbo", model_ref(VIGGLE))
    record = store.installed_record(model_ref(VIGGLE))
    item = next(x for x in record["files"] if x["name"] == "scheduler/scheduler_config.json")
    assert "Viggle" in item["cache_path"]
    assert record["generationProfile"] == profile["generationProfile"]


@pytest.mark.parametrize(
    "filename", ["model_index.json", "transformer/diffusion_pytorch_model.safetensors"]
)
def test_dependency_bytes_must_match_their_original_publisher_hash(tmp_path, local_hub, filename):
    from oryn_engine.models import ModelStoreError

    hub = copy.copy(local_hub)
    original = hub.download

    def altered(repository, name, revision, cache_dir, progress=None):
        path = Path(original(repository, name, revision, cache_dir, progress))
        if name == filename:
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(data)
        return str(path)

    hub.download = altered
    store = store_for(tmp_path, hub)
    with pytest.raises(ModelStoreError) as error:
        store.install("install_corrupt", model_ref())
    assert error.value.code == "MODEL_INTEGRITY_FAILED"
    assert not store.inventory()["models"]


def test_audit_negotiates_editing_from_the_pinned_base_and_records_sources(tmp_path, local_hub):
    from oryn_engine.audit_metadata import MetadataSnapshot
    from oryn_engine.metadata_audit import inspect_row

    snap = MetadataSnapshot(
        tmp_path / "snapshot.json",
        NATURAL["model"]["repository"],
        NATURAL["model"]["revision"],
        client=local_hub,
    )
    result = inspect_row(
        {"row": 1, "input": NATURAL["model"]["repository"], "task": "image-to-image"}, snap
    )
    assert result["status"] == "metadata-compatible", result
    assert result["requirements"]["contract"]["id"] == "image-editing-v1"
    assert result["composition"] == NATURAL
    assert {x["source"]["repository"] for x in result["selectedFiles"]} == {
        NATURAL["base"]["repository"],
        NATURAL["model"]["repository"],
    }
    assert all(x.get("blobId") for x in result["selectedFiles"])
    assert result["executionVerified"] is False


@pytest.fixture(scope="module")
def lightning_checkpoint(tmp_path_factory, local_hub):
    from diffusers import QwenImageTransformer2DModel
    from qwen_image_edit_fixture import write_checkpoint
    from safetensors.torch import save_file

    root = write_checkpoint(
        tmp_path_factory.mktemp("lightning-complete"), local_hub.base / "processor"
    )
    model = QwenImageTransformer2DModel.from_pretrained(root / "transformer", local_files_only=True)
    value = binding(ORIGINALS[-1]["repository"], ORIGINALS[-1]["revision"])
    (root / "oryn_adapter").mkdir()
    save_file(arrays(model, alpha=True), root / adapter_name(value))
    return root, value


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_original_alpha_format_runs_complete_edit_2511_graph(
    lightning_checkpoint, tmp_path, monkeypatch, backend
):
    import numpy as np
    import torch
    from PIL import Image

    from oryn_engine.runtime_packs.diffusers_worker import (
        _generate_image,
        _pipeline,
        _validate_model,
    )

    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Apple Silicon")
    root, composition = lightning_checkpoint
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _validate_model(root)
    pipe, _ = _pipeline(
        root, "image-editing-v1", torch, backend_request=backend, composition=composition
    )
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64), (150, 30, 100)).save(source)
    value = {
        "prompt": "Make the image blue",
        "negativePrompt": "",
        "width": 64,
        "height": 64,
        "steps": 4,
        "guidanceScale": 1.0,
        "seed": 7,
    }
    enabled, disabled = tmp_path / "enabled.png", tmp_path / "disabled.png"
    with pytest.raises(ValueError, match="requires steps"):
        _generate_image(
            pipe, "image-editing-v1", {**value, "steps": 3}, source, enabled, None, torch
        )
    _generate_image(pipe, "image-editing-v1", value, source, enabled, None, torch)
    for module in pipe._oryn_lora_modules:
        module.enabled = False
    _generate_image(pipe, "image-editing-v1", value, source, disabled, None, torch)
    assert Image.open(enabled).size == (64, 64)
    assert np.any(np.asarray(Image.open(enabled)) != np.asarray(Image.open(disabled)))
