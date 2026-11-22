import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file, save_model

from oryn_engine import depth3_tasks, shared_tasks
from oryn_engine.generated_capabilities import normalized_transformers_config

FIXTURES = Path(__file__).parent / "fixtures/depth3"


def original(name="DA3-SMALL"):
    return json.loads((FIXTURES / (name + ".json")).read_text())


def compact(kind):
    config = original("DA3NESTED-GIANT-LARGE-1.1" if kind == "nested" else "DA3-SMALL")
    if kind == "mono":
        config = original("DA3MONO-LARGE")
    branches = (
        [config["config"]]
        if kind != "nested"
        else list(config["config"][name] for name in ("anyview", "metric"))
    )
    for branch in branches:
        dual = "cam_dec" in branch
        branch["net"].update(
            name="vits",
            out_layers=[5, 7, 9, 11],
            alt_start=4 if dual else -1,
            qknorm_start=4 if dual else -1,
            rope_start=4 if dual else -1,
        )
        for head in ("head", "gs_head"):
            if head in branch:
                branch[head].update(
                    dim_in=768 if dual else 384, features=16, out_channels=[16, 16, 16, 16]
                )
        if dual:
            branch["cam_dec"]["dim_in"] = 768
            branch["cam_enc"]["dim_out"] = 384
    return config


def reference_graph(config):
    # Construct the upstream numerical graph independently of the adapter factory.
    from depth_anything_3.model.cam_dec import CameraDec
    from depth_anything_3.model.cam_enc import CameraEnc
    from depth_anything_3.model.da3 import DepthAnything3Net, NestedDepthAnything3Net
    from depth_anything_3.model.dinov2.dinov2 import DinoV2
    from depth_anything_3.model.dpt import DPT
    from depth_anything_3.model.dualdpt import DualDPT
    from depth_anything_3.model.gsdpt import GSDPT

    def single(branch):
        dual = "cam_dec" in branch
        model = DepthAnything3Net(
            DinoV2(
                name="vits",
                out_layers=[5, 7, 9, 11],
                alt_start=4 if dual else -1,
                qknorm_start=4 if dual else -1,
                rope_start=4 if dual else -1,
                cat_token=dual,
            ),
            (DualDPT if dual else DPT)(
                dim_in=768 if dual else 384,
                features=16,
                out_channels=[16] * 4,
                output_dim=2 if dual else 1,
            ),
            CameraDec(dim_in=768) if dual else None,
            CameraEnc(dim_out=384) if dual else None,
        )
        if "gs_head" in branch:
            model.gs_head = GSDPT(dim_in=768, features=16, out_channels=[16] * 4, output_dim=38)
        return model

    if "anyview" not in config["config"]:
        return single(config["config"]).eval()
    model = NestedDepthAnything3Net.__new__(NestedDepthAnything3Net)
    torch.nn.Module.__init__(model)
    model.da3 = single(config["config"]["anyview"])
    model.da3_metric = single(config["config"]["metric"])
    return model.eval()


@pytest.fixture(scope="module", params=["dual", "mono", "nested"])
def native_fixture(request, tmp_path_factory):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(61628)
    root = tmp_path_factory.mktemp("depth3-" + request.param)
    config = compact(request.param)
    reference = reference_graph(config)
    wrapper = torch.nn.Module()
    wrapper.model = reference
    save_model(wrapper, root / "model.safetensors")
    (root / "config.json").write_text(json.dumps(config))
    yield root, normalized_transformers_config(config, "depth-estimation"), reference
    torch.set_num_threads(old)


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_complete_native_graph_parity(native_fixture, backend):
    if backend == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal unavailable")
    root, config, reference = native_fixture
    loaded, processor, selected = shared_tasks.load_task(root, "depth-estimation", backend)
    assert processor is None and selected == config
    for name, tensor in reference.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name].cpu(), tensor, rtol=0, atol=0)
    value = torch.linspace(-1, 1, 3 * 56 * 56).reshape(1, 1, 3, 56, 56)
    with torch.inference_mode():
        expected = reference(value)
        observed = loaded(value.to(backend))
    for name in ("depth", "depth_conf", "intrinsics", "extrinsics", "sky"):
        if name in expected:
            torch.testing.assert_close(observed[name].cpu(), expected[name], rtol=3e-3, atol=4e-4)


def test_full_graph_headers_require_auxiliary_and_metric_branches(native_fixture):
    root, config, reference = native_fixture
    shapes = {
        key: tuple(tensor.shape) for key, tensor in load_file(root / "model.safetensors").items()
    }
    depth3_tasks.validate_weight_shapes(config, shapes)
    prefixes = ["backbone", "head", "cam_dec", "cam_enc", "gs_head", "da3.", "da3_metric."]
    for prefix in prefixes:
        keys = [key for key in shapes if prefix in key]
        if keys:
            broken = {key: shape for key, shape in shapes.items() if key not in keys}
            with pytest.raises(ValueError, match="INCOMPLETE"):
                depth3_tasks.validate_weight_shapes(config, broken)
    for bad in ({**shapes, "ignored.weight": (1,)}, {key: (1,) for key in shapes}):
        with pytest.raises(ValueError, match="GEOMETRY"):
            depth3_tasks.validate_weight_shapes(config, bad)
    # Header metadata cannot invent sharing for a missing native parameter.
    assert set(reference.state_dict())


@pytest.mark.parametrize("name", [path.stem for path in sorted(FIXTURES.glob("*.json"))])
def test_original_config_is_metadata_only(name):
    raw = original(name)
    selected = normalized_transformers_config(raw, "depth-estimation")
    depth3_tasks.validate_config(selected)
    assert selected["model_type"] == "depth_anything_3"
    assert normalized_transformers_config(raw, "image-classification") == raw


@pytest.mark.parametrize(
    "path,value",
    [
        ("config.__object__.path", "repository.remote"),
        ("config.net.__object__.name", "Injected"),
        ("config.net.name", "unknown"),
        ("config.net.out_layers", [0, 1, 2, 99]),
        ("config.net.out_layers", [5, 7, 7, 11]),
        ("config.net.alt_start", True),
        ("config.net.qknorm_start", 3),
        ("config.net.rope_start", False),
        ("config.net.cat_token", False),
        ("config.head.dim_in", 384),
        ("config.head.output_dim", 1),
        ("config.head.out_channels", [16, 16, 16, 1000000]),
        ("config.head.activation", "relu"),
        ("config.cam_dec.dim_in", 384),
        ("config.cam_enc.dim_out", 768),
        ("auto_map", {"AutoModel": "remote.Model"}),
        ("extra", True),
    ],
)
def test_closed_configuration(path, value):
    config = normalized_transformers_config(original(), "depth-estimation")
    cursor = config
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = value
    with pytest.raises(ValueError):
        depth3_tasks.validate_config(config)


def test_metadata_normalization_never_imports_native_runtime():
    code = (
        "import sys,json; "
        "from oryn_engine.generated_capabilities import normalized_transformers_config; "
        "c=normalized_transformers_config(json.load(sys.stdin),'depth-estimation'); "
        "assert c['model_type']=='depth_anything_3'; "
        "assert not {'torch','depth_anything_3','cv2'} & sys.modules.keys()"
    )
    subprocess.run(
        [sys.executable, "-c", code], input=json.dumps(original()), text=True, check=True
    )


def test_native_dynamic_construction_is_disabled():
    from depth_anything_3.cfg import create_object

    with pytest.raises(ValueError, match="DYNAMIC_CONSTRUCTION_DISABLED"):
        create_object({"__object__": {"path": "os", "name": "system"}})


@pytest.mark.parametrize("size", [(40, 20), (70, 11), (1007, 321), (1536, 4096)])
def test_pixels_match_native_resize_and_normalization(size):
    raw = np.random.default_rng(61).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    # Independent numerical reference for the pinned upper_bound_resize path.
    scale = 504 / max(size)
    target = tuple(max(1, int(round(axis * scale))) for axis in size)
    expected = cv2.resize(
        raw, target, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    )
    target2 = tuple(max(14, ((axis + 7) // 14) * 14) for axis in target)
    if target2 != target:
        expected = cv2.resize(
            expected,
            target2,
            interpolation=(
                cv2.INTER_CUBIC
                if target2[0] > target[0] or target2[1] > target[1]
                else cv2.INTER_AREA
            ),
        )
    expected = (
        expected.astype(np.float32) / 255 - np.array([0.485, 0.456, 0.406], dtype=np.float32)
    ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    observed = depth3_tasks.pixels(Image.fromarray(raw))
    np.testing.assert_allclose(observed[0, 0].permute(1, 2, 0).numpy(), expected, atol=2e-7)


def test_degenerate_input_and_invalid_output_fail_closed():
    with pytest.raises(ValueError, match="GEOMETRY"):
        depth3_tasks.pixels(Image.new("RGB", (4096, 1)))
    for tensor in (
        torch.tensor([[[[float("nan")]]]]),
        torch.tensor([[[[-1.0]]]]),
        torch.ones(1, 2, 1, 1),
    ):
        with pytest.raises(ValueError, match="OUTPUT_INVALID"):
            depth3_tasks.execute(
                lambda *args, tensor=tensor, **kwargs: SimpleNamespace(depth=tensor),
                Image.new("RGB", (40, 10)),
                "cpu",
            )


def test_shared_execution_returns_depth_contract(native_fixture, tmp_path):
    root, _, reference = native_fixture
    path = tmp_path / "input.png"
    image = Image.new("RGB", (90, 10), color=(50, 80, 100))
    image.save(path)
    events = []
    result = shared_tasks.execute_task(
        root,
        "depth-estimation",
        {"imageArtifactId": "artifact_aaaaaaaaaaaaaaaaaaaaaaaa"},
        "cpu",
        path,
        None,
        lambda *args: events.append(args),
    )
    with torch.inference_mode():
        depth = reference(depth3_tasks.pixels(image)).depth[0, 0]
    expected = {
        "width": 504,
        "height": 56,
        "minimum": float(depth.min()),
        "maximum": float(depth.max()),
        "mean": float(depth.mean()),
    }
    assert result == pytest.approx(expected)
    assert events


@pytest.mark.parametrize("damage", ["integer", "nan", "alias-conflict", "duplicate-shard"])
def test_loader_rejects_invalid_tensor_values(native_fixture, tmp_path, damage):
    root, config, _ = native_fixture
    weights = load_file(root / "model.safetensors")
    key = next(iter(weights))
    if damage == "integer":
        weights[key] = weights[key].long()
    elif damage == "nan":
        weights[key].reshape(-1)[0] = float("nan")
    elif damage == "alias-conflict":
        _, names, aliases, _ = depth3_tasks._weights(
            config, {k: tuple(v.shape) for k, v in weights.items()}
        )
        if not aliases:
            assert config["config"]["net"]["cat_token"] is False
            return
        alias, source = next(iter(aliases.items()))
        weights["model." + alias] = weights[names[source]] + 1
    else:
        save_file({key: weights[key]}, tmp_path / "extra.safetensors")
    save_file(weights, tmp_path / "model.safetensors")
    with pytest.raises(ValueError):
        depth3_tasks.load(tmp_path, config)


@pytest.mark.parametrize("damage", [None, "missing-weights", "remote-code", "wrong-task"])
def test_metadata_discovery_uses_shared_depth_route(tmp_path, damage):
    from test_models import FakeHub

    from oryn_engine.models import ModelStore, ModelStoreError
    from oryn_engine.runtime_packs.execution import PackRuntime

    config = original()
    if damage == "remote-code":
        config["auto_map"] = {"AutoModel": "repository.Model"}
    files = [("config.json", 100, "b" * 64)]
    if damage != "missing-weights":
        files.append(("model.safetensors", 100, "c" * 64))

    class Hub(FakeHub):
        def read_json(self, repository, revision, filename):
            if filename == "config.json":
                return config
            raise OSError(filename)

    hub = Hub(
        config=config,
        files=files,
        pipeline_tag=("image-classification" if damage == "wrong-task" else "depth-estimation"),
    )
    store = ModelStore(tmp_path, hub=hub, runtime_available=True)
    store._pack_runtime_locator = SimpleNamespace(
        resolve=lambda _: PackRuntime(Path("/python"), None)
    )
    request = {"provider": "huggingface", "repository": "test/depth3", "revision": "main"}
    if damage == "remote-code":
        with pytest.raises(ModelStoreError, match="metadata"):
            store.inspect(request)
        return
    result = store.inspect(request)
    assert not hub.downloaded
    if damage:
        assert not result["executable"]
    else:
        assert result["executable"], result
        assert result["runtimePack"]["taskContract"] == "depth-estimation-v1"
        assert result["runtimePack"]["runnerId"] == "transformers.shared-tasks"
