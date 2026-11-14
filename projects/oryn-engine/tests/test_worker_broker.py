from pathlib import Path
from types import SimpleNamespace

import pytest

from oryn_engine.runtime_packs.broker import WorkerBroker
from oryn_engine.runtime_packs.diffusers_worker import WorkerFailure, _generate_image


def broker_request(tmp_path):
    broker = object.__new__(WorkerBroker)
    broker.token = "private-launch-token"
    broker.roots = {key: (tmp_path,) for key in ("models", "inputs", "outputs")}
    request = {
        "token": broker.token,
        "packId": "oryn.mlx-lm",
        "version": "0.1.0",
        "command": "generate",
        "payload": {"modelRoot": str(tmp_path)},
        "timeoutSeconds": 10,
    }
    return broker, request


@pytest.mark.parametrize(
    "change",
    [
        {"token": "untrusted"},
        {"command": "install"},
        {"arguments": ["/bin/sh"]},
        {"timeoutSeconds": float("nan")},
        {"timeoutSeconds": 100000},
    ],
)
def test_broker_rejects_untrusted_authority_commands_and_limits(tmp_path, change):
    broker, request = broker_request(tmp_path)
    with pytest.raises(ValueError):
        broker._validate(request | change)


def test_broker_resolves_symlinks_and_scopes_each_data_path(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    escaped = tmp_path / "private"
    escaped.mkdir()
    (allowed / "escape").symlink_to(escaped)
    broker, request = broker_request(allowed)
    assert broker._validate(request) == request
    for field in ("modelRoot", "inputPath", "outputPath", "previewPath"):
        for value in (str(escaped), str(allowed / "escape"), "/etc/passwd", "relative"):
            changed = request | {"payload": request["payload"] | {field: value}}
            with pytest.raises(ValueError):
                broker._validate(changed)


def test_filtered_diffusion_result_is_not_published_as_a_successful_black_image(tmp_path):
    class Pipe:
        def __call__(self, **_options):
            return SimpleNamespace(nsfw_content_detected=[True], images=[])

    torch = SimpleNamespace(
        Generator=lambda **_kwargs: SimpleNamespace(manual_seed=lambda _seed: None)
    )
    value = {
        "seed": 7,
        "prompt": "a bird",
        "negativePrompt": "",
        "width": 256,
        "height": 256,
        "steps": 12,
        "guidanceScale": 5,
    }
    output = Path(tmp_path) / "image.png"
    with pytest.raises(WorkerFailure) as error:
        _generate_image(Pipe(), "text-to-image-v1", value, None, output, None, torch)
    assert error.value.code == "MODEL_OUTPUT_FILTERED"
    assert not output.exists()
