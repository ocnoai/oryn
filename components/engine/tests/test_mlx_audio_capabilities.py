"""Build-time family discovery reads bundled source without importing its code."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def builder():
    path = Path(__file__).parents[1] / "scripts/mlx_audio_capabilities.py"
    spec = importlib.util.spec_from_file_location("mlx_audio_capabilities_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def test_catalog_extracts_direct_alias_inheritance_and_wrapper_without_importing(tmp_path):
    source(tmp_path, "tts/utils.py", 'MODEL_REMAPPING = {"renamed": "direct"}\n')
    source(tmp_path, "stt/utils.py", 'MODEL_REMAPPING = {"recognizer": "wrapper"}\n')
    source(
        tmp_path,
        "tts/models/direct/__init__.py",
        "raise RuntimeError('never import')\nfrom .model import Engine as Model\n",
    )
    source(
        tmp_path,
        "tts/models/direct/model.py",
        "class Engine:\n"
        "    def generate(self, text, voice='native-voice', speed=1.0, **kwargs): pass\n",
    )
    source(
        tmp_path,
        "tts/models/child/__init__.py",
        "from mlx_audio.tts.models.direct.model import Engine\nclass Model(Engine): pass\n",
    )
    source(tmp_path, "stt/models/wrapper/__init__.py", "from .model import Model\n")
    source(
        tmp_path,
        "stt/models/wrapper/model.py",
        """class Aligner:
    def generate(self, audio, text): pass
class Recognizer:
    def generate(self, audio, language=None): pass
class Model:
    def __init__(self, config):
        self._model = Aligner(config) if config.align else Recognizer(config)
    def __getattr__(self, name):
        return getattr(self._model, name)
""",
    )
    catalog = builder().build_catalog(tmp_path, "1.2.3")
    tts = catalog["models"]["text-to-speech"]
    assert set(tts) == {"direct", "renamed", "child"}
    assert tts["direct"]["defaults"]["voice"] == "native-voice"
    assert tts["child"]["parameters"] == ["text", "voice", "speed"]
    stt = catalog["models"]["automatic-speech-recognition"]
    assert set(stt) == {"wrapper", "recognizer"}
    assert stt["recognizer"]["parameters"] == ["audio", "language"]
    assert catalog["version"] == "1.2.3"
    assert set(catalog["sources"]) == {str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*.py")}


def test_catalog_excludes_ambiguous_wrappers_and_other_contracts(tmp_path):
    source(tmp_path, "tts/utils.py", "MODEL_REMAPPING = {}\n")
    source(tmp_path, "stt/utils.py", 'MODEL_REMAPPING = {"music_alias": "moss_music"}\n')
    source(
        tmp_path,
        "stt/models/moss_music/__init__.py",
        "class Model:\n    def generate(self, audio, prompt=None): pass\n",
    )
    source(
        tmp_path,
        "tts/models/llama/__init__.py",
        "class Model:\n    def generate(self, text): pass\n",
    )
    source(
        tmp_path,
        "stt/models/aligner/__init__.py",
        "class Model:\n    def generate(self, audio, text): pass\n",
    )
    source(
        tmp_path,
        "stt/models/wrapper/__init__.py",
        """class First:
    def generate(self, audio): pass
class Second:
    def generate(self, audio, language=None): pass
class Model:
    def __init__(self, config):
        self._model = First(config) if config.first else Second(config)
    def __getattr__(self, name):
        return getattr(self._model, name)
""",
    )
    catalog = builder().build_catalog(tmp_path, "1.2.3")
    assert catalog["models"] == {"text-to-speech": {}, "automatic-speech-recognition": {}}


def test_catalog_binds_reviewed_companion_data_to_families_and_aliases(tmp_path):
    (tmp_path / "stt/models").mkdir(parents=True)
    source(tmp_path, "tts/utils.py", 'MODEL_REMAPPING = {"kitten": "kitten_tts"}\n')
    source(tmp_path, "stt/utils.py", "MODEL_REMAPPING = {}\n")
    for family in ("fish_drama3", "kitten_tts", "soprano"):
        source(
            tmp_path,
            f"tts/models/{family}/__init__.py",
            "class Model:\n    def generate(self, text): pass\n",
        )
    models = builder().build_catalog(tmp_path, "1.2.3")["models"]["text-to-speech"]
    assert models["fish_drama3"]["requiredFiles"] == [
        "codec.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    assert models["kitten"]["requiredFiles"] == ["voices.npz"]
    assert models["soprano"]["requiredFiles"] == ["tokenizer.json", "tokenizer_config.json"]
