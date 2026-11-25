"""
Backend registry: MODEL_BACKEND picks the backend, a wrong one stops the start with a clear message.
"""

import re
import tomllib

from pathlib import Path

import pytest

from loguru import logger

from app.core.backends import registry
from app.core.backends.fake import FakeBackend
from app.core.backends.registry import BACKENDS, BackendSpec, build_backend
from app.core.config import Settings
from app.core.exceptions import BackendError
from scripts import fetch_weights

EXTRA_BACKENDS = sorted(name for name, spec in BACKENDS.items() if spec.extra)
PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def make_settings(**values) -> Settings:
    # No .env: the test sees the defaults and its own values only
    return Settings(_env_file=None, **values)


def test_registry_selects_fake():
    records = []
    sink = logger.add(lambda message: records.append(message.record), level="WARNING")
    try:
        backend = build_backend(make_settings(MODEL_BACKEND="fake", MODEL_NAME=None))
    finally:
        logger.remove(sink)

    assert isinstance(backend, FakeBackend)
    assert (backend.name, backend.model, backend.device) == ("fake", "fake", "cpu")
    # Synthetic answers must not pass for a measurement unnoticed
    assert any(record["level"].name == "WARNING" for record in records)


def test_registry_unknown_backend_fails():
    message = f"Unknown backend 'yolo9000', expected one of: {', '.join(sorted(BACKENDS))}"
    with pytest.raises(BackendError, match=re.escape(message)):
        build_backend(make_settings(MODEL_BACKEND="yolo9000"))

    message = "Unknown model 'large' of backend 'fake', expected one of: fake"
    with pytest.raises(BackendError, match=re.escape(message)):
        build_backend(make_settings(MODEL_BACKEND="fake", MODEL_NAME="large"))


@pytest.mark.parametrize("name", EXTRA_BACKENDS)
def test_registry_missing_extra_fails(name, monkeypatch):
    # None of the extra libraries is installed, as in the fast environment
    monkeypatch.setattr(registry, "find_spec", lambda package: None)

    message = f"Backend '{name}' requires the '{BACKENDS[name].extra}' extra"
    with pytest.raises(BackendError, match=re.escape(message)):
        build_backend(make_settings(MODEL_BACKEND=name))


def test_registry_missing_library_on_import_fails(tmp_path, monkeypatch):
    # The backend module imports a library of its extra, and the library is not installed
    (tmp_path / "alv_probe_backend.py").write_text("import alv_missing_library\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(BACKENDS, "probe", BackendSpec("alv_probe_backend:ProbeBackend", "probe"))

    with pytest.raises(BackendError, match=re.escape("Backend 'probe' requires the 'probe' extra")):
        build_backend(make_settings(MODEL_BACKEND="probe"))


def test_registry_lists_asr_backends():
    # ASR-3: the backends of the service, the fake one among them
    assert sorted(BACKENDS) == ["fake", "gigaam", "gigaam_onnx", "tone", "vosk", "whisper"]
    assert EXTRA_BACKENDS == ["gigaam", "gigaam_onnx", "vosk", "whisper"]


def test_gigaam_onnx_requires_extra(monkeypatch):
    # AC-44: onnx-asr is not installed, as in the fast environment
    real_find_spec = registry.find_spec
    monkeypatch.setattr(registry, "find_spec", lambda package: None if package == "onnx_asr" else real_find_spec(package))

    with pytest.raises(BackendError, match=re.escape("Backend 'gigaam_onnx' requires the 'gigaam_onnx' extra")):
        build_backend(make_settings(MODEL_BACKEND="gigaam_onnx"))


@pytest.mark.parametrize("name", EXTRA_BACKENDS)
def test_registry_extra_is_declared(name):
    # The extra the error message names must exist: `uv sync --extra <name>` has to work
    with open(PYPROJECT, "rb") as source:
        extras = tomllib.load(source)["project"]["optional-dependencies"]

    spec = BACKENDS[name]
    assert spec.extra in extras
    assert spec.packages


@pytest.mark.parametrize("name", EXTRA_BACKENDS)
def test_registry_backend_module_exists(name):
    # A typo in the path would pass for a missing extra
    module_name, class_name = BACKENDS[name].path.split(":")
    source = Path(__file__).resolve().parents[1] / Path(*module_name.split(".")).with_suffix(".py")
    assert f"class {class_name}(Backend)" in source.read_text(encoding="utf-8")


def test_registry_tone_needs_no_extra():
    # T-one runs on the core onnxruntime: the model is checked before any weights are loaded
    assert BACKENDS["tone"].extra is None

    message = "Unknown model 'large' of backend 'tone', expected one of: t-one"
    with pytest.raises(BackendError, match=re.escape(message)):
        build_backend(make_settings(MODEL_BACKEND="tone", MODEL_NAME="large"))


def test_fetch_weights_fake_and_unknown():
    assert fetch_weights.fetch(make_settings(MODEL_BACKEND="fake", MODEL_NAME=None)) == "fetch_weights: fake/fake -> no weights"

    with pytest.raises(BackendError, match=re.escape("Unknown backend 'yolo9000'")):
        fetch_weights.fetch(make_settings(MODEL_BACKEND="yolo9000"))
