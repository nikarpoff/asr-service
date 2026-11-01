"""
Backend registry: MODEL_BACKEND picks the backend, a wrong one stops the start with a clear message.
"""

import re

import pytest

from loguru import logger

from app.core.backends import registry
from app.core.backends.fake import FakeBackend
from app.core.backends.registry import BACKENDS, BackendSpec, build_backend
from app.core.config import Settings
from app.core.exceptions import BackendError

EXTRA_BACKENDS = sorted(name for name, spec in BACKENDS.items() if spec.extra)


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
