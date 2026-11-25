"""
The gigaam_onnx backend without onnx-asr and the network: the library and the downloads are fakes.

The real weights and CUDA are checked on the model machine (tests/test_model_machine.py).
"""

import importlib
import re
import sys
import types

from pathlib import Path
from typing import List

import numpy as np
import pytest

from app.core.backends import registry
from app.core.backends.registry import build_backend
from app.core.config import Settings, GIGAAM_ONNX_SETTINGS as CFG
from app.core.exceptions import BackendError
from scripts import fetch_weights

MODULE = "app.core.backends.gigaam_onnx"


class FakeSession:
    def __init__(self, providers: List[str]) -> None:
        self.providers = providers

    def get_providers(self) -> List[str]:
        return list(self.providers)


class FakeModel:
    """What onnx_asr.load_model gives: an adapter with the model sessions inside"""

    def __init__(self, name: str, path: str, providers: List[str], taken: List[str]) -> None:
        self.name, self.path, self.providers = name, path, providers
        preprocessor = types.SimpleNamespace(_preprocessor=FakeSession(taken))
        self.asr = types.SimpleNamespace(_model=FakeSession(taken), _preprocessor=preprocessor, config={"version": "v3"})
        self.calls = []

    def recognize(self, waveform: np.ndarray, sample_rate: int = 16000) -> str:
        self.calls.append((waveform, sample_rate))
        return "где дверь"


class FakeHub:
    """snapshot_download that writes the asked files and remembers the calls"""

    def __init__(self, cache: Path) -> None:
        self.cache = cache
        self.calls = []

    def __call__(self, repo_id, revision=None, allow_patterns=None, local_dir=None, local_files_only=False):
        self.calls.append({"repo_id": repo_id, "revision": revision, "allow_patterns": list(allow_patterns), "local_dir": local_dir, "local_files_only": local_files_only})
        target = Path(local_dir) if local_dir is not None else self.cache / "snapshots" / revision
        if local_files_only and not target.is_dir():
            raise backend_module().LocalEntryNotFoundError("not in the cache")
        target.mkdir(parents=True, exist_ok=True)
        for name in allow_patterns:
            (target / name).write_bytes(b"onnx")
        return str(target)


def backend_module():
    return sys.modules[MODULE]


@pytest.fixture
def onnx_asr(monkeypatch, tmp_path):
    # The library is a fake module; the backend module is imported against it and dropped after the test
    state = types.SimpleNamespace(loaded=[], taken=["CPUExecutionProvider"])

    def load_model(model, path=None, *, providers=None, **kwargs):
        loaded = FakeModel(model, path, providers, state.taken)
        state.loaded.append(loaded)
        return loaded

    fake = types.ModuleType("onnx_asr")
    fake.load_model = load_model
    monkeypatch.setitem(sys.modules, "onnx_asr", fake)
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    real_find_spec = registry.find_spec
    monkeypatch.setattr(registry, "find_spec", lambda package: object() if package == "onnx_asr" else real_find_spec(package))

    module = importlib.import_module(MODULE)
    state.hub = FakeHub(tmp_path / "hf-cache")
    monkeypatch.setattr(module, "snapshot_download", state.hub)
    yield state
    sys.modules.pop(MODULE, None)


def make_settings(**values) -> Settings:
    # No .env: the test sees the defaults and its own values only
    return Settings(_env_file=None, **{"MODEL_BACKEND": "gigaam_onnx", "MODEL_NAME": None, "TARGET_PLATFORM": "cpu", **values})


def test_gigaam_onnx_models(onnx_asr, tmp_path):
    backend = build_backend(make_settings(ARTIFACTS_PATH=tmp_path))
    assert (backend.name, backend.model, backend.device) == ("gigaam_onnx", "v3_ctc", "cpu")
    assert type(backend).models == frozenset({"v3_ctc", "v3_rnnt"})
    assert onnx_asr.loaded[-1].name == "gigaam-v3-ctc"

    build_backend(make_settings(MODEL_NAME="v3_rnnt", ARTIFACTS_PATH=tmp_path))
    assert onnx_asr.loaded[-1].name == "gigaam-v3-rnnt"

    message = "Unknown model 'v3_e2e_ctc' of backend 'gigaam_onnx', expected one of: v3_ctc, v3_rnnt"
    with pytest.raises(BackendError, match=re.escape(message)):
        build_backend(make_settings(MODEL_NAME="v3_e2e_ctc", ARTIFACTS_PATH=tmp_path))


@pytest.mark.parametrize("model", ["v3_ctc", "v3_rnnt"])
def test_gigaam_onnx_fetch_full_precision_files(onnx_asr, tmp_path, model):
    build_backend(make_settings(MODEL_NAME=model, ARTIFACTS_PATH=tmp_path))

    (call, ) = onnx_asr.hub.calls
    assert (call["repo_id"], call["revision"]) == ("istupakov/gigaam-v3-onnx", CFG.REVISION)
    assert call["local_dir"] == tmp_path / "gigaam_onnx"
    patterns = call["allow_patterns"]
    assert {"config.json", "v3_vocab.txt"} <= set(patterns)
    assert not [name for name in patterns if "int8" in name or "e2e" in name or "*" in name]
    expected = {"v3_ctc": {"v3_ctc.onnx"}, "v3_rnnt": {"v3_rnnt_encoder.onnx", "v3_rnnt_decoder.onnx", "v3_rnnt_joint.onnx"}}[model]
    assert {name for name in patterns if name.endswith(".onnx")} == expected


def test_gigaam_onnx_loads_from_artifacts(onnx_asr, tmp_path):
    build_backend(make_settings(ARTIFACTS_PATH=tmp_path))
    assert onnx_asr.loaded[-1].path == str(tmp_path / "gigaam_onnx")

    # The weights are there: the next start goes to the network no more
    build_backend(make_settings(ARTIFACTS_PATH=tmp_path))
    assert len(onnx_asr.hub.calls) == 1


def test_gigaam_onnx_fetch_without_artifacts_uses_cache(onnx_asr):
    first = build_backend(make_settings(ARTIFACTS_PATH=None))
    # Nothing in the cache: a lookup without the network, then the download
    assert [call["local_files_only"] for call in onnx_asr.hub.calls] == [True, False]
    assert all(call["local_dir"] is None for call in onnx_asr.hub.calls)
    assert onnx_asr.loaded[-1].path.endswith(CFG.REVISION)

    build_backend(make_settings(ARTIFACTS_PATH=None))
    assert [call["local_files_only"] for call in onnx_asr.hub.calls] == [True, False, True]
    assert first.params()["revision"] == CFG.REVISION


def test_gigaam_onnx_fetch_weights_line(onnx_asr, tmp_path):
    line = fetch_weights.fetch(make_settings(ARTIFACTS_PATH=tmp_path))
    assert line == f"fetch_weights: gigaam_onnx/v3_ctc -> {tmp_path / 'gigaam_onnx'}"
    assert not onnx_asr.loaded, "fetch_weights must not load the model"


def test_gigaam_onnx_cuda_without_cuda_provider_fails(onnx_asr, tmp_path):
    # onnxruntime without CUDA falls back to CPU silently: on cuda it is an error (MA-1)
    onnx_asr.taken = ["CPUExecutionProvider"]
    with pytest.raises(BackendError, match="needs CUDAExecutionProvider"):
        build_backend(make_settings(TARGET_PLATFORM="cuda", ARTIFACTS_PATH=tmp_path))
    assert onnx_asr.loaded[-1].providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_gigaam_onnx_cuda_params(onnx_asr, tmp_path):
    onnx_asr.taken = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    backend = build_backend(make_settings(TARGET_PLATFORM="cuda", ARTIFACTS_PATH=tmp_path))

    params = backend.params()
    assert (params["backend"], params["name"], params["device"]) == ("gigaam_onnx", "v3_ctc", "cuda")
    assert params["providers"][0] == "CUDAExecutionProvider"
    assert params["revision"] == CFG.REVISION


def test_gigaam_onnx_cpu_params_and_recognize(onnx_asr, tmp_path):
    backend = build_backend(make_settings(ARTIFACTS_PATH=tmp_path))
    assert onnx_asr.loaded[-1].providers == ["CPUExecutionProvider"]
    assert backend.params()["providers"] == ["CPUExecutionProvider"]

    audio = np.zeros(16000, dtype=np.float64)
    assert backend.recognize(audio, "ru") == "где дверь"
    waveform, sample_rate = onnx_asr.loaded[-1].calls[-1]
    assert (waveform.dtype, sample_rate) == (np.float32, 16000)
