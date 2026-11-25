"""
Tests of the model machine (G-3): the real weights of every backend, on CUDA where the backend has it.

The thresholds are those of the smoke test AC-68: WER of 30 % at most on tests/data/questions.jsonl,
and two seconds of silence give no words. A test on CUDA checks the GPU in its first line, before any
weights: without a GPU it fails in seconds with "CUDA is not available", it is never skipped (AC-30).
The backend libraries are imported inside the tests only, so the module is collected in the fast
environment too.

The weights come from ARTIFACTS_PATH when it is set (scripts/fetch_weights.py), else from the caches.
"""

import os

from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

from app.core.backends import Backend, build_backend
from app.core.config import Settings, AUDIO_SETTINGS
from bench import wer

pytestmark = pytest.mark.model

MANIFEST = Path(__file__).resolve().parent / "data" / "questions.jsonl"
MAX_WER = 0.30
CUDA_PROVIDER = "CUDAExecutionProvider"

# (backend, model) on CUDA; the ids are "<backend>-<model>"
CUDA_MODELS = [
    ("gigaam_onnx", "v3_ctc"),
    ("gigaam_onnx", "v3_rnnt"),
    ("whisper", "medium"),
    ("whisper", "large-v3-turbo"),
]


def require_cuda(onnxruntime_provider: bool = False) -> None:
    """Fails the test at once when there is no GPU: a skip would pass for a check."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and not visible.strip():
        pytest.fail("CUDA is not available: CUDA_VISIBLE_DEVICES is empty", pytrace=False)

    if find_spec("torch") is not None:
        import torch

        if not torch.cuda.is_available():
            pytest.fail("CUDA is not available: torch.cuda.is_available() is False", pytrace=False)

    if onnxruntime_provider or find_spec("torch") is None:
        import onnxruntime

        providers = onnxruntime.get_available_providers()
        if CUDA_PROVIDER not in providers:
            pytest.fail(f"CUDA is not available: no {CUDA_PROVIDER} in onnxruntime {onnxruntime.__version__} {providers}", pytrace=False)


def make_backend(name: str, model: str, device: str) -> Backend:
    # No .env and no fake of the fast tests: the settings of the model are given here
    artifacts = os.environ.get("ARTIFACTS_PATH") or None
    return build_backend(Settings(_env_file=None, MODEL_BACKEND=name, MODEL_NAME=model, TARGET_PLATFORM=device, ARTIFACTS_PATH=artifacts))


def check_questions(backend: Backend) -> None:
    """WER on the manifest and silence past the speech detector, as AC-68."""
    backend.warmup()
    scores = wer.evaluate(backend, MANIFEST)

    report = ", ".join(f"{item['id']}: '{item['reference']}' -> '{item['hypothesis']}'" for item in scores["items"])
    assert scores["overall"]["n"] == len(wer.read_manifest(MANIFEST))
    assert scores["overall"]["wer"] <= MAX_WER, f"WER {scores['overall']['wer']} is over {MAX_WER}: {report}"

    # The model itself, past the speech detector: silence must not turn into words
    silence = np.zeros(2 * AUDIO_SETTINGS.SAMPLE_RATE, dtype=np.float32)
    assert backend.recognize(silence, "ru") == ""


@pytest.mark.parametrize(("name", "model"), CUDA_MODELS)
def test_backend_recognizes_questions_cuda(name, model):
    require_cuda(onnxruntime_provider=name == "gigaam_onnx")
    backend = make_backend(name, model, "cuda")
    try:
        assert (backend.name, backend.model, backend.device) == (name, model, "cuda")
        check_questions(backend)
    finally:
        backend.close()


def test_gigaam_onnx_cuda_provider_cuda():
    # AC-43: the sessions onnxruntime really created, not the asked providers; a fallback to CPU fails
    require_cuda(onnxruntime_provider=True)
    backend = make_backend("gigaam_onnx", "v3_ctc", "cuda")
    try:
        sessions = list(backend._sessions())
        assert sessions, "no onnxruntime sessions in the model"
        assert [session.get_providers()[0] for session in sessions] == [CUDA_PROVIDER] * len(sessions)
        assert backend.gigaam.asr._model.get_providers()[0] == CUDA_PROVIDER
        assert backend.params()["device"] == "cuda"
        assert backend.params()["providers"][0] == CUDA_PROVIDER
    finally:
        backend.close()


def test_vosk_ru_recognizes_questions():
    # MA-3: the large Vosk model runs on CPU only
    backend = make_backend("vosk", "ru", "cpu")
    try:
        assert (backend.name, backend.model, backend.device) == ("vosk", "ru", "cpu")
        check_questions(backend)
    finally:
        backend.close()


def test_gigaam_onnx_recognizes_questions_cpu():
    # MA-1 on CPU: the same weights without a GPU
    backend = make_backend("gigaam_onnx", "v3_ctc", "cpu")
    try:
        assert (backend.name, backend.model, backend.device) == ("gigaam_onnx", "v3_ctc", "cpu")
        assert backend.params()["providers"][0] == "CPUExecutionProvider"
        check_questions(backend)
    finally:
        backend.close()
