"""
Smoke of a real model: the CPU model recognizes the test questions with WER of 30 % at most (ASR-3).

The model is T-one: it runs on the core onnxruntime, so the test needs no extra, only the
network or a cache with the weights (~140 MB). The recordings go the way of a request:
decoding, 16 kHz mono, the speech detector, the model, the normalization.
"""

from pathlib import Path

import numpy as np
import pytest

from app.core.backends import build_backend
from app.core.config import Settings, AUDIO_SETTINGS
from bench import wer

pytestmark = pytest.mark.model

MANIFEST = Path(__file__).resolve().parent / "data" / "questions.jsonl"
MAX_WER = 0.30


@pytest.fixture(scope="module")
def backend():
    # No .env and no fake of the fast tests: the settings of the model are given here
    backend = build_backend(Settings(_env_file=None, MODEL_BACKEND="tone", MODEL_NAME=None, TARGET_PLATFORM="cpu", ARTIFACTS_PATH=None))
    yield backend
    backend.close()


def test_cpu_model_recognizes_questions(backend):
    assert (backend.name, backend.device) == ("tone", "cpu")
    backend.warmup()

    scores = wer.evaluate(backend, MANIFEST)

    report = ", ".join(f"{item['id']}: '{item['reference']}' -> '{item['hypothesis']}'" for item in scores["items"])
    assert scores["overall"]["n"] == len(wer.read_manifest(MANIFEST))
    assert scores["overall"]["wer"] <= MAX_WER, f"WER {scores['overall']['wer']} is over {MAX_WER}: {report}"


def test_cpu_model_keeps_silence_empty(backend):
    # The model itself, past the speech detector: silence must not turn into words
    silence = np.zeros(2 * AUDIO_SETTINGS.SAMPLE_RATE, dtype=np.float32)
    assert backend.recognize(silence, "ru") == ""
