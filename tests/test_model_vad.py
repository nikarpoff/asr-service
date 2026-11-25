"""
The speech detector on street noise (MA-4): the speech is kept, the noise alone is not speech.

T-one runs on the core onnxruntime and Silero VAD is a part of the core, so the tests need no
extra, only the network or a cache with the T-one weights (~140 MB), as the smoke test AC-68.
"""

from pathlib import Path

import numpy as np
import pytest

from app.core.audio import decode, parse_media_type, to_model_input
from app.core.backends import build_backend
from app.core.config import Settings, AUDIO_SETTINGS
from app.core.vad import SileroVad
from bench import wer

pytestmark = pytest.mark.model

DATA = Path(__file__).resolve().parent / "data"
MANIFEST = DATA / "questions.jsonl"
MAX_WER = 0.30
NOISE_SECONDS = 0.3                     # street noise before the speech starts in q003 and q004
NOISE_TOTAL_SECONDS = 2.0


def load_samples(name: str) -> np.ndarray:
    # The way of a request: WAV decoding, then 16 kHz mono
    audio = decode((DATA / "questions" / name).read_bytes(), parse_media_type("audio/wav"), 30.0)
    return to_model_input(audio.samples, audio.sample_rate)


@pytest.fixture(scope="module")
def backend():
    # No .env and no fake of the fast tests: the settings of the model are given here
    backend = build_backend(Settings(_env_file=None, MODEL_BACKEND="tone", MODEL_NAME=None, TARGET_PLATFORM="cpu", ARTIFACTS_PATH=None))
    yield backend
    backend.close()


def test_vad_keeps_noisy_speech(backend):
    vad = SileroVad()

    backend.vad = vad
    with_vad = wer.evaluate(backend, MANIFEST)
    backend.vad = None
    without_vad = wer.evaluate(backend, MANIFEST)
    backend.vad = vad

    report = ", ".join(f"{item['id']}: '{item['reference']}' -> '{item['hypothesis']}'" for item in with_vad["items"])
    hypotheses = {item["id"]: item["hypothesis"] for item in with_vad["items"]}
    assert hypotheses["q004"] == "окно", report
    assert with_vad["overall"]["wer"] <= without_vad["overall"]["wer"], f"WER with the detector {with_vad['overall']['wer']} is over {without_vad['overall']['wer']} without it: {report}"
    assert with_vad["overall"]["wer"] <= MAX_WER, f"WER {with_vad['overall']['wer']} is over {MAX_WER}: {report}"


def test_vad_street_noise_is_not_speech():
    # The noise before the speech, repeated up to the length of a short question
    head = int(NOISE_SECONDS * AUDIO_SETTINGS.SAMPLE_RATE)
    noise = np.concatenate([load_samples(name)[:head] for name in ("q003.wav", "q004.wav")])
    total = int(NOISE_TOTAL_SECONDS * AUDIO_SETTINGS.SAMPLE_RATE)
    audio = np.tile(noise, -(-total // noise.shape[0]))[:total]

    vad = SileroVad()
    assert vad.speech_segments(audio) == [], f"max speech probability {vad.probabilities(audio).max():.3f}"
    assert vad.trim(audio) is None
