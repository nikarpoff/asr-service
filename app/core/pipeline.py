"""
Recognition of one request: decode → 16 kHz mono → speech detector → model → normalization.

Without speech the model is not called (ASR-2): it is faster, and a model would make a text
up from silence.
"""

from loguru import logger

from app.core.audio import decode, MediaType, to_model_input
from app.core.backends import Backend
from app.core.config import AUDIO_SETTINGS
from app.core.entities import Recognition
from app.core.text import normalize
from app.utils.middleware import current_timings


def recognize(backend: Backend, body: bytes, media: MediaType, language: str, max_seconds: float) -> Recognition:
    """Blocking: call it from a worker thread. The stages go to the request timings."""
    timings = current_timings()

    # 1. Decode and bring to the model input
    with timings.stage("decode"):
        audio = decode(body, media, max_seconds)
        samples = to_model_input(audio.samples, audio.sample_rate)

    # 2. Trim the silence at the edges; no speech at all - no model call
    if backend.vad is not None:
        with timings.stage("vad"):
            samples = backend.vad.trim(samples)
        if samples is None:
            logger.info("No speech in {:.2f} s of audio", audio.duration_s)
            return Recognition("", "", language, False, audio.duration_s)

    # 3. Recognize
    with timings.stage("inference"):
        text = backend.recognize(samples, language)

    logger.info("Recognized {:.2f} s of audio ({:.2f} s of speech): {} chars", audio.duration_s, samples.shape[0] / AUDIO_SETTINGS.SAMPLE_RATE, len(text))
    return Recognition(text, normalize(text), language, True, audio.duration_s)
