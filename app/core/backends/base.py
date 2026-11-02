"""
Model backends: interchangeable speech recognition models.

A backend is chosen by MODEL_BACKEND, built once at startup with its weights loaded,
and called from the engine worker threads only. The speech detector is built with it:
the backend is the one object the shared engine builds and warms up at startup, and
every model gets the same trimmed audio.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, FrozenSet, Optional

import numpy as np

from app.core.config import get_service_settings, Settings, AUDIO_SETTINGS
from app.core.vad import SileroVad

DEFAULT_LANGUAGE = "ru"


class Backend(ABC):
    """A speech recognition model behind the service API, with Silero VAD in front of it"""
    name: ClassVar[str]                                     # the MODEL_BACKEND value
    default_model: ClassVar[str]                            # the model when MODEL_NAME is unset
    models: ClassVar[FrozenSet[str]] = frozenset()          # known MODEL_NAME values; empty takes any
    languages: ClassVar[FrozenSet[str]] = frozenset({DEFAULT_LANGUAGE})     # values of the `language` parameter

    def __init__(self, settings: Settings, model: str) -> None:
        # Models are loaded here, at startup: the first request must not pay for loading them
        self.settings = settings
        self.model = model
        self.device = settings.TARGET_PLATFORM
        self.vad: Optional[SileroVad] = SileroVad() if get_service_settings().VAD_ENABLED else None

    @abstractmethod
    def recognize(self, audio: np.ndarray, language: str) -> str:
        """Text of the speech in 16 kHz mono float32 audio. Blocking: call it from a worker thread."""
        ...

    def warmup(self) -> None:
        """An idle call of the detector and the model on a typical input. Blocking: call it from a worker thread."""
        # Quiet noise of a question length: the detector sees no speech, so the model is called apart
        generator = np.random.default_rng(0)
        audio = (0.01 * generator.standard_normal(int(AUDIO_SETTINGS.WARMUP_SECONDS * AUDIO_SETTINGS.SAMPLE_RATE))).astype(np.float32)
        if self.vad is not None:
            self.vad.trim(audio)
        self.recognize(audio, DEFAULT_LANGUAGE)

    def params(self) -> Dict[str, Any]:
        """Backend parameters that affect the answers, for /v1/info."""
        return {
            "language": DEFAULT_LANGUAGE,
            "sample_rate": AUDIO_SETTINGS.SAMPLE_RATE,
            "vad_enabled": self.vad is not None,
        }

    def close(self) -> None:
        """Releases the model at shutdown."""
