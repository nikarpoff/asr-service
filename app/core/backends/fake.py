import time

from typing import Any, Dict

import numpy as np

from loguru import logger

from app.core.backends.base import Backend
from app.core.config import get_asr_settings, Settings


class FakeBackend(Backend):
    """Deterministic stand-in for the model: no weights, no GPU; FAKE_TEXT for any speech, FAKE_LATENCY_MS imitates the inference time"""
    name = "fake"
    default_model = "fake"
    models = frozenset({"fake"})

    def __init__(self, settings: Settings, model: str) -> None:
        super().__init__(settings, model)
        self.latency_ms = settings.FAKE_LATENCY_MS
        self.text = get_asr_settings().FAKE_TEXT
        logger.warning("Fake backend is on: the answers are synthetic, do not measure or deploy them")
        logger.info("Fake model ready: latency_ms={} text='{}'", self.latency_ms, self.text)

    def recognize(self, audio: np.ndarray, language: str) -> str:
        """Blocking: call it from a worker thread."""
        self._imitate_inference()
        return self.text

    def params(self) -> Dict[str, Any]:
        return {**super().params(), "latency_ms": self.latency_ms}

    def _imitate_inference(self) -> None:
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000)
