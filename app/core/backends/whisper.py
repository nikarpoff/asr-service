"""
Whisper on CTranslate2 through faster-whisper.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from faster_whisper import WhisperModel
from faster_whisper.utils import download_model
from loguru import logger

from app.core.backends.base import Backend
from app.core.config import Settings, WHISPER_SETTINGS as CFG


class WhisperBackend(Backend):
    """Whisper with a fixed language and the greedy search: speed matters more than the last percent of quality"""
    name = "whisper"
    default_model = "small"
    models = frozenset({"small", "medium", "large-v3-turbo"})

    def __init__(self, settings: Settings, model: str) -> None:
        super().__init__(settings, model)
        self.compute_type = CFG.COMPUTE_TYPE_CUDA if self.device == "cuda" else CFG.COMPUTE_TYPE_CPU

        path = self.fetch(settings, model)
        self.whisper = WhisperModel(str(path), device=self.device, compute_type=self.compute_type)
        logger.info("Whisper model ready: model={} path='{}' device={} compute_type={}", model, path, self.device, self.compute_type)

    @classmethod
    def fetch(cls, settings: Settings, model: str) -> Optional[Path]:
        root = cls.weights_root(settings)
        if root is None:
            return Path(download_model(model))

        path = root / model
        if not (path / "model.bin").is_file():
            logger.info("Downloading Whisper '{}' weights into '{}'", model, path)
            download_model(model, output_dir=str(path))
        return path

    def recognize(self, audio: np.ndarray, language: str) -> str:
        """Blocking: call it from a worker thread."""
        # The speech detector has already run in the pipeline; the previous text is not a context:
        # a question is one short segment, and a context only feeds the made-up continuations
        segments, _ = self.whisper.transcribe(
            audio,
            language=language,
            beam_size=CFG.BEAM_SIZE,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    def params(self) -> Dict[str, Any]:
        return {**super().params(), "beam_size": CFG.BEAM_SIZE, "compute_type": self.compute_type}

    def close(self) -> None:
        self.whisper = None
