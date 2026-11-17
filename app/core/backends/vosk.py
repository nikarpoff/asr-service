"""
Vosk: Kaldi models that run on CPU.
"""

import json
import shutil
import tempfile
import urllib.request
import zipfile

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from loguru import logger
from vosk import KaldiRecognizer, Model, SetLogLevel

from app.core.backends.base import Backend
from app.core.config import Settings, AUDIO_SETTINGS, VOSK_SETTINGS as CFG


class VoskBackend(Backend):
    """Vosk recognizer; the model is shared, a recognizer is created per question"""
    name = "vosk"
    default_model = "small-ru"
    models = frozenset(CFG.ARCHIVES)

    def __init__(self, settings: Settings, model: str) -> None:
        super().__init__(settings, model)
        if self.device == "cuda":
            # /v1/info must show where the model really runs
            logger.warning("Vosk runs on cpu only, TARGET_PLATFORM=cuda is ignored")
            self.device = "cpu"

        SetLogLevel(-1)         # Kaldi writes its own log to stderr past loguru
        if self.weights_root(settings) is None:
            # The library downloads into its own cache, ~/.cache/vosk, and finds the model there by name
            self.vosk = Model(model_name=CFG.ARCHIVES[model])
        else:
            self.vosk = Model(model_path=str(self.fetch(settings, model)))
        logger.info("Vosk model ready: model={} archive={} device={}", model, CFG.ARCHIVES[model], self.device)

    @classmethod
    def fetch(cls, settings: Settings, model: str) -> Optional[Path]:
        archive = CFG.ARCHIVES[model]
        root = cls.weights_root(settings)
        if root is None:
            # Only the library knows its cache: loading the model by name is its way to download one
            Model(model_name=archive)
            return None

        path = root / archive
        if not path.is_dir():
            logger.info("Downloading Vosk '{}' weights into '{}'", archive, root)
            cls._download(archive, root)
        return path

    def recognize(self, audio: np.ndarray, language: str) -> str:
        """Blocking: call it from a worker thread."""
        recognizer = KaldiRecognizer(self.vosk, AUDIO_SETTINGS.SAMPLE_RATE)
        pcm = np.clip(np.rint(audio * 32768.0), -32768, 32767).astype(np.int16)
        recognizer.AcceptWaveform(pcm.tobytes())
        return json.loads(recognizer.FinalResult()).get("text", "")

    def params(self) -> Dict[str, Any]:
        return {**super().params(), "archive": CFG.ARCHIVES[self.model]}

    def close(self) -> None:
        self.vosk = None

    @staticmethod
    def _download(archive: str, root: Path) -> None:
        # The archive holds one directory named as the archive; it appears under its name only when complete
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as workspace:
            packed = Path(workspace) / f"{archive}.zip"
            with urllib.request.urlopen(f"{CFG.URL}/{archive}.zip") as response, open(packed, "wb") as target:
                shutil.copyfileobj(response, target)
            with zipfile.ZipFile(packed) as source:
                source.extractall(workspace)
            (Path(workspace) / archive).rename(root / archive)
