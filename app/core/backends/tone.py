"""
T-one: a streaming CTC model for Russian telephony speech, run offline over the whole question.

The acoustic model is one ONNX file with the feature extraction inside, so the backend
needs the core onnxruntime only and has no extra. The `tone` package of the authors is not
used: it pulls pyctcdecode (numpy<2) and KenLM (LGPL, built from source) for the beam
search with a language model. Decoding here is the greedy one of that package.
"""

from itertools import groupby
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import onnxruntime as ort

from huggingface_hub import hf_hub_download
from loguru import logger

from app.core.audio import resample
from app.core.backends.base import Backend
from app.core.config import Settings, AUDIO_SETTINGS, TONE_SETTINGS as CFG


class ToneBackend(Backend):
    """T-one acoustic model on onnxruntime with greedy CTC decoding"""
    name = "tone"
    default_model = "t-one"
    models = frozenset({"t-one"})

    def __init__(self, settings: Settings, model: str) -> None:
        super().__init__(settings, model)
        providers = ["CPUExecutionProvider"]
        if self.device == "cuda":
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
            else:
                # /v1/info must show where the model really runs
                logger.warning("T-one: this onnxruntime build has no CUDA provider, the model runs on cpu")
                self.device = "cpu"

        path = self.fetch(settings, model)
        self.session = ort.InferenceSession(str(path), providers=providers)
        logger.info("T-one model ready: path='{}' device={} providers={}", path, self.device, self.session.get_providers())

    @classmethod
    def fetch(cls, settings: Settings, model: str) -> Optional[Path]:
        root = cls.weights_root(settings)
        if root is None:
            return Path(hf_hub_download(CFG.REPOSITORY, CFG.FILENAME, revision=CFG.REVISION))

        path = root / CFG.FILENAME
        if not path.is_file():
            logger.info("Downloading T-one weights into '{}'", root)
            hf_hub_download(CFG.REPOSITORY, CFG.FILENAME, revision=CFG.REVISION, local_dir=root)
        return path

    def recognize(self, audio: np.ndarray, language: str) -> str:
        """Blocking: call it from a worker thread."""
        # 1. The model takes 8 kHz int16 samples in chunks of 300 ms
        samples = resample(audio, AUDIO_SETTINGS.SAMPLE_RATE, CFG.SAMPLE_RATE)
        signal = np.clip(np.rint(samples * 32768.0), -32768, 32767).astype(np.int32)

        # 2. Silence around the speech, as in the offline pipeline of the authors, and up to whole chunks
        signal = np.pad(signal, (CFG.CHUNK_SAMPLES, CFG.CHUNK_SAMPLES))
        signal = np.pad(signal, (0, -signal.shape[0] % CFG.CHUNK_SAMPLES))

        # 3. Chunk by chunk with the state carried over
        state = np.zeros((1, CFG.STATE_SIZE), dtype=np.float16)
        logprobs = []
        for chunk in signal.reshape(-1, CFG.CHUNK_SAMPLES):
            output, state = self.session.run(None, {"signal": chunk[None, :, None], "state": state})
            logprobs.append(output[0])

        return self._decode(np.concatenate(logprobs))

    def params(self) -> Dict[str, Any]:
        return {**super().params(), "model_sample_rate": CFG.SAMPLE_RATE, "decoder": "greedy", "revision": CFG.REVISION}

    def close(self) -> None:
        self.session = None

    @staticmethod
    def _decode(logprobs: np.ndarray) -> str:
        # Greedy CTC: the best token of every frame, repeats collapsed, blanks dropped
        tokens = [token for token, _ in groupby(logprobs.argmax(axis=-1).tolist())]
        text = "".join(CFG.LABELS[token] for token in tokens if token < len(CFG.LABELS))
        return " ".join(text.split())
