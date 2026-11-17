"""
GigaAM: Conformer models for Russian, CTC and RNN-T heads.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import gigaam
import numpy as np
import torch

from loguru import logger

from app.core.backends.base import Backend
from app.core.config import Settings, GIGAAM_SETTINGS as CFG


class GigaAMBackend(Backend):
    """GigaAM on torch; FP16 encoder on CUDA"""
    name = "gigaam"
    default_model = "v3_ctc"
    models = frozenset({"v2_ctc", "v2_rnnt", "v3_ctc", "v3_rnnt", "v3_e2e_ctc", "v3_e2e_rnnt"})

    def __init__(self, settings: Settings, model: str) -> None:
        super().__init__(settings, model)
        root = self.weights_root(settings)
        self.gigaam = gigaam.load_model(
            model,
            fp16_encoder=CFG.FP16_ENCODER,
            device=self.device,
            download_root=str(root) if root else None,
        )
        logger.info("GigaAM model ready: model={} device={}", model, self.device)

    @classmethod
    def fetch(cls, settings: Settings, model: str) -> Optional[Path]:
        root = cls.weights_root(settings) or Path(gigaam._CACHE_DIR)
        # The checkpoint and, for the models that have one, the tokenizer; files already there are kept
        _, path = gigaam._download_model(model, str(root))
        gigaam._download_tokenizer(model, str(root))
        return Path(path)

    @torch.inference_mode()
    def recognize(self, audio: np.ndarray, language: str) -> str:
        """Blocking: call it from a worker thread."""
        parameter = next(self.gigaam.parameters())
        wav = torch.from_numpy(audio).to(parameter.device).to(parameter.dtype).unsqueeze(0)
        length = torch.full([1], wav.shape[-1], device=parameter.device)

        encoded, encoded_length = self.gigaam.forward(wav, length)
        text, _, _ = self.gigaam.decoding.decode(self.gigaam.head, encoded, encoded_length)[0]
        return text

    def params(self) -> Dict[str, Any]:
        return {**super().params(), "fp16_encoder": CFG.FP16_ENCODER and self.device == "cuda"}

    def close(self) -> None:
        self.gigaam = None
