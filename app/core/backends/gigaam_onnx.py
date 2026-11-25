"""
GigaAM-v3 on onnxruntime through onnx-asr, with the weights of the ONNX export on Hugging Face.

The `gigaam` library downloads from the CDN of the authors only; this backend takes the same
v3 models exported to ONNX (`istupakov/gigaam-v3-onnx`, `base_model: ai-sage/GigaAM-v3`) at a
pinned revision. The weights are fetched ahead into ARTIFACTS_PATH, and the model is loaded from
that directory: the library does not go to the network at startup.

On CUDA a silent fallback of onnxruntime to CPU is an error, not a warning: the measurements of
the model machine must not pass on CPU unnoticed.
"""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import onnx_asr

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from loguru import logger

from app.core.backends.base import Backend
from app.core.config import Settings, AUDIO_SETTINGS, GIGAAM_ONNX_SETTINGS as CFG
from app.core.exceptions import BackendError

CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


class GigaAMOnnxBackend(Backend):
    """GigaAM-v3 CTC and RNN-T on onnxruntime; CUDA through CUDAExecutionProvider only"""
    name = "gigaam_onnx"
    default_model = "v3_ctc"
    models = frozenset(CFG.MODELS)

    def __init__(self, settings: Settings, model: str) -> None:
        super().__init__(settings, model)
        self.providers = [CUDA_PROVIDER, CPU_PROVIDER] if self.device == "cuda" else [CPU_PROVIDER]

        path = self.fetch(settings, model)
        self.gigaam = onnx_asr.load_model(CFG.MODELS[model], path=str(path), providers=self.providers)

        # The providers onnxruntime took, not the asked ones: a session without CUDA runs on CPU silently
        self.session_providers = [session.get_providers() for session in self._sessions()]
        if self.device == "cuda":
            fallen = [providers for providers in self.session_providers if not providers or providers[0] != CUDA_PROVIDER]
            if not self.session_providers or fallen:
                raise BackendError(
                    f"GigaAM ONNX on cuda needs {CUDA_PROVIDER}, the sessions run on {self.session_providers}"
                )
        logger.info("GigaAM ONNX model ready: model={} path='{}' device={} providers={}", model, path, self.device, self.session_providers)

    @staticmethod
    def files(model: str) -> List[str]:
        """Files of the model in the repository: full precision only, no int8 and no e2e."""
        return [*CFG.COMMON_FILES, *CFG.FILES[model]]

    @classmethod
    def fetch(cls, settings: Settings, model: str) -> Optional[Path]:
        files = cls.files(model)
        root = cls.weights_root(settings)
        if root is not None and cls._complete(root, files):
            return root

        if root is None:
            # The snapshot of the pinned revision in the cache, without a request when it is there
            try:
                path = Path(snapshot_download(CFG.REPOSITORY, revision=CFG.REVISION, allow_patterns=files, local_files_only=True))
                if cls._complete(path, files):
                    return path
            except LocalEntryNotFoundError:
                pass

        logger.info("Downloading GigaAM ONNX '{}' weights into '{}'", model, root or "the Hugging Face cache")
        path = Path(snapshot_download(CFG.REPOSITORY, revision=CFG.REVISION, allow_patterns=files, local_dir=root))
        return root if root is not None else path

    def recognize(self, audio: np.ndarray, language: str) -> str:
        """Blocking: call it from a worker thread."""
        return self.gigaam.recognize(audio.astype(np.float32, copy=False), sample_rate=AUDIO_SETTINGS.SAMPLE_RATE)

    def params(self) -> Dict[str, Any]:
        return {
            **super().params(),
            "backend": self.name,
            "name": self.model,
            "device": self.device,
            "providers": self.session_providers[0] if self.session_providers else [],
            "revision": CFG.REVISION,
        }

    def close(self) -> None:
        self.gigaam = None

    def _sessions(self) -> Iterator[Any]:
        # onnx-asr keeps the sessions of the model (CTC; encoder, decoder and joint of RNN-T) and of its
        # feature extractor in private attributes: any attribute with get_providers is an onnxruntime session
        asr = self.gigaam.asr
        holders: List[Any] = [asr]
        preprocessor = getattr(asr, "_preprocessor", None)
        if preprocessor is not None:
            holders.extend([preprocessor, getattr(preprocessor, "preprocessor", None)])
        for holder in holders:
            for value in vars(holder).values() if holder is not None and hasattr(holder, "__dict__") else ():
                if callable(getattr(value, "get_providers", None)):
                    yield value

    @staticmethod
    def _complete(path: Path, files: List[str]) -> bool:
        return all((path / file).is_file() for file in files)
