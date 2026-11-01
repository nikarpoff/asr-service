"""
Model backends: interchangeable implementations of the service model.

A backend is chosen by MODEL_BACKEND, built once at startup with its weights loaded,
and called from the engine worker threads only.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, FrozenSet

from app.core.config import Settings


class Backend(ABC):
    """A model behind the service API"""
    name: ClassVar[str]                                     # the MODEL_BACKEND value
    default_model: ClassVar[str]                            # the model when MODEL_NAME is unset
    models: ClassVar[FrozenSet[str]] = frozenset()          # known MODEL_NAME values; empty takes any

    def __init__(self, settings: Settings, model: str) -> None:
        # Models are loaded here, at startup: the first request must not pay for loading them
        self.settings = settings
        self.model = model
        self.device = settings.TARGET_PLATFORM

    @abstractmethod
    def warmup(self) -> None:
        """An idle call on a typical input. Blocking: call it from a worker thread."""
        ...

    def params(self) -> Dict[str, Any]:
        """Backend parameters that affect the answers, for /v1/info."""
        return {}

    def close(self) -> None:
        """Releases the model at shutdown."""
