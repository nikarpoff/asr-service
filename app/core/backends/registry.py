"""
Registry of the model backends.

Backend modules are imported lazily, only the chosen one: the libraries of the real
backends are optional extras (torch, AGPL and GPL code among them), and the service
installs and runs its fast tests without them.
"""

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import Dict, Optional, Tuple, Type

from app.core.backends.base import Backend
from app.core.config import Settings
from app.core.exceptions import BackendError


@dataclass(frozen=True)
class BackendSpec:
    """Where a backend lives and what it needs installed"""
    path: str                               # "<module>:<class>"
    extra: Optional[str] = None             # pyproject extra with the backend libraries
    packages: Tuple[str, ...] = ()          # top-level packages of the extra, checked before the import


# Backends of the service (ASR-3); the key is the MODEL_BACKEND value
BACKENDS: Dict[str, BackendSpec] = {
    "fake": BackendSpec("app.core.backends.fake:FakeBackend"),
    "gigaam": BackendSpec("app.core.backends.gigaam:GigaAMBackend", "gigaam", ("gigaam", )),
    "tone": BackendSpec("app.core.backends.tone:ToneBackend", "tone", ("tone", )),
    "whisper": BackendSpec("app.core.backends.whisper:WhisperBackend", "whisper", ("faster_whisper", )),
    "vosk": BackendSpec("app.core.backends.vosk:VoskBackend", "vosk", ("vosk", )),
}


def build_backend(settings: Settings) -> Backend:
    """The backend of MODEL_BACKEND with the model of MODEL_NAME loaded. Blocking."""
    name = settings.MODEL_BACKEND
    spec = BACKENDS.get(name)
    if spec is None:
        raise BackendError(f"Unknown backend '{name}', expected one of: {', '.join(sorted(BACKENDS))}")

    backend_class = _load_class(name, spec)

    model = settings.MODEL_NAME or backend_class.default_model
    if backend_class.models and model not in backend_class.models:
        raise BackendError(
            f"Unknown model '{model}' of backend '{name}', expected one of: {', '.join(sorted(backend_class.models))}"
        )

    return backend_class(settings, model)


def _load_class(name: str, spec: BackendSpec) -> Type[Backend]:
    missing_extra = BackendError(f"Backend '{name}' requires the '{spec.extra}' extra")

    if any(find_spec(package) is None for package in spec.packages):
        raise missing_extra

    module_name, class_name = spec.path.split(":")
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as e:
        # A library of the extra is missing, not the backend module itself
        if spec.extra and e.name and e.name != module_name and not e.name.startswith("app."):
            raise missing_extra from e
        raise BackendError(f"Backend '{name}' is not available: {e}") from e

    return getattr(module, class_name)
