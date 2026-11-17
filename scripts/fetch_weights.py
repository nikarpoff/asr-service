"""
Скачивает веса модели текущих MODEL_BACKEND и MODEL_NAME, не загружая её в память (DEP-6).

С ARTIFACTS_PATH веса ложатся в `<ARTIFACTS_PATH>/<бэкенд>/`, откуда их берёт сервис при
старте: после этого он стартует без сети. Без ARTIFACTS_PATH - в кэш библиотеки модели.
Файлы, которые уже на месте, не скачиваются заново.

    MODEL_BACKEND=whisper MODEL_NAME=small ARTIFACTS_PATH=./model uv run python -m scripts.fetch_weights
"""

import argparse
import sys

from typing import Optional

from loguru import logger

from app.core.backends import registry
from app.core.backends.registry import BACKENDS
from app.core.config import Settings
from app.core.exceptions import BackendError


def fetch(settings: Settings) -> str:
    """Строка итога: бэкенд, модель и место весов."""
    name = settings.MODEL_BACKEND
    spec = BACKENDS.get(name)
    if spec is None:
        raise BackendError(f"Unknown backend '{name}', expected one of: {', '.join(sorted(BACKENDS))}")

    # Тот же ленивый импорт, что при старте сервиса: без extra - та же понятная ошибка
    backend_class = registry._load_class(name, spec)
    model = settings.MODEL_NAME or backend_class.default_model
    if backend_class.models and model not in backend_class.models:
        raise BackendError(
            f"Unknown model '{model}' of backend '{name}', expected one of: {', '.join(sorted(backend_class.models))}"
        )

    path = backend_class.fetch(settings, model)
    if path is not None:
        place = str(path)
    elif name == "fake":
        place = "no weights"
    else:
        place = "library cache"
    return f"fetch_weights: {name}/{model} -> {place}"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Скачивание весов модели из настроек (MODEL_BACKEND, MODEL_NAME, ARTIFACTS_PATH)")
    parser.add_argument("--backend", default=None, help="бэкенд вместо MODEL_BACKEND")
    parser.add_argument("--model", default=None, help="модель вместо MODEL_NAME")
    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    overrides = {key: value for key, value in (("MODEL_BACKEND", args.backend), ("MODEL_NAME", args.model)) if value}
    try:
        print(fetch(Settings(**overrides)))
    except BackendError as e:
        print(f"fetch_weights: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
