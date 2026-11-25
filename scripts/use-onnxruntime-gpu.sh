#!/usr/bin/env bash
# Заменяет onnxruntime ядра на onnxruntime-gpu из профиля gpu. Запускать после каждого
# `uv sync --extra gpu ...`.
#
# Оба пакета ставят один и тот же модуль `onnxruntime`, а uv ставит их оба: маркер extra в
# зависимостях ядра этого не меняет. Файлы перетирают друг друга, и без замены в окружении
# может оказаться CPU-сборка без CUDAExecutionProvider. Скрипт удаляет onnxruntime и заново
# ставит onnxruntime-gpu той версии, что в uv.lock, без зависимостей: они уже стоят.
#
#     uv sync --frozen --extra gpu --extra gigaam_onnx && scripts/use-onnxruntime-gpu.sh
#
# Окружение - VIRTUAL_ENV, иначе .venv сервиса.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${VIRTUAL_ENV:-$PWD/.venv}/bin/python"

# Версия onnxruntime-gpu из uv.lock: та же, что ставит `uv sync --frozen`
version=$("$PYTHON" -c '
import tomllib
with open("uv.lock", "rb") as source:
    packages = tomllib.load(source)["package"]
print(next(package["version"] for package in packages if package["name"] == "onnxruntime-gpu"))
')

uv pip uninstall --python "$PYTHON" onnxruntime
uv pip install --python "$PYTHON" --reinstall --no-deps "onnxruntime-gpu==${version}"

# Список провайдеров - свойство сборки, видеокарта для проверки не нужна
"$PYTHON" -c '
import sys
import onnxruntime
providers = onnxruntime.get_available_providers()
if "CUDAExecutionProvider" not in providers:
    sys.exit(f"use-onnxruntime-gpu: no CUDAExecutionProvider in onnxruntime {onnxruntime.__version__}: {providers}")
print(f"use-onnxruntime-gpu: onnxruntime-gpu {onnxruntime.__version__}, providers {providers}")
'
