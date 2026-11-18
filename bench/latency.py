"""
Замер задержки распознавания в процессе, без HTTP (SVC-10).

Прогрев, затем N повторов полного конвейера запроса на фиксированной записи:
декодирование → 16 кГц моно → детектор речи → модель → нормализация.
Результат - JSON общей схемы EXP-2 в stdout (и в файл `--out`, если задан).

    uv run python -m bench.latency --backend fake --n 50 --warmup 5
"""

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import time

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from app.core import pipeline
from app.core.audio import parse_media_type
from app.core.backends import Backend, build_backend
from app.core.config import get_service_settings, get_project, Settings

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO = ROOT / "tests" / "data" / "questions" / "q001.wav"


class Hardware(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cpu: str = Field(description="Модель процессора")
    gpu: Optional[str] = Field(description="Модель видеокарты; null без неё")


class BenchResult(BaseModel):
    """Результат замера сервиса, схема EXP-2"""
    model_config = ConfigDict(extra="forbid")

    service: str
    backend: str
    model: str
    device: Literal["cpu", "cuda"]
    git_sha: str
    n: int = Field(ge=1)
    warmup: int = Field(ge=0)
    p50_ms: float
    p95_ms: float
    mean_ms: float
    std_ms: float
    hardware: Hardware
    started_at: str


def percentile(values: list[float], q: float) -> float:
    """Перцентиль nearest-rank: x[ceil(q·n) - 1] по отсортированной выборке."""
    ordered = sorted(values)
    return ordered[max(math.ceil(q * len(ordered)) - 1, 0)]


def sample_std(values: list[float]) -> float:
    """Выборочное стандартное отклонение (n - 1); 0 для одного значения."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def git_sha(settings: Settings) -> str:
    """GIT_SHA образа, иначе текущий коммит рабочей копии."""
    if settings.GIT_SHA != "unknown":
        return settings.GIT_SHA
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def hardware() -> Hardware:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass

    gpu = None
    try:
        names = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()
        gpu = names[0].strip() if names else None
    except (OSError, subprocess.CalledProcessError):
        pass
    return Hardware(cpu=cpu, gpu=gpu)


def build(backend: str, model: Optional[str], device: str) -> tuple[Settings, Backend]:
    """Бэкенд с настройками из окружения и .env, поверх которых - аргументы командной строки."""
    overrides = {"MODEL_BACKEND": backend, "TARGET_PLATFORM": device}
    if model:
        overrides["MODEL_NAME"] = model
    settings = Settings(**overrides)
    return settings, build_backend(settings)


def measure(backend: Backend, body: bytes, n: int, warmup: int, language: str = "ru") -> list[float]:
    """Длительности n вызовов конвейера в мс после warmup холостых."""
    media = parse_media_type("audio/wav")
    max_seconds = get_service_settings().MAX_AUDIO_SECONDS
    for _ in range(warmup):
        pipeline.recognize(backend, body, media, language, max_seconds)

    durations = []
    for _ in range(n):
        started = time.perf_counter()
        pipeline.recognize(backend, body, media, language, max_seconds)
        durations.append((time.perf_counter() - started) * 1000)
    return durations


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Замер задержки распознавания, JSON схемы EXP-2")
    parser.add_argument("--backend", default="fake", help="бэкенд из реестра (MODEL_BACKEND)")
    parser.add_argument("--model", default=None, help="модель бэкенда (MODEL_NAME); по умолчанию - модель бэкенда")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="устройство (TARGET_PLATFORM)")
    parser.add_argument("--n", type=int, default=50, help="число замеряемых повторов")
    parser.add_argument("--warmup", type=int, default=5, help="число холостых вызовов перед замером")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO, help="фиксированная запись WAV")
    parser.add_argument("--out", type=Path, default=None, help="файл для JSON, кроме stdout")
    args = parser.parse_args(argv)
    if args.n < 1 or args.warmup < 0:
        parser.error("--n must be at least 1 and --warmup at least 0")

    # Логи - в stderr, JSON - в stdout
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    settings, backend = build(args.backend, args.model, args.device)
    try:
        durations = measure(backend, args.audio.read_bytes(), args.n, args.warmup)
    finally:
        backend.close()

    result = BenchResult(
        service=get_project().name,
        backend=backend.name,
        model=backend.model,
        device=backend.device,
        git_sha=git_sha(settings),
        n=args.n,
        warmup=args.warmup,
        p50_ms=round(percentile(durations, 0.50), 1),
        p95_ms=round(percentile(durations, 0.95), 1),
        mean_ms=round(statistics.fmean(durations), 1),
        std_ms=round(sample_std(durations), 1),
        hardware=hardware(),
        started_at=started_at,
    )
    output = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
    print(output)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
