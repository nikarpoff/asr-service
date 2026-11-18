"""
WER и CER распознавания по манифесту вопросов EXP-1 (ASR-7).

Каждая запись проходит тот же конвейер, что и запрос к сервису: декодирование → 16 кГц моно →
детектор речи → модель. Эталон и гипотеза нормализуются по ASR-4 (EXP-4). Ошибки
суммируются по условию (микроусреднение): WER - правки слов на число слов эталона, CER -
правки символов, пробелы между словами включены, на число символов эталона.

    uv run python -m bench.wer --manifest <questions.jsonl> --backend gigaam --out wer.json
"""

import argparse
import json
import sys

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from loguru import logger

from app.core import pipeline
from app.core.audio import parse_media_type
from app.core.backends import Backend
from app.core.config import get_service_settings, get_project
from app.core.text import normalize
from bench.latency import build, git_sha

CONDITIONS = ("quiet", "street_noise")


def edit_distance(reference: Sequence, hypothesis: Sequence) -> int:
    """Расстояние Левенштейна: замены, вставки и удаления стоят по 1."""
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_item != hyp_item)))
        previous = current
    return previous[-1]


def score_item(reference: str, hypothesis: str) -> dict:
    """Ошибки одной пары после нормализации ASR-4."""
    reference, hypothesis = normalize(reference), normalize(hypothesis)
    return {
        "reference": reference,
        "hypothesis": hypothesis,
        "words": len(reference.split()),
        "word_errors": edit_distance(reference.split(), hypothesis.split()),
        "chars": len(reference),
        "char_errors": edit_distance(reference, hypothesis),
    }


def summarize(items: list[dict]) -> dict:
    """Итог по набору пар: микроусреднение; null, если эталон пуст."""
    words = sum(item["words"] for item in items)
    chars = sum(item["chars"] for item in items)
    word_errors = sum(item["word_errors"] for item in items)
    char_errors = sum(item["char_errors"] for item in items)
    return {
        "n": len(items),
        "words": words,
        "word_errors": word_errors,
        "wer": round(word_errors / words, 4) if words else None,
        "chars": chars,
        "char_errors": char_errors,
        "cer": round(char_errors / chars, 4) if chars else None,
    }


def read_manifest(path: Path) -> list[dict]:
    """Строки questions.jsonl; путь к аудио - от каталога манифеста."""
    entries = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        entry = json.loads(line)
        missing = {"id", "audio", "text", "condition"} - entry.keys()
        if missing:
            raise ValueError(f"{path}:{number}: no fields {sorted(missing)}")
        if entry["condition"] not in CONDITIONS:
            raise ValueError(f"{path}:{number}: condition must be one of {CONDITIONS}, got '{entry['condition']}'")
        entries.append(entry)
    return entries


def evaluate(backend: Backend, manifest: Path, language: str = "ru") -> dict:
    """Распознаёт все записи манифеста и считает WER и CER по условиям."""
    media = parse_media_type("audio/wav")
    max_seconds = get_service_settings().MAX_AUDIO_SECONDS

    items = []
    for entry in read_manifest(manifest):
        body = (manifest.parent / entry["audio"]).read_bytes()
        recognition = pipeline.recognize(backend, body, media, language, max_seconds)
        items.append({"id": entry["id"], "condition": entry["condition"], **score_item(entry["text"], recognition.text)})

    return {
        "conditions": {
            condition: summarize([item for item in items if item["condition"] == condition])
            for condition in CONDITIONS
        },
        "overall": summarize(items),
        "items": items,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="WER и CER распознавания по манифесту вопросов EXP-1")
    parser.add_argument("--manifest", type=Path, required=True, help="questions.jsonl; пути к аудио - от его каталога")
    parser.add_argument("--backend", default="fake", help="бэкенд из реестра (MODEL_BACKEND)")
    parser.add_argument("--model", default=None, help="модель бэкенда (MODEL_NAME); по умолчанию - модель бэкенда")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="устройство (TARGET_PLATFORM)")
    parser.add_argument("--out", type=Path, default=None, help="файл для JSON, кроме stdout")
    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    settings, backend = build(args.backend, args.model, args.device)
    try:
        scores = evaluate(backend, args.manifest)
    finally:
        backend.close()

    result = {
        "service": get_project().name,
        "backend": backend.name,
        "model": backend.model,
        "device": backend.device,
        "git_sha": git_sha(settings),
        "manifest": str(args.manifest),
        "vad_enabled": backend.vad is not None,
        "started_at": started_at,
        **scores,
    }
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
