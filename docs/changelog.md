# Changelog

### v0.1.0
- Добавлены копии эталонных контрактов из репозитория Шлюза в `tests/contract/`.
- Добавлена выдержка из общей спецификации в `docs/spec.md`.
- Добавлен каркас сервиса по `ainergy-python-style` (`--ml`): FastAPI, uv, Python 3.12 (`requires-python = ">=3.12,<3.13"`, `.python-version`), профили `cpu` и `gpu` для torch.
- Добавлен общий слой сервисов ИИ (SVC): Bearer-аутентификация, ошибки `{"detail": ...}`, заголовки `X-Request-ID` и `Server-Timing` на чистых ASGI-middleware, `request_id` в каждой строке лога, движок с пулом `WORKERS` и очередью `QUEUE_MAX` (сверх неё - 503 с `Retry-After: 1`), реестр бэкендов с ленивым импортом и бэкендом `fake` (`FAKE_LATENCY_MS`), прогрев тремя холостыми вызовами, `/health` и `/v1/info`.
- Добавлены образы `Dockerfile` (CUDA) и `Dockerfile.cpu` с аргументами `EXTRAS` и `GIT_SHA`, `compose.yml` и `compose-cpu.yml` на `127.0.0.1:8102` в сети `alv-net`, `.env.example`.
- Добавлен CI в GitHub Actions: быстрые тесты и публикация образов в GHCR.
- README переписан на русском по разделам SVC-11.
- Добавлен эндпоинт `POST /v1/recognize` (ASR-1): тела `audio/wav` (PCM 8-32 бит, 8-48 кГц, моно и стерео), `audio/L16` и `multipart/form-data` с полем `file`, параметр `language`, коды 400, 413 (4 МиБ или `MAX_AUDIO_SECONDS`), 415 и 422; тело читалось в память с проверкой лимита по ходу чтения.
- Добавлено приведение записи к 16 кГц моно и детектор речи Silero VAD v6.2.3 в формате ONNX на onnxruntime, без torch (ASR-2): без речи ответ стал `200` с `text=""` и `speech=false` без вызова модели, а `timings` - без `inference_ms`.
- Добавлена нормализация текста ASR-4 по общей с Шлюзом таблице `tests/contract/normalize.cases.json`.
- Бэкенд `fake` стал отвечать текстом `FAKE_TEXT` (по умолчанию «Где дверь?»); добавлены настройки `VAD_ENABLED`, `MAX_AUDIO_SECONDS` и `FAKE_TEXT` в отдельном классе `ServiceSettings`.
- Добавлены скрипты `bench/latency.py` (JSON EXP-2) и `bench/wer.py` (WER и CER по условиям `quiet` и `street_noise`), игрушечный манифест вопросов `tests/data/` с источниками записей в `SOURCES.md`.
- Добавлены тесты API распознавания, звука, нормализации, контракта и WER; README дополнен разделами о распознавании и замерах.
