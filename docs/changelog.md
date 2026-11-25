# Changelog

### v0.2.0
- Порог детектора речи `VAD_SETTINGS.THRESHOLD` снижен с 0,5 до 0,3: на уличном шуме детектор больше не отрезает начало слова (`q004` - «окно», а не «но»), а шум без речи речью по-прежнему не считается. Добавлены тесты `tests/test_model_vad.py` (маркер `model`), README описывает порог и замер.
- Добавлен бэкенд `gigaam_onnx`: GigaAM-v3 `v3_ctc` и `v3_rnnt` в ONNX на onnx-asr 0.12 (extra `gigaam_onnx`), веса `istupakov/gigaam-v3-onnx` на закреплённой ревизии, только файлы полной точности. `fetch_weights` скачивает их в `ARTIFACTS_PATH/gigaam_onnx/` или в кэш Hugging Face, сервис загружает модель из этого каталога без сети. На `cuda` откат onnxruntime на CPU останавливает старт ошибкой; `/v1/info` показывает провайдеры сессий и ревизию.
- Профиль `gpu` ставит `onnxruntime-gpu` той же версии, что `onnxruntime` ядра (оба - 1.32.0); `scripts/use-onnxruntime-gpu.sh` после `uv sync` заменяет CPU-сборку на CUDA-сборку. CUDA-образ вызывает его и по умолчанию собирается с `gigaam_onnx` (`EXTRAS="gpu gigaam gigaam_onnx whisper vosk"`).
- Добавлены тесты машины моделей `tests/test_model_machine.py`: GigaAM-v3 в ONNX и Whisper `medium`, `large-v3-turbo` на CUDA, провайдер CUDA у `gigaam_onnx`, Vosk `ru` и `gigaam_onnx` на CPU. Без видеокарты тесты на CUDA падают за секунды, а не пропускаются. README описывает среду и команды.

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
- Добавлены настоящие бэкенды ASR-3: `gigaam` (GigaAM v2 и v3, CTC и RNN-T, FP16-энкодер на CUDA), `tone` (T-one на onnxruntime ядра, без extra), `whisper` (faster-whisper, `language=ru`, `beam_size=1`, int8 на CPU) и `vosk` (`small-ru`, `ru`); extras `gigaam`, `whisper`, `vosk` с ленивым импортом.
- Профили `cpu` и `gpu` перешли на torch 2.11 с torchaudio 2.11: последний torchaudio собран под эту версию torch, а он нужен GigaAM. GigaAM ставится из коммита репозитория авторов.
- Добавлен `scripts/fetch_weights.py`: веса `MODEL_BACKEND`/`MODEL_NAME` скачиваются заранее в `ARTIFACTS_PATH/<бэкенд>/` или в кэш библиотеки; оттуда их берёт сервис при старте.
- Образы по умолчанию собираются со всеми бэкендами профиля (`EXTRAS="cpu gigaam whisper vosk"` и `"gpu gigaam whisper vosk"`), веса в образ не входят.
- Добавлены смоук-тест настоящей CPU-модели (`tests/test_model_smoke.py`, маркер `model`, T-one, WER не больше 30 %) и проверки реестра: у каждого extra-бэкенда есть свой extra и модуль.
- README: таблица бэкендов и моделей, источники весов, таблица лицензий, сверенных по первоисточникам.
