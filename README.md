# asr-service

Сервис распознавания речи голосового ассистента для людей с нарушениями зрения. Принимает запись вопроса пользователя и возвращает текст и его нормализованную форму. Перед моделью запись приводится к 16 кГц моно, а тишина по краям обрезается детектором речи; если речи нет, модель не вызывается. Сервис не хранит записи и не разбирает смысл вопроса: намерение определяет Шлюз (`assistant-low-vision-platform`), он же - единственный потребитель API. Сейчас распознавание работает на бэкенде `fake`; настоящие модели добавляются следующим шагом. Требования - в [docs/spec.md](docs/spec.md).

## Назначение

Детектор, ASR и TTS устроены одинаково (раздел SVC спецификации), поэтому Шлюз работает с любым бэкендом любого из них одинаково:

- модель выбирается настройками `MODEL_BACKEND` и `MODEL_NAME`, загружается и прогревается при старте (3 холостых вызова), `/health` отвечает только после прогрева;
- инференс идёт в пуле из `WORKERS` потоков, ещё `QUEUE_MAX` запросов могут ждать; следующий запрос сразу получает 503: устаревшее описание сцены пользователю бесполезно;
- вход читается в память, временных файлов нет;
- у каждого запроса есть идентификатор `X-Request-ID`, он попадает в каждую строку лога о запросе;
- каждый ответ, включая потоковый, несёт заголовок `Server-Timing` с временем этапов;
- `/v1/info` описывает конфигурацию (сервис, версия, коммит, бэкенд, модель, устройство), Шлюз включает её в свой `config_id`.

Распознавание `POST /v1/recognize` (ASR-1, ASR-2, ASR-4):

1. Тело читается в память с проверкой лимита 4 МиБ по ходу чтения.
2. WAV, `audio/L16` или файл из multipart декодируется, проверяется длительность (`MAX_AUDIO_SECONDS`), каналы усредняются в моно, частота приводится к 16 кГц.
3. Детектор речи Silero VAD (ONNX на onnxruntime, без torch) обрезает тишину по краям с запасом 100 мс. Если речи нет, ответ - `text=""` и `speech=false`, модель не вызывается: это быстрее и защищает от выдуманного текста на тишине.
4. Модель распознаёт обрезанную запись.
5. Текст нормализуется так же, как в Шлюзе (PLT-4): нижний регистр, `ё` → `е`, знаки препинания (категории Unicode `P*`) заменяются пробелом, символы (`S*`, например `№`) остаются, пробелы схлопываются. Правило закреплено таблицей `tests/contract/normalize.cases.json`, общей с Шлюзом.

## API

| Метод | Путь | |
| --- | --- | --- |
| `GET` | `/health` | `{"status": "ok"}`, когда модель загружена и прогрета |
| `GET` | `/v1/info` | конфигурация сервиса для `config_id` Шлюза |
| `POST` | `/v1/recognize` | текст речи в записи |
| `GET` | `/docs` | Swagger UI |

Запросы к `/v1/*` требуют `Authorization: Bearer <AUTH_TOKEN>`; если `AUTH_TOKEN` не задан, аутентификация выключена.

Заголовок `X-Request-ID` запроса возвращается в ответе; без него сервис создаёт свой (8 hex-символов). Значение с пробелами, управляющими символами или длиннее 128 символов заменяется новым: оно пишется в каждую строку лога.

Заголовок `Server-Timing` перечисляет этапы, известные до начала тела ответа, и `total` - время до ответа: `decode;dur=2.1, inference;dur=11.6, total;dur=15.0`.

```bash
curl -H "Authorization: Bearer $AUTH_TOKEN" -H "X-Request-ID: 3f2c1ab0" \
  "http://localhost:8102/v1/info"
```

```json
{
  "service": "asr-service",
  "version": "0.1.0",
  "git_sha": "3f2c1ab",
  "backend": "fake",
  "model": "fake",
  "device": "cpu",
  "warmed_up": true,
  "params": {"language": "ru", "sample_rate": 16000, "vad_enabled": true, "latency_ms": 0}
}
```

| Поле | |
| --- | --- |
| `service`, `version` | имя и версия из `pyproject.toml` |
| `git_sha` | коммит, из которого собран образ (аргумент сборки `GIT_SHA`) |
| `backend`, `model` | `MODEL_BACKEND` и модель бэкенда (`MODEL_NAME` или модель по умолчанию) |
| `device` | `TARGET_PLATFORM`: `cpu` или `cuda` |
| `warmed_up` | модель прошла холостые вызовы |
| `params` | параметры, от которых зависит ответ: язык по умолчанию, частота входа модели, включён ли детектор речи; у `fake` ещё `latency_ms` |

### POST /v1/recognize

Тело - одно из:

| `Content-Type` | Тело |
| --- | --- |
| `audio/wav` (также `audio/x-wav`, `audio/wave`, `audio/vnd.wave`) | WAV PCM 8, 16, 24 или 32 бит, 8-48 кГц, моно или стерео |
| `audio/L16; rate=16000; channels=1` | PCM s16le без заголовка, так шлёт Шлюз; `rate` 8000-48000 обязателен, `channels` 1 или 2, по умолчанию 1 |
| `multipart/form-data` | поле `file` с WAV или `audio/L16`; файл с типом `application/octet-stream` или без типа читается как WAV |

Параметр запроса `language` - код языка ISO 639-1, по умолчанию `ru`. Бэкенд принимает только свои языки, у `fake` - `ru`.

```bash
# WAV
curl -H "Authorization: Bearer $AUTH_TOKEN" -H "Content-Type: audio/wav" \
  --data-binary @tests/data/questions/q001.wav "http://localhost:8102/v1/recognize?language=ru"

# PCM s16le 16 кГц моно, как шлёт Шлюз
curl -H "Authorization: Bearer $AUTH_TOKEN" -H "Content-Type: audio/L16; rate=16000; channels=1" \
  --data-binary @question.pcm "http://localhost:8102/v1/recognize"

# multipart
curl -H "Authorization: Bearer $AUTH_TOKEN" -F "file=@tests/data/questions/q001.wav" \
  "http://localhost:8102/v1/recognize"
```

```json
{
  "text": "Где дверь?",
  "normalized": "где дверь",
  "language": "ru",
  "speech": true,
  "audio_duration_ms": 1830,
  "model": {"backend": "gigaam", "name": "v2_ctc", "device": "cuda"},
  "timings": {"decode_ms": 1.2, "vad_ms": 4.8, "inference_ms": 38.5, "total_ms": 46.0},
  "rtf": 0.025
}
```

Запись без речи - тоже `200`, без этапа `inference_ms`:

```json
{
  "text": "",
  "normalized": "",
  "language": "ru",
  "speech": false,
  "audio_duration_ms": 1500,
  "model": {"backend": "gigaam", "name": "v2_ctc", "device": "cuda"},
  "timings": {"decode_ms": 0.9, "vad_ms": 4.4, "total_ms": 5.6},
  "rtf": 0.004
}
```

| Поле | |
| --- | --- |
| `text` | текст, как его выдала модель; пустой, если речи нет |
| `normalized` | `text` после нормализации ASR-4 |
| `language` | язык распознавания |
| `speech` | детектор нашёл речь; `false` - модель не вызывалась |
| `audio_duration_ms` | длительность присланной записи, до обрезки тишины |
| `model` | бэкенд, модель и устройство |
| `timings` | этапы в мс: `decode_ms` (разбор тела, моно, 16 кГц), `vad_ms`, `inference_ms`, `total_ms` (с прихода запроса); те же этапы - в `Server-Timing` |
| `rtf` | `total_ms`, делённое на длительность записи |

Примеры ответов лежат в `tests/contract/` и совпадают с копией в Шлюзе.

## Коды ответов

Ошибки приходят одним предложением на английском: `{"detail": "<причина>"}`.

| Код | Когда |
| --- | --- |
| `200` | успех |
| `400` | запись пустая (`Audio is empty`) или битая (`Audio could not be decoded`) |
| `401` | нет токена или неверный токен: `{"detail": "Invalid token"}` |
| `413` | запись длиннее `MAX_AUDIO_SECONDS` (`Audio is longer than 30 seconds`) или тело больше 4 МиБ, 4 194 304 байт (`Audio is larger than 4194304 bytes`) |
| `415` | `Content-Type` не `audio/wav`, `audio/L16` или `multipart/form-data`, у `audio/L16` нет `rate`, WAV не PCM, частота вне 8-48 кГц, больше двух каналов |
| `422` | параметры не прошли проверку: `language` не код из двух строчных букв (`{"detail": "Invalid request: <что не так> at <где>"}`), язык не поддерживается бэкендом, в multipart нет поля `file` |
| `500` | внутренняя ошибка: `{"detail": "Internal server error"}`, подробности только в логе |
| `503` | очередь заполнена: `{"detail": "Service is busy"}` и `Retry-After: 1`; у `/health` - модель ещё не прогрета |

Коды `400`, `413`, `415` и `422` относятся к эндпоинту распознавания.

## Конфигурация

Все переменные необязательные, значения по умолчанию - в таблице и в [.env.example](.env.example).

| Переменная | По умолчанию | |
| --- | --- | --- |
| `MODEL_BACKEND` | `fake` | бэкенд из реестра (раздел «Бэкенды и модели») |
| `MODEL_NAME` | - | модель бэкенда (раздел «Бэкенды и модели»); не задана - модель бэкенда по умолчанию |
| `TARGET_PLATFORM` | `cpu` | `cpu` или `cuda` |
| `ARTIFACTS_PATH` | - | каталог весов: бэкенд берёт их из `<ARTIFACTS_PATH>/<бэкенд>/` и докачивает недостающие; не задан - веса скачиваются в кэш библиотеки |
| `FAKE_LATENCY_MS` | `0` | время инференса, которое изображает `fake`, мс на вызов |
| `FAKE_TEXT` | `Где дверь?` | текст, который `fake` «распознаёт» в любой речи |
| `VAD_ENABLED` | `True` | обрезка тишины детектором речи; без речи - пустой текст без вызова модели |
| `MAX_AUDIO_SECONDS` | `30` | самая длинная запись, с; длиннее - 413 |
| `AUTH_TOKEN` | - | Bearer-токен; не задан - аутентификация выключена |
| `WORKERS` | `1` | потоков инференса |
| `QUEUE_MAX` | `4` | запросов в ожидании потока; сверх этого - 503 |
| `GIT_SHA` | `unknown` | коммит для `/v1/info`; задаёт аргумент сборки образа, в `.env` не пишется |
| `LOG_LEVEL` | `INFO` | `TRACE`, `DEBUG`, `INFO`, `SUCCESS`, `WARNING`, `ERROR`, `CRITICAL` |
| `LOG_DIAGNOSE` | `False` | значения переменных в трейсбеках; только для отладки - в логи попадут данные запросов и токены |

Пороги моделей и эвристик - не переменные окружения, а константы в `app/core/config/constants.py`: лимит тела 4 МиБ, диапазон частот и каналов, пороги детектора речи (вероятность 0,3, речь не короче 250 мс, пауза 100 мс, запас 100 мс по краям).

**Порог детектора речи - 0,3, а не 0,5 (значение Silero по умолчанию).** Порог общий для всех бэкендов. Замер 2026-11-22 на T-one (CPU, манифест `tests/data/questions.jsonl`, `bench.wer.evaluate`): при 0,5 начало речи в записи `q004` с уличным шумом находится на 0,54 с, звук «о» (вероятность речи 0,39-0,42 на 0,35-0,45 с) отрезается, и «окно» распознаётся как «но» - WER 0,25 против 0,0 без детектора. При 0,3 начало речи - 0,35 с, WER 0,0, как без детектора. Порог 0,3 взят из проекта автора, где он проверен на его данных; он оставляет наибольший запас на шуме. Шум до начала речи (первые 0,3 с `q003` и `q004`) детектор по-прежнему речью не считает: вероятность речи там не выше 0,02. Остальные пороги - значения Silero: ограничение длины речи не нужно, длину вопроса ограничивают Шлюз и `MAX_AUDIO_SECONDS`, а длинная пауза на обрезку от первой до последней речи почти не влияет. Оба случая проверяют тесты `tests/test_model_vad.py` (маркер `model`).

Настройки распознавания (`VAD_ENABLED`, `MAX_AUDIO_SECONDS`, `FAKE_TEXT`) читает свой класс `ServiceSettings` (`app/core/config/service.py`) из того же окружения и `.env`: класс `Settings` общий для трёх сервисов ИИ и одинаков в них.

## Бэкенды и модели

Бэкенд выбирается `MODEL_BACKEND`. Реестр (`app/core/backends/registry.py`) импортирует модуль только выбранного бэкенда, поэтому библиотеки настоящих моделей - необязательные extras: без них сервис устанавливается и проходит быстрые тесты. Неизвестный бэкенд, неизвестная модель или бэкенд без своего extra останавливают старт с понятным сообщением, например `Backend 'whisper' requires the 'whisper' extra`.

| `MODEL_BACKEND` | Extra | `MODEL_NAME` (по умолчанию - первая) | Устройство | Заметки |
| --- | --- | --- | --- | --- |
| `fake` | - | `fake` | - | текст `FAKE_TEXT` на любую речь, без весов и видеокарты |
| `gigaam` | `gigaam` + профиль `cpu` или `gpu` (torch, torchaudio) | `v3_ctc`, `v3_rnnt`, `v3_e2e_ctc`, `v3_e2e_rnnt`, `v2_ctc`, `v2_rnnt` | `cpu`, `cuda` (энкодер в FP16) | GigaAM, Conformer для русского языка; `e2e` - с пунктуацией |
| `tone` | - (ядро, onnxruntime) | `t-one` | `cpu`; `cuda` - если в onnxruntime есть CUDA-провайдер, иначе `cpu` с предупреждением | T-one, потоковая CTC-модель 8 кГц; запрос распознаётся целиком, поблочно по 300 мс, жадное декодирование без языковой модели |
| `whisper` | `whisper` | `small`, `medium`, `large-v3-turbo` | `cpu` (int8), `cuda` (float16) | faster-whisper (CTranslate2), `language=ru`, `beam_size=1`, без контекста прошлых фраз и без своего VAD |
| `vosk` | `vosk` | `small-ru` (`vosk-model-small-ru-0.22`, 45 МБ), `ru` (`vosk-model-ru-0.42`, 1,8 ГБ) | только `cpu`; при `cuda` - предупреждение | Kaldi; распознаватель создаётся на каждый запрос, модель общая |

T-one работает на onnxruntime из ядра, поэтому своего extra у него нет: это CPU-модель смоук-теста AC-68. Пакет авторов `tone` не используется: он тянет pyctcdecode (numpy<2) и KenLM, который собирается из исходников. Модель ONNX сама считает признаки, а жадное CTC-декодирование - то же, что в пакете авторов. Версия весов закреплена в `app/core/config/constants.py` (`TONE_SETTINGS.REVISION`).

GigaAM ставится из коммита репозитория авторов, а не с PyPI: релиз 0.1.0 на PyPI требует `torch<=2.5.1`. torchaudio собирается под одну версию torch, последняя версия torchaudio - 2.11, поэтому профили `cpu` и `gpu` держат torch 2.11. Своя версия onnxruntime GigaAM (1.23, для экспорта в ONNX, сервису не нужен) переопределена версией ядра в `[tool.uv] override-dependencies`.

Установка extras:

```bash
uv sync --extra whisper                    # Whisper
uv sync --extra vosk                       # Vosk
uv sync --extra cpu --extra gigaam         # GigaAM на CPU
```

На `fake` сервис пишет предупреждение при старте, а `/v1/info` показывает `backend: fake`, поэтому подмена модели в замерах сразу видна.

**Веса.** В образ веса не входят. При старте бэкенд берёт их из `ARTIFACTS_PATH/<бэкенд>/` и докачивает, если их там нет. Без `ARTIFACTS_PATH` веса загружаются в кэш библиотеки: `~/.cache/huggingface` (T-one, Whisper), `~/.cache/gigaam`, `~/.cache/vosk`. Заранее, для старта без сети, их скачивает `scripts/fetch_weights.py`: модель он не загружает в память. Этот скрипт вызывает `deployments/fetch-weights.sh` из образа сервиса.

```bash
MODEL_BACKEND=whisper MODEL_NAME=small ARTIFACTS_PATH=./model uv run python -m scripts.fetch_weights
# fetch_weights: whisper/small -> model/whisper/small
```

| Источник весов | Где лежат |
| --- | --- |
| T-one | Hugging Face [`t-tech/T-one`](https://huggingface.co/t-tech/T-one), `model.onnx` (~140 МБ) |
| Whisper | Hugging Face: [`Systran/faster-whisper-small`](https://huggingface.co/Systran/faster-whisper-small), [`Systran/faster-whisper-medium`](https://huggingface.co/Systran/faster-whisper-medium), [`mobiuslabsgmbh/faster-whisper-large-v3-turbo`](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo) |
| GigaAM | CDN авторов `cdn.chatwm.opensmodel.sberdevices.ru/GigaAM` (адрес задан в пакете `gigaam`) |
| Vosk | [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) |

Лицензии сверены по первоисточникам (файлы `LICENSE` репозиториев, карточки моделей на Hugging Face, список моделей Vosk) 2026-11-17. Бэкендов под GPL и AGPL и весов с некоммерческой лицензией среди них нет; библиотеки настоящих моделей всё равно подключаются только как extras.

| Модель | Лицензия | Коммерческое использование |
| --- | --- | --- |
| `fake` (код сервиса, без весов) | Apache-2.0 | да |
| Silero VAD v6.2.3, `app/core/vad/silero_vad.onnx` (детектор речи, входит в ядро) | MIT ([app/core/vad/LICENSE](app/core/vad/LICENSE)) | да |
| T-one, веса [`t-tech/T-one`](https://huggingface.co/t-tech/T-one) | Apache-2.0 (карточка модели и [LICENSE](https://github.com/voicekit-team/T-one/blob/main/LICENSE) репозитория) | да |
| GigaAM v2 и v3, библиотека [`gigaam`](https://github.com/salute-developers/GigaAM) и веса | MIT ([LICENSE](https://github.com/salute-developers/GigaAM/blob/main/LICENSE), карточка [`ai-sage/GigaAM-v3`](https://huggingface.co/ai-sage/GigaAM-v3)) | да |
| Whisper small, medium, large-v3-turbo (веса OpenAI в формате CTranslate2) | MIT (карточки моделей на Hugging Face, [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE) OpenAI) | да |
| faster-whisper и CTranslate2 | MIT ([faster-whisper](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE), [CTranslate2](https://github.com/OpenNMT/CTranslate2/blob/master/LICENSE)) | да |
| Vosk: библиотека и модели `vosk-model-small-ru-0.22`, `vosk-model-ru-0.42` | Apache-2.0 ([COPYING](https://github.com/alphacep/vosk-api/blob/master/COPYING), [список моделей](https://alphacephei.com/vosk/models)) | да |
| torch, torchaudio (профили `cpu`, `gpu`) | BSD-3-Clause | да |

## Запуск

Порт на хосте - `127.0.0.1:8102`, сеть Docker `alv-net` compose создаёт сам. Каталог `./model` монтируется в `/opt/model` под веса.

На CPU:

```bash
cp .env.example .env
docker compose -f compose-cpu.yml up -d --build
curl http://localhost:8102/health
```

На GPU (нужен NVIDIA Container Toolkit; в `.env` - `TARGET_PLATFORM=cuda`):

```bash
docker compose up -d --build
```

Аргументы сборки берутся из окружения: `GIT_SHA=$(git rev-parse --short HEAD)` попадает в `/v1/info`, `EXTRAS` задаёт extras образа (по умолчанию профиль `cpu` или `gpu` и все бэкенды: `cpu gigaam whisper vosk`; `EXTRAS=""` - лёгкий образ с `fake` и `tone`). Настоящая модель задаётся в `.env` (`MODEL_BACKEND`, `MODEL_NAME`); веса при этом скачиваются в `./model`, если задан `ARTIFACTS_PATH=/opt/model`, иначе - в кэш внутри контейнера.

Локально:

```bash
uv sync
uv run fastapi dev app/main.py
uv run pytest -q -m "not model"
```

Тесты с маркером `model` нужны весам и в быстрый прогон не входят. Смоук настоящей модели (AC-68) - T-one на CPU, extras ему не нужны, нужна сеть или кэш с весами:

```bash
uv run pytest tests/test_model_smoke.py -q -m model
``` CI (GitHub Actions) прогоняет быстрые тесты на каждый пуш и PR, а на пуш в `dev`, `main` и тег `v*` публикует образы в `ghcr.io/nikarpoff/asr-service` с тегами `latest` (ветка `main`), `sha-<7>` и `v<semver>` и суффиксом варианта: `latest-cpu`, `v0.1.0-cuda`. На `dev` собирается только `-cpu`.

### Устройство

```
app/
├── main.py                 # настройки -> логирование -> FastAPI(lifespan) -> /health -> middleware -> роутеры
├── api_v1/
│   ├── views.py            # /v1/info, /v1/recognize, чтение тела с лимитом
│   ├── schemas.py          # контракт API (pydantic)
│   └── errors.py           # AppError, 422 и 500 -> {"detail": ...}
├── core/
│   ├── config/             # Settings (env, общий), ServiceSettings (env распознавания), версия, константы
│   ├── auth/               # Bearer-токен
│   ├── audio/              # WAV, L16 и multipart -> отсчёты; моно и 16 кГц
│   ├── vad/                # Silero VAD на onnxruntime, веса и лицензия MIT
│   ├── text/               # нормализация ASR-4
│   ├── backends/           # реестр, базовый класс с recognize() и детектором речи; fake, gigaam, tone, whisper, vosk
│   ├── pipeline.py         # запрос: декодирование -> VAD -> модель -> нормализация, этапы timings
│   ├── entities.py         # результат распознавания
│   ├── engine.py           # бэкенд в пуле WORKERS, очередь QUEUE_MAX, прогрев
│   └── exceptions.py       # доменные ошибки со status_code
└── utils/                  # lifespan, логирование, middleware X-Request-ID и Server-Timing
scripts/
└── fetch_weights.py        # веса MODEL_BACKEND/MODEL_NAME заранее, в ARTIFACTS_PATH или кэш
bench/
├── latency.py              # замер задержки, JSON EXP-2
└── wer.py                  # WER и CER по манифесту вопросов
tests/
├── contract/               # эталонные примеры из Шлюза; правятся только синхронизацией
├── data/                   # игрушечный манифест вопросов и записи, источники в SOURCES.md
└── test_*.py
```

## Замеры

Время этапов - в заголовке `Server-Timing` каждого ответа, а у ответа распознавания ещё и в поле `timings`. Сами замеры и сравнение моделей - задача `002-hypothesis-experiment`; скрипты пишут JSON в stdout и, если задан `--out`, в файл.

**Задержка** (SVC-10) - `bench/latency.py`, в процессе, без HTTP: прогрев `--warmup` вызовов, затем `--n` повторов полного конвейера запроса (декодирование, 16 кГц моно, детектор речи, модель, нормализация) на фиксированной записи `--audio` (по умолчанию `tests/data/questions/q001.wav`). Перцентили - nearest-rank, стандартное отклонение - выборочное.

```bash
uv run python -m bench.latency --backend gigaam --model v2_ctc --device cuda --n 50 --warmup 5 --out latency.json
```

```json
{"service": "asr-service", "backend": "gigaam", "model": "v2_ctc", "device": "cuda", "git_sha": "3f2c1ab", "n": 50, "warmup": 5, "p50_ms": 41.8, "p95_ms": 52.3, "mean_ms": 43.1, "std_ms": 4.6, "hardware": {"cpu": "AMD Ryzen 7 5700X", "gpu": "NVIDIA GeForce RTX 3070"}, "started_at": "2026-11-01T10:00:00Z"}
```

`git_sha` берётся из `GIT_SHA`, а без него - из текущего коммита рабочей копии.

**Качество** (ASR-7) - `bench/wer.py` по манифесту вопросов EXP-1 (`questions.jsonl`, путь к аудио - от каталога манифеста). Каждая запись проходит тот же конвейер, что запрос к сервису, эталон и гипотеза нормализуются по ASR-4. WER - правки слов на число слов эталона, CER - правки символов (с пробелами между словами) на число символов эталона; ошибки суммируются по условию (микроусреднение) отдельно для `quiet` и `street_noise` и по всему набору.

```bash
FAKE_TEXT=дверь uv run python -m bench.wer --manifest tests/data/questions.jsonl --backend fake --out wer.json
```

```json
{
  "service": "asr-service", "backend": "fake", "model": "fake", "device": "cpu", "git_sha": "3f2c1ab",
  "manifest": "tests/data/questions.jsonl", "vad_enabled": true, "started_at": "2026-11-01T10:00:00Z",
  "conditions": {
    "quiet": {"n": 2, "words": 2, "word_errors": 1, "wer": 0.5, "chars": 9, "char_errors": 5, "cer": 0.5556},
    "street_noise": {"n": 2, "words": 2, "word_errors": 2, "wer": 1.0, "chars": 7, "char_errors": 9, "cer": 1.2857}
  },
  "overall": {"n": 4, "words": 4, "word_errors": 3, "wer": 0.75, "chars": 16, "char_errors": 14, "cer": 0.875},
  "items": [{"id": "q001", "condition": "quiet", "reference": "дверь", "hypothesis": "дверь", "words": 1, "word_errors": 0, "chars": 5, "char_errors": 0}]
}
```

Если условия нет в манифесте, его `wer` и `cer` - `null`. В `items` - эталон и гипотеза каждой записи; набор EXP-1 хранится вне git, поэтому и отчёты по нему в git не попадают.
