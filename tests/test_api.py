"""
Service API: the shared layer (health, info, auth, request id, Server-Timing, errors) and /v1/recognize.
"""

import io
import json
import os
import re
import struct
import tomllib
import wave

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import AsyncIterator, Iterator, List

import numpy as np
import pytest

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from loguru import logger

from app.api_v1.errors import register_error_handlers
from app.api_v1.schemas import InfoResponse, RecognizeResponse
from app.utils.middleware import current_timings, RequestIdMiddleware, ServerTimingMiddleware

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "tests" / "contract"
SPEECH = ROOT / "tests" / "data" / "questions" / "q001.wav"     # "дверь", 1.1 s at 16 kHz mono

WAV = {"Content-Type": "audio/wav"}
L16 = {"Content-Type": "audio/L16; rate=16000; channels=1"}

GENERATED_ID = re.compile(r"[0-9a-f]{8}")
SERVER_TIMING = re.compile(r"(?:[a-z_]+;dur=\d+\.\d, )*total;dur=\d+\.\d")


@pytest.fixture(scope="module")
def probe(client: TestClient) -> Iterator[TestClient]:
    """The shared middlewares, error handlers and engine behind the routes the services add later."""
    engine = client.app.state.engine
    app = FastAPI()
    register_error_handlers(app)
    app.add_middleware(ServerTimingMiddleware)
    app.add_middleware(RequestIdMiddleware)

    @app.get("/probe/log")
    async def log_lines() -> dict:
        logger.info("probe: event loop")
        await engine.run(lambda: logger.info("probe: worker thread"))
        return {}

    @app.get("/probe/stream")
    async def stream() -> StreamingResponse:
        with current_timings().stage("prepare"):
            pass

        async def chunks() -> AsyncIterator[bytes]:
            yield b"first"
            yield b"second"

        return StreamingResponse(chunks(), media_type="audio/L16; rate=24000; channels=1")

    @app.get("/probe/timings")
    async def timings() -> dict:
        with current_timings().stage("decode"):
            pass
        return current_timings().as_dict()

    @app.get("/probe/busy")
    async def busy() -> dict:
        with engine.admit():
            return {}

    @app.get("/probe/params")
    async def params(n: int = Query(..., ge=1)) -> dict:
        return {"n": n}

    @app.get("/probe/error")
    async def error() -> dict:
        raise RuntimeError("probe failure")

    with TestClient(app, raise_server_exceptions=False) as probe_client:
        yield probe_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    # Not before the warmup
    engine = client.app.state.engine
    engine.warmed_up = False
    try:
        assert client.get("/health").status_code == 503
    finally:
        engine.warmed_up = True


def test_info(client, auth):
    response = client.get("/v1/info", headers=auth)
    assert response.status_code == 200
    info = response.json()

    # The shape of the reference example: the gateway builds its config_id of these keys
    example = json.loads((CONTRACT / "info.response.json").read_text())
    InfoResponse.model_validate(example)
    assert info.keys() == example.keys()

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert info["service"] == project["name"] == example["service"]
    assert info["version"] == project["version"]
    assert info["git_sha"] == os.environ["GIT_SHA"]
    assert (info["backend"], info["model"], info["device"]) == ("fake", "fake", "cpu")
    assert info["warmed_up"] is True
    assert info["params"]["latency_ms"] == 0


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong-token"}, {"Authorization": "Basic dGVzdC10b2tlbg=="}],
    ids=["no_token", "wrong_token", "not_bearer"],
)
def test_info_unauthorized_401(client, headers):
    response = client.get("/v1/info", headers=headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid token"}
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_request_id_echoed(client, auth, probe):
    for path, headers in (("/health", {}), ("/v1/info", auth), ("/v1/info", {})):
        response = client.get(path, headers={**headers, "X-Request-ID": "req-0001"})
        assert response.headers["X-Request-ID"] == "req-0001", path

    for body, headers in ((SPEECH.read_bytes(), auth), (b"", auth), (b"", {})):
        response = client.post("/v1/recognize", content=body, headers={**headers, **WAV, "X-Request-ID": "req-0003"})
        assert response.headers["X-Request-ID"] == "req-0003", response.status_code

    # Every log line of the request carries the id, the worker thread lines too
    records = []
    sink = logger.add(lambda message: records.append(message.record), level="DEBUG")
    try:
        response = probe.get("/probe/log", headers={"X-Request-ID": "req-0002"})
    finally:
        logger.remove(sink)

    assert response.headers["X-Request-ID"] == "req-0002"
    lines = [record for record in records if record["message"].startswith("probe:")]
    assert len(lines) == 2
    assert all(record["extra"]["request_id"] == "req-0002" for record in lines)


def test_request_id_generated(client, auth):
    ids = [client.get("/v1/info", headers=auth).headers["X-Request-ID"] for _ in range(2)]
    assert all(GENERATED_ID.fullmatch(request_id) for request_id in ids)
    assert ids[0] != ids[1]

    # An id that would break a log line is replaced as well
    for bad_id in ("has spaces in it", "x" * 200):
        response = client.get("/health", headers={"X-Request-ID": bad_id})
        assert GENERATED_ID.fullmatch(response.headers["X-Request-ID"])


def test_server_timing_header(client, auth, probe):
    responses = {
        "health": client.get("/health"),
        "info": client.get("/v1/info", headers=auth),
        "unauthorized": client.get("/v1/info"),
        "not_found": client.get("/v1/missing", headers=auth),
        "timings": probe.get("/probe/timings"),
        "stream": probe.get("/probe/stream"),
        "internal_error": probe.get("/probe/error"),
        "recognize": client.post("/v1/recognize", content=SPEECH.read_bytes(), headers={**auth, **WAV}),
        "recognize_error": client.post("/v1/recognize", content=b"", headers={**auth, **WAV}),
    }
    for name, response in responses.items():
        assert SERVER_TIMING.fullmatch(response.headers.get("Server-Timing", "")), name
        assert response.headers.get("X-Request-ID"), name

    # The same stages go to the `timings` of a body
    timings = responses["timings"]
    assert timings.headers["Server-Timing"].startswith("decode;dur=")
    assert timings.json().keys() == {"decode_ms", "total_ms"}

    # The recognition stages go to the header and to the body alike
    recognize = responses["recognize"]
    assert recognize.headers["Server-Timing"].startswith("decode;dur=")
    stages = [stage.split(";")[0] for stage in recognize.headers["Server-Timing"].split(", ")]
    assert stages == ["decode", "vad", "inference", "total"]
    assert [f"{stage}_ms" for stage in stages] == list(recognize.json()["timings"])

    # A streaming response gets the stages done before its body
    stream = responses["stream"]
    assert stream.headers["Server-Timing"].startswith("prepare;dur=")
    assert stream.content == b"firstsecond"

    # The 500 comes from outside of the middlewares and keeps the headers
    error = responses["internal_error"]
    assert error.status_code == 500
    assert error.json() == {"detail": "Internal server error"}


def test_busy_503(client, probe):
    engine = client.app.state.engine
    places = engine.settings.WORKERS + engine.settings.QUEUE_MAX

    with ExitStack() as stack:
        for _ in range(places):
            stack.enter_context(engine.admit())
        response = probe.get("/probe/busy")

    assert response.status_code == 503
    assert response.json() == {"detail": "Service is busy"}
    assert response.headers["Retry-After"] == "1"

    # The places are given back
    assert probe.get("/probe/busy").status_code == 200


def test_validation_error_422(probe):
    response = probe.get("/probe/params", params={"n": 0})
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request: Input should be greater than or equal to 1 at query.n"}


# ---------------------------------------------------------------------------
# POST /v1/recognize
# ---------------------------------------------------------------------------

def wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    """16-bit PCM WAV of float samples, (frames,) or (frames, channels)."""
    frames = samples if samples.ndim == 2 else samples[:, np.newaxis]
    pcm = np.clip(np.round(frames * 32767), -32768, 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(frames.shape[1])
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm.tobytes())
    return buffer.getvalue()


def pcm_of(wav: bytes) -> bytes:
    with wave.open(io.BytesIO(wav)) as reader:
        return reader.readframes(reader.getnframes())


@pytest.fixture
def backend_calls(client, monkeypatch) -> List[np.ndarray]:
    """Audio the backend model got in the test."""
    backend = client.app.state.engine.backend
    calls = []
    original = backend.recognize

    def spy(audio: np.ndarray, language: str) -> str:
        calls.append(audio)
        return original(audio, language)

    monkeypatch.setattr(backend, "recognize", spy)
    return calls


def assert_recognized(response, duration_ms: int) -> None:
    assert response.status_code == 200, response.text
    body = response.json()
    RecognizeResponse.model_validate(body)

    assert body["text"] == "Где дверь?"
    assert body["normalized"] == "где дверь"
    assert (body["language"], body["speech"]) == ("ru", True)
    assert body["model"] == {"backend": "fake", "name": "fake", "device": "cpu"}
    assert abs(body["audio_duration_ms"] - duration_ms) <= 1
    assert list(body["timings"]) == ["decode_ms", "vad_ms", "inference_ms", "total_ms"]
    assert body["rtf"] == round(body["timings"]["total_ms"] / body["audio_duration_ms"], 3)


def test_recognize_ok(client, auth, backend_calls):
    wav = SPEECH.read_bytes()

    requests = {
        "wav": dict(content=wav, headers={**auth, **WAV}),
        "wav_language": dict(content=wav, headers={**auth, "Content-Type": "audio/x-wav"}, params={"language": "ru"}),
        "l16": dict(content=pcm_of(wav), headers={**auth, **L16}),
        "multipart": dict(files={"file": ("q001.wav", wav, "audio/wav")}, headers=auth),
        "multipart_octet_stream": dict(files={"file": ("q001.wav", wav, "application/octet-stream")}, headers=auth),
        "multipart_l16": dict(files={"file": ("q001.pcm", pcm_of(wav), L16["Content-Type"])}, headers=auth),
    }
    for name, request in requests.items():
        backend_calls.clear()
        response = client.post("/v1/recognize", **request)
        assert_recognized(response, 1100)

        # The model gets 16 kHz mono trimmed to the speech with its pads
        assert len(backend_calls) == 1, name
        audio = backend_calls[0]
        assert audio.dtype == np.float32 and audio.ndim == 1
        assert 0.4 * 16000 < audio.shape[0] < 1.1 * 16000, name

    # Stereo at 48 kHz: the duration is that of the audio sent
    speech = np.frombuffer(pcm_of(wav), dtype="<i2").astype(np.float32) / 32768
    upsampled = np.interp(np.arange(speech.shape[0] * 3) / 3, np.arange(speech.shape[0]), speech)
    stereo = np.stack([upsampled, upsampled], axis=1)
    assert_recognized(client.post("/v1/recognize", content=wav_bytes(stereo, 48000), headers={**auth, **WAV}), 1100)


def test_recognize_empty_400(client, auth, backend_calls):
    header_only = wav_bytes(np.zeros(0, dtype=np.float32))
    cases = {
        "empty_wav": (b"", WAV, "Audio is empty"),
        "empty_l16": (b"", L16, "Audio is empty"),
        "no_frames": (header_only, WAV, "Audio is empty"),
        "not_wav": (b"definitely not a riff file", WAV, "Audio could not be decoded"),
        "truncated_header": (SPEECH.read_bytes()[:30], WAV, "Audio could not be decoded"),
        "odd_l16": (b"\x00\x01\x02", L16, "Audio could not be decoded"),
    }
    for name, (body, headers, detail) in cases.items():
        response = client.post("/v1/recognize", content=body, headers={**auth, **headers})
        assert response.status_code == 400, name
        assert response.json() == {"detail": detail}, name

    # The contract example is the corrupt audio answer
    error = json.loads((CONTRACT / "error.response.json").read_text())
    assert client.post("/v1/recognize", content=b"RIFF0000WAVEjunk", headers={**auth, **WAV}).json() == error

    response = client.post("/v1/recognize", files={"file": ("q.wav", b"", "audio/wav")}, headers=auth)
    assert response.status_code == 400
    assert backend_calls == []


def test_recognize_too_long_413(client, auth):
    limit = 4 * 1024 * 1024

    # Larger than 4 MiB: refused by Content-Length and, without it, while reading
    too_large = bytes(limit + 2)
    response = client.post("/v1/recognize", content=too_large, headers={**auth, **L16})
    assert response.status_code == 413
    assert response.json() == {"detail": f"Audio is larger than {limit} bytes"}

    def chunks() -> Iterator[bytes]:
        for _ in range(5):
            yield bytes(1024 * 1024)

    response = client.post("/v1/recognize", content=chunks(), headers={**auth, **L16})
    assert response.status_code == 413
    assert response.json() == {"detail": f"Audio is larger than {limit} bytes"}

    # Longer than MAX_AUDIO_SECONDS, though small: 31 s of 16 kHz is under 1 MiB
    for body, headers in (
        (bytes(31 * 16000 * 2), L16),
        (wav_bytes(np.zeros(31 * 8000, dtype=np.float32), 8000), WAV),
    ):
        response = client.post("/v1/recognize", content=body, headers={**auth, **headers})
        assert response.status_code == 413
        assert response.json() == {"detail": "Audio is longer than 30 seconds"}

    # 30 s exactly is taken
    response = client.post("/v1/recognize", content=bytes(30 * 16000 * 2), headers={**auth, **L16})
    assert response.status_code == 200


def float_wav(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    """IEEE float WAV: not PCM, the service does not take it."""
    data = samples.astype("<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, sample_rate, sample_rate * 4, 4, 32)
    return b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(data)) + b"WAVE" \
        + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data


def test_recognize_unsupported_media_415(client, auth):
    wav = SPEECH.read_bytes()
    cases = {
        "no_content_type": (wav, {}),
        "text": (wav, {"Content-Type": "text/plain"}),
        "json": (b"{}", {"Content-Type": "application/json"}),
        "mp3": (wav, {"Content-Type": "audio/mpeg"}),
        "l16_no_rate": (pcm_of(wav), {"Content-Type": "audio/L16"}),
        "l16_96k": (pcm_of(wav), {"Content-Type": "audio/L16; rate=96000; channels=1"}),
        "l16_six_channels": (pcm_of(wav)[:12000], {"Content-Type": "audio/L16; rate=16000; channels=6"}),
        "wav_float": (float_wav(np.zeros(1600, dtype=np.float32)), WAV),
        "wav_96k": (wav_bytes(np.zeros(9600, dtype=np.float32), 96000), WAV),
        "wav_4k": (wav_bytes(np.zeros(4000, dtype=np.float32), 4000), WAV),
        "wav_six_channels": (wav_bytes(np.zeros((1600, 6), dtype=np.float32)), WAV),
    }
    for name, (body, headers) in cases.items():
        response = client.post("/v1/recognize", content=body, headers={**auth, **headers})
        assert response.status_code == 415, name
        assert response.json()["detail"], name

    response = client.post("/v1/recognize", files={"file": ("q.mp3", wav, "audio/mpeg")}, headers=auth)
    assert response.status_code == 415
    assert response.json() == {"detail": "Content-Type must be audio/wav, audio/L16 or multipart/form-data"}


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong-token"}, {"Authorization": "Basic dGVzdC10b2tlbg=="}],
    ids=["no_token", "wrong_token", "not_bearer"],
)
def test_recognize_unauthorized_401(client, headers, backend_calls):
    response = client.post("/v1/recognize", content=SPEECH.read_bytes(), headers={**headers, **WAV})
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid token"}
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert backend_calls == []


def test_recognize_bad_params_422(client, auth, backend_calls):
    wav = SPEECH.read_bytes()
    for language in ("russian", "RU", "r", ""):
        response = client.post("/v1/recognize", content=wav, headers={**auth, **WAV}, params={"language": language})
        assert response.status_code == 422, language
        assert response.json()["detail"].startswith("Invalid request: "), language

    # A well-formed language the backend does not recognize
    response = client.post("/v1/recognize", content=wav, headers={**auth, **WAV}, params={"language": "en"})
    assert response.status_code == 422
    assert response.json() == {"detail": "Language 'en' is not supported by backend 'fake'"}

    # Multipart without the field `file`
    response = client.post("/v1/recognize", files={"audio": ("q.wav", wav, "audio/wav")}, headers=auth)
    assert response.status_code == 422
    assert response.json() == {"detail": "Multipart field 'file' is missing"}
    assert backend_calls == []


def test_recognize_busy_503(client, auth, backend_calls):
    engine = client.app.state.engine
    places = engine.settings.WORKERS + engine.settings.QUEUE_MAX

    with ExitStack() as stack:
        for _ in range(places):
            stack.enter_context(engine.admit())
        response = client.post("/v1/recognize", content=SPEECH.read_bytes(), headers={**auth, **WAV})

    assert response.status_code == 503
    assert response.json() == {"detail": "Service is busy"}
    assert response.headers["Retry-After"] == "1"
    assert backend_calls == []

    # The places are given back
    assert client.post("/v1/recognize", content=SPEECH.read_bytes(), headers={**auth, **WAV}).status_code == 200


def test_recognize_busy_503_concurrent(client, auth, monkeypatch):
    # Real requests over the capacity while the model works: the extra ones get 503 at once
    engine = client.app.state.engine
    places = engine.settings.WORKERS + engine.settings.QUEUE_MAX
    monkeypatch.setattr(engine.backend, "latency_ms", 300)

    wav = SPEECH.read_bytes()
    with ThreadPoolExecutor(max_workers=places + 2) as pool:
        futures = [
            pool.submit(client.post, "/v1/recognize", content=wav, headers={**auth, **WAV})
            for _ in range(places + 2)
        ]
        codes = sorted(future.result().status_code for future in futures)

    assert codes.count(503) >= 2
    assert set(codes) <= {200, 503}


def test_recognize_silence_returns_empty_text(client, auth, backend_calls):
    example = json.loads((CONTRACT / "recognize.silence.response.json").read_text())
    generator = np.random.default_rng(1)
    silences = {
        "zeros_wav": (wav_bytes(np.zeros(24000, dtype=np.float32)), WAV),
        "zeros_l16": (bytes(24000 * 2), L16),
        "quiet_noise": (wav_bytes(0.003 * generator.standard_normal(24000).astype(np.float32)), WAV),
    }
    for name, (body, headers) in silences.items():
        response = client.post("/v1/recognize", content=body, headers={**auth, **headers})
        assert response.status_code == 200, name
        result = response.json()
        RecognizeResponse.model_validate(result)

        # The shape of the contract example: no inference stage
        assert result.keys() == example.keys()
        assert (result["text"], result["normalized"], result["speech"]) == ("", "", False), name
        assert result["timings"].keys() == example["timings"].keys(), name
        assert result["audio_duration_ms"] == 1500

    # The model is not called on silence
    assert backend_calls == []
