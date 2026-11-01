"""
Shared service layer: health, info, auth, request id, Server-Timing and errors.
"""

import json
import os
import re
import tomllib

from contextlib import ExitStack
from pathlib import Path
from typing import AsyncIterator, Iterator

import pytest

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from loguru import logger

from app.api_v1.errors import register_error_handlers
from app.api_v1.schemas import InfoResponse
from app.utils.middleware import current_timings, RequestIdMiddleware, ServerTimingMiddleware

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "tests" / "contract"

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
    }
    for name, response in responses.items():
        assert SERVER_TIMING.fullmatch(response.headers.get("Server-Timing", "")), name
        assert response.headers.get("X-Request-ID"), name

    # The same stages go to the `timings` of a body
    timings = responses["timings"]
    assert timings.headers["Server-Timing"].startswith("decode;dur=")
    assert timings.json().keys() == {"decode_ms", "total_ms"}

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
