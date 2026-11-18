"""
Request context: the request id and the stage timings.

Both middlewares are pure ASGI. BaseHTTPMiddleware would buffer a streaming response,
and the gateway would get the synthesized speech only at its end.
"""

import re
import time
import uuid

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Iterator, Optional

from loguru import logger
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = "X-Request-ID"
SERVER_TIMING_HEADER = "Server-Timing"
NO_REQUEST_ID = "-"                         # log lines outside of a request

# Visible ASCII only: the id goes to every log line of the request
_REQUEST_ID = re.compile(r"[!-~]{1,128}")

_timings: ContextVar[Optional["StageTimings"]] = ContextVar("timings", default=None)


class StageTimings:
    """Durations of the request stages in milliseconds, in the order they ran"""

    def __init__(self) -> None:
        self._started = time.perf_counter()
        self._stages: Dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Times the block as the stage `name`."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - started) * 1000)

    def add(self, name: str, duration_ms: float) -> None:
        self._stages[name] = self._stages.get(name, 0.0) + duration_ms

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._started) * 1000

    def as_dict(self) -> Dict[str, float]:
        """The `timings` of a response body: stage -> ms, with `total_ms` since the request came."""
        timings = {f"{name}_ms": round(duration, 1) for name, duration in self._stages.items()}
        timings["total_ms"] = round(self.elapsed_ms(), 1)
        return timings

    def header(self) -> str:
        """Server-Timing value: the stages known so far and `total` up to now."""
        stages = [f"{name};dur={duration:.1f}" for name, duration in self._stages.items()]
        stages.append(f"total;dur={self.elapsed_ms():.1f}")
        return ", ".join(stages)


def current_timings() -> StageTimings:
    """Timings of the current request; a detached one outside of a request (warmup, bench)."""
    return _timings.get() or StageTimings()


def context_headers(scope: Scope) -> Dict[str, str]:
    """Headers of the request context, for the responses made outside of the middlewares (500)."""
    state = scope.get("state") or {}
    headers = {}
    if request_id := state.get("request_id"):
        headers[REQUEST_ID_HEADER] = request_id
    if timings := state.get("timings"):
        headers[SERVER_TIMING_HEADER] = timings.header()
    return headers


class RequestIdMiddleware:
    """Takes X-Request-ID or makes one, returns it and binds it to every log line of the request"""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get(REQUEST_ID_HEADER, "")
        request_id = incoming if _REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex[:8]
        # The 500 handler runs outside of the middlewares and reads it from the scope
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        with logger.contextualize(request_id=request_id):
            await self.app(scope, receive, send_with_id)


class ServerTimingMiddleware:
    """Puts the stages known before the body into Server-Timing, streaming responses included"""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        timings = StageTimings()
        scope.setdefault("state", {})["timings"] = timings

        async def send_with_timing(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[SERVER_TIMING_HEADER] = timings.header()
            await send(message)

        token = _timings.set(timings)
        try:
            await self.app(scope, receive, send_with_timing)
        finally:
            _timings.reset(token)
