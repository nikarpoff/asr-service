"""
Inference engine: the backend on a bounded thread pool.

The pool runs WORKERS calls at once and QUEUE_MAX more requests may wait for it.
A request over that is refused at once with 503: a scene description that waited
in a queue is of no use to the user anymore.
"""

import asyncio
import contextvars

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Iterator, TypeVar

from loguru import logger

from app.core.backends import build_backend
from app.core.config import Settings, ENGINE_SETTINGS as CFG
from app.core.exceptions import ServiceBusyError

T = TypeVar("T")


class InferenceEngine:
    """Owns the backend and its thread pool; admits up to WORKERS + QUEUE_MAX requests"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # The model is loaded here: build the engine off the event loop
        self.backend = build_backend(settings)
        self.warmed_up = False

        self._thread_pool = ThreadPoolExecutor(
            max_workers=settings.WORKERS, thread_name_prefix="infer"
        )
        self._capacity = settings.WORKERS + settings.QUEUE_MAX
        self._admitted = 0

    async def startup(self) -> None:
        # Kernels, allocators and caches get ready before the first request
        for _ in range(CFG.WARMUP_CALLS):
            await self.run(self.backend.warmup)
        self.warmed_up = True

        logger.info(
            "Engine ready: backend={} model={} device={} workers={} queue={}",
            self.backend.name,
            self.backend.model,
            self.backend.device,
            self.settings.WORKERS,
            self.settings.QUEUE_MAX,
        )

    async def shutdown(self) -> None:
        self.warmed_up = False
        self._thread_pool.shutdown(wait=True, cancel_futures=True)
        self.backend.close()

    @contextmanager
    def admit(self) -> Iterator[None]:
        """A place in the engine for one request, held for all its calls; ServiceBusyError when there is none."""
        # The event loop is the only caller: the counter needs no lock
        if self._admitted >= self._capacity:
            raise ServiceBusyError("Service is busy")

        self._admitted += 1
        try:
            yield
        finally:
            self._admitted -= 1

    async def run(self, fn: Callable[..., T], *args: Any) -> T:
        """`fn(*args)` on the pool in the request context: its log lines keep the request id."""
        loop = asyncio.get_running_loop()
        context = contextvars.copy_context()
        return await loop.run_in_executor(self._thread_pool, partial(context.run, fn, *args))
