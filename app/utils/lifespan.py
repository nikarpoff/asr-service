import asyncio

from fastapi import FastAPI
from contextlib import asynccontextmanager

from loguru import logger

from app.core.config import get_settings
from app.core.engine import InferenceEngine
from app.core.exceptions import BackendError


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")

    # 1. Build the engine off the event loop: the backend loads its model there
    try:
        engine = await asyncio.to_thread(InferenceEngine, get_settings())
    except BackendError as e:
        logger.error("Bad configuration: {}", e)
        raise SystemExit(1) from None

    # 2. Warm up: the service takes connections only after that
    await engine.startup()
    app.state.engine = engine

    try:
        yield
    finally:
        logger.info("Shutting down...")
        await engine.shutdown()
