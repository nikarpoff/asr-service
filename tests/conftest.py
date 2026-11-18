import os

from typing import Dict, Iterator

import pytest

from fastapi.testclient import TestClient

TOKEN = "test-token"

# The settings are read once, at the app import: the environment is set before it.
# Variables win over a local .env, so the tests run the same with it and without it.
os.environ.update({
    "MODEL_BACKEND": "fake",
    "MODEL_NAME": "fake",
    "TARGET_PLATFORM": "cpu",
    "FAKE_LATENCY_MS": "0",
    "AUTH_TOKEN": TOKEN,
    "WORKERS": "1",
    "QUEUE_MAX": "4",
    "GIT_SHA": "abc1234",
})


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    from app.main import app

    # `with` runs the lifespan: the engine is built and warmed up
    with TestClient(app) as client:
        yield client


@pytest.fixture
def auth() -> Dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}
