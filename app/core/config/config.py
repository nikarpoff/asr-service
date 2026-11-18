import tomllib

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# The service root: pyproject.toml is the one source of the name and the version
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # 1. Model
    MODEL_BACKEND: str = Field("fake", description="Backend from the registry; fake answers without weights and GPU")
    MODEL_NAME: Optional[str] = Field(None, description="Model of the backend; the backend default when unset")
    TARGET_PLATFORM: Literal["cuda", "cpu"] = Field("cpu", description="Inference device")
    ARTIFACTS_PATH: Optional[Path] = Field(None, description="Local model weights root; when unset weights are downloaded into the cache")
    FAKE_LATENCY_MS: int = Field(0, ge=0, description="Inference time the fake backend imitates on every call")

    # 2. Service settings
    AUTH_TOKEN: Optional[str] = Field(None, description="Bearer token of /v1/*; auth is off when unset")
    WORKERS: int = Field(1, ge=1, description="Inference threads: requests processed at once")
    QUEUE_MAX: int = Field(4, ge=0, description="Requests waiting for a free thread; the next ones get 503 at once")
    GIT_SHA: str = Field("unknown", description="Commit the image is built from; set by the GIT_SHA build argument")

    # 3. Logging settings
    LOG_LEVEL: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = Field("INFO", description="Minimum level a message needs to be logged")
    LOG_DIAGNOSE: bool = Field(False, description="Add variable values to logged tracebacks; debug only - request payloads and auth tokens end up in the logs")

    # An empty variable keeps the default
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()


@dataclass(frozen=True)
class Project:
    """Name and version of the service from pyproject.toml"""
    name: str
    version: str


@lru_cache
def get_project() -> Project:
    """The same version goes to FastAPI and /v1/info"""
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as source:
        project = tomllib.load(source)["project"]
    return Project(name=project["name"], version=project["version"])
