"""
Settings of this service on top of the shared ones.

The shared Settings are the same in the three AI services byte for byte; the
variables only the recognition reads live here. They come from the same environment
and .env.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceSettings(BaseSettings):
    # 1. Recognition
    VAD_ENABLED: bool = Field(True, description="Trim the silence with Silero VAD; audio without speech gets an empty text without a model call")
    MAX_AUDIO_SECONDS: float = Field(30.0, gt=0, description="Longest audio accepted; a longer one gets 413")

    # 2. Fake backend
    FAKE_TEXT: str = Field("Где дверь?", description="Text the fake backend recognizes in any speech")

    # An empty variable keeps the default
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")


@lru_cache
def get_service_settings() -> ServiceSettings:
    return ServiceSettings()
