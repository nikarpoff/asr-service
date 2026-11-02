from typing import Any, Dict, Literal

from pydantic import BaseModel, ConfigDict, Field


class InfoResponse(BaseModel):
    """Service configuration; the gateway puts it into its config_id"""
    model_config = ConfigDict(extra="forbid")

    service: str = Field(description="Service name")
    version: str = Field(description="Service version, the same as in pyproject.toml")
    git_sha: str = Field(description="Commit the image is built from")
    backend: str = Field(description="Model backend, MODEL_BACKEND")
    model: str = Field(description="Model of the backend, MODEL_NAME")
    device: Literal["cpu", "cuda"] = Field(description="Inference device, TARGET_PLATFORM")
    warmed_up: bool = Field(description="The model has passed its warmup calls")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters that affect the answers")


class ModelInfo(BaseModel):
    """The model that recognized the audio"""
    model_config = ConfigDict(extra="forbid")

    backend: str = Field(description="Model backend, MODEL_BACKEND")
    name: str = Field(description="Model of the backend, MODEL_NAME")
    device: Literal["cpu", "cuda"] = Field(description="Inference device, TARGET_PLATFORM")


class RecognizeResponse(BaseModel):
    """Recognized text of the audio; an empty one when the audio has no speech"""
    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="Text as the model wrote it")
    normalized: str = Field(description="Text normalized by ASR-4: lower case, ё as е, no punctuation, single spaces")
    language: str = Field(description="Language of the recognition")
    speech: bool = Field(description="The speech detector found speech; false means the model was not called")
    audio_duration_ms: int = Field(ge=0, description="Duration of the audio as sent")
    model: ModelInfo = Field(description="The model behind the answer")
    timings: Dict[str, float] = Field(description="Stage durations in ms: decode_ms, vad_ms, inference_ms and total_ms")
    rtf: float = Field(ge=0, description="Real-time factor: total_ms divided by the audio duration")


class ErrorResponse(BaseModel):
    """Body of every error response"""
    model_config = ConfigDict(extra="forbid")

    detail: str = Field(description="Short English sentence without a final period")
