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
