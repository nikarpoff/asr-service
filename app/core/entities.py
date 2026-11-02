from dataclasses import dataclass


@dataclass(frozen=True)
class Recognition:
    """What the pipeline heard in one request"""
    text: str
    normalized: str
    language: str
    speech: bool
    audio_duration_s: float
