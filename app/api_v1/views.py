from fastapi import (
    APIRouter,
    Request,
    Depends,
    HTTPException,
    Query,
    status,
)

from app.core.audio import parse_media_type
from app.core.auth import verify_token
from app.core.config import get_asr_settings, get_project, AUDIO_SETTINGS
from app.core.engine import InferenceEngine
from app.core.exceptions import InputTooLargeError, InvalidInputError, InvalidParamsError
from app.core import pipeline
from app.api_v1.schemas import ErrorResponse, InfoResponse, ModelInfo, RecognizeResponse
from app.utils.middleware import current_timings

router = APIRouter(prefix="/v1", dependencies=[Depends(verify_token)])

TOO_LARGE = f"Audio is larger than {AUDIO_SETTINGS.MAX_BYTES} bytes"


def get_engine(request: Request) -> InferenceEngine:
    # Prefer app.state singleton created on startup
    svc = getattr(request.app.state, "engine", None)
    if isinstance(svc, InferenceEngine):
        return svc
    raise HTTPException(500, "service is not initialized")


@router.get("/info",
    responses={
        status.HTTP_401_UNAUTHORIZED: {"description": "No token or a wrong one"},
    },
)
async def info(engine: InferenceEngine = Depends(get_engine)) -> InfoResponse:
    """Service configuration for the gateway config_id"""
    project = get_project()
    backend = engine.backend
    return InfoResponse(
        service=project.name,
        version=project.version,
        git_sha=engine.settings.GIT_SHA,
        backend=backend.name,
        model=backend.model,
        device=backend.device,
        warmed_up=engine.warmed_up,
        params=backend.params(),
    )


@router.post("/recognize",
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse, "description": "Empty or corrupt audio"},
        status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse, "description": "No token or a wrong one"},
        status.HTTP_413_CONTENT_TOO_LARGE: {"model": ErrorResponse, "description": "Audio longer than MAX_AUDIO_SECONDS or larger than 4 MiB"},
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"model": ErrorResponse, "description": "Not audio/wav, audio/L16 or multipart/form-data, or a format of them the service does not take"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse, "description": "Invalid parameters"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse, "description": "The queue is full; retry after Retry-After seconds"},
    },
)
async def recognize(
    request: Request,
    language: str = Query("ru", pattern=r"^[a-z]{2}$", description="Язык речи, код ISO 639-1"),
    engine: InferenceEngine = Depends(get_engine),
) -> RecognizeResponse:
    """Text of the speech in a WAV, audio/L16 or multipart body"""
    media = parse_media_type(request.headers.get("content-type"))
    backend = engine.backend
    if language not in backend.languages:
        raise InvalidParamsError(f"Language '{language}' is not supported by backend '{backend.name}'")

    # A busy service refuses before reading the body
    with engine.admit():
        body = await _read_body(request, AUDIO_SETTINGS.MAX_BYTES)
        result = await engine.run(
            pipeline.recognize, backend, body, media, language, get_asr_settings().MAX_AUDIO_SECONDS,
        )

    timings = current_timings().as_dict()
    return RecognizeResponse(
        text=result.text,
        normalized=result.normalized,
        language=result.language,
        speech=result.speech,
        audio_duration_ms=round(result.audio_duration_s * 1000),
        model=ModelInfo(backend=backend.name, name=backend.model, device=backend.device),
        timings=timings,
        rtf=round(timings["total_ms"] / (result.audio_duration_s * 1000), 3),
    )


async def _read_body(request: Request, limit: int) -> bytes:
    """The body in memory, refused as soon as it passes `limit` bytes."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise InputTooLargeError(TOO_LARGE)

    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise InputTooLargeError(TOO_LARGE)

    if not body:
        raise InvalidInputError("Audio is empty")
    return bytes(body)
