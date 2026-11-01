from fastapi import (
    APIRouter,
    Request,
    Depends,
    HTTPException,
    status,
)
from app.core.auth import verify_token
from app.core.config import get_project
from app.core.engine import InferenceEngine
from app.api_v1.schemas import InfoResponse

router = APIRouter(prefix="/v1", dependencies=[Depends(verify_token)])


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
