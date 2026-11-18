from fastapi import FastAPI, APIRouter, HTTPException, Request, status

from app.api_v1 import router
from app.api_v1.errors import register_error_handlers
from app.core.config import get_project, get_settings
from app.utils.lifespan import lifespan
from app.utils.logging import configure_logging
from app.utils.middleware import RequestIdMiddleware, ServerTimingMiddleware


# Get config
settings = get_settings()
project = get_project()

# Configure app and uvicorn loggers
configure_logging(settings.LOG_LEVEL, settings.LOG_DIAGNOSE)

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan, title=project.name, version=project.version)

# Health check router
health_router = APIRouter()

@health_router.get("/health", include_in_schema=False)
async def health(request: Request) -> dict:
    # The lifespan warms the model up before the first connection; this only guards it
    engine = getattr(request.app.state, "engine", None)
    if engine is None or not engine.warmed_up:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Service is warming up")
    return {"status": "ok"}

register_error_handlers(app)

# Request id is the outer one: the timing and the error lines are logged with it
app.add_middleware(ServerTimingMiddleware)
app.add_middleware(RequestIdMiddleware)

app.include_router(router)
app.include_router(health_router)
