"""
Exception handlers mapping errors to HTTP responses: always {"detail": "<message>"}.
"""

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger

from app.core.exceptions import AppError
from app.utils.middleware import context_headers, NO_REQUEST_ID

INTERNAL_DETAIL = "Internal server error"


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    detail = str(exc) if exc.public else INTERNAL_DETAIL
    where = f"{request.method} {request.url.path}"
    logger.info(f"{where} -> {exc.status_code}: {exc.__class__.__name__}: {str(exc)[:200]}")

    return JSONResponse(status_code=exc.status_code, content={"detail": detail}, headers=exc.headers or None)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # One sentence instead of the FastAPI list: the gateway reads `detail` as a string
    error = exc.errors()[0]
    location = ".".join(str(part) for part in error["loc"])
    detail = f"Invalid request: {error['msg']} at {location}"
    logger.info(f"{request.method} {request.url.path} -> 422: {detail[:200]}")

    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": detail})


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # It runs outside of the middlewares: the request context comes from the scope
    headers = context_headers(request.scope)
    with logger.contextualize(request_id=getattr(request.state, "request_id", NO_REQUEST_ID)):
        logger.opt(exception=exc).error(
            f"{request.method} {request.url.path} -> unhandled exception "
            f"{type(exc).__name__}: {exc}"
        )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": INTERNAL_DETAIL},
        headers=headers,
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
