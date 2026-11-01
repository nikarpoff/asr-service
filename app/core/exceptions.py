"""
Domain errors
"""

from typing import ClassVar, Dict


class AppError(Exception):
    """Base app error; its message is hidden from the client."""
    status_code = 500
    public = False
    headers: ClassVar[Dict[str, str]] = {}

class InvalidInputError(AppError):
    """The input is empty or corrupt."""
    status_code = 400
    public = True

class InputTooLargeError(AppError):
    """The input exceeds its limit."""
    status_code = 413
    public = True

class UnsupportedMediaTypeError(AppError):
    """The Content-Type is not one the endpoint takes."""
    status_code = 415
    public = True

class InvalidParamsError(AppError):
    """The request parameters failed validation."""
    status_code = 422
    public = True

class ServiceBusyError(AppError):
    """The queue is full: a waiting request would only get stale."""
    status_code = 503
    public = True
    headers = {"Retry-After": "1"}

class BackendError(AppError):
    """The configured backend cannot be built: unknown, unsupported model or missing extra."""
