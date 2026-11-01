import inspect
import logging
import sys

from loguru import logger

from app.utils.middleware import NO_REQUEST_ID

# Access-log paths to mute
MUTED_ACCESS_PATHS = ("/health", "/metrics")

# Standard loggers sent through loguru: their lines get the request id too
INTERCEPTED_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "[{extra[request_id]}] "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)


class InterceptHandler(logging.Handler):
    """Standard logging records to loguru, in the context of the caller"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find the frame that made the call, so the record keeps its origin
        frame, depth = inspect.currentframe(), 0
        while frame is not None and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging(level: str = "INFO", diagnose: bool = False) -> None:
    logger.remove()
    logger.configure(extra={"request_id": NO_REQUEST_ID})
    logger.add(
        sys.stderr,
        level=level,
        format=LOG_FORMAT,
        backtrace=False,
        diagnose=diagnose,
    )

    for name in INTERCEPTED_LOGGERS:
        std_logger = logging.getLogger(name)
        std_logger.handlers = [InterceptHandler()]
        std_logger.propagate = False

    logging.getLogger("uvicorn.access").addFilter(
        lambda record: not any(path in record.getMessage() for path in MUTED_ACCESS_PATHS)
    )

    if diagnose:
        logger.warning(
            "LOG_DIAGNOSE is on: tracebacks will include variable values, "
            "request payloads and auth tokens among them"
        )
