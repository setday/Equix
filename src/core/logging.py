"""Logging configuration for the Equix application."""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import config
from src.core.interfaces import LoggerInterface


class LoguruLogger:
    """Loguru-based logger implementation."""

    def __init__(self) -> None:
        """Initialize the logger."""
        self._configure_logger()

    def _configure_logger(self) -> None:
        """Configure loguru logger based on configuration."""
        # Remove default handler
        logger.remove()

        # Add console handler
        logger.add(
            sys.stdout,
            level=config.logging.level,
            format=config.logging.format,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

        # Add file handler if not in development
        if not config.is_development:
            log_file = config.logging.log_dir / config.logging.log_file
            logger.add(
                log_file,
                level=config.logging.level,
                format=config.logging.format,
                rotation=config.logging.rotation,
                retention=config.logging.retention,
                compression="gz",
                backtrace=True,
                diagnose=True,
            )

        # Add error file handler
        error_log_file = config.logging.log_dir / "error.log"
        logger.add(
            error_log_file,
            level="ERROR",
            format=config.logging.format,
            rotation=config.logging.rotation,
            retention=config.logging.retention,
            compression="gz",
            backtrace=True,
            diagnose=True,
        )

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        logger.critical(message, **kwargs)

    @contextmanager
    def contextualize(self, **kwargs: Any) -> Any:
        """Add context to all log messages within the block."""
        with logger.contextualize(**kwargs):
            yield

    def bind(self, **kwargs: Any) -> Any:
        """Bind context to logger."""
        return logger.bind(**kwargs)


class StructuredLogger:
    """Structured logger with additional metadata."""

    def __init__(self, logger_impl: LoguruLogger) -> None:
        """Initialize with logger implementation."""
        self._logger = logger_impl

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._logger.info(message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self._logger.error(message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._logger.warning(message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._logger.debug(message, **kwargs)

    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        response_time: float,
        **kwargs: Any,
    ) -> None:
        """Log HTTP request."""
        self._logger.info(
            "HTTP request completed",
            method=method,
            path=path,
            status_code=status_code,
            response_time=response_time,
            **kwargs,
        )

    def log_model_load(
        self,
        model_name: str,
        load_time: float,
        success: bool,
        **kwargs: Any,
    ) -> None:
        """Log model loading."""
        level = "info" if success else "error"
        getattr(self._logger, level)(
            f"Model {model_name} {'loaded' if success else 'failed to load'}",
            model_name=model_name,
            load_time=load_time,
            success=success,
            **kwargs,
        )

    def log_document_processing(
        self,
        document_id: str,
        operation: str,
        processing_time: float,
        success: bool,
        **kwargs: Any,
    ) -> None:
        """Log document processing."""
        level = "info" if success else "error"
        getattr(self._logger, level)(
            f"Document {operation} {'completed' if success else 'failed'}",
            document_id=document_id,
            operation=operation,
            processing_time=processing_time,
            success=success,
            **kwargs,
        )

    def log_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log error with context."""
        self._logger.error(
            f"Error occurred: {error}",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {},
            **kwargs,
        )


# Global logger instances
_loguru_logger = LoguruLogger()
logger_impl: LoggerInterface = _loguru_logger
structured_logger = StructuredLogger(_loguru_logger)


def get_logger() -> LoggerInterface:
    """Get the application logger."""
    return logger_impl


def get_structured_logger() -> StructuredLogger:
    """Get the structured logger."""
    return structured_logger


# Configure standard library logging to use loguru
class InterceptHandler(logging.Handler):
    """Intercept standard library logs and redirect to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record through loguru."""
        # Get corresponding loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            new_frame = frame.f_back
            if new_frame is None:
                break
            frame = new_frame
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


# Set up interception of standard library logging
def setup_logging_interception() -> None:
    """Set up interception of standard library logging."""
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Silence noisy loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


# Initialize logging interception
setup_logging_interception()
