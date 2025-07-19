"""Base service implementations for the Equix application."""

from __future__ import annotations

import asyncio
import time
import traceback
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from src.config import config
from src.core.exceptions import EquixError, ServiceError, ValidationError as EquixValidationError
from src.core.logging import get_structured_logger, StructuredLogger

logger: StructuredLogger = get_structured_logger()


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str
    version: str
    timestamp: float
    uptime: float
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str
    message: str
    error_code: str | None = None
    details: dict[str, Any] | None = None
    timestamp: float


class BaseService(ABC):
    """Base class for all services."""
    
    def __init__(
        self,
        title: str,
        description: str,
        version: str = "1.0.0",
    ) -> None:
        """Initialize the service.
        
        Args:
            title: Service title
            description: Service description  
            version: Service version
        """
        self.title = title
        self.description = description
        self.version = version
        self.start_time = time.time()
        self._app: FastAPI | None = None
        self._is_healthy = False

    @abstractmethod
    async def setup(self) -> None:
        """Set up the service resources."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up the service resources."""
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Perform health check and return status details."""
        pass

    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> Any:
        """FastAPI lifespan context manager."""
        try:
            await self.setup()
            self._is_healthy = True
            logger.info(f"Service {self.title} started successfully")
            yield
        except Exception as e:
            logger.error(f"Failed to start service {self.title}: {e}")
            raise
        finally:
            try:
                await self.cleanup()
                logger.info(f"Service {self.title} shutdown successfully")
            except Exception as e:
                logger.error(f"Error during service {self.title} cleanup: {e}")

    def create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        if self._app is not None:
            return self._app

        self._app = FastAPI(
            title=self.title,
            description=self.description,
            version=self.version,
            lifespan=self.lifespan,
        )

        # Add middleware
        self._add_middleware()
        
        # Add exception handlers
        self._add_exception_handlers()
        
        # Add common routes
        self._add_common_routes()
        
        # Add service-specific routes
        self._add_routes()

        return self._app

    def _add_middleware(self) -> None:
        """Add middleware to the FastAPI app."""
        if not self._app:
            return

        # CORS middleware
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=config.service.cors_origins,
            allow_credentials=True,
            allow_methods=config.service.cors_methods,
            allow_headers=config.service.cors_headers,
        )

        # Trusted host middleware for production
        if config.is_production:
            self._app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=["localhost", "127.0.0.1", config.service.host],
            )

        # Request logging middleware
        @self._app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            
            try:
                response = await call_next(request)
                process_time = time.time() - start_time
                
                logger.log_request(
                    method=request.method,
                    path=str(request.url.path),
                    status_code=response.status_code,
                    response_time=process_time,
                    user_agent=request.headers.get("user-agent"),
                    client_ip=request.client.host if request.client else None,
                )
                
                # Add timing header
                response.headers["X-Process-Time"] = str(process_time)
                return response
                
            except Exception as e:
                process_time = time.time() - start_time
                
                logger.log_request(
                    method=request.method,
                    path=str(request.url.path),
                    status_code=500,
                    response_time=process_time,
                    error=str(e),
                )
                raise

    def _add_exception_handlers(self) -> None:
        """Add exception handlers to the FastAPI app."""
        if not self._app:
            return

        @self._app.exception_handler(EquixError)
        async def equix_error_handler(request: Request, exc: EquixError):
            """Handle Equix-specific errors."""
            logger.error(
                f"Equix error in {request.method} {request.url.path}: {exc.message}",
                error_type=type(exc).__name__,
                error_code=exc.error_code,
                details=exc.details,
            )
            
            # Map error types to HTTP status codes
            status_code = self._get_status_code_for_error(exc)
            
            return JSONResponse(
                status_code=status_code,
                content=ErrorResponse(
                    error=type(exc).__name__,
                    message=exc.message,
                    error_code=exc.error_code,
                    details=exc.details,
                    timestamp=time.time(),
                ).model_dump(),
            )

        @self._app.exception_handler(ValidationError)
        async def validation_error_handler(request: Request, exc: ValidationError):
            """Handle Pydantic validation errors."""
            logger.error(
                f"Validation error in {request.method} {request.url.path}: {exc}",
                error_details=exc.errors(),
            )
            
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=ErrorResponse(
                    error="ValidationError",
                    message="Invalid input data",
                    details={"validation_errors": exc.errors()},
                    timestamp=time.time(),
                ).model_dump(),
            )

        @self._app.exception_handler(Exception)
        async def general_error_handler(request: Request, exc: Exception):
            """Handle unexpected errors."""
            logger.error(
                f"Unexpected error in {request.method} {request.url.path}: {exc}",
                error_type=type(exc).__name__,
                traceback=traceback.format_exc(),
            )
            
            # Don't expose internal errors in production
            if config.is_production:
                message = "Internal server error"
                details = None
            else:
                message = str(exc)
                details = {"traceback": traceback.format_exc()}
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="InternalServerError",
                    message=message,
                    details=details,
                    timestamp=time.time(),
                ).model_dump(),
            )

    def _get_status_code_for_error(self, exc: EquixError) -> int:
        """Map Equix errors to HTTP status codes."""
        from src.core.exceptions import (
            ConfigurationError,
            DocumentProcessingError,
            ModelError,
            ResourceNotFoundError,
            SecurityError,
            ServiceUnavailableError,
            ValidationError as EquixValidationError,
        )
        
        error_map = {
            EquixValidationError: status.HTTP_400_BAD_REQUEST,
            ResourceNotFoundError: status.HTTP_404_NOT_FOUND,
            SecurityError: status.HTTP_403_FORBIDDEN,
            ServiceUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
            ModelError: status.HTTP_503_SERVICE_UNAVAILABLE,
            DocumentProcessingError: status.HTTP_422_UNPROCESSABLE_ENTITY,
            ConfigurationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        }
        
        for error_type, status_code in error_map.items():
            if isinstance(exc, error_type):
                return status_code
        
        return status.HTTP_500_INTERNAL_SERVER_ERROR

    def _add_common_routes(self) -> None:
        """Add common routes like health check."""
        if not self._app:
            return

        @self._app.get("/health", response_model=HealthResponse)
        async def health_endpoint() -> HealthResponse:
            """Health check endpoint."""
            try:
                details = await self.health_check() if self._is_healthy else {}
                
                return HealthResponse(
                    status="healthy" if self._is_healthy else "unhealthy",
                    version=self.version,
                    timestamp=time.time(),
                    uptime=time.time() - self.start_time,
                    details=details,
                )
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                raise ServiceError(
                    f"Health check failed: {e}",
                    "HEALTH_CHECK_ERROR",
                ) from e

        @self._app.get("/metrics")
        async def metrics_endpoint():
            """Metrics endpoint for monitoring."""
            return {
                "service": self.title,
                "version": self.version,
                "uptime": time.time() - self.start_time,
                "memory_usage": self._get_memory_usage(),
                "request_count": getattr(self, "_request_count", 0),
            }

    @abstractmethod
    def _add_routes(self) -> None:
        """Add service-specific routes."""
        pass

    def _get_memory_usage(self) -> dict[str, float]:
        """Get memory usage information."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss": memory_info.rss / 1024 / 1024,  # MB
                "vms": memory_info.vms / 1024 / 1024,  # MB
                "percent": process.memory_percent(),
            }
        except ImportError:
            return {"rss": 0.0, "vms": 0.0, "percent": 0.0}

    def run(
        self,
        host: str | None = None,
        port: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Run the service."""
        import uvicorn
        
        app = self.create_app()
        
        uvicorn.run(
            app,
            host=host or config.service.host,
            port=port or config.service.port,
            **kwargs,
        )
