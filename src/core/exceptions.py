"""Core exceptions for the Equix application."""

from __future__ import annotations

from typing import Any


class EquixError(Exception):
    """Base exception for all Equix-related errors."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code for categorization
            details: Additional error details for debugging
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ConfigurationError(EquixError):
    """Exception raised for configuration-related errors."""

    pass


class ModelError(EquixError):
    """Base exception for model-related errors."""

    pass


class ModelLoadError(ModelError):
    """Exception raised when model loading fails."""

    pass


class ModelInferenceError(ModelError):
    """Exception raised during model inference."""

    pass


class DocumentError(EquixError):
    """Base exception for document processing errors."""

    pass


class DocumentProcessingError(DocumentError):
    """Exception raised during general document processing."""

    pass


class LayoutExtractionError(DocumentError):
    """Exception raised during layout extraction."""

    pass


class PDFReadError(DocumentError):
    """Exception raised when PDF reading fails."""

    pass


class ValidationError(EquixError):
    """Exception raised for input validation errors."""

    pass


class ServiceError(EquixError):
    """Base exception for service-level errors."""

    pass


class ServiceUnavailableError(ServiceError):
    """Exception raised when a service is unavailable."""

    pass


class ResourceNotFoundError(EquixError):
    """Exception raised when a required resource is not found."""

    pass


class SecurityError(EquixError):
    """Exception raised for security-related issues."""

    pass
