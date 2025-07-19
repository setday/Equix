"""Core interfaces and protocols for the Equix application."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from src.base.layout import Layout


class DocumentProcessor(Protocol):
    """Protocol for document processing components."""

    def process(self, document_path: Path) -> Any:
        """Process a document and return results."""
        ...


class ModelInterface(ABC):
    """Abstract base class for all ML models."""

    @abstractmethod
    def load(self) -> None:
        """Load the model and its dependencies."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model to free memory."""
        pass


class LayoutExtractorInterface(ModelInterface):
    """Interface for layout extraction models."""

    @abstractmethod
    def extract_layout(self, image: Image.Image) -> list[dict[str, Any]]:
        """Extract layout information from an image.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            List of layout blocks with bounding boxes and types
        """
        pass


class LLMInterface(ModelInterface):
    """Interface for Large Language Models."""

    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        max_tokens: int = 200,
    ) -> str:
        """Generate a text response from a prompt.
        
        Args:
            prompt: Input text prompt
            context: Optional context information
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        pass


class VisionLanguageModelInterface(ModelInterface):
    """Interface for Vision-Language Models."""

    @abstractmethod
    def process_image_with_text(
        self,
        prompt: str,
        image: Image.Image | None = None,
        max_tokens: int = 200,
    ) -> str:
        """Process an image with a text prompt.
        
        Args:
            image: PIL Image to analyze
            prompt: Text prompt describing the task
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        pass


class PDFReaderInterface(ABC):
    """Interface for PDF reading components."""

    @abstractmethod
    def read_pdf(self, pdf_path: Path) -> list[Image.Image]:
        """Read a PDF file and convert to images.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of PIL Images, one per page
        """
        pass

    @abstractmethod
    def get_page_count(self, pdf_path: Path) -> int:
        """Get the number of pages in a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Number of pages
        """
        pass


class LayoutServiceInterface(ABC):
    """Interface for layout extraction services."""

    @abstractmethod
    async def extract_layout(self, document_path: Path) -> Layout:
        """Extract layout from a document.
        
        Args:
            document_path: Path to the document
            
        Returns:
            Extracted layout structure
        """
        pass


class InformationServiceInterface(ABC):
    """Interface for information extraction services."""

    @abstractmethod
    async def extract_information(
        self,
        document_path: Path,
        layout: Layout,
        prompt: str,
    ) -> str:
        """Extract information from a document based on a prompt.
        
        Args:
            document_path: Path to the document
            layout: Document layout structure
            prompt: Information extraction prompt
            
        Returns:
            Extracted information as text
        """
        pass


class CacheInterface(Protocol):
    """Protocol for caching implementations."""

    async def get(self, key: str) -> Any:
        """Get a value from cache."""
        ...

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set a value in cache with optional TTL."""
        ...

    async def delete(self, key: str) -> None:
        """Delete a value from cache."""
        ...

    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        ...


class LoggerInterface(Protocol):
    """Protocol for logging implementations."""

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        ...

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        ...

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        ...

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        ...

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        ...


class MetricsInterface(Protocol):
    """Protocol for metrics collection."""

    def increment_counter(self, name: str, tags: dict[str, str] | None = None) -> None:
        """Increment a counter metric."""
        ...

    def record_gauge(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Record a gauge metric."""
        ...

    def record_histogram(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Record a histogram metric."""
        ...

    def time_function(self, name: str, tags: dict[str, str] | None = None) -> Any:
        """Decorator to time function execution."""
        ...
