"""Layout extraction service implementation."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import UploadFile, Form
from pydantic import BaseModel, Field

from src.base.layout import Layout
from src.config import config
from src.core.container import container
from src.core.exceptions import DocumentProcessingError, LayoutExtractionError, ValidationError
from src.core.interfaces import LayoutExtractorInterface, LayoutServiceInterface, PDFReaderInterface
from src.core.logging import get_structured_logger
from src.services.base import BaseService
from src.tools.models.layout_extractor import LayoutExtractor
from src.tools.pdf_reader import PDFReader

from src.core.logging import get_structured_logger, StructuredLogger

logger: StructuredLogger = get_structured_logger()


class LayoutExtractionRequest(BaseModel):
    """Request model for layout extraction."""
    
    normalize_coordinates: bool = Field(
        default=True,
        description="Whether to normalize coordinates to [0,1] range"
    )
    include_confidence: bool = Field(
        default=True,
        description="Whether to include confidence scores"
    )
    min_confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold"
    )


class LayoutExtractionResponse(BaseModel):
    """Response model for layout extraction."""
    
    layout: dict[str, Any] = Field(description="Extracted layout structure")
    document_id: str = Field(description="Unique document identifier")
    processing_time: float = Field(description="Processing time in seconds")
    page_count: int = Field(description="Number of pages processed")
    block_count: int = Field(description="Number of layout blocks found")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "layout": {
                    "blocks": [
                        {
                            "id": 0,
                            "type": "TEXT",
                            "specification": "UNKNOWN",
                            "bounding_box": {
                                "x": 0.1,
                                "y": 0.2,
                                "width": 0.8,
                                "height": 0.05
                            },
                            "page_number": 0,
                            "confidence": 0.95
                        }
                    ],
                    "page_count": 1
                },
                "document_id": "doc_12345",
                "processing_time": 2.5,
                "page_count": 1,
                "block_count": 5,
                "metadata": {
                    "model": "detr-layout-detection",
                    "version": "1.0.0"
                }
            }
        }


class LayoutExtractionService(BaseService, LayoutServiceInterface):
    """Service for extracting layout from documents."""

    def __init__(self) -> None:
        """Initialize the layout extraction service."""
        super().__init__(
            title="Equix Layout Extraction API",
            description="API for extracting layout structure from PDF documents using computer vision models.",
            version="1.0.0",
        )
        self._layout_extractor: LayoutExtractorInterface | None = None
        self._pdf_reader: PDFReaderInterface | None = None

    async def setup(self) -> None:
        """Set up service resources."""
        logger.info("Setting up layout extraction service...")
        
        try:
            # Register dependencies
            container.register_singleton(LayoutExtractorInterface, LayoutExtractor) # type: ignore
            container.register_singleton(PDFReaderInterface, PDFReader) # type: ignore
            
            # Get instances
            self._layout_extractor = container.get(LayoutExtractorInterface) # type: ignore
            self._pdf_reader = container.get(PDFReaderInterface) # type: ignore

            # Ensure models are loaded
            if hasattr(self._layout_extractor, 'load'):
                self._layout_extractor.load()
            
            logger.info("Layout extraction service setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup layout extraction service: {e}")
            raise DocumentProcessingError(
                f"Service setup failed: {e}",
                "SERVICE_SETUP_ERROR",
                {"original_error": str(e)},
            ) from e

    async def cleanup(self) -> None:
        """Clean up service resources."""
        logger.info("Cleaning up layout extraction service...")
        
        try:
            if self._layout_extractor and hasattr(self._layout_extractor, 'unload'):
                self._layout_extractor.unload()
            
            logger.info("Layout extraction service cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        details = {
            "layout_extractor": "unknown",
            "pdf_reader": "unknown",
        }
        
        try:
            if self._layout_extractor:
                if hasattr(self._layout_extractor, 'is_loaded'):
                    details["layout_extractor"] = "loaded" if self._layout_extractor.is_loaded() else "not_loaded"
                else:
                    details["layout_extractor"] = "available"
            
            if self._pdf_reader:
                details["pdf_reader"] = "available"
                
        except Exception as e:
            logger.error(f"Health check error: {e}")
            details["error"] = str(e)
        
        return details

    async def extract_layout(self, document_path: Path) -> Layout:
        """Extract layout from a document.
        
        Args:
            document_path: Path to the document
            
        Returns:
            Extracted layout structure
        """
        if self._layout_extractor is None or self._pdf_reader is None:
            raise DocumentProcessingError(
                "Service not properly initialized",
                "SERVICE_NOT_INITIALIZED",
            )

        try:
            # Read PDF
            images = self._pdf_reader.read_pdf(document_path)
            
            # Extract layout from each page
            all_blocks = []
            for page_number, image in enumerate(images):
                page_blocks = self._layout_extractor.extract_layout(image)
                
                # Add page number and normalize coordinates
                for block in page_blocks:
                    block["page_number"] = page_number
                    self._normalize_bounding_box(block, image.size)
                
                all_blocks.extend(page_blocks)
            
            # Create layout
            layout_data = {
                "blocks": all_blocks,
                "page_count": len(images),
            }
            
            return Layout.from_dict(layout_data) # type: ignore
            
        except Exception as e:
            raise LayoutExtractionError(
                f"Failed to extract layout: {e}",
                "LAYOUT_EXTRACTION_ERROR",
                {"document_path": str(document_path), "original_error": str(e)},
            ) from e

    def _normalize_bounding_box(self, block: dict[str, Any], image_size: tuple[int, int]) -> None:
        """Normalize bounding box coordinates to [0,1] range.
        
        Args:
            block: Layout block dictionary
            image_size: (width, height) of the image
        """
        bbox = block.get("bounding_box", {})
        if not bbox:
            return
        
        width, height = image_size
        
        bbox["x"] = bbox.get("x", 0) / width
        bbox["y"] = bbox.get("y", 0) / height
        bbox["width"] = bbox.get("width", 0) / width
        bbox["height"] = bbox.get("height", 0) / height

    def _add_routes(self) -> None:
        """Add service-specific routes."""
        if not self._app:
            return

        @self._app.post("/layout-extraction", response_model=LayoutExtractionResponse)
        async def extract_layout_endpoint(
            document: UploadFile,
            normalize_coordinates: bool = Form(True),
            include_confidence: bool = Form(True),
            min_confidence: float = Form(0.3),
        ) -> LayoutExtractionResponse:
            """Extract layout from uploaded PDF document."""
            
            # Validate file
            self._validate_upload(document)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                content = await document.read()
                tmp_file.write(content)
                tmp_path = Path(tmp_file.name)

            processing_start = time.time()
            document_id = f"doc_{int(time.time() * 1000)}"
            
            try:
                # Extract layout
                layout = await self.extract_layout(tmp_path)
                
                processing_time = time.time() - processing_start
                
                # Filter blocks by confidence if needed
                layout_dict = layout.to_dict()
                if include_confidence and min_confidence > 0:
                    layout_dict["blocks"] = [
                        block for block in layout_dict["blocks"]
                        if block.get("confidence", 1.0) >= min_confidence
                    ]
                
                # Remove confidence scores if not requested
                if not include_confidence:
                    for block in layout_dict["blocks"]:
                        block.pop("confidence", None)
                
                logger.log_document_processing(
                    document_id=document_id,
                    operation="layout_extraction",
                    processing_time=processing_time,
                    success=True,
                    page_count=layout_dict["page_count"],
                    block_count=len(layout_dict["blocks"]),
                )
                
                return LayoutExtractionResponse(
                    layout=layout_dict,
                    document_id=document_id,
                    processing_time=processing_time,
                    page_count=layout_dict["page_count"],
                    block_count=len(layout_dict["blocks"]),
                    metadata={
                        "model": config.models.layout_detection_model,
                        "version": self.version,
                        "min_confidence": min_confidence,
                        "normalized": normalize_coordinates,
                    },
                )
                
            except Exception as e:
                processing_time = time.time() - processing_start
                
                logger.log_document_processing(
                    document_id=document_id,
                    operation="layout_extraction",
                    processing_time=processing_time,
                    success=False,
                    error=str(e),
                )
                raise
                
            finally:
                # Clean up temporary file
                try:
                    tmp_path.unlink()
                except Exception as e:
                    logger.error(f"Failed to clean up temporary file {tmp_path}: {e}")

    def _validate_upload(self, document: UploadFile) -> None:
        """Validate uploaded document.
        
        Args:
            document: Uploaded file
            
        Raises:
            ValidationError: If validation fails
        """
        if not document.filename:
            raise ValidationError(
                "No filename provided",
                "NO_FILENAME",
            )
        
        # Check file extension
        file_path = Path(document.filename)
        if file_path.suffix.lower() not in config.data.supported_formats:
            raise ValidationError(
                f"Unsupported file format: {file_path.suffix}",
                "UNSUPPORTED_FORMAT",
                {
                    "provided_format": file_path.suffix,
                    "supported_formats": config.data.supported_formats,
                },
            )
        
        # Check content type
        if document.content_type and not document.content_type.startswith(("application/pdf", "image/")):
            raise ValidationError(
                f"Invalid content type: {document.content_type}",
                "INVALID_CONTENT_TYPE",
                {"content_type": document.content_type},
            )


# Factory function for service creation
def create_layout_service() -> LayoutExtractionService:
    """Create and return a layout extraction service instance."""
    return LayoutExtractionService()
