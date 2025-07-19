"""Information extraction service implementation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.base.layout import Layout
from src.config import config
from src.core.container import container
from src.core.exceptions import DocumentProcessingError, ModelInferenceError
from src.core.interfaces import InformationServiceInterface, VisionLanguageModelInterface
from src.core.logging import get_structured_logger
from src.services.base import BaseService
from src.tools.models.vllm_model import VLLMModel

from src.core.logging import get_structured_logger, StructuredLogger

logger: StructuredLogger = get_structured_logger()


class InformationExtractionRequest(BaseModel):
    """Request model for information extraction."""
    
    prompt: str = Field(description="Question or prompt for information extraction")
    document_id: str = Field(description="Document identifier from layout extraction")
    max_tokens: int = Field(
        default=200,
        ge=1,
        le=2048,
        description="Maximum tokens to generate"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )


class InformationExtractionResponse(BaseModel):
    """Response model for information extraction."""
    
    answer: str = Field(description="Generated answer")
    processing_time: float = Field(description="Processing time in seconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "answer": "The document discusses machine learning applications in computer vision...",
                "processing_time": 1.5,
                "metadata": {
                    "model": "meta-llama/Llama-3.2-11B-Vision",
                    "tokens_generated": 45,
                    "prompt_length": 120
                }
            }
        }


class InformationExtractionService(BaseService, InformationServiceInterface):
    """Service for extracting information from documents using LLM."""

    def __init__(self) -> None:
        """Initialize the information extraction service."""
        super().__init__(
            title="Equix Information Extraction API",
            description="API for extracting information from documents using vision-language models.",
            version="1.0.0",
        )
        self._vllm_model: VisionLanguageModelInterface | None = None
        self._document_cache: dict[str, Any] = {}

    async def setup(self) -> None:
        """Set up service resources."""
        logger.info("Setting up information extraction service...")
        
        try:
            # Register dependencies
            container.register_singleton(VisionLanguageModelInterface, VLLMModel)
            
            # Get instances
            self._vllm_model = container.get(VisionLanguageModelInterface) # type: ignore
            
            # Ensure model is loaded
            if hasattr(self._vllm_model, 'load'):
                self._vllm_model.load()
            
            logger.info("Information extraction service setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup information extraction service: {e}")
            raise DocumentProcessingError(
                f"Service setup failed: {e}",
                "SERVICE_SETUP_ERROR",
                {"original_error": str(e)},
            ) from e

    async def cleanup(self) -> None:
        """Clean up service resources."""
        logger.info("Cleaning up information extraction service...")
        
        try:
            if self._vllm_model and hasattr(self._vllm_model, 'unload'):
                self._vllm_model.unload()
            
            # Clear document cache
            self._document_cache.clear()
            
            logger.info("Information extraction service cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        details = {
            "vllm_model": "unknown",
            "cache_size": len(self._document_cache),
        }
        
        try:
            if self._vllm_model:
                if hasattr(self._vllm_model, 'is_loaded'):
                    details["vllm_model"] = "loaded" if self._vllm_model.is_loaded() else "not_loaded"
                else:
                    details["vllm_model"] = "available"
                    
        except Exception as e:
            logger.error(f"Health check error: {e}")
            details["error"] = str(e)
        
        return details

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
        if not self._vllm_model:
            raise DocumentProcessingError(
                "Service not properly initialized",
                "SERVICE_NOT_INITIALIZED",
            )

        try:
            # For now, use the baseline approach with document as image
            # This could be enhanced to use layout information for context
            from src.tools.pdf_reader import PDFReader
            
            pdf_reader = PDFReader()
            pdf_reader.load_pdf(document_path)
            
            # Convert to single image for baseline model
            document_image = pdf_reader.as_single_image()
            
            # Generate response using VLM
            response = self._vllm_model.process_image_with_text(
                image=document_image,
                prompt=prompt,
                max_tokens=200,  # This could be configurable
            )
            
            return response
            
        except Exception as e:
            raise ModelInferenceError(
                f"Failed to extract information: {e}",
                "INFORMATION_EXTRACTION_ERROR",
                {
                    "document_path": str(document_path),
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "original_error": str(e),
                },
            ) from e

    def _add_routes(self) -> None:
        """Add service-specific routes."""
        if not self._app:
            return

        @self._app.post("/information-extraction", response_model=InformationExtractionResponse)
        async def extract_information_endpoint(
            request: InformationExtractionRequest,
        ) -> InformationExtractionResponse:
            """Extract information from a document based on a prompt."""
            
            processing_start = time.time()
            
            try:
                # For now, we'll use a simplified approach
                # In a full implementation, you'd retrieve the document and layout
                # from the document_id (e.g., from cache or storage)
                
                # Mock document path - in real implementation, resolve from document_id
                document_path = Path(request.document_id)
                
                # Mock layout - in real implementation, load from cache
                from src.base.layout import Layout
                layout = Layout.from_dict({"blocks": [], "page_count": 1}) # type: ignore
                
                # Extract information
                answer = await self.extract_information(
                    document_path=document_path,
                    layout=layout,
                    prompt=request.prompt,
                )
                
                processing_time = time.time() - processing_start
                
                logger.log_document_processing(
                    document_id=request.document_id,
                    operation="information_extraction",
                    processing_time=processing_time,
                    success=True,
                    prompt_length=len(request.prompt),
                    answer_length=len(answer),
                )
                
                return InformationExtractionResponse(
                    answer=answer,
                    processing_time=processing_time,
                    metadata={
                        "model": config.models.baseline_model,
                        "version": self.version,
                        "prompt_length": len(request.prompt),
                        "answer_length": len(answer),
                        "max_tokens": request.max_tokens,
                        "temperature": request.temperature,
                    },
                )
                
            except Exception as e:
                processing_time = time.time() - processing_start
                
                logger.log_document_processing(
                    document_id=request.document_id,
                    operation="information_extraction",
                    processing_time=processing_time,
                    success=False,
                    error=str(e),
                )
                raise


# Factory function for service creation
def create_information_service() -> InformationExtractionService:
    """Create and return an information extraction service instance."""
    return InformationExtractionService()
