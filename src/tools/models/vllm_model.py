from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel
from transformers import AutoProcessor

from src.config import config
from src.core.exceptions import ModelLoadError
from src.core.interfaces import VisionLanguageModelInterface
from src.core.logging import get_logger

logger = get_logger()


class VLLMModel(VisionLanguageModelInterface):
    def __init__(
        self,
        model_path: Path | None = None,
        processor_path: Path | None = None,
        device: str | None = None,
    ):
        """
        A vllm model class that uses a model from specified path to extract information from documents.
        Will be used as Base line.

        Args:
            model_path: Path to the model files
            processor_path: Path to the processor files
            device: Device to run the model on
        """
        self.model_path = model_path or (
            config.model_dir / "baseline_model" / "model"
        )
        self.processor_path = processor_path or (
            config.model_dir / "baseline_model" / "processor"
        )

        # Device configuration
        if device is None:
            device = config.models.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Model components
        self.model: AutoModel | None = None
        self.processor: AutoProcessor | None = None
        self._is_loaded = False

    def load(self) -> None:
        """Load the model and processor."""
        if self._is_loaded:
            logger.info("Layout extractor already loaded")
            return

        try:
            self._validate_paths()

            logger.info(f"Loading layout extractor model from {self.model_path}")

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.processor_path,
                use_fast=True,
            )

            # Load model
            self.model = AutoModel.from_pretrained(self.model_path)
            self.model.eval()
            self.model.to(self.device)

            self._is_loaded = True

            logger.info(
                "Layout extractor loaded successfully",
                device=self.device,
                model_path=str(self.model_path),
                detection_threshold=self.detection_threshold,
            )

        except Exception as e:
            raise ModelLoadError(
                f"Failed to load layout extractor: {e}",
                "LAYOUT_EXTRACTOR_LOAD_ERROR",
                {
                    "model_path": str(self.model_path),
                    "processor_path": str(self.processor_path),
                    "device": self.device,
                    "original_error": str(e),
                },
            ) from e

    def unload(self) -> None:
        """Unload the model to free memory."""
        if not self._is_loaded:
            return

        try:
            if self.model is not None:
                self.model.cpu()
                del self.model
                self.model = None

            if self.processor is not None:
                del self.processor
                self.processor = None

            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._is_loaded = False

            logger.info("Layout extractor unloaded successfully")

        except Exception as e:
            logger.error(f"Error unloading layout extractor: {e}")

    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self._is_loaded and self.model is not None and self.processor is not None

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
        message = f"<|image|><|begin_of_text|>{prompt}"

        inputs = self.processor(image, message, return_tensors="pt")
        outputs = self.model.generate(**inputs)

        text = self.processor.decode(outputs[0], skip_special_tokens=True)

        return str(text)


# Create a global VLLM model based on a default model
global_vllm_model = VLLMModel()
