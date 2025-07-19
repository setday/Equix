"""Improved layout extractor with proper error handling and interface implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoImageProcessor
from transformers import DetrForSegmentation

from src.config import config
from src.core.exceptions import ModelInferenceError
from src.core.exceptions import ModelLoadError
from src.core.interfaces import LayoutExtractorInterface
from src.core.logging import get_logger

logger = get_logger()


class LayoutExtractor(LayoutExtractorInterface):
    """Enhanced layout extractor with proper error handling and interface implementation."""

    def __init__(
        self,
        model_path: Path | None = None,
        processor_path: Path | None = None,
        device: str | None = None,
        detection_threshold: float | None = None,
    ) -> None:
        """Initialize the layout extractor.

        Args:
            model_path: Path to the model files
            processor_path: Path to the processor files
            device: Device to run the model on
            detection_threshold: Confidence threshold for detections
        """
        self.model_path = model_path or (
            config.model_dir / "layout_detection_model" / "model"
        )
        self.processor_path = processor_path or (
            config.model_dir / "layout_detection_model" / "processor"
        )

        # Device configuration
        if device is None:
            device = config.models.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.detection_threshold = (
            detection_threshold or config.models.detection_threshold
        )

        # Model components
        self.model: DetrForSegmentation | None = None
        self.processor: AutoImageProcessor | None = None
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
            self.processor = AutoImageProcessor.from_pretrained(
                self.processor_path,
                use_fast=True,
            )

            # Load model
            self.model = DetrForSegmentation.from_pretrained(self.model_path)
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

    def extract_layout(self, image: Image.Image) -> list[dict[str, Any]]:
        """Extract layout information from an image.

        Args:
            image: PIL Image to analyze

        Returns:
            List of layout blocks with bounding boxes and types
        """
        if not self.is_loaded():
            self.load()

        try:
            return self._perform_inference(image)

        except Exception as e:
            raise ModelInferenceError(
                f"Layout extraction failed: {e}",
                "LAYOUT_EXTRACTION_INFERENCE_ERROR",
                {
                    "image_size": image.size,
                    "image_mode": image.mode,
                    "detection_threshold": self.detection_threshold,
                    "original_error": str(e),
                },
            ) from e

    def make_layout(self, image: Image.Image) -> list[dict[str, Any]]:
        """Legacy method for compatibility."""
        return self.extract_layout(image)

    def _validate_paths(self) -> None:
        """Validate that model and processor paths exist."""
        if not self.model_path.exists():
            raise ModelLoadError(
                f"Model path does not exist: {self.model_path}",
                "MODEL_PATH_NOT_FOUND",
                {"model_path": str(self.model_path)},
            )

        if not self.processor_path.exists():
            raise ModelLoadError(
                f"Processor path does not exist: {self.processor_path}",
                "PROCESSOR_PATH_NOT_FOUND",
                {"processor_path": str(self.processor_path)},
            )

    def _perform_inference(self, image: Image.Image) -> list[dict[str, Any]]:
        """Perform model inference on the image.

        Args:
            image: Input image

        Returns:
            List of detected layout blocks
        """
        if not self.model or not self.processor:
            raise ModelInferenceError(
                "Model or processor not loaded",
                "MODEL_NOT_LOADED",
            )

        # Prepare inputs
        inputs = self.processor(image, return_tensors="pt").to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process outputs
        bboxes = self.processor.post_process_object_detection(
            outputs,
            threshold=self.detection_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        # Format results
        result = []
        for score, label, box in zip(
            bboxes["scores"],
            bboxes["labels"],
            bboxes["boxes"],
        ):
            if score < self.detection_threshold:
                continue

            bbox = box.tolist()

            result.append(
                {
                    "type": label.item(),
                    "bounding_box": {
                        "x": bbox[0],
                        "y": bbox[1],
                        "width": bbox[2] - bbox[0],
                        "height": bbox[3] - bbox[1],
                    },
                    "confidence": score.item(),
                },
            )

        logger.debug(
            "Layout extraction completed",
            blocks_found=len(result),
            image_size=image.size,
            threshold=self.detection_threshold,
        )

        return result

    def set_detection_threshold(self, threshold: float) -> None:
        """Set the detection threshold.

        Args:
            threshold: New threshold value (0.0 to 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")

        self.detection_threshold = threshold
        logger.info(f"Detection threshold updated to {threshold}")


# Global layout extractor instance for backward compatibility
global_layout_extractor = LayoutExtractor()
