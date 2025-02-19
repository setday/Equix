from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoImageProcessor
from transformers import DetrForSegmentation

from src.config import config


class LayoutExtractor:
    def __init__(
        self,
        model_path: Path = config.model_dir / "layout_detection_model" / "model",
        processor_path: Path = config.model_dir
        / "layout_detection_model"
        / "processor",
    ):
        """
        A layout extractor class that uses a model from specified path to extract layout information.

        :param model_path: The path to the model.
        :param processor_path: The path to the processor.
        """

        assert model_path.exists(), (
            f"Model path {model_path} does not exist. "
            "Please download the model using `scripts/models/load_models`."
        )
        assert processor_path.exists(), (
            f"Processor path {processor_path} does not exist. "
            "Remove the model folder `models/layout_detection_model` "
            "and download the model again using `scripts/models/load_models`."
        )

        self.model = DetrForSegmentation.from_pretrained(model_path).eval()
        self.processor = AutoImageProcessor.from_pretrained(processor_path)

        self.detection_threshold = 0.5

    def make_layout(
        self,
        image: Image.Image,
    ) -> list[dict[str, Any]]:
        """
        Make layout of the image.

        :param image: The input image.

        """

        inputs = self.processor(image, return_tensors="pt")
        outputs = self.model(**inputs)

        bboxes = self.processor.post_process_object_detection(
            outputs,
            threshold=self.detection_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        result = []

        for score, label, box in zip(
            bboxes["scores"],
            bboxes["labels"],
            bboxes["boxes"],
        ):
            if score < self.detection_threshold:
                continue

            result.append(
                {
                    "block_type": label.item(),
                    "bbox": box.tolist(),
                },
            )

        return result


# Create a global layout extractor based on a default model
global_layout_extractor = LayoutExtractor()
