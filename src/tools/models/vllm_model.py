from __future__ import annotations

from pathlib import Path

from PIL import Image
from transformers import AutoModel
from transformers import AutoProcessor

from src.config import config


class VLLMModel:
    def __init__(
        self,
        model_path: Path = config.model_dir / "baseline_model" / "model",
        processor_path: Path = config.model_dir / "baseline_model" / "processor",
    ):
        """
        A vllm model class that uses a model from specified path to extract information from documents.
        Will be used as Base line.

        :param model_path: The path to the model.
        :param processor_path: The path to the processor.
        """

        assert model_path.exists()
        assert processor_path.exists()

        self.model = AutoModel.from_pretrained(model_path).eval()
        self.processor = AutoProcessor.from_pretrained(processor_path)

    def ask_for(
        self,
        prompt: str,
        image: Image.Image | None = None,
    ) -> str:
        """
        Ask for information about document.

        :param prompt: The prompt to ask for information.
        :return: The information about the document.
        """
        message = f"<|image|><|begin_of_text|>{prompt}"

        inputs = self.processor(image, message, return_tensors="pt")
        outputs = self.model.generate(**inputs)

        text = self.processor.decode(outputs[0], skip_special_tokens=True)

        return str(text)


# Create a global VLLM model based on a default model
global_vllm_model = VLLMModel()
