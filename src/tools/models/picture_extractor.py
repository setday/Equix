from __future__ import annotations

from pathlib import Path

from PIL import Image
from transformers import AutoModel
from transformers import AutoTokenizer

from src.config import config


class PictureExtractor:
    def __init__(
        self,
        model_path: Path = config.model_dir
        / "picture_information_extraction_model"
        / "model",
        processor_path: Path = config.model_dir
        / "picture_information_extraction_model"
        / "processor",
    ):
        """
        A picture extractor class that uses a model from specified path to extract information from pictures.

        :param model_path: The path to the model.
        :param processor_path: The path to the processor.
        """

        assert model_path.exists()
        assert processor_path.exists()

        self.model = AutoModel.from_pretrained(model_path).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.processor = self.model.init_processor(self.tokenizer)

    def ask_for(
        self,
        image: Image.Image,
        prompt: str,
    ) -> str:
        """
        Ask for information about the image.

        :param image: The input image.
        :param prompt: The prompt to ask for information.
        :return: The information about the image.
        """

        messages = [
            {"role": "user", "content": f"<|image|> {prompt}"},
            {"role": "assistant", "content": ""},
        ]

        inputs = self.processor(messages, images=[image], videos=None)
        inputs.update(
            {
                "tokenizer": self.tokenizer,
                "max_new_tokens": 100,
                "decode_text": True,
            },
        )

        outputs = self.model.generate(**inputs)

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return str(text)


# Create a global picture extractor based on a default model
global_picture_extractor = PictureExtractor()
