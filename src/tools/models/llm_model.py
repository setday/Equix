from __future__ import annotations

from pathlib import Path

from transformers import AutoModel
from transformers import AutoTokenizer

from src.base.layout import Layout
from src.config import config


class LLMModel:
    def __init__(
        self,
        model_path: Path = config.model_dir / "llm_model" / "model",
        tokenizer_path: Path = config.model_dir / "llm_model" / "tokenizer",
    ):
        """
        A picture extractor class that uses a model from specified path to extract information from pictures.

        :param model_path: The path to the model.
        :param tokenizer_path: The path to the tokenizer.
        """

        assert model_path.exists()
        assert tokenizer_path.exists()

        self.model = AutoModel.from_pretrained(model_path).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def ask_for(
        self,
        prompt: str,
        layout: Layout | None = None,
    ) -> str:
        """
        Ask for information about document.

        :param prompt: The prompt to ask for information.
        :return: The information about the document.
        """
        messages = [
            (
                {"role": "system", "content": f"Document: {layout.to_text()}"}
                if layout
                else {}
            ),
            {"role": "user", "content": f"{prompt}"},
            {"role": "assistant", "content": ""},
        ]

        inputs = self.tokenizer(
            messages,
            max_new_tokens=200,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        outputs = self.model.generate(**inputs)

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return str(text)


# Create a global LLM model based on a default model
global_llm_model = LLMModel()
