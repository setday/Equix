from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModel
from transformers import AutoTokenizer

from src.config import config


class TextEmbedder:
    def __init__(
        self,
        model_path: Path = config.model_dir / "embedding_model" / "model",
        tokenizer_path: Path = config.model_dir / "embedding_model" / "tokenizer",
    ):
        """
        A text embedder class that uses a model from specified path to embed text.

        :param model_path: The path to the model.
        :param tokenizer_path: The path to the tokenizer.
        """

        assert model_path.exists(), (
            f"Model path {model_path} does not exist. "
            "Please download the model using `scripts/models/load_models`."
        )
        assert tokenizer_path.exists(), (
            f"Tokenizer path {tokenizer_path} does not exist. "
            "Remove the model folder `models/embedding_model` "
            "and download the model again using `scripts/models/load_models`."
        )

        self.model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def embed_text(self, text: str) -> torch.Tensor:
        """
        Embed the input text.

        :param text: The input text.
        :return: The embeddings of the text.
        """

        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        outputs = self.model(**inputs)

        return outputs.last_hidden_state.mean(dim=1)[0]

    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        """
        Embed the input texts.

        :param texts: The input texts.
        :return: The embeddings of the texts.
        """

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        outputs = self.model(**inputs)

        return outputs.last_hidden_state.mean(dim=1)


# Create a global text embedder based on a default model
global_text_embedder = TextEmbedder()
