from __future__ import annotations

import pytest

from modeling.metrics import text_similarity_score
from src.tools.text_embedder import global_text_embedder
from src.tools.text_embedder import TextEmbedder


@pytest.fixture  # type: ignore
def text_embedder() -> TextEmbedder:
    return TextEmbedder()


def test_text_embedder(text_embedder: TextEmbedder) -> None:
    assert global_text_embedder is not None

    text = "Hello, world!"

    embedding = text_embedder.embed_text(text)

    assert embedding.size() == (384,)

    text1 = "Hello, world!"
    text2 = "Hello, world!"

    embeddings = text_embedder.embed_texts([text1, text2])

    assert embeddings.size() == (2, 384)

    assert text_similarity_score([text1], [text2]) == 1.0

    text1 = "Hello, world!"
    text2 = "Hi, universe!"

    similarity_score = text_similarity_score([text1], [text2])

    assert 0.6 < similarity_score < 1.0
