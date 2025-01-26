from __future__ import annotations

import pytest

from modeling.metrics import _embeddings_similarity_score
from modeling.metrics import boxes_similarity_score
from modeling.metrics import exact_similarity_score
from modeling.metrics import numeric_similarity_score


@pytest.mark.parametrize(  # type: ignore
    "test_input,expected",
    [
        (([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), 1.0),
        (([1, 2, 3, 4, 5], [-1, -2, -3, -4, -5]), -21.0),
        (([0, 1], [0, 1]), 1.0),
        (([1, 2, 3, 4, 5], [1, 2, 3, 4, 6]), 0.9),
    ],
)
def test_numeric_similarity_score(test_input, expected) -> None:
    # Test of R^2 coefficient of determination

    assert numeric_similarity_score(test_input[0], test_input[1]) == expected


@pytest.mark.parametrize(  # type: ignore
    "test_input,expected",
    [
        (([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), 1.0),
        ((["a", "b", "c", "d", "e"], ["a", "b", "c", "d", "e"]), 1.0),
        ((["a", "b", "c", "d", "e"], ["a", "b", "c", "d", "f"]), 0.8),
    ],
)
def test_exact_similarity_score(test_input, expected) -> None:
    # Test of F1 score

    assert exact_similarity_score(test_input[0], test_input[1]) == expected


@pytest.mark.parametrize(  # type: ignore
    "test_input,expected",
    [
        (
            (
                [(0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)],
                [(0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)],
            ),
            1.0,
        ),
        (
            (
                [(0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)],
                [(0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 2.0)],
            ),
            0.5,
        ),
        (
            (
                [(0.0, 0.0, 1.0, 1.0), (1.0, 1.0, 2.0, 2.0), (2.0, 2.0, 3.0, 3.0)],
                [(1.0, 1.0, 2.0, 2.0), (2.0, 2.0, 3.0, 3.0), (0.0, 0.0, 1.0, 1.0)],
            ),
            1.0,
        ),
        (([(0.0, 0.0, 2.0, 2.0)], [(1.0, 1.0, 3.0, 3.0)]), 0.0),
    ],
)
def test_boxes_similarity_score(test_input, expected) -> None:
    # Test of MOTP metric

    assert boxes_similarity_score(test_input[0], test_input[1]) == expected


@pytest.mark.parametrize(  # type: ignore
    "test_input,expected",
    [
        (([1, 0.5, 0.2], [1, 0.5, 0.2]), 1.0),
        (([1, 0.5, 0.2], [0.5, 1, 0.2]), 0.8),
        (([1, 0.5, 0.2], [-1, -0.5, -0.2]), -1.0),
        (([1, 0, 0], [0, 0, 1]), 0.0),
    ],
)
def test_embedding_similarity_score(test_input, expected) -> None:
    # Test of cosine similarity

    assert (
        abs(_embeddings_similarity_score(test_input[0], test_input[1]) - expected)
        < 1e-2
    )
