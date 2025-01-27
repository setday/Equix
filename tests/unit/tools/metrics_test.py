from __future__ import annotations

import pytest  # noqa

from modeling.metrics import _embeddings_similarity_score
from modeling.metrics import boxes_similarity_score
from modeling.metrics import exact_similarity_score
from modeling.metrics import numeric_similarity_score


def test_numeric_similarity_score() -> None:
    # Test of R^2 coefficient of determination

    test1_y_true = [1, 2, 3, 4, 5]
    test1_y_pred = [1, 2, 3, 4, 5]

    assert numeric_similarity_score(test1_y_true, test1_y_pred) == 1.0

    test2_y_true = [1, 2, 3, 4, 5]
    test2_y_pred = [-1, -2, -3, -4, -5]

    assert numeric_similarity_score(test2_y_true, test2_y_pred) == -21.0

    test3_y_true = [0, 1]
    test3_y_pred = [0, 1]

    assert numeric_similarity_score(test3_y_true, test3_y_pred) == 1.0

    test4_y_true = [1, 2, 3, 4, 5]
    test4_y_pred = [1, 2, 3, 4, 6]

    assert numeric_similarity_score(test4_y_true, test4_y_pred) == 0.9


def test_exact_similarity_score() -> None:
    # Test of F1 score

    test1_y_true = [1, 2, 3, 4, 5]
    test1_y_pred = [1, 2, 3, 4, 5]

    assert exact_similarity_score(test1_y_true, test1_y_pred) == 1.0

    test2_y_true = ["a", "b", "c", "d", "e"]
    test2_y_pred = ["a", "b", "c", "d", "e"]

    assert exact_similarity_score(test2_y_true, test2_y_pred) == 1.0

    test3_y_true = ["a", "b", "c", "d", "e"]
    test3_y_pred = ["a", "b", "c", "d", "f"]

    assert exact_similarity_score(test3_y_true, test3_y_pred) == 0.8


def test_boxes_similarity_score() -> None:
    # Test of MOTP metric

    test1_y_true = [(0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)]
    test1_y_pred = [(0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)]

    assert boxes_similarity_score(test1_y_true, test1_y_pred) == 1.0

    test2_y_true = [(0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)]
    test2_y_pred = [(0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 2.0)]

    assert boxes_similarity_score(test2_y_true, test2_y_pred) == 0.5

    test3_y_true = [(0.0, 0.0, 1.0, 1.0), (1.0, 1.0, 2.0, 2.0), (2.0, 2.0, 3.0, 3.0)]
    test3_y_pred = [(1.0, 1.0, 2.0, 2.0), (2.0, 2.0, 3.0, 3.0), (0.0, 0.0, 1.0, 1.0)]

    assert boxes_similarity_score(test3_y_true, test3_y_pred) == 1.0

    test4_y_true = [(0.0, 0.0, 2.0, 2.0)]
    test4_y_pred = [(1.0, 1.0, 3.0, 3.0)]

    assert boxes_similarity_score(test4_y_true, test4_y_pred) == 0.0


def test_embedding_similarity_score() -> None:
    # Test of cosine similarity

    test1_embedding1 = [1, 0.5, 0.2]
    test1_embedding2 = [1, 0.5, 0.2]

    assert _embeddings_similarity_score(test1_embedding1, test1_embedding2) == 1.0

    test2_embedding1 = [1, 0.5, 0.2]
    test2_embedding2 = [0.5, 1, 0.2]

    assert (
        abs(_embeddings_similarity_score(test2_embedding1, test2_embedding2) - 0.8)
        < 1e-2
    )

    test3_embedding1 = [1, 0.5, 0.2]
    test3_embedding2 = [-1, -0.5, -0.2]

    assert _embeddings_similarity_score(test3_embedding1, test3_embedding2) == -1.0

    test4_embedding1 = [1, 0, 0]
    test4_embedding2 = [0, 0, 1]

    assert _embeddings_similarity_score(test4_embedding1, test4_embedding2) == 0.0
