from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity

from src.tools.models.text_embedder import global_text_embedder


def numeric_similarity_score(
    y_true: Iterable[float] | Iterable[int],
    y_pred: Iterable[float] | Iterable[int],
) -> float:
    """
    Calculate the similarity score between two numeric sequences.
    Score is based on the R^2 coefficient of determination.

    :param y_true: The ground truth target values.
    :param y_pred: The predicted target values.
    :return: The similarity score.
    """

    result = r2_score(y_true, y_pred, force_finite=True)

    assert isinstance(
        result,
        float,
    ), f"Expected float, got {type(result)} in numeric_similarity_score function."

    return result


def exact_similarity_score(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
) -> float:
    """
    Calculate the similarity score between two sequences.
    Score is based on the F1 score.

    :param y_true: The ground truth target values.
    :param y_pred: The predicted target values.
    :return: The similarity score.
    """

    result = f1_score(y_true, y_pred, average="micro")

    assert isinstance(
        result,
        float,
    ), f"Expected float, got {type(result)} in exact_similarity_score function."

    return result


def _iou_score(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """
    Calculate the Intersection over Union (IoU) score between two bounding boxes.

    :param box_a: The first bounding box.
    :param box_b: The second bounding box.
    :return: The IoU score.
    """

    assert (
        len(box_a) == 4 and len(box_b) == 4
    ), f"Bounding boxes must have 4 elements to find iou, got {len(box_a)} and {len(box_b)}."

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    ixmin = max(box_a[0], box_b[0])
    iymin = max(box_a[1], box_b[1])
    ixmax = min(box_a[2], box_b[2])
    iymax = min(box_a[3], box_b[3])

    inter_area = max(0, ixmax - ixmin) * max(0, iymax - iymin)

    iou = inter_area / float(area_a + area_b - inter_area)

    return iou


def boxes_similarity_score(
    y_true: list[tuple[float, float, float, float]],
    y_pred: list[tuple[float, float, float, float]],
    threshold: float = 0.5,
) -> float:
    """
    Calculate the similarity score between two sequences of bounding boxes.
    Score is based on the MOTP metric.

    :param y_true: The ground truth bounding boxes.
    :param y_pred: The predicted bounding boxes.
    :param threshold: The IOU threshold for matching.
    :return: The similarity score.
    """

    dist_sum = 0.0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    iou_pairs = []

    for tidx, true_bbox in enumerate(y_true):
        for pidx, pred_bbox in enumerate(y_pred):
            iou = _iou_score(true_bbox, pred_bbox)
            if iou > threshold:
                iou_pairs.append((tidx, pidx, iou))

    iou_pairs.sort(key=lambda x: x[2], reverse=True)

    marked_true = set()
    marked_pred = set()

    for [tidx, pidx, iou] in iou_pairs:
        if tidx not in marked_true and pidx not in marked_pred:
            dist_sum += iou
            match_count += 1

            marked_true.add(tidx)
            marked_pred.add(pidx)

    boxes_count = len(y_true) + len(y_pred) - match_count
    motp = dist_sum / boxes_count

    return motp


def _embeddings_similarity_score(
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> float:
    """
    Calculate the similarity score between two sequences of embeddings.
    Score is based on the cosine similarity.

    :param y_true: The ground truth embeddings.
    :param y_pred: The predicted embeddings.
    :return: The similarity score.
    """

    result = cosine_similarity([y_true], [y_pred])[0][0]
    result = max(-1.0, min(1.0, result))

    assert isinstance(
        result,
        float,
    ), f"Expected float, got {type(result)} in _embeddings_similarity_score function."

    return result


def _text_similarity_score(y_true: str, y_pred: str) -> float:
    """
    Calculate the similarity score between two text sequences.
    Score is based on the cosine similarity of the embeddings.

    :param y_true: The ground truth text.
    :param y_pred: The predicted text.
    :return: The similarity score.
    """

    y_true_embedding = [
        float(ee) for ee in global_text_embedder.embed_text(y_true).detach()
    ]
    y_pred_embedding = [
        float(ee) for ee in global_text_embedder.embed_text(y_pred).detach()
    ]

    return _embeddings_similarity_score(y_true_embedding, y_pred_embedding)


def text_similarity_score(
    y_true: list[str],
    y_pred: list[str],
) -> float:
    """
    Calculate the similarity score between two sequences of texts.
    Score is based on the cosine similarity of the embeddings.

    :param y_true: The ground truth texts.
    :param y_pred: The predicted texts.
    :return: The similarity score.
    """

    assert len(y_true) == len(y_pred), (
        "Expected equal length of y_true and y_pred, "
        f"got {len(y_true)} and {len(y_pred)}."
    )

    similarity_score_generator = (
        _text_similarity_score(true_text, pred_text)
        for true_text, pred_text in zip(y_true, y_pred)
    )

    return sum(similarity_score_generator) / len(y_true)
