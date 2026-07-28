"""Inner/outer metric primitives for the fresh VEATIC Phase 02 benchmark."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def binary_ranking_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float | int | None]:
    valid = np.isfinite(score)
    labels = y_true[valid].astype(np.uint8)
    scores = score[valid].astype(np.float64)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if not len(labels) or positives == 0 or negatives == 0:
        return {
            "rows": int(len(labels)),
            "positives": positives,
            "negatives": negatives,
            "prevalence": positives / len(labels) if len(labels) else None,
            "raw_pr_auc": None,
            "roc_auc": None,
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        pr_auc = float(average_precision_score(labels, scores))
        roc_auc = float(roc_auc_score(labels, scores))
    return {
        "rows": int(len(labels)),
        "positives": positives,
        "negatives": negatives,
        "prevalence": positives / len(labels),
        "raw_pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }


def probability_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float | None]:
    valid = np.isfinite(probability)
    labels = y_true[valid].astype(np.uint8)
    probabilities = np.clip(probability[valid].astype(np.float64), 0.0, 1.0)
    if not len(labels):
        return {"brier": None}
    return {"brier": float(brier_score_loss(labels, probabilities))}


def binary_ranking_and_probability_metrics_fast(
    y_true: np.ndarray,
    score: np.ndarray,
    *,
    probability: bool,
) -> dict[str, float | int | None]:
    """Compute Stage A metrics with one stable sort instead of two sklearn sorts.

    NumPy performs the sort and cumulative sums in native code and releases the GIL, so
    independent configuration columns can be evaluated concurrently without copying the
    prediction matrices into child processes. The equations match sklearn's binary
    average-precision and ROC trapezoid definitions, including tied-score grouping.
    """

    valid = np.isfinite(score)
    labels = y_true[valid].astype(np.uint8, copy=False)
    scores = score[valid].astype(np.float64, copy=False)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    result: dict[str, float | int | None] = {
        "rows": int(len(labels)),
        "positives": positives,
        "negatives": negatives,
        "prevalence": positives / len(labels) if len(labels) else None,
        "raw_pr_auc": None,
        "roc_auc": None,
    }
    if len(labels) and positives and negatives:
        order = np.argsort(scores, kind="mergesort")[::-1]
        ordered_scores = scores[order]
        ordered_labels = labels[order]
        distinct = np.flatnonzero(np.diff(ordered_scores))
        threshold_indices = np.concatenate(
            [distinct, np.asarray([len(ordered_scores) - 1], dtype=np.int64)]
        )
        true_positives = np.cumsum(ordered_labels, dtype=np.float64)[threshold_indices]
        false_positives = 1 + threshold_indices.astype(np.float64) - true_positives
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        recall_delta = np.diff(np.concatenate([np.asarray([0.0]), recall]))
        result["raw_pr_auc"] = float(np.sum(recall_delta * precision))

        true_positive_rate = np.concatenate([np.asarray([0.0]), recall])
        false_positive_rate = np.concatenate([np.asarray([0.0]), false_positives / negatives])
        result["roc_auc"] = float(np.trapezoid(true_positive_rate, false_positive_rate))
    if probability:
        probabilities = np.clip(scores, 0.0, 1.0)
        result["brier"] = (
            float(np.mean((labels.astype(np.float64) - probabilities) ** 2))
            if len(labels)
            else None
        )
    else:
        result["brier"] = None
    return result
