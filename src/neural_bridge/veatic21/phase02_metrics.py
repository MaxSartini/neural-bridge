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
