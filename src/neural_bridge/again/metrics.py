"""Metrics used by the canonical AGAIN event and continuous comparisons."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def top_recall(y_true: np.ndarray, scores: np.ndarray, frac: float) -> float:
    """Return event recall inside the highest-scoring fraction of rows."""

    positives = int(np.sum(y_true == 1))
    if positives == 0 or len(y_true) == 0:
        return math.nan
    keep = max(1, int(math.ceil(len(y_true) * frac)))
    order = np.argsort(-scores)[:keep]
    return float(np.sum(y_true[order] == 1) / positives)


def threshold_from_train(y_train: np.ndarray, train_scores: np.ndarray) -> float:
    """Choose the F1-optimal threshold using training rows only."""

    if len(np.unique(y_train)) < 2:
        return float(np.median(train_scores))
    candidates = np.unique(np.quantile(train_scores, np.linspace(0.05, 0.95, 19)))
    best = (float("-inf"), float(candidates[0]))
    for threshold in candidates:
        prediction = (train_scores >= threshold).astype(int)
        score = f1_score(y_train, prediction, zero_division=0)
        if score > best[0]:
            best = (float(score), float(threshold))
    return best[1]


def event_metrics(
    train_y: np.ndarray,
    train_scores: np.ndarray,
    test_y: np.ndarray,
    test_scores: np.ndarray,
) -> dict[str, float | int]:
    """Score pooled valid test rows using a train-only decision threshold."""

    threshold = threshold_from_train(train_y, train_scores)
    prediction = (test_scores >= threshold).astype(np.int8)
    two_classes = len(np.unique(test_y)) > 1
    return {
        "decision_threshold_train_only": threshold,
        "pr_auc": float(average_precision_score(test_y, test_scores)) if two_classes else math.nan,
        "roc_auc": float(roc_auc_score(test_y, test_scores)) if two_classes else math.nan,
        "f1": float(f1_score(test_y, prediction, zero_division=0)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(test_y, prediction)) if two_classes else math.nan
        ),
        "precision": float(precision_score(test_y, prediction, zero_division=0)),
        "recall": float(recall_score(test_y, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(test_y, prediction)),
        "top_1pct_recall": top_recall(test_y, test_scores, 0.01),
        "top_5pct_recall": top_recall(test_y, test_scores, 0.05),
        "top_10pct_recall": top_recall(test_y, test_scores, 0.10),
        "predicted_positive_count": int(prediction.sum()),
    }


def continuous_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if not len(y_true):
        return {"mae": math.nan, "mse": math.nan, "pearson": math.nan, "spearman": math.nan}
    pearson = (
        math.nan
        if np.std(y_true) == 0 or np.std(prediction) == 0
        else float(np.corrcoef(y_true, prediction)[0, 1])
    )
    spearman = float(spearmanr(y_true, prediction).statistic)
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "mse": float(mean_squared_error(y_true, prediction)),
        "pearson": pearson,
        "spearman": spearman,
    }


def top_fraction_lift(values: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    if not len(values):
        return math.nan
    keep = max(1, math.ceil(len(values) * fraction))
    selected = values[np.argsort(-scores)[:keep]]
    return float(np.mean(selected) - np.mean(values))
