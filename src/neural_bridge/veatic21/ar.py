"""Fold-owned autoregressive modeling utilities for VEATIC 2.1 Phase 02."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from neural_bridge.veatic21.contracts import (
    AR_OPTIMIZER_LEARNING_RATE,
    AR_OPTIMIZER_MAX_ITERATIONS,
    AR_OPTIMIZER_TOLERANCE,
)
from neural_bridge.veatic21.evidence import average_precision_skill


@dataclass(frozen=True)
class ARModel:
    """A standardized logistic model whose fitted state is fully serializable."""

    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    regularization: float
    iterations: int
    converged: bool
    final_gradient_norm: float
    device: str


@dataclass(frozen=True)
class SplitCell:
    """One outer/inner partition with globally indexed row ownership."""

    protocol: str
    fold: int
    seed: int
    outer_train: np.ndarray
    inner_train: np.ndarray
    inner_validation: np.ndarray
    outer_test: np.ndarray


def derive_seed(source_digest: str, label: str) -> int:
    """Derive a fresh uint32 seed from the sealed VEATIC target identity."""

    payload = f"veatic21-phase02\0{source_digest}\0{label}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def derive_lag_depths(*, pacf_decay_lag: int, target_width: int) -> tuple[int, ...]:
    """Generate a compact VEATIC-only lag family from Phase 01 label landmarks."""

    if pacf_decay_lag < 1 or target_width < 1:
        raise ValueError("lag landmarks must be positive")
    powers = []
    value = 1
    while value <= target_width:
        powers.append(value)
        value *= 2
    return tuple(sorted({0, *powers, pacf_decay_lag, target_width}))


def common_history_mask(
    video_id: np.ndarray,
    row_index: np.ndarray,
    target_valid: np.ndarray,
    *,
    max_depth: int,
) -> np.ndarray:
    """Return the exact common candidate mask without crossing a video boundary."""

    video_id = np.asarray(video_id)
    row_index = np.asarray(row_index)
    target_valid = np.asarray(target_valid, dtype=np.bool_)
    if not (video_id.shape == row_index.shape == target_valid.shape):
        raise ValueError("common-history arrays must align")
    if max_depth < 0:
        raise ValueError("max_depth must be nonnegative")
    mask = target_valid & (row_index >= max_depth)
    for index in np.flatnonzero(mask):
        start = index - max_depth
        if start < 0 or video_id[start] != video_id[index]:
            raise ValueError("row_index does not encode a contiguous causal history")
        expected = np.arange(row_index[index] - max_depth, row_index[index] + 1)
        if not np.array_equal(row_index[start : index + 1], expected):
            raise ValueError("nonsequential history within a video")
    return mask


def build_ar_features(arousal: np.ndarray, indices: np.ndarray, *, depth: int) -> np.ndarray:
    """Build `[a[t], a[t-1], ..., a[t-depth]]` on already-audited rows."""

    arousal = np.asarray(arousal, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if depth < 0 or np.any(indices < depth):
        raise ValueError("indices do not support the requested AR depth")
    output = np.column_stack([arousal[indices - lag] for lag in range(depth + 1)])
    if not np.isfinite(output).all():
        raise ValueError("AR features must be finite")
    return output


def build_trailing_slope_feature(
    arousal: np.ndarray, indices: np.ndarray, *, width: int
) -> np.ndarray:
    """Build the registered univariate causal trailing-slope control."""

    arousal = np.asarray(arousal, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if width < 2 or np.any(indices < width - 1):
        raise ValueError("indices do not support the requested slope width")
    x = np.arange(width, dtype=np.float64)
    x -= np.mean(x)
    denominator = float(np.dot(x, x))
    values = np.asarray(
        [np.dot(x, arousal[index - width + 1 : index + 1]) / denominator for index in indices]
    )
    return values.reshape(-1, 1)


def _split_count(size: int, heldout_fraction: float) -> int:
    if size < 4 or not 0.0 < heldout_fraction < 1.0:
        raise ValueError("split requires at least four units and a proper fraction")
    return min(size - 2, max(1, int(round(size * heldout_fraction))))


def grouped_split_cell(
    eligible_indices: np.ndarray,
    video_id: np.ndarray,
    *,
    fold: int,
    seed: int,
    test_fraction: float,
    validation_fraction: float,
) -> SplitCell:
    """Create disjoint video-grouped outer and inner partitions."""

    eligible_indices = np.asarray(eligible_indices, dtype=np.int64)
    videos = np.unique(np.asarray(video_id)[eligible_indices])
    rng = np.random.default_rng(seed)
    ordered = rng.permutation(videos)
    test_count = _split_count(len(ordered), test_fraction)
    test_videos = ordered[:test_count]
    outer_train_videos = ordered[test_count:]
    inner_rng = np.random.default_rng(derive_seed(str(seed), "inner-grouped"))
    inner_ordered = inner_rng.permutation(outer_train_videos)
    validation_count = _split_count(len(inner_ordered), validation_fraction)
    validation_videos = inner_ordered[:validation_count]
    inner_train_videos = inner_ordered[validation_count:]
    eligible_videos = np.asarray(video_id)[eligible_indices]

    def rows(selected: np.ndarray) -> np.ndarray:
        return eligible_indices[np.isin(eligible_videos, selected)]

    cell = SplitCell(
        protocol="grouped_video",
        fold=fold,
        seed=seed,
        outer_train=rows(outer_train_videos),
        inner_train=rows(inner_train_videos),
        inner_validation=rows(validation_videos),
        outer_test=rows(test_videos),
    )
    validate_split_cell(cell)
    return cell


def blocked_split_cell(
    eligible_indices: np.ndarray,
    video_id: np.ndarray,
    *,
    seed: int,
    test_fraction: float,
    validation_fraction: float,
) -> SplitCell:
    """Create per-video forward-time outer and nested inner partitions."""

    eligible_indices = np.asarray(eligible_indices, dtype=np.int64)
    video_id = np.asarray(video_id)
    inner_train: list[np.ndarray] = []
    inner_validation: list[np.ndarray] = []
    outer_test: list[np.ndarray] = []
    for video in np.unique(video_id[eligible_indices]):
        rows = eligible_indices[video_id[eligible_indices] == video]
        test_count = _split_count(len(rows), test_fraction)
        outer_rows = rows[:-test_count]
        outer_test.append(rows[-test_count:])
        validation_count = _split_count(len(outer_rows), validation_fraction)
        inner_train.append(outer_rows[:-validation_count])
        inner_validation.append(outer_rows[-validation_count:])
    train = np.concatenate(inner_train)
    validation = np.concatenate(inner_validation)
    cell = SplitCell(
        protocol="blocked_temporal",
        fold=0,
        seed=seed,
        outer_train=np.sort(np.concatenate((train, validation))),
        inner_train=np.sort(train),
        inner_validation=np.sort(validation),
        outer_test=np.sort(np.concatenate(outer_test)),
    )
    validate_split_cell(cell)
    return cell


def validate_split_cell(cell: SplitCell) -> None:
    """Fail closed on overlap, empty ownership, or incomplete outer-train nesting."""

    arrays = (
        cell.outer_train,
        cell.inner_train,
        cell.inner_validation,
        cell.outer_test,
    )
    if any(array.ndim != 1 or len(array) == 0 for array in arrays):
        raise ValueError("every split partition must be a nonempty vector")
    if len(np.intersect1d(cell.outer_train, cell.outer_test)):
        raise ValueError("outer train/test overlap")
    if len(np.intersect1d(cell.inner_train, cell.inner_validation)):
        raise ValueError("inner train/validation overlap")
    nested = np.sort(np.concatenate((cell.inner_train, cell.inner_validation)))
    if not np.array_equal(nested, np.sort(cell.outer_train)):
        raise ValueError("inner partitions do not exactly own outer training")


def fit_logistic_mlx(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    regularization: float,
    max_iterations: int = AR_OPTIMIZER_MAX_ITERATIONS,
) -> ARModel:
    """Fit standardized ridge logistic regression with deterministic MLX Adam on GPU."""

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    if features.ndim != 2 or labels.shape != (len(features),):
        raise ValueError("training features and labels must align")
    if not np.isfinite(features).all() or len(np.unique(labels)) != 2:
        raise ValueError("training requires finite features and both classes")
    if regularization < 0.0:
        raise ValueError("regularization cannot be negative")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    mx.set_default_device(mx.gpu)
    x = mx.array(features)
    y = mx.array(labels)
    mean = mx.mean(x, axis=0)
    scale = mx.sqrt(mx.mean(mx.square(x - mean), axis=0))
    scale = mx.maximum(scale, mx.array(1e-7, dtype=scale.dtype))
    standardized = (x - mean) / scale
    design = mx.concatenate((mx.ones((len(features), 1)), standardized), axis=1)
    weights = mx.zeros((design.shape[1],), dtype=mx.float32)
    first_moment = mx.zeros_like(weights)
    second_moment = mx.zeros_like(weights)
    penalty = mx.concatenate((mx.zeros((1,)), mx.ones((design.shape[1] - 1,))))
    converged = False
    iterations = 0
    gradient_norm = float("inf")
    beta1 = 0.9
    beta2 = 0.999
    for iteration in range(1, max_iterations + 1):
        logits = mx.clip(design @ weights, -30.0, 30.0)
        probabilities = mx.sigmoid(logits)
        gradient = design.T @ (probabilities - y) / len(features)
        gradient = gradient + regularization * penalty * weights
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * mx.square(gradient)
        corrected_first = first_moment / (1.0 - beta1**iteration)
        corrected_second = second_moment / (1.0 - beta2**iteration)
        step = AR_OPTIMIZER_LEARNING_RATE * corrected_first / (mx.sqrt(corrected_second) + 1e-8)
        weights = weights - step
        mx.eval(weights, gradient)
        iterations = iteration
        gradient_norm = float(mx.sqrt(mx.sum(mx.square(gradient))).item())
        if gradient_norm <= AR_OPTIMIZER_TOLERANCE:
            converged = True
            break
    mx.eval(mean, scale, weights)
    return ARModel(
        mean=np.asarray(mean, dtype=np.float64),
        scale=np.asarray(scale, dtype=np.float64),
        weights=np.asarray(weights, dtype=np.float64),
        regularization=float(regularization),
        iterations=iterations,
        converged=converged,
        final_gradient_norm=gradient_norm,
        device="gpu:0",
    )


def predict_logistic_mlx(model: ARModel, features: np.ndarray) -> np.ndarray:
    """Score a fitted model with MLX on the registered GPU device."""

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != len(model.mean):
        raise ValueError("prediction feature width does not match the model")
    mx.set_default_device(mx.gpu)
    x = mx.array(features)
    standardized = (x - mx.array(model.mean, dtype=mx.float32)) / mx.array(
        model.scale, dtype=mx.float32
    )
    design = mx.concatenate((mx.ones((len(features), 1)), standardized), axis=1)
    probabilities = mx.sigmoid(mx.clip(design @ mx.array(model.weights), -30.0, 30.0))
    mx.eval(probabilities)
    output = np.asarray(probabilities, dtype=np.float64)
    if output.shape != (len(features),) or not np.isfinite(output).all():
        raise ValueError("invalid probability output")
    return output


def select_decision_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Select the maximum-F1 threshold using only the supplied training rows."""

    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if not len(thresholds):
        raise ValueError("decision-threshold selection requires score variation")
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = np.flatnonzero(f1 == np.max(f1))
    return float(np.max(thresholds[best]))


def spike_metrics(
    labels: np.ndarray, scores: np.ndarray, *, decision_threshold: float
) -> dict[str, float | int]:
    """Compute the complete Phase 02 held-out spike metric stack."""

    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or len(np.unique(labels)) != 2:
        raise ValueError("spike metrics require aligned scores and both classes")
    prevalence = float(np.mean(labels))
    predictions = scores >= decision_threshold
    output: dict[str, float | int] = {
        "rows": len(labels),
        "positives": int(np.sum(labels)),
        "prevalence": prevalence,
        "analytic_chance_pr_auc": prevalence,
        "pr_auc": float(average_precision_score(labels, scores)),
        "average_precision_skill": float(average_precision_skill(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "decision_threshold": float(decision_threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "brier": float(brier_score_loss(labels, scores)),
    }
    order = np.argsort(-scores, kind="stable")
    for fraction in (0.01, 0.05, 0.10):
        count = max(1, int(math.ceil(len(labels) * fraction)))
        selected = labels[order[:count]]
        key = f"top_{int(fraction * 100)}pct"
        precision_at = float(np.mean(selected))
        output[f"{key}_event_recall"] = float(np.sum(selected) / np.sum(labels))
        output[f"{key}_event_lift"] = precision_at / prevalence
    return output
