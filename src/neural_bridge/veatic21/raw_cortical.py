"""Full-width raw-cortical and matched-control utilities for VEATIC Phase 03."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class RawDiscriminantModel:
    """Training-owned full-width standardized diagonal-centroid classifier."""

    mean: np.ndarray
    scale: np.ndarray
    direction: np.ndarray
    projection_bias: float
    positive_rows: int
    negative_rows: int
    device: str


def derive_phase03_seed(source_digest: str, label: str) -> int:
    payload = f"veatic21-phase03\0{source_digest}\0{label}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def fit_raw_discriminant_mlx(features: np.ndarray, labels: np.ndarray) -> RawDiscriminantModel:
    """Fit every full-width statistic on the supplied training rows using MLX GPU."""

    features = np.asarray(features)
    labels = np.asarray(labels, dtype=np.int8)
    if features.ndim != 2 or labels.shape != (len(features),):
        raise ValueError("raw discriminant inputs must be an aligned matrix/vector")
    if not np.isfinite(features).all() or len(np.unique(labels)) != 2:
        raise ValueError("raw discriminant requires finite features and both classes")
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    mx.set_default_device(mx.gpu)
    x = mx.array(features).astype(mx.float32)
    mean = mx.mean(x, axis=0)
    second_moment = mx.mean(mx.square(x), axis=0)
    variance = mx.maximum(second_moment - mx.square(mean), 0.0)
    scale = mx.maximum(mx.sqrt(variance), 1e-6)
    positive_sum = x.T @ mx.array(labels, dtype=mx.float32)
    total_sum = mean * len(features)
    positive_mean = positive_sum / len(positive)
    negative_mean = (total_sum - positive_sum) / len(negative)
    direction = (positive_mean - negative_mean) / scale
    direction_norm = mx.sqrt(mx.sum(mx.square(direction)))
    direction = direction / mx.maximum(direction_norm, 1e-12)
    standardized_weight = direction / scale
    projection_bias = -mx.sum(mean * standardized_weight)
    mx.eval(mean, scale, direction, projection_bias)
    model = RawDiscriminantModel(
        mean=np.asarray(mean, dtype=np.float32),
        scale=np.asarray(scale, dtype=np.float32),
        direction=np.asarray(direction, dtype=np.float32),
        projection_bias=float(projection_bias.item()),
        positive_rows=len(positive),
        negative_rows=len(negative),
        device="gpu:0",
    )
    mx.clear_cache()
    return model


def predict_raw_discriminant_mlx(model: RawDiscriminantModel, features: np.ndarray) -> np.ndarray:
    """Project raw rows through the fixed training-owned discriminant on MLX GPU."""

    features = np.asarray(features)
    if features.ndim != 2 or features.shape[1] != len(model.mean):
        raise ValueError("raw prediction width does not match fitted full-width model")
    if not np.isfinite(features).all():
        raise ValueError("raw prediction features must be finite")
    mx.set_default_device(mx.gpu)
    x = mx.array(features).astype(mx.float32)
    weight = mx.array(model.direction) / mx.array(model.scale)
    scores = x @ weight + model.projection_bias
    mx.eval(scores)
    output = np.asarray(scores, dtype=np.float64)
    mx.clear_cache()
    if output.shape != (len(features),) or not np.isfinite(output).all():
        raise ValueError("invalid raw discriminant scores")
    return output


def expand_control_to_width(base: np.ndarray, *, width: int) -> np.ndarray:
    """Deterministically tile a nuisance control to the declared matched input width."""

    base = np.asarray(base)
    if base.ndim != 2 or not 0 < base.shape[1] <= width:
        raise ValueError("control base width must be positive and no wider than the target")
    if not np.isfinite(base).all():
        raise ValueError("control base must be finite")
    columns = np.arange(width, dtype=np.int64) % base.shape[1]
    return np.asarray(base[:, columns], dtype=np.float16)


def within_partition_video_shuffle(
    partition: np.ndarray,
    video_id: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Return a nonidentity within-video circular content permutation for one partition."""

    partition = np.asarray(partition, dtype=np.int64)
    video_id = np.asarray(video_id)
    source = partition.copy()
    rng = np.random.default_rng(seed)
    for video in np.unique(video_id[partition]):
        positions = np.flatnonzero(video_id[partition] == video)
        if len(positions) < 2:
            continue
        shift = int(rng.integers(1, len(positions)))
        source[positions] = partition[np.roll(positions, shift)]
    if len(partition) > 1 and np.array_equal(source, partition):
        raise ValueError("registered shuffled control was accidentally identity")
    return source


def within_video_label_permutation(
    labels: np.ndarray,
    video_id: np.ndarray,
    indices: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Circularly permute labels within video while preserving per-video support."""

    labels = np.asarray(labels, dtype=np.int8)
    video_id = np.asarray(video_id)
    indices = np.asarray(indices, dtype=np.int64)
    if labels.shape != (len(indices),):
        raise ValueError("permutation labels and indices must align")
    output = labels.copy()
    rng = np.random.default_rng(seed)
    for video in np.unique(video_id[indices]):
        positions = np.flatnonzero(video_id[indices] == video)
        if len(positions) < 2:
            continue
        shift = int(rng.integers(1, len(positions)))
        output[positions] = labels[np.roll(positions, shift)]
    if not np.array_equal(np.sort(output), np.sort(labels)):
        raise ValueError("label permutation changed global class support")
    return output


def shape_matched_random(
    rows: int,
    width: int,
    *,
    seed: int,
    chunk_rows: int = 128,
) -> np.ndarray:
    """Generate deterministic Rademacher input with the exact real shape and dtype."""

    if rows < 1 or width < 1 or chunk_rows < 1:
        raise ValueError("random control dimensions must be positive")
    rng = np.random.default_rng(seed)
    output = np.empty((rows, width), dtype=np.float16)
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        values = rng.integers(0, 2, size=(stop - start, width), dtype=np.int8)
        output[start:stop] = values * 2 - 1
    return output
