"""Locked Stage A utilities for the zero-label deployment-bridge experiment.

This module contains only prespecified data contracts, feature construction,
MLX model definitions, deterministic controls, metrics, and the fail-closed
Stage A verdict. The orchestration entry point lives in
``run_again_dense_2hz_zero_label_deployment_stage_a.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from backend.scripts import again_dense_2hz_phase4_pca_bridge as phase4
from backend.scripts import run_again_dense_2hz_phase5_continuous_residual_blocked as continuous
from backend.scripts import run_again_dense_2hz_phase5_learned_heads as mlx_base
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_blocked as temporal
from backend.scripts import run_again_dense_2hz_zero_label_deployment_stage0 as stage0


SCHEMA_VERSION = "again_dense_2hz_zero_label_deployment_stage_a_v1"
AUTHORIZATION = "explicit_user_authorization_20260714_after_stage0_pass"
OUTER_FOLDS = (1, 2, 3)
SEEDS = stage0.STAGE_A_SEEDS
GROUP_NAME = "20260718_20260719_20260720"
LANES = stage0.STAGE_A_LANES
ZERO_LABEL_LANES = tuple(lane for lane in LANES if lane != "phase7_ar_assisted_teacher_ceiling")
CANDIDATES = ("video_distilled_temporal", "video_closed_loop_rollout")
CONTROLS = (
    "video_supervised_temporal",
    "video_supervised_current_row",
    "no_video_closed_loop_persistence",
    "sequence_shuffled_video",
    "video_label_permutation",
)
REQUIRED_METRICS = stage0.REQUIRED_METRICS
PCA_WIDTH = 256
PCA_FAMILY = "temporal_mean_2s"
WINDOW_ROWS = 5
DIAGNOSTIC_WIDTH = 53
HIDDEN = 64
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 80
PATIENCE = 12
BATCH_SIZE = 8192
GRAD_CLIP = 1.0
EVENT_QUANTILE = 0.90
FIRST_30_SECONDS = 30.0
MIN_DELTAS = {
    "pooled_continuous_spearman": 0.002,
    "top_5pct_true_future_movement_lift": 0.001,
    "training_q90_future_event_pr_auc": 0.002,
}


def canonical_digest(value: Any, *, digest_size: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=digest_size).hexdigest()


def array_digest(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(values)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(json.dumps(list(arr.shape)).encode("ascii"))
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_mlx_gpu() -> str:
    mlx_base.require_mlx()
    device = str(mlx_base.mx.default_device())
    if "gpu" not in device.lower():
        raise RuntimeError(f"Stage A requires MLX GPU/MPS, got {device}")
    return device


def row_ids(df: pd.DataFrame) -> np.ndarray:
    return (
        df["video_id"].astype(str)
        + "|"
        + df["time_seconds"].map(lambda value: f"{float(value):.6f}")
    ).to_numpy(dtype=str)


def indices_for_videos(df: pd.DataFrame, video_ids: Iterable[str]) -> np.ndarray:
    wanted = {str(value) for value in video_ids}
    return np.flatnonzero(df["video_id"].astype(str).isin(wanted).to_numpy()).astype(np.int64)


def valid_target_mask(df: pd.DataFrame, values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return (
        valid.astype(bool)
        & df["label_available"].to_numpy(dtype=bool)
        & np.isfinite(values)
    )


def teacher_compatibility_mask(df: pd.DataFrame, hard_mask: np.ndarray) -> np.ndarray:
    return hard_mask & df["ar_context_available"].to_numpy(dtype=bool)


def split_rows(
    df: pd.DataFrame,
    split_manifest: Mapping[str, Any],
    fold: int,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    record = next(item for item in split_manifest["stage_a"] if int(item["fold"]) == int(fold))
    train_idx = indices_for_videos(df, record["train_videos"])
    test_idx = indices_for_videos(df, record["test_videos"])
    if len(set(train_idx.tolist()) & set(test_idx.tolist())):
        raise RuntimeError(f"Outer fold {fold} has overlapping rows")
    return train_idx, test_idx, record


def three_way_teacher_rows(
    df: pd.DataFrame,
    outer_record: Mapping[str, Any],
) -> list[tuple[np.ndarray, np.ndarray, Mapping[str, Any]]]:
    outer_test = {str(value) for value in outer_record["test_videos"]}
    outer_train = [str(value) for value in outer_record["train_videos"]]
    records: list[tuple[np.ndarray, np.ndarray, Mapping[str, Any]]] = []
    teacher_folds = outer_record["teacher_crossfit_test_folds"]
    expected_digests = outer_record["teacher_crossfit_fold_digests"]
    for fold_index, test_fold in enumerate(teacher_folds):
        test_fold_set = {str(value) for value in test_fold}
        train_fold = [value for value in outer_train if value not in test_fold_set]
        record = {
            "fold": fold_index + 1,
            "train_videos": train_fold,
            "test_videos": list(test_fold),
            "train_digest": stage0.videos_digest(train_fold),
            "test_digest": stage0.videos_digest(list(test_fold)),
            "fold_digest": expected_digests[fold_index],
        }
        train_idx = indices_for_videos(df, train_fold)
        test_idx = indices_for_videos(df, test_fold)
        train_videos = set(df.loc[train_idx, "video_id"].astype(str))
        test_videos = set(df.loc[test_idx, "video_id"].astype(str))
        if train_videos & test_videos or train_videos & outer_test or test_videos & outer_test:
            raise RuntimeError("Nested teacher cross-fit violated video ownership")
        records.append((train_idx, test_idx, record))
    return records


@dataclass(frozen=True)
class PcaView:
    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_scores: np.ndarray
    test_scores: np.ndarray
    component_path: Path
    score_path: Path
    metadata: Mapping[str, Any]

    def scores_for(self, indices: np.ndarray) -> np.ndarray:
        lookup = np.full(
            int(max(self.train_idx.max(initial=0), self.test_idx.max(initial=0))) + 1,
            -1,
            dtype=np.int64,
        )
        combined = np.concatenate([self.train_idx, self.test_idx])
        lookup[combined] = np.arange(len(combined), dtype=np.int64)
        positions = lookup[np.asarray(indices, dtype=np.int64)]
        if np.any(positions < 0):
            raise KeyError(f"PCA view {self.name} does not contain all requested rows")
        scores = np.concatenate([self.train_scores, self.test_scores], axis=0)
        return np.asarray(scores[positions], dtype=np.float32)


def fit_or_load_pca_view(
    *,
    accessor: phase4.CorticalVariantAccessor,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    output_root: Path,
    name: str,
    seed: int,
    batch_size: int = 384,
    oversampling: int = 32,
    power_iterations: int = 1,
) -> PcaView:
    if len(set(train_idx.tolist()) & set(test_idx.tolist())):
        raise ValueError(f"PCA split {name} has overlapping rows")
    all_idx = np.concatenate([train_idx, test_idx]).astype(np.int64)
    result = phase4.streaming_randomized_pca_fit(
        accessor,
        train_idx,
        all_idx,
        width=PCA_WIDTH,
        seed=int(seed),
        output_root=output_root,
        fit_key=name,
        batch_size=int(batch_size),
        oversampling=int(oversampling),
        power_iterations=int(power_iterations),
    )
    backend = str(result.metadata.get("pca_backend", ""))
    if "mlx_gpu" not in backend:
        raise RuntimeError(f"PCA {name} did not use MLX GPU matmul: {backend}")
    if result.metadata.get("train_idx_digest") != phase4.array_digest(train_idx):
        raise RuntimeError(f"PCA {name} train-row digest mismatch")
    scores = np.asarray(result.scores)
    if tuple(scores.shape) != (len(all_idx), PCA_WIDTH):
        raise RuntimeError(f"PCA {name} score shape mismatch: {scores.shape}")
    return PcaView(
        name=name,
        train_idx=train_idx.copy(),
        test_idx=test_idx.copy(),
        train_scores=np.asarray(scores[: len(train_idx)], dtype=np.float32),
        test_scores=np.asarray(scores[len(train_idx) :], dtype=np.float32),
        component_path=Path(result.component_path),
        score_path=Path(result.score_path),
        metadata=dict(result.metadata),
    )


@dataclass(frozen=True)
class VideoFeatures:
    row_idx: np.ndarray
    video_id: np.ndarray
    time_seconds: np.ndarray
    x_temporal: np.ndarray
    x_current: np.ndarray
    history_mask: np.ndarray
    feature_names: tuple[str, ...]


def _causal_sequence_with_mask(
    current: np.ndarray,
    row_idx: np.ndarray,
    video_id: np.ndarray,
    *,
    window_rows: int = WINDOW_ROWS,
) -> tuple[np.ndarray, np.ndarray]:
    row_to_pos = {int(row): pos for pos, row in enumerate(row_idx)}
    sequence = np.zeros((len(row_idx), window_rows, current.shape[1]), dtype=np.float32)
    mask = np.zeros((len(row_idx), window_rows), dtype=np.float32)
    for pos, row in enumerate(row_idx):
        for offset in range(window_rows):
            lag = window_rows - 1 - offset
            previous = int(row) - lag
            previous_pos = row_to_pos.get(previous)
            if previous_pos is None or str(video_id[previous_pos]) != str(video_id[pos]):
                continue
            sequence[pos, offset] = current[previous_pos]
            mask[pos, offset] = 1.0
    return sequence.reshape(len(row_idx), -1), mask


def _video_time_features(
    df: pd.DataFrame,
    row_idx: np.ndarray,
) -> np.ndarray:
    times = df.loc[row_idx, "time_seconds"].to_numpy(dtype=np.float32)
    videos = df.loc[row_idx, "video_id"].astype(str)
    durations = df.groupby(df["video_id"].astype(str))["time_seconds"].max().clip(lower=0.5)
    denom = videos.map(durations).to_numpy(dtype=np.float32)
    fraction = np.clip(times / denom, 0.0, 1.0)
    return np.column_stack([np.log1p(np.maximum(times, 0.0)), fraction]).astype(np.float32)


def build_video_features(
    df: pd.DataFrame,
    row_idx: np.ndarray,
    pca_scores: np.ndarray,
    diagnostics: np.ndarray,
) -> VideoFeatures:
    idx = np.asarray(row_idx, dtype=np.int64)
    if tuple(pca_scores.shape) != (len(idx), PCA_WIDTH):
        raise ValueError(f"Expected PCA shape {(len(idx), PCA_WIDTH)}, got {pca_scores.shape}")
    if tuple(diagnostics.shape) != (len(idx), DIAGNOSTIC_WIDTH):
        raise ValueError(
            f"Expected diagnostic shape {(len(idx), DIAGNOSTIC_WIDTH)}, got {diagnostics.shape}"
        )
    videos = df.loc[idx, "video_id"].astype(str).to_numpy()
    sequence, history_mask = _causal_sequence_with_mask(pca_scores, idx, videos)
    time_features = _video_time_features(df, idx)
    temporal_x = np.concatenate(
        [sequence, diagnostics.astype(np.float32), history_mask, time_features], axis=1
    ).astype(np.float32)
    current_x = np.concatenate(
        [pca_scores.astype(np.float32), diagnostics.astype(np.float32), history_mask[:, -1:], time_features],
        axis=1,
    ).astype(np.float32)
    temporal_names = tuple(
        [f"pca256_lag{lag}_{column}" for lag in range(4, -1, -1) for column in range(PCA_WIDTH)]
        + [f"temporal_diagnostic_{column}" for column in range(DIAGNOSTIC_WIDTH)]
        + [f"history_available_lag{lag}" for lag in range(4, -1, -1)]
        + ["log1p_time_seconds", "video_time_fraction"]
    )
    stage0.validate_inference_feature_names(temporal_names)
    return VideoFeatures(
        row_idx=idx,
        video_id=videos,
        time_seconds=df.loc[idx, "time_seconds"].to_numpy(dtype=np.float32),
        x_temporal=temporal_x,
        x_current=current_x,
        history_mask=history_mask,
        feature_names=temporal_names,
    )


def standardize_train_only(
    train_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(train_x, axis=0).astype(np.float32)
    std = np.nanstd(train_x, axis=0).astype(np.float32)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    train = ((np.nan_to_num(train_x, nan=0.0) - mean) / std).astype(np.float32)
    test = ((np.nan_to_num(test_x, nan=0.0) - mean) / std).astype(np.float32)
    return train, test, mean, std


def deterministic_video_mapping(
    video_ids: Sequence[str],
    row_counts: Mapping[str, int],
    namespace: str,
) -> dict[str, str]:
    values = sorted({str(value) for value in video_ids})
    if len(values) < 2:
        raise ValueError("A video control mapping requires at least two videos")
    ordered = sorted(
        values,
        key=lambda value: (
            int(row_counts[value]),
            hashlib.blake2b(f"{namespace}|{value}".encode(), digest_size=16).hexdigest(),
        ),
    )
    mapping = {value: ordered[(index + 1) % len(ordered)] for index, value in enumerate(ordered)}
    if any(source == target for source, target in mapping.items()):
        raise RuntimeError("Control video mapping contains an identity assignment")
    return mapping


def reassign_video_sequences(
    values: np.ndarray,
    video_ids: np.ndarray,
    namespace: str,
) -> tuple[np.ndarray, Mapping[str, str]]:
    videos = np.asarray(video_ids, dtype=str)
    row_counts = {video: int(np.sum(videos == video)) for video in np.unique(videos)}
    mapping = deterministic_video_mapping(list(row_counts), row_counts, namespace)
    output = np.empty_like(values)
    for recipient, donor in mapping.items():
        recipient_pos = np.flatnonzero(videos == recipient)
        donor_pos = np.flatnonzero(videos == donor)
        sample = np.rint(np.linspace(0, len(donor_pos) - 1, len(recipient_pos))).astype(np.int64)
        output[recipient_pos] = values[donor_pos[sample]]
    return output, mapping


def permute_video_targets(
    targets: np.ndarray,
    video_ids: np.ndarray,
    namespace: str,
) -> tuple[np.ndarray, Mapping[str, str]]:
    reassigned, mapping = reassign_video_sequences(targets[:, None], video_ids, namespace)
    return reassigned[:, 0], mapping


class VideoScalarHead(mlx_base.nn.Module):
    """Direct scalar-output video head with the fixed five-row causal conv shape."""

    def __init__(self, input_dim: int, *, temporal_context: bool):
        super().__init__()
        self.temporal_context = bool(temporal_context)
        if self.temporal_context:
            self.sequence_width = WINDOW_ROWS * PCA_WIDTH
            self.conv = mlx_base.nn.Linear(PCA_WIDTH * 3, HIDDEN)
            self.post = mlx_base.nn.Linear(HIDDEN + input_dim - self.sequence_width, HIDDEN)
        else:
            self.sequence_width = 0
            self.layers = [
                mlx_base.nn.Linear(input_dim, HIDDEN),
                mlx_base.nn.Linear(HIDDEN, HIDDEN),
            ]
        self.out = mlx_base.nn.Linear(HIDDEN, 1)

    def __call__(self, x: Any) -> Any:
        if self.temporal_context:
            seq = x[:, : self.sequence_width].reshape((x.shape[0], WINDOW_ROWS, PCA_WIDTH))
            extra = x[:, self.sequence_width :]
            padded = mlx_base.mx.concatenate(
                [mlx_base.mx.zeros((x.shape[0], 2, PCA_WIDTH), dtype=x.dtype), seq], axis=1
            )
            hidden_rows = []
            for position in range(WINDOW_ROWS):
                window = padded[:, position : position + 3, :].reshape((x.shape[0], PCA_WIDTH * 3))
                hidden_rows.append(mlx_base.nn.gelu(self.conv(window)))
            hidden = mlx_base.mx.concatenate([hidden_rows[-1], extra], axis=1)
            hidden = mlx_base.nn.gelu(self.post(hidden))
        else:
            hidden = x
            for layer in self.layers:
                hidden = mlx_base.nn.gelu(layer(hidden))
        return self.out(hidden)[:, 0]


@dataclass(frozen=True)
class ScalarModelResult:
    train_prediction: np.ndarray
    test_prediction: np.ndarray
    checkpoint_path: Path
    checkpoint_sha256: str
    mean: np.ndarray
    std: np.ndarray
    curves: tuple[Mapping[str, Any], ...]
    best_epoch: int
    best_validation_loss: float


def _inner_video_split(video_ids: np.ndarray, namespace: str) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(
        np.unique(video_ids).astype(str),
        key=lambda value: hashlib.blake2b(f"{namespace}|{value}".encode(), digest_size=16).hexdigest(),
    )
    n_val = max(1, int(math.ceil(len(unique) * 0.20)))
    val_videos = set(unique[:n_val])
    val = np.flatnonzero(np.isin(video_ids.astype(str), list(val_videos))).astype(np.int64)
    train = np.flatnonzero(~np.isin(video_ids.astype(str), list(val_videos))).astype(np.int64)
    if not len(train) or not len(val):
        raise RuntimeError("Inner grouped-video split is empty")
    return train, val


def _predict_scalar(model: VideoScalarHead, x: np.ndarray, batch_size: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    if hasattr(model, "eval"):
        model.eval()
    for start in range(0, len(x), batch_size):
        out = model(mlx_base.mx.array(x[start : start + batch_size], dtype=mlx_base.mx.float32))
        mlx_base.mx.eval(out)
        chunks.append(np.asarray(out, dtype=np.float32))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def train_scalar_model(
    *,
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_target: np.ndarray,
    train_loss_mask: np.ndarray,
    train_video_id: np.ndarray,
    temporal_context: bool,
    seed: int,
    checkpoint_path: Path,
    namespace: str,
    weighted_huber: bool,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> ScalarModelResult:
    require_mlx_gpu()
    train_std, test_std, mean, std = standardize_train_only(train_x, test_x)
    eligible = np.flatnonzero(train_loss_mask & np.isfinite(train_target)).astype(np.int64)
    if len(eligible) < 2:
        raise RuntimeError(f"Not enough eligible training rows for {namespace}")
    relative_train, relative_val = _inner_video_split(train_video_id[eligible], namespace)
    inner_train = eligible[relative_train]
    inner_val = eligible[relative_val]
    target = np.asarray(train_target, dtype=np.float32)
    if weighted_huber:
        abs_target = np.abs(target[inner_train])
        q80, q90 = np.quantile(abs_target, [0.80, 0.90])
    else:
        q80 = q90 = math.inf
    mlx_base.mx.random.seed(int(seed))
    model = VideoScalarHead(train_std.shape[1], temporal_context=temporal_context)
    optimizer = mlx_base.optim.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(int(seed) + 130003)

    def loss_fn(model_obj: Any, xb: Any, yb: Any, wb: Any) -> Any:
        pred = model_obj(xb)
        return mlx_base.mx.mean(mlx_base.nn.losses.huber_loss(pred, yb, delta=1.0) * wb)

    loss_and_grad = mlx_base.nn.value_and_grad(model, loss_fn)
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    curves: list[Mapping[str, Any]] = []
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(max_epochs) + 1):
        if hasattr(model, "train"):
            model.train()
        order = rng.permutation(inner_train)
        epoch_losses: list[float] = []
        for start in range(0, len(order), int(batch_size)):
            rows = order[start : start + int(batch_size)]
            weights = (
                1.0
                + (np.abs(target[rows]) >= q80).astype(np.float32)
                + (np.abs(target[rows]) >= q90).astype(np.float32)
            )
            loss, grads = loss_and_grad(
                model,
                mlx_base.mx.array(train_std[rows], dtype=mlx_base.mx.float32),
                mlx_base.mx.array(target[rows], dtype=mlx_base.mx.float32),
                mlx_base.mx.array(weights, dtype=mlx_base.mx.float32),
            )
            grads, _ = mlx_base.optim.clip_grad_norm(grads, GRAD_CLIP)
            optimizer.update(model, grads)
            mlx_base.mx.eval(loss, model.parameters(), optimizer.state)
            epoch_losses.append(float(np.asarray(loss)))
        val_pred = _predict_scalar(model, train_std[inner_val], int(batch_size))
        residual = val_pred - target[inner_val]
        abs_residual = np.abs(residual)
        huber = np.where(abs_residual <= 1.0, 0.5 * residual * residual, abs_residual - 0.5)
        val_loss = float(np.mean(huber))
        curves.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)) if epoch_losses else math.nan,
                "inner_validation_huber": val_loss,
            }
        )
        if math.isfinite(val_loss) and val_loss < best_loss:
            model.save_weights(str(checkpoint_path))
            best_loss = val_loss
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= int(patience):
            break
    if not checkpoint_path.exists():
        raise RuntimeError(f"No checkpoint written for {namespace}")
    restored = VideoScalarHead(train_std.shape[1], temporal_context=temporal_context)
    _ = restored(mlx_base.mx.array(train_std[:2], dtype=mlx_base.mx.float32))
    restored.load_weights(str(checkpoint_path))
    train_prediction = _predict_scalar(restored, train_std, int(batch_size))
    test_prediction = _predict_scalar(restored, test_std, int(batch_size))
    return ScalarModelResult(
        train_prediction=train_prediction,
        test_prediction=test_prediction,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=file_sha256(checkpoint_path),
        mean=mean,
        std=std,
        curves=tuple(curves),
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
    )


def own_prediction_ar_features(
    predictions: np.ndarray,
    video_ids: np.ndarray,
    training_median: float,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    pred = np.asarray(predictions, dtype=np.float32)
    videos = np.asarray(video_ids, dtype=str)
    features = np.zeros((len(pred), 7), dtype=np.float32)
    prior_sources: list[str] = []
    previous_video: str | None = None
    history: list[float] = []
    for index, (value, video) in enumerate(zip(pred, videos, strict=True)):
        if video != previous_video:
            history = []
            previous_video = video
        lag1 = history[-1] if len(history) >= 1 else training_median
        lag2 = history[-2] if len(history) >= 2 else training_median
        lag4 = history[-4] if len(history) >= 4 else training_median
        features[index] = [value, lag1, lag2, lag4, value - lag1, value - lag2, value - lag4]
        prior_sources.append("earlier_prediction" if history else "training_median_initialization")
        history.append(float(value))
    audit = {
        "teacher_forcing_ratio": 0.0,
        "teacher_forced_state_reads": 0,
        "cross_video_state_carry": 0,
        "state_reset_per_video": True,
        "initial_state": "training_arousal_median",
        "delta_initialization": 0.0,
        "all_finite": bool(np.isfinite(features).all()),
        "prior_sources_digest": canonical_digest(prior_sources),
    }
    stage0.validate_rollout_dependencies(
        [
            {
                "video_id": str(video),
                "row_index": int(index),
                "state_source": (
                    "prediction"
                    if prior_sources[index] == "earlier_prediction"
                    else "train_median_initialization"
                ),
                "source_row_index": (
                    int(index - 1) if prior_sources[index] == "earlier_prediction" else None
                ),
                "source_video_id": str(video),
                "teacher_forced": False,
                "observed_response_read": False,
            }
            for index, video in enumerate(videos)
        ]
    )
    return features, audit


def event_labels(train_values: np.ndarray, test_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    threshold = float(np.quantile(train_values[np.isfinite(train_values)], EVENT_QUANTILE))
    return train_values >= threshold, test_values >= threshold, threshold


def top_lift(y_true: np.ndarray, prediction: np.ndarray, fraction: float) -> float:
    count = max(1, int(math.ceil(len(y_true) * float(fraction))))
    chosen = np.argsort(-prediction, kind="mergesort")[:count]
    return float(np.mean(y_true[chosen]) - np.mean(y_true))


def score_prediction(
    *,
    train_values: np.ndarray,
    test_values: np.ndarray,
    prediction: np.ndarray,
    time_seconds: np.ndarray,
) -> Mapping[str, Any]:
    if len(test_values) != len(prediction) or len(time_seconds) != len(prediction):
        raise ValueError("Prediction scorer row alignment mismatch")
    _, test_events, threshold = event_labels(train_values, test_values)
    stage0.require_event_gate_defined(test_events)
    first30 = np.asarray(time_seconds) < FIRST_30_SECONDS
    stage0.require_event_gate_defined(test_events[first30])
    row = {
        "pooled_continuous_spearman": float(continuous.spearman(test_values, prediction)),
        "top_1pct_true_future_movement_lift": top_lift(test_values, prediction, 0.01),
        "top_5pct_true_future_movement_lift": top_lift(test_values, prediction, 0.05),
        "top_10pct_true_future_movement_lift": top_lift(test_values, prediction, 0.10),
        "training_q90_future_event_pr_auc": float(average_precision_score(test_events, prediction)),
        "event_threshold": threshold,
        "event_prevalence": float(np.mean(test_events)),
        "first30_pooled_continuous_spearman": float(
            continuous.spearman(test_values[first30], prediction[first30])
        ),
        "first30_top_5pct_true_future_movement_lift": top_lift(
            test_values[first30], prediction[first30], 0.05
        ),
        "first30_training_q90_future_event_pr_auc": float(
            average_precision_score(test_events[first30], prediction[first30])
        ),
        "prediction_rows": int(len(prediction)),
        "prediction_finite_fraction": float(np.mean(np.isfinite(prediction))),
        "prediction_digest": array_digest(prediction.astype(np.float32)),
    }
    return row


def ensemble_predictions(predictions: Sequence[np.ndarray]) -> np.ndarray:
    if len(predictions) != 3:
        raise ValueError("Stage A ensembles require exactly three members")
    shapes = {tuple(np.asarray(values).shape) for values in predictions}
    if len(shapes) != 1:
        raise ValueError("Stage A ensemble members are not aligned")
    return np.mean(np.stack(predictions), axis=0).astype(np.float32)


def scored_row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["stage"],
        row["split_digest"],
        row["fold"],
        row["lane"],
        row["row_type"],
        row["seed_or_group"],
        row["cold_start_policy"],
    )


def _lane_means(frame: pd.DataFrame, metric: str) -> dict[str, float]:
    return {
        str(lane): float(rows[metric].mean())
        for lane, rows in frame.groupby("lane", sort=False)
    }


def _max_positive_contribution(values: Sequence[float]) -> float:
    positive = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(np.sum(positive))
    return float(np.max(positive) / total) if total > 0 else math.inf


def compute_stage_a_verdict(rows: pd.DataFrame, audit_pass: bool) -> Mapping[str, Any]:
    members = rows[rows["row_type"] == "member"].copy()
    ensembles = rows[rows["row_type"] == "ensemble"].copy()
    unique_keys = {scored_row_key(row) for row in rows.to_dict(orient="records")}
    scope_pass = bool(
        len(rows) == 96
        and len(members) == 72
        and len(ensembles) == 24
        and len(unique_keys) == 96
        and set(rows["fold"].astype(int)) == set(OUTER_FOLDS)
        and set(rows["lane"].astype(str)) == set(LANES)
        and set(members["seed_or_group"].astype(str)) == {str(seed) for seed in SEEDS}
        and set(ensembles["seed_or_group"].astype(str)) == {GROUP_NAME}
    )
    lane_metrics = {metric: _lane_means(ensembles, metric) for metric in REQUIRED_METRICS}
    compatibility_lane_metrics = {
        metric: _lane_means(ensembles, f"compat_{metric}") for metric in REQUIRED_METRICS
    }
    strongest_controls = {
        metric: max(CONTROLS, key=lambda lane: lane_metrics[metric][lane])
        for metric in REQUIRED_METRICS
    }
    candidate_results: dict[str, Any] = {}
    for candidate in CANDIDATES:
        checks: dict[str, bool] = {}
        deltas_by_metric: dict[str, list[float]] = {}
        retentions: dict[str, float] = {}
        first30_deltas: dict[str, list[float]] = {}
        for metric in REQUIRED_METRICS:
            control = strongest_controls[metric]
            aggregate_delta = lane_metrics[metric][candidate] - lane_metrics[metric][control]
            fold_deltas: list[float] = []
            for fold in OUTER_FOLDS:
                indexed = ensembles[ensembles["fold"].astype(int) == fold].set_index("lane")
                fold_deltas.append(float(indexed.loc[candidate, metric] - indexed.loc[control, metric]))
            deltas_by_metric[metric] = fold_deltas
            anchor = compatibility_lane_metrics[metric]["no_video_closed_loop_persistence"]
            teacher = compatibility_lane_metrics[metric]["phase7_ar_assisted_teacher_ceiling"]
            denominator = teacher - anchor
            retention = (
                (compatibility_lane_metrics[metric][candidate] - anchor) / denominator
                if denominator > 0
                else math.nan
            )
            retentions[metric] = float(retention)
            checks[f"{metric}_aggregate_minimum"] = aggregate_delta >= MIN_DELTAS[metric]
            checks[f"{metric}_wins_3_of_3"] = all(delta > 0 for delta in fold_deltas)
            checks[f"{metric}_paired_median_positive"] = float(np.median(fold_deltas)) > 0
            checks[f"{metric}_max_fold_contribution_at_most_0_60"] = (
                _max_positive_contribution(fold_deltas) <= 0.60
            )
            checks[f"{metric}_retention_at_least_0_50"] = math.isfinite(retention) and retention >= 0.50
            for false_signal in ("sequence_shuffled_video", "video_label_permutation"):
                checks[f"{metric}_beats_{false_signal}"] = (
                    lane_metrics[metric][candidate] > lane_metrics[metric][false_signal]
                )
            first_metric = {
                "pooled_continuous_spearman": "first30_pooled_continuous_spearman",
                "top_5pct_true_future_movement_lift": "first30_top_5pct_true_future_movement_lift",
                "training_q90_future_event_pr_auc": "first30_training_q90_future_event_pr_auc",
            }[metric]
            first_means = _lane_means(ensembles, first_metric)
            first_control = max(CONTROLS, key=lambda lane: first_means[lane])
            fold_first: list[float] = []
            for fold in OUTER_FOLDS:
                indexed = ensembles[ensembles["fold"].astype(int) == fold].set_index("lane")
                fold_first.append(
                    float(indexed.loc[candidate, first_metric] - indexed.loc[first_control, first_metric])
                )
            first30_deltas[metric] = fold_first
            checks[f"{metric}_first30_aggregate_positive"] = (
                first_means[candidate] > first_means[first_control]
            )
            checks[f"{metric}_first30_wins_at_least_2_of_3"] = sum(value > 0 for value in fold_first) >= 2
        member_means = {
            metric: float(members[members["lane"] == candidate][metric].mean())
            for metric in REQUIRED_METRICS
        }
        ensemble_means = {metric: lane_metrics[metric][candidate] for metric in REQUIRED_METRICS}
        uplift = {metric: ensemble_means[metric] - member_means[metric] for metric in REQUIRED_METRICS}
        checks["ensemble_spearman_uplift_at_least_0_001"] = (
            uplift["pooled_continuous_spearman"] >= 0.001
        )
        checks["ensemble_top5_uplift_positive"] = uplift["top_5pct_true_future_movement_lift"] > 0
        checks["ensemble_event_uplift_positive"] = uplift["training_q90_future_event_pr_auc"] > 0
        if candidate == "video_closed_loop_rollout":
            candidate_audits = rows[rows["lane"] == candidate]
            checks["h2_teacher_forcing_zero"] = bool(
                (candidate_audits["teacher_forcing_ratio"].fillna(0.0) == 0.0).all()
            )
            checks["h2_cross_video_carry_zero"] = bool(
                (candidate_audits["cross_video_state_carry"].fillna(0).astype(int) == 0).all()
            )
            checks["h2_rollout_finite"] = bool(candidate_audits["rollout_all_finite"].fillna(False).all())
        failed = [name for name, passed in checks.items() if not passed]
        candidate_results[candidate] = {
            "qualified": not failed,
            "checks": checks,
            "failed_gates": failed,
            "retention": retentions,
            "deltas_by_fold": deltas_by_metric,
            "first30_deltas_by_fold": first30_deltas,
            "ensemble_uplift": uplift,
        }
    qualified = [candidate for candidate in CANDIDATES if candidate_results[candidate]["qualified"]]
    winner: str | None = None
    if len(qualified) == 1:
        winner = qualified[0]
    elif len(qualified) == 2:
        minima = {
            candidate: min(candidate_results[candidate]["retention"].values())
            for candidate in qualified
        }
        if abs(minima[qualified[0]] - minima[qualified[1]]) < 0.02:
            winner = "video_distilled_temporal"
        else:
            winner = max(minima, key=minima.get)
    failed_global = []
    if not scope_pass:
        failed_global.append("exact_96_row_scope")
    if not audit_pass:
        failed_global.append("stage_a_audit")
    if not qualified:
        failed_global.append("no_candidate_passed_all_continuation_gates")
    passed = bool(scope_pass and audit_pass and winner is not None)
    return {
        "schema_version": SCHEMA_VERSION,
        "stage_a_pass": passed,
        "stage_b_authorized": False,
        "locked_winner": winner if passed else None,
        "rows_expected": 96,
        "rows_actual": int(len(rows)),
        "member_rows": int(len(members)),
        "ensemble_rows": int(len(ensembles)),
        "scope_pass": scope_pass,
        "audit_pass": bool(audit_pass),
        "strongest_zero_label_control": strongest_controls,
        "lane_metric_means": lane_metrics,
        "compatibility_lane_metric_means": compatibility_lane_metrics,
        "candidate_results": candidate_results,
        "failed_gates": failed_global,
        "phase7_ar_assisted_claim_unchanged": True,
        "deployment_claim_promoted": False,
    }


def implementation_freeze_manifest() -> Mapping[str, Any]:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "authorization": AUTHORIZATION,
        "stage": "stage_a",
        "outer_folds": list(OUTER_FOLDS),
        "seeds": list(SEEDS),
        "ensemble": {"members": list(SEEDS), "weights": [1 / 3, 1 / 3, 1 / 3]},
        "lanes": list(LANES),
        "required_metrics": list(REQUIRED_METRICS),
        "pca": {
            "source_family": PCA_FAMILY,
            "width": PCA_WIDTH,
            "oversampling": 32,
            "power_iterations": 1,
            "outer_train_only": True,
            "nested_teacher_train_only": True,
        },
        "input": {
            "sequence_window_rows": WINDOW_ROWS,
            "pca_channels": PCA_WIDTH,
            "diagnostics_width": DIAGNOSTIC_WIDTH,
            "history_mask_width": WINDOW_ROWS,
            "time_features": ["log1p_time_seconds", "video_time_fraction"],
        },
        "model": {
            "hidden": HIDDEN,
            "optimizer": "mlx.optimizers.AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "gradient_clip": GRAD_CLIP,
            "h1_loss": "weighted_huber_to_standardized_crossfit_teacher_regression_score",
            "h1_checkpoint_selection": "inner_grouped_video_teacher_huber_only",
            "supervised_loss": "weighted_huber_to_hard_future_movement",
            "h2_state_loss": "huber_to_current_arousal_with_zero_teacher_forcing",
        },
        "controls": {
            "sequence_shuffle": "deterministic_whole_video_donor_with_linear_row_resampling",
            "label_permutation": "deterministic_whole_video_teacher_target_donor_with_linear_row_resampling",
            "no_video": "zero_pca_and_diagnostics_keep_only_time_metadata",
        },
        "cold_start": {
            "prediction_starts_at_row0": True,
            "h1_missing_history": "zeros_plus_explicit_mask",
            "h2_initial_state": "outer_training_arousal_median",
            "h2_initial_deltas": 0.0,
            "teacher_forcing_ratio": 0.0,
        },
        "event_threshold": {"quantile": EVENT_QUANTILE, "fit_scope": "outer_training_only"},
        "hardware": "mlx_gpu_mps",
        "cpu_fallback": False,
        "rows": {"member": 72, "ensemble": 24, "total": 96},
        "heldout_hyperparameter_search": False,
        "member_selection": False,
        "weight_search": False,
        "stage_b_authorized": False,
    }
    return {**manifest, "implementation_freeze_digest": canonical_digest(manifest)}
