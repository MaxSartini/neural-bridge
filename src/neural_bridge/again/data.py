"""Dense-row targets, leakage-safe splits, and causal representations for AGAIN."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .contracts import ROW_STEP_SECONDS, SPIKE_TARGET, Protocol, Split, TargetSpec

HORIZON_ROWS = (1, 2, 4, 6)
AR_FEATURE_COLUMNS = (
    "arousal",
    "arousal_lag_1row",
    "arousal_lag_2row",
    "arousal_lag_4row",
    "arousal_delta_prev_1row",
    "arousal_delta_prev_2row",
    "arousal_delta_prev_4row",
)


def add_targets_and_ar_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Build the historical AGAIN 2 Hz targets without using future feature rows."""

    required = {"video_id", "row_index", "time_seconds", "label_available", "arousal"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"dense table is missing required columns: {missing}")
    if rows.duplicated(["video_id", "row_index"]).any():
        raise ValueError("duplicate (video_id, row_index) identities")

    out = rows.sort_values(["video_id", "row_index"]).reset_index(drop=True).copy()
    for horizon in HORIZON_ROWS:
        out[f"future_arousal_delta_p{horizon}rows"] = np.nan
        out[f"target_mask_arousal_delta_p{horizon}rows"] = False
        out[f"horizon_seconds_p{horizon}rows"] = horizon * ROW_STEP_SECONDS
    out["future_arousal_max_delta_rows_2_6"] = np.nan
    out["target_mask_arousal_spike_rows_2_6"] = False
    for lag in (1, 2, 4):
        out[f"arousal_lag_{lag}row"] = np.nan
        out[f"arousal_lag_{lag}row_available"] = False
        out[f"arousal_delta_prev_{lag}row"] = np.nan

    for _, positions in out.groupby("video_id", sort=False).groups.items():
        idx = np.asarray(positions, dtype=np.int64)
        available = out.loc[idx, "label_available"].to_numpy(dtype=bool)
        arousal = out.loc[idx, "arousal"].to_numpy(dtype=np.float64)
        for horizon in HORIZON_ROWS:
            if len(idx) <= horizon:
                continue
            source = idx[:-horizon]
            feasible = available[:-horizon] & available[horizon:]
            delta = arousal[horizon:] - arousal[:-horizon]
            out.loc[source, f"future_arousal_delta_p{horizon}rows"] = np.where(
                feasible, delta, np.nan
            )
            out.loc[source, f"target_mask_arousal_delta_p{horizon}rows"] = feasible
        for lag in (1, 2, 4):
            if len(idx) <= lag:
                continue
            destination = idx[lag:]
            feasible = available[lag:] & available[:-lag]
            out.loc[destination, f"arousal_lag_{lag}row"] = np.where(
                feasible, arousal[:-lag], np.nan
            )
            out.loc[destination, f"arousal_lag_{lag}row_available"] = feasible
            out.loc[destination, f"arousal_delta_prev_{lag}row"] = np.where(
                feasible, arousal[lag:] - arousal[:-lag], np.nan
            )
        if len(idx) > 6:
            source = idx[:-6]
            future = np.vstack([arousal[offset : len(idx) - 6 + offset] for offset in range(2, 7)])
            future_available = np.vstack(
                [available[offset : len(idx) - 6 + offset] for offset in range(2, 7)]
            )
            feasible = available[:-6] & np.all(future_available, axis=0)
            values = np.full(len(source), np.nan, dtype=np.float64)
            values[feasible] = np.max(future[:, feasible] - arousal[:-6][None, feasible], axis=0)
            out.loc[source, "future_arousal_max_delta_rows_2_6"] = values
            out.loc[source, "target_mask_arousal_spike_rows_2_6"] = feasible

    lag_flags = [f"arousal_lag_{lag}row_available" for lag in (1, 2, 4)]
    out["ar_context_available"] = out[lag_flags].all(axis=1)
    durations = out.groupby("video_id")["time_seconds"].transform("max").replace(0, np.nan)
    out["video_time_fraction"] = out["time_seconds"] / durations
    return out


def future_max_delta(
    rows: pd.DataFrame, *, start: int = 4, end: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Maximum future arousal increase over an inclusive causal forecast window."""

    arousal = rows["arousal"].to_numpy(dtype=np.float64)
    labels = rows["label_available"].to_numpy(dtype=bool)
    values = np.full(len(rows), np.nan, dtype=np.float64)
    mask = np.zeros(len(rows), dtype=bool)
    for _, group in rows.groupby("video_id", sort=False):
        idx = group.index.to_numpy(dtype=np.int64)
        if len(idx) <= end:
            continue
        base = idx[: len(idx) - end]
        future = np.vstack(
            [arousal[idx[offset : len(idx) - end + offset]] for offset in range(start, end + 1)]
        )
        future_labels = np.vstack(
            [labels[idx[offset : len(idx) - end + offset]] for offset in range(start, end + 1)]
        )
        valid = (
            labels[base]
            & np.all(future_labels, axis=0)
            & np.isfinite(arousal[base])
            & np.all(np.isfinite(future), axis=0)
        )
        output = np.full(len(base), np.nan, dtype=np.float64)
        output[valid] = np.max(future[:, valid] - arousal[base][valid][None, :], axis=0)
        values[base] = output
        mask[base] = valid
    return values, mask


def add_redesigned_targets(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Reproduce the selected AGAIN 4–10-row target and its locked residual target."""

    output = rows.copy()
    future, valid_future = future_max_delta(output)
    output["future_arousal_max_delta_rows_4_10"] = future
    output["target_mask_future_arousal_max_delta_rows_4_10"] = valid_future
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        trailing_2s = np.nanmean(
            np.vstack(
                [
                    output["arousal"].to_numpy(dtype=np.float64),
                    output["arousal_lag_1row"].to_numpy(dtype=np.float64),
                    output["arousal_lag_2row"].to_numpy(dtype=np.float64),
                    output["arousal_lag_4row"].to_numpy(dtype=np.float64),
                ]
            ),
            axis=0,
        )
    trailing_4s = np.full(len(output), np.nan, dtype=np.float64)
    for _, group in output.groupby("video_id", sort=False):
        idx = group.index.to_numpy(dtype=np.int64)
        trailing_4s[idx] = (
            pd.Series(output.loc[idx, "arousal"].to_numpy(dtype=np.float64))
            .rolling(window=9, min_periods=9)
            .mean()
            .to_numpy()
        )
    ar_features = np.column_stack(
        [
            output["arousal"].to_numpy(dtype=np.float64),
            output["arousal_lag_1row"].to_numpy(dtype=np.float64),
            trailing_2s,
            trailing_4s,
            output["arousal_delta_prev_4row"].to_numpy(dtype=np.float64),
            output["video_time_fraction"].to_numpy(dtype=np.float64),
        ]
    )
    residual_valid = (
        output["label_available"].to_numpy(dtype=bool)
        & output["ar_context_available"].to_numpy(dtype=bool)
        & valid_future
        & np.isfinite(future)
        & np.all(np.isfinite(ar_features), axis=1)
    )
    raw_split = _raw_splits(output, residual_valid, "blocked_temporal_70_30", 1)
    if not raw_split:
        raise ValueError("redesigned target has no valid blocked-temporal split")
    _, train_idx, test_idx = raw_split[0]
    scaler = StandardScaler().fit(ar_features[train_idx])
    model = Ridge(alpha=10.0).fit(scaler.transform(ar_features[train_idx]), future[train_idx])
    scored_idx = np.concatenate([train_idx, test_idx])
    residual = np.full(len(output), np.nan, dtype=np.float64)
    residual[scored_idx] = future[scored_idx] - model.predict(
        scaler.transform(ar_features[scored_idx])
    )
    output["residual_future_max_delta_rows_4_10"] = residual
    output["target_mask_residual_future_max_delta_rows_4_10"] = np.isfinite(residual)
    return output, {
        "policy": "train_only_simple_ar_residualizer_inside_redesigned_blocked_split",
        "ridge_alpha": 10.0,
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
    }


def target_mask(rows: pd.DataFrame, target: TargetSpec) -> np.ndarray:
    return (
        rows["label_available"].to_numpy(dtype=bool)
        & rows["ar_context_available"].to_numpy(dtype=bool)
        & rows[target.mask_column].to_numpy(dtype=bool)
        & np.isfinite(rows[target.value_column].to_numpy(dtype=np.float64))
    )


def _transform_target(values: np.ndarray, target: TargetSpec) -> np.ndarray:
    if target.transform == "positive_delta":
        return np.maximum(values, 0.0)
    if target.transform == "abs_movement":
        return np.abs(values)
    return values


def _threshold_labels(
    values: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, target: TargetSpec
) -> tuple[np.ndarray, np.ndarray, float]:
    transformed = _transform_target(values, target)
    train_values = transformed[train_idx]
    train_values = train_values[np.isfinite(train_values)]
    if not len(train_values):
        raise ValueError(f"no finite training values for {target.name}")
    threshold = float(np.quantile(train_values, target.quantile))
    return (
        (transformed[train_idx] >= threshold).astype(np.int8),
        (transformed[test_idx] >= threshold).astype(np.int8),
        threshold,
    )


def _raw_splits(
    rows: pd.DataFrame, mask: np.ndarray, protocol: Protocol, n_splits: int
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    eligible = np.flatnonzero(mask)
    if protocol == "grouped_video":
        groups = rows.loc[eligible, "video_id"].astype(str).to_numpy()
        folds = min(n_splits, len(np.unique(groups)))
        if folds < 2:
            return []
        return [
            (fold, eligible[train], eligible[test])
            for fold, (train, test) in enumerate(
                GroupKFold(folds).split(eligible, groups=groups), start=1
            )
        ]

    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for _, group in rows.loc[mask].groupby("video_id", sort=False):
        idx = group.index.to_numpy(dtype=np.int64)
        if len(idx) < 4:
            continue
        cutoff = max(1, min(len(idx) - 1, math.floor(len(idx) * 0.70)))
        train_parts.append(idx[:cutoff])
        test_parts.append(idx[cutoff:])
    if not train_parts:
        return []
    return [(1, np.concatenate(train_parts), np.concatenate(test_parts))]


def build_splits(
    rows: pd.DataFrame,
    *,
    target: TargetSpec = SPIKE_TARGET,
    protocols: Sequence[Protocol] = ("grouped_video", "blocked_temporal_70_30"),
    n_splits: int = 5,
) -> list[Split]:
    """Create outer splits and fit every event threshold on outer-train rows only."""

    mask = target_mask(rows, target)
    values = rows[target.value_column].to_numpy(dtype=np.float64)
    result: list[Split] = []
    for protocol in protocols:
        for fold, train_idx, test_idx in _raw_splits(rows, mask, protocol, n_splits):
            _validate_split(rows, protocol, train_idx, test_idx)
            train_y, test_y, threshold = _threshold_labels(values, train_idx, test_idx, target)
            if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
                continue
            result.append(
                Split(protocol, fold, target, train_idx, test_idx, train_y, test_y, threshold)
            )
    return result


def _validate_split(
    rows: pd.DataFrame, protocol: Protocol, train_idx: np.ndarray, test_idx: np.ndarray
) -> None:
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError(f"row leakage in {protocol}")
    if protocol == "grouped_video":
        train_videos = set(rows.loc[train_idx, "video_id"].astype(str))
        test_videos = set(rows.loc[test_idx, "video_id"].astype(str))
        if train_videos & test_videos:
            raise ValueError("grouped-video leakage")
        return
    for video in rows.loc[np.concatenate([train_idx, test_idx]), "video_id"].unique():
        tr = rows.loc[np.intersect1d(train_idx, rows.index[rows.video_id == video]), "row_index"]
        te = rows.loc[np.intersect1d(test_idx, rows.index[rows.video_id == video]), "row_index"]
        if len(tr) and len(te) and tr.max() >= te.min():
            raise ValueError(f"blocked-temporal ordering violation for {video}")


def inner_validation(rows: pd.DataFrame, split: Split) -> tuple[np.ndarray, np.ndarray, str]:
    """Return indices relative to outer train; never inspect outer-test rows."""

    groups = rows.loc[split.train_idx, "video_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 3:
        folds = min(3, len(unique_groups))
        for train, val in GroupKFold(folds).split(np.arange(len(groups)), split.train_y, groups):
            if len(np.unique(split.train_y[train])) > 1 and len(np.unique(split.train_y[val])) > 1:
                return train, val, f"inner_grouped_video_{folds}fold_first_valid"
    train_parts, val_parts = [], []
    for group in unique_groups:
        relative = np.flatnonzero(groups == group)
        if len(relative) < 4:
            continue
        cutoff = max(1, min(len(relative) - 1, math.floor(len(relative) * 0.80)))
        train_parts.append(relative[:cutoff])
        val_parts.append(relative[cutoff:])
    if train_parts:
        train, val = np.concatenate(train_parts), np.concatenate(val_parts)
        if len(np.unique(split.train_y[train])) > 1 and len(np.unique(split.train_y[val])) > 1:
            return train, val, "inner_blocked_temporal_outer_train_80_20"
    all_train = np.arange(len(split.train_idx))
    return all_train, all_train, "inner_fallback_train_resubstitution"


def ar_matrix(rows: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    values = rows.loc[indices, list(AR_FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("AR matrix contains non-finite values")
    return values


def fold_safe_pca(
    train_x: np.ndarray, test_x: np.ndarray, *, width: int, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit scaling and PCA only on one outer fold's training rows."""

    if width <= 0 or width > min(train_x.shape):
        raise ValueError(f"invalid PCA width {width} for training shape {train_x.shape}")
    scaler = StandardScaler().fit(train_x)
    train_scaled = scaler.transform(train_x)
    test_scaled = scaler.transform(test_x)
    pca = PCA(n_components=width, svd_solver="randomized", random_state=seed).fit(train_scaled)
    return (
        pca.transform(train_scaled).astype(np.float32),
        pca.transform(test_scaled).astype(np.float32),
        {
            "fit_scope": "outer_train_only",
            "width": width,
            "seed": seed,
            "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        },
    )


def causal_history(
    current: np.ndarray,
    row_index: np.ndarray,
    video_id: np.ndarray,
    *,
    window_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return zero-padded past-to-current sequences and an explicit availability mask."""

    if window_rows < 1:
        raise ValueError("window_rows must be positive")
    sequence = np.zeros((len(current), window_rows, current.shape[1]), dtype=np.float32)
    available = np.zeros((len(current), window_rows), dtype=bool)
    lookup = {
        (str(video), int(row)): pos
        for pos, (video, row) in enumerate(zip(video_id, row_index, strict=True))
    }
    for pos, (video, row) in enumerate(zip(video_id, row_index, strict=True)):
        for offset in range(window_rows):
            lag = window_rows - 1 - offset
            source = lookup.get((str(video), int(row) - lag))
            if source is not None:
                sequence[pos, offset] = current[source]
                available[pos, offset] = True
    return sequence, available


def causal_summary(sequence: np.ndarray, available: np.ndarray) -> np.ndarray:
    """Current, causal mean, slope, std, and history-availability features."""

    if sequence.ndim != 3 or available.shape != sequence.shape[:2]:
        raise ValueError("sequence/mask shapes do not agree")
    count = available.sum(axis=1, keepdims=True).clip(min=1).astype(np.float32)
    masked = sequence * available[..., None]
    mean = masked.sum(axis=1) / count
    current = sequence[:, -1]
    first = np.argmax(available, axis=1)
    earliest = sequence[np.arange(len(sequence)), first]
    slope = (current - earliest) / np.maximum(count - 1.0, 1.0)
    variance = (((sequence - mean[:, None]) ** 2) * available[..., None]).sum(axis=1) / count
    availability = (count / sequence.shape[1]).astype(np.float32)
    return np.concatenate([current, mean, slope, np.sqrt(variance), availability], axis=1)
