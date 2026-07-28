"""Fresh VEATIC-only causal response-history features for Phase 02."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CausalHistory:
    levels: np.ndarray
    available: np.ndarray
    deltas: np.ndarray
    rolling_mean: np.ndarray
    rolling_std: np.ndarray
    rolling_min: np.ndarray
    rolling_max: np.ndarray
    rolling_slope: np.ndarray
    rolling_fraction: np.ndarray


def build_causal_history(
    arousal: np.ndarray,
    video_id: np.ndarray,
    row_index: np.ndarray,
    *,
    max_depth: int,
) -> CausalHistory:
    """Build causal level, difference, and rolling tensors without dropping cold-start rows."""

    if max_depth < 1 or not (arousal.shape == video_id.shape == row_index.shape):
        raise ValueError("invalid causal-history inputs")
    rows = len(arousal)
    levels = np.empty((rows, max_depth + 1), dtype=np.float32)
    available = np.empty((rows, max_depth + 1), dtype=np.float32)
    rolling_mean = np.empty((rows, max_depth), dtype=np.float32)
    rolling_std = np.empty((rows, max_depth), dtype=np.float32)
    rolling_min = np.empty((rows, max_depth), dtype=np.float32)
    rolling_max = np.empty((rows, max_depth), dtype=np.float32)
    rolling_slope = np.empty((rows, max_depth), dtype=np.float32)
    rolling_fraction = np.empty((rows, max_depth), dtype=np.float32)

    for video in np.unique(video_id):
        owned = np.flatnonzero(video_id == video)
        if not np.array_equal(row_index[owned], np.arange(len(owned))):
            raise ValueError(f"nonsequential row identity for video {int(video)}")
        values = arousal[owned].astype(np.float64)
        for local_row, global_row in enumerate(owned):
            for lag in range(max_depth + 1):
                source = max(local_row - lag, 0)
                levels[global_row, lag] = values[source]
                available[global_row, lag] = float(local_row >= lag)
            for depth in range(1, max_depth + 1):
                count = min(local_row + 1, depth)
                window = values[local_row - count + 1 : local_row + 1]
                rolling_mean[global_row, depth - 1] = np.mean(window)
                rolling_std[global_row, depth - 1] = np.std(window)
                rolling_min[global_row, depth - 1] = np.min(window)
                rolling_max[global_row, depth - 1] = np.max(window)
                rolling_fraction[global_row, depth - 1] = count / depth
                if count < 2:
                    rolling_slope[global_row, depth - 1] = 0.0
                else:
                    x = np.arange(count, dtype=np.float64)
                    centered = x - np.mean(x)
                    rolling_slope[global_row, depth - 1] = np.sum(
                        centered * (window - np.mean(window))
                    ) / np.sum(centered**2)
    deltas = levels[:, :-1] - levels[:, 1:]
    return CausalHistory(
        levels=levels,
        available=available,
        deltas=deltas,
        rolling_mean=rolling_mean,
        rolling_std=rolling_std,
        rolling_min=rolling_min,
        rolling_max=rolling_max,
        rolling_slope=rolling_slope,
        rolling_fraction=rolling_fraction,
    )


def feature_names(form: str, depth: int) -> tuple[str, ...]:
    if depth < 1:
        raise ValueError("history depth must be positive")
    if form == "current_only":
        return ("current_arousal",)
    if form in {"raw_levels_with_availability_mask", "raw_sequence_with_availability_mask"}:
        return (
            *(f"level_lag_{lag:02d}" for lag in range(depth + 1)),
            *(f"available_lag_{lag:02d}" for lag in range(1, depth + 1)),
        )
    if form == "level_and_first_difference":
        return (
            "current_arousal",
            *(f"delta_lag_{lag:02d}" for lag in range(1, depth + 1)),
            *(f"available_lag_{lag:02d}" for lag in range(1, depth + 1)),
        )
    summary = (
        *(f"mean_depth_{window:02d}" for window in range(1, depth + 1)),
        *(f"std_depth_{window:02d}" for window in range(1, depth + 1)),
        *(f"min_depth_{window:02d}" for window in range(1, depth + 1)),
        *(f"max_depth_{window:02d}" for window in range(1, depth + 1)),
        *(f"slope_depth_{window:02d}" for window in range(1, depth + 1)),
        *(f"fraction_depth_{window:02d}" for window in range(1, depth + 1)),
    )
    if form == "causal_rolling_summary":
        return ("current_arousal", "previous_delta", *summary)
    if form == "combined_levels_differences_summaries":
        return (
            *(f"level_lag_{lag:02d}" for lag in range(depth + 1)),
            *(f"delta_lag_{lag:02d}" for lag in range(1, depth + 1)),
            *(f"available_lag_{lag:02d}" for lag in range(1, depth + 1)),
            *summary,
        )
    raise ValueError(f"unknown Phase 02 feature form: {form}")


def build_feature_matrix(history: CausalHistory, form: str, depth: int) -> np.ndarray:
    """Materialize one registered feature form/depth from the causal history cache."""

    if depth < 1 or depth >= history.levels.shape[1]:
        raise ValueError("history depth is outside the registered cache")
    if form == "current_only":
        matrix = history.levels[:, :1]
    elif form in {"raw_levels_with_availability_mask", "raw_sequence_with_availability_mask"}:
        matrix = np.column_stack(
            [history.levels[:, : depth + 1], history.available[:, 1 : depth + 1]]
        )
    elif form == "level_and_first_difference":
        matrix = np.column_stack(
            [
                history.levels[:, :1],
                history.deltas[:, :depth],
                history.available[:, 1 : depth + 1],
            ]
        )
    else:
        summary = np.column_stack(
            [
                history.rolling_mean[:, :depth],
                history.rolling_std[:, :depth],
                history.rolling_min[:, :depth],
                history.rolling_max[:, :depth],
                history.rolling_slope[:, :depth],
                history.rolling_fraction[:, :depth],
            ]
        )
        if form == "causal_rolling_summary":
            matrix = np.column_stack([history.levels[:, :1], history.deltas[:, :1], summary])
        elif form == "combined_levels_differences_summaries":
            matrix = np.column_stack(
                [
                    history.levels[:, : depth + 1],
                    history.deltas[:, :depth],
                    history.available[:, 1 : depth + 1],
                    summary,
                ]
            )
        else:
            raise ValueError(f"unknown Phase 02 feature form: {form}")
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[1] != len(feature_names(form, depth)) or not np.isfinite(matrix).all():
        raise ValueError("causal feature schema/finiteness mismatch")
    return matrix


def standardize_from_owner(
    features: np.ndarray, owner: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if features.ndim != 2 or owner.shape != (len(features),) or not np.any(owner):
        raise ValueError("invalid feature-standardization ownership")
    mean = np.mean(features[owner], axis=0, dtype=np.float64)
    std = np.std(features[owner], axis=0, dtype=np.float64)
    std[std < np.finfo(np.float32).eps] = 1.0
    transformed = ((features - mean) / std).astype(np.float32)
    return transformed, mean, std
