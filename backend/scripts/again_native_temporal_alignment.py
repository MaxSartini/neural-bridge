"""Native temporal-grid alignment utilities for AGAIN TRIBE outputs.

These helpers keep TRIBE predictions on their real/native row grid. They align
continuous arousal annotations to prediction timestamps or segments, and build
future targets using seconds-based windows instead of row offsets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np


TimingSource = Literal["explicit", "model_tr", "inferred_duration_even_spacing", "unknown"]
InterpolationMethod = Literal["nearest", "linear", "window_mean"]


@dataclass(frozen=True)
class TribeTimeGrid:
    row_start_times: np.ndarray
    row_end_times: np.ndarray
    row_center_times: np.ndarray
    timing_source: TimingSource
    timing_confidence: str
    seconds_per_prediction: float | None
    prediction_rate_hz: float | None
    notes: str = ""


def infer_tribe_time_grid(
    prediction_shape: Sequence[int],
    aligned_video_duration: float,
    explicit_timestamps: Sequence[float] | None = None,
    explicit_segments: Sequence[tuple[float, float]] | None = None,
    model_tr: float | None = None,
) -> TribeTimeGrid:
    """Infer prediction row timing without fabricating extra rows."""

    if not prediction_shape:
        raise ValueError("prediction_shape must include a row dimension")
    row_count = int(prediction_shape[0])
    if row_count <= 0:
        raise ValueError("prediction row count must be positive")
    duration = _positive_float(aligned_video_duration, "aligned_video_duration")

    if explicit_segments is not None:
        if len(explicit_segments) != row_count:
            raise ValueError("explicit_segments length must match prediction row count")
        starts = np.asarray([float(start) for start, _end in explicit_segments], dtype=np.float64)
        ends = np.asarray([float(end) for _start, end in explicit_segments], dtype=np.float64)
        _validate_monotonic_segments(starts, ends)
        return _grid(
            starts,
            ends,
            "explicit",
            "high",
            "explicit segment start/end times supplied",
        )

    if explicit_timestamps is not None:
        centers = np.asarray(explicit_timestamps, dtype=np.float64)
        if len(centers) != row_count:
            raise ValueError("explicit_timestamps length must match prediction row count")
        starts, ends = _bounds_from_centers(centers, duration)
        return _grid(starts, ends, "explicit", "high", "explicit row center timestamps supplied")

    if model_tr is not None:
        tr = _positive_float(model_tr, "model_tr")
        starts = np.arange(row_count, dtype=np.float64) * tr
        ends = np.minimum(starts + tr, duration)
        timing_confidence = "high" if abs((row_count * tr) - duration) <= max(1.0, tr) else "medium"
        return _grid(starts, ends, "model_tr", timing_confidence, f"model TR={tr:g}s supplied")

    starts = np.linspace(0.0, duration, row_count, endpoint=False, dtype=np.float64)
    ends = np.linspace(duration / row_count, duration, row_count, endpoint=True, dtype=np.float64)
    return _grid(
        starts,
        ends,
        "inferred_duration_even_spacing",
        "medium",
        "timestamps inferred by evenly spacing existing prediction rows over aligned duration",
    )


def align_arousal_to_tribe_grid(
    annotation_times: Sequence[float],
    arousal_values: Sequence[float],
    row_center_times: Sequence[float],
    row_start_times: Sequence[float] | None = None,
    row_end_times: Sequence[float] | None = None,
    method: InterpolationMethod = "linear",
) -> dict[str, Any]:
    """Align annotation arousal to each TRIBE row center or segment."""

    times, values = _clean_series(annotation_times, arousal_values)
    centers = np.asarray(row_center_times, dtype=np.float64)
    if method == "nearest":
        indices = np.searchsorted(times, centers, side="left")
        left = np.maximum(indices - 1, 0)
        right = np.minimum(indices, len(times) - 1)
        choose_right = np.abs(times[right] - centers) < np.abs(times[left] - centers)
        nearest = np.where(choose_right, right, left)
        aligned = values[nearest]
        valid = (centers >= times[0]) & (centers <= times[-1])
    elif method == "linear":
        aligned = np.interp(centers, times, values)
        valid = (centers >= times[0]) & (centers <= times[-1])
        aligned = np.where(valid, aligned, np.nan)
    elif method == "window_mean":
        if row_start_times is None or row_end_times is None:
            raise ValueError("window_mean requires row_start_times and row_end_times")
        starts = np.asarray(row_start_times, dtype=np.float64)
        ends = np.asarray(row_end_times, dtype=np.float64)
        aligned = np.full(len(centers), np.nan, dtype=np.float64)
        valid = np.zeros(len(centers), dtype=bool)
        for index, (start, end) in enumerate(zip(starts, ends)):
            mask = (times >= start) & (times <= end)
            if np.any(mask):
                aligned[index] = float(np.mean(values[mask]))
                valid[index] = True
            elif start <= centers[index] <= end and times[0] <= centers[index] <= times[-1]:
                aligned[index] = float(np.interp(centers[index], times, values))
                valid[index] = True
    else:
        raise ValueError(f"Unsupported interpolation method: {method}")

    return {
        "arousal": aligned,
        "valid": valid,
        "interpolation_method": method,
        "missing_or_invalid_count": int(np.sum(~valid)),
        "uses_future_labels_as_features": False,
    }


def build_seconds_based_future_targets(
    annotation_times: Sequence[float],
    arousal_values: Sequence[float],
    row_center_times: Sequence[float],
    current_arousal: Sequence[float],
    spike_threshold: float,
    change_threshold: float,
    aligned_context_end_seconds: float | None = None,
    spike_window: tuple[float, float] = (1.0, 3.0),
    change_horizon_seconds: float = 3.0,
) -> dict[str, np.ndarray]:
    """Build future labels with real-second windows, never row offsets."""

    times, values = _clean_series(annotation_times, arousal_values)
    centers = np.asarray(row_center_times, dtype=np.float64)
    current = np.asarray(current_arousal, dtype=np.float64)
    if len(current) != len(centers):
        raise ValueError("current_arousal length must match row_center_times")
    start_offset, end_offset = spike_window
    if start_offset < 0 or end_offset <= start_offset:
        raise ValueError("spike_window must be positive and increasing")
    context_end = float(times[-1]) if aligned_context_end_seconds is None else min(
        float(times[-1]),
        _positive_float(aligned_context_end_seconds, "aligned_context_end_seconds"),
    )

    spike_labels = np.full(len(centers), np.nan, dtype=np.float64)
    spike_feasible = np.zeros(len(centers), dtype=bool)
    change_labels = np.full(len(centers), np.nan, dtype=np.float64)
    change_feasible = np.zeros(len(centers), dtype=bool)
    change_values = np.full(len(centers), np.nan, dtype=np.float64)

    for index, center in enumerate(centers):
        if not np.isfinite(current[index]):
            continue
        future_start = center + start_offset
        future_end = center + end_offset
        if future_end > context_end:
            continue
        future_mask = (times >= future_start) & (times <= future_end)
        if np.any(future_mask):
            future_max = float(np.max(values[future_mask]))
            spike_labels[index] = float(future_max - current[index] >= spike_threshold)
            spike_feasible[index] = True

        change_time = center + change_horizon_seconds
        if times[0] <= change_time <= context_end:
            future_value = float(np.interp(change_time, times, values))
            delta = future_value - current[index]
            change_values[index] = delta
            change_labels[index] = float(abs(delta) >= change_threshold)
            change_feasible[index] = True

    return {
        "future_spike_1_3s": spike_labels,
        "future_spike_1_3s_feasible": spike_feasible,
        "future_change_p3s": change_labels,
        "future_change_p3s_value": change_values,
        "future_change_p3s_feasible": change_feasible,
        "target_window_units": np.asarray(["seconds"], dtype=object),
    }


def build_native_grid_manifest(
    dataset_name: str,
    video_id: str,
    video_path: str,
    prediction_shape: Sequence[int],
    aligned_video_duration: float,
    annotation_times: Sequence[float],
    arousal_values: Sequence[float],
    alignment_policy: str,
    aligned_context_end_seconds: float | None = None,
    explicit_timestamps: Sequence[float] | None = None,
    explicit_segments: Sequence[tuple[float, float]] | None = None,
    model_tr: float | None = None,
    interpolation_method: InterpolationMethod = "linear",
    spike_threshold: float = 0.05,
    change_threshold: float = 0.05,
) -> tuple[list[dict[str, Any]], TribeTimeGrid]:
    grid = infer_tribe_time_grid(
        prediction_shape=prediction_shape,
        aligned_video_duration=aligned_video_duration,
        explicit_timestamps=explicit_timestamps,
        explicit_segments=explicit_segments,
        model_tr=model_tr,
    )
    context_end = aligned_context_end_seconds
    if context_end is None and alignment_policy == "drop_last_3s_video_keep_annotation_start":
        context_end = min(float(aligned_video_duration), float(np.max(annotation_times)))
    aligned = align_arousal_to_tribe_grid(
        annotation_times,
        arousal_values,
        grid.row_center_times,
        grid.row_start_times,
        grid.row_end_times,
        method=interpolation_method,
    )
    targets = build_seconds_based_future_targets(
        annotation_times,
        arousal_values,
        grid.row_center_times,
        aligned["arousal"],
        spike_threshold=spike_threshold,
        change_threshold=change_threshold,
        aligned_context_end_seconds=context_end,
    )
    rows: list[dict[str, Any]] = []
    for index in range(len(grid.row_center_times)):
        rows.append(
            {
                "dataset_name": dataset_name,
                "video_id": video_id,
                "video_path": video_path,
                "prediction_row_index": index,
                "tribe_time_start_seconds": float(grid.row_start_times[index]),
                "tribe_time_end_seconds": float(grid.row_end_times[index]),
                "tribe_time_center_seconds": float(grid.row_center_times[index]),
                "arousal": _none_if_nan(aligned["arousal"][index]),
                "arousal_interpolation_method": aligned["interpolation_method"],
                "timing_source": grid.timing_source,
                "timing_confidence": grid.timing_confidence,
                "future_spike_1_3s": _none_if_nan(targets["future_spike_1_3s"][index]),
                "future_change_p3s": _none_if_nan(targets["future_change_p3s"][index]),
                "future_change_p3s_value": _none_if_nan(targets["future_change_p3s_value"][index]),
                "future_spike_1_3s_feasible": bool(targets["future_spike_1_3s_feasible"][index]),
                "future_change_p3s_feasible": bool(targets["future_change_p3s_feasible"][index]),
                "alignment_policy": alignment_policy,
                "aligned_context_end_seconds": context_end,
                "dropped_by_alignment": bool(context_end is not None and grid.row_center_times[index] > context_end),
                "notes": grid.notes,
            }
        )
    return rows, grid


def require_benchmarkable_timing(grid: TribeTimeGrid) -> None:
    if grid.timing_source == "unknown":
        raise ValueError("Cannot benchmark with unknown TRIBE timing source")
    if grid.timing_confidence not in {"high", "medium"}:
        raise ValueError(f"Cannot benchmark with timing confidence={grid.timing_confidence}")


def _grid(
    starts: np.ndarray,
    ends: np.ndarray,
    source: TimingSource,
    confidence: str,
    notes: str,
) -> TribeTimeGrid:
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    _validate_monotonic_segments(starts, ends)
    centers = (starts + ends) / 2.0
    widths = ends - starts
    seconds_per_prediction = float(np.median(widths)) if widths.size else None
    prediction_rate_hz = (1.0 / seconds_per_prediction) if seconds_per_prediction and seconds_per_prediction > 0 else None
    return TribeTimeGrid(
        row_start_times=starts,
        row_end_times=ends,
        row_center_times=centers,
        timing_source=source,
        timing_confidence=confidence,
        seconds_per_prediction=seconds_per_prediction,
        prediction_rate_hz=prediction_rate_hz,
        notes=notes,
    )


def _bounds_from_centers(centers: np.ndarray, duration: float) -> tuple[np.ndarray, np.ndarray]:
    if np.any(~np.isfinite(centers)):
        raise ValueError("explicit timestamps must be finite")
    if np.any(np.diff(centers) <= 0):
        raise ValueError("explicit timestamps must be strictly increasing")
    if len(centers) == 1:
        half = duration / 2.0
        return np.asarray([max(0.0, centers[0] - half)]), np.asarray([min(duration, centers[0] + half)])
    midpoints = (centers[:-1] + centers[1:]) / 2.0
    starts = np.concatenate([[max(0.0, centers[0] - (midpoints[0] - centers[0]))], midpoints])
    ends = np.concatenate([midpoints, [min(duration, centers[-1] + (centers[-1] - midpoints[-1]))]])
    return starts, ends


def _clean_series(times: Sequence[float], values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if len(t) != len(v):
        raise ValueError("annotation_times and arousal_values length mismatch")
    mask = np.isfinite(t) & np.isfinite(v)
    t = t[mask]
    v = v[mask]
    if len(t) < 2:
        raise ValueError("at least two finite annotation samples are required")
    order = np.argsort(t)
    t = t[order]
    v = v[order]
    unique_times, inverse = np.unique(t, return_inverse=True)
    if len(unique_times) != len(t):
        sums = np.zeros(len(unique_times), dtype=np.float64)
        counts = np.zeros(len(unique_times), dtype=np.float64)
        np.add.at(sums, inverse, v)
        np.add.at(counts, inverse, 1.0)
        t = unique_times
        v = sums / counts
    return t, v


def _positive_float(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _validate_monotonic_segments(starts: np.ndarray, ends: np.ndarray) -> None:
    if len(starts) != len(ends):
        raise ValueError("segment starts and ends length mismatch")
    if np.any(~np.isfinite(starts)) or np.any(~np.isfinite(ends)):
        raise ValueError("segment times must be finite")
    if np.any(ends <= starts):
        raise ValueError("segment ends must be greater than starts")
    if len(starts) > 1 and np.any(np.diff(starts) < 0):
        raise ValueError("segment starts must be monotonic")


def _none_if_nan(value: Any) -> Any:
    try:
        number = float(value)
    except Exception:
        return value
    return None if not np.isfinite(number) else number
