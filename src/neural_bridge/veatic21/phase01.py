"""Phase 01 label, dynamics, target, and split-ownership audit.

This module deliberately cannot load the cortical representation.  It derives the
prospective VEATIC 2.1 target and ownership design from labels and non-cortical audit
arrays only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neural_bridge.veatic21.bundle import (
    DEFAULT_BUNDLE_ROOT,
    EXPECTED_VIDEO_IDS,
    BundleError,
    _seal_tree,
    assert_safe_delete_target,
)

DEFAULT_PHASE01_ROOT = Path(
    "/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/"
    "phase-01-alignment-dynamics-targets-splits"
)
ROW_HZ = 2.0
ROW_SECONDS = 1.0 / ROW_HZ
QUANTILE_AUDIT_GRID = tuple(round(value / 100, 2) for value in range(50, 100))
ACF_CROSSING_LEVELS = (0.95, 0.90, 0.75, 0.50, 0.25, 0.10, 0.0)
GROUPED_FOLD_COUNT_AUDIT = tuple(range(3, 9))
BLOCKED_TRAIN_FRACTION_AUDIT = tuple(round(value / 100, 2) for value in range(50, 90, 5))
INNER_TRAIN_FRACTION_AUDIT = (0.70, 0.75, 0.80, 0.85)
CONFIRMATION_FRACTION_AUDIT = tuple(round(value / 100, 2) for value in range(10, 31, 5))

FORBIDDEN_ARRAY_KEYS = frozenset(
    {"cortical_prediction", "temporal_diagnostics53", "tribe_grouped_video_feature"}
)
ALLOWED_AUDIT_ARRAY_KEYS = (
    "time_seconds",
    "sample_frame_indices",
    "sample_time_seconds",
    "source_frame_position",
    "source_floor_frame_index",
    "source_ceil_frame_index",
    "source_interp_alpha",
    "source_arousal",
    "source_valence",
    "arousal",
    "valence",
    "luma_mean",
    "luma_std",
    "frame_luma_std_mean",
    "motion_absdiff_mean",
    "black_frame_fraction",
    "duplicate_frame_fraction",
    "quality_black_frame_flag",
    "quality_duplicate_frame_flag",
    "quality_exclusion_flag",
    "quality_weight_suggested",
)


@dataclass(frozen=True)
class Phase01Video:
    video_id: str
    row_index: NDArray[np.int64]
    time_seconds: NDArray[np.float64]
    arousal: NDArray[np.float64]
    valence: NDArray[np.float64]
    audit_arrays: dict[str, NDArray[np.generic]]

    @property
    def row_count(self) -> int:
        return len(self.row_index)


@dataclass(frozen=True)
class TrajectoryFamily:
    washout_rows: int
    horizon_rows: int
    eligible: NDArray[np.bool_]
    endpoint_delta: NDArray[np.float64]
    max_positive_delta: NDArray[np.float64]
    max_negative_delta: NDArray[np.float64]
    max_absolute_delta: NDArray[np.float64]
    total_variation: NDArray[np.float64]
    onset_surprise: NDArray[np.float64]


def _sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _read_rows(path: Path, video_id: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise BundleError(f"empty Phase 01 rows for video {video_id}")
    for expected_index, row in enumerate(rows):
        if row["video_id"] != video_id or int(row["row_index"]) != expected_index:
            raise BundleError(f"Phase 01 row identity failure for video {video_id}")
    return rows


def load_phase01_video(video_id: str, *, bundle_root: Path = DEFAULT_BUNDLE_ROOT) -> Phase01Video:
    """Load labels and allowlisted audit arrays without evaluating forbidden values."""

    if video_id not in EXPECTED_VIDEO_IDS:
        raise BundleError(f"unexpected VEATIC video ID: {video_id}")
    directory = bundle_root / "per_video" / video_id
    rows = _read_rows(directory / "rows.csv", video_id)
    row_count = len(rows)
    row_index = np.arange(row_count, dtype=np.int64)
    times = np.asarray([float(row["time_seconds"]) for row in rows], dtype=np.float64)
    arousal = np.asarray([float(row["arousal"]) for row in rows], dtype=np.float64)
    valence = np.asarray([float(row["valence"]) for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(arousal)) or not np.all(np.isfinite(valence)):
        raise BundleError(f"nonfinite Phase 01 label for video {video_id}")
    if not np.allclose(np.diff(times), ROW_SECONDS, rtol=0.0, atol=1e-7):
        raise BundleError(f"Phase 01 cadence failure for video {video_id}")

    audit_arrays: dict[str, NDArray[np.generic]] = {}
    payload_path = directory / "tribe_v2_cortical_predictions.npz"
    with np.load(payload_path, allow_pickle=False) as payload:
        missing = sorted(set(ALLOWED_AUDIT_ARRAY_KEYS).difference(payload.files))
        if missing:
            raise BundleError(f"Phase 01 audit arrays missing for video {video_id}: {missing}")
        for key in ALLOWED_AUDIT_ARRAY_KEYS:
            if key in FORBIDDEN_ARRAY_KEYS:
                raise BundleError(f"forbidden Phase 01 array entered allowlist: {key}")
            value = np.asarray(payload[key])
            if value.shape[0] != row_count:
                raise BundleError(f"Phase 01 audit row mismatch for video {video_id}: {key}")
            audit_arrays[key] = value
    for name, csv_values in (("arousal", arousal), ("valence", valence)):
        if not np.allclose(
            np.asarray(audit_arrays[name], dtype=np.float64),
            csv_values,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise BundleError(f"Phase 01 label equality failure for video {video_id}: {name}")
    return Phase01Video(video_id, row_index, times, arousal, valence, audit_arrays)


def _load_worker(task: tuple[str, str]) -> Phase01Video:
    video_id, root = task
    return load_phase01_video(video_id, bundle_root=Path(root))


def load_phase01_videos(
    *,
    bundle_root: Path = DEFAULT_BUNDLE_ROOT,
    video_ids: Sequence[str] = EXPECTED_VIDEO_IDS,
    workers: int = 1,
) -> list[Phase01Video]:
    if workers < 1:
        raise ValueError("workers must be positive")
    tasks = [(video_id, str(bundle_root)) for video_id in video_ids]
    if workers == 1:
        videos = [_load_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            videos = list(pool.map(_load_worker, tasks))
    if [video.video_id for video in videos] != list(video_ids):
        raise BundleError("Phase 01 loader changed video order")
    return videos


def derive_trajectory_family(
    values: NDArray[np.float64], *, washout_rows: int, horizon_rows: int
) -> TrajectoryFamily:
    """Derive complementary summaries from one intact future trajectory."""

    if washout_rows < 0 or horizon_rows < 1:
        raise ValueError("washout_rows must be nonnegative and horizon_rows must be positive")
    row_count = len(values)
    eligible = np.zeros(row_count, dtype=np.bool_)
    output = [np.full(row_count, np.nan, dtype=np.float64) for _ in range(7)]
    endpoint, max_positive, max_negative, max_absolute, variation, onset, _ = output
    for row_index in range(row_count):
        first = row_index + washout_rows + 1
        stop = first + horizon_rows
        if stop > row_count:
            continue
        future = values[first:stop]
        relative = future - values[row_index]
        steps = np.diff(np.concatenate(([values[first - 1]], future)))
        eligible[row_index] = True
        endpoint[row_index] = relative[-1]
        max_positive[row_index] = max(0.0, float(np.max(relative)))
        max_negative[row_index] = min(0.0, float(np.min(relative)))
        max_absolute[row_index] = float(np.max(np.abs(relative)))
        variation[row_index] = float(np.sum(np.abs(steps)))
        # Surprise is the largest step relative to the preceding target-window step scale.
        scale = float(np.median(np.abs(steps))) + np.finfo(np.float64).eps
        onset[row_index] = float(np.max(np.abs(steps)) / scale)
    return TrajectoryFamily(
        washout_rows,
        horizon_rows,
        eligible,
        endpoint,
        max_positive,
        max_negative,
        max_absolute,
        variation,
        onset,
    )


def autocorrelation_profile(
    videos: Sequence[Phase01Video], *, label: str, max_lag_rows: int
) -> list[dict[str, Any]]:
    """Compute per-video and pair-weighted ACF without crossing boundaries."""

    if label not in {"arousal", "valence"}:
        raise ValueError(f"unsupported label: {label}")
    profile: list[dict[str, Any]] = []
    for lag in range(1, max_lag_rows + 1):
        correlations: list[float] = []
        weights: list[int] = []
        for video in videos:
            values = getattr(video, label)
            if len(values) <= lag + 2:
                continue
            left, right = values[:-lag], values[lag:]
            if np.std(left) == 0 or np.std(right) == 0:
                continue
            correlations.append(float(np.corrcoef(left, right)[0, 1]))
            weights.append(len(left))
        profile.append(
            {
                "label": label,
                "lag_rows": lag,
                "lag_seconds": lag * ROW_SECONDS,
                "eligible_videos": len(correlations),
                "eligible_pairs": int(sum(weights)),
                "median_video_correlation": _optional_float(np.median(correlations)),
                "pair_weighted_correlation": _optional_float(
                    np.average(correlations, weights=weights) if correlations else None
                ),
            }
        )
    return profile


def partial_autocorrelation_profile(
    videos: Sequence[Phase01Video], *, label: str, max_lag_rows: int
) -> list[dict[str, Any]]:
    """Estimate lag-specific partial correlation using within-video OLS residuals."""

    if label not in {"arousal", "valence"}:
        raise ValueError(f"unsupported label: {label}")
    records = []
    for lag in range(1, max_lag_rows + 1):
        values_by_video: list[float] = []
        weights: list[int] = []
        for video in videos:
            values = getattr(video, label)
            if len(values) <= lag + 3:
                continue
            current = values[lag:]
            distant = values[:-lag]
            if lag == 1:
                residual_current, residual_distant = current, distant
            else:
                controls = np.column_stack(
                    [values[lag - offset : -offset] for offset in range(1, lag)]
                )
                design = np.column_stack((np.ones(len(controls)), controls))
                residual_current = (
                    current - design @ np.linalg.lstsq(design, current, rcond=None)[0]
                )
                residual_distant = (
                    distant - design @ np.linalg.lstsq(design, distant, rcond=None)[0]
                )
            if np.std(residual_current) == 0 or np.std(residual_distant) == 0:
                continue
            values_by_video.append(float(np.corrcoef(residual_current, residual_distant)[0, 1]))
            weights.append(len(current))
        records.append(
            {
                "label": label,
                "lag_rows": lag,
                "lag_seconds": lag * ROW_SECONDS,
                "eligible_videos": len(values_by_video),
                "pair_weighted_partial_correlation": _optional_float(
                    np.average(values_by_video, weights=weights) if values_by_video else None
                ),
                "median_video_partial_correlation": _optional_float(
                    np.median(values_by_video) if values_by_video else None
                ),
            }
        )
    return records


def _optional_float(value: Any) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


def _quantile_dict(values: NDArray[np.float64]) -> dict[str, float]:
    return {
        str(quantile): float(np.quantile(values, quantile))
        for quantile in (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    }


def describe_labels_and_audits(videos: Sequence[Phase01Video]) -> dict[str, Any]:
    """Describe all allowed Phase 01 inputs across rows and videos."""

    row_counts = np.asarray([video.row_count for video in videos], dtype=np.float64)
    result: dict[str, Any] = {
        "video_count": len(videos),
        "total_rows": int(np.sum(row_counts)),
        "duration_rows": _quantile_dict(row_counts),
        "duration_seconds": _quantile_dict(row_counts * ROW_SECONDS),
        "labels": {},
        "audit_arrays": {},
    }
    for label in ("arousal", "valence"):
        pooled = np.concatenate([getattr(video, label) for video in videos])
        step = np.concatenate([np.diff(getattr(video, label)) for video in videos])
        per_video_mean = np.asarray(
            [np.mean(getattr(video, label)) for video in videos], dtype=np.float64
        )
        per_video_std = np.asarray(
            [np.std(getattr(video, label)) for video in videos], dtype=np.float64
        )
        result["labels"][label] = {
            "level_quantiles": _quantile_dict(pooled),
            "step_delta_quantiles": _quantile_dict(step),
            "absolute_step_delta_quantiles": _quantile_dict(np.abs(step)),
            "per_video_mean_quantiles": _quantile_dict(per_video_mean),
            "per_video_std_quantiles": _quantile_dict(per_video_std),
            "constant_video_count": int(np.sum(per_video_std == 0)),
        }
    for key in ALLOWED_AUDIT_ARRAY_KEYS:
        if key in {"arousal", "valence", "sample_frame_indices", "sample_time_seconds"}:
            continue
        arrays = [np.asarray(video.audit_arrays[key]) for video in videos]
        pooled = np.concatenate([value.reshape(-1) for value in arrays]).astype(np.float64)
        result["audit_arrays"][key] = {
            "shape_tail": list(arrays[0].shape[1:]),
            "dtype": str(arrays[0].dtype),
            "finite_fraction": float(np.mean(np.isfinite(pooled))),
            "quantiles": _quantile_dict(pooled[np.isfinite(pooled)]),
        }
    return result


def _acf_crossings(profile: Sequence[dict[str, Any]]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for level in ACF_CROSSING_LEVELS:
        record = next(
            (
                item
                for item in profile
                if item["pair_weighted_correlation"] is not None
                and item["pair_weighted_correlation"] <= level
            ),
            None,
        )
        result[str(level)] = None if record is None else int(record["lag_rows"])
    return result


def derive_geometry_registry(
    videos: Sequence[Phase01Video], acf: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Derive a compact VEATIC-only geometry range plus named comparison anchors."""

    minimum_rows = min(video.row_count for video in videos)
    candidates: set[tuple[int, int, str]] = set()
    for label in ("arousal", "valence"):
        crossings = _acf_crossings(acf[label])
        history = sorted(
            {
                value
                for key, value in crossings.items()
                if key in {"0.9", "0.75", "0.5", "0.25"} and value is not None
            }
        )
        # The strongest short, medium, and long label-decay scales become audit anchors.
        for horizon in history[:4]:
            if horizon < minimum_rows:
                candidates.add((0, max(1, horizon), f"{label}_acf_decay"))
        for washout_level in ("0.95", "0.9"):
            washout = crossings[washout_level]
            for horizon_level in ("0.75", "0.5"):
                endpoint = crossings[horizon_level]
                if washout is None or endpoint is None:
                    continue
                horizon = max(1, endpoint - washout)
                if washout + horizon < minimum_rows:
                    candidates.add((washout, horizon, f"{label}_acf_washout"))
    registry = [
        {
            "washout_rows": washout,
            "washout_seconds": washout * ROW_SECONDS,
            "horizon_rows": horizon,
            "horizon_seconds": horizon * ROW_SECONDS,
            "derivation": derivation,
            "veatic_selected": True,
        }
        for washout, horizon, derivation in sorted(candidates)
    ]
    registry.append(
        {
            "washout_rows": 4,
            "washout_seconds": 2.0,
            "horizon_rows": 7,
            "horizon_seconds": 3.5,
            "derivation": "AGAIN rows +4 through +10 comparability anchor only",
            "veatic_selected": False,
        }
    )
    return registry


def trajectory_support_table(
    videos: Sequence[Phase01Video],
    *,
    geometries: Sequence[tuple[int, int]],
    quantiles: Sequence[float] = QUANTILE_AUDIT_GRID,
) -> list[dict[str, Any]]:
    """Audit target and threshold support without using cortical outcomes."""

    metrics = (
        "endpoint_delta",
        "max_positive_delta",
        "max_negative_magnitude",
        "max_absolute_delta",
        "total_variation",
        "onset_surprise",
    )
    records: list[dict[str, Any]] = []
    for label in ("arousal", "valence"):
        for washout, horizon in geometries:
            per_metric: dict[str, list[tuple[str, NDArray[np.float64]]]] = {
                metric: [] for metric in metrics
            }
            for video in videos:
                family = derive_trajectory_family(
                    getattr(video, label), washout_rows=washout, horizon_rows=horizon
                )
                eligible = family.eligible
                values = {
                    "endpoint_delta": family.endpoint_delta[eligible],
                    "max_positive_delta": family.max_positive_delta[eligible],
                    "max_negative_magnitude": -family.max_negative_delta[eligible],
                    "max_absolute_delta": family.max_absolute_delta[eligible],
                    "total_variation": family.total_variation[eligible],
                    "onset_surprise": family.onset_surprise[eligible],
                }
                for metric in metrics:
                    per_metric[metric].append((video.video_id, values[metric]))
            for metric, values_by_video in per_metric.items():
                nonempty = [(key, value) for key, value in values_by_video if len(value)]
                pooled = np.concatenate([value for _, value in nonempty])
                for quantile in quantiles:
                    threshold = float(np.quantile(pooled, quantile))
                    counts = [int(np.sum(value >= threshold)) for _, value in nonempty]
                    records.append(
                        {
                            "label": label,
                            "washout_rows": washout,
                            "horizon_rows": horizon,
                            "metric": metric,
                            "quantile": quantile,
                            "exploratory_global_threshold": threshold,
                            "eligible_rows": len(pooled),
                            "eligible_videos": len(nonempty),
                            "event_rows": int(sum(counts)),
                            "event_videos": int(sum(count > 0 for count in counts)),
                            "zero_event_videos": int(sum(count == 0 for count in counts)),
                        }
                    )
    return records


def blocked_membership(
    eligible: NDArray[np.bool_], *, outer_train_fraction: float, inner_train_fraction: float
) -> NDArray[np.int8]:
    """Return 1=inner train, 2=inner val, 3=outer test for one video."""

    if not 0 < outer_train_fraction < 1 or not 0 < inner_train_fraction < 1:
        raise ValueError("split fractions must lie strictly between zero and one")
    indices = np.flatnonzero(eligible)
    membership = np.zeros(len(eligible), dtype=np.int8)
    outer_boundary = int(np.floor(len(indices) * outer_train_fraction))
    inner_boundary = int(np.floor(outer_boundary * inner_train_fraction))
    membership[indices[:inner_boundary]] = 1
    membership[indices[inner_boundary:outer_boundary]] = 2
    membership[indices[outer_boundary:]] = 3
    return membership


def audit_split_candidates(
    videos: Sequence[Phase01Video], geometries: Sequence[tuple[int, int]]
) -> dict[str, Any]:
    blocked: list[dict[str, Any]] = []
    for washout, horizon in geometries:
        masks = [
            derive_trajectory_family(
                video.arousal, washout_rows=washout, horizon_rows=horizon
            ).eligible
            for video in videos
        ]
        for outer in BLOCKED_TRAIN_FRACTION_AUDIT:
            for inner in INNER_TRAIN_FRACTION_AUDIT:
                counts = np.zeros(4, dtype=np.int64)
                participating = np.zeros(4, dtype=np.int64)
                strict = True
                for mask in masks:
                    membership = blocked_membership(
                        mask,
                        outer_train_fraction=outer,
                        inner_train_fraction=inner,
                    )
                    for part in (1, 2, 3):
                        row_indices = np.flatnonzero(membership == part)
                        counts[part] += len(row_indices)
                        participating[part] += bool(len(row_indices))
                    train = np.flatnonzero(membership == 1)
                    val = np.flatnonzero(membership == 2)
                    test = np.flatnonzero(membership == 3)
                    strict &= (
                        bool(len(train) and len(val) and len(test))
                        and train[-1] < val[0]
                        and val[-1] < test[0]
                    )
                blocked.append(
                    {
                        "washout_rows": washout,
                        "horizon_rows": horizon,
                        "outer_train_fraction": outer,
                        "inner_train_fraction_within_outer_train": inner,
                        "inner_train_rows": int(counts[1]),
                        "inner_validation_rows": int(counts[2]),
                        "outer_test_rows": int(counts[3]),
                        "inner_train_videos": int(participating[1]),
                        "inner_validation_videos": int(participating[2]),
                        "outer_test_videos": int(participating[3]),
                        "all_eligible_videos_participate": bool(
                            np.all(participating[1:] == len(videos))
                        ),
                        "strict_forward_time_zero_overlap": bool(strict),
                    }
                )
    row_counts = np.asarray([video.row_count for video in videos], dtype=np.int64)
    grouped = []
    for folds in GROUPED_FOLD_COUNT_AUDIT:
        assignments = _balanced_fold_assignments(videos, folds=folds)
        fold_rows = [int(np.sum(row_counts[assignments == fold])) for fold in range(folds)]
        fold_videos = [int(np.sum(assignments == fold)) for fold in range(folds)]
        grouped.append(
            {
                "fold_count": folds,
                "fold_rows": fold_rows,
                "fold_videos": fold_videos,
                "minimum_test_videos": min(fold_videos),
                "row_imbalance_ratio": max(fold_rows) / min(fold_rows),
                "whole_video_only": True,
            }
        )
    confirmation = [
        {
            "fraction": fraction,
            "video_count": int(round(len(videos) * fraction)),
            "remaining_development_videos": len(videos) - int(round(len(videos) * fraction)),
            "whole_video_only": True,
        }
        for fraction in CONFIRMATION_FRACTION_AUDIT
    ]
    return {
        "blocked_forward": blocked,
        "grouped_video": grouped,
        "zero_label_confirmation": confirmation,
        "historical_comparability_anchor": {
            "name": "AGAIN blocked_temporal_70_30 plus inner 80/20",
            "transfers_as_selection": False,
            "meaning": "all eligible videos; earlier/later rows within each video",
        },
    }


def _balanced_fold_assignments(videos: Sequence[Phase01Video], *, folds: int) -> NDArray[np.int64]:
    """Assign whole videos deterministically while greedily balancing row counts."""

    if folds < 2 or folds > len(videos):
        raise ValueError("invalid grouped fold count")
    totals = np.zeros(folds, dtype=np.int64)
    assignments = np.zeros(len(videos), dtype=np.int64)
    ordered = sorted(range(len(videos)), key=lambda index: (-videos[index].row_count, index))
    for index in ordered:
        fold = int(np.argmin(totals))
        assignments[index] = fold
        totals[fold] += videos[index].row_count
    return assignments


def build_overlap_ledger(registry: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    histories = (1, 2, 4, 8, 16, 32, 64)
    return [
        {
            "history_rows": history,
            "history_relative_rows": [-history + 1, 0],
            "washout_rows": item["washout_rows"],
            "target_relative_rows": [
                item["washout_rows"] + 1,
                item["washout_rows"] + item["horizon_rows"],
            ],
            "history_target_overlap": False,
        }
        for item in registry
        for history in histories
    ]


def shared_computation_contract() -> dict[str, Any]:
    return {
        "share_allowed": [
            "one intact future trajectory primitive per label and geometry",
            "row identity and causal context indices",
            "split membership when row mask and ownership are identical",
            "later fold-owned scaling/PCA only for identical training ownership",
        ],
        "must_remain_separate": [
            "target-specific thresholds and AR opponents",
            "event and continuous losses, heads, checkpoints, metrics, and confirmation",
            "arousal and valence scientific claims",
            "grouped-video and blocked-forward conclusions",
        ],
        "combined_challenger_gate": (
            "allowed only after independently confirmed event and continuous specialists; "
            "must beat each specialist on its own locked endpoint"
        ),
        "zero_label_gate": (
            "allowed only after supervised specialists and combined challenger are resolved"
        ),
    }


def analyze_phase01(videos: Sequence[Phase01Video]) -> dict[str, Any]:
    max_lag = min(120, min(video.row_count for video in videos) - 4)
    descriptions = describe_labels_and_audits(videos)
    acf = {
        label: autocorrelation_profile(videos, label=label, max_lag_rows=max_lag)
        for label in ("arousal", "valence")
    }
    pacf = {
        label: partial_autocorrelation_profile(videos, label=label, max_lag_rows=min(32, max_lag))
        for label in ("arousal", "valence")
    }
    registry = derive_geometry_registry(videos, acf)
    geometries = sorted(
        {(int(item["washout_rows"]), int(item["horizon_rows"])) for item in registry}
    )
    return {
        "summary": {
            "schema_version": "veatic21_phase01_result_v1",
            "status": "descriptive_complete_no_cortical_values_read",
            "created_at": datetime.now(UTC).isoformat(),
            "video_count": len(videos),
            "total_rows": sum(video.row_count for video in videos),
            "row_hz": ROW_HZ,
            "forbidden_array_keys_read": [],
        },
        "label_and_audit_distributions": descriptions,
        "autocorrelation": acf,
        "partial_autocorrelation": pacf,
        "geometry_registry": registry,
        "threshold_support": trajectory_support_table(videos, geometries=geometries),
        "split_candidate_audit": audit_split_candidates(videos, geometries),
        "target_overlap_ledger": build_overlap_ledger(registry),
        "shared_computation_contract": shared_computation_contract(),
    }


def _artifact_manifest(root: Path, names: Iterable[str]) -> dict[str, Any]:
    files = [
        {"path": name, "bytes": (root / name).stat().st_size, "sha256": _sha256(root / name)}
        for name in names
    ]
    return {"schema_version": "veatic21_phase01_artifact_manifest_v1", "files": files}


def run_phase01(
    *,
    bundle_root: Path = DEFAULT_BUNDLE_ROOT,
    output_root: Path = DEFAULT_PHASE01_ROOT,
    registration_path: Path,
    workers: int = 12,
) -> dict[str, Any]:
    """Run and atomically publish the registered label-only Phase 01 analysis."""

    output_root = output_root.resolve()
    if output_root.exists():
        raise BundleError(f"refusing to overwrite existing Phase 01 root: {output_root}")
    assert_safe_delete_target(output_root)
    videos = load_phase01_videos(bundle_root=bundle_root, workers=workers)
    analysis = analyze_phase01(videos)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent)
    )
    try:
        names = []
        for key, value in analysis.items():
            name = key.replace("_", "-") + ".json"
            _write_json(temporary / name, value)
            names.append(name)
        _write_json(
            temporary / "provenance.json",
            {
                "bundle_root": str(bundle_root.resolve()),
                "bundle_manifest_sha256": _sha256(bundle_root / "bundle-manifest.json"),
                "registration_path": str(registration_path.resolve()),
                "registration_sha256": _sha256(registration_path.resolve()),
                "workers": workers,
                "cortical_values_read": False,
                "predictive_model_fit": False,
            },
        )
        names.append("provenance.json")
        _write_json(temporary / "artifact-manifest.json", _artifact_manifest(temporary, names))
        os.replace(temporary, output_root)
        _seal_tree(output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return verify_phase01(output_root=output_root)


def verify_phase01(*, output_root: Path = DEFAULT_PHASE01_ROOT) -> dict[str, Any]:
    root = output_root.resolve()
    summary = json.loads((root / "summary.json").read_text())
    manifest = json.loads((root / "artifact-manifest.json").read_text())
    for record in manifest["files"]:
        path = root / record["path"]
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise BundleError(f"Phase 01 artifact mismatch: {path}")
    if summary["status"] != "descriptive_complete_no_cortical_values_read":
        raise BundleError("Phase 01 summary is not a clean descriptive result")
    return {
        "status": "pass",
        "phase01_root": str(root),
        "video_count": summary["video_count"],
        "total_rows": summary["total_rows"],
        "artifact_manifest_sha256": _sha256(root / "artifact-manifest.json"),
    }


def benchmark_phase01_label_audit(
    *,
    bundle_root: Path = DEFAULT_BUNDLE_ROOT,
    worker_counts: Sequence[int],
    repeats: int = 3,
) -> dict[str, Any]:
    """Benchmark the complete real-input loader and descriptive label/audit pass."""

    results: dict[str, list[float]] = {}
    checksums: set[str] = set()
    for workers in worker_counts:
        durations = []
        for _ in range(repeats):
            started = time.perf_counter()
            videos = load_phase01_videos(bundle_root=bundle_root, workers=workers)
            description = describe_labels_and_audits(videos)
            acf = autocorrelation_profile(videos, label="arousal", max_lag_rows=64)
            payload = json.dumps({"description": description, "acf": acf}, sort_keys=True).encode()
            checksums.add(hashlib.sha256(payload).hexdigest())
            durations.append(time.perf_counter() - started)
        results[str(workers)] = durations
    if len(checksums) != 1:
        raise BundleError("Phase 01 benchmark topologies produced unequal results")
    medians = {key: float(np.median(value)) for key, value in results.items()}
    selected = min(medians, key=medians.__getitem__)
    return {
        "workload": "124-video allowlisted NPZ/CSV load, full distribution audit, arousal ACF64",
        "topology_repeat_end_to_end_seconds": results,
        "median_seconds": medians,
        "selected_workers": int(selected),
        "selected_median_seconds": medians[selected],
        "result_sha256": next(iter(checksums)),
        "gpu_disposition": "not_applicable_for_csv_zip_decompression_and_small_label_statistics",
    }
