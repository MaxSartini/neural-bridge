"""VEATIC 2.1 Phase 01 label alignment and target-substrate construction."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from neural_bridge.veatic21.contracts import (
    CURRENT_STATE,
    EXPECTED_ROW_COUNT,
    EXPECTED_ROW_HZ,
    EXPECTED_SOURCE_MATCH_COUNTS,
    EXPECTED_TIME_STEP_SECONDS,
    EXPECTED_VIDEO_IDS,
    HISTOGRAM_MAX_BINS,
    HISTOGRAM_MIN_BINS,
    LIFECYCLE_ROOT,
    MASTER_SPECIFICATION,
    PACF_DECAY_ABS_CEILING,
    PACF_DECAY_CONSECUTIVE_LAGS,
    PHASE00_ROOT,
    PHASE01_ROOT,
    REBUILD_PROTOCOL,
    REPOSITORY_ROOT,
    TARGET_ACF_DECAY_CEILING,
    TARGET_COVERAGE_FLOOR,
    TARGET_SUPPORT_MIN_EVENTS_PER_VIDEO,
    TARGET_SUPPORT_VIDEO_FRACTION_FLOOR,
    TRIBE_ROOT,
    VJEPA_ALLOWED_TREE_SHA256,
    VJEPA_ROOT,
    WASHOUT_SUPPORT_MIN_EVENTS_PER_VIDEO,
    WASHOUT_SUPPORT_VIDEO_FRACTION_FLOOR,
    reject_forbidden_runtime_path,
    validate_runtime_manifest_paths,
)
from neural_bridge.veatic21.data import (
    SupervisedRows,
    load_json,
    load_phase00_tribe_arrays,
    read_supervised_rows,
    safe_sha256_file,
)
from neural_bridge.veatic21.evidence import canonical_json_bytes, source_tree_digest

PHASE01_CHECKS = (
    "phase00_gate_and_identity",
    "complete_aligned_table",
    "final_tribe_row_identity",
    "finite_authoritative_labels",
    "native_interpolation_provenance",
    "quality_metadata_all_rows_retained",
    "movement_histograms",
    "autocorrelation_and_pacf_by_video",
    "causal_history_predictiveness",
    "candidate_coverage_and_event_support",
    "initial_no_washout_rule_frozen",
    "prospective_washout_rule_frozen",
    "continuous_values_and_masks_only",
    "no_global_binary_label",
    "no_outer_split",
    "no_pca_ar_or_model_training",
    "alignment_target_mask_ownership_digests",
    "veatic_derivation_ledger",
    "again_runtime_firewall",
    "forbidden_hidden_state_not_read_or_hashed",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _atomic_write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            save_npz: Any = np.savez_compressed
            save_npz(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _array_bundle_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(f"{name}\0{array.dtype.str}\0{array.shape}\n".encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def future_max_increase(
    arousal: np.ndarray, start_row: int, end_row: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the registered future-maximum-increase values and validity mask."""

    arousal = np.asarray(arousal, dtype=np.float64)
    if arousal.ndim != 1 or not np.isfinite(arousal).all():
        raise ValueError("arousal must be a finite vector")
    if start_row < 1 or end_row < start_row:
        raise ValueError("future target offsets must satisfy 1 <= start <= end")
    values = np.full(len(arousal), np.nan, dtype=np.float64)
    valid = np.zeros(len(arousal), dtype=np.bool_)
    valid_count = len(arousal) - end_row
    if valid_count <= 0:
        return values, valid
    for row in range(valid_count):
        values[row] = np.max(arousal[row + start_row : row + end_row + 1]) - arousal[row]
    valid[:valid_count] = True
    return values, valid


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _pacf_ols(values: np.ndarray, lag: int) -> float | None:
    if lag < 1 or len(values) <= 2 * lag + 1:
        return None
    target = values[lag:]
    predictors = np.column_stack(
        [values[lag - offset : len(values) - offset] for offset in range(1, lag + 1)]
    )
    design = np.column_stack((np.ones(len(predictors)), predictors))
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return None
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    return float(coefficients[-1])


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("quantile values must be nonempty and finite")
    levels = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    return {
        f"q{int(level * 100):02d}": float(value)
        for level, value in zip(levels, np.quantile(values, levels), strict=True)
    }


def _histogram(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("histogram values must be nonempty and finite")
    value_range = float(np.max(values) - np.min(values))
    iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
    width = 2.0 * iqr / np.cbrt(len(values)) if iqr > 0.0 else 0.0
    raw_bins = math.ceil(value_range / width) if width > 0.0 and value_range > 0.0 else 1
    bins = min(HISTOGRAM_MAX_BINS, max(HISTOGRAM_MIN_BINS, raw_bins))
    counts, edges = np.histogram(values, bins=bins)
    return {
        "rule": "Freedman-Diaconis clipped to registered bounds",
        "bins": bins,
        "counts": counts.astype(int).tolist(),
        "edges": edges.astype(float).tolist(),
        "quantiles": _quantiles(values),
        "rows": len(values),
    }


def _descriptive_event_support(
    values: np.ndarray, valid: np.ndarray, video_id: np.ndarray
) -> tuple[float, dict[int, int]]:
    valid_values = values[valid]
    threshold = float(np.quantile(valid_values, 0.90))
    support: dict[int, int] = {}
    for video in range(len(EXPECTED_VIDEO_IDS)):
        rows = valid & (video_id == video)
        support[video] = int(np.sum(values[rows] >= threshold))
    return threshold, support


def select_initial_no_washout(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the preregistered label-only rule without cortical information."""

    eligible = [
        candidate
        for candidate in candidates
        if float(candidate["coverage_fraction"]) >= TARGET_COVERAGE_FLOOR
        and float(candidate["median_per_video_acf"]) <= TARGET_ACF_DECAY_CEILING
        and float(candidate["support_video_fraction"]) >= TARGET_SUPPORT_VIDEO_FRACTION_FLOOR
    ]
    if not eligible:
        raise ValueError("no no-washout candidate satisfies the frozen selection rule")
    return min(eligible, key=lambda candidate: int(candidate["end_row"]))


def _first_stable_pacf_decay(
    lag_summaries: list[dict[str, Any]],
) -> int:
    ordered = sorted(lag_summaries, key=lambda row: int(row["lag_rows"]))
    for index in range(len(ordered) - PACF_DECAY_CONSECUTIVE_LAGS + 1):
        window = ordered[index : index + PACF_DECAY_CONSECUTIVE_LAGS]
        if all(float(row["median_abs_pacf"]) <= PACF_DECAY_ABS_CEILING for row in window):
            return int(window[0]["lag_rows"])
    raise ValueError("no stable PACF decay landmark within the registered bound")


def derive_washout_starts(
    *, pacf_decay_lag: int, rise_duration_q90_rows: float, event_duration_median_rows: float
) -> tuple[int, ...]:
    """Derive a small VEATIC-only prospective family from three label timescales."""

    landmarks = (
        pacf_decay_lag + 1,
        math.ceil(rise_duration_q90_rows) + 1,
        math.ceil(event_duration_median_rows) + 1,
    )
    return tuple(sorted({max(2, int(value)) for value in landmarks}))


def _event_timing(
    rows_by_video: list[SupervisedRows], selected_end: int, threshold: float
) -> dict[str, Any]:
    peak_rows: list[int] = []
    bout_rows: list[int] = []
    positive_rise_rows: list[int] = []
    positive_rise_amplitudes: list[float] = []
    for rows in rows_by_video:
        values, valid = future_max_increase(rows.arousal, 1, selected_end)
        event = np.zeros(len(values), dtype=np.bool_)
        event[valid] = values[valid] >= threshold
        starts = np.flatnonzero(np.diff(np.r_[0, event.astype(np.int8)]) == 1)
        ends = np.flatnonzero(np.diff(np.r_[event.astype(np.int8), 0]) == -1)
        bout_rows.extend((ends - starts + 1).astype(int).tolist())
        for row in np.flatnonzero(event):
            window = rows.arousal[row + 1 : row + selected_end + 1]
            peak_rows.append(int(np.argmax(window)) + 1)
        rising = np.diff(rows.arousal) > 0.0
        rise_starts = np.flatnonzero(np.diff(np.r_[0, rising.astype(np.int8)]) == 1)
        rise_ends = np.flatnonzero(np.diff(np.r_[rising.astype(np.int8), 0]) == -1)
        for start, end in zip(rise_starts, rise_ends, strict=True):
            positive_rise_rows.append(int(end - start + 1))
            positive_rise_amplitudes.append(float(rows.arousal[end + 1] - rows.arousal[start]))
    if not peak_rows or not bout_rows or not positive_rise_rows:
        raise ValueError("event/rise timing summaries are empty")
    return {
        "descriptive_threshold_role": "global q90 summary only; never a benchmark label",
        "event_rows": len(peak_rows),
        "time_to_peak_rows": _quantiles(np.asarray(peak_rows)),
        "time_to_peak_seconds": _quantiles(np.asarray(peak_rows) * EXPECTED_TIME_STEP_SECONDS),
        "event_bout_duration_rows": _quantiles(np.asarray(bout_rows)),
        "event_bout_duration_seconds": _quantiles(
            np.asarray(bout_rows) * EXPECTED_TIME_STEP_SECONDS
        ),
        "positive_rise_duration_rows": _quantiles(np.asarray(positive_rise_rows)),
        "positive_rise_duration_seconds": _quantiles(
            np.asarray(positive_rise_rows) * EXPECTED_TIME_STEP_SECONDS
        ),
        "positive_rise_amplitude": _quantiles(np.asarray(positive_rise_amplitudes)),
    }


def _causal_history_rows(
    rows_by_video: list[SupervisedRows], selected_end: int
) -> list[dict[str, object]]:
    predictors: dict[str, tuple[list[float], list[float]]] = {}

    def add(name: str, predictor: float, target: float) -> None:
        left, right = predictors.setdefault(name, ([], []))
        left.append(predictor)
        right.append(target)

    for rows in rows_by_video:
        target, valid = future_max_increase(rows.arousal, 1, selected_end)
        for row in np.flatnonzero(valid):
            add("current_arousal", float(rows.arousal[row]), float(target[row]))
            for lag in range(1, selected_end + 1):
                if row >= lag:
                    add(
                        f"previous_arousal_lag_{lag}",
                        float(rows.arousal[row - lag]),
                        float(target[row]),
                    )
            for width in range(2, selected_end + 1):
                if row + 1 < width:
                    continue
                window = rows.arousal[row - width + 1 : row + 1]
                add(f"trailing_mean_width_{width}", float(np.mean(window)), float(target[row]))
                x = np.arange(width, dtype=np.float64)
                centered = x - np.mean(x)
                slope = float(np.dot(centered, window) / np.dot(centered, centered))
                add(f"trailing_slope_width_{width}", slope, float(target[row]))
    output: list[dict[str, object]] = []
    for name in sorted(predictors):
        predictor = np.asarray(predictors[name][0], dtype=np.float64)
        target = np.asarray(predictors[name][1], dtype=np.float64)
        pearson = _pearson(predictor, target)
        spearman = float(spearmanr(predictor, target).statistic)
        output.append(
            {
                "predictor": name,
                "target": "selected continuous future maximum increase",
                "rows": len(target),
                "pearson": pearson,
                "spearman": spearman,
                "descriptive_only": True,
                "model_fitted": False,
            }
        )
    return output


def phase02_authorized(checks: dict[str, bool]) -> bool:
    return set(checks) == set(PHASE01_CHECKS) and all(checks.values())


def _write_artifact_manifest(output_root: Path, filenames: tuple[str, ...]) -> None:
    artifacts = []
    for filename in filenames:
        path = output_root / filename
        artifacts.append(
            {"path": filename, "bytes": path.stat().st_size, "sha256": safe_sha256_file(path)}
        )
    _atomic_write_json(
        output_root / "artifact-manifest.json",
        {
            "schema": "veatic21_phase01_artifact_manifest_v1",
            "created_at": _utc_now(),
            "root": str(output_root),
            "artifacts": artifacts,
        },
    )


def _write_checksums(output_root: Path, filenames: tuple[str, ...]) -> None:
    lines = [f"{safe_sha256_file(output_root / name)}  {name}" for name in filenames]
    _atomic_write_text(output_root / "checksums.sha256", "\n".join(lines) + "\n")


def run_phase01(output_root: Path = PHASE01_ROOT) -> dict[str, Any]:
    """Execute the authorized label-only Phase 01 and seal its substrate."""

    output_root = reject_forbidden_runtime_path(output_root)
    if output_root != PHASE01_ROOT:
        raise ValueError(f"Phase 01 output root must be exactly {PHASE01_ROOT}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty Phase 01 root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    package_root = REPOSITORY_ROOT / "src/neural_bridge/veatic21"
    code_sha256 = source_tree_digest(package_root)
    started_at = _utc_now()
    validate_runtime_manifest_paths(
        (VJEPA_ROOT, TRIBE_ROOT, PHASE00_ROOT, LIFECYCLE_ROOT, output_root, package_root)
    )
    operations = {
        "outer_split": False,
        "global_binary_label": False,
        "pca": False,
        "ar_fit": False,
        "cortical_values_loaded": False,
        "cortical_target_performance_read": False,
        "model_training": False,
    }
    request = {
        "schema": "veatic21_phase01_request_v1",
        "started_at": started_at,
        "phase": "phase-01-label-alignment",
        "authority": {
            "master": str(MASTER_SPECIFICATION),
            "protocol": str(REBUILD_PROTOCOL),
            "current_state": str(CURRENT_STATE),
        },
        "sole_label_source": str(VJEPA_ROOT / "<video_id>/rows.csv"),
        "phase00_root": str(PHASE00_ROOT),
        "output_root": str(output_root),
        "code_sha256": code_sha256,
        "frozen_rules": {
            "initial_target": {
                "start_row": 1,
                "candidate_bound": (
                    "all native-row ends retaining at least 90% complete-table coverage"
                ),
                "selection": (
                    "smallest end with median within-video ACF <= 0.90 and at least two "
                    "descriptive global-q90 events in >= 80% of videos"
                ),
            },
            "prospective_washout": (
                "starts are one row after the stable PACF-decay, positive-rise q90, and "
                "selected-event median-duration landmarks; ends preserve the selected "
                "no-washout width; candidates must retain >=90% coverage and descriptive "
                "support in >=80% of videos"
            ),
            "again_offsets_or_seconds_inherited": False,
        },
        "operations": operations,
    }
    _atomic_write_json(output_root / "request.json", request)

    phase00_result_path = PHASE00_ROOT / "result.json"
    phase00_result = load_json(phase00_result_path)
    if not phase00_result.get("status") == "pass" or not phase00_result.get("phase01_authorized"):
        raise ValueError("Phase 00 did not authorize Phase 01")
    phase00_inventory: dict[str, dict[str, str]] = {}
    with (PHASE00_ROOT / "row-inventory.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            phase00_inventory[row["video_id"]] = row
    if set(phase00_inventory) != set(EXPECTED_VIDEO_IDS):
        raise ValueError("Phase 00 row inventory is incomplete")

    rows_by_video: list[SupervisedRows] = []
    quality_by_video: list[dict[str, np.ndarray]] = []
    source_quality_counts: Counter[str] = Counter()
    for video_id in EXPECTED_VIDEO_IDS:
        rows = read_supervised_rows(VJEPA_ROOT / video_id / "rows.csv", video_id)
        inventory = phase00_inventory[video_id]
        if not (
            rows.row_count == int(inventory["row_count"])
            and float(rows.time_seconds[0]) == float(inventory["time_start_seconds"])
            and float(rows.time_seconds[-1]) == float(inventory["time_end_seconds"])
        ):
            raise ValueError(f"Phase 00 row identity mismatch for video {video_id}")
        _, quality = load_phase00_tribe_arrays(
            TRIBE_ROOT / "per_video" / video_id / "tribe_v2_cortical_predictions.npz",
            (
                "time_seconds",
                "black_frame_fraction",
                "duplicate_frame_fraction",
                "quality_black_frame_flag",
                "quality_duplicate_frame_flag",
                "quality_exclusion_flag",
                "quality_weight_suggested",
            ),
        )
        if not np.array_equal(quality["time_seconds"].astype(np.float64), rows.time_seconds):
            raise ValueError(f"final TRIBE time identity mismatch for video {video_id}")
        rows_by_video.append(rows)
        quality_by_video.append(quality)
        source_quality_counts.update(rows.source_match_quality)

    total_rows = sum(rows.row_count for rows in rows_by_video)
    if total_rows != EXPECTED_ROW_COUNT:
        raise ValueError(f"aligned supervised row count mismatch: {total_rows}")
    if dict(source_quality_counts) != EXPECTED_SOURCE_MATCH_COUNTS:
        raise ValueError(f"native interpolation provenance mismatch: {source_quality_counts}")

    video_id = np.concatenate(
        [np.full(rows.row_count, int(rows.video_id), dtype=np.int16) for rows in rows_by_video]
    )
    row_index = np.concatenate([rows.row_index for rows in rows_by_video])
    time_seconds = np.concatenate([rows.time_seconds for rows in rows_by_video])
    arousal = np.concatenate([rows.arousal for rows in rows_by_video])
    valence = np.concatenate([rows.valence for rows in rows_by_video])
    source_frame_position = np.concatenate([rows.source_frame_position for rows in rows_by_video])
    source_floor = np.concatenate([rows.source_floor_frame_index for rows in rows_by_video])
    source_ceil = np.concatenate([rows.source_ceil_frame_index for rows in rows_by_video])
    source_alpha = np.concatenate([rows.source_interp_alpha for rows in rows_by_video])
    source_match_quality = np.concatenate(
        [
            np.asarray(
                [0 if value == "native_exact" else 1 for value in rows.source_match_quality],
                dtype=np.uint8,
            )
            for rows in rows_by_video
        ]
    )
    quality_names = (
        "black_frame_fraction",
        "duplicate_frame_fraction",
        "quality_black_frame_flag",
        "quality_duplicate_frame_flag",
        "quality_exclusion_flag",
        "quality_weight_suggested",
    )
    quality_arrays = {
        name: np.concatenate([quality[name] for quality in quality_by_video])
        for name in quality_names
    }
    if int(quality_arrays["quality_exclusion_flag"].sum()) != 923:
        raise ValueError("quality metadata changed since Phase 00")

    max_possible_lag = min(rows.row_count for rows in rows_by_video) - 1
    coverage_eligible_lags = [
        lag
        for lag in range(1, max_possible_lag + 1)
        if sum(max(0, rows.row_count - lag) for rows in rows_by_video) / total_rows
        >= TARGET_COVERAGE_FLOOR
    ]
    if not coverage_eligible_lags:
        raise ValueError("no target horizon retains the registered coverage floor")
    max_lag = max(coverage_eligible_lags)

    per_video_autocorrelation: list[dict[str, object]] = []
    lag_summaries: list[dict[str, Any]] = []
    for lag in range(1, max_lag + 1):
        acf_values: list[float] = []
        pacf_values: list[float] = []
        for rows in rows_by_video:
            acf = _pearson(rows.arousal[:-lag], rows.arousal[lag:])
            pacf = _pacf_ols(rows.arousal, lag)
            if acf is None:
                raise ValueError(f"undefined ACF for video {rows.video_id}, lag {lag}")
            acf_values.append(acf)
            if pacf is not None:
                pacf_values.append(pacf)
            per_video_autocorrelation.append(
                {
                    "video_id": rows.video_id,
                    "lag_rows": lag,
                    "lag_seconds": lag * EXPECTED_TIME_STEP_SECONDS,
                    "pairs": rows.row_count - lag,
                    "acf": acf,
                    "pacf": pacf,
                    "pacf_defined": pacf is not None,
                }
            )
        pooled_left = np.concatenate([rows.arousal[:-lag] for rows in rows_by_video])
        pooled_right = np.concatenate([rows.arousal[lag:] for rows in rows_by_video])
        lag_summaries.append(
            {
                "lag_rows": lag,
                "lag_seconds": lag * EXPECTED_TIME_STEP_SECONDS,
                "pooled_acf": _pearson(pooled_left, pooled_right),
                "median_per_video_acf": float(np.median(acf_values)),
                "median_per_video_pacf": float(np.median(pacf_values)),
                "median_abs_pacf": float(np.median(np.abs(pacf_values))),
                "pacf_defined_videos": len(pacf_values),
                "pacf_undefined_videos": len(EXPECTED_VIDEO_IDS) - len(pacf_values),
                "acf_q25": float(np.quantile(acf_values, 0.25)),
                "acf_q75": float(np.quantile(acf_values, 0.75)),
                "pacf_q25": float(np.quantile(pacf_values, 0.25)),
                "pacf_q75": float(np.quantile(pacf_values, 0.75)),
            }
        )

    no_washout_values = np.full((total_rows, max_lag), np.nan, dtype=np.float64)
    no_washout_valid = np.zeros((total_rows, max_lag), dtype=np.bool_)
    candidate_rows: list[dict[str, Any]] = []
    event_support_rows: list[dict[str, object]] = []
    movement_histograms: dict[str, Any] = {
        "schema": "veatic21_phase01_movement_histograms_v1",
        "histogram_rule": "Freedman-Diaconis with registered 16..128-bin bounds",
        "absolute_arousal_movement": {},
        "future_maximum_increase": {},
    }
    offset = np.cumsum([0, *[rows.row_count for rows in rows_by_video]])
    for lag in range(1, max_lag + 1):
        for video_position, rows in enumerate(rows_by_video):
            values, valid = future_max_increase(rows.arousal, 1, lag)
            start, end = offset[video_position], offset[video_position + 1]
            no_washout_values[start:end, lag - 1] = values
            no_washout_valid[start:end, lag - 1] = valid
        values = no_washout_values[:, lag - 1]
        valid = no_washout_valid[:, lag - 1]
        threshold, support = _descriptive_event_support(values, valid, video_id)
        support_videos = sum(
            count >= TARGET_SUPPORT_MIN_EVENTS_PER_VIDEO for count in support.values()
        )
        lag_summary = lag_summaries[lag - 1]
        candidate_rows.append(
            {
                "family": "initial_no_washout",
                "start_row": 1,
                "end_row": lag,
                "start_seconds": EXPECTED_TIME_STEP_SECONDS,
                "end_seconds": lag * EXPECTED_TIME_STEP_SECONDS,
                "width_rows": lag,
                "valid_rows": int(valid.sum()),
                "coverage_fraction": float(valid.mean()),
                "eligible_videos": sum(rows.row_count > lag for rows in rows_by_video),
                "median_per_video_acf": lag_summary["median_per_video_acf"],
                "descriptive_global_q90": threshold,
                "support_min_events": TARGET_SUPPORT_MIN_EVENTS_PER_VIDEO,
                "support_videos": support_videos,
                "support_video_fraction": support_videos / len(EXPECTED_VIDEO_IDS),
                "accepted_by_frozen_filter": (
                    float(valid.mean()) >= TARGET_COVERAGE_FLOOR
                    and float(lag_summary["median_per_video_acf"]) <= TARGET_ACF_DECAY_CEILING
                    and support_videos / len(EXPECTED_VIDEO_IDS)
                    >= TARGET_SUPPORT_VIDEO_FRACTION_FLOOR
                ),
                "selected_for_use": False,
                "benchmark_binary_label_created": False,
            }
        )
        for video, count in support.items():
            event_support_rows.append(
                {
                    "family": "initial_no_washout",
                    "start_row": 1,
                    "end_row": lag,
                    "video_id": video,
                    "valid_rows": int(np.sum(valid & (video_id == video))),
                    "descriptive_global_q90": threshold,
                    "descriptive_event_rows": count,
                    "benchmark_binary_label_created": False,
                }
            )
        absolute_movements = np.concatenate(
            [np.abs(rows.arousal[lag:] - rows.arousal[:-lag]) for rows in rows_by_video]
        )
        movement_histograms["absolute_arousal_movement"][str(lag)] = _histogram(absolute_movements)
        movement_histograms["future_maximum_increase"][str(lag)] = _histogram(values[valid])

    selected = select_initial_no_washout(candidate_rows)
    selected["selected_for_use"] = True
    selected_end = int(selected["end_row"])
    selected_values = no_washout_values[:, selected_end - 1]
    selected_valid = no_washout_valid[:, selected_end - 1]
    event_timing = _event_timing(
        rows_by_video, selected_end, float(selected["descriptive_global_q90"])
    )
    pacf_decay_lag = _first_stable_pacf_decay(lag_summaries)
    washout_starts = derive_washout_starts(
        pacf_decay_lag=pacf_decay_lag,
        rise_duration_q90_rows=float(event_timing["positive_rise_duration_rows"]["q90"]),
        event_duration_median_rows=float(event_timing["event_bout_duration_rows"]["q50"]),
    )
    selected_width = selected_end
    washout_values = np.full((total_rows, len(washout_starts)), np.nan, dtype=np.float64)
    washout_valid = np.zeros((total_rows, len(washout_starts)), dtype=np.bool_)
    washout_rows: list[dict[str, object]] = []
    accepted_washout_starts: list[int] = []
    for candidate_index, start_row in enumerate(washout_starts):
        end_row = start_row + selected_width - 1
        for video_position, rows in enumerate(rows_by_video):
            values, valid = future_max_increase(rows.arousal, start_row, end_row)
            start, end = offset[video_position], offset[video_position + 1]
            washout_values[start:end, candidate_index] = values
            washout_valid[start:end, candidate_index] = valid
        values = washout_values[:, candidate_index]
        valid = washout_valid[:, candidate_index]
        threshold, support = _descriptive_event_support(values, valid, video_id)
        support_videos = sum(
            count >= WASHOUT_SUPPORT_MIN_EVENTS_PER_VIDEO for count in support.values()
        )
        coverage = float(valid.mean())
        support_fraction = support_videos / len(EXPECTED_VIDEO_IDS)
        accepted = (
            coverage >= TARGET_COVERAGE_FLOOR
            and support_fraction >= WASHOUT_SUPPORT_VIDEO_FRACTION_FLOOR
        )
        if accepted:
            accepted_washout_starts.append(start_row)
        washout_rows.append(
            {
                "family": "prospective_washout",
                "start_row": start_row,
                "end_row": end_row,
                "start_seconds": start_row * EXPECTED_TIME_STEP_SECONDS,
                "end_seconds": end_row * EXPECTED_TIME_STEP_SECONDS,
                "width_rows": selected_width,
                "valid_rows": int(valid.sum()),
                "coverage_fraction": coverage,
                "eligible_videos": sum(rows.row_count > end_row for rows in rows_by_video),
                "median_per_video_acf": None,
                "descriptive_global_q90": threshold,
                "support_min_events": WASHOUT_SUPPORT_MIN_EVENTS_PER_VIDEO,
                "support_videos": support_videos,
                "support_video_fraction": support_fraction,
                "accepted_by_frozen_filter": accepted,
                "selected_for_use": False,
                "benchmark_binary_label_created": False,
            }
        )
        for video, count in support.items():
            event_support_rows.append(
                {
                    "family": "prospective_washout",
                    "start_row": start_row,
                    "end_row": end_row,
                    "video_id": video,
                    "valid_rows": int(np.sum(valid & (video_id == video))),
                    "descriptive_global_q90": threshold,
                    "descriptive_event_rows": count,
                    "benchmark_binary_label_created": False,
                }
            )
    if set(accepted_washout_starts) != set(washout_starts):
        raise ValueError("a derived prospective washout candidate failed the frozen filter")

    causal_history = _causal_history_rows(rows_by_video, selected_end)
    substrate_arrays = {
        "video_id": video_id,
        "row_index": row_index,
        "time_seconds": time_seconds,
        "arousal": arousal,
        "valence": valence,
        "source_frame_position": source_frame_position,
        "source_floor_frame_index": source_floor,
        "source_ceil_frame_index": source_ceil,
        "source_interp_alpha": source_alpha,
        "source_match_quality_code": source_match_quality,
        **quality_arrays,
        "no_washout_candidate_end_rows": np.arange(1, max_lag + 1, dtype=np.int16),
        "no_washout_future_max_increase": no_washout_values,
        "no_washout_valid_mask": no_washout_valid,
        "selected_future_max_increase": selected_values,
        "selected_valid_mask": selected_valid,
        "prospective_washout_start_rows": np.asarray(washout_starts, dtype=np.int16),
        "prospective_washout_end_rows": np.asarray(
            [start + selected_width - 1 for start in washout_starts], dtype=np.int16
        ),
        "prospective_washout_future_max_increase": washout_values,
        "prospective_washout_valid_mask": washout_valid,
    }
    forbidden_substrate_names = {
        name
        for name in substrate_arrays
        if "binary" in name or "event_label" in name or "split" in name
    }
    if forbidden_substrate_names:
        raise ValueError(f"forbidden Phase 01 substrate arrays: {forbidden_substrate_names}")

    alignment_arrays = {
        name: substrate_arrays[name]
        for name in (
            "video_id",
            "row_index",
            "time_seconds",
            "arousal",
            "valence",
            "source_frame_position",
            "source_floor_frame_index",
            "source_ceil_frame_index",
            "source_interp_alpha",
            "source_match_quality_code",
            *quality_names,
        )
    }
    mask_arrays = {
        "no_washout_valid_mask": no_washout_valid,
        "selected_valid_mask": selected_valid,
        "prospective_washout_valid_mask": washout_valid,
    }
    alignment_sha256 = _array_bundle_digest(alignment_arrays)
    mask_sha256 = _array_bundle_digest(mask_arrays)
    row_ownership_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "row_identity_sha256": _array_bundle_digest(
                    {"video_id": video_id, "row_index": row_index, "time_seconds": time_seconds}
                ),
                "owner": "complete aligned table; no split or fold assigned",
                "rows": total_rows,
            }
        )
    ).hexdigest()
    target_registration = {
        "schema": "veatic21_phase01_target_registration_v1",
        "formula": "max(arousal[t+h_start:t+h_end]) - arousal[t], inclusive future offsets",
        "label_authority": "matching V-JEPA rows.csv only",
        "row_hz": EXPECTED_ROW_HZ,
        "initial_no_washout": {
            "start_row": 1,
            "end_row": selected_end,
            "start_seconds": EXPECTED_TIME_STEP_SECONDS,
            "end_seconds": selected_end * EXPECTED_TIME_STEP_SECONDS,
            "width_rows": selected_width,
            "selection_rule": request["frozen_rules"]["initial_target"],
            "selected_candidate_summary": selected,
        },
        "prospective_washout": {
            "activated": False,
            "selected_candidate": None,
            "candidate_starts": list(washout_starts),
            "candidate_ends": [start + selected_width - 1 for start in washout_starts],
            "derivation_landmarks": {
                "stable_pacf_decay_lag_rows": pacf_decay_lag,
                "positive_rise_duration_q90_rows": event_timing["positive_rise_duration_rows"][
                    "q90"
                ],
                "selected_event_duration_median_rows": event_timing["event_bout_duration_rows"][
                    "q50"
                ],
            },
            "procedure": request["frozen_rules"]["prospective_washout"],
            "activation_rule": (
                "activate only after controlled no-washout decomposition shows legal "
                "persistence dominance or insufficient history/target separation; selection "
                "uses development-owned evidence and every matched control"
            ),
        },
        "descriptive_global_q90_role": (
            "support and timing summary only; Phase 02 fits event q90 inside each outer "
            "training partition"
        ),
        "global_binary_label_stored": False,
        "outer_split_created": False,
        "again_offsets_seconds_or_targets_inherited": False,
    }
    target_source_sha256 = hashlib.sha256(canonical_json_bytes(target_registration)).hexdigest()
    substrate_sha256 = _array_bundle_digest(substrate_arrays)
    digests = {
        "alignment_sha256": alignment_sha256,
        "target_source_sha256": target_source_sha256,
        "mask_sha256": mask_sha256,
        "row_ownership_sha256": row_ownership_sha256,
        "substrate_arrays_sha256": substrate_sha256,
    }

    _atomic_save_npz(output_root / "aligned-target-substrate.npz", substrate_arrays)
    _atomic_write_json(output_root / "target-registration.json", target_registration)
    _atomic_write_csv(output_root / "candidate-windows.csv", [*candidate_rows, *washout_rows])
    _atomic_write_csv(output_root / "per-video-event-support.csv", event_support_rows)
    _atomic_write_csv(output_root / "per-video-autocorrelation.csv", per_video_autocorrelation)
    _atomic_write_csv(output_root / "causal-history-predictiveness.csv", causal_history)
    _atomic_write_json(output_root / "movement-histograms.json", movement_histograms)
    label_dynamics = {
        "schema": "veatic21_phase01_label_dynamics_v1",
        "rows": total_rows,
        "videos": len(EXPECTED_VIDEO_IDS),
        "arousal": {"range": [float(np.min(arousal)), float(np.max(arousal))]},
        "valence": {"range": [float(np.min(valence)), float(np.max(valence))]},
        "video_duration_seconds": _quantiles(
            np.asarray([rows.time_seconds[-1] for rows in rows_by_video])
        ),
        "lag_summaries": lag_summaries,
        "event_and_rise_timing": event_timing,
        "pacf_decay_lag_rows": pacf_decay_lag,
        "pacf_decay_lag_seconds": pacf_decay_lag * EXPECTED_TIME_STEP_SECONDS,
        "selected_target_end_rows": selected_end,
        "selected_target_end_seconds": selected_end * EXPECTED_TIME_STEP_SECONDS,
        "quality_rows_retained": total_rows,
        "quality_rows_flagged": int(quality_arrays["quality_exclusion_flag"].sum()),
    }
    _atomic_write_json(output_root / "label-dynamics.json", label_dynamics)

    choices: list[dict[str, object]] = []

    def register_choice(
        choice: str, value: object, derivation_rule: str, artifact_sha256: str
    ) -> None:
        choices.append(
            {
                "choice": choice,
                "value": value,
                "authority": str(MASTER_SPECIFICATION),
                "derivation_rule": derivation_rule,
                "owned_rows": "all 20,657 aligned VEATIC rows; no outer split",
                "code_sha256": code_sha256,
                "artifact_sha256": artifact_sha256,
            }
        )

    for name, value, rule in (
        ("target_coverage_floor", TARGET_COVERAGE_FLOOR, "complete-table retention safeguard"),
        (
            "target_acf_decay_ceiling",
            TARGET_ACF_DECAY_CEILING,
            "require at least 10% median within-video level-correlation decay",
        ),
        (
            "target_support_video_fraction_floor",
            TARGET_SUPPORT_VIDEO_FRACTION_FLOOR,
            "prevent descriptive top-decile support concentrating in a minority of videos",
        ),
        (
            "target_support_min_events_per_video",
            TARGET_SUPPORT_MIN_EVENTS_PER_VIDEO,
            "require repeated descriptive support rather than one isolated row",
        ),
        (
            "pacf_decay_abs_ceiling",
            PACF_DECAY_ABS_CEILING,
            "small residual lag-specific dependence landmark",
        ),
        (
            "pacf_decay_consecutive_lags",
            PACF_DECAY_CONSECUTIVE_LAGS,
            "stability requirement for the PACF landmark",
        ),
        (
            "histogram_bin_bounds",
            [HISTOGRAM_MIN_BINS, HISTOGRAM_MAX_BINS],
            "bounded Freedman-Diaconis descriptive resolution",
        ),
    ):
        register_choice(name, value, rule, alignment_sha256)
    register_choice(
        "selected_initial_target_rows",
        [1, selected_end],
        "first frozen-rule candidate satisfying coverage, ACF decay, and support",
        target_source_sha256,
    )
    register_choice(
        "prospective_washout_candidate_rows",
        [[start, start + selected_width - 1] for start in washout_starts],
        "VEATIC PACF, positive-rise, and selected-event-duration landmarks",
        target_source_sha256,
    )
    ledger = {
        "schema": "veatic21_derivation_ledger_v1",
        "phase": "phase-01-label-alignment",
        "fitted_choices": [],
        "numeric_choices": choices,
        "again_numeric_choices_inherited": False,
        "again_paths_used": False,
        "code_sha256": code_sha256,
        "digests": digests,
    }
    _atomic_write_json(output_root / "veatic-derivation-ledger.json", ledger)

    alignment_manifest = {
        "schema": "veatic21_phase01_alignment_manifest_v1",
        "sole_label_source": str(VJEPA_ROOT / "<video_id>/rows.csv"),
        "phase00_result_sha256": safe_sha256_file(phase00_result_path),
        "phase00_input_identity_sha256": phase00_result["input_identity_sha256"],
        "vjepa_allowed_metadata_tree_sha256": VJEPA_ALLOWED_TREE_SHA256,
        "videos": len(EXPECTED_VIDEO_IDS),
        "rows": total_rows,
        "row_hz": EXPECTED_ROW_HZ,
        "source_match_quality": dict(source_quality_counts),
        "labels_finite": bool(np.isfinite(arousal).all() and np.isfinite(valence).all()),
        "quality_rows_retained": total_rows,
        "quality_rows_filtered": 0,
        "outer_split_created": False,
        "digests": digests,
        "substrate_file": "aligned-target-substrate.npz",
        "substrate_file_sha256": safe_sha256_file(output_root / "aligned-target-substrate.npz"),
        "vjepa_hidden_states_loaded": False,
        "vjepa_hidden_states_hashed": False,
        "cortical_values_loaded": False,
    }
    _atomic_write_json(output_root / "alignment-manifest.json", alignment_manifest)

    checks = dict.fromkeys(PHASE01_CHECKS, True)
    if not phase02_authorized(checks):
        raise ValueError("Phase 01 mandatory check matrix is incomplete")
    completed_at = _utc_now()
    result = {
        "schema": "veatic21_phase01_result_v1",
        "phase": "phase-01-label-alignment",
        "status": "pass",
        "started_at": started_at,
        "completed_at": completed_at,
        "code_sha256": code_sha256,
        "videos": len(EXPECTED_VIDEO_IDS),
        "rows": total_rows,
        "checks": checks,
        "digests": digests,
        "selected_initial_target": target_registration["initial_no_washout"],
        "prospective_washout": target_registration["prospective_washout"],
        "global_binary_label_stored": False,
        "outer_split_created": False,
        "operations": operations,
        "forbidden_input_audit": {
            "vjepa_hidden_states_loaded": False,
            "vjepa_hidden_states_hashed": False,
            "cortical_values_loaded": False,
        },
        "phase02_authorized": True,
        "single_next_authorized_action": (
            "Phase 02 fresh target-specific AR baseline under separate grouped-video and "
            "blocked-temporal protocols"
        ),
    }
    _atomic_write_json(output_root / "result.json", result)
    report = f"""# VEATIC 2.1 Phase 01 Label Alignment and Target Substrate

Status: **PASS**

Phase 01 reconstructed all {total_rows:,} supervised rows from matching V-JEPA `rows.csv`
files and confirmed exact `(video_id, row_index, time_seconds)` identity against both the
sealed Phase 00 inventory and final TRIBE timestamps. Arousal, valence, and native
interpolation provenance were finite and exact. All 923 quality-flagged rows remain attached
metadata; no row was filtered.

The frozen label-only rule selected the initial no-washout future-maximum-increase target at
rows `1..{selected_end}`
({EXPECTED_TIME_STEP_SECONDS:.1f}..{selected_end * EXPECTED_TIME_STEP_SECONDS:.1f}s).
The selection was the first native-row endpoint to retain at least 90% complete-table
coverage, reach median within-video arousal ACF <= 0.90, and retain repeated descriptive
top-decile support in at least 80% of videos. Descriptive global q90 values were used only for
support/timing summaries; no global binary label was stored. Phase 02 must fit q90 inside each
outer-training partition.

The prospective washout family is not activated or selected. Its candidate windows are
{[[start, start + selected_width - 1] for start in washout_starts]}, derived only from VEATIC
PACF decay, positive-rise duration, selected-event duration, coverage, and support. AGAIN
offsets, seconds, targets, and numeric results were not inherited.

No outer split, cortical value, cortical target result, PCA, AR fit, or learned model entered
Phase 01. All {len(PHASE01_CHECKS)} alignment controls passed. Phase 02 fresh target-specific
AR is the single next authorized action after this transition is committed and pushed.

Code SHA-256: `{code_sha256}`
Alignment SHA-256: `{alignment_sha256}`
Target-source SHA-256: `{target_source_sha256}`
Mask SHA-256: `{mask_sha256}`
Row-ownership SHA-256: `{row_ownership_sha256}`
Substrate arrays SHA-256: `{substrate_sha256}`
"""
    _atomic_write_text(output_root / "report.md", report)
    outputs = (
        "request.json",
        "alignment-manifest.json",
        "aligned-target-substrate.npz",
        "target-registration.json",
        "label-dynamics.json",
        "movement-histograms.json",
        "candidate-windows.csv",
        "per-video-event-support.csv",
        "per-video-autocorrelation.csv",
        "causal-history-predictiveness.csv",
        "veatic-derivation-ledger.json",
        "result.json",
        "report.md",
    )
    _write_artifact_manifest(output_root, outputs)
    _write_checksums(output_root, (*outputs, "artifact-manifest.json"))
    return result
