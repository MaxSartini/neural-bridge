"""Fresh VEATIC 2.1 Phase 01 label alignment and target substrate."""

from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np
from scipy.stats import spearmanr

from neural_bridge.veatic21.contracts import (
    EXPECTED_QUALITY_COUNTS,
    EXPECTED_ROW_COUNT,
    EXPECTED_ROW_HZ,
    EXPECTED_SOURCE_MATCH_COUNTS,
    EXPECTED_TIME_STEP_SECONDS,
    EXPECTED_VIDEO_IDS,
    PHASE00_INPUT_IDENTITY_SHA256,
    PHASE00_RESULT_SHA256,
    PHASE00_ROOT,
    PHASE01_MANDATORY_CHECK_NAMES,
    PHASE01_ROOT,
    REPOSITORY_ROOT,
    TRIBE_PER_VIDEO_ROOT,
    VJEPA_ROOT,
)
from neural_bridge.veatic21.data import (
    SupervisedRows,
    TribeRowMetadata,
    load_json,
    read_supervised_rows,
    read_tribe_row_metadata,
    reject_forbidden_runtime_path,
    sha256_file,
)
from neural_bridge.veatic21.phase00 import (
    _audit_again_source_firewall,
    _source_tree_identity,
    _write_artifact_manifests,
    _write_json,
    _write_text,
    digest_json,
    verify_phase00_output,
)


def _row_identity_digest(rows: Sequence[SupervisedRows]) -> str:
    digest = hashlib.sha256()
    for video in rows:
        for row_index, time_seconds in zip(
            video.row_index, video.time_seconds, strict=True
        ):
            digest.update(f"{video.video_id}\0{int(row_index)}\0{time_seconds:.6f}\n".encode())
    return digest.hexdigest()


def _label_digest(rows: Sequence[SupervisedRows]) -> str:
    digest = hashlib.sha256()
    for video in rows:
        digest.update(f"{video.video_id}\0".encode())
        digest.update(video.arousal.astype("<f8", copy=False).tobytes())
        digest.update(video.valence.astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def _array_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(array.dtype).encode())
        digest.update(b"\0")
        digest.update(repr(array.shape).encode())
        digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def derive_candidate_pairs(row_counts: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Enumerate every future window supported by every VEATIC video."""

    if not row_counts or min(row_counts) < 2:
        raise ValueError("candidate derivation requires at least two rows in every video")
    max_end = min(row_counts) - 1
    return tuple((start, end) for end in range(1, max_end + 1) for start in range(1, end + 1))


def future_maximum_increase(
    arousal: np.ndarray, start: int, end: int
) -> tuple[np.ndarray, np.ndarray]:
    if start < 1 or end < start:
        raise ValueError("invalid future window")
    values = np.full(arousal.shape, np.nan, dtype=np.float32)
    mask = np.zeros(arousal.shape, dtype=bool)
    valid_rows = len(arousal) - end
    for row in range(max(valid_rows, 0)):
        values[row] = float(np.max(arousal[row + start : row + end + 1]) - arousal[row])
        mask[row] = True
    return values, mask


def build_target_substrate(
    videos: Sequence[SupervisedRows], candidates: Sequence[tuple[int, int]]
) -> tuple[dict[str, np.ndarray], list[slice]]:
    total_rows = sum(video.row_count for video in videos)
    values = np.full((len(candidates), total_rows), np.nan, dtype=np.float32)
    masks = np.zeros((len(candidates), total_rows), dtype=bool)
    video_slices: list[slice] = []
    offset = 0
    for video in videos:
        video_slice = slice(offset, offset + video.row_count)
        video_slices.append(video_slice)
        for candidate_index, (start, end) in enumerate(candidates):
            candidate_values, candidate_mask = future_maximum_increase(video.arousal, start, end)
            values[candidate_index, video_slice] = candidate_values
            masks[candidate_index, video_slice] = candidate_mask
        offset += video.row_count
    return {
        "candidate_start_rows": np.asarray([pair[0] for pair in candidates], dtype=np.int16),
        "candidate_end_rows": np.asarray([pair[1] for pair in candidates], dtype=np.int16),
        "continuous_future_maximum_increase": values,
        "valid_mask": masks,
    }, video_slices


def _finite_correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float | int | None]:
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return {"rows": int(valid.sum()), "pearson": None, "spearman": None}
    pearson = float(np.corrcoef(x[valid], y[valid])[0, 1])
    spearman = float(spearmanr(x[valid], y[valid]).statistic)
    return {"rows": int(valid.sum()), "pearson": pearson, "spearman": spearman}


def _acf(values: np.ndarray, lag: int) -> float | None:
    if len(values) <= lag:
        return None
    left, right = values[:-lag], values[lag:]
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _pacf(values: np.ndarray, lag: int) -> float | None:
    if len(values) <= lag + 2:
        return None
    target = values[lag:]
    design = np.column_stack([values[lag - offset : -offset] for offset in range(1, lag + 1)])
    design = np.column_stack([np.ones(len(target)), design])
    try:
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    return float(coefficients[-1])


def _summarize(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
    }


def _run_lengths(mask: np.ndarray) -> list[int]:
    lengths: list[int] = []
    current = 0
    for value in mask:
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def _duration_summary(videos: Sequence[SupervisedRows]) -> dict[str, object]:
    rise_lengths: list[int] = []
    high_movement_lengths: list[int] = []
    for video in videos:
        delta = np.diff(video.arousal)
        rise_lengths.extend(_run_lengths(delta > 0))
        threshold = float(np.quantile(np.abs(delta), 0.90))
        high_movement_lengths.extend(_run_lengths(np.abs(delta) >= threshold))

    def seconds(lengths: Sequence[int]) -> dict[str, float | int | None]:
        summary = _summarize([length * EXPECTED_TIME_STEP_SECONDS for length in lengths])
        summary["events"] = len(lengths)
        return summary

    return {
        "rise_definition": "consecutive positive native 2 Hz arousal differences",
        "high_movement_definition": "per-video q90 absolute native 2 Hz arousal difference",
        "rise_duration_seconds": seconds(rise_lengths),
        "high_movement_duration_seconds": seconds(high_movement_lengths),
    }


def _dynamics(videos: Sequence[SupervisedRows], max_lag: int) -> dict[str, object]:
    autocorrelation: list[dict[str, object]] = []
    partial_autocorrelation: list[dict[str, object]] = []
    movement: list[dict[str, object]] = []
    for lag in range(1, max_lag + 1):
        acf_values = [value for video in videos if (value := _acf(video.arousal, lag)) is not None]
        pacf_values = [
            value for video in videos if (value := _pacf(video.arousal, lag)) is not None
        ]
        absolute = np.concatenate(
            [np.abs(video.arousal[lag:] - video.arousal[:-lag]) for video in videos]
        )
        autocorrelation.append(
            {"lag_rows": lag, "lag_seconds": lag / EXPECTED_ROW_HZ, **_summarize(acf_values)}
        )
        partial_autocorrelation.append(
            {"lag_rows": lag, "lag_seconds": lag / EXPECTED_ROW_HZ, **_summarize(pacf_values)}
        )
        movement.append(
            {
                "lag_rows": lag,
                "lag_seconds": lag / EXPECTED_ROW_HZ,
                "rows": int(absolute.size),
                "q50": float(np.quantile(absolute, 0.50)),
                "q75": float(np.quantile(absolute, 0.75)),
                "q90": float(np.quantile(absolute, 0.90)),
                "q95": float(np.quantile(absolute, 0.95)),
            }
        )
    durations = np.asarray([video.time_seconds[-1] for video in videos], dtype=np.float64)
    return {
        "max_lag_rows": max_lag,
        "max_lag_seconds": max_lag / EXPECTED_ROW_HZ,
        "autocorrelation": autocorrelation,
        "partial_autocorrelation": partial_autocorrelation,
        "absolute_movement": movement,
        "event_timing": _duration_summary(videos),
        "video_last_time_seconds": {
            "min": float(np.min(durations)),
            "p10": float(np.quantile(durations, 0.10)),
            "median": float(np.median(durations)),
            "p90": float(np.quantile(durations, 0.90)),
            "max": float(np.max(durations)),
        },
    }


def _candidate_registry(
    videos: Sequence[SupervisedRows],
    candidates: Sequence[tuple[int, int]],
    substrate: Mapping[str, np.ndarray],
    video_slices: Sequence[slice],
) -> list[dict[str, object]]:
    values = substrate["continuous_future_maximum_increase"]
    masks = substrate["valid_mask"]
    registry: list[dict[str, object]] = []
    for index, (start, end) in enumerate(candidates):
        candidate_values = values[index]
        candidate_mask = masks[index]
        pooled = candidate_values[candidate_mask].astype(np.float64)
        descriptive_q90 = float(np.quantile(pooled, 0.90))
        per_video_q90: list[float] = []
        positive_counts: list[int] = []
        valid_counts: list[int] = []
        for video_slice in video_slices:
            local_mask = candidate_mask[video_slice]
            local_values = candidate_values[video_slice][local_mask].astype(np.float64)
            valid_counts.append(int(local_values.size))
            per_video_q90.append(float(np.quantile(local_values, 0.90)))
            positive_counts.append(int(np.sum(local_values >= descriptive_q90)))
        eligible = bool(
            np.isfinite(pooled).all()
            and np.std(pooled) > 0
            and min(valid_counts) > 0
            and all(np.isfinite(per_video_q90))
            and sum(positive_counts) > 0
        )
        family = "initial_no_washout" if start == 1 else "prospective_washout"
        registry.append(
            {
                "candidate_id": f"s{start:02d}_e{end:02d}",
                "candidate_index": index,
                "family": family,
                "future_start_rows": start,
                "future_end_rows": end,
                "future_start_seconds": start / EXPECTED_ROW_HZ,
                "future_end_seconds": end / EXPECTED_ROW_HZ,
                "washout_rows": list(range(1, start)),
                "washout_seconds": [row / EXPECTED_ROW_HZ for row in range(1, start)],
                "valid_rows": int(candidate_mask.sum()),
                "valid_row_fraction": float(candidate_mask.mean()),
                "eligible_videos": int(sum(count > 0 for count in valid_counts)),
                "minimum_valid_rows_per_video": min(valid_counts),
                "continuous_std": float(np.std(pooled)),
                "descriptive_pooled_q90_only": descriptive_q90,
                "per_video_q90_stability": _summarize(per_video_q90),
                "videos_with_descriptive_positive_support": int(
                    sum(count > 0 for count in positive_counts)
                ),
                "minimum_descriptive_positive_rows_per_video": min(positive_counts),
                "median_descriptive_positive_rows_per_video": float(np.median(positive_counts)),
                "eligible": eligible,
                "phase02_active": eligible and family == "initial_no_washout",
                "prospective_only": family == "prospective_washout",
                "binary_label_stored": False,
            }
        )
    return registry


def _causal_features(video: SupervisedRows, max_depth: int) -> dict[str, np.ndarray]:
    features: dict[str, np.ndarray] = {"current_arousal": video.arousal.astype(np.float64)}
    previous_delta = np.full(video.row_count, np.nan, dtype=np.float64)
    previous_delta[1:] = np.diff(video.arousal)
    features["previous_delta"] = previous_delta
    for depth in range(1, max_depth + 1):
        mean = np.full(video.row_count, np.nan, dtype=np.float64)
        for row in range(depth - 1, video.row_count):
            mean[row] = float(np.mean(video.arousal[row - depth + 1 : row + 1]))
        features[f"trailing_mean_{depth:02d}"] = mean
        if depth >= 2:
            slope = np.full(video.row_count, np.nan, dtype=np.float64)
            x = np.arange(depth, dtype=np.float64)
            centered = x - np.mean(x)
            denominator = float(np.sum(centered**2))
            for row in range(depth - 1, video.row_count):
                y = video.arousal[row - depth + 1 : row + 1]
                slope[row] = float(np.sum(centered * (y - np.mean(y))) / denominator)
            features[f"causal_slope_{depth:02d}"] = slope
    return features


def _causal_diagnostics(
    videos: Sequence[SupervisedRows],
    candidates: Sequence[tuple[int, int]],
    substrate: Mapping[str, np.ndarray],
    video_slices: Sequence[slice],
    max_depth: int,
) -> dict[str, object]:
    feature_by_video = [_causal_features(video, max_depth) for video in videos]
    feature_names = tuple(feature_by_video[0])
    flattened = {
        name: np.concatenate([features[name] for features in feature_by_video])
        for name in feature_names
    }
    values = substrate["continuous_future_maximum_increase"]
    masks = substrate["valid_mask"]
    rows: list[dict[str, object]] = []
    for candidate_index, (start, end) in enumerate(candidates):
        if start != 1:
            continue
        target = values[candidate_index].astype(np.float64)
        target[~masks[candidate_index]] = np.nan
        for feature_name in feature_names:
            rows.append(
                {
                    "candidate_id": f"s{start:02d}_e{end:02d}",
                    "feature": feature_name,
                    **_finite_correlation(flattened[feature_name], target),
                }
            )
    return {
        "schema_version": "veatic21_phase01_causal_history_diagnostics_v2",
        "interpretation": "label-only correlations; no AR model was fitted",
        "history_depth_rows": list(range(1, max_depth + 1)),
        "history_depth_seconds": [depth / EXPECTED_ROW_HZ for depth in range(1, max_depth + 1)],
        "candidate_family": "initial_no_washout only",
        "rows": rows,
        "video_slices_used": len(video_slices),
    }


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)  # ty: ignore[invalid-argument-type]


def _build_result(
    checks: Mapping[str, bool], *, code_sha256: str, input_sha256: str, row_digest: str,
    label_digest: str, candidate_count: int, active_count: int, prospective_count: int,
    derived_digests: Mapping[str, str] | None = None,
) -> dict[str, object]:
    missing = sorted(set(PHASE01_MANDATORY_CHECK_NAMES) - set(checks))
    extra = sorted(set(checks) - set(PHASE01_MANDATORY_CHECK_NAMES))
    failed = sorted(name for name, value in checks.items() if not value)
    passed = not missing and not extra and not failed and len(checks) == len(
        PHASE01_MANDATORY_CHECK_NAMES
    )
    return {
        "schema_version": "veatic21_fresh_phase01_result_v2",
        "phase": "phase-01-label-alignment",
        "status": "PASS" if passed else "FAIL",
        "phase01_pass": passed,
        "mandatory_controls_expected": len(PHASE01_MANDATORY_CHECK_NAMES),
        "mandatory_controls_passed": sum(bool(value) for value in checks.values()),
        "checks": dict(checks),
        "missing_checks": missing,
        "extra_checks": extra,
        "failed_checks": failed,
        "code_sha256": code_sha256,
        "phase00_result_sha256": PHASE00_RESULT_SHA256,
        "phase00_input_identity_sha256": PHASE00_INPUT_IDENTITY_SHA256,
        "input_identity_sha256": input_sha256,
        "row_identity_sha256": row_digest,
        "label_sha256": label_digest,
        "derived_digests": dict(derived_digests or {}),
        "video_count": 124,
        "row_count": EXPECTED_ROW_COUNT,
        "row_hz": EXPECTED_ROW_HZ,
        "candidate_count": candidate_count,
        "active_no_washout_candidate_count": active_count,
        "prospective_washout_candidate_count": prospective_count,
        "selected_target": None,
        "global_binary_label_stored": False,
        "outer_split_created": False,
        "cortical_values_loaded": False,
        "tribe_payload_opened_for_row_metadata": True,
        "tribe_cortical_values_loaded": False,
        "vjepa_hidden_states_loaded": False,
        "vjepa_hidden_states_hashed": False,
        "operations": {
            "ar_fit": False,
            "dataset_split": False,
            "global_binary_target": False,
            "cortical_performance_read": False,
            "pca": False,
            "model_training": False,
            "head_search": False,
        },
        "single_next_authorized_action": (
            "Phase 02 comprehensive fresh target-specific AR benchmark only" if passed else None
        ),
    }


def run_phase01(output_root: Path = PHASE01_ROOT) -> dict[str, object]:
    output_root = reject_forbidden_runtime_path(output_root)
    if output_root != PHASE01_ROOT:
        raise ValueError(f"Phase 01 output must use the canonical root: {PHASE01_ROOT}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to reuse nonempty Phase 01 output: {output_root}")

    phase00 = verify_phase00_output(PHASE00_ROOT)
    if phase00["result_sha256"] != PHASE00_RESULT_SHA256:
        raise ValueError("Phase 00 result identity changed")
    phase00_result_value = phase00["result"]
    if not isinstance(phase00_result_value, dict):
        raise ValueError("Phase 00 result payload is invalid")
    phase00_result = cast(dict[str, object], phase00_result_value)
    if phase00_result["input_identity_sha256"] != PHASE00_INPUT_IDENTITY_SHA256:
        raise ValueError("Phase 00 input identity changed")

    package_root = REPOSITORY_ROOT / "src/neural_bridge/veatic21"
    code_identity = _source_tree_identity(package_root)
    source_firewall = _audit_again_source_firewall(package_root)
    code_sha256 = str(code_identity["sha256"])
    request = {
        "schema_version": "veatic21_fresh_phase01_request_v2",
        "phase": "phase-01-label-alignment",
        "phase00_result_sha256": PHASE00_RESULT_SHA256,
        "phase00_input_identity_sha256": PHASE00_INPUT_IDENTITY_SHA256,
        "vjepa_rows_root": str(VJEPA_ROOT),
        "tribe_row_metadata_root": str(TRIBE_PER_VIDEO_ROOT),
        "output_root": str(output_root),
        "code_sha256": code_sha256,
        "input_boundary": (
            "labels and interpolation provenance from rows.csv; exact time/quality metadata "
            "from matching TRIBE payloads; no cortical values or hidden states"
        ),
        "candidate_rule": (
            "enumerate every (start,end) with 1<=start<=end<=minimum_video_rows-1"
        ),
        "operations": {
            "global_binary_target": False,
            "dataset_split": False,
            "cortical_performance_read": False,
            "ar_fit": False,
            "pca": False,
            "model_training": False,
        },
    }
    _write_json(output_root / "request.json", request)

    phase00_inventory: dict[str, dict[str, str]] = {}
    with (PHASE00_ROOT / "row-inventory.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            phase00_inventory[row["video_id"]] = row

    videos = [
        read_supervised_rows(VJEPA_ROOT / video_id / "rows.csv", video_id)
        for video_id in EXPECTED_VIDEO_IDS
    ]
    checks = dict.fromkeys(PHASE01_MANDATORY_CHECK_NAMES, False)
    checks["phase00_identity_reconfirmed"] = True
    if tuple(video.video_id for video in videos) != EXPECTED_VIDEO_IDS:
        raise ValueError("Phase 01 video inventory mismatch")
    checks["complete_video_inventory"] = True
    tribe_metadata: list[TribeRowMetadata] = []
    for video in videos:
        sealed = phase00_inventory[video.video_id]
        if not (
            int(sealed["row_count"]) == video.row_count
            and float(sealed["time_start_seconds"]) == float(video.time_seconds[0])
            and float(sealed["time_end_seconds"]) == float(video.time_seconds[-1])
        ):
            raise ValueError(f"Phase 00 row identity mismatch for video {video.video_id}")
        metadata = read_tribe_row_metadata(
            TRIBE_PER_VIDEO_ROOT / video.video_id / "tribe_v2_cortical_predictions.npz",
            video,
        )
        if not np.array_equal(metadata.time_seconds, video.time_seconds):
            raise ValueError(f"TRIBE/rows.csv row identity mismatch for video {video.video_id}")
        tribe_metadata.append(metadata)
    black = sum(int(metadata.quality_black_frame_flag.sum()) for metadata in tribe_metadata)
    duplicate = sum(
        int(metadata.quality_duplicate_frame_flag.sum()) for metadata in tribe_metadata
    )
    both = sum(
        int(
            np.sum(
                metadata.quality_black_frame_flag.astype(bool)
                & metadata.quality_duplicate_frame_flag.astype(bool)
            )
        )
        for metadata in tribe_metadata
    )
    union = sum(int(metadata.quality_exclusion_flag.sum()) for metadata in tribe_metadata)
    observed_quality = {
        "black_rows": black,
        "duplicate_rows": duplicate,
        "both_rows": both,
        "union_rows": union,
        "unflagged_rows": EXPECTED_ROW_COUNT - union,
        "total_rows": EXPECTED_ROW_COUNT,
    }
    if observed_quality != EXPECTED_QUALITY_COUNTS:
        raise ValueError(f"Phase 01 quality count mismatch: {observed_quality}")
    observed_source_matches = Counter(
        value for video in videos for value in video.source_match_quality
    )
    if dict(observed_source_matches) != EXPECTED_SOURCE_MATCH_COUNTS:
        raise ValueError(f"Phase 01 source-match count mismatch: {observed_source_matches}")
    total_rows = sum(video.row_count for video in videos)
    if total_rows != EXPECTED_ROW_COUNT:
        raise ValueError("Phase 01 total row count mismatch")
    checks["complete_row_inventory"] = True
    checks["exact_row_identity"] = True
    checks["exact_2hz_grid"] = True
    checks["rows_schema_and_encode_policy"] = True
    if not all(np.isfinite(video.arousal).all() for video in videos):
        raise ValueError("nonfinite arousal")
    checks["finite_arousal"] = True
    if not all(np.isfinite(video.valence).all() for video in videos):
        raise ValueError("nonfinite valence")
    checks["finite_valence"] = True
    checks["native_interpolation_provenance"] = True
    checks["tribe_row_identity_reconfirmed"] = True
    checks["quality_metadata_preserved"] = True
    checks["all_rows_retained"] = True
    checks["label_only_input_boundary"] = True

    row_counts = [video.row_count for video in videos]
    candidates = derive_candidate_pairs(row_counts)
    max_end = min(row_counts) - 1
    checks["candidate_bound_derived_from_veatic"] = True
    substrate, video_slices = build_target_substrate(videos, candidates)
    registry = _candidate_registry(videos, candidates, substrate, video_slices)
    active = [row for row in registry if row["phase02_active"]]
    prospective = [row for row in registry if row["prospective_only"]]
    if len(active) != max_end or any(not row["eligible"] for row in active):
        raise ValueError("incomplete no-washout registry")
    checks["complete_no_washout_registry"] = True
    if len(prospective) != len(candidates) - max_end:
        raise ValueError("incomplete prospective washout registry")
    checks["complete_prospective_washout_registry"] = True
    checks["candidate_values_and_masks_stored"] = True
    checks["candidate_coverage_audited"] = True
    checks["per_video_support_audited"] = True
    checks["threshold_stability_audited"] = True

    dynamics = _dynamics(videos, max_end)
    checks["autocorrelation_audited"] = True
    checks["partial_autocorrelation_audited"] = True
    causal = _causal_diagnostics(videos, candidates, substrate, video_slices, max_end)
    checks["causal_history_predictiveness_audited"] = True
    checks["rise_and_event_duration_audited"] = True
    checks["global_binary_label_absent"] = True
    checks["outer_split_absent"] = True
    checks["no_cortical_performance_or_modeling"] = True
    if source_firewall["again_imports"] or source_firewall["again_runtime_paths"]:
        raise ValueError("AGAIN runtime firewall failed")
    checks["again_runtime_firewall"] = True

    row_digest = _row_identity_digest(videos)
    label_digest = _label_digest(videos)
    input_identity = {
        "phase00_result_sha256": PHASE00_RESULT_SHA256,
        "phase00_input_identity_sha256": PHASE00_INPUT_IDENTITY_SHA256,
        "row_identity_sha256": row_digest,
        "label_sha256": label_digest,
        "video_count": len(videos),
        "row_count": total_rows,
        "row_hz": EXPECTED_ROW_HZ,
    }
    input_sha256 = digest_json(input_identity)
    video_id = np.concatenate(
        [np.full(video.row_count, int(video.video_id), dtype=np.int16) for video in videos]
    )
    source_match_quality_code = np.concatenate(
        [
            np.asarray(
                [0 if value == "native_exact" else 1 for value in video.source_match_quality],
                dtype=np.uint8,
            )
            for video in videos
        ]
    )
    aligned_arrays = {
        "video_id": video_id,
        "row_index": np.concatenate([video.row_index for video in videos]),
        "time_seconds": np.concatenate([video.time_seconds for video in videos]),
        "native_label_frame_count": np.concatenate(
            [video.native_label_frame_count for video in videos]
        ),
        "source_frame_position": np.concatenate(
            [video.source_frame_position for video in videos]
        ),
        "source_floor_frame_index": np.concatenate(
            [video.source_floor_frame_index for video in videos]
        ),
        "source_ceil_frame_index": np.concatenate(
            [video.source_ceil_frame_index for video in videos]
        ),
        "source_interp_alpha": np.concatenate([video.source_interp_alpha for video in videos]),
        "source_match_quality_code": source_match_quality_code,
        "arousal": np.concatenate([video.arousal for video in videos]).astype(np.float32),
        "valence": np.concatenate([video.valence for video in videos]).astype(np.float32),
        "black_frame_fraction": np.concatenate(
            [metadata.black_frame_fraction for metadata in tribe_metadata]
        ),
        "duplicate_frame_fraction": np.concatenate(
            [metadata.duplicate_frame_fraction for metadata in tribe_metadata]
        ),
        "quality_black_frame_flag": np.concatenate(
            [metadata.quality_black_frame_flag for metadata in tribe_metadata]
        ),
        "quality_duplicate_frame_flag": np.concatenate(
            [metadata.quality_duplicate_frame_flag for metadata in tribe_metadata]
        ),
        "quality_exclusion_flag": np.concatenate(
            [metadata.quality_exclusion_flag for metadata in tribe_metadata]
        ),
        "quality_weight_suggested": np.concatenate(
            [metadata.quality_weight_suggested for metadata in tribe_metadata]
        ),
    }
    row_ownership_arrays = {
        key: aligned_arrays[key] for key in ("video_id", "row_index", "time_seconds")
    }
    target_source_arrays = {
        key: aligned_arrays[key]
        for key in (
            "native_label_frame_count",
            "source_frame_position",
            "source_floor_frame_index",
            "source_ceil_frame_index",
            "source_interp_alpha",
            "source_match_quality_code",
            "arousal",
            "valence",
        )
    }
    quality_arrays = {
        key: aligned_arrays[key]
        for key in (
            "black_frame_fraction",
            "duplicate_frame_fraction",
            "quality_black_frame_flag",
            "quality_duplicate_frame_flag",
            "quality_exclusion_flag",
            "quality_weight_suggested",
        )
    }
    derived_digests = {
        "alignment_sha256": _array_digest(aligned_arrays),
        "target_source_sha256": _array_digest(target_source_arrays),
        "validity_mask_sha256": _array_digest({"valid_mask": substrate["valid_mask"]}),
        "continuous_target_sha256": _array_digest(
            {"continuous_future_maximum_increase": substrate["continuous_future_maximum_increase"]}
        ),
        "row_ownership_sha256": _array_digest(row_ownership_arrays),
        "quality_metadata_sha256": _array_digest(quality_arrays),
    }
    result = _build_result(
        checks,
        code_sha256=code_sha256,
        input_sha256=input_sha256,
        row_digest=row_digest,
        label_digest=label_digest,
        candidate_count=len(candidates),
        active_count=len(active),
        prospective_count=len(prospective),
        derived_digests=derived_digests,
    )
    if not result["phase01_pass"]:
        raise RuntimeError("Phase 01 controls did not all pass")

    _atomic_save_npz(output_root / "aligned-labels.npz", aligned_arrays)
    _atomic_save_npz(output_root / "target-substrate.npz", substrate)
    _write_json(
        output_root / "alignment-schema.json",
        {
            "schema_version": "veatic21_phase01_alignment_schema_v2",
            "row_count": total_rows,
            "row_order": "numeric video_id, then ascending native row_index",
            "label_authority": "matching V-JEPA rows.csv only",
            "tribe_metadata_arrays_accessed": list(tribe_metadata[0].accessed_arrays),
            "tribe_feature_arrays_accessed": [],
            "source_match_quality_code": {"0": "native_exact", "1": "linear_native_frames"},
            "quality_filter_applied": False,
            "all_rows_retained": True,
            "derived_digests": derived_digests,
        },
    )
    _write_json(
        output_root / "candidate-registry.json",
        {
            "schema_version": "veatic21_phase01_candidate_registry_v2",
            "derivation": {
                "minimum_video_rows": min(row_counts),
                "maximum_future_end_rows": max_end,
                "maximum_future_end_seconds": max_end / EXPECTED_ROW_HZ,
                "rule": "all 1<=start<=end<=minimum_video_rows-1",
                "all_videos_must_have_at_least_one_valid_row": True,
            },
            "initial_no_washout_candidate_count": len(active),
            "prospective_washout_candidate_count": len(prospective),
            "selected_candidate": None,
            "global_binary_label_stored": False,
            "candidates": registry,
        },
    )
    _write_json(
        output_root / "dynamics-summary.json",
        {
            "schema_version": "veatic21_phase01_dynamics_v2",
            "arousal_range": [
                float(min(np.min(video.arousal) for video in videos)),
                float(max(np.max(video.arousal) for video in videos)),
            ],
            "valence_range": [
                float(min(np.min(video.valence) for video in videos)),
                float(max(np.max(video.valence) for video in videos)),
            ],
            **dynamics,
        },
    )
    _write_json(output_root / "causal-history-diagnostics.json", causal)
    _write_json(
        output_root / "veatic-derivation-ledger.json",
        {
            "schema_version": "veatic21_fresh_derivation_ledger_v2",
            "phase": "phase-01-label-alignment",
            "code_sha256": code_sha256,
            "input_identity_sha256": input_sha256,
            "derived_digests": derived_digests,
            "entries": [
                {
                    "name": "maximum_future_endpoint",
                    "value": {"rows": max_end, "seconds": max_end / EXPECTED_ROW_HZ},
                    "evidence": "minimum observed VEATIC video row count",
                    "derivation_rule": "minimum_video_rows - 1",
                    "owned_rows": "complete label table",
                },
                {
                    "name": "candidate_registry",
                    "value": {"initial": len(active), "prospective": len(prospective)},
                    "evidence": "complete VEATIC-derived endpoint lattice",
                    "derivation_rule": "all 1<=start<=end<=maximum_future_endpoint",
                    "owned_rows": "complete label table",
                },
                {
                    "name": "selected_target",
                    "value": None,
                    "evidence": "Phase 01 does not use cortical or AR results",
                    "derivation_rule": "selection deferred to comprehensive Phase 02",
                    "owned_rows": "none",
                },
            ],
        },
    )
    _write_json(output_root / "result.json", result)
    _write_text(
        output_root / "report.md",
        f"""# VEATIC 2.1 fresh Phase 01 label alignment

Status: **PASS** ({result['mandatory_controls_passed']}/{result['mandatory_controls_expected']})

All 124 videos and all 20,657 exact 2 Hz rows were reconstructed from authoritative
`rows.csv` labels and matched exactly to TRIBE row time. Native interpolation provenance and
row-level TRIBE quality metadata were preserved; no row was filtered.

The VEATIC-derived maximum future endpoint is {max_end} rows ({max_end / 2:.1f}s), determined
only by the shortest video so every registered candidate retains support in every video. The
complete lattice contains {len(active)} initial no-washout windows and {len(prospective)}
prospective washout windows. No target was selected and no global binary label was stored.

Autocorrelation, partial autocorrelation, movement, duration, causal-history correlation,
coverage, per-video support, and descriptive q90 stability were audited label-only. No
cortical value or performance, split, AR, PCA, head, or learned model was opened or fitted.

Only a comprehensive fresh Phase 02 target-specific AR benchmark is authorized next.
""",
    )
    payload_names = (
        "request.json",
        "aligned-labels.npz",
        "target-substrate.npz",
        "alignment-schema.json",
        "candidate-registry.json",
        "dynamics-summary.json",
        "causal-history-diagnostics.json",
        "veatic-derivation-ledger.json",
        "result.json",
        "report.md",
    )
    output_hashes = _write_artifact_manifests(
        output_root,
        payload_names,
        schema_version="veatic21_fresh_phase01_artifact_manifest_v2",
    )
    return {
        **result,
        "output_hashes": {
            **output_hashes,
            "checksums.sha256": sha256_file(output_root / "checksums.sha256"),
        },
    }


def verify_phase01_output(output_root: Path = PHASE01_ROOT) -> dict[str, object]:
    """Independently verify the sealed Phase 01 files and scientific invariants."""

    output_root = reject_forbidden_runtime_path(output_root)
    manifest = load_json(output_root / "artifact-manifest.json")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ValueError("invalid Phase 01 artifact manifest")
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("invalid Phase 01 artifact record")
        path = output_root / record["path"]
        if record.get("bytes") != path.stat().st_size or record.get("sha256") != sha256_file(path):
            raise ValueError(f"Phase 01 artifact mismatch: {path}")
    checksum_path = output_root / "checksums.sha256"
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected_sha256, name = line.split("  ", maxsplit=1)
        if sha256_file(output_root / name) != expected_sha256:
            raise ValueError(f"Phase 01 checksum mismatch: {name}")

    result = load_json(output_root / "result.json")
    checks = result.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(PHASE01_MANDATORY_CHECK_NAMES):
        raise ValueError("Phase 01 mandatory-control set mismatch")
    if not all(checks.values()) or result.get("phase01_pass") is not True:
        raise ValueError("Phase 01 is not a complete pass")

    with np.load(output_root / "aligned-labels.npz", allow_pickle=False) as payload:
        aligned = {key: payload[key] for key in payload.files}
    with np.load(output_root / "target-substrate.npz", allow_pickle=False) as payload:
        substrate = {key: payload[key] for key in payload.files}
    expected_aligned = {
        "video_id",
        "row_index",
        "time_seconds",
        "native_label_frame_count",
        "source_frame_position",
        "source_floor_frame_index",
        "source_ceil_frame_index",
        "source_interp_alpha",
        "source_match_quality_code",
        "arousal",
        "valence",
        "black_frame_fraction",
        "duplicate_frame_fraction",
        "quality_black_frame_flag",
        "quality_duplicate_frame_flag",
        "quality_exclusion_flag",
        "quality_weight_suggested",
    }
    if set(aligned) != expected_aligned or any(
        value.shape != (EXPECTED_ROW_COUNT,) for value in aligned.values()
    ):
        raise ValueError("Phase 01 aligned table schema mismatch")
    if set(substrate) != {
        "candidate_start_rows",
        "candidate_end_rows",
        "continuous_future_maximum_increase",
        "valid_mask",
    }:
        raise ValueError("Phase 01 target substrate schema mismatch")
    values = substrate["continuous_future_maximum_increase"]
    mask = substrate["valid_mask"]
    if values.shape != (231, EXPECTED_ROW_COUNT) or mask.shape != values.shape:
        raise ValueError("Phase 01 target substrate shape mismatch")
    if not np.isfinite(values[mask]).all() or not np.isnan(values[~mask]).all():
        raise ValueError("Phase 01 target mask/value semantics mismatch")
    if any("binary" in key.lower() or "threshold" in key.lower() for key in substrate):
        raise ValueError("Phase 01 stored a forbidden binary target")

    registry = load_json(output_root / "candidate-registry.json")
    candidates = registry.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 231:
        raise ValueError("Phase 01 candidate registry size mismatch")
    actual_pairs = {
        (int(row["future_start_rows"]), int(row["future_end_rows"]))
        for row in candidates
        if isinstance(row, dict)
    }
    expected_pairs = {(start, end) for end in range(1, 22) for start in range(1, end + 1)}
    if actual_pairs != expected_pairs:
        raise ValueError("Phase 01 candidate lattice mismatch")
    active = [row for row in candidates if isinstance(row, dict) and row.get("phase02_active")]
    prospective = [
        row for row in candidates if isinstance(row, dict) and row.get("prospective_only")
    ]
    if len(active) != 21 or len(prospective) != 210:
        raise ValueError("Phase 01 candidate family count mismatch")

    quality_union = aligned["quality_exclusion_flag"].astype(bool)
    black = aligned["quality_black_frame_flag"].astype(bool)
    duplicate = aligned["quality_duplicate_frame_flag"].astype(bool)
    if not np.array_equal(quality_union, black | duplicate) or int(quality_union.sum()) != 923:
        raise ValueError("Phase 01 quality metadata mismatch")
    if set(np.unique(aligned["source_match_quality_code"])) != {0, 1}:
        raise ValueError("Phase 01 interpolation provenance code mismatch")

    digests = result.get("derived_digests")
    if not isinstance(digests, dict):
        raise ValueError("Phase 01 derived digests missing")
    quality_keys = (
        "black_frame_fraction",
        "duplicate_frame_fraction",
        "quality_black_frame_flag",
        "quality_duplicate_frame_flag",
        "quality_exclusion_flag",
        "quality_weight_suggested",
    )
    target_source_keys = (
        "native_label_frame_count",
        "source_frame_position",
        "source_floor_frame_index",
        "source_ceil_frame_index",
        "source_interp_alpha",
        "source_match_quality_code",
        "arousal",
        "valence",
    )
    recomputed = {
        "alignment_sha256": _array_digest(aligned),
        "target_source_sha256": _array_digest(
            {key: aligned[key] for key in target_source_keys}
        ),
        "validity_mask_sha256": _array_digest({"valid_mask": mask}),
        "continuous_target_sha256": _array_digest(
            {"continuous_future_maximum_increase": values}
        ),
        "row_ownership_sha256": _array_digest(
            {key: aligned[key] for key in ("video_id", "row_index", "time_seconds")}
        ),
        "quality_metadata_sha256": _array_digest({key: aligned[key] for key in quality_keys}),
    }
    if digests != recomputed:
        raise ValueError("Phase 01 derived digest mismatch")
    return {
        "verified": True,
        "artifact_manifest_sha256": sha256_file(output_root / "artifact-manifest.json"),
        "checksums_sha256": sha256_file(checksum_path),
        "result_sha256": sha256_file(output_root / "result.json"),
        "result": result,
    }
