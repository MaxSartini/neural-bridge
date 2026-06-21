"""Build AGAIN manifests from audited per-video boundary recommendations."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SPIKE_THRESHOLDS = (0.05, 0.075)
CHANGE_THRESHOLDS = (0.05, 0.075)
BOUNDARY_POLICY = "use_annotation_covered_video_time_only"


@dataclass(frozen=True)
class AnnotationSeries:
    times: np.ndarray
    arousal: np.ndarray
    frame_indices: np.ndarray
    rows: list[dict[str, str]]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _clean_for_csv(row.get(key, "")) for key in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean_for_json(payload), indent=2, sort_keys=True), encoding="utf-8")


def load_annotation_series(manifest_proposal_path: Path) -> dict[str, AnnotationSeries]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv_rows(manifest_proposal_path):
        video_id = row.get("video_id", "")
        if video_id:
            grouped[video_id].append(row)

    series: dict[str, AnnotationSeries] = {}
    for video_id, rows in grouped.items():
        times: list[float] = []
        arousal: list[float] = []
        frame_indices: list[float] = []
        kept_rows: list[dict[str, str]] = []
        for row in rows:
            t = safe_float(row.get("time_start_seconds"))
            y = safe_float(row.get("arousal"))
            if t is None or y is None:
                continue
            times.append(t)
            arousal.append(y)
            frame_index = safe_float(row.get("frame_index"))
            frame_indices.append(frame_index if frame_index is not None else math.nan)
            kept_rows.append(row)
        if len(times) < 2:
            continue
        t_arr = np.asarray(times, dtype=np.float64)
        y_arr = np.asarray(arousal, dtype=np.float64)
        frame_arr = np.asarray(frame_indices, dtype=np.float64)
        order = np.argsort(t_arr)
        t_arr = t_arr[order]
        y_arr = y_arr[order]
        frame_arr = frame_arr[order]
        kept_rows = [kept_rows[int(index)] for index in order]
        unique_times, inverse = np.unique(t_arr, return_inverse=True)
        if len(unique_times) != len(t_arr):
            sums = np.zeros(len(unique_times), dtype=np.float64)
            frame_sums = np.zeros(len(unique_times), dtype=np.float64)
            counts = np.zeros(len(unique_times), dtype=np.float64)
            np.add.at(sums, inverse, y_arr)
            np.add.at(frame_sums, inverse, np.nan_to_num(frame_arr, nan=0.0))
            np.add.at(counts, inverse, 1.0)
            t_arr = unique_times
            y_arr = sums / counts
            frame_arr = frame_sums / counts
        series[video_id] = AnnotationSeries(times=t_arr, arousal=y_arr, frame_indices=frame_arr, rows=kept_rows)
    return series


def build_boundary_aligned_manifest(
    boundary_rows: Iterable[dict[str, str]],
    annotations_by_video: dict[str, AnnotationSeries],
    *,
    sampling_hz: float = 1.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if sampling_hz <= 0 or not math.isfinite(sampling_hz):
        raise ValueError("sampling_hz must be positive")
    if abs(sampling_hz - 1.0) > 1e-9:
        raise ValueError("Only 1Hz manifests are currently supported by this builder")

    manifest_rows: list[dict[str, Any]] = []
    video_summary_rows: list[dict[str, Any]] = []
    missing_annotation_videos: list[str] = []
    non_policy_videos: list[str] = []

    for boundary in boundary_rows:
        video_id = boundary.get("video_id", "")
        if not video_id:
            continue
        policy = boundary.get("recommended_policy", "")
        if policy != BOUNDARY_POLICY:
            non_policy_videos.append(video_id)
            continue
        series = annotations_by_video.get(video_id)
        if series is None:
            missing_annotation_videos.append(video_id)
            continue

        grid_start = safe_int(boundary.get("one_hz_grid_start_second"))
        grid_end = safe_int(boundary.get("one_hz_grid_end_second"))
        target_safe_end = safe_int(boundary.get("one_hz_target_safe_end_second"))
        benchmark_start = safe_float(boundary.get("recommended_benchmark_start_seconds"))
        benchmark_end = safe_float(boundary.get("recommended_benchmark_end_seconds"))
        if grid_start is None or grid_end is None or target_safe_end is None:
            raise ValueError(f"Missing 1Hz grid fields for video {video_id}")
        if benchmark_start is None or benchmark_end is None:
            raise ValueError(f"Missing benchmark boundary fields for video {video_id}")
        if grid_end < grid_start:
            raise ValueError(f"Invalid 1Hz grid for video {video_id}")

        times = np.arange(grid_start, grid_end + 1, dtype=np.float64)
        current = np.interp(times, series.times, series.arousal)
        frame_indices = interpolate_frame_indices(series.times, series.frame_indices, times)
        source_row = series.rows[0] if series.rows else {}
        for index, (time_s, arousal, frame_index) in enumerate(zip(times, current, frame_indices)):
            target_info = seconds_based_targets(series.times, series.arousal, float(time_s), float(arousal), benchmark_end)
            row: dict[str, Any] = {
                "dataset_name": "AGAIN_cleaned",
                "video_id": video_id,
                "video_path": boundary.get("video_path", ""),
                "sampling_rate_hz": 1.0,
                "time_start_seconds": float(time_s),
                "aligned_time_seconds": float(time_s),
                "original_video_time_seconds": float(time_s),
                "frame_index": int(round(float(frame_index))) if math.isfinite(float(frame_index)) else "",
                "timestamp_index": index,
                "arousal": float(arousal),
                "valence": "",
                "participant_id": boundary.get("participant_id", source_row.get("participant_id", "")),
                "session_id": boundary.get("session_id", source_row.get("session_id", "")),
                "game": boundary.get("game", source_row.get("game", "")),
                "genre": source_row.get("genre", ""),
                "aggregate_method": source_row.get("aggregate_method", "none_participant_session_label"),
                "split_group": source_row.get("split_group", "not_assigned"),
                "alignment_policy": BOUNDARY_POLICY,
                "recommended_encode_start_seconds": safe_float(boundary.get("recommended_encode_start_seconds")),
                "recommended_encode_end_seconds": safe_float(boundary.get("recommended_encode_end_seconds")),
                "recommended_benchmark_start_seconds": benchmark_start,
                "recommended_benchmark_end_seconds": benchmark_end,
                "target_safe_end_future_1_3s_seconds": safe_float(boundary.get("target_safe_end_future_1_3s_seconds")),
                "trim_start_seconds": safe_float(boundary.get("trim_start_seconds")),
                "trim_end_seconds": safe_float(boundary.get("trim_end_seconds")),
                "boundary_confidence": boundary.get("boundary_confidence", ""),
                "boundary_notes": boundary.get("notes", ""),
                "arousal_interpolation_method": "linear_from_native_annotations",
                "label_is_interpolated": not bool(np.any(np.isclose(series.times, time_s, atol=1e-6))),
                "target_feasible_future_spike_1_3s": bool(time_s <= target_safe_end),
                "target_feasible_future_change_p3s": bool(time_s <= target_safe_end),
                "dropped_by_alignment": False,
                "source_metadata": json.dumps(
                    {
                        "boundary_policy_source": "again_video_boundary_recommendations.csv",
                        "annotation_source": "again_manifest_proposal.csv",
                        "source_metadata": source_row.get("source_metadata", ""),
                    },
                    sort_keys=True,
                ),
                "alignment_status": "boundary_audited_annotation_covered",
            }
            row.update(target_info)
            manifest_rows.append(row)

        video_summary_rows.append(
            {
                "video_id": video_id,
                "video_path": boundary.get("video_path", ""),
                "boundary_confidence": boundary.get("boundary_confidence", ""),
                "recommended_benchmark_start_seconds": benchmark_start,
                "recommended_benchmark_end_seconds": benchmark_end,
                "one_hz_rows": int(len(times)),
                "target_feasible_1hz_rows": int(np.sum(times <= target_safe_end)),
                "annotation_start_seconds": safe_float(boundary.get("annotation_start_seconds")),
                "annotation_end_seconds": safe_float(boundary.get("annotation_end_seconds")),
                "trim_end_seconds": safe_float(boundary.get("trim_end_seconds")),
                "notes": boundary.get("notes", ""),
            }
        )

    summary = {
        "schema_version": "again_boundary_aligned_manifest_summary_v1",
        "sampling_rate_hz": 1.0,
        "alignment_policy": BOUNDARY_POLICY,
        "videos_in_manifest": len(video_summary_rows),
        "manifest_rows": len(manifest_rows),
        "target_feasible_rows_future_spike_1_3s": sum(1 for row in manifest_rows if row["target_feasible_future_spike_1_3s"]),
        "target_feasible_rows_future_change_p3s": sum(1 for row in manifest_rows if row["target_feasible_future_change_p3s"]),
        "missing_annotation_videos": missing_annotation_videos,
        "non_policy_videos": non_policy_videos,
        "final_benchmark_manifest_created": False,
        "tribe_encoding_run": False,
        "models_trained": False,
        "veatic_outputs_modified": False,
    }
    return manifest_rows, video_summary_rows, summary


def seconds_based_targets(
    annotation_times: np.ndarray,
    arousal_values: np.ndarray,
    time_s: float,
    current_arousal: float,
    context_end_seconds: float,
) -> dict[str, Any]:
    future_start = time_s + 1.0
    future_end = time_s + 3.0
    feasible = future_end <= context_end_seconds
    result: dict[str, Any] = {
        "future_spike_1_3s_delta": "",
        "future_change_p3s_value": "",
    }
    for threshold in SPIKE_THRESHOLDS:
        result[f"future_spike_1_3s_ge_{threshold:g}"] = ""
    for threshold in CHANGE_THRESHOLDS:
        result[f"future_change_p3s_movement_ge_{threshold:g}"] = ""
    if not feasible:
        return result

    future_mask = (annotation_times >= future_start) & (annotation_times <= future_end)
    if np.any(future_mask):
        spike_delta = float(np.max(arousal_values[future_mask]) - current_arousal)
        result["future_spike_1_3s_delta"] = spike_delta
        for threshold in SPIKE_THRESHOLDS:
            result[f"future_spike_1_3s_ge_{threshold:g}"] = bool(spike_delta >= threshold)

    change_time = time_s + 3.0
    if annotation_times[0] <= change_time <= context_end_seconds:
        future_value = float(np.interp(change_time, annotation_times, arousal_values))
        change = future_value - current_arousal
        result["future_change_p3s_value"] = change
        for threshold in CHANGE_THRESHOLDS:
            result[f"future_change_p3s_movement_ge_{threshold:g}"] = bool(abs(change) >= threshold)
    return result


def interpolate_frame_indices(source_times: np.ndarray, source_frame_indices: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    finite = np.isfinite(source_times) & np.isfinite(source_frame_indices)
    if np.sum(finite) < 2:
        return np.full(len(target_times), math.nan, dtype=np.float64)
    return np.interp(target_times, source_times[finite], source_frame_indices[finite])


def build_report(summary: dict[str, Any], output_root: Path) -> str:
    return "\n".join(
        [
            "# AGAIN Boundary-Aligned 1Hz Manifest",
            "",
            "## Summary",
            "",
            f"- alignment policy: `{summary['alignment_policy']}`",
            f"- videos: `{summary['videos_in_manifest']}`",
            f"- rows: `{summary['manifest_rows']}`",
            f"- future spike/change feasible rows: `{summary['target_feasible_rows_future_spike_1_3s']}`",
            "",
            "## Boundary Rule",
            "",
            "Rows start at the audited benchmark start and stop at each video's annotation-covered end time.",
            "The post-annotation tail is not used for benchmark rows, even when it contains visible motion.",
            "Future targets are seconds-based and are only marked feasible through `annotation_end_seconds - 3s`.",
            "",
            "## Guardrails",
            "",
            "tribe_encoding_run=false",
            "models_trained=false",
            "veatic_outputs_modified=false",
            "final_benchmark_manifest_created=false",
            "",
            "## Files",
            "",
            f"- `{output_root / 'again_boundary_aligned_1hz_manifest.csv'}`",
            f"- `{output_root / 'again_boundary_aligned_video_summary.csv'}`",
            f"- `{output_root / 'again_boundary_aligned_manifest_summary.json'}`",
            "",
        ]
    )


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def _clean_for_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _clean_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_for_json(item) for item in value]
    return value


def _clean_for_csv(value: Any) -> Any:
    value = _clean_for_json(value)
    if isinstance(value, bool):
        return str(value).lower()
    return value


def run_builder(
    *,
    boundary_recommendations_path: Path,
    manifest_proposal_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root exists and is non-empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    boundary_rows = read_csv_rows(boundary_recommendations_path)
    annotations = load_annotation_series(manifest_proposal_path)
    manifest_rows, video_summary_rows, summary = build_boundary_aligned_manifest(boundary_rows, annotations)
    summary = {
        **summary,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "boundary_recommendations_path": str(boundary_recommendations_path),
        "manifest_proposal_path": str(manifest_proposal_path),
        "output_root": str(output_root),
    }
    write_csv(output_root / "again_boundary_aligned_1hz_manifest.csv", manifest_rows)
    write_csv(output_root / "again_boundary_aligned_video_summary.csv", video_summary_rows)
    write_json(output_root / "again_boundary_aligned_manifest_summary.json", summary)
    (output_root / "again_boundary_aligned_manifest_report.md").write_text(build_report(summary, output_root), encoding="utf-8")
    run_manifest = {
        "schema_version": "again_boundary_aligned_manifest_run_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files_written": [
            str(output_root / "again_boundary_aligned_1hz_manifest.csv"),
            str(output_root / "again_boundary_aligned_video_summary.csv"),
            str(output_root / "again_boundary_aligned_manifest_summary.json"),
            str(output_root / "again_boundary_aligned_manifest_report.md"),
            str(output_root / "run_manifest.json"),
        ],
        **summary,
    }
    write_json(output_root / "run_manifest.json", run_manifest)
    return run_manifest
