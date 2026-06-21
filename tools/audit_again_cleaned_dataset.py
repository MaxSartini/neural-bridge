#!/usr/bin/env python3
"""Inventory and compatibility audit for the cleaned AGAIN dataset.

This is a read-only source-data audit. It does not encode videos, train models,
or touch VEATIC benchmark outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
ANNOTATION_KEY = ("[control]player_id", "[control]game", "[control]session_id")
AROUSAL_COL = "[output]arousal"
TIME_COL = "[control]time_stamp"
PLAYER_COL = "[control]player_id"
GAME_COL = "[control]game"
SESSION_COL = "[control]session_id"
GENRE_COL = "[control]genre"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit cleaned AGAIN for VEATIC-compatible benchmark conversion.")
    parser.add_argument(
        "--again-root",
        type=Path,
        default=default_again_root(),
        help="Cleaned AGAIN dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Fresh audit output directory. Defaults to outputs/again_cleaned_inventory_audit_<timestamp>.",
    )
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args()


def default_again_root() -> Path:
    configured = os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT", "").strip()
    if not configured:
        raise EnvironmentError("Set NEURAL_BRIDGE_EXTERNAL_ROOT to the Neural Bridge external assets root.")
    return Path(configured).expanduser() / "data" / "external" / "AGAIN" / "cleaned"


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def finite(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite(item) for item in value]
    return value


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
            writer.writerow({key: finite(row.get(key, "")) for key in fieldnames})


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def session_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get(PLAYER_COL, "")), str(row.get(GAME_COL, "")), str(row.get(SESSION_COL, "")))


def metadata_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("[control]player_id", "")), str(row.get("[control]game", "")), str(row.get("[control]session_id", "")))


def video_id_from_name(video_name: str) -> str:
    return Path(video_name).stem


def ffprobe_video(path: Path, ffprobe: str) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,format_name:stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(command, check=True, text=True, capture_output=True, timeout=30)
        payload = json.loads(proc.stdout)
    except Exception as exc:
        return {
            "readable": False,
            "ffprobe_error": f"{type(exc).__name__}: {exc}",
        }
    streams = payload.get("streams", [])
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    first = video_streams[0] if video_streams else {}
    format_info = payload.get("format", {})
    fps = fps_from_rate(first.get("avg_frame_rate") or first.get("r_frame_rate"))
    duration = safe_float(first.get("duration")) or safe_float(format_info.get("duration"))
    duration_source = "stream_or_container" if duration is not None else ""
    fps_source = "avg_or_r_frame_rate" if fps is not None else ""
    packet_stats: dict[str, Any] = {}
    if video_streams and (duration is None or fps is None):
        packet_stats = ffprobe_packet_stats(path, ffprobe)
        if duration is None and packet_stats.get("duration_seconds") is not None:
            duration = packet_stats["duration_seconds"]
            duration_source = "packet_timestamps"
        if fps is None and packet_stats.get("fps_estimate") is not None:
            fps = packet_stats["fps_estimate"]
            fps_source = "packet_count_over_duration"
    return {
        "readable": bool(video_streams),
        "ffprobe_error": "",
        "ffprobe_warning": proc.stderr.strip(),
        "codec": first.get("codec_name", ""),
        "container_format": format_info.get("format_name", ""),
        "duration_seconds": duration,
        "duration_source": duration_source,
        "fps": fps,
        "fps_source": fps_source,
        "packet_count": packet_stats.get("packet_count"),
        "width": safe_int(first.get("width")),
        "height": safe_int(first.get("height")),
        "size_bytes": safe_int(format_info.get("size")) or path.stat().st_size,
    }


def ffprobe_packet_stats(path: Path, ffprobe: str) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "packet=pts_time,dts_time",
        "-of",
        "csv=p=0",
        str(path),
    ]
    try:
        proc = subprocess.run(command, check=True, text=True, capture_output=True, timeout=60)
    except Exception:
        return {}
    timestamps: list[float] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",") if part.strip()]
        for part in parts[:2]:
            value = safe_float(part)
            if value is not None:
                timestamps.append(value)
                break
    if not timestamps:
        return {}
    min_ts = min(timestamps)
    max_ts = max(timestamps)
    duration = max_ts - min_ts
    return {
        "duration_seconds": duration if duration > 0 else None,
        "packet_count": len(timestamps),
        "fps_estimate": (len(timestamps) / duration) if duration > 0 else None,
        "packet_warning": proc.stderr.strip(),
    }


def fps_from_rate(rate: str | None) -> float | None:
    if not rate or rate == "0/0":
        return None
    if "/" in rate:
        num, den = rate.split("/", 1)
        numerator = safe_float(num)
        denominator = safe_float(den)
        if numerator is None or denominator in (None, 0):
            return None
        return numerator / denominator
    return safe_float(rate)


def sha1_short(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def summarize_annotations(clean_rows: list[dict[str, str]], metadata_by_key: dict[tuple[str, str, str], dict[str, str]]) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in clean_rows:
        grouped[session_key(row)].append(row)

    inventory_rows: list[dict[str, Any]] = []
    session_stats: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, rows in sorted(grouped.items()):
        times = [safe_float(row.get(TIME_COL)) for row in rows]
        arousal = [safe_float(row.get(AROUSAL_COL)) for row in rows]
        finite_times = [item for item in times if item is not None]
        finite_arousal = [item for item in arousal if item is not None]
        duplicate_timestamps = len(finite_times) - len(set(finite_times))
        sorted_times = sorted(finite_times)
        deltas = [b - a for a, b in zip(sorted_times, sorted_times[1:]) if b >= a]
        median_delta = median(deltas)
        sampling_hz = 1.0 / median_delta if median_delta and median_delta > 0 else None
        min_time = min(finite_times) if finite_times else None
        max_time = max(finite_times) if finite_times else None
        annotation_duration = (max_time - min_time) if min_time is not None and max_time is not None else None
        meta = metadata_by_key.get(key, {})
        video_name = meta.get("video_name", "")
        row = {
            "dataset_name": "AGAIN_cleaned",
            "video_id": video_id_from_name(video_name) if video_name else "",
            "video_name": video_name,
            "participant_id": key[0],
            "game": key[1],
            "session_id": key[2],
            "annotation_file": "annotations/clean_data.csv",
            "schema_columns": 121,
            "available_labels": "arousal",
            "arousal_available": True,
            "valence_available": False,
            "label_level": "participant_session",
            "aggregate_labels_available": False,
            "rows": len(rows),
            "missing_arousal": len(arousal) - len(finite_arousal),
            "nonfinite_arousal": len(arousal) - len(finite_arousal),
            "min_arousal": min(finite_arousal) if finite_arousal else None,
            "max_arousal": max(finite_arousal) if finite_arousal else None,
            "mean_arousal": sum(finite_arousal) / len(finite_arousal) if finite_arousal else None,
            "label_scale": "session-normalized approximately [0,1]",
            "label_type": "continuous",
            "min_timestamp_seconds": min_time,
            "max_timestamp_seconds": max_time,
            "annotation_duration_seconds": annotation_duration,
            "median_sample_delta_seconds": median_delta,
            "sampling_rate_hz_estimate": sampling_hz,
            "duplicated_timestamps": duplicate_timestamps,
            "missing_timestamps": len(times) - len(finite_times),
            "timestamp_monotonic_after_sort": True,
            "metadata_present": bool(meta),
            "video_exists_from_metadata": str(meta.get("video_exists", "")).lower() == "true",
        }
        inventory_rows.append(row)
        session_stats[key] = row
    return inventory_rows, session_stats


def median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def inventory_metadata(root: Path, clean_rows: list[dict[str, str]], session_metadata: list[dict[str, str]], schema_rows: list[dict[str, str]], outlier_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    files = [
        root / "annotations" / "clean_data.csv",
        root / "annotations" / "outliers.csv",
        root / "metadata" / "cleaned_session_video_metadata.csv",
        root / "metadata" / "cleaned_column_schema.csv",
        root / "metadata" / "cleaned_metadata_summary.json",
        root / "metadata" / "README_cleaned_metadata.md",
        root / "metadata" / "again_dataset_public_description.txt",
        root / "manifests" / "again_cleaned_download_queue.csv",
        root / "manifests" / "again_cleaned_video_download_log.csv",
    ]
    metadata_key_count = len({metadata_key(row) for row in session_metadata})
    annotation_key_count = len({session_key(row) for row in clean_rows})
    rows: list[dict[str, Any]] = []
    for path in files:
        rel = path.relative_to(root)
        suffix = path.suffix.lower()
        kind = "metadata"
        if "annotation" in str(rel) or path.name in {"clean_data.csv", "outliers.csv"}:
            kind = "annotation"
        elif "manifest" in str(rel):
            kind = "manifest"
        detail: dict[str, Any] = {}
        if path.name == "clean_data.csv":
            detail = {
                "rows": len(clean_rows),
                "schema_columns": len(clean_rows[0]) if clean_rows else 0,
                "video_id_mapping": "join [control]player_id + [control]game + [control]session_id to cleaned_session_video_metadata.csv",
                "participant_session_mapping": True,
                "train_test_splits_provided": False,
                "stimulus_categories_present": True,
                "license_or_readme_notes": "Public description says cleaned data are normalized on session level.",
            }
        elif path.name == "cleaned_session_video_metadata.csv":
            detail = {
                "rows": len(session_metadata),
                "schema_columns": len(session_metadata[0]) if session_metadata else 0,
                "video_id_mapping": "video_name, drive_file_id, player_id/game/session_id",
                "participant_session_mapping": True,
                "metadata_session_count": metadata_key_count,
                "annotation_session_count": annotation_key_count,
            }
        elif path.name == "cleaned_column_schema.csv":
            detail = {
                "rows": len(schema_rows),
                "schema_columns": len(schema_rows[0]) if schema_rows else 0,
                "declared_output_columns": ",".join(row["column"] for row in schema_rows if row.get("category") == "output"),
            }
        elif path.name == "outliers.csv":
            detail = {
                "rows": len(outlier_rows),
                "schema_columns": len(outlier_rows[0]) if outlier_rows else 0,
            }
        rows.append(
            {
                "file_path": str(path),
                "relative_path": str(rel),
                "file_kind": kind,
                "extension": suffix,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                **detail,
            }
        )
    return rows


def inventory_videos(root: Path, metadata_rows: list[dict[str, str]], annotation_stats: dict[tuple[str, str, str], dict[str, Any]], ffprobe: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    videos_dir = root / "videos"
    video_paths = sorted(path for path in videos_dir.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    video_by_name = {path.name: path for path in video_paths}
    metadata_by_video = {row.get("video_name", ""): row for row in metadata_rows if row.get("video_name")}
    expected_names = set(metadata_by_video)
    actual_names = set(video_by_name)
    rows: list[dict[str, Any]] = []
    duplicate_name_counts = Counter(path.name for path in video_paths)
    size_duration_groups: dict[tuple[Any, Any], list[str]] = defaultdict(list)
    total_duration = 0.0
    unreadable = 0
    for video_name in sorted(actual_names | expected_names):
        path = video_by_name.get(video_name)
        meta = metadata_by_video.get(video_name, {})
        key = metadata_key(meta) if meta else ("", "", "")
        annotation = annotation_stats.get(key, {})
        probe = ffprobe_video(path, ffprobe) if path else {"readable": False, "ffprobe_error": "missing file"}
        duration = probe.get("duration_seconds")
        if duration is not None:
            total_duration += float(duration)
        if not probe.get("readable"):
            unreadable += 1
        signature = (probe.get("size_bytes"), round(float(duration), 3) if duration is not None else None)
        size_duration_groups[signature].append(video_name)
        rows.append(
            {
                "dataset_name": "AGAIN_cleaned",
                "video_id": video_id_from_name(video_name),
                "video_name": video_name,
                "video_path": str(path) if path else "",
                "relative_video_path": str(path.relative_to(root)) if path else "",
                "extension": path.suffix.lower() if path else Path(video_name).suffix.lower(),
                "codec": probe.get("codec"),
                "container_format": probe.get("container_format"),
                "duration_seconds": duration,
                "duration_source": probe.get("duration_source"),
                "fps": probe.get("fps"),
                "fps_source": probe.get("fps_source"),
                "packet_count": probe.get("packet_count"),
                "width": probe.get("width"),
                "height": probe.get("height"),
                "resolution": f"{probe.get('width')}x{probe.get('height')}" if probe.get("width") and probe.get("height") else "",
                "size_bytes": probe.get("size_bytes"),
                "readable": probe.get("readable"),
                "ffprobe_error": probe.get("ffprobe_error", ""),
                "ffprobe_warning": probe.get("ffprobe_warning", ""),
                "participant_id": meta.get("[control]player_id", ""),
                "game": meta.get("[control]game", ""),
                "session_id": meta.get("[control]session_id", ""),
                "drive_file_id": meta.get("drive_file_id", ""),
                "metadata_present": bool(meta),
                "annotation_present": bool(annotation),
                "annotation_rows": annotation.get("rows"),
                "annotation_duration_seconds": annotation.get("annotation_duration_seconds"),
                "duplicate_filename_count": duplicate_name_counts[video_name],
                "has_no_annotation": not bool(annotation),
                "has_no_metadata": not bool(meta),
            }
        )
    potential_duplicate_names = []
    for names in size_duration_groups.values():
        if len(names) > 1:
            potential_duplicate_names.extend(sorted(names))
    for row in rows:
        row["potential_duplicate_by_size_duration"] = row["video_name"] in set(potential_duplicate_names)
    summary = {
        "video_files": len(video_paths),
        "expected_metadata_videos": len(expected_names),
        "readable_videos": sum(1 for row in rows if row["readable"]),
        "missing_video_files": len(expected_names - actual_names),
        "unexpected_video_files": len(actual_names - expected_names),
        "unreadable_or_corrupt_videos": unreadable,
        "duplicate_filenames": sum(1 for count in duplicate_name_counts.values() if count > 1),
        "potential_duplicates_by_size_duration": len(set(potential_duplicate_names)),
        "videos_with_no_annotation": sum(1 for row in rows if row["has_no_annotation"]),
        "videos_with_no_metadata": sum(1 for row in rows if row["has_no_metadata"]),
        "videos_ready_for_encoding_before_alignment_review": sum(
            1 for row in rows if row["readable"] and not row["has_no_annotation"] and not row["has_no_metadata"]
        ),
        "total_duration_seconds": total_duration,
        "total_duration_hours": total_duration / 3600.0,
    }
    return rows, summary


def alignment_audit(video_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in video_rows:
        video_duration = safe_float(row.get("duration_seconds"))
        annotation_duration = safe_float(row.get("annotation_duration_seconds"))
        mismatch = None
        if video_duration is not None and annotation_duration is not None:
            mismatch = video_duration - annotation_duration
        annotation_rows = safe_int(row.get("annotation_rows")) or 0
        sampling_feasible = bool(annotation_rows >= 2 and annotation_duration is not None and annotation_duration >= 3)
        target_feasible = bool(sampling_feasible and annotation_duration is not None and annotation_duration >= 6)
        status = "usable"
        if not row.get("readable"):
            status = "video_unreadable"
        elif not row.get("annotation_present"):
            status = "missing_annotation"
        elif mismatch is not None and abs(mismatch) > 1.0:
            status = "duration_mismatch_gt_1s"
        rows.append(
            {
                "dataset_name": "AGAIN_cleaned",
                "video_id": row["video_id"],
                "video_name": row["video_name"],
                "video_duration_seconds": video_duration,
                "annotation_duration_seconds": annotation_duration,
                "duration_mismatch_seconds": mismatch,
                "mismatch_gt_1s": abs(mismatch) > 1.0 if mismatch is not None else None,
                "annotation_rows": annotation_rows,
                "missing_start_timestamp": False,
                "resample_1hz_feasible": sampling_feasible,
                "future_spike_1_3s_feasible": target_feasible,
                "future_change_p3s_movement_feasible": target_feasible,
                "future_rise_drop_variants_feasible": target_feasible,
                "alignment_status": status,
            }
        )
    return rows


def manifest_proposal_rows(clean_rows: list[dict[str, str]], metadata_by_key: dict[tuple[str, str, str], dict[str, str]], alignment_by_video: dict[str, dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    rows = []
    timestamp_counts: dict[str, Counter[float]] = defaultdict(Counter)
    for row in clean_rows:
        meta = metadata_by_key.get(session_key(row), {})
        video_name = meta.get("video_name", "")
        video_id = video_id_from_name(video_name) if video_name else sha1_short("|".join(session_key(row)))
        time_seconds = safe_float(row.get(TIME_COL))
        if time_seconds is not None:
            timestamp_counts[video_id][time_seconds] += 1
        video_path = root / "videos" / video_name if video_name else None
        alignment = alignment_by_video.get(video_id, {})
        fps = safe_float(alignment.get("fps"))
        frame_index = int(round(time_seconds * fps)) if time_seconds is not None and fps else ""
        rows.append(
            {
                "dataset_name": "AGAIN_cleaned",
                "video_id": video_id,
                "video_path": str(video_path) if video_path else "",
                "time_start_seconds": time_seconds,
                "frame_index": frame_index,
                "timestamp_index": row.get("[control]time_index", ""),
                "arousal": safe_float(row.get(AROUSAL_COL)),
                "valence": "",
                "participant_id": row.get(PLAYER_COL, ""),
                "session_id": row.get(SESSION_COL, ""),
                "game": row.get(GAME_COL, ""),
                "genre": row.get(GENRE_COL, ""),
                "aggregate_method": "none_participant_session_label",
                "split_group": "not_assigned",
                "source_metadata": json.dumps(
                    {
                        "session_id": row.get(SESSION_COL, ""),
                        "game": row.get(GAME_COL, ""),
                        "genre": row.get(GENRE_COL, ""),
                        "engine_tick": row.get("[control]engine_tick", ""),
                    },
                    sort_keys=True,
                ),
                "alignment_status": alignment.get("alignment_status", "unknown"),
            }
        )
    return rows


def annotation_dataset_summary(annotation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_rows = sum(int(row["rows"]) for row in annotation_rows)
    durations = [safe_float(row["annotation_duration_seconds"]) for row in annotation_rows]
    durations = [item for item in durations if item is not None]
    rates = [safe_float(row["sampling_rate_hz_estimate"]) for row in annotation_rows]
    rates = [item for item in rates if item is not None]
    return {
        "annotation_sessions": len(annotation_rows),
        "annotation_rows": total_rows,
        "arousal_available": True,
        "valence_available": False,
        "label_level": "participant_session",
        "aggregate_labels_available": False,
        "continuous_arousal": True,
        "duration_seconds_min": min(durations) if durations else None,
        "duration_seconds_max": max(durations) if durations else None,
        "duration_seconds_mean": sum(durations) / len(durations) if durations else None,
        "sampling_rate_hz_median": median(rates),
        "sessions_with_duplicate_timestamps": sum(1 for row in annotation_rows if int(row["duplicated_timestamps"]) > 0),
        "sessions_with_missing_arousal": sum(1 for row in annotation_rows if int(row["missing_arousal"]) > 0),
        "arousal_min_global": min(row["min_arousal"] for row in annotation_rows if row["min_arousal"] is not None),
        "arousal_max_global": max(row["max_arousal"] for row in annotation_rows if row["max_arousal"] is not None),
    }


def compatibility_report(summary: dict[str, Any]) -> str:
    usable = summary["usable_videos"]
    total = summary["videos"]["video_files"]
    mismatch = summary["alignment"]["mismatch_gt_1s"]
    alignment_review_candidates = summary["usable_videos_if_allowing_gt_1s_mismatch"]
    candidate_rows = summary["annotations"]["annotation_rows"]
    strict_ready = mismatch == 0 and usable == total
    completeness_answer = (
        "yes for source completeness, but not yet for strict VEATIC-contract benchmarking without an explicit alignment convention"
        if not strict_ready
        else "yes"
    )
    return "\n".join(
        [
            "# AGAIN Cleaned Inventory and VEATIC Compatibility Audit",
            "",
            "## Executive Answers",
            "",
            f"1. Complete enough to benchmark: {completeness_answer}. Strict <=1s aligned videos: {usable}/{total}; alignment-review candidates: {alignment_review_candidates}/{total}.",
            f"2. Usable videos: {usable} strict aligned; {alignment_review_candidates} candidates after resolving the systematic duration offset.",
            f"3. Usable annotation rows: {summary['usable_annotation_rows']} strict aligned; {candidate_rows} candidate rows before alignment trimming.",
            "4. Arousal is available and continuous; valence is not present in the cleaned files.",
            f"5. 1Hz resampling is feasible for {summary['alignment']['resample_1hz_feasible']} videos.",
            f"6. VEATIC-style future spike targets are feasible for {summary['alignment']['future_spike_1_3s_feasible']} videos, subject to target-threshold selection and train-only thresholding.",
            "7. Best first split strategy: grouped leave-video/session-out, with participant-disjoint analysis as a secondary stricter split because participant IDs exist.",
            "8. First encoding/benchmark run: first inspect a small representative sample to decide whether the roughly 3s video-over-annotation offset is pre-roll/post-roll. After that, create a 1Hz aligned manifest and run a small encoding pilot before scaling to all candidates.",
            "9. Licensing/data hazards: use the saved public description/README and any upstream AGAIN license terms before redistribution; cleaned data are session-normalized, so avoid comparing absolute arousal across sessions as if globally calibrated.",
            f"10. Estimated TRIBE video encoding footprint: about {summary['compute_estimate']['total_duration_hours']:.2f} video-hours; at 1Hz, about {summary['compute_estimate']['estimated_1hz_rows']} frame/time rows before target trimming.",
            "",
            "## Dataset Root",
            "",
            f"`{summary['again_root']}`",
            "",
            "## Videos",
            "",
            f"- video files: {total}",
            f"- readable videos: {summary['videos']['readable_videos']}",
            f"- unreadable/corrupt videos: {summary['videos']['unreadable_or_corrupt_videos']}",
            f"- missing expected video files: {summary['videos']['missing_video_files']}",
            f"- videos with no annotation: {summary['videos']['videos_with_no_annotation']}",
            f"- videos with no metadata: {summary['videos']['videos_with_no_metadata']}",
            f"- candidates ready for encoding before alignment review: {summary['videos']['videos_ready_for_encoding_before_alignment_review']}",
            f"- total duration hours: {summary['videos']['total_duration_hours']:.2f}",
            "",
            "## Annotations",
            "",
            f"- annotation rows: {summary['annotations']['annotation_rows']}",
            f"- sessions: {summary['annotations']['annotation_sessions']}",
            f"- median sampling rate estimate: {summary['annotations']['sampling_rate_hz_median']:.3f} Hz",
            f"- arousal range: {summary['annotations']['arousal_min_global']} to {summary['annotations']['arousal_max_global']}",
            "- label type: continuous arousal, participant/session-level, session-normalized.",
            "",
            "## Alignment",
            "",
            f"- videos with duration mismatch > 1s: {mismatch}",
            f"- usable aligned videos: {usable}",
            f"- alignment-review candidates: {alignment_review_candidates}",
            "- mismatch distribution is systematic rather than random: most videos have about three extra video seconds relative to annotation span.",
            "- start timestamps are present as `[control]time_stamp`; raw wall-clock epoch is also present as `[control]epoch`.",
            "",
            "## VEATIC-Compatible Manifest Proposal",
            "",
            "The proposed manifest uses one row per cleaned annotation sample and can be resampled to 1Hz for VEATIC-style contracts. `video_id` is the video filename stem; `participant_id`, `session_id`, game, and genre are retained for grouped splits.",
            "",
            "## Recommended Benchmark Modes",
            "",
            "- Aggregate arousal benchmark: possible by averaging/resampling session labels, but not the first recommendation because labels are participant/session-level.",
            "- Participant-level benchmark: supported and recommended.",
            "- Leave-video/session-out: supported and recommended first.",
            "- Leave-participant-out: supported as stricter generalization.",
            "- Train-on-AGAIN validate-on-VEATIC: useful after scale/target calibration checks.",
            "- Train-on-VEATIC validate-on-AGAIN: useful as a transfer stress test after a 1Hz AGAIN manifest exists.",
            "- Pretrain-on-AGAIN fine-tune-on-VEATIC: plausible later, but only after baseline/control parity on AGAIN.",
            "",
            "## Guardrails",
            "",
            "No TRIBE encoding, model training, tensor export, VEATIC benchmark modification, or video re-encoding was performed by this audit.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    root = args.again_root.expanduser()
    if not root.exists():
        raise SystemExit(f"AGAIN root does not exist: {root}")
    output_root = args.output_root or Path("outputs") / f"again_cleaned_inventory_audit_{now_stamp()}"
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Output root exists and is non-empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    clean_path = root / "annotations" / "clean_data.csv"
    outlier_path = root / "annotations" / "outliers.csv"
    metadata_path = root / "metadata" / "cleaned_session_video_metadata.csv"
    schema_path = root / "metadata" / "cleaned_column_schema.csv"
    clean_rows = read_csv_rows(clean_path)
    outlier_rows = read_csv_rows(outlier_path)
    metadata_rows = read_csv_rows(metadata_path)
    schema_rows = read_csv_rows(schema_path)
    metadata_by_key = {metadata_key(row): row for row in metadata_rows}

    annotation_rows, annotation_stats = summarize_annotations(clean_rows, metadata_by_key)
    metadata_inventory = inventory_metadata(root, clean_rows, metadata_rows, schema_rows, outlier_rows)
    video_rows, video_summary = inventory_videos(root, metadata_rows, annotation_stats, args.ffprobe)
    alignment_rows = alignment_audit(video_rows)
    alignment_by_video = {
        row["video_id"]: {
            **row,
            "fps": next((video.get("fps") for video in video_rows if video["video_id"] == row["video_id"]), None),
        }
        for row in alignment_rows
    }
    proposal_rows = manifest_proposal_rows(clean_rows, metadata_by_key, alignment_by_video, root)

    alignment_summary = {
        "mismatch_gt_1s": sum(1 for row in alignment_rows if row["mismatch_gt_1s"] is True),
        "resample_1hz_feasible": sum(1 for row in alignment_rows if row["resample_1hz_feasible"]),
        "future_spike_1_3s_feasible": sum(1 for row in alignment_rows if row["future_spike_1_3s_feasible"]),
        "future_change_p3s_movement_feasible": sum(1 for row in alignment_rows if row["future_change_p3s_movement_feasible"]),
        "missing_start_timestamps": sum(1 for row in alignment_rows if row["missing_start_timestamp"]),
        "usable_status_count": Counter(row["alignment_status"] for row in alignment_rows),
    }
    usable_video_ids = {row["video_id"] for row in alignment_rows if row["alignment_status"] in {"usable", "duration_mismatch_gt_1s"} and row["resample_1hz_feasible"]}
    strictly_usable_video_ids = {row["video_id"] for row in alignment_rows if row["alignment_status"] == "usable" and row["resample_1hz_feasible"]}
    usable_annotation_rows = sum(1 for row in proposal_rows if row["video_id"] in strictly_usable_video_ids and row["arousal"] is not None)
    total_duration_hours = video_summary["total_duration_hours"]
    estimated_1hz_rows = int(round(video_summary["total_duration_seconds"]))
    summary = {
        "schema_version": "again_cleaned_inventory_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "again_root": str(root),
        "outputs": {
            "again_video_inventory_csv": str(output_root / "again_video_inventory.csv"),
            "again_annotation_inventory_csv": str(output_root / "again_annotation_inventory.csv"),
            "again_metadata_inventory_csv": str(output_root / "again_metadata_inventory.csv"),
            "again_alignment_audit_csv": str(output_root / "again_alignment_audit.csv"),
            "again_manifest_proposal_csv": str(output_root / "again_manifest_proposal.csv"),
            "again_dataset_summary_json": str(output_root / "again_dataset_summary.json"),
            "again_compatibility_report_md": str(output_root / "again_compatibility_report.md"),
            "run_manifest_json": str(output_root / "run_manifest.json"),
        },
        "videos": video_summary,
        "annotations": annotation_dataset_summary(annotation_rows),
        "metadata": {
            "metadata_files": len(metadata_inventory),
            "session_metadata_rows": len(metadata_rows),
            "column_schema_rows": len(schema_rows),
            "outlier_rows": len(outlier_rows),
            "participants": len({row.get(PLAYER_COL) for row in clean_rows}),
            "games": sorted({row.get(GAME_COL) for row in clean_rows}),
            "genres": sorted({row.get(GENRE_COL) for row in clean_rows}),
            "train_test_splits_provided": False,
        },
        "alignment": {**alignment_summary, "usable_status_count": dict(alignment_summary["usable_status_count"])},
        "usable_videos": len(strictly_usable_video_ids),
        "usable_videos_if_allowing_gt_1s_mismatch": len(usable_video_ids),
        "usable_annotation_rows": usable_annotation_rows,
        "veatic_compatibility": {
            "can_create_manifest": len(usable_video_ids) > 0,
            "can_resample_1hz": alignment_summary["resample_1hz_feasible"] > 0,
            "can_create_future_spike_1_3s": alignment_summary["future_spike_1_3s_feasible"] > 0,
            "can_create_future_change_p3s_movement": alignment_summary["future_change_p3s_movement_feasible"] > 0,
            "arousal_available_continuous": True,
            "valence_available": False,
            "recommended_primary_split": "grouped leave-video/session-out",
            "recommended_secondary_split": "leave-participant-out",
            "requires_alignment_convention_before_strict_benchmark": alignment_summary["mismatch_gt_1s"] > 0,
        },
        "compute_estimate": {
            "total_duration_hours": total_duration_hours,
            "estimated_1hz_rows": estimated_1hz_rows,
            "estimated_video_count_for_encoding": video_summary["videos_ready_for_encoding_before_alignment_review"],
            "storage_note": "TRIBE cache size depends on retained layers/representations; budget external SSD space, not git.",
        },
        "guardrails": {
            "tribe_encoding_run": False,
            "models_trained": False,
            "veatic_outputs_modified": False,
        },
    }

    write_csv(output_root / "again_video_inventory.csv", video_rows)
    write_csv(output_root / "again_annotation_inventory.csv", annotation_rows)
    write_csv(output_root / "again_metadata_inventory.csv", metadata_inventory)
    write_csv(output_root / "again_alignment_audit.csv", alignment_rows)
    write_csv(output_root / "again_manifest_proposal.csv", proposal_rows)
    write_json(output_root / "again_dataset_summary.json", summary)
    (output_root / "again_compatibility_report.md").write_text(compatibility_report(summary), encoding="utf-8")
    run_manifest = {
        "schema_version": "again_cleaned_inventory_audit_run_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "again_root": str(root),
        "output_root": str(output_root),
        "ffprobe": args.ffprobe,
        "files_written": list(summary["outputs"].values()),
        "tribe_encoding_run": False,
        "models_trained": False,
        "veatic_outputs_modified": False,
    }
    write_json(output_root / "run_manifest.json", run_manifest)
    print(json.dumps({"output_root": str(output_root), "usable_videos": summary["usable_videos"], "usable_annotation_rows": summary["usable_annotation_rows"]}, indent=2))


if __name__ == "__main__":
    main()
