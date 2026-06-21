#!/usr/bin/env python3
"""Diagnose AGAIN cleaned video-vs-annotation alignment offsets.

This is an audit-only script. It reads the cleaned AGAIN source and the prior
inventory audit, samples lightweight frame diagnostics, compares timing-only
alignment policies, and writes preview artifacts. It does not encode videos,
train models, or create a final benchmark manifest.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_PREVIOUS_AUDIT = Path("outputs/again_cleaned_inventory_audit_20260621_123531")
RANDOM_SEED = 20260621


def default_again_root() -> Path:
    configured = os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT", "").strip()
    if not configured:
        raise EnvironmentError("Set NEURAL_BRIDGE_EXTERNAL_ROOT to the Neural Bridge external assets root.")
    return Path(configured).expanduser() / "data" / "external" / "AGAIN" / "cleaned"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit AGAIN cleaned alignment offsets.")
    parser.add_argument("--again-root", type=Path, default=default_again_root())
    parser.add_argument("--previous-audit-root", type=Path, default=DEFAULT_PREVIOUS_AUDIT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to outputs/again_alignment_offset_diagnosis_<timestamp>.",
    )
    return parser.parse_args()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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
            writer.writerow({key: clean_value(row.get(key, "")) for key in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_value(payload), indent=2, sort_keys=True), encoding="utf-8")


def clean_value(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: clean_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_value(item) for item in value]
    return value


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


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100.0
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def load_previous_outputs(previous_root: Path) -> dict[str, Any]:
    required = {
        "video_inventory": previous_root / "again_video_inventory.csv",
        "annotation_inventory": previous_root / "again_annotation_inventory.csv",
        "alignment_audit": previous_root / "again_alignment_audit.csv",
        "manifest_proposal": previous_root / "again_manifest_proposal.csv",
        "summary": previous_root / "again_dataset_summary.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise SystemExit(f"Missing previous audit files: {missing}")
    return {
        "video_inventory": read_csv_rows(required["video_inventory"]),
        "annotation_inventory": read_csv_rows(required["annotation_inventory"]),
        "alignment_audit": read_csv_rows(required["alignment_audit"]),
        "manifest_proposal_path": required["manifest_proposal"],
        "summary": json.loads(required["summary"].read_text(encoding="utf-8")),
        "paths": {key: str(path) for key, path in required.items()},
    }


def build_distribution_rows(video_rows: list[dict[str, str]], alignment_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    video_by_id = {row["video_id"]: row for row in video_rows}
    rows: list[dict[str, Any]] = []
    mismatches: list[float] = []
    by_game: dict[str, list[float]] = defaultdict(list)
    by_ext: dict[str, list[float]] = defaultdict(list)
    for row in alignment_rows:
        mismatch = safe_float(row.get("duration_mismatch_seconds"))
        video_duration = safe_float(row.get("video_duration_seconds"))
        annotation_duration = safe_float(row.get("annotation_duration_seconds"))
        video = video_by_id.get(row["video_id"], {})
        game = video.get("game", "")
        ext = video.get("extension", "")
        if mismatch is not None:
            mismatches.append(mismatch)
            by_game[game].append(mismatch)
            by_ext[ext].append(mismatch)
        rows.append(
            {
                "video_id": row["video_id"],
                "video_name": row["video_name"],
                "video_path": video.get("video_path", ""),
                "extension": ext,
                "game": game,
                "participant_id": video.get("participant_id", ""),
                "session_id": video.get("session_id", ""),
                "video_duration_seconds": video_duration,
                "annotation_duration_seconds": annotation_duration,
                "video_minus_annotation_seconds": mismatch,
                "abs_mismatch_seconds": abs(mismatch) if mismatch is not None else None,
                "near_plus_3s": abs(mismatch - 3.0) <= 0.5 if mismatch is not None else None,
                "near_minus_3s": abs(mismatch + 3.0) <= 0.5 if mismatch is not None else None,
                "near_0s": abs(mismatch) <= 0.5 if mismatch is not None else None,
            }
        )
    summary = {
        "count": len(mismatches),
        "mean": statistics.mean(mismatches) if mismatches else None,
        "median": statistics.median(mismatches) if mismatches else None,
        "std": statistics.pstdev(mismatches) if len(mismatches) > 1 else 0.0,
        "min": min(mismatches) if mismatches else None,
        "max": max(mismatches) if mismatches else None,
        "percentiles": {f"p{pct}": percentile(mismatches, pct) for pct in [1, 5, 25, 50, 75, 95, 99]},
        "near_plus_3s_count": sum(1 for item in mismatches if abs(item - 3.0) <= 0.5),
        "near_minus_3s_count": sum(1 for item in mismatches if abs(item + 3.0) <= 0.5),
        "near_0s_count": sum(1 for item in mismatches if abs(item) <= 0.5),
        "by_extension": summarize_groups(by_ext),
        "by_game": summarize_groups(by_game),
    }
    return rows, summary


def summarize_groups(groups: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, values in sorted(groups.items()):
        result[key or "unknown"] = {
            "count": len(values),
            "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return result


def select_representative_videos(distribution_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in distribution_rows if row["video_minus_annotation_seconds"] is not None]
    median_mismatch = statistics.median([row["video_minus_annotation_seconds"] for row in valid])
    selected: dict[str, dict[str, Any]] = {}

    def add(rows: list[dict[str, Any]], bucket: str) -> None:
        for row in rows:
            entry = selected.setdefault(row["video_id"], {**row, "selection_reasons": []})
            entry["selection_reasons"].append(bucket)

    add(sorted(valid, key=lambda row: abs(row["video_minus_annotation_seconds"] - median_mismatch))[:10], "closest_to_median")
    add(sorted(valid, key=lambda row: abs(row["video_minus_annotation_seconds"]))[:5], "smallest_mismatch")
    add(sorted(valid, key=lambda row: row["video_minus_annotation_seconds"], reverse=True)[:5], "largest_positive_mismatch")
    negative = [row for row in valid if row["video_minus_annotation_seconds"] < 0]
    add(sorted(negative, key=lambda row: row["video_minus_annotation_seconds"])[:5], "largest_negative_mismatch")
    rng = random.Random(RANDOM_SEED)
    add(rng.sample(valid, k=min(5, len(valid))), "random_seed_20260621")
    ordered = list(selected.values())
    for row in ordered:
        row["selection_reasons"] = "|".join(row["selection_reasons"])
    return sorted(ordered, key=lambda row: (row["selection_reasons"], row["video_id"]))


def sample_frame(video_path: Path, time_seconds: float) -> np.ndarray | None:
    frame = sample_frame_ffmpeg(video_path, time_seconds)
    if frame is not None:
        return frame
    return sample_frame_opencv(video_path, time_seconds)


def sample_frame_ffmpeg(video_path: Path, time_seconds: float) -> np.ndarray | None:
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, time_seconds):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    try:
        proc = subprocess.run(command, check=True, capture_output=True, timeout=20)
        image = Image.open(io.BytesIO(proc.stdout)).convert("RGB")
        return np.array(image)
    except Exception:
        return None


def sample_frame_opencv(video_path: Path, time_seconds: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_seconds) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def frame_metrics(frame: np.ndarray | None, previous: np.ndarray | None) -> dict[str, Any]:
    if frame is None:
        return {
            "frame_readable": False,
            "mean_brightness": None,
            "near_black": None,
            "mean_absdiff_from_previous": None,
        }
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))
    diff = None
    if previous is not None:
        prev = cv2.resize(previous, (frame.shape[1], frame.shape[0]))
        diff = float(np.mean(np.abs(frame.astype(np.float32) - prev.astype(np.float32))))
    return {
        "frame_readable": True,
        "mean_brightness": brightness,
        "near_black": brightness < 10.0,
        "mean_absdiff_from_previous": diff,
    }


def make_contact_sheet(video: dict[str, Any], output_dir: Path) -> tuple[list[dict[str, Any]], Path | None]:
    video_path = Path(str(video["video_path"]))
    duration = safe_float(video.get("video_duration_seconds")) or 0.0
    samples = [
        ("first_frame", 0.0),
        ("t_1s", 1.0),
        ("t_2s", 2.0),
        ("t_3s", 3.0),
        ("t_4s", 4.0),
        ("last_minus_4s", max(0.0, duration - 4.0)),
        ("last_minus_3s", max(0.0, duration - 3.0)),
        ("last_minus_2s", max(0.0, duration - 2.0)),
        ("last_minus_1s", max(0.0, duration - 1.0)),
        ("last_frame", max(0.0, duration - 0.05)),
    ]
    frames: list[tuple[str, float, np.ndarray | None, dict[str, Any]]] = []
    previous: np.ndarray | None = None
    for label, seconds in samples:
        frame = sample_frame(video_path, seconds)
        metrics = frame_metrics(frame, previous)
        frames.append((label, seconds, frame, metrics))
        if frame is not None:
            previous = frame
    sheet_path = output_dir / f"{video['video_id']}.jpg"
    create_sheet(sheet_path, video, frames)
    rows = []
    for label, seconds, _frame, metrics in frames:
        rows.append(
            {
                "video_id": video["video_id"],
                "sample_label": label,
                "sample_time_seconds": seconds,
                **metrics,
                "contact_sheet_path": str(sheet_path),
            }
        )
    return rows, sheet_path


def create_sheet(path: Path, video: dict[str, Any], frames: list[tuple[str, float, np.ndarray | None, dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    thumb_w, thumb_h = 240, 150
    label_h = 42
    cols = 5
    rows = math.ceil(len(frames) / cols)
    header_h = 58
    sheet = Image.new("RGB", (cols * thumb_w, header_h + rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    title = f"{video['video_id']} | mismatch={video['video_minus_annotation_seconds']:.3f}s | {video.get('selection_reasons', '')}"
    draw.text((8, 8), title[:180], fill=(0, 0, 0), font=font)
    draw.text((8, 28), str(video.get("video_path", ""))[:180], fill=(50, 50, 50), font=font)
    for idx, (label, seconds, frame, metrics) in enumerate(frames):
        col = idx % cols
        row = idx // cols
        x = col * thumb_w
        y = header_h + row * (thumb_h + label_h)
        if frame is None:
            thumb = Image.new("RGB", (thumb_w, thumb_h), (30, 30, 30))
        else:
            image = Image.fromarray(frame)
            image.thumbnail((thumb_w, thumb_h))
            thumb = Image.new("RGB", (thumb_w, thumb_h), "black")
            thumb.paste(image, ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2))
        sheet.paste(thumb, (x, y))
        text = f"{label} {seconds:.2f}s\nbright={metrics.get('mean_brightness')}\ndiff={metrics.get('mean_absdiff_from_previous')}"
        draw.text((x + 4, y + thumb_h + 2), text[:80], fill=(0, 0, 0), font=font)
    sheet.save(path, quality=88)


def diagnose_selected_videos(
    selected_rows: list[dict[str, Any]],
    annotation_rows: list[dict[str, str]],
    contact_sheet_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annotation_by_video = {row["video_id"]: row for row in annotation_rows}
    inspection_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for video in selected_rows:
        frame_metrics_rows, sheet_path = make_contact_sheet(video, contact_sheet_dir)
        frame_rows.extend(frame_metrics_rows)
        annotation = annotation_by_video.get(video["video_id"], {})
        first_static = sequence_static_or_black(frame_metrics_rows, ["first_frame", "t_1s", "t_2s", "t_3s"])
        last_static = sequence_static_or_black(frame_metrics_rows, ["last_minus_3s", "last_minus_2s", "last_minus_1s", "last_frame"])
        first_ts = safe_float(annotation.get("min_timestamp_seconds"))
        last_ts = safe_float(annotation.get("max_timestamp_seconds"))
        video_duration = safe_float(video.get("video_duration_seconds"))
        annotation_duration = safe_float(video.get("annotation_duration_seconds"))
        mismatch = safe_float(video.get("video_minus_annotation_seconds"))
        inspection_rows.append(
            {
                **video,
                "contact_sheet_path": str(sheet_path) if sheet_path else "",
                "annotation_first_timestamp": first_ts,
                "annotation_last_timestamp": last_ts,
                "annotation_duration_seconds": annotation_duration,
                "video_duration_seconds": video_duration,
                "video_minus_annotation_seconds": mismatch,
                "annotation_sampling_step_seconds": annotation.get("median_sample_delta_seconds", ""),
                "duplicated_timestamps": annotation.get("duplicated_timestamps", ""),
                "missing_timestamps": annotation.get("missing_timestamps", ""),
                "annotation_starts_at_zero": abs(first_ts or 0.0) <= 0.25,
                "annotation_stops_roughly_3s_before_video_end": abs((mismatch or 0.0) - 3.0) <= 0.75,
                "annotation_start_plus_3_aligns_end": abs(((video_duration or 0.0) - ((annotation_duration or 0.0) + 3.0))) <= 1.0,
                "trim_first_3s_aligns_duration": abs(((video_duration or 0.0) - 3.0) - (annotation_duration or 0.0)) <= 1.0,
                "trim_last_3s_aligns_duration": abs(((video_duration or 0.0) - 3.0) - (annotation_duration or 0.0)) <= 1.0,
                "first_3s_static_or_black": first_static,
                "last_3s_static_or_black": last_static,
                "pre_roll_evidence": first_static and not last_static,
                "post_roll_evidence": last_static and not first_static,
            }
        )
    return inspection_rows, frame_rows


def sequence_static_or_black(frame_rows: list[dict[str, Any]], labels: list[str]) -> bool:
    subset = [row for row in frame_rows if row["sample_label"] in labels and row["frame_readable"]]
    if len(subset) < 3:
        return False
    near_black_count = sum(1 for row in subset if row["near_black"])
    diffs = [safe_float(row.get("mean_absdiff_from_previous")) for row in subset[1:]]
    finite_diffs = [item for item in diffs if item is not None]
    low_motion_count = sum(1 for item in finite_diffs if item < 4.0)
    return near_black_count >= 2 or (finite_diffs and low_motion_count >= max(2, len(finite_diffs) - 1))


def evaluate_policy(policy: str, alignment_rows: list[dict[str, str]]) -> dict[str, Any]:
    residuals: list[float] = []
    usable_rows = 0
    dropped_first_total = 0.0
    dropped_last_total = 0.0
    future_feasible = 0
    for row in alignment_rows:
        video_duration = safe_float(row.get("video_duration_seconds")) or 0.0
        annotation_duration = safe_float(row.get("annotation_duration_seconds")) or 0.0
        mismatch = video_duration - annotation_duration
        dropped_first, dropped_last, aligned_duration, residual = policy_effect(policy, video_duration, annotation_duration)
        residuals.append(residual)
        dropped_first_total += dropped_first
        dropped_last_total += dropped_last
        if aligned_duration >= 6.0:
            future_feasible += 1
        usable_rows += max(0, int(math.floor(aligned_duration)))
    residual_within_1 = sum(1 for item in residuals if abs(item) <= 1.0)
    return {
        "alignment_policy": policy,
        "usable_videos": residual_within_1,
        "usable_1hz_rows_estimate": usable_rows,
        "mean_residual_mismatch_seconds": statistics.mean(residuals) if residuals else None,
        "median_residual_mismatch_seconds": statistics.median(residuals) if residuals else None,
        "max_abs_residual_mismatch_seconds": max(abs(item) for item in residuals) if residuals else None,
        "videos_with_residual_mismatch_lte_1s": residual_within_1,
        "dropped_first_seconds_total": dropped_first_total,
        "dropped_last_seconds_total": dropped_last_total,
        "future_spike_1_3s_target_feasible_videos": future_feasible,
        "future_change_p3s_target_feasible_videos": future_feasible,
        "global_simple": policy in {
            "no_shift_trim_to_min_duration",
            "drop_first_3s_video_align_annotations_to_video_t_minus_3",
            "drop_last_3s_video_keep_annotation_start",
            "annotation_start_plus_3s",
        },
        "per_video": policy.startswith("per_video") or policy == "center_trim_video_to_annotation_duration",
        "risk_of_leakage_or_target_distortion": policy_risk(policy),
        "recommendation_status": policy_status(policy),
    }


def policy_effect(policy: str, video_duration: float, annotation_duration: float) -> tuple[float, float, float, float]:
    mismatch = video_duration - annotation_duration
    if policy == "no_shift_trim_to_min_duration":
        dropped_first = 0.0
        dropped_last = max(0.0, mismatch)
        aligned_duration = min(video_duration, annotation_duration)
        residual = 0.0
    elif policy in {"drop_first_3s_video_align_annotations_to_video_t_minus_3", "annotation_start_plus_3s"}:
        dropped_first = 3.0
        dropped_last = max(0.0, video_duration - dropped_first - annotation_duration)
        aligned_duration = min(max(0.0, video_duration - dropped_first), annotation_duration)
        residual = (video_duration - dropped_first) - annotation_duration
    elif policy == "drop_last_3s_video_keep_annotation_start":
        dropped_first = 0.0
        dropped_last = 3.0
        aligned_duration = min(max(0.0, video_duration - dropped_last), annotation_duration)
        residual = (video_duration - dropped_last) - annotation_duration
    elif policy == "center_trim_video_to_annotation_duration":
        trim = max(0.0, mismatch)
        dropped_first = trim / 2.0
        dropped_last = trim / 2.0
        aligned_duration = annotation_duration
        residual = 0.0
    elif policy == "per_video_end_trim_to_annotation_duration":
        dropped_first = 0.0
        dropped_last = max(0.0, mismatch)
        aligned_duration = annotation_duration
        residual = 0.0
    elif policy == "per_video_start_trim_to_annotation_duration":
        dropped_first = max(0.0, mismatch)
        dropped_last = 0.0
        aligned_duration = annotation_duration
        residual = 0.0
    else:
        raise ValueError(policy)
    return dropped_first, dropped_last, max(0.0, aligned_duration), residual


def policy_risk(policy: str) -> str:
    if policy == "no_shift_trim_to_min_duration":
        return "low leakage risk; may keep pre-roll if mismatch is at video start"
    if policy == "drop_first_3s_video_align_annotations_to_video_t_minus_3":
        return "low leakage risk; target distortion if extra duration is actually post-roll"
    if policy == "drop_last_3s_video_keep_annotation_start":
        return "low leakage risk; target distortion if extra duration is actually pre-roll"
    if policy == "annotation_start_plus_3s":
        return "low leakage risk; equivalent to global start shift and needs pre-roll evidence"
    if policy == "center_trim_video_to_annotation_duration":
        return "low leakage risk but less interpretable; trims both ends per video"
    if policy == "per_video_end_trim_to_annotation_duration":
        return "low leakage risk; per-video timing transform must be documented"
    if policy == "per_video_start_trim_to_annotation_duration":
        return "low leakage risk; per-video timing transform must be documented"
    return "unknown"


def policy_status(policy: str) -> str:
    if policy == "drop_last_3s_video_keep_annotation_start":
        return "primary_candidate_if_post_roll_supported"
    if policy == "drop_first_3s_video_align_annotations_to_video_t_minus_3":
        return "primary_candidate_if_pre_roll_supported"
    if policy == "per_video_end_trim_to_annotation_duration":
        return "fallback_candidate_if_post_roll_supported"
    if policy == "per_video_start_trim_to_annotation_duration":
        return "fallback_candidate_if_pre_roll_supported"
    if policy == "no_shift_trim_to_min_duration":
        return "safe_baseline_preview_only"
    return "diagnostic_only"


def recommend_policy(inspection_rows: list[dict[str, Any]], distribution_summary: dict[str, Any]) -> dict[str, str]:
    post_votes = sum(1 for row in inspection_rows if row.get("post_roll_evidence"))
    pre_votes = sum(1 for row in inspection_rows if row.get("pre_roll_evidence"))
    starts_at_zero = sum(1 for row in inspection_rows if row.get("annotation_starts_at_zero"))
    stops_early = sum(1 for row in inspection_rows if row.get("annotation_stops_roughly_3s_before_video_end"))
    if post_votes > pre_votes:
        likely_cause = "post_roll_or_annotation_end_offset"
        primary = "drop_last_3s_video_keep_annotation_start"
        fallback = "per_video_end_trim_to_annotation_duration"
    elif pre_votes > post_votes:
        likely_cause = "pre_roll_or_annotation_start_offset"
        primary = "drop_first_3s_video_align_annotations_to_video_t_minus_3"
        fallback = "per_video_start_trim_to_annotation_duration"
    elif starts_at_zero >= max(1, len(inspection_rows) // 2) and stops_early >= max(1, len(inspection_rows) // 2):
        likely_cause = "likely_post_roll_or_annotation_end_offset_from_timing"
        primary = "drop_last_3s_video_keep_annotation_start"
        fallback = "per_video_end_trim_to_annotation_duration"
    else:
        likely_cause = "ambiguous_global_duration_offset"
        primary = "no_shift_trim_to_min_duration"
        fallback = "per_video_end_trim_to_annotation_duration"
    return {
        "likely_cause": likely_cause,
        "primary_policy": primary,
        "fallback_policy": fallback,
        "pre_roll_evidence_votes": str(pre_votes),
        "post_roll_evidence_votes": str(post_votes),
        "annotation_starts_at_zero_votes": str(starts_at_zero),
        "annotation_stops_early_votes": str(stops_early),
        "median_mismatch_seconds": str(distribution_summary.get("median")),
    }


def build_manifest_preview(
    manifest_path: Path,
    preview_video_ids: set[str],
    video_by_id: dict[str, dict[str, Any]],
    alignment_policy: str,
) -> list[dict[str, Any]]:
    best_by_video_second: dict[str, dict[int, tuple[float, dict[str, str]]]] = defaultdict(dict)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            video_id = row.get("video_id", "")
            if video_id not in preview_video_ids:
                continue
            annotation_time = safe_float(row.get("time_start_seconds"))
            if annotation_time is None:
                continue
            second = int(round(annotation_time))
            delta = abs(annotation_time - second)
            current = best_by_video_second[video_id].get(second)
            if current is None or delta < current[0]:
                best_by_video_second[video_id][second] = (delta, row)
    preview_rows: list[dict[str, Any]] = []
    for video_id in sorted(best_by_video_second):
        video = video_by_id[video_id]
        video_duration = safe_float(video.get("video_duration_seconds")) or 0.0
        annotation_duration = safe_float(video.get("annotation_duration_seconds")) or 0.0
        dropped_first, dropped_last, aligned_duration, _residual = policy_effect(alignment_policy, video_duration, annotation_duration)
        max_second = int(math.floor(aligned_duration))
        for second in sorted(best_by_video_second[video_id]):
            if second > max_second:
                continue
            row = best_by_video_second[video_id][second][1]
            annotation_time = safe_float(row.get("time_start_seconds")) or 0.0
            original_video_time = annotation_time + dropped_first
            dropped = original_video_time < dropped_first or original_video_time > (video_duration - dropped_last)
            preview_rows.append(
                {
                    "dataset_name": "AGAIN_cleaned",
                    "video_id": video_id,
                    "video_path": row.get("video_path", ""),
                    "original_video_time_seconds": original_video_time,
                    "aligned_time_seconds": annotation_time,
                    "annotation_time_seconds": annotation_time,
                    "arousal": row.get("arousal", ""),
                    "alignment_policy": alignment_policy,
                    "alignment_offset_seconds": dropped_first,
                    "dropped_by_alignment": dropped,
                    "target_feasible_future_spike_1_3s": (annotation_duration - annotation_time) >= 3.0,
                    "target_feasible_future_change_p3s": (annotation_duration - annotation_time) >= 3.0,
                }
            )
    return preview_rows


def build_report(summary: dict[str, Any]) -> str:
    rec = summary["recommendation"]
    policy = summary["recommended_policy_comparison"]
    pilot = summary["first_encoding_pilot_video_ids"]
    return "\n".join(
        [
            "# AGAIN Alignment Offset Diagnosis",
            "",
            "## Executive Answers",
            "",
            f"1. Likely cause of the ~3s mismatch: {rec['likely_cause']}. Visual diagnostics are lightweight heuristics; timing evidence shows annotations start at 0 and usually stop about 3s before video duration.",
            f"2. Systematic enough for a global policy: {'yes' if summary['mismatch_distribution']['near_plus_3s_count'] >= 900 else 'borderline'}; median mismatch is {summary['mismatch_distribution']['median']:.3f}s.",
            f"3. Recommended policy: `{rec['primary_policy']}`. Fallback: `{rec['fallback_policy']}`.",
            f"4. Survivors under recommended policy: {policy['usable_videos']} videos and about {policy['usable_1hz_rows_estimate']} 1Hz rows.",
            f"5. 1Hz arousal rows can be built safely after documenting the alignment policy: {'yes' if policy['usable_videos'] > 0 else 'no'}.",
            f"6. VEATIC-style future spike/change targets can be built after alignment: {'yes' if policy['future_spike_1_3s_target_feasible_videos'] > 0 else 'no'}.",
            "7. Small TRIBE encoding pilot: approved only after manual review of the contact sheets confirms the extra seconds are end padding/post-roll; do not scale to all videos yet.",
            f"8. First encoding pilot videos: {', '.join(pilot)}.",
            "9. Do not run full TRIBE encoding, do not train models, do not create a final manifest, and do not compare to VEATIC until the alignment convention is frozen.",
            "",
            "## Mismatch Distribution",
            "",
            f"- count: {summary['mismatch_distribution']['count']}",
            f"- mean: {summary['mismatch_distribution']['mean']:.3f}s",
            f"- median: {summary['mismatch_distribution']['median']:.3f}s",
            f"- std: {summary['mismatch_distribution']['std']:.3f}s",
            f"- min/max: {summary['mismatch_distribution']['min']:.3f}s / {summary['mismatch_distribution']['max']:.3f}s",
            f"- near +3s: {summary['mismatch_distribution']['near_plus_3s_count']}",
            f"- near -3s: {summary['mismatch_distribution']['near_minus_3s_count']}",
            f"- near 0s: {summary['mismatch_distribution']['near_0s_count']}",
            "",
            "## Visual Diagnostics",
            "",
            f"- representative videos inspected: {summary['representative_video_count']}",
            f"- contact sheets: `{summary['outputs']['contact_sheets_dir']}`",
            f"- pre-roll evidence votes: {rec['pre_roll_evidence_votes']}",
            f"- post-roll evidence votes: {rec['post_roll_evidence_votes']}",
            "",
            "## Policy Comparison",
            "",
            f"The recommended policy is global/simple, uses only video and annotation timing, and does not inspect labels for fitting. The per-video end-trim fallback is more exact but should be used only if the global 3s policy leaves too many residual mismatches after manual review.",
            "",
            "## Guardrails",
            "",
            "tribe_encoding_run=false",
            "models_trained=false",
            "final_manifest_created=false",
            "veatic_outputs_modified=false",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    root = args.again_root.expanduser()
    previous_root = args.previous_audit_root
    if not root.exists():
        raise SystemExit(f"AGAIN root not found: {root}")
    if not previous_root.exists():
        raise SystemExit(f"Previous audit root not found: {previous_root}")
    output_root = args.output_root or Path("outputs") / f"again_alignment_offset_diagnosis_{now_stamp()}"
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Output root exists and is non-empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    contact_sheet_dir = output_root / "contact_sheets"

    previous = load_previous_outputs(previous_root)
    video_rows = previous["video_inventory"]
    annotation_rows = previous["annotation_inventory"]
    alignment_rows = previous["alignment_audit"]
    distribution_rows, distribution_summary = build_distribution_rows(video_rows, alignment_rows)
    selected_rows = select_representative_videos(distribution_rows)
    inspection_rows, frame_rows = diagnose_selected_videos(selected_rows, annotation_rows, contact_sheet_dir)

    policies = [
        "no_shift_trim_to_min_duration",
        "drop_first_3s_video_align_annotations_to_video_t_minus_3",
        "drop_last_3s_video_keep_annotation_start",
        "annotation_start_plus_3s",
        "center_trim_video_to_annotation_duration",
        "per_video_end_trim_to_annotation_duration",
        "per_video_start_trim_to_annotation_duration",
    ]
    policy_rows = [evaluate_policy(policy, alignment_rows) for policy in policies]
    recommendation = recommend_policy(inspection_rows, distribution_summary)
    recommended_policy = recommendation["primary_policy"]
    recommended_policy_row = next(row for row in policy_rows if row["alignment_policy"] == recommended_policy)

    video_by_id = {row["video_id"]: row for row in distribution_rows}
    first_25_video_ids = [row["video_id"] for row in distribution_rows[:25]]
    pilot_ids = [row["video_id"] for row in selected_rows[:25]]
    preview_video_ids = set(first_25_video_ids) | set(pilot_ids)
    manifest_preview_rows = build_manifest_preview(
        previous["manifest_proposal_path"],
        preview_video_ids,
        video_by_id,
        recommended_policy,
    )

    outputs = {
        "again_offset_mismatch_distribution_csv": str(output_root / "again_offset_mismatch_distribution.csv"),
        "again_selected_video_inspection_csv": str(output_root / "again_selected_video_inspection.csv"),
        "again_alignment_policy_comparison_csv": str(output_root / "again_alignment_policy_comparison.csv"),
        "again_aligned_manifest_preview_csv": str(output_root / "again_aligned_manifest_preview.csv"),
        "again_offset_diagnosis_summary_json": str(output_root / "again_offset_diagnosis_summary.json"),
        "again_offset_diagnosis_report_md": str(output_root / "again_offset_diagnosis_report.md"),
        "contact_sheets_dir": str(contact_sheet_dir),
        "run_manifest_json": str(output_root / "run_manifest.json"),
    }
    summary = {
        "schema_version": "again_alignment_offset_diagnosis_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "again_root": str(root),
        "previous_audit_root": str(previous_root),
        "previous_files_loaded": previous["paths"],
        "mismatch_distribution": distribution_summary,
        "representative_video_count": len(selected_rows),
        "recommendation": recommendation,
        "recommended_policy_comparison": recommended_policy_row,
        "fallback_policy_comparison": next(row for row in policy_rows if row["alignment_policy"] == recommendation["fallback_policy"]),
        "first_encoding_pilot_video_ids": pilot_ids[:25],
        "outputs": outputs,
        "guardrails": {
            "tribe_encoding_run": False,
            "models_trained": False,
            "final_manifest_created": False,
            "veatic_outputs_modified": False,
        },
    }

    write_csv(output_root / "again_offset_mismatch_distribution.csv", distribution_rows)
    write_csv(output_root / "again_selected_video_inspection.csv", inspection_rows)
    write_csv(output_root / "again_frame_diagnostics.csv", frame_rows)
    write_csv(output_root / "again_alignment_policy_comparison.csv", policy_rows)
    write_csv(output_root / "again_aligned_manifest_preview.csv", manifest_preview_rows)
    write_json(output_root / "again_offset_diagnosis_summary.json", summary)
    (output_root / "again_offset_diagnosis_report.md").write_text(build_report(summary), encoding="utf-8")
    run_manifest = {
        "schema_version": "again_alignment_offset_diagnosis_run_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "again_root": str(root),
        "previous_audit_root": str(previous_root),
        "output_root": str(output_root),
        "files_written": list(outputs.values()) + [str(output_root / "again_frame_diagnostics.csv")],
        "tribe_encoding_run": False,
        "models_trained": False,
        "final_manifest_created": False,
        "veatic_outputs_modified": False,
    }
    write_json(output_root / "run_manifest.json", run_manifest)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "recommended_alignment_policy": recommended_policy,
                "usable_videos": recommended_policy_row["usable_videos"],
                "usable_1hz_rows_estimate": recommended_policy_row["usable_1hz_rows_estimate"],
                "representative_videos_inspected": len(selected_rows),
                "likely_cause": recommendation["likely_cause"],
                "small_encoding_pilot_approved": "manual_review_required_before_pilot",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
