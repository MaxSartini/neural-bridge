#!/usr/bin/env python3
"""Audit per-video usable boundaries for the cleaned AGAIN dataset.

This is a lightweight visual/timing audit. It does not run TRIBE, train models,
or create a final benchmark manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


DEFAULT_INVENTORY_ROOT = Path("outputs/again_cleaned_inventory_audit_20260621_123531")


def default_again_root() -> Path:
    configured = os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT", "").strip()
    if not configured:
        raise EnvironmentError("Set NEURAL_BRIDGE_EXTERNAL_ROOT to the Neural Bridge external assets root.")
    return Path(configured).expanduser() / "data" / "external" / "AGAIN" / "cleaned"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_frame(cap: cv2.VideoCapture, time_seconds: float, duration_seconds: float) -> tuple[bool, np.ndarray | None]:
    seek_time = min(max(float(time_seconds), 0.0), max(float(duration_seconds) - 0.05, 0.0))
    cap.set(cv2.CAP_PROP_POS_MSEC, seek_time * 1000.0)
    ok, frame = cap.read()
    if ok and frame is not None:
        return True, frame
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(seek_time * fps))))
    ok, frame = cap.read()
    return bool(ok and frame is not None), frame if ok else None


def frame_metrics(frame: np.ndarray | None) -> dict[str, Any]:
    if frame is None:
        return {
            "frame_readable": False,
            "mean_brightness": None,
            "std_brightness": None,
            "dark_pixel_fraction": None,
            "near_black": None,
            "low_detail": None,
            "gray_small": None,
        }
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_AREA)
    mean = float(np.mean(gray_small))
    std = float(np.std(gray_small))
    dark_fraction = float(np.mean(gray_small < 20))
    return {
        "frame_readable": True,
        "mean_brightness": mean,
        "std_brightness": std,
        "dark_pixel_fraction": dark_fraction,
        "near_black": bool(mean < 12.0 or dark_fraction > 0.85),
        "low_detail": bool(std < 5.0),
        "gray_small": gray_small,
    }


def mean_absdiff(previous: np.ndarray | None, current: np.ndarray | None) -> float | None:
    if previous is None or current is None:
        return None
    return float(np.mean(np.abs(current.astype(np.float32) - previous.astype(np.float32))))


def unique_times(times: list[tuple[str, float]], duration_seconds: float) -> list[tuple[str, float]]:
    seen: set[float] = set()
    out: list[tuple[str, float]] = []
    for label, raw_time in times:
        if not math.isfinite(float(raw_time)):
            continue
        clipped = min(max(float(raw_time), 0.0), max(float(duration_seconds) - 0.05, 0.0))
        rounded = round(clipped, 3)
        if rounded in seen:
            continue
        seen.add(rounded)
        out.append((label, rounded))
    return out


def sample_plan(video_duration: float, annotation_end: float) -> list[tuple[str, float]]:
    samples: list[tuple[str, float]] = []
    for t in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
        samples.append((f"start_{t:g}s", t))
    for delta in (-3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0):
        label = "annotation_end" if delta == 0 else f"annotation_end_{delta:+g}s"
        samples.append((label, annotation_end + delta))
    for delta in (-5.0, -4.0, -3.0, -2.0, -1.0, -0.05):
        samples.append((f"video_end_{delta:g}s", video_duration + delta))
    return unique_times(samples, video_duration)


def bool_fraction(values: list[Any]) -> float | None:
    clean = [bool(item) for item in values if item is not None and not pd.isna(item)]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def median_clean(values: list[float | None]) -> float | None:
    clean = [float(item) for item in values if item is not None and math.isfinite(float(item))]
    if not clean:
        return None
    return float(np.median(clean))


def build_contact_sheet(
    *,
    video_id: str,
    sample_rows: list[dict[str, Any]],
    frames: dict[float, np.ndarray],
    output_path: Path,
) -> None:
    tiles = []
    for row in sample_rows:
        t = float(row["sample_time_seconds"])
        frame = frames.get(t)
        if frame is None:
            tile = np.zeros((120, 180, 3), dtype=np.uint8)
        else:
            tile = cv2.resize(frame, (180, 120), interpolation=cv2.INTER_AREA)
        label = f"{row['sample_label']} {t:.1f}s"
        cv2.rectangle(tile, (0, 0), (180, 18), (0, 0, 0), thickness=-1)
        cv2.putText(tile, label[:28], (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
        tiles.append(tile)
    if not tiles:
        return
    cols = 4
    rows = int(math.ceil(len(tiles) / cols))
    blank = np.zeros_like(tiles[0])
    while len(tiles) < rows * cols:
        tiles.append(blank.copy())
    sheet_rows = [np.concatenate(tiles[i * cols : (i + 1) * cols], axis=1) for i in range(rows)]
    sheet = np.concatenate(sheet_rows, axis=0)
    cv2.putText(sheet, video_id[:60], (6, sheet.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def reason_join(reasons: list[str]) -> str:
    return ";".join(sorted(set(reasons)))


def audit_video(row: pd.Series) -> tuple[dict[str, Any], list[dict[str, Any]], dict[float, np.ndarray]]:
    video_id = str(row["video_id"])
    video_path = str(row["video_path"])
    duration = float(row["duration_seconds"])
    ann_start = float(row["min_timestamp_seconds"])
    ann_end = float(row["max_timestamp_seconds"])
    mismatch = duration - ann_end
    samples = sample_plan(duration, ann_end)

    cap = cv2.VideoCapture(video_path)
    sample_rows: list[dict[str, Any]] = []
    frames_for_sheet: dict[float, np.ndarray] = {}
    previous_gray_by_zone: dict[str, np.ndarray | None] = defaultdict(lambda: None)
    for label, sample_time in samples:
        ok, frame = read_frame(cap, sample_time, duration)
        metrics = frame_metrics(frame if ok else None)
        zone = label.split("_")[0]
        previous_gray = previous_gray_by_zone[zone]
        diff = mean_absdiff(previous_gray, metrics.get("gray_small"))
        if metrics.get("gray_small") is not None:
            previous_gray_by_zone[zone] = metrics["gray_small"]
        if frame is not None:
            frames_for_sheet[sample_time] = frame
        sample_rows.append(
            {
                "video_id": video_id,
                "video_path": video_path,
                "sample_label": label,
                "sample_time_seconds": sample_time,
                "relative_to_annotation_end_seconds": sample_time - ann_end,
                "relative_to_video_end_seconds": sample_time - duration,
                "frame_readable": metrics["frame_readable"],
                "mean_brightness": metrics["mean_brightness"],
                "std_brightness": metrics["std_brightness"],
                "dark_pixel_fraction": metrics["dark_pixel_fraction"],
                "near_black": metrics["near_black"],
                "low_detail": metrics["low_detail"],
                "mean_absdiff_from_previous_in_zone": diff,
            }
        )
    cap.release()

    start_rows = [item for item in sample_rows if item["sample_time_seconds"] <= 3.05]
    post_annotation_rows = [item for item in sample_rows if item["sample_time_seconds"] > ann_end + 0.25]
    tail_rows = [item for item in sample_rows if item["sample_time_seconds"] >= duration - 3.05]

    start_black_fraction = bool_fraction([item["near_black"] for item in start_rows])
    start_low_motion = median_clean([item["mean_absdiff_from_previous_in_zone"] for item in start_rows[1:]])
    post_black_fraction = bool_fraction([item["near_black"] for item in post_annotation_rows])
    post_low_motion = median_clean([item["mean_absdiff_from_previous_in_zone"] for item in post_annotation_rows[1:]])
    tail_black_fraction = bool_fraction([item["near_black"] for item in tail_rows])
    tail_low_motion = median_clean([item["mean_absdiff_from_previous_in_zone"] for item in tail_rows[1:]])

    start_nuisance = bool((start_black_fraction or 0.0) >= 0.75 or (start_low_motion is not None and start_low_motion < 1.0))
    post_annotation_black_or_static = bool(
        (post_black_fraction is not None and post_black_fraction >= 0.5)
        or (post_low_motion is not None and post_low_motion < 1.0)
        or (tail_black_fraction is not None and tail_black_fraction >= 0.5)
        or (tail_low_motion is not None and tail_low_motion < 1.0)
    )
    post_annotation_dynamic = bool(
        post_annotation_rows
        and not post_annotation_black_or_static
        and any(
            item["frame_readable"]
            and item["near_black"] is False
            and (item["mean_absdiff_from_previous_in_zone"] is None or item["mean_absdiff_from_previous_in_zone"] > 3.0)
            for item in post_annotation_rows
        )
    )

    reasons: list[str] = []
    if ann_start > 0.5:
        reasons.append("annotation_starts_late")
    if start_nuisance:
        reasons.append("first_3s_visual_nuisance_candidate")
    if mismatch > 1.0:
        reasons.append("video_extends_beyond_annotation")
    if post_annotation_black_or_static:
        reasons.append("post_annotation_tail_black_or_static")
    if post_annotation_dynamic:
        reasons.append("post_annotation_tail_dynamic_manual_review")
    if not post_annotation_black_or_static and not post_annotation_dynamic:
        reasons.append("post_annotation_tail_inconclusive")
    if mismatch < 0:
        reasons.append("annotation_exceeds_video")
    if abs(mismatch - 3.0) > 0.5:
        reasons.append("duration_mismatch_outlier")

    recommended_start = ann_start if ann_start > 0.5 else 0.0
    recommended_end = min(duration, ann_end)
    target_safe_end = max(recommended_start, recommended_end - 3.0)
    one_hz_start = int(math.ceil(recommended_start))
    one_hz_end = int(math.floor(recommended_end))
    one_hz_target_end = int(math.floor(target_safe_end))
    usable_rows_1hz = max(0, one_hz_end - one_hz_start + 1)
    target_rows_1hz = max(0, one_hz_target_end - one_hz_start + 1)

    recommendation = {
        "dataset_name": "AGAIN_cleaned",
        "video_id": video_id,
        "video_path": video_path,
        "game": row.get("game", ""),
        "participant_id": row.get("participant_id", ""),
        "session_id": row.get("session_id", ""),
        "video_duration_seconds": duration,
        "annotation_start_seconds": ann_start,
        "annotation_end_seconds": ann_end,
        "video_minus_annotation_end_seconds": mismatch,
        "recommended_encode_start_seconds": recommended_start,
        "recommended_encode_end_seconds": recommended_end,
        "recommended_benchmark_start_seconds": recommended_start,
        "recommended_benchmark_end_seconds": recommended_end,
        "target_safe_end_future_1_3s_seconds": target_safe_end,
        "trim_start_seconds": recommended_start,
        "trim_end_seconds": max(0.0, duration - recommended_end),
        "one_hz_grid_start_second": one_hz_start,
        "one_hz_grid_end_second": one_hz_end,
        "one_hz_target_safe_end_second": one_hz_target_end,
        "usable_1hz_rows_estimate": usable_rows_1hz,
        "target_feasible_1hz_rows_estimate": target_rows_1hz,
        "start_black_fraction_first_3s": start_black_fraction,
        "start_median_absdiff_first_3s": start_low_motion,
        "tail_black_fraction_last_3s": tail_black_fraction,
        "tail_median_absdiff_last_3s": tail_low_motion,
        "post_annotation_black_fraction": post_black_fraction,
        "post_annotation_median_absdiff": post_low_motion,
        "start_visual_nuisance_candidate": start_nuisance,
        "post_annotation_tail_black_or_static": post_annotation_black_or_static,
        "post_annotation_tail_dynamic_manual_review": post_annotation_dynamic,
        "boundary_confidence": (
            "high" if mismatch > 1 and ann_start <= 0.5 and post_annotation_black_or_static
            else "medium" if mismatch > 1 and ann_start <= 0.5
            else "review"
        ),
        "recommended_policy": "use_annotation_covered_video_time_only",
        "notes": reason_join(reasons),
    }
    return recommendation, sample_rows, frames_for_sheet


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--again-root", type=Path, default=default_again_root())
    parser.add_argument("--inventory-root", type=Path, default=DEFAULT_INVENTORY_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-contact-sheets", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    output_root = args.output_root or Path(f"outputs/again_video_boundary_audit_{timestamp()}")
    output_root.mkdir(parents=True, exist_ok=False)

    video_inventory = pd.read_csv(args.inventory_root / "again_video_inventory.csv")
    annotation_inventory = pd.read_csv(args.inventory_root / "again_annotation_inventory.csv")
    merged = video_inventory.merge(
        annotation_inventory[
            [
                "video_id",
                "min_timestamp_seconds",
                "max_timestamp_seconds",
                "annotation_duration_seconds",
                "rows",
                "duplicated_timestamps",
                "missing_timestamps",
            ]
        ],
        on="video_id",
        how="left",
        suffixes=("", "_annotation_inventory"),
    )
    merged = merged[(merged["readable"] == True) & (merged["annotation_present"] == True)].copy()  # noqa: E712
    merged = merged.sort_values(["video_id"]).reset_index(drop=True)
    if args.limit:
        merged = merged.head(args.limit)

    recommendations: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    contact_candidates: list[tuple[int, dict[str, Any], list[dict[str, Any]], dict[float, np.ndarray]]] = []

    for index, (_, row) in enumerate(merged.iterrows(), start=1):
        rec, sample_rows, frames = audit_video(row)
        recommendations.append(rec)
        samples.extend(sample_rows)
        risk = 0
        if rec["boundary_confidence"] != "high":
            risk += 3
        if rec["post_annotation_tail_dynamic_manual_review"]:
            risk += 5
        if rec["start_visual_nuisance_candidate"]:
            risk += 2
        if "duration_mismatch_outlier" in str(rec["notes"]):
            risk += 2
        if risk:
            contact_candidates.append((risk, rec, sample_rows, frames))
        if index % 50 == 0 or index == len(merged):
            print(json.dumps({"progress": f"{index}/{len(merged)}", "video_id": rec["video_id"]}), flush=True)

    contact_root = output_root / "flagged_contact_sheets"
    for _, rec, sample_rows, frames in sorted(contact_candidates, key=lambda item: (-item[0], item[1]["video_id"]))[
        : max(0, args.max_contact_sheets)
    ]:
        safe_id = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in rec["video_id"])
        sheet_path = contact_root / f"{safe_id}.jpg"
        build_contact_sheet(video_id=rec["video_id"], sample_rows=sample_rows, frames=frames, output_path=sheet_path)
        rec["flagged_contact_sheet_path"] = str(sheet_path)

    recommendation_fields = list(recommendations[0].keys()) if recommendations else []
    if "flagged_contact_sheet_path" not in recommendation_fields:
        recommendation_fields.append("flagged_contact_sheet_path")
    write_csv(output_root / "again_video_boundary_recommendations.csv", recommendations, recommendation_fields)
    write_csv(output_root / "again_video_boundary_frame_samples.csv", samples)

    reason_counts = Counter()
    for rec in recommendations:
        for reason in str(rec["notes"]).split(";"):
            if reason:
                reason_counts[reason] += 1
    confidence_counts = Counter(str(rec["boundary_confidence"]) for rec in recommendations)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "again_root": str(args.again_root),
        "inventory_root": str(args.inventory_root),
        "output_root": str(output_root),
        "guardrails": {
            "tribe_encoding_run": False,
            "models_trained": False,
            "final_manifest_created": False,
            "veatic_outputs_modified": False,
        },
        "videos_audited": len(recommendations),
        "boundary_confidence_counts": dict(confidence_counts),
        "reason_counts": dict(reason_counts),
        "recommended_policy": "use_annotation_covered_video_time_only",
        "mean_trim_end_seconds": float(np.mean([rec["trim_end_seconds"] for rec in recommendations])),
        "median_trim_end_seconds": float(np.median([rec["trim_end_seconds"] for rec in recommendations])),
        "total_usable_1hz_rows_estimate": int(sum(rec["usable_1hz_rows_estimate"] for rec in recommendations)),
        "total_target_feasible_1hz_rows_estimate": int(
            sum(rec["target_feasible_1hz_rows_estimate"] for rec in recommendations)
        ),
        "contact_sheets_written": len(list(contact_root.glob("*.jpg"))) if contact_root.exists() else 0,
    }
    (output_root / "again_video_boundary_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "script": "tools/audit_again_video_boundaries.py",
                "created_at": summary["created_at"],
                "inputs": {
                    "again_root": str(args.again_root),
                    "inventory_root": str(args.inventory_root),
                },
                "outputs": {
                    "recommendations": str(output_root / "again_video_boundary_recommendations.csv"),
                    "frame_samples": str(output_root / "again_video_boundary_frame_samples.csv"),
                    "summary": str(output_root / "again_video_boundary_summary.json"),
                    "report": str(output_root / "again_video_boundary_report.md"),
                },
                "guardrails": summary["guardrails"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report = [
        "# AGAIN Video Boundary Audit",
        "",
        "## Executive Answer",
        "",
        "Use annotation-covered video time only. Do not assume the final seconds are always black; "
        "end each video at its annotation end timestamp and flag any unannotated dynamic tail for review.",
        "",
        "## Summary",
        "",
        f"- videos audited: `{summary['videos_audited']}`",
        f"- boundary confidence counts: `{summary['boundary_confidence_counts']}`",
        f"- median end trim: `{summary['median_trim_end_seconds']:.3f}s`",
        f"- mean end trim: `{summary['mean_trim_end_seconds']:.3f}s`",
        f"- usable 1Hz rows estimate: `{summary['total_usable_1hz_rows_estimate']}`",
        f"- target-feasible 1Hz rows estimate: `{summary['total_target_feasible_1hz_rows_estimate']}`",
        f"- flagged contact sheets written: `{summary['contact_sheets_written']}`",
        "",
        "## Policy",
        "",
        "- `recommended_encode_start_seconds`: annotation start, rounded down to 0 when the first timestamp is within 0.5s of zero.",
        "- `recommended_encode_end_seconds`: annotation end timestamp.",
        "- `target_safe_end_future_1_3s_seconds`: annotation end minus 3s.",
        "- Post-annotation video frames are not used for benchmark rows even when they contain visible content.",
        "- Beginning frames are not trimmed automatically; videos with likely first-3s nuisance content are flagged for manual review.",
        "",
        "## Reason Counts",
        "",
    ]
    for reason, count in sorted(reason_counts.items()):
        report.append(f"- `{reason}`: `{count}`")
    report.extend(
        [
            "",
            "## Guardrails",
            "",
            "- tribe_encoding_run=false",
            "- models_trained=false",
            "- final_manifest_created=false",
            "- veatic_outputs_modified=false",
        ]
    )
    (output_root / "again_video_boundary_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
