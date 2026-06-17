"""Build a validated VEATIC temporal benchmark manifest from a local dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = Path(os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT", str(ROOT / "external_assets"))).expanduser()
DEFAULT_ROOT = str(EXTERNAL_ROOT / "datasets" / "veatic")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi")


def run_ffprobe(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    info = json.loads(result.stdout)
    stream = (info.get("streams") or [{}])[0]
    fmt = info.get("format") or {}
    duration = stream.get("duration") or fmt.get("duration")
    return {
        "duration_seconds": float(duration) if duration not in (None, "N/A") else None,
        "nb_frames": int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else None,
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "r_frame_rate": stream.get("r_frame_rate"),
    }


def parse_rate(rate: str | None) -> float | None:
    if not rate or rate == "0/0":
        return None
    if "/" in rate:
        num, den = rate.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f else None
    return float(rate)


def load_rating(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {path}")
    return np.asarray(data[:, 1], dtype=np.float32)


def find_video(video_dir: Path, video_id: str) -> Path | None:
    for extension in VIDEO_EXTENSIONS:
        candidate = video_dir / f"{video_id}{extension}"
        if candidate.exists():
            return candidate
    return None


def row_indices(length: int, fps: float, sample_hz: float | None) -> list[int]:
    if sample_hz is None or sample_hz <= 0:
        return list(range(length))
    step = max(1, int(round(fps / sample_hz)))
    return list(range(0, length, step))


def official_split(frame_index: int, length: int, split_fraction: float) -> str:
    train_cutoff = int(length * split_fraction)
    return "train" if frame_index < train_cutoff else "test"


def blocked_gap_split(
    frame_index: int,
    length: int,
    train_fraction: float,
    gap_fraction: float,
) -> str:
    train_end = int(length * train_fraction)
    gap_end = min(length, int(length * (train_fraction + gap_fraction)))
    if frame_index < train_end:
        return "train"
    if frame_index < gap_end:
        return "gap"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--output", default="benchmarks/veatic/veatic_manifest.jsonl")
    parser.add_argument("--report", default="benchmarks/veatic/veatic_manifest.report.json")
    parser.add_argument("--sample-hz", type=float, default=1.0)
    parser.add_argument("--split", type=float, default=0.7, help="Official VEATIC train fraction.")
    parser.add_argument("--blocked-train-fraction", type=float, default=0.6)
    parser.add_argument("--blocked-gap-fraction", type=float, default=0.1)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    expected_video_dir = root / "video"
    alternate_video_dir = root / "videos"
    video_dir = expected_video_dir if expected_video_dir.exists() else alternate_video_dir
    rating_dir = root / "rating_averaged"
    output = Path(args.output).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    folder_structure = {
        "root_exists": root.exists(),
        "expected_video_dir": str(expected_video_dir),
        "expected_video_dir_exists": expected_video_dir.exists(),
        "alternate_video_dir": str(alternate_video_dir),
        "alternate_video_dir_exists": alternate_video_dir.exists(),
        "selected_video_dir": str(video_dir),
        "selected_video_dir_exists": video_dir.exists(),
        "rating_averaged_dir_exists": rating_dir.exists(),
    }
    valence_files = sorted(rating_dir.glob("*_valence.csv"))
    if not valence_files:
        raise FileNotFoundError(f"No VEATIC valence files found under {rating_dir}")

    rows: list[dict[str, Any]] = []
    videos: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for valence_path in valence_files:
        video_id = valence_path.name.removesuffix("_valence.csv")
        arousal_path = rating_dir / f"{video_id}_arousal.csv"
        video_path = find_video(video_dir, video_id)
        if not arousal_path.exists() or video_path is None:
            rejected.append(
                {
                    "video_id": video_id,
                    "reason": "missing_arousal_or_video",
                    "valence_path": str(valence_path),
                    "arousal_path": str(arousal_path),
                    "video_path": str(video_path) if video_path else None,
                }
            )
            continue

        valence = load_rating(valence_path)
        arousal = load_rating(arousal_path)
        if len(valence) != len(arousal):
            rejected.append({"video_id": video_id, "reason": "valence_arousal_length_mismatch"})
            continue

        media = run_ffprobe(video_path)
        fps = parse_rate(media.get("avg_frame_rate")) or parse_rate(media.get("r_frame_rate"))
        if not fps or not math.isfinite(fps):
            rejected.append({"video_id": video_id, "reason": "unknown_fps", "media": media})
            continue
        expected_duration = len(valence) / fps
        actual_duration = media.get("duration_seconds")
        duration_delta = None if actual_duration is None else abs(actual_duration - expected_duration)
        nb_frames_delta = None
        if media.get("nb_frames") is not None:
            nb_frames_delta = int(media["nb_frames"]) - int(len(valence))
        alignment_status = "ok"
        if actual_duration is not None and duration_delta is not None and duration_delta > max(1.0 / fps * 3.0, 0.15):
            alignment_status = "duration_mismatch"
            if args.strict:
                rejected.append(
                    {
                        "video_id": video_id,
                        "reason": alignment_status,
                        "actual_duration_seconds": actual_duration,
                        "expected_duration_seconds": expected_duration,
                        "duration_delta_seconds": duration_delta,
                    }
                )
                continue

        selected = row_indices(len(valence), fps, args.sample_hz)
        for frame_index in selected:
            row_official_split = official_split(frame_index, len(valence), args.split)
            row_blocked_split = blocked_gap_split(
                frame_index,
                len(valence),
                args.blocked_train_fraction,
                args.blocked_gap_fraction,
            )
            rows.append(
                {
                    "schema_version": "veatic_temporal_window_v1",
                    "dataset": "veatic",
                    "stimulus_id": f"{video_id}:{frame_index:06d}",
                    "video_id": video_id,
                    "frame_index": int(frame_index),
                    "time_start_seconds": float(frame_index / fps),
                    "time_end_seconds": float((frame_index + 1) / fps),
                    "sampling_frequency_hz": float(args.sample_hz) if args.sample_hz else float(fps),
                    "split": row_official_split,
                    "splits": {
                        "official_70_30": row_official_split,
                        "blocked_temporal_gap": row_blocked_split,
                        "leave_video_out_group": video_id,
                    },
                    "targets": {
                        "valence": float(valence[frame_index]),
                        "arousal": float(arousal[frame_index]),
                    },
                    "media_path": str(video_path),
                    "source_annotation": {
                        "valence": str(valence_path),
                        "arousal": str(arousal_path),
                    },
                }
            )
        videos.append(
            {
                "video_id": video_id,
                "frames": int(len(valence)),
                "fps": fps,
                "duration_seconds": actual_duration,
                "expected_duration_seconds_from_labels": expected_duration,
                "duration_delta_seconds": duration_delta,
                "ffprobe_nb_frames": media.get("nb_frames"),
                "label_frames": int(len(valence)),
                "nb_frames_delta": nb_frames_delta,
                "alignment_status": alignment_status,
                "manifest_rows": len(selected),
                "official_split_counts": {
                    "train": sum(1 for frame_index in selected if official_split(frame_index, len(valence), args.split) == "train"),
                    "test": sum(1 for frame_index in selected if official_split(frame_index, len(valence), args.split) == "test"),
                },
                "blocked_gap_split_counts": {
                    "train": sum(
                        1
                        for frame_index in selected
                        if blocked_gap_split(
                            frame_index,
                            len(valence),
                            args.blocked_train_fraction,
                            args.blocked_gap_fraction,
                        )
                        == "train"
                    ),
                    "gap": sum(
                        1
                        for frame_index in selected
                        if blocked_gap_split(
                            frame_index,
                            len(valence),
                            args.blocked_train_fraction,
                            args.blocked_gap_fraction,
                        )
                        == "gap"
                    ),
                    "test": sum(
                        1
                        for frame_index in selected
                        if blocked_gap_split(
                            frame_index,
                            len(valence),
                            args.blocked_train_fraction,
                            args.blocked_gap_fraction,
                        )
                        == "test"
                    ),
                },
                "media_path": str(video_path),
            }
        )

    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = {
        "schema_version": "veatic_manifest_report_v1",
        "root": str(root),
        "video_dir": str(video_dir),
        "rating_dir": str(rating_dir),
        "folder_structure": folder_structure,
        "output": str(output),
        "sample_hz": args.sample_hz,
        "split_contract": {
            "mode_a_official_veatic": {
                "field": "splits.official_70_30",
                "train_fraction": args.split,
                "purpose": "Comparable to the official VEATIC first-70-percent / last-30-percent frame protocol.",
            },
            "mode_b_blocked_temporal_gap": {
                "field": "splits.blocked_temporal_gap",
                "train_fraction": args.blocked_train_fraction,
                "gap_fraction": args.blocked_gap_fraction,
                "purpose": "Tests temporal tracking while avoiding adjacent-frame leakage through a middle buffer.",
            },
            "mode_c_leave_video_out": {
                "field": "splits.leave_video_out_group",
                "purpose": "Tests generalization to unseen videos by holding out whole video IDs.",
            },
        },
        "valid_videos": len(videos),
        "rejected_videos": len(rejected),
        "rows": len(rows),
        "videos": videos,
        "rejected": rejected,
        "strict_duration_validation": bool(args.strict),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(output), "report": str(report_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
