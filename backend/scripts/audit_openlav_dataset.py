"""Audit OpenLAV files and labels before expensive neuro inference."""

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos-dir",
        default="/Volumes/onn. Drive/Neural Bridge/datasets/openlav_videos",
    )
    parser.add_argument(
        "--tools-dir",
        default="/Volumes/onn. Drive/Neural Bridge/datasets/openlav_tools",
    )
    parser.add_argument("--output", default="benchmarks/openlav/dataset_audit.json")
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir).expanduser().resolve()
    tools_dir = Path(args.tools_dir).expanduser().resolve()
    label_path = tools_dir / "export" / "video_data.csv"
    valid_path = tools_dir / "processed_data" / "valid_data.csv"
    labels = read_rows(label_path)
    valid = read_rows(valid_path)
    videos = sorted(videos_dir.glob("*.webm"))
    video_by_code = {path.stem: path for path in videos}
    label_by_code = {row["video_code"]: row for row in labels}
    missing_videos = sorted(set(label_by_code) - set(video_by_code))
    unlabelled_videos = sorted(set(video_by_code) - set(label_by_code))
    zero_size = [str(path) for path in videos if path.stat().st_size == 0]
    partials = sorted(str(path) for path in videos_dir.glob("*.part"))

    targets = ("valence", "arousal")
    target_summary = {}
    for target in targets:
        values = [float(row[target]) for row in labels if row.get(target, "") != ""]
        target_summary[target] = {
            "count": len(values),
            "missing": len(labels) - len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
        }

    valid_codes = Counter(row.get("video_code", row.get("video", "")) for row in valid)
    valid_codes.pop("", None)
    worker_key = next(
        (key for key in ("worker_id", "worker", "participant", "participant_id") if valid and key in valid[0]),
        None,
    )
    report = {
        "schema_version": "openlav_dataset_audit_v1",
        "ready_for_video_level_calibration": not (
            missing_videos or unlabelled_videos or zero_size or partials
        ),
        "paths": {
            "videos_dir": str(videos_dir),
            "labels": str(label_path),
            "valid_participant_rows": str(valid_path),
        },
        "counts": {
            "videos": len(videos),
            "official_label_rows": len(labels),
            "official_unique_video_codes": len(label_by_code),
            "valid_participant_video_rows": len(valid),
            "valid_participants": len({row[worker_key] for row in valid}) if worker_key else None,
            "source_urls": len({row["source_URL"] for row in labels}),
            "authors": len({row["author"] for row in labels}),
        },
        "ratings_per_video": {
            "min": min(valid_codes.values()) if valid_codes else None,
            "max": max(valid_codes.values()) if valid_codes else None,
            "median": statistics.median(valid_codes.values()) if valid_codes else None,
        },
        "targets": target_summary,
        "integrity": {
            "missing_videos": missing_videos,
            "unlabelled_videos": unlabelled_videos,
            "zero_size_files": zero_size,
            "partial_files": partials,
        },
        "limitations": [
            "OpenLAV validates short video affect; it does not establish text, audio-only, financial, or political generalization.",
            "Participant rows are not independent neuro samples because each video shares one population-average TRIBE response.",
            "Stimulus/source-family grouped holdouts are required to prevent leakage.",
        ],
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["ready_for_video_level_calibration"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
