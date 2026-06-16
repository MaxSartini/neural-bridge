"""Parallel lightweight OpenLAV preflight audit.

This intentionally avoids TRIBE, MLX, V-JEPA, and LLM inference. It is safe to
run while downloads or other lightweight reporting tasks are active.
"""

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=width,height,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return {"error": result.stderr.strip() or f"ffprobe exited {result.returncode}"}
    return json.loads(result.stdout)


def audit_video(video: Path) -> dict[str, Any]:
    info = ffprobe(video)
    row: dict[str, Any] = {
        "stimulus_id": video.stem,
        "path": str(video),
        "bytes": video.stat().st_size,
        "sha256": sha256(video),
        "ffprobe": info,
    }
    if "format" in info:
        row["duration_seconds"] = float(info["format"].get("duration", 0.0))
    return row


def audit_status(status_path: Path) -> dict[str, Any]:
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"stimulus_id": status_path.parent.name, "status_error": str(exc)}
    contract = status.get("model_contract", {})
    return {
        "stimulus_id": status_path.parent.name,
        "complete": bool(status.get("complete")),
        "error": status.get("error"),
        "video_num_frames": contract.get("video_num_frames"),
        "video_extraction_contract": contract.get("video_extraction_contract"),
        "feature_schema": _feature_schema(status_path.parent),
    }


def _feature_schema(cache_dir: Path) -> str | None:
    metadata = cache_dir / "neuro_response_ir.json"
    if not metadata.exists():
        return None
    try:
        return json.loads(metadata.read_text(encoding="utf-8")).get("feature_contract", {}).get("schema_version")
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", default="/Volumes/onn. Drive/Neural Bridge/datasets/openlav_videos")
    parser.add_argument("--cache-dir", default="/Volumes/onn. Drive/Neural Bridge/benchmarks/openlav/tribe_cache")
    parser.add_argument("--output", default="benchmarks/openlav/preflight_audit.json")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--expected-video-frames", type=int, default=64)
    parser.add_argument(
        "--expected-extraction-contract",
        default="official_64_frame_exact_chunked_attention",
    )
    parser.add_argument("--expected-feature-schema", default="neuro_calibration_features_v2")
    args = parser.parse_args()

    videos = sorted(Path(args.videos_dir).expanduser().glob("*.webm"))
    statuses = sorted(Path(args.cache_dir).expanduser().glob("*/cache_status.json"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        video_rows = list(pool.map(audit_video, videos))
        status_rows = list(pool.map(audit_status, statuses))

    current_contract = [
        row for row in status_rows
        if row.get("video_num_frames") == args.expected_video_frames
        and row.get("video_extraction_contract") == args.expected_extraction_contract
        and row.get("feature_schema") in {args.expected_feature_schema, None}
    ]
    complete_current = [
        row for row in current_contract
        if row.get("complete")
        and row.get("feature_schema") == args.expected_feature_schema
    ]
    incomplete_current = [
        row for row in current_contract
        if not row.get("complete")
        or row.get("feature_schema") != args.expected_feature_schema
    ]
    legacy_rows = [
        row for row in status_rows
        if row not in current_contract
    ]
    report = {
        "schema_version": "openlav_preflight_audit_v1",
        "expected_contract": {
            "video_num_frames": args.expected_video_frames,
            "video_extraction_contract": args.expected_extraction_contract,
            "feature_schema": args.expected_feature_schema,
        },
        "videos": {
            "count": len(video_rows),
            "zero_byte": [row["stimulus_id"] for row in video_rows if row["bytes"] == 0],
            "ffprobe_errors": [row for row in video_rows if "error" in row.get("ffprobe", {})],
            "duration_seconds": {
                "min": min((row.get("duration_seconds", 0.0) for row in video_rows), default=0.0),
                "max": max((row.get("duration_seconds", 0.0) for row in video_rows), default=0.0),
            },
        },
        "cache": {
            "status_files": len(status_rows),
            "complete": sum(1 for row in status_rows if row.get("complete")),
            "current_contract_rows": len(current_contract),
            "complete_current_contract": len(complete_current),
            "incomplete_current_contract": incomplete_current,
            "legacy_or_rejected_rows": len(legacy_rows),
            "legacy_or_rejected_contracts": sorted({
                f"{row.get('video_num_frames')}::{row.get('video_extraction_contract')}::{row.get('feature_schema')}"
                for row in legacy_rows
            }),
            "incomplete": [row for row in status_rows if not row.get("complete")],
            "contracts": sorted({
                f"{row.get('video_num_frames')}::{row.get('video_extraction_contract')}::{row.get('feature_schema')}"
                for row in status_rows
            }),
        },
        "parallelism_policy": {
            "safe_parallel": ["hashing", "ffprobe metadata", "status parsing", "manifest/report generation"],
            "unsafe_parallel": ["cortical V-JEPA", "subcortical V-JEPA", "large MLX/Qwen inference"],
        },
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
