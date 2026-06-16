"""Wait for the OpenLAV cache to complete, then build and score the benchmark."""

import argparse
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "backend" / ".venv" / "bin" / "python"
CACHE = Path("/Volumes/onn. Drive/Neural Bridge/benchmarks/openlav/tribe_cache")
STATUS = ROOT / "benchmarks" / "openlav" / "full_run_status.json"
EXPECTED_VIDEO_FRAMES = 64
EXPECTED_EXTRACTION_CONTRACT = "official_64_frame_exact_chunked_attention"
EXPECTED_FEATURE_SCHEMA = "neuro_calibration_features_v2"


def feature_schema(cache_dir: Path) -> str | None:
    metadata = cache_dir / "neuro_response_ir.json"
    if not metadata.exists():
        return None
    try:
        return json.loads(metadata.read_text(encoding="utf-8")).get("feature_contract", {}).get("schema_version")
    except (OSError, json.JSONDecodeError):
        return None


def is_current_complete(status_path: Path) -> bool:
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    contract = status.get("model_contract", {})
    return (
        bool(status.get("complete"))
        and contract.get("video_num_frames") == EXPECTED_VIDEO_FRAMES
        and contract.get("video_extraction_contract") == EXPECTED_EXTRACTION_CONTRACT
        and feature_schema(status_path.parent) == EXPECTED_FEATURE_SCHEMA
    )


def completed_count() -> int:
    complete = 0
    for path in CACHE.glob("*/cache_status.json"):
        complete += is_current_complete(path)
    return complete


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=int, default=188)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=18.0)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout_hours * 3600
    while True:
        complete = completed_count()
        status = {
            "complete_cached_stimuli": complete,
            "completion_contract": {
                "video_num_frames": EXPECTED_VIDEO_FRAMES,
                "video_extraction_contract": EXPECTED_EXTRACTION_CONTRACT,
                "feature_schema": EXPECTED_FEATURE_SCHEMA,
            },
            "expected_stimuli": args.expected,
            "ready_to_finalize": complete >= args.expected,
            "updated_unix_seconds": time.time(),
        }
        STATUS.parent.mkdir(parents=True, exist_ok=True)
        STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status), flush=True)
        if complete >= args.expected:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Only {complete}/{args.expected} stimuli completed before timeout")
        time.sleep(args.poll_seconds)

    manifest = ROOT / "benchmarks" / "openlav" / "calibration_manifest.jsonl"
    run(
        [
            str(PYTHON),
            str(ROOT / "backend" / "scripts" / "build_openlav_calibration_manifest.py"),
            "--require-complete",
        ]
    )
    run(
        [
            str(PYTHON),
            str(ROOT / "backend" / "scripts" / "run_openlav_benchmark.py"),
            str(manifest),
        ]
    )
    status["benchmark_complete"] = True
    status["benchmark_output"] = str(ROOT / "benchmarks" / "openlav" / "openlav_benchmark.json")
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
