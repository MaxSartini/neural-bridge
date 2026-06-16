"""Automatically fill missing VEATIC cortical_fast_default TRIBE cache entries.

This controller intentionally runs one extraction worker at a time.  If another
run_veatic_tribe_cache.py process is already active, it waits for that process
to finish before auditing cache coverage and launching the next missing batch.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "backend" / ".venv" / "bin" / "python"
RUN_SCRIPT = ROOT / "backend" / "scripts" / "run_veatic_tribe_cache.py"


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def audit_cache(report_path: Path, cache_root: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    missing: list[str] = []
    corrupt: list[str] = []
    incomplete: list[str] = []
    valid = 0

    videos = sorted(
        report["videos"],
        key=lambda item: (float(item["duration_seconds"]), int(item["video_id"])),
    )
    for video in videos:
        video_id = str(video["video_id"])
        raw_path = cache_root / video_id / "tribe_raw_output.npz"
        status_path = cache_root / video_id / "cache_status.json"
        raw_ok = False
        marked_complete = False

        if raw_path.exists():
            try:
                with np.load(raw_path) as bundle:
                    raw_ok = (
                        "predictions" in bundle.files
                        and bundle["predictions"].ndim == 2
                        and bundle["predictions"].shape[1] == 20484
                    )
            except Exception:
                corrupt.append(video_id)

        if status_path.exists():
            try:
                marked_complete = bool(json.loads(status_path.read_text(encoding="utf-8")).get("complete"))
            except Exception:
                marked_complete = False

        if raw_ok and marked_complete:
            valid += 1
            continue

        missing.append(video_id)
        if raw_path.exists() and not raw_ok and video_id not in corrupt:
            corrupt.append(video_id)
        elif raw_path.exists() or status_path.exists():
            incomplete.append(video_id)

    return {
        "valid_complete": valid,
        "target": len(videos),
        "remaining": len(missing),
        "missing": missing,
        "corrupt": corrupt,
        "incomplete": incomplete,
    }


def active_extraction_processes() -> list[str]:
    result = subprocess.run(
        ["pgrep", "-af", "backend/scripts/run_veatic_tribe_cache.py"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def wait_for_existing_worker(poll_seconds: int) -> None:
    while True:
        active = active_extraction_processes()
        if not active:
            return
        log(f"waiting for existing extraction worker: {active[0]}")
        time.sleep(poll_seconds)


def run_batch(
    batch: list[str],
    cache_root: Path,
    logs_dir: Path,
    batch_index: int,
    dry_run: bool,
) -> int:
    logs_dir.mkdir(parents=True, exist_ok=True)
    batch_label = f"{batch_index:03d}_{batch[0]}_{batch[-1]}"
    log_path = logs_dir / f"veatic_auto_batch_{batch_label}_mlx_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    cmd = [
        str(PYTHON),
        "-u",
        str(RUN_SCRIPT),
        "--run-mode",
        "cortical_fast_default",
        "--video-ids",
        ",".join(batch),
        "--limit",
        "0",
        "--cache-dir",
        str(cache_root),
        "--video-encoder-backend",
        "mlx",
        "--mps-memory-fraction",
        "0.35",
        "--mps-high-watermark",
        "0.45",
        "--mps-low-watermark",
        "0.25",
        "--attention-query-chunk-size",
        "128",
    ]
    log(f"starting batch {batch_index}: {','.join(batch)}")
    log(f"batch log: {log_path}")
    if dry_run:
        log("dry run: " + " ".join(cmd))
        return 0

    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="", flush=True)
        return process.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(ROOT / "benchmarks" / "veatic" / "veatic_manifest_1hz.report.json"))
    parser.add_argument("--cache-dir", default="/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache")
    parser.add_argument("--logs-dir", default=str(ROOT / "logs"))
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report_path = Path(args.report).expanduser().resolve()
    cache_root = Path(args.cache_dir).expanduser().resolve()
    logs_dir = Path(args.logs_dir).expanduser().resolve()

    batch_index = 1
    while True:
        wait_for_existing_worker(args.poll_seconds)
        audit = audit_cache(report_path, cache_root)
        log(
            "coverage: "
            f"{audit['valid_complete']}/{audit['target']} valid, "
            f"{audit['remaining']} remaining, "
            f"{len(audit['corrupt'])} corrupt, "
            f"{len(audit['incomplete'])} incomplete"
        )
        if audit["valid_complete"] == audit["target"] and not audit["corrupt"] and not audit["incomplete"]:
            log("cache coverage complete")
            return

        if audit["corrupt"]:
            log("corrupt entries will be retried by video id: " + ",".join(audit["corrupt"]))
        if audit["incomplete"]:
            log("incomplete entries will be retried by video id: " + ",".join(audit["incomplete"]))

        batch = audit["missing"][: args.batch_size]
        if not batch:
            raise SystemExit("No runnable missing batch found, but coverage is incomplete.")
        if args.max_batches and batch_index > args.max_batches:
            log("max batch limit reached")
            return

        code = run_batch(batch, cache_root, logs_dir, batch_index, args.dry_run)
        if code != 0:
            log(f"batch {batch_index} failed with exit code {code}; will audit and retry after wait")
            time.sleep(args.poll_seconds)
        else:
            log(f"batch {batch_index} finished")
        batch_index += 1


if __name__ == "__main__":
    main()
