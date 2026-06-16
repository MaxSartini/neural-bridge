"""Run the OpenLAV TRIBE cache one isolated, resumable stimulus at a time."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "backend" / "scripts" / "run_openlav_tribe_cache.py"
READINESS_GATE = ROOT / "backend" / "scripts" / "openlav_scientific_readiness_gate.py"
PYTHON = ROOT / "backend" / ".venv" / "bin" / "python"
EXTERNAL_ROOT = Path("/Volumes/onn. Drive/Neural Bridge")


def swap_used_gb() -> float:
    if sys.platform != "darwin":
        return 0.0
    output = subprocess.check_output(
        ["sysctl", "-n", "vm.swapusage"], text=True
    )
    match = re.search(r"used = ([0-9.]+)([MG])", output)
    if not match:
        return 0.0
    value = float(match.group(1))
    return value / 1024 if match.group(2) == "M" else value


def wait_for_resources(args: argparse.Namespace) -> None:
    while True:
        system_free_gb = shutil.disk_usage("/").free / 1e9
        external_free_gb = shutil.disk_usage(EXTERNAL_ROOT).free / 1e9
        current_swap_gb = swap_used_gb()
        if (
            system_free_gb >= args.min_system_free_gb
            and external_free_gb >= args.min_external_free_gb
            and current_swap_gb <= args.max_swap_used_gb
        ):
            return
        print(
            json.dumps(
                {
                    "status": "waiting_for_resources",
                    "system_free_gb": round(system_free_gb, 2),
                    "required_system_free_gb": args.min_system_free_gb,
                    "external_free_gb": round(external_free_gb, 2),
                    "required_external_free_gb": args.min_external_free_gb,
                    "swap_used_gb": round(current_swap_gb, 2),
                    "max_swap_used_gb": args.max_swap_used_gb,
                }
            ),
            flush=True,
        )
        time.sleep(args.resource_poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos-dir",
        default="/Volumes/onn. Drive/Neural Bridge/datasets/openlav_videos",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--cooldown-seconds", type=float, default=5.0)
    parser.add_argument("--min-system-free-gb", type=float, default=3.0)
    parser.add_argument("--min-external-free-gb", type=float, default=20.0)
    parser.add_argument("--resource-poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-swap-used-gb", type=float, default=8.0)
    parser.add_argument(
        "--item-timeout-seconds",
        type=float,
        default=900.0,
        help="Maximum wall time for one stimulus attempt before retry/fail.",
    )
    parser.add_argument(
        "--skip-readiness-gate",
        action="store_true",
        help="Bypass scientific/exactness preflight. Use only for local debugging.",
    )
    args = parser.parse_args()

    videos = sorted(Path(args.videos_dir).expanduser().resolve().glob("*.webm"))
    stop = args.offset + args.limit if args.limit else len(videos)
    selected = videos[args.offset:stop]
    failures = []
    child_env = os.environ.copy()
    external_paths = {
        "HF_HOME": EXTERNAL_ROOT / "cache" / "huggingface",
        "TMPDIR": EXTERNAL_ROOT / "tmp",
        "TRIBE_CACHE_DIR": EXTERNAL_ROOT / "cache" / "tribev2",
        "TRIBE_VIDEO_WINDOW_CACHE_DIR": EXTERNAL_ROOT / "cache" / "tribev2" / "video_windows",
    }
    for name, path in external_paths.items():
        path.mkdir(parents=True, exist_ok=True)
        child_env[name] = str(path)

    if not args.skip_readiness_gate:
        subprocess.run(
            [str(PYTHON), str(READINESS_GATE)],
            cwd=ROOT,
            env=child_env,
            check=True,
        )

    for position, video in enumerate(selected, start=args.offset):
        command = [
            str(PYTHON),
            str(RUNNER),
            "--videos-dir",
            str(Path(args.videos_dir).expanduser().resolve()),
            "--offset",
            str(position),
            "--limit",
            "1",
        ]
        succeeded = False
        for attempt in range(args.retries + 1):
            wait_for_resources(args)
            print(
                json.dumps(
                    {
                        "batch_progress": f"{position + 1}/{len(videos)}",
                        "stimulus_id": video.stem,
                        "attempt": attempt + 1,
                    }
                ),
                flush=True,
            )
            try:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=child_env,
                    check=False,
                    timeout=args.item_timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                print(
                    json.dumps(
                        {
                            "stimulus_id": video.stem,
                            "timeout_seconds": args.item_timeout_seconds,
                            "status": "retrying" if attempt < args.retries else "failed_timeout",
                        }
                    ),
                    flush=True,
                )
                time.sleep(args.cooldown_seconds)
                continue
            if result.returncode == 0:
                succeeded = True
                break
            print(
                json.dumps(
                    {
                        "stimulus_id": video.stem,
                        "returncode": result.returncode,
                        "status": "retrying" if attempt < args.retries else "failed",
                    }
                ),
                flush=True,
            )
            time.sleep(args.cooldown_seconds)
        if not succeeded:
            failures.append(video.stem)
        time.sleep(args.cooldown_seconds)

    report = {
        "selected": len(selected),
        "completed_or_cached": len(selected) - len(failures),
        "failures": failures,
    }
    print(json.dumps(report, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
