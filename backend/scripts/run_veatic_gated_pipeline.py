"""Run the VEATIC cortical-fast extraction, benchmark, and gate workflow.

The pipeline is deliberately conservative:
- it resumes extraction instead of deleting caches,
- it validates the cache before scoring,
- it runs the benchmark only on complete cache entries,
- it writes a machine-readable gate report and a Markdown summary,
- it does not claim success unless real cortical beats the required controls.
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


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "backend" / ".venv" / "bin" / "python"
VEATIC_REPORT = ROOT / "benchmarks" / "veatic" / "veatic_manifest_1hz.report.json"
DEFAULT_CACHE = Path("/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache")
DEFAULT_OUTPUT = ROOT / "benchmarks" / "veatic" / "veatic_neuro_benchmark_50video_cortical_fast_default.json"
TARGET_PATTERNS = (
    "arousal__future_change",
    "arousal__residual_future",
    "arousal__event_future_spike",
    "arousal__event_future_drop",
    "arousal__event_trend_reversal",
    "arousal__event_peak_onset",
    "arousal__event_recovery_onset",
)


def load_target_videos(report_path: Path, target: int) -> list[str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    videos = sorted(
        report["videos"],
        key=lambda item: (float(item["duration_seconds"]), int(item["video_id"])),
    )
    return [str(video["video_id"]) for video in videos[:target]]


def read_status(cache_dir: Path, video_id: str) -> dict[str, Any] | None:
    path = cache_dir / video_id / "cache_status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"video_id": video_id, "complete": False, "error": str(exc)}


def has_predictions(cache_dir: Path, video_id: str) -> bool:
    raw = cache_dir / video_id / "tribe_raw_output.npz"
    if not raw.exists():
        return False
    try:
        import numpy as np

        with np.load(raw) as bundle:
            return "predictions" in bundle.files
    except Exception:
        return False


def cache_audit(cache_dir: Path, video_ids: list[str], run_mode: str) -> dict[str, Any]:
    rows = []
    for video_id in video_ids:
        status = read_status(cache_dir, video_id)
        if status is None:
            rows.append({"video_id": video_id, "state": "missing"})
            continue
        contract = status.get("contract", {})
        rows.append(
            {
                "video_id": video_id,
                "state": "complete" if status.get("complete") else "failed" if status.get("error") else "incomplete",
                "error": status.get("error"),
                "run_mode": contract.get("run_mode"),
                "subcortical_enabled": contract.get("subcortical_enabled"),
                "has_predictions": has_predictions(cache_dir, video_id),
                "timings_seconds": status.get("timings_seconds"),
            }
        )
    complete = [row for row in rows if row["state"] == "complete" and row["has_predictions"]]
    failed = [row for row in rows if row["state"] == "failed"]
    incomplete = [row for row in rows if row["state"] in {"missing", "incomplete"}]
    contract_mismatches = [
        row
        for row in rows
        if row["state"] == "complete"
        and (row["run_mode"] != run_mode or row["subcortical_enabled"] is not False)
    ]
    return {
        "target": len(video_ids),
        "complete": len(complete),
        "failed": failed,
        "incomplete": incomplete,
        "contract_mismatches": contract_mismatches,
        "rows": rows,
    }


def run_stream(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n\n## {datetime.now().isoformat()} COMMAND: {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return process.wait()


def run_extract_until_complete(args: argparse.Namespace, video_ids: list[str], log_path: Path) -> dict[str, Any]:
    cache_dir = Path(args.cache_dir)
    for attempt in range(1, args.max_resume_attempts + 1):
        audit = cache_audit(cache_dir, video_ids, args.run_mode)
        if audit["complete"] >= len(video_ids):
            return {"status": "complete", "attempts": attempt - 1, "audit": audit}
        if audit["failed"]:
            return {"status": "failed", "attempts": attempt - 1, "audit": audit}
        command = [
            str(PYTHON),
            "backend/scripts/run_veatic_tribe_cache.py",
            "--run-mode",
            args.run_mode,
            "--limit",
            str(args.target),
            "--cache-dir",
            str(cache_dir),
        ]
        if args.force_extract:
            command.append("--force")
        code = run_stream(command, log_path)
        audit = cache_audit(cache_dir, video_ids, args.run_mode)
        if code == 0 and audit["complete"] >= len(video_ids):
            return {"status": "complete", "attempts": attempt, "audit": audit}
        if audit["failed"]:
            return {"status": "failed", "attempts": attempt, "audit": audit, "return_code": code}
        if code != 0 and not args.resume_on_clean_exit_only:
            time.sleep(args.retry_sleep_seconds)
            continue
        if code != 0:
            return {"status": "exited_nonzero", "attempts": attempt, "audit": audit, "return_code": code}
        time.sleep(args.retry_sleep_seconds)
    return {
        "status": "attempt_limit_reached",
        "attempts": args.max_resume_attempts,
        "audit": cache_audit(cache_dir, video_ids, args.run_mode),
    }


def run_benchmark(args: argparse.Namespace, log_path: Path) -> int:
    command = [
        str(PYTHON),
        "backend/scripts/run_veatic_neuro_benchmark.py",
        "--run-mode",
        args.run_mode,
        "--cache-dir",
        str(args.cache_dir),
        "--output",
        str(args.output),
    ]
    return run_stream(command, log_path)


def metric_for_target(target_name: str) -> tuple[str, bool]:
    if "__event_" in target_name:
        return "f1", True
    return "mae", False


def beats(real: float | None, control: float | None, higher_is_better: bool) -> bool:
    if real is None or control is None:
        return False
    return real > control if higher_is_better else real < control


def evaluate_gate(benchmark_path: Path) -> dict[str, Any]:
    report = json.loads(benchmark_path.read_text(encoding="utf-8"))
    checks = []
    passing = []
    for mode_name, mode in report.get("modes", {}).items():
        aggregate = mode.get("aggregate", mode)
        targets = aggregate.get("targets", {})
        for target_name, table in targets.items():
            if not isinstance(table, dict):
                continue
            if not any(target_name.startswith(pattern) for pattern in TARGET_PATTERNS):
                continue
            metric, higher_is_better = metric_for_target(target_name)
            real = table.get("autoregressive_plus_cortical_global", {})
            autoregressive = table.get("autoregressive", {})
            shuffled = table.get("autoregressive_plus_shuffled_cortical_global", {})
            gaussian = table.get("autoregressive_plus_random_gaussian_cortical_global", {})
            row = {
                "mode": mode_name,
                "target": target_name,
                "metric": metric,
                "higher_is_better": higher_is_better,
                "real": real.get(metric) if isinstance(real, dict) else None,
                "autoregressive": autoregressive.get(metric) if isinstance(autoregressive, dict) else None,
                "shuffled": shuffled.get(metric) if isinstance(shuffled, dict) else None,
                "random_gaussian": gaussian.get(metric) if isinstance(gaussian, dict) else None,
            }
            row["beats_autoregressive"] = beats(row["real"], row["autoregressive"], higher_is_better)
            row["beats_shuffled"] = beats(row["real"], row["shuffled"], higher_is_better)
            row["beats_random_gaussian"] = beats(row["real"], row["random_gaussian"], higher_is_better)
            row["passes"] = row["beats_autoregressive"] and row["beats_shuffled"] and row["beats_random_gaussian"]
            checks.append(row)
            if row["passes"]:
                passing.append(row)
    robust_passing = [row for row in passing if row["mode"] != "mode_a_official_veatic_70_30"]
    return {
        "benchmark": str(benchmark_path),
        "total_checks": len(checks),
        "passing_checks": passing,
        "robust_passing_checks": robust_passing,
        "decision": "scale_candidate" if robust_passing else "do_not_scale_yet",
        "interpretation": (
            "At least one non-official arousal dynamics target beat autoregressive, shuffled cortical, and random Gaussian controls."
            if robust_passing
            else "No non-official arousal dynamics target passed all required controls. Do not scale as evidence yet."
        ),
        "checks": checks,
    }


def write_gate_report(gate: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    lines = [
        "# VEATIC Gated Pipeline Report",
        "",
        f"Benchmark: `{gate['benchmark']}`",
        f"Decision: `{gate['decision']}`",
        "",
        gate["interpretation"],
        "",
        "## Passing Checks",
        "",
    ]
    if gate["passing_checks"]:
        for row in gate["passing_checks"]:
            lines.append(
                f"- {row['mode']} / {row['target']} / {row['metric']}: "
                f"real={row['real']} autoregressive={row['autoregressive']} "
                f"shuffled={row['shuffled']} random={row['random_gaussian']}"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Robust Non-Official Passing Checks", ""])
    if gate["robust_passing_checks"]:
        for row in gate["robust_passing_checks"]:
            lines.append(f"- {row['mode']} / {row['target']} / {row['metric']}")
    else:
        lines.append("- None.")
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=50)
    parser.add_argument("--run-mode", default="cortical_fast_default")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(VEATIC_REPORT))
    parser.add_argument("--log", default="logs/veatic_gated_pipeline.log")
    parser.add_argument("--gate-output", default="benchmarks/veatic/veatic_gated_pipeline_gate.json")
    parser.add_argument("--max-resume-attempts", type=int, default=20)
    parser.add_argument("--retry-sleep-seconds", type=int, default=20)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--resume-on-clean-exit-only", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    video_ids = load_target_videos(Path(args.report), args.target)
    log_path = ROOT / args.log
    extract_result = {"status": "skipped"}
    if not args.skip_extract:
        extract_result = run_extract_until_complete(args, video_ids, log_path)
        if extract_result["status"] != "complete":
            print(json.dumps({"extract_result": extract_result}, indent=2))
            raise SystemExit(1)

    final_audit = cache_audit(Path(args.cache_dir), video_ids, args.run_mode)
    if final_audit["complete"] < len(video_ids) or final_audit["failed"] or final_audit["contract_mismatches"]:
        print(json.dumps({"cache_audit": final_audit}, indent=2))
        raise SystemExit(1)

    benchmark_code = run_benchmark(args, log_path)
    if benchmark_code != 0:
        raise SystemExit(benchmark_code)
    gate = evaluate_gate(Path(args.output))
    gate["extract_result"] = extract_result
    gate["cache_audit"] = final_audit
    gate["created_at"] = datetime.now().isoformat()
    write_gate_report(gate, ROOT / args.gate_output)
    print(json.dumps({"gate_output": str(ROOT / args.gate_output), "decision": gate["decision"]}, indent=2))


if __name__ == "__main__":
    main()

