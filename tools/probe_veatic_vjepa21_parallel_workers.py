#!/usr/bin/env python3
"""Run concurrent V-JEPA 2.1 MLX TRIBE workers on short VEATIC videos."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from probe_veatic_vjepa21_real_video_smoke import (  # noqa: E402
    ffprobe,
    load_manifest_rows,
    resolve_external_path,
    write_csv,
    write_json,
)
from probe_veatic_vjepa21_speed_ablation import row_from_variant  # noqa: E402


def load_short_videos(report_path: Path, count: int) -> list[dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return sorted(report["videos"], key=lambda item: (float(item["duration_seconds"]), int(item["video_id"])))[:count]


def run_one_worker(
    *,
    worker_index: int,
    selected_video: dict[str, Any],
    manifest_path: str,
    external_root: str,
    external_output_root: str,
    image_size: int,
    window_batch_size: int,
) -> dict[str, Any]:
    from backend.app.config import Config

    external_root_path = Path(external_root)
    manifest_rows_all = load_manifest_rows(Path(manifest_path))
    video_id = str(selected_video["video_id"])
    manifest_rows = [row for row in manifest_rows_all if str(row.get("video_id")) == video_id]
    if not manifest_rows:
        raise ValueError(f"No manifest rows found for video_id={video_id}")
    video_path = resolve_external_path(manifest_rows[0]["media_path"], external_root_path)
    metadata = ffprobe(video_path)
    old_npz = external_root_path / "benchmarks" / "veatic" / "tribe_cache" / video_id / "tribe_raw_output.npz"
    Config.NEURAL_BRIDGE_EXTERNAL_ROOT = external_root

    row = row_from_variant(
        variant={
            "variant": f"parallel256_worker{worker_index}_video{video_id}",
            "image_size": image_size,
            "window_batch_size": window_batch_size,
        },
        video_id=video_id,
        video_path=video_path,
        manifest_rows=manifest_rows,
        selected_video=selected_video,
        metadata=metadata,
        external_output_root=Path(external_output_root),
        old_npz=old_npz,
    )
    row["worker_index"] = worker_index
    return row


def best_effort_command(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        return (proc.stdout + proc.stderr).strip()
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        return str(exc)


def write_report(path: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    lines = [
        "# VEATIC V-JEPA 2.1 MLX Parallel Worker Smoke",
        "",
        "## Verdict",
        f"- Workers requested: `{manifest['workers_requested']}`",
        f"- Successful workers: `{sum(1 for row in rows if row.get('success'))}`",
        f"- Parallel wall time seconds: `{manifest['parallel_wall_time_seconds']}`",
        f"- Sum worker elapsed seconds: `{manifest['sum_worker_encoding_time_seconds']}`",
        f"- Throughput ratio vs sequential worker sum: `{manifest['throughput_ratio_vs_worker_sum']}`",
        "- Benchmark run: `false`",
        "- Models trained: `false`",
        "- Accuracy claim made: `false`",
        "",
        "## Worker Results",
        "| worker | video | duration | rows | shape | success | time_s | sec/window | nonfinite |",
        "|---:|---:|---:|---:|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {worker} | {video} | {duration} | {rows} | {shape} | {success} | {time} | {spw} | {nonfinite} |".format(
                worker=row.get("worker_index"),
                video=row.get("video_id"),
                duration=row.get("manifest_duration_seconds"),
                rows=row.get("new_temporal_rows"),
                shape=row.get("new_shape"),
                success=str(row.get("success")).lower(),
                time=row.get("encoding_time_seconds"),
                spw=row.get("seconds_per_internal_window"),
                nonfinite=row.get("new_nonfinite_count"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- This tests whether independent video-level worker processes improve throughput.",
            "- It does not test benchmark accuracy or model quality.",
            "- If the throughput ratio is near 1.0, the GPU/Metal path is saturated and more workers do not help.",
            "- If the throughput ratio is much greater than 1.0, video-level parallelism is useful for bulk encoding.",
            "",
            "## Guardrails",
            "- full_veatic_encoding_run: `false`",
            "- again_encoding_run: `false`",
            "- benchmark_run: `false`",
            "- models_trained: `false`",
            "- old_veatic_cache_modified: `false`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("benchmarks/veatic/veatic_manifest_124_complete_20260616.report.json"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--video-count", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--window-batch-size", type=int, default=1)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.video_count < args.workers:
        args.video_count = args.workers

    from backend.app.config import Config

    external_root = Path(Config.NEURAL_BRIDGE_EXTERNAL_ROOT).expanduser()
    output_root = ROOT / "outputs" / f"veatic_vjepa21_parallel_workers_{args.timestamp}"
    external_output_root = external_root / "benchmarks" / "veatic" / f"vjepa21_mlx_parallel_workers_{args.timestamp}"
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty output root: {output_root}")
    if external_output_root.exists() and any(external_output_root.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty external root: {external_output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    external_output_root.mkdir(parents=True, exist_ok=False)

    selected = load_short_videos(args.report, args.video_count)[: args.workers]
    started = time.time()
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                run_one_worker,
                worker_index=index + 1,
                selected_video=video,
                manifest_path=str(args.manifest),
                external_root=str(external_root),
                external_output_root=str(external_output_root),
                image_size=args.image_size,
                window_batch_size=args.window_batch_size,
            )
            for index, video in enumerate(selected)
        ]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
            write_csv(output_root / "veatic_vjepa21_parallel_worker_results.csv", sorted(rows, key=lambda item: int(item["worker_index"])))
    wall = time.time() - started
    rows = sorted(rows, key=lambda item: int(item["worker_index"]))
    worker_sum = sum(float(row.get("encoding_time_seconds") or 0.0) for row in rows)
    throughput_ratio = worker_sum / wall if wall > 0 else None

    manifest = {
        "created_at": datetime.now().isoformat(),
        "output_root": str(output_root),
        "external_output_root": str(external_output_root),
        "workers_requested": args.workers,
        "video_count": len(selected),
        "selected_videos": [
            {
                "video_id": str(video["video_id"]),
                "duration_seconds": video.get("duration_seconds"),
                "fps": video.get("fps"),
                "manifest_rows": video.get("manifest_rows"),
            }
            for video in selected
        ],
        "image_size": args.image_size,
        "window_batch_size": args.window_batch_size,
        "parallel_wall_time_seconds": round(wall, 3),
        "sum_worker_encoding_time_seconds": round(worker_sum, 3),
        "throughput_ratio_vs_worker_sum": round(throughput_ratio, 4) if throughput_ratio else None,
        "system": {
            "mlx_default_device": best_effort_command([sys.executable, "-c", "import mlx.core as mx; print(mx.default_device())"]),
            "pmset_batt": best_effort_command(["pmset", "-g", "batt"]),
            "pmset_therm": best_effort_command(["pmset", "-g", "therm"]),
        },
        "guardrails": {
            "real_veatic_videos_encoded": len(selected),
            "full_veatic_encoding_run": False,
            "again_encoding_run": False,
            "benchmark_run": False,
            "models_trained": False,
            "old_veatic_cache_modified": False,
            "vjepa21_claim_made": False,
        },
    }
    write_csv(output_root / "veatic_vjepa21_parallel_worker_results.csv", rows)
    write_json(output_root / "run_manifest.json", manifest)
    write_report(output_root / "veatic_vjepa21_parallel_worker_report.md", rows, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
