#!/usr/bin/env python3
"""Fresh no-cache V-JEPA 2.1 microbatch benchmark.

Runs the same decoded video windows through one MLX V-JEPA 2.1 model owner with
different microbatch sizes. This never reads or writes feature caches.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.mlx_vjepa2_cortical import _decode_video_grid_ffmpeg, _sample_decoded_grid  # noqa: E402
from backend.app.services.mlx_vjepa21_cortical import MlxVjepa21FeatureModel  # noqa: E402
from backend.scripts.again_sparse_tribe_teacher_500 import (  # noqa: E402
    CLIP_DURATION_SECONDS,
    DECODE_FPS,
    FRAMES_PER_CLIP,
    IMAGE_SIZE,
    group_mean_vitg_layers,
    mlx_memory_snapshot,
    selected_vitg_hidden_indices,
    sparse_window_frame_times,
    validate_single_gpu_runtime,
)


def local_external_root() -> Path:
    env_root = os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    for candidate in (
        Path("/Users/maxsartini/neural_bridge_scratch/external_root"),
        Path("/Volumes/onn. Drive/Neural Bridge"),
    ):
        if candidate.exists():
            return candidate
    raise EnvironmentError("Set NEURAL_BRIDGE_EXTERNAL_ROOT to the Neural Bridge external assets root.")


def ffprobe_duration(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(completed.stdout.strip())


def default_vjepa_weights_dir(dtype: str) -> Path:
    root = local_external_root()
    if dtype == "bfloat16":
        names = ("vitg_bfloat16_from_source", "vitg_bfloat16")
    elif dtype == "float32":
        names = ("vitg_float32_from_source", "vitg_float32")
    else:
        names = ("vitg_float16_from_source", "vitg")
    for name in names:
        candidate = root / "models" / "vjepa21_mlx" / name
        if (candidate / "model.safetensors").exists():
            return candidate
    return root / "models" / "vjepa21_mlx" / names[0]


def pick_sample_video() -> Path:
    candidates = [
        local_external_root() / "data" / "external" / "AGAIN" / "cleaned",
        local_external_root() / "datasets" / "AGAIN" / "cleaned",
        ROOT / "data" / "external" / "AGAIN" / "cleaned",
    ]
    for root in candidates:
        if root.exists():
            for suffix in ("*.webm", "*.mp4", "*.mkv", "*.mov"):
                found = sorted(root.rglob(suffix))
                if found:
                    return found[0]
    raise FileNotFoundError("No sample AGAIN video found; pass --video-path")


def prepare_windows(
    video_path: Path,
    *,
    count: int,
    image_size: int,
    full_video: bool = False,
    center_hz: float = 1.0,
    max_windows: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    decode_started = time.perf_counter()
    grid = _decode_video_grid_ffmpeg(video_path, fps=DECODE_FPS, image_size=image_size)
    decode_seconds = time.perf_counter() - decode_started
    try:
        duration = ffprobe_duration(video_path)
    except Exception:  # noqa: BLE001 - fallback to decoded grid duration
        duration = float(grid.shape[0] / DECODE_FPS)
    if full_video:
        if center_hz <= 0:
            raise ValueError("center_hz must be positive")
        expected = max(1, int(math.floor(duration * center_hz)))
        centers = np.linspace(0, duration, expected + 1, dtype=np.float32)[1:]
        if max_windows is not None:
            centers = centers[: max(1, int(max_windows))]
    else:
        start = max(CLIP_DURATION_SECONDS, 4.0)
        stop = max(start, duration - 1.0)
        if count == 1:
            centers = np.asarray([(start + stop) / 2.0], dtype=np.float32)
        else:
            centers = np.linspace(start, stop, count, dtype=np.float32)
    prep_started = time.perf_counter()
    windows = np.stack(
        [
            _sample_decoded_grid(grid, fps=DECODE_FPS, times=sparse_window_frame_times(float(center)))
            for center in centers
        ],
        axis=0,
    )
    prep_seconds = time.perf_counter() - prep_started
    return windows, {
        "decode_seconds": decode_seconds,
        "preprocess_seconds": prep_seconds,
        "duration_seconds": duration,
        "window_count": int(windows.shape[0]),
        "video_path": str(video_path),
        "full_video": bool(full_video),
        "center_hz": float(center_hz),
        "first_center_seconds": float(centers[0]) if len(centers) else None,
        "last_center_seconds": float(centers[-1]) if len(centers) else None,
    }


def run_variant(
    *,
    weights_dir: Path,
    windows: np.ndarray,
    selected_indices: list[int],
    image_size: int,
    input_dtype: str,
    compile_encoder: bool,
    microbatch_size: int,
    warmup: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    validate_single_gpu_runtime(
        gpu_workers=1,
        preprocess_workers=0,
        ready_queue_max_size=max(1, microbatch_size),
        writer_queue_max_size=4,
        microbatch_size=microbatch_size,
    )
    model = MlxVjepa21FeatureModel(
        str(weights_dir),
        image_size=image_size,
        compile_encoder=compile_encoder,
        input_dtype=input_dtype,
    )
    if warmup:
        _ = model.predict_hidden_states(windows[:1], selected_indices)
    outputs = []
    forward_started = time.perf_counter()
    for start in range(0, windows.shape[0], microbatch_size):
        batch = windows[start : start + microbatch_size]
        hidden = model.predict_hidden_states(batch, selected_indices)
        outputs.append(np.stack([group_mean_vitg_layers(np.asarray(item, dtype=np.float32)) for item in hidden], axis=0))
    forward_seconds = time.perf_counter() - forward_started
    features = np.concatenate(outputs, axis=0)
    stats = {
        "compile_encoder": bool(compile_encoder),
        "microbatch_size": int(microbatch_size),
        "input_dtype": input_dtype,
        "forward_seconds": forward_seconds,
        "total_end_to_end_seconds_shared_decode": forward_seconds,
        "seconds_per_window": forward_seconds / max(1, windows.shape[0]),
        **mlx_memory_snapshot(),
    }
    return features, stats


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], prep: dict[str, Any]) -> None:
    safe_rows = [row for row in rows if int(row.get("nonfinite_output_count", 0)) == 0 and math.isfinite(float(row.get("mean_abs_diff", 0.0)))]
    best = min(safe_rows, key=lambda row: float(row["seconds_per_window"])) if safe_rows else {}
    lines = [
        "# V-JEPA 2.1 Single-GPU Microbatch Benchmark",
        "",
        f"Created: `{datetime.now().isoformat(timespec='seconds')}`",
        f"Video: `{prep['video_path']}`",
        f"Windows: `{prep['window_count']}`",
        f"Full video: `{prep['full_video']}`",
        f"Center Hz: `{prep['center_hz']}`",
        f"Decode seconds: `{prep['decode_seconds']:.4f}`",
        f"Window preprocessing seconds: `{prep['preprocess_seconds']:.4f}`",
        "",
        "| Compile | Microbatch | Seconds/window | Forward seconds | Total incl shared decode/prep | Finite fraction | Nonfinite | Max abs diff | Mean abs diff |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {compile_encoder} | {microbatch_size} | {seconds_per_window:.4f} | {forward_seconds:.4f} | {total_end_to_end_seconds_shared_decode:.4f} | {finite_output_fraction:.6f} | {nonfinite_output_count} | {max_abs_diff:.6g} | {mean_abs_diff:.6g} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            (
                f"Fastest measured setting: `compile={best.get('compile_encoder')}`, "
                f"`microbatch_size={best.get('microbatch_size')}`, "
                f"`{float(best.get('seconds_per_window', math.nan)):.4f}s/window`."
            )
            if best
            else "No production-safe variant in this run: at least one output contained non-finite values or invalid diffs.",
            "",
            "Keep a microbatch only if outputs are fully finite, the output diff stays within tolerance, and Activity Monitor shows no swap/memory-pressure regression.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument("--weights-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--window-count", type=int, default=6)
    parser.add_argument("--full-video", action="store_true", help="Benchmark every center generated from the video's duration and --center-hz.")
    parser.add_argument("--center-hz", type=float, default=1.0, help="Center cadence used with --full-video.")
    parser.add_argument("--max-windows", type=int, default=None, help="Optional cap for a quick prefix of --full-video centers.")
    parser.add_argument("--input-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--microbatch-sizes", default="1,2,3")
    parser.add_argument("--compile-modes", default="false,true")
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or ROOT / "outputs" / f"vjepa21_single_gpu_microbatch_{stamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    video_path = (args.video_path or pick_sample_video()).expanduser().resolve()
    weights_dir = (args.weights_dir or default_vjepa_weights_dir(args.input_dtype)).expanduser().resolve()
    windows, prep = prepare_windows(
        video_path,
        count=args.window_count,
        image_size=args.image_size,
        full_video=args.full_video,
        center_hz=args.center_hz,
        max_windows=args.max_windows,
    )
    selected = selected_vitg_hidden_indices()
    microbatch_sizes = [int(value.strip()) for value in args.microbatch_sizes.split(",") if value.strip()]
    compile_modes = [value.strip().lower() in {"1", "true", "yes"} for value in args.compile_modes.split(",") if value.strip()]
    rows: list[dict[str, Any]] = []
    baseline: np.ndarray | None = None
    for compile_encoder in compile_modes:
        for microbatch_size in microbatch_sizes:
            features, stats = run_variant(
                weights_dir=weights_dir,
                windows=windows,
                selected_indices=selected,
                image_size=args.image_size,
                input_dtype=args.input_dtype,
                compile_encoder=compile_encoder,
                microbatch_size=microbatch_size,
                warmup=not args.no_warmup,
            )
            if baseline is None:
                baseline = features
                max_abs_diff = 0.0
                mean_abs_diff = 0.0
            else:
                diff = np.abs(features - baseline)
                finite_diff = diff[np.isfinite(diff)]
                max_abs_diff = float(np.max(finite_diff)) if finite_diff.size else math.nan
                mean_abs_diff = float(np.mean(finite_diff)) if finite_diff.size else math.nan
            rows.append(
                {
                    "video_path": str(video_path),
                    "weights_dir": str(weights_dir),
                    "image_size": args.image_size,
                    "window_count": int(windows.shape[0]),
                    "finite_output_fraction": float(np.isfinite(features).sum() / features.size),
                    "nonfinite_output_count": int(features.size - np.isfinite(features).sum()),
                    "max_abs_diff": max_abs_diff,
                    "mean_abs_diff": mean_abs_diff,
                    **prep,
                    **stats,
                }
            )
    for row in rows:
        row["total_end_to_end_seconds_shared_decode"] = (
            float(row["decode_seconds"]) + float(row["preprocess_seconds"]) + float(row["forward_seconds"])
        )
    write_csv(output_root / "vjepa21_microbatch_benchmark.csv", rows)
    (output_root / "prep_manifest.json").write_text(json.dumps(prep, indent=2) + "\n", encoding="utf-8")
    write_report(output_root / "vjepa21_microbatch_benchmark.md", rows, prep)
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
