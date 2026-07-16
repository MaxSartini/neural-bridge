#!/usr/bin/env python3
"""Time one VEATIC video through compiled MLX V-JEPA 2.1 only."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.services.mlx_vjepa21_cortical import MlxVjepa21FeatureModel  # noqa: E402


IMAGE_SIZE = 256
DECODE_HZ = 16.0
ROW_HZ = 2.0
FRAMES_PER_CLIP = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("/Volumes/onn. Drive/Neural Bridge/models/vjepa21_mlx/vitg"),
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(result.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid video duration: {duration}")
    return duration


def exact_2hz_time_seconds(duration_seconds: float) -> np.ndarray:
    final_index = int(math.floor(float(duration_seconds) * ROW_HZ + 1e-9))
    return (np.arange(final_index + 1, dtype=np.float32) / np.float32(ROW_HZ)).astype(
        np.float32
    )


def sample_plan(time_seconds: np.ndarray, decoded_frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    trailing_offsets = np.arange(
        FRAMES_PER_CLIP - 1,
        -1,
        -1,
        dtype=np.float32,
    ) / np.float32(DECODE_HZ)
    sample_times = np.maximum(
        np.float32(0.0),
        time_seconds[:, None] - trailing_offsets[None, :],
    ).astype(np.float32)
    indices = np.rint(sample_times.astype(np.float64) * DECODE_HZ).astype(np.int64)
    indices = np.clip(indices, 0, decoded_frame_count - 1).astype(np.int32)
    return sample_times, indices


def decode_video_software(path: Path) -> np.ndarray:
    short_side = int(256.0 / 224.0 * IMAGE_SIZE)
    video_filter = (
        f"fps={DECODE_HZ:.8f},"
        f"scale='if(gt(iw,ih),-2,{short_side})':'if(gt(iw,ih),{short_side},-2)',"
        f"crop={IMAGE_SIZE}:{IMAGE_SIZE}"
    )
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-an",
            "-vf",
            video_filter,
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    frame_bytes = IMAGE_SIZE * IMAGE_SIZE * 3
    if len(result.stdout) < frame_bytes or len(result.stdout) % frame_bytes:
        raise RuntimeError(f"Invalid decoded byte count: {len(result.stdout)}")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        -1,
        IMAGE_SIZE,
        IMAGE_SIZE,
        3,
    ).copy()


def mlx_memory_snapshot() -> dict[str, int | None]:
    fields = {
        "active_memory_bytes": "get_active_memory",
        "peak_memory_bytes": "get_peak_memory",
        "cache_memory_bytes": "get_cache_memory",
    }
    metal = getattr(mx, "metal", None)
    snapshot: dict[str, int | None] = {}
    for field, method_name in fields.items():
        method = getattr(mx, method_name, None)
        if not callable(method) and metal is not None:
            method = getattr(metal, method_name, None)
        try:
            snapshot[field] = int(method()) if callable(method) else None
        except Exception:
            snapshot[field] = None
    return snapshot


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    weights_dir = args.weights_dir.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not (weights_dir / "model.safetensors").is_file():
        raise FileNotFoundError(weights_dir / "model.safetensors")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to replace non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    process_started = time.perf_counter()
    duration_seconds = ffprobe_duration(video_path)
    time_seconds = exact_2hz_time_seconds(duration_seconds)

    decode_started = time.perf_counter()
    decoded = decode_video_software(video_path)
    decode_seconds = time.perf_counter() - decode_started
    sample_time_seconds, sample_frame_indices = sample_plan(time_seconds, len(decoded))

    model_load_started = time.perf_counter()
    model = MlxVjepa21FeatureModel(
        str(weights_dir),
        IMAGE_SIZE,
        compile_encoder=True,
        input_dtype="float16",
    )
    model_load_seconds = time.perf_counter() - model_load_started
    selected = [0, 2, 4, 6, 8, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 34, 36, 38, 40]

    warmup_windows = decoded[sample_frame_indices[:1]]
    warmup_started = time.perf_counter()
    warmup_states, warmup_diagnostics = (
        model.predict_hidden_states_with_compact_temporal_diagnostics(warmup_windows, selected)
    )
    compile_warmup_seconds = time.perf_counter() - warmup_started
    if not np.isfinite(warmup_states).all() or not np.isfinite(warmup_diagnostics).all():
        raise RuntimeError("V-JEPA warm-up produced non-finite values")
    del warmup_states, warmup_diagnostics, warmup_windows

    reset_peak = getattr(mx, "reset_peak_memory", None)
    if not callable(reset_peak):
        reset_peak = getattr(getattr(mx, "metal", None), "reset_peak_memory", None)
    if callable(reset_peak):
        reset_peak()

    feature_parts: list[np.ndarray] = []
    diagnostic_parts: list[np.ndarray] = []
    row_forward_seconds: list[float] = []
    sampling_seconds = 0.0
    forward_started = time.perf_counter()
    for row_index in range(len(time_seconds)):
        sampling_started = time.perf_counter()
        windows = decoded[sample_frame_indices[row_index : row_index + 1]]
        sampling_seconds += time.perf_counter() - sampling_started
        row_started = time.perf_counter()
        states, diagnostics = model.predict_hidden_states_with_compact_temporal_diagnostics(
            windows,
            selected,
        )
        row_elapsed = time.perf_counter() - row_started
        if not np.isfinite(states).all() or not np.isfinite(diagnostics).all():
            raise RuntimeError(f"V-JEPA produced non-finite values at row {row_index}")
        feature_parts.append(states.astype(np.float16))
        diagnostic_parts.append(diagnostics.astype(np.float32))
        row_forward_seconds.append(row_elapsed)
        if (row_index + 1) % 5 == 0 or row_index + 1 == len(time_seconds):
            print(
                json.dumps(
                    {
                        "stage": "vjepa21_only",
                        "rows": f"{row_index + 1}/{len(time_seconds)}",
                        "elapsed_seconds": round(sum(row_forward_seconds), 3),
                        "last_row_seconds": round(row_elapsed, 3),
                    }
                ),
                flush=True,
            )
    forward_loop_seconds = time.perf_counter() - forward_started
    forward_seconds = float(sum(row_forward_seconds))
    features = np.concatenate(feature_parts, axis=0)
    diagnostics53 = np.concatenate(diagnostic_parts, axis=0)
    memory = mlx_memory_snapshot()

    cache_path = output_root / "vjepa21_compact_cache.npz"
    write_started = time.perf_counter()
    np.savez_compressed(
        cache_path,
        features=features,
        temporal_diagnostics53=diagnostics53,
        time_seconds=time_seconds,
        sample_frame_indices=sample_frame_indices,
        sample_time_seconds=sample_time_seconds,
        selected_state_indices=np.asarray(selected, dtype=np.int16),
    )
    write_seconds = time.perf_counter() - write_started
    process_seconds = time.perf_counter() - process_started
    operational_cold_seconds = (
        decode_seconds + model_load_seconds + compile_warmup_seconds + forward_loop_seconds + write_seconds
    )
    resident_model_video_seconds = decode_seconds + forward_loop_seconds + write_seconds

    report = {
        "schema_version": "veatic_vjepa21_only_one_video_runtime_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "video": str(video_path),
        "video_sha256": file_sha256(video_path),
        "video_duration_seconds": duration_seconds,
        "row_hz": ROW_HZ,
        "row_count": int(len(time_seconds)),
        "time_start_seconds": float(time_seconds[0]),
        "time_end_seconds": float(time_seconds[-1]),
        "decode_hz": DECODE_HZ,
        "decoded_frame_count": int(len(decoded)),
        "image_size": IMAGE_SIZE,
        "frames_per_clip": FRAMES_PER_CLIP,
        "microbatch_size": 1,
        "dtype": "float16",
        "compile_encoder": True,
        "mlx_device": str(mx.default_device()),
        "weights_dir": str(weights_dir),
        "weights_sha256": file_sha256(weights_dir / "model.safetensors"),
        "selected_state_indices": selected,
        "features_shape": list(features.shape),
        "diagnostics53_shape": list(diagnostics53.shape),
        "outputs_finite": bool(np.isfinite(features).all() and np.isfinite(diagnostics53).all()),
        "runtime_seconds": {
            "decode": decode_seconds,
            "model_load": model_load_seconds,
            "compile_warmup_one_window": compile_warmup_seconds,
            "window_sampling_total": sampling_seconds,
            "steady_full_video_forward_sum": forward_seconds,
            "steady_full_video_loop": forward_loop_seconds,
            "cache_write": write_seconds,
            "cold_operational_total": operational_cold_seconds,
            "resident_compiled_model_video_total": resident_model_video_seconds,
            "observed_process_total": process_seconds,
        },
        "throughput": {
            "forward_seconds_per_row_mean": float(np.mean(row_forward_seconds)),
            "forward_seconds_per_row_median": float(np.median(row_forward_seconds)),
            "forward_seconds_per_row_min": float(np.min(row_forward_seconds)),
            "forward_seconds_per_row_max": float(np.max(row_forward_seconds)),
            "rows_per_second": float(len(time_seconds) / forward_seconds),
            "resident_real_time_factor": float(resident_model_video_seconds / duration_seconds),
            "projected_seconds_per_source_video_minute": float(
                np.mean(row_forward_seconds) * ROW_HZ * 60.0
            ),
        },
        "memory": memory,
        "cache": {
            "path": str(cache_path),
            "bytes": int(cache_path.stat().st_size),
        },
        "guardrails": {
            "vjepa21_only": True,
            "tribe_executed": False,
            "neural_bridge_head_executed": False,
            "single_gpu_lane": True,
            "protected_veatic_cache_modified": False,
        },
    }
    write_json(output_root / "runtime_report.json", report)
    markdown = f"""# VEATIC MLX V-JEPA 2.1 one-video runtime

- Video: `{video_path.name}` ({duration_seconds:.3f} source seconds)
- Rows: `{len(time_seconds)}` at exact 2 Hz (`{time_seconds[0]:.1f}`–`{time_seconds[-1]:.1f}` s)
- V-JEPA-only: **yes**
- TRIBE executed: **no**
- MLX device: `{mx.default_device()}`
- Outputs finite: **{report['outputs_finite']}**

## Runtime

- Decode: `{decode_seconds:.3f}` s
- Model load: `{model_load_seconds:.3f}` s
- Compile warm-up: `{compile_warmup_seconds:.3f}` s
- Steady full-video forward: `{forward_seconds:.3f}` s
- Compact-cache write: `{write_seconds:.3f}` s
- Cold operational total: `{operational_cold_seconds:.3f}` s
- With resident compiled model: `{resident_model_video_seconds:.3f}` s
- Mean / median per row: `{np.mean(row_forward_seconds):.3f}` / `{np.median(row_forward_seconds):.3f}` s
- Projected V-JEPA forward per source minute: `{report['throughput']['projected_seconds_per_source_video_minute']:.3f}` s

## Output

- Selected states: `{list(features.shape)}` float16
- Canonical compact diagnostics: `{list(diagnostics53.shape)}` float32
- Cache: `{cache_path}` (`{cache_path.stat().st_size}` bytes)
"""
    (output_root / "runtime_report.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)

    del model, decoded, features, diagnostics53
    mx.clear_cache()
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
