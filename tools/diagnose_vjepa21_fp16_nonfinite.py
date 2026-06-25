#!/usr/bin/env python3
"""Locate where V-JEPA 2.1 float16 first produces non-finite values."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.mlx_vjepa2_cortical import _decode_video_grid_ffmpeg, _sample_decoded_grid  # noqa: E402
from backend.app.services.mlx_vjepa21_cortical import MlxVjepa21FeatureModel, _mlx_array_with_dtype, _preprocess_video_batch  # noqa: E402
from backend.scripts.again_sparse_tribe_teacher_500 import DECODE_FPS, IMAGE_SIZE, sparse_window_frame_times  # noqa: E402
from tools.benchmark_vjepa21_single_gpu_microbatch import default_vjepa_weights_dir, pick_sample_video  # noqa: E402


def finite_summary(array: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(array)
    finite_values = array[finite]
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite_count": int(finite.sum()),
        "total_count": int(array.size),
        "finite_fraction": float(finite.sum() / array.size) if array.size else 1.0,
        "nan_count": int(np.isnan(array).sum()),
        "posinf_count": int(np.isposinf(array).sum()),
        "neginf_count": int(np.isneginf(array).sum()),
        "finite_min": float(np.min(finite_values)) if finite_values.size else None,
        "finite_max": float(np.max(finite_values)) if finite_values.size else None,
    }


def diagnose_dtype(video_path: Path, weights_dir: Path, dtype: str, timestamp: float) -> list[dict[str, Any]]:
    grid = _decode_video_grid_ffmpeg(video_path, fps=DECODE_FPS, image_size=IMAGE_SIZE)
    window = _sample_decoded_grid(grid, fps=DECODE_FPS, times=sparse_window_frame_times(timestamp))
    preprocessed = _preprocess_video_batch(window, IMAGE_SIZE)
    rows: list[dict[str, Any]] = []
    rows.append({"dtype": dtype, "stage": "decoded_window", **finite_summary(window)})
    rows.append({"dtype": dtype, "stage": "preprocessed_window", **finite_summary(preprocessed)})
    model = MlxVjepa21FeatureModel(str(weights_dir), image_size=IMAGE_SIZE, compile_encoder=False, input_dtype=dtype)
    video = _mlx_array_with_dtype(preprocessed, dtype)
    encoder = model.encoder
    batch, _channels, frames, height, width = video.shape
    temporal = frames // encoder.config.tubelet_size
    h_patches = height // encoder.config.patch_size
    w_patches = width // encoder.config.patch_size
    hidden = encoder.patch_embed(video)
    if encoder.config.modality_embedding:
        hidden = hidden + mx.broadcast_to(encoder.video_mod_embed, (batch, 1, encoder.config.hidden_size))
    mx.eval(hidden)
    rows.append({"dtype": dtype, "stage": "patch_embed_plus_video_mod", **finite_summary(np.asarray(hidden.astype(mx.float32)))})
    for index, block in enumerate(encoder.blocks, start=1):
        start = time.perf_counter()
        hidden = block(hidden, temporal=temporal, h_patches=h_patches, w_patches=w_patches)
        mx.eval(hidden)
        arr = np.asarray(hidden.astype(mx.float32))
        row = {"dtype": dtype, "stage": f"block_{index:02d}", "seconds": time.perf_counter() - start, **finite_summary(arr)}
        rows.append(row)
        if row["finite_fraction"] < 1.0:
            # Continue a few more blocks would be expensive and noisy; first failure is the important diagnostic.
            break
    mx.clear_cache()
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument("--timestamp", type=float, default=4.0)
    parser.add_argument("--dtypes", default="float16,bfloat16,float32")
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or ROOT / "outputs" / f"vjepa21_fp16_nonfinite_diagnostic_{stamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    video_path = (args.video_path or pick_sample_video()).expanduser().resolve()
    all_rows: list[dict[str, Any]] = []
    for dtype in [value.strip() for value in args.dtypes.split(",") if value.strip()]:
        weights_dir = default_vjepa_weights_dir(dtype)
        rows = diagnose_dtype(video_path, weights_dir, dtype, args.timestamp)
        for row in rows:
            row["video_path"] = str(video_path)
            row["weights_dir"] = str(weights_dir)
            row["timestamp"] = args.timestamp
        all_rows.extend(rows)
    write_csv(output_root / "vjepa21_fp16_nonfinite_diagnostic.csv", all_rows)
    (output_root / "manifest.json").write_text(
        json.dumps({"video_path": str(video_path), "timestamp": args.timestamp, "rows": len(all_rows)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
