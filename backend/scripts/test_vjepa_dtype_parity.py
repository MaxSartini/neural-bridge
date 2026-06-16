"""Compare V-JEPA2 selected hidden states across dtype contracts.

This is a bounded single-window scientific contract test. It does not run TRIBE
end-to-end; it compares the exact V-JEPA features consumed by NeuralSet/TRIBE
for the same 64 sampled frames.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from moviepy import VideoFileClip


MODEL = Path("/Volumes/onn. Drive/Neural Bridge/models/cortical-upstream/facebook-vjepa2-vitg-fpc64-256")
VIDEO = Path("/Volumes/onn. Drive/Neural Bridge/datasets/openlav_videos/VID_1001.webm")


def load_frames(video_path: Path, frame_count: int, duration: float) -> np.ndarray:
    clip = VideoFileClip(str(video_path))
    try:
        times = np.linspace(0, min(duration, clip.duration), frame_count, endpoint=False)
        return np.asarray([clip.get_frame(float(t)).astype(np.uint8) for t in times])
    finally:
        clip.close()


def run_dtype(
    dtype: str,
    frames: np.ndarray,
    device: str,
    memory_fraction: float,
    model_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    os.environ["TRIBE_VIDEO_DTYPE"] = dtype
    os.environ.setdefault("TRIBE_MPS_CHUNKED_ATTENTION", "true")
    os.environ.setdefault("TRIBE_MPS_ATTENTION_QUERY_CHUNK_SIZE", "256")
    os.environ.setdefault("TRIBE_VJEPA_SELECTIVE_HIDDEN_STATES", "true")
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(memory_fraction)
        torch.mps.empty_cache()
    from neuralset.extractors.video import _HFVideoModel

    started = time.perf_counter()
    model = _HFVideoModel(
        model_name=str(model_path),
        pretrained=True,
        num_frames=frames.shape[0],
        cache_n_layers=20,
        layers="all",
        layer_aggregation="mean",
    )
    model.model.to(device)
    with torch.inference_mode():
        selected = model.predict_hidden_states(frames)
        selected = selected.detach().float().cpu().numpy()
    elapsed = time.perf_counter() - started
    stats = {
        "dtype": dtype,
        "device": device,
        "seconds": round(elapsed, 3),
        "shape": list(selected.shape),
        "finite": bool(np.isfinite(selected).all()),
    }
    if device == "mps":
        stats["mps_driver_allocated_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 3)
        torch.mps.empty_cache()
    del model
    gc.collect()
    return selected, stats


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    flat_ref = reference.reshape(-1).astype(np.float64)
    flat_candidate = candidate.reshape(-1).astype(np.float64)
    corr = float(np.corrcoef(flat_ref, flat_candidate)[0, 1])
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "reference_std": float(np.std(flat_ref)),
        "candidate_std": float(np.std(flat_candidate)),
        "pearson": corr,
        "relative_rmse_to_reference_std": float(np.sqrt(np.mean(delta * delta)) / max(np.std(flat_ref), 1e-12)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(VIDEO))
    parser.add_argument("--model", default=str(MODEL))
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--memory-fraction", type=float, default=float(os.environ.get("TRIBE_MPS_MEMORY_FRACTION", "0.50")))
    parser.add_argument("--reference-dtype", default="float32")
    parser.add_argument("--candidate-dtype", default="float16")
    parser.add_argument("--relative-rmse-tolerance", type=float, default=0.01)
    parser.add_argument("--pearson-tolerance", type=float, default=0.999)
    parser.add_argument("--output", default="benchmarks/openlav/vjepa_dtype_parity.json")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve()
    frames = load_frames(video_path, args.frames, args.duration)
    report: dict[str, Any] = {
        "schema_version": "vjepa_dtype_parity_v1",
        "video": str(video_path),
        "model": str(model_path),
        "frames": args.frames,
        "duration_seconds": args.duration,
        "device": args.device,
        "reference_dtype": args.reference_dtype,
        "candidate_dtype": args.candidate_dtype,
        "tolerances": {
            "relative_rmse_to_reference_std": args.relative_rmse_tolerance,
            "pearson": args.pearson_tolerance,
        },
    }
    try:
        reference, reference_stats = run_dtype(
            args.reference_dtype, frames, args.device, args.memory_fraction, model_path
        )
        candidate, candidate_stats = run_dtype(
            args.candidate_dtype, frames, args.device, args.memory_fraction, model_path
        )
        metrics = compare(reference, candidate)
        report.update({
            "success": True,
            "reference": reference_stats,
            "candidate": candidate_stats,
            "metrics": metrics,
            "passes_contract": (
                metrics["relative_rmse_to_reference_std"] <= args.relative_rmse_tolerance
                and metrics["pearson"] >= args.pearson_tolerance
            ),
        })
    except Exception as exc:
        report.update({
            "success": False,
            "passes_contract": False,
            "error": f"{type(exc).__name__}: {exc}",
        })
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report.get("passes_contract"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
