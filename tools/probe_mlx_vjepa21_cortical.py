#!/usr/bin/env python3
"""Smoke-test the MLX V-JEPA 2.1 ViT-g path used by TRIBE video encoding."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.mlx_vjepa21_cortical import MlxVjepa21FeatureModel, MlxVjepa21Video


def default_weights_dir() -> Path:
    configured = os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT", "").strip()
    if not configured:
        raise EnvironmentError("Set NEURAL_BRIDGE_EXTERNAL_ROOT to the Neural Bridge external assets root.")
    return Path(configured).expanduser() / "models" / "vjepa21_mlx" / "vitg"


def parse_indices(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def make_frames(num_frames: int, image_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 256, size=(num_frames, image_size, image_size, 3), dtype=np.uint8)
    # Add deterministic spatial structure so the smoke input is not pure noise.
    ramp = np.linspace(0, 255, image_size, dtype=np.uint8)
    frames[:, :, :, 0] = ramp[None, None, :]
    frames[:, :, :, 1] = ramp[None, :, None]
    return frames


def direct_model_probe(weights_dir: Path, image_size: int, num_frames: int, selected: list[int]) -> dict:
    frames = make_frames(num_frames=num_frames, image_size=image_size, seed=21)
    started = time.time()
    model = MlxVjepa21FeatureModel(str(weights_dir), image_size=image_size)
    load_seconds = time.time() - started
    started = time.time()
    states = model.predict_hidden_states(frames, selected)
    forward_seconds = time.time() - started
    return {
        "mode": "direct_model",
        "weights_dir": str(weights_dir),
        "image_size": image_size,
        "num_frames": num_frames,
        "selected_indices": selected,
        "output_shape": list(states.shape),
        "output_dtype": str(states.dtype),
        "finite": bool(np.isfinite(states).all()),
        "mean": float(np.mean(states)),
        "std": float(np.std(states)),
        "load_seconds": round(load_seconds, 3),
        "forward_seconds": round(forward_seconds, 3),
    }


def extractor_probe(weights_dir: Path, image_size: int, num_frames: int) -> dict:
    import imageio.v2 as imageio
    from neuralset.events import etypes as evts

    frames = make_frames(num_frames=num_frames, image_size=image_size, seed=22)
    with tempfile.TemporaryDirectory(prefix="vjepa21_tribe_smoke_") as tmp:
        video_path = Path(tmp) / "smoke.mp4"
        imageio.mimsave(video_path, list(frames), fps=max(1, num_frames))
        extractor = MlxVjepa21Video(
            mlx_weights_dir=str(weights_dir),
            image_size=image_size,
            frequency=1,
            clip_duration=1.0,
            max_imsize=image_size,
            cache_model_name=f"mlx-vjepa21-smoke-image{image_size}",
        )
        event = evts.Video(
            start=0.0,
            timeline="vjepa21_smoke",
            duration=1.0,
            filepath=video_path,
            frequency=1.0,
            offset=0.0,
        )
        started = time.time()
        outputs = list(extractor._get_data([event]))
        elapsed = time.time() - started
    if len(outputs) != 1:
        raise RuntimeError(f"expected one TimedArray from extractor, got {len(outputs)}")
    data = np.asarray(outputs[0].data)
    return {
        "mode": "tribe_extractor",
        "weights_dir": str(weights_dir),
        "image_size": image_size,
        "num_frames": num_frames,
        "output_shape": list(data.shape),
        "output_dtype": str(data.dtype),
        "finite": bool(np.isfinite(data).all()),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "elapsed_seconds": round(elapsed, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=default_weights_dir())
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--selected-indices", default="0,20,40")
    parser.add_argument("--through-extractor", action="store_true")
    args = parser.parse_args()

    weights_dir = args.weights_dir.expanduser().resolve()
    selected = parse_indices(args.selected_indices)
    results = {
        "success": True,
        "weights_dir": str(weights_dir),
        "direct_model": direct_model_probe(weights_dir, args.image_size, args.num_frames, selected),
    }
    if args.through_extractor:
        results["tribe_extractor"] = extractor_probe(weights_dir, args.image_size, args.num_frames)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
