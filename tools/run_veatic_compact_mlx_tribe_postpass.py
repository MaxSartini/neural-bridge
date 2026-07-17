#!/usr/bin/env python3
"""Run cache-only MLX TRIBE v2 over compact VEATIC V-JEPA 2.1 folders."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.services.mlx_tribe_encoder import MlxTribeEncoder
from backend.scripts.again_sparse_tribe_teacher_500 import group_mean_vitg_layers


EXPECTED_FEATURE_TAIL = (20, 1, 1408)
EXPECTED_CORTICAL_WIDTH = 20484


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--expected-videos", type=int, default=124)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def natural_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name), path.name
    except ValueError:
        return 10**12, path.name


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def output_complete(output_dir: Path) -> bool:
    status_path = output_dir / "status.json"
    cache_path = output_dir / "tribe_v2_cortical_predictions.npz"
    if not status_path.is_file() or not cache_path.is_file():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "complete":
            return False
        with np.load(cache_path, allow_pickle=False) as bundle:
            cortical = bundle["cortical_prediction"]
            rows = bundle["time_seconds"].shape[0]
        return cortical.shape == (rows, EXPECTED_CORTICAL_WIDTH)
    except Exception:
        return False


def group_features(features: np.ndarray) -> np.ndarray:
    rows = [group_mean_vitg_layers(np.asarray(row, dtype=np.float32)) for row in features]
    return np.stack(rows, axis=0).astype(np.float16)


def process_video(
    input_dir: Path,
    output_dir: Path,
    encoder: MlxTribeEncoder,
) -> dict[str, object]:
    started = time.perf_counter()
    input_path = input_dir / "vjepa21_hidden_states.npz"
    with np.load(input_path, allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    features = arrays.pop("features")
    if features.ndim != 4 or tuple(features.shape[1:]) != EXPECTED_FEATURE_TAIL:
        raise RuntimeError(f"{input_dir.name}: unexpected feature shape {features.shape}")
    time_seconds = np.asarray(arrays["time_seconds"], dtype=np.float32)
    if len(features) != len(time_seconds):
        raise RuntimeError(f"{input_dir.name}: feature/time row mismatch")

    grouped = group_features(features)
    video = np.asarray(grouped, dtype=np.float32).transpose(1, 2, 0)[None]
    cortical = encoder.predict({"video": video}, pool_outputs=False)
    cortical_rows = np.asarray(cortical[0].T, dtype=np.float16)
    if cortical_rows.shape != (len(time_seconds), EXPECTED_CORTICAL_WIDTH):
        raise RuntimeError(f"{input_dir.name}: unexpected cortical shape {cortical_rows.shape}")
    if not np.isfinite(cortical_rows).all():
        raise RuntimeError(f"{input_dir.name}: non-finite TRIBE output")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_arrays = {
        key: value
        for key, value in arrays.items()
        if key != "temporal_diagnostics53"
    }
    output_arrays.update(
        cortical_prediction=cortical_rows,
        tribe_grouped_video_feature=grouped,
        temporal_diagnostics53=np.asarray(arrays["temporal_diagnostics53"], dtype=np.float32),
        time_seconds=time_seconds,
    )
    output_path = output_dir / "tribe_v2_cortical_predictions.npz"
    atomic_npz(output_path, output_arrays)
    elapsed = time.perf_counter() - started
    manifest: dict[str, object] = {
        "schema_version": "veatic_compact_mlx_tribe_v2_postpass_v1",
        "video_id": input_dir.name,
        "status": "complete",
        "row_count": int(len(time_seconds)),
        "input_cache": str(input_path),
        "output_cache": str(output_path),
        "cortical_shape": list(cortical_rows.shape),
        "grouped_shape": list(grouped.shape),
        "runtime_seconds": elapsed,
        "finished_at": utc_now(),
    }
    atomic_json(output_dir / "manifest.json", manifest)
    atomic_json(output_dir / "status.json", manifest)
    print(json.dumps(manifest, sort_keys=True), flush=True)
    del features, grouped, video, cortical, cortical_rows, arrays, output_arrays
    mx.clear_cache()
    gc.collect()
    return manifest


def main() -> int:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    encoder = MlxTribeEncoder(str(model_dir), dtype="float16")
    completed: set[str] = {
        path.parent.name
        for path in output_root.glob("*/status.json")
        if output_complete(path.parent)
    }
    failures: dict[str, str] = {}
    while len(completed) < args.expected_videos:
        candidates = sorted(input_root.iterdir(), key=natural_key) if input_root.exists() else []
        progressed = False
        for input_dir in candidates:
            if not input_dir.is_dir() or input_dir.name in completed:
                continue
            if not (input_dir / "_UPLOAD_COMPLETE.json").is_file():
                continue
            if not (input_dir / "vjepa21_hidden_states.npz").is_file():
                continue
            try:
                process_video(input_dir, output_root / input_dir.name, encoder)
                completed.add(input_dir.name)
                failures.pop(input_dir.name, None)
                progressed = True
            except Exception as exc:
                failures[input_dir.name] = f"{type(exc).__name__}: {exc}"
                atomic_json(
                    output_root / f"{input_dir.name}.failed.json",
                    {"video_id": input_dir.name, "status": "failed", "error": failures[input_dir.name]},
                )
                print(json.dumps({"video_id": input_dir.name, "error": failures[input_dir.name]}), flush=True)
        atomic_json(
            output_root / "run_status.json",
            {
                "schema_version": "veatic_compact_mlx_tribe_v2_run_v1",
                "expected_videos": args.expected_videos,
                "completed_videos": len(completed),
                "failures": failures,
                "updated_at": utc_now(),
            },
        )
        if len(completed) >= args.expected_videos:
            break
        if not progressed:
            time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
