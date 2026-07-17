#!/usr/bin/env python3
"""Run one video through MLX V-JEPA 2.1, MLX TRIBE, and Neural Bridge.

This is an inference-only smoke path.  It reconstructs the frozen zero-label
head scaler from the canonical AGAIN training substrate, verifies that the
three frozen heads reproduce their sealed predictions, and then applies the
same transforms to a new video's 2 Hz cortical rows.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.services.mlx_tribe_encoder import MlxTribeEncoder  # noqa: E402
from backend.app.services.mlx_vjepa2_cortical import (  # noqa: E402
    _decode_video_grid_ffmpeg,
    _sample_decoded_grid,
)
from backend.app.services.mlx_vjepa21_cortical import MlxVjepa21FeatureModel  # noqa: E402
from backend.scripts import again_sparse_tribe_teacher_500 as sparse  # noqa: E402
from backend.scripts import again_zero_label_deployment_stage_a as stage_a  # noqa: E402
from backend.scripts import (  # noqa: E402
    run_again_dense_2hz_zero_label_deployment_stage_a as stage_a_runner,
)
from backend.scripts import (  # noqa: E402
    run_again_dense_2hz_zero_label_direct_supervised_locked_confirmation as locked_runner,
)


ROW_HZ = 2.0
DECODE_HZ = 16.0
FRAMES_PER_CLIP = 64
IMAGE_SIZE = 256
EXPECTED_CORTICAL_WIDTH = 20_484
PRIMARY_LANE = "video_supervised_temporal"
SEEDS = (20260721, 20260722, 20260723)


def parse_args() -> argparse.Namespace:
    external = Path(
        os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT", str(REPO_ROOT))
    ).expanduser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, default=external)
    parser.add_argument(
        "--dense-root",
        type=Path,
        default=REPO_ROOT / ".cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz",
    )
    parser.add_argument(
        "--stage0-root",
        type=Path,
        default=REPO_ROOT / "evidence/zero_label_video_only_deployment_stage0_20260714",
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--compile-vjepa", action="store_true")
    parser.add_argument("--skip-locked-reproduction", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe_duration(path: Path) -> float:
    proc = subprocess.run(
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
    duration = float(proc.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid video duration: {duration}")
    return duration


def compact_temporal_diagnostics(temporal_std: np.ndarray) -> dict[str, np.ndarray]:
    std32 = np.asarray(temporal_std, dtype=np.float32)
    if std32.ndim != 4 or std32.shape[1] != 20 or std32.shape[2] != 32:
        raise ValueError(f"Unexpected temporal std shape: {std32.shape}")
    return {
        "temporal_std_global": std32.mean(axis=(1, 2, 3), dtype=np.float32),
        "temporal_std_by_state": std32.mean(axis=(2, 3), dtype=np.float32),
        "temporal_std_by_state_token": std32.mean(axis=3, dtype=np.float32),
        "temporal_std_by_state_feature": std32.mean(axis=2, dtype=np.float32),
    }


def encode_vjepa(
    *,
    video_path: Path,
    time_seconds: np.ndarray,
    weights_dir: Path,
    dtype: str,
    microbatch_size: int,
    compile_encoder: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    decode_started = time.perf_counter()
    decoded = _decode_video_grid_ffmpeg(video_path, fps=DECODE_HZ, image_size=IMAGE_SIZE)
    decode_seconds = time.perf_counter() - decode_started
    model = MlxVjepa21FeatureModel(
        str(weights_dir),
        IMAGE_SIZE,
        compile_encoder=compile_encoder,
        input_dtype=dtype,
    )
    selected = sparse.selected_vitg_hidden_indices()
    feature_parts: list[np.ndarray] = []
    diagnostic_parts: dict[str, list[np.ndarray]] = {
        "temporal_std_global": [],
        "temporal_std_by_state": [],
        "temporal_std_by_state_token": [],
        "temporal_std_by_state_feature": [],
    }
    forward_started = time.perf_counter()
    for start in range(0, len(time_seconds), microbatch_size):
        batch_times = time_seconds[start : start + microbatch_size]
        windows = np.stack(
            [
                _sample_decoded_grid(
                    decoded,
                    fps=DECODE_HZ,
                    times=sparse.sparse_window_frame_times(float(row_time)),
                )
                for row_time in batch_times
            ],
            axis=0,
        )
        states, _temporal_mean, temporal_std = model.predict_hidden_states_with_temporal_stats(
            windows,
            selected,
        )
        if not np.isfinite(states).all() or not np.isfinite(temporal_std).all():
            raise RuntimeError("V-JEPA 2.1 produced non-finite values")
        feature_parts.append(states.astype(np.float16))
        compact = compact_temporal_diagnostics(temporal_std)
        for key, value in compact.items():
            diagnostic_parts[key].append(value)
        done = min(start + len(batch_times), len(time_seconds))
        print(json.dumps({"stage": "vjepa21_mlx", "rows": f"{done}/{len(time_seconds)}"}), flush=True)
    forward_seconds = time.perf_counter() - forward_started
    features = np.concatenate(feature_parts, axis=0)
    diagnostics = {key: np.concatenate(parts, axis=0) for key, parts in diagnostic_parts.items()}
    del model, decoded
    mx.clear_cache()
    gc.collect()
    return features, diagnostics, {
        "decode_seconds": decode_seconds,
        "forward_seconds": forward_seconds,
    }


def run_tribe(
    *,
    features: np.ndarray,
    model_dir: Path,
    dtype: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    grouped = np.stack(
        [sparse.group_mean_vitg_layers(np.asarray(row, dtype=np.float32)) for row in features],
        axis=0,
    ).astype(np.float32)
    video = grouped.transpose(1, 2, 0)[None]
    started = time.perf_counter()
    encoder = MlxTribeEncoder(str(model_dir), dtype=dtype)
    cortical = encoder.predict({"video": video}, pool_outputs=False)
    forward_seconds = time.perf_counter() - started
    cortical_rows = np.asarray(cortical[0].T, dtype=np.float32)
    if cortical_rows.shape != (len(features), EXPECTED_CORTICAL_WIDTH):
        raise RuntimeError(f"Unexpected MLX TRIBE output shape: {cortical_rows.shape}")
    if not np.isfinite(cortical_rows).all():
        raise RuntimeError("MLX TRIBE produced non-finite values")
    del encoder
    mx.clear_cache()
    gc.collect()
    return grouped, cortical_rows, forward_seconds


def causal_trailing_mean_2s(cortical: np.ndarray) -> np.ndarray:
    rows = []
    for index in range(len(cortical)):
        rows.append(cortical[max(0, index - 3) : index + 1].mean(axis=0, dtype=np.float32))
    return np.stack(rows).astype(np.float32)


def diagnostic_matrix(diagnostics: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            diagnostics["temporal_std_global"].reshape(-1, 1),
            diagnostics["temporal_std_by_state"],
            diagnostics["temporal_std_by_state_token"].mean(axis=1, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def load_split_rows(dense_root: Path, stage0_root: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    row_path = dense_root / "labels_aligned_2hz.parquet"
    df = pd.read_parquet(row_path, columns=["video_id", "time_seconds"])
    _, split_manifest, _ = stage_a_runner.load_stage0(stage0_root)
    train_idx, locked_idx, _ = locked_runner.split_indices(df, split_manifest)
    return df, train_idx, locked_idx


def head_scaler_and_reproduction(
    *,
    dense_root: Path,
    stage0_root: Path,
    bridge_root: Path,
    skip_reproduction: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    df, train_idx, locked_idx = load_split_rows(dense_root, stage0_root)
    scores = np.load(
        bridge_root / "pca/features/locked_confirmation__temporal_mean_2s__scores_w256.npy",
        mmap_mode="r",
    )
    diagnostics = np.load(
        dense_root / "_derived/temporal_diagnostics_summary_features.npy",
        mmap_mode="r",
    )
    train_features = stage_a.build_video_features(
        df,
        train_idx,
        np.asarray(scores[: len(train_idx)], dtype=np.float32),
        np.asarray(diagnostics[train_idx], dtype=np.float32),
    )
    train_x = train_features.x_temporal
    mean = np.nanmean(train_x, axis=0).astype(np.float32)
    std = np.nanstd(train_x, axis=0).astype(np.float32)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    del train_x, train_features
    gc.collect()
    audit: dict[str, Any] = {
        "train_rows": int(len(train_idx)),
        "locked_rows": int(len(locked_idx)),
        "feature_width": int(len(mean)),
        "locked_reproduction_skipped": bool(skip_reproduction),
    }
    if skip_reproduction:
        return mean, std, audit
    locked_features = stage_a.build_video_features(
        df,
        locked_idx,
        np.asarray(scores[len(train_idx) :], dtype=np.float32),
        np.asarray(diagnostics[locked_idx], dtype=np.float32),
    )
    locked_x = ((np.nan_to_num(locked_features.x_temporal, nan=0.0) - mean) / std).astype(np.float32)
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        checkpoint = bridge_root / "models" / PRIMARY_LANE / f"seed{seed}.npz"
        model = stage_a.VideoScalarHead(locked_x.shape[1], temporal_context=True)
        _ = model(stage_a.mlx_base.mx.array(locked_x[:2], dtype=stage_a.mlx_base.mx.float32))
        model.load_weights(str(checkpoint))
        actual = stage_a._predict_scalar(model, locked_x, stage_a.BATCH_SIZE)
        sealed = pd.read_parquet(
            bridge_root / "predictions" / PRIMARY_LANE / f"seed{seed}.parquet",
            columns=["prediction"],
        )["prediction"].to_numpy(dtype=np.float32)
        delta = np.abs(actual - sealed)
        per_seed[str(seed)] = {
            "checkpoint_sha256": file_sha256(checkpoint),
            "max_abs_error": float(delta.max()),
            "mean_abs_error": float(delta.mean()),
            "passed": bool(float(delta.max()) <= 1e-6),
        }
        del model, actual, sealed, delta
    audit["per_seed"] = per_seed
    audit["passed"] = all(record["passed"] for record in per_seed.values())
    del locked_x, locked_features, scores, diagnostics
    mx.clear_cache()
    gc.collect()
    if not audit["passed"]:
        raise RuntimeError(f"Frozen head reproduction failed: {audit}")
    return mean, std, audit


def bridge_predict(
    *,
    cortical_rows: np.ndarray,
    diagnostics: dict[str, np.ndarray],
    time_seconds: np.ndarray,
    bridge_root: Path,
    head_mean: np.ndarray,
    head_std: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, str]]:
    with np.load(
        bridge_root / "pca/pca_components/locked_confirmation__temporal_mean_2s__components_w256.npz"
    ) as pca:
        components = np.asarray(pca["components"], dtype=np.float32)
        pca_mean = np.asarray(pca["mean"], dtype=np.float32)
        pca_std = np.asarray(pca["std"], dtype=np.float32)
    base = causal_trailing_mean_2s(cortical_rows)
    pca_scores = ((base - pca_mean) / pca_std) @ components.T
    df = pd.DataFrame({"video_id": ["beat_unseen"] * len(time_seconds), "time_seconds": time_seconds})
    features = stage_a.build_video_features(
        df,
        np.arange(len(df), dtype=np.int64),
        pca_scores.astype(np.float32),
        diagnostic_matrix(diagnostics),
    )
    x = ((np.nan_to_num(features.x_temporal, nan=0.0) - head_mean) / head_std).astype(np.float32)
    members = []
    checkpoint_hashes: dict[str, str] = {}
    for seed in SEEDS:
        checkpoint = bridge_root / "models" / PRIMARY_LANE / f"seed{seed}.npz"
        model = stage_a.VideoScalarHead(x.shape[1], temporal_context=True)
        _ = model(stage_a.mlx_base.mx.array(x[:2], dtype=stage_a.mlx_base.mx.float32))
        model.load_weights(str(checkpoint))
        members.append(stage_a._predict_scalar(model, x, stage_a.BATCH_SIZE))
        checkpoint_hashes[str(seed)] = file_sha256(checkpoint)
        del model
    member_matrix = np.stack(members, axis=1).astype(np.float32)
    prediction = member_matrix.mean(axis=1, dtype=np.float32)
    ranks = pd.Series(prediction).rank(method="average", pct=True).to_numpy(dtype=np.float32)
    top_count = max(1, int(math.ceil(0.05 * len(prediction))))
    top_idx = np.argsort(prediction)[-top_count:]
    spike = np.zeros(len(prediction), dtype=bool)
    spike[top_idx] = True
    result = pd.DataFrame(
        {
            "time_seconds": time_seconds,
            "future_arousal_movement_score": prediction,
            "within_video_percentile": ranks,
            "relative_top_5pct_spike_candidate": spike,
            **{
                f"member_seed_{seed}": member_matrix[:, index]
                for index, seed in enumerate(SEEDS)
            },
        }
    )
    return result, checkpoint_hashes


def main() -> int:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    external_root = args.external_root.expanduser().resolve()
    dense_root = args.dense_root.expanduser().resolve()
    stage0_root = args.stage0_root.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is non-empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    vjepa_dir = external_root / "models/vjepa21_mlx/vitg"
    tribe_dir = external_root / "models/tribe-mlx/zimengxiong-tribev2-mlx"
    bridge_root = external_root / "outputs/again_dense_2hz_zero_label_direct_supervised_locked_confirm_20260715"
    started = time.perf_counter()
    duration = ffprobe_duration(video_path)
    row_count = int(round(duration * ROW_HZ))
    if row_count < 5:
        raise ValueError("Video is too short for the five-row Neural Bridge context")
    if row_count > 1024:
        raise ValueError("This smoke path requires <=1024 rows (<=512 seconds)")
    time_seconds = np.arange(1, row_count + 1, dtype=np.float32) / ROW_HZ
    features, diagnostics, vjepa_runtime = encode_vjepa(
        video_path=video_path,
        time_seconds=time_seconds,
        weights_dir=vjepa_dir,
        dtype=args.dtype,
        microbatch_size=args.microbatch_size,
        compile_encoder=args.compile_vjepa,
    )
    np.savez_compressed(
        output_root / "vjepa21_cache_compact.npz",
        features=features,
        time_seconds=time_seconds,
        selected_state_indices=np.asarray(sparse.selected_vitg_hidden_indices(), dtype=np.int16),
        **{key: value.astype(np.float16) for key, value in diagnostics.items()},
    )
    grouped, cortical_rows, tribe_seconds = run_tribe(
        features=features,
        model_dir=tribe_dir,
        dtype=args.dtype,
    )
    np.savez_compressed(
        output_root / "tribe_v2_cortical_predictions.npz",
        cortical_prediction=cortical_rows.astype(np.float16),
        time_seconds=time_seconds,
        tribe_grouped_video_feature=grouped.astype(np.float16),
    )
    head_mean, head_std, reproduction = head_scaler_and_reproduction(
        dense_root=dense_root,
        stage0_root=stage0_root,
        bridge_root=bridge_root,
        skip_reproduction=args.skip_locked_reproduction,
    )
    np.savez(output_root / "reconstructed_frozen_head_scaler.npz", mean=head_mean, std=head_std)
    predictions, checkpoint_hashes = bridge_predict(
        cortical_rows=cortical_rows,
        diagnostics=diagnostics,
        time_seconds=time_seconds,
        bridge_root=bridge_root,
        head_mean=head_mean,
        head_std=head_std,
    )
    predictions.to_csv(output_root / "neural_bridge_predictions.csv", index=False)
    top = predictions.nlargest(min(10, len(predictions)), "future_arousal_movement_score")
    summary = {
        "schema_version": "beat_single_video_neural_bridge_mlx_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "experimental_cross_domain_unlabeled_video_inference",
        "validation_status": "not_external_validation_without_labels",
        "input_video": str(video_path),
        "input_sha256": file_sha256(video_path),
        "duration_seconds": duration,
        "row_hz": ROW_HZ,
        "row_count": row_count,
        "modalities": ["video"],
        "vjepa_backend": "mlx_vjepa_2_1_vitg",
        "vjepa_weights": str(vjepa_dir),
        "vjepa_weights_sha256": file_sha256(vjepa_dir / "model.safetensors"),
        "vjepa_feature_shape": list(features.shape),
        "temporal_diagnostics_width": 53,
        "tribe_backend": "mlx_tribe_v2",
        "tribe_weights": str(tribe_dir / f"tribev2_mlx_{args.dtype}.npz"),
        "tribe_pool_outputs": False,
        "cortical_prediction_shape": list(cortical_rows.shape),
        "neural_bridge_lane": PRIMARY_LANE,
        "neural_bridge_target": "future_arousal_max_delta_rows_4_10",
        "neural_bridge_interpretation": "relative future arousal-movement ranking 2-5 seconds ahead",
        "checkpoint_hashes": checkpoint_hashes,
        "locked_prediction_reproduction": reproduction,
        "relative_spike_policy": "within-video top 5 percent; provisional ranking marker, not calibrated threshold",
        "top_candidates": top[["time_seconds", "future_arousal_movement_score", "within_video_percentile"]].to_dict("records"),
        "runtime_seconds": {
            **vjepa_runtime,
            "tribe_forward_seconds": tribe_seconds,
            "total": time.perf_counter() - started,
        },
        "artifacts": {
            "vjepa": str(output_root / "vjepa21_cache_compact.npz"),
            "tribe": str(output_root / "tribe_v2_cortical_predictions.npz"),
            "head_scaler": str(output_root / "reconstructed_frozen_head_scaler.npz"),
            "predictions": str(output_root / "neural_bridge_predictions.csv"),
        },
    }
    write_json(output_root / "run_manifest.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
