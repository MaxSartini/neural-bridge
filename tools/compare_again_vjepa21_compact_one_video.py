#!/usr/bin/env python3
"""Compare one AGAIN H100 cache with paired full-stat and compact MLX encodes."""

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
from backend.app.services.mlx_vjepa21_cortical import MlxVjepa21FeatureModel  # noqa: E402
from backend.scripts import again_sparse_tribe_teacher_500 as sparse  # noqa: E402
from tools.run_single_video_neural_bridge_mlx import (  # noqa: E402
    bridge_predict,
    head_scaler_and_reproduction,
)


IMAGE_SIZE = 256
DECODE_HZ = 16.0
EXPECTED_STATES = 20
EXPECTED_TUBELETS = 32
EXPECTED_HIDDEN = 1408
EXPECTED_CORTICAL = 20_484


def parse_args() -> argparse.Namespace:
    external = Path(
        os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT", str(REPO_ROOT))
    ).expanduser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--reference-dense", type=Path, required=True)
    parser.add_argument("--reference-postpass", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, default=external)
    parser.add_argument(
        "--dense-root",
        type=Path,
        default=external / "cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz",
    )
    parser.add_argument(
        "--stage0-root",
        type=Path,
        default=REPO_ROOT / "evidence/zero_label_video_only_deployment_stage0_20260714",
    )
    parser.add_argument(
        "--bridge-root",
        type=Path,
        default=external / "outputs/again_dense_2hz_zero_label_direct_supervised_locked_confirm_20260715",
    )
    parser.add_argument("--microbenchmark-repetitions", type=int, default=3)
    parser.add_argument("--skip-bridge", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def decode_video_software(path: Path) -> np.ndarray:
    short_side = int(256.0 / 224.0 * IMAGE_SIZE)
    video_filter = (
        f"fps={DECODE_HZ:.8f},"
        f"scale='if(gt(iw,ih),-2,{short_side})':'if(gt(iw,ih),{short_side},-2)',"
        f"crop={IMAGE_SIZE}:{IMAGE_SIZE}"
    )
    command = [
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
    ]
    result = subprocess.run(command, check=False, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    frame_bytes = IMAGE_SIZE * IMAGE_SIZE * 3
    if len(result.stdout) < frame_bytes or len(result.stdout) % frame_bytes:
        raise RuntimeError(f"Invalid decoded byte count: {len(result.stdout)}")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        -1, IMAGE_SIZE, IMAGE_SIZE, 3
    ).copy()


def h100_float16_preprocess(windows: np.ndarray) -> np.ndarray:
    """Match the retained H100 cache's float16 normalization arithmetic."""
    array = np.asarray(windows)
    if array.ndim != 5 or tuple(array.shape[2:]) != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Unexpected window shape: {array.shape}")
    normalized = array.astype(np.float16) / np.float16(255.0)
    normalized = np.transpose(normalized, (0, 4, 1, 2, 3))
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float16)[None, :, None, None, None]
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float16)[None, :, None, None, None]
    return ((normalized - mean) / std).astype(np.float16, copy=False)


def canonical_diag53_from_cached_std(temporal_std: np.ndarray) -> np.ndarray:
    """Reproduce dense-f16 -> H100 postpass-f16 -> benchmark-f32 diagnostics."""
    cached = np.asarray(temporal_std, dtype=np.float16).astype(np.float32)
    if cached.ndim != 4 or tuple(cached.shape[1:3]) != (EXPECTED_STATES, EXPECTED_TUBELETS):
        raise ValueError(f"Unexpected temporal std shape: {cached.shape}")
    global_std = cached.mean(axis=(1, 2, 3), dtype=np.float32)[:, None]
    by_state = cached.mean(axis=(2, 3), dtype=np.float32).astype(np.float16).astype(np.float32)
    by_state_token = cached.mean(axis=3, dtype=np.float32).astype(np.float16).astype(np.float32)
    by_token = by_state_token.mean(axis=1, dtype=np.float32)
    return np.concatenate((global_std, by_state, by_token), axis=1).astype(np.float32)


def diag53_from_postpass(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as bundle:
        global_std = np.asarray(bundle["temporal_std_global"], dtype=np.float32)[:, None]
        by_state = np.asarray(bundle["temporal_std_by_state"], dtype=np.float32)
        by_state_token = np.asarray(bundle["temporal_std_by_state_token"], dtype=np.float32)
    by_token = by_state_token.mean(axis=1, dtype=np.float32)
    return np.concatenate((global_std, by_state, by_token), axis=1).astype(np.float32)


def full_forward(
    model: MlxVjepa21FeatureModel,
    preprocessed: np.ndarray,
    selected: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    video = mx.array(preprocessed, dtype=mx.float16)
    states, temporal_mean, temporal_std = model.encoder.selected_states_with_temporal_stats(
        video, selected
    )
    states = states.astype(mx.float16)
    temporal_mean = temporal_mean.astype(mx.float16)
    temporal_std = temporal_std.astype(mx.float16)
    mx.eval(states, temporal_mean, temporal_std)
    states_np = np.asarray(states).copy()
    mean_np = np.asarray(temporal_mean).copy()
    std_np = np.asarray(temporal_std).copy()
    return states_np, mean_np, std_np, canonical_diag53_from_cached_std(std_np)


def compact_forward(
    model: MlxVjepa21FeatureModel,
    preprocessed: np.ndarray,
    selected: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    video = mx.array(preprocessed, dtype=mx.float16)
    states, diagnostics = model.encoder.selected_states_with_compact_temporal_diagnostics(
        video, selected
    )
    states = states.astype(mx.float16)
    diagnostics = diagnostics.astype(mx.float32)
    mx.eval(states, diagnostics)
    return np.asarray(states).copy(), np.asarray(diagnostics).copy()


def group_features(features: np.ndarray) -> np.ndarray:
    rows = [sparse.group_mean_vitg_layers(np.asarray(row, dtype=np.float32)) for row in features]
    return np.stack(rows, axis=0).astype(np.float16)


def mlx_tribe_predict(encoder: MlxTribeEncoder, grouped: np.ndarray) -> tuple[np.ndarray, float]:
    video = np.asarray(grouped, dtype=np.float32).transpose(1, 2, 0)[None]
    started = time.perf_counter()
    prediction = encoder.predict({"video": video}, pool_outputs=False)
    cortical = np.asarray(prediction[0].T, dtype=np.float32)
    elapsed = time.perf_counter() - started
    if cortical.shape != (len(grouped), EXPECTED_CORTICAL):
        raise RuntimeError(f"Unexpected TRIBE output shape: {cortical.shape}")
    return cortical, elapsed


def array_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference64 = np.asarray(reference, dtype=np.float64)
    candidate64 = np.asarray(candidate, dtype=np.float64)
    if reference64.shape != candidate64.shape:
        return {"shape_match": False, "reference_shape": list(reference64.shape), "candidate_shape": list(candidate64.shape)}
    delta = candidate64 - reference64
    flat_reference = reference64.reshape(-1)
    flat_candidate = candidate64.reshape(-1)
    rmse = float(np.sqrt(np.mean(np.square(delta))))
    reference_std = float(np.std(flat_reference))
    denominator = float(np.linalg.norm(flat_reference) * np.linalg.norm(flat_candidate))
    cosine = float(np.dot(flat_reference, flat_candidate) / denominator) if denominator else 1.0
    return {
        "shape_match": True,
        "shape": list(reference64.shape),
        "finite": bool(np.isfinite(reference64).all() and np.isfinite(candidate64).all()),
        "exact_fraction": float(np.mean(reference64 == candidate64)),
        "max_abs_error": float(np.max(np.abs(delta))),
        "mean_abs_error": float(np.mean(np.abs(delta))),
        "rmse": rmse,
        "normalized_rmse": float(rmse / max(reference_std, 1e-12)),
        "cosine": cosine,
    }


def score_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    metrics = array_metrics(reference, candidate)
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    top_count = max(1, int(math.ceil(0.05 * len(ref))))
    ref_top = set(np.argsort(ref)[-top_count:].tolist())
    cand_top = set(np.argsort(cand)[-top_count:].tolist())
    metrics.update(
        {
            "spearman": float(pd.Series(ref).corr(pd.Series(cand), method="spearman")),
            "top5_count": top_count,
            "top5_identical": ref_top == cand_top,
            "top5_overlap": len(ref_top & cand_top),
        }
    )
    return metrics


def diag_dict_from_53(diagnostics: np.ndarray) -> dict[str, np.ndarray]:
    diagnostics = np.asarray(diagnostics, dtype=np.float32)
    by_token = diagnostics[:, 21:]
    return {
        "temporal_std_global": diagnostics[:, 0],
        "temporal_std_by_state": diagnostics[:, 1:21],
        "temporal_std_by_state_token": np.repeat(by_token[:, None, :], EXPECTED_STATES, axis=1),
    }


def bridge_scores(
    *,
    cortical: np.ndarray,
    diagnostics: np.ndarray,
    time_seconds: np.ndarray,
    bridge_root: Path,
    head_mean: np.ndarray,
    head_std: np.ndarray,
) -> pd.DataFrame:
    predictions, _hashes = bridge_predict(
        cortical_rows=np.asarray(cortical, dtype=np.float32),
        diagnostics=diag_dict_from_53(diagnostics),
        time_seconds=time_seconds,
        bridge_root=bridge_root,
        head_mean=head_mean,
        head_std=head_std,
    )
    return predictions


def ensure_fresh_outputs(output_root: Path) -> dict[str, Path]:
    paths = {
        "full": output_root / "legacy_full_mlx",
        "compact": output_root / "compact_mlx",
        "h100_mlx_tribe": output_root / "fresh_mlx_tribe_from_h100",
    }
    for path in paths.values():
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"Refusing to replace non-empty output lane: {path}")
        path.mkdir(parents=True, exist_ok=True)
    return paths


def main() -> int:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    reference_dense = args.reference_dense.expanduser().resolve()
    reference_postpass = args.reference_postpass.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    external_root = args.external_root.expanduser().resolve()
    dense_root = args.dense_root.expanduser().resolve()
    stage0_root = args.stage0_root.expanduser().resolve()
    bridge_root = args.bridge_root.expanduser().resolve()
    if args.microbenchmark_repetitions < 1:
        raise ValueError("microbenchmark repetitions must be positive")
    for path in (video_path, reference_dense):
        if not path.is_file():
            raise FileNotFoundError(path)
    lane_paths = ensure_fresh_outputs(output_root)

    with np.load(reference_dense, allow_pickle=False) as dense:
        reference_features = np.asarray(dense["features"], dtype=np.float16)
        reference_std = np.asarray(dense["temporal_std"], dtype=np.float16)
        time_seconds = np.asarray(dense["time_seconds"], dtype=np.float32)
        sample_frame_indices = np.asarray(dense["sample_frame_indices"], dtype=np.int32)
        sample_time_seconds = np.asarray(dense["sample_time_seconds"], dtype=np.float32)
        selected_indices = np.asarray(dense["selected_state_indices"], dtype=np.int16)
    reference_diag = diag53_from_postpass(reference_postpass / "vjepa_temporal_diagnostics.npz")
    reference_dense_diag = canonical_diag53_from_cached_std(reference_std)
    del reference_std

    with np.load(reference_postpass / "tribe_v2_cortical_predictions.npz", allow_pickle=False) as post:
        reference_cortical = np.asarray(post["cortical_prediction"], dtype=np.float16)
        reference_grouped = np.asarray(post["tribe_grouped_video_feature"], dtype=np.float16)
        reference_post_times = np.asarray(post["time_seconds"], dtype=np.float32)

    if not np.array_equal(time_seconds, reference_post_times):
        raise RuntimeError("Dense and postpass timestamps differ")
    if tuple(reference_features.shape[1:]) != (EXPECTED_STATES, 1, EXPECTED_HIDDEN):
        raise RuntimeError(f"Unexpected reference feature shape: {reference_features.shape}")
    if sample_frame_indices.shape != (len(time_seconds), 64):
        raise RuntimeError(f"Unexpected sample index shape: {sample_frame_indices.shape}")

    decode_started = time.perf_counter()
    decoded = decode_video_software(video_path)
    decode_seconds = time.perf_counter() - decode_started
    if int(sample_frame_indices.max()) >= len(decoded):
        raise RuntimeError("Reference sample index exceeds decoded frame grid")
    decoded_frame_count = int(len(decoded))
    decoded_sha256 = hashlib.sha256(decoded.tobytes()).hexdigest()

    weights_dir = external_root / "models/vjepa21_mlx/vitg"
    model_load_started = time.perf_counter()
    model = MlxVjepa21FeatureModel(str(weights_dir), IMAGE_SIZE, input_dtype="float16")
    model_load_seconds = time.perf_counter() - model_load_started
    selected = [int(value) for value in selected_indices]

    first_windows = decoded[sample_frame_indices[:1]]
    first_preprocessed = h100_float16_preprocess(first_windows)
    warm_full_started = time.perf_counter()
    warm_full = full_forward(model, first_preprocessed, selected)
    warm_full_seconds = time.perf_counter() - warm_full_started
    del warm_full
    warm_compact_started = time.perf_counter()
    warm_compact = compact_forward(model, first_preprocessed, selected)
    warm_compact_seconds = time.perf_counter() - warm_compact_started
    del warm_compact
    gc.collect()

    micro_full: list[float] = []
    micro_compact: list[float] = []
    for repetition in range(args.microbenchmark_repetitions):
        order = ("full", "compact") if repetition % 2 == 0 else ("compact", "full")
        for lane in order:
            started = time.perf_counter()
            result = (
                full_forward(model, first_preprocessed, selected)
                if lane == "full"
                else compact_forward(model, first_preprocessed, selected)
            )
            elapsed = time.perf_counter() - started
            (micro_full if lane == "full" else micro_compact).append(elapsed)
            del result

    full_features_parts: list[np.ndarray] = []
    full_mean_parts: list[np.ndarray] = []
    full_std_parts: list[np.ndarray] = []
    full_diag_parts: list[np.ndarray] = []
    compact_features_parts: list[np.ndarray] = []
    compact_diag_parts: list[np.ndarray] = []
    full_seconds = 0.0
    compact_seconds = 0.0
    sampling_preprocess_seconds = 0.0
    for row_index in range(len(time_seconds)):
        sample_started = time.perf_counter()
        windows = decoded[sample_frame_indices[row_index : row_index + 1]]
        preprocessed = h100_float16_preprocess(windows)
        sampling_preprocess_seconds += time.perf_counter() - sample_started
        order = ("full", "compact") if row_index % 2 == 0 else ("compact", "full")
        for lane in order:
            started = time.perf_counter()
            if lane == "full":
                states, temporal_mean, temporal_std, diagnostics = full_forward(
                    model, preprocessed, selected
                )
                full_seconds += time.perf_counter() - started
                full_features_parts.append(states)
                full_mean_parts.append(temporal_mean)
                full_std_parts.append(temporal_std)
                full_diag_parts.append(diagnostics)
            else:
                states, diagnostics = compact_forward(model, preprocessed, selected)
                compact_seconds += time.perf_counter() - started
                compact_features_parts.append(states)
                compact_diag_parts.append(diagnostics)
        if (row_index + 1) % 5 == 0 or row_index + 1 == len(time_seconds):
            print(
                json.dumps(
                    {
                        "stage": "paired_encode",
                        "rows": f"{row_index + 1}/{len(time_seconds)}",
                        "legacy_full_seconds": round(full_seconds, 3),
                        "compact_seconds": round(compact_seconds, 3),
                    }
                ),
                flush=True,
            )

    full_features = np.concatenate(full_features_parts, axis=0)
    full_mean = np.concatenate(full_mean_parts, axis=0)
    full_std = np.concatenate(full_std_parts, axis=0)
    full_diag = np.concatenate(full_diag_parts, axis=0)
    compact_features = np.concatenate(compact_features_parts, axis=0)
    compact_diag = np.concatenate(compact_diag_parts, axis=0)
    del full_features_parts, full_mean_parts, full_std_parts, full_diag_parts
    del compact_features_parts, compact_diag_parts, model, decoded
    mx.clear_cache()
    gc.collect()

    full_write_started = time.perf_counter()
    full_cache_path = lane_paths["full"] / "vjepa21_cache_full.npz"
    np.savez_compressed(
        full_cache_path,
        features=full_features,
        temporal_mean=full_mean,
        temporal_std=full_std,
        time_seconds=time_seconds,
        sample_frame_indices=sample_frame_indices,
        sample_time_seconds=sample_time_seconds,
        selected_state_indices=selected_indices,
    )
    full_write_seconds = time.perf_counter() - full_write_started
    compact_write_started = time.perf_counter()
    compact_cache_path = lane_paths["compact"] / "vjepa21_cache_compact.npz"
    np.savez_compressed(
        compact_cache_path,
        features=compact_features,
        temporal_diagnostics53=compact_diag,
        time_seconds=time_seconds,
        sample_frame_indices=sample_frame_indices,
        sample_time_seconds=sample_time_seconds,
        selected_state_indices=selected_indices,
    )
    compact_write_seconds = time.perf_counter() - compact_write_started

    full_grouped = group_features(full_features)
    compact_grouped = group_features(compact_features)
    regrouped_reference = group_features(reference_features)
    tribe_model_dir = external_root / "models/tribe-mlx/zimengxiong-tribev2-mlx"
    tribe_load_started = time.perf_counter()
    tribe_encoder = MlxTribeEncoder(str(tribe_model_dir), dtype="float16")
    tribe_load_seconds = time.perf_counter() - tribe_load_started
    h100_to_mlx_cortical, h100_to_mlx_tribe_seconds = mlx_tribe_predict(
        tribe_encoder, reference_grouped
    )
    full_cortical, full_tribe_seconds = mlx_tribe_predict(tribe_encoder, full_grouped)
    compact_cortical, compact_tribe_seconds = mlx_tribe_predict(tribe_encoder, compact_grouped)
    del tribe_encoder
    mx.clear_cache()
    gc.collect()

    np.savez_compressed(
        lane_paths["h100_mlx_tribe"] / "tribe_v2_cortical_predictions.npz",
        cortical_prediction=h100_to_mlx_cortical.astype(np.float16),
        tribe_grouped_video_feature=reference_grouped,
        time_seconds=time_seconds,
    )
    np.savez_compressed(
        lane_paths["full"] / "tribe_v2_cortical_predictions.npz",
        cortical_prediction=full_cortical.astype(np.float16),
        tribe_grouped_video_feature=full_grouped,
        time_seconds=time_seconds,
    )
    np.savez_compressed(
        lane_paths["compact"] / "tribe_v2_cortical_predictions.npz",
        cortical_prediction=compact_cortical.astype(np.float16),
        tribe_grouped_video_feature=compact_grouped,
        time_seconds=time_seconds,
    )

    comparisons = {
        "reference_dense_diag_vs_reference_postpass_diag": array_metrics(reference_diag, reference_dense_diag),
        "reference_features_regrouped_vs_reference_postpass_grouped": array_metrics(reference_grouped, regrouped_reference),
        "legacy_full_vs_compact_features": array_metrics(full_features, compact_features),
        "legacy_full_vs_compact_diag53": array_metrics(full_diag, compact_diag),
        "legacy_full_vs_compact_grouped": array_metrics(full_grouped, compact_grouped),
        "legacy_full_vs_compact_cortical": array_metrics(full_cortical, compact_cortical),
        "h100_vs_legacy_full_features": array_metrics(reference_features, full_features),
        "h100_vs_compact_features": array_metrics(reference_features, compact_features),
        "h100_vs_legacy_full_diag53": array_metrics(reference_diag, full_diag),
        "h100_vs_compact_diag53": array_metrics(reference_diag, compact_diag),
        "h100_postpass_vs_mlx_tribe_on_h100_features": array_metrics(
            reference_cortical, h100_to_mlx_cortical
        ),
        "h100_postpass_vs_compact_end_to_end_cortical": array_metrics(
            reference_cortical, compact_cortical
        ),
    }

    bridge_audit: dict[str, Any] = {"skipped": bool(args.skip_bridge)}
    if not args.skip_bridge:
        bridge_started = time.perf_counter()
        try:
            head_mean, head_std, scaler_audit = head_scaler_and_reproduction(
                dense_root=dense_root,
                stage0_root=stage0_root,
                bridge_root=bridge_root,
                skip_reproduction=True,
            )
            bridge_frames = {
                "h100_original": bridge_scores(
                    cortical=reference_cortical,
                    diagnostics=reference_diag,
                    time_seconds=time_seconds,
                    bridge_root=bridge_root,
                    head_mean=head_mean,
                    head_std=head_std,
                ),
                "h100_features_mlx_tribe": bridge_scores(
                    cortical=h100_to_mlx_cortical,
                    diagnostics=reference_diag,
                    time_seconds=time_seconds,
                    bridge_root=bridge_root,
                    head_mean=head_mean,
                    head_std=head_std,
                ),
                "legacy_full_mlx": bridge_scores(
                    cortical=full_cortical,
                    diagnostics=full_diag,
                    time_seconds=time_seconds,
                    bridge_root=bridge_root,
                    head_mean=head_mean,
                    head_std=head_std,
                ),
                "compact_mlx": bridge_scores(
                    cortical=compact_cortical,
                    diagnostics=compact_diag,
                    time_seconds=time_seconds,
                    bridge_root=bridge_root,
                    head_mean=head_mean,
                    head_std=head_std,
                ),
            }
            for name, frame in bridge_frames.items():
                frame.to_csv(output_root / f"bridge_predictions_{name}.csv", index=False)
            score_column = "future_arousal_movement_score"
            comparisons["legacy_full_vs_compact_bridge_score"] = score_metrics(
                bridge_frames["legacy_full_mlx"][score_column].to_numpy(),
                bridge_frames["compact_mlx"][score_column].to_numpy(),
            )
            comparisons["h100_vs_compact_bridge_score"] = score_metrics(
                bridge_frames["h100_original"][score_column].to_numpy(),
                bridge_frames["compact_mlx"][score_column].to_numpy(),
            )
            comparisons["h100_postpass_vs_mlx_tribe_bridge_score"] = score_metrics(
                bridge_frames["h100_original"][score_column].to_numpy(),
                bridge_frames["h100_features_mlx_tribe"][score_column].to_numpy(),
            )
            bridge_audit = {
                "skipped": False,
                "passed": True,
                "scaler": scaler_audit,
                "seconds": time.perf_counter() - bridge_started,
            }
        except Exception as exc:  # Preserve the completed encoder comparison if head replay fails.
            bridge_audit = {
                "skipped": False,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "seconds": time.perf_counter() - bridge_started,
            }

    feature_match = comparisons["legacy_full_vs_compact_features"]
    diag_match = comparisons["legacy_full_vs_compact_diag53"]
    grouped_match = comparisons["legacy_full_vs_compact_grouped"]
    cortical_match = comparisons["legacy_full_vs_compact_cortical"]
    gates = {
        "same_timestamps": bool(np.array_equal(time_seconds, reference_post_times)),
        "full_vs_compact_features": bool(
            feature_match["max_abs_error"] <= 5e-4 and feature_match["cosine"] >= 0.999999
        ),
        "full_vs_compact_diag53": bool(
            np.allclose(full_diag, compact_diag, rtol=1e-5, atol=1e-6)
        ),
        "full_vs_compact_grouped": bool(
            grouped_match["max_abs_error"] <= 5e-4 and grouped_match["cosine"] >= 0.999999
        ),
        "full_vs_compact_cortical": bool(
            np.allclose(full_cortical, compact_cortical, rtol=1e-5, atol=1e-5)
        ),
    }
    if "legacy_full_vs_compact_bridge_score" in comparisons:
        score_match = comparisons["legacy_full_vs_compact_bridge_score"]
        gates["full_vs_compact_bridge_score"] = bool(
            score_match["max_abs_error"] <= 1e-6
            and score_match["spearman"] >= 0.999999
            and score_match["top5_identical"]
        )
    gates["compaction_preserves_outputs"] = all(gates.values())

    micro_full_median = float(np.median(micro_full))
    micro_compact_median = float(np.median(micro_compact))
    full_operational = full_seconds + full_write_seconds
    compact_operational = compact_seconds + compact_write_seconds
    runtime = {
        "shared_decode_seconds": decode_seconds,
        "shared_model_load_seconds": model_load_seconds,
        "shared_sampling_preprocess_seconds": sampling_preprocess_seconds,
        "warmup_full_seconds": warm_full_seconds,
        "warmup_compact_seconds": warm_compact_seconds,
        "microbenchmark_full_seconds": micro_full,
        "microbenchmark_compact_seconds": micro_compact,
        "microbenchmark_full_median_seconds": micro_full_median,
        "microbenchmark_compact_median_seconds": micro_compact_median,
        "microbenchmark_speedup_full_over_compact": micro_full_median / micro_compact_median,
        "full_video_full_compute_seconds": full_seconds,
        "full_video_compact_compute_seconds": compact_seconds,
        "full_video_compute_speedup_full_over_compact": full_seconds / compact_seconds,
        "full_cache_write_seconds": full_write_seconds,
        "compact_cache_write_seconds": compact_write_seconds,
        "full_operational_compute_plus_write_seconds": full_operational,
        "compact_operational_compute_plus_write_seconds": compact_operational,
        "operational_speedup_full_over_compact": full_operational / compact_operational,
        "tribe_model_load_seconds": tribe_load_seconds,
        "tribe_h100_features_seconds": h100_to_mlx_tribe_seconds,
        "tribe_full_features_seconds": full_tribe_seconds,
        "tribe_compact_features_seconds": compact_tribe_seconds,
        "reference_h100_forward_seconds": 51.68305730819702,
        "reference_h100_microbatch_size": 64,
        "mac_microbatch_size": 1,
    }
    report = {
        "schema_version": "again_vjepa21_compact_one_video_comparison_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "video": str(video_path),
        "video_sha256": file_sha256(video_path),
        "reference_dense": str(reference_dense),
        "reference_dense_sha256": file_sha256(reference_dense),
        "reference_postpass": str(reference_postpass),
        "row_count": int(len(time_seconds)),
        "time_start": float(time_seconds[0]),
        "time_end": float(time_seconds[-1]),
        "row_hz": 2.0,
        "selected_state_indices": selected,
        "decoded_grid_frame_count": decoded_frame_count,
        "reference_max_sample_frame_index": int(sample_frame_indices.max()),
        "decoded_rgb_grid_sha256": decoded_sha256,
        "preprocessing": "H100-compatible software ffmpeg + exact stored frame indices + float16 normalization",
        "runtime": runtime,
        "cache_bytes": {
            "reference_h100_dense": reference_dense.stat().st_size,
            "fresh_full_mlx": full_cache_path.stat().st_size,
            "fresh_compact_mlx": compact_cache_path.stat().st_size,
        },
        "comparisons": comparisons,
        "gates": gates,
        "bridge_audit": bridge_audit,
        "artifacts": {
            "fresh_full_cache": str(full_cache_path),
            "fresh_compact_cache": str(compact_cache_path),
            "report_json": str(output_root / "comparison_report.json"),
            "report_markdown": str(output_root / "comparison_report.md"),
        },
        "interpretation_boundary": (
            "Full-vs-compact isolates compaction on identical Mac frames and preprocessing. "
            "H100-vs-MLX additionally includes encoder implementation/backend differences."
        ),
    }
    write_json(output_root / "comparison_report.json", report)
    markdown = f"""# AGAIN one-video V-JEPA 2.1 compact-cache comparison

- Video: `{video_path.name}`
- Rows: `{len(time_seconds)}` at 2 Hz (`{time_seconds[0]:.1f}`–`{time_seconds[-1]:.1f}` s)
- Original caches modified: **no**
- Compaction preservation gate: **{gates['compaction_preserves_outputs']}**

## Runtime

- Full-stat MLX compute: `{full_seconds:.3f}` s
- Compact MLX compute: `{compact_seconds:.3f}` s
- Compute speedup: `{full_seconds / compact_seconds:.3f}x`
- Full compute + write: `{full_operational:.3f}` s
- Compact compute + write: `{compact_operational:.3f}` s
- Operational speedup: `{full_operational / compact_operational:.3f}x`
- Original H100 forward (microbatch 64): `51.683` s

## Exact isolation: fresh full-stat MLX vs fresh compact MLX

- Selected-state max absolute error: `{feature_match['max_abs_error']:.9g}`
- Diagnostic-53 max absolute error: `{diag_match['max_abs_error']:.9g}`
- Grouped TRIBE-input max absolute error: `{grouped_match['max_abs_error']:.9g}`
- Cortical-output max absolute error: `{cortical_match['max_abs_error']:.9g}`

See `comparison_report.json` for the separate H100↔MLX backend comparison and every metric.
"""
    (output_root / "comparison_report.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if gates["compaction_preserves_outputs"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
