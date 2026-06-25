"""AGAIN sparse ViT-G/TRIBE teacher pilot.

This module runs a bounded 500-window sparse teacher pilot on AGAIN. It uses
the real V-JEPA 2.1 ViT-G MLX encoder and the converted TRIBE MLX cortical
head, but only on queued sparse causal windows. It does not dense-encode full
videos and does not touch VEATIC outputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import resource
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import mlx.core as mx
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from backend.app.services.mlx_tribe_encoder import MlxTribeEncoder
from backend.app.services.mlx_vjepa2_cortical import _decode_video_grid_ffmpeg, _sample_decoded_grid
from backend.app.services.mlx_vjepa21_cortical import MlxVjepa21FeatureModel
from backend.scripts.again_real_scout_selector_validation import (
    BUDGETS,
    SELECTOR_BASES,
    compute_label_vectors,
    score_rows,
    select_random_times_for_target_coverage,
    select_times_for_budget,
    selected_intervals,
)
from backend.scripts.again_scout_sparse_pipeline import (
    AGAIN_ALIGNMENT_POLICY,
    AGAIN_DATASET_NAME,
    assert_again_only_output_path,
    external_root,
    read_csv_rows,
    safe_float,
    safe_int,
    write_csv_rows,
    write_json,
)
from backend.scripts.mlx_runtime_config import configure_mlx_memory_from_env


SCHEMA_VERSION = "again_sparse_tribe_teacher_v2"
BENCHMARK_MODE = "again_sparse_vitg_tribe_teacher_small_pca_confirmatory"
SCOUT_VALIDATION_ROOT = Path("outputs/again_real_scout_selector_validation_20260621_230938_n50_covmatched")
SCOUT_VALIDATION_VIDEO_COUNT = 50
FULL_AGAIN_VIDEO_COUNT = 995
SMALL_PCA_WIDTH_CANDIDATES = (8, 16, 32, 64)
LOCKED_CONFIRMATORY_PCA_WIDTH = 32

CAUSAL_RELATIVE_SECONDS = (-2.0, -1.0, 0.0)
CLIP_DURATION_SECONDS = 4.0
FRAMES_PER_CLIP = 64
IMAGE_SIZE = 256
DECODE_FPS = FRAMES_PER_CLIP / CLIP_DURATION_SECONDS
VITG_CACHE_N_LAYERS = 20
VITG_GROUP_LAYERS = (0.5, 0.75, 1.0)
TRIBE_HEAD_POOL_POLICY = "mean_over_100_output_timesteps"

ARM_WINDOW_BUDGETS_500 = {
    "hybrid_top5_selected": 250,
    "coverage_matched_random_to_hybrid": 100,
    "oracle_upper_bound": 60,
    "low_salience_background": 60,
    "sparse_anchor_windows": 30,
}
ARM_WINDOW_BUDGETS_1000 = {
    "hybrid_top5_selected": 450,
    "coverage_matched_random_to_hybrid": 200,
    "fixed_random_same_budget": 120,
    "oracle_upper_bound": 100,
    "low_salience_background": 80,
    "sparse_anchor_windows": 50,
}
ARM_WINDOW_BUDGETS_2000 = {
    "hybrid_top5_selected": 900,
    "coverage_matched_random_to_hybrid": 400,
    "fixed_random_same_budget": 240,
    "oracle_upper_bound": 200,
    "low_salience_background": 160,
    "sparse_anchor_windows": 100,
}
ARM_WINDOW_BUDGETS = ARM_WINDOW_BUDGETS_500
ORACLE_EVALUATION_ARM = "oracle_upper_bound_with_background_controls"
EVALUATION_ARMS = tuple(dict.fromkeys([*ARM_WINDOW_BUDGETS_2000, *ARM_WINDOW_BUDGETS_1000, *ARM_WINDOW_BUDGETS_500, ORACLE_EVALUATION_ARM]))


@dataclass(frozen=True)
class SparseTeacherConfig:
    max_actual_windows: int = 500
    random_seed: int = 20260621
    selector_validation_root: Path = SCOUT_VALIDATION_ROOT
    image_size: int = IMAGE_SIZE
    frames_per_clip: int = FRAMES_PER_CLIP
    clip_duration_seconds: float = CLIP_DURATION_SECONDS
    report_date: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label: str = "again_sparse_tribe_teacher_500"
    run_title: str = "AGAIN Sparse TRIBE Teacher 500"
    arm_window_budgets: dict[str, int] | None = None
    causal_relative_seconds: tuple[float, ...] = CAUSAL_RELATIVE_SECONDS
    teacher_dtype: str = "bfloat16"
    allow_cache_backfill: bool = False
    gpu_workers: int = 1
    preprocess_workers: int = 0
    ready_queue_max_size: int = 4
    writer_queue_max_size: int = 4
    microbatch_size: int = 1
    compile_encoder: bool = True


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_teacher_dtype(dtype: str) -> str:
    if dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError(f"Unsupported teacher dtype: {dtype}")
    return dtype


def validate_single_gpu_runtime(
    *,
    gpu_workers: int,
    preprocess_workers: int,
    ready_queue_max_size: int,
    writer_queue_max_size: int,
    microbatch_size: int,
) -> dict[str, Any]:
    """Validate the Apple-Silicon-safe V-JEPA runtime shape."""
    if int(gpu_workers) != 1:
        raise ValueError(
            "V-JEPA 2.1 MLX extraction must use exactly one GPU owner on Apple Silicon; "
            f"received gpu_workers={gpu_workers}."
        )
    if int(preprocess_workers) < 0:
        raise ValueError("preprocess_workers must be >= 0")
    if int(ready_queue_max_size) < 1:
        raise ValueError("ready_queue_max_size must be >= 1")
    if int(writer_queue_max_size) < 1:
        raise ValueError("writer_queue_max_size must be >= 1")
    if int(microbatch_size) < 1:
        raise ValueError("microbatch_size must be >= 1")
    return {
        "gpu_workers": int(gpu_workers),
        "preprocess_workers": int(preprocess_workers),
        "ready_queue_max_size": int(ready_queue_max_size),
        "writer_queue_max_size": int(writer_queue_max_size),
        "microbatch_size": int(microbatch_size),
        "single_gpu_owner": True,
        "platform_machine": platform.machine(),
        "platform_system": platform.system(),
    }


def mlx_memory_snapshot() -> dict[str, int | None]:
    fields = {
        "mlx_active_memory_bytes": "get_active_memory",
        "mlx_peak_memory_bytes": "get_peak_memory",
        "mlx_cache_memory_bytes": "get_cache_memory",
    }
    snapshot: dict[str, int | None] = {}
    metal = getattr(mx, "metal", None)
    for field, name in fields.items():
        func = getattr(metal, name, None) if metal is not None else None
        if callable(func):
            try:
                snapshot[field] = int(func())
            except Exception:  # noqa: BLE001 - diagnostic only
                snapshot[field] = None
        else:
            snapshot[field] = None
    return snapshot


def vjepa21_vitg_dir_for_dtype(root: Path, dtype: str) -> Path:
    dtype = validate_teacher_dtype(dtype)
    if dtype == "bfloat16":
        for name in ("vitg_bfloat16_from_source", "vitg_bfloat16"):
            candidate = root / "models" / "vjepa21_mlx" / name
            if (candidate / "model.safetensors").exists():
                return candidate
        raise FileNotFoundError(root / "models" / "vjepa21_mlx" / "vitg_bfloat16_from_source" / "model.safetensors")
    if dtype == "float32":
        for name in ("vitg_float32_from_source", "vitg_float32"):
            candidate = root / "models" / "vjepa21_mlx" / name
            if (candidate / "model.safetensors").exists():
                return candidate
        raise FileNotFoundError(root / "models" / "vjepa21_mlx" / "vitg_float32_from_source" / "model.safetensors")
    for name in ("vitg_float16_from_source", "vitg"):
        candidate = root / "models" / "vjepa21_mlx" / name
        if (candidate / "model.safetensors").exists():
            return candidate
    return root / "models" / "vjepa21_mlx" / "vitg"


def tribe_weights_path_for_dtype(model_dir: Path, dtype: str) -> Path:
    dtype = validate_teacher_dtype(dtype)
    path = model_dir / f"tribev2_mlx_{dtype}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def causal_relative_seconds_for_actual_window_hz(actual_window_hz: float, *, lookback_seconds: float = 2.0) -> tuple[float, ...]:
    if actual_window_hz <= 0 or not math.isfinite(actual_window_hz):
        raise ValueError("actual_window_hz must be positive")
    step = 1.0 / float(actual_window_hz)
    values: list[float] = []
    current = -float(lookback_seconds)
    while current < -1e-9:
        values.append(round(current, 6))
        current += step
    values.append(0.0)
    return tuple(values)


def arm_budgets_for_window_budget(max_actual_windows: int) -> dict[str, int]:
    if max_actual_windows >= 2000:
        return dict(ARM_WINDOW_BUDGETS_2000)
    if max_actual_windows >= 1000:
        return dict(ARM_WINDOW_BUDGETS_1000)
    return dict(ARM_WINDOW_BUDGETS_500)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    write_csv_rows(path, list(rows))


def group_rows_by_video(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["video_id"]), []).append(row)
    for video_rows in grouped.values():
        video_rows.sort(key=lambda item: safe_float(item.get("time_start_seconds")))
    return grouped


def row_key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row["video_id"]), float(safe_float(row.get("time_start_seconds")))


def row_by_time(rows: list[dict[str, Any]]) -> dict[float, dict[str, Any]]:
    return {float(safe_float(row.get("time_start_seconds"))): row for row in rows}


def selected_vitg_hidden_indices(num_hidden_layers: int = 40, cache_n_layers: int = VITG_CACHE_N_LAYERS) -> list[int]:
    n_states = num_hidden_layers + 1
    return [int(round(index)) for index in np.linspace(0, n_states - 1, cache_n_layers)]


def group_mean_vitg_layers(raw_layers: np.ndarray) -> np.ndarray:
    """Match neuralset group_mean over cached 20 V-JEPA layer means."""
    if raw_layers.ndim == 3:
        raw_layers = raw_layers.mean(axis=1)
    if raw_layers.ndim != 2:
        raise ValueError(f"Expected V-JEPA layer means with shape [layers, hidden], got {raw_layers.shape}")
    n_layers = int(raw_layers.shape[0])
    layer_indices = [int(value * (n_layers - 1)) for value in VITG_GROUP_LAYERS]
    layer_indices = sorted(set(layer_indices))
    if len(layer_indices) < 2:
        raise ValueError(f"Cannot group V-JEPA layers from indices: {layer_indices}")
    layer_indices[-1] += 1
    groups = [raw_layers[start:end].mean(axis=0) for start, end in zip(layer_indices[:-1], layer_indices[1:])]
    return np.stack(groups).astype(np.float32)


def fingerprint_payload(
    *,
    video_id: str,
    video_path: str,
    actual_clip_timestamp: float,
    vjepa21_sha256: str,
    tribe_sha256: str,
    selector_arm: str = "",
    selector_config_hash: str = "",
    strict_selector_fingerprint: bool = True,
    teacher_dtype: str = "bfloat16",
) -> dict[str, Any]:
    clip_end = max(0.0, float(actual_clip_timestamp))
    clip_start = max(0.0, clip_end - CLIP_DURATION_SECONDS)
    payload = {
        "dataset": AGAIN_DATASET_NAME,
        "video_id": video_id,
        "video_path": video_path,
        "actual_clip_timestamp": round(float(actual_clip_timestamp), 6),
        "clip_duration_seconds": CLIP_DURATION_SECONDS,
        "frame_count": FRAMES_PER_CLIP,
        "resolution": IMAGE_SIZE,
        "vjepa_model_name": "vjepa2_1_vit_giant_384_mlx_image256",
        "vjepa_checkpoint_sha256": vjepa21_sha256,
        "mlx_checkpoint_hash": vjepa21_sha256,
        "hidden_layer_selection": selected_vitg_hidden_indices(),
        "layer_grouping_policy": "cache20_group_mean_layers_0.5_0.75_1.0",
        "preprocessing_version": "ffmpeg_videotoolbox_square_256_imagenet_norm_v1",
        "dtype": f"{validate_teacher_dtype(teacher_dtype)}_weights_{validate_teacher_dtype(teacher_dtype)}_encoder_input_float32_cached_features",
        "tribe_head_version": "zimengxiong_tribev2_mlx",
        "tribe_head_sha256": tribe_sha256,
        "tribe_head_pool_policy": TRIBE_HEAD_POOL_POLICY,
    }
    if strict_selector_fingerprint:
        payload["clip_start"] = round(float(clip_start), 6)
        payload["clip_end"] = round(float(clip_end), 6)
        payload["selector_arm"] = selector_arm
        payload["selector_config_hash"] = selector_config_hash
    return payload


def role_name_for_relative_seconds(relative: float) -> str:
    if abs(relative) < 1e-9:
        return "T"
    prefix = "T+" if relative > 0 else "T-"
    magnitude = abs(float(relative))
    text = f"{magnitude:g}"
    return f"{prefix}{text}"


def selector_config_hash(arm_window_budgets: dict[str, int], selector_root: Path, causal_relative_seconds: tuple[float, ...] = CAUSAL_RELATIVE_SECONDS) -> str:
    payload = {
        "arm_window_budgets": arm_window_budgets,
        "causal_relative_seconds": list(causal_relative_seconds),
        "selector_bases": SELECTOR_BASES,
        "selector_root_name": selector_root.name,
        "top5_budget": BUDGETS["top5pct"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def cache_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def scout_model_name_from_row(row: dict[str, Any]) -> str:
    name = str(row.get("scout_model_name") or row.get("scout_model") or "").strip()
    if name:
        return name
    if str(row.get("vjepa_l_novelty_z", "")).strip():
        return "vjepa21_vitl_dgrauet_mlx_scout"
    if str(row.get("vjepa_b_novelty_z", "")).strip():
        return "vjepa21_vitb_lukasugar_mlx_scout"
    return ""


def scout_novelty_value(row: dict[str, Any], default: float = 0.0) -> float:
    value = row.get("scout_novelty_z")
    if value is None or str(value).strip() == "":
        value = row.get("vjepa_l_novelty_z")
    if value is None or str(value).strip() == "":
        value = row.get("vjepa_b_novelty_z")
    return safe_float(value, default)


def queue_row(
    *,
    source_row: dict[str, Any],
    selector_arm: str,
    selector_role: str,
    center_timestamp: float,
    actual_clip_timestamp: float,
    temporal_role: str,
    source_selector_score: float,
    candidate_region_id: str,
    oracle_used_for_selection: bool,
    fingerprint: str,
    legacy_fingerprint: str,
    selector_config_digest: str,
) -> dict[str, Any]:
    clip_end = max(0.0, float(actual_clip_timestamp))
    clip_start = max(0.0, clip_end - CLIP_DURATION_SECONDS)
    scout_novelty = scout_novelty_value(source_row, default=0.0)
    scout_model_name = scout_model_name_from_row(source_row)
    return {
        "dataset_name": AGAIN_DATASET_NAME,
        "video_id": source_row["video_id"],
        "video_path": source_row["video_path"],
        "participant_id": source_row.get("participant_id", ""),
        "session_id": source_row.get("session_id", ""),
        "game": source_row.get("game", ""),
        "genre": source_row.get("genre", ""),
        "selector_arm": selector_arm,
        "selector_role": selector_role,
        "center_timestamp": float(center_timestamp),
        "actual_clip_timestamp": float(actual_clip_timestamp),
        "relative_position_to_center": temporal_role,
        "clip_start": clip_start,
        "clip_end": clip_end,
        "temporal_role": temporal_role,
        "source_selector_score": float(source_selector_score),
        "candidate_region_id": candidate_region_id,
        "spike_label_eval_only": source_row.get("future_spike_1_3s_ge_0.05", ""),
        "arousal_delta_eval_only": source_row.get("future_spike_1_3s_delta", ""),
        "future_change_p3s_eval_only": source_row.get("future_change_p3s_value", ""),
        "pre_spike_2s_eval_only": source_row.get("pre_spike_2s", ""),
        "pre_spike_4s_eval_only": source_row.get("pre_spike_4s", ""),
        "pre_spike_6s_eval_only": source_row.get("pre_spike_6s", ""),
        "pre_spike_8s_eval_only": source_row.get("pre_spike_8s", ""),
        "telemetry_change_z": source_row.get("telemetry_change_z", ""),
        "scout_model_name": scout_model_name,
        "scout_novelty_z": scout_novelty,
        "vjepa_l_novelty_z": source_row.get("vjepa_l_novelty_z", ""),
        "vjepa_b_novelty_z": source_row.get("vjepa_b_novelty_z", ""),
        "cheap_video_audio_z": source_row.get("cheap_video_audio_z", ""),
        "arousal": source_row.get("arousal", ""),
        "oracle_used_for_selection": bool(oracle_used_for_selection),
        "deployable_control_or_oracle": selector_role,
        "cache_fingerprint": fingerprint,
        "legacy_cache_fingerprint": legacy_fingerprint,
        "selector_config_hash": selector_config_digest,
        "final_predictive_feature_row": temporal_role != "T+1",
        "future_row_diagnostic_only": temporal_role == "T+1",
    }


def round_robin_take(per_video: dict[str, list[dict[str, Any]]], center_limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    video_ids = sorted(per_video)
    while len(selected) < center_limit and any(per_video.values()):
        progressed = False
        for video_id in video_ids:
            if len(selected) >= center_limit:
                break
            if per_video[video_id]:
                selected.append(per_video[video_id].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def selector_candidates(
    rows_by_video: dict[str, list[dict[str, Any]]],
    *,
    arm: str,
    center_limit: int | None,
    rng: random.Random,
) -> list[dict[str, Any]]:
    per_video: dict[str, list[dict[str, Any]]] = {}
    top5 = BUDGETS["top5pct"]
    for video_id, rows in rows_by_video.items():
        rows = [row for row in rows if safe_float(row.get("time_start_seconds")) >= 2.0]
        if not rows:
            continue
        duration = float(max(safe_float(row.get("time_start_seconds")) for row in rows) + 1.0)
        if arm == "hybrid_top5_selected":
            scores = score_rows(rows, SELECTOR_BASES["hybrid_telemetry_video_audio_vjepa_b"], rng)
            selected_times = set(select_times_for_budget(rows, scores, top5))
            candidates = [
                {**row, "_selector_score": float(score)}
                for row, score in zip(rows, scores)
                if safe_float(row.get("time_start_seconds")) in selected_times
            ]
            candidates.sort(key=lambda item: (-safe_float(item.get("_selector_score")), safe_float(item.get("time_start_seconds"))))
        elif arm == "coverage_matched_random_to_hybrid":
            hybrid_scores = score_rows(rows, SELECTOR_BASES["hybrid_telemetry_video_audio_vjepa_b"], rng)
            hybrid_times = select_times_for_budget(rows, hybrid_scores, top5)
            target_duration = sum(end - start for start, end in selected_intervals(hybrid_times, duration=duration))
            selected_times = select_random_times_for_target_coverage(
                rows,
                duration=duration,
                target_duration=target_duration,
                rng=rng,
            )
            selected_set = set(selected_times)
            random_scores = {time_s: rng.random() for time_s in selected_set}
            candidates = [
                {**row, "_selector_score": float(random_scores[safe_float(row.get("time_start_seconds"))])}
                for row in rows
                if safe_float(row.get("time_start_seconds")) in selected_set
            ]
            candidates.sort(key=lambda item: (-safe_float(item.get("_selector_score")), safe_float(item.get("time_start_seconds"))))
        elif arm == "fixed_random_same_budget":
            shuffled = list(rows)
            rng.shuffle(shuffled)
            candidates = [{**row, "_selector_score": float(rng.random())} for row in shuffled]
        elif arm == "oracle_upper_bound":
            scores = score_rows(rows, "oracle", rng)
            candidates = [{**row, "_selector_score": float(score)} for row, score in zip(rows, scores)]
            candidates.sort(key=lambda item: (-safe_float(item.get("_selector_score")), safe_float(item.get("time_start_seconds"))))
        elif arm == "low_salience_background":
            scores = score_rows(rows, SELECTOR_BASES["hybrid_telemetry_video_audio_vjepa_b"], rng)
            candidates = [{**row, "_selector_score": float(score)} for row, score in zip(rows, scores)]
            candidates.sort(key=lambda item: (safe_float(item.get("_selector_score")), safe_float(item.get("time_start_seconds"))))
        elif arm == "sparse_anchor_windows":
            anchors = []
            for target in np.arange(15.0, duration, 30.0):
                row = min(rows, key=lambda item: abs(safe_float(item.get("time_start_seconds")) - target))
                anchors.append({**row, "_selector_score": 0.0})
            candidates = anchors
        else:
            raise ValueError(f"Unknown selector arm: {arm}")
        per_video[video_id] = candidates
    if center_limit is None:
        center_limit = sum(len(rows) for rows in per_video.values())
    return round_robin_take(per_video, center_limit)


def build_sparse_teacher_queue(
    feature_rows: list[dict[str, Any]],
    *,
    max_actual_windows: int,
    rng: random.Random,
    vjepa21_sha256: str,
    tribe_sha256: str,
    arm_window_budgets: dict[str, int] | None = None,
    selector_config_digest: str = "",
    causal_relative_seconds: tuple[float, ...] = CAUSAL_RELATIVE_SECONDS,
    teacher_dtype: str = "bfloat16",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if arm_window_budgets is None:
        arm_window_budgets = ARM_WINDOW_BUDGETS_500
    rows_by_video = group_rows_by_video(feature_rows)
    rows_lookup = {row_key(row): row for row in feature_rows}
    queue: list[dict[str, Any]] = []
    unique_fingerprints: set[str] = set()
    arm_unique_counts: dict[str, int] = {}
    arm_queue_counts: dict[str, int] = {}
    final_centers: dict[str, int] = {}
    roles = {float(relative): role_name_for_relative_seconds(float(relative)) for relative in causal_relative_seconds}

    for arm, window_budget in arm_window_budgets.items():
        candidates = selector_candidates(rows_by_video, arm=arm, center_limit=None, rng=rng)
        selector_role = "oracle" if arm == "oracle_upper_bound" else "control" if "random" in arm or "background" in arm or "anchor" in arm else "deployable"
        arm_fingerprints: set[str] = set()
        accepted_centers = 0
        for center_index, center_row in enumerate(candidates):
            center_timestamp = float(safe_float(center_row.get("time_start_seconds")))
            burst_rows: list[dict[str, Any]] = []
            burst_fingerprints: set[str] = set()
            for relative in causal_relative_seconds:
                actual_timestamp = max(0.0, center_timestamp + relative)
                actual_row = rows_lookup.get((str(center_row["video_id"]), actual_timestamp), center_row)
                payload = fingerprint_payload(
                    video_id=str(center_row["video_id"]),
                    video_path=str(center_row["video_path"]),
                    actual_clip_timestamp=actual_timestamp,
                    vjepa21_sha256=vjepa21_sha256,
                    tribe_sha256=tribe_sha256,
                    selector_arm=arm,
                    selector_config_hash=selector_config_digest,
                    strict_selector_fingerprint=True,
                    teacher_dtype=teacher_dtype,
                )
                legacy_payload = fingerprint_payload(
                    video_id=str(center_row["video_id"]),
                    video_path=str(center_row["video_path"]),
                    actual_clip_timestamp=actual_timestamp,
                    vjepa21_sha256=vjepa21_sha256,
                    tribe_sha256=tribe_sha256,
                    strict_selector_fingerprint=False,
                    teacher_dtype=teacher_dtype,
                )
                fingerprint = cache_fingerprint(payload)
                legacy_fingerprint = cache_fingerprint(legacy_payload)
                burst_fingerprints.add(fingerprint)
                burst_rows.append(
                    queue_row(
                        source_row=actual_row,
                        selector_arm=arm,
                        selector_role=selector_role,
                        center_timestamp=center_timestamp,
                        actual_clip_timestamp=actual_timestamp,
                        temporal_role=roles[float(relative)],
                        source_selector_score=safe_float(center_row.get("_selector_score"), 0.0),
                        candidate_region_id=f"{arm}_{center_index:04d}",
                        oracle_used_for_selection=arm == "oracle_upper_bound",
                        fingerprint=fingerprint,
                        legacy_fingerprint=legacy_fingerprint,
                        selector_config_digest=selector_config_digest,
                    )
                )
            new_for_arm = burst_fingerprints - arm_fingerprints
            new_for_global = burst_fingerprints - unique_fingerprints
            if len(arm_fingerprints) + len(new_for_arm) > window_budget:
                continue
            if len(unique_fingerprints) + len(new_for_global) > max_actual_windows:
                continue
            arm_fingerprints.update(burst_fingerprints)
            unique_fingerprints.update(burst_fingerprints)
            queue.extend(burst_rows)
            accepted_centers += 1
        final_centers[arm] = accepted_centers
        arm_unique_counts[arm] = len(arm_fingerprints)
        arm_queue_counts[arm] = sum(1 for row in queue if row["selector_arm"] == arm)
    summary = {
        "max_actual_windows": max_actual_windows,
        "queued_rows": len(queue),
        "unique_actual_windows": len(unique_fingerprints),
        "arm_unique_window_counts": arm_unique_counts,
        "arm_queue_row_counts": arm_queue_counts,
        "arm_center_counts": final_centers,
        "arm_window_budgets": arm_window_budgets,
        "selector_config_hash": selector_config_digest,
        "video_count": len(rows_by_video),
        "video_ids": sorted(rows_by_video),
        "future_rows_included": False,
        "causal_roles": list(causal_relative_seconds),
        "causal_role_names": [roles[float(relative)] for relative in causal_relative_seconds],
        "teacher_dtype": validate_teacher_dtype(teacher_dtype),
    }
    return queue, summary


def cache_path_for(external_cache_root: Path, fingerprint: str) -> Path:
    return external_cache_root / "encoded_windows" / fingerprint[:2] / f"{fingerprint}.npz"


def load_cached_window(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path) as bundle:
        return np.asarray(bundle["cortical_prediction"], dtype=np.float32), np.asarray(bundle["grouped_video_feature"], dtype=np.float32)


def write_cached_window(path: Path, cortical: np.ndarray, grouped: np.ndarray, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cortical_prediction=cortical.astype(np.float32),
        grouped_video_feature=grouped.astype(np.float32),
        fingerprint_payload=json.dumps(payload, sort_keys=True),
    )


def sparse_window_frame_times(actual_clip_timestamp: float) -> list[float]:
    subtimes = [index / FRAMES_PER_CLIP * CLIP_DURATION_SECONDS for index in reversed(range(FRAMES_PER_CLIP))]
    return [max(0.0, float(actual_clip_timestamp) - delta) for delta in subtimes]


def runtime_row(
    *,
    video_id: str,
    row: dict[str, Any],
    cache_hit: bool,
    cache_hit_mode: str,
    success: bool,
    decode_seconds_for_video: float = 0.0,
    preprocess_seconds: float = 0.0,
    queue_wait_seconds: float = 0.0,
    forward_seconds: float = 0.0,
    forward_batch_seconds: float = 0.0,
    tribe_head_seconds: float = 0.0,
    write_seconds: float = 0.0,
    microbatch_size: int = 1,
    error: str = "",
) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "cache_fingerprint": str(row["cache_fingerprint"]),
        "selector_arm": row.get("selector_arm", ""),
        "actual_clip_timestamp": row.get("actual_clip_timestamp", ""),
        "cache_hit": bool(cache_hit),
        "cache_hit_mode": cache_hit_mode,
        "success": bool(success),
        "decode_seconds_for_video": float(decode_seconds_for_video),
        "preprocess_seconds": float(preprocess_seconds),
        "queue_wait_seconds": float(queue_wait_seconds),
        "forward_seconds": float(forward_seconds),
        "forward_batch_seconds": float(forward_batch_seconds),
        "tribe_head_seconds": float(tribe_head_seconds),
        "write_seconds": float(write_seconds),
        "microbatch_size": int(microbatch_size),
        "error": error,
    }


def expensive_window_identity_from_payload(payload: dict[str, Any]) -> str:
    identity = {
        "dataset": payload.get("dataset"),
        "video_id": payload.get("video_id"),
        "video_path": payload.get("video_path"),
        "actual_clip_timestamp": round(float(payload.get("actual_clip_timestamp", 0.0)), 6),
        "clip_duration_seconds": payload.get("clip_duration_seconds", CLIP_DURATION_SECONDS),
        "frame_count": payload.get("frame_count", FRAMES_PER_CLIP),
        "resolution": payload.get("resolution", IMAGE_SIZE),
        "vjepa_model_name": payload.get("vjepa_model_name"),
        "hidden_layer_selection": payload.get("hidden_layer_selection"),
        "layer_grouping_policy": payload.get("layer_grouping_policy"),
        "preprocessing_version": payload.get("preprocessing_version"),
        "dtype": payload.get("dtype"),
        "tribe_head_version": payload.get("tribe_head_version"),
        "tribe_head_pool_policy": payload.get("tribe_head_pool_policy"),
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def expensive_window_identity_for_row(row: dict[str, Any], *, video_path: Path, teacher_dtype: str = "bfloat16") -> str:
    payload = fingerprint_payload(
        video_id=str(row["video_id"]),
        video_path=str(video_path),
        actual_clip_timestamp=float(row["actual_clip_timestamp"]),
        vjepa21_sha256="ignored_for_expensive_window_identity",
        tribe_sha256="ignored_for_expensive_window_identity",
        strict_selector_fingerprint=False,
        teacher_dtype=teacher_dtype,
    )
    return expensive_window_identity_from_payload(payload)


def build_expensive_window_cache_index(external_cache_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    cache_dir = external_cache_root / "encoded_windows"
    if not cache_dir.exists():
        return index
    for path in cache_dir.glob("*/*.npz"):
        try:
            with np.load(path) as bundle:
                raw_payload = bundle.get("fingerprint_payload")
                if raw_payload is None:
                    continue
                payload = json.loads(str(raw_payload.item() if hasattr(raw_payload, "item") else raw_payload))
        except Exception:  # noqa: BLE001 - malformed old cache entries are ignored
            continue
        index.setdefault(expensive_window_identity_from_payload(payload), path)
    return index


def encode_sparse_windows(
    queue: list[dict[str, Any]],
    *,
    external_cache_root: Path,
    vjepa_weights_dir: Path,
    tribe_model_dir: Path,
    vjepa_sha256: str,
    tribe_sha256: str,
    teacher_dtype: str = "bfloat16",
    allow_cache_backfill: bool = False,
    gpu_workers: int = 1,
    preprocess_workers: int = 0,
    ready_queue_max_size: int = 4,
    writer_queue_max_size: int = 4,
    microbatch_size: int = 1,
    compile_encoder: bool = True,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    teacher_dtype = validate_teacher_dtype(teacher_dtype)
    runtime_config = validate_single_gpu_runtime(
        gpu_workers=gpu_workers,
        preprocess_workers=preprocess_workers,
        ready_queue_max_size=ready_queue_max_size,
        writer_queue_max_size=writer_queue_max_size,
        microbatch_size=microbatch_size,
    )
    effective_microbatch_size = min(int(microbatch_size), int(ready_queue_max_size))
    unique_by_fp: dict[str, dict[str, Any]] = {}
    for row in queue:
        unique_by_fp.setdefault(str(row["cache_fingerprint"]), row)
    selected_indices = selected_vitg_hidden_indices()
    cache_identity_index = build_expensive_window_cache_index(external_cache_root) if allow_cache_backfill else {}
    model: MlxVjepa21FeatureModel | None = None
    tribe: MlxTribeEncoder | None = None
    features_by_fp: dict[str, np.ndarray] = {}
    runtime_rows: list[dict[str, Any]] = []
    start_all = time.perf_counter()
    cache_hits = 0
    legacy_cache_hits = 0
    physical_cache_hits = 0
    failures = 0

    by_video: dict[str, list[dict[str, Any]]] = {}
    for row in unique_by_fp.values():
        by_video.setdefault(str(row["video_id"]), []).append(row)

    for video_index, (video_id, rows) in enumerate(sorted(by_video.items()), start=1):
        rows.sort(key=lambda item: safe_float(item.get("actual_clip_timestamp")))
        video_path = Path(rows[0]["video_path"])
        miss_rows: list[tuple[int, dict[str, Any], Path]] = []
        decode_seconds = 0.0
        decoded_grid: np.ndarray | None = None
        for local_index, row in enumerate(rows, start=1):
            fp = str(row["cache_fingerprint"])
            path = cache_path_for(external_cache_root, fp)
            cached = load_cached_window(path)
            cache_hit_mode = "strict"
            if allow_cache_backfill and cached is None and row.get("legacy_cache_fingerprint"):
                legacy_path = cache_path_for(external_cache_root, str(row["legacy_cache_fingerprint"]))
                cached = load_cached_window(legacy_path)
                if cached is not None:
                    cache_hit_mode = "legacy_backfilled_to_strict"
            if allow_cache_backfill and cached is None:
                physical_path = cache_identity_index.get(expensive_window_identity_for_row(row, video_path=video_path, teacher_dtype=teacher_dtype))
                if physical_path is not None and physical_path != path:
                    cached = load_cached_window(physical_path)
                    if cached is not None:
                        cache_hit_mode = "physical_window_backfilled_to_strict"
            if cached is not None:
                cortical, grouped = cached
                if cache_hit_mode in {"legacy_backfilled_to_strict", "physical_window_backfilled_to_strict"}:
                    write_cached_window(
                        path,
                        cortical,
                        grouped,
                        fingerprint_payload(
                            video_id=video_id,
                            video_path=str(video_path),
                            actual_clip_timestamp=float(row["actual_clip_timestamp"]),
                            vjepa21_sha256=vjepa_sha256,
                            tribe_sha256=tribe_sha256,
                            selector_arm=str(row["selector_arm"]),
                            selector_config_hash=str(row.get("selector_config_hash", "")),
                            strict_selector_fingerprint=True,
                            teacher_dtype=teacher_dtype,
                        ),
                    )
                    cache_identity_index.setdefault(expensive_window_identity_for_row(row, video_path=video_path, teacher_dtype=teacher_dtype), path)
                if cache_hit_mode == "legacy_backfilled_to_strict":
                    legacy_cache_hits += 1
                if cache_hit_mode == "physical_window_backfilled_to_strict":
                    physical_cache_hits += 1
                features_by_fp[fp] = cortical
                cache_hits += 1
                runtime_rows.append(
                    runtime_row(
                        video_id=video_id,
                        row=row,
                        cache_hit=True,
                        cache_hit_mode=cache_hit_mode,
                        success=True,
                    )
                )
                continue
            miss_rows.append((local_index, row, path))

        if miss_rows:
            decode_started = time.perf_counter()
            decoded_grid = _decode_video_grid_ffmpeg(video_path, fps=DECODE_FPS, image_size=IMAGE_SIZE)
            decode_seconds = time.perf_counter() - decode_started
        for batch_start in range(0, len(miss_rows), effective_microbatch_size):
            batch = miss_rows[batch_start : batch_start + effective_microbatch_size]
            try:
                if model is None:
                    model = MlxVjepa21FeatureModel(
                        str(vjepa_weights_dir),
                        IMAGE_SIZE,
                        compile_encoder=compile_encoder,
                        input_dtype=teacher_dtype,
                    )
                if tribe is None:
                    tribe = MlxTribeEncoder(str(tribe_model_dir), dtype=teacher_dtype)
                if decoded_grid is None:
                    raise RuntimeError("decoded video grid missing for uncached windows")
                preprocess_started = time.perf_counter()
                windows = [
                    _sample_decoded_grid(
                        decoded_grid,
                        fps=DECODE_FPS,
                        times=sparse_window_frame_times(float(row["actual_clip_timestamp"])),
                    )
                    for _local_index, row, _path in batch
                ]
                stacked_windows = np.stack(windows, axis=0)
                preprocess_seconds = time.perf_counter() - preprocess_started
                forward_started = time.perf_counter()
                hidden = model.predict_hidden_states(stacked_windows, selected_indices)
                forward_batch_seconds = time.perf_counter() - forward_started
                if not np.isfinite(hidden).all():
                    raise ValueError("V-JEPA 2.1 produced non-finite hidden states; refusing to cache batch")
                grouped_batch = np.stack(
                    [group_mean_vitg_layers(np.asarray(hidden[index], dtype=np.float32)) for index in range(hidden.shape[0])],
                    axis=0,
                )
                head_started = time.perf_counter()
                pred = tribe.predict({"video": grouped_batch[..., None]})
                cortical_batch = np.asarray(pred, dtype=np.float32).mean(axis=-1)
                if not np.isfinite(cortical_batch).all():
                    raise ValueError("TRIBE head produced non-finite cortical predictions; refusing to cache batch")
                head_seconds = time.perf_counter() - head_started
                write_started = time.perf_counter()
                for batch_index, (_local_index, row, path) in enumerate(batch):
                    actual = float(row["actual_clip_timestamp"])
                    cortical = np.asarray(cortical_batch[batch_index], dtype=np.float32)
                    grouped = np.asarray(grouped_batch[batch_index], dtype=np.float32)
                    write_cached_window(
                        path,
                        cortical,
                        grouped,
                        fingerprint_payload(
                            video_id=video_id,
                            video_path=str(video_path),
                            actual_clip_timestamp=actual,
                            vjepa21_sha256=vjepa_sha256,
                            tribe_sha256=tribe_sha256,
                            selector_arm=str(row["selector_arm"]),
                            selector_config_hash=str(row.get("selector_config_hash", "")),
                            strict_selector_fingerprint=True,
                            teacher_dtype=teacher_dtype,
                        ),
                    )
                    if allow_cache_backfill:
                        cache_identity_index.setdefault(expensive_window_identity_for_row(row, video_path=video_path, teacher_dtype=teacher_dtype), path)
                    features_by_fp[str(row["cache_fingerprint"])] = cortical
                write_seconds = time.perf_counter() - write_started
                per_window_forward = forward_batch_seconds / len(batch)
                per_window_preprocess = preprocess_seconds / len(batch)
                per_window_head = head_seconds / len(batch)
                per_window_write = write_seconds / len(batch)
                for batch_index, (local_index, row, _path) in enumerate(batch):
                    runtime_rows.append(
                        runtime_row(
                            video_id=video_id,
                            row=row,
                            cache_hit=False,
                            cache_hit_mode="miss_encoded",
                            success=True,
                            decode_seconds_for_video=decode_seconds if batch_start == 0 and batch_index == 0 else 0.0,
                            preprocess_seconds=per_window_preprocess,
                            forward_seconds=per_window_forward,
                            forward_batch_seconds=forward_batch_seconds,
                            tribe_head_seconds=per_window_head,
                            write_seconds=per_window_write,
                            microbatch_size=len(batch),
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - per-batch failure is reported per window
                failures += len(batch)
                for batch_index, (local_index, row, _path) in enumerate(batch):
                    runtime_rows.append(
                        runtime_row(
                            video_id=video_id,
                            row=row,
                            cache_hit=False,
                            cache_hit_mode="miss_failed",
                            success=False,
                            decode_seconds_for_video=decode_seconds if batch_start == 0 and batch_index == 0 else 0.0,
                            microbatch_size=len(batch),
                            error=str(exc),
                        )
                    )
        mx.clear_cache()
        print(
            json.dumps(
                {
                    "progress_video": f"{video_index}/{len(by_video)}",
                    "video_id": video_id,
                    "windows_for_video": len(rows),
                    "encoded_or_cached": len(features_by_fp),
                    "cache_hits": cache_hits,
                    "legacy_cache_hits": legacy_cache_hits,
                    "physical_cache_hits": physical_cache_hits,
                    "microbatch_size": effective_microbatch_size,
                    "failures": failures,
                }
            ),
            flush=True,
        )

    summary = {
        "unique_actual_windows": len(unique_by_fp),
        "successful_windows": len(features_by_fp),
        "cache_hits": cache_hits,
        "legacy_cache_hits": legacy_cache_hits,
        "physical_cache_hits": physical_cache_hits,
        "failed_windows": failures,
        "total_runtime_seconds": time.perf_counter() - start_all,
        "seconds_per_successful_window": (time.perf_counter() - start_all) / len(features_by_fp) if features_by_fp else math.nan,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "teacher_dtype": teacher_dtype,
        "allow_cache_backfill": bool(allow_cache_backfill),
        "compile_encoder": bool(compile_encoder),
        **runtime_config,
        **mlx_memory_snapshot(),
    }
    return features_by_fp, runtime_rows, summary


def bool_label(value: Any) -> int:
    return 1 if str(value).strip().lower() in {"true", "1", "yes"} else 0


def build_center_rows(
    queue: list[dict[str, Any]],
    features_by_fp: dict[str, np.ndarray],
    *,
    causal_relative_seconds: tuple[float, ...] = CAUSAL_RELATIVE_SECONDS,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    role_names = [role_name_for_relative_seconds(relative) for relative in causal_relative_seconds]
    if "T" not in role_names:
        raise ValueError("causal_relative_seconds must include 0.0/T for current-window evaluation")
    grouped: dict[tuple[str, str, float], dict[str, dict[str, Any]]] = {}
    for row in queue:
        if row["cache_fingerprint"] not in features_by_fp:
            continue
        key = (str(row["selector_arm"]), str(row["video_id"]), float(row["center_timestamp"]))
        grouped.setdefault(key, {})[str(row["temporal_role"])] = row

    rows: list[dict[str, Any]] = []
    features: dict[str, dict[str, np.ndarray]] = {}
    for (arm, video_id, center), role_rows in sorted(grouped.items()):
        if not all(role in role_rows for role in role_names):
            continue
        current = features_by_fp[role_rows["T"]["cache_fingerprint"]]
        previous_role = next((role for role in reversed(role_names) if role != "T" and role in role_rows), "T")
        prev = features_by_fp[role_rows[previous_role]["cache_fingerprint"]]
        past2_role = role_names[0]
        past2 = features_by_fp[role_rows[past2_role]["cache_fingerprint"]]
        causal_stack = np.stack([features_by_fp[role_rows[role]["cache_fingerprint"]] for role in role_names], axis=0)
        row0 = role_rows["T"]
        center_id = f"{arm}|{video_id}|{center:.3f}"
        rows.append(
            {
                "center_id": center_id,
                "selector_arm": arm,
                "selector_role": row0["selector_role"],
                "video_id": video_id,
                "participant_id": row0.get("participant_id", ""),
                "session_id": row0.get("session_id", ""),
                "game": row0.get("game", ""),
                "genre": row0.get("genre", ""),
                "center_timestamp": center,
                "spike_label": bool_label(row0.get("spike_label_eval_only")),
                "pre_spike_2s": bool_label(row0.get("pre_spike_2s_eval_only")),
                "pre_spike_4s": bool_label(row0.get("pre_spike_4s_eval_only")),
                "pre_spike_6s": bool_label(row0.get("pre_spike_6s_eval_only")),
                "pre_spike_8s": bool_label(row0.get("pre_spike_8s_eval_only")),
                "telemetry_change_z": safe_float(row0.get("telemetry_change_z"), 0.0),
                "scout_model_name": row0.get("scout_model_name", ""),
                "scout_novelty_z": scout_novelty_value(row0, default=0.0),
                "vjepa_l_novelty_z": safe_float(row0.get("vjepa_l_novelty_z"), 0.0),
                "vjepa_b_novelty_z": safe_float(row0.get("vjepa_b_novelty_z"), 0.0),
                "cheap_video_audio_z": safe_float(row0.get("cheap_video_audio_z"), 0.0),
                "arousal": safe_float(row0.get("arousal"), 0.0),
            }
        )
        features[center_id] = {
            "raw_sparse_T_minus_2": past2,
            "raw_sparse_T_minus_1": prev,
            "raw_sparse_current": current,
            "sparse_delta": current - prev,
            "sparse_pca64_delta_analogue": current - prev,
            "sparse_causal_roles": causal_stack,
            "sparse_causal_past2s_raw_mean": np.mean(causal_stack, axis=0),
        }
    return rows, features


def add_ar_features(center_rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]]) -> None:
    by_vid = group_rows_by_video(feature_rows)
    arousal_by_vid = {
        video_id: {safe_float(row.get("time_start_seconds")): safe_float(row.get("arousal"), 0.0) for row in rows}
        for video_id, rows in by_vid.items()
    }
    for row in center_rows:
        series = arousal_by_vid.get(str(row["video_id"]), {})
        t = float(row["center_timestamp"])
        current = series.get(t, safe_float(row.get("arousal"), 0.0))
        lag1 = series.get(t - 1.0, current)
        lag2 = series.get(t - 2.0, lag1)
        row["arousal_current"] = current
        row["arousal_lag1"] = lag1
        row["arousal_lag2"] = lag2
        row["arousal_delta1"] = current - lag1
        row["arousal_delta2"] = lag1 - lag2


def finite_matrix(values: list[list[float]]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def top_recall(y_true: np.ndarray, scores: np.ndarray, frac: float) -> float:
    positives = int(np.sum(y_true == 1))
    if positives == 0 or len(y_true) == 0:
        return math.nan
    keep = max(1, int(math.ceil(len(y_true) * frac)))
    order = np.argsort(-scores)[:keep]
    return float(np.sum(y_true[order] == 1) / positives)


def threshold_from_train(y_train: np.ndarray, train_scores: np.ndarray) -> float:
    if len(np.unique(y_train)) < 2:
        return float(np.median(train_scores))
    candidates = np.unique(np.quantile(train_scores, np.linspace(0.05, 0.95, 19)))
    best = (float("-inf"), float(candidates[0]))
    for threshold in candidates:
        pred = (train_scores >= threshold).astype(int)
        score = f1_score(y_train, pred, zero_division=0)
        if score > best[0]:
            best = (float(score), float(threshold))
    return best[1]


def metric_row(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (scores >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else math.nan,
        "roc_auc": roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else math.nan,
        "f1": f1_score(y_true, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred) if len(np.unique(y_true)) > 1 else math.nan,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
        "top_1pct_recall": top_recall(y_true, scores, 0.01),
        "top_5pct_recall": top_recall(y_true, scores, 0.05),
        "top_10pct_recall": top_recall(y_true, scores, 0.10),
        "predicted_positive_rate": float(np.mean(pred)) if len(pred) else math.nan,
        "predicted_positive_count": int(np.sum(pred)),
        "event_count": int(np.sum(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else math.nan,
    }


def mlx_standardize_fit_apply(
    X_fit: np.ndarray,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    fit = mx.array(X_fit.astype(np.float32, copy=False))
    mean = mx.mean(fit, axis=0, keepdims=True)
    centered = fit - mean
    std = mx.sqrt(mx.mean(centered * centered, axis=0, keepdims=True) + 1e-6)
    outputs = []
    for array in arrays:
        transformed = (mx.array(array.astype(np.float32, copy=False)) - mean) / std
        mx.eval(transformed)
        outputs.append(np.asarray(transformed, dtype=np.float32))
    return tuple(outputs)


def mlx_nipals_pca_components(
    X_train_fit: np.ndarray,
    *,
    pca_width: int,
    random_seed: int,
    iterations: int = 12,
) -> tuple[np.ndarray, int]:
    actual_width = max(1, min(int(pca_width), X_train_fit.shape[0] - 1, X_train_fit.shape[1]))
    X = mx.array(X_train_fit.astype(np.float32, copy=False))
    rng = np.random.default_rng(random_seed)
    components = []
    for _component_index in range(actual_width):
        init = rng.normal(size=(X_train_fit.shape[1],)).astype(np.float32)
        p = mx.array(init)
        p = p / mx.sqrt(mx.sum(p * p) + 1e-8)
        for _ in range(iterations):
            t = X @ p
            denom = mx.sum(t * t) + 1e-8
            p = (mx.transpose(X) @ t) / denom
            p = p / mx.sqrt(mx.sum(p * p) + 1e-8)
        t = X @ p
        X = X - mx.expand_dims(t, 1) * mx.expand_dims(p, 0)
        components.append(p)
        mx.eval(X)
    stacked = mx.stack(components, axis=0)
    mx.eval(stacked)
    return np.asarray(stacked, dtype=np.float32), actual_width


def mlx_pca_fit_transform(
    X_train_fit: np.ndarray,
    X_train_apply: np.ndarray,
    X_test_apply: np.ndarray,
    *,
    pca_width: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    fit_std, train_std, test_std = mlx_standardize_fit_apply(X_train_fit, X_train_fit, X_train_apply, X_test_apply)
    # fit_std is intentionally used for component fitting; train_std/test_std
    # preserve the requested application row order.
    components, actual_width = mlx_nipals_pca_components(
        fit_std,
        pca_width=pca_width,
        random_seed=random_seed,
    )
    comp = mx.array(components)
    train_scores = mx.array(train_std) @ mx.transpose(comp)
    test_scores = mx.array(test_std) @ mx.transpose(comp)
    mx.eval(train_scores, test_scores)
    return np.asarray(train_scores, dtype=np.float32), np.asarray(test_scores, dtype=np.float32), actual_width


def fit_predict_mlx_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    rng: np.random.Generator,
    alpha: float = 1.0,
    max_iter: int = 120,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    del rng
    X_train, X_test = mlx_standardize_fit_apply(X_train, X_train, X_test)
    xtr = mx.array(X_train.astype(np.float32, copy=False))
    xte = mx.array(X_test.astype(np.float32, copy=False))
    y = mx.array(y_train.astype(np.float32, copy=False))
    k_train = xtr @ mx.transpose(xtr)
    n_train = int(X_train.shape[0])
    system = k_train + float(alpha) * mx.eye(n_train)
    alpha_vec = mx.zeros_like(y)
    residual = y - system @ alpha_vec
    direction = residual
    rs_old = mx.sum(residual * residual)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        ap = system @ direction
        step = rs_old / (mx.sum(direction * ap) + 1e-8)
        alpha_vec = alpha_vec + step * direction
        residual = residual - step * ap
        rs_new = mx.sum(residual * residual)
        direction = residual + (rs_new / (rs_old + 1e-8)) * direction
        rs_old = rs_new
    train_scores = k_train @ alpha_vec
    test_scores = (xte @ mx.transpose(xtr)) @ alpha_vec
    mx.eval(train_scores, test_scores)
    return (
        np.asarray(train_scores, dtype=np.float32),
        np.asarray(test_scores, dtype=np.float32),
        {"ridge_solver": "mlx_conjugate_gradient_dual", "ridge_iterations": iterations, "pca_width_actual": 0},
    )


def causal_pca_mean_features(
    causal_roles: np.ndarray,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    pca_width: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    causal_train_fit = causal_roles[train_idx].reshape(-1, causal_roles.shape[-1])
    causal_train_apply = causal_roles[train_idx].reshape(-1, causal_roles.shape[-1])
    causal_test_apply = causal_roles[test_idx].reshape(-1, causal_roles.shape[-1])
    train_pc_flat, test_pc_flat, actual_width = mlx_pca_fit_transform(
        causal_train_fit,
        causal_train_apply,
        causal_test_apply,
        pca_width=pca_width,
        random_seed=random_seed,
    )
    role_count = int(causal_roles.shape[1])
    train_pc = train_pc_flat.reshape(len(train_idx), role_count, actual_width).mean(axis=1)
    test_pc = test_pc_flat.reshape(len(test_idx), role_count, actual_width).mean(axis=1)
    return train_pc, test_pc, actual_width


def select_pca_width_with_inner_video_validation(
    *,
    causal_roles: np.ndarray,
    ar_base: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outer_train_idx: np.ndarray,
    candidate_widths: tuple[int, ...] = SMALL_PCA_WIDTH_CANDIDATES,
    random_seed: int,
) -> dict[str, Any]:
    """Select PCA width using only outer-train rows and grouped inner validation."""
    train_groups = groups[outer_train_idx]
    unique_groups = sorted(set(str(item) for item in train_groups))
    if len(unique_groups) < 2 or len(np.unique(y[outer_train_idx])) < 2:
        width = min(candidate_widths)
        return {
            "selected_width": width,
            "selected_actual_width": 0,
            "inner_validation_strategy": "fallback_smallest_width_no_valid_group_split",
            "inner_validation_pr_auc": math.nan,
            "candidate_scores": [],
            "test_labels_used_for_selection": False,
        }
    inner_splits = min(3, len(unique_groups))
    splitter = GroupKFold(n_splits=inner_splits)
    train_positions = np.arange(len(outer_train_idx))
    candidate_scores: list[dict[str, Any]] = []
    for width in candidate_widths:
        fold_scores = []
        actual_widths = []
        for inner_fold, (inner_train_pos, inner_val_pos) in enumerate(
            splitter.split(np.zeros(len(outer_train_idx)), y[outer_train_idx], train_groups),
            start=1,
        ):
            inner_train_idx = outer_train_idx[train_positions[inner_train_pos]]
            inner_val_idx = outer_train_idx[train_positions[inner_val_pos]]
            y_inner_train = y[inner_train_idx]
            y_inner_val = y[inner_val_idx]
            if len(np.unique(y_inner_train)) < 2 or len(np.unique(y_inner_val)) < 2:
                continue
            inner_train_pc, inner_val_pc, actual_width = causal_pca_mean_features(
                causal_roles,
                train_idx=inner_train_idx,
                test_idx=inner_val_idx,
                pca_width=width,
                random_seed=random_seed + width * 100 + inner_fold,
            )
            X_inner_train = np.concatenate([ar_base[inner_train_idx], inner_train_pc], axis=1)
            X_inner_val = np.concatenate([ar_base[inner_val_idx], inner_val_pc], axis=1)
            _train_scores, val_scores, _fit_info = fit_predict_mlx_ridge(
                X_inner_train,
                y_inner_train,
                X_inner_val,
                rng=np.random.default_rng(random_seed + width * 1000 + inner_fold),
            )
            fold_scores.append(float(average_precision_score(y_inner_val, val_scores)))
            actual_widths.append(int(actual_width))
        candidate_scores.append(
            {
                "requested_width": int(width),
                "mean_inner_pr_auc": float(np.mean(fold_scores)) if fold_scores else math.nan,
                "inner_folds": len(fold_scores),
                "mean_actual_width": float(np.mean(actual_widths)) if actual_widths else 0.0,
            }
        )
    valid = [row for row in candidate_scores if math.isfinite(safe_float(row.get("mean_inner_pr_auc")))]
    if not valid:
        selected = min(candidate_widths)
        selected_row = {"mean_inner_pr_auc": math.nan, "mean_actual_width": 0.0}
        strategy = "fallback_smallest_width_no_valid_inner_scores"
    else:
        selected_row = max(valid, key=lambda row: (safe_float(row.get("mean_inner_pr_auc")), -int(row["requested_width"])))
        selected = int(selected_row["requested_width"])
        strategy = "grouped_video_inner_validation_train_only"
    return {
        "selected_width": int(selected),
        "selected_actual_width": int(round(safe_float(selected_row.get("mean_actual_width"), 0.0))),
        "inner_validation_strategy": strategy,
        "inner_validation_pr_auc": safe_float(selected_row.get("mean_inner_pr_auc")),
        "candidate_scores": candidate_scores,
        "test_labels_used_for_selection": False,
    }


def pca_requested_width_for_lane(lane_name: str, selected_width: int = 0) -> int:
    if "pca_train_selected" in lane_name:
        return int(selected_width)
    for width in (*SMALL_PCA_WIDTH_CANDIDATES, 128):
        if f"pca{width}" in lane_name:
            return int(width)
    return 0


def evaluate_arm(
    arm: str,
    center_rows: list[dict[str, Any]],
    sparse_features: dict[str, dict[str, np.ndarray]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [row for row in center_rows if row["selector_arm"] == arm]
    if len(rows) < 12 or len({row["video_id"] for row in rows}) < 3:
        return [], []
    y = np.asarray([int(row["spike_label"]) for row in rows], dtype=int)
    groups = np.asarray([row["video_id"] for row in rows])
    n_splits = min(5, len(set(groups)))
    if len(np.unique(y)) < 2 or n_splits < 2:
        return [], []

    base = {
        "timestamp_only_baseline": finite_matrix([[row["center_timestamp"]] for row in rows]),
        "AR_only": finite_matrix([[row["arousal_current"], row["arousal_lag1"], row["arousal_lag2"], row["arousal_delta1"], row["arousal_delta2"]] for row in rows]),
        "telemetry_change_only": finite_matrix([[row["telemetry_change_z"]] for row in rows]),
        "VJEPA_B_scout_only": finite_matrix([[row["scout_novelty_z"]] for row in rows]),
        "AR_plus_telemetry_change_plus_VJEPA_B": finite_matrix(
            [
                [
                    row["arousal_current"],
                    row["arousal_lag1"],
                    row["arousal_lag2"],
                    row["arousal_delta1"],
                    row["arousal_delta2"],
                    row["telemetry_change_z"],
                    row["scout_novelty_z"],
                ]
                for row in rows
            ]
        ),
    }
    raw = np.stack([sparse_features[row["center_id"]]["raw_sparse_current"] for row in rows]).astype(np.float32)
    delta = np.stack([sparse_features[row["center_id"]]["sparse_delta"] for row in rows]).astype(np.float32)
    causal_roles = np.stack([sparse_features[row["center_id"]]["sparse_causal_roles"] for row in rows], axis=0).astype(np.float32)
    raw_causal_mean = causal_roles.mean(axis=1)
    lane_names = [
        "majority_prevalence_baseline",
        "timestamp_only_baseline",
        "AR_only",
        "telemetry_change_only",
        "VJEPA_B_scout_only",
        "AR_plus_telemetry_change_plus_VJEPA_B",
        "AR_plus_raw_sparse_current",
        "AR_plus_raw_sparse_causal_past2s_mean",
        "AR_plus_sparse_pca64_delta_analogue",
        *[f"AR_plus_sparse_pca{width}_causal_past2s_mean" for width in SMALL_PCA_WIDTH_CANDIDATES],
        "AR_plus_sparse_pca_train_selected_causal_past2s_mean",
        "AR_plus_sparse_pca128_causal_past2s_mean",
        "AR_plus_telemetry_VJEPA_B_sparse_pca32_causal_past2s_mean",
        "AR_plus_telemetry_VJEPA_B_sparse_pca_train_selected_causal_past2s_mean",
        "AR_plus_telemetry_VJEPA_B_sparse_pca128_causal_past2s_mean",
        "control_split_local_shuffled_sparse_pca32",
        "control_random_gaussian_sparse_pca32",
        "control_split_local_shuffled_sparse_pca_train_selected",
        "control_random_gaussian_sparse_pca_train_selected",
        "control_split_local_shuffled_sparse_pca128",
        "control_random_gaussian_sparse_pca128",
    ]
    fold_rows: list[dict[str, Any]] = []
    lane_accum: dict[str, list[dict[str, Any]]] = {name: [] for name in lane_names}
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(gkf.split(np.zeros(len(rows)), y, groups), start=1):
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        rng = np.random.default_rng(17 + fold)
        pca64_train, pca64_test, pca64_width = mlx_pca_fit_transform(
            delta[train_idx],
            delta[train_idx],
            delta[test_idx],
            pca_width=64,
            random_seed=1000 + fold,
        )
        causal_train_pc, causal_test_pc, pca128_width = causal_pca_mean_features(
            causal_roles,
            train_idx=train_idx,
            test_idx=test_idx,
            pca_width=128,
            random_seed=2000 + fold,
        )
        small_pca_by_width: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
        for width in SMALL_PCA_WIDTH_CANDIDATES:
            small_pca_by_width[width] = causal_pca_mean_features(
                causal_roles,
                train_idx=train_idx,
                test_idx=test_idx,
                pca_width=width,
                random_seed=3000 + fold * 100 + width,
            )
        selection = select_pca_width_with_inner_video_validation(
            causal_roles=causal_roles,
            ar_base=base["AR_only"],
            y=y,
            groups=groups,
            outer_train_idx=train_idx,
            random_seed=4000 + fold,
        )
        selected_width = int(selection["selected_width"])
        selected_train_pc, selected_test_pc, selected_actual_width = small_pca_by_width[selected_width]
        shuffled_train_pc = causal_train_pc[rng.permutation(len(causal_train_pc))]
        shuffled_test_pc = causal_test_pc[rng.permutation(len(causal_test_pc))]
        random_train_pc = rng.normal(size=causal_train_pc.shape).astype(np.float32)
        random_test_pc = rng.normal(size=causal_test_pc.shape).astype(np.float32)
        pca32_train_pc, pca32_test_pc, pca32_actual_width = small_pca_by_width[LOCKED_CONFIRMATORY_PCA_WIDTH]
        shuffled_pca32_train_pc = pca32_train_pc[rng.permutation(len(pca32_train_pc))]
        shuffled_pca32_test_pc = pca32_test_pc[rng.permutation(len(pca32_test_pc))]
        random_pca32_train_pc = rng.normal(size=pca32_train_pc.shape).astype(np.float32)
        random_pca32_test_pc = rng.normal(size=pca32_test_pc.shape).astype(np.float32)
        shuffled_selected_train_pc = selected_train_pc[rng.permutation(len(selected_train_pc))]
        shuffled_selected_test_pc = selected_test_pc[rng.permutation(len(selected_test_pc))]
        random_selected_train_pc = rng.normal(size=selected_train_pc.shape).astype(np.float32)
        random_selected_test_pc = rng.normal(size=selected_test_pc.shape).astype(np.float32)
        lane_matrices = {
            "timestamp_only_baseline": (base["timestamp_only_baseline"][train_idx], base["timestamp_only_baseline"][test_idx], 0, 0, {}),
            "AR_only": (base["AR_only"][train_idx], base["AR_only"][test_idx], 0, 0, {}),
            "telemetry_change_only": (base["telemetry_change_only"][train_idx], base["telemetry_change_only"][test_idx], 0, 0, {}),
            "VJEPA_B_scout_only": (base["VJEPA_B_scout_only"][train_idx], base["VJEPA_B_scout_only"][test_idx], 0, 0, {}),
            "AR_plus_telemetry_change_plus_VJEPA_B": (
                base["AR_plus_telemetry_change_plus_VJEPA_B"][train_idx],
                base["AR_plus_telemetry_change_plus_VJEPA_B"][test_idx],
                0,
                0,
                {},
            ),
            "AR_plus_raw_sparse_current": (
                np.concatenate([base["AR_only"][train_idx], raw[train_idx]], axis=1),
                np.concatenate([base["AR_only"][test_idx], raw[test_idx]], axis=1),
                0,
                0,
                {},
            ),
            "AR_plus_raw_sparse_causal_past2s_mean": (
                np.concatenate([base["AR_only"][train_idx], raw_causal_mean[train_idx]], axis=1),
                np.concatenate([base["AR_only"][test_idx], raw_causal_mean[test_idx]], axis=1),
                0,
                0,
                {},
            ),
            "AR_plus_sparse_pca64_delta_analogue": (
                np.concatenate([base["AR_only"][train_idx], pca64_train], axis=1),
                np.concatenate([base["AR_only"][test_idx], pca64_test], axis=1),
                64,
                pca64_width,
                {},
            ),
            **{
                f"AR_plus_sparse_pca{width}_causal_past2s_mean": (
                    np.concatenate([base["AR_only"][train_idx], train_pc], axis=1),
                    np.concatenate([base["AR_only"][test_idx], test_pc], axis=1),
                    width,
                    actual_width,
                    {},
                )
                for width, (train_pc, test_pc, actual_width) in small_pca_by_width.items()
            },
            "AR_plus_sparse_pca_train_selected_causal_past2s_mean": (
                np.concatenate([base["AR_only"][train_idx], selected_train_pc], axis=1),
                np.concatenate([base["AR_only"][test_idx], selected_test_pc], axis=1),
                selected_width,
                selected_actual_width,
                selection,
            ),
            "AR_plus_sparse_pca128_causal_past2s_mean": (
                np.concatenate([base["AR_only"][train_idx], causal_train_pc], axis=1),
                np.concatenate([base["AR_only"][test_idx], causal_test_pc], axis=1),
                128,
                pca128_width,
                {},
            ),
            "AR_plus_telemetry_VJEPA_B_sparse_pca32_causal_past2s_mean": (
                np.concatenate([base["AR_plus_telemetry_change_plus_VJEPA_B"][train_idx], pca32_train_pc], axis=1),
                np.concatenate([base["AR_plus_telemetry_change_plus_VJEPA_B"][test_idx], pca32_test_pc], axis=1),
                LOCKED_CONFIRMATORY_PCA_WIDTH,
                pca32_actual_width,
                {},
            ),
            "AR_plus_telemetry_VJEPA_B_sparse_pca_train_selected_causal_past2s_mean": (
                np.concatenate([base["AR_plus_telemetry_change_plus_VJEPA_B"][train_idx], selected_train_pc], axis=1),
                np.concatenate([base["AR_plus_telemetry_change_plus_VJEPA_B"][test_idx], selected_test_pc], axis=1),
                selected_width,
                selected_actual_width,
                selection,
            ),
            "AR_plus_telemetry_VJEPA_B_sparse_pca128_causal_past2s_mean": (
                np.concatenate([base["AR_plus_telemetry_change_plus_VJEPA_B"][train_idx], causal_train_pc], axis=1),
                np.concatenate([base["AR_plus_telemetry_change_plus_VJEPA_B"][test_idx], causal_test_pc], axis=1),
                128,
                pca128_width,
                {},
            ),
            "control_split_local_shuffled_sparse_pca32": (
                np.concatenate([base["AR_only"][train_idx], shuffled_pca32_train_pc], axis=1),
                np.concatenate([base["AR_only"][test_idx], shuffled_pca32_test_pc], axis=1),
                LOCKED_CONFIRMATORY_PCA_WIDTH,
                pca32_actual_width,
                {},
            ),
            "control_random_gaussian_sparse_pca32": (
                np.concatenate([base["AR_only"][train_idx], random_pca32_train_pc], axis=1),
                np.concatenate([base["AR_only"][test_idx], random_pca32_test_pc], axis=1),
                LOCKED_CONFIRMATORY_PCA_WIDTH,
                pca32_actual_width,
                {},
            ),
            "control_split_local_shuffled_sparse_pca_train_selected": (
                np.concatenate([base["AR_only"][train_idx], shuffled_selected_train_pc], axis=1),
                np.concatenate([base["AR_only"][test_idx], shuffled_selected_test_pc], axis=1),
                selected_width,
                selected_actual_width,
                selection,
            ),
            "control_random_gaussian_sparse_pca_train_selected": (
                np.concatenate([base["AR_only"][train_idx], random_selected_train_pc], axis=1),
                np.concatenate([base["AR_only"][test_idx], random_selected_test_pc], axis=1),
                selected_width,
                selected_actual_width,
                selection,
            ),
            "control_split_local_shuffled_sparse_pca128": (
                np.concatenate([base["AR_only"][train_idx], shuffled_train_pc], axis=1),
                np.concatenate([base["AR_only"][test_idx], shuffled_test_pc], axis=1),
                128,
                pca128_width,
                {},
            ),
            "control_random_gaussian_sparse_pca128": (
                np.concatenate([base["AR_only"][train_idx], random_train_pc], axis=1),
                np.concatenate([base["AR_only"][test_idx], random_test_pc], axis=1),
                128,
                pca128_width,
                {},
            ),
        }
        train_prevalence = float(np.mean(y_train))
        train_scores = np.full(len(y_train), train_prevalence, dtype=np.float32)
        test_scores = np.full(len(y_test), train_prevalence, dtype=np.float32)
        threshold = threshold_from_train(y_train, train_scores)
        prevalence_row = {
            "selector_arm": arm,
            "fold": fold,
            "model_lane": "majority_prevalence_baseline",
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_event_count": int(np.sum(y_train)),
            "test_event_count": int(np.sum(y_test)),
            "pca_width_requested": 0,
            "pca_width_actual": 0,
            "pca_backend": "",
            "ridge_backend": "constant_train_prevalence",
            "ridge_iterations": 0,
            "decision_threshold_train_only": threshold,
            **metric_row(y_test, test_scores, threshold),
        }
        fold_rows.append(prevalence_row)
        lane_accum["majority_prevalence_baseline"].append(prevalence_row)
        for lane_name, (X_train_lane, X_test_lane, requested_pca_width, actual_pca_width, selection_info) in lane_matrices.items():
            train_scores, test_scores, fit_info = fit_predict_mlx_ridge(
                X_train_lane,
                y_train,
                X_test_lane,
                rng=rng,
            )
            threshold = threshold_from_train(y_train, train_scores)
            row = {
                "selector_arm": arm,
                "fold": fold,
                "model_lane": lane_name,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "train_event_count": int(np.sum(y_train)),
                "test_event_count": int(np.sum(y_test)),
                "pca_width_requested": requested_pca_width,
                "pca_width_actual": actual_pca_width,
                "pca_width_selected_by_inner_validation": bool(selection_info),
                "inner_validation_strategy": selection_info.get("inner_validation_strategy", ""),
                "inner_validation_pr_auc": selection_info.get("inner_validation_pr_auc", ""),
                "inner_validation_candidate_scores_json": json.dumps(selection_info.get("candidate_scores", []), sort_keys=True),
                "test_labels_used_for_selection": selection_info.get("test_labels_used_for_selection", ""),
                "pca_backend": "mlx_nipals_power_iteration" if actual_pca_width else "",
                "ridge_backend": fit_info["ridge_solver"],
                "ridge_iterations": fit_info["ridge_iterations"],
                "decision_threshold_train_only": threshold,
                **metric_row(y_test, test_scores, threshold),
            }
            fold_rows.append(row)
            lane_accum[lane_name].append(row)
    lane_rows: list[dict[str, Any]] = []
    for lane_name, values in lane_accum.items():
        if not values:
            continue
        lane_rows.append(
            {
                "selector_arm": arm,
                "model_lane": lane_name,
                "folds": len(values),
                "mean_pr_auc": float(np.nanmean([safe_float(row.get("pr_auc")) for row in values])),
                "mean_roc_auc": float(np.nanmean([safe_float(row.get("roc_auc")) for row in values])),
                "mean_f1": float(np.nanmean([safe_float(row.get("f1")) for row in values])),
                "mean_balanced_accuracy": float(np.nanmean([safe_float(row.get("balanced_accuracy")) for row in values])),
                "mean_top_5pct_recall": float(np.nanmean([safe_float(row.get("top_5pct_recall")) for row in values])),
                "mean_top_10pct_recall": float(np.nanmean([safe_float(row.get("top_10pct_recall")) for row in values])),
                "mean_pca_width_actual": float(np.nanmean([safe_float(row.get("pca_width_actual"), 0.0) for row in values])),
                "selected_widths": ",".join(str(int(safe_float(row.get("pca_width_requested"), 0))) for row in values if row.get("pca_width_selected_by_inner_validation")),
                "mean_inner_validation_pr_auc": float(np.nanmean([safe_float(row.get("inner_validation_pr_auc")) for row in values if row.get("pca_width_selected_by_inner_validation")])) if any(row.get("pca_width_selected_by_inner_validation") for row in values) else math.nan,
            }
        )
    return lane_rows, fold_rows


def add_gate_rows(lane_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by = {(row["selector_arm"], row["model_lane"]): row for row in lane_rows}
    comparisons = [
        ("sparse_pca128_vs_ar", "hybrid_top5_selected", "AR_plus_sparse_pca128_causal_past2s_mean", "AR_only"),
        ("sparse_pca128_vs_ar_tel_scout", "hybrid_top5_selected", "AR_plus_telemetry_VJEPA_B_sparse_pca128_causal_past2s_mean", "AR_plus_telemetry_change_plus_VJEPA_B"),
        ("pca128_vs_raw_current", "hybrid_top5_selected", "AR_plus_sparse_pca128_causal_past2s_mean", "AR_plus_raw_sparse_current"),
        ("pca32_locked_vs_ar", "hybrid_top5_selected", "AR_plus_sparse_pca32_causal_past2s_mean", "AR_only"),
        ("pca32_locked_vs_raw_current", "hybrid_top5_selected", "AR_plus_sparse_pca32_causal_past2s_mean", "AR_plus_raw_sparse_current"),
        ("pca32_locked_vs_raw_causal_mean", "hybrid_top5_selected", "AR_plus_sparse_pca32_causal_past2s_mean", "AR_plus_raw_sparse_causal_past2s_mean"),
        ("pca32_locked_vs_ar_tel_scout", "hybrid_top5_selected", "AR_plus_sparse_pca32_causal_past2s_mean", "AR_plus_telemetry_change_plus_VJEPA_B"),
        ("pca32_fusion_vs_ar_tel_scout", "hybrid_top5_selected", "AR_plus_telemetry_VJEPA_B_sparse_pca32_causal_past2s_mean", "AR_plus_telemetry_change_plus_VJEPA_B"),
        ("pca32_locked_vs_shuffled_control", "hybrid_top5_selected", "AR_plus_sparse_pca32_causal_past2s_mean", "control_split_local_shuffled_sparse_pca32"),
        ("pca32_locked_vs_random_control", "hybrid_top5_selected", "AR_plus_sparse_pca32_causal_past2s_mean", "control_random_gaussian_sparse_pca32"),
        ("pca_train_selected_vs_ar", "hybrid_top5_selected", "AR_plus_sparse_pca_train_selected_causal_past2s_mean", "AR_only"),
        ("pca_train_selected_vs_ar_tel_scout", "hybrid_top5_selected", "AR_plus_sparse_pca_train_selected_causal_past2s_mean", "AR_plus_telemetry_change_plus_VJEPA_B"),
        ("pca_train_selected_fusion_vs_ar_tel_scout", "hybrid_top5_selected", "AR_plus_telemetry_VJEPA_B_sparse_pca_train_selected_causal_past2s_mean", "AR_plus_telemetry_change_plus_VJEPA_B"),
        ("pca_train_selected_vs_raw_current", "hybrid_top5_selected", "AR_plus_sparse_pca_train_selected_causal_past2s_mean", "AR_plus_raw_sparse_current"),
        ("pca_train_selected_vs_raw_causal_mean", "hybrid_top5_selected", "AR_plus_sparse_pca_train_selected_causal_past2s_mean", "AR_plus_raw_sparse_causal_past2s_mean"),
        ("pca_train_selected_vs_pca64_delta", "hybrid_top5_selected", "AR_plus_sparse_pca_train_selected_causal_past2s_mean", "AR_plus_sparse_pca64_delta_analogue"),
        ("pca_train_selected_vs_shuffled_control", "hybrid_top5_selected", "AR_plus_sparse_pca_train_selected_causal_past2s_mean", "control_split_local_shuffled_sparse_pca_train_selected"),
        ("pca_train_selected_vs_random_control", "hybrid_top5_selected", "AR_plus_sparse_pca_train_selected_causal_past2s_mean", "control_random_gaussian_sparse_pca_train_selected"),
        ("pca_train_selected_vs_coverage_random_selected", "hybrid_top5_selected", "AR_plus_sparse_pca_train_selected_causal_past2s_mean", "AR_plus_sparse_pca_train_selected_causal_past2s_mean"),
        ("pca_train_selected_vs_fixed_random_selected", "hybrid_top5_selected", "AR_plus_sparse_pca_train_selected_causal_past2s_mean", "AR_plus_sparse_pca_train_selected_causal_past2s_mean"),
        ("pca32_locked_vs_coverage_random_pca32", "hybrid_top5_selected", "AR_plus_sparse_pca32_causal_past2s_mean", "AR_plus_sparse_pca32_causal_past2s_mean"),
        ("pca32_locked_vs_fixed_random_pca32", "hybrid_top5_selected", "AR_plus_sparse_pca32_causal_past2s_mean", "AR_plus_sparse_pca32_causal_past2s_mean"),
        ("hybrid_sparse_vs_coverage_random_sparse", "hybrid_top5_selected", "AR_plus_sparse_pca128_causal_past2s_mean", "AR_plus_sparse_pca128_causal_past2s_mean"),
    ]
    for gate, arm, lhs, rhs in comparisons:
        lhs_row = by.get((arm, lhs))
        if gate in {
            "hybrid_sparse_vs_coverage_random_sparse",
            "pca_train_selected_vs_coverage_random_selected",
            "pca32_locked_vs_coverage_random_pca32",
        }:
            rhs_arm = "coverage_matched_random_to_hybrid"
        elif gate in {"pca_train_selected_vs_fixed_random_selected", "pca32_locked_vs_fixed_random_pca32"}:
            rhs_arm = "fixed_random_same_budget"
        else:
            rhs_arm = arm
        rhs_row = by.get((rhs_arm, rhs))
        if not lhs_row or not rhs_row:
            out.append({"gate": gate, "status": "not_evaluable", "notes": "missing lane"})
            continue
        delta = safe_float(lhs_row.get("mean_pr_auc")) - safe_float(rhs_row.get("mean_pr_auc"))
        out.append(
            {
                "gate": gate,
                "lhs_selector_arm": arm,
                "lhs_model_lane": lhs,
                "rhs_selector_arm": rhs_arm,
                "rhs_model_lane": rhs,
                "mean_pr_auc_delta": delta,
                "pass": bool(math.isfinite(delta) and delta > 0),
            }
        )
    delta_over_ar_comparisons = [
        (
            "pca32_locked_delta_over_ar_vs_coverage_random_delta_over_ar",
            "hybrid_top5_selected",
            "coverage_matched_random_to_hybrid",
            "AR_plus_sparse_pca32_causal_past2s_mean",
        ),
        (
            "pca32_locked_delta_over_ar_vs_fixed_random_delta_over_ar",
            "hybrid_top5_selected",
            "fixed_random_same_budget",
            "AR_plus_sparse_pca32_causal_past2s_mean",
        ),
        (
            "pca_train_selected_delta_over_ar_vs_coverage_random_delta_over_ar",
            "hybrid_top5_selected",
            "coverage_matched_random_to_hybrid",
            "AR_plus_sparse_pca_train_selected_causal_past2s_mean",
        ),
        (
            "pca_train_selected_delta_over_ar_vs_fixed_random_delta_over_ar",
            "hybrid_top5_selected",
            "fixed_random_same_budget",
            "AR_plus_sparse_pca_train_selected_causal_past2s_mean",
        ),
    ]
    for gate, lhs_arm, rhs_arm, lane in delta_over_ar_comparisons:
        lhs_row = by.get((lhs_arm, lane))
        lhs_ar = by.get((lhs_arm, "AR_only"))
        rhs_row = by.get((rhs_arm, lane))
        rhs_ar = by.get((rhs_arm, "AR_only"))
        if not lhs_row or not lhs_ar or not rhs_row or not rhs_ar:
            out.append({"gate": gate, "status": "not_evaluable", "notes": "missing lane"})
            continue
        lhs_delta = safe_float(lhs_row.get("mean_pr_auc")) - safe_float(lhs_ar.get("mean_pr_auc"))
        rhs_delta = safe_float(rhs_row.get("mean_pr_auc")) - safe_float(rhs_ar.get("mean_pr_auc"))
        delta = lhs_delta - rhs_delta
        out.append(
            {
                "gate": gate,
                "lhs_selector_arm": lhs_arm,
                "lhs_model_lane": lane,
                "rhs_selector_arm": rhs_arm,
                "rhs_model_lane": lane,
                "lhs_delta_over_ar": lhs_delta,
                "rhs_delta_over_ar": rhs_delta,
                "mean_pr_auc_delta": delta,
                "pass": bool(math.isfinite(delta) and delta > 0),
            }
        )
    return out


def report_lines_queue(queue_summary: dict[str, Any], output_root: Path, external_cache_root: Path, *, run_title: str) -> list[str]:
    def portable_path(path: Path) -> str:
        root_text = os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT")
        if not root_text:
            return str(path)
        try:
            relative = path.resolve().relative_to(Path(root_text).resolve())
        except ValueError:
            return str(path)
        return f"${{NEURAL_BRIDGE_EXTERNAL_ROOT}}/{relative}"

    return [
        f"# {run_title} Queue",
        "",
        f"- benchmark_mode: `{BENCHMARK_MODE}`",
        f"- max actual encoded windows: `{queue_summary['max_actual_windows']}`",
        f"- queued rows: `{queue_summary['queued_rows']}`",
        f"- unique actual windows: `{queue_summary['unique_actual_windows']}`",
        f"- selector video count: `{queue_summary.get('video_count')}`",
        "- subset decision: existing 100-video scout/selector cache was not available; reused the corrected 50-video selector subset and expanded sparse coverage within it.",
        f"- causal roles: `{queue_summary['causal_roles']}`",
        f"- selector config hash: `{queue_summary.get('selector_config_hash')}`",
        f"- teacher dtype: `{queue_summary.get('teacher_dtype', 'bfloat16')}`",
        f"- future rows included: `{str(queue_summary['future_rows_included']).lower()}`",
        f"- output root: `{output_root}`",
        f"- external cache root: `{portable_path(external_cache_root)}`",
        "",
        "## Arm Counts",
        *[
            f"- {arm}: {count} unique actual windows; "
            f"{queue_summary['arm_queue_row_counts'].get(arm, 0)} queued causal rows; "
            f"{queue_summary['arm_center_counts'].get(arm, 0)} candidate centers"
            for arm, count in sorted(queue_summary["arm_unique_window_counts"].items())
        ],
    ]


def report_lines_runtime(runtime_summary: dict[str, Any], *, run_title: str) -> list[str]:
    successful = safe_float(runtime_summary.get("successful_windows"), 0.0)
    total_runtime = safe_float(runtime_summary.get("total_runtime_seconds"), 0.0)
    sec_per = total_runtime / successful if successful else math.nan
    return [
        f"# {run_title} Runtime",
        "",
        f"- unique actual windows: `{runtime_summary.get('unique_actual_windows')}`",
        f"- successful windows: `{runtime_summary.get('successful_windows')}`",
        f"- teacher dtype: `{runtime_summary.get('teacher_dtype', 'bfloat16')}`",
        f"- loose cache backfill enabled: `{str(runtime_summary.get('allow_cache_backfill', False)).lower()}`",
        f"- cache hits: `{runtime_summary.get('cache_hits')}`",
        f"- legacy cache hits backfilled to strict fingerprints: `{runtime_summary.get('legacy_cache_hits')}`",
        f"- physical-window cache hits backfilled to strict fingerprints: `{runtime_summary.get('physical_cache_hits', 0)}`",
        f"- failed windows: `{runtime_summary.get('failed_windows')}`",
        f"- total runtime seconds: `{total_runtime:.3f}`",
        f"- seconds per successful window: `{sec_per:.3f}`",
        f"- projected 1000-window runtime seconds: `{sec_per * 1000:.1f}`",
        f"- projected 2000-window runtime seconds: `{sec_per * 2000:.1f}`",
        f"- peak RSS bytes: `{runtime_summary.get('peak_rss_bytes')}`",
    ]


def report_lines_results(
    lane_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    runtime_summary: dict[str, Any],
    *,
    run_title: str = "AGAIN Sparse TRIBE Teacher 500",
    max_actual_windows: int = 500,
) -> list[str]:
    by = {(row["selector_arm"], row["model_lane"]): row for row in lane_rows}

    def pr(arm: str, lane: str) -> str:
        row = by.get((arm, lane))
        return "n/a" if not row else f"{100 * safe_float(row.get('mean_pr_auc')):.2f}%"

    def gate(gate_name: str) -> str:
        row = next((item for item in gate_rows if item["gate"] == gate_name), None)
        if not row or "pass" not in row:
            return "not evaluable"
        return f"{'pass' if row['pass'] else 'fail'} (delta {100 * safe_float(row.get('mean_pr_auc_delta')):.2f} pp)"

    def width_line(width: int) -> str:
        lane = f"AR_plus_sparse_pca{width}_causal_past2s_mean"
        row = by.get(("hybrid_top5_selected", lane))
        if not row:
            return f"- AR + sparse PCA{width}: `n/a`"
        return (
            f"- AR + sparse PCA{width}: PR-AUC `{100 * safe_float(row.get('mean_pr_auc')):.2f}%`, "
            f"mean actual width `{safe_float(row.get('mean_pca_width_actual')):.1f}`"
        )

    selected = by.get(("hybrid_top5_selected", "AR_plus_sparse_pca_train_selected_causal_past2s_mean"))
    selected_widths = selected.get("selected_widths", "") if selected else ""
    selected_inner = selected.get("mean_inner_validation_pr_auc", math.nan) if selected else math.nan

    return [
        f"# {run_title} Small PCA Results",
        "",
        "## Scope",
        "",
        f"- This sparse teacher pilot uses `{SCOUT_VALIDATION_VIDEO_COUNT}` selected AGAIN videos out of `{FULL_AGAIN_VIDEO_COUNT}` total videos.",
        f"- That is `{100 * SCOUT_VALIDATION_VIDEO_COUNT / FULL_AGAIN_VIDEO_COUNT:.1f}%` of the dataset.",
        f"- Actual expensive-window budget: `{max_actual_windows}`.",
        "- Existing 100-video selector cache was not available, so this run expands sparse coverage on the corrected 50-video selector subset.",
        "- AR-only in this report is computed only on the same sparse pilot center rows as the sparse TRIBE/PCA rows.",
        "- It is not the full-AGAIN AR baseline and must not be read as a 995-video comparison.",
        "- AR + sparse PCA128 cannot be tested against all 995 videos until matching sparse PCA rows exist for that full scope.",
        "",
        "## Executive Verdict",
        f"- Completed sparse ViT-G/TRIBE windows: `{runtime_summary.get('successful_windows')}`",
        f"- Hybrid AR-only PR-AUC: `{pr('hybrid_top5_selected', 'AR_only')}`",
        f"- Hybrid AR + raw sparse current PR-AUC: `{pr('hybrid_top5_selected', 'AR_plus_raw_sparse_current')}`",
        f"- Hybrid AR + raw sparse causal mean PR-AUC: `{pr('hybrid_top5_selected', 'AR_plus_raw_sparse_causal_past2s_mean')}`",
        f"- Hybrid AR + sparse PCA128 causal PR-AUC: `{pr('hybrid_top5_selected', 'AR_plus_sparse_pca128_causal_past2s_mean')}`",
        f"- Hybrid AR + locked sparse PCA32 causal PR-AUC: `{pr('hybrid_top5_selected', 'AR_plus_sparse_pca32_causal_past2s_mean')}`",
        f"- Hybrid AR + train-selected sparse PCA causal PR-AUC: `{pr('hybrid_top5_selected', 'AR_plus_sparse_pca_train_selected_causal_past2s_mean')}`",
        f"- Train-selected PCA widths by grouped outer fold: `{selected_widths or 'n/a'}`",
        f"- Mean inner-validation PR-AUC for selected-width lane: `{100 * safe_float(selected_inner):.2f}%`" if math.isfinite(safe_float(selected_inner)) else "- Mean inner-validation PR-AUC for selected-width lane: `n/a`",
        f"- Hybrid AR + telemetry + V-JEPA-B + locked sparse PCA32 causal PR-AUC: `{pr('hybrid_top5_selected', 'AR_plus_telemetry_VJEPA_B_sparse_pca32_causal_past2s_mean')}`",
        f"- Hybrid AR + telemetry + V-JEPA-B + train-selected sparse PCA causal PR-AUC: `{pr('hybrid_top5_selected', 'AR_plus_telemetry_VJEPA_B_sparse_pca_train_selected_causal_past2s_mean')}`",
        f"- Hybrid AR + telemetry + V-JEPA-B + sparse PCA128 causal PR-AUC: `{pr('hybrid_top5_selected', 'AR_plus_telemetry_VJEPA_B_sparse_pca128_causal_past2s_mean')}`",
        f"- Coverage-random AR + sparse PCA128 causal PR-AUC: `{pr('coverage_matched_random_to_hybrid', 'AR_plus_sparse_pca128_causal_past2s_mean')}`",
        f"- Oracle+background AR + sparse PCA128 causal PR-AUC: `{pr(ORACLE_EVALUATION_ARM, 'AR_plus_sparse_pca128_causal_past2s_mean')}`",
        "",
        "## Smaller PCA Width Re-analysis",
        "",
        "- This section is cache-only: it reuses existing sparse TRIBE window features and fits PCA on train rows only.",
        "- Reported PCA-width rows are AR + sparse PCA lanes, not PCA-only lanes.",
        "- Candidate widths are `8`, `16`, `32`, and `64`; the selected-width lane uses grouped train/inner validation only.",
        *[width_line(width) for width in SMALL_PCA_WIDTH_CANDIDATES],
        "",
        "## Gate Summary",
        f"- sparse PCA128 vs AR-only: {gate('sparse_pca128_vs_ar')}",
        f"- sparse PCA128 vs AR + telemetry + V-JEPA-B: {gate('sparse_pca128_vs_ar_tel_scout')}",
        f"- sparse PCA128 vs raw sparse current: {gate('pca128_vs_raw_current')}",
        f"- locked PCA32 vs AR-only: {gate('pca32_locked_vs_ar')}",
        f"- locked PCA32 vs raw sparse current: {gate('pca32_locked_vs_raw_current')}",
        f"- locked PCA32 vs raw sparse causal mean: {gate('pca32_locked_vs_raw_causal_mean')}",
        f"- locked PCA32 vs AR + telemetry + V-JEPA-B: {gate('pca32_locked_vs_ar_tel_scout')}",
        f"- AR + telemetry + V-JEPA-B + locked PCA32 vs AR + telemetry + V-JEPA-B: {gate('pca32_fusion_vs_ar_tel_scout')}",
        f"- AR + locked PCA32 vs AR + shuffled sparse PCA32 control: {gate('pca32_locked_vs_shuffled_control')}",
        f"- AR + locked PCA32 vs AR + random Gaussian PCA32 control: {gate('pca32_locked_vs_random_control')}",
        f"- AR + train-selected small PCA vs AR-only: {gate('pca_train_selected_vs_ar')}",
        f"- AR + train-selected small PCA vs AR + telemetry + V-JEPA-B: {gate('pca_train_selected_vs_ar_tel_scout')}",
        f"- AR + telemetry + V-JEPA-B + train-selected small PCA vs AR + telemetry + V-JEPA-B: {gate('pca_train_selected_fusion_vs_ar_tel_scout')}",
        f"- AR + train-selected small PCA vs AR + raw sparse current: {gate('pca_train_selected_vs_raw_current')}",
        f"- AR + train-selected small PCA vs AR + raw sparse causal mean: {gate('pca_train_selected_vs_raw_causal_mean')}",
        f"- AR + train-selected small PCA vs AR + PCA64-delta analogue: {gate('pca_train_selected_vs_pca64_delta')}",
        f"- AR + train-selected small PCA vs AR + shuffled sparse control: {gate('pca_train_selected_vs_shuffled_control')}",
        f"- AR + train-selected small PCA vs AR + random Gaussian sparse control: {gate('pca_train_selected_vs_random_control')}",
        f"- AR + train-selected small PCA vs fixed coverage-random AR + selected small PCA: {gate('pca_train_selected_vs_coverage_random_selected')}",
        f"- AR + train-selected small PCA vs true same-budget fixed-random AR + selected small PCA: {gate('pca_train_selected_vs_fixed_random_selected')}",
        f"- locked PCA32 vs coverage-random PCA32: {gate('pca32_locked_vs_coverage_random_pca32')}",
        f"- AR + locked PCA32 vs true same-budget fixed-random AR + PCA32: {gate('pca32_locked_vs_fixed_random_pca32')}",
        f"- hybrid sparse vs coverage-random sparse: {gate('hybrid_sparse_vs_coverage_random_sparse')}",
        "",
        "## Delta-over-AR Matched-Arm Checks",
        "",
        "- These checks compare each arm's improvement over its own AR baseline, avoiding unfair absolute PR-AUC comparisons across selector distributions.",
        f"- AR + locked PCA32 hybrid delta-over-AR vs coverage-random delta-over-AR: {gate('pca32_locked_delta_over_ar_vs_coverage_random_delta_over_ar')}",
        f"- AR + locked PCA32 hybrid delta-over-AR vs true fixed-random delta-over-AR: {gate('pca32_locked_delta_over_ar_vs_fixed_random_delta_over_ar')}",
        f"- AR + train-selected small PCA hybrid delta-over-AR vs coverage-random delta-over-AR: {gate('pca_train_selected_delta_over_ar_vs_coverage_random_delta_over_ar')}",
        f"- AR + train-selected small PCA hybrid delta-over-AR vs true fixed-random delta-over-AR: {gate('pca_train_selected_delta_over_ar_vs_fixed_random_delta_over_ar')}",
        "",
        "## Decision Rule",
        "- This is a sparse teacher pilot only, not final AGAIN proof.",
        "- Promote nothing unless the sparse lane beats AR, raw sparse current/causal mean, cheap AR+telemetry+V-JEPA-B, shuffled/random nuisance controls, and matched-random sparse controls.",
        "- If a lane beats AR but loses to raw sparse, shuffled/random, or matched-random controls, treat it as non-confirmed sparse-sample signal.",
        "- Do not approve full AGAIN scaling from this sparse pilot alone.",
    ]


def run_sparse_teacher_500(
    *,
    output_root: Path,
    external_cache_root: Path,
    config: SparseTeacherConfig,
) -> dict[str, Any]:
    assert_again_only_output_path(output_root)
    assert_again_only_output_path(external_cache_root)
    output_root.mkdir(parents=True, exist_ok=False)
    external_cache_root.mkdir(parents=True, exist_ok=True)
    mlx_memory_config = configure_mlx_memory_from_env(mx, label="again_sparse_vitg_tribe_teacher")
    teacher_dtype = validate_teacher_dtype(config.teacher_dtype)

    root = external_root()
    vjepa_weights_dir = vjepa21_vitg_dir_for_dtype(root, teacher_dtype)
    tribe_model_dir = root / "models" / "tribe-mlx" / "zimengxiong-tribev2-mlx"
    vjepa_sha = sha256_file(vjepa_weights_dir / "model.safetensors")
    tribe_weights_path = tribe_weights_path_for_dtype(tribe_model_dir, teacher_dtype)
    tribe_sha = sha256_file(tribe_weights_path)
    arm_window_budgets = dict(config.arm_window_budgets) if config.arm_window_budgets else arm_budgets_for_window_budget(config.max_actual_windows)
    selector_digest = selector_config_hash(arm_window_budgets, config.selector_validation_root, config.causal_relative_seconds)

    feature_path = config.selector_validation_root / "again_real_scout_feature_rows.csv"
    feature_rows = read_csv_rows(feature_path)
    labels_by_video = group_rows_by_video(feature_rows)
    for video_rows in labels_by_video.values():
        labels = compute_label_vectors(video_rows, threshold=0.05)
        for row, pre2, pre4, pre6, pre8 in zip(
            video_rows,
            labels["pre_spike_2s"],
            labels["pre_spike_4s"],
            labels["pre_spike_6s"],
            labels["pre_spike_8s"],
        ):
            row["pre_spike_2s"] = str(bool(pre2)).lower()
            row["pre_spike_4s"] = str(bool(pre4)).lower()
            row["pre_spike_6s"] = str(bool(pre6)).lower()
            row["pre_spike_8s"] = str(bool(pre8)).lower()
    feature_rows = [row for rows in labels_by_video.values() for row in rows]

    queue, queue_summary = build_sparse_teacher_queue(
        feature_rows,
        max_actual_windows=config.max_actual_windows,
        rng=random.Random(config.random_seed),
        vjepa21_sha256=vjepa_sha,
        tribe_sha256=tribe_sha,
        arm_window_budgets=arm_window_budgets,
        selector_config_digest=selector_digest,
        causal_relative_seconds=config.causal_relative_seconds,
        teacher_dtype=teacher_dtype,
    )
    write_csv(output_root / f"{config.run_label}_queue.csv", queue)
    write_json(output_root / f"{config.run_label}_queue_summary.json", queue_summary)
    queue_report = "\n".join(report_lines_queue(queue_summary, output_root, external_cache_root, run_title=config.run_title)) + "\n"
    Path("reports").mkdir(exist_ok=True)
    queue_report_path = Path("reports") / f"{config.run_label}_queue_{config.report_date}.md"
    queue_report_path.write_text(queue_report, encoding="utf-8")

    features_by_fp, runtime_rows, runtime_summary = encode_sparse_windows(
        queue,
        external_cache_root=external_cache_root,
        vjepa_weights_dir=vjepa_weights_dir,
        tribe_model_dir=tribe_model_dir,
        vjepa_sha256=vjepa_sha,
        tribe_sha256=tribe_sha,
        teacher_dtype=teacher_dtype,
        allow_cache_backfill=config.allow_cache_backfill,
        gpu_workers=config.gpu_workers,
        preprocess_workers=config.preprocess_workers,
        ready_queue_max_size=config.ready_queue_max_size,
        writer_queue_max_size=config.writer_queue_max_size,
        microbatch_size=config.microbatch_size,
        compile_encoder=config.compile_encoder,
    )
    write_csv(output_root / f"{config.run_label}_runtime.csv", runtime_rows)
    write_json(output_root / f"{config.run_label}_runtime_summary.json", runtime_summary)
    runtime_report_path = Path("reports") / f"{config.run_label}_runtime_{config.report_date}.md"
    runtime_report_path.write_text("\n".join(report_lines_runtime(runtime_summary, run_title=config.run_title)) + "\n", encoding="utf-8")

    center_rows, sparse_features = build_center_rows(queue, features_by_fp, causal_relative_seconds=config.causal_relative_seconds)
    add_ar_features(center_rows, feature_rows)
    write_csv(output_root / f"{config.run_label}_center_rows.csv", center_rows)
    eval_center_rows = list(center_rows)
    for row in center_rows:
        if row["selector_arm"] in {"oracle_upper_bound", "low_salience_background", "sparse_anchor_windows"}:
            copied = dict(row)
            copied["selector_arm"] = ORACLE_EVALUATION_ARM
            copied["selector_role"] = "oracle"
            eval_center_rows.append(copied)
    lane_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for arm in EVALUATION_ARMS:
        arm_lane_rows, arm_fold_rows = evaluate_arm(arm, eval_center_rows, sparse_features)
        lane_rows.extend(arm_lane_rows)
        fold_rows.extend(arm_fold_rows)
    gate_rows = add_gate_rows(lane_rows)
    write_csv(output_root / f"{config.run_label}_lane_results.csv", lane_rows)
    write_csv(output_root / f"{config.run_label}_fold_results.csv", fold_rows)
    write_csv(output_root / f"{config.run_label}_gate_checks.csv", gate_rows)

    results_report_path = Path("reports") / f"{config.run_label}_small_pca_results_{config.report_date}.md"
    results_report_path.write_text(
        "\n".join(
            report_lines_results(
                lane_rows,
                gate_rows,
                runtime_summary,
                run_title=config.run_title,
                max_actual_windows=config.max_actual_windows,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_mode": BENCHMARK_MODE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "again_only": True,
        "mlx_only": True,
        "cuda_dependencies_installed": False,
        "dense_again_vitg_encoding_run": False,
        "actual_vitg_tribe_window_budget": config.max_actual_windows,
        "arm_window_budgets": arm_window_budgets,
        "causal_relative_seconds": list(config.causal_relative_seconds),
        "selector_config_hash": selector_digest,
        "actual_unique_windows_queued": queue_summary["unique_actual_windows"],
        "actual_successful_windows": runtime_summary["successful_windows"],
        "models_trained": True,
        "training_scope": "small_sparse_teacher_pilot_grouped_video_cv",
        "alignment_policy": AGAIN_ALIGNMENT_POLICY,
        "selector_validation_root": str(config.selector_validation_root),
        "output_root": str(output_root),
        "external_cache_root": str(external_cache_root),
        "vjepa21_weights_dir": str(vjepa_weights_dir),
        "vjepa21_model_sha256": vjepa_sha,
        "tribe_model_dir": str(tribe_model_dir),
        "tribe_weights_path": str(tribe_weights_path),
        "tribe_model_sha256": tribe_sha,
        "teacher_dtype": teacher_dtype,
        "allow_cache_backfill": bool(config.allow_cache_backfill),
        "gpu_workers": int(config.gpu_workers),
        "preprocess_workers": int(config.preprocess_workers),
        "ready_queue_max_size": int(config.ready_queue_max_size),
        "writer_queue_max_size": int(config.writer_queue_max_size),
        "microbatch_size": int(config.microbatch_size),
        "compile_encoder": bool(config.compile_encoder),
        "single_gpu_owner": True,
        "mlx_memory_config": mlx_memory_config,
        "queue_report": str(queue_report_path),
        "runtime_report": str(runtime_report_path),
        "results_report": str(results_report_path),
        "no_future_rows_in_final_features": True,
        "tribe_head_pool_policy": TRIBE_HEAD_POOL_POLICY,
        "pca_train_only": True,
    }
    write_json(output_root / "run_manifest.json", run_manifest)
    return {
        "manifest": run_manifest,
        "queue_summary": queue_summary,
        "runtime_summary": runtime_summary,
        "lane_rows": lane_rows,
        "gate_rows": gate_rows,
    }
