#!/usr/bin/env python3
"""Cache-only TRIBE v2 post-pass over dense H100 V-JEPA 2.1 caches.

This script only creates a downstream-ready TRIBE v2 output bundle.
It does not decode video, run V-JEPA, fit PCA, train bridge models, run
benchmarks, create deltas, or inspect labels beyond preserving existing row
alignment fields.

Input contract:
  output/cache/<video_id>/vjepa21_hidden_states.npz
    features: [rows, 20, 1, 1408], float16
    temporal_mean: [rows, 20, 32, 1408], float16
    temporal_std: [rows, 20, 32, 1408], float16

TRIBE v2 video input contract:
  [B, L, D, T] == [1, 2, 1408, rows]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from neuraltrain.models.common import Mlp, SubjectLayers
from neuraltrain.models.transformer import TransformerEncoder


TRIBE_GROUP_LAYERS = (0.5, 0.75, 1.0)
EXPECTED_FEATURE_SHAPE_TAIL = (20, 1, 1408)
EXPECTED_GROUPED_SHAPE_TAIL = (2, 1408)
EXPECTED_CORTICAL_WIDTH = 20484
EXPECTED_ROW_RATE_HZ = 2.0
EXPECTED_CLIP_SECONDS = 4.0
EXPECTED_IMAGE_SIZE = 256
EXPECTED_DECODE_HZ = 16.0
EXPECTED_FRAMES_PER_CLIP = 64
MEDIAN_SAMPLE_WINDOW_SECONDS = 3.9375

REQUIRED_CACHE_FILES = (
    "vjepa21_hidden_states.npz",
    "rows.csv",
    "manifest.json",
    "status.json",
    "preprocessing.json",
)

REQUIRED_NPZ_KEYS = (
    "features",
    "time_seconds",
    "sample_frame_indices",
    "sample_time_seconds",
    "selected_state_indices",
    "luma_mean",
    "luma_std",
    "frame_luma_std_mean",
    "motion_absdiff_mean",
    "black_frame_fraction",
    "duplicate_frame_fraction",
)

REQUIRED_TEMPORAL_KEYS = ("temporal_mean", "temporal_std")

SUCCESS_OUTPUT_FILES = (
    "tribe_v2_cortical_predictions.npz",
    "baseline_features_rowlevel.npz",
    "vjepa_temporal_diagnostics.npz",
    "manifest.json",
    "status.json",
    "input_mapping.json",
    "diagnostics.json",
    "rows_aligned.csv",
)


class StageError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


@dataclass
class CacheBundle:
    cache_dir: Path
    rows_csv: list[dict[str, str]]
    source_manifest: dict[str, Any]
    source_status: dict[str, Any]
    preprocessing: dict[str, Any]
    arrays: dict[str, np.ndarray]
    rows_csv_count: int


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_json_load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise StageError("load_cache", f"Cannot read JSON {path}: {exc}") from exc


def read_rows_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception as exc:
        raise StageError("load_cache", f"Cannot read rows CSV {path}: {exc}") from exc


def array_stats(arr: np.ndarray, prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        f"{prefix}_shape": list(arr.shape),
        f"{prefix}_dtype": str(arr.dtype),
        f"{prefix}_nan_count": 0,
        f"{prefix}_inf_count": 0,
    }
    if arr.size == 0:
        out.update({
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
        })
        return out
    if np.issubdtype(arr.dtype, np.floating):
        finite_arr = arr.astype(np.float32, copy=False)
        out[f"{prefix}_nan_count"] = int(np.isnan(finite_arr).sum())
        out[f"{prefix}_inf_count"] = int(np.isinf(finite_arr).sum())
    else:
        finite_arr = arr
    finite_mask = np.isfinite(finite_arr) if np.issubdtype(finite_arr.dtype, np.floating) else np.ones(arr.shape, dtype=bool)
    if not finite_mask.any():
        out.update({
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
        })
        return out
    values = finite_arr[finite_mask].astype(np.float32, copy=False)
    out.update({
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_mean": float(values.mean(dtype=np.float64)),
        f"{prefix}_std": float(values.std(dtype=np.float64)),
    })
    return out


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.savez(tmp, **arrays)
    actual_tmp = Path(str(tmp) + ".npz") if not str(tmp).endswith(".npz") else tmp
    actual_tmp.replace(path)


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_csv_dicts(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def try_write_parquet(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        return False
    table = pa.Table.from_pylist(rows)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
    return True


def load_tribe_model_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("tribev2_model_file", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import TRIBE model file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_tribe_model(tribe_model_py: Path, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    module = load_tribe_model_module(tribe_model_py)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    build_args = ckpt["model_build_args"]
    state = {key.removeprefix("model."): value for key, value in ckpt["state_dict"].items()}
    config = module.FmriEncoder(
        projector=Mlp(norm_layer="layer", activation_layer="gelu"),
        combiner=None,
        encoder=TransformerEncoder(
            heads=8,
            depth=8,
            attn_dropout=0.0,
            ff_dropout=0.0,
            layer_dropout=0.0,
        ),
        time_pos_embedding=True,
        subject_embedding=False,
        subject_layers=SubjectLayers(n_subjects=1, bias=True, average_subjects=False),
        hidden=1152,
        max_seq_len=1024,
        dropout=0.0,
        extractor_aggregation="cat",
        layer_aggregation="cat",
        linear_baseline=False,
        modality_dropout=0.0,
        temporal_dropout=0.0,
        low_rank_head=2048,
        temporal_smoothing=None,
    )
    model = config.build(**build_args)
    model.load_state_dict(state, strict=True)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def group_mean_tribe_video_layers(features: np.ndarray) -> np.ndarray:
    """Return [rows, 2, 1408] grouped video features for TRIBE."""
    if features.ndim != 4 or tuple(features.shape[1:]) != EXPECTED_FEATURE_SHAPE_TAIL:
        raise StageError("adapter", f"Expected features shape [rows,20,1,1408], got {features.shape}")
    raw = features[:, :, 0, :].astype(np.float32, copy=False)
    n_layers = int(raw.shape[1])
    layer_indices = [int(value * (n_layers - 1)) for value in TRIBE_GROUP_LAYERS]
    layer_indices = sorted(set(layer_indices))
    if len(layer_indices) < 2:
        raise StageError("adapter", f"Cannot form TRIBE layer groups from {layer_indices}")
    layer_indices[-1] += 1
    groups = [raw[:, start:end, :].mean(axis=1) for start, end in zip(layer_indices[:-1], layer_indices[1:])]
    grouped = np.stack(groups, axis=1)
    if tuple(grouped.shape[1:]) != EXPECTED_GROUPED_SHAPE_TAIL:
        raise StageError("adapter", f"Grouped TRIBE feature shape mismatch: {grouped.shape}")
    if not np.isfinite(grouped).all():
        raise StageError("adapter", "Grouped TRIBE feature contains NaN or inf")
    return grouped.astype(np.float16, copy=False)


def complete_cache_dirs(cache_root: Path, *, fail_on_missing_cache: bool) -> list[Path]:
    if not cache_root.exists():
        if fail_on_missing_cache:
            raise FileNotFoundError(f"Cache root does not exist: {cache_root}")
        return []
    dirs = []
    for path in sorted(cache_root.iterdir()):
        if not path.is_dir():
            continue
        status_path = path / "status.json"
        feature_path = path / "vjepa21_hidden_states.npz"
        if not status_path.exists() or not feature_path.exists():
            continue
        try:
            status = json.loads(status_path.read_text())
        except json.JSONDecodeError:
            continue
        if status.get("status") == "complete":
            dirs.append(path)
    return dirs


def choose_smoke_dirs(cache_dirs: list[Path]) -> list[Path]:
    with_counts: list[tuple[int, Path, float]] = []
    for d in cache_dirs:
        rows_path = d / "rows.csv"
        try:
            rows = sum(1 for _ in rows_path.open()) - 1
        except Exception:
            rows = 0
        max_black = 0.0
        try:
            with np.load(d / "vjepa21_hidden_states.npz") as bundle:
                if "black_frame_fraction" in bundle:
                    max_black = float(np.nanmax(bundle["black_frame_fraction"]))
                elif "duplicate_frame_fraction" in bundle:
                    max_black = float(np.nanmax(bundle["duplicate_frame_fraction"]))
        except Exception:
            pass
        with_counts.append((rows, d, max_black))
    if not with_counts:
        return []
    by_rows = sorted(with_counts, key=lambda item: item[0])
    chosen = [by_rows[0][1], by_rows[len(by_rows) // 2][1], by_rows[-1][1]]
    quality = sorted(with_counts, key=lambda item: item[2], reverse=True)[0][1]
    chosen.append(quality)
    unique: list[Path] = []
    seen = set()
    for d in chosen:
        if d.name not in seen:
            unique.append(d)
            seen.add(d.name)
    return unique


def load_and_validate_cache(cache_dir: Path, *, require_temporal: bool, limit_rows: int = 0) -> CacheBundle:
    missing_files = [name for name in REQUIRED_CACHE_FILES if not (cache_dir / name).exists()]
    if missing_files:
        raise StageError("load_cache", f"Missing required cache files: {missing_files}")
    rows_csv = read_rows_csv(cache_dir / "rows.csv")
    source_manifest = safe_json_load(cache_dir / "manifest.json")
    source_status = safe_json_load(cache_dir / "status.json")
    preprocessing = safe_json_load(cache_dir / "preprocessing.json")
    if source_status.get("status") != "complete":
        raise StageError("validate_cache", f"Source status is not complete: {source_status.get('status')}")
    row_hz = float(preprocessing.get("row_hz", preprocessing.get("row_rate_hz", EXPECTED_ROW_RATE_HZ)))
    image_size = int(preprocessing.get("image_size", EXPECTED_IMAGE_SIZE))
    decode_hz = float(preprocessing.get("decode_hz", EXPECTED_DECODE_HZ))
    frames_per_clip = int(preprocessing.get("frames_per_clip", EXPECTED_FRAMES_PER_CLIP))
    if abs(row_hz - EXPECTED_ROW_RATE_HZ) > 1e-6:
        raise StageError("validate_cache", f"Cache row_hz {row_hz} does not match expected 2Hz rows")
    if image_size != EXPECTED_IMAGE_SIZE:
        raise StageError("validate_cache", f"Cache image_size {image_size} does not match expected 256px V-JEPA 2.1 run")
    if abs(decode_hz - EXPECTED_DECODE_HZ) > 1e-6:
        raise StageError("validate_cache", f"Cache decode_hz {decode_hz} does not match expected 16Hz sampled clips")
    if frames_per_clip != EXPECTED_FRAMES_PER_CLIP:
        raise StageError("validate_cache", f"Cache frames_per_clip {frames_per_clip} does not match expected 64")
    required_keys = list(REQUIRED_NPZ_KEYS)
    if require_temporal:
        required_keys.extend(REQUIRED_TEMPORAL_KEYS)
    try:
        with np.load(cache_dir / "vjepa21_hidden_states.npz") as bundle:
            missing_keys = [key for key in required_keys if key not in bundle]
            if missing_keys:
                raise StageError("load_cache", f"Missing required NPZ keys: {missing_keys}")
            arrays = {key: np.asarray(bundle[key]) for key in required_keys}
    except StageError:
        raise
    except Exception as exc:
        raise StageError("load_cache", f"Cannot load {cache_dir / 'vjepa21_hidden_states.npz'}: {exc}") from exc

    if limit_rows > 0:
        rows_csv = rows_csv[:limit_rows]
        for key, arr in list(arrays.items()):
            if key in {"selected_state_indices"}:
                continue
            if arr.ndim > 0 and arr.shape[0] >= limit_rows:
                arrays[key] = arr[:limit_rows]

    features = arrays["features"]
    row_count = int(features.shape[0])
    if features.ndim != 4 or tuple(features.shape[1:]) != EXPECTED_FEATURE_SHAPE_TAIL:
        raise StageError("validate_cache", f"Expected features [rows,20,1,1408], got {features.shape}")
    if features.dtype != np.float16:
        # Safe but explicit: downstream storage remains float16 after adapter.
        if not np.issubdtype(features.dtype, np.floating):
            raise StageError("validate_cache", f"features dtype is not floating: {features.dtype}")
    if len(rows_csv) != row_count:
        raise StageError("validate_cache", f"rows.csv count {len(rows_csv)} != features rows {row_count}")
    for key in ("time_seconds", "sample_frame_indices", "sample_time_seconds") + tuple(REQUIRED_NPZ_KEYS[5:]):
        if arrays[key].shape[0] != row_count:
            raise StageError("validate_cache", f"{key} rows {arrays[key].shape[0]} != features rows {row_count}")
    if arrays["sample_frame_indices"].shape != (row_count, 64):
        raise StageError("validate_cache", f"sample_frame_indices expected [rows,64], got {arrays['sample_frame_indices'].shape}")
    if arrays["sample_time_seconds"].shape != (row_count, 64):
        raise StageError("validate_cache", f"sample_time_seconds expected [rows,64], got {arrays['sample_time_seconds'].shape}")
    if arrays["selected_state_indices"].shape != (20,):
        raise StageError("validate_cache", f"selected_state_indices expected [20], got {arrays['selected_state_indices'].shape}")
    if require_temporal:
        if arrays["temporal_mean"].shape != (row_count, 20, 32, 1408):
            raise StageError("validate_cache", f"temporal_mean expected [rows,20,32,1408], got {arrays['temporal_mean'].shape}")
        if arrays["temporal_std"].shape != (row_count, 20, 32, 1408):
            raise StageError("validate_cache", f"temporal_std expected [rows,20,32,1408], got {arrays['temporal_std'].shape}")

    time_seconds = arrays["time_seconds"].astype(np.float32, copy=False)
    if not np.isfinite(time_seconds).all():
        raise StageError("validate_cache", "time_seconds contains NaN or inf")
    if row_count and abs(float(time_seconds[0])) > 1e-3:
        raise StageError("validate_cache", f"First timestamp is not approximately 0.0: {time_seconds[0]}")
    if row_count > 1:
        steps = np.diff(time_seconds)
        if not np.all(steps >= -1e-6):
            raise StageError("validate_cache", "time_seconds is not monotonic increasing")
        median_step = float(np.median(steps))
        if abs(median_step - 0.5) > 0.03:
            raise StageError("validate_cache", f"Median timestamp step is not approximately 0.5: {median_step}")
    if not np.isfinite(features).all():
        raise StageError("validate_cache", "features contains NaN or inf")
    return CacheBundle(cache_dir, rows_csv, source_manifest, source_status, preprocessing, arrays, len(rows_csv))


def create_temporal_diagnostics(bundle: CacheBundle) -> dict[str, np.ndarray]:
    try:
        temporal_mean = bundle.arrays["temporal_mean"]
        temporal_std = bundle.arrays["temporal_std"]
    except KeyError as exc:
        raise StageError("temporal_diagnostics", f"Missing temporal tensor: {exc}") from exc
    if not np.isfinite(temporal_mean).all() or not np.isfinite(temporal_std).all():
        raise StageError("temporal_diagnostics", "temporal_mean or temporal_std contains NaN/inf")
    std32 = temporal_std.astype(np.float32, copy=False)
    mean32 = temporal_mean.astype(np.float32, copy=False)
    diagnostics = {
        "temporal_std_global": std32.mean(axis=(1, 2, 3)).astype(np.float32, copy=False),
        "temporal_std_by_state": std32.mean(axis=(2, 3)).astype(np.float16, copy=False),
        "temporal_std_by_state_token": std32.mean(axis=3).astype(np.float16, copy=False),
        "temporal_mean_by_state_feature": mean32.mean(axis=2).astype(np.float16, copy=False),
        "temporal_std_by_state_feature": std32.mean(axis=2).astype(np.float16, copy=False),
    }
    return diagnostics


def create_quality_flags(bundle: CacheBundle) -> dict[str, np.ndarray]:
    arrays = bundle.arrays
    black = arrays["black_frame_fraction"].astype(np.float32, copy=False)
    duplicate = arrays["duplicate_frame_fraction"].astype(np.float32, copy=False)
    # Conservative flags only. Do not discard quiet/static gameplay automatically;
    # preserve all rows and let downstream experiments choose their filters.
    black_flag = black >= 0.5
    duplicate_flag = duplicate >= 0.95
    exclusion_flag = black_flag | duplicate_flag
    quality_weight = 1.0 - np.maximum(black, duplicate)
    quality_weight = np.clip(quality_weight, 0.0, 1.0).astype(np.float32, copy=False)
    return {
        "quality_black_frame_flag": black_flag.astype(np.uint8, copy=False),
        "quality_duplicate_frame_flag": duplicate_flag.astype(np.uint8, copy=False),
        "quality_exclusion_flag": exclusion_flag.astype(np.uint8, copy=False),
        "quality_weight_suggested": quality_weight,
    }


def create_rows_aligned(bundle: CacheBundle) -> list[dict[str, Any]]:
    arrays = bundle.arrays
    time_seconds = arrays["time_seconds"].astype(np.float32, copy=False)
    sample_time_seconds = arrays["sample_time_seconds"].astype(np.float32, copy=False)
    quality_flags = create_quality_flags(bundle)
    out = []
    video_id = bundle.cache_dir.name
    for idx, t in enumerate(time_seconds):
        source_row = bundle.rows_csv[idx] if idx < len(bundle.rows_csv) else {}
        clip_start = float(np.nanmin(sample_time_seconds[idx])) if sample_time_seconds.shape[0] else max(0.0, float(t) - MEDIAN_SAMPLE_WINDOW_SECONDS)
        clip_end = float(np.nanmax(sample_time_seconds[idx])) if sample_time_seconds.shape[0] else float(t)
        out.append({
            "video_id": video_id,
            "row_index": idx,
            "time_seconds": f"{float(t):.6f}",
            "source_row_index": source_row.get("row_index", idx),
            "duration_seconds": f"{float(time_seconds[-1]) if len(time_seconds) else 0.0:.6f}",
            "row_rate_hz": f"{EXPECTED_ROW_RATE_HZ:.1f}",
            "temporal_semantics": "causal_trailing_clip",
            "clip_window_start_seconds": f"{clip_start:.6f}",
            "clip_window_end_seconds": f"{min(clip_end, float(t)):.6f}",
            "is_start_clamped": str(bool(clip_start <= 1e-6 and float(t) < MEDIAN_SAMPLE_WINDOW_SECONDS)).lower(),
            "black_frame_fraction": f"{float(arrays['black_frame_fraction'][idx]):.9f}",
            "duplicate_frame_fraction": f"{float(arrays['duplicate_frame_fraction'][idx]):.9f}",
            "quality_black_frame_flag": int(quality_flags["quality_black_frame_flag"][idx]),
            "quality_duplicate_frame_flag": int(quality_flags["quality_duplicate_frame_flag"][idx]),
            "quality_exclusion_flag": int(quality_flags["quality_exclusion_flag"][idx]),
            "quality_weight_suggested": f"{float(quality_flags['quality_weight_suggested'][idx]):.9f}",
            "motion_absdiff_mean": f"{float(arrays['motion_absdiff_mean'][idx]):.9f}",
            "luma_mean": f"{float(arrays['luma_mean'][idx]):.9f}",
            "luma_std": f"{float(arrays['luma_std'][idx]):.9f}",
            "frame_luma_std_mean": f"{float(arrays['frame_luma_std_mean'][idx]):.9f}",
        })
    return out


def write_rows_aligned(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "video_id",
        "row_index",
        "time_seconds",
        "source_row_index",
        "duration_seconds",
        "row_rate_hz",
        "temporal_semantics",
        "clip_window_start_seconds",
        "clip_window_end_seconds",
        "is_start_clamped",
        "black_frame_fraction",
        "duplicate_frame_fraction",
        "quality_black_frame_flag",
        "quality_duplicate_frame_flag",
        "quality_exclusion_flag",
        "quality_weight_suggested",
        "motion_absdiff_mean",
        "luma_mean",
        "luma_std",
        "frame_luma_std_mean",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def source_cache_fingerprint(bundle: CacheBundle) -> str | None:
    for source in (bundle.source_manifest, bundle.preprocessing, bundle.source_status):
        for key in (
            "cache_fingerprint",
            "fingerprint",
            "cache_sha256",
            "vjepa21_hidden_states_sha256",
            "feature_sha256",
        ):
            value = source.get(key)
            if value:
                return str(value)
    return None


def output_complete(out_dir: Path) -> bool:
    if not all((out_dir / name).exists() for name in SUCCESS_OUTPUT_FILES):
        return False
    try:
        status = json.loads((out_dir / "status.json").read_text())
        if status.get("status") != "success":
            return False
        with np.load(out_dir / "tribe_v2_cortical_predictions.npz") as bundle:
            cortical = bundle["cortical_prediction"]
            time_seconds = bundle["time_seconds"]
            grouped = bundle["tribe_grouped_video_feature"]
        with np.load(out_dir / "baseline_features_rowlevel.npz") as baseline:
            baseline_grouped = baseline["tribe_grouped_video_feature"]
            baseline_time = baseline["time_seconds"]
        with np.load(out_dir / "vjepa_temporal_diagnostics.npz") as temporal:
            temporal_rows = temporal["temporal_std_global"].shape[0]
        if cortical.ndim != 2 or cortical.shape[1] != EXPECTED_CORTICAL_WIDTH:
            return False
        if time_seconds.shape[0] != cortical.shape[0]:
            return False
        if grouped.shape != (cortical.shape[0], 2, 1408):
            return False
        if baseline_grouped.shape != grouped.shape or baseline_time.shape[0] != cortical.shape[0]:
            return False
        if temporal_rows != cortical.shape[0]:
            return False
    except Exception:
        return False
    return True


def write_failure(
    out_dir: Path,
    failed_path: Path,
    *,
    video_id: str,
    cache_path: Path,
    stage: str,
    exc: BaseException,
    started_at: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_path = out_dir / "traceback.txt"
    tb_path.write_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    status = {
        "video_id": video_id,
        "status": "failed",
        "started_at": started_at,
        "finished_at": utc_now(),
        "runtime_seconds": None,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "stage": stage,
        "safe_to_retry": True,
        "traceback_path": str(tb_path),
    }
    atomic_write_text(out_dir / "status.json", json.dumps(status, indent=2, sort_keys=True) + "\n")
    failure_row = {
        "video_id": video_id,
        "cache_path": str(cache_path),
        "error_stage": stage,
        "error_message": str(exc),
        "safe_to_retry": True,
    }
    append_jsonl(failed_path, failure_row)
    return status


def process_video(
    cache_dir: Path,
    per_video_root: Path,
    manifest_jsonl: Path,
    failed_jsonl: Path,
    model: torch.nn.Module | None,
    device: torch.device,
    args: argparse.Namespace,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    started_at = utc_now()
    video_id = cache_dir.name
    out_dir = per_video_root / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_complete(out_dir) and not args.force:
        result = {
            "video_id": video_id,
            "status": "skipped",
            "rows": json.loads((out_dir / "manifest.json").read_text()).get("row_count"),
            "output_path": str(out_dir / "tribe_v2_cortical_predictions.npz"),
            "runtime_seconds": 0.0,
        }
        append_jsonl(manifest_jsonl, result)
        return result
    if args.dry_run and out_dir.exists() and (out_dir / "status.json").exists() and not args.force:
        # Dry run should validate source cache even if an old failed output exists.
        pass

    try:
        bundle = load_and_validate_cache(cache_dir, require_temporal=args.require_temporal_diagnostics, limit_rows=args.limit_rows)
        features = bundle.arrays["features"]
        time_seconds = bundle.arrays["time_seconds"].astype(np.float32, copy=False)
        grouped = group_mean_tribe_video_layers(features)
        temporal_diags = create_temporal_diagnostics(bundle) if args.save_compact_temporal_diagnostics else {}
        quality_flags = create_quality_flags(bundle)
        rows_aligned = create_rows_aligned(bundle)
        if len(rows_aligned) != grouped.shape[0]:
            raise StageError("validate_cache", "rows_aligned row count does not match grouped features")
        if args.dry_run:
            result = {
                "video_id": video_id,
                "status": "dry_run_validated",
                "rows": int(len(time_seconds)),
                "source_cache": str(cache_dir),
                "adapter_output_shape": list(grouped.shape),
                "temporal_diagnostics_shapes": {key: list(value.shape) for key, value in temporal_diags.items()},
                "runtime_seconds": time.time() - started,
            }
            append_jsonl(manifest_jsonl, result)
            return result

        if model is None:
            raise StageError("tribe_forward", "TRIBE model was not loaded")
        video = torch.from_numpy(grouped.transpose(1, 2, 0)[None]).to(device=device)
        video = video.to(dtype=torch.float16 if device.type == "cuda" else torch.float32)
        if list(video.shape) != [1, 2, 1408, len(time_seconds)]:
            raise StageError("adapter", f"Unexpected TRIBE input shape {list(video.shape)}")
        batch = SimpleNamespace(
            data={
                "video": video,
                "subject_id": torch.zeros((1,), dtype=torch.long, device=device),
            }
        )
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            cortical = model(batch, pool_outputs=False)
        cortical_rows = cortical[0].transpose(0, 1).detach().cpu().numpy().astype(np.float16, copy=False)
        if cortical_rows.shape != (len(time_seconds), EXPECTED_CORTICAL_WIDTH):
            raise StageError("tribe_forward", f"Unexpected cortical row shape {cortical_rows.shape}")
        if not np.isfinite(cortical_rows).all():
            raise StageError("tribe_forward", "TRIBE output contains NaN or inf")

        temporal_path = out_dir / "vjepa_temporal_diagnostics.npz"
        atomic_savez(temporal_path, **temporal_diags)
        output_path = out_dir / "tribe_v2_cortical_predictions.npz"
        atomic_savez(
            output_path,
            cortical_prediction=cortical_rows,
            time_seconds=time_seconds.astype(np.float32, copy=False),
            tribe_grouped_video_feature=grouped.astype(np.float16, copy=False),
            temporal_std_global=temporal_diags["temporal_std_global"].astype(np.float32, copy=False),
            temporal_std_by_state=temporal_diags["temporal_std_by_state"].astype(np.float16, copy=False),
            temporal_std_by_state_token=temporal_diags["temporal_std_by_state_token"].astype(np.float16, copy=False),
            luma_mean=bundle.arrays["luma_mean"].astype(np.float32, copy=False),
            luma_std=bundle.arrays["luma_std"].astype(np.float32, copy=False),
            frame_luma_std_mean=bundle.arrays["frame_luma_std_mean"].astype(np.float32, copy=False),
            motion_absdiff_mean=bundle.arrays["motion_absdiff_mean"].astype(np.float32, copy=False),
            black_frame_fraction=bundle.arrays["black_frame_fraction"].astype(np.float32, copy=False),
            duplicate_frame_fraction=bundle.arrays["duplicate_frame_fraction"].astype(np.float32, copy=False),
            quality_black_frame_flag=quality_flags["quality_black_frame_flag"],
            quality_duplicate_frame_flag=quality_flags["quality_duplicate_frame_flag"],
            quality_exclusion_flag=quality_flags["quality_exclusion_flag"],
            quality_weight_suggested=quality_flags["quality_weight_suggested"],
            selected_state_indices=bundle.arrays["selected_state_indices"].astype(np.int16, copy=False),
            tribe_group_layers=np.asarray(TRIBE_GROUP_LAYERS, dtype=np.float32),
            sample_frame_indices=bundle.arrays["sample_frame_indices"].astype(np.int32, copy=False),
            sample_time_seconds=bundle.arrays["sample_time_seconds"].astype(np.float32, copy=False),
        )
        baseline_path = out_dir / "baseline_features_rowlevel.npz"
        atomic_savez(
            baseline_path,
            tribe_grouped_video_feature=grouped.astype(np.float16, copy=False),
            time_seconds=time_seconds.astype(np.float32, copy=False),
            motion_absdiff_mean=bundle.arrays["motion_absdiff_mean"].astype(np.float32, copy=False),
            luma_mean=bundle.arrays["luma_mean"].astype(np.float32, copy=False),
            luma_std=bundle.arrays["luma_std"].astype(np.float32, copy=False),
            frame_luma_std_mean=bundle.arrays["frame_luma_std_mean"].astype(np.float32, copy=False),
            black_frame_fraction=bundle.arrays["black_frame_fraction"].astype(np.float32, copy=False),
            duplicate_frame_fraction=bundle.arrays["duplicate_frame_fraction"].astype(np.float32, copy=False),
            quality_black_frame_flag=quality_flags["quality_black_frame_flag"],
            quality_duplicate_frame_flag=quality_flags["quality_duplicate_frame_flag"],
            quality_exclusion_flag=quality_flags["quality_exclusion_flag"],
            quality_weight_suggested=quality_flags["quality_weight_suggested"],
            temporal_std_global=temporal_diags["temporal_std_global"].astype(np.float32, copy=False),
            temporal_std_by_state=temporal_diags["temporal_std_by_state"].astype(np.float16, copy=False),
            temporal_std_by_state_token=temporal_diags["temporal_std_by_state_token"].astype(np.float16, copy=False),
            sample_frame_indices=bundle.arrays["sample_frame_indices"].astype(np.int32, copy=False),
            sample_time_seconds=bundle.arrays["sample_time_seconds"].astype(np.float32, copy=False),
        )
        rows_path = out_dir / "rows_aligned.csv"
        write_rows_aligned(rows_path, rows_aligned)

        diagnostics = {}
        diagnostics.update(array_stats(features, "input"))
        diagnostics.update(array_stats(grouped, "adapter"))
        diagnostics.update(array_stats(cortical_rows, "output"))
        diagnostics.update({
            "temporal_std_global_nan_count": int(np.isnan(temporal_diags["temporal_std_global"]).sum()),
            "temporal_std_global_inf_count": int(np.isinf(temporal_diags["temporal_std_global"]).sum()),
            "temporal_std_global_mean": float(temporal_diags["temporal_std_global"].astype(np.float32).mean(dtype=np.float64)),
            "temporal_std_global_std": float(temporal_diags["temporal_std_global"].astype(np.float32).std(dtype=np.float64)),
            "temporal_std_by_state_shape": list(temporal_diags["temporal_std_by_state"].shape),
            "temporal_std_by_state_token_shape": list(temporal_diags["temporal_std_by_state_token"].shape),
            "temporal_mean_by_state_feature_shape": list(temporal_diags["temporal_mean_by_state_feature"].shape),
            "temporal_std_by_state_feature_shape": list(temporal_diags["temporal_std_by_state_feature"].shape),
            "black_frame_fraction_mean": float(bundle.arrays["black_frame_fraction"].mean(dtype=np.float64)),
            "black_frame_fraction_max": float(bundle.arrays["black_frame_fraction"].max()),
            "duplicate_frame_fraction_mean": float(bundle.arrays["duplicate_frame_fraction"].mean(dtype=np.float64)),
            "duplicate_frame_fraction_max": float(bundle.arrays["duplicate_frame_fraction"].max()),
            "quality_black_frame_flag_count": int(quality_flags["quality_black_frame_flag"].sum()),
            "quality_duplicate_frame_flag_count": int(quality_flags["quality_duplicate_frame_flag"].sum()),
            "quality_exclusion_flag_count": int(quality_flags["quality_exclusion_flag"].sum()),
            "quality_exclusion_fraction": float(quality_flags["quality_exclusion_flag"].mean(dtype=np.float64)),
            "quality_weight_suggested_mean": float(quality_flags["quality_weight_suggested"].mean(dtype=np.float64)),
            "quality_weight_suggested_min": float(quality_flags["quality_weight_suggested"].min()),
            "motion_absdiff_mean_mean": float(bundle.arrays["motion_absdiff_mean"].mean(dtype=np.float64)),
            "motion_absdiff_mean_max": float(bundle.arrays["motion_absdiff_mean"].max()),
            "luma_mean_mean": float(bundle.arrays["luma_mean"].mean(dtype=np.float64)),
            "luma_mean_min": float(bundle.arrays["luma_mean"].min()),
            "luma_mean_max": float(bundle.arrays["luma_mean"].max()),
            "first_timestamp": float(time_seconds[0]) if len(time_seconds) else None,
            "last_timestamp": float(time_seconds[-1]) if len(time_seconds) else None,
            "timestamp_step_median": float(np.median(np.diff(time_seconds))) if len(time_seconds) > 1 else None,
            "timestamp_step_min": float(np.min(np.diff(time_seconds))) if len(time_seconds) > 1 else None,
            "timestamp_step_max": float(np.max(np.diff(time_seconds))) if len(time_seconds) > 1 else None,
            "time_monotonic": bool(np.all(np.diff(time_seconds) >= -1e-6)) if len(time_seconds) > 1 else True,
        })
        atomic_write_text(out_dir / "diagnostics.json", json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")

        source_npz = cache_dir / "vjepa21_hidden_states.npz"
        input_mapping = {
            "video_id": video_id,
            "source_cache_file": str(source_npz),
            "source_key": "features",
            "source_shape": list(features.shape),
            "source_dtype": str(features.dtype),
            "adapter_function": "group_mean_tribe_video_layers",
            "adapter_steps": [
                "load vjepa21_hidden_states.npz key features",
                "squeeze axis 2: [rows,20,1,1408] -> [rows,20,1408]",
                "group selected states using TRIBE group layers [0.5,0.75,1.0]",
                "produce tribe_grouped_video_feature [rows,2,1408]",
                "transpose to TRIBE input [1,2,1408,rows]",
            ],
            "tribe_group_layers": list(TRIBE_GROUP_LAYERS),
            "selected_state_indices": bundle.arrays["selected_state_indices"].astype(int).tolist(),
            "adapter_output_shape": list(grouped.shape),
            "tribe_input_shape": list(video.shape),
            "tribe_expected_layout": "[B,L,D,T]",
            "row_rate_hz": EXPECTED_ROW_RATE_HZ,
            "vjepa_image_size": EXPECTED_IMAGE_SIZE,
            "decode_hz": EXPECTED_DECODE_HZ,
            "frames_per_clip": EXPECTED_FRAMES_PER_CLIP,
            "clip_window_seconds": EXPECTED_CLIP_SECONDS,
            "temporal_semantics": "causal_trailing_clip",
            "median_sample_window": "approximately t - 3.9375s through t",
            "first_rows_clamped_at_video_start": True,
            "cache_schema": bundle.preprocessing.get("cache_schema"),
            "vjepa_model_name": bundle.preprocessing.get("model_name"),
            "vjepa_model_hash": bundle.preprocessing.get("model_sha256"),
            "vjepa_run_contract": "V-JEPA 2.1 ViT-g, 256px image_size, 2Hz output rows, 16Hz decode, 64-frame causal trailing clips",
            "tribe_model_name": "official facebookresearch/tribev2 FmriEncoderModel",
            "tribe_checkpoint_hash": run_metadata.get("tribe_checkpoint_hash"),
            "repo_commit_hash": run_metadata.get("repo_commit_hash"),
            "script_path": str(Path(__file__)),
            "script_commit_hash": run_metadata.get("script_sha256"),
            "temporal_diagnostics_created": True,
            "temporal_diagnostic_reductions": {
                "temporal_std_global": "temporal_std mean over states, tokens, and features",
                "temporal_std_by_state": "temporal_std mean over tokens and features",
                "temporal_std_by_state_token": "temporal_std mean over features",
                "temporal_mean_by_state_feature": "temporal_mean mean over token dimension",
                "temporal_std_by_state_feature": "temporal_std mean over token dimension",
            },
        }
        atomic_write_text(out_dir / "input_mapping.json", json.dumps(input_mapping, indent=2, sort_keys=True) + "\n")

        finished_at = utc_now()
        runtime_seconds = time.time() - started
        output_hash = sha256_file(output_path)
        manifest = {
            "video_id": video_id,
            "source_video_filename": bundle.preprocessing.get("video_name") or bundle.source_manifest.get("video_name"),
            "source_video_sha256": bundle.preprocessing.get("video_sha256") or bundle.source_manifest.get("video_sha256"),
            "source_cache_path": str(cache_dir),
            "source_cache_sha256": source_cache_fingerprint(bundle),
            "vjepa_model_name": bundle.preprocessing.get("model_name"),
            "vjepa_model_hash": bundle.preprocessing.get("model_sha256"),
            "vjepa_schema": bundle.preprocessing.get("cache_schema"),
            "vjepa_run_contract": "V-JEPA 2.1 ViT-g, 256px image_size, 2Hz output rows, 16Hz decode, 64-frame causal trailing clips",
            "vjepa_image_size": EXPECTED_IMAGE_SIZE,
            "decode_hz": EXPECTED_DECODE_HZ,
            "frames_per_clip": EXPECTED_FRAMES_PER_CLIP,
            "row_rate_hz": EXPECTED_ROW_RATE_HZ,
            "tribe_model_name": "official facebookresearch/tribev2 FmriEncoderModel",
            "tribe_checkpoint_hash": run_metadata.get("tribe_checkpoint_hash"),
            "repo_commit_hash": run_metadata.get("repo_commit_hash"),
            "script_path": str(Path(__file__)),
            "script_commit_hash": run_metadata.get("script_sha256"),
            "created_at": finished_at,
            "runtime_seconds": runtime_seconds,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" and torch.cuda.is_available() else None,
            "row_count": int(len(time_seconds)),
            "duration_seconds": float(time_seconds[-1]) if len(time_seconds) else 0.0,
            "input_shape": list(features.shape),
            "adapter_output_shape": list(grouped.shape),
            "tribe_input_shape": list(video.shape),
            "cortical_output_shape": list(cortical_rows.shape),
            "temporal_diagnostics_shapes": {key: list(value.shape) for key, value in temporal_diags.items()},
            "quality_exclusion_flag_count": int(quality_flags["quality_exclusion_flag"].sum()),
            "quality_exclusion_fraction": float(quality_flags["quality_exclusion_flag"].mean(dtype=np.float64)),
            "dtype": "float16",
            "output_hash": output_hash,
            "status": "success",
            "rows_csv_count": bundle.rows_csv_count,
            "features_row_count": int(features.shape[0]),
            "time_seconds_count": int(time_seconds.shape[0]),
            "output_row_count": int(cortical_rows.shape[0]),
            "temporal_diagnostics_row_count": int(temporal_diags["temporal_std_global"].shape[0]),
        }
        atomic_write_text(out_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        status = {
            "video_id": video_id,
            "status": "success",
            "started_at": started_at,
            "finished_at": finished_at,
            "runtime_seconds": runtime_seconds,
            "error": None,
            "outputs_written": True,
            "nan_inf_checks_passed": True,
            "row_alignment_passed": True,
            "temporal_diagnostics_written": True,
        }
        atomic_write_text(out_dir / "status.json", json.dumps(status, indent=2, sort_keys=True) + "\n")

        result = {
            "video_id": video_id,
            "status": "success",
            "rows": int(len(time_seconds)),
            "duration_seconds": float(time_seconds[-1]) if len(time_seconds) else 0.0,
            "output_path": str(output_path),
            "baseline_features_path": str(baseline_path),
            "cortical_shape": list(cortical_rows.shape),
            "temporal_diagnostics_path": str(temporal_path),
            "temporal_diagnostics_shapes": {key: list(value.shape) for key, value in temporal_diags.items()},
            "time_start": float(time_seconds[0]) if len(time_seconds) else None,
            "time_end": float(time_seconds[-1]) if len(time_seconds) else None,
            "runtime_seconds": runtime_seconds,
            "source_cache_hash": source_cache_fingerprint(bundle),
            "output_hash": output_hash,
            "error": None,
        }
        append_jsonl(manifest_jsonl, result)
        return result
    except Exception as exc:
        stage = exc.stage if isinstance(exc, StageError) else "unknown"
        status = write_failure(out_dir, failed_jsonl, video_id=video_id, cache_path=cache_dir, stage=stage, exc=exc, started_at=started_at)
        result = {
            "video_id": video_id,
            "status": "failed",
            "rows": None,
            "duration_seconds": None,
            "output_path": None,
            "cortical_shape": None,
            "temporal_diagnostics_path": None,
            "temporal_diagnostics_shapes": None,
            "time_start": None,
            "time_end": None,
            "runtime_seconds": time.time() - started,
            "source_cache_hash": None,
            "output_hash": None,
            "error": status["error_message"],
        }
        append_jsonl(manifest_jsonl, result)
        return result


def write_global_schema(output_root: Path) -> None:
    schema = {
        "per_video_required_files": list(SUCCESS_OUTPUT_FILES),
        "tribe_v2_cortical_predictions.npz": {
            "cortical_prediction": "[rows,20484] float16, unpooled 2Hz TRIBE cortical predictions",
            "time_seconds": "[rows] float32",
            "tribe_grouped_video_feature": "[rows,2,1408] float16 adapted representation fed to TRIBE",
            "temporal_std_global": "[rows] float32 compact temporal instability",
            "temporal_std_by_state": "[rows,20] float16 compact temporal instability by selected state",
            "temporal_std_by_state_token": "[rows,20,32] float16 compact temporal instability by state/token",
            "quality_signals": ["luma_mean", "luma_std", "frame_luma_std_mean", "motion_absdiff_mean", "black_frame_fraction", "duplicate_frame_fraction"],
            "quality_flags": ["quality_black_frame_flag", "quality_duplicate_frame_flag", "quality_exclusion_flag", "quality_weight_suggested"],
            "sample_frame_indices": "[rows,64] int32",
            "sample_time_seconds": "[rows,64] float32",
        },
        "baseline_features_rowlevel.npz": {
            "tribe_grouped_video_feature": "[rows,2,1408] float16",
            "time_seconds": "[rows] float32",
            "quality_signals": ["motion_absdiff_mean", "luma_mean", "luma_std", "frame_luma_std_mean", "black_frame_fraction", "duplicate_frame_fraction"],
            "quality_flags": ["quality_black_frame_flag", "quality_duplicate_frame_flag", "quality_exclusion_flag", "quality_weight_suggested"],
            "temporal_std_global": "[rows] float32",
            "temporal_std_by_state": "[rows,20] float16",
            "temporal_std_by_state_token": "[rows,20,32] float16",
            "sample_frame_indices": "[rows,64] int32",
            "sample_time_seconds": "[rows,64] float32",
        },
        "vjepa_temporal_diagnostics.npz": {
            "temporal_std_global": "[rows] mean over states/tokens/features",
            "temporal_std_by_state": "[rows,20] mean over tokens/features",
            "temporal_std_by_state_token": "[rows,20,32] mean over features",
            "temporal_mean_by_state_feature": "[rows,20,1408] mean over token dimension",
            "temporal_std_by_state_feature": "[rows,20,1408] mean over token dimension",
        },
        "global_files": {
            "video_metadata.csv": "one row per video with status, row counts, quality summaries, source/cache/output hashes",
            "row_index.parquet": "one row per 2Hz output row when pyarrow is available",
            "row_index.csv": "CSV fallback/mirror for row_index.parquet",
            "splits_by_video.json": "deterministic train/val/test video split manifest; no model training",
            "splits_quality_filtered.json": "deterministic split after simple quality warning filter; no model training",
            "splits_duration_balanced.json": "duration-balanced deterministic split; no model training",
            "BASELINE_READINESS.md": "maps saved arrays to later baseline/control families",
        },
        "non_goals": ["PCA", "bridge training", "benchmarking", "delta/spike analysis", "V-JEPA extraction", "raw video decode"],
    }
    atomic_write_text(output_root / "output_schema.json", json.dumps(schema, indent=2, sort_keys=True) + "\n")
    readme = """# TRIBE v2 Post-pass Output Schema

This bundle is a cache-only TRIBE v2 working-output dataset. It was produced
from existing V-JEPA 2.1 cache folders without decoding raw video or running
V-JEPA.

No PCA, bridge training, benchmarking, delta prediction, spike detection, or
exploratory modelling is performed by this post-pass.

Per-video outputs live under `per_video/<video_id>/`.

`tribe_v2_cortical_predictions.npz` contains:
- `cortical_prediction`: `[rows, 20484]` float16 unpooled 2Hz cortical predictions.
- `time_seconds`: `[rows]` float32 row timestamps.
- `tribe_grouped_video_feature`: `[rows, 2, 1408]` float16 adapted features fed to TRIBE.
- compact temporal std diagnostics, row-level quality signals, and sample frame/time arrays.
- quality flags: `quality_black_frame_flag`, `quality_duplicate_frame_flag`,
  `quality_exclusion_flag`, and `quality_weight_suggested`. These preserve
  black/static-row handling for later benchmarking without dropping rows here.

`baseline_features_rowlevel.npz` contains the compact non-cortical row-level
features needed for later local controls and nuisance baselines without loading
the full upstream V-JEPA cache.

`vjepa_temporal_diagnostics.npz` contains compact reductions of the large V-JEPA
temporal tensors. It intentionally does not copy full `features`,
`all_layer_features`, `temporal_mean`, or `temporal_std` from the upstream cache.

Global files include `video_metadata.csv`, `row_index.csv`, optional
`row_index.parquet` when pyarrow is installed, split manifests, and
`BASELINE_READINESS.md`.
"""
    atomic_write_text(output_root / "README_OUTPUT_SCHEMA.md", readme)


def summarize_outputs(output_root: Path) -> dict[str, Any]:
    per_video = output_root / "per_video"
    completed = []
    failed = []
    total_rows = 0
    total_cortical_size = 0
    total_grouped_size = 0
    total_temporal_size = 0
    runtimes = []
    warnings = []
    for status_path in per_video.glob("*/status.json"):
        data = json.loads(status_path.read_text())
        video_dir = status_path.parent
        if data.get("status") == "success":
            completed.append(video_dir.name)
            manifest = json.loads((video_dir / "manifest.json").read_text())
            total_rows += int(manifest.get("row_count") or 0)
            runtimes.append(float(manifest.get("runtime_seconds") or 0))
            pred = video_dir / "tribe_v2_cortical_predictions.npz"
            temporal = video_dir / "vjepa_temporal_diagnostics.npz"
            if pred.exists():
                total_cortical_size += pred.stat().st_size
                # grouped feature is inside the same archive; exact internal bytes are rows*2*1408*2.
                total_grouped_size += int(manifest.get("row_count") or 0) * 2 * 1408 * 2
            if temporal.exists():
                total_temporal_size += temporal.stat().st_size
            diagnostics_path = video_dir / "diagnostics.json"
            if diagnostics_path.exists():
                diag = json.loads(diagnostics_path.read_text())
                if float(diag.get("black_frame_fraction_max") or 0) > 0.5 or float(diag.get("duplicate_frame_fraction_max") or 0) > 0.95:
                    warnings.append(video_dir.name)
        elif data.get("status") == "failed":
            failed.append(video_dir.name)
    total_size = sum(path.stat().st_size for path in output_root.rglob("*") if path.is_file())
    summary = {
        "total_videos_expected": 995,
        "total_videos_completed": len(completed),
        "total_videos_failed": len(failed),
        "total_rows_completed": total_rows,
        "total_cortical_output_size": total_cortical_size,
        "total_grouped_feature_size_estimated_uncompressed": total_grouped_size,
        "total_temporal_diagnostics_size": total_temporal_size,
        "total_output_folder_size": total_size,
        "average_runtime_per_video": sum(runtimes) / len(runtimes) if runtimes else None,
        "average_runtime_per_row": sum(runtimes) / total_rows if total_rows else None,
        "failed_video_list": failed,
        "warning_video_list": warnings,
        "output_completeness_checklist": {
            "per_video_required_files": list(SUCCESS_OUTPUT_FILES),
            "global_required_files": [
                "global_run_metadata.json",
                "tribe_v2_postpass_manifest.jsonl",
                "failed_videos.jsonl",
                "summary_report.json",
                "output_schema.json",
                "README_OUTPUT_SCHEMA.md",
                "video_metadata.csv",
                "row_index.csv",
                "row_index.parquet",
                "splits_by_video.json",
                "splits_quality_filtered.json",
                "splits_duration_balanced.json",
                "BASELINE_READINESS.md",
            ],
        },
    }
    atomic_write_text(output_root / "summary_report.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def deterministic_three_way_split(video_ids: list[str]) -> dict[str, list[str]]:
    ordered = sorted(video_ids)
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for idx, video_id in enumerate(ordered):
        bucket = idx % 20
        if bucket < 14:
            train.append(video_id)
        elif bucket < 17:
            val.append(video_id)
        else:
            test.append(video_id)
    return {"train": train, "val": val, "test": test}


def duration_balanced_split(video_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    ordered = sorted(video_rows, key=lambda row: float(row.get("duration_seconds") or 0), reverse=True)
    buckets = {"train": [], "val": [], "test": []}
    totals = {"train": 0.0, "val": 0.0, "test": 0.0}
    target_order = ["train", "train", "train", "train", "train", "train", "train", "val", "val", "test"]
    for idx, row in enumerate(ordered):
        target = target_order[idx % len(target_order)]
        video_id = str(row["video_id"])
        buckets[target].append(video_id)
        totals[target] += float(row.get("duration_seconds") or 0)
    return {**buckets, "duration_seconds_by_split": totals}


def write_baseline_readiness_outputs(output_root: Path) -> None:
    per_video = output_root / "per_video"
    video_rows: list[dict[str, Any]] = []
    row_index_rows: list[dict[str, Any]] = []
    for manifest_path in sorted(per_video.glob("*/manifest.json")):
        video_dir = manifest_path.parent
        status_path = video_dir / "status.json"
        diagnostics_path = video_dir / "diagnostics.json"
        rows_path = video_dir / "rows_aligned.csv"
        try:
            manifest = json.loads(manifest_path.read_text())
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else {}
        except Exception:
            continue

        video_id = video_dir.name
        video_rows.append({
            "video_id": video_id,
            "source_filename": manifest.get("source_video_filename"),
            "duration_seconds": manifest.get("duration_seconds"),
            "row_count": manifest.get("row_count"),
            "black_frame_fraction_mean": diagnostics.get("black_frame_fraction_mean"),
            "black_frame_fraction_max": diagnostics.get("black_frame_fraction_max"),
            "duplicate_frame_fraction_mean": diagnostics.get("duplicate_frame_fraction_mean"),
            "duplicate_frame_fraction_max": diagnostics.get("duplicate_frame_fraction_max"),
            "quality_black_frame_flag_count": diagnostics.get("quality_black_frame_flag_count"),
            "quality_duplicate_frame_flag_count": diagnostics.get("quality_duplicate_frame_flag_count"),
            "quality_exclusion_flag_count": diagnostics.get("quality_exclusion_flag_count"),
            "quality_exclusion_fraction": diagnostics.get("quality_exclusion_fraction"),
            "quality_weight_suggested_mean": diagnostics.get("quality_weight_suggested_mean"),
            "quality_weight_suggested_min": diagnostics.get("quality_weight_suggested_min"),
            "motion_absdiff_mean_mean": diagnostics.get("motion_absdiff_mean_mean"),
            "motion_absdiff_mean_max": diagnostics.get("motion_absdiff_mean_max"),
            "luma_mean_mean": diagnostics.get("luma_mean_mean"),
            "luma_mean_min": diagnostics.get("luma_mean_min"),
            "luma_mean_max": diagnostics.get("luma_mean_max"),
            "source_video_sha256": manifest.get("source_video_sha256"),
            "source_cache_hash": manifest.get("source_cache_sha256"),
            "output_hash": manifest.get("output_hash"),
            "status": status.get("status", manifest.get("status")),
            "cortical_output_path": str(video_dir / "tribe_v2_cortical_predictions.npz"),
            "baseline_features_path": str(video_dir / "baseline_features_rowlevel.npz"),
            "temporal_diagnostics_path": str(video_dir / "vjepa_temporal_diagnostics.npz"),
        })

        if rows_path.exists():
            with rows_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    row_index_rows.append({
                        "video_id": video_id,
                        "row_index": int(row["row_index"]),
                        "time_seconds": float(row["time_seconds"]),
                        "clip_window_start_seconds": float(row["clip_window_start_seconds"]),
                        "clip_window_end_seconds": float(row["clip_window_end_seconds"]),
                        "temporal_semantics": row["temporal_semantics"],
                        "black_frame_fraction": float(row["black_frame_fraction"]),
                        "duplicate_frame_fraction": float(row["duplicate_frame_fraction"]),
                        "quality_black_frame_flag": int(row["quality_black_frame_flag"]),
                        "quality_duplicate_frame_flag": int(row["quality_duplicate_frame_flag"]),
                        "quality_exclusion_flag": int(row["quality_exclusion_flag"]),
                        "quality_weight_suggested": float(row["quality_weight_suggested"]),
                        "motion_absdiff_mean": float(row["motion_absdiff_mean"]),
                        "luma_mean": float(row["luma_mean"]),
                        "luma_std": float(row["luma_std"]),
                        "frame_luma_std_mean": float(row["frame_luma_std_mean"]),
                        "output_file_path": str(video_dir / "tribe_v2_cortical_predictions.npz"),
                    })

    video_fields = [
        "video_id",
        "source_filename",
        "duration_seconds",
        "row_count",
        "black_frame_fraction_mean",
        "black_frame_fraction_max",
        "duplicate_frame_fraction_mean",
        "duplicate_frame_fraction_max",
        "quality_black_frame_flag_count",
        "quality_duplicate_frame_flag_count",
        "quality_exclusion_flag_count",
        "quality_exclusion_fraction",
        "quality_weight_suggested_mean",
        "quality_weight_suggested_min",
        "motion_absdiff_mean_mean",
        "motion_absdiff_mean_max",
        "luma_mean_mean",
        "luma_mean_min",
        "luma_mean_max",
        "source_video_sha256",
        "source_cache_hash",
        "output_hash",
        "status",
        "cortical_output_path",
        "baseline_features_path",
        "temporal_diagnostics_path",
    ]
    write_csv_dicts(output_root / "video_metadata.csv", video_rows, video_fields)

    row_fields = [
        "video_id",
        "row_index",
        "time_seconds",
        "clip_window_start_seconds",
        "clip_window_end_seconds",
        "temporal_semantics",
        "black_frame_fraction",
        "duplicate_frame_fraction",
        "quality_black_frame_flag",
        "quality_duplicate_frame_flag",
        "quality_exclusion_flag",
        "quality_weight_suggested",
        "motion_absdiff_mean",
        "luma_mean",
        "luma_std",
        "frame_luma_std_mean",
        "output_file_path",
    ]
    write_csv_dicts(output_root / "row_index.csv", row_index_rows, row_fields)
    parquet_written = try_write_parquet(output_root / "row_index.parquet", row_index_rows)
    if not parquet_written:
        atomic_write_text(
            output_root / "row_index.parquet.unavailable.json",
            json.dumps({
                "status": "not_written",
                "reason": "pyarrow is not installed in this environment",
                "csv_fallback": "row_index.csv",
            }, indent=2, sort_keys=True) + "\n",
        )

    completed_video_ids = [str(row["video_id"]) for row in video_rows if row.get("status") == "success"]
    quality_video_ids = [
        str(row["video_id"])
        for row in video_rows
        if row.get("status") == "success"
        and float(row.get("black_frame_fraction_max") or 0) <= 0.5
        and float(row.get("duplicate_frame_fraction_max") or 0) <= 0.95
    ]
    split_common = {
        "created_at": utc_now(),
        "note": "Split manifests only. No model training or benchmarking was run.",
        "split_method": "deterministic sorted video IDs unless otherwise specified",
    }
    atomic_write_text(
        output_root / "splits_by_video.json",
        json.dumps({**split_common, "splits": deterministic_three_way_split(completed_video_ids)}, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(
        output_root / "splits_quality_filtered.json",
        json.dumps({
            **split_common,
            "quality_filter": "black_frame_fraction_max <= 0.5 and duplicate_frame_fraction_max <= 0.95",
            "splits": deterministic_three_way_split(quality_video_ids),
        }, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(
        output_root / "splits_duration_balanced.json",
        json.dumps({
            **split_common,
            "split_method": "greedy duration-descending round-robin with about 70/20/10 assignment cadence",
            "splits": duration_balanced_split([row for row in video_rows if row.get("status") == "success"]),
        }, indent=2, sort_keys=True) + "\n",
    )

    atomic_write_text(
        output_root / "labels_aligned_2hz.README.md",
        "# labels_aligned_2hz.parquet\n\n"
        "This TRIBE post-pass does not create labels unless an explicit annotation "
        "alignment source is provided. No PCA, bridge training, benchmarking, delta "
        "target creation, or label interpolation is performed in this run.\n",
    )
    baseline_readme = """# Baseline Readiness

This output bundle is prepared for later local baselines and controls. It does
not run PCA, bridge training, benchmarking, spike/delta analysis, or label
alignment.

Saved arrays support:

- Autoregressive baseline: use future local label/annotation alignment with `row_index.csv` or `row_index.parquet` timestamps.
- Quality/motion/luma baseline: `baseline_features_rowlevel.npz` quality arrays plus `video_metadata.csv` summaries.
- Black/static-row controls: use `quality_exclusion_flag` to exclude flagged rows,
  or `quality_weight_suggested` to downweight them. This post-pass records the
  flags but does not drop rows.
- V-JEPA grouped baseline: `tribe_grouped_video_feature [rows,2,1408]`.
- TRIBE cortical baseline: `cortical_prediction [rows,20484]`.
- TRIBE plus temporal diagnostics baseline: `cortical_prediction` plus `vjepa_temporal_diagnostics.npz` compact temporal arrays.
- Shuffled/shifted controls: `row_index` timestamps, `sample_frame_indices`, `sample_time_seconds`, and per-video split manifests.

The bundle is intended to be sufficient for local PCA, bridge training,
benchmarking, delta testing, spike detection, controls, and reporting without
the full upstream V-JEPA cache. The upstream cache remains the authoritative
archive for full hidden states and full temporal tensors.
"""
    atomic_write_text(output_root / "BASELINE_READINESS.md", baseline_readme)


def environment_metadata(args: argparse.Namespace) -> dict[str, Any]:
    script_path = Path(__file__)
    try:
        repo_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        repo_commit = None
    checkpoint = Path(args.checkpoint)
    return {
        "dataset_name": "AGAIN_cleaned",
        "num_expected_videos": 995,
        "row_rate_hz": EXPECTED_ROW_RATE_HZ,
        "vjepa_encoder": "V-JEPA 2.1 ViT-g target_encoder",
        "vjepa_schema": "again_dense_vjepa21_vitg_temporal_pool_v2",
        "tribe_checkpoint": str(checkpoint),
        "tribe_checkpoint_hash": sha256_file(checkpoint) if checkpoint.exists() else None,
        "adapter_function": "group_mean_tribe_video_layers",
        "created_at": utc_now(),
        "repo_commit": repo_commit,
        "repo_commit_hash": repo_commit,
        "script_sha256": sha256_file(script_path) if script_path.exists() else None,
        "command_used": " ".join(sys.argv),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "precision": "float16",
        "cache_only": True,
        "forbid_video_decode": bool(args.forbid_video_decode),
        "forbid_vjepa": bool(args.forbid_vjepa),
        "pca_run": False,
        "bridge_training_run": False,
        "benchmark_run": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default="output/cache")
    parser.add_argument("--output-root", default="tribe_v2_outputs")
    parser.add_argument("--tribe-model-py", default="external_models/tribev2-official/tribev2/model.py")
    parser.add_argument("--checkpoint", default="models/tribe_v2/tribev2_pytorch_float16.ckpt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--cache-only", action="store_true", default=True)
    parser.add_argument("--forbid-video-decode", action="store_true", default=True)
    parser.add_argument("--forbid-vjepa", action="store_true", default=True)
    parser.add_argument("--fail-on-missing-cache", action="store_true", default=True)
    parser.add_argument("--save-grouped-video-feature", action="store_true", default=True)
    parser.add_argument("--save-quality-signals", action="store_true", default=True)
    parser.add_argument("--save-sample-times", action="store_true", default=True)
    parser.add_argument("--save-compact-temporal-diagnostics", action="store_true", default=True)
    parser.add_argument("--require-temporal-diagnostics", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.cache_only or not args.forbid_video_decode or not args.forbid_vjepa:
        raise SystemExit("Refusing to run without cache-only/no-video/no-VJEPA guardrails")
    cache_root = Path(args.cache_root)
    output_root = Path(args.output_root)
    per_video_root = output_root / "per_video"
    per_video_root.mkdir(parents=True, exist_ok=True)
    write_global_schema(output_root)
    manifest_jsonl = output_root / "tribe_v2_postpass_manifest.jsonl"
    failed_jsonl = output_root / "failed_videos.jsonl"
    failed_jsonl.touch(exist_ok=True)

    run_metadata = environment_metadata(args)
    dirs = complete_cache_dirs(cache_root, fail_on_missing_cache=args.fail_on_missing_cache)
    if args.smoke_test:
        dirs = choose_smoke_dirs(dirs)
    elif args.limit_videos:
        dirs = dirs[: args.limit_videos]

    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available() and not args.dry_run:
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(requested_device if not args.dry_run else "cpu")
    model = None if args.dry_run else build_tribe_model(Path(args.tribe_model_py), Path(args.checkpoint), device)
    run_metadata.update({
        "num_expected_videos": len(dirs),
        "num_completed": 0,
        "num_failed": 0,
        "total_rows": 0,
        "total_duration_seconds": 0.0,
    })
    atomic_write_text(output_root / "global_run_metadata.json", json.dumps(run_metadata, indent=2, sort_keys=True) + "\n")

    completed = failed = total_rows = 0
    smoke_rows: list[dict[str, Any]] = []
    for index, cache_dir in enumerate(dirs, start=1):
        result = process_video(cache_dir, per_video_root, manifest_jsonl, failed_jsonl, model, device, args, run_metadata)
        if result.get("status") in ("success", "skipped", "dry_run_validated"):
            completed += 1
            total_rows += int(result.get("rows") or 0)
        elif result.get("status") == "failed":
            failed += 1
        smoke_rows.append(result)
        print(
            f"tribe_video={index}/{len(dirs)} name={cache_dir.name} status={result.get('status')} "
            f"rows={result.get('rows', '')} seconds={result.get('runtime_seconds', '')}",
            flush=True,
        )

    run_metadata.update({
        "num_completed": completed,
        "num_failed": failed,
        "total_rows": total_rows,
        "finished_at": utc_now(),
    })
    atomic_write_text(output_root / "global_run_metadata.json", json.dumps(run_metadata, indent=2, sort_keys=True) + "\n")
    summary = summarize_outputs(output_root)
    write_baseline_readiness_outputs(output_root)
    summary = summarize_outputs(output_root)
    if args.smoke_test:
        report = {
            "videos_tested": [row.get("video_id") for row in smoke_rows],
            "rows_per_video": {row.get("video_id"): row.get("rows") for row in smoke_rows},
            "runtime_per_video": {row.get("video_id"): row.get("runtime_seconds") for row in smoke_rows},
            "statuses": {row.get("video_id"): row.get("status") for row in smoke_rows},
            "estimated_runtime_for_995_videos_seconds": (
                (sum(float(row.get("runtime_seconds") or 0) for row in smoke_rows) / len(smoke_rows)) * 995
                if smoke_rows else None
            ),
            "summary": summary,
        }
        atomic_write_text(output_root / "smoke_test_report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
