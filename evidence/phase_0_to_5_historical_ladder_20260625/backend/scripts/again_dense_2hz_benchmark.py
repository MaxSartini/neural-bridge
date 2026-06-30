"""Dense AGAIN 2Hz label alignment and first-pass benchmarks.

This module consumes the H100-generated V-JEPA 2.1 / TRIBE v2 cache only. It
does not decode raw video, run V-JEPA, or run TRIBE. The benchmark contract is
true 2Hz-on-2Hz: row timing comes from the dense cache row index and target
labels are aligned onto those exact 0.5s rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

try:
    import mlx.core as mx
except Exception:  # pragma: no cover - exercised only on non-MLX machines.
    mx = None

from backend.scripts.again_boundary_manifest import BOUNDARY_POLICY
from backend.scripts.again_scout_sparse_pipeline import assert_again_only_output_path
from backend.scripts.again_sparse_tribe_teacher_500 import threshold_from_train, top_recall


DEFAULT_DENSE_ROOT = Path(".cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz")
DEFAULT_ANNOTATION_PATH = Path("outputs/again_cleaned_inventory_audit_20260621_123531/again_manifest_proposal.csv")
DEFAULT_BOUNDARY_PATH = Path("outputs/again_video_boundary_audit_20260621_204520/again_video_boundary_recommendations.csv")
SCHEMA_VERSION = "again_dense_2hz_benchmark_v1"
ROW_RATE_HZ = 2.0
ROW_STEP_SECONDS = 0.5
HORIZON_ROWS = (1, 2, 4, 6)
DEFAULT_RIDGE_ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)
AR_FEATURE_COLUMNS = (
    "arousal",
    "arousal_lag_1row",
    "arousal_lag_2row",
    "arousal_lag_4row",
    "arousal_delta_prev_1row",
    "arousal_delta_prev_2row",
    "arousal_delta_prev_4row",
)
QUALITY_FEATURE_COLUMNS = (
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
)
TIME_FEATURE_COLUMNS = (
    "time_seconds",
    "video_time_fraction",
    "row_index",
    "sin_time_10s",
    "cos_time_10s",
    "sin_time_30s",
    "cos_time_30s",
)


@dataclass(frozen=True)
class TargetSpec:
    name: str
    value_column: str
    mask_column: str
    threshold_mode: str = "train_quantile"
    quantile: float = 0.90
    transform: str = "positive_delta"


TARGET_SPECS = (
    TargetSpec(
        name="arousal_spike_rows_2_6_train_q90",
        value_column="future_arousal_max_delta_rows_2_6",
        mask_column="target_mask_arousal_spike_rows_2_6",
        transform="positive_delta",
    ),
    TargetSpec(
        name="arousal_delta_p2rows_train_q90",
        value_column="future_arousal_delta_p2rows",
        mask_column="target_mask_arousal_delta_p2rows",
        transform="positive_delta",
    ),
    TargetSpec(
        name="arousal_abs_delta_p4rows_train_q90",
        value_column="future_arousal_delta_p4rows",
        mask_column="target_mask_arousal_delta_p4rows",
        transform="abs_movement",
    ),
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def clean_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean_csv(row.get(key, "")) for key in fields})


def clean_csv(value: Any) -> Any:
    value = clean_json(value)
    if isinstance(value, bool):
        return str(value).lower()
    return "" if value is None else value


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dense_cache_only(dense_root: Path) -> None:
    required = [
        dense_root / "row_index.parquet",
        dense_root / "per_video",
        dense_root / "global_run_metadata.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Dense H100 cache is incomplete; missing {missing}")
    metadata = json.loads((dense_root / "global_run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("cache_only") is not True or metadata.get("forbid_vjepa") is not True:
        raise ValueError("Dense root metadata does not describe the cache-only H100 postpass contract")


def read_row_index(dense_root: Path) -> pd.DataFrame:
    parquet_path = dense_root / "row_index.parquet"
    csv_path = dense_root / "row_index.csv"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"No row_index.parquet or row_index.csv under {dense_root}")
    required = {"video_id", "row_index", "time_seconds", "clip_window_start_seconds", "clip_window_end_seconds"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Row index missing required columns: {sorted(missing)}")
    if df.duplicated(["video_id", "row_index"]).any():
        raise ValueError("Duplicate video_id,row_index keys in dense row index")
    if not np.isfinite(df["time_seconds"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Non-finite time_seconds in dense row index")
    return df.sort_values(["video_id", "row_index"]).reset_index(drop=True)


def read_boundary_table(boundary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(boundary_path)
    if "video_id" not in df:
        raise ValueError(f"Boundary table lacks video_id: {boundary_path}")
    return df.drop_duplicates("video_id", keep="first").set_index("video_id", drop=False)


def read_annotation_table(annotation_path: Path) -> pd.DataFrame:
    usecols = [
        "dataset_name",
        "video_id",
        "video_path",
        "time_start_seconds",
        "frame_index",
        "timestamp_index",
        "arousal",
        "valence",
        "participant_id",
        "session_id",
        "game",
        "genre",
        "aggregate_method",
        "split_group",
        "source_metadata",
        "alignment_status",
    ]
    df = pd.read_csv(annotation_path, usecols=lambda col: col in set(usecols), low_memory=False)
    for column in ("time_start_seconds", "frame_index", "arousal", "valence"):
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["video_id", "time_start_seconds", "arousal"]).copy()
    df = df.sort_values(["video_id", "time_start_seconds"]).reset_index(drop=True)
    return df


def _nearest_source(times: np.ndarray, target_times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos = np.searchsorted(times, target_times, side="left")
    prev_idx = np.clip(pos - 1, 0, len(times) - 1)
    next_idx = np.clip(pos, 0, len(times) - 1)
    prev_times = times[prev_idx]
    next_times = times[next_idx]
    use_next = np.abs(next_times - target_times) < np.abs(target_times - prev_times)
    nearest_times = np.where(use_next, next_times, prev_times)
    nearest_idx = np.where(use_next, next_idx, prev_idx)
    return nearest_times, prev_times, next_times, nearest_idx.astype(np.int64)


def _interp_or_nan(source_times: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    finite = np.isfinite(source_times) & np.isfinite(values)
    if np.sum(finite) < 2:
        return np.full(len(target_times), np.nan, dtype=np.float64)
    return np.interp(target_times, source_times[finite], values[finite], left=np.nan, right=np.nan)


def build_labels_aligned_2hz(
    *,
    dense_root: Path,
    annotation_path: Path = DEFAULT_ANNOTATION_PATH,
    boundary_path: Path = DEFAULT_BOUNDARY_PATH,
    max_alignment_tolerance_seconds: float = 0.30,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ensure_dense_cache_only(dense_root)
    row_index = read_row_index(dense_root)
    boundaries = read_boundary_table(boundary_path)
    annotations = read_annotation_table(annotation_path)
    grouped_annotations = {video_id: group.copy() for video_id, group in annotations.groupby("video_id", sort=False)}
    boundary_columns = {
        "video_path",
        "game",
        "participant_id",
        "session_id",
        "recommended_policy",
        "recommended_benchmark_start_seconds",
        "recommended_benchmark_end_seconds",
        "target_safe_end_future_1_3s_seconds",
        "annotation_start_seconds",
        "annotation_end_seconds",
        "boundary_confidence",
        "notes",
    }
    chunks: list[pd.DataFrame] = []
    missing_annotation_videos: list[str] = []
    non_policy_videos: list[str] = []
    no_boundary_videos: list[str] = []
    for video_id, video_rows in row_index.groupby("video_id", sort=False):
        out = video_rows.copy()
        out["dataset_name"] = "AGAIN_cleaned"
        out["row_rate_hz"] = ROW_RATE_HZ
        out["alignment_policy"] = BOUNDARY_POLICY
        out["label_alignment_method"] = "linear_interpolation_from_native_again_annotations_to_dense_2hz_rows"
        out["native_label_preserves_subsecond_movement"] = True
        out["label_available"] = False
        out["boundary_inclusion_reason"] = "uninitialized"
        out["alignment_distance_seconds"] = np.nan
        out["source_annotation_time_nearest"] = np.nan
        out["source_annotation_time_prev"] = np.nan
        out["source_annotation_time_next"] = np.nan
        out["source_annotation_nearest_index"] = -1
        out["source_annotation_frame_nearest"] = np.nan
        out["arousal"] = np.nan
        out["valence"] = np.nan
        out["participant_id"] = ""
        out["session_id"] = ""
        out["game"] = ""
        out["genre"] = ""
        out["split_group"] = "not_assigned"
        out["source_annotation_alignment_status"] = ""
        boundary = boundaries.loc[video_id] if video_id in boundaries.index else None
        if boundary is None:
            no_boundary_videos.append(video_id)
            out["boundary_inclusion_reason"] = "missing_boundary_recommendation"
            chunks.append(out)
            continue
        for column in boundary_columns:
            if column in boundary.index:
                out[column] = boundary.get(column)
        if str(boundary.get("recommended_policy", "")) != BOUNDARY_POLICY:
            non_policy_videos.append(video_id)
            out["boundary_inclusion_reason"] = "non_policy_boundary_recommendation"
            chunks.append(out)
            continue
        ann = grouped_annotations.get(video_id)
        if ann is None or ann.empty:
            missing_annotation_videos.append(video_id)
            out["boundary_inclusion_reason"] = "missing_annotation"
            chunks.append(out)
            continue

        times = ann["time_start_seconds"].to_numpy(dtype=np.float64)
        arousal = ann["arousal"].to_numpy(dtype=np.float64)
        valence = ann["valence"].to_numpy(dtype=np.float64) if "valence" in ann else np.full(len(ann), np.nan)
        frames = ann["frame_index"].to_numpy(dtype=np.float64) if "frame_index" in ann else np.full(len(ann), np.nan)
        target_times = out["time_seconds"].to_numpy(dtype=np.float64)
        nearest, prev_times, next_times, nearest_idx = _nearest_source(times, target_times)
        distance = np.abs(nearest - target_times)
        benchmark_start = float(pd.to_numeric(boundary.get("recommended_benchmark_start_seconds"), errors="coerce"))
        benchmark_end = float(pd.to_numeric(boundary.get("recommended_benchmark_end_seconds"), errors="coerce"))
        within_boundary = (target_times >= benchmark_start - 1e-9) & (target_times <= benchmark_end + 1e-9)
        within_annotation = (target_times >= np.nanmin(times) - 1e-9) & (target_times <= np.nanmax(times) + 1e-9)
        within_tolerance = distance <= max_alignment_tolerance_seconds
        label_available = within_boundary & within_annotation & within_tolerance
        out["label_available"] = label_available
        out["boundary_inclusion_reason"] = np.where(
            label_available,
            "included_boundary_and_tolerance",
            np.where(~within_boundary, "outside_recommended_benchmark_boundary", "outside_alignment_tolerance_or_annotation"),
        )
        out["alignment_distance_seconds"] = distance
        out["source_annotation_time_nearest"] = nearest
        out["source_annotation_time_prev"] = prev_times
        out["source_annotation_time_next"] = next_times
        out["source_annotation_nearest_index"] = nearest_idx
        out["source_annotation_frame_nearest"] = frames[nearest_idx]
        out["arousal"] = np.where(label_available, _interp_or_nan(times, arousal, target_times), np.nan)
        out["valence"] = np.where(label_available, _interp_or_nan(times, valence, target_times), np.nan)
        source_row = ann.iloc[0].to_dict()
        for col in ("participant_id", "session_id", "game", "genre", "split_group", "alignment_status"):
            if col in source_row:
                target_col = "source_annotation_alignment_status" if col == "alignment_status" else col
                out[target_col] = source_row[col]
        chunks.append(out)

    manifest = pd.concat(chunks, ignore_index=True)
    manifest = add_dense_2hz_targets_and_ar_features(manifest)
    summary = summarize_label_manifest(
        manifest,
        row_index=row_index,
        annotation_path=annotation_path,
        boundary_path=boundary_path,
        missing_annotation_videos=missing_annotation_videos,
        non_policy_videos=non_policy_videos,
        no_boundary_videos=no_boundary_videos,
        max_alignment_tolerance_seconds=max_alignment_tolerance_seconds,
    )
    return manifest, summary


def add_dense_2hz_targets_and_ar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["video_id", "row_index"]).reset_index(drop=True).copy()
    for rows in HORIZON_ROWS:
        seconds = rows * ROW_STEP_SECONDS
        col = f"future_arousal_delta_p{rows}rows"
        mask_col = f"target_mask_arousal_delta_p{rows}rows"
        val_col = f"future_valence_delta_p{rows}rows"
        val_mask_col = f"target_mask_valence_delta_p{rows}rows"
        out[col] = np.nan
        out[mask_col] = False
        out[val_col] = np.nan
        out[val_mask_col] = False
        out[f"horizon_seconds_p{rows}rows"] = seconds
    out["future_arousal_max_delta_rows_2_6"] = np.nan
    out["target_mask_arousal_spike_rows_2_6"] = False
    out["future_arousal_abs_movement_p4rows"] = np.nan
    out["target_mask_arousal_abs_movement_p4rows"] = False
    ar_cols = ["arousal_lag_1row", "arousal_lag_2row", "arousal_lag_4row"]
    for col in ar_cols:
        out[col] = np.nan
        out[f"{col}_available"] = False
    out["arousal_delta_prev_1row"] = np.nan
    out["arousal_delta_prev_2row"] = np.nan
    out["arousal_delta_prev_4row"] = np.nan
    for video_id, idx in out.groupby("video_id", sort=False).groups.items():
        positions = np.asarray(idx, dtype=np.int64)
        video = out.loc[positions]
        label_mask = video["label_available"].to_numpy(dtype=bool)
        arousal = video["arousal"].to_numpy(dtype=np.float64)
        valence = video["valence"].to_numpy(dtype=np.float64)
        for rows in HORIZON_ROWS:
            if len(video) <= rows:
                continue
            src = positions[:-rows]
            dst = positions[rows:]
            feasible = label_mask[:-rows] & label_mask[rows:]
            delta = arousal[rows:] - arousal[:-rows]
            out.loc[src, f"future_arousal_delta_p{rows}rows"] = np.where(feasible, delta, np.nan)
            out.loc[src, f"target_mask_arousal_delta_p{rows}rows"] = feasible
            val_feasible = feasible & np.isfinite(valence[:-rows]) & np.isfinite(valence[rows:])
            out.loc[src, f"future_valence_delta_p{rows}rows"] = np.where(val_feasible, valence[rows:] - valence[:-rows], np.nan)
            out.loc[src, f"target_mask_valence_delta_p{rows}rows"] = val_feasible
        for lag in (1, 2, 4):
            if len(video) <= lag:
                continue
            dst_pos = positions[lag:]
            feasible_lag = label_mask[lag:] & label_mask[:-lag]
            lag_values = np.where(feasible_lag, arousal[:-lag], np.nan)
            out.loc[dst_pos, f"arousal_lag_{lag}row"] = lag_values
            out.loc[dst_pos, f"arousal_lag_{lag}row_available"] = feasible_lag
            out.loc[dst_pos, f"arousal_delta_prev_{lag}row"] = np.where(feasible_lag, arousal[lag:] - arousal[:-lag], np.nan)
        if len(video) > 6:
            base = positions[:-6]
            future = np.vstack([arousal[r : len(video) - 6 + r] for r in range(2, 7)])
            future_masks = np.vstack([label_mask[r : len(video) - 6 + r] for r in range(2, 7)])
            feasible = label_mask[:-6] & np.all(future_masks, axis=0)
            max_delta = np.full(len(base), np.nan, dtype=np.float64)
            if np.any(feasible):
                max_delta[feasible] = np.max(future[:, feasible] - arousal[:-6][None, feasible], axis=0)
            out.loc[base, "future_arousal_max_delta_rows_2_6"] = np.where(feasible, max_delta, np.nan)
            out.loc[base, "target_mask_arousal_spike_rows_2_6"] = feasible
        mask4 = out.loc[positions, "target_mask_arousal_delta_p4rows"].to_numpy(dtype=bool)
        val4 = out.loc[positions, "future_arousal_delta_p4rows"].to_numpy(dtype=np.float64)
        out.loc[positions, "future_arousal_abs_movement_p4rows"] = np.where(mask4, np.abs(val4), np.nan)
        out.loc[positions, "target_mask_arousal_abs_movement_p4rows"] = mask4
    out["ar_context_available"] = out[[f"arousal_lag_{lag}row_available" for lag in (1, 2, 4)]].all(axis=1)
    durations = out.groupby("video_id")["time_seconds"].transform("max").replace(0, np.nan)
    out["video_time_fraction"] = out["time_seconds"] / durations
    out["sin_time_10s"] = np.sin(2 * np.pi * out["time_seconds"] / 10.0)
    out["cos_time_10s"] = np.cos(2 * np.pi * out["time_seconds"] / 10.0)
    out["sin_time_30s"] = np.sin(2 * np.pi * out["time_seconds"] / 30.0)
    out["cos_time_30s"] = np.cos(2 * np.pi * out["time_seconds"] / 30.0)
    return out


def summarize_label_manifest(
    manifest: pd.DataFrame,
    *,
    row_index: pd.DataFrame,
    annotation_path: Path,
    boundary_path: Path,
    missing_annotation_videos: Sequence[str],
    non_policy_videos: Sequence[str],
    no_boundary_videos: Sequence[str],
    max_alignment_tolerance_seconds: float,
) -> dict[str, Any]:
    labeled = manifest["label_available"].to_numpy(dtype=bool)
    labels_per_video = manifest.groupby("video_id")["label_available"].sum()
    first_times = manifest.groupby("video_id")["time_seconds"].first()
    movement = manifest.loc[manifest["target_mask_arousal_delta_p1rows"].astype(bool), "future_arousal_delta_p1rows"]
    abs_movement = movement.abs().dropna()
    hist_counts, hist_edges = np.histogram(abs_movement.to_numpy(dtype=np.float64), bins=[0, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, np.inf])
    target_coverage = {}
    for spec in TARGET_SPECS:
        mask = manifest[spec.mask_column].astype(bool)
        target_coverage[spec.name] = {
            "source_column": spec.value_column,
            "mask_column": spec.mask_column,
            "rows": int(mask.sum()),
            "videos": int(manifest.loc[mask, "video_id"].nunique()),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dense_root_contract": "H100 V-JEPA 2.1 ViT-G + TRIBE v2 cache-only postpass",
        "true_2hz_on_2hz": True,
        "row_rate_hz": ROW_RATE_HZ,
        "row_step_seconds": ROW_STEP_SECONDS,
        "dense_rows": int(len(row_index)),
        "manifest_rows": int(len(manifest)),
        "labeled_rows": int(np.sum(labeled)),
        "unlabeled_rows": int(len(manifest) - np.sum(labeled)),
        "videos_total": int(manifest["video_id"].nunique()),
        "videos_with_labels": int((labels_per_video > 0).sum()),
        "videos_with_zero_labeled_rows": int((labels_per_video == 0).sum()),
        "videos_with_any_unlabeled_rows": int(manifest.loc[~labeled, "video_id"].nunique()),
        "missing_annotation_videos": sorted(set(missing_annotation_videos)),
        "non_policy_videos": sorted(set(non_policy_videos)),
        "no_boundary_videos": sorted(set(no_boundary_videos)),
        "max_alignment_tolerance_seconds": max_alignment_tolerance_seconds,
        "rows_outside_alignment_tolerance_or_annotation": int((manifest["boundary_inclusion_reason"] == "outside_alignment_tolerance_or_annotation").sum()),
        "rows_outside_boundary": int((manifest["boundary_inclusion_reason"] == "outside_recommended_benchmark_boundary").sum()),
        "rows_missing_ar_context": int((manifest["label_available"].astype(bool) & ~manifest["ar_context_available"].astype(bool)).sum()),
        "first_timestamp_counts": {str(key): int(value) for key, value in first_times.value_counts().sort_index().items()},
        "target_coverage": target_coverage,
        "alignment_distance_summary": finite_summary(manifest.loc[labeled, "alignment_distance_seconds"]),
        "half_second_abs_arousal_movement_histogram": {
            "bin_edges": [float(edge) if math.isfinite(float(edge)) else "inf" for edge in hist_edges],
            "counts": [int(v) for v in hist_counts],
            "rows": int(len(abs_movement)),
        },
        "between_second_rows_preserved": True,
        "small_spike_targets_created": True,
        "annotation_path": str(annotation_path),
        "annotation_sha256": sha256_file(annotation_path) if annotation_path.exists() else "",
        "boundary_path": str(boundary_path),
        "boundary_sha256": sha256_file(boundary_path) if boundary_path.exists() else "",
        "vjepa_encoding_run": False,
        "tribe_encoding_run": False,
        "pca_run": False,
        "benchmark_run": False,
    }


def finite_summary(values: pd.Series | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.quantile(arr, 0.5)),
        "p90": float(np.quantile(arr, 0.9)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def write_label_report(path: Path, summary: dict[str, Any], manifest_path: Path) -> None:
    target_lines = [
        f"- `{name}`: rows `{info['rows']}`, videos `{info['videos']}`, source `{info['source_column']}`"
        for name, info in summary["target_coverage"].items()
    ]
    hist = summary["half_second_abs_arousal_movement_histogram"]
    hist_lines = [
        f"- `{hist['bin_edges'][i]}` to `{hist['bin_edges'][i + 1]}`: `{count}` rows"
        for i, count in enumerate(hist["counts"])
    ]
    lines = [
        "# AGAIN Dense 2Hz Label Alignment",
        "",
        "## Scope",
        "",
        "- This is a true 2Hz-on-2Hz supervised alignment for the dense H100 AGAIN cache.",
        "- It uses saved dense-cache `time_seconds`; it does not collapse labels to 1Hz.",
        "- No V-JEPA/TRIBE re-encoding, PCA, bridge training, or benchmark fitting was performed.",
        "",
        "## Coverage",
        "",
        f"- dense rows: `{summary['dense_rows']}`",
        f"- manifest rows: `{summary['manifest_rows']}`",
        f"- labeled rows: `{summary['labeled_rows']}`",
        f"- unlabeled rows: `{summary['unlabeled_rows']}`",
        f"- videos total: `{summary['videos_total']}`",
        f"- videos with labels: `{summary['videos_with_labels']}`",
        f"- videos with zero labeled rows: `{summary['videos_with_zero_labeled_rows']}`",
        f"- videos with any unlabeled rows: `{summary['videos_with_any_unlabeled_rows']}`",
        f"- rows outside boundary: `{summary['rows_outside_boundary']}`",
        f"- rows outside tolerance/annotation: `{summary['rows_outside_alignment_tolerance_or_annotation']}`",
        f"- rows missing AR context: `{summary['rows_missing_ar_context']}`",
        f"- first timestamp counts: `{summary['first_timestamp_counts']}`",
        "",
        "## Targets",
        "",
        *target_lines,
        "",
        "Targets store continuous future movement values and masks. Binary event thresholds are selected inside each train fold during the benchmark; test labels do not set thresholds.",
        "",
        "## 0.5s Movement Histogram",
        "",
        "Absolute arousal movement over +1 dense row (+0.5s):",
        "",
        *hist_lines,
        "",
        "## Guardrails",
        "",
        "- between-second label movements preserved: `true`",
        "- small-spike targets created: `true`",
        "- primary row index source: dense cache `row_index.parquet`",
        f"- output manifest: `{manifest_path}`",
        "- vjepa_encoding_run=`false`",
        "- tribe_encoding_run=`false`",
        "- pca_run=`false`",
        "- benchmark_run=`false`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def load_labels(dense_root: Path) -> pd.DataFrame:
    path = dense_root / "labels_aligned_2hz.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing labels_aligned_2hz.parquet; run build_again_labels_aligned_2hz.py first: {path}")
    df = pd.read_parquet(path)
    if df.duplicated(["video_id", "row_index"]).any():
        raise ValueError("Duplicate video_id,row_index keys in aligned labels")
    if abs(float(df["row_rate_hz"].dropna().median()) - ROW_RATE_HZ) > 1e-6:
        raise ValueError("Aligned label manifest is not 2Hz")
    return df.sort_values(["video_id", "row_index"]).reset_index(drop=True)


def target_base_mask(df: pd.DataFrame, spec: TargetSpec) -> np.ndarray:
    mask = (
        df["label_available"].to_numpy(dtype=bool)
        & df["ar_context_available"].to_numpy(dtype=bool)
        & df[spec.mask_column].to_numpy(dtype=bool)
        & np.isfinite(df[spec.value_column].to_numpy(dtype=np.float64))
    )
    return mask


def threshold_labels(values: np.ndarray, train_mask: np.ndarray, eval_mask: np.ndarray, spec: TargetSpec) -> tuple[np.ndarray, np.ndarray, float]:
    if spec.transform == "abs_movement":
        transformed = np.abs(values)
    elif spec.transform == "positive_delta":
        transformed = np.maximum(values, 0.0)
    else:
        transformed = values
    train_values = transformed[train_mask]
    train_values = train_values[np.isfinite(train_values)]
    if len(train_values) == 0:
        raise ValueError(f"No finite train values for target {spec.name}")
    threshold = float(np.quantile(train_values, spec.quantile))
    y_train = (transformed[train_mask] >= threshold).astype(int)
    y_eval = (transformed[eval_mask] >= threshold).astype(int)
    return y_train, y_eval, threshold


def grouped_video_splits(df: pd.DataFrame, base_mask: np.ndarray, *, n_splits: int) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    idx = np.flatnonzero(base_mask)
    groups = df.loc[idx, "video_id"].astype(str).to_numpy()
    videos = np.unique(groups)
    splits = min(n_splits, len(videos))
    if splits < 2:
        return []
    out = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=splits).split(idx, groups=groups), start=1):
        out.append(("grouped_video", fold, idx[tr], idx[te]))
    return out


def blocked_temporal_split(df: pd.DataFrame, base_mask: np.ndarray, *, train_fraction: float = 0.70) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for _video_id, group in df.loc[base_mask].groupby("video_id", sort=False):
        indices = group.index.to_numpy(dtype=np.int64)
        if len(indices) < 4:
            continue
        cutoff = max(1, min(len(indices) - 1, int(math.floor(len(indices) * train_fraction))))
        train_parts.append(indices[:cutoff])
        test_parts.append(indices[cutoff:])
    if not train_parts or not test_parts:
        return []
    return [("blocked_temporal_70_30", 1, np.concatenate(train_parts), np.concatenate(test_parts))]


def validate_split(df: pd.DataFrame, protocol: str, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    if len(set(train_idx).intersection(set(test_idx))):
        raise ValueError(f"Row leakage in {protocol}")
    if protocol == "grouped_video":
        train_videos = set(df.loc[train_idx, "video_id"].astype(str))
        test_videos = set(df.loc[test_idx, "video_id"].astype(str))
        overlap = train_videos.intersection(test_videos)
        if overlap:
            raise ValueError(f"Grouped-video leakage: {sorted(overlap)[:5]}")


def feature_matrix(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df.loc[:, list(columns)].to_numpy(dtype=np.float32)


def safe_scale_fit_predict(train_x: np.ndarray, test_x: np.ndarray, train_y: np.ndarray, *, alpha: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_x = np.nan_to_num(train_x, nan=0.0, posinf=0.0, neginf=0.0)
    test_x = np.nan_to_num(test_x, nan=0.0, posinf=0.0, neginf=0.0)
    if mx is not None:
        return mlx_ridge_fit_predict(train_x, test_x, train_y, alpha=alpha)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    test_scaled = scaler.transform(test_x)
    model = Ridge(alpha=alpha, solver="lsqr", tol=1e-6)
    model.fit(train_scaled, train_y.astype(np.float32))
    return (
        model.predict(train_scaled).astype(np.float32),
        model.predict(test_scaled).astype(np.float32),
        {"ridge_backend": "sklearn_ridge_lsqr_cpu_fallback", "ridge_alpha": alpha},
    )


def inner_validation_relative_split(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    groups = df.loc[train_idx, "video_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 3:
        splits = min(3, len(unique_groups))
        for inner_train, inner_val in GroupKFold(n_splits=splits).split(np.arange(len(train_idx)), y_train, groups):
            if len(np.unique(y_train[inner_train])) >= 2 and len(np.unique(y_train[inner_val])) >= 2:
                return inner_train, inner_val, f"inner_grouped_video_{splits}fold_first_valid"
    # Fallback remains train-only: use the later portion of each video's outer-train rows as validation.
    inner_train_parts: list[np.ndarray] = []
    inner_val_parts: list[np.ndarray] = []
    for group in unique_groups:
        rel = np.flatnonzero(groups == group)
        if len(rel) < 4:
            continue
        cutoff = max(1, min(len(rel) - 1, int(math.floor(len(rel) * 0.80))))
        inner_train_parts.append(rel[:cutoff])
        inner_val_parts.append(rel[cutoff:])
    if inner_train_parts and inner_val_parts:
        inner_train = np.concatenate(inner_train_parts)
        inner_val = np.concatenate(inner_val_parts)
        if len(np.unique(y_train[inner_train])) >= 2 and len(np.unique(y_train[inner_val])) >= 2:
            return inner_train, inner_val, "inner_blocked_temporal_outer_train_80_20"
    return np.arange(len(train_idx)), np.arange(len(train_idx)), "inner_fallback_train_resubstitution"


def select_alpha_train_only(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha_grid: Sequence[float],
) -> dict[str, Any]:
    valid_grid = [float(alpha) for alpha in alpha_grid if float(alpha) > 0 and math.isfinite(float(alpha))]
    if not valid_grid:
        valid_grid = [1.0]
    inner_train, inner_val, strategy = inner_validation_relative_split(df, train_idx, y_train)
    rows: list[dict[str, Any]] = []
    best_alpha = valid_grid[0]
    best_score = float("-inf")
    for alpha in valid_grid:
        try:
            _train_scores, val_scores, info = safe_scale_fit_predict(
                x_train[inner_train],
                x_train[inner_val],
                y_train[inner_train],
                alpha=alpha,
            )
            score = average_precision_score(y_train[inner_val], val_scores) if len(np.unique(y_train[inner_val])) > 1 else math.nan
            row = {
                "alpha": alpha,
                "inner_validation_pr_auc": score,
                "inner_strategy": strategy,
                **info,
            }
        except Exception as exc:
            score = math.nan
            row = {
                "alpha": alpha,
                "inner_validation_pr_auc": math.nan,
                "inner_strategy": strategy,
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        if math.isfinite(score) and score > best_score:
            best_score = float(score)
            best_alpha = alpha
    return {
        "selected_alpha": best_alpha,
        "inner_validation_pr_auc": best_score if math.isfinite(best_score) else math.nan,
        "inner_validation_strategy": strategy,
        "alpha_grid": valid_grid,
        "alpha_selection_rows": rows,
    }


def mlx_ridge_fit_predict(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    *,
    alpha: float,
    max_iter: int = 220,
    tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    xtr = mx.array(train_x.astype(np.float32, copy=False), dtype=mx.float32)
    xte = mx.array(test_x.astype(np.float32, copy=False), dtype=mx.float32)
    ytr = mx.array(train_y.astype(np.float32, copy=False), dtype=mx.float32)
    mean = mx.mean(xtr, axis=0)
    std = mx.sqrt(mx.var(xtr, axis=0) + 1e-6)
    xtr = (xtr - mean) / std
    xte = (xte - mean) / std
    y_mean = mx.mean(ytr)
    y_centered = ytr - y_mean
    n_features = int(train_x.shape[1])
    gram = mx.transpose(xtr) @ xtr
    rhs = mx.transpose(xtr) @ y_centered
    system = gram + float(alpha) * mx.eye(n_features, dtype=mx.float32)
    weights = mx.zeros((n_features,), dtype=mx.float32)
    residual = rhs - system @ weights
    direction = residual
    rs_old = mx.sum(residual * residual)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        ap = system @ direction
        step = rs_old / (mx.sum(direction * ap) + 1e-8)
        weights = weights + step * direction
        residual = residual - step * ap
        rs_new = mx.sum(residual * residual)
        if float(np.asarray(rs_new)) < tol:
            rs_old = rs_new
            break
        direction = residual + (rs_new / (rs_old + 1e-8)) * direction
        rs_old = rs_new
    train_scores = xtr @ weights + y_mean
    test_scores = xte @ weights + y_mean
    mx.eval(train_scores, test_scores)
    return (
        np.asarray(train_scores, dtype=np.float32),
        np.asarray(test_scores, dtype=np.float32),
        {
            "ridge_backend": "mlx_primal_conjugate_gradient",
            "ridge_alpha": alpha,
            "feature_width": n_features,
            "ridge_iterations": iterations,
            "mlx_available": True,
        },
    )


def metric_row(y_true: np.ndarray, scores: np.ndarray, decision_threshold: float) -> dict[str, Any]:
    pred = (scores >= decision_threshold).astype(int)
    unique = np.unique(y_true)
    return {
        "pr_auc": average_precision_score(y_true, scores) if len(unique) > 1 else math.nan,
        "roc_auc": roc_auc_score(y_true, scores) if len(unique) > 1 else math.nan,
        "f1": f1_score(y_true, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred) if len(unique) > 1 else math.nan,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
        "top_1pct_recall": top_recall(y_true, scores, 0.01),
        "top_5pct_recall": top_recall(y_true, scores, 0.05),
        "top_10pct_recall": top_recall(y_true, scores, 0.10),
        "predicted_positive_rate": float(np.mean(pred)) if len(pred) else math.nan,
        "predicted_positive_count": int(np.sum(pred)),
    }


def regression_metric_row(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    if len(y_true) == 0:
        return {"mae": math.nan, "mse": math.nan, "pearson": math.nan}
    if np.std(y_true) == 0 or np.std(scores) == 0:
        pearson = math.nan
    else:
        pearson = float(np.corrcoef(y_true, scores)[0, 1])
    return {
        "mae": float(mean_absolute_error(y_true, scores)),
        "mse": float(mean_squared_error(y_true, scores)),
        "pearson": pearson,
    }


def evaluate_lanes(
    df: pd.DataFrame,
    feature_map: dict[str, np.ndarray],
    *,
    target_specs: Sequence[TargetSpec] = TARGET_SPECS,
    n_splits: int = 5,
    alpha: float = 1.0,
    alpha_grid: Sequence[float] | None = DEFAULT_RIDGE_ALPHA_GRID,
    random_seed: int = 20260625,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    fold_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_seed)
    for spec in target_specs:
        base_mask = target_base_mask(df, spec)
        splits = grouped_video_splits(df, base_mask, n_splits=n_splits) + blocked_temporal_split(df, base_mask)
        values = df[spec.value_column].to_numpy(dtype=np.float64)
        for protocol, fold, train_idx, test_idx in splits:
            validate_split(df, protocol, train_idx, test_idx)
            split_train_mask = np.zeros(len(df), dtype=bool)
            split_test_mask = np.zeros(len(df), dtype=bool)
            split_train_mask[train_idx] = True
            split_test_mask[test_idx] = True
            y_train, y_test, target_threshold = threshold_labels(values, split_train_mask, split_test_mask, spec)
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            for lane, x in feature_map.items():
                if lane == "random_matched_feature_control":
                    width = int(feature_map.get("raw_cortical_only", x).shape[1])
                    x_train = rng.normal(size=(len(train_idx), width)).astype(np.float32)
                    x_test = rng.normal(size=(len(test_idx), width)).astype(np.float32)
                else:
                    x_train = x[train_idx]
                    x_test = x[test_idx]
                    if lane == "shuffled_cortical_control" or lane == "shuffled_temporal_diagnostics_control":
                        x_train = x_train[rng.permutation(len(x_train))]
                        x_test = x_test[rng.permutation(len(x_test))]
                if alpha_grid is None:
                    selection = {
                        "selected_alpha": float(alpha),
                        "inner_validation_pr_auc": math.nan,
                        "inner_validation_strategy": "fixed_cli_alpha",
                        "alpha_grid": [float(alpha)],
                        "alpha_selection_rows": [],
                    }
                else:
                    selection = select_alpha_train_only(df, train_idx, x_train, y_train, alpha_grid)
                train_scores, test_scores, fit_info = safe_scale_fit_predict(x_train, x_test, y_train, alpha=selection["selected_alpha"])
                decision_threshold = threshold_from_train(y_train, train_scores)
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "target_name": spec.name,
                    "target_value_column": spec.value_column,
                    "target_mask_column": spec.mask_column,
                    "target_threshold_train_only": target_threshold,
                    "target_threshold_quantile": spec.quantile,
                    "target_transform": spec.transform,
                    "validation_protocol": protocol,
                    "fold": fold,
                    "model_lane": lane,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "train_videos": int(df.loc[train_idx, "video_id"].nunique()),
                    "test_videos": int(df.loc[test_idx, "video_id"].nunique()),
                    "train_event_count": int(np.sum(y_train)),
                    "test_event_count": int(np.sum(y_test)),
                    "train_positive_rate": float(np.mean(y_train)),
                    "test_positive_rate": float(np.mean(y_test)),
                    "feature_width": int(x_train.shape[1]),
                    "decision_threshold_train_only": decision_threshold,
                    "selected_ridge_alpha_train_only": selection["selected_alpha"],
                    "inner_validation_pr_auc": selection["inner_validation_pr_auc"],
                    "inner_validation_strategy": selection["inner_validation_strategy"],
                    "ridge_alpha_grid_json": json.dumps(selection["alpha_grid"]),
                    "ridge_alpha_selection_json": json.dumps(clean_json(selection["alpha_selection_rows"]), sort_keys=True),
                    "uses_future_features": False,
                    "uses_train_only_transform": True,
                    "vjepa_encoding_run": False,
                    "tribe_encoding_run": False,
                    "pca_run": False,
                    **fit_info,
                    **metric_row(y_test, test_scores, decision_threshold),
                    **{f"delta_{k}": v for k, v in regression_metric_row(values[test_idx], test_scores).items()},
                }
                fold_rows.append(row)
    fold_df = pd.DataFrame(fold_rows)
    if fold_df.empty:
        return fold_df, pd.DataFrame(), {"fold_rows": 0}
    summary_df = (
        fold_df.groupby(["target_name", "validation_protocol", "model_lane"], dropna=False)
        .agg(
            folds=("fold", "count"),
            rows_test_total=("n_test", "sum"),
            mean_pr_auc=("pr_auc", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_f1=("f1", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_top_1pct_recall=("top_1pct_recall", "mean"),
            mean_top_5pct_recall=("top_5pct_recall", "mean"),
            mean_top_10pct_recall=("top_10pct_recall", "mean"),
            mean_delta_mae=("delta_mae", "mean"),
            mean_delta_mse=("delta_mse", "mean"),
            mean_delta_pearson=("delta_pearson", "mean"),
        )
        .reset_index()
    )
    gates = promotion_gates(summary_df)
    return fold_df, summary_df, gates


def promotion_gates(summary_df: pd.DataFrame) -> dict[str, Any]:
    gates: dict[str, Any] = {"gate_scope": "grouped_video_only", "targets": {}}
    grouped = summary_df[summary_df["validation_protocol"] == "grouped_video"]
    for target, target_df in grouped.groupby("target_name"):
        by_lane = {row["model_lane"]: float(row["mean_pr_auc"]) for _, row in target_df.iterrows()}
        ar = by_lane.get("AR_only")
        raw = by_lane.get("raw_cortical_only")
        ar_raw = by_lane.get("AR_plus_raw_cortical")
        shuffled = by_lane.get("shuffled_cortical_control")
        random_control = by_lane.get("random_matched_feature_control")
        gates["targets"][target] = {
            "ar_only_pr_auc": ar,
            "raw_cortical_pr_auc": raw,
            "ar_plus_raw_cortical_pr_auc": ar_raw,
            "shuffled_cortical_pr_auc": shuffled,
            "random_matched_pr_auc": random_control,
            "raw_beats_ar": bool(raw is not None and ar is not None and raw > ar),
            "ar_plus_raw_beats_ar": bool(ar_raw is not None and ar is not None and ar_raw > ar),
            "ar_plus_raw_beats_shuffled": bool(ar_raw is not None and shuffled is not None and ar_raw > shuffled),
            "strict_raw_promotion_pass": bool(
                raw is not None
                and ar is not None
                and shuffled is not None
                and random_control is not None
                and raw > ar
                and raw > shuffled
                and raw > random_control
            ),
            "strict_ar_plus_raw_promotion_pass": bool(
                ar_raw is not None
                and ar is not None
                and shuffled is not None
                and random_control is not None
                and ar_raw > ar
                and ar_raw > shuffled
                and ar_raw > random_control
            ),
        }
    return gates


def default_output_root(prefix: str) -> Path:
    return Path("outputs") / f"{prefix}_{utc_stamp()}"


def parse_alpha_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("alpha grid cannot be empty")
    if any((not math.isfinite(value)) or value <= 0 for value in values):
        raise ValueError(f"alpha grid must contain positive finite values: {text}")
    return values


def build_ar_feature_map(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {"AR_only": feature_matrix(df, AR_FEATURE_COLUMNS)}


def build_small_control_features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "timestamp_video_time_only_control": feature_matrix(df, TIME_FEATURE_COLUMNS),
        "quality_motion_luma_only_control": feature_matrix(df, QUALITY_FEATURE_COLUMNS),
    }


def local_npz_path(dense_root: Path, video_id: str, filename: str) -> Path:
    return dense_root / "per_video" / video_id / filename


def load_or_build_raw_cortical_projection(
    dense_root: Path,
    row_df: pd.DataFrame,
    *,
    projection_dim: int,
    random_seed: int,
    force: bool = False,
) -> np.ndarray:
    derived = dense_root / "_derived"
    derived.mkdir(parents=True, exist_ok=True)
    path = derived / f"raw_cortical_block_summary_b{projection_dim}.npy"
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("rows") == len(row_df) and meta.get("block_count") == projection_dim:
            return np.load(path, mmap_mode="r")
    if projection_dim <= 0:
        raise ValueError("projection_dim/block count must be positive")
    block_indices = np.array_split(np.arange(20484), projection_dim)
    width = projection_dim * 2
    tmp_path = path.with_suffix(".tmp.npy")
    arr = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float32, shape=(len(row_df), width))
    for video_id, group in row_df.groupby("video_id", sort=False):
        npz_path = local_npz_path(dense_root, str(video_id), "tribe_v2_cortical_predictions.npz")
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing cortical cache for {video_id}: {npz_path}")
        with np.load(npz_path) as npz:
            cortical = np.asarray(npz["cortical_prediction"], dtype=np.float32)
            times = np.asarray(npz["time_seconds"], dtype=np.float64)
        if cortical.shape[0] != len(group):
            raise ValueError(f"Row mismatch for {video_id}: cortical {cortical.shape[0]} vs labels {len(group)}")
        if not np.allclose(times, group["time_seconds"].to_numpy(dtype=np.float64), atol=1e-6):
            raise ValueError(f"Time mismatch for {video_id}")
        means = np.stack([cortical[:, indices].mean(axis=1) for indices in block_indices], axis=1)
        stds = np.stack([cortical[:, indices].std(axis=1) for indices in block_indices], axis=1)
        arr[group.index.to_numpy(dtype=np.int64)] = np.concatenate([means, stds], axis=1)
    arr.flush()
    tmp_path.replace(path)
    write_json(
        meta_path,
        {
            "schema_version": SCHEMA_VERSION,
            "feature": "raw_cortical_block_summary",
            "source": "tribe_v2_cortical_predictions.npz:cortical_prediction [rows,20484]",
            "rows": len(row_df),
            "block_count": projection_dim,
            "width": width,
            "pca_run": False,
            "train_fit": False,
            "label_free_deterministic_block_summary": True,
            "summary_per_block": ["mean", "std"],
        },
    )
    return np.load(path, mmap_mode="r")


def load_or_build_temporal_diagnostic_features(dense_root: Path, row_df: pd.DataFrame, *, force: bool = False) -> np.ndarray:
    derived = dense_root / "_derived"
    derived.mkdir(parents=True, exist_ok=True)
    path = derived / "temporal_diagnostics_summary_features.npy"
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("rows") == len(row_df):
            return np.load(path, mmap_mode="r")
    width = 1 + 20 + 32
    tmp_path = path.with_suffix(".tmp.npy")
    arr = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float32, shape=(len(row_df), width))
    for video_id, group in row_df.groupby("video_id", sort=False):
        npz_path = local_npz_path(dense_root, str(video_id), "vjepa_temporal_diagnostics.npz")
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing temporal diagnostics for {video_id}: {npz_path}")
        with np.load(npz_path) as npz:
            global_std = np.asarray(npz["temporal_std_global"], dtype=np.float32).reshape(-1, 1)
            by_state = np.asarray(npz["temporal_std_by_state"], dtype=np.float32)
            by_token = np.asarray(npz["temporal_std_by_state_token"], dtype=np.float32).mean(axis=1)
        if global_std.shape[0] != len(group):
            raise ValueError(f"Temporal row mismatch for {video_id}")
        arr[group.index.to_numpy(dtype=np.int64)] = np.concatenate([global_std, by_state, by_token], axis=1)
    arr.flush()
    tmp_path.replace(path)
    write_json(
        meta_path,
        {
            "schema_version": SCHEMA_VERSION,
            "feature": "temporal_diagnostics_summary",
            "rows": len(row_df),
            "width": width,
            "columns": ["temporal_std_global"] + [f"temporal_std_state_{i}" for i in range(20)] + [f"temporal_std_token_mean_{i}" for i in range(32)],
            "pca_run": False,
            "vjepa_encoding_run": False,
            "tribe_encoding_run": False,
        },
    )
    return np.load(path, mmap_mode="r")


def run_ar_baseline(
    *,
    dense_root: Path,
    output_root: Path,
    n_splits: int = 5,
    ridge_alpha: float = 1.0,
    ridge_alpha_grid: Sequence[float] | None = DEFAULT_RIDGE_ALPHA_GRID,
) -> dict[str, Any]:
    assert_again_only_output_path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root exists and is non-empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    df = load_labels(dense_root)
    feature_map = build_ar_feature_map(df)
    fold_df, summary_df, gates = evaluate_lanes(df, feature_map, n_splits=n_splits, alpha=ridge_alpha, alpha_grid=ridge_alpha_grid)
    fold_path = output_root / "again_dense_2hz_ar_fold_metrics.csv"
    summary_path = output_root / "again_dense_2hz_ar_summary_metrics.csv"
    fold_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "again_dense_2hz_ar_baseline",
        "true_2hz_on_2hz": True,
        "dense_root": str(dense_root),
        "labels_path": str(dense_root / "labels_aligned_2hz.parquet"),
        "rows": int(len(df)),
        "labeled_rows": int(df["label_available"].sum()),
        "feature_lanes": list(feature_map),
        "targets": [spec.name for spec in TARGET_SPECS],
        "validation_protocols": sorted(fold_df["validation_protocol"].unique().tolist()) if not fold_df.empty else [],
        "ridge_alpha_grid": list(ridge_alpha_grid) if ridge_alpha_grid is not None else [ridge_alpha],
        "ridge_alpha_selection": "train_only_inner_validation" if ridge_alpha_grid is not None else "fixed_cli_alpha",
        "promotion_gates": gates,
        "vjepa_encoding_run": False,
        "tribe_encoding_run": False,
        "pca_run": False,
        "models_trained": True,
    }
    write_json(output_root / "summary.json", summary)
    report_path = Path("reports") / f"again_dense_2hz_ar_baseline_{utc_stamp()}.md"
    write_benchmark_report(report_path, summary, summary_df, gates, title="AGAIN Dense 2Hz AR Baseline")
    write_json(output_root / "run_manifest.json", {**summary, "report_path": str(report_path)})
    return {**summary, "output_root": str(output_root), "report_path": str(report_path)}


def run_raw_cortical_benchmark(
    *,
    dense_root: Path,
    output_root: Path,
    projection_dim: int = 256,
    random_seed: int = 20260625,
    n_splits: int = 5,
    ridge_alpha: float = 1.0,
    ridge_alpha_grid: Sequence[float] | None = DEFAULT_RIDGE_ALPHA_GRID,
    force_features: bool = False,
) -> dict[str, Any]:
    assert_again_only_output_path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root exists and is non-empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    df = load_labels(dense_root)
    raw = load_or_build_raw_cortical_projection(dense_root, df, projection_dim=projection_dim, random_seed=random_seed, force=force_features)
    temporal = load_or_build_temporal_diagnostic_features(dense_root, df, force=force_features)
    ar = feature_matrix(df, AR_FEATURE_COLUMNS)
    time_features = feature_matrix(df, TIME_FEATURE_COLUMNS)
    quality = feature_matrix(df, QUALITY_FEATURE_COLUMNS)
    feature_map = {
        "AR_only": ar,
        "raw_cortical_only": raw,
        "AR_plus_raw_cortical": np.concatenate([ar, raw], axis=1),
        "temporal_diagnostics_only": temporal,
        "AR_plus_temporal_diagnostics": np.concatenate([ar, temporal], axis=1),
        "AR_plus_raw_cortical_plus_temporal_diagnostics": np.concatenate([ar, raw, temporal], axis=1),
        "shuffled_cortical_control": raw,
        "shuffled_temporal_diagnostics_control": temporal,
        "random_matched_feature_control": np.zeros((len(df), projection_dim), dtype=np.float32),
        "timestamp_video_time_only_control": time_features,
        "quality_motion_luma_only_control": quality,
    }
    fold_df, summary_df, gates = evaluate_lanes(
        df,
        feature_map,
        n_splits=n_splits,
        alpha=ridge_alpha,
        alpha_grid=ridge_alpha_grid,
        random_seed=random_seed,
    )
    fold_path = output_root / "again_dense_2hz_raw_cortical_fold_metrics.csv"
    summary_path = output_root / "again_dense_2hz_raw_cortical_summary_metrics.csv"
    gate_path = output_root / "promotion_gates.json"
    fold_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    write_json(gate_path, gates)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "again_dense_2hz_raw_cortical_vs_ar",
        "true_2hz_on_2hz": True,
        "dense_root": str(dense_root),
        "labels_path": str(dense_root / "labels_aligned_2hz.parquet"),
        "rows": int(len(df)),
        "labeled_rows": int(df["label_available"].sum()),
        "projection_dim": projection_dim,
        "raw_cortical_source": "per_video/<video_id>/tribe_v2_cortical_predictions.npz:cortical_prediction [rows,20484]",
        "raw_cortical_model_lane": "label_free_deterministic_block_summary_not_pca_bridge",
        "temporal_diagnostics_used": True,
        "feature_lanes": list(feature_map),
        "targets": [spec.name for spec in TARGET_SPECS],
        "validation_protocols": sorted(fold_df["validation_protocol"].unique().tolist()) if not fold_df.empty else [],
        "ridge_alpha_grid": list(ridge_alpha_grid) if ridge_alpha_grid is not None else [ridge_alpha],
        "ridge_alpha_selection": "train_only_inner_validation" if ridge_alpha_grid is not None else "fixed_cli_alpha",
        "promotion_gates": gates,
        "vjepa_encoding_run": False,
        "tribe_encoding_run": False,
        "pca_run": False,
        "bridge_training_run": False,
        "models_trained": True,
    }
    write_json(output_root / "summary.json", summary)
    report_path = Path("reports") / f"again_dense_2hz_raw_cortical_vs_ar_{utc_stamp()}.md"
    write_benchmark_report(report_path, summary, summary_df, gates, title="AGAIN Dense 2Hz Raw Cortical vs AR")
    write_json(output_root / "run_manifest.json", {**summary, "report_path": str(report_path)})
    return {**summary, "output_root": str(output_root), "report_path": str(report_path)}


def write_benchmark_report(path: Path, summary: dict[str, Any], summary_df: pd.DataFrame, gates: dict[str, Any], *, title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "## Scope",
        "",
        "- This benchmark uses the dense H100 AGAIN cache with true 0.5s row-level targets.",
        "- Saved `time_seconds` values are used directly; no 1Hz fallback was used.",
        "- No V-JEPA/TRIBE re-encoding was performed.",
        "- Binary event thresholds are selected inside each train split from continuous future-label movement.",
        "- This is not PCA bridge training.",
        "",
        "## Coverage",
        "",
        f"- rows: `{summary['rows']}`",
        f"- labeled rows: `{summary['labeled_rows']}`",
        f"- targets: `{summary['targets']}`",
        f"- validation protocols: `{summary['validation_protocols']}`",
        f"- ridge alpha selection: `{summary.get('ridge_alpha_selection', 'fixed_cli_alpha')}`",
        f"- ridge alpha grid: `{summary.get('ridge_alpha_grid', [])}`",
        "- ridge backend: per-fold `mlx_primal_conjugate_gradient` when MLX is available",
        "",
        "## Lane Summary",
        "",
    ]
    if summary_df.empty:
        lines.append("No valid fold rows were produced.")
    else:
        for _, row in summary_df.iterrows():
            lines.append(
                f"- `{row['target_name']}` / `{row['validation_protocol']}` / `{row['model_lane']}`: "
                f"folds `{int(row['folds'])}`, PR-AUC `{100 * row['mean_pr_auc']:.2f}%`, "
                f"ROC-AUC `{100 * row['mean_roc_auc']:.2f}%`, F1 `{100 * row['mean_f1']:.2f}%`"
            )
    lines.extend(["", "## Promotion Gates", ""])
    for target, gate in gates.get("targets", {}).items():
        lines.append(
            f"- `{target}`: raw beats AR `{gate.get('raw_beats_ar')}`, "
            f"AR+raw beats AR `{gate.get('ar_plus_raw_beats_ar')}`, "
            f"strict raw pass `{gate.get('strict_raw_promotion_pass')}`, "
            f"strict AR+raw pass `{gate.get('strict_ar_plus_raw_promotion_pass')}`"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Raw cortical lanes use fixed label-free cortical block mean/std summaries for computational safety when modelling the 20,484 vertex output.",
            "- Temporal diagnostics are non-PCA causal cache diagnostics, not learned bridge features.",
            "- Promotion requires grouped-video wins over AR and nuisance controls; raw cortical losing is a valid outcome.",
            "",
            "## Guardrails",
            "",
            f"- vjepa_encoding_run=`{summary['vjepa_encoding_run']}`",
            f"- tribe_encoding_run=`{summary['tribe_encoding_run']}`",
            f"- pca_run=`{summary['pca_run']}`",
            f"- models_trained=`{summary['models_trained']}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_labels_cli(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build true 2Hz AGAIN labels aligned to the dense H100 row index.")
    parser.add_argument("--dense-root", type=Path, default=DEFAULT_DENSE_ROOT)
    parser.add_argument("--annotation-path", type=Path, default=DEFAULT_ANNOTATION_PATH)
    parser.add_argument("--boundary-path", type=Path, default=DEFAULT_BOUNDARY_PATH)
    parser.add_argument("--max-alignment-tolerance-seconds", type=float, default=0.30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    manifest, summary = build_labels_aligned_2hz(
        dense_root=args.dense_root,
        annotation_path=args.annotation_path,
        boundary_path=args.boundary_path,
        max_alignment_tolerance_seconds=args.max_alignment_tolerance_seconds,
    )
    if args.dry_run:
        print(json.dumps(clean_json(summary), indent=2, sort_keys=True))
        return
    manifest_path = args.dense_root / "labels_aligned_2hz.parquet"
    manifest.to_parquet(manifest_path, index=False)
    summary_path = args.dense_root / "labels_aligned_2hz_summary.json"
    write_json(summary_path, summary)
    report_path = Path("reports") / f"again_labels_aligned_2hz_{utc_stamp()}.md"
    write_label_report(report_path, summary, manifest_path)
    print(json.dumps({"manifest_path": str(manifest_path), "summary_path": str(summary_path), "report_path": str(report_path)}, indent=2))


def ar_baseline_cli(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run AR-only baseline on dense AGAIN 2Hz aligned labels.")
    parser.add_argument("--dense-root", type=Path, default=DEFAULT_DENSE_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--ridge-alpha-grid", default=",".join(str(x) for x in DEFAULT_RIDGE_ALPHA_GRID))
    parser.add_argument("--fixed-ridge-alpha", action="store_true")
    args = parser.parse_args(argv)
    output_root = args.output_root or default_output_root("again_dense_2hz_ar_baseline")
    alpha_grid = None if args.fixed_ridge_alpha else parse_alpha_grid(args.ridge_alpha_grid)
    manifest = run_ar_baseline(
        dense_root=args.dense_root,
        output_root=output_root,
        n_splits=args.n_splits,
        ridge_alpha=args.ridge_alpha,
        ridge_alpha_grid=alpha_grid,
    )
    print(json.dumps(clean_json(manifest), indent=2, sort_keys=True))


def raw_cortical_cli(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run raw cortical/temporal diagnostics vs AR on dense AGAIN 2Hz labels.")
    parser.add_argument("--dense-root", type=Path, default=DEFAULT_DENSE_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--random-seed", type=int, default=20260625)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--ridge-alpha-grid", default=",".join(str(x) for x in DEFAULT_RIDGE_ALPHA_GRID))
    parser.add_argument("--fixed-ridge-alpha", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args(argv)
    output_root = args.output_root or default_output_root("again_dense_2hz_raw_cortical_benchmark")
    alpha_grid = None if args.fixed_ridge_alpha else parse_alpha_grid(args.ridge_alpha_grid)
    manifest = run_raw_cortical_benchmark(
        dense_root=args.dense_root,
        output_root=output_root,
        projection_dim=args.projection_dim,
        random_seed=args.random_seed,
        n_splits=args.n_splits,
        ridge_alpha=args.ridge_alpha,
        ridge_alpha_grid=alpha_grid,
        force_features=args.force_features,
    )
    print(json.dumps(clean_json(manifest), indent=2, sort_keys=True))
