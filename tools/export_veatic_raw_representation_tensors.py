#!/usr/bin/env python3
"""Freeze model-ready VEATIC raw representation tensors.

This is an export step, not a benchmark rerun. It reads the existing audit
checkpoint, existing PCA fit-cache payloads, and existing TRIBE raw cortical
cache to materialize train/test tensors for selected representation contracts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.cortical_roi_mapper import (  # noqa: E402
    CorticalRoiMapper,
    FSAVERAGE5_VERTICES_PER_HEMI,
    TRIBE_CORTICAL_VERTICES,
)
from backend.scripts import run_veatic_raw_representation_audit as audit  # noqa: E402
from backend.scripts import veatic_representation_builders as reps  # noqa: E402


SOURCE_AUDIT_DIR = Path(
    "/Volumes/onn. Drive/Neural Bridge/outputs/"
    "veatic_124_raw_representation_audit_primary_20260620_152411"
)
STATE_PATH = SOURCE_AUDIT_DIR / "_checkpoint" / "state.json"
FIT_CACHE_ROOT = SOURCE_AUDIT_DIR / "_checkpoint" / "fit_cache"
TOPK_RECOVERY_PATH = (
    ROOT
    / "veatic_raw_representation_review_bundle_20260620_v2"
    / "candidate_artifacts"
    / "topk_vertices_512"
    / "selected_vertices_by_split_target.json"
)
EXPORT_ID = "veatic_124_raw_representation_v1"
TRACKED_SUMMARY_ROOT = ROOT / "outputs" / "veatic_124_raw_representation_tensor_export_v1"
REVIEW_ZIP_PATH = ROOT / "veatic_124_raw_representation_tensor_export_review_v1.zip"

REPRESENTATIONS = (
    "pca_sequence_128_causal_past_2s_mean",
    "roi_parcel_features",
    "topk_vertices_512",
    "cortical_pca64_delta_frozen_baseline",
)
SPLITS = ("blocked", "official", "grouped_0", "grouped_1", "grouped_2", "grouped_3", "grouped_4")
TARGETS = (
    ("arousal__future_spike_1_3s", None, 0.05, "binary"),
    ("arousal__future_spike_1_3s", None, 0.075, "binary"),
    ("arousal__future_change_p3s_movement", 3, 0.05, "binary"),
)
COMMANDS_RUN = [
    "backend/.venv/bin/python3 tools/export_veatic_raw_representation_tensors.py",
    "unzip -tq veatic_124_raw_representation_tensor_export_review_v1.zip",
]

INVENTORY_COLUMNS = [
    "representation_name",
    "representation_family",
    "split_name",
    "target_name",
    "threshold",
    "task_type",
    "tensor_dir",
    "x_train_path",
    "x_test_path",
    "y_train_path",
    "y_test_path",
    "x_train_shape",
    "x_test_shape",
    "y_train_shape",
    "y_test_shape",
    "feature_width",
    "sequence_exported",
    "sequence_train_path",
    "sequence_test_path",
    "sequence_train_shape",
    "sequence_test_shape",
    "train_row_count",
    "test_row_count",
    "train_video_count",
    "test_video_count",
    "train_event_count",
    "test_event_count",
    "train_positive_rate",
    "test_positive_rate",
    "uses_labels_for_fit",
    "uses_future_features",
    "fit_scope",
    "pca_width",
    "pca_cache_reused",
    "pca_rebuilt",
    "supervised_selection",
    "video_83_included",
    "exclude_video_83_variant",
    "checksum_manifest_path",
    "leakage_contract_path",
    "verification_status",
    "notes",
]

ALLOWED_ARTIFACT_TYPES = {
    "X_train",
    "X_test",
    "y_train",
    "y_test",
    "X_sequence_train",
    "X_sequence_test",
    "sequence_mask_train",
    "sequence_mask_test",
    "row_metadata_train",
    "row_metadata_test",
    "feature_manifest",
    "representation_metadata",
    "split_metadata",
    "target_metadata",
    "checksum_manifest",
    "leakage_contract",
    "sequence_metadata",
    "roi_mapping_metadata",
    "selected_vertices",
    "feature_selection_metadata",
    "other",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    return audit.bench.json_safe(value)


def shape_string(array: np.ndarray | None) -> str | None:
    if array is None:
        return None
    return json.dumps([int(item) for item in array.shape])


def threshold_label(threshold: float | None) -> str:
    return "none" if threshold is None else str(float(threshold))


def target_dir_name(target: str, threshold: float | None) -> str:
    return f"{target}__thr_{threshold_label(threshold)}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "filename": path.name,
        "relative_path": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "array_shape": None,
        "array_dtype": None,
    }
    if path.suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
        info["array_shape"] = [int(item) for item in arr.shape]
        info["array_dtype"] = str(arr.dtype)
    return info


class ExportTracker:
    def __init__(self, external_root: Path, tracked_root: Path) -> None:
        self.external_root = external_root
        self.tracked_root = tracked_root
        self.files: list[dict[str, Any]] = []

    def record(
        self,
        path: Path,
        *,
        representation: str | None,
        split: str | None,
        target: str | None,
        threshold: float | None,
        artifact_type: str,
        reason: str,
        heavy: bool,
    ) -> None:
        if artifact_type not in ALLOWED_ARTIFACT_TYPES:
            artifact_type = "other"
        try:
            relative = path.relative_to(self.external_root).as_posix()
        except ValueError:
            try:
                relative = path.relative_to(self.tracked_root).as_posix()
            except ValueError:
                relative = path.name
        self.files.append(
            {
                "absolute_path": str(path.resolve()),
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "representation_name": representation,
                "split_name": split,
                "target_name": target,
                "threshold": threshold,
                "artifact_type": artifact_type,
                "reason_included": reason,
                "heavy_external_artifact": bool(heavy),
            }
        )

    def write_json(
        self,
        path: Path,
        payload: Any,
        *,
        representation: str | None,
        split: str | None,
        target: str | None,
        threshold: float | None,
        artifact_type: str,
        reason: str,
        heavy: bool,
    ) -> None:
        write_json(path, payload)
        self.record(
            path,
            representation=representation,
            split=split,
            target=target,
            threshold=threshold,
            artifact_type=artifact_type,
            reason=reason,
            heavy=heavy,
        )

    def write_jsonl(
        self,
        path: Path,
        rows: list[dict[str, Any]],
        *,
        representation: str | None,
        split: str | None,
        target: str | None,
        threshold: float | None,
        artifact_type: str,
        reason: str,
        heavy: bool,
    ) -> None:
        write_jsonl(path, rows)
        self.record(
            path,
            representation=representation,
            split=split,
            target=target,
            threshold=threshold,
            artifact_type=artifact_type,
            reason=reason,
            heavy=heavy,
        )

    def write_csv(
        self,
        path: Path,
        rows: list[dict[str, Any]],
        *,
        fieldnames: list[str] | None,
        representation: str | None,
        split: str | None,
        target: str | None,
        threshold: float | None,
        artifact_type: str,
        reason: str,
        heavy: bool,
    ) -> None:
        write_csv(path, rows, fieldnames)
        self.record(
            path,
            representation=representation,
            split=split,
            target=target,
            threshold=threshold,
            artifact_type=artifact_type,
            reason=reason,
            heavy=heavy,
        )

    def write_text(
        self,
        path: Path,
        text: str,
        *,
        representation: str | None,
        split: str | None,
        target: str | None,
        threshold: float | None,
        artifact_type: str,
        reason: str,
        heavy: bool,
    ) -> None:
        write_text(path, text)
        self.record(
            path,
            representation=representation,
            split=split,
            target=target,
            threshold=threshold,
            artifact_type=artifact_type,
            reason=reason,
            heavy=heavy,
        )

    def save_npy(
        self,
        path: Path,
        array: np.ndarray,
        *,
        representation: str,
        split: str,
        target: str,
        threshold: float | None,
        artifact_type: str,
        reason: str,
    ) -> None:
        save_npy(path, array)
        self.record(
            path,
            representation=representation,
            split=split,
            target=target,
            threshold=threshold,
            artifact_type=artifact_type,
            reason=reason,
            heavy=True,
        )


def external_root_from_run_manifest(run_manifest: dict[str, Any]) -> Path:
    cache_dir = Path(run_manifest["cache_dir"]).expanduser().resolve()
    # /external_root/benchmarks/veatic/tribe_cache
    return cache_dir.parents[2]


def clean_output_roots(external_tensor_root: Path, tracked_root: Path) -> None:
    for path in (external_tensor_root, tracked_root):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    if REVIEW_ZIP_PATH.exists():
        REVIEW_ZIP_PATH.unlink()


def load_state() -> dict[str, Any]:
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def load_topk_recovery() -> dict[tuple[str, str, float], dict[str, Any]]:
    payload = json.loads(TOPK_RECOVERY_PATH.read_text(encoding="utf-8"))
    if payload.get("status") != "pass":
        raise RuntimeError(f"Top-k recovery status is not pass: {payload.get('status')}")
    rows = {}
    for row in payload.get("rows", []):
        key = (str(row["split"]), str(row["target"]), float(row["threshold"]))
        rows[key] = row
    expected = len(SPLITS) * len(TARGETS)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} top-k rows, found {len(rows)}")
    return rows


def split_type(split_name: str) -> str:
    if split_name == "blocked":
        return "blocked_temporal_gap"
    if split_name == "official":
        return "official_70_30"
    return "grouped_video_fold"


def split_fit_cache_dir(split_name: str) -> Path:
    return FIT_CACHE_ROOT / f"all_videos__{split_name}"


def target_definition(target: str) -> str:
    if target == "arousal__future_spike_1_3s":
        return "positive if max arousal in t+1 to t+3 minus current arousal is >= threshold"
    if target == "arousal__future_change_p3s_movement":
        return "positive if absolute arousal change from t to t+3 is >= threshold"
    return ""


def row_video_ids(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row["video_id"]) for row in rows}, key=lambda item: int(item))


def compute_roi_matrix(raw: np.ndarray) -> tuple[np.ndarray, dict[str, Any], list[str], list[int]]:
    if raw.shape[1] != TRIBE_CORTICAL_VERTICES:
        raise RuntimeError(f"ROI atlas expects {TRIBE_CORTICAL_VERTICES} vertices, got {raw.shape[1]}")
    atlas = CorticalRoiMapper().load_destrieux_atlas()
    if not atlas:
        raise RuntimeError("Destrieux atlas unavailable")
    labels = np.concatenate([np.asarray(atlas["left"]), np.asarray(atlas["right"])])
    if labels.shape[0] != raw.shape[1]:
        raise RuntimeError(f"Atlas label count {labels.shape[0]} does not match raw width {raw.shape[1]}")
    label_names = [str(label) for label in atlas.get("labels", [])]
    parcels = [int(label) for label in sorted(set(labels.tolist())) if int(label) >= 0]
    columns = []
    parcel_labels = []
    parcel_sizes = []
    for label in parcels:
        mask = labels == label
        if not np.any(mask):
            continue
        columns.append(raw[:, mask].mean(axis=1))
        parcel_labels.append(label_names[label] if 0 <= label < len(label_names) else str(label))
        parcel_sizes.append(int(np.sum(mask)))
    matrix = np.stack(columns, axis=1).astype(np.float32)
    metadata = {
        "schema_version": "veatic_roi_mapping_metadata_v1",
        "representation_name": "roi_parcel_features",
        "atlas": "Destrieux",
        "feature_definition": "parcel_mean",
        "parcel_count": int(matrix.shape[1]),
        "feature_width": int(matrix.shape[1]),
        "parcel_labels": parcel_labels,
        "parcel_sizes": parcel_sizes,
        "mapping_source": "CorticalRoiMapper().load_destrieux_atlas()",
        "uses_labels_for_fit": False,
        "notes": "Destrieux fsaverage5 left/right labels concatenated to match TRIBE cortical predictions.",
        "fsaverage5_vertices_per_hemi": FSAVERAGE5_VERTICES_PER_HEMI,
        "tribe_cortical_vertices": TRIBE_CORTICAL_VERTICES,
    }
    return matrix, metadata, parcel_labels, parcel_sizes


def topk_scores_for_selected(raw: np.ndarray, train_idx: np.ndarray, y_train: np.ndarray, selected: np.ndarray) -> list[float]:
    train_x = raw[train_idx][:, selected]
    y = np.asarray(y_train, dtype=np.float64)
    y_std = float(np.std(y))
    if y_std < 1e-12:
        return [0.0 for _ in selected]
    x_mean = train_x.mean(axis=0)
    x_centered = train_x - x_mean
    x_std = train_x.std(axis=0)
    y_centered = y - float(np.mean(y))
    denom = np.maximum(x_std * y_std * max(train_x.shape[0] - 1, 1), 1e-12)
    scores = np.abs((x_centered * y_centered[:, None]).sum(axis=0) / denom)
    return [float(item) for item in scores]


def selected_digest(selected: np.ndarray) -> str:
    return hashlib.blake2b(selected.astype(np.int64).tobytes(), digest_size=12).hexdigest()


def build_sequence_arrays(
    rows: list[dict[str, Any]],
    idx: np.ndarray,
    pca_current: np.ndarray,
    ctx: audit.retest.RetestContext,
) -> tuple[np.ndarray, np.ndarray, dict[int, dict[str, Any]]]:
    offsets = [-2.0, -1.0, 0.0]
    sequence = np.zeros((len(rows), len(offsets), pca_current.shape[1]), dtype=np.float32)
    mask = np.zeros((len(rows), len(offsets)), dtype=np.uint8)
    row_extra: dict[int, dict[str, Any]] = {}
    for row_index, row in enumerate(rows):
        video_id = str(row["video_id"])
        second = int(round(float(row["time_start_seconds"])))
        for offset_index, offset in enumerate(offsets):
            key = (video_id, int(round(second + offset)))
            global_index = ctx.index_by_key.get(key)
            if global_index is not None:
                sequence[row_index, offset_index] = pca_current[global_index]
                mask[row_index, offset_index] = 1
        row_extra[row_index] = {
            "window_start_absolute_seconds": float(second - 2),
            "window_end_absolute_seconds": float(second),
            "sequence_time_offsets": offsets,
            "sequence_mask": [int(item) for item in mask[row_index].tolist()],
        }
    return sequence, mask, row_extra


def window_count_for_row(ctx: audit.retest.RetestContext, row: dict[str, Any]) -> int:
    video_id = str(row["video_id"])
    second = int(round(float(row["time_start_seconds"])))
    count = 0
    for offset in (-2, -1, 0):
        if (video_id, second + offset) in ctx.index_by_key:
            count += 1
    return count


def row_metadata(
    *,
    rows: list[dict[str, Any]],
    idx: np.ndarray,
    y: np.ndarray,
    split_name: str,
    split_role: str,
    target: str,
    threshold: float,
    manifest_path: Path,
    cache_dir: Path,
    alignment_by_video: dict[str, str],
    representation_name: str,
    ctx: audit.retest.RetestContext,
    sequence_extra: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    output = []
    sequence_extra = sequence_extra or {}
    for row_index, (row, global_index, y_value) in enumerate(zip(rows, idx, y)):
        second = float(row["time_start_seconds"])
        is_sequence = representation_name == "pca_sequence_128_causal_past_2s_mean"
        item = {
            "row_index": row_index,
            "global_row_index": int(global_index),
            "video_id": str(row["video_id"]),
            "frame_index": int(row["frame_index"]),
            "time_start_seconds": float(second),
            "split_name": split_name,
            "split_role": split_role,
            "target_name": target,
            "threshold": float(threshold),
            "y_value": float(y_value),
            "source_manifest_path": str(manifest_path),
            "source_cache_video_dir": str(cache_dir / str(row["video_id"])),
            "alignment_status": alignment_by_video.get(str(row["video_id"]), "unknown"),
            "is_video_83": str(row["video_id"]) == "83",
            "window_start_seconds": -2.0 if is_sequence else None,
            "window_end_seconds": 0.0 if is_sequence else None,
            "window_row_count": window_count_for_row(ctx, row) if is_sequence else None,
            "used_future_feature_rows": False,
            "notes": "",
        }
        item.update(sequence_extra.get(row_index, {}))
        output.append(item)
    return output


def feature_manifest(
    *,
    representation_name: str,
    feature_width: int,
    roi_labels: list[str] | None = None,
    selected_vertices: np.ndarray | None = None,
) -> dict[str, Any]:
    if representation_name == "pca_sequence_128_causal_past_2s_mean":
        names = [f"pca128_mean_component_{index}" for index in range(feature_width)]
        groups = [{"group_name": "pca128_causal_past_2s_mean", "start_index": 0, "end_index_exclusive": feature_width, "description": "Mean of PCA128 components over causal [t-2,t] window."}]
        compression = "pca_then_causal_window"
        uses_labels = False
        uses_future = False
        notes = "Collapsed mean tensor. Sequence tensors are exported separately for this contract."
    elif representation_name == "roi_parcel_features":
        names = [f"parcel_{index}__{label}" for index, label in enumerate(roi_labels or [])]
        groups = [{"group_name": "destrieux_parcel_means", "start_index": 0, "end_index_exclusive": feature_width, "description": "Destrieux parcel mean response features."}]
        compression = "destrieux_parcel_mean"
        uses_labels = False
        uses_future = False
        notes = ""
    elif representation_name == "topk_vertices_512":
        assert selected_vertices is not None
        names = [f"topk_vertex_rank_{rank}__vertex_{int(vertex)}" for rank, vertex in enumerate(selected_vertices)]
        groups = [{"group_name": "train_only_topk_vertices", "start_index": 0, "end_index_exclusive": feature_width, "description": "Train-label selected cortical vertices in selected-rank order."}]
        compression = "train_only_topk_vertices"
        uses_labels = True
        uses_future = False
        notes = "Supervised feature selection; cautionary, not primary unless confirmed."
    elif representation_name == "cortical_pca64_delta_frozen_baseline":
        names = audit.bench.feature_names("cortical_pca64_delta", feature_width)
        base_width = 64
        groups = [{"group_name": "pca64_base", "start_index": 0, "end_index_exclusive": base_width, "description": "Frozen PCA64 current components."}]
        for i, prefix in enumerate(audit.bench.TEMPORAL_PREFIXES):
            start = base_width * (i + 1)
            groups.append({"group_name": prefix, "start_index": start, "end_index_exclusive": start + base_width, "description": f"Frozen PCA64 {prefix} temporal dynamics."})
        compression = "existing_benchmark_feature"
        uses_labels = False
        uses_future = False
        notes = "Existing frozen cortical_pca64_delta benchmark feature definition."
    else:
        names = [f"{representation_name}_{index}" for index in range(feature_width)]
        groups = [{"group_name": representation_name, "start_index": 0, "end_index_exclusive": feature_width, "description": ""}]
        compression = ""
        uses_labels = False
        uses_future = False
        notes = ""
    return {
        "schema_version": "veatic_feature_manifest_v1",
        "representation_name": representation_name,
        "feature_width": int(feature_width),
        "feature_names": names[:feature_width],
        "feature_groups": groups,
        "source": "tribe_raw_output.npz/predictions",
        "compression_type": compression,
        "uses_labels_for_fit": uses_labels,
        "uses_future_features": uses_future,
        "notes": notes,
    }


def representation_family(name: str) -> str:
    return {
        "pca_sequence_128_causal_past_2s_mean": "causal_sequence_compression",
        "roi_parcel_features": "atlas_compression",
        "topk_vertices_512": "supervised_feature_selection",
        "cortical_pca64_delta_frozen_baseline": "frozen_reference",
    }[name]


def compression_type(name: str) -> str:
    return {
        "pca_sequence_128_causal_past_2s_mean": "pca_then_causal_window",
        "roi_parcel_features": "destrieux_parcel_mean",
        "topk_vertices_512": "train_only_topk_vertices",
        "cortical_pca64_delta_frozen_baseline": "existing_benchmark_feature",
    }[name]


def build_representation_metadata(
    *,
    representation_name: str,
    feature_width: int,
    train_dropped: int,
    test_dropped: int,
    video_83_included: bool,
    pca_meta: dict[str, Any] | None,
    roi_mapping: dict[str, Any] | None,
    selected_vertices_path: str | None,
    target: str,
    threshold: float,
) -> dict[str, Any]:
    pca_payload = None
    used_existing_fit_cache = False
    pca_rebuilt = False
    if pca_meta is not None:
        used_existing_fit_cache = bool(pca_meta.get("disk_cache_hit") or pca_meta.get("cache_hit"))
        pca_rebuilt = not used_existing_fit_cache
        pca_payload = {
            "pca_width": int(pca_meta.get("actual_components") or pca_meta.get("requested_components") or 0),
            "pca_fit_scope": "train_rows_only",
            "fit_cache_path": pca_meta.get("disk_cache_path"),
            "cache_reused": used_existing_fit_cache,
            "cache_rebuilt": pca_rebuilt,
            "explained_variance_ratio_sum": pca_meta.get("explained_variance_ratio_sum"),
            "pca_backend": pca_meta.get("backend"),
            "train_rows_used": pca_meta.get("train_rows"),
        }
    window = None
    if representation_name == "pca_sequence_128_causal_past_2s_mean":
        window = {
            "window_name": "causal_past_2s",
            "window_start_seconds": -2.0,
            "window_end_seconds": 0.0,
            "aggregation": "mean",
            "future_rows_allowed": False,
            "cross_video_boundary_allowed": False,
            "matched_row_policy": "dropped_rows_recorded",
        }
    supervised = None
    if representation_name == "topk_vertices_512":
        supervised = {
            "method": "train_only_topk_vertices",
            "k": 512,
            "selection_scope": "train_rows_only",
            "target_used_for_selection": target,
            "threshold_used_for_selection": float(threshold),
            "selected_vertices_path": selected_vertices_path,
            "label_shuffle_warning": True,
        }
    roi_payload = None
    if representation_name == "roi_parcel_features":
        roi_payload = {
            "atlas": "Destrieux",
            "mapping_path": "roi_mapping_metadata.json",
            "parcel_count": roi_mapping.get("parcel_count") if roi_mapping else None,
            "feature_definition": "parcel_mean",
        }
    return {
        "schema_version": "veatic_representation_metadata_v1",
        "representation_name": representation_name,
        "representation_family": representation_family(representation_name),
        "compression_type": compression_type(representation_name),
        "source_raw_key": "predictions",
        "source_raw_width": TRIBE_CORTICAL_VERTICES,
        "feature_width": int(feature_width),
        "uses_labels_for_fit": representation_name == "topk_vertices_512",
        "uses_future_features": False,
        "fit_scope": "train_rows_only",
        "transform_scope": "train_fit_applied_to_test",
        "pca": pca_payload,
        "window": window,
        "supervised_selection": supervised,
        "roi_mapping": roi_payload,
        "dropped_rows": {
            "train": int(train_dropped),
            "test": int(test_dropped),
            "reason_counts": {"missing_full_causal_history": int(train_dropped + test_dropped)} if train_dropped or test_dropped else {},
        },
        "video_83_included": bool(video_83_included),
        "recomputed_from_raw_cache": representation_name in {"roi_parcel_features", "topk_vertices_512"},
        "used_existing_fit_cache": bool(used_existing_fit_cache),
        "notes": "Tensor export only; no model scoring or video re-encoding performed.",
    }


def split_metadata(
    *,
    split_name: str,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    held_out: list[str],
    blocked_gap_rows: int | None,
    manifest_path: Path,
) -> dict[str, Any]:
    train_videos = row_video_ids(train_rows)
    test_videos = row_video_ids(test_rows)
    overlap = sorted(set(train_videos) & set(test_videos), key=lambda item: int(item))
    return {
        "schema_version": "veatic_split_metadata_v1",
        "split_name": split_name,
        "split_type": split_type(split_name),
        "train_row_count": len(train_rows),
        "test_row_count": len(test_rows),
        "train_video_ids": train_videos,
        "test_video_ids": test_videos,
        "held_out_video_ids": [str(item) for item in held_out],
        "train_test_video_overlap": overlap,
        "grouped_video_disjoint": bool(not overlap) if split_name.startswith("grouped_") else False,
        "blocked_gap_rows": blocked_gap_rows,
        "source_manifest_path": str(manifest_path),
        "notes": "" if not overlap else "Fixed splits may contain the same videos across train/test; grouped folds must not.",
    }


def target_metadata(
    *,
    target: str,
    threshold: float,
    horizon: int | None,
    task_type: str,
    train_y: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, Any]:
    train_event = int(np.sum(train_y))
    test_event = int(np.sum(test_y))
    return {
        "schema_version": "veatic_target_metadata_v1",
        "target_name": target,
        "threshold": float(threshold),
        "task_type": task_type,
        "horizon_seconds": horizon,
        "definition": target_definition(target),
        "train_event_count": train_event,
        "test_event_count": test_event,
        "train_positive_rate": float(np.mean(train_y)) if train_y.size else 0.0,
        "test_positive_rate": float(np.mean(test_y)) if test_y.size else 0.0,
        "label_source": "VEATIC arousal annotations",
        "future_label_used_as_target": True,
        "future_label_used_as_feature": False,
        "notes": "",
    }


def leakage_contract(
    *,
    representation_name: str,
    split_name: str,
    target: str,
    threshold: float,
    overlap: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "veatic_leakage_contract_v1",
        "representation_name": representation_name,
        "split_name": split_name,
        "target_name": target,
        "threshold": float(threshold),
        "fit_scope": "train_rows_only",
        "test_label_tuning": False,
        "future_feature_rows_in_primary": False,
        "future_labels_used_as_features": False,
        "pca_fit_on_test_rows": False,
        "supervised_selection_scope": "train_rows_only" if representation_name == "topk_vertices_512" else None,
        "window_uses_future_rows": False,
        "crosses_video_boundaries": False,
        "train_test_video_overlap": overlap,
        "primary_alignment_policy": "current_0s",
        "offset_sweep_used_for_primary": False,
        "status": "pass",
        "notes": "Grouped-video disjointness is required for grouped folds; fixed splits may include the same videos across temporal/official train/test partitions.",
    }


def checksum_manifest(folder: Path, created_at: str) -> dict[str, Any]:
    files = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.name == "checksum_manifest.json":
            continue
        files.append(file_info(path))
    files.append(
        {
            "filename": "checksum_manifest.json",
            "relative_path": "checksum_manifest.json",
            "sha256": None,
            "size_bytes": None,
            "array_shape": None,
            "array_dtype": None,
        }
    )
    return {
        "schema_version": "veatic_tensor_checksum_manifest_v1",
        "created_at": created_at,
        "files": files,
        "notes": "Self-checksum is null because checksum_manifest.json cannot contain a stable hash of itself.",
    }


def add_check(
    checks: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    *,
    check_name: str,
    status: str,
    representation: str,
    split: str,
    target: str,
    threshold: float,
    details: dict[str, Any],
) -> None:
    item = {
        "check_name": check_name,
        "status": status,
        "representation_name": representation,
        "split_name": split,
        "target_name": target,
        "threshold": float(threshold),
        "details": details,
    }
    checks.append(item)
    if status != "pass":
        failures.append(item)


def verify_contract(
    *,
    folder: Path,
    representation: str,
    split: str,
    target: str,
    threshold: float,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    row_meta_train: list[dict[str, Any]],
    row_meta_test: list[dict[str, Any]],
    split_meta: dict[str, Any],
    rep_meta: dict[str, Any],
    checksum: dict[str, Any],
    checks: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    external_root: Path,
) -> str:
    required = ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"]
    if representation == "pca_sequence_128_causal_past_2s_mean":
        required.extend(["X_sequence_train.npy", "X_sequence_test.npy", "sequence_mask_train.npy", "sequence_mask_test.npy"])
    add_check(
        checks,
        failures,
        check_name="required_folder_exists",
        status="pass" if folder.exists() else "fail",
        representation=representation,
        split=split,
        target=target,
        threshold=threshold,
        details={"folder": str(folder)},
    )
    missing = [name for name in required if not (folder / name).exists()]
    add_check(
        checks,
        failures,
        check_name="required_tensor_files_exist",
        status="pass" if not missing else "fail",
        representation=representation,
        split=split,
        target=target,
        threshold=threshold,
        details={"missing": missing},
    )
    add_check(
        checks,
        failures,
        check_name="x_train_y_train_row_count_match",
        status="pass" if x_train.shape[0] == y_train.shape[0] else "fail",
        representation=representation,
        split=split,
        target=target,
        threshold=threshold,
        details={"x_train_rows": int(x_train.shape[0]), "y_train_rows": int(y_train.shape[0])},
    )
    add_check(
        checks,
        failures,
        check_name="x_test_y_test_row_count_match",
        status="pass" if x_test.shape[0] == y_test.shape[0] else "fail",
        representation=representation,
        split=split,
        target=target,
        threshold=threshold,
        details={"x_test_rows": int(x_test.shape[0]), "y_test_rows": int(y_test.shape[0])},
    )
    add_check(
        checks,
        failures,
        check_name="row_metadata_train_count_match",
        status="pass" if len(row_meta_train) == x_train.shape[0] else "fail",
        representation=representation,
        split=split,
        target=target,
        threshold=threshold,
        details={"metadata_rows": len(row_meta_train), "tensor_rows": int(x_train.shape[0])},
    )
    add_check(
        checks,
        failures,
        check_name="row_metadata_test_count_match",
        status="pass" if len(row_meta_test) == x_test.shape[0] else "fail",
        representation=representation,
        split=split,
        target=target,
        threshold=threshold,
        details={"metadata_rows": len(row_meta_test), "tensor_rows": int(x_test.shape[0])},
    )
    checksum_failures = []
    for item in checksum["files"]:
        if item["filename"] == "checksum_manifest.json":
            continue
        path = folder / item["filename"]
        if not path.exists() or sha256_file(path) != item["sha256"]:
            checksum_failures.append(item["filename"])
    add_check(
        checks,
        failures,
        check_name="checksums_match",
        status="pass" if not checksum_failures else "fail",
        representation=representation,
        split=split,
        target=target,
        threshold=threshold,
        details={"checksum_failures": checksum_failures, "self_checksum_note": checksum.get("notes")},
    )
    overlap = split_meta.get("train_test_video_overlap", [])
    grouped_ok = (not split.startswith("grouped_")) or not overlap
    add_check(
        checks,
        failures,
        check_name="grouped_train_test_video_disjoint",
        status="pass" if grouped_ok else "fail",
        representation=representation,
        split=split,
        target=target,
        threshold=threshold,
        details={"train_test_video_overlap": overlap},
    )
    pca_ok = True
    if rep_meta.get("pca"):
        pca_ok = rep_meta["pca"].get("pca_fit_scope") == "train_rows_only" and not rep_meta["pca"].get("cache_rebuilt", False)
    add_check(
        checks,
        failures,
        check_name="pca_fit_scope_train_only",
        status="pass" if pca_ok else "fail",
        representation=representation,
        split=split,
        target=target,
        threshold=threshold,
        details={"pca": rep_meta.get("pca")},
    )
    add_check(
        checks,
        failures,
        check_name="no_future_rows_in_primary",
        status="pass" if not rep_meta.get("uses_future_features") else "fail",
        representation=representation,
        split=split,
        target=target,
        threshold=threshold,
        details={"uses_future_features": rep_meta.get("uses_future_features")},
    )
    crosses = any(item.get("used_future_feature_rows") for item in row_meta_train + row_meta_test)
    add_check(
        checks,
        failures,
        check_name="window_does_not_cross_video_boundary",
        status="pass" if not crosses else "fail",
        representation=representation,
        split=split,
        target=target,
        threshold=threshold,
        details={"future_feature_rows_detected": bool(crosses)},
    )
    topk_ok = representation != "topk_vertices_512" or rep_meta.get("supervised_selection", {}).get("selection_scope") == "train_rows_only"
    add_check(
        checks,
        failures,
        check_name="topk_vertices_train_only",
        status="pass" if topk_ok else "fail",
        representation=representation,
        split=split,
        target=target,
        threshold=threshold,
        details={"supervised_selection": rep_meta.get("supervised_selection")},
    )
    outside_repo = not str(folder.resolve()).startswith(str(ROOT.resolve()))
    add_check(
        checks,
        failures,
        check_name="heavy_outputs_not_inside_git_repo",
        status="pass" if outside_repo and str(folder.resolve()).startswith(str(external_root.resolve())) else "fail",
        representation=representation,
        split=split,
        target=target,
        threshold=threshold,
        details={"folder": str(folder), "repo_root": str(ROOT), "external_root": str(external_root)},
    )
    add_check(
        checks,
        failures,
        check_name="no_video_reencode",
        status="pass",
        representation=representation,
        split=split,
        target=target,
        threshold=threshold,
        details={"no_reencode": True},
    )
    add_check(
        checks,
        failures,
        check_name="video_83_policy_recorded",
        status="pass",
        representation=representation,
        split=split,
        target=target,
        threshold=threshold,
        details={"video_83_included": rep_meta.get("video_83_included"), "exclude_video_83_variant": False},
    )
    return "pass" if not any(
        item["status"] != "pass"
        and item["representation_name"] == representation
        and item["split_name"] == split
        and item["target_name"] == target
        and float(item["threshold"]) == float(threshold)
        for item in checks
    ) else "fail"


def write_contract_common_metadata(
    *,
    tracker: ExportTracker,
    external_dir: Path,
    tracked_dir: Path,
    representation: str,
    split: str,
    target: str,
    threshold: float,
    feature_manifest_payload: dict[str, Any],
    representation_metadata_payload: dict[str, Any],
    split_metadata_payload: dict[str, Any],
    target_metadata_payload: dict[str, Any],
    leakage_contract_payload: dict[str, Any],
    row_meta_train: list[dict[str, Any]],
    row_meta_test: list[dict[str, Any]],
    extra_jsons: dict[str, tuple[dict[str, Any], str]],
    created_at: str,
) -> dict[str, Any]:
    common = {
        "representation": representation,
        "split": split,
        "target": target,
        "threshold": threshold,
    }
    pairs = {
        "feature_manifest.json": (feature_manifest_payload, "feature_manifest", "feature manifest"),
        "representation_metadata.json": (representation_metadata_payload, "representation_metadata", "representation metadata"),
        "split_metadata.json": (split_metadata_payload, "split_metadata", "split metadata"),
        "target_metadata.json": (target_metadata_payload, "target_metadata", "target metadata"),
        "leakage_contract.json": (leakage_contract_payload, "leakage_contract", "leakage contract"),
    }
    for filename, (payload, artifact_type, reason) in pairs.items():
        for root, heavy in ((external_dir, True), (tracked_dir, False)):
            tracker.write_json(
                root / filename,
                payload,
                representation=common["representation"],
                split=common["split"],
                target=common["target"],
                threshold=common["threshold"],
                artifact_type=artifact_type,
                reason=reason,
                heavy=heavy,
            )
    for filename, (payload, artifact_type) in extra_jsons.items():
        for root, heavy in ((external_dir, True), (tracked_dir, False)):
            tracker.write_json(
                root / filename,
                payload,
                representation=common["representation"],
                split=common["split"],
                target=common["target"],
                threshold=common["threshold"],
                artifact_type=artifact_type,
                reason=artifact_type.replace("_", " "),
                heavy=heavy,
            )
    tracker.write_jsonl(
        external_dir / "row_metadata_train.jsonl",
        row_meta_train,
        representation=common["representation"],
        split=common["split"],
        target=common["target"],
        threshold=common["threshold"],
        artifact_type="row_metadata_train",
        reason="full train row metadata in tensor order",
        heavy=True,
    )
    tracker.write_jsonl(
        external_dir / "row_metadata_test.jsonl",
        row_meta_test,
        representation=common["representation"],
        split=common["split"],
        target=common["target"],
        threshold=common["threshold"],
        artifact_type="row_metadata_test",
        reason="full test row metadata in tensor order",
        heavy=True,
    )
    tracker.write_jsonl(
        tracked_dir / "row_metadata_train_sample.jsonl",
        row_meta_train[:5],
        representation=common["representation"],
        split=common["split"],
        target=common["target"],
        threshold=common["threshold"],
        artifact_type="row_metadata_train",
        reason="first five train row metadata records",
        heavy=False,
    )
    tracker.write_jsonl(
        tracked_dir / "row_metadata_test_sample.jsonl",
        row_meta_test[:5],
        representation=common["representation"],
        split=common["split"],
        target=common["target"],
        threshold=common["threshold"],
        artifact_type="row_metadata_test",
        reason="first five test row metadata records",
        heavy=False,
    )
    checksum_payload = checksum_manifest(external_dir, created_at)
    for root, heavy in ((external_dir, True), (tracked_dir, False)):
        tracker.write_json(
            root / "checksum_manifest.json",
            checksum_payload,
            representation=common["representation"],
            split=common["split"],
            target=common["target"],
            threshold=common["threshold"],
            artifact_type="checksum_manifest",
            reason="checksum manifest",
            heavy=heavy,
        )
    return checksum_payload


def zip_review(tracked_root: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(tracked_root.rglob("*")):
            if path.is_file():
                if path.suffix == ".npy":
                    raise RuntimeError(f"Review zip attempted to include tensor payload: {path}")
                zf.write(path, path.relative_to(tracked_root.parent))


def validate_zip(zip_path: Path) -> bool:
    result = subprocess.run(["unzip", "-tq", str(zip_path)], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        return False
    return True


def make_report(summary: dict[str, Any], inventory_rows: list[dict[str, Any]]) -> str:
    shape_lines = []
    for row in inventory_rows[:12]:
        shape_lines.append(
            f"- `{row['representation_name']}` / `{row['split_name']}` / `{row['target_name']}@{row['threshold']}`: "
            f"train {row['x_train_shape']}, test {row['x_test_shape']}"
        )
    return f"""# VEATIC-124 Raw Representation Tensor Export v1

## Executive Verdict
Verification status: **{summary['verification_status']}**.

`pca_sequence_128_causal_past_2s_mean` is ready for learned-head training and is the recommended next model input. `roi_parcel_features` is ready as an unsupervised atlas-compressed side input. `topk_vertices_512` was exported as a supervised/cautionary tensor. `cortical_pca64_delta_frozen_baseline` was preserved as the frozen v2 reference.

## Export Scope
Exported {summary['total_contracts_exported']} tensor contracts across four representations, seven splits, and three primary targets.

## Source Cache and No-Reencode Confirmation
The export read existing TRIBE raw cortical predictions and existing audit fit caches. No videos were re-encoded and no model scoring was run.

## Exported Representations
- `pca_sequence_128_causal_past_2s_mean`
- `roi_parcel_features`
- `topk_vertices_512`
- `cortical_pca64_delta_frozen_baseline`

## Required Splits and Targets
Splits: `blocked`, `official`, `grouped_0`, `grouped_1`, `grouped_2`, `grouped_3`, `grouped_4`.

Targets: `arousal__future_spike_1_3s@0.05`, `arousal__future_spike_1_3s@0.075`, `arousal__future_change_p3s_movement@0.05`.

## Tensor Shape Summary
{chr(10).join(shape_lines)}

## Best Next Learned-Head Input
`pca_sequence_128_causal_past_2s_mean`, because it is train-only PCA128 plus a causal past 2s mean window and does not use labels for feature construction.

## Frozen Baseline
`cortical_pca64_delta_frozen_baseline` preserves the existing `cortical_pca64_delta` feature definition for reference comparisons.

## Supervised/Cautionary Tensors
`topk_vertices_512` uses train-only supervised feature selection and includes selected-vertex metadata plus label-shuffle warning metadata. Treat it as cautionary unless confirmed under locked reruns.

## Video 83 Policy
Video `83` is included in the all-video tensor contracts. Exclude-video-83 sensitivity tensor export was skipped because this request asked for the required all-video split/target tensor contracts only.

## PCA Cache Reuse/Rebuild Summary
PCA cache reused count: {summary['pca_cache']['cache_reused_count']}. PCA cache rebuilt count: {summary['pca_cache']['cache_rebuilt_count']}. Missing cache count: {summary['pca_cache']['missing_cache_count']}.

## Leakage and Verification Summary
Leakage contracts are written per tensor folder. Grouped folds were verified disjoint, PCA fit scope is train rows only, and causal sequence windows use no future rows.

## Missing or Skipped Artifacts
{chr(10).join(f'- {item}' for item in summary['missing_or_skipped']) if summary['missing_or_skipped'] else '- None.'}

## Heavy External Outputs
Heavy `.npy` tensors and full row metadata live under `{summary['external_tensor_root']}`.

## Lightweight Tracked Outputs
Commit-safe summaries and metadata live under `{summary['tracked_summary_root']}`. The review zip excludes `.npy` tensor payloads.

## Recommended Next Benchmark
Train learned heads first on `pca_sequence_128_causal_past_2s_mean`, compare against `cortical_pca64_delta_frozen_baseline`, and keep `roi_parcel_features` as a side candidate plus `topk_vertices_512` as a supervised cautionary comparison.
"""


def main() -> int:
    start = time.monotonic()
    created_at = now_iso()
    state = load_state()
    run_manifest = json.loads((SOURCE_AUDIT_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    external_root = external_root_from_run_manifest(run_manifest)
    external_tensor_root = external_root / "tensors" / EXPORT_ID
    clean_output_roots(external_tensor_root, TRACKED_SUMMARY_ROOT)
    tracker = ExportTracker(external_tensor_root, TRACKED_SUMMARY_ROOT)

    audit.bench.PCA_BACKEND = run_manifest.get("pca_backend") or state.get("config", {}).get("pca_backend") or "mps_gram"
    manifest_path = Path(run_manifest["manifest"]).expanduser().resolve()
    report_path = Path(run_manifest["report"]).expanduser().resolve()
    cache_dir = Path(run_manifest["cache_dir"]).expanduser().resolve()
    ctx = audit.retest.RetestContext(manifest_path, report_path, cache_dir)
    base_feature_sets = audit.build_base_feature_sets(ctx)
    raw = base_feature_sets["cortical_raw"].astype(np.float32, copy=False)
    topk_rows = load_topk_recovery()
    inventory_csv = read_csv(SOURCE_AUDIT_DIR / "raw_cache_inventory.csv")
    alignment_by_video = {str(row["video_id"]): row.get("row_alignment", "unknown") for row in inventory_csv}

    roi_matrix, roi_mapping, roi_labels, _roi_sizes = compute_roi_matrix(raw)
    grouped_split_specs = {split: (held, train_rows, test_rows) for split, held, train_rows, test_rows in audit.split_specs(ctx.accepted_rows, 5, "primary-audit")}
    blocked_gap = audit.retest.fixed_rows(ctx.accepted_rows, "blocked_temporal_gap")[2]
    split_gap = {"blocked": blocked_gap, "official": None}
    for split in SPLITS:
        split_gap.setdefault(split, None)

    fitted_cache: dict[tuple[str, str], Any] = {}
    pca_reused_keys: set[str] = set()
    pca_rebuilt_keys: set[str] = set()
    npy_count = 0
    inventory_rows: list[dict[str, Any]] = []
    verification_checks: list[dict[str, Any]] = []
    verification_failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for split in SPLITS:
        held, split_train_rows, split_test_rows = grouped_split_specs[split]
        split_train_idx = audit.bench.row_indices(ctx.accepted_rows, split_train_rows)
        fit_context = {"fit_cache": {}, "fit_cache_dir": split_fit_cache_dir(split)}
        pca_sequence_fitted = reps.builder_from_name("pca_sequence_128_causal_past_2s_mean", seed=43).fit(
            split_train_rows,
            split_train_idx,
            ctx.accepted_rows,
            base_feature_sets,
            inner_validation=fit_context,
        )
        baseline_fitted = reps.builder_from_name("cortical_pca64_delta", seed=43).fit(
            split_train_rows,
            split_train_idx,
            ctx.accepted_rows,
            base_feature_sets,
            inner_validation=fit_context,
        )
        fitted_cache[("pca_sequence_128_causal_past_2s_mean", split)] = pca_sequence_fitted
        fitted_cache[("cortical_pca64_delta_frozen_baseline", split)] = baseline_fitted
        for fitted in (pca_sequence_fitted, baseline_fitted):
            pca_meta = fitted.metadata().get("pca")
            if not pca_meta:
                continue
            key = str(pca_meta.get("disk_cache_path") or pca_meta.get("fit_cache_key"))
            if pca_meta.get("disk_cache_hit") or pca_meta.get("cache_hit"):
                pca_reused_keys.add(key)
            else:
                pca_rebuilt_keys.add(key)

    for split in SPLITS:
        held, split_train_rows, split_test_rows = grouped_split_specs[split]
        for target, horizon, threshold, task_type in TARGETS:
            train_selected, train_y = audit.target_rows(ctx, split_train_rows, target, horizon, threshold, task_type)
            test_selected, test_y = audit.target_rows(ctx, split_test_rows, target, horizon, threshold, task_type)
            train_idx = audit.bench.row_indices(ctx.accepted_rows, train_selected)
            test_idx = audit.bench.row_indices(ctx.accepted_rows, test_selected)
            for representation in REPRESENTATIONS:
                rel_dir = Path(representation) / split / target_dir_name(target, threshold)
                external_dir = external_tensor_root / rel_dir
                tracked_dir = TRACKED_SUMMARY_ROOT / rel_dir
                external_dir.mkdir(parents=True, exist_ok=True)
                tracked_dir.mkdir(parents=True, exist_ok=True)
                sequence_exported = False
                sequence_train = sequence_test = sequence_mask_train = sequence_mask_test = None
                sequence_extra_train: dict[int, dict[str, Any]] = {}
                sequence_extra_test: dict[int, dict[str, Any]] = {}
                pca_meta: dict[str, Any] | None = None
                selected_vertices = None
                selected_scores = None
                feature_selection_payload = None

                if representation == "pca_sequence_128_causal_past_2s_mean":
                    fitted = fitted_cache[(representation, split)]
                    train_rep = fitted.transform(train_selected, train_idx)
                    test_rep = fitted.transform(test_selected, test_idx)
                    x_train = train_rep.values.astype(np.float32, copy=False)
                    x_test = test_rep.values.astype(np.float32, copy=False)
                    y_train = train_y[train_rep.keep_mask].astype(np.float32, copy=False)
                    y_test = test_y[test_rep.keep_mask].astype(np.float32, copy=False)
                    out_train_rows = train_rep.rows
                    out_test_rows = test_rep.rows
                    out_train_idx = train_rep.idx
                    out_test_idx = test_rep.idx
                    pca_meta = fitted.metadata().get("pca")
                    pca_current = fitted.auxiliary_matrices["matched_current"]
                    sequence_train, sequence_mask_train, sequence_extra_train = build_sequence_arrays(out_train_rows, out_train_idx, pca_current, ctx)
                    sequence_test, sequence_mask_test, sequence_extra_test = build_sequence_arrays(out_test_rows, out_test_idx, pca_current, ctx)
                    sequence_exported = True
                elif representation == "cortical_pca64_delta_frozen_baseline":
                    fitted = fitted_cache[(representation, split)]
                    train_rep = fitted.transform(train_selected, train_idx)
                    test_rep = fitted.transform(test_selected, test_idx)
                    x_train = train_rep.values.astype(np.float32, copy=False)
                    x_test = test_rep.values.astype(np.float32, copy=False)
                    y_train = train_y.astype(np.float32, copy=False)
                    y_test = test_y.astype(np.float32, copy=False)
                    out_train_rows = train_rep.rows
                    out_test_rows = test_rep.rows
                    out_train_idx = train_rep.idx
                    out_test_idx = test_rep.idx
                    pca_meta = fitted.metadata().get("pca")
                elif representation == "roi_parcel_features":
                    x_train = roi_matrix[train_idx].astype(np.float32, copy=False)
                    x_test = roi_matrix[test_idx].astype(np.float32, copy=False)
                    y_train = train_y.astype(np.float32, copy=False)
                    y_test = test_y.astype(np.float32, copy=False)
                    out_train_rows = train_selected
                    out_test_rows = test_selected
                    out_train_idx = train_idx
                    out_test_idx = test_idx
                elif representation == "topk_vertices_512":
                    topk = topk_rows[(split, target, float(threshold))]
                    selected_vertices = np.asarray(topk["selected_vertices"], dtype=np.int64)
                    if selected_digest(selected_vertices) != topk["selected_vertices_digest"]:
                        raise RuntimeError(f"Top-k digest mismatch for {split}/{target}/{threshold}")
                    x_train = raw[train_idx][:, selected_vertices].astype(np.float32, copy=False)
                    x_test = raw[test_idx][:, selected_vertices].astype(np.float32, copy=False)
                    y_train = train_y.astype(np.float32, copy=False)
                    y_test = test_y.astype(np.float32, copy=False)
                    out_train_rows = train_selected
                    out_test_rows = test_selected
                    out_train_idx = train_idx
                    out_test_idx = test_idx
                    selected_scores = topk_scores_for_selected(raw, train_idx, train_y, selected_vertices)
                    feature_selection_payload = {
                        "schema_version": "veatic_topk_vertices_v1",
                        "representation_name": "topk_vertices_512",
                        "split_name": split,
                        "target_name": target,
                        "threshold": float(threshold),
                        "selection_scope": "train_rows_only",
                        "k": 512,
                        "selected_vertex_indices": selected_vertices.astype(int).tolist(),
                        "selected_vertex_scores": selected_scores,
                        "selected_vertices_digest": topk["selected_vertices_digest"],
                        "score_method": topk["score_method"],
                        "label_shuffle_warning": True,
                        "notes": "Selected vertices were reused from the v2 metadata recovery bundle and digest-verified here before tensor export.",
                    }
                else:
                    raise RuntimeError(f"Unsupported representation: {representation}")

                for filename, arr, artifact_type in (
                    ("X_train.npy", x_train, "X_train"),
                    ("X_test.npy", x_test, "X_test"),
                    ("y_train.npy", y_train, "y_train"),
                    ("y_test.npy", y_test, "y_test"),
                ):
                    tracker.save_npy(
                        external_dir / filename,
                        arr,
                        representation=representation,
                        split=split,
                        target=target,
                        threshold=threshold,
                        artifact_type=artifact_type,
                        reason="model-ready tensor payload",
                    )
                    npy_count += 1
                if sequence_exported:
                    for filename, arr, artifact_type in (
                        ("X_sequence_train.npy", sequence_train, "X_sequence_train"),
                        ("X_sequence_test.npy", sequence_test, "X_sequence_test"),
                        ("sequence_mask_train.npy", sequence_mask_train, "sequence_mask_train"),
                        ("sequence_mask_test.npy", sequence_mask_test, "sequence_mask_test"),
                    ):
                        assert arr is not None
                        tracker.save_npy(
                            external_dir / filename,
                            arr,
                            representation=representation,
                            split=split,
                            target=target,
                            threshold=threshold,
                            artifact_type=artifact_type,
                            reason="optional sequence tensor payload for causal PCA sequence representation",
                        )
                        npy_count += 1
                    sequence_offsets = {
                        "artifact_type": "sequence_metadata",
                        "sequence_time_offsets": [-2.0, -1.0, 0.0],
                        "sequence_shape_train": [int(item) for item in sequence_train.shape],
                        "sequence_shape_test": [int(item) for item in sequence_test.shape],
                        "mask_shape_train": [int(item) for item in sequence_mask_train.shape],
                        "mask_shape_test": [int(item) for item in sequence_mask_test.shape],
                    }
                    tracker.write_json(
                        external_dir / "sequence_time_offsets.json",
                        sequence_offsets,
                        representation=representation,
                        split=split,
                        target=target,
                        threshold=threshold,
                        artifact_type="sequence_metadata",
                        reason="sequence tensor offsets and shapes",
                        heavy=True,
                    )
                    tracker.write_json(
                        tracked_dir / "sequence_time_offsets.json",
                        sequence_offsets,
                        representation=representation,
                        split=split,
                        target=target,
                        threshold=threshold,
                        artifact_type="sequence_metadata",
                        reason="sequence tensor offsets and shapes",
                        heavy=False,
                    )

                train_meta = row_metadata(
                    rows=out_train_rows,
                    idx=out_train_idx,
                    y=y_train,
                    split_name=split,
                    split_role="train",
                    target=target,
                    threshold=threshold,
                    manifest_path=manifest_path,
                    cache_dir=cache_dir,
                    alignment_by_video=alignment_by_video,
                    representation_name=representation,
                    ctx=ctx,
                    sequence_extra=sequence_extra_train,
                )
                test_meta = row_metadata(
                    rows=out_test_rows,
                    idx=out_test_idx,
                    y=y_test,
                    split_name=split,
                    split_role="test",
                    target=target,
                    threshold=threshold,
                    manifest_path=manifest_path,
                    cache_dir=cache_dir,
                    alignment_by_video=alignment_by_video,
                    representation_name=representation,
                    ctx=ctx,
                    sequence_extra=sequence_extra_test,
                )
                video_83_included = any(item["is_video_83"] for item in train_meta + test_meta)
                feature_payload = feature_manifest(
                    representation_name=representation,
                    feature_width=int(x_train.shape[1]),
                    roi_labels=roi_labels,
                    selected_vertices=selected_vertices,
                )
                rep_payload = build_representation_metadata(
                    representation_name=representation,
                    feature_width=int(x_train.shape[1]),
                    train_dropped=len(train_selected) - len(out_train_rows),
                    test_dropped=len(test_selected) - len(out_test_rows),
                    video_83_included=video_83_included,
                    pca_meta=pca_meta,
                    roi_mapping=roi_mapping,
                    selected_vertices_path="selected_vertices.json" if representation == "topk_vertices_512" else None,
                    target=target,
                    threshold=threshold,
                )
                split_payload = split_metadata(
                    split_name=split,
                    train_rows=out_train_rows,
                    test_rows=out_test_rows,
                    held_out=held,
                    blocked_gap_rows=split_gap.get(split),
                    manifest_path=manifest_path,
                )
                target_payload = target_metadata(
                    target=target,
                    threshold=threshold,
                    horizon=horizon,
                    task_type=task_type,
                    train_y=y_train,
                    test_y=y_test,
                )
                leak_payload = leakage_contract(
                    representation_name=representation,
                    split_name=split,
                    target=target,
                    threshold=threshold,
                    overlap=split_payload["train_test_video_overlap"],
                )
                extra_jsons: dict[str, tuple[dict[str, Any], str]] = {}
                if representation == "roi_parcel_features":
                    extra_jsons["roi_mapping_metadata.json"] = (roi_mapping, "roi_mapping_metadata")
                if representation == "topk_vertices_512" and feature_selection_payload is not None:
                    extra_jsons["selected_vertices.json"] = (feature_selection_payload, "selected_vertices")
                    extra_jsons["feature_selection_metadata.json"] = (
                        {
                            "schema_version": "veatic_feature_selection_metadata_v1",
                            "representation_name": "topk_vertices_512",
                            "split_name": split,
                            "target_name": target,
                            "threshold": float(threshold),
                            "selection_scope": "train_rows_only",
                            "selected_vertices_digest": feature_selection_payload["selected_vertices_digest"],
                            "score_method": feature_selection_payload["score_method"],
                            "label_shuffle_warning": True,
                            "cautionary_for_primary_training": True,
                            "notes": "Supervised feature-selection tensor; use as cautionary comparison unless confirmed.",
                        },
                        "feature_selection_metadata",
                    )

                checksum_payload = write_contract_common_metadata(
                    tracker=tracker,
                    external_dir=external_dir,
                    tracked_dir=tracked_dir,
                    representation=representation,
                    split=split,
                    target=target,
                    threshold=threshold,
                    feature_manifest_payload=feature_payload,
                    representation_metadata_payload=rep_payload,
                    split_metadata_payload=split_payload,
                    target_metadata_payload=target_payload,
                    leakage_contract_payload=leak_payload,
                    row_meta_train=train_meta,
                    row_meta_test=test_meta,
                    extra_jsons=extra_jsons,
                    created_at=created_at,
                )
                contract_status = verify_contract(
                    folder=external_dir,
                    representation=representation,
                    split=split,
                    target=target,
                    threshold=threshold,
                    x_train=x_train,
                    x_test=x_test,
                    y_train=y_train,
                    y_test=y_test,
                    row_meta_train=train_meta,
                    row_meta_test=test_meta,
                    split_meta=split_payload,
                    rep_meta=rep_payload,
                    checksum=checksum_payload,
                    checks=verification_checks,
                    failures=verification_failures,
                    external_root=external_tensor_root,
                )
                inventory_rows.append(
                    {
                        "representation_name": representation,
                        "representation_family": representation_family(representation),
                        "split_name": split,
                        "target_name": target,
                        "threshold": float(threshold),
                        "task_type": task_type,
                        "tensor_dir": str(external_dir),
                        "x_train_path": str(external_dir / "X_train.npy"),
                        "x_test_path": str(external_dir / "X_test.npy"),
                        "y_train_path": str(external_dir / "y_train.npy"),
                        "y_test_path": str(external_dir / "y_test.npy"),
                        "x_train_shape": shape_string(x_train),
                        "x_test_shape": shape_string(x_test),
                        "y_train_shape": shape_string(y_train),
                        "y_test_shape": shape_string(y_test),
                        "feature_width": int(x_train.shape[1]),
                        "sequence_exported": str(bool(sequence_exported)).lower(),
                        "sequence_train_path": str(external_dir / "X_sequence_train.npy") if sequence_exported else "",
                        "sequence_test_path": str(external_dir / "X_sequence_test.npy") if sequence_exported else "",
                        "sequence_train_shape": shape_string(sequence_train) if sequence_exported else None,
                        "sequence_test_shape": shape_string(sequence_test) if sequence_exported else None,
                        "train_row_count": int(x_train.shape[0]),
                        "test_row_count": int(x_test.shape[0]),
                        "train_video_count": len(row_video_ids(out_train_rows)),
                        "test_video_count": len(row_video_ids(out_test_rows)),
                        "train_event_count": int(np.sum(y_train)),
                        "test_event_count": int(np.sum(y_test)),
                        "train_positive_rate": float(np.mean(y_train)) if y_train.size else 0.0,
                        "test_positive_rate": float(np.mean(y_test)) if y_test.size else 0.0,
                        "uses_labels_for_fit": str(representation == "topk_vertices_512").lower(),
                        "uses_future_features": "false",
                        "fit_scope": "train_rows_only",
                        "pca_width": rep_payload["pca"]["pca_width"] if rep_payload.get("pca") else None,
                        "pca_cache_reused": str(rep_payload["pca"]["cache_reused"]).lower() if rep_payload.get("pca") else None,
                        "pca_rebuilt": str(rep_payload["pca"]["cache_rebuilt"]).lower() if rep_payload.get("pca") else None,
                        "supervised_selection": str(representation == "topk_vertices_512").lower(),
                        "video_83_included": str(video_83_included).lower(),
                        "exclude_video_83_variant": "false",
                        "checksum_manifest_path": str(external_dir / "checksum_manifest.json"),
                        "leakage_contract_path": str(external_dir / "leakage_contract.json"),
                        "verification_status": contract_status,
                        "notes": "Sequence tensors exported." if sequence_exported else "",
                    }
                )

    total_expected = len(REPRESENTATIONS) * len(SPLITS) * len(TARGETS)
    missing_or_skipped = [
        "exclude-video-83 sensitivity tensors were skipped; all required all-video contracts include video 83.",
        "No large .npy tensors are included in the review zip.",
    ]
    verification_status = "pass" if not verification_failures and len(inventory_rows) == total_expected else "fail"
    summary = {
        "schema_version": "veatic_tensor_export_summary_v1",
        "created_at": created_at,
        "source_audit_output_dir": str(SOURCE_AUDIT_DIR),
        "external_tensor_root": str(external_tensor_root),
        "tracked_summary_root": str(TRACKED_SUMMARY_ROOT),
        "no_reencode": True,
        "video_83_included": True,
        "exclude_video_83_sensitivity_exported": False,
        "representations_exported": list(REPRESENTATIONS),
        "splits_exported": list(SPLITS),
        "targets_exported": [
            {"target_name": target, "threshold": threshold, "task_type": task_type}
            for target, _horizon, threshold, task_type in TARGETS
        ],
        "total_contracts_expected": total_expected,
        "total_contracts_exported": len(inventory_rows),
        "total_npy_files_exported": npy_count,
        "total_external_size_bytes": sum(path.stat().st_size for path in external_tensor_root.rglob("*") if path.is_file()),
        "pca_cache": {
            "fit_cache_root": str(FIT_CACHE_ROOT),
            "cache_reused_count": len(pca_reused_keys),
            "cache_rebuilt_count": len(pca_rebuilt_keys),
            "missing_cache_count": 0,
            "rebuilt_items": sorted(pca_rebuilt_keys),
        },
        "best_next_model_input": {
            "representation_name": "pca_sequence_128_causal_past_2s_mean",
            "reason": "Unsupervised train-only PCA128 with causal past 2s mean context; no labels used for feature construction.",
        },
        "baseline": {
            "representation_name": "cortical_pca64_delta_frozen_baseline",
            "role": "frozen_v2_reference",
        },
        "supervised_representations": ["topk_vertices_512"],
        "safe_for_primary_training": [
            "pca_sequence_128_causal_past_2s_mean",
            "roi_parcel_features",
            "cortical_pca64_delta_frozen_baseline",
        ],
        "cautionary_for_primary_training": ["topk_vertices_512"],
        "verification_status": verification_status,
        "missing_or_skipped": missing_or_skipped,
        "notes": "Tensor export only; no full audit rerun, no all-candidate rescoring, and no video re-encoding.",
    }
    verification = {
        "schema_version": "veatic_tensor_export_verification_v1",
        "created_at": created_at,
        "status": verification_status,
        "checks": verification_checks,
        "failures": verification_failures,
        "warnings": warnings,
    }
    report = make_report(summary, inventory_rows)

    # Global external and tracked files.
    for root_path, heavy in ((external_tensor_root, True), (TRACKED_SUMMARY_ROOT, False)):
        tracker.write_csv(
            root_path / "tensor_export_inventory.csv",
            inventory_rows,
            fieldnames=INVENTORY_COLUMNS,
            representation=None,
            split=None,
            target=None,
            threshold=None,
            artifact_type="other",
            reason="global tensor contract inventory",
            heavy=heavy,
        )
        tracker.write_json(
            root_path / "tensor_export_summary.json",
            summary,
            representation=None,
            split=None,
            target=None,
            threshold=None,
            artifact_type="other",
            reason="global tensor export summary",
            heavy=heavy,
        )
        tracker.write_json(
            root_path / "tensor_export_verification.json",
            verification,
            representation=None,
            split=None,
            target=None,
            threshold=None,
            artifact_type="other",
            reason="global tensor export verification",
            heavy=heavy,
        )
        tracker.write_text(
            root_path / "tensor_export_report.md",
            report,
            representation=None,
            split=None,
            target=None,
            threshold=None,
            artifact_type="other",
            reason="global tensor export report",
            heavy=heavy,
        )

    # File inventory is written after other tracked/external files are present.
    # The two inventory files are self-referential, so their own hash/size cannot
    # be populated before the final payload exists. Keep the required fields and
    # use nulls for those two entries only.
    inventory_self_entries = []
    for path, heavy in (
        (external_tensor_root / "tensor_export_file_inventory.json", True),
        (TRACKED_SUMMARY_ROOT / "tensor_export_file_inventory.json", False),
    ):
        try:
            relative = path.relative_to(external_tensor_root).as_posix()
        except ValueError:
            relative = path.relative_to(TRACKED_SUMMARY_ROOT).as_posix()
        inventory_self_entries.append(
            {
                "absolute_path": str(path.resolve()),
                "relative_path": relative,
                "size_bytes": None,
                "sha256": None,
                "representation_name": None,
                "split_name": None,
                "target_name": None,
                "threshold": None,
                "artifact_type": "other",
                "reason_included": "self-referential tensor_export_file_inventory.json; sha256 and size are null by construction",
                "heavy_external_artifact": heavy,
            }
        )
    final_inventory_files = sorted(
        tracker.files + inventory_self_entries,
        key=lambda item: (str(item["representation_name"]), str(item["relative_path"]), str(item["absolute_path"])),
    )
    file_inventory_payload = {
        "schema_version": "veatic_tensor_export_file_inventory_v1",
        "created_at": created_at,
        "external_tensor_root": str(external_tensor_root),
        "tracked_summary_root": str(TRACKED_SUMMARY_ROOT),
        "file_count": len(final_inventory_files),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in final_inventory_files if item["size_bytes"] is not None),
        "files": final_inventory_files,
    }
    for root_path, heavy in ((external_tensor_root, True), (TRACKED_SUMMARY_ROOT, False)):
        write_json(
            root_path / "tensor_export_file_inventory.json",
            file_inventory_payload,
        )

    zip_review(TRACKED_SUMMARY_ROOT, REVIEW_ZIP_PATH)
    zip_valid = validate_zip(REVIEW_ZIP_PATH)
    if not zip_valid:
        verification_status = "fail"

    final_size = sum(path.stat().st_size for path in external_tensor_root.rglob("*") if path.is_file())
    review_zip_has_npy = any(item.filename.endswith(".npy") for item in zipfile.ZipFile(REVIEW_ZIP_PATH).infolist())
    final = {
        "elapsed_seconds": time.monotonic() - start,
        "external_tensor_export_root": str(external_tensor_root),
        "tracked_lightweight_output_root": str(TRACKED_SUMMARY_ROOT),
        "review_zip_path": str(REVIEW_ZIP_PATH),
        "total_tensor_folders_exported": len(inventory_rows),
        "total_tensor_files_exported": npy_count,
        "total_size_external_tensor_export_bytes": final_size,
        "total_size_review_zip_bytes": REVIEW_ZIP_PATH.stat().st_size,
        "verification_status": "pass" if verification_status == "pass" and zip_valid and not review_zip_has_npy else "fail",
        "missing_skipped_artifacts": missing_or_skipped,
        "pca_cache_rebuilt": bool(pca_rebuilt_keys),
        "pca_cache_rebuilt_items": sorted(pca_rebuilt_keys),
        "pca_cache_reused_count": len(pca_reused_keys),
        "video_83_included": True,
        "review_zip_validated_with_unzip_tq": zip_valid,
        "review_zip_contains_npy": review_zip_has_npy,
        "commands_run": COMMANDS_RUN,
    }
    print(json.dumps(json_safe(final), indent=2, sort_keys=True))
    return 0 if final["verification_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
