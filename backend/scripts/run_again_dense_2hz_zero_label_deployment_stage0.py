#!/usr/bin/env python3
"""Freeze the planning-only zero-label deployment bridge Stage 0 contracts."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts import again_dense_2hz_benchmark as base  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_redesigned_target_blocked as redesigned  # noqa: E402

SCHEMA_VERSION = "again_dense_2hz_zero_label_deployment_stage0_v1"
PREREGISTRATION = "docs/zero_label_video_only_deployment_bridge_pilot_preregistration.md"
SPLIT_NAMESPACE = "neural_bridge_zero_label_v1_20260714"
TEACHER_NAMESPACE = "neural_bridge_zero_label_teacher_crossfit_v1_20260714"
TARGET_NAME = "future_arousal_max_delta_rows_4_10"
TARGET_MASK = "target_mask_future_arousal_max_delta_rows_4_10"
EVENT_QUANTILE = 0.90
EXPECTED_VIDEOS = 995
DEVELOPMENT_VIDEOS = 696
LOCKED_VIDEOS = 299
STAGE_A_FOLDS = 3
STAGE_A_TEST_VIDEOS = 232
STAGE_B_PANEL_SIZES = (60, 60, 60, 60, 59)
STAGE_A_SEEDS = (20260718, 20260719, 20260720)
STAGE_B_SEEDS = (20260721, 20260722, 20260723)
REQUIRED_METRICS = (
    "pooled_continuous_spearman",
    "top_5pct_true_future_movement_lift",
    "training_q90_future_event_pr_auc",
)
STAGE_A_LANES = (
    "video_distilled_temporal",
    "video_closed_loop_rollout",
    "video_supervised_temporal",
    "video_supervised_current_row",
    "no_video_closed_loop_persistence",
    "sequence_shuffled_video",
    "video_label_permutation",
    "phase7_ar_assisted_teacher_ceiling",
)
STAGE_B_LANES = (
    "locked_stage_a_winner",
    "video_supervised_temporal",
    "video_supervised_current_row",
    "no_video_closed_loop_persistence",
    "sequence_shuffled_video",
    "video_label_permutation",
    "phase7_ar_assisted_teacher_ceiling",
)
FORBIDDEN_EXACT_FIELDS = frozenset(
    {
        "arousal",
        "valence",
        "label_available",
        "ar_context_available",
        TARGET_NAME,
        TARGET_MASK,
        "event_label",
        "teacher_score",
        "teacher_hidden_state",
        "ar_score",
        "ar_reg",
    }
)
FORBIDDEN_FIELD_TOKENS = (
    "arousal_lag",
    "arousal_delta",
    "ground_truth",
    "future_",
    "target_",
    "teacher_",
)
DEFAULT_DENSE_ROOT = REPO_ROOT / ".cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "evidence/zero_label_video_only_deployment_stage0_20260714"
STAGE0_ARTIFACT_NAMES = frozenset(
    {
        "split_manifest.json",
        "target_identity_manifest.json",
        "feature_policy_manifest.json",
        "dry_run_matrix.csv",
        "stage0_result.json",
    }
)


def canonical_digest(value: Any, *, digest_size: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=digest_size).hexdigest()


def array_digest(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(values)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(json.dumps(list(arr.shape)).encode("ascii"))
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def source_digest(function: Any) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def hashed_video_order(video_ids: Iterable[str], namespace: str = SPLIT_NAMESPACE) -> list[str]:
    unique = {str(video_id) for video_id in video_ids}

    def key(video_id: str) -> tuple[str, str]:
        raw = f"{namespace}|{video_id}".encode("utf-8")
        return hashlib.blake2b(raw, digest_size=16).hexdigest(), video_id

    return sorted(unique, key=key)


def partition_exact(values: Sequence[str], sizes: Sequence[int]) -> list[list[str]]:
    if sum(sizes) != len(values):
        raise ValueError(f"Partition sizes {tuple(sizes)} do not cover {len(values)} values")
    out: list[list[str]] = []
    cursor = 0
    for size in sizes:
        out.append(list(values[cursor : cursor + size]))
        cursor += size
    return out


def three_way_crossfit(video_ids: Sequence[str], namespace: str) -> list[list[str]]:
    ordered = hashed_video_order(video_ids, namespace)
    base_size, remainder = divmod(len(ordered), 3)
    sizes = tuple(base_size + (1 if fold < remainder else 0) for fold in range(3))
    return partition_exact(ordered, sizes)


def videos_digest(video_ids: Sequence[str]) -> str:
    return canonical_digest(list(video_ids))


def build_split_manifest(video_ids: Sequence[str]) -> dict[str, Any]:
    ordered = hashed_video_order(video_ids)
    if len(ordered) != EXPECTED_VIDEOS:
        raise ValueError(f"Expected {EXPECTED_VIDEOS} unique videos, got {len(ordered)}")
    development = ordered[:DEVELOPMENT_VIDEOS]
    locked = ordered[DEVELOPMENT_VIDEOS:]
    stage_a_tests = partition_exact(development, (STAGE_A_TEST_VIDEOS,) * STAGE_A_FOLDS)
    stage_a: list[dict[str, Any]] = []
    for fold, test_videos in enumerate(stage_a_tests, 1):
        test_set = set(test_videos)
        train_videos = [video for video in development if video not in test_set]
        teacher_folds = three_way_crossfit(
            train_videos, f"{TEACHER_NAMESPACE}|stage_a|fold{fold}"
        )
        stage_a.append(
            {
                "fold": fold,
                "train_videos": train_videos,
                "test_videos": test_videos,
                "train_count": len(train_videos),
                "test_count": len(test_videos),
                "train_digest": videos_digest(train_videos),
                "test_digest": videos_digest(test_videos),
                "split_digest": canonical_digest({"train": train_videos, "test": test_videos}),
                "teacher_crossfit_test_folds": teacher_folds,
                "teacher_crossfit_fold_digests": [videos_digest(values) for values in teacher_folds],
            }
        )
    stage_b_panels = partition_exact(locked, STAGE_B_PANEL_SIZES)
    stage_b_teacher = three_way_crossfit(development, f"{TEACHER_NAMESPACE}|stage_b")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "namespace": SPLIT_NAMESPACE,
        "all_video_count": len(ordered),
        "all_video_digest": videos_digest(ordered),
        "development_videos": development,
        "development_count": len(development),
        "development_digest": videos_digest(development),
        "locked_videos": locked,
        "locked_count": len(locked),
        "locked_digest": videos_digest(locked),
        "stage_a": stage_a,
        "stage_b": {
            "train_videos": development,
            "train_count": len(development),
            "train_digest": videos_digest(development),
            "panels": [
                {
                    "panel": panel,
                    "test_videos": values,
                    "test_count": len(values),
                    "test_digest": videos_digest(values),
                    "split_digest": canonical_digest(
                        {"train": development, "panel": panel, "test": values}
                    ),
                }
                for panel, values in enumerate(stage_b_panels, 1)
            ],
            "teacher_crossfit_test_folds": stage_b_teacher,
            "teacher_crossfit_fold_digests": [videos_digest(values) for values in stage_b_teacher],
        },
        "historically_untouched": False,
        "prospectively_locked_for_this_method": True,
    }
    validate_split_manifest(manifest)
    return manifest


def validate_split_manifest(manifest: dict[str, Any]) -> None:
    development = set(manifest["development_videos"])
    locked = set(manifest["locked_videos"])
    if development & locked or len(development) != DEVELOPMENT_VIDEOS or len(locked) != LOCKED_VIDEOS:
        raise ValueError("Development/locked split is overlapping or incomplete")
    seen_tests: set[str] = set()
    for fold in manifest["stage_a"]:
        train = set(fold["train_videos"])
        test = set(fold["test_videos"])
        if train & test or train | test != development:
            raise ValueError(f"Stage A fold {fold['fold']} is not video-disjoint and complete")
        if seen_tests & test:
            raise ValueError("Stage A test folds overlap")
        seen_tests |= test
        teacher_tests = [set(values) for values in fold["teacher_crossfit_test_folds"]]
        if set().union(*teacher_tests) != train or any(values & test for values in teacher_tests):
            raise ValueError("Stage A teacher cross-fit ownership is invalid")
        if any(teacher_tests[i] & teacher_tests[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Stage A teacher cross-fit folds overlap")
    if seen_tests != development:
        raise ValueError("Stage A test folds do not cover development videos exactly once")
    panels = [set(panel["test_videos"]) for panel in manifest["stage_b"]["panels"]]
    if set().union(*panels) != locked:
        raise ValueError("Stage B panels do not cover locked videos")
    if any(panels[i] & panels[j] for i in range(5) for j in range(i + 1, 5)):
        raise ValueError("Stage B panels overlap")
    teacher_tests = [set(values) for values in manifest["stage_b"]["teacher_crossfit_test_folds"]]
    if set().union(*teacher_tests) != development:
        raise ValueError("Stage B teacher cross-fit does not cover development videos")
    if any(values & locked for values in teacher_tests):
        raise ValueError("Locked video entered Stage B teacher cross-fit")


def row_ids(df: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [f"{video_id}:{int(row_index)}" for video_id, row_index in zip(df["video_id"], df["row_index"])],
        dtype="U128",
    )


def split_target_record(
    df: pd.DataFrame,
    values: np.ndarray,
    valid: np.ndarray,
    train_videos: Sequence[str],
    test_videos: Sequence[str],
    split_id: str,
) -> dict[str, Any]:
    videos = df["video_id"].astype(str).to_numpy()
    label_available = df["label_available"].to_numpy(dtype=bool)
    finite_valid = valid & label_available & np.isfinite(values)
    train_mask = np.isin(videos, np.asarray(train_videos)) & finite_valid
    test_mask = np.isin(videos, np.asarray(test_videos)) & finite_valid
    if not train_mask.any() or not test_mask.any():
        raise ValueError(f"Target split {split_id} has no valid train or test rows")
    spec = redesigned.target_specs()[0]
    y_train, y_test, threshold = base.threshold_labels(values, train_mask, test_mask, spec)
    ids = row_ids(df)
    test_indices = np.flatnonzero(test_mask)
    first30 = df.loc[test_indices, "time_seconds"].to_numpy(dtype=np.float64) <= 30.0
    y_test_first30 = y_test[first30]
    require_event_gate_defined(y_test)
    require_event_gate_defined(y_test_first30)
    return {
        "split_id": split_id,
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_row_ids_digest": array_digest(ids[train_mask]),
        "test_row_ids_digest": array_digest(ids[test_mask]),
        "train_mask_digest": array_digest(train_mask.astype(np.uint8)),
        "test_mask_digest": array_digest(test_mask.astype(np.uint8)),
        "train_values_digest": array_digest(values[train_mask].astype(np.float64)),
        "test_values_digest": array_digest(values[test_mask].astype(np.float64)),
        "event_threshold_train_q90": threshold,
        "event_train_positive": int(y_train.sum()),
        "event_train_negative": int(len(y_train) - y_train.sum()),
        "event_test_positive": int(y_test.sum()),
        "event_test_negative": int(len(y_test) - y_test.sum()),
        "event_pr_auc_defined": event_gate_defined(y_test),
        "first30_test_rows": int(len(y_test_first30)),
        "first30_event_positive": int(y_test_first30.sum()),
        "first30_event_negative": int(len(y_test_first30) - y_test_first30.sum()),
        "first30_event_pr_auc_defined": event_gate_defined(y_test_first30),
    }


def build_target_identity_manifest(df: pd.DataFrame, split_manifest: dict[str, Any]) -> dict[str, Any]:
    values, valid = redesigned.future_max_delta(df, 4, 10)
    ids = row_ids(df)
    records: list[dict[str, Any]] = []
    for fold in split_manifest["stage_a"]:
        records.append(
            split_target_record(
                df,
                values,
                valid,
                fold["train_videos"],
                fold["test_videos"],
                f"stage_a_fold{fold['fold']}",
            )
        )
    for panel in split_manifest["stage_b"]["panels"]:
        records.append(
            split_target_record(
                df,
                values,
                valid,
                split_manifest["stage_b"]["train_videos"],
                panel["test_videos"],
                f"stage_b_panel{panel['panel']}",
            )
        )
    score_mask = valid & df["label_available"].to_numpy(dtype=bool) & np.isfinite(values)
    identity = {
        "target_name": TARGET_NAME,
        "value_column": TARGET_NAME,
        "mask_column": TARGET_MASK,
        "event_transform": "positive_delta",
        "event_quantile": EVENT_QUANTILE,
        "rows": len(df),
        "shape": list(values.shape),
        "all_row_ids_digest": array_digest(ids),
        "score_row_ids_digest": array_digest(ids[score_mask]),
        "value_array_digest": array_digest(values.astype(np.float64)),
        "mask_array_digest": array_digest(valid.astype(np.uint8)),
        "score_mask_digest": array_digest(score_mask.astype(np.uint8)),
        "valid_score_rows": int(score_mask.sum()),
        "builder_function": "run_again_dense_2hz_phase5_redesigned_target_blocked.future_max_delta",
        "builder_source_sha256": source_digest(redesigned.future_max_delta),
        "event_scorer_function": "again_dense_2hz_benchmark.threshold_labels",
        "event_scorer_source_sha256": source_digest(base.threshold_labels),
        "expected_prediction_identity_fields": [
            "split_id",
            "video_id",
            "row_index",
            "row_id",
            "target_identity_digest",
            "prediction",
        ],
        "prediction_table_created_in_stage0": False,
        "hard_outcome_entered_in_inference_block": False,
        "split_records": records,
        "phase7_name_value_hazard_resolved": True,
        "residual_target_used": False,
    }
    identity["target_identity_digest"] = canonical_digest(identity)
    return identity


def build_feature_policy(dense_root: Path, rows: int) -> dict[str, Any]:
    labels_path = dense_root / "labels_aligned_2hz.parquet"
    diagnostics_path = dense_root / "_derived/temporal_diagnostics_summary_features.npy"
    diagnostics_meta_path = diagnostics_path.with_suffix(".json")
    for path in (labels_path, diagnostics_path, diagnostics_meta_path):
        if not path.exists():
            raise FileNotFoundError(path)
    diagnostics = np.load(diagnostics_path, mmap_mode="r")
    diagnostics_meta = json.loads(diagnostics_meta_path.read_text(encoding="utf-8"))
    if diagnostics.shape != (rows, 53) or diagnostics_meta.get("width") != 53:
        raise ValueError(f"Expected diagnostics shape {(rows, 53)}, got {diagnostics.shape}")
    return {
        "schema_version": SCHEMA_VERSION,
        "dense_root": display_path(dense_root),
        "frozen_substrate_only": True,
        "vjepa_tribe_reencoding": False,
        "dense_cache_mutation": False,
        "hardware_for_later_fitting": "mlx_gpu_mps",
        "cpu_fallback": False,
        "pca": {
            "source": "frozen_predicted_cortical_fmri_row_features",
            "causal_pooling": "temporal_mean_2s",
            "width": 256,
            "width_search": False,
            "basis_policy": "fit_new_basis_inside_each_outer_training_pool_only",
            "teacher_basis_policy": "fit_inside_each_nested_teacher_crossfit_training_partition_only",
            "locked_or_outer_test_rows_in_fit": False,
            "reuse_incompatible_phase7_fold_score_matrices": False,
            "reason": "Phase 7 fold score matrices use different coordinate systems and cannot be concatenated for the 696/299 split",
        },
        "diagnostics": {
            "path": display_path(diagnostics_path),
            "metadata_path": display_path(diagnostics_meta_path),
            "shape": list(diagnostics.shape),
            "npy_sha256": file_digest(diagnostics_path),
            "metadata_sha256": file_digest(diagnostics_meta_path),
            "video_derived_only": True,
        },
        "labels_artifact": {
            "path": display_path(labels_path),
            "sha256": file_digest(labels_path),
            "inference_access": False,
            "scorer_join_after_prediction_checksum_only": True,
        },
        "h1_allowed_inputs": [
            "pca256_current_row",
            "pca256_same_video_lag_1",
            "pca256_same_video_lag_2",
            "pca256_same_video_lag_3",
            "pca256_same_video_lag_4",
            "temporal_diagnostics_53_current_row",
            "history_availability_masks",
            "time_seconds",
            "video_time_fraction_from_video_metadata",
        ],
        "h2_additional_inputs": ["own_prior_predictions", "lags_and_deltas_of_own_prior_predictions"],
        "video_id_policy": "split_order_reset_metadata_only_never_learned_feature",
        "forbidden_exact_fields": sorted(FORBIDDEN_EXACT_FIELDS),
        "forbidden_field_tokens": list(FORBIDDEN_FIELD_TOKENS),
        "candidate_inference_labels_loaded": False,
        "teacher_scores_available_at_candidate_inference": False,
    }


def cold_start_policy(lane: str) -> str:
    if lane in {"video_closed_loop_rollout", "no_video_closed_loop_persistence"}:
        return "self_prediction_state_train_median_init_video_reset"
    if lane == "phase7_ar_assisted_teacher_ceiling":
        return "observed_arousal_non_deployable_ceiling"
    if lane == "locked_stage_a_winner":
        return "locked_stage_a_policy"
    return "zero_padded_same_video_history_with_masks"


def dry_run_matrix(split_manifest: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in split_manifest["stage_a"]:
        for lane in STAGE_A_LANES:
            for seed in STAGE_A_SEEDS:
                rows.append(
                    {
                        "stage": "stage_a",
                        "split_digest": fold["split_digest"],
                        "fold_or_panel": fold["fold"],
                        "lane": lane,
                        "row_type": "member",
                        "seed_or_group": seed,
                        "cold_start_policy": cold_start_policy(lane),
                    }
                )
            rows.append(
                {
                    "stage": "stage_a",
                    "split_digest": fold["split_digest"],
                    "fold_or_panel": fold["fold"],
                    "lane": lane,
                    "row_type": "ensemble",
                    "seed_or_group": ",".join(map(str, STAGE_A_SEEDS)),
                    "cold_start_policy": cold_start_policy(lane),
                }
            )
    for panel in split_manifest["stage_b"]["panels"]:
        for lane in STAGE_B_LANES:
            for seed in STAGE_B_SEEDS:
                rows.append(
                    {
                        "stage": "stage_b",
                        "split_digest": panel["split_digest"],
                        "fold_or_panel": panel["panel"],
                        "lane": lane,
                        "row_type": "member",
                        "seed_or_group": seed,
                        "cold_start_policy": cold_start_policy(lane),
                    }
                )
            rows.append(
                {
                    "stage": "stage_b",
                    "split_digest": panel["split_digest"],
                    "fold_or_panel": panel["panel"],
                    "lane": lane,
                    "row_type": "ensemble",
                    "seed_or_group": ",".join(map(str, STAGE_B_SEEDS)),
                    "cold_start_policy": cold_start_policy(lane),
                }
            )
    frame = pd.DataFrame(rows)
    validate_dry_run_matrix(frame)
    return frame


def validate_dry_run_matrix(frame: pd.DataFrame) -> None:
    keys = [
        "stage",
        "split_digest",
        "fold_or_panel",
        "lane",
        "row_type",
        "seed_or_group",
        "cold_start_policy",
    ]
    if frame.duplicated(keys).any():
        raise ValueError("Dry-run matrix has duplicate uniqueness keys")
    counts = frame.groupby(["stage", "row_type"]).size().to_dict()
    expected = {
        ("stage_a", "member"): 72,
        ("stage_a", "ensemble"): 24,
        ("stage_b", "member"): 105,
        ("stage_b", "ensemble"): 35,
    }
    if counts != expected or len(frame) != 236:
        raise ValueError(f"Dry-run matrix mismatch: {counts}, total={len(frame)}")


@dataclass(frozen=True)
class VideoOnlyInferenceBlock:
    row_ids: np.ndarray
    video_ids: np.ndarray
    row_indices: np.ndarray
    pca_sequence: np.ndarray
    diagnostics: np.ndarray
    history_available: np.ndarray
    time_features: np.ndarray

    def validate(self) -> None:
        count = len(self.row_ids)
        if any(len(value) != count for value in (self.video_ids, self.row_indices, self.pca_sequence, self.diagnostics, self.history_available, self.time_features)):
            raise ValueError("Inference block row counts do not match")
        if self.pca_sequence.shape[1:] != (5, 256):
            raise ValueError("Expected causal PCA sequence shape rows x 5 x 256")
        if self.diagnostics.shape[1:] != (53,) or self.history_available.shape[1:] != (5,):
            raise ValueError("Diagnostics/history mask shape mismatch")
        if not all(np.isfinite(value).all() for value in (self.pca_sequence, self.diagnostics, self.time_features)):
            raise ValueError("Inference block contains nonfinite features")


def inference_block_field_names() -> set[str]:
    return {field.name for field in fields(VideoOnlyInferenceBlock)}


def validate_inference_feature_names(names: Iterable[str]) -> None:
    for raw_name in names:
        name = str(raw_name).lower()
        if name in FORBIDDEN_EXACT_FIELDS or any(token in name for token in FORBIDDEN_FIELD_TOKENS):
            raise ValueError(f"Forbidden held-out inference field: {raw_name}")
        if "arousal" in name or "valence" in name:
            raise ValueError(f"Observed-response field forbidden at inference: {raw_name}")


def causal_history_indices(video_ids: Sequence[str], row_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    if len(video_ids) != len(row_indices):
        raise ValueError("video_ids and row_indices must align")
    lookup = {(str(video), int(row)): idx for idx, (video, row) in enumerate(zip(video_ids, row_indices))}
    indices = np.full((len(video_ids), 5), -1, dtype=np.int64)
    available = np.zeros((len(video_ids), 5), dtype=bool)
    for idx, (video, row) in enumerate(zip(video_ids, row_indices)):
        for column, lag in enumerate(range(5)):
            source = lookup.get((str(video), int(row) - lag))
            if source is not None:
                indices[idx, column] = source
                available[idx, column] = True
    return indices, available


def validate_rollout_dependencies(records: Sequence[dict[str, Any]]) -> None:
    previous_video: str | None = None
    for record in records:
        video = str(record["video_id"])
        row = int(record["row_index"])
        source = record.get("state_source")
        source_row = record.get("source_row_index")
        if video != previous_video:
            if source != "train_median_initialization" or source_row is not None:
                raise ValueError("First row of each video must reset to train-median initialization")
        else:
            if source != "prediction" or source_row is None or int(source_row) >= row:
                raise ValueError("Rollout state must come only from an earlier same-video prediction")
            if str(record.get("source_video_id", video)) != video:
                raise ValueError("Rollout state cannot cross a video boundary")
        if record.get("teacher_forced") or record.get("observed_response_read"):
            raise ValueError("Teacher-forced or observed-response rollout state is forbidden")
        previous_video = video


def validate_prediction_seal(manifest: dict[str, Any]) -> None:
    if manifest.get("labels_loaded_before_checksum") is not False:
        raise ValueError("Prediction checksum must be sealed before labels are loaded")
    if not manifest.get("prediction_sha256"):
        raise ValueError("Prediction checksum is missing")
    validate_inference_feature_names(manifest.get("prediction_columns", ()))


def event_gate_defined(labels: Sequence[int] | np.ndarray) -> bool:
    unique = np.unique(np.asarray(labels, dtype=int))
    return bool(np.array_equal(unique, np.asarray([0, 1])))


def require_event_gate_defined(labels: Sequence[int] | np.ndarray) -> None:
    if not event_gate_defined(labels):
        raise ValueError("Required event PR-AUC slice has only one outcome class")


def stage0_static_manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "stage0_planning_contracts_only",
        "preregistration": PREREGISTRATION,
        "authorized": ["manifests", "dry_run_matrix", "contract_tests"],
        "forbidden": [
            "model_fitting",
            "teacher_score_generation",
            "heldout_prediction_scoring",
            "stage_a",
            "stage_b",
            "claim_promotion",
        ],
        "required_metrics": list(REQUIRED_METRICS),
        "teacher_incremental_gain_retention_minimum": 0.50,
        "absolute_phase7_parity_required": False,
        "stage_a_expected_rows": 96,
        "stage_b_expected_rows": 140,
        "accelerator_for_later_training": "mlx_gpu_mps",
        "no_cpu_fallback": True,
        "model_training_performed": False,
        "teacher_scores_generated": False,
        "heldout_predictions_scored": False,
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_stage0(dense_root: Path, output_root: Path, *, overwrite: bool = False) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        present = {path.name for path in output_root.iterdir()}
        if not overwrite or not present <= STAGE0_ARTIFACT_NAMES:
            raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    df = base.load_labels(dense_root)
    video_ids = df["video_id"].astype(str).unique().tolist()
    split_manifest = build_split_manifest(video_ids)
    target_manifest = build_target_identity_manifest(df, split_manifest)
    feature_policy = build_feature_policy(dense_root, len(df))
    matrix = dry_run_matrix(split_manifest)
    write_json(output_root / "split_manifest.json", split_manifest)
    write_json(output_root / "target_identity_manifest.json", target_manifest)
    write_json(output_root / "feature_policy_manifest.json", feature_policy)
    matrix.to_csv(output_root / "dry_run_matrix.csv", index=False)
    artifacts = {
        name: file_digest(output_root / name)
        for name in (
            "split_manifest.json",
            "target_identity_manifest.json",
            "feature_policy_manifest.json",
            "dry_run_matrix.csv",
        )
    }
    result = {
        **stage0_static_manifest(),
        "stage0_pass": True,
        "output_root": display_path(output_root),
        "dataset_rows": len(df),
        "dataset_videos": len(video_ids),
        "stage_a_matrix_rows": int((matrix["stage"] == "stage_a").sum()),
        "stage_b_matrix_rows": int((matrix["stage"] == "stage_b").sum()),
        "target_identity_digest": target_manifest["target_identity_digest"],
        "development_split_digest": split_manifest["development_digest"],
        "locked_split_digest": split_manifest["locked_digest"],
        "artifact_sha256": artifacts,
        "failed_contracts": [],
        "stage_a_authorized": False,
    }
    write_json(output_root / "stage0_result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-root", default=str(DEFAULT_DENSE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print(json.dumps(stage0_static_manifest(), indent=2, sort_keys=True))
        return 0
    result = run_stage0(Path(args.dense_root), Path(args.output_root), overwrite=args.overwrite)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
