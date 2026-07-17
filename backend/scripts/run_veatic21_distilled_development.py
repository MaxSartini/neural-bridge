#!/usr/bin/env python3
"""Run the bounded VEATIC 2.1 arousal/valence development matrix.

The runner consumes the completed compact cache only.  It fits a fresh
train-only PCA inside each grouped-video split and applies the promoted AGAIN
temporal recipe and matched controls.  All 124 VEATIC videos participate in
grouped cross-validation; the genuine zero-label cold-start confirmation is a
separate future video that is absent from VEATIC and every model-selection run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.scripts import again_dense_2hz_phase4_pca_bridge as phase4  # noqa: E402
from backend.scripts import again_zero_label_deployment_stage_a as zero_label  # noqa: E402
from backend.scripts import veatic21_compact_cache as compact  # noqa: E402
from backend.scripts import veatic21_distilled_program as program  # noqa: E402
from backend.scripts import veatic21_targets as targets  # noqa: E402


RUN_SCHEMA_VERSION = "veatic21_distilled_development_v1"
PCA_FAMILY = "temporal_mean_2s"
PCA_SEED = 20260716
DEFAULT_TARGETS = (
    targets.AROUSAL_FUTURE_MAX_DELTA,
    targets.VALENCE_FUTURE_RISE_MAGNITUDE,
    targets.VALENCE_FUTURE_DROP_MAGNITUDE,
    targets.VALENCE_FUTURE_MAX_ABS_MOVEMENT,
)
VIDEO_LANES = (
    "video_supervised_temporal",
    "video_supervised_current_row",
    "diagnostics_only_supervised_temporal",
    "sequence_shuffled_supervised_temporal",
    "label_permutation_supervised_temporal",
    "no_video_supervised_temporal",
)
PRIVILEGED_LANES = (
    "frozen_ar",
    "ar_plus_temporal_residual",
    "ar_plus_shuffled_pca_residual",
    "ar_plus_random_pca_residual",
    "ar_plus_train_only_video_mean_residual",
    "ar_plus_diagnostics_only_residual",
    "ar_plus_label_permutation_residual",
)
ALL_LANES = VIDEO_LANES + PRIVILEGED_LANES
ZERO_LABEL_CONTROLS = (
    "video_supervised_current_row",
    "diagnostics_only_supervised_temporal",
    "sequence_shuffled_supervised_temporal",
    "label_permutation_supervised_temporal",
    "no_video_supervised_temporal",
)
AR_CONTROLS = (
    "ar_plus_shuffled_pca_residual",
    "ar_plus_random_pca_residual",
    "ar_plus_train_only_video_mean_residual",
    "ar_plus_diagnostics_only_residual",
    "ar_plus_label_permutation_residual",
)
METRICS = (
    "pooled_continuous_spearman",
    "top_5pct_true_future_movement_lift",
    "training_q90_future_event_pr_auc",
)
FIRST30_METRICS = (
    "first30_pooled_continuous_spearman",
    "first30_top_5pct_true_future_movement_lift",
    "first30_training_q90_future_event_pr_auc",
)
PCA_SEQUENCE_WIDTH = program.WINDOW_ROWS * program.PCA_WIDTH
VIDEO_BLOCK_WIDTH = PCA_SEQUENCE_WIDTH + program.DIAGNOSTIC_WIDTH


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def parse_names(text: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in text.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one name is required")
    return values


def stable_contract_identity(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return the restart-stable portion of a run contract.

    Wall-clock metadata is deliberately excluded.  The resulting digest is the
    identity used by checkpoints and prediction shards, so an interrupted run
    can resume only when every scientific setting and input seal still agrees.
    """

    identity = dict(contract)
    identity.pop("created_at", None)
    identity.pop("contract_digest", None)
    return identity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--shared-derived-root",
        type=Path,
        default=None,
        help="Optional shared root for the sealed cortical memmap and fold-safe PCA artifacts.",
    )
    parser.add_argument("--identity-manifest", type=Path, default=compact.DEFAULT_IDENTITY_MANIFEST)
    parser.add_argument("--targets", type=parse_names, default=DEFAULT_TARGETS)
    parser.add_argument("--seeds", type=parse_ints, default=program.DEVELOPMENT_SEEDS)
    parser.add_argument("--folds", type=int, default=program.DEVELOPMENT_FOLDS)
    parser.add_argument(
        "--pca-family",
        choices=("temporal_mean_2s", "current", "delta"),
        default=PCA_FAMILY,
    )
    parser.add_argument(
        "--temporal-head",
        choices=("short_conv", "flat_mlp"),
        default="short_conv",
    )
    parser.add_argument("--max-epochs", type=int, default=zero_label.MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=zero_label.PATIENCE)
    parser.add_argument("--batch-size", type=int, default=zero_label.BATCH_SIZE)
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one development fold, target and seed while retaining all matched lanes.",
    )
    return parser


def _dense_rows_and_diagnostics(
    cache: compact.Veatic21CompactCache,
) -> tuple[list[dict[str, Any]], pd.DataFrame, np.ndarray]:
    rows: list[dict[str, Any]] = []
    diagnostics: list[np.ndarray] = []
    global_index = 0
    for block in cache.iter_videos():
        columns = block.columns
        count = block.row_count
        diagnostics.append(np.asarray(columns["temporal_diagnostics53"], dtype=np.float32))
        for local_index in range(count):
            time_seconds = float(columns["time_seconds"][local_index])
            rows.append(
                {
                    "schema_version": "veatic21_dense_row_v1",
                    "dataset": "veatic21",
                    "stimulus_id": f"{block.video_id}:{local_index:06d}",
                    "video_id": block.video_id,
                    "row_index": local_index,
                    "global_row_index": global_index,
                    "frame_index": int(round(time_seconds * compact.ROW_HZ)),
                    "time_seconds": time_seconds,
                    "time_start_seconds": time_seconds,
                    "time_end_seconds": time_seconds + 1.0 / compact.ROW_HZ,
                    "sampling_frequency_hz": compact.ROW_HZ,
                    "label_available": True,
                    "targets": {
                        "arousal": float(columns["arousal"][local_index]),
                        "valence": float(columns["valence"][local_index]),
                    },
                    "arousal": float(columns["arousal"][local_index]),
                    "valence": float(columns["valence"][local_index]),
                    "quality_exclusion_flag": bool(columns["quality_exclusion_flag"][local_index]),
                    "quality_black_frame_flag": bool(columns["quality_black_frame_flag"][local_index]),
                    "quality_duplicate_frame_flag": bool(columns["quality_duplicate_frame_flag"][local_index]),
                    "quality_weight_suggested": float(columns["quality_weight_suggested"][local_index]),
                    "media_path": block.identity.media_path,
                    "source_annotation": dict(block.identity.source_annotation),
                    "source_video_sha256": block.provenance.video_sha256,
                    "row_plan_sha256": block.provenance.row_plan_sha256,
                    "model_sha256": block.provenance.model_sha256,
                }
            )
            global_index += 1
    frame = pd.DataFrame(
        {
            "video_id": [row["video_id"] for row in rows],
            "row_index": [row["row_index"] for row in rows],
            "time_seconds": [row["time_seconds"] for row in rows],
            "label_available": [row["label_available"] for row in rows],
            "arousal": [row["arousal"] for row in rows],
            "valence": [row["valence"] for row in rows],
            "quality_exclusion_flag": [row["quality_exclusion_flag"] for row in rows],
        }
    )
    frame.index = np.arange(len(frame), dtype=np.int64)
    diagnostic_matrix = np.concatenate(diagnostics, axis=0).astype(np.float32, copy=False)
    if len(frame) != compact.EXPECTED_TOTAL_ROWS or diagnostic_matrix.shape != (len(frame), 53):
        raise RuntimeError("Dense table assembly did not preserve the sealed cache row contract")
    duplicated = frame.duplicated(subset=["video_id", "row_index", "time_seconds"])
    if duplicated.any():
        raise RuntimeError("Dense table contains duplicate row identities")
    return rows, frame, diagnostic_matrix


def _target_arrays(
    augmented_rows: Sequence[Mapping[str, Any]], target_name: str
) -> tuple[np.ndarray, np.ndarray]:
    values = np.full(len(augmented_rows), np.nan, dtype=np.float32)
    valid = np.zeros(len(augmented_rows), dtype=bool)
    mask_name = f"target_mask_{target_name}"
    for index, row in enumerate(augmented_rows):
        masks = row["target_masks"]
        if not bool(masks.get(mask_name, False)):
            continue
        value = row["targets"].get(target_name)
        if value is None or not math.isfinite(float(value)):
            continue
        values[index] = float(value)
        valid[index] = True
    return values, valid


def _target_axis(target_name: str) -> str:
    return "arousal" if target_name.startswith("future_arousal") else "valence"


def _metric_orientation(
    target_name: str,
    train_values: np.ndarray,
    test_values: np.ndarray,
    prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orient every scored endpoint so larger means a stronger future event."""

    if target_name == targets.VALENCE_FUTURE_SIGNED_DROP:
        return -train_values, -test_values, -prediction
    return train_values, test_values, prediction


def _fit_or_load_quality_safe_pca(
    *,
    accessor: phase4.CorticalVariantAccessor,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    quality_valid: np.ndarray,
    output_root: Path,
    name: str,
    seed: int,
) -> zero_label.PcaView:
    """Fit PCA on quality-valid outer training rows and transform every row.

    Excluded rows are transformed solely so a later valid row can retain its
    complete causal history.  They never participate in PCA fitting, supervised
    loss, target-threshold fitting, or scoring.
    """

    fit_idx = np.asarray(train_idx, dtype=np.int64)[quality_valid[train_idx]]
    all_idx = np.concatenate([train_idx, test_idx]).astype(np.int64)
    if not len(fit_idx) or len(set(train_idx.tolist()) & set(test_idx.tolist())):
        raise RuntimeError(f"Invalid quality-safe PCA ownership for {name}")
    result = phase4.streaming_randomized_pca_fit(
        accessor,
        fit_idx,
        all_idx,
        width=program.PCA_WIDTH,
        seed=int(seed),
        output_root=output_root,
        fit_key=name,
        batch_size=384,
        oversampling=32,
        power_iterations=1,
    )
    backend = str(result.metadata.get("pca_backend", ""))
    if "mlx_gpu" not in backend:
        raise RuntimeError(f"PCA {name} did not use MLX GPU matmul: {backend}")
    if result.metadata.get("train_idx_digest") != phase4.array_digest(fit_idx):
        raise RuntimeError(f"PCA {name} quality-valid fit-row digest mismatch")
    scores = np.asarray(result.scores)
    if tuple(scores.shape) != (len(all_idx), program.PCA_WIDTH):
        raise RuntimeError(f"PCA {name} score shape mismatch: {scores.shape}")
    metadata = {
        **dict(result.metadata),
        "fit_quality_valid_only": True,
        "fit_row_count": int(len(fit_idx)),
        "fit_row_digest": phase4.array_digest(fit_idx),
        "transformed_excluded_rows_for_causal_history_only": True,
        "transform_row_digest": phase4.array_digest(all_idx),
    }
    atomic_json(output_root / f"{name}_quality_audit.json", metadata)
    return zero_label.PcaView(
        name=name,
        train_idx=np.asarray(train_idx, dtype=np.int64).copy(),
        test_idx=np.asarray(test_idx, dtype=np.int64).copy(),
        train_scores=np.asarray(scores[: len(train_idx)], dtype=np.float32),
        test_scores=np.asarray(scores[len(train_idx) :], dtype=np.float32),
        component_path=Path(result.component_path),
        score_path=Path(result.score_path),
        metadata=metadata,
    )


def _control_features(
    train_features: zero_label.VideoFeatures,
    test_features: zero_label.VideoFeatures,
    *,
    fold: int,
    target_name: str,
    seed: int,
    train_fit_mask: np.ndarray,
) -> tuple[Mapping[str, tuple[np.ndarray, np.ndarray]], Mapping[str, Any]]:
    zero_shuffled_train, zero_train_map = program.whole_video_reassignment(
        train_features.x_temporal,
        train_features.video_id,
        namespace=f"veatic21|fold{fold}|{target_name}|sequence_shuffle|train",
    )
    zero_shuffled_test, zero_test_map = program.whole_video_reassignment(
        test_features.x_temporal,
        test_features.video_id,
        namespace=f"veatic21|fold{fold}|{target_name}|sequence_shuffle|test",
    )
    no_video_train = train_features.x_temporal.copy()
    no_video_test = test_features.x_temporal.copy()
    no_video_train[:, :VIDEO_BLOCK_WIDTH] = 0.0
    no_video_test[:, :VIDEO_BLOCK_WIDTH] = 0.0

    shuffled_pca_train, shuffled_pca_train_map = program.whole_video_reassignment(
        train_features.x_temporal[:, :PCA_SEQUENCE_WIDTH],
        train_features.video_id,
        namespace=f"veatic21|fold{fold}|{target_name}|pca_shuffle|train",
    )
    shuffled_pca_test, shuffled_pca_test_map = program.whole_video_reassignment(
        test_features.x_temporal[:, :PCA_SEQUENCE_WIDTH],
        test_features.video_id,
        namespace=f"veatic21|fold{fold}|{target_name}|pca_shuffle|test",
    )
    shuffled_train = train_features.x_temporal.copy()
    shuffled_test = test_features.x_temporal.copy()
    shuffled_train[:, :PCA_SEQUENCE_WIDTH] = shuffled_pca_train
    shuffled_test[:, :PCA_SEQUENCE_WIDTH] = shuffled_pca_test

    random_train_part, random_test_part, _ = program.train_matched_random_features(
        train_features.x_temporal[:, :PCA_SEQUENCE_WIDTH],
        test_features.x_temporal[:, :PCA_SEQUENCE_WIDTH],
        seed=int(seed) + int(fold) * 1009,
    )
    random_train = train_features.x_temporal.copy()
    random_test = test_features.x_temporal.copy()
    random_train[:, :PCA_SEQUENCE_WIDTH] = random_train_part
    random_test[:, :PCA_SEQUENCE_WIDTH] = random_test_part

    diagnostics_train = train_features.x_temporal.copy()
    diagnostics_test = test_features.x_temporal.copy()
    diagnostics_train[:, :PCA_SEQUENCE_WIDTH] = 0.0
    diagnostics_test[:, :PCA_SEQUENCE_WIDTH] = 0.0

    current_slice = slice(PCA_SEQUENCE_WIDTH - program.PCA_WIDTH, PCA_SEQUENCE_WIDTH)
    train_current = train_features.x_temporal[:, current_slice]
    valid_fit = np.asarray(train_fit_mask, dtype=bool)
    if valid_fit.shape != (len(train_current),) or not np.any(valid_fit):
        raise RuntimeError("Train-only video-mean control has no quality-valid fit rows")
    global_mean = np.mean(train_current[valid_fit], axis=0, dtype=np.float64).astype(np.float32)
    video_means: dict[str, np.ndarray] = {}
    for video_id in np.unique(train_features.video_id):
        rows = (train_features.video_id == video_id) & valid_fit
        video_means[str(video_id)] = (
            np.mean(train_current[rows], axis=0, dtype=np.float64).astype(np.float32)
            if np.any(rows)
            else global_mean
        )
    mean_train = train_features.x_temporal.copy()
    mean_test = test_features.x_temporal.copy()
    for lag in range(program.WINDOW_ROWS):
        block = slice(lag * program.PCA_WIDTH, (lag + 1) * program.PCA_WIDTH)
        mean_train[:, block] = np.vstack(
            [video_means[str(video_id)] for video_id in train_features.video_id]
        )
        mean_test[:, block] = global_mean

    features = {
        "zero_label_shuffled": (zero_shuffled_train, zero_shuffled_test),
        "shuffled_pca": (shuffled_train, shuffled_test),
        "no_video": (no_video_train, no_video_test),
        "random_pca": (random_train, random_test),
        "train_only_video_mean": (mean_train, mean_test),
        "diagnostics_only": (diagnostics_train, diagnostics_test),
    }
    audit = {
        "zero_label_sequence_shuffle": {
            "train_mapping": zero_train_map,
            "test_mapping": zero_test_map,
            "train_digest": program.array_digest(zero_shuffled_train),
            "test_digest": program.array_digest(zero_shuffled_test),
        },
        "pca_sequence_shuffle": {
            "train_mapping": shuffled_pca_train_map,
            "test_mapping": shuffled_pca_test_map,
            "train_digest": program.array_digest(shuffled_train),
            "test_digest": program.array_digest(shuffled_test),
            "diagnostics_retained": True,
        },
        "random_pca": {
            "seed": int(seed) + int(fold) * 1009,
            "train_digest": program.array_digest(random_train),
            "test_digest": program.array_digest(random_test),
            "diagnostics_retained": True,
        },
        "train_only_video_mean": {
            "uses_test_rows_for_mean": False,
            "global_mean_digest": program.array_digest(global_mean),
            "per_video_mean_digests": {
                video_id: program.array_digest(value) for video_id, value in video_means.items()
            },
            "train_digest": program.array_digest(mean_train),
            "test_digest": program.array_digest(mean_test),
        },
        "diagnostics_only": {
            "pca_zeroed": True,
            "diagnostics_retained": True,
            "train_digest": program.array_digest(diagnostics_train),
            "test_digest": program.array_digest(diagnostics_test),
        },
        "no_video": {
            "pca_and_diagnostics_zeroed": True,
            "time_and_history_mask_retained": True,
            "train_digest": program.array_digest(no_video_train),
            "test_digest": program.array_digest(no_video_test),
        },
    }
    return features, audit


def _prediction_paths(
    root: Path, target_name: str, lane: str, seed: int
) -> tuple[Path, Path, Path]:
    lane_root = root / "predictions" / target_name / lane
    return (
        lane_root / f"seed{seed}.npz",
        lane_root / f"seed{seed}.json",
        root / "checkpoints" / target_name / lane / f"seed{seed}.npz",
    )


def _train_or_load_scalar(
    *,
    root: Path,
    target_name: str,
    lane: str,
    seed: int,
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_target: np.ndarray,
    train_valid: np.ndarray,
    train_video_ids: np.ndarray,
    temporal_context: bool,
    inner_ownership: program.InnerVideoOwnership,
    contract_digest: str,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    prediction_path, metadata_path, checkpoint_path = _prediction_paths(
        root, target_name, lane, seed
    )
    fit_mask = np.asarray(train_valid, dtype=bool)
    if fit_mask.shape != (len(train_x),):
        raise ValueError(f"Training mask is not aligned for {target_name}/{lane}/seed{seed}")
    fit_mask_digest = program.array_digest(fit_mask.astype(np.uint8))
    inner_split_namespace = inner_ownership.namespace
    eligible_video_ids = np.asarray(train_video_ids, dtype=str)[fit_mask]
    relative_train, relative_validation = zero_label._inner_video_split(  # noqa: SLF001
        eligible_video_ids, inner_split_namespace
    )
    actual_inner_train_videos = set(eligible_video_ids[relative_train].tolist())
    actual_inner_validation_videos = set(eligible_video_ids[relative_validation].tolist())
    if actual_inner_train_videos != set(inner_ownership.inner_train_videos) or (
        actual_inner_validation_videos != set(inner_ownership.inner_validation_videos)
    ):
        raise RuntimeError(
            f"Trainer inner ownership diverged for {target_name}/{lane}/seed{seed}"
        )
    if prediction_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("contract_digest") != contract_digest:
            raise RuntimeError(f"Existing prediction contract mismatch: {prediction_path}")
        if metadata.get("inner_split_namespace") != inner_split_namespace:
            raise RuntimeError(f"Existing prediction inner split mismatch: {prediction_path}")
        if metadata.get("inner_ownership_digest") != inner_ownership.digest:
            raise RuntimeError(f"Existing prediction ownership digest mismatch: {prediction_path}")
        if metadata.get("training_fit_mask_digest") != fit_mask_digest:
            raise RuntimeError(f"Existing prediction fit-mask mismatch: {prediction_path}")
        if metadata.get("prediction_file_sha256") != file_sha256(prediction_path):
            raise RuntimeError(f"Existing prediction file checksum mismatch: {prediction_path}")
        stored_checkpoint = Path(str(metadata.get("checkpoint_path", "")))
        if not stored_checkpoint.exists() or file_sha256(stored_checkpoint) != metadata.get(
            "checkpoint_sha256"
        ):
            raise RuntimeError(f"Existing checkpoint checksum mismatch: {stored_checkpoint}")
        with np.load(prediction_path, allow_pickle=False) as bundle:
            train_prediction = np.asarray(bundle["train_prediction"], dtype=np.float32)
            test_prediction = np.asarray(bundle["test_prediction"], dtype=np.float32)
        if train_prediction.shape != (len(train_x),) or test_prediction.shape != (len(test_x),):
            raise RuntimeError(f"Existing prediction shape mismatch: {prediction_path}")
        if not np.isfinite(train_prediction).all() or not np.isfinite(test_prediction).all():
            raise RuntimeError(f"Existing prediction contains non-finite values: {prediction_path}")
        if metadata.get("train_prediction_digest") != program.array_digest(train_prediction):
            raise RuntimeError(f"Existing train prediction digest mismatch: {prediction_path}")
        if metadata.get("test_prediction_digest") != program.array_digest(test_prediction):
            raise RuntimeError(f"Existing test prediction digest mismatch: {prediction_path}")
        return train_prediction, test_prediction, metadata
    masked_train_x = np.asarray(train_x, dtype=np.float32).copy()
    masked_train_x[~fit_mask] = np.nan
    result = zero_label.train_scalar_model(
        train_x=masked_train_x,
        test_x=test_x,
        train_target=train_target,
        train_loss_mask=train_valid,
        train_video_id=train_video_ids,
        temporal_context=temporal_context,
        seed=int(seed),
        checkpoint_path=checkpoint_path,
        namespace=inner_split_namespace,
        weighted_huber=True,
        batch_size=int(batch_size),
        max_epochs=int(max_epochs),
        patience=int(patience),
    )
    train_prediction = result.train_prediction.astype(np.float32)
    test_prediction = result.test_prediction.astype(np.float32)
    metadata = {
        "schema_version": RUN_SCHEMA_VERSION,
        "target": target_name,
        "lane": lane,
        "seed": int(seed),
        "contract_digest": contract_digest,
        "checkpoint_path": str(result.checkpoint_path),
        "checkpoint_sha256": result.checkpoint_sha256,
        "inner_split_namespace": inner_split_namespace,
        "inner_ownership_digest": inner_ownership.digest,
        "training_fit_mask_digest": fit_mask_digest,
        "standardization_fit_scope": "eligible_outer_training_rows_only",
        "train_prediction_digest": program.array_digest(train_prediction),
        "test_prediction_digest": program.array_digest(test_prediction),
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "finished_at": utc_now(),
    }
    atomic_npz(prediction_path, train_prediction=train_prediction, test_prediction=test_prediction)
    metadata["prediction_file_sha256"] = file_sha256(prediction_path)
    atomic_json(metadata_path, metadata)
    return train_prediction, test_prediction, metadata


def _score_row(
    *,
    target_name: str,
    lane: str,
    fold: int,
    seed_or_group: str,
    row_type: str,
    split_digest: str,
    train_values: np.ndarray,
    test_values: np.ndarray,
    prediction: np.ndarray,
    test_times: np.ndarray,
    checkpoint_sha256: str,
    frozen_ar_digest: str,
) -> dict[str, Any]:
    oriented_train, oriented_test, oriented_prediction = _metric_orientation(
        target_name,
        np.asarray(train_values),
        np.asarray(test_values),
        np.asarray(prediction),
    )
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "target": target_name,
        "target_axis": _target_axis(target_name),
        "lane": lane,
        "fold": int(fold),
        "seed_or_group": str(seed_or_group),
        "row_type": row_type,
        "split_digest": split_digest,
        "train_rows": int(len(train_values)),
        "test_rows": int(len(test_values)),
        "checkpoint_sha256": checkpoint_sha256,
        "frozen_ar_prediction_digest": frozen_ar_digest,
        **program.score_prediction(
            train_values=oriented_train,
            test_values=oriented_test,
            prediction=oriented_prediction,
            time_seconds=test_times,
        ),
    }


def _matrix_audit(
    score_frame: pd.DataFrame,
    *,
    expected_targets: Sequence[str],
    expected_fold_ids: Sequence[int],
    seeds: Sequence[int],
) -> Mapping[str, Any]:
    member_seeds = tuple(str(int(seed)) for seed in seeds)
    ensemble_group = "_".join(member_seeds)
    expected_member = {
        (str(target), int(fold), str(lane), seed, "member")
        for target in expected_targets
        for fold in expected_fold_ids
        for lane in ALL_LANES
        for seed in member_seeds
    }
    expected_ensemble = {
        (str(target), int(fold), str(lane), ensemble_group, "ensemble")
        for target in expected_targets
        for fold in expected_fold_ids
        for lane in ALL_LANES
    }
    expected = expected_member | expected_ensemble
    key_columns = ["target", "fold", "lane", "seed_or_group", "row_type"]
    actual = {
        (str(row.target), int(row.fold), str(row.lane), str(row.seed_or_group), str(row.row_type))
        for row in score_frame[key_columns].itertuples(index=False)
    }
    duplicate_rows = int(score_frame.duplicated(subset=key_columns).sum())
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    return {
        "matrix_complete": bool(not missing and not unexpected and duplicate_rows == 0),
        "expected_rows": int(len(expected)),
        "actual_rows": int(len(score_frame)),
        "duplicate_rows": duplicate_rows,
        "missing_cells": [list(cell) for cell in missing],
        "unexpected_cells": [list(cell) for cell in unexpected],
    }


def _summary(
    score_frame: pd.DataFrame,
    *,
    expected_targets: Sequence[str],
    expected_fold_ids: Sequence[int],
    seeds: Sequence[int],
) -> Mapping[str, Any]:
    matrix_audit = _matrix_audit(
        score_frame,
        expected_targets=expected_targets,
        expected_fold_ids=expected_fold_ids,
        seeds=seeds,
    )
    ensembles = score_frame[score_frame["row_type"] == "ensemble"].copy()
    if not matrix_audit["matrix_complete"]:
        return {
            "schema_version": RUN_SCHEMA_VERSION,
            **matrix_audit,
            "ensemble_rows": int(len(ensembles)),
            "targets": {},
        }
    def finite_mean(values: pd.Series) -> float | None:
        array = values.to_numpy(dtype=np.float64)
        finite = array[np.isfinite(array)]
        return float(np.mean(finite)) if len(finite) == len(array) and len(finite) else None

    def strongest_lane(
        lane_names: Sequence[str], lane_means: Mapping[str, Mapping[str, float | None]], metric: str
    ) -> str | None:
        candidates = [
            lane for lane in lane_names if lane_means.get(lane, {}).get(metric) is not None
        ]
        return (
            max(candidates, key=lambda lane: float(lane_means[lane][metric]))
            if candidates
            else None
        )

    all_metrics = METRICS + FIRST30_METRICS
    required_wins = max(1, len(expected_fold_ids) - 1)
    target_summaries: dict[str, Any] = {}
    for target_name, target_rows in ensembles.groupby("target", sort=False):
        member_rows = score_frame[
            (score_frame["target"] == target_name) & (score_frame["row_type"] == "member")
        ]
        lane_means = {
            lane: {metric: finite_mean(rows[metric]) for metric in all_metrics}
            for lane, rows in target_rows.groupby("lane", sort=False)
        }
        strongest_zero = {
            metric: strongest_lane(ZERO_LABEL_CONTROLS, lane_means, metric)
            for metric in all_metrics
        }
        strongest_ar_control = {
            metric: strongest_lane(AR_CONTROLS, lane_means, metric)
            for metric in all_metrics
        }
        zero_deltas: dict[str, float | None] = {}
        ar_deltas: dict[str, float | None] = {}
        fold_wins_zero: dict[str, int] = {}
        fold_wins_ar: dict[str, int] = {}
        fold_comparisons_zero: dict[str, int] = {}
        fold_comparisons_ar: dict[str, int] = {}
        for metric in all_metrics:
            zero_control = strongest_zero[metric]
            ar_control_lane = strongest_ar_control[metric]
            zero_real_mean = lane_means["video_supervised_temporal"][metric]
            ar_real_mean = lane_means["ar_plus_temporal_residual"][metric]
            ar_base_mean = lane_means["frozen_ar"][metric]
            zero_control_mean = (
                lane_means[zero_control][metric] if zero_control is not None else None
            )
            ar_control_mean = (
                lane_means[ar_control_lane][metric] if ar_control_lane is not None else None
            )
            zero_deltas[metric] = (
                float(zero_real_mean - zero_control_mean)
                if zero_real_mean is not None and zero_control_mean is not None
                else None
            )
            ar_deltas[metric] = (
                float(ar_real_mean - max(ar_base_mean, ar_control_mean))
                if ar_real_mean is not None
                and ar_base_mean is not None
                and ar_control_mean is not None
                else None
            )
            zero_wins = 0
            ar_wins = 0
            zero_comparisons = 0
            ar_comparisons = 0
            for fold_id in expected_fold_ids:
                def fold_value(lane: str | None) -> float | None:
                    if lane is None:
                        return None
                    values = target_rows[
                        (target_rows["lane"] == lane) & (target_rows["fold"] == int(fold_id))
                    ][metric].to_numpy(dtype=np.float64)
                    if len(values) != 1 or not np.isfinite(values[0]):
                        return None
                    return float(values[0])

                zero_real = fold_value("video_supervised_temporal")
                zero_control_value = fold_value(zero_control)
                if zero_real is not None and zero_control_value is not None:
                    zero_comparisons += 1
                    zero_wins += int(zero_real > zero_control_value)
                ar_real = fold_value("ar_plus_temporal_residual")
                ar_base = fold_value("frozen_ar")
                ar_control_value = fold_value(ar_control_lane)
                if ar_real is not None and ar_base is not None and ar_control_value is not None:
                    ar_comparisons += 1
                    ar_wins += int(ar_real > max(ar_base, ar_control_value))
            fold_wins_zero[metric] = zero_wins
            fold_wins_ar[metric] = ar_wins
            fold_comparisons_zero[metric] = zero_comparisons
            fold_comparisons_ar[metric] = ar_comparisons

        ensemble_uplift: dict[str, dict[str, float | None]] = {}
        for candidate_lane in ("video_supervised_temporal", "ar_plus_temporal_residual"):
            candidate_members = member_rows[member_rows["lane"] == candidate_lane]
            ensemble_uplift[candidate_lane] = {}
            for metric in all_metrics:
                member_mean = finite_mean(candidate_members[metric])
                ensemble_mean = lane_means[candidate_lane][metric]
                ensemble_uplift[candidate_lane][metric] = (
                    float(ensemble_mean - member_mean)
                    if ensemble_mean is not None and member_mean is not None
                    else None
                )

        def metric_gate(
            metrics: Sequence[str],
            deltas: Mapping[str, float | None],
            wins: Mapping[str, int],
            comparisons: Mapping[str, int],
        ) -> bool:
            return bool(
                all(deltas[metric] is not None and float(deltas[metric]) > 0.0 for metric in metrics)
                and all(wins[metric] >= required_wins for metric in metrics)
                and all(comparisons[metric] == len(expected_fold_ids) for metric in metrics)
            )

        zero_pooled_pass = metric_gate(
            METRICS, zero_deltas, fold_wins_zero, fold_comparisons_zero
        )
        zero_cold_pass = metric_gate(
            FIRST30_METRICS, zero_deltas, fold_wins_zero, fold_comparisons_zero
        )
        ar_pooled_pass = metric_gate(METRICS, ar_deltas, fold_wins_ar, fold_comparisons_ar)
        ar_first30_pass = metric_gate(
            FIRST30_METRICS, ar_deltas, fold_wins_ar, fold_comparisons_ar
        )
        zero_ensemble_pass = all(
            ensemble_uplift["video_supervised_temporal"][metric] is not None
            and float(ensemble_uplift["video_supervised_temporal"][metric]) > 0.0
            for metric in METRICS
        )
        ar_ensemble_pass = all(
            ensemble_uplift["ar_plus_temporal_residual"][metric] is not None
            and float(ensemble_uplift["ar_plus_temporal_residual"][metric]) > 0.0
            for metric in METRICS
        )
        target_summaries[str(target_name)] = {
            "lane_means": lane_means,
            "strongest_zero_label_control": strongest_zero,
            "strongest_ar_control": strongest_ar_control,
            "zero_label_deltas": zero_deltas,
            "ar_residual_deltas": ar_deltas,
            "zero_label_fold_wins": fold_wins_zero,
            "ar_residual_fold_wins": fold_wins_ar,
            "zero_label_fold_comparisons": fold_comparisons_zero,
            "ar_residual_fold_comparisons": fold_comparisons_ar,
            "ensemble_uplift_over_members": ensemble_uplift,
            "zero_label_pooled_pass": zero_pooled_pass,
            "zero_label_first30_cold_start_pass": zero_cold_pass,
            "zero_label_ensemble_uplift_pass": zero_ensemble_pass,
            "ar_residual_pooled_pass": ar_pooled_pass,
            "ar_residual_first30_pass": ar_first30_pass,
            "ar_residual_ensemble_uplift_pass": ar_ensemble_pass,
            "zero_label_development_pass": bool(
                zero_pooled_pass and zero_cold_pass and zero_ensemble_pass
            ),
            "ar_residual_development_pass": bool(
                ar_pooled_pass and ar_first30_pass and ar_ensemble_pass
            ),
        }
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        **matrix_audit,
        "ensemble_rows": int(len(ensembles)),
        "targets": target_summaries,
    }


def _report_text(summary: Mapping[str, Any]) -> str:
    lines = [
        "# VEATIC 2.1 Distilled Development",
        "",
        f"Matrix complete: `{summary['matrix_complete']}`",
        "",
    ]
    for target_name, result in summary["targets"].items():
        lines.extend(
            [
                f"## `{target_name}`",
                "",
                f"- Zero-label development pass: `{result['zero_label_development_pass']}`",
                f"- Zero-label pooled pass: `{result['zero_label_pooled_pass']}`",
                f"- Zero-label first-30-second pass: `{result['zero_label_first30_cold_start_pass']}`",
                f"- Zero-label ensemble uplift pass: `{result['zero_label_ensemble_uplift_pass']}`",
                f"- Privileged AR-residual development pass: `{result['ar_residual_development_pass']}`",
                f"- Privileged pooled pass: `{result['ar_residual_pooled_pass']}`",
                f"- Privileged first-30-second pass: `{result['ar_residual_first30_pass']}`",
                f"- Privileged ensemble uplift pass: `{result['ar_residual_ensemble_uplift_pass']}`",
                f"- Zero-label deltas: `{json.dumps(result['zero_label_deltas'], sort_keys=True)}`",
                f"- AR-residual deltas: `{json.dumps(result['ar_residual_deltas'], sort_keys=True)}`",
                f"- Zero-label fold wins: `{json.dumps(result['zero_label_fold_wins'], sort_keys=True)}`",
                f"- AR-residual fold wins: `{json.dumps(result['ar_residual_fold_wins'], sort_keys=True)}`",
                "",
            ]
        )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    shared_derived_root = (
        args.shared_derived_root.expanduser().resolve()
        if args.shared_derived_root is not None
        else output_root / "derived_substrate"
    )
    shared_derived_root.mkdir(parents=True, exist_ok=True)
    source = compact.Veatic21CompactCache(
        args.cache_root,
        upstream_root=args.upstream_root,
        identity_manifest_path=args.identity_manifest,
        verify_checksums=not args.skip_checksums,
    )
    cache_report = source.validate()
    atomic_json(output_root / "cache_validation.json", cache_report.to_dict())

    # Avoid re-hashing gigabytes while assembling rows after the full seal above.
    row_source = compact.Veatic21CompactCache(
        args.cache_root,
        upstream_root=args.upstream_root,
        identity_manifest_path=args.identity_manifest,
        verify_checksums=False,
    )
    source_rows, frame, diagnostics = _dense_rows_and_diagnostics(row_source)
    row_counts = {str(key): int(value) for key, value in frame.groupby("video_id").size().items()}
    panel = program.deterministic_video_panel(
        row_counts, reserved_count=0
    )
    folds = program.balanced_grouped_video_folds(
        row_counts,
        panel.development_videos,
        fold_count=int(args.folds),
    )
    augmented = targets.build_veatic21_targets(
        source_rows,
        build_events=False,
    )
    selected_targets = tuple(args.targets)
    unknown = sorted(set(selected_targets) - set(targets.CONTINUOUS_TARGET_NAMES))
    if unknown:
        raise ValueError(f"Unknown or non-continuous targets: {unknown}")
    seeds = tuple(int(seed) for seed in args.seeds)
    selected_folds = folds
    if args.smoke:
        selected_targets = selected_targets[:1]
        seeds = seeds[:1]
        selected_folds = folds[:1]
    run_contract = {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": utc_now(),
        "cache": cache_report.to_dict(),
        "cache_checksums_verified": bool(not args.skip_checksums),
        "implementation_sha256": {
            "runner": file_sha256(Path(__file__).resolve()),
            "compact_cache": file_sha256(Path(compact.__file__).resolve()),
            "targets": file_sha256(Path(targets.__file__).resolve()),
            "program": file_sha256(Path(program.__file__).resolve()),
            "again_phase4_pca": file_sha256(Path(phase4.__file__).resolve()),
            "again_zero_label_temporal": file_sha256(Path(zero_label.__file__).resolve()),
        },
        "program": dict(program.contract_manifest()),
        "targets": augmented.contract,
        "panel": asdict(panel),
        "folds": [asdict(fold) for fold in folds],
        "selected_targets": list(selected_targets),
        "selected_folds": [fold.fold for fold in selected_folds],
        "seeds": list(seeds),
        "lanes": list(ALL_LANES),
        "pca_family": str(args.pca_family),
        "pca_seed": PCA_SEED,
        "temporal_head": str(args.temporal_head),
        "shared_derived_root": str(shared_derived_root),
        "quality_policy": {
            "pca_fit": "outer-training rows with quality_exclusion_flag=false only",
            "supervised_standardization_and_loss": "eligible outer-training rows with quality_exclusion_flag=false only",
            "scoring": "eligible held-out rows with quality_exclusion_flag=false only",
            "excluded_row_transform": "permitted only as frozen causal history for a later valid row",
        },
        "frozen_ar_policy": {
            "model_family": "target/fold/seed-specific MLX neural AR7",
            "features": "same-axis current, lag1/2/4, current-minus-lag1/2/4",
            "common_inner_video_ownership": True,
            "exact_identity_reused_across_real_and_all_matched_controls": True,
            "ridge_reference_promotable": False,
        },
        "event_metric_threshold_policy": (
            "refit q90 from eligible outer-training continuous targets for every fold; "
            "inventory-level augmented event labels are not used for model fitting or scoring"
        ),
        "internal_reserved_video_count": 0,
        "all_veatic_videos_enter_grouped_cross_validation": True,
        "external_zero_label_confirmation_pending": True,
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "smoke": bool(args.smoke),
    }
    contract_digest = program.canonical_digest(stable_contract_identity(run_contract))
    run_contract["contract_digest"] = contract_digest
    contract_path = output_root / "run_contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        existing_digest = program.canonical_digest(stable_contract_identity(existing))
        if existing_digest != contract_digest:
            raise RuntimeError("Output root already contains a different run contract")
    else:
        atomic_json(contract_path, run_contract)
    if args.dry_run or args.audit_only:
        return run_contract

    zero_label.require_mlx_gpu()
    cortical = phase4.load_or_build_cortical_memmap(
        args.cache_root.expanduser().resolve(),
        frame,
        output_root=shared_derived_root / "cortical_memmap",
    )
    accessor = phase4.CorticalVariantAccessor(
        cortical, frame, base_family=str(args.pca_family)
    )
    quality_valid = ~frame["quality_exclusion_flag"].to_numpy(dtype=bool)
    all_scores: list[dict[str, Any]] = []

    for fold in selected_folds:
        fold_root = output_root / f"fold{fold.fold}"
        train_idx = program.indices_for_videos(
            frame["video_id"].to_numpy(dtype=str), fold.train_videos
        )
        test_idx = program.indices_for_videos(
            frame["video_id"].to_numpy(dtype=str), fold.test_videos
        )
        if set(frame.loc[train_idx, "video_id"].astype(str)) & set(
            frame.loc[test_idx, "video_id"].astype(str)
        ):
            raise RuntimeError("Outer grouped fold has video overlap")
        if panel.reserved_videos:
            raise RuntimeError("Internal VEATIC reservation is forbidden for the all-video CV run")
        pca_view = _fit_or_load_quality_safe_pca(
            accessor=accessor,
            train_idx=train_idx,
            test_idx=test_idx,
            quality_valid=quality_valid,
            output_root=(
                shared_derived_root / "pca" / str(args.pca_family) / f"fold{fold.fold}"
            ),
            name=f"veatic21_dev_fold{fold.fold}_{args.pca_family}",
            seed=PCA_SEED,
        )
        train_features = zero_label.build_video_features(
            frame,
            train_idx,
            pca_view.train_scores,
            diagnostics[train_idx],
        )
        test_features = zero_label.build_video_features(
            frame,
            test_idx,
            pca_view.test_scores,
            diagnostics[test_idx],
        )
        forbidden_response_tokens = (
            "arousal",
            "valence",
            "response",
            "teacher",
            "target",
            "label",
        )
        offending_features = sorted(
            name
            for name in train_features.feature_names
            if any(token in name.lower() for token in forbidden_response_tokens)
        )
        if offending_features:
            raise RuntimeError(
                f"Zero-label feature schema contains forbidden response inputs: {offending_features}"
            )
        atomic_json(
            fold_root / "audit" / "zero_label_inference_inputs.json",
            {
                "schema_version": RUN_SCHEMA_VERSION,
                "contract_digest": contract_digest,
                "fold": int(fold.fold),
                "feature_names": list(train_features.feature_names),
                "feature_name_digest": program.canonical_digest(
                    list(train_features.feature_names)
                ),
                "forbidden_response_tokens": list(forbidden_response_tokens),
                "offending_features": offending_features,
                "uses_observed_arousal": False,
                "uses_observed_valence": False,
                "uses_response_history": False,
                "uses_teacher_score": False,
                "uses_labels_at_inference": False,
                "prediction_starts_at_row0": True,
                "missing_history_policy": "zero_pca_history_plus_explicit_history_mask",
                "pass": True,
            },
        )

        for target_name in selected_targets:
            values, target_valid = _target_arrays(augmented.rows, target_name)
            train_valid = target_valid[train_idx] & quality_valid[train_idx]
            test_valid = target_valid[test_idx] & quality_valid[test_idx]
            if np.count_nonzero(train_valid) < 100 or np.count_nonzero(test_valid) < 50:
                raise RuntimeError(f"Insufficient valid rows for fold {fold.fold} target {target_name}")
            axis = _target_axis(target_name)
            signal = frame[axis].to_numpy(dtype=np.float32)
            ar_features_all, ar_context_valid, ar_history_audit = (
                program.canonical_ar_history_features(
                signal,
                frame["video_id"].to_numpy(dtype=str),
                )
            )
            privileged_train_valid = train_valid & ar_context_valid[train_idx]
            privileged_test_valid = test_valid & ar_context_valid[test_idx]
            if np.count_nonzero(privileged_train_valid) < 100 or np.count_nonzero(
                privileged_test_valid
            ) < 50:
                raise RuntimeError(
                    f"Insufficient full-context AR rows for fold {fold.fold} target {target_name}"
                )
            train_target = values[train_idx]
            test_target = values[test_idx]
            member_predictions: dict[str, list[np.ndarray]] = {lane: [] for lane in ALL_LANES}
            member_checkpoint_hashes: dict[str, list[str]] = {lane: [] for lane in ALL_LANES}
            member_frozen_ar_digests: list[str] = []

            for seed in seeds:
                use_temporal_conv = str(args.temporal_head) == "short_conv"
                ownership = program.build_member_inner_video_ownership(
                    train_features.video_id,
                    outer_fold=fold.fold,
                    target_name=target_name,
                    seed=seed,
                )
                inner_split_namespace = ownership.namespace
                frozen_ar_train, frozen_ar_test, frozen_ar_metadata = _train_or_load_scalar(
                    root=fold_root,
                    target_name=target_name,
                    lane="frozen_ar",
                    seed=seed,
                    train_x=ar_features_all[train_idx],
                    test_x=ar_features_all[test_idx],
                    train_target=train_target,
                    train_valid=privileged_train_valid,
                    train_video_ids=train_features.video_id,
                    temporal_context=False,
                    inner_ownership=ownership,
                    contract_digest=contract_digest,
                    batch_size=args.batch_size,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                )
                frozen_ar_identity = program.frozen_ar_prediction_identity(
                    ownership=ownership,
                    outer_fold=fold.fold,
                    target_name=target_name,
                    seed=seed,
                    model_family="mlx_target_specific_neural_ar7",
                    checkpoint_digest=str(frozen_ar_metadata["checkpoint_sha256"]),
                    train_prediction=frozen_ar_train,
                    test_prediction=frozen_ar_test,
                )
                frozen_ar_digest = str(frozen_ar_identity["identity_digest"])
                member_frozen_ar_digests.append(frozen_ar_digest)
                atomic_json(
                    fold_root / "audit" / target_name / f"seed{seed}_frozen_ar_identity.json",
                    {
                        "schema_version": RUN_SCHEMA_VERSION,
                        "contract_digest": contract_digest,
                        "ownership": ownership.audit_manifest(),
                        "ar_feature_contract": ar_history_audit,
                        "frozen_ar_identity": frozen_ar_identity,
                        "checkpoint": dict(frozen_ar_metadata),
                        "train_valid_digest": program.array_digest(
                            privileged_train_valid.astype(np.uint8)
                        ),
                        "test_valid_digest": program.array_digest(
                            privileged_test_valid.astype(np.uint8)
                        ),
                    },
                )
                controls, control_audit = _control_features(
                    train_features,
                    test_features,
                    fold=fold.fold,
                    target_name=target_name,
                    seed=seed,
                    train_fit_mask=quality_valid[train_idx],
                )
                permuted_target, _ = program.whole_video_reassignment(
                    train_target[:, None],
                    train_features.video_id,
                    namespace=f"veatic21|fold{fold.fold}|{target_name}|label_permutation|seed{seed}",
                )
                permuted_target = permuted_target[:, 0]
                permuted_valid = train_valid & np.isfinite(permuted_target)
                residual_target = train_target - frozen_ar_train
                permuted_residual, residual_label_mapping = program.whole_video_reassignment(
                    residual_target[:, None],
                    train_features.video_id,
                    namespace=(
                        f"veatic21|fold{fold.fold}|{target_name}|"
                        f"residual_label_permutation|seed{seed}"
                    ),
                )
                permuted_residual = permuted_residual[:, 0]
                permuted_residual_valid = privileged_train_valid & np.isfinite(
                    permuted_residual
                )
                control_audit = {
                    **dict(control_audit),
                    "video_label_permutation": {
                        "mapping": residual_label_mapping,
                        "target_digest": program.array_digest(permuted_residual),
                        "heldout_scoring_uses_true_targets": True,
                    },
                }
                atomic_json(
                    fold_root
                    / "audit"
                    / target_name
                    / f"seed{seed}_matched_controls.json",
                    control_audit,
                )
                lane_specs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]] = {
                    "video_supervised_temporal": (
                        train_features.x_temporal,
                        test_features.x_temporal,
                        train_target,
                        train_valid,
                        use_temporal_conv,
                    ),
                    "video_supervised_current_row": (
                        train_features.x_current,
                        test_features.x_current,
                        train_target,
                        train_valid,
                        False,
                    ),
                    "diagnostics_only_supervised_temporal": (
                        controls["diagnostics_only"][0],
                        controls["diagnostics_only"][1],
                        train_target,
                        train_valid,
                        use_temporal_conv,
                    ),
                    "sequence_shuffled_supervised_temporal": (
                        controls["zero_label_shuffled"][0],
                        controls["zero_label_shuffled"][1],
                        train_target,
                        train_valid,
                        use_temporal_conv,
                    ),
                    "label_permutation_supervised_temporal": (
                        train_features.x_temporal,
                        test_features.x_temporal,
                        permuted_target,
                        permuted_valid,
                        use_temporal_conv,
                    ),
                    "no_video_supervised_temporal": (
                        controls["no_video"][0],
                        controls["no_video"][1],
                        train_target,
                        train_valid,
                        use_temporal_conv,
                    ),
                    "ar_plus_temporal_residual": (
                        train_features.x_temporal,
                        test_features.x_temporal,
                        residual_target,
                        privileged_train_valid,
                        use_temporal_conv,
                    ),
                    "ar_plus_shuffled_pca_residual": (
                        controls["shuffled_pca"][0],
                        controls["shuffled_pca"][1],
                        residual_target,
                        privileged_train_valid,
                        use_temporal_conv,
                    ),
                    "ar_plus_random_pca_residual": (
                        controls["random_pca"][0],
                        controls["random_pca"][1],
                        residual_target,
                        privileged_train_valid,
                        use_temporal_conv,
                    ),
                    "ar_plus_train_only_video_mean_residual": (
                        controls["train_only_video_mean"][0],
                        controls["train_only_video_mean"][1],
                        residual_target,
                        privileged_train_valid,
                        use_temporal_conv,
                    ),
                    "ar_plus_diagnostics_only_residual": (
                        controls["diagnostics_only"][0],
                        controls["diagnostics_only"][1],
                        residual_target,
                        privileged_train_valid,
                        use_temporal_conv,
                    ),
                    "ar_plus_label_permutation_residual": (
                        train_features.x_temporal,
                        test_features.x_temporal,
                        permuted_residual,
                        permuted_residual_valid,
                        use_temporal_conv,
                    ),
                }
                for lane, (train_x, test_x, lane_target, lane_valid, temporal_context) in lane_specs.items():
                    _, lane_test, metadata = _train_or_load_scalar(
                        root=fold_root,
                        target_name=target_name,
                        lane=lane,
                        seed=seed,
                        train_x=train_x,
                        test_x=test_x,
                        train_target=lane_target,
                        train_valid=lane_valid,
                        train_video_ids=train_features.video_id,
                        temporal_context=temporal_context,
                        inner_ownership=ownership,
                        contract_digest=contract_digest,
                        batch_size=args.batch_size,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                    )
                    if lane.startswith("ar_plus_"):
                        lane_test = frozen_ar_test + lane_test
                    member_predictions[lane].append(lane_test.astype(np.float32))
                    member_checkpoint_hashes[lane].append(str(metadata["checkpoint_sha256"]))
                    if lane in VIDEO_LANES:
                        lane_train_values = train_target[train_valid]
                        lane_test_values = test_target[test_valid]
                        lane_prediction = lane_test[test_valid]
                        lane_times = test_features.time_seconds[test_valid]
                        lane_ar_digest = "not_applicable_zero_label"
                    else:
                        lane_train_values = train_target[privileged_train_valid]
                        lane_test_values = test_target[privileged_test_valid]
                        lane_prediction = lane_test[privileged_test_valid]
                        lane_times = test_features.time_seconds[privileged_test_valid]
                        lane_ar_digest = frozen_ar_digest
                    all_scores.append(
                        _score_row(
                            target_name=target_name,
                            lane=lane,
                            fold=fold.fold,
                            seed_or_group=str(seed),
                            row_type="member",
                            split_digest=fold.digest,
                            train_values=lane_train_values,
                            test_values=lane_test_values,
                            prediction=lane_prediction,
                            test_times=lane_times,
                            checkpoint_sha256=str(metadata["checkpoint_sha256"]),
                            frozen_ar_digest=lane_ar_digest,
                        )
                    )

                shared_frozen_ar_digest = program.require_shared_frozen_ar_identity(
                    {
                        lane: frozen_ar_identity
                        for lane in PRIVILEGED_LANES
                        if lane != "frozen_ar"
                    }
                )
                if shared_frozen_ar_digest != frozen_ar_digest:
                    raise RuntimeError("Frozen AR identity changed across matched residual lanes")
                member_predictions["frozen_ar"].append(frozen_ar_test.astype(np.float32))
                member_checkpoint_hashes["frozen_ar"].append(
                    str(frozen_ar_metadata["checkpoint_sha256"])
                )
                all_scores.append(
                    _score_row(
                        target_name=target_name,
                        lane="frozen_ar",
                        fold=fold.fold,
                        seed_or_group=str(seed),
                        row_type="member",
                        split_digest=fold.digest,
                        train_values=train_target[privileged_train_valid],
                        test_values=test_target[privileged_test_valid],
                        prediction=frozen_ar_test[privileged_test_valid],
                        test_times=test_features.time_seconds[privileged_test_valid],
                        checkpoint_sha256=str(frozen_ar_metadata["checkpoint_sha256"]),
                        frozen_ar_digest=frozen_ar_digest,
                    )
                )
                atomic_csv(output_root / "scores.partial.csv", pd.DataFrame(all_scores))

            group_name = "_".join(str(seed) for seed in seeds)
            frozen_ar_ensemble_digest = program.canonical_digest(member_frozen_ar_digests)
            for lane in ALL_LANES:
                predictions = member_predictions[lane]
                if len(predictions) == 1:
                    ensemble = predictions[0]
                elif len(predictions) == 3:
                    ensemble = zero_label.ensemble_predictions(predictions)
                else:
                    ensemble = np.mean(np.stack(predictions, axis=0), axis=0).astype(np.float32)
                lane_valid_mask = test_valid if lane in VIDEO_LANES else privileged_test_valid
                lane_train_values = (
                    train_target[train_valid]
                    if lane in VIDEO_LANES
                    else train_target[privileged_train_valid]
                )
                lane_test_values = test_target[lane_valid_mask]
                lane_times = test_features.time_seconds[lane_valid_mask]
                lane_ar_digest = (
                    "not_applicable_zero_label"
                    if lane in VIDEO_LANES
                    else frozen_ar_ensemble_digest
                )
                ensemble_path = (
                    fold_root
                    / "ensemble_predictions"
                    / target_name
                    / f"{lane}__seeds_{group_name}.npz"
                )
                atomic_npz(
                    ensemble_path,
                    global_row_index=test_idx.astype(np.int64),
                    video_id=test_features.video_id.astype(str),
                    time_seconds=test_features.time_seconds.astype(np.float32),
                    target=test_target.astype(np.float32),
                    target_valid=lane_valid_mask.astype(np.uint8),
                    prediction=ensemble.astype(np.float32),
                )
                atomic_json(
                    ensemble_path.with_suffix(".json"),
                    {
                        "schema_version": RUN_SCHEMA_VERSION,
                        "contract_digest": contract_digest,
                        "target": target_name,
                        "lane": lane,
                        "fold": int(fold.fold),
                        "seeds": list(seeds),
                        "member_checkpoint_hashes": member_checkpoint_hashes[lane],
                        "prediction_digest": program.array_digest(ensemble),
                        "frozen_ar_prediction_digest": lane_ar_digest,
                        "payload_sha256": file_sha256(ensemble_path),
                    },
                )
                all_scores.append(
                    _score_row(
                        target_name=target_name,
                        lane=lane,
                        fold=fold.fold,
                        seed_or_group=group_name,
                        row_type="ensemble",
                        split_digest=fold.digest,
                        train_values=lane_train_values,
                        test_values=lane_test_values,
                        prediction=ensemble[lane_valid_mask],
                        test_times=lane_times,
                        checkpoint_sha256=program.canonical_digest(member_checkpoint_hashes[lane]),
                        frozen_ar_digest=lane_ar_digest,
                    )
                )
            atomic_csv(output_root / "scores.partial.csv", pd.DataFrame(all_scores))

    score_frame = pd.DataFrame(all_scores)
    atomic_csv(output_root / "scores.csv", score_frame)
    summary = _summary(
        score_frame,
        expected_targets=selected_targets,
        expected_fold_ids=[fold.fold for fold in selected_folds],
        seeds=seeds,
    )
    failed_gates: list[str] = []
    if not summary["matrix_complete"]:
        failed_gates.append("matrix_complete")
    for target_name, target_summary in summary["targets"].items():
        if not target_summary["zero_label_development_pass"]:
            failed_gates.append(f"{target_name}:zero_label_development_pass")
        if not target_summary["ar_residual_development_pass"]:
            failed_gates.append(f"{target_name}:ar_residual_development_pass")
    if args.smoke:
        failed_gates.append("smoke_non_promotable")
    summary = {
        **summary,
        "contract_digest": contract_digest,
        "cache_dataset_fingerprint": cache_report.dataset_fingerprint_sha256,
        "internal_reserved_video_count": 0,
        "external_zero_label_confirmation_pending": True,
        "promotable": bool(not args.smoke and not failed_gates),
        "failed_gates": failed_gates,
    }
    atomic_json(output_root / "summary.json", summary)
    (output_root / "summary.md").write_text(_report_text(summary), encoding="utf-8")
    return summary


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
