"""VEATIC 2.1 Phase 05 learned frozen-AR residual bridge."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from sklearn.metrics import average_precision_score

from neural_bridge.veatic21.ar import (
    build_ar_features,
    fit_logistic_mlx,
    predict_logistic_mlx,
    spike_metrics,
)
from neural_bridge.veatic21.contracts import (
    AR_FINAL_OPTIMIZER_MAX_ITERATIONS,
    CLUSTER_BOOTSTRAP_RESAMPLES,
    CURRENT_STATE,
    EXPECTED_ROW_COUNT,
    EXPECTED_ROW_HZ,
    EXPECTED_VIDEO_IDS,
    MASTER_SPECIFICATION,
    PHASE00_ROOT,
    PHASE01_ROOT,
    PHASE02_ROOT,
    PHASE03_ROOT,
    PHASE04_ROOT,
    PHASE05_ROOT,
    REBUILD_PROTOCOL,
    REPOSITORY_ROOT,
    reject_forbidden_runtime_path,
    validate_runtime_manifest_paths,
)
from neural_bridge.veatic21.evidence import (
    paired_video_bootstrap_raw_pr_auc_delta,
    per_video_pr_auc,
    sha256_file,
    source_tree_digest,
)
from neural_bridge.veatic21.phase03 import (
    FAMILY_NAMES,
    PHASE00_ALLOWED_INPUT_MANIFEST_SHA256,
    PHASE00_RESULT_SHA256,
    PHASE01_SUBSTRATE_FILE_SHA256,
    PHASE02_MODEL_MANIFEST_SHA256,
    PHASE02_PREDICTION_MANIFEST_SHA256,
    PHASE02_RESULT_SHA256,
    PHASE02_SPLIT_MANIFEST_SHA256,
    PHASE02_SPLIT_OWNERSHIP_SHA256,
    _load_ar_model,
    _load_cells,
)
from neural_bridge.veatic21.phase04 import (
    _array_bundle_digest,
    _artifact_inventory,
    _atomic_save_npz,
    _atomic_write_csv,
    _atomic_write_json,
    _atomic_write_text,
)
from neural_bridge.veatic21.raw_cortical import within_video_label_permutation
from neural_bridge.veatic21.residual import (
    ResidualCheckpoint,
    ResidualRecipe,
    derive_phase05_seed,
    derive_residual_recipes,
    fit_residual_checkpoint_mlx,
    predict_residual_mlx,
)

PHASE02_ARTIFACT_MANIFEST_SHA256 = (
    "67263687318aa4f08f378320121e198bd4091a2c9546aa2c99958ec9789956cb"
)
PHASE02_AR_DOMINANCE_SHA256 = "21e4e081094df6b4b2b2c3e206deae44f05d501c875e89e0a189d95cc1739595"
PHASE02_SELECTED_HYPERPARAMETERS_SHA256 = (
    "44e4df31b09e36326b3e50b4369814b71be527dbe9dd2df2f2a48494ad2061b3"
)
PHASE03_RESULT_SHA256 = "8c0839d8eb8ba5c20e4c13ae83367b0fe4e0e383b7e0c3b074b80f7a5cf38c16"
PHASE04_RESULT_SHA256 = "922f181ded0b9125de43242558bd6e66113bcebf24316ca38d7767b1965f8da4"
PHASE04_CHECKSUMS_SHA256 = "bc195a879ddc2be9c37005c285b2560f69c2d4e7831b3182d6a95739f7631924"
PHASE04_PCA_ACCURACY_SHA256 = "d55ae7ebd9a4c0ea15e7307edbc6e643283aa4d1be3813e1fab8d1ddf5a28a37"
PHASE04_PROJECTION_MANIFEST_SHA256 = (
    "03cefe1a72d72021bc08ca2fcb2731d32a2deebc9fced5706407a3afb0f9ec4d"
)
PHASE04_PREDICTION_MANIFEST_SHA256 = (
    "ee3c73f873fe11fcab88fba840ceb7b6f63b5369933c877adf20668a99969623"
)
PHASE04_FINAL_MODEL_MANIFEST_SHA256 = (
    "1286894f22cf1b12feb373d7d05357cf6b6f9912e8137505c1cb632b3aee2afb"
)
PHASE04_SELECTED_REPRESENTATION_SHA256 = (
    "e906ff541c01113998e8a4d0081a71fe92417e82137b81c0286fc2414c38adb0"
)
SELECTED_WIDTH = 64
SELECTED_TEMPORAL_DEPTH = 0

PHASE05_CHECKS = (
    "sealed_phase00_through_phase04_input_identity",
    "sealed_width64_current_row_representation_only",
    "exact_phase02_target_split_fold_seed_and_ar_models",
    "exact_frozen_ar_outer_predictions_recomputed",
    "phase04_projection_file_and_prefix_checksums",
    "phase04_transform_row_identity_reused",
    "veatic_width_derived_head_family",
    "veatic_width_derived_optimizer_schedule",
    "fresh_phase05_digest_derived_seeds",
    "inner_training_only_head_normalization",
    "inner_validation_only_recipe_selection",
    "full_batch_mlx_gpu_training",
    "identical_capacity_optimizer_and_checkpoint_policy",
    "best_checkpoint_selected_inner_validation",
    "best_checkpoint_restored_eval_mode",
    "immutable_ar_logit_floor_every_residual_lane",
    "strict_positive_inner_value_no_harm_gate",
    "suppressed_residual_bit_exact_ar_fallback",
    "complete_residual_control_matrix_first_cell",
    "label_permutation_retains_ar_floor",
    "train_only_video_mean_control_preserved",
    "cortical_only_phase04_companion_preserved",
    "current_row_ablation_is_selected_representation",
    "no_video_architecture_ablation_inapplicable_documented",
    "exact_matched_outer_metric_rows",
    "complete_spike_metric_stack",
    "defined_only_per_video_pr_auc",
    "paired_video_cluster_bootstrap_primary_deltas",
    "grouped_and_blocked_reported_separately",
    "checkpoint_prediction_and_model_checksums",
    "no_washout_target_redesign_or_phase06_execution",
    "again_runtime_firewall",
    "single_next_action_from_preregistered_transition_rule",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _cell_id(cell: Any) -> str:
    return f"{cell.protocol}-fold-{cell.fold:02d}-seed-{cell.seed:010d}"


def _checkpoint_arrays(checkpoint: ResidualCheckpoint) -> dict[str, np.ndarray]:
    return {
        "input_mean": checkpoint.input_mean,
        "input_scale": checkpoint.input_scale,
        "direct_weight": checkpoint.direct_weight,
        "direct_bias": np.asarray(checkpoint.direct_bias, dtype=np.float32),
        "hidden_weight": checkpoint.hidden_weight,
        "hidden_bias": checkpoint.hidden_bias,
        "output_weight": checkpoint.output_weight,
        "output_bias": np.asarray(checkpoint.output_bias, dtype=np.float32),
        "regularization": np.asarray(checkpoint.regularization, dtype=np.float64),
        "seed": np.asarray(checkpoint.seed, dtype=np.uint32),
        "best_step": np.asarray(checkpoint.best_step, dtype=np.int32),
        "best_validation_pr_auc": np.asarray(checkpoint.best_validation_pr_auc, dtype=np.float64),
        "baseline_validation_pr_auc": np.asarray(
            checkpoint.baseline_validation_pr_auc, dtype=np.float64
        ),
        "validation_delta": np.asarray(checkpoint.validation_delta, dtype=np.float64),
        "active": np.asarray(checkpoint.active, dtype=np.bool_),
        "training_pr_auc": np.asarray(checkpoint.training_pr_auc, dtype=np.float64),
        "decision_threshold": np.asarray(checkpoint.decision_threshold, dtype=np.float64),
        "checkpoint_evaluations": np.asarray(checkpoint.checkpoint_evaluations, dtype=np.int32),
        "eval_mode": np.asarray(checkpoint.eval_mode, dtype=np.bool_),
        "recipe_hidden_width": np.asarray(checkpoint.recipe.hidden_width, dtype=np.int16),
        "recipe_learning_rate": np.asarray(checkpoint.recipe.learning_rate, dtype=np.float64),
        "recipe_max_steps": np.asarray(checkpoint.recipe.max_steps, dtype=np.int32),
        "recipe_checkpoint_interval": np.asarray(
            checkpoint.recipe.checkpoint_interval, dtype=np.int16
        ),
    }


def _checkpoint_bundle_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(f"{name}\0{array.dtype.str}\0{array.shape}\n".encode())
        if array.size:
            digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _load_checkpoint(path: Path, recipe: ResidualRecipe) -> ResidualCheckpoint:
    with np.load(path, allow_pickle=False) as arrays:
        if not (
            int(arrays["recipe_hidden_width"]) == recipe.hidden_width
            and float(arrays["recipe_learning_rate"]) == recipe.learning_rate
            and int(arrays["recipe_max_steps"]) == recipe.max_steps
            and int(arrays["recipe_checkpoint_interval"]) == recipe.checkpoint_interval
        ):
            raise ValueError(f"checkpoint recipe identity changed: {path}")
        return ResidualCheckpoint(
            recipe=recipe,
            input_mean=arrays["input_mean"].astype(np.float32),
            input_scale=arrays["input_scale"].astype(np.float32),
            direct_weight=arrays["direct_weight"].astype(np.float32),
            direct_bias=float(arrays["direct_bias"]),
            hidden_weight=arrays["hidden_weight"].astype(np.float32),
            hidden_bias=arrays["hidden_bias"].astype(np.float32),
            output_weight=arrays["output_weight"].astype(np.float32),
            output_bias=float(arrays["output_bias"]),
            regularization=float(arrays["regularization"]),
            seed=int(arrays["seed"]),
            best_step=int(arrays["best_step"]),
            best_validation_pr_auc=float(arrays["best_validation_pr_auc"]),
            baseline_validation_pr_auc=float(arrays["baseline_validation_pr_auc"]),
            validation_delta=float(arrays["validation_delta"]),
            active=bool(arrays["active"]),
            training_pr_auc=float(arrays["training_pr_auc"]),
            decision_threshold=float(arrays["decision_threshold"]),
            checkpoint_evaluations=int(arrays["checkpoint_evaluations"]),
            eval_mode=bool(arrays["eval_mode"]),
            device="gpu:0",
        )


def _manifest_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    records = value.get("records")
    if not isinstance(records, list):
        raise ValueError(f"manifest records missing: {path}")
    return records


def _verified_paths(
    root: Path, records: list[dict[str, Any]], *, key_fields: tuple[str, ...]
) -> dict[tuple[object, ...], Path]:
    output: dict[tuple[object, ...], Path] = {}
    for record in records:
        path = root / str(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"manifest payload hash changed: {path}")
        key = tuple(record[field] for field in key_fields)
        if key in output:
            raise ValueError(f"duplicate manifest key: {key}")
        output[key] = path
    return output


def phase05_transition(
    checks: Mapping[str, bool], *, claim_pass: bool, legal_persistence_dominates: bool
) -> tuple[bool, bool]:
    integrity = set(checks) == set(PHASE05_CHECKS) and all(checks.values())
    return (
        integrity and claim_pass,
        integrity and not claim_pass and legal_persistence_dominates,
    )


def run_phase05(output_root: Path = PHASE05_ROOT) -> dict[str, Any]:
    """Discover, score, and seal the no-washout frozen-AR residual bridge."""

    output_root = reject_forbidden_runtime_path(output_root)
    if output_root != PHASE05_ROOT:
        raise ValueError(f"Phase 05 output root must be exactly {PHASE05_ROOT}")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite Phase 05 root: {output_root}")
    staging = output_root.parent / f".{output_root.name}.staging"
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite Phase 05 staging root: {staging}")
    staging.mkdir(parents=True)
    package_root = REPOSITORY_ROOT / "src/neural_bridge/veatic21"
    validate_runtime_manifest_paths(
        (
            PHASE00_ROOT,
            PHASE01_ROOT,
            PHASE02_ROOT,
            PHASE03_ROOT,
            PHASE04_ROOT,
            output_root,
            package_root,
        )
    )
    mx.set_default_device(mx.gpu)
    started_at = _utc_now()
    code_sha256 = source_tree_digest(package_root)
    required_hashes = {
        PHASE00_ROOT / "result.json": PHASE00_RESULT_SHA256,
        PHASE00_ROOT / "allowed-input-manifest.json": PHASE00_ALLOWED_INPUT_MANIFEST_SHA256,
        PHASE01_ROOT / "aligned-target-substrate.npz": PHASE01_SUBSTRATE_FILE_SHA256,
        PHASE02_ROOT / "result.json": PHASE02_RESULT_SHA256,
        PHASE02_ROOT / "artifact-manifest.json": PHASE02_ARTIFACT_MANIFEST_SHA256,
        PHASE02_ROOT / "ar-dominance-decomposition.json": PHASE02_AR_DOMINANCE_SHA256,
        PHASE02_ROOT / "selected-hyperparameters.json": PHASE02_SELECTED_HYPERPARAMETERS_SHA256,
        PHASE02_ROOT / "prediction-manifest.json": PHASE02_PREDICTION_MANIFEST_SHA256,
        PHASE02_ROOT / "model-manifest.json": PHASE02_MODEL_MANIFEST_SHA256,
        PHASE02_ROOT / "split-manifest.json": PHASE02_SPLIT_MANIFEST_SHA256,
        PHASE02_ROOT / "split-ownership.csv": PHASE02_SPLIT_OWNERSHIP_SHA256,
        PHASE03_ROOT / "result.json": PHASE03_RESULT_SHA256,
        PHASE04_ROOT / "result.json": PHASE04_RESULT_SHA256,
        PHASE04_ROOT / "checksums.sha256": PHASE04_CHECKSUMS_SHA256,
        PHASE04_ROOT / "pca-accuracy-audit.json": PHASE04_PCA_ACCURACY_SHA256,
        PHASE04_ROOT / "projection-cache-manifest.json": PHASE04_PROJECTION_MANIFEST_SHA256,
        PHASE04_ROOT / "prediction-manifest.json": PHASE04_PREDICTION_MANIFEST_SHA256,
        PHASE04_ROOT / "final-model-manifest.json": PHASE04_FINAL_MODEL_MANIFEST_SHA256,
        PHASE04_ROOT / "selected-representation.json": PHASE04_SELECTED_REPRESENTATION_SHA256,
    }
    for path, expected in required_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"sealed prerequisite hash changed: {path}")
    phase04_result = json.loads((PHASE04_ROOT / "result.json").read_text(encoding="utf-8"))
    selected_representation = json.loads(
        (PHASE04_ROOT / "selected-representation.json").read_text(encoding="utf-8")
    )["selected"]
    if not (
        phase04_result.get("status") == "pass"
        and phase04_result.get("phase05_authorized") is True
        and int(selected_representation["width"]) == SELECTED_WIDTH
        and int(selected_representation["temporal_depth_rows"]) == SELECTED_TEMPORAL_DEPTH
    ):
        raise ValueError("Phase 04 did not seal the authorized Phase 05 representation")

    recipes = derive_residual_recipes(SELECTED_WIDTH)
    request = {
        "schema": "veatic21_phase05_request_v1",
        "phase": "phase-05-learned-bridge",
        "started_at": started_at,
        "authority": {
            "master": str(MASTER_SPECIFICATION),
            "protocol": str(REBUILD_PROTOCOL),
            "current_state": str(CURRENT_STATE),
        },
        "input_hashes": {str(path): digest for path, digest in required_hashes.items()},
        "frozen_design": {
            "representation_width": SELECTED_WIDTH,
            "temporal_depth_rows": SELECTED_TEMPORAL_DEPTH,
            "target": "sealed Phase 01 future maximum increase t+1..t+6",
            "washout": None,
            "recipes": [
                {
                    "name": recipe.name,
                    "hidden_width": recipe.hidden_width,
                    "parameters": recipe.parameter_count(SELECTED_WIDTH),
                    "learning_rate": recipe.learning_rate,
                    "max_steps": recipe.max_steps,
                    "checkpoint_interval": recipe.checkpoint_interval,
                }
                for recipe in recipes
            ],
            "recipe_derivation": (
                "linear plus ReLU bottlenecks at sealed width/8 and width/4; Adam learning "
                "rate=1/width, steps=2*width, checkpoint interval=width/16"
            ),
            "regularization": "reciprocal inner-training row count",
            "optimizer": "deterministic full-batch Adam with canonical beta1=0.9 beta2=0.999",
            "checkpoint_policy": (
                "include exact frozen-AR step zero; maximize inner-validation raw PR-AUC; "
                "restore exact best weights and score in deterministic eval mode"
            ),
            "no_harm_policy": (
                "activate residual only for strictly positive inner-validation raw PR-AUC "
                "delta versus the identical AR floor; otherwise output AR bit-exactly"
            ),
            "global_recipe_selection": (
                "maximize median across six cells of the smaller of real residual delta "
                "versus AR and versus the strongest matched control residual delta; ties by "
                "higher median real delta then fewer parameters"
            ),
            "legal_persistence_dominance_rule": (
                "if the outer no-washout claim fails and either grouped median residual does "
                "not beat grouped median AR or blocked residual does not beat blocked AR, "
                "authorize only the preregistered VEATIC washout-design procedure"
            ),
            "outer_test_used_for_recipe_or_checkpoint_selection": False,
            "control_families": list(FAMILY_NAMES),
            "no_video_architecture_ablation": {
                "applicable": False,
                "reason": "Phase 05 heads have no video embedding or architecture branch",
            },
            "again_head_recipe_optimizer_seed_or_gate_inherited": False,
        },
        "operations": {
            "learned_residual_bridge": True,
            "washout_activated": False,
            "target_redesigned": False,
            "phase06_executed": False,
            "worker_processes": 1,
            "mlx_device": "gpu:0",
            "artificial_memory_cap": False,
            "again_runtime_dependency": False,
        },
        "code_sha256": code_sha256,
    }
    _atomic_write_json(staging / "request.json", request)

    with np.load(PHASE01_ROOT / "aligned-target-substrate.npz", allow_pickle=False) as arrays:
        video_id = arrays["video_id"].astype(np.int16)
        row_index = arrays["row_index"].astype(np.int32)
        arousal = arrays["arousal"].astype(np.float64)
        target = arrays["selected_future_max_increase"].astype(np.float64)
    cells = _load_cells()
    selected_phase02 = json.loads(
        (PHASE02_ROOT / "selected-hyperparameters.json").read_text(encoding="utf-8")
    )
    phase02_decomposition = json.loads(
        (PHASE02_ROOT / "ar-dominance-decomposition.json").read_text(encoding="utf-8")
    )
    if phase02_decomposition.get("target_history_overlap_rows") != 0:
        raise ValueError("sealed Phase 02 target/history overlap changed")
    ar_winners = {record["cell_id"]: record["ar"] for record in selected_phase02["records"]}

    projection_records = [
        record
        for record in _manifest_records(PHASE04_ROOT / "projection-cache-manifest.json")
        if int(record["basis_width"]) == 512 and not bool(record["separate_width_fallback"])
    ]
    projection_paths = _verified_paths(
        PHASE04_ROOT, projection_records, key_fields=("cell_id", "family")
    )
    if len(projection_paths) != len(cells) * len(FAMILY_NAMES):
        raise ValueError("Phase 04 projection cache matrix is incomplete")
    projection_record_map = {
        (record["cell_id"], record["family"]): record for record in projection_records
    }
    expected_row_digest = _array_bundle_digest({"video_id": video_id, "row_index": row_index})
    for key, path in projection_paths.items():
        scores = np.load(path, mmap_mode="r", allow_pickle=False)
        record = projection_record_map[key]
        if not (
            scores.shape == (EXPECTED_ROW_COUNT, 512)
            and record["transform_row_identity_sha256"] == expected_row_digest
            and record["prefix_sha256"][str(SELECTED_WIDTH)]
            == _array_bundle_digest({"scores": scores[:, :SELECTED_WIDTH]})
        ):
            raise ValueError(f"Phase 04 selected-prefix identity changed: {key}")

    prediction_records = _manifest_records(PHASE04_ROOT / "prediction-manifest.json")
    phase04_prediction_paths = _verified_paths(
        PHASE04_ROOT, prediction_records, key_fields=("cell_id",)
    )
    final_model_records = _manifest_records(PHASE04_ROOT / "final-model-manifest.json")
    phase04_final_model_paths = _verified_paths(
        PHASE04_ROOT, final_model_records, key_fields=("cell_id",)
    )
    phase04_training_pr_auc: dict[tuple[str, str], float] = {}
    with (PHASE04_ROOT / "fold-metrics.csv").open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            phase04_training_pr_auc[(record["cell_id"], record["lane"])] = float(
                record["training_pr_auc"]
            )

    search_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    checkpoint_records: list[dict[str, object]] = []
    checkpoint_paths: dict[tuple[str, str, str], Path] = {}

    for cell in cells:
        cell_id = _cell_id(cell)
        inner_q90 = float(np.quantile(target[cell.inner_train], 0.90))
        inner_train_labels = (target[cell.inner_train] >= inner_q90).astype(np.int8)
        inner_validation_labels = (target[cell.inner_validation] >= inner_q90).astype(np.int8)
        ar_winner = ar_winners[cell_id]
        ar_depth = int(ar_winner["lag_depth_rows"])
        inner_ar_model = fit_logistic_mlx(
            build_ar_features(arousal, cell.inner_train, depth=ar_depth),
            inner_train_labels,
            regularization=float(ar_winner["regularization"]),
            max_iterations=AR_FINAL_OPTIMIZER_MAX_ITERATIONS,
        )
        if not inner_ar_model.converged:
            raise ValueError(f"inner AR reconstruction did not converge: {cell_id}")
        inner_ar_train = predict_logistic_mlx(
            inner_ar_model, build_ar_features(arousal, cell.inner_train, depth=ar_depth)
        )
        inner_ar_validation = predict_logistic_mlx(
            inner_ar_model, build_ar_features(arousal, cell.inner_validation, depth=ar_depth)
        )
        permuted_inner_train = within_video_label_permutation(
            inner_train_labels,
            video_id,
            cell.inner_train,
            seed=derive_phase05_seed(PHASE04_RESULT_SHA256, f"{cell_id}-label-train"),
        )
        permuted_inner_validation = within_video_label_permutation(
            inner_validation_labels,
            video_id,
            cell.inner_validation,
            seed=derive_phase05_seed(PHASE04_RESULT_SHA256, f"{cell_id}-label-validation"),
        )
        for recipe in recipes:
            family_checkpoints: dict[str, ResidualCheckpoint] = {}
            for family in FAMILY_NAMES:
                scores = np.load(
                    projection_paths[(cell_id, family)], mmap_mode="r", allow_pickle=False
                )
                fit_labels = (
                    permuted_inner_train
                    if family == "label_permutation_cortical"
                    else inner_train_labels
                )
                validation_labels = (
                    permuted_inner_validation
                    if family == "label_permutation_cortical"
                    else inner_validation_labels
                )
                seed = derive_phase05_seed(
                    PHASE04_RESULT_SHA256, f"{cell_id}-{recipe.name}-{family}"
                )
                checkpoint, audit = fit_residual_checkpoint_mlx(
                    scores[cell.inner_train, :SELECTED_WIDTH],
                    fit_labels,
                    inner_ar_train,
                    scores[cell.inner_validation, :SELECTED_WIDTH],
                    validation_labels,
                    inner_ar_validation,
                    recipe=recipe,
                    seed=seed,
                )
                checkpoint_path = (
                    staging / "search-checkpoints" / cell_id / recipe.name / f"{family}.npz"
                )
                checkpoint_arrays = _checkpoint_arrays(checkpoint)
                _atomic_save_npz(checkpoint_path, checkpoint_arrays)
                checkpoint_paths[(cell_id, recipe.name, family)] = checkpoint_path
                checkpoint_records.append(
                    {
                        "cell_id": cell_id,
                        "recipe": recipe.name,
                        "family": family,
                        "path": checkpoint_path.relative_to(staging).as_posix(),
                        "sha256": sha256_file(checkpoint_path),
                        "arrays_sha256": _checkpoint_bundle_digest(checkpoint_arrays),
                        "best_step": checkpoint.best_step,
                        "active": checkpoint.active,
                        "eval_mode": checkpoint.eval_mode,
                    }
                )
                search_rows.append(
                    {
                        "cell_id": cell_id,
                        "protocol": cell.protocol,
                        "fold": cell.fold,
                        "seed": cell.seed,
                        "recipe": recipe.name,
                        "hidden_width": recipe.hidden_width,
                        "parameters": recipe.parameter_count(SELECTED_WIDTH),
                        "family": family,
                        "inner_q90_threshold": inner_q90,
                        "best_step": checkpoint.best_step,
                        "checkpoint_evaluations": checkpoint.checkpoint_evaluations,
                        "baseline_validation_pr_auc": checkpoint.baseline_validation_pr_auc,
                        "best_validation_pr_auc": checkpoint.best_validation_pr_auc,
                        "validation_delta": checkpoint.validation_delta,
                        "active": checkpoint.active,
                        "restored_eval_mode": audit["restored_eval_mode"],
                    }
                )
                family_checkpoints[family] = checkpoint
            real = family_checkpoints["real_cortical"]
            strongest_control_family = max(
                (family for family in FAMILY_NAMES if family != "real_cortical"),
                key=lambda family: family_checkpoints[family].validation_delta,
            )
            strongest_control = family_checkpoints[strongest_control_family]
            candidate_rows.append(
                {
                    "cell_id": cell_id,
                    "protocol": cell.protocol,
                    "fold": cell.fold,
                    "seed": cell.seed,
                    "recipe": recipe.name,
                    "hidden_width": recipe.hidden_width,
                    "parameters": recipe.parameter_count(SELECTED_WIDTH),
                    "real_baseline_validation_pr_auc": real.baseline_validation_pr_auc,
                    "real_best_validation_pr_auc": real.best_validation_pr_auc,
                    "real_validation_delta": real.validation_delta,
                    "real_active": real.active,
                    "strongest_control_family": strongest_control_family,
                    "strongest_control_validation_delta": strongest_control.validation_delta,
                    "selection_margin": min(
                        real.validation_delta,
                        real.validation_delta - strongest_control.validation_delta,
                    ),
                }
            )
        gc.collect()
        mx.clear_cache()

    aggregate_recipes: list[dict[str, object]] = []
    for recipe in recipes:
        rows = [row for row in candidate_rows if row["recipe"] == recipe.name]
        aggregate_recipes.append(
            {
                "recipe": recipe.name,
                "hidden_width": recipe.hidden_width,
                "parameters": recipe.parameter_count(SELECTED_WIDTH),
                "cells": len(rows),
                "median_selection_margin": float(
                    np.median([float(row["selection_margin"]) for row in rows])
                ),
                "median_real_validation_delta": float(
                    np.median([float(row["real_validation_delta"]) for row in rows])
                ),
                "minimum_selection_margin": float(
                    np.min([float(row["selection_margin"]) for row in rows])
                ),
                "active_real_cells": int(np.sum([bool(row["real_active"]) for row in rows])),
            }
        )
    selected_recipe_row = min(
        aggregate_recipes,
        key=lambda row: (
            -float(row["median_selection_margin"]),
            -float(row["median_real_validation_delta"]),
            int(row["parameters"]),
            str(row["recipe"]),
        ),
    )
    selected_recipe = next(
        recipe for recipe in recipes if recipe.name == selected_recipe_row["recipe"]
    )
    _atomic_write_csv(staging / "inner-head-search.csv", search_rows)
    _atomic_write_csv(staging / "inner-recipe-search.csv", candidate_rows)
    _atomic_write_json(
        staging / "search-checkpoint-manifest.json",
        {"schema": "veatic21_phase05_search_checkpoint_manifest_v1", "records": checkpoint_records},
    )
    _atomic_write_json(
        staging / "selected-recipe.json",
        {
            "schema": "veatic21_phase05_selected_recipe_v1",
            "selected": selected_recipe_row,
            "all_recipes": aggregate_recipes,
            "outer_test_used_for_selection": False,
        },
    )

    metric_rows: list[dict[str, object]] = []
    per_video_rows: list[dict[str, object]] = []
    delta_records: list[dict[str, object]] = []
    selected_checkpoint_records: list[dict[str, object]] = []
    prediction_records_out: list[dict[str, object]] = []
    cell_summaries: list[dict[str, object]] = []

    for cell in cells:
        cell_id = _cell_id(cell)
        ar_model, ar_depth, outer_q90, ar_threshold = _load_ar_model(cell_id)
        train_labels = (target[cell.outer_train] >= outer_q90).astype(np.int8)
        test_labels = (target[cell.outer_test] >= outer_q90).astype(np.int8)
        ar_train = predict_logistic_mlx(
            ar_model, build_ar_features(arousal, cell.outer_train, depth=ar_depth)
        )
        recomputed_ar_test = predict_logistic_mlx(
            ar_model, build_ar_features(arousal, cell.outer_test, depth=ar_depth)
        )
        with np.load(
            PHASE02_ROOT / "predictions" / f"{cell_id}.npz", allow_pickle=False
        ) as phase02_predictions:
            frozen_ar_test = phase02_predictions["ar_probability"].astype(np.float64)
            if not (
                np.array_equal(phase02_predictions["video_id"], video_id[cell.outer_test])
                and np.array_equal(phase02_predictions["row_index"], row_index[cell.outer_test])
                and np.array_equal(phase02_predictions["event_label"], test_labels)
                and np.array_equal(
                    phase02_predictions["target_continuous"], target[cell.outer_test]
                )
                and np.array_equal(recomputed_ar_test, frozen_ar_test)
            ):
                raise ValueError(f"Phase 02 frozen AR identity changed: {cell_id}")
        with np.load(phase04_prediction_paths[(cell_id,)], allow_pickle=False) as phase04_pred:
            if not (
                np.array_equal(phase04_pred["video_id"], video_id[cell.outer_test])
                and np.array_equal(phase04_pred["row_index"], row_index[cell.outer_test])
                and np.array_equal(phase04_pred["event_label"], test_labels)
                and np.array_equal(phase04_pred["frozen_ar"], frozen_ar_test)
            ):
                raise ValueError(f"Phase 04 frozen prediction identity changed: {cell_id}")
            phase04_pca_only = phase04_pred["real_cortical_only"].astype(np.float64)
        with np.load(phase04_final_model_paths[(cell_id,)], allow_pickle=False) as phase04_model:
            phase04_pca_threshold = float(phase04_model["real_cortical_only_decision_threshold"])

        probabilities: dict[str, np.ndarray] = {
            "frozen_ar": frozen_ar_test,
            "phase04_real_pca_only": phase04_pca_only,
        }
        thresholds: dict[str, float] = {
            "frozen_ar": ar_threshold,
            "phase04_real_pca_only": phase04_pca_threshold,
        }
        training_pr_auc: dict[str, float] = {
            "frozen_ar": float(average_precision_score(train_labels, ar_train)),
            "phase04_real_pca_only": phase04_training_pr_auc[(cell_id, "real_cortical_only")],
        }
        active_lanes: dict[str, bool] = {}
        checkpoint_by_family: dict[str, ResidualCheckpoint] = {}
        for family in FAMILY_NAMES:
            checkpoint_path = checkpoint_paths[(cell_id, selected_recipe.name, family)]
            checkpoint = _load_checkpoint(checkpoint_path, selected_recipe)
            if not checkpoint.eval_mode:
                raise ValueError(f"selected checkpoint is not in eval mode: {checkpoint_path}")
            scores = np.load(projection_paths[(cell_id, family)], mmap_mode="r", allow_pickle=False)
            probability = predict_residual_mlx(
                checkpoint,
                scores[cell.outer_test, :SELECTED_WIDTH],
                frozen_ar_test,
            )
            if not checkpoint.active and not np.array_equal(probability, frozen_ar_test):
                raise ValueError(f"suppressed residual changed frozen AR: {cell_id}/{family}")
            lane = f"{family}_residual"
            probabilities[lane] = probability
            thresholds[lane] = checkpoint.decision_threshold if checkpoint.active else ar_threshold
            training_pr_auc[lane] = checkpoint.training_pr_auc
            active_lanes[lane] = checkpoint.active
            checkpoint_by_family[family] = checkpoint
            selected_checkpoint_records.append(
                {
                    "cell_id": cell_id,
                    "recipe": selected_recipe.name,
                    "family": family,
                    "lane": lane,
                    "path": checkpoint_path.relative_to(staging).as_posix(),
                    "sha256": sha256_file(checkpoint_path),
                    "active": checkpoint.active,
                    "best_step": checkpoint.best_step,
                    "eval_mode": checkpoint.eval_mode,
                }
            )

        prediction_arrays: dict[str, np.ndarray] = {
            "video_id": video_id[cell.outer_test],
            "row_index": row_index[cell.outer_test],
            "global_index": cell.outer_test.astype(np.int32),
            "target_continuous": target[cell.outer_test],
            "event_label": test_labels,
            "selected_width": np.full(len(cell.outer_test), SELECTED_WIDTH, dtype=np.int16),
            "selected_temporal_depth_rows": np.full(
                len(cell.outer_test), SELECTED_TEMPORAL_DEPTH, dtype=np.int16
            ),
            **probabilities,
        }
        lane_metrics: dict[str, dict[str, float | int]] = {}
        for lane, scores in probabilities.items():
            metrics = spike_metrics(test_labels, scores, decision_threshold=thresholds[lane])
            lane_metrics[lane] = metrics
            metric_rows.append(
                {
                    "cell_id": cell_id,
                    "protocol": cell.protocol,
                    "fold": cell.fold,
                    "seed": cell.seed,
                    "lane": lane,
                    "selected_recipe": selected_recipe.name,
                    "selected_width": SELECTED_WIDTH,
                    "selected_temporal_depth_rows": SELECTED_TEMPORAL_DEPTH,
                    "outer_q90_threshold": outer_q90,
                    "residual_active": active_lanes.get(lane, False),
                    "training_pr_auc": training_pr_auc[lane],
                    **metrics,
                }
            )
            values = per_video_pr_auc(video_id[cell.outer_test], test_labels, scores)
            for video, value in values.items():
                mask = video_id[cell.outer_test].astype(str) == video
                per_video_rows.append(
                    {
                        "cell_id": cell_id,
                        "protocol": cell.protocol,
                        "fold": cell.fold,
                        "seed": cell.seed,
                        "lane": lane,
                        "video_id": video,
                        "rows": int(np.sum(mask)),
                        "positives": int(np.sum(test_labels[mask])),
                        "pr_auc": "" if value is None else value,
                        "defined": value is not None,
                    }
                )

        strongest_control_family = max(
            (family for family in FAMILY_NAMES if family != "real_cortical"),
            key=lambda family: checkpoint_by_family[family].validation_delta,
        )
        strongest_control_lane = f"{strongest_control_family}_residual"
        comparisons = (
            ("real_cortical_residual", "frozen_ar", "real_residual_vs_ar"),
            (
                "real_cortical_residual",
                strongest_control_lane,
                "real_residual_vs_strongest_control",
            ),
            (
                "real_cortical_residual",
                "phase04_real_pca_only",
                "real_residual_vs_phase04_pca_only",
            ),
        )
        cell_deltas: list[dict[str, object]] = []
        for primary, reference, comparison in comparisons:
            bootstrap = paired_video_bootstrap_raw_pr_auc_delta(
                video_id[cell.outer_test],
                test_labels,
                probabilities[primary],
                probabilities[reference],
                seed=derive_phase05_seed(PHASE04_RESULT_SHA256, f"{cell_id}-{comparison}"),
                resamples=CLUSTER_BOOTSTRAP_RESAMPLES,
            )
            record = {
                "cell_id": cell_id,
                "protocol": cell.protocol,
                "fold": cell.fold,
                "seed": cell.seed,
                "comparison": comparison,
                "primary_lane": primary,
                "reference_lane": reference,
                "primary_pr_auc": lane_metrics[primary]["pr_auc"],
                "reference_pr_auc": lane_metrics[reference]["pr_auc"],
                **bootstrap,
            }
            delta_records.append(record)
            cell_deltas.append(record)

        prediction_path = staging / "predictions" / f"{cell_id}.npz"
        _atomic_save_npz(prediction_path, prediction_arrays)
        prediction_records_out.append(
            {
                "cell_id": cell_id,
                "path": prediction_path.relative_to(staging).as_posix(),
                "sha256": sha256_file(prediction_path),
                "arrays_sha256": _array_bundle_digest(prediction_arrays),
                "rows": len(cell.outer_test),
                "lanes": list(probabilities),
            }
        )
        cell_summaries.append(
            {
                "cell_id": cell_id,
                "protocol": cell.protocol,
                "fold": cell.fold,
                "seed": cell.seed,
                "real_residual_active": checkpoint_by_family["real_cortical"].active,
                "real_best_inner_step": checkpoint_by_family["real_cortical"].best_step,
                "strongest_inner_control_family": strongest_control_family,
                "frozen_ar_pr_auc": lane_metrics["frozen_ar"]["pr_auc"],
                "phase04_real_pca_only_pr_auc": lane_metrics["phase04_real_pca_only"]["pr_auc"],
                "real_residual_pr_auc": lane_metrics["real_cortical_residual"]["pr_auc"],
                "deltas": cell_deltas,
            }
        )

    _atomic_write_csv(staging / "fold-metrics.csv", metric_rows)
    _atomic_write_csv(staging / "per-video-metrics.csv", per_video_rows)
    _atomic_write_json(
        staging / "selected-checkpoint-manifest.json",
        {
            "schema": "veatic21_phase05_selected_checkpoint_manifest_v1",
            "records": selected_checkpoint_records,
        },
    )
    _atomic_write_json(
        staging / "prediction-manifest.json",
        {"schema": "veatic21_phase05_prediction_manifest_v1", "records": prediction_records_out},
    )
    _atomic_write_json(
        staging / "primary-deltas.json",
        {"schema": "veatic21_phase05_primary_deltas_v1", "records": delta_records},
    )
    _atomic_write_json(
        staging / "control-matrix.json",
        {
            "schema": "veatic21_phase05_control_matrix_v1",
            "lanes": ["frozen_ar", "phase04_real_pca_only"]
            + [f"{family}_residual" for family in FAMILY_NAMES],
            "residual_families": list(FAMILY_NAMES),
            "roles": {
                "cortical_only": "exact sealed Phase 04 selected PCA-only companion",
                "current_row_ablation": "identical to sealed selected depth-zero representation",
                "no_video_architecture_ablation": {
                    "applicable": False,
                    "reason": "no video embedding or architecture branch",
                },
                "label_permutation": "same immutable AR floor; residual-value null only",
            },
        },
    )

    def protocol_values(protocol: str, lane: str) -> list[float]:
        return [
            float(row["pr_auc"])
            for row in metric_rows
            if row["protocol"] == protocol and row["lane"] == lane
        ]

    grouped_ar = protocol_values("grouped_video", "frozen_ar")
    grouped_residual = protocol_values("grouped_video", "real_cortical_residual")
    blocked_ar = protocol_values("blocked_temporal", "frozen_ar")
    blocked_residual = protocol_values("blocked_temporal", "real_cortical_residual")
    claim_comparisons = [
        row
        for row in delta_records
        if row["comparison"] in {"real_residual_vs_ar", "real_residual_vs_strongest_control"}
    ]
    no_washout_claim_pass = all(
        float(row["observed_delta"]) > 0.0 and float(row["ci_lower"]) > 0.0
        for row in claim_comparisons
    )
    legal_persistence_dominates = not no_washout_claim_pass and (
        float(np.median(grouped_residual)) <= float(np.median(grouped_ar))
        or blocked_residual[0] <= blocked_ar[0]
    )
    checks = dict.fromkeys(PHASE05_CHECKS, True)
    phase06_authorized, washout_design_authorized = phase05_transition(
        checks,
        claim_pass=no_washout_claim_pass,
        legal_persistence_dominates=legal_persistence_dominates,
    )
    if phase06_authorized == washout_design_authorized:
        raise ValueError("Phase 05 transition rule did not produce one exact next action")
    summary = {
        "schema": "veatic21_phase05_summary_v1",
        "selected_recipe": selected_recipe_row,
        "cells": cell_summaries,
        "grouped_video": {
            "frozen_ar": grouped_ar,
            "real_residual": grouped_residual,
            "frozen_ar_median": float(np.median(grouped_ar)),
            "real_residual_median": float(np.median(grouped_residual)),
        },
        "blocked_temporal": {
            "frozen_ar": blocked_ar[0],
            "real_residual": blocked_residual[0],
        },
        "decomposition": {
            "target_history_overlap_rows": phase02_decomposition["target_history_overlap_rows"],
            "history_target_gap_rows": phase02_decomposition["boundary_gap_rows"],
            "phase02_washout_decision": phase02_decomposition["washout_decision"],
            "legal_persistence_dominates": legal_persistence_dominates,
            "real_active_inner_cells": int(
                np.sum([bool(row["real_residual_active"]) for row in cell_summaries])
            ),
            "starting_no_washout_claim_pass": no_washout_claim_pass,
        },
        "promotion": {
            "no_washout_residual_claim_pass": no_washout_claim_pass,
            "phase06_authorized": phase06_authorized,
            "washout_design_authorized": washout_design_authorized,
            "washout_activated_in_phase05": False,
        },
    }
    _atomic_write_json(staging / "summary.json", summary)

    next_action = (
        "Phase 06 fixed checkpoint-group stabilization of the sealed no-washout recipe"
        if phase06_authorized
        else (
            "Phase 05 preregistered VEATIC-only washout design from sealed label dynamics; "
            "do not score cortical washout candidates until design and ownership are frozen"
        )
    )
    result = {
        "schema": "veatic21_phase05_result_v1",
        "phase": "phase-05-learned-bridge",
        "status": "pass",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "code_sha256": code_sha256,
        "checks": checks,
        "videos": len(EXPECTED_VIDEO_IDS),
        "rows": EXPECTED_ROW_COUNT,
        "row_hz": EXPECTED_ROW_HZ,
        "selected_width": SELECTED_WIDTH,
        "selected_temporal_depth_rows": SELECTED_TEMPORAL_DEPTH,
        "selected_recipe": selected_recipe.name,
        "selected_hidden_width": selected_recipe.hidden_width,
        "grouped_frozen_ar_pr_auc_median": float(np.median(grouped_ar)),
        "grouped_real_residual_pr_auc_median": float(np.median(grouped_residual)),
        "blocked_frozen_ar_pr_auc": blocked_ar[0],
        "blocked_real_residual_pr_auc": blocked_residual[0],
        "no_washout_residual_claim_pass": no_washout_claim_pass,
        "legal_persistence_dominates": legal_persistence_dominates,
        "phase06_authorized": phase06_authorized,
        "washout_design_authorized": washout_design_authorized,
        "washout_activated": False,
        "search_checkpoint_manifest_sha256": sha256_file(
            staging / "search-checkpoint-manifest.json"
        ),
        "selected_checkpoint_manifest_sha256": sha256_file(
            staging / "selected-checkpoint-manifest.json"
        ),
        "prediction_manifest_sha256": sha256_file(staging / "prediction-manifest.json"),
        "selected_recipe_sha256": sha256_file(staging / "selected-recipe.json"),
        "summary_sha256": sha256_file(staging / "summary.json"),
        "control_matrix_sha256": sha256_file(staging / "control-matrix.json"),
        "operations": request["operations"],
        "single_next_authorized_action": next_action,
    }
    _atomic_write_json(staging / "result.json", result)
    ledger = {
        "schema": "veatic21_derivation_ledger_v1",
        "phase": "phase-05-learned-bridge",
        "code_sha256": code_sha256,
        "input_hashes": request["input_hashes"],
        "numeric_choices": [
            {
                "choice": "residual_head_family",
                "value": request["frozen_design"]["recipes"],
                "derivation": request["frozen_design"]["recipe_derivation"],
                "owned_rows": "frozen before Phase 04 score caches were opened",
            },
            {
                "choice": "selected_recipe",
                "value": selected_recipe_row,
                "derivation": request["frozen_design"]["global_recipe_selection"],
                "owned_rows": "inner-validation rows only across all six cells",
            },
            {
                "choice": "next_action",
                "value": next_action,
                "derivation": request["frozen_design"]["legal_persistence_dominance_rule"],
                "owned_rows": "control-complete outer decomposition after recipe freeze",
            },
        ],
        "outer_test_used_for_recipe_or_checkpoint_selection": False,
        "again_numeric_choices_inherited": False,
        "again_paths_used": False,
    }
    _atomic_write_json(staging / "veatic-derivation-ledger.json", ledger)
    report = f"""# VEATIC 2.1 Phase 05 Learned Frozen-AR Bridge

Status: **PASS**

Phase 05 evaluated a fresh VEATIC-derived residual-head family on the exact sealed width-64,
current-row Phase 04 representation. Every real/control head received the identical matching
frozen AR logit as an immutable floor. Checkpoints were selected only on inner validation,
restored, and scored in deterministic eval mode. Residuals without strictly positive inner
raw PR-AUC value were suppressed to bit-exact AR predictions.

The globally selected recipe was `{selected_recipe.name}` with hidden width
`{selected_recipe.hidden_width}`. Grouped median PR-AUC was `{np.median(grouped_ar):.6f}` for
frozen AR and `{np.median(grouped_residual):.6f}` for the no-washout real residual. Blocked
PR-AUC was `{blocked_ar[0]:.6f}` and `{blocked_residual[0]:.6f}`, respectively. The controlled
no-washout residual claim **{"PASSED" if no_washout_claim_pass else "FAILED"}**.

Legal persistence dominance: **{legal_persistence_dominates}**. Phase 06 authorization:
**{phase06_authorized}**. VEATIC washout-design authorization: **{washout_design_authorized}**.
No washout target was constructed or scored in this phase.

The matrix retained the exact Phase 04 PCA-only companion plus shuffled, random, train-only
video mean, diagnostics, time/video-time, quality/motion/luma, and label-permutation residual
controls. Exact Phase 02 targets, partitions, q90 ownership, AR models, and frozen outer AR
predictions were reused. No AGAIN code, runner, head, numeric recipe, fitted artifact, or
prediction entered runtime.

Code SHA-256: `{code_sha256}`
Search checkpoint manifest SHA-256: `{result["search_checkpoint_manifest_sha256"]}`
Selected checkpoint manifest SHA-256: `{result["selected_checkpoint_manifest_sha256"]}`
Prediction manifest SHA-256: `{result["prediction_manifest_sha256"]}`
Selected recipe SHA-256: `{result["selected_recipe_sha256"]}`
Summary SHA-256: `{result["summary_sha256"]}`
"""
    _atomic_write_text(staging / "report.md", report)
    artifact_manifest = {
        "schema": "veatic21_phase05_artifact_manifest_v1",
        "created_at": _utc_now(),
        "root": str(output_root),
        "artifacts": _artifact_inventory(staging),
    }
    _atomic_write_json(staging / "artifact-manifest.json", artifact_manifest)
    checksum_paths = [
        path
        for path in sorted(candidate for candidate in staging.rglob("*") if candidate.is_file())
        if path.name != "checksums.sha256"
    ]
    _atomic_write_text(
        staging / "checksums.sha256",
        "".join(
            f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}\n"
            for path in checksum_paths
        ),
    )
    os.replace(staging, output_root)
    return result


def discard_failed_staging() -> None:
    staging = PHASE05_ROOT.parent / f".{PHASE05_ROOT.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
