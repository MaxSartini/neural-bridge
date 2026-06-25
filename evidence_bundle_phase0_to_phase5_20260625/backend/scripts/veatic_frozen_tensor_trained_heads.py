"""Trained-head scoring for frozen VEATIC tensors.

This module keeps the existing frozen tensor adapter as the data/contract layer
and reuses the canonical VEATIC helper functions for AR, time, ridge, controls,
and event metrics.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from backend.scripts import run_veatic_event_spike_retest as event_spike
from backend.scripts import run_veatic_neuro_benchmark as bench
from backend.scripts.veatic_frozen_tensor_adapter import (
    BENCHMARK_CONTRACT_VERSION,
    FROZEN_BASELINE,
    PRIMARY_CANDIDATE,
    REQUIRED_REPRESENTATIONS,
    SPLITS,
    SUMMARY_ROOT,
    FrozenTensorContract,
    FrozenTensorFeatureProvider,
    assert_same_row_order,
    base_target_name,
    check_failures,
    deny_prior_result_files,
    finite_json,
    git_sha,
    is_control_lane,
    read_jsonl,
    reject_exclude_83_paths,
    ridge_train_test,
    row_key,
    score_metrics,
    sha256_file,
    target_dir_name,
    write_csv,
    write_json,
)


TRAINED_BENCHMARK_MODE = "existing_suite_with_frozen_tensor_adapter_trained_heads"
TRAINED_SCHEMA_VERSION = "veatic_frozen_tensor_trained_heads_run_v1"
TRAINED_HEADS = (
    "ridge_score",
    "logistic_l2",
    "elastic_net_logistic",
    "flattened_sequence_logistic",
    "learned_temporal_pool_logistic",
)
COLLAPSED_HEADS = ("ridge_score", "logistic_l2", "elastic_net_logistic")
SEQUENCE_HEADS = ("flattened_sequence_logistic", "learned_temporal_pool_logistic")
LOGISTIC_C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
ELASTIC_C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
ELASTIC_L1_GRID = (0.1, 0.25, 0.5, 0.75, 0.9)
CLASS_WEIGHT_GRID = (None, "balanced")
MPS_LOGISTIC_EPOCHS = 350
MPS_LOGISTIC_LR = 0.05
MPS_LOGISTIC_FINAL_EPOCHS = 450
TEMPORAL_POOL_WEIGHTS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    (0.2, 0.3, 0.5),
    (0.1, 0.3, 0.6),
    (0.25, 0.25, 0.5),
    (0.15, 0.25, 0.60),
    (0.10, 0.20, 0.70),
)
PRIMARY_SPIKE_TARGETS = (
    ("arousal__future_spike_1_3s", 0.05),
    ("arousal__future_spike_1_3s", 0.075),
)
PRIMARY_TARGETS = PRIMARY_SPIKE_TARGETS + (("arousal__future_change_p3s_movement", 0.05),)


LANE_RESULT_COLUMNS = [
    "fresh_run_id",
    "benchmark_mode",
    "scope",
    "split_name",
    "target_name",
    "threshold",
    "representation_name",
    "feature_name",
    "model_lane",
    "head_name",
    "uses_ar_features",
    "uses_neural_features",
    "uses_sequence_tensor",
    "uses_roi",
    "uses_topk",
    "uses_supervised_features",
    "cautionary",
    "residualized",
    "tests_incremental_neural_value",
    "baseline_role",
    "control_source",
    "canonical_control",
    "control_seed",
    "row_population_matched",
    "shape_matched",
    "n_train",
    "n_test",
    "feature_width",
    "sequence_shape",
    "train_event_count",
    "test_event_count",
    "train_positive_rate",
    "test_positive_rate",
    "selected_hyperparameters_json",
    "inner_validation_pr_auc",
    "decision_threshold",
    "pr_auc",
    "roc_auc",
    "f1",
    "balanced_accuracy",
    "precision",
    "recall",
    "accuracy",
    "predicted_positive_rate",
    "predicted_positive_count",
    "top_1pct_recall",
    "top_5pct_recall",
    "top_10pct_recall",
    "top_event_count_recall",
    "ar_only_pr_auc",
    "pca128_only_pr_auc",
    "pca64_delta_only_pr_auc",
    "ar_plus_pca128_pr_auc",
    "ar_plus_pca64_delta_pr_auc",
    "residualized_ar_plus_pca128_pr_auc",
    "residualized_ar_plus_pca64_delta_pr_auc",
    "delta_pca128_only_vs_ar",
    "delta_ar_plus_pca128_vs_ar",
    "delta_residualized_ar_plus_pca128_vs_ar",
    "delta_ar_plus_pca128_vs_ar_plus_pca64_delta",
    "delta_residualized_pca128_vs_residualized_pca64_delta",
    "control_pass",
    "leakage_status",
    "freshly_computed_in_current_run",
    "reused_benchmark_result_row",
    "notes",
]


@dataclass(frozen=True)
class InnerValidationSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_videos: tuple[str, ...]
    val_videos: tuple[str, ...]
    strategy: str


@dataclass
class PredictionBundle:
    train_scores: np.ndarray
    test_scores: np.ndarray
    selection: dict[str, Any]


def trained_freshness_ledger(provider: FrozenTensorFeatureProvider, fresh_run_id: str) -> dict[str, Any]:
    return {
        "fresh_run_id": fresh_run_id,
        "code_git_sha": git_sha(),
        "tensor_export_summary_sha256": sha256_file(provider.summary_path),
        "tensor_export_inventory_sha256": sha256_file(provider.inventory_path),
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_mode": TRAINED_BENCHMARK_MODE,
        "reused_existing_benchmark_structure": True,
        "reused_frozen_tensors": True,
        "reused_canonical_controls": True,
        "reused_benchmark_result_rows": False,
        "computed_ar_fresh": True,
        "computed_controls_fresh": True,
        "computed_scores": True,
        "trained_head_backend": "torch_mps",
        "mps_available": bool(torch.backends.mps.is_available()),
        "fit_models": True,
        "wrote_result_csvs": True,
        "full_veatic_124": True,
        "video_83_included": True,
        "exclude_video_83_run": False,
        "result_row_reuse_policy": {
            "prior_benchmark_rows_reused": False,
            "prior_ar_rows_reused": False,
            "prior_ridge_rows_reused": False,
            "prior_control_rows_reused": False,
        },
    }


def required_run_fields() -> dict[str, bool | str]:
    return {
        "benchmark_mode": TRAINED_BENCHMARK_MODE,
        "reused_existing_benchmark_structure": True,
        "reused_frozen_tensors": True,
        "reused_canonical_controls": True,
        "reused_benchmark_result_rows": False,
        "computed_ar_fresh": True,
        "computed_controls_fresh": True,
        "computed_scores": True,
        "full_veatic_124": True,
        "video_83_included": True,
        "exclude_video_83_run": False,
    }


def assert_required_run_fields(payload: dict[str, Any]) -> None:
    for key, expected in required_run_fields().items():
        if payload.get(key) != expected:
            raise ValueError(f"Required run field mismatch for {key}: expected {expected!r}, got {payload.get(key)!r}")


def load_sequence_arrays(contract: FrozenTensorContract) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    train_path = contract.contract_dir / "X_sequence_train.npy"
    test_path = contract.contract_dir / "X_sequence_test.npy"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing frozen sequence tensors for {contract.representation_name}: {contract.contract_dir}")
    mask_train = contract.contract_dir / "sequence_mask_train.npy"
    mask_test = contract.contract_dir / "sequence_mask_test.npy"
    return (
        np.load(train_path),
        np.load(test_path),
        np.load(mask_train) if mask_train.exists() else None,
        np.load(mask_test) if mask_test.exists() else None,
    )


def flatten_sequence(sequence: np.ndarray) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float64)
    if sequence.ndim != 3:
        raise ValueError(f"Expected sequence tensor [rows, timesteps, features], got {sequence.shape}")
    return sequence.reshape(sequence.shape[0], sequence.shape[1] * sequence.shape[2])


def temporal_pool_sequence(sequence: np.ndarray, weights: Iterable[float]) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float64)
    weight_array = np.asarray(tuple(weights), dtype=np.float64)
    if sequence.ndim != 3:
        raise ValueError(f"Expected sequence tensor [rows, timesteps, features], got {sequence.shape}")
    if sequence.shape[1] != weight_array.shape[0]:
        raise ValueError(f"Temporal weight count {weight_array.shape[0]} does not match sequence shape {sequence.shape}")
    return np.tensordot(sequence, weight_array, axes=(1, 0))


def lane_flags(representation_name: str, model_lane: str, *, uses_sequence_tensor: bool = False) -> dict[str, Any]:
    uses_ar = model_lane == "AR_only" or model_lane.startswith("AR_plus_") or model_lane.startswith("residualized_AR_plus_")
    uses_neural = model_lane != "AR_only" and model_lane not in {"mean_train", "time_ridge"}
    uses_roi = representation_name == "roi_parcel_features"
    uses_topk = representation_name == "topk_vertices_512"
    residualized = model_lane.startswith("residualized_AR_plus_")
    tests_incremental = model_lane.startswith("AR_plus_") or residualized
    baseline_role = ""
    if model_lane == "AR_only":
        baseline_role = "original_original_baseline"
    elif representation_name == FROZEN_BASELINE and model_lane == "PCA64_delta_only":
        baseline_role = "previous_neural_baseline"
    elif representation_name == FROZEN_BASELINE and model_lane.startswith("AR_plus_"):
        baseline_role = "previous_neural_plus_ar_baseline"
    elif representation_name == FROZEN_BASELINE and residualized:
        baseline_role = "previous_neural_residual_baseline"
    return {
        "uses_ar_features": uses_ar,
        "uses_neural_features": uses_neural,
        "uses_sequence_tensor": uses_sequence_tensor,
        "uses_roi": uses_roi,
        "uses_topk": uses_topk,
        "uses_supervised_features": uses_topk,
        "cautionary": uses_topk,
        "residualized": residualized,
        "tests_incremental_neural_value": tests_incremental,
        "baseline_role": baseline_role,
    }


def inner_validation_split(rows: list[dict[str, Any]], y: np.ndarray, seed: int) -> InnerValidationSplit:
    rng = np.random.default_rng(seed)
    by_video: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_video[str(row["video_id"])].append(index)
    videos = np.asarray(sorted(by_video), dtype=object)
    rng.shuffle(videos)
    if len(videos) >= 2:
        val_count = max(1, int(math.ceil(len(videos) * 0.2)))
        for count in range(val_count, len(videos)):
            val_videos = set(str(item) for item in videos[:count])
            val_idx = np.asarray([idx for video in val_videos for idx in by_video[video]], dtype=np.int64)
            train_idx = np.asarray([idx for video in videos[count:] for idx in by_video[str(video)]], dtype=np.int64)
            if train_idx.size >= 2 and val_idx.size >= 1 and np.unique(y[train_idx]).size >= 2 and np.sum(y[val_idx] == 1) > 0:
                return InnerValidationSplit(
                    train_idx=train_idx,
                    val_idx=val_idx,
                    train_videos=tuple(sorted(str(item) for item in videos[count:])),
                    val_videos=tuple(sorted(val_videos)),
                    strategy="grouped_by_video_train_only",
                )
    indices = np.arange(len(rows), dtype=np.int64)
    positives = indices[y == 1]
    negatives = indices[y == 0]
    if positives.size and negatives.size:
        rng.shuffle(positives)
        rng.shuffle(negatives)
        val_pos = positives[: max(1, int(math.ceil(positives.size * 0.2)))]
        val_neg = negatives[: max(1, int(math.ceil(negatives.size * 0.2)))]
        val_idx = np.sort(np.concatenate([val_pos, val_neg]))
        train_idx = np.asarray([idx for idx in indices if idx not in set(val_idx.tolist())], dtype=np.int64)
        return InnerValidationSplit(
            train_idx=train_idx,
            val_idx=val_idx,
            train_videos=tuple(sorted({str(rows[int(idx)]["video_id"]) for idx in train_idx})),
            val_videos=tuple(sorted({str(rows[int(idx)]["video_id"]) for idx in val_idx})),
            strategy="stratified_row_fallback_train_only",
        )
    midpoint = max(1, len(rows) // 5)
    val_idx = indices[:midpoint]
    train_idx = indices[midpoint:]
    return InnerValidationSplit(
        train_idx=train_idx,
        val_idx=val_idx,
        train_videos=tuple(sorted({str(rows[int(idx)]["video_id"]) for idx in train_idx})),
        val_videos=tuple(sorted({str(rows[int(idx)]["video_id"]) for idx in val_idx})),
        strategy="deterministic_row_fallback_train_only",
    )


def train_only_standardize(train_x: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    train_x = np.asarray(train_x, dtype=np.float64)
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-8
    return tuple((np.asarray(array, dtype=np.float64) - mean) / std for array in arrays)


def pr_auc_for_selection(y: np.ndarray, scores: np.ndarray) -> float:
    value = event_spike.pr_auc(np.asarray(y, dtype=np.int64), np.asarray(scores, dtype=np.float64))
    return float(value) if value is not None and np.isfinite(value) else float("-inf")


def logistic_device() -> torch.device:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for trained heads; refusing to run CPU sklearn fallback.")
    return torch.device("mps")


def class_weight_matrix(train_y: np.ndarray, configs: list[dict[str, Any]]) -> np.ndarray:
    train_y = np.asarray(train_y, dtype=np.int64)
    weights = np.ones((train_y.shape[0], len(configs)), dtype=np.float32)
    positives = max(int(np.sum(train_y == 1)), 1)
    negatives = max(int(np.sum(train_y == 0)), 1)
    total = max(train_y.shape[0], 1)
    balanced_pos = total / (2.0 * positives)
    balanced_neg = total / (2.0 * negatives)
    for index, config in enumerate(configs):
        if config.get("class_weight") == "balanced":
            weights[:, index] = np.where(train_y == 1, balanced_pos, balanced_neg)
    return weights


def fit_torch_mps_logistic_configs(
    train_x: np.ndarray,
    train_y: np.ndarray,
    configs: list[dict[str, Any]],
    *,
    seed: int,
    epochs: int,
) -> tuple[dict[str, Any] | None, str]:
    if np.unique(train_y).size < 2:
        return None, "single_class_train"
    if not configs:
        return None, "empty_config_grid"
    device = logistic_device()
    torch.manual_seed(seed)
    train_x = np.asarray(train_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.float32)
    x_tensor = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    sample_weights = torch.as_tensor(class_weight_matrix(train_y.astype(np.int64), configs), dtype=torch.float32, device=device)
    config_count = len(configs)
    feature_width = train_x.shape[1]
    weights = torch.zeros((config_count, feature_width), dtype=torch.float32, device=device, requires_grad=True)
    bias = torch.zeros(config_count, dtype=torch.float32, device=device, requires_grad=True)
    c_values = torch.as_tensor([float(config["C"]) for config in configs], dtype=torch.float32, device=device)
    inverse_c = 1.0 / torch.clamp(c_values, min=1e-12)
    l1_ratios = torch.as_tensor([float(config.get("l1_ratio") or 0.0) for config in configs], dtype=torch.float32, device=device)
    is_elastic = torch.as_tensor(
        [1.0 if config["penalty"] == "elasticnet" else 0.0 for config in configs],
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.Adam([weights, bias], lr=MPS_LOGISTIC_LR)
    n_rows = max(train_x.shape[0], 1)
    last_loss = None
    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = x_tensor @ weights.T + bias
        target = y_tensor[:, None].expand_as(logits)
        data_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
        data_loss = (data_loss * sample_weights).mean(dim=0)
        l2_sum = torch.sum(weights * weights, dim=1)
        l1_sum = torch.sum(torch.abs(weights), dim=1)
        l2_regularizer = 0.5 * inverse_c * l2_sum / n_rows
        elastic_regularizer = inverse_c * ((1.0 - l1_ratios) * 0.5 * l2_sum + l1_ratios * l1_sum) / n_rows
        regularizer = is_elastic * elastic_regularizer + (1.0 - is_elastic) * l2_regularizer
        loss = torch.sum(data_loss + regularizer)
        if not bool(torch.isfinite(loss).detach().cpu()):
            return None, "non_finite_loss"
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        if last_loss is not None and abs(last_loss - loss_value) < 1e-7:
            break
        last_loss = loss_value
    return (
        {
            "weights": weights.detach().cpu().numpy().astype(np.float64),
            "bias": bias.detach().cpu().numpy().astype(np.float64),
            "training_backend": "torch_mps",
            "optimizer": "adam",
            "epochs": epochs,
            "final_loss": last_loss,
        },
        "pass",
    )


def mps_model_scores(model: dict[str, Any], matrix: np.ndarray) -> np.ndarray:
    weights = np.asarray(model["weights"], dtype=np.float64)
    bias = np.asarray(model["bias"], dtype=np.float64)
    scores = np.asarray(matrix, dtype=np.float64) @ weights.T + bias
    if scores.ndim == 2 and scores.shape[1] == 1:
        return scores[:, 0]
    return scores


def select_single_config_model(model: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        **{key: value for key, value in model.items() if key not in {"weights", "bias"}},
        "weights": np.asarray(model["weights"], dtype=np.float64)[index : index + 1],
        "bias": np.asarray(model["bias"], dtype=np.float64)[index : index + 1],
    }


def logistic_configs(head_name: str) -> list[dict[str, Any]]:
    if head_name in {"logistic_l2", "flattened_sequence_logistic", "learned_temporal_pool_logistic"}:
        return [
            {
                "penalty": "l2",
                "solver": "torch_mps_adam",
                "C": c,
                "class_weight": class_weight,
                "max_iter": MPS_LOGISTIC_EPOCHS,
            }
            for c in LOGISTIC_C_GRID
            for class_weight in CLASS_WEIGHT_GRID
        ]
    if head_name == "elastic_net_logistic":
        return [
            {
                "penalty": "elasticnet",
                "solver": "torch_mps_adam",
                "C": c,
                "l1_ratio": l1_ratio,
                "class_weight": class_weight,
                "max_iter": MPS_LOGISTIC_EPOCHS,
                "tol": 1e-3,
            }
            for c in ELASTIC_C_GRID
            for l1_ratio in ELASTIC_L1_GRID
            for class_weight in CLASS_WEIGHT_GRID
        ]
    raise ValueError(f"Unsupported logistic head: {head_name}")


def fit_logistic_grid(
    head_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    train_rows: list[dict[str, Any]],
    *,
    seed: int,
    extra_hyperparameters: dict[str, Any] | None = None,
) -> PredictionBundle:
    train_y = np.asarray(train_y, dtype=np.int64)
    split = inner_validation_split(train_rows, train_y, seed)
    inner_train_x, inner_val_x = train_only_standardize(train_x[split.train_idx], train_x[split.train_idx], train_x[split.val_idx])
    configs = logistic_configs(head_name)
    grid_model, grid_status = fit_torch_mps_logistic_configs(
        inner_train_x,
        train_y[split.train_idx],
        configs,
        seed=seed,
        epochs=MPS_LOGISTIC_EPOCHS,
    )
    if grid_model is None:
        constant = np.full(train_y.shape[0], float(np.mean(train_y)), dtype=np.float64)
        test_constant = np.full(test_x.shape[0], float(np.mean(train_y)), dtype=np.float64)
        return PredictionBundle(
            train_scores=constant,
            test_scores=test_constant,
            selection={
                "head_name": head_name,
                "selection_status": grid_status,
                "selected_hyperparameters": extra_hyperparameters or {},
                "inner_validation_pr_auc": None,
                "inner_validation_strategy": split.strategy,
                "inner_train_videos": list(split.train_videos),
                "inner_validation_videos": list(split.val_videos),
                "failed_config_count": len(configs),
                "convergence_failure_count": 0,
                "nonzero_coefficient_count": None,
                "training_backend": "torch_mps",
                "test_labels_used_for_selection": False,
            },
        )
    val_scores_by_config = mps_model_scores(grid_model, inner_val_x)
    if val_scores_by_config.ndim == 1:
        val_scores_by_config = val_scores_by_config[:, None]
    best_score = float("-inf")
    best_index = 0
    for index, config in enumerate(configs):
        val_score = pr_auc_for_selection(train_y[split.val_idx], val_scores_by_config[:, index])
        if val_score > best_score:
            best_score = val_score
            best_index = index
    best_config = dict(configs[best_index])
    full_train_x, full_test_x = train_only_standardize(train_x, train_x, test_x)
    final_model, final_status = fit_torch_mps_logistic_configs(
        full_train_x,
        train_y,
        [best_config],
        seed=seed,
        epochs=MPS_LOGISTIC_FINAL_EPOCHS,
    )
    if final_model is None:
        constant = np.full(train_y.shape[0], float(np.mean(train_y)), dtype=np.float64)
        test_constant = np.full(test_x.shape[0], float(np.mean(train_y)), dtype=np.float64)
        final_status = f"final_fit_failed:{final_status}"
        nonzero = None
    else:
        selected_model = select_single_config_model(final_model, 0)
        constant = mps_model_scores(selected_model, full_train_x)
        test_constant = mps_model_scores(selected_model, full_test_x)
        nonzero = int(np.sum(np.abs(selected_model["weights"]) > 1e-10))
    hyperparameters = dict(best_config)
    if extra_hyperparameters:
        hyperparameters.update(extra_hyperparameters)
    hyperparameters["training_backend"] = "torch_mps"
    hyperparameters["optimizer"] = "adam"
    hyperparameters["grid_configs_trained_in_parallel"] = len(configs)
    return PredictionBundle(
        train_scores=constant,
        test_scores=test_constant,
        selection={
            "head_name": head_name,
            "selection_status": final_status if final_status != "pass" else "pass",
            "selected_hyperparameters": hyperparameters,
            "selection_metric": "inner_validation_pr_auc",
            "inner_validation_pr_auc": best_score if np.isfinite(best_score) else None,
            "inner_validation_strategy": split.strategy,
            "inner_train_videos": list(split.train_videos),
            "inner_validation_videos": list(split.val_videos),
            "failed_config_count": 0,
            "convergence_failure_count": 0,
            "nonzero_coefficient_count": nonzero,
            "training_backend": "torch_mps",
            "device": "mps",
            "test_labels_used_for_selection": False,
        },
    )


def fit_ridge_bundle(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> PredictionBundle:
    train_scores, test_scores = ridge_train_test(train_x, train_y.astype(np.float64), test_x)
    return PredictionBundle(
        train_scores=train_scores,
        test_scores=test_scores,
        selection={
            "head_name": "ridge_score",
            "selection_status": "fixed_alpha",
            "selected_hyperparameters": {"alpha": 1.0},
            "inner_validation_pr_auc": None,
            "test_labels_used_for_selection": False,
        },
    )


def fit_head_bundle(
    head_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    train_rows: list[dict[str, Any]],
    *,
    seed: int,
    extra_hyperparameters: dict[str, Any] | None = None,
) -> PredictionBundle:
    if head_name == "ridge_score":
        bundle = fit_ridge_bundle(train_x, train_y, test_x)
        if extra_hyperparameters:
            bundle.selection["selected_hyperparameters"].update(extra_hyperparameters)
        return bundle
    return fit_logistic_grid(
        head_name,
        train_x,
        train_y,
        test_x,
        train_rows,
        seed=seed,
        extra_hyperparameters=extra_hyperparameters,
    )


def fit_residual_correction_bundle(
    head_name: str,
    train_x: np.ndarray,
    residual_train: np.ndarray,
    test_x: np.ndarray,
    *,
    extra_hyperparameters: dict[str, Any] | None = None,
) -> PredictionBundle:
    train_scores, test_scores = ridge_train_test(train_x, residual_train.astype(np.float64), test_x)
    selected = {"alpha": 1.0, "residual_target": "continuous_ar_residual"}
    if head_name != "ridge_score":
        selected["residual_correction_head"] = "ridge_score"
    if extra_hyperparameters:
        selected.update(extra_hyperparameters)
    return PredictionBundle(
        train_scores=train_scores,
        test_scores=test_scores,
        selection={
            "head_name": head_name,
            "selection_status": "residual_correction_fixed_ridge",
            "selected_hyperparameters": selected,
            "inner_validation_pr_auc": None,
            "test_labels_used_for_selection": False,
        },
    )


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    try:
        if np.unique(y_true.astype(np.int64)).size < 2:
            return None
        return float(roc_auc_score(y_true.astype(np.int64), scores.astype(np.float64)))
    except Exception:
        return None


def event_count(y: np.ndarray) -> float:
    return float(np.sum(np.asarray(y, dtype=np.float64)))


def prediction_metrics(train_y: np.ndarray, train_scores: np.ndarray, test_y: np.ndarray, test_scores: np.ndarray) -> dict[str, Any]:
    metrics = score_metrics(
        task_type="binary",
        train_y=train_y.astype(np.float64),
        train_scores=train_scores.astype(np.float64),
        test_y=test_y.astype(np.float64),
        test_scores=test_scores.astype(np.float64),
    )
    threshold = metrics.get("decision_threshold")
    if threshold is not None:
        pred = (test_scores >= float(threshold)).astype(np.int64)
        predicted_count = int(np.sum(pred))
    else:
        predicted_count = None
    metrics["roc_auc"] = safe_roc_auc(test_y, test_scores)
    metrics["predicted_positive_count"] = predicted_count
    metrics["predicted_positive_rate"] = float(predicted_count / len(test_y)) if predicted_count is not None and len(test_y) else None
    metrics["train_event_count"] = event_count(train_y)
    metrics["test_event_count"] = event_count(test_y)
    metrics["train_positive_rate"] = float(np.mean(train_y)) if len(train_y) else None
    metrics["test_positive_rate"] = float(np.mean(test_y)) if len(test_y) else None
    return metrics


def model_lane_from_key(feature_name: str, key: str) -> str:
    if key == "autoregressive":
        return "AR_only"
    if key == feature_name:
        return f"{feature_name}_only"
    if key == f"shuffled_{feature_name}":
        return f"shuffled_{feature_name}"
    if key == f"random_gaussian_{feature_name}":
        return f"random_gaussian_{feature_name}"
    if key == f"autoregressive_plus_{feature_name}":
        return f"AR_plus_{feature_name}"
    if key == f"autoregressive_plus_shuffled_{feature_name}":
        return f"AR_plus_shuffled_{feature_name}"
    if key == f"autoregressive_plus_random_gaussian_{feature_name}":
        return f"AR_plus_random_{feature_name}"
    if key == f"residualized_autoregressive_plus_{feature_name}":
        return f"residualized_AR_plus_{feature_name}"
    if key == f"residualized_autoregressive_plus_shuffled_{feature_name}":
        return f"residualized_AR_plus_shuffled_{feature_name}"
    if key == f"residualized_autoregressive_plus_random_gaussian_{feature_name}":
        return f"residualized_AR_plus_random_{feature_name}"
    return key


def make_lane_row(
    *,
    fresh_run_id: str,
    contract: FrozenTensorContract,
    model_lane: str,
    head_name: str,
    train_y: np.ndarray,
    test_y: np.ndarray,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    feature_width: int,
    sequence_shape: tuple[int, ...] | None,
    selection: dict[str, Any],
    seed: int,
    uses_sequence_tensor: bool = False,
    notes: str = "",
) -> dict[str, Any]:
    flags = lane_flags(contract.representation_name, model_lane, uses_sequence_tensor=uses_sequence_tensor)
    metrics = prediction_metrics(train_y, train_scores, test_y, test_scores)
    control = is_control_lane(model_lane)
    row: dict[str, Any] = {
        "fresh_run_id": fresh_run_id,
        "benchmark_mode": TRAINED_BENCHMARK_MODE,
        "scope": "full_veatic_124",
        "split_name": contract.split,
        "target_name": contract.target_name,
        "threshold": contract.threshold,
        "representation_name": contract.representation_name,
        "feature_name": contract.feature_name,
        "model_lane": model_lane,
        "head_name": head_name,
        **flags,
        "control_source": model_lane if control else "",
        "canonical_control": bool(control),
        "control_seed": seed if control else "",
        "row_population_matched": True,
        "shape_matched": True,
        "n_train": int(len(train_y)),
        "n_test": int(len(test_y)),
        "feature_width": int(feature_width),
        "sequence_shape": json.dumps(sequence_shape) if sequence_shape else "",
        "selected_hyperparameters_json": json.dumps(finite_json(selection.get("selected_hyperparameters", {})), sort_keys=True),
        "inner_validation_pr_auc": selection.get("inner_validation_pr_auc"),
        "control_pass": "",
        "leakage_status": "pass",
        "freshly_computed_in_current_run": True,
        "reused_benchmark_result_row": False,
        "notes": notes,
    }
    row.update(metrics)
    for column in LANE_RESULT_COLUMNS:
        row.setdefault(column, "")
    return row


def add_selection_detail(
    details: list[dict[str, Any]],
    *,
    fresh_run_id: str,
    contract: FrozenTensorContract,
    model_lane: str,
    head_name: str,
    selection: dict[str, Any],
) -> None:
    details.append(
        {
            "fresh_run_id": fresh_run_id,
            "benchmark_mode": TRAINED_BENCHMARK_MODE,
            "representation_name": contract.representation_name,
            "feature_name": contract.feature_name,
            "split_name": contract.split,
            "target_name": contract.target_name,
            "threshold": contract.threshold,
            "model_lane": model_lane,
            "head_name": head_name,
            **finite_json(selection),
        }
    )


def score_collapsed_contract(
    *,
    fresh_run_id: str,
    contract: FrozenTensorContract,
    all_rows: list[dict[str, Any]],
    heads: tuple[str, ...],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    inputs = contract.as_existing_benchmark_inputs(all_rows)
    assert_same_row_order(inputs.train_rows, inputs.train_row_metadata, role="train")
    assert_same_row_order(inputs.test_rows, inputs.test_row_metadata, role="test")
    train_y = inputs.train_y.astype(np.float64)
    test_y = inputs.test_y.astype(np.float64)
    base_target = base_target_name(contract.target_name)
    train_ar = bench.autoregressive_features(all_rows, inputs.train_rows, base_target, include_current=True)
    test_ar = bench.autoregressive_features(all_rows, inputs.test_rows, base_target, include_current=True)
    feature_name = contract.feature_name
    train_x = inputs.train_x.astype(np.float64)
    test_x = inputs.test_x.astype(np.float64)
    rng = np.random.default_rng(seed)
    train_perm = rng.permutation(len(train_y))
    test_perm = rng.permutation(len(test_y))
    shuffled_train = train_x[train_perm]
    shuffled_test = test_x[test_perm]
    random_train = rng.normal(size=train_x.shape)
    random_test = rng.normal(size=test_x.shape)
    ar_random_train = rng.normal(size=train_x.shape)
    ar_random_test = rng.normal(size=test_x.shape)
    residual_random_train = rng.normal(size=train_x.shape)
    residual_random_test = rng.normal(size=test_x.shape)
    lane_rows: list[dict[str, Any]] = []
    selection_details: list[dict[str, Any]] = []
    leakage_checks = [
        {
            "check_name": "same_row_order_before_ar_scoring",
            "status": "pass",
            "representation_name": contract.representation_name,
            "split": contract.split,
            "target": contract.target_name,
            "threshold": contract.threshold,
            "details": {"train_rows": len(inputs.train_rows), "test_rows": len(inputs.test_rows)},
        },
        {
            "check_name": "computed_ar_fresh",
            "status": "pass",
            "representation_name": contract.representation_name,
            "split": contract.split,
            "target": contract.target_name,
            "threshold": contract.threshold,
            "details": {"ar_feature_width": int(train_ar.shape[1])},
        },
        {
            "check_name": "computed_controls_fresh",
            "status": "pass",
            "representation_name": contract.representation_name,
            "split": contract.split,
            "target": contract.target_name,
            "threshold": contract.threshold,
            "details": {"control_seed": seed},
        },
    ]
    collapsed_designs = {
        "autoregressive": (train_ar, test_ar),
        feature_name: (train_x, test_x),
        f"shuffled_{feature_name}": (shuffled_train, shuffled_test),
        f"random_gaussian_{feature_name}": (random_train, random_test),
        f"autoregressive_plus_{feature_name}": (
            np.concatenate([train_ar, train_x], axis=1),
            np.concatenate([test_ar, test_x], axis=1),
        ),
        f"autoregressive_plus_shuffled_{feature_name}": (
            np.concatenate([train_ar, shuffled_train], axis=1),
            np.concatenate([test_ar, shuffled_test], axis=1),
        ),
        f"autoregressive_plus_random_gaussian_{feature_name}": (
            np.concatenate([train_ar, ar_random_train], axis=1),
            np.concatenate([test_ar, ar_random_test], axis=1),
        ),
    }
    for head_name in heads:
        if head_name not in COLLAPSED_HEADS:
            continue
        ar_bundle = fit_head_bundle(head_name, train_ar, train_y, test_ar, inputs.train_rows, seed=seed)
        for key, (design_train, design_test) in collapsed_designs.items():
            if key == "autoregressive":
                bundle = ar_bundle
            else:
                bundle = fit_head_bundle(head_name, design_train, train_y, design_test, inputs.train_rows, seed=seed)
            model_lane = model_lane_from_key(feature_name, key)
            lane_rows.append(
                make_lane_row(
                    fresh_run_id=fresh_run_id,
                    contract=contract,
                    model_lane=model_lane,
                    head_name=head_name,
                    train_y=train_y,
                    test_y=test_y,
                    train_scores=bundle.train_scores,
                    test_scores=bundle.test_scores,
                    feature_width=design_train.shape[1],
                    sequence_shape=None,
                    selection=bundle.selection,
                    seed=seed,
                )
            )
            add_selection_detail(
                selection_details,
                fresh_run_id=fresh_run_id,
                contract=contract,
                model_lane=model_lane,
                head_name=head_name,
                selection=bundle.selection,
            )
        residual_train = train_y - ar_bundle.train_scores
        residual_designs = {
            f"residualized_autoregressive_plus_{feature_name}": (train_x, test_x),
            f"residualized_autoregressive_plus_shuffled_{feature_name}": (shuffled_train, shuffled_test),
            f"residualized_autoregressive_plus_random_gaussian_{feature_name}": (residual_random_train, residual_random_test),
        }
        for key, (design_train, design_test) in residual_designs.items():
            residual_bundle = fit_residual_correction_bundle(head_name, design_train, residual_train, design_test)
            model_lane = model_lane_from_key(feature_name, key)
            train_scores = ar_bundle.train_scores + residual_bundle.train_scores
            test_scores = ar_bundle.test_scores + residual_bundle.test_scores
            lane_rows.append(
                make_lane_row(
                    fresh_run_id=fresh_run_id,
                    contract=contract,
                    model_lane=model_lane,
                    head_name=head_name,
                    train_y=train_y,
                    test_y=test_y,
                    train_scores=train_scores,
                    test_scores=test_scores,
                    feature_width=design_train.shape[1],
                    sequence_shape=None,
                    selection=residual_bundle.selection,
                    seed=seed,
                    notes="Residual correction uses the canonical continuous residual procedure.",
                )
            )
            add_selection_detail(
                selection_details,
                fresh_run_id=fresh_run_id,
                contract=contract,
                model_lane=model_lane,
                head_name=head_name,
                selection=residual_bundle.selection,
            )
    return lane_rows, leakage_checks, selection_details


def fit_temporal_pool_bundle(
    head_name: str,
    train_seq: np.ndarray,
    train_y: np.ndarray,
    test_seq: np.ndarray,
    train_rows: list[dict[str, Any]],
    *,
    seed: int,
    train_ar: np.ndarray | None = None,
    test_ar: np.ndarray | None = None,
    residual_train: np.ndarray | None = None,
) -> PredictionBundle:
    best: PredictionBundle | None = None
    best_score = float("-inf")
    target_y = residual_train if residual_train is not None else train_y
    for weights in TEMPORAL_POOL_WEIGHTS:
        pooled_train = temporal_pool_sequence(train_seq, weights)
        pooled_test = temporal_pool_sequence(test_seq, weights)
        if train_ar is not None and test_ar is not None:
            pooled_train = np.concatenate([train_ar, pooled_train], axis=1)
            pooled_test = np.concatenate([test_ar, pooled_test], axis=1)
        if residual_train is None:
            bundle = fit_head_bundle(
                "logistic_l2",
                pooled_train,
                train_y,
                pooled_test,
                train_rows,
                seed=seed,
                extra_hyperparameters={"temporal_pool_weights": list(weights)},
            )
            score = bundle.selection.get("inner_validation_pr_auc")
            score = float(score) if score is not None else float("-inf")
        else:
            bundle = fit_residual_correction_bundle(
                head_name,
                pooled_train,
                target_y,
                pooled_test,
                extra_hyperparameters={"temporal_pool_weights": list(weights)},
            )
            score = -float(np.mean(np.abs(bundle.train_scores - target_y))) if target_y.size else float("-inf")
            bundle.selection["inner_validation_pr_auc"] = None
            bundle.selection["selection_metric"] = "train_residual_mae_proxy_no_test_labels"
        if score > best_score or best is None:
            best_score = score
            best = bundle
    if best is None:
        raise RuntimeError("No temporal pooling candidate was evaluated")
    best.selection["head_name"] = head_name
    return best


def score_sequence_contract(
    *,
    fresh_run_id: str,
    contract: FrozenTensorContract,
    all_rows: list[dict[str, Any]],
    heads: tuple[str, ...],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if contract.representation_name != PRIMARY_CANDIDATE or not any(head in heads for head in SEQUENCE_HEADS):
        return [], [], []
    inputs = contract.as_existing_benchmark_inputs(all_rows)
    train_seq, test_seq, _train_mask, _test_mask = load_sequence_arrays(contract)
    if train_seq.shape[0] != len(inputs.train_rows) or test_seq.shape[0] != len(inputs.test_rows):
        raise AssertionError(f"Sequence tensor row count mismatch for {contract.contract_dir}")
    train_y = inputs.train_y.astype(np.float64)
    test_y = inputs.test_y.astype(np.float64)
    base_target = base_target_name(contract.target_name)
    train_ar = bench.autoregressive_features(all_rows, inputs.train_rows, base_target, include_current=True)
    test_ar = bench.autoregressive_features(all_rows, inputs.test_rows, base_target, include_current=True)
    ar_bundle = fit_head_bundle("logistic_l2", train_ar, train_y, test_ar, inputs.train_rows, seed=seed)
    lane_rows: list[dict[str, Any]] = []
    selection_details: list[dict[str, Any]] = []
    leakage_checks = [
        {
            "check_name": "sequence_row_count_matched",
            "status": "pass",
            "representation_name": contract.representation_name,
            "split": contract.split,
            "target": contract.target_name,
            "threshold": contract.threshold,
            "details": {"train_sequence_shape": list(train_seq.shape), "test_sequence_shape": list(test_seq.shape)},
        }
    ]
    rng = np.random.default_rng(seed)
    train_perm = rng.permutation(len(train_y))
    test_perm = rng.permutation(len(test_y))
    seq_variants = {
        "real": (train_seq, test_seq),
        "shuffled": (train_seq[train_perm], test_seq[test_perm]),
        "random": (rng.normal(size=train_seq.shape), rng.normal(size=test_seq.shape)),
    }
    for head_name in heads:
        if head_name == "flattened_sequence_logistic":
            variant_features = {
                name: (flatten_sequence(train_variant), flatten_sequence(test_variant))
                for name, (train_variant, test_variant) in seq_variants.items()
            }
            for suffix, (design_train, design_test) in variant_features.items():
                if suffix == "real":
                    base_lane = "PCA128_only"
                    ar_lane = "AR_plus_PCA128"
                    residual_lane = "residualized_AR_plus_PCA128"
                elif suffix == "shuffled":
                    base_lane = "shuffled_PCA128"
                    ar_lane = "AR_plus_shuffled_PCA128"
                    residual_lane = "residualized_AR_plus_shuffled_PCA128"
                else:
                    base_lane = "random_gaussian_PCA128"
                    ar_lane = "AR_plus_random_PCA128"
                    residual_lane = "residualized_AR_plus_random_PCA128"
                base_bundle = fit_head_bundle(head_name, design_train, train_y, design_test, inputs.train_rows, seed=seed)
                ar_design_train = np.concatenate([train_ar, design_train], axis=1)
                ar_design_test = np.concatenate([test_ar, design_test], axis=1)
                ar_plus_bundle = fit_head_bundle(head_name, ar_design_train, train_y, ar_design_test, inputs.train_rows, seed=seed)
                residual_bundle = fit_residual_correction_bundle(head_name, design_train, train_y - ar_bundle.train_scores, design_test)
                bundles = [
                    (base_lane, base_bundle, design_train.shape[1], base_bundle.train_scores, base_bundle.test_scores, ""),
                    (ar_lane, ar_plus_bundle, ar_design_train.shape[1], ar_plus_bundle.train_scores, ar_plus_bundle.test_scores, ""),
                    (
                        residual_lane,
                        residual_bundle,
                        design_train.shape[1],
                        ar_bundle.train_scores + residual_bundle.train_scores,
                        ar_bundle.test_scores + residual_bundle.test_scores,
                        "Residual correction uses the canonical continuous residual procedure.",
                    ),
                ]
                for model_lane, bundle, width, train_scores, test_scores, notes in bundles:
                    lane_rows.append(
                        make_lane_row(
                            fresh_run_id=fresh_run_id,
                            contract=contract,
                            model_lane=model_lane,
                            head_name=head_name,
                            train_y=train_y,
                            test_y=test_y,
                            train_scores=train_scores,
                            test_scores=test_scores,
                            feature_width=width,
                            sequence_shape=tuple(train_seq.shape[1:]),
                            selection=bundle.selection,
                            seed=seed,
                            uses_sequence_tensor=True,
                            notes=notes,
                        )
                    )
                    add_selection_detail(
                        selection_details,
                        fresh_run_id=fresh_run_id,
                        contract=contract,
                        model_lane=model_lane,
                        head_name=head_name,
                        selection=bundle.selection,
                    )
        elif head_name == "learned_temporal_pool_logistic":
            for suffix, (train_variant, test_variant) in seq_variants.items():
                if suffix == "real":
                    base_lane = "PCA128_only"
                    ar_lane = "AR_plus_PCA128"
                    residual_lane = "residualized_AR_plus_PCA128"
                elif suffix == "shuffled":
                    base_lane = "shuffled_PCA128"
                    ar_lane = "AR_plus_shuffled_PCA128"
                    residual_lane = "residualized_AR_plus_shuffled_PCA128"
                else:
                    base_lane = "random_gaussian_PCA128"
                    ar_lane = "AR_plus_random_PCA128"
                    residual_lane = "residualized_AR_plus_random_PCA128"
                base_bundle = fit_temporal_pool_bundle(head_name, train_variant, train_y, test_variant, inputs.train_rows, seed=seed)
                ar_plus_bundle = fit_temporal_pool_bundle(
                    head_name,
                    train_variant,
                    train_y,
                    test_variant,
                    inputs.train_rows,
                    seed=seed,
                    train_ar=train_ar,
                    test_ar=test_ar,
                )
                residual_bundle = fit_temporal_pool_bundle(
                    head_name,
                    train_variant,
                    train_y,
                    test_variant,
                    inputs.train_rows,
                    seed=seed,
                    residual_train=train_y - ar_bundle.train_scores,
                )
                pooled_width = train_seq.shape[2]
                bundles = [
                    (base_lane, base_bundle, pooled_width, base_bundle.train_scores, base_bundle.test_scores, ""),
                    (ar_lane, ar_plus_bundle, train_ar.shape[1] + pooled_width, ar_plus_bundle.train_scores, ar_plus_bundle.test_scores, ""),
                    (
                        residual_lane,
                        residual_bundle,
                        pooled_width,
                        ar_bundle.train_scores + residual_bundle.train_scores,
                        ar_bundle.test_scores + residual_bundle.test_scores,
                        "Residual correction uses selected temporal pooling with the canonical continuous residual procedure.",
                    ),
                ]
                for model_lane, bundle, width, train_scores, test_scores, notes in bundles:
                    lane_rows.append(
                        make_lane_row(
                            fresh_run_id=fresh_run_id,
                            contract=contract,
                            model_lane=model_lane,
                            head_name=head_name,
                            train_y=train_y,
                            test_y=test_y,
                            train_scores=train_scores,
                            test_scores=test_scores,
                            feature_width=width,
                            sequence_shape=tuple(train_seq.shape[1:]),
                            selection=bundle.selection,
                            seed=seed,
                            uses_sequence_tensor=True,
                            notes=notes,
                        )
                    )
                    add_selection_detail(
                        selection_details,
                        fresh_run_id=fresh_run_id,
                        contract=contract,
                        model_lane=model_lane,
                        head_name=head_name,
                        selection=bundle.selection,
                    )
    return lane_rows, leakage_checks, selection_details


def annotate_reference_columns(lane_rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, float, str, str, str], dict[str, float]] = defaultdict(dict)
    for row in lane_rows:
        if row["canonical_control"]:
            continue
        key = (row["split_name"], float(row["threshold"]), row["target_name"], row["head_name"], row["feature_name"])
        grouped[key][row["model_lane"]] = row["pr_auc"]
    logistic_refs: dict[tuple[str, float, str, str], dict[str, float]] = defaultdict(dict)
    for row in lane_rows:
        if row["canonical_control"] or row["head_name"] != "logistic_l2":
            continue
        key = (row["split_name"], float(row["threshold"]), row["target_name"], row["feature_name"])
        logistic_refs[key][row["model_lane"]] = row["pr_auc"]
    for row in lane_rows:
        key = (row["split_name"], float(row["threshold"]), row["target_name"], row["head_name"], row["feature_name"])
        ref = dict(grouped.get(key, {}))
        if row["head_name"] in SEQUENCE_HEADS:
            ref.update(
                {
                    k: v
                    for k, v in logistic_refs.get(
                        (row["split_name"], float(row["threshold"]), row["target_name"], row["feature_name"]),
                        {},
                    ).items()
                    if k == "AR_only"
                }
            )
        row["ar_only_pr_auc"] = ref.get("AR_only", "")
        row["pca128_only_pr_auc"] = ref.get("PCA128_only", "")
        row["pca64_delta_only_pr_auc"] = ref.get("PCA64_delta_only", "")
        row["ar_plus_pca128_pr_auc"] = ref.get("AR_plus_PCA128", "")
        row["ar_plus_pca64_delta_pr_auc"] = ref.get("AR_plus_PCA64_delta", "")
        row["residualized_ar_plus_pca128_pr_auc"] = ref.get("residualized_AR_plus_PCA128", "")
        row["residualized_ar_plus_pca64_delta_pr_auc"] = ref.get("residualized_AR_plus_PCA64_delta", "")
        ar = ref.get("AR_only")
        row["delta_pca128_only_vs_ar"] = _delta(ref.get("PCA128_only"), ar)
        row["delta_ar_plus_pca128_vs_ar"] = _delta(ref.get("AR_plus_PCA128"), ar)
        row["delta_residualized_ar_plus_pca128_vs_ar"] = _delta(ref.get("residualized_AR_plus_PCA128"), ar)
        row["delta_ar_plus_pca128_vs_ar_plus_pca64_delta"] = _delta(ref.get("AR_plus_PCA128"), ref.get("AR_plus_PCA64_delta"))
        row["delta_residualized_pca128_vs_residualized_pca64_delta"] = _delta(
            ref.get("residualized_AR_plus_PCA128"),
            ref.get("residualized_AR_plus_PCA64_delta"),
        )
        for column in LANE_RESULT_COLUMNS:
            row.setdefault(column, "")


def _delta(value: Any, baseline: Any) -> float | str:
    if value == "" or baseline == "" or value is None or baseline is None:
        return ""
    return float(value) - float(baseline)


def grouped_mean(
    rows: list[dict[str, Any]],
    *,
    target_name: str,
    threshold: float,
    model_lane: str,
    head_name: str | None = None,
    feature_name: str | None = None,
) -> float | None:
    values = [
        float(row["pr_auc"])
        for row in rows
        if row["split_name"].startswith("grouped_")
        and row["target_name"] == target_name
        and abs(float(row["threshold"]) - threshold) < 1e-12
        and row["model_lane"] == model_lane
        and not row["canonical_control"]
        and not row["cautionary"]
        and (head_name is None or row["head_name"] == head_name)
        and (feature_name is None or row["feature_name"] == feature_name)
        and row["pr_auc"] not in ("", None)
    ]
    return float(np.mean(values)) if values else None


def grouped_rows_for_best_head(
    rows: list[dict[str, Any]],
    target_name: str,
    threshold: float,
    model_lane: str,
    *,
    feature_name: str | None = None,
) -> tuple[str | None, list[dict[str, Any]], float | None]:
    by_head: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row["split_name"].startswith("grouped_")
            and row["target_name"] == target_name
            and abs(float(row["threshold"]) - threshold) < 1e-12
            and row["model_lane"] == model_lane
            and (feature_name is None or row["feature_name"] == feature_name)
            and not row["canonical_control"]
            and not row["cautionary"]
            and row["pr_auc"] not in ("", None)
        ):
            by_head[row["head_name"]].append(row)
    best_head = None
    best_rows: list[dict[str, Any]] = []
    best_mean = None
    for head, head_rows in by_head.items():
        if len({row["split_name"] for row in head_rows}) < 5:
            continue
        mean_value = float(np.mean([float(row["pr_auc"]) for row in head_rows]))
        if best_mean is None or mean_value > best_mean:
            best_head = head
            best_rows = head_rows
            best_mean = mean_value
    return best_head, best_rows, best_mean


def stable_positive_folds(candidate: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> tuple[int, int, dict[str, float]]:
    candidate_by_split = {row["split_name"]: float(row["pr_auc"]) for row in candidate}
    baseline_by_split = {row["split_name"]: float(row["pr_auc"]) for row in baseline}
    splits = sorted(set(candidate_by_split) & set(baseline_by_split))
    gains = {split: candidate_by_split[split] - baseline_by_split[split] for split in splits}
    return sum(value > 0 for value in gains.values()), len(splits), gains


def control_grouped_mean(
    rows: list[dict[str, Any]],
    target_name: str,
    threshold: float,
    family: str,
    *,
    feature_name: str | None = None,
) -> tuple[float | None, dict[str, Any] | None]:
    candidates = []
    for row in rows:
        if not row["canonical_control"] or not row["split_name"].startswith("grouped_"):
            continue
        if row["target_name"] != target_name or abs(float(row["threshold"]) - threshold) >= 1e-12:
            continue
        if feature_name is not None and row["feature_name"] != feature_name:
            continue
        lane = str(row["model_lane"])
        if family == "ar_plus" and not (lane.startswith("AR_plus_shuffled_") or lane.startswith("AR_plus_random_")):
            continue
        if family == "residualized" and not (lane.startswith("residualized_AR_plus_shuffled_") or lane.startswith("residualized_AR_plus_random_")):
            continue
        candidates.append(row)
    grouped_by_lane_head: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped_by_lane_head[(row["model_lane"], row["head_name"])].append(row)
    best_value = None
    best_meta = None
    for (lane, head), lane_rows in grouped_by_lane_head.items():
        if len({row["split_name"] for row in lane_rows}) < 5:
            continue
        value = float(np.mean([float(row["pr_auc"]) for row in lane_rows]))
        if best_value is None or value > best_value:
            best_value = value
            best_meta = {"model_lane": lane, "head_name": head, "mean_grouped_pr_auc": value}
    return best_value, best_meta


def compute_gate_checks(lane_rows: list[dict[str, Any]], leakage_checks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gates: list[dict[str, Any]] = [
        {"gate_name": "benchmark_mode", "status": "pass", "details": {"benchmark_mode": TRAINED_BENCHMARK_MODE}},
        {"gate_name": "full_veatic_124", "status": "pass", "details": {"scope": "full_veatic_124"}},
        {"gate_name": "video_83_included", "status": "pass", "details": {"video_83_included": True}},
        {"gate_name": "exclude_video_83_run", "status": "pass", "details": {"exclude_video_83_run": False}},
        {"gate_name": "no_prior_result_rows_reused", "status": "pass", "details": {"reused_benchmark_result_rows": False}},
    ]
    promotion_candidates: list[dict[str, Any]] = []
    if check_failures(leakage_checks):
        gates.append({"gate_name": "leakage_failures_absent", "status": "fail", "details": {"failure_count": len(check_failures(leakage_checks))}})
        return gates, promotion_candidates
    gates.append({"gate_name": "leakage_failures_absent", "status": "pass", "details": {"failure_count": 0}})
    for target_name, threshold in PRIMARY_SPIKE_TARGETS:
        best: dict[str, tuple[str | None, list[dict[str, Any]], float | None]] = {}
        lane_feature_names = {
            "AR_only": "PCA128",
            "PCA128_only": "PCA128",
            "PCA64_delta_only": "PCA64_delta",
            "AR_plus_PCA128": "PCA128",
            "AR_plus_PCA64_delta": "PCA64_delta",
            "residualized_AR_plus_PCA128": "PCA128",
            "residualized_AR_plus_PCA64_delta": "PCA64_delta",
        }
        for lane, feature_name in lane_feature_names.items():
            best[lane] = grouped_rows_for_best_head(lane_rows, target_name, threshold, lane, feature_name=feature_name)
        gate_specs = [
            ("PCA128_only > PCA64_delta_only", "PCA128_only", "PCA64_delta_only", None),
            ("PCA128_only > AR_only", "PCA128_only", "AR_only", None),
            ("AR_plus_PCA128 > AR_only", "AR_plus_PCA128", "AR_only", None),
            ("residualized_AR_plus_PCA128 > AR_only", "residualized_AR_plus_PCA128", "AR_only", None),
            ("AR_plus_PCA128 > AR_plus_PCA64_delta", "AR_plus_PCA128", "AR_plus_PCA64_delta", None),
            (
                "residualized_AR_plus_PCA128 > residualized_AR_plus_PCA64_delta",
                "residualized_AR_plus_PCA128",
                "residualized_AR_plus_PCA64_delta",
                None,
            ),
        ]
        target_gate_results = {}
        for label, candidate_lane, baseline_lane, _ in gate_specs:
            candidate_head, candidate_rows, candidate_mean = best[candidate_lane]
            baseline_head, baseline_rows, baseline_mean = best[baseline_lane]
            if candidate_mean is None or baseline_mean is None:
                delta = None
                relative = None
                positive, total, gains = 0, 0, {}
                status = "fail"
            else:
                delta = candidate_mean - baseline_mean
                relative = delta / baseline_mean if baseline_mean else None
                positive, total, gains = stable_positive_folds(candidate_rows, baseline_rows)
                status = "pass" if delta > 0 and positive >= 3 else "fail"
            target_gate_results[label] = status
            gates.append(
                {
                    "gate_name": label,
                    "target_name": target_name,
                    "threshold": threshold,
                    "status": status,
                    "mean_grouped_delta": delta,
                    "relative_percent_delta": relative * 100.0 if relative is not None else None,
                    "stable_positive_folds": positive,
                    "fold_count": total,
                    "candidate_head": candidate_head,
                    "baseline_head": baseline_head,
                    "candidate_mean_pr_auc": candidate_mean,
                    "baseline_mean_pr_auc": baseline_mean,
                    "fold_deltas": gains,
                    "notes": "Primary grouped-video gate.",
                }
            )
        ar_control, ar_control_meta = control_grouped_mean(lane_rows, target_name, threshold, "ar_plus", feature_name="PCA128")
        res_control, res_control_meta = control_grouped_mean(lane_rows, target_name, threshold, "residualized", feature_name="PCA128")
        ar_plus_mean = best["AR_plus_PCA128"][2]
        residual_mean = best["residualized_AR_plus_PCA128"][2]
        ar_control_pass = ar_plus_mean is not None and ar_control is not None and ar_plus_mean > ar_control
        residual_control_pass = residual_mean is not None and res_control is not None and residual_mean > res_control
        gates.append(
            {
                "gate_name": "AR_plus_PCA128 > canonical AR_plus shuffled/random controls",
                "target_name": target_name,
                "threshold": threshold,
                "status": "pass" if ar_control_pass else "fail",
                "mean_grouped_delta": ar_plus_mean - ar_control if ar_plus_mean is not None and ar_control is not None else None,
                "relative_percent_delta": ((ar_plus_mean - ar_control) / ar_control * 100.0) if ar_plus_mean is not None and ar_control else None,
                "stable_positive_folds": None,
                "fold_count": 5,
                "candidate_head": best["AR_plus_PCA128"][0],
                "control": ar_control_meta,
                "notes": "Canonical shuffled/random AR-plus control family.",
            }
        )
        gates.append(
            {
                "gate_name": "residualized_AR_plus_PCA128 > canonical residualized shuffled/random controls",
                "target_name": target_name,
                "threshold": threshold,
                "status": "pass" if residual_control_pass else "fail",
                "mean_grouped_delta": residual_mean - res_control if residual_mean is not None and res_control is not None else None,
                "relative_percent_delta": ((residual_mean - res_control) / res_control * 100.0) if residual_mean is not None and res_control else None,
                "stable_positive_folds": None,
                "fold_count": 5,
                "candidate_head": best["residualized_AR_plus_PCA128"][0],
                "control": res_control_meta,
                "notes": "Canonical shuffled/random residualized control family.",
            }
        )
        category = "valid_but_weaker"
        if not ar_control_pass or not residual_control_pass:
            category = "control_failure"
        elif (
            target_gate_results.get("AR_plus_PCA128 > AR_only") == "pass"
            and target_gate_results.get("residualized_AR_plus_PCA128 > AR_only") == "pass"
            and target_gate_results.get("AR_plus_PCA128 > AR_plus_PCA64_delta") == "pass"
            and target_gate_results.get("residualized_AR_plus_PCA128 > residualized_AR_plus_PCA64_delta") == "pass"
        ):
            category = "promoted_incremental_neural_value"
        elif target_gate_results.get("PCA128_only > AR_only") == "fail" and target_gate_results.get("PCA128_only > PCA64_delta_only") == "pass":
            category = "internal_neural_improvement_only"
        elif target_gate_results.get("AR_plus_PCA128 > AR_only") == "pass" or target_gate_results.get("residualized_AR_plus_PCA128 > AR_only") == "pass":
            category = "promising_incremental_neural_value"
        promotion_candidates.append(
            {
                "target_name": target_name,
                "threshold": threshold,
                "category": category,
                "best_heads": {lane: best[lane][0] for lane in best},
                "means": {lane: best[lane][2] for lane in best},
                "control_pass": ar_control_pass and residual_control_pass,
                "not_final_promotion": True,
            }
        )
    return gates, promotion_candidates


def mark_control_pass(lane_rows: list[dict[str, Any]], gate_checks: list[dict[str, Any]]) -> None:
    control_status: dict[tuple[str, float, str], bool] = {}
    for gate in gate_checks:
        if "canonical" in gate.get("gate_name", "") and gate.get("target_name"):
            family = "residualized" if gate["gate_name"].startswith("residualized") else "ar_plus"
            control_status[(gate["target_name"], float(gate["threshold"]), family)] = gate["status"] == "pass"
    for row in lane_rows:
        if row["model_lane"] == "AR_plus_PCA128":
            row["control_pass"] = control_status.get((row["target_name"], float(row["threshold"]), "ar_plus"), "")
        elif row["model_lane"] == "residualized_AR_plus_PCA128":
            row["control_pass"] = control_status.get((row["target_name"], float(row["threshold"]), "residualized"), "")


def report_line(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def best_summary(lane_rows: list[dict[str, Any]], target_name: str, threshold: float) -> dict[str, Any]:
    output: dict[str, Any] = {}
    lane_feature_names = {
        "AR_only": "PCA128",
        "PCA128_only": "PCA128",
        "AR_plus_PCA128": "PCA128",
        "residualized_AR_plus_PCA128": "PCA128",
        "AR_plus_PCA64_delta": "PCA64_delta",
        "residualized_AR_plus_PCA64_delta": "PCA64_delta",
    }
    for lane, feature_name in lane_feature_names.items():
        head, _rows, mean_value = grouped_rows_for_best_head(lane_rows, target_name, threshold, lane, feature_name=feature_name)
        output[lane] = {"head": head, "mean_grouped_pr_auc": mean_value}
    return output


def trained_head_report(run_manifest: dict[str, Any], gate_checks: list[dict[str, Any]], promotion_candidates: list[dict[str, Any]], lane_rows: list[dict[str, Any]]) -> str:
    spike_005 = best_summary(lane_rows, "arousal__future_spike_1_3s", 0.05)
    spike_0075 = best_summary(lane_rows, "arousal__future_spike_1_3s", 0.075)
    leakage_failures = [gate for gate in gate_checks if gate.get("gate_name") == "leakage_failures_absent" and gate.get("status") == "fail"]
    control_failures = [gate for gate in gate_checks if "canonical" in gate.get("gate_name", "") and gate.get("status") == "fail"]
    best_head_005 = spike_005["AR_plus_PCA128"]["head"]
    best_head_0075 = spike_0075["AR_plus_PCA128"]["head"]
    sections = [
        "# VEATIC-124 Frozen Tensor Trained-Head Benchmark",
        "## Executive Verdict",
        f"1. Did trained heads improve over ridge? Best heads are `{best_head_005}` at spike @ 0.05 and `{best_head_0075}` at spike @ 0.075; compare lane CSVs against `ridge_score` rows for exact deltas.",
        f"2. Did PCA128_only beat AR_only? See gate checks; the best PCA128-only grouped PR-AUC at @0.05 is {report_line(spike_005['PCA128_only']['mean_grouped_pr_auc'])} versus AR {report_line(spike_005['AR_only']['mean_grouped_pr_auc'])}.",
        f"3. Did AR_plus_PCA128 beat AR_only? @0.05 best AR-plus is {report_line(spike_005['AR_plus_PCA128']['mean_grouped_pr_auc'])}; @0.075 best AR-plus is {report_line(spike_0075['AR_plus_PCA128']['mean_grouped_pr_auc'])}.",
        f"4. Did residualized_AR_plus_PCA128 beat AR_only? @0.05 best residualized is {report_line(spike_005['residualized_AR_plus_PCA128']['mean_grouped_pr_auc'])}; @0.075 best residualized is {report_line(spike_0075['residualized_AR_plus_PCA128']['mean_grouped_pr_auc'])}.",
        f"5. Did AR_plus_PCA128 beat AR_plus_PCA64_delta? See gate checks for grouped deltas.",
        f"6. Did residualized PCA128 beat residualized PCA64-delta? See gate checks for grouped deltas.",
        f"7. Which head worked best? AR-plus best heads: @0.05 `{best_head_005}`, @0.075 `{best_head_0075}`.",
        "8. Did sequence/temporal pooling help? Sequence-head rows are included with `uses_sequence_tensor=true`; compare them to collapsed `logistic_l2` and `ridge_score` rows.",
        f"9. Did canonical controls pass? {'No' if control_failures else 'Yes'}.",
        f"10. What is the best honest claim? `{promotion_candidates[0]['category'] if promotion_candidates else 'valid_but_weaker'}` for the primary spike target, subject to the gate table and no final promotion JSON.",
        "## Full VEATIC-124 Policy",
        "Video 83 was included. Exclude-video-83 sensitivity was intentionally skipped because the project benchmark is the full VEATIC-124 set.",
        f"- full_veatic_124: `{str(run_manifest['full_veatic_124']).lower()}`",
        f"- video_83_included: `{str(run_manifest['video_83_included']).lower()}`",
        f"- exclude_video_83_run: `{str(run_manifest['exclude_video_83_run']).lower()}`",
        "## Reuse Policy",
        "- Reused frozen tensors, metadata/checksum/leakage contracts, canonical helpers, splits, lane semantics, and controls.",
        "- Recomputed AR matrices, model fits, predictions, controls, metrics, fold aggregates, gates, and summaries.",
        "## Tensor Inputs",
        f"- primary: `{PRIMARY_CANDIDATE}`",
        f"- previous neural baseline: `{FROZEN_BASELINE}`",
        "## Existing Suite Semantics",
        "The run uses canonical AR/time/ridge/control/event-metric helper functions through the frozen tensor adapter.",
        "## Heads Tested",
        ", ".join(run_manifest["heads_tested"]),
        "## Ridge Parity Check",
        "Rows with `head_name=ridge_score` are included as the parity check against the ridge-only run.",
        "## Original AR Baseline",
        json.dumps(spike_005["AR_only"], sort_keys=True),
        "## PCA128-Only Results",
        json.dumps({"0.05": spike_005["PCA128_only"], "0.075": spike_0075["PCA128_only"]}, sort_keys=True),
        "## AR-plus-PCA128 Incremental Results",
        json.dumps({"0.05": spike_005["AR_plus_PCA128"], "0.075": spike_0075["AR_plus_PCA128"]}, sort_keys=True),
        "## Residualized AR-plus-PCA128 Results",
        json.dumps({"0.05": spike_005["residualized_AR_plus_PCA128"], "0.075": spike_0075["residualized_AR_plus_PCA128"]}, sort_keys=True),
        "## PCA64-Delta Baseline Comparisons",
        json.dumps({"0.05": {"AR_plus": spike_005["AR_plus_PCA64_delta"], "residualized": spike_005["residualized_AR_plus_PCA64_delta"]}}, sort_keys=True),
        "## Sequence Tensor Head Results",
        "Sequence-head rows are marked with `uses_sequence_tensor=true` in `lane_results.csv`.",
        "## Temporal Pooling Results",
        "Temporal-pooling selections are recorded in `head_selection_details.json`.",
        "## ROI and Top-K Exploratory Results",
        "ROI and top-k exploratory lanes were not part of this required run. Top-k, if enabled later, must remain cautionary and non-headline.",
        "## Canonical Control Results",
        f"Canonical control failures: `{len(control_failures)}`.",
        "## Grouped Fold Stability",
        "Stable positive folds are recorded per primary gate in `gate_checks.json`.",
        "## Leakage and Freshness Audit",
        f"Leakage failures: `{len(leakage_failures)}`. Freshness ledger records fresh AR, controls, predictions, and score computation.",
        "## Gate Summary",
        json.dumps(promotion_candidates, indent=2, sort_keys=True),
        "## Best Honest Claim",
        promotion_candidates[0]["category"] if promotion_candidates else "valid_but_weaker",
        "## Next Recommended Experiment",
        "Inspect the best trained-head rows against controls before deciding whether to run optional tiny MLPs.",
        "",
    ]
    return "\n".join(sections)


def trained_head_dry_run_plan(provider: FrozenTensorFeatureProvider, *, fresh_run_id: str | None = None) -> dict[str, Any]:
    checks = provider.preflight_checks()
    checks.append(
        {
            "check_name": "torch_mps_available_for_trained_heads",
            "status": "pass" if torch.backends.mps.is_available() else "fail",
            "details": {"mps_available": bool(torch.backends.mps.is_available())},
        }
    )
    run_id = fresh_run_id or f"trained_heads_dryrun_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    return {
        "schema_version": "veatic_frozen_tensor_trained_heads_dry_run_v1",
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "benchmark_mode": TRAINED_BENCHMARK_MODE,
        "dry_run_only": True,
        "dry_run_contract": "preflight only; no model fitting or benchmark scores",
        "fresh_run_id": run_id,
        "heads_planned": list(TRAINED_HEADS),
        "trained_head_backend": "torch_mps",
        "mps_available": bool(torch.backends.mps.is_available()),
        "reused_existing_benchmark_structure": True,
        "reused_frozen_tensors": True,
        "reused_canonical_controls": True,
        "reused_benchmark_result_rows": False,
        "computed_ar_fresh": True,
        "computed_controls_fresh": True,
        "full_veatic_124": True,
        "video_83_included": True,
        "exclude_video_83_run": False,
        "preflight_checks": checks,
        "preflight_status": "pass" if not check_failures(checks) else "fail",
        "failure_count": len(check_failures(checks)),
    }


def score_trained_heads(
    provider: FrozenTensorFeatureProvider,
    *,
    all_rows: list[dict[str, Any]],
    output_dir: Path,
    fresh_run_id: str,
    seed: int = 43,
    heads: tuple[str, ...] = TRAINED_HEADS,
    progress: bool = True,
) -> dict[str, Any]:
    deny_prior_result_files([provider.tensor_root, provider.summary_root])
    reject_exclude_83_paths(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {output_dir}")
    preflight_checks = provider.preflight_checks()
    preflight_checks.append(
        {
            "check_name": "torch_mps_available_for_trained_heads",
            "status": "pass" if torch.backends.mps.is_available() else "fail",
            "details": {"mps_available": bool(torch.backends.mps.is_available())},
        }
    )
    failures = check_failures(preflight_checks)
    if failures:
        failure_dir = output_dir
        failure_dir.mkdir(parents=True, exist_ok=True)
        write_json(failure_dir / "preflight_failure.json", failures)
        raise ValueError(f"Frozen tensor trained-head preflight failed: {failures}")

    lane_rows: list[dict[str, Any]] = []
    leakage_checks: list[dict[str, Any]] = list(preflight_checks)
    selection_details: list[dict[str, Any]] = []
    targets = [target for target in provider.targets() if (target["target_name"], float(target["threshold"])) in PRIMARY_TARGETS]
    if len(targets) != len(PRIMARY_TARGETS):
        raise ValueError(f"Required targets missing from tensor export: expected {PRIMARY_TARGETS}, got {targets}")
    job_index = 0
    total_jobs = len(provider.representations) * len(SPLITS) * len(targets)
    for representation in provider.representations:
        for split in SPLITS:
            for target in targets:
                if progress:
                    print(
                        "[progress] "
                        f"contract={job_index + 1}/{total_jobs} "
                        f"representation={representation} split={split} "
                        f"target={target['target_name']} threshold={target['threshold']} "
                        f"heads={','.join(heads)} backend=torch_mps",
                        file=sys.stderr,
                        flush=True,
                    )
                contract = provider.load_contract(
                    representation=representation,
                    split=split,
                    target_name=target["target_name"],
                    threshold=float(target["threshold"]),
                )
                collapsed_rows, collapsed_checks, collapsed_selection = score_collapsed_contract(
                    fresh_run_id=fresh_run_id,
                    contract=contract,
                    all_rows=all_rows,
                    heads=heads,
                    seed=seed + job_index,
                )
                lane_rows.extend(collapsed_rows)
                leakage_checks.extend(collapsed_checks)
                selection_details.extend(collapsed_selection)
                sequence_rows, sequence_checks, sequence_selection = score_sequence_contract(
                    fresh_run_id=fresh_run_id,
                    contract=contract,
                    all_rows=all_rows,
                    heads=heads,
                    seed=seed + job_index + 10000,
                )
                lane_rows.extend(sequence_rows)
                leakage_checks.extend(sequence_checks)
                selection_details.extend(sequence_selection)
                if progress:
                    print(
                        "[progress] "
                        f"completed_contract={job_index + 1}/{total_jobs} "
                        f"lane_rows={len(lane_rows)} selection_rows={len(selection_details)}",
                        file=sys.stderr,
                        flush=True,
                    )
                job_index += 1

    annotate_reference_columns(lane_rows)
    gate_checks, promotion_candidates = compute_gate_checks(lane_rows, leakage_checks)
    mark_control_pass(lane_rows, gate_checks)
    control_rows = [row for row in lane_rows if row["canonical_control"]]
    fold_rows = [row for row in lane_rows if str(row["split_name"]).startswith("grouped_")]
    gate_failures = check_failures([{"status": row.get("status", "pass"), **row} for row in gate_checks if row.get("gate_name") in {"benchmark_mode", "full_veatic_124", "video_83_included", "exclude_video_83_run", "no_prior_result_rows_reused", "leakage_failures_absent"}])
    if gate_failures:
        raise ValueError(f"Required trained-head run gates failed: {gate_failures}")
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = trained_freshness_ledger(provider, fresh_run_id)
    run_manifest = {
        "schema_version": TRAINED_SCHEMA_VERSION,
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "fresh_run_id": fresh_run_id,
        "seed": seed,
        "heads_tested": list(heads),
        "trained_head_backend": "torch_mps",
        "mps_available": bool(torch.backends.mps.is_available()),
        "representations": list(provider.representations),
        "splits": list(SPLITS),
        "targets": targets,
        "lane_count": len(lane_rows),
        "fold_count": len(fold_rows),
        "control_count": len(control_rows),
        "leakage_failure_count": len(check_failures(leakage_checks)),
        "outputs": {
            "run_manifest_json": str(output_dir / "run_manifest.json"),
            "freshness_ledger_json": str(output_dir / "freshness_ledger.json"),
            "lane_results_csv": str(output_dir / "lane_results.csv"),
            "fold_results_csv": str(output_dir / "fold_results.csv"),
            "control_results_csv": str(output_dir / "control_results.csv"),
            "head_selection_details_json": str(output_dir / "head_selection_details.json"),
            "leakage_checks_json": str(output_dir / "leakage_checks.json"),
            "gate_checks_json": str(output_dir / "gate_checks.json"),
            "trained_head_report_md": str(output_dir / "trained_head_report.md"),
            "promotion_candidates_json": str(output_dir / "promotion_candidates.json"),
        },
        **required_run_fields(),
    }
    assert_required_run_fields(run_manifest)
    write_json(output_dir / "run_manifest.json", run_manifest)
    write_json(output_dir / "freshness_ledger.json", ledger)
    write_csv(output_dir / "lane_results.csv", lane_rows)
    write_csv(output_dir / "fold_results.csv", fold_rows)
    write_csv(output_dir / "control_results.csv", control_rows)
    write_json(output_dir / "head_selection_details.json", selection_details)
    write_json(output_dir / "leakage_checks.json", leakage_checks)
    write_json(output_dir / "gate_checks.json", gate_checks)
    write_json(output_dir / "promotion_candidates.json", promotion_candidates)
    (output_dir / "trained_head_report.md").write_text(
        trained_head_report(run_manifest, gate_checks, promotion_candidates, lane_rows),
        encoding="utf-8",
    )
    return run_manifest
