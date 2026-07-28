"""VEATIC-derived Phase 02 AR benchmark preregistration without outer scoring."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np

from neural_bridge.veatic21.contracts import (
    EXPECTED_ROW_COUNT,
    EXPECTED_VIDEO_IDS,
    PHASE01_ARTIFACT_MANIFEST_SHA256,
    PHASE01_CHECKSUMS_SHA256,
    PHASE01_RESULT_SHA256,
    PHASE01_ROOT,
    PHASE02_REGISTRATION_ROOT,
    REPOSITORY_ROOT,
)
from neural_bridge.veatic21.data import load_json, reject_forbidden_runtime_path, sha256_file
from neural_bridge.veatic21.phase00 import (
    _audit_again_source_firewall,
    _source_tree_identity,
    _write_artifact_manifests,
    _write_json,
    _write_text,
    digest_json,
)
from neural_bridge.veatic21.phase01 import verify_phase01_output


def derive_protocol_counts(video_count: int, active_target_count: int) -> dict[str, int]:
    """Derive split/repeat/seed counts from VEATIC support, not inherited constants."""

    if video_count < 8 or active_target_count < 2:
        raise ValueError("insufficient support for nested Phase 02 protocols")
    minimum_test_videos = math.ceil(math.sqrt(video_count))
    grouped_outer_folds = video_count // minimum_test_videos
    grouped_repeats = math.ceil(math.log2(grouped_outer_folds))
    grouped_inner_folds = math.ceil(math.sqrt(grouped_outer_folds))
    blocked_time_blocks = grouped_inner_folds
    blocked_forward_folds = math.ceil(math.log2(blocked_time_blocks))
    finalist_seeds = math.ceil(math.log2(active_target_count))
    return {
        "minimum_test_videos": minimum_test_videos,
        "grouped_outer_folds": grouped_outer_folds,
        "grouped_repeats": grouped_repeats,
        "grouped_inner_folds": grouped_inner_folds,
        "blocked_time_blocks": blocked_time_blocks,
        "blocked_forward_folds": blocked_forward_folds,
        "finalist_seeds": finalist_seeds,
    }


def deterministic_seed(identity: str, *parts: object) -> int:
    payload = "\0".join((identity, *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") & 0x7FFF_FFFF


def balanced_group_folds(
    video_ids: Sequence[int],
    row_counts: Mapping[int, int],
    *,
    fold_count: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    """Create deterministic row-balanced folds without using target outcomes."""

    if fold_count < 2 or len(video_ids) < fold_count:
        raise ValueError("invalid grouped fold count")
    order = sorted(
        video_ids,
        key=lambda video_id: hashlib.sha256(f"{seed}\0{video_id}".encode()).digest(),
    )
    folds: list[list[int]] = [[] for _ in range(fold_count)]
    totals = [0] * fold_count
    base, remainder = divmod(len(order), fold_count)
    capacities = [base + int(index < remainder) for index in range(fold_count)]
    for video_id in order:
        eligible = [
            index for index, fold in enumerate(folds) if len(fold) < capacities[index]
        ]
        destination = min(eligible, key=lambda index: (totals[index], index))
        folds[destination].append(video_id)
        totals[destination] += row_counts[video_id]
    return tuple(tuple(sorted(fold)) for fold in folds)


def _grouped_split_registry(
    video_ids: Sequence[int],
    row_counts: Mapping[int, int],
    *,
    input_identity: str,
    outer_folds: int,
    repeats: int,
    inner_folds: int,
) -> list[dict[str, object]]:
    registry: list[dict[str, object]] = []
    all_videos = set(video_ids)
    for repeat in range(repeats):
        repeat_seed = deterministic_seed(input_identity, "grouped", repeat)
        folds = balanced_group_folds(
            video_ids, row_counts, fold_count=outer_folds, seed=repeat_seed
        )
        for outer_fold, test_videos in enumerate(folds):
            train_videos = tuple(sorted(all_videos - set(test_videos)))
            inner_seed = deterministic_seed(
                input_identity, "grouped", repeat, outer_fold, "inner"
            )
            inner = balanced_group_folds(
                train_videos, row_counts, fold_count=inner_folds, seed=inner_seed
            )
            registry.append(
                {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "partition_seed": repeat_seed,
                    "inner_partition_seed": inner_seed,
                    "train_videos": list(train_videos),
                    "test_videos": list(test_videos),
                    "train_rows_before_target_mask": sum(row_counts[v] for v in train_videos),
                    "test_rows_before_target_mask": sum(row_counts[v] for v in test_videos),
                    "inner_validation_video_folds": [list(fold) for fold in inner],
                }
            )
    return registry


def _blocked_fold_registry(
    *, block_count: int, blocked_folds: int, input_identity: str
) -> list[dict[str, object]]:
    first_test_block = block_count - blocked_folds
    return [
        {
            "outer_fold": fold,
            "test_block_index": test_block,
            "inner_validation_block_index": test_block - 1,
            "block_count": block_count,
            "row_assignment": (
                "per video on native row index; target window must end before its assigned "
                "boundary; boundary-crossing rows are purged and recorded"
            ),
            "partition_seed": deterministic_seed(input_identity, "blocked", fold),
        }
        for fold, test_block in enumerate(range(first_test_block, block_count))
    ]


def _grouped_support(
    registry: Sequence[Mapping[str, object]],
    video_id: np.ndarray,
    active_values: np.ndarray,
    active_masks: np.ndarray,
    candidate_ids: Sequence[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in registry:
        train_videos = np.asarray(split["train_videos"], dtype=np.int16)
        test_videos = np.asarray(split["test_videos"], dtype=np.int16)
        train_owner = np.isin(video_id, train_videos)
        test_owner = np.isin(video_id, test_videos)
        for target_index, candidate_id in enumerate(candidate_ids):
            train = train_owner & active_masks[target_index]
            test = test_owner & active_masks[target_index]
            threshold = float(np.quantile(active_values[target_index, train], 0.90))
            train_events = int(np.sum(active_values[target_index, train] >= threshold))
            rows.append(
                {
                    "repeat": split["repeat"],
                    "outer_fold": split["outer_fold"],
                    "candidate_id": candidate_id,
                    "train_rows": int(train.sum()),
                    "test_rows": int(test.sum()),
                    "train_q90_event_rows": train_events,
                    "train_non_event_rows": int(train.sum()) - train_events,
                    "outer_test_labels_opened": False,
                }
            )
    return rows


def _blocked_row_masks(
    video_id: np.ndarray,
    row_index: np.ndarray,
    row_counts: Mapping[int, int],
    target_end: int,
    test_block: int,
    block_count: int,
) -> dict[str, np.ndarray]:
    inner_train = np.zeros(video_id.shape, dtype=bool)
    inner_validation = np.zeros(video_id.shape, dtype=bool)
    outer_train = np.zeros(video_id.shape, dtype=bool)
    outer_test = np.zeros(video_id.shape, dtype=bool)
    purged = np.zeros(video_id.shape, dtype=bool)
    for video, count in row_counts.items():
        owned = video_id == video
        index = row_index[owned]
        inner_boundary = math.floor(count * (test_block - 1) / block_count)
        test_boundary = math.floor(count * test_block / block_count)
        test_end_boundary = math.floor(count * (test_block + 1) / block_count)
        inner_train[owned] = index + target_end < inner_boundary
        inner_validation[owned] = (index >= inner_boundary) & (
            index + target_end < test_boundary
        )
        outer_train[owned] = index + target_end < test_boundary
        outer_test[owned] = (index >= test_boundary) & (index < test_end_boundary)
        within_span = index < test_end_boundary
        assigned = inner_train[owned] | inner_validation[owned] | outer_test[owned]
        purged[owned] = within_span & ~assigned
    return {
        "inner_train": inner_train,
        "inner_validation": inner_validation,
        "outer_train": outer_train,
        "outer_test": outer_test,
        "purged": purged,
    }


def _blocked_support(
    registry: Sequence[Mapping[str, object]],
    video_id: np.ndarray,
    row_index: np.ndarray,
    row_counts: Mapping[int, int],
    active_values: np.ndarray,
    active_masks: np.ndarray,
    candidate_ids: Sequence[str],
    target_ends: Sequence[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in registry:
        for target_index, (candidate_id, target_end) in enumerate(
            zip(candidate_ids, target_ends, strict=True)
        ):
            masks = _blocked_row_masks(
                video_id,
                row_index,
                row_counts,
                target_end,
                cast(int, split["test_block_index"]),
                cast(int, split["block_count"]),
            )
            valid = active_masks[target_index]
            inner_train = masks["inner_train"] & valid
            inner_validation = masks["inner_validation"] & valid
            outer_train = masks["outer_train"] & valid
            outer_test = masks["outer_test"] & valid
            threshold = float(np.quantile(active_values[target_index, outer_train], 0.90))
            train_events = int(np.sum(active_values[target_index, outer_train] >= threshold))
            rows.append(
                {
                    "outer_fold": split["outer_fold"],
                    "candidate_id": candidate_id,
                    "target_end_rows": target_end,
                    "inner_train_rows": int(inner_train.sum()),
                    "inner_validation_rows": int(inner_validation.sum()),
                    "outer_train_rows": int(outer_train.sum()),
                    "outer_test_rows": int(outer_test.sum()),
                    "purged_boundary_rows": int((masks["purged"] & valid).sum()),
                    "outer_train_q90_event_rows": train_events,
                    "outer_train_non_event_rows": int(outer_train.sum()) - train_events,
                    "outer_test_labels_opened": False,
                }
            )
    return rows


def _search_registration(
    *, history_depths: Sequence[int], finalist_seeds: Sequence[int]
) -> dict[str, object]:
    feature_forms = [
        "current_only",
        "raw_levels_with_availability_mask",
        "level_and_first_difference",
        "causal_rolling_summary",
        "combined_levels_differences_summaries",
        "raw_sequence_with_availability_mask",
    ]
    return {
        "question": (
            "best defensible target-specific response-history floor for every active VEATIC "
            "no-washout spike candidate"
        ),
        "target_threshold": {
            "quantile": 0.90,
            "ownership": "refit inside every applicable training partition",
            "global_binary_target_stored": False,
        },
        "history_depth_rows": list(history_depths),
        "history_depth_seconds": [depth / 2 for depth in history_depths],
        "history_padding": (
            "causal missing lags are filled from the earliest available/current level with "
            "an explicit availability mask; target-valid rows are never silently discarded"
        ),
        "feature_forms": feature_forms,
        "analytic_controls": [
            "training_prevalence_constant",
            "current_arousal_rank",
            "previous_delta_rank",
            "causal_trailing_mean_rank",
            "causal_slope_rank",
            "time_and_video_time_only",
        ],
        "model_families": [
            "continuous_ridge",
            "event_logistic_l2",
            "event_elastic_net",
            "event_mlp",
            "event_gru",
        ],
        "linear_solvers": {
            "continuous_ridge": (
                "MLX float32 primal normal-equation solve with unpenalized intercept; "
                "training-owned target-valid rows only"
            ),
            "event_logistic_l2": (
                "MLX float32 full-batch accelerated gradient with a training-covariance "
                "Lipschitz step; no stochastic seed"
            ),
            "event_elastic_net": (
                "MLX float32 proximal accelerated gradient with the same Lipschitz rule"
            ),
            "convergence_tolerance": "1/sqrt(training_target_valid_rows)",
            "intercept_penalty": False,
            "metric_precision": "float64 after predictions are transferred for scoring",
        },
        "regularization": {
            "linear_scale": "training trace(X'X)/(n_features*n_rows)",
            "linear_multipliers": [
                1e-6,
                1e-5,
                1e-4,
                1e-3,
                1e-2,
                1e-1,
                1.0,
                10.0,
                100.0,
                1000.0,
            ],
            "elastic_l1_ratios": [0.25, 0.5, 0.75, 1.0],
            "boundary_rule": "expand one decade when an inner winner lies at either edge",
        },
        "nonlinear_capacity": {
            "width_rule": (
                "powers of two from half through four times next_power_of_two(sqrt(input_dim)), "
                "capped by next_power_of_two(sqrt(inner_train_rows))"
            ),
            "mlp_depths": [1, 2, 3],
            "activations": ["relu", "gelu", "tanh"],
            "dropout_rule": "0, 1/sqrt(input_dim), 2/sqrt(input_dim), each capped at 0.5",
            "gru_layers": [1, 2],
            "optimizers": ["adamw", "sgd_nesterov"],
            "optimizer_constants": {
                "adamw_beta1": "1 - 1/tail_bins",
                "adamw_beta2": "1 - 1/minimum_evaluation_rows",
                "adamw_epsilon": "numpy.float32 machine epsilon",
                "sgd_nesterov_momentum": "1 - 1/tail_bins",
            },
            "learning_rate_rule": "{0.25,1,4,16}/sqrt(inner_train_rows)",
            "batch_rule": (
                "half, one, and two times the largest power of two not exceeding "
                "sqrt(inner_train_rows)"
            ),
        },
        "staged_search": [
            {
                "stage": "A_complete_linear_screen",
                "scope": "all targets, protocols, outer folds, history depths and feature forms",
                "budget": "analytic plus complete ridge and logistic-L2 regularization grids",
                "pruning": "none",
            },
            {
                "stage": "B_stratified_family_expansion",
                "scope": (
                    "ceil(sqrt(number_of_feature_sets)) inner finalists per target/protocol/"
                    "outer fold, retaining at least one candidate per feature form and low/mid/"
                    "high history region"
                ),
                "budget": (
                    "elastic-net grid plus one-factor-at-a-time MLP/GRU width, depth, activation, "
                    "dropout, optimizer, learning-rate and batch screens"
                ),
            },
            {
                "stage": "C_joint_nonlinear_escalation",
                "scope": "ceil(sqrt(Stage-B finalists)) candidates per nonlinear family",
                "budget": "joint settings at four times the data-derived base update budget",
            },
            {
                "stage": "D_fresh_seed_finalists",
                "scope": "best inner candidate from every model family",
                "budget": (
                    "five fresh VEATIC-hash-derived seeds at sixteen times the data-derived base "
                    "update budget"
                ),
            },
        ],
        "update_budget_rule": {
            "base": "next_power_of_two(sqrt(inner_train_rows))",
            "stages": [1, 4, 16],
            "learning_curve_cadence": "base/16 updates, minimum one",
            "plateau_patience": "base/4 updates",
            "undertraining_rule": (
                "double the final budget before disposition when the inner curve has not plateaued"
            ),
        },
        "calibration_candidates": ["native_probability", "temperature", "platt"],
        "calibration_ownership": (
            "fit on inner-validation predictions only with the registered convex accelerated "
            "solver; outer labels never calibrate"
        ),
        "decision_threshold": (
            "maximize inner-validation F1; ties choose higher precision, then the higher "
            "threshold; used only for precision/recall/F1, never ranking"
        ),
        "uncertainty": {
            "cluster_unit": "whole video",
            "bootstrap_replicates_rule": (
                "next_power_of_two(minimum_evaluation_rows), yielding 1024"
            ),
            "confidence_interval": "two-sided percentile 95%",
            "seed_rule": "SHA-256 of registration identity, protocol, target, and comparison",
        },
        "selection": (
            "mean inner raw PR-AUC; one-standard-error set resolved by Brier, then smaller "
            "history/capacity; outer scores never select"
        ),
        "finalist_seeds": list(finalist_seeds),
        "label_permutation_control": (
            "same selected family/configuration with training and inner labels deterministically "
            "permuted; held-out labels remain true"
        ),
        "search_sufficiency": [
            "every registered family completed or excluded by its frozen inner-only rule",
            "edge winners receive the declared boundary expansion",
            "finalist learning curves plateau or receive doubled budget",
            "all five fresh seeds complete or are dispositioned as invalid",
            "fold/seed variability and remaining uncertainty are reported",
            "simple-history, time-only, prevalence and label-permutation controls are complete",
            "no negative or winner is declared while an applicable family remains undertrained",
        ],
    }


def run_phase02_registration(
    output_root: Path = PHASE02_REGISTRATION_ROOT,
) -> dict[str, object]:
    output_root = reject_forbidden_runtime_path(output_root)
    if output_root != PHASE02_REGISTRATION_ROOT:
        raise ValueError(f"Phase 02 registration must use the canonical root: {output_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to reuse Phase 02 registration: {output_root}")

    phase01 = verify_phase01_output(PHASE01_ROOT)
    if phase01["result_sha256"] != PHASE01_RESULT_SHA256:
        raise ValueError("Phase 01 result identity changed")
    if phase01["artifact_manifest_sha256"] != PHASE01_ARTIFACT_MANIFEST_SHA256:
        raise ValueError("Phase 01 artifact manifest identity changed")
    if phase01["checksums_sha256"] != PHASE01_CHECKSUMS_SHA256:
        raise ValueError("Phase 01 checksum identity changed")

    package_root = REPOSITORY_ROOT / "src/neural_bridge/veatic21"
    code_identity = _source_tree_identity(package_root)
    source_firewall = _audit_again_source_firewall(package_root)
    if source_firewall["again_imports"] or source_firewall["again_runtime_paths"]:
        raise ValueError("AGAIN source firewall failed")
    code_sha256 = str(code_identity["sha256"])
    input_identity = digest_json(
        {
            "phase01_result_sha256": PHASE01_RESULT_SHA256,
            "phase01_artifact_manifest_sha256": PHASE01_ARTIFACT_MANIFEST_SHA256,
            "phase01_checksums_sha256": PHASE01_CHECKSUMS_SHA256,
        }
    )
    _write_json(
        output_root / "request.json",
        {
            "schema_version": "veatic21_phase02_registration_request_v2",
            "phase": "phase-02-target-specific-ar-registration",
            "output_root": str(output_root),
            "phase01_root": str(PHASE01_ROOT),
            "input_identity_sha256": input_identity,
            "code_sha256": code_sha256,
            "outer_model_scores_opened": False,
            "cortical_values_opened": False,
        },
    )

    with np.load(PHASE01_ROOT / "aligned-labels.npz", allow_pickle=False) as payload:
        video_id = payload["video_id"].astype(np.int16)
        row_index = payload["row_index"].astype(np.int32)
    with np.load(PHASE01_ROOT / "target-substrate.npz", allow_pickle=False) as payload:
        starts = payload["candidate_start_rows"].astype(int)
        active_indices = np.flatnonzero(starts == 1)
        active_values = payload["continuous_future_maximum_increase"][active_indices]
        active_masks = payload["valid_mask"][active_indices]
    candidate_registry = load_json(PHASE01_ROOT / "candidate-registry.json")
    candidates = candidate_registry["candidates"]
    if not isinstance(candidates, list):
        raise ValueError("invalid Phase 01 candidate registry")
    active_candidates = [row for row in candidates if row.get("phase02_active")]
    candidate_ids = [str(row["candidate_id"]) for row in active_candidates]
    target_ends = [int(row["future_end_rows"]) for row in active_candidates]
    if len(candidate_ids) != 21 or target_ends != list(range(1, 22)):
        raise ValueError("Phase 02 active candidate family mismatch")

    numeric_video_ids = tuple(int(video) for video in EXPECTED_VIDEO_IDS)
    row_counts = {video: int(np.sum(video_id == video)) for video in numeric_video_ids}
    counts = derive_protocol_counts(len(numeric_video_ids), len(candidate_ids))
    grouped = _grouped_split_registry(
        numeric_video_ids,
        row_counts,
        input_identity=input_identity,
        outer_folds=counts["grouped_outer_folds"],
        repeats=counts["grouped_repeats"],
        inner_folds=counts["grouped_inner_folds"],
    )
    block_count = counts["blocked_time_blocks"]
    blocked = _blocked_fold_registry(
        block_count=block_count,
        blocked_folds=counts["blocked_forward_folds"],
        input_identity=input_identity,
    )
    grouped_support = _grouped_support(
        grouped, video_id, active_values, active_masks, candidate_ids
    )
    blocked_support = _blocked_support(
        blocked,
        video_id,
        row_index,
        row_counts,
        active_values,
        active_masks,
        candidate_ids,
        target_ends,
    )
    tail_bins = round(1 / (1 - 0.90))
    minimum_event_rows = tail_bins**2
    minimum_evaluation_rows = minimum_event_rows * tail_bins
    if min(cast(int, row["test_rows"]) for row in grouped_support) < minimum_evaluation_rows:
        raise ValueError("grouped test support is inadequate")
    if min(
        cast(int, row["outer_test_rows"]) for row in blocked_support
    ) < minimum_evaluation_rows:
        raise ValueError("blocked test support is inadequate")
    if min(
        cast(int, row["inner_validation_rows"]) for row in blocked_support
    ) < minimum_evaluation_rows:
        raise ValueError("blocked inner-validation support is inadequate")
    if any(row["outer_test_labels_opened"] for row in [*grouped_support, *blocked_support]):
        raise ValueError("registration opened outer labels")

    finalist_seeds = [
        deterministic_seed(input_identity, "finalist", index)
        for index in range(counts["finalist_seeds"])
    ]
    history_depths = list(range(1, max(target_ends) + 1))
    registration = {
        "schema_version": "veatic21_phase02_experiment_registration_v2",
        "frozen_before_outer_model_scoring": True,
        "input_identity_sha256": input_identity,
        "code_sha256": code_sha256,
        "targets": candidate_ids,
        "prospective_washout_candidates_active": False,
        "protocol_derivation": {
            **counts,
            "minimum_test_videos_rule": "ceil(sqrt(124))",
            "grouped_outer_folds_rule": "floor(video_count/minimum_test_videos)",
            "grouped_repeats_rule": "ceil(log2(grouped_outer_folds))",
            "grouped_inner_folds_rule": "ceil(sqrt(grouped_outer_folds))",
            "blocked_time_blocks_rule": "grouped_inner_folds",
            "blocked_forward_folds_rule": "ceil(log2(blocked_time_blocks))",
            "finalist_seed_count_rule": "ceil(log2(active_target_count))",
        },
        "grouped_protocol": {
            "outer_folds": counts["grouped_outer_folds"],
            "repeats": counts["grouped_repeats"],
            "inner_folds": counts["grouped_inner_folds"],
            "assignment": "hash-seeded, row-balanced by video; no target outcomes used",
        },
        "blocked_protocol": {
            "block_count": block_count,
            "outer_folds": counts["blocked_forward_folds"],
            "test_blocks": [cast(int, row["test_block_index"]) for row in blocked],
            "inner_validation": "immediately preceding native-time block",
            "target_boundary_purge": True,
        },
        "support_gate": {
            "tail_bins": tail_bins,
            "minimum_expected_event_rows": minimum_event_rows,
            "minimum_evaluation_rows": minimum_evaluation_rows,
            "derivation": (
                "q90 defines ten tail bins; require ten-squared expected event rows and thus "
                "ten-cubed total evaluation rows"
            ),
        },
        "search": _search_registration(
            history_depths=history_depths, finalist_seeds=finalist_seeds
        ),
        "outer_metrics": [
            "raw_pr_auc",
            "event_prevalence",
            "analytic_chance",
            "average_precision_skill",
            "roc_auc",
            "precision",
            "recall",
            "f1",
            "brier",
            "top_1pct_event_recall_and_lift",
            "top_5pct_event_recall_and_lift",
            "top_10pct_event_recall_and_lift",
            "defined_only_per_video_pr_auc",
            "undefined_video_count",
            "fold_seed_positive_counts_and_medians",
            "paired_whole_video_cluster_bootstrap",
        ],
        "operations": {
            "outer_model_scoring": False,
            "ar_fit": False,
            "cortical_read": False,
            "pca": False,
            "head_training": False,
            "washout_scoring": False,
        },
    }
    _write_json(output_root / "experiment-registration.json", registration)
    _write_json(
        output_root / "split-registry.json",
        {
            "schema_version": "veatic21_phase02_split_registry_v2",
            "input_identity_sha256": input_identity,
            "grouped": grouped,
            "blocked": blocked,
        },
    )
    _write_json(
        output_root / "support-audit.json",
        {
            "schema_version": "veatic21_phase02_support_audit_v2",
            "outer_test_labels_opened": False,
            "grouped": grouped_support,
            "blocked": blocked_support,
        },
    )
    ledger = {
        "schema_version": "veatic21_fresh_derivation_ledger_v2",
        "phase": "phase-02-target-specific-ar-registration",
        "code_sha256": code_sha256,
        "input_identity_sha256": input_identity,
        "entries": [
            {
                "name": key,
                "value": value,
                "evidence": "sealed VEATIC Phase 01 video/target support",
                "derivation_rule": registration["protocol_derivation"].get(f"{key}_rule"),
                "owned_rows": "Phase 01 identity and target-valid masks; no outer scores",
            }
            for key, value in counts.items()
        ],
    }
    _write_json(output_root / "veatic-derivation-ledger.json", ledger)
    result = {
        "schema_version": "veatic21_phase02_registration_result_v2",
        "status": "PASS",
        "registration_pass": True,
        "code_sha256": code_sha256,
        "input_identity_sha256": input_identity,
        "video_count": len(numeric_video_ids),
        "row_count": EXPECTED_ROW_COUNT,
        "active_target_count": len(candidate_ids),
        "prospective_washout_target_count": 210,
        "grouped_outer_cells": len(grouped),
        "blocked_outer_cells": len(blocked),
        "grouped_support_rows": len(grouped_support),
        "blocked_support_rows": len(blocked_support),
        "outer_model_scores_opened": False,
        "outer_test_labels_opened": False,
        "cortical_values_opened": False,
        "global_binary_target_stored": False,
        "operations": registration["operations"],
        "single_next_authorized_action": (
            "Execute the frozen comprehensive Phase 02 AR benchmark only"
        ),
    }
    _write_json(output_root / "result.json", result)
    _write_text(
        output_root / "report.md",
        f"""# VEATIC 2.1 Phase 02 AR experiment registration

Status: **PASS**

The registration freezes all {len(candidate_ids)} active no-washout targets before any outer
model score. It defines {counts['grouped_repeats']} independent {counts['grouped_outer_folds']}-fold
grouped-video partitions, {counts['grouped_inner_folds']} inner grouped folds per outer cell,
and {counts['blocked_forward_folds']} expanding blocked-forward folds over
{counts['blocked_time_blocks']} native-time blocks. Every count and seed is
derived from the sealed VEATIC video/target support.

The search covers all 21 causal history depths, six causal feature forms, analytic controls,
continuous ridge, L2 logistic, elastic-net logistic, MLP, and GRU families. Regularization,
capacity, optimizer, learning-rate, batch, update-budget, calibration, boundary expansion,
undertraining recovery, and five fresh finalist seeds are frozen by VEATIC-derived formulas.

No AR model, outer score, cortical value, PCA, learned bridge/head, or prospective washout
candidate was opened. Only execution of this exact registered Phase 02 AR benchmark is next.
""",
    )
    output_hashes = _write_artifact_manifests(
        output_root,
        (
            "request.json",
            "experiment-registration.json",
            "split-registry.json",
            "support-audit.json",
            "veatic-derivation-ledger.json",
            "result.json",
            "report.md",
        ),
        schema_version="veatic21_phase02_registration_artifact_manifest_v2",
    )
    return {
        **result,
        "output_hashes": {
            **output_hashes,
            "checksums.sha256": sha256_file(output_root / "checksums.sha256"),
        },
    }


def verify_phase02_registration(
    output_root: Path = PHASE02_REGISTRATION_ROOT,
) -> dict[str, object]:
    """Verify the frozen registration without opening any outer model result."""

    output_root = reject_forbidden_runtime_path(output_root)
    manifest = load_json(output_root / "artifact-manifest.json")
    if manifest.get("schema_version") != "veatic21_phase02_registration_artifact_manifest_v2":
        raise ValueError("Phase 02 registration manifest schema mismatch")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ValueError("Phase 02 registration manifest records missing")
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("invalid Phase 02 registration artifact record")
        path = output_root / record["path"]
        if record.get("bytes") != path.stat().st_size or record.get("sha256") != sha256_file(path):
            raise ValueError(f"Phase 02 registration artifact mismatch: {path}")
    checksum_path = output_root / "checksums.sha256"
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", maxsplit=1)
        if sha256_file(output_root / name) != expected:
            raise ValueError(f"Phase 02 registration checksum mismatch: {name}")

    result = load_json(output_root / "result.json")
    registration = load_json(output_root / "experiment-registration.json")
    splits = load_json(output_root / "split-registry.json")
    support = load_json(output_root / "support-audit.json")
    if not (
        result.get("registration_pass") is True
        and result.get("outer_model_scores_opened") is False
        and result.get("outer_test_labels_opened") is False
        and result.get("cortical_values_opened") is False
        and registration.get("frozen_before_outer_model_scoring") is True
        and registration.get("prospective_washout_candidates_active") is False
    ):
        raise ValueError("Phase 02 registration result is not promotable")
    operations = result.get("operations")
    if not isinstance(operations, dict) or any(operations.values()):
        raise ValueError("Phase 02 registration performed a forbidden operation")
    targets = registration.get("targets")
    if targets != [f"s01_e{end:02d}" for end in range(1, 22)]:
        raise ValueError("Phase 02 active target registry mismatch")
    grouped = splits.get("grouped")
    blocked = splits.get("blocked")
    if not isinstance(grouped, list) or len(grouped) != 40:
        raise ValueError("Phase 02 grouped split count mismatch")
    if not isinstance(blocked, list) or [row.get("test_block_index") for row in blocked] != [2, 3]:
        raise ValueError("Phase 02 blocked split count mismatch")
    for repeat in range(4):
        repeat_rows = [row for row in grouped if row.get("repeat") == repeat]
        test_videos = [video for row in repeat_rows for video in row["test_videos"]]
        if sorted(test_videos) != list(range(124)):
            raise ValueError(f"Phase 02 grouped coverage mismatch for repeat {repeat}")
    grouped_support = support.get("grouped")
    blocked_support = support.get("blocked")
    if not isinstance(grouped_support, list) or not isinstance(blocked_support, list):
        raise ValueError("Phase 02 support audit missing")
    if len(grouped_support) != 840 or len(blocked_support) != 42:
        raise ValueError("Phase 02 support audit cell count mismatch")
    if min(cast(int, row["test_rows"]) for row in grouped_support) < 1000:
        raise ValueError("Phase 02 grouped support gate failed")
    if min(cast(int, row["outer_test_rows"]) for row in blocked_support) < 1000:
        raise ValueError("Phase 02 blocked support gate failed")
    if min(cast(int, row["inner_validation_rows"]) for row in blocked_support) < 1000:
        raise ValueError("Phase 02 blocked inner support gate failed")
    if any(row.get("outer_test_labels_opened") for row in [*grouped_support, *blocked_support]):
        raise ValueError("Phase 02 registration opened outer labels")
    return {
        "verified": True,
        "artifact_manifest_sha256": sha256_file(output_root / "artifact-manifest.json"),
        "checksums_sha256": sha256_file(checksum_path),
        "result_sha256": sha256_file(output_root / "result.json"),
        "registration_sha256": sha256_file(output_root / "experiment-registration.json"),
        "result": result,
    }
