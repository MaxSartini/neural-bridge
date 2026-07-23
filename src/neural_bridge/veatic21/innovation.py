"""Control-complete causal-innovation redesign for VEATIC 2.1."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import FeatureRows
from .controls import (
    _lane_summary,
    _quarantine_partial,
    _replay_frozen_ar,
    _require_self_digest,
    _run_control_cell,
)
from .data import CanonicalSubstrate
from .evidence import (
    atomic_write_json,
    digest_json,
    paired_video_bootstrap_raw_pr_auc_delta,
    sha256_file,
    source_tree_digest,
)
from .pca_cache import load_event_pca_projection
from .preregistration import benchmark_partition_mask
from .protocol import (
    causal_ar_features,
    event_labels,
    fit_event_threshold,
    future_target_values,
    target_support_mask,
)
from .runner import (
    _causal_video_means,
    _circular_permute_labels,
    _lane_rng,
    _permute_within_video,
)
from .stage1 import _causal_design, _owned_rows, _target

_SCHEMA = "veatic21_innovation_control_plan_v1"
_REAL = "real_prior_centered_innovation_residual"
_MATCHED = (
    "sequence_shuffled_residual",
    "random_pca_residual",
    "causal_prefix_video_mean_residual",
    "diagnostics_only_residual",
    "label_permutation_residual",
)
_CURRENT = "current_innovation_only_residual"
_LANES = (_REAL, *_MATCHED, _CURRENT)


def _target_control_ranking(
    baseline_summary: Mapping[str, Any],
    control_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    real = {
        (str(row["target"]), int(row["fold"]), int(row["seed"])): float(
            row["inner_average_precision_skill_delta_vs_frozen_ar"]
        )
        for row in baseline_summary["records"]
        if row.get("lane") == "fixed_pca512"
    }
    controls: dict[str, dict[tuple[str, int, int], float]] = defaultdict(dict)
    for row in control_summary["records"]:
        controls[str(row["lane"])][
            (str(row["target"]), int(row["fold"]), int(row["seed"]))
        ] = float(row["inner_average_precision_skill_delta_vs_frozen_ar"])
    matched = tuple(control_summary["lane_summaries"])
    matched = tuple(lane for lane in matched if lane != "current_row_only_residual")
    ranking = []
    for target in sorted({key[0] for key in real}):
        keys = [key for key in real if key[0] == target]
        real_mean = float(np.mean([real[key] for key in keys]))
        lane_means = {
            lane: float(np.mean([controls[lane][key] for key in keys])) for lane in matched
        }
        strongest = max(lane_means, key=lambda lane: lane_means[lane])
        ranking.append(
            {
                "target": target,
                "real_mean_delta_vs_ar": real_mean,
                "strongest_matched_control": strongest,
                "strongest_control_mean_delta_vs_ar": lane_means[strongest],
                "real_minus_strongest_control": real_mean - lane_means[strongest],
            }
        )
    ranking.sort(key=lambda row: (-float(row["real_minus_strongest_control"]), row["target"]))
    return ranking


def build_innovation_control_plan(
    preregistration: Mapping[str, Any],
    recipe_plan: Mapping[str, Any],
    recipe_selection: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    failed_control_summary: Mapping[str, Any],
    pca_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Register one compact redesign with controls present from its first cell."""

    _require_self_digest(preregistration, "preregistration_sha256")
    _require_self_digest(recipe_plan, "plan_sha256")
    _require_self_digest(recipe_selection, "resolution_sha256")
    _require_self_digest(baseline_summary, "summary_sha256")
    _require_self_digest(failed_control_summary, "summary_sha256")
    _require_self_digest(pca_manifest, "manifest_sha256")
    if failed_control_summary.get("all_gates_pass") is not False:
        raise ValueError("innovation redesign requires the recorded failed control verdict")
    if failed_control_summary.get("completed_new_cells") != failed_control_summary.get(
        "expected_new_cells"
    ):
        raise ValueError("innovation redesign requires a complete control matrix")
    if recipe_selection.get("plan_sha256") != recipe_plan.get("plan_sha256"):
        raise ValueError("innovation artifacts do not share the recipe plan")
    if pca_manifest.get("manifest_sha256") != recipe_plan.get("artifacts", {}).get(
        "pca_manifest_sha256"
    ):
        raise ValueError("innovation artifacts do not share the PCA manifest")

    ranking = _target_control_ranking(baseline_summary, failed_control_summary)
    targets = [str(row["target"]) for row in ranking[:2]]
    folds = [int(value) for value in recipe_plan["matrix"]["folds"]]
    seeds = [int(value) for value in recipe_plan["matrix"]["comparison_seeds"]]
    cells_per_lane = len(targets) * len(folds) * len(seeds)
    plan: dict[str, Any] = {
        "schema": _SCHEMA,
        "purpose": "control_complete_prior_centered_causal_innovation_redesign",
        "artifacts": {
            "baseline_summary_sha256": baseline_summary["summary_sha256"],
            "control_runtime_code_sha256": sha256_file(Path(__file__).with_name("controls.py")),
            "failed_control_summary_sha256": failed_control_summary["summary_sha256"],
            "innovation_code_sha256": sha256_file(Path(__file__)),
            "pca_manifest_sha256": pca_manifest["manifest_sha256"],
            "preregistration_sha256": preregistration["preregistration_sha256"],
            "recipe_selection_sha256": recipe_selection["resolution_sha256"],
            "veatic21_source_tree_sha256": source_tree_digest(Path(__file__).parent),
        },
        "target_selection": {
            "scope": "failed_control_comparison_evidence_only",
            "rule": "top_two_real_minus_strongest_matched_control",
            "ranking": ranking,
            "selected_targets": targets,
            "selection_is_development_only": True,
        },
        "architecture": {
            "source": "fixed_fold_owned_pca512",
            "centering": "subtract_strictly_prior_causal_prefix_mean_with_cold_start_indicator",
            "temporal_design": "centered_current_plus_five_centered_current_minus_past_blocks",
            "static_video_level_removed": True,
            "future_features_forbidden": True,
            "video_boundary_crossing_forbidden": True,
        },
        "selected_recipe": dict(recipe_selection["selected_recipe"]),
        "matrix": {
            "targets": targets,
            "folds": folds,
            "comparison_seeds": seeds,
            "real_lane": _REAL,
            "matched_control_lanes": list(_MATCHED),
            "ablation_lane": _CURRENT,
            "trained_lanes": list(_LANES),
            "cells_per_lane": cells_per_lane,
            "expected_cells": cells_per_lane * len(_LANES),
            "worker_count": 1,
            "backend": "mlx",
            "sealed_tail_labels": True,
        },
        "gates": {
            "real_mean_raw_pr_auc_minus_ar_positive": True,
            "real_mean_raw_pr_auc_minus_strongest_control_positive": True,
            "paired_median_real_minus_strongest_control_positive": True,
            "every_target_mean_real_minus_strongest_control_positive": True,
            "at_least_four_of_five_fold_means_positive": True,
            "real_mean_minus_registered_ablation_positive": True,
            "label_permutation_mean_raw_pr_auc_delta_vs_ar_nonpositive": True,
            "bootstrap_lower_ci_vs_ar_positive": True,
            "bootstrap_lower_ci_vs_strongest_control_positive": True,
            "exact_matrix_and_artifact_audits_pass": True,
        },
        "metrics": {
            "primary": "raw_pr_auc",
            "cross_prevalence_companion": "average_precision_skill",
            "reported": [
                "analytic_chance_pr_auc",
                "top_1_5_10_percent_event_recall",
                "brier_score",
                "per_video_pr_auc_defined_only",
            ],
        },
        "uncertainty": {
            "primary_metric": "raw_pr_auc",
            "cluster": "target_fold_seed_video",
            "confidence_interval": 0.95,
            "resamples": 2_048,
            "seed": 20_260_723,
        },
        "stability_authorized_only_after_all_gates_pass": True,
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    plan["plan_sha256"] = digest_json(plan)
    return plan


def write_innovation_control_plan(path: Path, plan: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(plan))


def build_current_innovation_control_plan(
    preregistration: Mapping[str, Any],
    prior_plan: Mapping[str, Any],
    prior_summary: Mapping[str, Any],
    recipe_selection: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    pca_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Promote the diagnostic current-innovation lane into a control-complete candidate."""

    _require_self_digest(preregistration, "preregistration_sha256")
    _require_self_digest(prior_plan, "plan_sha256")
    _require_self_digest(prior_summary, "summary_sha256")
    _require_self_digest(recipe_selection, "resolution_sha256")
    _require_self_digest(baseline_summary, "summary_sha256")
    _require_self_digest(pca_manifest, "manifest_sha256")
    if prior_summary.get("plan_sha256") != prior_plan.get("plan_sha256"):
        raise ValueError("current-innovation redesign does not bind the prior redesign")
    if prior_summary.get("all_gates_pass") is not False:
        raise ValueError("current-innovation redesign requires the recorded failed verdict")
    eligible = [
        str(target)
        for target, passed in prior_summary["target_gate_results"].items()
        if passed is True
    ]
    if eligible != ["arousal_positive_max_0p5_1s_train_q900"]:
        raise ValueError("current-innovation target selection changed")
    recipe_plan_path_sha = prior_plan.get("artifacts", {}).get("pca_manifest_sha256")
    if recipe_plan_path_sha != pca_manifest.get("manifest_sha256"):
        raise ValueError("current-innovation artifacts do not share the PCA manifest")

    real_lane = "real_current_innovation_residual"
    ablation_lane = "uncentered_current_row_residual"
    lanes = (real_lane, *_MATCHED, ablation_lane)
    folds = [int(value) for value in prior_plan["matrix"]["folds"]]
    seeds = [int(value) for value in prior_plan["matrix"]["comparison_seeds"]]
    cells_per_lane = len(folds) * len(seeds)
    plan: dict[str, Any] = {
        "schema": _SCHEMA,
        "purpose": "control_complete_current_innovation_redesign",
        "artifacts": {
            "baseline_summary_sha256": baseline_summary["summary_sha256"],
            "control_runtime_code_sha256": sha256_file(Path(__file__).with_name("controls.py")),
            "innovation_code_sha256": sha256_file(Path(__file__)),
            "pca_manifest_sha256": pca_manifest["manifest_sha256"],
            "preregistration_sha256": preregistration["preregistration_sha256"],
            "prior_innovation_plan_sha256": prior_plan["plan_sha256"],
            "prior_innovation_summary_sha256": prior_summary["summary_sha256"],
            "recipe_selection_sha256": recipe_selection["resolution_sha256"],
            "veatic21_source_tree_sha256": source_tree_digest(Path(__file__).parent),
        },
        "target_selection": {
            "scope": "prior_control_complete_innovation_matrix_only",
            "rule": "only_target_with_positive_prior_target_gate",
            "selected_targets": eligible,
            "selection_is_development_only": True,
        },
        "architecture": {
            "source": "fixed_fold_owned_pca512",
            "centering": "subtract_strictly_prior_causal_prefix_mean_with_cold_start_indicator",
            "temporal_design": "current_innovation_only",
            "current_innovation_only": True,
            "static_video_level_removed": True,
            "future_features_forbidden": True,
            "video_boundary_crossing_forbidden": True,
        },
        "selected_recipe": dict(recipe_selection["selected_recipe"]),
        "matrix": {
            "targets": eligible,
            "folds": folds,
            "comparison_seeds": seeds,
            "real_lane": real_lane,
            "matched_control_lanes": list(_MATCHED),
            "ablation_lane": ablation_lane,
            "trained_lanes": list(lanes),
            "cells_per_lane": cells_per_lane,
            "expected_cells": cells_per_lane * len(lanes),
            "worker_count": 1,
            "backend": "mlx",
            "sealed_tail_labels": True,
        },
        "gates": {
            "real_mean_raw_pr_auc_minus_ar_positive": True,
            "real_mean_raw_pr_auc_minus_strongest_control_positive": True,
            "paired_median_real_minus_strongest_control_positive": True,
            "every_target_mean_real_minus_strongest_control_positive": True,
            "at_least_four_of_five_fold_means_positive": True,
            "real_mean_minus_registered_ablation_positive": True,
            "label_permutation_mean_raw_pr_auc_delta_vs_ar_nonpositive": True,
            "bootstrap_lower_ci_vs_ar_positive": True,
            "bootstrap_lower_ci_vs_strongest_control_positive": True,
            "exact_matrix_and_artifact_audits_pass": True,
        },
        "metrics": {
            "primary": "raw_pr_auc",
            "cross_prevalence_companion": "average_precision_skill",
            "reported": [
                "analytic_chance_pr_auc",
                "top_1_5_10_percent_event_recall",
                "brier_score",
                "per_video_pr_auc_defined_only",
            ],
        },
        "uncertainty": {
            "primary_metric": "raw_pr_auc",
            "cluster": "target_fold_seed_video",
            "confidence_interval": 0.95,
            "resamples": 2_048,
            "seed": 20_260_723,
        },
        "stability_authorized_only_after_all_gates_pass": True,
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    plan["plan_sha256"] = digest_json(plan)
    return plan


def _strict_prior_center(
    values: np.ndarray,
    video_id: np.ndarray,
    row_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    videos = video_id.astype(str)
    centered = np.empty_like(values)
    available = np.zeros((len(values), 1), dtype=np.float32)
    for video in sorted(set(videos), key=int):
        positions = np.flatnonzero(videos == video)
        ordered = positions[np.argsort(row_index[positions], kind="stable")]
        cumulative = np.zeros(values.shape[1], dtype=np.float64)
        for count, position in enumerate(ordered):
            if count == 0:
                centered[position] = values[position]
            else:
                centered[position] = values[position] - cumulative / count
                available[position, 0] = 1.0
            cumulative += values[position]
    return centered, available


def _innovation_design(
    values: np.ndarray,
    video_id: np.ndarray,
    row_index: np.ndarray,
    context_rows: tuple[int, ...],
    *,
    current_only: bool,
) -> np.ndarray:
    centered, prefix_available = _strict_prior_center(values, video_id, row_index)
    if current_only:
        return np.concatenate([centered, prefix_available], axis=1)
    temporal = _causal_design(
        centered,
        video_id,
        row_index,
        family="frozen_ar_plus_causal_temporal_residual",
        context_rows=context_rows,
    )
    return np.concatenate([temporal, prefix_available], axis=1)


def _design_for_lane(
    lane: str,
    projected: np.ndarray,
    diagnostics: np.ndarray,
    binary: np.ndarray,
    video_id: np.ndarray,
    row_index: np.ndarray,
    context_rows: tuple[int, ...],
    seed: int,
    *,
    real_lane: str = _REAL,
    ablation_lane: str = _CURRENT,
    current_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    targets = binary
    if lane == real_lane:
        values = projected
    elif lane == "sequence_shuffled_residual":
        values = _permute_within_video(projected, video_id, _lane_rng(seed, lane))
    elif lane == "random_pca_residual":
        values = _lane_rng(seed, lane).standard_normal(projected.shape).astype(np.float32)
    elif lane == "causal_prefix_video_mean_residual":
        values = _causal_video_means(projected, video_id, row_index).astype(np.float32)
    elif lane == "diagnostics_only_residual":
        values = diagnostics
    elif lane == "label_permutation_residual":
        values = projected
        targets = _circular_permute_labels(
            binary,
            video_id,
            row_index,
            _lane_rng(seed, lane),
        )
    elif lane == ablation_lane:
        if current_only:
            return projected, targets
        return (
            _innovation_design(
                projected,
                video_id,
                row_index,
                context_rows,
                current_only=True,
            ),
            targets,
        )
    else:
        raise ValueError(f"unknown innovation lane: {lane}")
    return (
        _innovation_design(
            values,
            video_id,
            row_index,
            context_rows,
            current_only=current_only,
        ),
        targets,
    )


def run_innovation_control_program(
    substrate: CanonicalSubstrate,
    preregistration: Mapping[str, Any],
    calibration: Mapping[str, Any],
    pca_manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    pca_root: Path,
    baseline_root: Path,
    output_dir: Path,
    *,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the compact redesign and all matched controls in one matrix."""

    _require_self_digest(plan, "plan_sha256")
    _require_self_digest(baseline_summary, "summary_sha256")
    if plan.get("schema") != _SCHEMA:
        raise ValueError("innovation runner requires the current plan schema")
    if plan.get("artifacts", {}).get("innovation_code_sha256") != sha256_file(Path(__file__)):
        raise ValueError("innovation plan does not bind the current runner code")
    if plan.get("artifacts", {}).get("control_runtime_code_sha256") != sha256_file(
        Path(__file__).with_name("controls.py")
    ):
        raise ValueError("innovation plan does not bind the shared control runtime")
    if plan.get("artifacts", {}).get("veatic21_source_tree_sha256") != source_tree_digest(
        Path(__file__).parent
    ):
        raise ValueError("innovation plan does not bind the complete VEATIC runtime")
    request = {
        "schema": "veatic21_innovation_control_run_request_v1",
        "plan_sha256": plan["plan_sha256"],
        "worker_count": 1,
        "backend": "mlx",
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    request_sha256 = digest_json(request)
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "request.json"
    if request_path.is_file():
        if digest_json(json.loads(request_path.read_text(encoding="utf-8"))) != request_sha256:
            raise RuntimeError("refusing innovation resume because request changed")
    else:
        atomic_write_json(request_path, request)

    matrix = plan["matrix"]
    expected_keys = {
        (str(target), int(fold), int(seed))
        for target in matrix["targets"]
        for fold in matrix["folds"]
        for seed in matrix["comparison_seeds"]
    }
    baseline = {
        (str(row["target"]), int(row["fold"]), int(row["seed"])): row
        for row in baseline_summary["records"]
        if row.get("lane") == "fixed_pca512"
        and str(row["target"]) in set(matrix["targets"])
    }
    if set(baseline) != expected_keys:
        raise ValueError("innovation baseline does not cover its exact target panel")

    all_features = substrate.load_features(
        substrate.video_ids,
        ("tribe_cortical", "diagnostics_only"),
    )
    development_mask = benchmark_partition_mask(all_features, preregistration["split"], "train")
    features = all_features.subset(development_mask)
    pca_features = FeatureRows(
        video_id=features.video_id,
        row_index=features.row_index,
        time_seconds=features.time_seconds,
        quality_eligible=features.quality_eligible,
        representations={"tribe_cortical": features.representations["tribe_cortical"]},
    )
    labels = substrate.load_labels(
        substrate.video_ids,
        row_indices=_owned_rows(all_features.video_id, all_features.row_index, development_mask),
        stage="innovation_controls_benchmark_train_labels_only",
    )
    projected_by_fold = {
        int(fold): load_event_pca_projection(
            pca_features,
            preregistration,
            pca_manifest,
            pca_root,
            fold=int(fold),
            width=512,
        )
        for fold in matrix["folds"]
    }
    diagnostics = features.representations["diagnostics_only"]
    context_rows = tuple(int(value) for value in plan["selected_recipe"]["context_rows"])
    current_only = bool(plan["architecture"].get("current_innovation_only", False))
    records: list[dict[str, Any]] = []
    for target_name in matrix["targets"]:
        target = _target(calibration, str(target_name))
        future = future_target_values(labels, target)
        support = target_support_mask(features, target)
        ar_values, ar_available = causal_ar_features(labels, target)
        ar_matrix = np.concatenate([ar_values, ar_available.astype(np.float64)], axis=1)
        for fold in matrix["folds"]:
            validation_videos = preregistration["split"]["inner_grouped_video_folds"][int(fold)]
            validation_mask = np.isin(features.video_id.astype(str), validation_videos) & support
            train_mask = ~np.isin(features.video_id.astype(str), validation_videos) & support
            threshold = fit_event_threshold(future, train_mask, target)
            binary = event_labels(future, threshold)
            projected = projected_by_fold[int(fold)]
            for seed in matrix["comparison_seeds"]:
                key = (str(target_name), int(fold), int(seed))
                source_record = baseline[key]
                source_dir = baseline_root / str(source_record["cell_directory"])
                metrics_path = source_dir / "metrics.json"
                if sha256_file(metrics_path) != source_record["cell_metrics_sha256"]:
                    raise ValueError("innovation source real metrics changed")
                source_artifacts = {
                    name: sha256_file(source_dir / name)
                    for name in (
                        "best-checkpoint.npz",
                        "preprocessing.npz",
                        "training-curve.json",
                        "validation-predictions.npz",
                    )
                }
                ar_logits = _replay_frozen_ar(
                    ar_matrix,
                    source_dir,
                    validation_mask,
                    binary,
                    features.video_id,
                    features.row_index,
                )
                for lane in matrix["trained_lanes"]:
                    design, lane_targets = _design_for_lane(
                        str(lane),
                        projected,
                        diagnostics,
                        binary,
                        features.video_id,
                        features.row_index,
                        context_rows,
                        int(seed),
                        real_lane=str(matrix["real_lane"]),
                        ablation_lane=str(matrix["ablation_lane"]),
                        current_only=current_only,
                    )
                    cell_dir = (
                        output_dir
                        / "targets"
                        / str(target_name)
                        / f"fold-{int(fold)}"
                        / f"seed-{int(seed)}"
                        / str(lane)
                    )
                    _quarantine_partial(cell_dir, output_dir, plan["plan_sha256"])
                    metrics = _run_control_cell(
                        projected=projected,
                        diagnostics=diagnostics,
                        binary=binary,
                        ar_logits=ar_logits,
                        train_mask=train_mask,
                        validation_mask=validation_mask,
                        video_id=features.video_id,
                        row_index=features.row_index,
                        plan=plan,
                        output_dir=cell_dir,
                        lane=str(lane),
                        target=str(target_name),
                        fold=int(fold),
                        seed=int(seed),
                        source_metrics_sha256=str(source_record["cell_metrics_sha256"]),
                        source_artifact_sha256=source_artifacts,
                        design_override=design,
                        lane_targets_override=lane_targets,
                    )
                    record = {
                        "lane": str(lane),
                        "target": str(target_name),
                        "fold": int(fold),
                        "seed": int(seed),
                        "inner_average_precision_skill_delta_vs_frozen_ar": metrics[
                            "inner_average_precision_skill_delta_vs_frozen_ar"
                        ],
                        "event_prevalence": metrics["event_prevalence"],
                        "fresh_ar_pr_auc": metrics["fresh_ar_pr_auc"],
                        "lane_pr_auc": metrics["control_pr_auc"],
                        "fresh_ar_brier_score": metrics["fresh_ar_brier_score"],
                        "lane_brier_score": metrics["control_brier_score"],
                        "fresh_ar_top_event_recall": metrics["fresh_ar_top_event_recall"],
                        "lane_top_event_recall": metrics["control_top_event_recall"],
                        "fresh_ar_defined_per_video_mean_pr_auc": metrics[
                            "fresh_ar_defined_per_video_mean_pr_auc"
                        ],
                        "lane_defined_per_video_mean_pr_auc": metrics[
                            "control_defined_per_video_mean_pr_auc"
                        ],
                        "best_epoch": metrics["best_epoch"],
                        "cell_metrics_sha256": sha256_file(cell_dir / "metrics.json"),
                        "cell_directory": str(cell_dir.relative_to(output_dir)),
                    }
                    records.append(record)
                    if progress is not None:
                        progress(
                            {
                                "schema": "veatic21_innovation_progress_v1",
                                "completed_cells": len(records),
                                "expected_cells": int(matrix["expected_cells"]),
                                "last_cell": record,
                                "worker_count": 1,
                                "benchmark_test_labels_accessed": False,
                            }
                        )
    return _summarize_innovation(plan, request_sha256, records, output_dir)


def _paired_prediction_panel(
    output_dir: Path,
    records: list[dict[str, Any]],
    real_lane: str,
    reference_lane: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indexed = {
        (str(row["lane"]), str(row["target"]), int(row["fold"]), int(row["seed"])): row
        for row in records
    }
    keys = sorted(
        (str(row["target"]), int(row["fold"]), int(row["seed"]))
        for row in records
        if row["lane"] == real_lane
    )
    clusters = []
    targets = []
    primary = []
    reference = []
    for target, fold, seed in keys:
        real_row = indexed[(real_lane, target, fold, seed)]
        with np.load(
            output_dir / str(real_row["cell_directory"]) / "validation-predictions.npz",
            allow_pickle=False,
        ) as arrays:
            video = arrays["video_id"].astype(str)
            row_index = np.asarray(arrays["row_index"])
            truth = np.asarray(arrays["target"], dtype=np.int8)
            real_scores = np.asarray(arrays["control_score"], dtype=np.float64)
            ar_scores = np.asarray(arrays["ar_score"], dtype=np.float64)
        if reference_lane is None:
            reference_scores = ar_scores
        else:
            reference_row = indexed[(reference_lane, target, fold, seed)]
            with np.load(
                output_dir
                / str(reference_row["cell_directory"])
                / "validation-predictions.npz",
                allow_pickle=False,
            ) as arrays:
                if not (
                    np.array_equal(arrays["video_id"].astype(str), video)
                    and np.array_equal(arrays["row_index"], row_index)
                    and np.array_equal(arrays["target"], truth)
                ):
                    raise ValueError("innovation control predictions are not exactly paired")
                reference_scores = np.asarray(arrays["control_score"], dtype=np.float64)
        clusters.append(
            np.asarray(
                [f"{target}|{fold}|{seed}|{value}" for value in video],
                dtype=str,
            )
        )
        targets.append(truth)
        primary.append(real_scores)
        reference.append(reference_scores)
    return (
        np.concatenate(clusters),
        np.concatenate(targets),
        np.concatenate(primary),
        np.concatenate(reference),
    )


def _summarize_innovation(
    plan: Mapping[str, Any],
    request_sha256: str,
    records: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    matrix = plan["matrix"]
    if len(records) != int(matrix["expected_cells"]):
        raise RuntimeError("innovation program did not complete its exact matrix")
    by_lane = {
        str(lane): [row for row in records if row["lane"] == lane]
        for lane in matrix["trained_lanes"]
    }
    lane_summaries = {}
    for lane, rows in by_lane.items():
        lane_summary = _lane_summary(rows)
        lane_summary.update(
            {
                "mean_raw_pr_auc": float(np.mean([row["lane_pr_auc"] for row in rows])),
                "mean_matched_fresh_ar_pr_auc": float(
                    np.mean([row["fresh_ar_pr_auc"] for row in rows])
                ),
                "mean_raw_pr_auc_delta_vs_frozen_ar": float(
                    np.mean(
                        [row["lane_pr_auc"] - row["fresh_ar_pr_auc"] for row in rows]
                    )
                ),
                "mean_brier_score": float(np.mean([row["lane_brier_score"] for row in rows])),
                "mean_defined_per_video_pr_auc": float(
                    np.mean([row["lane_defined_per_video_mean_pr_auc"] for row in rows])
                ),
                "mean_top_event_recall": {
                    key: float(
                        np.mean([row["lane_top_event_recall"][key] for row in rows])
                    )
                    for key in ("top_1_percent", "top_5_percent", "top_10_percent")
                },
            }
        )
        lane_summaries[lane] = lane_summary
    deltas = {
        lane: {
            (str(row["target"]), int(row["fold"]), int(row["seed"])): float(
                row["inner_average_precision_skill_delta_vs_frozen_ar"]
            )
            for row in rows
        }
        for lane, rows in by_lane.items()
    }
    real_lane = str(matrix["real_lane"])
    ablation_lane = str(matrix["ablation_lane"])
    real = deltas[real_lane]
    strongest = max(
        matrix["matched_control_lanes"],
        key=lambda lane: lane_summaries[str(lane)]["mean_raw_pr_auc"],
    )
    raw_pr_auc = {
        lane: {
            (str(row["target"]), int(row["fold"]), int(row["seed"])): float(
                row["lane_pr_auc"]
            )
            for row in rows
        }
        for lane, rows in by_lane.items()
    }
    fresh_ar_pr_auc = {
        (str(row["target"]), int(row["fold"]), int(row["seed"])): float(
            row["fresh_ar_pr_auc"]
        )
        for row in by_lane[real_lane]
    }
    paired = np.asarray(
        [raw_pr_auc[real_lane][key] - raw_pr_auc[str(strongest)][key] for key in sorted(real)]
    )
    target_results = {}
    for target in matrix["targets"]:
        keys = [key for key in sorted(real) if key[0] == target]
        real_mean = float(np.mean([raw_pr_auc[real_lane][key] for key in keys]))
        best = max(
            float(np.mean([raw_pr_auc[str(lane)][key] for key in keys]))
            for lane in matrix["matched_control_lanes"]
        )
        target_results[str(target)] = real_mean - best > 0.0
    fold_results = {}
    for fold in matrix["folds"]:
        keys = [key for key in sorted(real) if key[1] == int(fold)]
        real_mean = float(np.mean([raw_pr_auc[real_lane][key] for key in keys]))
        best = max(
            float(np.mean([raw_pr_auc[str(lane)][key] for key in keys]))
            for lane in matrix["matched_control_lanes"]
        )
        fold_results[str(fold)] = real_mean - best > 0.0
    real_mean = float(np.mean(list(raw_pr_auc[real_lane].values())))
    ar_mean = float(np.mean(list(fresh_ar_pr_auc.values())))
    strongest_mean = float(np.mean(list(raw_pr_auc[str(strongest)].values())))
    current_mean = float(np.mean(list(raw_pr_auc[ablation_lane].values())))
    label_mean = float(np.mean(list(raw_pr_auc["label_permutation_residual"].values())))
    uncertainty = plan["uncertainty"]
    ar_panel = _paired_prediction_panel(output_dir, records, real_lane, None)
    control_panel = _paired_prediction_panel(output_dir, records, real_lane, str(strongest))
    bootstrap_vs_ar = paired_video_bootstrap_raw_pr_auc_delta(
        *ar_panel,
        seed=int(uncertainty["seed"]),
        resamples=int(uncertainty["resamples"]),
    )
    bootstrap_vs_control = paired_video_bootstrap_raw_pr_auc_delta(
        *control_panel,
        seed=int(uncertainty["seed"]) + 1,
        resamples=int(uncertainty["resamples"]),
    )
    positive_folds = sum(fold_results.values())
    gates = {
        "real_mean_raw_pr_auc_minus_ar_positive": real_mean - ar_mean > 0.0,
        "real_mean_raw_pr_auc_minus_strongest_control_positive": (
            real_mean - strongest_mean > 0.0
        ),
        "paired_median_real_minus_strongest_control_positive": float(np.median(paired)) > 0.0,
        "every_target_mean_real_minus_strongest_control_positive": all(
            target_results.values()
        ),
        "at_least_four_of_five_fold_means_positive": positive_folds >= 4,
        "real_mean_minus_registered_ablation_positive": real_mean - current_mean > 0.0,
        "label_permutation_mean_raw_pr_auc_delta_vs_ar_nonpositive": label_mean - ar_mean <= 0.0,
        "bootstrap_lower_ci_vs_ar_positive": float(bootstrap_vs_ar["ci_lower"]) > 0.0,
        "bootstrap_lower_ci_vs_strongest_control_positive": (
            float(bootstrap_vs_control["ci_lower"]) > 0.0
        ),
        "exact_matrix_and_artifact_audits_pass": True,
    }
    summary: dict[str, Any] = {
        "schema": "veatic21_innovation_control_summary_v1",
        "request_sha256": request_sha256,
        "plan_sha256": plan["plan_sha256"],
        "completed_cells": len(records),
        "expected_cells": int(matrix["expected_cells"]),
        "lane_summaries": lane_summaries,
        "strongest_matched_control": str(strongest),
        "primary_metric": "raw_pr_auc",
        "real_mean_raw_pr_auc": real_mean,
        "matched_fresh_ar_mean_raw_pr_auc": ar_mean,
        "real_minus_ar_raw_pr_auc": real_mean - ar_mean,
        "real_minus_strongest_control_raw_pr_auc": real_mean - strongest_mean,
        "paired_median_real_minus_strongest_control": float(np.median(paired)),
        "positive_fold_count": positive_folds,
        "credible_fold_gate": positive_folds >= 4,
        "strong_fold_gate": positive_folds == 5,
        "paired_video_cluster_bootstrap_vs_ar": bootstrap_vs_ar,
        "paired_video_cluster_bootstrap_vs_strongest_control": bootstrap_vs_control,
        "target_gate_results": target_results,
        "fold_gate_results": fold_results,
        "gate_results": gates,
        "all_gates_pass": all(gates.values()),
        "records": records,
        "worker_count": 1,
        "backend": "mlx",
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    summary["summary_sha256"] = digest_json(summary)
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


__all__ = [
    "build_current_innovation_control_plan",
    "build_innovation_control_plan",
    "run_innovation_control_program",
    "write_innovation_control_plan",
]
