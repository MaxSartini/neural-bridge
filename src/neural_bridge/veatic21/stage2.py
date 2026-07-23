"""Train-only target shortlisting and fixed-PCA screening for VEATIC 2.1."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from .data import CanonicalSubstrate
from .evidence import atomic_write_json, digest_json, sha256_file
from .stage1 import Stage1CellConfig, run_stage1_discovery_cell

_SCHEMA = "veatic21_stage2_pca_screen_v1"


def _require_self_digest(record: Mapping[str, Any], field: str) -> None:
    expected = record.get(field)
    if not isinstance(expected, str):
        raise ValueError(f"artifact is missing {field}")
    payload = dict(record)
    payload.pop(field)
    if digest_json(payload) != expected:
        raise ValueError(f"artifact failed its {field} integrity check")


def select_train_only_target_shortlist(
    ar_benchmark: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Select one stable fresh-AR target per preregistered train quantile."""

    _require_self_digest(ar_benchmark, "summary_sha256")
    _require_self_digest(plan, "plan_sha256")
    if plan.get("purpose") != "spike_discovery":
        raise ValueError("target shortlisting requires the spike-discovery plan")
    if plan.get("artifacts", {}).get("ar_benchmark_sha256") != ar_benchmark.get("summary_sha256"):
        raise ValueError("target shortlisting artifacts do not share one AR benchmark")
    if ar_benchmark.get("completed_cells") != ar_benchmark.get("expected_cells"):
        raise ValueError("target shortlisting requires every fresh-AR cell")
    if ar_benchmark.get("invalid_cells") != 0:
        raise ValueError("target shortlisting cannot use invalid fresh-AR cells")

    targets = {str(row["name"]): row for row in plan["matrix"]["targets"]}
    expected_per_target = len(plan["matrix"]["folds"]) * len(plan["matrix"]["comparison_seeds"])
    records_by_target: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in ar_benchmark["records"]:
        target = str(record["target"])
        if target not in targets:
            raise ValueError(f"AR benchmark contains an unregistered target: {target}")
        if record.get("status") != "complete":
            raise ValueError(f"AR benchmark target is incomplete: {target}")
        records_by_target[target].append(record)
    if set(records_by_target) != set(targets):
        raise ValueError("AR benchmark does not cover every registered target")

    rankings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for target, target_records in records_by_target.items():
        if len(target_records) != expected_per_target:
            raise ValueError(f"AR target has the wrong fold/seed cell count: {target}")
        skills = np.asarray(
            [float(record["fresh_ar_average_precision_skill"]) for record in target_records]
        )
        prevalences = np.asarray([float(record["event_prevalence"]) for record in target_records])
        quantile = float(targets[target]["quantile"])
        rankings[f"{quantile:.3f}"].append(
            {
                "target": target,
                "train_quantile": quantile,
                "mean_fresh_ar_average_precision_skill": float(np.mean(skills)),
                "minimum_fresh_ar_average_precision_skill": float(np.min(skills)),
                "standard_deviation_fresh_ar_average_precision_skill": float(np.std(skills)),
                "mean_event_prevalence": float(np.mean(prevalences)),
                "cell_count": len(target_records),
            }
        )

    selected: list[dict[str, Any]] = []
    ranked_groups: list[dict[str, Any]] = []
    for quantile_key in sorted(rankings, key=float, reverse=True):
        ranked = sorted(
            rankings[quantile_key],
            key=lambda row: (
                -float(row["mean_fresh_ar_average_precision_skill"]),
                -float(row["minimum_fresh_ar_average_precision_skill"]),
                float(row["standard_deviation_fresh_ar_average_precision_skill"]),
                str(row["target"]),
            ),
        )
        selected.append(dict(ranked[0]))
        ranked_groups.append(
            {
                "train_quantile": float(quantile_key),
                "ranked_targets": ranked,
                "selected_target": ranked[0]["target"],
            }
        )

    return {
        "selection_scope": "benchmark_train_inner_grouped_video_folds_only",
        "selection_unit": "exactly_one_target_per_preregistered_train_quantile",
        "primary_rank": "mean_fresh_ar_average_precision_skill_descending",
        "tie_breaks": [
            "minimum_fresh_ar_average_precision_skill_descending",
            "standard_deviation_fresh_ar_average_precision_skill_ascending",
            "target_name_ascending",
        ],
        "quantile_groups": ranked_groups,
        "selected": selected,
        "selected_targets": [str(row["target"]) for row in selected],
        "sealed_tail_labels_used": False,
    }


def build_stage2_pca_screen(
    ar_benchmark: Mapping[str, Any],
    plan: Mapping[str, Any],
    executor_request: Mapping[str, Any],
    *,
    executor_request_sha256: str,
) -> dict[str, Any]:
    """Register a fixed-nuisance PCA-width screen over the train-only shortlist."""

    shortlist = select_train_only_target_shortlist(ar_benchmark, plan)
    if executor_request.get("benchmark_test_labels_accessed") is not False:
        raise ValueError("executor validation must not access benchmark-test labels")
    if executor_request.get("promotable") is not False:
        raise ValueError("executor validation must remain non-promotable")
    source_config = executor_request.get("config")
    if not isinstance(source_config, Mapping):
        raise ValueError("executor validation is missing its configuration")
    recipe_fields = (
        "head_family",
        "hidden_width",
        "learning_rate",
        "weight_decay",
        "residual_logit_cap",
        "batch_rows",
        "context_rows",
        "minimum_epochs",
        "plateau_patience",
        "nonconvergence_patience",
    )
    recipe = {field: source_config[field] for field in recipe_fields}
    if recipe["head_family"] != "frozen_ar_plus_causal_temporal_residual":
        raise ValueError("Stage-2 PCA isolation requires the causal temporal residual")
    capacity_key = f"{recipe['head_family']}:{int(recipe['hidden_width'])}"
    safe_batches = plan["capacity"]["safe_batch_rows_by_head_hidden_width"]
    if capacity_key not in safe_batches or int(recipe["batch_rows"]) > int(
        safe_batches[capacity_key]
    ):
        raise ValueError("executor nuisance recipe exceeds the current MLX capacity")
    policy = plan["checkpoint_policy"]
    if (
        int(recipe["minimum_epochs"]) != int(policy["minimum_epochs_before_termination"])
        or int(recipe["plateau_patience"]) != int(policy["plateau_patience_epochs"])
        or int(recipe["nonconvergence_patience"]) != int(policy["nonconvergence_patience_epochs"])
    ):
        raise ValueError("executor nuisance recipe violates the checkpoint plan")

    folds = [int(row["fold"]) for row in plan["matrix"]["folds"]]
    width_sets = {
        tuple(int(width) for width in row["candidate_pca_widths"])
        for row in plan["matrix"]["folds"]
    }
    if len(width_sets) != 1:
        raise ValueError("Stage-2 PCA screen requires identical candidate widths by fold")
    pca_widths = list(next(iter(width_sets)))
    seeds = [int(seed) for seed in plan["matrix"]["comparison_seeds"]]
    expected_cells = len(shortlist["selected_targets"]) * len(folds) * len(seeds) * len(pca_widths)
    screen: dict[str, Any] = {
        "schema": _SCHEMA,
        "purpose": "fixed_pca_width_screen",
        "artifacts": {
            "ar_benchmark_sha256": ar_benchmark["summary_sha256"],
            "executor_validation_request_file_sha256": executor_request_sha256,
            "stage1_plan_sha256": plan["plan_sha256"],
        },
        "target_shortlist": shortlist,
        "fixed_nuisance_recipe": {
            **recipe,
            "source": "current_veatic_executor_validation_configuration_only",
            "source_score_used_for_selection": False,
        },
        "matrix": {
            "targets": shortlist["selected_targets"],
            "folds": folds,
            "comparison_seeds": seeds,
            "pca_widths": pca_widths,
            "expected_cells": expected_cells,
            "varied_axis": "fixed_fold_owned_pca_width",
            "sealed_tail_labels": True,
        },
        "selection_after_completion": {
            "primary_key": "mean_inner_average_precision_skill_delta_vs_frozen_ar",
            "tie_breaks": [
                "paired_video_cluster_bootstrap_lower_confidence_bound_descending",
                "median_delta_descending",
                "pca_width_ascending",
            ],
            "width_not_selected_until_all_expected_cells_complete": True,
        },
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    screen["screen_sha256"] = digest_json(screen)
    return screen


def write_stage2_pca_screen(path: Path, screen: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(screen))


def run_stage2_pca_screen(
    substrate: CanonicalSubstrate,
    preregistration: Mapping[str, Any],
    calibration: Mapping[str, Any],
    pca_manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    screen: Mapping[str, Any],
    pca_root: Path,
    output_dir: Path,
    *,
    progress=None,
) -> dict[str, Any]:
    """Run every registered fixed-PCA screen cell, resumable by completed cell."""

    _require_self_digest(screen, "screen_sha256")
    if screen.get("schema") != _SCHEMA:
        raise ValueError("Stage-2 PCA screen requires the current schema")
    if screen.get("artifacts", {}).get("stage1_plan_sha256") != plan.get("plan_sha256"):
        raise ValueError("Stage-2 PCA screen does not bind the supplied Stage-1 plan")
    request = {
        "schema": "veatic21_stage2_pca_run_request_v1",
        "screen_sha256": screen["screen_sha256"],
        "stage1_plan_sha256": plan["plan_sha256"],
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    request_sha256 = digest_json(request)
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "request.json"
    if request_path.is_file():
        saved = json.loads(request_path.read_text(encoding="utf-8"))
        if digest_json(saved) != request_sha256:
            raise RuntimeError("refusing Stage-2 PCA resume because the request changed")
    else:
        atomic_write_json(request_path, request)

    recipe = screen["fixed_nuisance_recipe"]
    records: list[dict[str, Any]] = []
    expected_cells = int(screen["matrix"]["expected_cells"])
    for target in screen["matrix"]["targets"]:
        for fold in screen["matrix"]["folds"]:
            for seed in screen["matrix"]["comparison_seeds"]:
                for pca_width in screen["matrix"]["pca_widths"]:
                    config = Stage1CellConfig(
                        target_name=str(target),
                        fold=int(fold),
                        seed=int(seed),
                        pca_width=int(pca_width),
                        head_family=cast(
                            Literal[
                                "frozen_ar_plus_causal_temporal_residual",
                                "frozen_ar_plus_gated_multiscale_temporal_residual",
                            ],
                            str(recipe["head_family"]),
                        ),
                        hidden_width=int(recipe["hidden_width"]),
                        learning_rate=float(recipe["learning_rate"]),
                        weight_decay=float(recipe["weight_decay"]),
                        residual_logit_cap=float(recipe["residual_logit_cap"]),
                        batch_rows=int(recipe["batch_rows"]),
                        context_rows=tuple(int(value) for value in recipe["context_rows"]),
                        minimum_epochs=int(recipe["minimum_epochs"]),
                        plateau_patience=int(recipe["plateau_patience"]),
                        nonconvergence_patience=int(recipe["nonconvergence_patience"]),
                    )
                    cell_dir = (
                        output_dir
                        / "targets"
                        / str(target)
                        / f"fold-{int(fold)}"
                        / f"seed-{int(seed)}"
                        / f"pca-{int(pca_width)}"
                    )
                    metrics = run_stage1_discovery_cell(
                        substrate,
                        preregistration,
                        calibration,
                        pca_manifest,
                        plan,
                        pca_root,
                        cell_dir,
                        config,
                    )
                    record = {
                        "target": str(target),
                        "fold": int(fold),
                        "seed": int(seed),
                        "pca_width": int(pca_width),
                        "inner_average_precision_skill_delta_vs_frozen_ar": metrics[
                            "inner_average_precision_skill_delta_vs_frozen_ar"
                        ],
                        "whole_fold_seed_uses_residual": metrics["whole_fold_seed_uses_residual"],
                        "best_epoch": metrics["best_epoch"],
                        "cell_metrics_sha256": sha256_file(cell_dir / "metrics.json"),
                        "cell_directory": str(cell_dir.relative_to(output_dir)),
                    }
                    records.append(record)
                    progress_record = {
                        "schema": "veatic21_stage2_pca_progress_v1",
                        "request_sha256": request_sha256,
                        "completed_cells": len(records),
                        "expected_cells": expected_cells,
                        "last_cell": record,
                        "benchmark_test_labels_accessed": False,
                    }
                    atomic_write_json(output_dir / "progress.json", progress_record)
                    if progress is not None:
                        progress(progress_record)

    summary: dict[str, Any] = {
        "schema": "veatic21_stage2_pca_summary_v1",
        "request_sha256": request_sha256,
        "screen_sha256": screen["screen_sha256"],
        "completed_cells": len(records),
        "expected_cells": expected_cells,
        "records": records,
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    summary["summary_sha256"] = digest_json(summary)
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


__all__ = [
    "build_stage2_pca_screen",
    "run_stage2_pca_screen",
    "select_train_only_target_shortlist",
    "write_stage2_pca_screen",
]
