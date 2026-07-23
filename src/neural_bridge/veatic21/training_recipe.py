"""Staged numeric training-recipe discovery for the selected VEATIC 2.1 head."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .data import CanonicalSubstrate
from .evidence import atomic_write_json, digest_json, sha256_file
from .stage1 import Stage1CellConfig, run_stage1_discovery_cell

_SCHEMA = "veatic21_training_recipe_plan_v1"
_HEAD = "frozen_ar_plus_causal_temporal_residual"
_REPRESENTATION = "fixed_pca512"
_BATCH_ROWS = 4096
_STAGE_AXES: tuple[tuple[str, tuple[int | float, ...]], ...] = (
    ("hidden_width", (64, 128, 256, 512)),
    ("learning_rate", (0.0001, 0.0003, 0.0009)),
    ("weight_decay", (0.0, 0.0001, 0.001)),
    ("residual_logit_cap", (0.25, 0.5, 1.0)),
)


def _require_self_digest(record: Mapping[str, Any], field: str) -> None:
    expected = record.get(field)
    if not isinstance(expected, str):
        raise ValueError(f"artifact is missing {field}")
    payload = dict(record)
    payload.pop(field)
    if digest_json(payload) != expected:
        raise ValueError(f"artifact failed its {field} integrity check")


def _recipe_from_screen(screen: Mapping[str, Any]) -> dict[str, Any]:
    source = screen.get("matched_recipe")
    if not isinstance(source, Mapping):
        raise ValueError("head-family screen is missing its matched recipe")
    names = (
        "hidden_width",
        "learning_rate",
        "weight_decay",
        "residual_logit_cap",
        "context_rows",
        "minimum_epochs",
        "plateau_patience",
        "nonconvergence_patience",
    )
    recipe = {name: deepcopy(source[name]) for name in names}
    recipe["batch_rows"] = _BATCH_ROWS
    return recipe


def _execution_plan(stage1_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the MLX execution contract without altering the sealed source plan."""

    plan = deepcopy(dict(stage1_plan))
    safe_batches = plan["capacity"]["safe_batch_rows_by_head_hidden_width"]
    for width in _STAGE_AXES[0][1]:
        safe_batches[f"{_HEAD}:{int(width)}"] = _BATCH_ROWS
    plan["capacity"]["training_recipe_batch_rows"] = _BATCH_ROWS
    plan["capacity"]["memory_fraction_cap"] = None
    plan["capacity"]["worker_count"] = 1
    plan["source_stage1_plan_sha256"] = stage1_plan["plan_sha256"]
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = digest_json(plan)
    return plan


def _baseline_records(
    summary: Mapping[str, Any],
    expected_keys: set[tuple[str, int, int]],
) -> list[dict[str, Any]]:
    records = [row for row in summary["records"] if row.get("lane") == _REPRESENTATION]
    indexed = {
        (str(row["target"]), int(row["fold"]), int(row["seed"])): row for row in records
    }
    if set(indexed) != expected_keys or len(records) != len(expected_keys):
        raise ValueError("baseline summary does not contain the exact PCA-512 cell panel")
    return [
        {
            "target": key[0],
            "fold": key[1],
            "seed": key[2],
            "inner_average_precision_skill_delta_vs_frozen_ar": float(
                indexed[key]["inner_average_precision_skill_delta_vs_frozen_ar"]
            ),
            "whole_fold_seed_uses_residual": bool(
                indexed[key]["whole_fold_seed_uses_residual"]
            ),
            "best_epoch": int(indexed[key]["best_epoch"]),
            "source_cell_directory": str(indexed[key]["cell_directory"]),
            "source_cell_metrics_sha256": str(indexed[key]["cell_metrics_sha256"]),
        }
        for key in sorted(expected_keys)
    ]


def build_training_recipe_plan(
    stage1_plan: Mapping[str, Any],
    representation_summary: Mapping[str, Any],
    representation_selection: Mapping[str, Any],
    head_screen: Mapping[str, Any],
    head_selection: Mapping[str, Any],
    pca_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Register four sequential one-axis gates and the reusable causal baseline."""

    _require_self_digest(stage1_plan, "plan_sha256")
    _require_self_digest(representation_summary, "summary_sha256")
    _require_self_digest(representation_selection, "selection_sha256")
    _require_self_digest(head_screen, "screen_sha256")
    _require_self_digest(head_selection, "selection_sha256")
    _require_self_digest(pca_manifest, "manifest_sha256")
    if stage1_plan.get("purpose") != "spike_discovery":
        raise ValueError("training-recipe discovery requires the spike-discovery plan")
    if representation_selection.get("selected_representation") != _REPRESENTATION:
        raise ValueError("training-recipe discovery requires selected fixed PCA-512")
    if head_selection.get("selected_head_family") != _HEAD:
        raise ValueError("training-recipe discovery requires the selected causal head")
    if representation_selection.get("summary_sha256") != representation_summary.get(
        "summary_sha256"
    ):
        raise ValueError("representation selection does not bind the baseline summary")
    if head_selection.get("screen_sha256") != head_screen.get("screen_sha256"):
        raise ValueError("head selection does not bind the supplied head screen")
    if pca_manifest.get("manifest_sha256") != stage1_plan.get("artifacts", {}).get(
        "pca_manifest_sha256"
    ):
        raise ValueError("training-recipe artifacts do not share the PCA manifest")
    if int(representation_summary.get("completed_cells", -1)) != int(
        representation_summary.get("expected_cells", -2)
    ):
        raise ValueError("training-recipe discovery requires a complete baseline summary")
    if representation_summary.get("benchmark_test_labels_accessed") is not False:
        raise ValueError("training-recipe discovery cannot use benchmark-test labels")

    targets = [str(value) for value in head_screen["matrix"]["targets"]]
    folds = [int(value) for value in head_screen["matrix"]["folds"]]
    seeds = [int(value) for value in head_screen["matrix"]["comparison_seeds"]]
    keys = {(target, fold, seed) for target in targets for fold in folds for seed in seeds}
    baseline_records = _baseline_records(representation_summary, keys)
    recipe = _recipe_from_screen(head_screen)
    if (
        int(recipe["hidden_width"]) != 64
        or float(recipe["learning_rate"]) != 0.0003
        or float(recipe["weight_decay"]) != 0.0001
        or float(recipe["residual_logit_cap"]) != 0.5
    ):
        raise ValueError("the reusable baseline is not the current VEATIC causal recipe")

    cell_count = len(keys)
    stages = []
    expected_new_cells = 0
    for index, (axis, values) in enumerate(_STAGE_AXES, start=1):
        baseline_value = recipe[axis]
        if baseline_value not in values:
            raise ValueError(f"baseline {axis} is missing from its candidate set")
        new_cells = (len(values) - 1) * cell_count
        expected_new_cells += new_cells
        stages.append(
            {
                "index": index,
                "axis": axis,
                "values": list(values),
                "baseline_value": baseline_value,
                "baseline_source": (
                    "verified_fixed_pca512_causal_records"
                    if index == 1
                    else "selected_records_from_previous_stage"
                ),
                "new_candidate_cells": new_cells,
                "selection_primary": (
                    "mean_inner_average_precision_skill_delta_vs_frozen_ar_descending"
                ),
                "tie_break": f"{axis}_ascending",
            }
        )

    execution_plan = _execution_plan(stage1_plan)
    plan: dict[str, Any] = {
        "schema": _SCHEMA,
        "purpose": "staged_numeric_training_recipe_discovery",
        "artifacts": {
            "head_family_screen_sha256": head_screen["screen_sha256"],
            "head_family_selection_sha256": head_selection["selection_sha256"],
            "pca_manifest_sha256": pca_manifest["manifest_sha256"],
            "representation_selection_sha256": representation_selection["selection_sha256"],
            "representation_summary_sha256": representation_summary["summary_sha256"],
            "source_stage1_plan_sha256": stage1_plan["plan_sha256"],
            "training_recipe_code_sha256": sha256_file(Path(__file__)),
        },
        "selected_representation": _REPRESENTATION,
        "selected_pca_width": 512,
        "selected_head_family": _HEAD,
        "initial_recipe": recipe,
        "execution_plan": execution_plan,
        "baseline_records": baseline_records,
        "matrix": {
            "targets": targets,
            "folds": folds,
            "comparison_seeds": seeds,
            "cells_per_candidate": cell_count,
            "stages": stages,
            "expected_new_cells": expected_new_cells,
            "worker_count": 1,
            "backend": "mlx",
            "sealed_tail_labels": True,
        },
        "selection_after_each_stage": {
            "requires_all_candidate_cells": True,
            "primary_key": "mean_inner_average_precision_skill_delta_vs_frozen_ar",
            "tie_break": "smaller_numeric_axis_value",
            "whole_fold_seed_no_harm": (
                "use_residual_only_when_inner_delta_is_positive_else_fresh_ar"
            ),
            "later_stages_inherit_only_the_prior_stage_winner": True,
        },
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    plan["plan_sha256"] = digest_json(plan)
    return plan


def write_training_recipe_plan(path: Path, plan: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(plan))


def _value_label(value: int | float) -> str:
    return format(value, ".8g").replace("-", "m").replace(".", "p")


def _candidate_summary(
    value: int | float,
    records: list[dict[str, Any]],
    *,
    reused: bool,
) -> dict[str, Any]:
    deltas = np.asarray(
        [float(row["inner_average_precision_skill_delta_vs_frozen_ar"]) for row in records]
    )
    return {
        "value": value,
        "cell_count": len(records),
        "mean_inner_average_precision_skill_delta_vs_frozen_ar": float(np.mean(deltas)),
        "median_inner_average_precision_skill_delta_vs_frozen_ar": float(np.median(deltas)),
        "positive_residual_cells": int(np.sum(deltas > 0.0)),
        "whole_fold_seed_ar_fallback_cells": int(np.sum(deltas <= 0.0)),
        "records_reused": reused,
    }


def _run_candidate(
    substrate: CanonicalSubstrate,
    preregistration: Mapping[str, Any],
    calibration: Mapping[str, Any],
    pca_manifest: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    pca_root: Path,
    candidate_dir: Path,
    recipe: Mapping[str, Any],
    targets: list[str],
    folds: list[int],
    seeds: list[int],
    progress: Callable[[Mapping[str, Any]], None] | None,
    completed_before: int,
    expected_cells: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for target in targets:
        for fold in folds:
            for seed in seeds:
                cell_dir = (
                    candidate_dir / "targets" / target / f"fold-{fold}" / f"seed-{seed}"
                )
                config = Stage1CellConfig(
                    target_name=target,
                    fold=fold,
                    seed=seed,
                    pca_width=512,
                    head_family=_HEAD,
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
                metrics = run_stage1_discovery_cell(
                    substrate,
                    preregistration,
                    calibration,
                    pca_manifest,
                    execution_plan,
                    pca_root,
                    cell_dir,
                    config,
                )
                record = {
                    "target": target,
                    "fold": fold,
                    "seed": seed,
                    "inner_average_precision_skill_delta_vs_frozen_ar": metrics[
                        "inner_average_precision_skill_delta_vs_frozen_ar"
                    ],
                    "whole_fold_seed_uses_residual": metrics[
                        "whole_fold_seed_uses_residual"
                    ],
                    "best_epoch": metrics["best_epoch"],
                    "cell_metrics_sha256": sha256_file(cell_dir / "metrics.json"),
                    "cell_directory": str(cell_dir),
                }
                records.append(record)
                if progress is not None:
                    progress(
                        {
                            "schema": "veatic21_training_recipe_progress_v1",
                            "completed_new_cells": completed_before + len(records),
                            "expected_new_cells": expected_cells,
                            "last_cell": record,
                            "worker_count": 1,
                            "benchmark_test_labels_accessed": False,
                        }
                    )
    return records


def run_training_recipe_program(
    substrate: CanonicalSubstrate,
    preregistration: Mapping[str, Any],
    calibration: Mapping[str, Any],
    pca_manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    pca_root: Path,
    output_dir: Path,
    *,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run all four gates sequentially with exactly one MLX worker."""

    _require_self_digest(plan, "plan_sha256")
    if plan.get("schema") != _SCHEMA:
        raise ValueError("training-recipe runner requires the current plan schema")
    if plan.get("artifacts", {}).get("training_recipe_code_sha256") != sha256_file(Path(__file__)):
        raise ValueError("training-recipe plan does not bind the current runner code")
    if plan.get("benchmark_test_labels_accessed") is not False:
        raise ValueError("training-recipe runner cannot use benchmark-test labels")
    request = {
        "schema": "veatic21_training_recipe_run_request_v1",
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
            raise RuntimeError("refusing training-recipe resume because request changed")
    else:
        atomic_write_json(request_path, request)

    matrix = plan["matrix"]
    targets = [str(value) for value in matrix["targets"]]
    folds = [int(value) for value in matrix["folds"]]
    seeds = [int(value) for value in matrix["comparison_seeds"]]
    current_recipe = deepcopy(dict(plan["initial_recipe"]))
    current_records = [dict(row) for row in plan["baseline_records"]]
    completed_new = 0
    stage_results: list[dict[str, Any]] = []

    for stage in matrix["stages"]:
        axis = str(stage["axis"])
        baseline_value = current_recipe[axis]
        records_by_value: dict[int | float, list[dict[str, Any]]] = {
            baseline_value: current_records
        }
        stage_dir = output_dir / f"stage-{int(stage['index'])}-{axis.replace('_', '-')}"
        for value in stage["values"]:
            if value == baseline_value:
                continue
            candidate_recipe = deepcopy(current_recipe)
            candidate_recipe[axis] = value
            records = _run_candidate(
                substrate,
                preregistration,
                calibration,
                pca_manifest,
                plan["execution_plan"],
                pca_root,
                stage_dir / f"candidate-{_value_label(value)}",
                candidate_recipe,
                targets,
                folds,
                seeds,
                progress,
                completed_new,
                int(matrix["expected_new_cells"]),
            )
            records_by_value[value] = records
            completed_new += len(records)

        ranking = [
            _candidate_summary(value, records, reused=value == baseline_value)
            for value, records in records_by_value.items()
        ]
        ranking.sort(
            key=lambda row: (
                -float(row["mean_inner_average_precision_skill_delta_vs_frozen_ar"]),
                float(row["value"]),
            )
        )
        selected_value = ranking[0]["value"]
        current_recipe[axis] = selected_value
        current_records = records_by_value[selected_value]
        result: dict[str, Any] = {
            "schema": "veatic21_training_recipe_stage_selection_v1",
            "plan_sha256": plan["plan_sha256"],
            "stage_index": int(stage["index"]),
            "axis": axis,
            "baseline_value": baseline_value,
            "selected_value": selected_value,
            "ranking": ranking,
            "selected_recipe": deepcopy(current_recipe),
            "benchmark_test_labels_accessed": False,
            "promotable": False,
        }
        result["selection_sha256"] = digest_json(result)
        atomic_write_json(stage_dir / "selection.json", result)
        stage_results.append(result)

    if completed_new != int(matrix["expected_new_cells"]):
        raise RuntimeError("training-recipe program did not complete its registered matrix")
    summary: dict[str, Any] = {
        "schema": "veatic21_training_recipe_summary_v1",
        "request_sha256": request_sha256,
        "plan_sha256": plan["plan_sha256"],
        "completed_new_cells": completed_new,
        "expected_new_cells": int(matrix["expected_new_cells"]),
        "stage_selections": stage_results,
        "selected_recipe": current_recipe,
        "selected_record_count": len(current_records),
        "worker_count": 1,
        "backend": "mlx",
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    summary["summary_sha256"] = digest_json(summary)
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


__all__ = [
    "build_training_recipe_plan",
    "run_training_recipe_program",
    "write_training_recipe_plan",
]
