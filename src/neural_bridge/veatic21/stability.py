"""Fixed-seed stability expansion for the retained VEATIC 2.1 spike recipe."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from .data import CanonicalSubstrate
from .evidence import atomic_write_json, digest_json, sha256_file
from .stage1 import Stage1CellConfig, run_stage1_discovery_cell

_SCHEMA = "veatic21_stability_plan_v1"


def _require_self_digest(record: Mapping[str, Any], field: str) -> None:
    expected = record.get(field)
    if not isinstance(expected, str):
        raise ValueError(f"artifact is missing {field}")
    payload = dict(record)
    payload.pop(field)
    if digest_json(payload) != expected:
        raise ValueError(f"artifact failed its {field} integrity check")


def build_stability_plan(
    preregistration: Mapping[str, Any],
    recipe_plan: Mapping[str, Any],
    recipe_selection: Mapping[str, Any],
    pca_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the retained recipe to the preregistered nine stability seeds."""

    _require_self_digest(preregistration, "preregistration_sha256")
    _require_self_digest(recipe_plan, "plan_sha256")
    _require_self_digest(recipe_selection, "resolution_sha256")
    _require_self_digest(pca_manifest, "manifest_sha256")
    if recipe_selection.get("plan_sha256") != recipe_plan.get("plan_sha256"):
        raise ValueError("recipe selection does not resolve the supplied recipe plan")
    if recipe_selection.get("benchmark_test_labels_accessed") is not False:
        raise ValueError("stability plan cannot use benchmark-test labels")
    if pca_manifest.get("manifest_sha256") != recipe_plan.get("artifacts", {}).get(
        "pca_manifest_sha256"
    ):
        raise ValueError("stability artifacts do not share the PCA manifest")
    seeds = [int(seed) for seed in preregistration["training"]["stability_seed_panel"]]
    if seeds != list(range(20_260_801, 20_260_810)):
        raise ValueError("stability requires the fixed preregistered nine-seed panel")
    ensembles = [
        list(map(int, group))
        for group in preregistration["training"]["checkpoint_ensembles"]
    ]
    if sorted(index for group in ensembles for index in group) != list(range(len(seeds))):
        raise ValueError("checkpoint ensembles must partition the stability panel")

    execution_plan = deepcopy(dict(recipe_plan["execution_plan"]))
    execution_plan["matrix"]["comparison_seeds"] = seeds
    execution_plan.pop("plan_sha256", None)
    execution_plan["plan_sha256"] = digest_json(execution_plan)
    targets = [str(value) for value in recipe_plan["matrix"]["targets"]]
    folds = [int(value) for value in recipe_plan["matrix"]["folds"]]
    plan: dict[str, Any] = {
        "schema": _SCHEMA,
        "purpose": "fixed_fold_seed_stability",
        "artifacts": {
            "pca_manifest_sha256": pca_manifest["manifest_sha256"],
            "preregistration_sha256": preregistration["preregistration_sha256"],
            "recipe_plan_sha256": recipe_plan["plan_sha256"],
            "recipe_selection_sha256": recipe_selection["resolution_sha256"],
            "stability_code_sha256": sha256_file(Path(__file__)),
        },
        "selected_recipe": dict(recipe_selection["selected_recipe"]),
        "selected_representation": recipe_selection["selected_representation"],
        "selected_head_family": recipe_selection["selected_head_family"],
        "execution_plan": execution_plan,
        "matrix": {
            "targets": targets,
            "folds": folds,
            "stability_seeds": seeds,
            "checkpoint_ensembles": ensembles,
            "expected_cells": len(targets) * len(folds) * len(seeds),
            "worker_count": 1,
            "backend": "mlx",
            "sealed_tail_labels": True,
        },
        "whole_fold_seed_no_harm": (
            "use_residual_only_when_inner_delta_is_positive_else_fresh_ar"
        ),
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    plan["plan_sha256"] = digest_json(plan)
    return plan


def write_stability_plan(path: Path, plan: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(plan))


def _quarantine_partial(cell_dir: Path, output_dir: Path, plan_sha256: str) -> None:
    if not cell_dir.exists() or (cell_dir / "metrics.json").is_file():
        return
    request_path = cell_dir / "request.json"
    state_path = cell_dir / "state.json"
    if not request_path.is_file() or not state_path.is_file():
        raise RuntimeError("refusing unrecognized partial stability cell")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if request.get("plan_sha256") != plan_sha256 or state.get("status") not in {
        "training",
        "failed",
    }:
        raise RuntimeError("refusing changed partial stability cell")
    quarantine = output_dir / "quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    base = "__".join(cell_dir.parts[-3:])
    index = 1
    destination = quarantine / f"{base}__attempt-{index}"
    while destination.exists():
        index += 1
        destination = quarantine / f"{base}__attempt-{index}"
    cell_dir.replace(destination)


def run_stability_program(
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
    """Train the retained recipe across all fixed stability folds and seeds."""

    _require_self_digest(plan, "plan_sha256")
    if plan.get("schema") != _SCHEMA:
        raise ValueError("stability runner requires the current plan schema")
    if plan.get("artifacts", {}).get("stability_code_sha256") != sha256_file(Path(__file__)):
        raise ValueError("stability plan does not bind the current runner code")
    request = {
        "schema": "veatic21_stability_run_request_v1",
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
            raise RuntimeError("refusing stability resume because request changed")
    else:
        atomic_write_json(request_path, request)

    recipe = plan["selected_recipe"]
    matrix = plan["matrix"]
    records: list[dict[str, Any]] = []
    for target in matrix["targets"]:
        for fold in matrix["folds"]:
            for seed in matrix["stability_seeds"]:
                cell_dir = output_dir / "targets" / str(target) / f"fold-{fold}" / f"seed-{seed}"
                _quarantine_partial(cell_dir, output_dir, plan["execution_plan"]["plan_sha256"])
                metrics = run_stage1_discovery_cell(
                    substrate,
                    preregistration,
                    calibration,
                    pca_manifest,
                    plan["execution_plan"],
                    pca_root,
                    cell_dir,
                    Stage1CellConfig(
                        target_name=str(target),
                        fold=int(fold),
                        seed=int(seed),
                        pca_width=512,
                        head_family=cast(
                            Literal[
                                "frozen_ar_plus_causal_temporal_residual",
                                "frozen_ar_plus_gated_multiscale_temporal_residual",
                            ],
                            plan["selected_head_family"],
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
                    ),
                )
                record = {
                    "target": str(target),
                    "fold": int(fold),
                    "seed": int(seed),
                    "inner_average_precision_skill_delta_vs_frozen_ar": metrics[
                        "inner_average_precision_skill_delta_vs_frozen_ar"
                    ],
                    "whole_fold_seed_uses_residual": metrics["whole_fold_seed_uses_residual"],
                    "best_epoch": metrics["best_epoch"],
                    "cell_metrics_sha256": sha256_file(cell_dir / "metrics.json"),
                    "cell_directory": str(cell_dir.relative_to(output_dir)),
                }
                records.append(record)
                if progress is not None:
                    progress(
                        {
                            "schema": "veatic21_stability_progress_v1",
                            "completed_cells": len(records),
                            "expected_cells": int(matrix["expected_cells"]),
                            "last_cell": record,
                            "worker_count": 1,
                            "benchmark_test_labels_accessed": False,
                        }
                    )
    deltas = np.asarray(
        [float(row["inner_average_precision_skill_delta_vs_frozen_ar"]) for row in records]
    )
    summary: dict[str, Any] = {
        "schema": "veatic21_stability_summary_v1",
        "request_sha256": request_sha256,
        "plan_sha256": plan["plan_sha256"],
        "completed_cells": len(records),
        "expected_cells": int(matrix["expected_cells"]),
        "mean_inner_average_precision_skill_delta_vs_frozen_ar": float(np.mean(deltas)),
        "median_inner_average_precision_skill_delta_vs_frozen_ar": float(np.median(deltas)),
        "positive_residual_cells": int(np.sum(deltas > 0.0)),
        "whole_fold_seed_ar_fallback_cells": int(np.sum(deltas <= 0.0)),
        "records": records,
        "worker_count": 1,
        "backend": "mlx",
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    summary["summary_sha256"] = digest_json(summary)
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


__all__ = ["build_stability_plan", "run_stability_program", "write_stability_plan"]
