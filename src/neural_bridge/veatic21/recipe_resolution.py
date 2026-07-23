"""Conservative resolution of the stopped VEATIC 2.1 numeric recipe sweep."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .evidence import atomic_write_json, digest_json, sha256_file


def _require_self_digest(record: Mapping[str, Any], field: str) -> None:
    expected = record.get(field)
    if not isinstance(expected, str):
        raise ValueError(f"artifact is missing {field}")
    payload = dict(record)
    payload.pop(field)
    if digest_json(payload) != expected:
        raise ValueError(f"artifact failed its {field} integrity check")


def resolve_stopped_training_recipe(
    plan: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    run_root: Path,
) -> dict[str, Any]:
    """Retain the complete baseline without claiming an incomplete sweep winner."""

    _require_self_digest(plan, "plan_sha256")
    _require_self_digest(baseline_summary, "summary_sha256")
    if plan.get("artifacts", {}).get("representation_summary_sha256") != baseline_summary.get(
        "summary_sha256"
    ):
        raise ValueError("stopped recipe plan does not bind the supplied baseline summary")
    baseline = {
        (str(row["target"]), int(row["fold"]), int(row["seed"])): float(
            row["inner_average_precision_skill_delta_vs_frozen_ar"]
        )
        for row in baseline_summary["records"]
        if row.get("lane") == "fixed_pca512"
    }
    expected = int(plan["matrix"]["cells_per_candidate"])
    if len(baseline) != expected:
        raise ValueError("resolution requires the complete verified PCA-512 causal baseline")

    candidate_root = run_root / "stage-1-hidden-width/candidate-128"
    candidate: dict[tuple[str, int, int], float] = {}
    for metrics_path in sorted(candidate_root.rglob("metrics.json")):
        state_path = metrics_path.with_name("state.json")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if state.get("status") != "complete" or state.get("metrics_sha256") != sha256_file(
            metrics_path
        ):
            raise ValueError("width-128 evidence contains an incomplete or changed cell")
        key = (str(metrics["target"]), int(metrics["fold"]), int(metrics["seed"]))
        if key not in baseline or key in candidate:
            raise ValueError("width-128 evidence is not an exact subset of the baseline panel")
        candidate[key] = float(metrics["inner_average_precision_skill_delta_vs_frozen_ar"])
    if not candidate:
        raise ValueError("resolution requires completed width-128 evidence")

    keys = sorted(candidate)
    baseline_values = np.asarray([baseline[key] for key in keys])
    candidate_values = np.asarray([candidate[key] for key in keys])
    paired = candidate_values - baseline_values
    by_target = {}
    for target in sorted({key[0] for key in keys}):
        values = np.asarray([paired[index] for index, key in enumerate(keys) if key[0] == target])
        by_target[target] = {
            "pair_count": len(values),
            "mean_width128_minus_width64": float(np.mean(values)),
            "width128_wins": int(np.sum(values > 0.0)),
            "width64_wins": int(np.sum(values < 0.0)),
        }

    resolution: dict[str, Any] = {
        "schema": "veatic21_training_recipe_resolution_v1",
        "plan_sha256": plan["plan_sha256"],
        "baseline_summary_sha256": baseline_summary["summary_sha256"],
        "resolution_code_sha256": sha256_file(Path(__file__)),
        "status": "stopped_incomplete_matrix_resolved_by_conservative_retention",
        "registered_expected_new_cells": int(plan["matrix"]["expected_new_cells"]),
        "completed_new_cells": len(candidate),
        "matrix_completion_fraction": len(candidate) / int(plan["matrix"]["expected_new_cells"]),
        "stopping_rule_was_preregistered": False,
        "stopping_reason": (
            "interim evidence showed material width harm and full sweep was inefficient"
        ),
        "width_evidence": {
            "pair_count": len(keys),
            "width128_mean_delta_vs_ar": float(np.mean(candidate_values)),
            "matched_width64_mean_delta_vs_ar": float(np.mean(baseline_values)),
            "paired_mean_width128_minus_width64": float(np.mean(paired)),
            "paired_median_width128_minus_width64": float(np.median(paired)),
            "width128_wins": int(np.sum(paired > 0.0)),
            "width64_wins": int(np.sum(paired < 0.0)),
            "ties": int(np.sum(paired == 0.0)),
            "by_target": by_target,
        },
        "selected_recipe": dict(plan["initial_recipe"]),
        "selection_kind": "retain_only_complete_validated_recipe_under_no_harm",
        "claims": {
            "hidden_width_128_rejected": True,
            "hidden_width_256_or_512_compared": False,
            "learning_rate_alternatives_compared": False,
            "weight_decay_alternatives_compared": False,
            "residual_cap_alternatives_compared": False,
            "global_numeric_optimum_claimed": False,
        },
        "selected_representation": "fixed_pca512",
        "selected_head_family": "frozen_ar_plus_causal_temporal_residual",
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    resolution["resolution_sha256"] = digest_json(resolution)
    return resolution


def write_training_recipe_resolution(path: Path, resolution: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(resolution))


__all__ = ["resolve_stopped_training_recipe", "write_training_recipe_resolution"]
