from __future__ import annotations

from pathlib import Path

from neural_bridge.veatic21.evidence import digest_json, sha256_file
from neural_bridge.veatic21.training_recipe import build_training_recipe_plan


def _seal(record: dict, field: str) -> dict:
    record[field] = digest_json(record)
    return record


def test_recipe_plan_is_staged_and_reuses_the_verified_baseline() -> None:
    manifest = _seal({"fixture": "pca"}, "manifest_sha256")
    stage1 = _seal(
        {
            "purpose": "spike_discovery",
            "schema": "veatic21_stage1_child_plan_v2",
            "artifacts": {"pca_manifest_sha256": manifest["manifest_sha256"]},
            "capacity": {"safe_batch_rows_by_head_hidden_width": {}},
        },
        "plan_sha256",
    )
    representation_summary = _seal(
        {
            "completed_cells": 1,
            "expected_cells": 1,
            "benchmark_test_labels_accessed": False,
            "records": [
                {
                    "lane": "fixed_pca512",
                    "target": "target",
                    "fold": 0,
                    "seed": 1,
                    "inner_average_precision_skill_delta_vs_frozen_ar": 0.01,
                    "whole_fold_seed_uses_residual": True,
                    "best_epoch": 2,
                    "cell_directory": "cell",
                    "cell_metrics_sha256": "metrics",
                }
            ],
        },
        "summary_sha256",
    )
    representation_selection = _seal(
        {
            "selected_representation": "fixed_pca512",
            "summary_sha256": representation_summary["summary_sha256"],
        },
        "selection_sha256",
    )
    head_screen = _seal(
        {
            "matched_recipe": {
                "hidden_width": 64,
                "learning_rate": 3e-4,
                "weight_decay": 1e-4,
                "residual_logit_cap": 0.5,
                "batch_rows": 4096,
                "context_rows": [1, 2, 4, 6, 10],
                "minimum_epochs": 50,
                "plateau_patience": 50,
                "nonconvergence_patience": 400,
            },
            "matrix": {"targets": ["target"], "folds": [0], "comparison_seeds": [1]},
        },
        "screen_sha256",
    )
    head_selection = _seal(
        {
            "screen_sha256": head_screen["screen_sha256"],
            "selected_head_family": "frozen_ar_plus_causal_temporal_residual",
        },
        "selection_sha256",
    )

    plan = build_training_recipe_plan(
        stage1,
        representation_summary,
        representation_selection,
        head_screen,
        head_selection,
        manifest,
    )

    assert [stage["axis"] for stage in plan["matrix"]["stages"]] == [
        "hidden_width",
        "learning_rate",
        "weight_decay",
        "residual_logit_cap",
    ]
    assert plan["matrix"]["expected_new_cells"] == 9
    assert plan["matrix"]["worker_count"] == 1
    assert plan["execution_plan"]["capacity"]["memory_fraction_cap"] is None
    assert plan["artifacts"]["training_recipe_code_sha256"] == sha256_file(
        Path(__file__).parents[2] / "src/neural_bridge/veatic21/training_recipe.py"
    )
    payload = dict(plan)
    expected = payload.pop("plan_sha256")
    assert digest_json(payload) == expected
