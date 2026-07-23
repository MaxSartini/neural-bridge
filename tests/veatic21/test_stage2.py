from __future__ import annotations

from neural_bridge.veatic21.evidence import digest_json
from neural_bridge.veatic21.stage2 import (
    build_stage2_pca_screen,
    select_stage2_pca_width,
    select_train_only_target_shortlist,
)


def _artifacts():
    targets = [
        {"name": "q950_short", "quantile": 0.95},
        {"name": "q950_long", "quantile": 0.95},
        {"name": "q800_short", "quantile": 0.80},
        {"name": "q800_long", "quantile": 0.80},
    ]
    plan = {
        "schema": "veatic21_stage1_child_plan_v2",
        "purpose": "spike_discovery",
        "artifacts": {"ar_benchmark_sha256": "pending"},
        "capacity": {
            "safe_batch_rows_by_head_hidden_width": {
                "frozen_ar_plus_causal_temporal_residual:64": 512
            }
        },
        "checkpoint_policy": {
            "minimum_epochs_before_termination": 50,
            "plateau_patience_epochs": 50,
            "nonconvergence_patience_epochs": 400,
        },
        "matrix": {
            "targets": targets,
            "folds": [{"fold": 0, "candidate_pca_widths": [64, 128]}],
            "comparison_seeds": [1, 2],
        },
    }
    scores = {
        "q950_short": [0.20, 0.18],
        "q950_long": [0.20, 0.10],
        "q800_short": [0.30, 0.28],
        "q800_long": [0.31, 0.29],
    }
    records = []
    for target, values in scores.items():
        for seed, value in zip((1, 2), values, strict=True):
            records.append(
                {
                    "target": target,
                    "fold": 0,
                    "seed": seed,
                    "status": "complete",
                    "fresh_ar_average_precision_skill": value,
                    "event_prevalence": 0.1,
                }
            )
    summary = {
        "expected_cells": 8,
        "completed_cells": 8,
        "invalid_cells": 0,
        "records": records,
    }
    summary["summary_sha256"] = digest_json(summary)
    plan["artifacts"]["ar_benchmark_sha256"] = summary["summary_sha256"]
    plan["plan_sha256"] = digest_json(plan)
    return summary, plan


def test_train_only_shortlist_selects_one_stable_target_per_quantile() -> None:
    summary, plan = _artifacts()

    shortlist = select_train_only_target_shortlist(summary, plan)

    assert shortlist["selected_targets"] == ["q950_short", "q800_long"]
    assert shortlist["sealed_tail_labels_used"] is False
    assert len(shortlist["quantile_groups"]) == 2


def test_stage2_pca_screen_binds_shortlist_axes_and_executor_recipe() -> None:
    summary, plan = _artifacts()
    executor_request = {
        "benchmark_test_labels_accessed": False,
        "promotable": False,
        "config": {
            "head_family": "frozen_ar_plus_causal_temporal_residual",
            "hidden_width": 64,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "residual_logit_cap": 0.5,
            "batch_rows": 256,
            "context_rows": [1, 2, 4, 6, 10],
            "minimum_epochs": 50,
            "plateau_patience": 50,
            "nonconvergence_patience": 400,
        },
    }

    screen = build_stage2_pca_screen(
        summary,
        plan,
        executor_request,
        executor_request_sha256="executor-request",
    )

    assert screen["matrix"]["targets"] == ["q950_short", "q800_long"]
    assert screen["matrix"]["expected_cells"] == 8
    assert screen["fixed_nuisance_recipe"]["source_score_used_for_selection"] is False
    assert screen["benchmark_test_labels_accessed"] is False
    payload = dict(screen)
    expected = payload.pop("screen_sha256")
    assert digest_json(payload) == expected


def test_stage2_pca_width_selection_uses_completed_mean_delta() -> None:
    summary, plan = _artifacts()
    executor_request = {
        "benchmark_test_labels_accessed": False,
        "promotable": False,
        "config": {
            "head_family": "frozen_ar_plus_causal_temporal_residual",
            "hidden_width": 64,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "residual_logit_cap": 0.5,
            "batch_rows": 256,
            "context_rows": [1, 2, 4, 6, 10],
            "minimum_epochs": 50,
            "plateau_patience": 50,
            "nonconvergence_patience": 400,
        },
    }
    screen = build_stage2_pca_screen(
        summary,
        plan,
        executor_request,
        executor_request_sha256="executor-request",
    )
    records = []
    for target in screen["matrix"]["targets"]:
        for fold in screen["matrix"]["folds"]:
            for seed in screen["matrix"]["comparison_seeds"]:
                for width in screen["matrix"]["pca_widths"]:
                    records.append(
                        {
                            "target": target,
                            "fold": fold,
                            "seed": seed,
                            "pca_width": width,
                            "inner_average_precision_skill_delta_vs_frozen_ar": (
                                0.01 if width == 128 else 0.001
                            ),
                            "whole_fold_seed_uses_residual": True,
                        }
                    )
    result_summary = {
        "screen_sha256": screen["screen_sha256"],
        "completed_cells": len(records),
        "expected_cells": len(records),
        "records": records,
        "benchmark_test_labels_accessed": False,
    }
    result_summary["summary_sha256"] = digest_json(result_summary)

    selection = select_stage2_pca_width(result_summary, screen)

    assert selection["selected_pca_width"] == 128
    assert selection["tie_break_invoked"] is False
    payload = dict(selection)
    expected = payload.pop("selection_sha256")
    assert digest_json(payload) == expected
