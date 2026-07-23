from __future__ import annotations

from neural_bridge.veatic21.evidence import digest_json
from neural_bridge.veatic21.head_training import (
    build_head_family_screen,
    select_head_family,
)


def _seal(record: dict, field: str) -> dict:
    record[field] = digest_json(record)
    return record


def test_head_screen_reuses_matched_causal_evidence() -> None:
    pca_manifest = _seal({"fixture": "pca"}, "manifest_sha256")
    plan = _seal(
        {"artifacts": {"pca_manifest_sha256": pca_manifest["manifest_sha256"]}},
        "plan_sha256",
    )
    representation_screen = _seal(
        {
            "matched_recipe": {
                "batch_rows": 4096,
                "context_rows": [1, 2, 4, 6, 10],
                "hidden_width": 64,
                "learning_rate": 3e-4,
                "weight_decay": 1e-4,
                "residual_logit_cap": 0.5,
                "minimum_epochs": 50,
                "plateau_patience": 50,
                "nonconvergence_patience": 400,
            },
            "matrix": {
                "targets": ["rare", "common"],
                "folds": [0],
                "comparison_seeds": [1, 2],
            },
        },
        "screen_sha256",
    )
    representation_summary = _seal(
        {"screen_sha256": representation_screen["screen_sha256"], "records": []},
        "summary_sha256",
    )
    representation_selection = _seal(
        {
            "selected_representation": "fixed_pca512",
            "summary_sha256": representation_summary["summary_sha256"],
        },
        "selection_sha256",
    )

    screen = build_head_family_screen(
        representation_selection,
        representation_summary,
        representation_screen,
        plan,
        pca_manifest,
    )

    assert screen["baseline_head"] == "frozen_ar_plus_causal_temporal_residual"
    assert screen["candidate_head"] == "frozen_ar_plus_gated_multiscale_temporal_residual"
    assert screen["matrix"]["expected_candidate_cells"] == 4
    assert screen["matrix"]["worker_count"] == 1
    assert screen["matched_recipe"]["batch_rows"] == 4096


def test_head_selection_uses_exact_paired_mean() -> None:
    screen = _seal({"fixture": "head"}, "screen_sha256")
    baseline_records = []
    gated_records = []
    for seed in (1, 2):
        baseline_records.append(
            {
                "lane": "fixed_pca512",
                "target": "target",
                "fold": 0,
                "seed": seed,
                "inner_average_precision_skill_delta_vs_frozen_ar": 0.01,
            }
        )
        gated_records.append(
            {
                "target": "target",
                "fold": 0,
                "seed": seed,
                "inner_average_precision_skill_delta_vs_frozen_ar": 0.02,
            }
        )
    baseline = _seal({"records": baseline_records}, "summary_sha256")
    gated = _seal(
        {
            "screen_sha256": screen["screen_sha256"],
            "completed_cells": 2,
            "expected_cells": 2,
            "records": gated_records,
        },
        "summary_sha256",
    )

    selection = select_head_family(gated, screen, baseline)

    assert selection["selected_head_family"] == (
        "frozen_ar_plus_gated_multiscale_temporal_residual"
    )
    assert selection["gated_wins"] == 2
    assert selection["paired_mean_gated_minus_causal"] == 0.01
