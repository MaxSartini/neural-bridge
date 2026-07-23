from __future__ import annotations

import numpy as np

from neural_bridge.veatic21.evidence import digest_json
from neural_bridge.veatic21.supervised_projection import (
    build_supervised_projection_screen,
    causal_context_indices,
    select_supervised_projection,
)


def _seal(record: dict, field: str) -> dict:
    record[field] = digest_json(record)
    return record


def test_causal_context_indices_never_cross_video_or_use_future_rows() -> None:
    video_id = np.asarray(["0", "0", "0", "1", "1", "1"])
    row_index = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int32)

    indices, available = causal_context_indices(video_id, row_index, context_rows=(1, 2))

    np.testing.assert_array_equal(indices[0], [0, 0])
    np.testing.assert_array_equal(indices[2], [1, 0])
    np.testing.assert_array_equal(indices[3], [3, 3])
    np.testing.assert_array_equal(indices[5], [4, 3])
    np.testing.assert_array_equal(available[0], [0.0, 0.0])
    np.testing.assert_array_equal(available[5], [1.0, 1.0])
    assert np.all(indices <= np.arange(len(indices))[:, None])
    for position in range(len(indices)):
        assert np.all(video_id[indices[position]] == video_id[position])


def test_supervised_projection_screen_is_matched_and_single_worker() -> None:
    plan = _seal(
        {
            "artifacts": {"pca_manifest_sha256": "pca"},
            "purpose": "spike_discovery",
        },
        "plan_sha256",
    )
    pca_manifest = _seal({"manifest": "fixture"}, "manifest_sha256")
    plan["artifacts"]["pca_manifest_sha256"] = pca_manifest["manifest_sha256"]
    plan.pop("plan_sha256")
    plan["plan_sha256"] = digest_json(plan)
    pca_screen = _seal(
        {
            "artifacts": {"stage1_plan_sha256": plan["plan_sha256"]},
            "fixed_nuisance_recipe": {
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
    pca_summary = _seal({"screen_sha256": pca_screen["screen_sha256"]}, "summary_sha256")
    pca_selection = _seal(
        {
            "selected_pca_width": 512,
            "summary_sha256": pca_summary["summary_sha256"],
        },
        "selection_sha256",
    )
    capacity = {
        "backend": "mlx",
        "worker_count": 1,
        "memory_fraction_cap": None,
        "source_width": 20_484,
        "projection_width": 512,
        "context_count": 5,
        "hidden_width": 64,
        "measurements": [],
        "selected_batch_rows": 4096,
        "selection_rule": "fixture",
    }

    screen = build_supervised_projection_screen(
        pca_selection,
        pca_summary,
        pca_screen,
        plan,
        pca_manifest,
        capacity,
    )

    assert screen["matrix"]["lanes"] == [
        "fixed_pca512",
        "supervised_bottleneck512",
    ]
    assert screen["matrix"]["expected_cells"] == 8
    assert screen["matrix"]["worker_count"] == 1
    assert screen["matched_recipe"]["batch_rows"] == 4096
    assert screen["matched_recipe"]["pca_and_supervised_lanes_use_identical_recipe"]
    assert screen["benchmark_test_labels_accessed"] is False


def test_supervised_projection_selection_keeps_pca_on_nonpositive_pairing() -> None:
    screen = _seal(
        {
            "matrix": {"targets": ["target"]},
            "benchmark_test_labels_accessed": False,
        },
        "screen_sha256",
    )
    records = []
    for seed in (1, 2):
        records.extend(
            [
                {
                    "target": "target",
                    "fold": 0,
                    "seed": seed,
                    "lane": "fixed_pca512",
                    "inner_average_precision_skill_delta_vs_frozen_ar": 0.02,
                },
                {
                    "target": "target",
                    "fold": 0,
                    "seed": seed,
                    "lane": "supervised_bottleneck512",
                    "inner_average_precision_skill_delta_vs_frozen_ar": 0.01,
                },
            ]
        )
    summary = _seal(
        {
            "screen_sha256": screen["screen_sha256"],
            "completed_cells": 4,
            "expected_cells": 4,
            "records": records,
            "benchmark_test_labels_accessed": False,
        },
        "summary_sha256",
    )

    selection = select_supervised_projection(summary, screen)

    assert selection["selected_representation"] == "fixed_pca512"
    assert selection["pca512_wins"] == 2
    assert selection["supervised_wins"] == 0
    assert selection["paired_mean_supervised_minus_pca512"] == -0.01
