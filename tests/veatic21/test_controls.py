from __future__ import annotations

import numpy as np

from neural_bridge.veatic21.controls import (
    _lane_design_and_targets,
    build_control_plan,
)
from neural_bridge.veatic21.evidence import digest_json


def _seal(record: dict, field: str) -> dict:
    record[field] = digest_json(record)
    return record


def test_control_plan_registers_complete_matched_panel(tmp_path) -> None:
    preregistration = _seal({"fixture": "prereg"}, "preregistration_sha256")
    manifest = _seal({"fixture": "pca"}, "manifest_sha256")
    baseline = _seal(
        {
            "completed_cells": 2,
            "expected_cells": 2,
            "records": [
                {"lane": "fixed_pca512", "target": "target", "fold": 0, "seed": seed}
                for seed in (1, 2)
            ],
        },
        "summary_sha256",
    )
    recipe_plan = _seal(
        {
            "artifacts": {
                "pca_manifest_sha256": manifest["manifest_sha256"],
                "representation_summary_sha256": baseline["summary_sha256"],
            },
            "execution_plan": {"fixture": "execution"},
            "matrix": {
                "targets": ["target"],
                "folds": [0],
                "comparison_seeds": [1, 2],
                "cells_per_candidate": 2,
            },
        },
        "plan_sha256",
    )
    selection = _seal(
        {
            "plan_sha256": recipe_plan["plan_sha256"],
            "selected_recipe": {"hidden_width": 64},
        },
        "resolution_sha256",
    )
    crosswalk = tmp_path / "crosswalk.md"
    crosswalk.write_text("controls", encoding="utf-8")

    plan = build_control_plan(
        preregistration,
        recipe_plan,
        selection,
        baseline,
        manifest,
        crosswalk,
    )

    assert plan["matrix"]["expected_new_cells"] == 12
    assert len(plan["matrix"]["matched_control_lanes"]) == 5
    assert plan["matrix"]["worker_count"] == 1
    assert plan["stability_must_not_resume_before_all_gates_pass"] is True


def test_control_transforms_are_deterministic_and_label_permutation_preserves_counts() -> None:
    projected = np.arange(24, dtype=np.float32).reshape(6, 4)
    diagnostics = np.arange(18, dtype=np.float32).reshape(6, 3)
    binary = np.asarray([0, 1, 1, 0, 0, 1], dtype=np.int8)
    video_id = np.asarray(["1", "1", "1", "2", "2", "2"])
    row_index = np.asarray([0, 1, 2, 0, 1, 2])
    args = (projected, diagnostics, binary, video_id, row_index, (1,), 7)

    random_a, _ = _lane_design_and_targets("random_pca_residual", *args)
    random_b, _ = _lane_design_and_targets("random_pca_residual", *args)
    _, permuted = _lane_design_and_targets("label_permutation_residual", *args)

    assert np.array_equal(random_a, random_b)
    for video in ("1", "2"):
        mask = video_id == video
        assert int(permuted[mask].sum()) == int(binary[mask].sum())
    assert not np.array_equal(permuted, binary)
