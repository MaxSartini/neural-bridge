from __future__ import annotations

from neural_bridge.veatic21.evidence import digest_json
from neural_bridge.veatic21.stability import build_stability_plan


def _seal(record: dict, field: str) -> dict:
    record[field] = digest_json(record)
    return record


def test_stability_plan_uses_only_fixed_stability_seeds() -> None:
    preregistration = _seal(
        {
            "training": {
                "stability_seed_panel": list(range(20_260_801, 20_260_810)),
                "checkpoint_ensembles": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            }
        },
        "preregistration_sha256",
    )
    manifest = _seal({"fixture": "pca"}, "manifest_sha256")
    execution = {
        "matrix": {"comparison_seeds": [1, 2, 3]},
        "plan_sha256": "old",
    }
    recipe_plan = _seal(
        {
            "artifacts": {"pca_manifest_sha256": manifest["manifest_sha256"]},
            "execution_plan": execution,
            "matrix": {"targets": ["target"], "folds": [0]},
        },
        "plan_sha256",
    )
    selection = _seal(
        {
            "plan_sha256": recipe_plan["plan_sha256"],
            "selected_recipe": {
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
            "selected_representation": "fixed_pca512",
            "selected_head_family": "frozen_ar_plus_causal_temporal_residual",
            "benchmark_test_labels_accessed": False,
        },
        "resolution_sha256",
    )

    plan = build_stability_plan(preregistration, recipe_plan, selection, manifest)

    assert plan["matrix"]["expected_cells"] == 9
    assert plan["matrix"]["worker_count"] == 1
    assert plan["execution_plan"]["matrix"]["comparison_seeds"] == list(
        range(20_260_801, 20_260_810)
    )
    assert plan["benchmark_test_labels_accessed"] is False
