from __future__ import annotations

from copy import deepcopy
from typing import cast

from neural_bridge.veatic21.contracts import (
    CURRENT_STATE,
    PHASE02_EXECUTOR_BACKTEST_REGISTRATION,
)
from neural_bridge.veatic21.data import load_json, sha256_file
from neural_bridge.veatic21.phase02_stage_a_backtest import (
    _compare_stage_a_unit,
    _select_candidate,
)
from neural_bridge.veatic21.phase02_stage_a_executor import _configuration_from_dict


def test_backtest_freeze_reaches_cpu_and_gpu_concurrency_ceiling() -> None:
    registration = load_json(PHASE02_EXECUTOR_BACKTEST_REGISTRATION)
    configurations = cast(list[dict[str, object]], registration["candidate_configurations"])
    total_gpu_concurrency = {
        cast(int, item["mlx_lanes"]) * cast(int, item["gpu_streams_per_lane"])
        for item in configurations
    }
    single_queue_metric_workers = {
        cast(int, item["metric_workers_per_lane"])
        for item in configurations
        if item["id"] != "reference_1p1s_1m"
        and item["mlx_lanes"] == 1
        and item["gpu_streams_per_lane"] == 1
    }

    assert {1, 2, 3, 4, 6, 8, 12} <= total_gpu_concurrency
    assert single_queue_metric_workers == {1, 4, 8, 12}
    for value in configurations:
        _configuration_from_dict(value).validate()
    assert sha256_file(PHASE02_EXECUTOR_BACKTEST_REGISTRATION) in CURRENT_STATE.read_text()


def _unit() -> dict[str, object]:
    return {
        "unit": {"unit_id": "00000_test"},
        "split_sha256": "split",
        "feature_names": ["x"],
        "feature_count": 1,
        "feature_matrix_sha256": "features",
        "scaler_sha256": "scaler",
        "target_thresholds_sha256": "thresholds",
        "train_row_counts": [10],
        "validation_row_counts": [5],
        "configuration_count": 1,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "records": [
            {
                "configuration_id": "config",
                "status": "completed",
                "disposition": "eligible_for_inner_aggregation",
                "converged": True,
                "rows": 5,
                "raw_pr_auc": 0.5,
                "roc_auc": 0.6,
                "brier": None,
            }
        ],
        "solver": {
            "backend": "mlx",
            "iterations": 8,
            "converged_mask": [[True]],
            "relative_residual_by_cell": [[1e-4]],
        },
    }


def _stats() -> dict[str, object]:
    return {
        "mismatch_count": 0,
        "mismatch_examples": [],
        "max_metric_absolute_difference": 0.0,
        "max_solver_absolute_difference": 0.0,
    }


def test_equivalence_comparator_ignores_compilation_provenance_within_tolerance() -> None:
    reference = _unit()
    candidate = deepcopy(reference)
    solver = cast(dict[str, object], candidate["solver"])
    solver["compiled_update_blocks"] = True
    residuals = cast(list[list[float]], solver["relative_residual_by_cell"])
    residuals[0][0] += 5e-8
    records = cast(list[dict[str, object]], candidate["records"])
    records[0]["raw_pr_auc"] = cast(float, records[0]["raw_pr_auc"]) + 5e-13
    stats = _stats()

    _compare_stage_a_unit(
        reference,
        candidate,
        metric_tolerance=1e-12,
        solver_tolerance=1e-7,
        stats=stats,
    )

    assert stats["mismatch_count"] == 0


def test_equivalence_comparator_rejects_metric_drift() -> None:
    reference = _unit()
    candidate = deepcopy(reference)
    records = cast(list[dict[str, object]], candidate["records"])
    records[0]["roc_auc"] = cast(float, records[0]["roc_auc"]) + 2e-12
    stats = _stats()

    _compare_stage_a_unit(
        reference,
        candidate,
        metric_tolerance=1e-12,
        solver_tolerance=1e-7,
        stats=stats,
    )

    assert stats["mismatch_count"] == 1


def test_selection_applies_three_percent_fewer_queue_tie_rule() -> None:
    base = {
        "eligible_without_saturation_gate": True,
        "gpu_utilization_mean_percent": 80.0,
        "peak_summed_mlx_active_memory_bytes_by_repetition": [100],
    }
    two_queue: dict[str, object] = {
        **base,
        "configuration": {
            "id": "two",
            "mlx_lanes": 1,
            "gpu_streams_per_lane": 2,
        },
        "median_work_units_per_second": 98.0,
    }
    four_queue: dict[str, object] = {
        **base,
        "configuration": {
            "id": "four",
            "mlx_lanes": 2,
            "gpu_streams_per_lane": 2,
        },
        "median_work_units_per_second": 100.0,
    }

    result = _select_candidate([two_queue, four_queue])

    assert result["status"] == "PASS"
    assert result["selected_configuration_id"] == "two"
    assert result["tie_within_three_percent_applied"] is True
