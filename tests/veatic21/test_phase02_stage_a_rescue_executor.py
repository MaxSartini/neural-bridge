from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from neural_bridge.veatic21.phase02_features import CausalHistory
from neural_bridge.veatic21.phase02_stage_a import StageAInputs, StageAPrepared
from neural_bridge.veatic21.phase02_stage_a_rescue import (
    _cell_batches,
    _solve_logistic_batch,
    _solve_ridge_batch,
    load_rescue_registry,
)
from neural_bridge.veatic21.phase02_stage_a_rescue_backtest import (
    _representative_selection,
    _selection_coverage,
)
from neural_bridge.veatic21.phase02_stage_a_rescue_backtest_verify import (
    EXPECTED_STAGE_COUNTS,
    verify_rescue_executor_backtest,
)
from neural_bridge.veatic21.phase02_stage_a_rescue_executor import (
    RescueExecutorConfiguration,
    deterministic_weighted_shards,
    rescue_unit_work_weight,
)
from neural_bridge.veatic21.phase02_stage_a_rescue_saturated import (
    RESCUE_MAIN_ROOT,
    verify_selected_rescue_executor,
)


def _synthetic_prepared() -> StageAPrepared:
    generator = np.random.default_rng(20260729)
    rows = 96
    features = generator.normal(size=(rows, 5)).astype(np.float32)
    features[:, -1] = 1
    masks = np.ones((rows, 3), dtype=bool)
    masks[:8, 1] = False
    masks[-12:, 2] = False
    labels = (generator.uniform(size=(rows, 3)) > 0.7).astype(np.float32)
    return StageAPrepared(
        train_masks=masks,
        validation_masks=masks,
        split_digest="synthetic",
        raw_features=features[:, :-1],
        mean=np.zeros(4, dtype=np.float32),
        std=np.ones(4, dtype=np.float32),
        x=features,
        thresholds=np.zeros(3, dtype=np.float32),
        labels=labels,
        preparation_seconds=0,
    )


def _synthetic_cells():
    base = load_rescue_registry()[0].cells[0]
    return tuple(
        replace(
            base,
            rescue_sequence=index,
            rescue_cell_identity_sha256=f"cell-{index}",
            target_index=index,
            regularization_index=index,
            regularization_value=(index + 1) * 1e-4,
            original_base_budget=8,
            original_maximum_budget=8,
            rescue_maximum_budget=32,
            convergence_tolerance=0.2,
        )
        for index in range(3)
    )


def test_sparse_ridge_batching_preserves_dispositions_and_values() -> None:
    prepared = _synthetic_prepared()
    cells = _synthetic_cells()
    values = np.stack(
        [np.linspace(index, index + 1, len(prepared.x), dtype=np.float32) for index in range(3)]
    )
    inputs = StageAInputs(
        arousal=np.zeros(len(prepared.x), dtype=np.float32),
        video_id=np.zeros(len(prepared.x), dtype=np.int16),
        row_index=np.arange(len(prepared.x), dtype=np.int32),
        time_seconds=np.arange(len(prepared.x), dtype=np.float64) / 2,
        active_values=values,
        active_masks=prepared.train_masks.T,
        candidate_ids=("s01_e01", "s01_e02", "s01_e03"),
        target_ends=(1, 2, 3),
        history=CausalHistory(
            levels=np.empty((0, 0), dtype=np.float32),
            available=np.empty((0, 0), dtype=bool),
            deltas=np.empty((0, 0), dtype=np.float32),
            rolling_mean=np.empty((0, 0), dtype=np.float32),
            rolling_std=np.empty((0, 0), dtype=np.float32),
            rolling_min=np.empty((0, 0), dtype=np.float32),
            rolling_max=np.empty((0, 0), dtype=np.float32),
            rolling_slope=np.empty((0, 0), dtype=np.float32),
            rolling_fraction=np.empty((0, 0), dtype=np.float32),
        ),
        grouped_splits=(),
        blocked_splits=(),
    )
    reference_predictions = []
    reference_records = []
    for cell in cells:
        prediction, record = _solve_ridge_batch(
            inputs, prepared, (cell,), compiled_update_blocks=False
        )
        reference_predictions.append(prediction[:, 0])
        reference_records.append(record[0])
    batched_predictions, batched_records = _solve_ridge_batch(
        inputs, prepared, cells, compiled_update_blocks=False
    )
    np.testing.assert_allclose(
        batched_predictions,
        np.column_stack(reference_predictions),
        rtol=0,
        atol=2e-6,
    )
    assert [record["converged"] for record in batched_records] == [
        record["converged"] for record in reference_records
    ]
    assert [record["iterations"] for record in batched_records] == [
        record["iterations"] for record in reference_records
    ]


def test_sparse_logistic_batching_and_compilation_preserve_values() -> None:
    prepared = _synthetic_prepared()
    cells = _synthetic_cells()
    reference, reference_records = _solve_logistic_batch(
        prepared, cells, compiled_update_blocks=False
    )
    compiled, compiled_records = _solve_logistic_batch(
        prepared, cells, compiled_update_blocks=True
    )
    np.testing.assert_allclose(compiled, reference, rtol=0, atol=2e-6)
    assert [record["converged"] for record in compiled_records] == [
        record["converged"] for record in reference_records
    ]
    assert [record["iterations"] for record in compiled_records] == [
        record["iterations"] for record in reference_records
    ]
    np.testing.assert_allclose(
        [record["final_diagnostic"] for record in compiled_records],
        [record["final_diagnostic"] for record in reference_records],
        rtol=0,
        atol=1e-7,
    )


def test_cell_batches_cover_each_cell_once() -> None:
    cells = _synthetic_cells()
    assert _cell_batches(cells, 2) == (cells[:2], cells[2:])
    with pytest.raises(ValueError, match="positive"):
        _cell_batches(cells, 0)


def test_weighted_shards_are_disjoint_complete_and_balanced() -> None:
    units = load_rescue_registry()[:24]
    shards = deterministic_weighted_shards(units, 3)
    flattened = [unit.rescue_unit_sequence for shard in shards for unit in shard]
    assert set(flattened) == {unit.rescue_unit_sequence for unit in units}
    assert len(flattened) == len(set(flattened))
    loads = [sum(rescue_unit_work_weight(unit) for unit in shard) for shard in shards]
    largest = max(rescue_unit_work_weight(unit) for unit in units)
    assert max(loads) - min(loads) <= largest


def test_executor_configuration_rejects_unsafe_stream_oversubscription() -> None:
    configuration = RescueExecutorConfiguration(
        id="unsafe",
        mlx_lanes=4,
        gpu_streams_per_lane=4,
        metric_workers_per_lane=1,
        cell_batch_size=16,
        compiled_update_blocks=False,
        fast_metrics=True,
        pipeline_depth=4,
    )
    with pytest.raises(ValueError, match="Metal streams"):
        configuration.validate()


def test_representative_selection_covers_every_registered_workload_axis() -> None:
    units = load_rescue_registry()
    selection = _representative_selection(units, 192, minimum_logistic=32)
    coverage = _selection_coverage(units, selection)
    assert coverage["model_families"] == ["continuous_ridge", "event_logistic_l2"]
    assert coverage["protocols"] == ["blocked", "grouped"]
    assert len(cast(list[str], coverage["feature_forms"])) == 6
    assert coverage["cell_count_bands"] == ["1", "17-41", "2-4", "5-8", "9-16"]
    assert coverage["candidate_ids"] == [f"s01_e{index:02d}" for index in range(1, 22)]
    assert coverage["regularization_indices"] == [0, 1, 2, 3, 4, 5, 6, 9]
    assert cast(int, coverage["logistic_units"]) >= 32


def test_completed_rescue_executor_backtest_passes_independent_verification() -> None:
    verification = verify_rescue_executor_backtest(write_verification=False)
    assert verification["status"] == "PASS"
    assert verification["candidate_count"] == sum(EXPECTED_STAGE_COUNTS.values()) == 79
    assert verification["outer_test_scores_opened"] is False
    assert verification["cortical_values_opened"] is False
    assert verification["aggregation_or_pruning_performed"] is False


def test_selected_rescue_executor_freezes_complete_main_identity() -> None:
    selected = verify_selected_rescue_executor()
    assert selected["status"] == "PASS"
    assert selected["main_rescue_units"] == 14_465
    assert selected["main_rescue_cells"] == 113_392
    assert selected["main_selection"] == {
        "start_inclusive": 0,
        "stop_exclusive": 14_465,
        "count": 14_465,
        "sequence_sha256": "54d9f376cdbddef97ce28f6faba25f9d1394d652fd5e37cc3b962e90f8319711",
    }
    assert selected["main_request_sha256"] == (
        "22f1dd5547b405fba1d62430ef5a1102a934895fe16c0f416a61416c998c1253"
    )
    assert str(RESCUE_MAIN_ROOT).endswith("stage-a-convergence-rescue/main-hardware-saturated")
