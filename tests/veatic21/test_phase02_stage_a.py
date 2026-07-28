from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from neural_bridge.veatic21.contracts import PHASE02_REGISTRATION_ROOT
from neural_bridge.veatic21.data import load_json
from neural_bridge.veatic21.phase02_features import (
    build_causal_history,
    build_feature_matrix,
    feature_names,
    standardize_from_owner,
)
from neural_bridge.veatic21.phase02_metrics import (
    binary_ranking_and_probability_metrics_fast,
    binary_ranking_metrics,
    probability_metrics,
)
from neural_bridge.veatic21.phase02_stage_a import (
    _logistic_screen,
    _ridge_screen,
    enumerate_stage_a_work_units,
)
from neural_bridge.veatic21.phase02_stage_a_executor import (
    ExecutorConfiguration,
    deterministic_pair_shards,
    pair_stage_a_units,
)


def _history(values: tuple[float, ...], max_depth: int = 3):
    rows = len(values)
    return build_causal_history(
        np.asarray(values, dtype=np.float32),
        np.zeros(rows, dtype=np.int16),
        np.arange(rows, dtype=np.int32),
        max_depth=max_depth,
    )


def test_causal_history_never_uses_future_values() -> None:
    before = _history((1.0, 2.0, 3.0, 4.0))
    after = _history((1.0, 2.0, 30.0, 40.0))

    for name in before.__dataclass_fields__:
        left = getattr(before, name)
        right = getattr(after, name)
        assert np.array_equal(left[:2], right[:2]), name


def test_cold_start_padding_is_explicit_and_retains_rows() -> None:
    history = _history((1.0, 2.0, 3.0, 4.0))
    assert np.array_equal(history.levels[0], [1.0, 1.0, 1.0, 1.0])
    assert np.array_equal(history.available[0], [1.0, 0.0, 0.0, 0.0])
    assert np.array_equal(history.available[2], [1.0, 1.0, 1.0, 0.0])


def test_every_registered_feature_form_matches_its_schema() -> None:
    history = _history((1.0, 2.0, 3.0, 4.0))
    forms = (
        "current_only",
        "raw_levels_with_availability_mask",
        "level_and_first_difference",
        "causal_rolling_summary",
        "combined_levels_differences_summaries",
        "raw_sequence_with_availability_mask",
    )
    for form in forms:
        matrix = build_feature_matrix(history, form, 3)
        assert matrix.shape == (4, len(feature_names(form, 3)))
        assert np.isfinite(matrix).all()


def test_standardization_is_owned_by_training_rows() -> None:
    features = np.asarray([[0.0], [2.0], [100.0]], dtype=np.float32)
    transformed, mean, std = standardize_from_owner(features, np.asarray([1, 1, 0], dtype=bool))
    assert np.allclose(mean, [1.0])
    assert np.allclose(std, [1.0])
    assert np.allclose(transformed[:2], [[-1.0], [1.0]])


def test_stage_a_registry_is_the_complete_frozen_matrix() -> None:
    registration = load_json(PHASE02_REGISTRATION_ROOT / "experiment-registration.json")
    splits = load_json(PHASE02_REGISTRATION_ROOT / "split-registry.json")
    units = enumerate_stage_a_work_units(registration, splits)
    assert len(units) == 40_824
    assert len({unit.unit_id for unit in units}) == len(units)
    assert {unit.model_family for unit in units} == {
        "continuous_ridge",
        "event_logistic_l2",
    }


def test_executor_pair_sharding_is_disjoint_complete_and_deterministic() -> None:
    registration = load_json(PHASE02_REGISTRATION_ROOT / "experiment-registration.json")
    splits = load_json(PHASE02_REGISTRATION_ROOT / "split-registry.json")
    units = enumerate_stage_a_work_units(registration, splits)[:48]
    pairs = pair_stage_a_units(units)
    first = deterministic_pair_shards(pairs, 4)
    second = deterministic_pair_shards(pairs, 4)

    assert first == second
    flattened = [unit for shard in first for pair in shard for unit in pair]
    assert len(flattened) == len(units)
    assert {unit.unit_id for unit in flattened} == {unit.unit_id for unit in units}
    for shard in first:
        for ridge, logistic in shard:
            assert logistic.sequence == ridge.sequence + 1
            assert ridge.model_family == "continuous_ridge"
            assert logistic.model_family == "event_logistic_l2"


def test_executor_configuration_bounds_total_gpu_concurrency() -> None:
    valid = ExecutorConfiguration(
        id="two-process-four-stream",
        mlx_lanes=2,
        gpu_streams_per_lane=2,
        metric_workers_per_lane=2,
        pair_cache=True,
        compiled_ridge_update_blocks=True,
        compiled_logistic_update_blocks=False,
        fast_metrics=True,
        pipeline_depth=4,
    )
    valid.validate()

    with pytest.raises(ValueError, match="total concurrent GPU streams"):
        ExecutorConfiguration(
            id="oversubscribed",
            mlx_lanes=4,
            gpu_streams_per_lane=4,
            metric_workers_per_lane=1,
            pair_cache=True,
            compiled_ridge_update_blocks=True,
            compiled_logistic_update_blocks=False,
            fast_metrics=True,
            pipeline_depth=4,
        ).validate()


def test_mlx_ridge_and_logistic_screens_fit_all_regularizers() -> None:
    rng = np.random.default_rng(7)
    rows = 96
    features = rng.normal(size=(rows, 3)).astype(np.float32)
    standardized, _, _ = standardize_from_owner(features, np.arange(rows) < 64)
    x = np.column_stack([standardized, np.ones(rows, dtype=np.float32)])
    train = np.zeros((rows, 2), dtype=bool)
    validation = np.zeros((rows, 2), dtype=bool)
    train[:64] = True
    validation[64:] = True
    continuous = np.vstack([features[:, 0] + 0.1 * rng.normal(size=rows), -features[:, 1]]).astype(
        np.float32
    )
    labels = (np.quantile(continuous[:, :64], 0.9, axis=1)[None, :] <= continuous.T).astype(
        np.float32
    )

    ridge, ridge_solver = _ridge_screen(x, continuous, train, validation)
    logistic, logistic_solver = _logistic_screen(x, labels, train, validation)

    assert ridge.shape == logistic.shape == (rows, 10, 2)
    assert np.isnan(ridge[:64]).all() and np.isfinite(ridge[64:]).all()
    assert np.isnan(logistic[:64]).all() and np.isfinite(logistic[64:]).all()
    assert np.all((logistic[64:] >= 0.0) & (logistic[64:] <= 1.0))
    assert ridge_solver["backend"] == "mlx_gpu_primal_conjugate_gradient"
    assert logistic_solver["backend"] == "mlx_gpu_full_batch_accelerated_gradient"


def test_compiled_solver_blocks_match_reference_solver_dispositions() -> None:
    rng = np.random.default_rng(91)
    rows = 128
    x = np.column_stack(
        [rng.normal(size=(rows, 4)).astype(np.float32), np.ones(rows, dtype=np.float32)]
    )
    train = np.zeros((rows, 3), dtype=bool)
    validation = np.zeros((rows, 3), dtype=bool)
    train[:88] = True
    validation[88:] = True
    continuous = rng.normal(size=(3, rows)).astype(np.float32)
    labels = (np.quantile(continuous[:, :88], 0.9, axis=1) <= continuous.T).astype(np.float32)

    ridge_reference, ridge_reference_solver = _ridge_screen(x, continuous, train, validation)
    ridge_compiled, ridge_compiled_solver = _ridge_screen(
        x, continuous, train, validation, compiled_update_blocks=True
    )
    logistic_reference, logistic_reference_solver = _logistic_screen(x, labels, train, validation)
    logistic_compiled, logistic_compiled_solver = _logistic_screen(
        x, labels, train, validation, compiled_update_blocks=True
    )

    assert ridge_reference_solver["converged_mask"] == ridge_compiled_solver["converged_mask"]
    assert logistic_reference_solver["converged_mask"] == logistic_compiled_solver["converged_mask"]
    assert np.allclose(ridge_reference, ridge_compiled, equal_nan=True, atol=1e-6, rtol=1e-6)
    assert np.allclose(logistic_reference, logistic_compiled, equal_nan=True, atol=1e-6, rtol=1e-6)


def test_fast_stage_a_metrics_match_sklearn_reference_with_ties() -> None:
    rng = np.random.default_rng(412)
    for rows in (31, 257, 4_801):
        labels = rng.integers(0, 2, size=rows, dtype=np.uint8)
        labels[0] = 0
        labels[1] = 1
        scores = np.round(rng.normal(size=rows), decimals=2)
        reference = {
            **binary_ranking_metrics(labels, scores),
            **probability_metrics(labels, 1 / (1 + np.exp(-scores))),
        }
        probability = 1 / (1 + np.exp(-scores))
        fast = binary_ranking_and_probability_metrics_fast(labels, probability, probability=True)
        probability_reference = {
            **binary_ranking_metrics(labels, probability),
            **probability_metrics(labels, probability),
        }
        for name in ("raw_pr_auc", "roc_auc", "brier"):
            assert np.isclose(
                cast(float, fast[name]),
                cast(float, probability_reference[name]),
                atol=1e-12,
                rtol=0,
            )
        ranking_fast = binary_ranking_and_probability_metrics_fast(
            labels, scores, probability=False
        )
        for name in ("raw_pr_auc", "roc_auc"):
            assert np.isclose(
                cast(float, ranking_fast[name]),
                cast(float, reference[name]),
                atol=1e-12,
                rtol=0,
            )
