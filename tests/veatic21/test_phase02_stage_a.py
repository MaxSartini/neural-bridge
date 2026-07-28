from __future__ import annotations

import numpy as np

from neural_bridge.veatic21.contracts import PHASE02_REGISTRATION_ROOT
from neural_bridge.veatic21.data import load_json
from neural_bridge.veatic21.phase02_features import (
    build_causal_history,
    build_feature_matrix,
    feature_names,
    standardize_from_owner,
)
from neural_bridge.veatic21.phase02_stage_a import (
    _logistic_screen,
    _ridge_screen,
    enumerate_stage_a_work_units,
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


def test_mlx_ridge_and_logistic_screens_fit_all_regularizers() -> None:
    rng = np.random.default_rng(7)
    rows = 96
    features = rng.normal(size=(rows, 3)).astype(np.float32)
    standardized, _, _ = standardize_from_owner(
        features, np.arange(rows) < 64
    )
    x = np.column_stack([standardized, np.ones(rows, dtype=np.float32)])
    train = np.zeros((rows, 2), dtype=bool)
    validation = np.zeros((rows, 2), dtype=bool)
    train[:64] = True
    validation[64:] = True
    continuous = np.vstack(
        [features[:, 0] + 0.1 * rng.normal(size=rows), -features[:, 1]]
    ).astype(np.float32)
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
