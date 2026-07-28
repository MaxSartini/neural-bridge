from __future__ import annotations

import numpy as np

from neural_bridge.veatic21.ar import (
    blocked_split_cell,
    build_ar_features,
    common_history_mask,
    derive_lag_depths,
    fit_logistic_mlx,
    grouped_split_cell,
    predict_logistic_mlx,
    select_decision_threshold,
    spike_metrics,
)


def _toy_rows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    video_id = np.repeat(np.arange(10), 20)
    row_index = np.tile(np.arange(20), 10)
    valid = row_index < 18
    return video_id, row_index, valid


def test_lag_family_is_derived_from_veatic_pacf_and_target_width() -> None:
    assert derive_lag_depths(pacf_decay_lag=4, target_width=6) == (0, 1, 2, 4, 6)


def test_common_history_mask_requires_target_validity_and_full_causal_history() -> None:
    video_id, row_index, valid = _toy_rows()
    mask = common_history_mask(video_id, row_index, valid, max_depth=6)

    assert np.all(row_index[mask] >= 6)
    assert np.all(row_index[mask] < 18)
    assert int(mask.sum()) == 10 * 12


def test_ar_features_are_current_then_strictly_causal_lags() -> None:
    arousal = np.arange(12, dtype=np.float64)
    features = build_ar_features(arousal, np.asarray((4, 8)), depth=3)

    assert features.tolist() == [[4.0, 3.0, 2.0, 1.0], [8.0, 7.0, 6.0, 5.0]]


def test_grouped_split_has_disjoint_video_owned_outer_and_inner_partitions() -> None:
    video_id, row_index, valid = _toy_rows()
    eligible = np.flatnonzero(common_history_mask(video_id, row_index, valid, max_depth=2))
    cell = grouped_split_cell(
        eligible,
        video_id,
        fold=0,
        seed=42,
        test_fraction=0.30,
        validation_fraction=0.30,
    )

    assert len(set(video_id[cell.outer_train]) & set(video_id[cell.outer_test])) == 0
    assert len(set(video_id[cell.inner_train]) & set(video_id[cell.inner_validation])) == 0
    assert len(np.unique(video_id[cell.outer_test])) == 3
    assert np.array_equal(
        np.sort(np.concatenate((cell.inner_train, cell.inner_validation))),
        np.sort(cell.outer_train),
    )


def test_blocked_split_is_forward_time_within_every_video() -> None:
    video_id, row_index, valid = _toy_rows()
    eligible = np.flatnonzero(common_history_mask(video_id, row_index, valid, max_depth=2))
    cell = blocked_split_cell(
        eligible,
        video_id,
        seed=42,
        test_fraction=0.30,
        validation_fraction=0.30,
    )

    for video in np.unique(video_id):
        train_rows = row_index[cell.outer_train[video_id[cell.outer_train] == video]]
        test_rows = row_index[cell.outer_test[video_id[cell.outer_test] == video]]
        assert np.max(train_rows) < np.min(test_rows)
        inner_train_rows = row_index[cell.inner_train[video_id[cell.inner_train] == video]]
        validation_rows = row_index[cell.inner_validation[video_id[cell.inner_validation] == video]]
        assert np.max(inner_train_rows) < np.min(validation_rows)


def test_mlx_gpu_logistic_fit_is_finite_and_directionally_correct() -> None:
    features = np.arange(-4, 5, dtype=np.float64).reshape(-1, 1)
    labels = (features[:, 0] > 0).astype(np.int8)
    model = fit_logistic_mlx(features, labels, regularization=0.01)
    scores = predict_logistic_mlx(model, features)

    assert model.device == "gpu:0"
    assert model.converged
    assert np.isfinite(scores).all()
    assert np.all(np.diff(scores) > 0)


def test_training_owned_threshold_and_spike_metrics_cover_required_stack() -> None:
    labels = np.asarray((0, 0, 0, 1, 1, 1), dtype=np.int8)
    scores = np.asarray((0.1, 0.2, 0.3, 0.7, 0.8, 0.9))
    threshold = select_decision_threshold(labels, scores)
    metrics = spike_metrics(labels, scores, decision_threshold=threshold)

    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["f1"] == 1.0
    for fraction in (1, 5, 10):
        assert f"top_{fraction}pct_event_recall" in metrics
        assert f"top_{fraction}pct_event_lift" in metrics
