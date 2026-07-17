from __future__ import annotations

import numpy as np
import pytest

from backend.scripts import veatic21_distilled_program as program


def test_all_video_panel_and_grouped_folds_are_deterministic_and_video_disjoint() -> None:
    row_counts = {str(index): 20 + (index % 9) for index in range(40)}
    first = program.deterministic_video_panel(row_counts, reserved_count=0)
    second = program.deterministic_video_panel(row_counts, reserved_count=0)
    assert first == second
    assert len(first.development_videos) == 40
    assert len(first.reserved_videos) == 0
    assert set(first.development_videos).isdisjoint(first.reserved_videos)
    assert set(first.development_videos) | set(first.reserved_videos) == set(row_counts)

    folds = program.balanced_grouped_video_folds(
        row_counts, first.development_videos, fold_count=4
    )
    held_out = []
    for fold in folds:
        assert set(fold.train_videos).isdisjoint(fold.test_videos)
        assert set(fold.train_videos) | set(fold.test_videos) == set(first.development_videos)
        held_out.extend(fold.test_videos)
    assert sorted(held_out, key=int) == sorted(first.development_videos, key=int)
    assert max(fold.test_rows for fold in folds) - min(fold.test_rows for fold in folds) <= max(
        row_counts.values()
    )


def test_observed_history_resets_between_videos_and_uses_train_initialization() -> None:
    signal = np.asarray([1.0, 1.5, 2.0, 10.0, 11.0], dtype=np.float32)
    videos = np.asarray(["1", "1", "1", "2", "2"])
    times = np.asarray([0.0, 0.5, 1.0, 0.0, 0.5], dtype=np.float32)
    features, audit = program.observed_history_features(
        signal, videos, times, training_initial_value=-3.0
    )
    assert features.shape == (5, 13)
    assert features[0, 1:4].tolist() == [-3.0, -3.0, -3.0]
    assert features[1, 1] == 1.0
    assert features[3, 1:4].tolist() == [-3.0, -3.0, -3.0]
    assert audit["video_resets"] == 2
    assert audit["cross_video_state_carry"] == 0


def test_canonical_ar_history_uses_exact_seven_features_and_masks_video_prefixes() -> None:
    signal = np.asarray([1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14], dtype=np.float32)
    videos = np.asarray(["a"] * 6 + ["b"] * 5)
    features, valid, audit = program.canonical_ar_history_features(signal, videos)

    assert features.shape == (11, 7)
    assert valid.tolist() == [False, False, False, False, True, True, False, False, False, False, True]
    np.testing.assert_allclose(features[4], [5, 4, 3, 1, 1, 2, 4])
    np.testing.assert_allclose(features[10], [14, 13, 12, 10, 1, 2, 4])
    assert audit["feature_width"] == 7
    assert audit["full_context_required"] is True
    assert audit["cross_video_state_carry"] == 0


def _ar_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    train_x = rng.normal(size=(144, 13)).astype(np.float32)
    test_x = rng.normal(size=(30, 13)).astype(np.float32)
    target = (
        0.8 * train_x[:, 0]
        - 0.2 * train_x[:, 2]
        + 0.05 * rng.normal(size=len(train_x))
    ).astype(np.float32)
    videos = np.repeat(np.arange(12).astype(str), 12)
    valid = np.ones(len(target), dtype=bool)
    return train_x, test_x, target, videos, valid


def test_common_inner_ownership_is_deterministic_disjoint_and_row_bound() -> None:
    _, _, _, videos, _ = _ar_fixture()
    first = program.build_inner_video_ownership(
        videos, namespace="fold2|arousal|seed20260717"
    )
    second = program.build_inner_video_ownership(
        videos, namespace="fold2|arousal|seed20260717"
    )
    assert first == second
    assert set(first.inner_train_videos).isdisjoint(first.inner_validation_videos)
    assert set(first.inner_train_videos) | set(first.inner_validation_videos) == set(videos)
    train_mask, validation_mask = first.row_masks(videos)
    assert np.all(train_mask ^ validation_mask)
    with pytest.raises(ValueError, match="row inventory/order"):
        first.row_masks(videos[::-1])

    member = program.build_member_inner_video_ownership(
        videos,
        outer_fold=2,
        target_name="future_arousal_max_delta_rows_4_10",
        seed=20260717,
    )
    assert "outer_fold2" in member.namespace
    assert "target=future_arousal_max_delta_rows_4_10" in member.namespace
    assert "seed=20260717" in member.namespace
    scopes = program.build_ar_crossfit_video_folds(member)
    assert {video for scope in scopes for video in scope.prediction_videos} == set(
        member.inner_train_videos
    )
    assert all(set(scope.fit_videos).isdisjoint(scope.prediction_videos) for scope in scopes)
    assert all(
        set(scope.fit_videos).isdisjoint(member.inner_validation_videos) for scope in scopes
    )


def test_frozen_ar_selection_and_train_scores_do_not_depend_on_test_rows() -> None:
    train_x, test_x, target, videos, valid = _ar_fixture()
    ownership = program.build_inner_video_ownership(
        videos, namespace="fold1|arousal|seed20260716"
    )
    first = program.fit_frozen_ar_train_only(
        train_x=train_x,
        test_x=test_x,
        train_target=target,
        train_valid=valid,
        train_video_ids=videos,
        ownership=ownership,
    )
    second = program.fit_frozen_ar_train_only(
        train_x=train_x,
        test_x=test_x * 1_000.0,
        train_target=target,
        train_valid=valid,
        train_video_ids=videos,
        ownership=ownership,
    )
    assert first.selected_alpha == second.selected_alpha
    np.testing.assert_allclose(first.train_prediction, second.train_prediction, atol=1e-7)
    assert first.audit["ownership_digest"] == ownership.digest
    assert first.audit["inner_validation_refit_before_residual_checkpoint_selection"] is False
    assert first.audit["residual_targets_use_final_outer_fit"] is False
    assert first.audit["all_outer_train_predictions_out_of_video_fit"] is True
    assert first.audit["model_family"] == "ridge_reference_contract_only"
    assert first.audit["promotable_veatic21_ar"] is False
    assert not np.allclose(first.test_prediction, second.test_prediction)


def test_inner_validation_targets_cannot_enter_ar_fits_used_for_residual_targets() -> None:
    train_x, test_x, target, videos, valid = _ar_fixture()
    ownership = program.build_inner_video_ownership(
        videos, namespace="fold3|valence_drop|seed20260718"
    )
    _, validation_mask = ownership.row_masks(videos)
    first = program.fit_frozen_ar_train_only(
        train_x=train_x,
        test_x=test_x,
        train_target=target,
        train_valid=valid,
        train_video_ids=videos,
        ownership=ownership,
        # A singleton grid isolates fit leakage from the legitimate act of
        # selecting a hyperparameter on inner validation.
        alpha_grid=(10.0,),
    )
    perturbed_target = target.copy()
    perturbed_target[validation_mask] = (
        10_000.0 + np.arange(np.count_nonzero(validation_mask), dtype=np.float32)
    )
    second = program.fit_frozen_ar_train_only(
        train_x=train_x,
        test_x=test_x,
        train_target=perturbed_target,
        train_valid=valid,
        train_video_ids=videos,
        ownership=ownership,
        alpha_grid=(10.0,),
    )

    # The selected inner model and every inner-train cross-fit model are
    # byte-identical: inner-validation labels never entered those fits.
    np.testing.assert_array_equal(first.inner_mean, second.inner_mean)
    np.testing.assert_array_equal(first.inner_std, second.inner_std)
    np.testing.assert_array_equal(first.inner_coef, second.inner_coef)
    assert first.inner_intercept == second.inner_intercept
    assert first.audit["inner_fit_digest"] == second.audit["inner_fit_digest"]
    assert [row["fit_digest"] for row in first.audit["crossfit"]] == [
        row["fit_digest"] for row in second.audit["crossfit"]
    ]
    np.testing.assert_array_equal(first.train_prediction, second.train_prediction)

    # The final outer-test model is intentionally different because it is the
    # separate model that may use all outer-training labels after selection.
    assert first.audit["final_fit_digest"] != second.audit["final_fit_digest"]
    assert not np.allclose(first.test_prediction, second.test_prediction)


def test_inner_validation_features_cannot_enter_inner_or_crossfit_ar_parameters() -> None:
    train_x, test_x, target, videos, valid = _ar_fixture()
    ownership = program.build_inner_video_ownership(
        videos, namespace="fold4|valence_rise|seed20260716"
    )
    train_mask, validation_mask = ownership.row_masks(videos)
    first = program.fit_frozen_ar_train_only(
        train_x=train_x,
        test_x=test_x,
        train_target=target,
        train_valid=valid,
        train_video_ids=videos,
        ownership=ownership,
        alpha_grid=(1.0,),
    )
    perturbed_x = train_x.copy()
    rng = np.random.default_rng(99)
    perturbed_x[validation_mask] = rng.normal(
        loc=500.0, scale=200.0, size=perturbed_x[validation_mask].shape
    )
    second = program.fit_frozen_ar_train_only(
        train_x=perturbed_x,
        test_x=test_x,
        train_target=target,
        train_valid=valid,
        train_video_ids=videos,
        ownership=ownership,
        alpha_grid=(1.0,),
    )
    np.testing.assert_array_equal(first.inner_mean, second.inner_mean)
    np.testing.assert_array_equal(first.inner_std, second.inner_std)
    np.testing.assert_array_equal(first.inner_coef, second.inner_coef)
    assert first.inner_intercept == second.inner_intercept
    assert first.audit["inner_fit_digest"] == second.audit["inner_fit_digest"]
    assert [row["fit_digest"] for row in first.audit["crossfit"]] == [
        row["fit_digest"] for row in second.audit["crossfit"]
    ]
    np.testing.assert_array_equal(
        first.train_prediction[train_mask], second.train_prediction[train_mask]
    )
    assert not np.allclose(
        first.train_prediction[validation_mask], second.train_prediction[validation_mask]
    )


def test_all_alpha_candidate_fits_are_inner_train_only_even_if_selection_data_changes() -> None:
    train_x, test_x, target, videos, valid = _ar_fixture()
    ownership = program.build_inner_video_ownership(
        videos, namespace="fold1|valence_movement|seed20260718"
    )
    _, validation_mask = ownership.row_masks(videos)
    first = program.fit_frozen_ar_train_only(
        train_x=train_x,
        test_x=test_x,
        train_target=target,
        train_valid=valid,
        train_video_ids=videos,
        ownership=ownership,
    )
    changed_x = train_x.copy()
    changed_target = target.copy()
    changed_x[validation_mask] *= -80.0
    changed_target[validation_mask] *= 150.0
    second = program.fit_frozen_ar_train_only(
        train_x=changed_x,
        test_x=test_x,
        train_target=changed_target,
        train_valid=valid,
        train_video_ids=videos,
        ownership=ownership,
    )
    assert [row["fit_digest"] for row in first.audit["alpha_selection"]] == [
        row["fit_digest"] for row in second.audit["alpha_selection"]
    ]


def test_crossfit_audit_proves_prediction_videos_are_excluded_from_each_fit() -> None:
    train_x, test_x, target, videos, valid = _ar_fixture()
    ownership = program.build_inner_video_ownership(
        videos, namespace="fold2|arousal|seed20260718"
    )
    result = program.fit_frozen_ar_train_only(
        train_x=train_x,
        test_x=test_x,
        train_target=target,
        train_valid=valid,
        train_video_ids=videos,
        ownership=ownership,
    )
    predicted_videos: set[str] = set()
    for row in result.audit["crossfit"]:
        assert set(row["fit_videos"]).isdisjoint(row["prediction_videos"])
        predicted_videos.update(row["prediction_videos"])
        assert len(row["mean"]) == train_x.shape[1]
        assert len(row["std"]) == train_x.shape[1]
        assert len(row["coef"]) == train_x.shape[1]
    assert predicted_videos == set(ownership.inner_train_videos)
    assert result.audit["ownership"]["inner_validation_videos"] == list(
        ownership.inner_validation_videos
    )
    assert len(result.audit["inner_mean"]) == train_x.shape[1]
    assert len(result.audit["final_coef"]) == train_x.shape[1]


def test_target_fold_seed_frozen_prediction_identity_must_match_all_controls() -> None:
    _, _, _, videos, _ = _ar_fixture()
    ownership = program.build_member_inner_video_ownership(
        videos,
        outer_fold=4,
        target_name="future_valence_drop_magnitude_rows_4_10",
        seed=20260718,
    )
    train_prediction = np.linspace(-1.0, 1.0, len(videos), dtype=np.float32)
    test_prediction = np.linspace(-0.5, 0.5, 36, dtype=np.float32)
    identity = program.frozen_ar_prediction_identity(
        ownership=ownership,
        outer_fold=4,
        target_name="future_valence_drop_magnitude_rows_4_10",
        seed=20260718,
        model_family="neural_ar_mlp",
        checkpoint_digest="abc123",
        train_prediction=train_prediction,
        test_prediction=test_prediction,
    )
    assert program.require_shared_frozen_ar_identity(
        {
            "real_residual": identity,
            "shuffled_residual": dict(identity),
            "random_residual": dict(identity),
        }
    ) == identity["identity_digest"]

    changed = program.frozen_ar_prediction_identity(
        ownership=ownership,
        outer_fold=4,
        target_name="future_valence_drop_magnitude_rows_4_10",
        seed=20260718,
        model_family="neural_ar_mlp",
        checkpoint_digest="different-checkpoint",
        train_prediction=train_prediction,
        test_prediction=test_prediction,
    )
    with pytest.raises(RuntimeError, match="do not share"):
        program.require_shared_frozen_ar_identity(
            {"real_residual": identity, "random_residual": changed}
        )


def test_whole_video_reassignment_never_uses_identity_donors() -> None:
    videos = np.asarray(["1"] * 3 + ["2"] * 5 + ["3"] * 4)
    values = np.arange(len(videos), dtype=np.float32)[:, None]
    reassigned, mapping = program.whole_video_reassignment(
        values, videos, namespace="matched-control"
    )
    assert reassigned.shape == values.shape
    assert all(recipient != donor for recipient, donor in mapping.items())
    assert not np.array_equal(reassigned, values)


def test_score_prediction_uses_training_only_event_threshold() -> None:
    train = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    test = np.linspace(0.0, 1.0, 40, dtype=np.float32)
    score = test.copy()
    metrics = program.score_prediction(
        train_values=train,
        test_values=test,
        prediction=score,
        time_seconds=np.arange(len(test), dtype=np.float32) * 0.5,
    )
    assert metrics["pooled_continuous_spearman"] > 0.99
    assert metrics["training_q90_future_event_pr_auc"] > 0.99
    assert metrics["event_threshold_train_only"] == np.quantile(train, 0.90)
