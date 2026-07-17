from __future__ import annotations

import math

import numpy as np
import pytest

from backend.scripts import veatic21_evaluation as evaluation


def _metric_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_rows: list[float] = []
    primary_rows: list[float] = []
    control_rows: list[float] = []
    videos: list[str] = []
    for video in range(30):
        target = np.linspace(-1.0, 1.0, 20, dtype=np.float64) + (video % 3) * 0.001
        target_rows.extend(target.tolist())
        primary_rows.extend((target + 0.002 * np.sin(np.arange(20))).tolist())
        control_rows.extend((-target + 0.002 * np.cos(np.arange(20))).tolist())
        videos.extend([f"video-{video:02d}"] * len(target))
    return (
        np.asarray(target_rows),
        np.asarray(primary_rows),
        np.asarray(control_rows),
        np.asarray(videos),
    )


def test_train_threshold_and_metric_triple_are_explicit_and_correct() -> None:
    training = np.arange(100, dtype=np.float64)
    threshold = evaluation.train_q90_threshold(training)
    assert threshold == pytest.approx(89.1)

    y_true = np.asarray([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
    scores = evaluation.score_end_state_metrics(
        y_true=y_true,
        prediction=y_true,
        event_threshold=0.75,
    )
    assert scores.spearman == pytest.approx(1.0)
    assert scores.train_q90_pr_auc == pytest.approx(1.0)
    assert scores.event_threshold == 0.75
    assert scores.event_prevalence == pytest.approx(2 / 6)
    assert scores.top_5pct_lift == pytest.approx(2.0 - np.mean(y_true))


def test_paired_whole_video_bootstrap_is_deterministic_and_strong() -> None:
    target, primary, control, videos = _metric_fixture()
    first = evaluation.paired_whole_video_bootstrap(
        y_true=target,
        primary_prediction=primary,
        control_prediction=control,
        video_ids=videos,
        event_threshold=0.6,
        resamples=250,
        seed=17,
    )
    second = evaluation.paired_whole_video_bootstrap(
        y_true=target,
        primary_prediction=primary,
        control_prediction=control,
        video_ids=videos,
        event_threshold=0.6,
        resamples=250,
        seed=17,
    )
    assert first == second
    assert first.video_count == 30
    assert set(first.metrics) == set(evaluation.END_STATE_METRICS)
    for summary in first.metrics.values():
        assert summary.point_delta > 0
        assert summary.lower_95_one_sided > 0
        assert summary.positive_fraction == 1.0


def test_metric_and_bootstrap_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="both classes"):
        evaluation.score_end_state_metrics(
            y_true=np.arange(10, dtype=float),
            prediction=np.arange(10, dtype=float),
            event_threshold=100.0,
        )
    with pytest.raises(ValueError, match="constant"):
        evaluation.score_end_state_metrics(
            y_true=np.arange(10, dtype=float),
            prediction=np.ones(10),
            event_threshold=5.0,
        )
    with pytest.raises(ValueError, match="row aligned"):
        evaluation.paired_whole_video_bootstrap(
            y_true=np.arange(10, dtype=float),
            primary_prediction=np.arange(10, dtype=float),
            control_prediction=np.arange(10, dtype=float),
            video_ids=np.asarray(["one"] * 9),
            event_threshold=5.0,
            resamples=100,
        )


def test_contribution_cap_matches_again_definition_and_fails_without_gain() -> None:
    even = np.full(15, 0.01)
    assert evaluation.max_positive_contribution(even) == pytest.approx(1 / 15)
    assert evaluation.contribution_cap_gate(even).passed

    concentrated = np.asarray([1.0] + [0.01] * 14)
    result = evaluation.contribution_cap_gate(concentrated)
    assert not result.passed
    assert result.diagnostics["max_positive_contribution"] > 0.25

    assert math.isinf(evaluation.max_positive_contribution(np.asarray([-1.0, 0.0])))
    assert not evaluation.contribution_cap_gate(np.asarray([-1.0, 0.0])).passed


def test_grouped_fold_seed_gate_enforces_exact_50_row_scope_and_thresholds() -> None:
    folds = np.repeat(np.arange(5), 10)
    seeds = np.tile(np.arange(10), 5)
    passing = evaluation.grouped_fold_seed_gates(
        delta_vs_ar=np.full(50, 0.0035),
        delta_vs_best_control=np.full(50, 0.0032),
        fold_ids=folds,
        seed_ids=seeds,
    )
    assert passing.passed
    assert passing.diagnostics["wins_vs_best_control"] == 50

    controls = np.full(50, 0.004)
    controls[:11] = -0.001
    inconsistent = evaluation.grouped_fold_seed_gates(
        delta_vs_ar=np.full(50, 0.004),
        delta_vs_best_control=controls,
        fold_ids=folds,
        seed_ids=seeds,
    )
    assert not inconsistent.passed
    assert "positive_vs_best_control_at_least_40_of_50" in inconsistent.failed_gates

    duplicate_seeds = seeds.copy()
    duplicate_seeds[-1] = duplicate_seeds[-2]
    bad_scope = evaluation.grouped_fold_seed_gates(
        delta_vs_ar=np.full(50, 0.004),
        delta_vs_best_control=np.full(50, 0.004),
        fold_ids=folds,
        seed_ids=duplicate_seeds,
    )
    assert not bad_scope.passed
    assert "exact_5_fold_x_10_seed_scope" in bad_scope.failed_gates


def _member_scope() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    folds: list[int] = []
    seeds: list[int] = []
    triples: list[int] = []
    for fold in range(5):
        for triple in range(3):
            for seed in range(triple * 3, triple * 3 + 3):
                folds.append(fold)
                triples.append(triple)
                seeds.append(seed)
    return np.asarray(folds), np.asarray(seeds), np.asarray(triples)


def _ensemble_scope() -> tuple[np.ndarray, np.ndarray]:
    return np.repeat(np.arange(5), 3), np.tile(np.arange(3), 5)


def test_fold_seed_triple_scope_is_exact_and_seed_membership_is_stable() -> None:
    folds, seeds, triples = _member_scope()
    result = evaluation.grouped_fold_seed_triple_scope(
        fold_ids=folds, seed_ids=seeds, triple_ids=triples
    )
    assert result.passed
    assert result.diagnostics["triple_to_seeds"] == {
        "0": ["0", "1", "2"],
        "1": ["3", "4", "5"],
        "2": ["6", "7", "8"],
    }

    corrupted = seeds.copy()
    corrupted[-1] = 0
    failed = evaluation.grouped_fold_seed_triple_scope(
        fold_ids=folds, seed_ids=corrupted, triple_ids=triples
    )
    assert not failed.passed
    assert "each_fold_triple_contains_its_same_three_seeds" in failed.failed_gates
    assert "seed_membership_is_disjoint_across_triples" in failed.failed_gates


def test_grouped_continuous_triple_gates_match_phase7_thresholds() -> None:
    folds, triples = _ensemble_scope()
    passing = evaluation.grouped_continuous_triple_gates(
        spearman_delta_vs_ar=np.full(15, 0.0030),
        spearman_delta_vs_best_control=np.full(15, 0.0025),
        top5_delta_vs_ar=np.full(15, 0.0015),
        top5_delta_vs_best_control=np.full(15, 0.0012),
        fold_ids=folds,
        triple_ids=triples,
    )
    assert passing.passed
    assert passing.diagnostics["wins_spearman_vs_ar"] == 15

    only_eleven_wins = np.full(15, 0.003)
    only_eleven_wins[:4] = -0.0001
    failed_wins = evaluation.grouped_continuous_triple_gates(
        spearman_delta_vs_ar=only_eleven_wins,
        spearman_delta_vs_best_control=np.full(15, 0.0025),
        top5_delta_vs_ar=np.full(15, 0.0015),
        top5_delta_vs_best_control=np.full(15, 0.0012),
        fold_ids=folds,
        triple_ids=triples,
    )
    assert not failed_wins.passed
    assert "spearman_positive_vs_ar_at_least_12_of_15" in failed_wins.failed_gates

    concentrated = np.full(15, 0.0001)
    concentrated[0] = 0.1
    failed_cap = evaluation.grouped_continuous_triple_gates(
        spearman_delta_vs_ar=concentrated,
        spearman_delta_vs_best_control=np.full(15, 0.0025),
        top5_delta_vs_ar=np.full(15, 0.0015),
        top5_delta_vs_best_control=np.full(15, 0.0012),
        fold_ids=folds,
        triple_ids=triples,
    )
    assert not failed_cap.passed
    assert (
        "single_fold_triple_spearman_contribution_at_most_0_25"
        in failed_cap.failed_gates
    )


def test_grouped_event_triple_gates_require_005_and_15_of_15() -> None:
    folds, triples = _ensemble_scope()
    passing = evaluation.grouped_event_triple_gates(
        pr_auc_delta_vs_ar=np.full(15, 0.006),
        pr_auc_delta_vs_best_control=np.full(15, 0.0055),
        fold_ids=folds,
        triple_ids=triples,
    )
    assert passing.passed

    one_loss = np.full(15, 0.006)
    one_loss[0] = -0.0001
    failed = evaluation.grouped_event_triple_gates(
        pr_auc_delta_vs_ar=one_loss,
        pr_auc_delta_vs_best_control=np.full(15, 0.006),
        fold_ids=folds,
        triple_ids=triples,
    )
    assert not failed.passed
    assert "positive_vs_ar_15_of_15" in failed.failed_gates


def _valence_fixture() -> tuple[np.ndarray, ...]:
    true_rise: list[float] = []
    true_drop: list[float] = []
    primary_rise: list[float] = []
    primary_drop: list[float] = []
    control_rise: list[float] = []
    control_drop: list[float] = []
    videos: list[str] = []
    folds: list[int] = []
    for video in range(25):
        for row in range(9):
            if row == 8:  # a genuine no-direction tie, excluded explicitly
                rise, drop = 0.0, 0.0
                p_rise, p_drop = 0.5, 0.5
            elif row % 2 == 0:
                rise, drop = 1.0, 0.0
                p_rise, p_drop = 0.9, 0.1
            else:
                rise, drop = 0.0, 1.0
                p_rise, p_drop = 0.1, 0.9
            true_rise.append(rise)
            true_drop.append(drop)
            primary_rise.append(p_rise)
            primary_drop.append(p_drop)
            control_rise.append(0.6)  # predicts rise for every directional row
            control_drop.append(0.4)
            videos.append(f"video-{video:02d}")
            folds.append(video % 5)
    return tuple(
        np.asarray(values)
        for values in (
            true_rise,
            true_drop,
            primary_rise,
            primary_drop,
            control_rise,
            control_drop,
            videos,
            folds,
        )
    )


def test_valence_direction_is_paired_fold_consistent_and_bootstrap_positive() -> None:
    (
        true_rise,
        true_drop,
        primary_rise,
        primary_drop,
        control_rise,
        control_drop,
        videos,
        folds,
    ) = _valence_fixture()
    result = evaluation.evaluate_valence_direction(
        true_rise_magnitude=true_rise,
        true_drop_magnitude=true_drop,
        primary_rise_prediction=primary_rise,
        primary_drop_prediction=primary_drop,
        control_rise_prediction=control_rise,
        control_drop_prediction=control_drop,
        video_ids=videos,
        fold_ids=folds,
        resamples=250,
        seed=91,
    )
    assert result.passed
    assert result.primary.balanced_accuracy == 1.0
    assert result.primary.macro_f1 == 1.0
    assert result.neutral_truth_rows_excluded == 25
    assert result.directional_rows == 200
    assert result.fold_wins == {"balanced_accuracy": 5, "macro_f1": 5}
    assert result.bootstrap["balanced_accuracy"].lower_95_one_sided > 0
    assert result.bootstrap["macro_f1"].lower_95_one_sided > 0


def test_valence_direction_derivation_preserves_ties_and_rejects_invalid_truth() -> None:
    directions = evaluation.derive_valence_direction(
        np.asarray([1.0, 0.0, 0.5]), np.asarray([0.0, 1.0, 0.5])
    )
    np.testing.assert_array_equal(directions, np.asarray([1, -1, 0], dtype=np.int8))
    with pytest.raises(ValueError, match="non-negative"):
        evaluation.derive_valence_direction(
            np.asarray([1.0, -0.1]),
            np.asarray([0.0, 1.0]),
            require_nonnegative=True,
        )
