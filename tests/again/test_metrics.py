from __future__ import annotations

import math

import numpy as np
import pytest

from neural_bridge.again.metrics import (
    continuous_metrics,
    event_metrics,
    threshold_from_train,
    top_fraction_lift,
    top_recall,
)

# These are properties, not recorded outputs: each assertion states something that
# must hold for any valid input, so none of the expected values were copied out of
# a passing run. metrics.py produces every headline figure in results/README.md.


def _balanced_train(seed: int = 11, size: int = 200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=size)
    scores = rng.normal(size=size) + labels * 0.8
    return labels, scores


def test_decision_threshold_is_identical_across_completely_different_test_folds() -> None:
    train_y, train_scores = _balanced_train()
    thresholds = set()
    for seed in range(6):
        rng = np.random.default_rng(seed)
        test_y = rng.integers(0, 2, size=300)
        test_scores = rng.normal(size=300) * 5.0 + 20.0
        thresholds.add(
            event_metrics(train_y, train_scores, test_y, test_scores)[
                "decision_threshold_train_only"
            ]
        )
    assert len(thresholds) == 1


def test_threshold_falls_back_to_the_train_median_when_only_one_class_is_present() -> None:
    rng = np.random.default_rng(3)
    scores = rng.normal(size=64)
    assert threshold_from_train(np.zeros(64, dtype=int), scores) == float(np.median(scores))


def test_threshold_stays_inside_the_observed_train_score_range() -> None:
    train_y, train_scores = _balanced_train()
    threshold = threshold_from_train(train_y, train_scores)
    assert train_scores.min() <= threshold <= train_scores.max()


def test_top_recall_finds_every_positive_once_the_whole_ranking_is_kept() -> None:
    rng = np.random.default_rng(5)
    labels = rng.integers(0, 2, size=150)
    assert top_recall(labels, rng.normal(size=150), 1.0) == 1.0


def test_top_recall_never_decreases_as_the_kept_fraction_grows() -> None:
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 2, size=400)
    scores = rng.normal(size=400)
    recalls = [top_recall(labels, scores, frac) for frac in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)]
    assert all(a <= b + 1e-12 for a, b in zip(recalls, recalls[1:]))


def test_top_recall_is_nan_rather_than_zero_when_there_are_no_positives() -> None:
    rng = np.random.default_rng(9)
    assert math.isnan(top_recall(np.zeros(50, dtype=int), rng.normal(size=50), 0.05))


def test_tied_scores_make_top_recall_depend_on_row_order() -> None:
    # np.argsort(-scores) breaks ties by array position, so with ties spanning the
    # cut-off the answer depends on how rows happen to be ordered. Identical scores
    # and an identical positive count give different recalls purely from placement.
    # Pinned to make the exposure visible, not because it is desirable.
    scores = np.ones(4)
    assert top_recall(np.array([1, 0, 0, 0]), scores, 0.25) == 1.0
    assert top_recall(np.array([0, 0, 0, 1]), scores, 0.25) == 0.0


def test_spearman_survives_any_strictly_monotonic_rescaling_of_the_prediction() -> None:
    # The defining property of a rank correlation: it sees order, not magnitude.
    rng = np.random.default_rng(13)
    truth = rng.normal(size=250)
    prediction = truth * 0.6 + rng.normal(size=250) * 0.8
    baseline = continuous_metrics(truth, prediction)["spearman"]
    for transform in (np.exp, lambda v: v * 3.0 + 7.0, lambda v: v**3):
        assert continuous_metrics(truth, transform(prediction))["spearman"] == pytest.approx(
            baseline
        )


def test_pearson_does_not_survive_that_same_rescaling() -> None:
    # The contrast that makes the test above meaningful: if both were invariant,
    # the property would be measuring nothing.
    rng = np.random.default_rng(13)
    truth = rng.normal(size=250)
    prediction = truth * 0.6 + rng.normal(size=250) * 0.8
    baseline = continuous_metrics(truth, prediction)["pearson"]
    assert continuous_metrics(truth, np.exp(prediction))["pearson"] != pytest.approx(baseline)


def test_continuous_metrics_are_all_nan_on_empty_input() -> None:
    empty = np.array([], dtype=float)
    assert all(math.isnan(value) for value in continuous_metrics(empty, empty).values())


def test_pearson_is_nan_when_a_side_has_no_variance() -> None:
    rng = np.random.default_rng(17)
    assert math.isnan(continuous_metrics(np.ones(30), rng.normal(size=30))["pearson"])


def test_lift_over_the_whole_population_is_zero() -> None:
    rng = np.random.default_rng(19)
    values = rng.normal(size=300)
    # Selecting everything means comparing the mean against itself.
    assert top_fraction_lift(values, rng.normal(size=300), 1.0) == pytest.approx(0.0, abs=1e-9)


def test_lift_is_positive_when_scores_rank_the_values_correctly() -> None:
    rng = np.random.default_rng(23)
    values = rng.normal(size=300)
    assert top_fraction_lift(values, values, 0.05) > 0.0


def test_perfectly_separable_scores_earn_a_perfect_pr_auc_and_roc_auc() -> None:
    train_y, train_scores = _balanced_train()
    rng = np.random.default_rng(29)
    test_y = np.concatenate([np.zeros(60, dtype=int), np.ones(60, dtype=int)])
    test_scores = np.concatenate([rng.uniform(0.0, 0.4, 60), rng.uniform(0.6, 1.0, 60)])
    metrics = event_metrics(train_y, train_scores, test_y, test_scores)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_predicted_positive_count_agrees_with_the_threshold_it_reports() -> None:
    train_y, train_scores = _balanced_train()
    rng = np.random.default_rng(31)
    test_scores = rng.normal(size=200)
    metrics = event_metrics(train_y, train_scores, rng.integers(0, 2, size=200), test_scores)
    threshold = metrics["decision_threshold_train_only"]
    assert metrics["predicted_positive_count"] == int((test_scores >= threshold).sum())


def test_single_class_test_fold_nans_the_ranking_metrics_but_not_the_thresholded_ones() -> None:
    # Asymmetry in event_metrics: pr_auc/roc_auc/balanced_accuracy are guarded by
    # `two_classes`, while f1/precision/recall/accuracy fall through to the
    # zero_division=0 path and return real numbers. Pinned so a change is visible;
    # whether the split is intended is a separate decision.
    train_y, train_scores = _balanced_train()
    rng = np.random.default_rng(37)
    metrics = event_metrics(train_y, train_scores, np.zeros(40, dtype=int), rng.normal(size=40))
    assert math.isnan(metrics["pr_auc"])
    assert math.isnan(metrics["roc_auc"])
    assert math.isnan(metrics["balanced_accuracy"])
    assert not math.isnan(metrics["f1"])
    assert not math.isnan(metrics["accuracy"])
