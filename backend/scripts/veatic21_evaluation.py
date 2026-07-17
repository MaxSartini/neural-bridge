"""Fail-closed evaluation contracts for the VEATIC 2.1 end state.

The helpers in this module are intentionally dataset-neutral.  They receive
already aligned arrays and explicit video/fold/seed ownership; they never read
or write cache artifacts.  Event scoring also requires an explicit threshold
derived from training data so a held-out panel cannot silently choose its own
event definition.

The grouped gates preserve the thresholds used by the canonical AGAIN
confirmations:

* grouped fold/seed compatibility: mean delta >= 0.003 and at least 40/50
  positive fold-seed comparisons against the strongest matched control;
* grouped continuous three-checkpoint ensembles: Spearman delta >= 0.002,
  top-5 lift delta >= 0.001, at least 12/15 positive fold-triples, all five
  fold means and paired medians positive, and no fold-triple contributing more
  than 25% of the positive gain;
* grouped event three-checkpoint ensembles: PR-AUC delta >= 0.005, 15/15
  positive fold-triples, all five fold means and paired medians positive, and
  the same 25% contribution cap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score


SCHEMA_VERSION = "veatic21_evaluation_v1"

SPEARMAN = "spearman"
TOP5_LIFT = "top_5pct_lift"
TRAIN_Q90_PR_AUC = "train_q90_pr_auc"
END_STATE_METRICS = (SPEARMAN, TOP5_LIFT, TRAIN_Q90_PR_AUC)

TRAIN_EVENT_QUANTILE = 0.90
ONE_SIDED_ALPHA = 0.05

CANONICAL_FOLD_SEED_FOLDS = 5
CANONICAL_FOLD_SEED_SEEDS = 10
CANONICAL_FOLD_SEED_ROWS = 50
CANONICAL_FOLD_SEED_MIN_DELTA = 0.003
CANONICAL_FOLD_SEED_CONTROL_WINS = 40

CANONICAL_GROUPED_FOLDS = 5
CANONICAL_TRIPLES = 3
CANONICAL_TRIPLE_SIZE = 3
CANONICAL_FOLD_TRIPLES = 15
CANONICAL_CONTINUOUS_SPEARMAN_DELTA = 0.002
CANONICAL_CONTINUOUS_TOP5_DELTA = 0.001
CANONICAL_CONTINUOUS_WINS = 12
CANONICAL_EVENT_DELTA = 0.005
CANONICAL_EVENT_WINS = 15
CANONICAL_GROUPED_CONTRIBUTION_CAP = 0.25
CANONICAL_DIRECTION_FOLD_WINS = 4


@dataclass(frozen=True)
class MetricScores:
    """The three primary end-state metrics for one prediction vector."""

    spearman: float
    top_5pct_lift: float
    train_q90_pr_auc: float
    event_threshold: float
    event_prevalence: float

    def as_dict(self) -> dict[str, float]:
        return {
            SPEARMAN: self.spearman,
            TOP5_LIFT: self.top_5pct_lift,
            TRAIN_Q90_PR_AUC: self.train_q90_pr_auc,
        }


@dataclass(frozen=True)
class BootstrapDeltaSummary:
    """Paired bootstrap distribution summary for one higher-is-better metric."""

    point_delta: float
    lower_95_one_sided: float
    median_delta: float
    mean_delta: float
    positive_fraction: float


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Whole-video paired bootstrap result for the end-state metric triple."""

    metrics: Mapping[str, BootstrapDeltaSummary]
    resamples: int
    seed: int
    video_count: int
    event_threshold: float


@dataclass(frozen=True)
class GateResult:
    """A fail-closed gate result with executable checks and diagnostics."""

    passed: bool
    checks: Mapping[str, bool]
    failed_gates: tuple[str, ...]
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class DirectionScores:
    balanced_accuracy: float
    macro_f1: float


@dataclass(frozen=True)
class ValenceDirectionEvaluation:
    """Paired rise/drop direction comparison against one matched control."""

    primary: DirectionScores
    control: DirectionScores
    aggregate_deltas: Mapping[str, float]
    fold_deltas: Mapping[str, tuple[float, ...]]
    fold_wins: Mapping[str, int]
    bootstrap: Mapping[str, BootstrapDeltaSummary]
    directional_rows: int
    neutral_truth_rows_excluded: int
    video_count: int
    fold_count: int
    checks: Mapping[str, bool]
    passed: bool
    failed_gates: tuple[str, ...]


def _float_vector(name: str, values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _id_vector(name: str, values: Sequence[Any] | np.ndarray, rows: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != rows:
        raise ValueError(f"{name} must be one-dimensional and row aligned")
    normalized = np.asarray([str(value) for value in array], dtype=object)
    if any(not value for value in normalized):
        raise ValueError(f"{name} contains an empty identifier")
    return normalized


def _aligned_float_vectors(**values: Sequence[float] | np.ndarray) -> dict[str, np.ndarray]:
    arrays = {name: _float_vector(name, value) for name, value in values.items()}
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("Evaluation arrays are not row aligned")
    return arrays


def train_q90_threshold(train_values: Sequence[float] | np.ndarray) -> float:
    """Derive the event threshold from an explicit, already filtered train vector."""

    values = _float_vector("train_values", train_values)
    if len(values) < 2:
        raise ValueError("A train-q90 threshold requires at least two training rows")
    return float(np.quantile(values, TRAIN_EVENT_QUANTILE))


def _spearman(y_true: np.ndarray, prediction: np.ndarray) -> float:
    if np.unique(y_true).size < 2 or np.unique(prediction).size < 2:
        raise ValueError("Spearman is undefined for a constant target or prediction")
    true_rank = rankdata(y_true, method="average")
    prediction_rank = rankdata(prediction, method="average")
    value = float(np.corrcoef(true_rank, prediction_rank)[0, 1])
    if not math.isfinite(value):
        raise ValueError("Spearman evaluation produced a non-finite value")
    return value


def _top5_lift(y_true: np.ndarray, prediction: np.ndarray) -> float:
    count = max(1, int(math.ceil(len(y_true) * 0.05)))
    selected = np.argsort(-prediction, kind="mergesort")[:count]
    value = float(np.mean(y_true[selected]) - np.mean(y_true))
    if not math.isfinite(value):
        raise ValueError("Top-5 lift evaluation produced a non-finite value")
    return value


def score_end_state_metrics(
    *,
    y_true: Sequence[float] | np.ndarray,
    prediction: Sequence[float] | np.ndarray,
    event_threshold: float,
) -> MetricScores:
    """Score Spearman, top-5 lift, and PR-AUC at an explicit train threshold."""

    arrays = _aligned_float_vectors(y_true=y_true, prediction=prediction)
    target = arrays["y_true"]
    score = arrays["prediction"]
    threshold = float(event_threshold)
    if not math.isfinite(threshold):
        raise ValueError("event_threshold must be a finite train-derived value")
    events = target >= threshold
    if np.unique(events).size != 2:
        raise ValueError("Train-threshold event PR-AUC requires both classes in evaluation")
    pr_auc = float(average_precision_score(events, score))
    if not math.isfinite(pr_auc):
        raise ValueError("Event PR-AUC evaluation produced a non-finite value")
    return MetricScores(
        spearman=_spearman(target, score),
        top_5pct_lift=_top5_lift(target, score),
        train_q90_pr_auc=pr_auc,
        event_threshold=threshold,
        event_prevalence=float(np.mean(events)),
    )


def _video_blocks(video_ids: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    unique = np.asarray(sorted(set(video_ids.tolist())), dtype=object)
    if len(unique) < 2:
        raise ValueError("Whole-video bootstrap requires at least two videos")
    blocks = {
        str(video): np.flatnonzero(video_ids == video).astype(np.int64) for video in unique
    }
    if any(not len(indices) for indices in blocks.values()):
        raise RuntimeError("Whole-video bootstrap constructed an empty video block")
    return unique, blocks


def _bootstrap_summary(point_delta: float, deltas: np.ndarray) -> BootstrapDeltaSummary:
    return BootstrapDeltaSummary(
        point_delta=float(point_delta),
        lower_95_one_sided=float(np.quantile(deltas, ONE_SIDED_ALPHA)),
        median_delta=float(np.median(deltas)),
        mean_delta=float(np.mean(deltas)),
        positive_fraction=float(np.mean(deltas > 0)),
    )


def paired_whole_video_bootstrap(
    *,
    y_true: Sequence[float] | np.ndarray,
    primary_prediction: Sequence[float] | np.ndarray,
    control_prediction: Sequence[float] | np.ndarray,
    video_ids: Sequence[Any] | np.ndarray,
    event_threshold: float,
    resamples: int = 5_000,
    seed: int = 20260716,
) -> PairedBootstrapResult:
    """Bootstrap paired primary-minus-control deltas by resampling whole videos.

    Rows from a sampled video are always taken together.  The same sampled
    video multiset is used for primary and control (and for all three metrics)
    in each iteration.  Undefined replicates fail closed rather than being
    silently dropped or replaced.
    """

    arrays = _aligned_float_vectors(
        y_true=y_true,
        primary_prediction=primary_prediction,
        control_prediction=control_prediction,
    )
    target = arrays["y_true"]
    primary = arrays["primary_prediction"]
    control = arrays["control_prediction"]
    videos = _id_vector("video_ids", video_ids, len(target))
    if int(resamples) != resamples or int(resamples) < 100:
        raise ValueError("Whole-video bootstrap requires an integer resamples >= 100")
    threshold = float(event_threshold)
    primary_point = score_end_state_metrics(
        y_true=target, prediction=primary, event_threshold=threshold
    ).as_dict()
    control_point = score_end_state_metrics(
        y_true=target, prediction=control, event_threshold=threshold
    ).as_dict()
    point_delta = {
        metric: float(primary_point[metric] - control_point[metric])
        for metric in END_STATE_METRICS
    }
    unique_videos, blocks = _video_blocks(videos)
    distributions = {
        metric: np.empty(int(resamples), dtype=np.float64) for metric in END_STATE_METRICS
    }
    rng = np.random.default_rng(int(seed))
    for iteration in range(int(resamples)):
        sampled = rng.choice(unique_videos, size=len(unique_videos), replace=True)
        take = np.concatenate([blocks[str(video)] for video in sampled])
        primary_scores = score_end_state_metrics(
            y_true=target[take], prediction=primary[take], event_threshold=threshold
        ).as_dict()
        control_scores = score_end_state_metrics(
            y_true=target[take], prediction=control[take], event_threshold=threshold
        ).as_dict()
        for metric in END_STATE_METRICS:
            distributions[metric][iteration] = (
                primary_scores[metric] - control_scores[metric]
            )
    return PairedBootstrapResult(
        metrics={
            metric: _bootstrap_summary(point_delta[metric], distributions[metric])
            for metric in END_STATE_METRICS
        },
        resamples=int(resamples),
        seed=int(seed),
        video_count=int(len(unique_videos)),
        event_threshold=threshold,
    )


def max_positive_contribution(deltas: Sequence[float] | np.ndarray) -> float:
    """Return the largest positive contribution divided by total positive gain.

    This matches the canonical AGAIN definition.  A vector with no positive
    gain returns infinity and therefore fails every finite contribution cap.
    """

    values = _float_vector("deltas", deltas)
    positive = values[values > 0]
    total = float(np.sum(positive))
    return float(np.max(positive) / total) if total > 0 else math.inf


def contribution_cap_gate(
    deltas: Sequence[float] | np.ndarray,
    *,
    cap: float = CANONICAL_GROUPED_CONTRIBUTION_CAP,
) -> GateResult:
    cap_value = float(cap)
    if not math.isfinite(cap_value) or not 0 < cap_value <= 1:
        raise ValueError("Contribution cap must be finite and in (0, 1]")
    contribution = max_positive_contribution(deltas)
    checks = {"max_positive_contribution_at_or_below_cap": contribution <= cap_value}
    failed = tuple(name for name, passed in checks.items() if not passed)
    return GateResult(
        passed=not failed,
        checks=checks,
        failed_gates=failed,
        diagnostics={"max_positive_contribution": contribution, "cap": cap_value},
    )


def _gate_result(checks: Mapping[str, bool], diagnostics: Mapping[str, Any]) -> GateResult:
    normalized = {str(name): bool(value) for name, value in checks.items()}
    failed = tuple(name for name, passed in normalized.items() if not passed)
    return GateResult(
        passed=not failed,
        checks=normalized,
        failed_gates=failed,
        diagnostics=dict(diagnostics),
    )


def grouped_fold_seed_gates(
    *,
    delta_vs_ar: Sequence[float] | np.ndarray,
    delta_vs_best_control: Sequence[float] | np.ndarray,
    fold_ids: Sequence[Any] | np.ndarray,
    seed_ids: Sequence[Any] | np.ndarray,
) -> GateResult:
    """Apply the canonical 5-fold x 10-seed grouped compatibility gates."""

    arrays = _aligned_float_vectors(
        delta_vs_ar=delta_vs_ar, delta_vs_best_control=delta_vs_best_control
    )
    ar = arrays["delta_vs_ar"]
    control = arrays["delta_vs_best_control"]
    folds = _id_vector("fold_ids", fold_ids, len(ar))
    seeds = _id_vector("seed_ids", seed_ids, len(ar))
    fold_set = sorted(set(folds.tolist()))
    seed_set = sorted(set(seeds.tolist()))
    keys = list(zip(folds.tolist(), seeds.tolist(), strict=True))
    expected_pairs = {(fold, seed) for fold in fold_set for seed in seed_set}
    actual_pairs = set(keys)
    exact_scope = bool(
        len(ar) == CANONICAL_FOLD_SEED_ROWS
        and len(fold_set) == CANONICAL_FOLD_SEED_FOLDS
        and len(seed_set) == CANONICAL_FOLD_SEED_SEEDS
        and len(actual_pairs) == len(keys)
        and actual_pairs == expected_pairs
    )
    wins_ar = int(np.count_nonzero(ar > 0))
    wins_control = int(np.count_nonzero(control > 0))
    checks = {
        "exact_5_fold_x_10_seed_scope": exact_scope,
        "mean_delta_vs_ar_at_least_0_003": float(np.mean(ar))
        >= CANONICAL_FOLD_SEED_MIN_DELTA,
        "mean_delta_vs_best_control_at_least_0_003": float(np.mean(control))
        >= CANONICAL_FOLD_SEED_MIN_DELTA,
        "positive_vs_best_control_at_least_40_of_50": wins_control
        >= CANONICAL_FOLD_SEED_CONTROL_WINS,
    }
    return _gate_result(
        checks,
        {
            "rows": int(len(ar)),
            "folds": fold_set,
            "seeds": seed_set,
            "mean_delta_vs_ar": float(np.mean(ar)),
            "mean_delta_vs_best_control": float(np.mean(control)),
            "wins_vs_ar": wins_ar,
            "wins_vs_best_control": wins_control,
        },
    )


def grouped_fold_seed_triple_scope(
    *,
    fold_ids: Sequence[Any] | np.ndarray,
    seed_ids: Sequence[Any] | np.ndarray,
    triple_ids: Sequence[Any] | np.ndarray,
) -> GateResult:
    """Validate the canonical 5-fold x 3-triple x 3-seed member matrix."""

    rows = len(np.asarray(fold_ids))
    folds = _id_vector("fold_ids", fold_ids, rows)
    seeds = _id_vector("seed_ids", seed_ids, rows)
    triples = _id_vector("triple_ids", triple_ids, rows)
    fold_set = sorted(set(folds.tolist()))
    triple_set = sorted(set(triples.tolist()))
    keys = list(zip(folds.tolist(), triples.tolist(), seeds.tolist(), strict=True))
    triple_to_seeds: dict[str, set[str]] = {}
    complete_cells = True
    for triple in triple_set:
        memberships = {
            seed for current_triple, seed in zip(triples, seeds, strict=True) if current_triple == triple
        }
        triple_to_seeds[triple] = memberships
        if len(memberships) != CANONICAL_TRIPLE_SIZE:
            complete_cells = False
        for fold in fold_set:
            cell = {
                seed
                for current_fold, current_triple, seed in zip(
                    folds, triples, seeds, strict=True
                )
                if current_fold == fold and current_triple == triple
            }
            if cell != memberships:
                complete_cells = False
    disjoint_triples = sum(len(values) for values in triple_to_seeds.values()) == len(
        set().union(*triple_to_seeds.values()) if triple_to_seeds else set()
    )
    expected_rows = (
        CANONICAL_GROUPED_FOLDS * CANONICAL_TRIPLES * CANONICAL_TRIPLE_SIZE
    )
    checks = {
        "exact_5_fold_x_3_triple_x_3_seed_scope": bool(
            rows == expected_rows
            and len(fold_set) == CANONICAL_GROUPED_FOLDS
            and len(triple_set) == CANONICAL_TRIPLES
            and len(set(keys)) == rows
        ),
        "each_fold_triple_contains_its_same_three_seeds": complete_cells,
        "seed_membership_is_disjoint_across_triples": disjoint_triples,
    }
    return _gate_result(
        checks,
        {
            "rows": int(rows),
            "folds": fold_set,
            "triples": triple_set,
            "triple_to_seeds": {
                triple: sorted(values) for triple, values in triple_to_seeds.items()
            },
        },
    )


def _fold_triple_scope(
    fold_ids: Sequence[Any] | np.ndarray,
    triple_ids: Sequence[Any] | np.ndarray,
    rows: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    folds = _id_vector("fold_ids", fold_ids, rows)
    triples = _id_vector("triple_ids", triple_ids, rows)
    fold_set = sorted(set(folds.tolist()))
    triple_set = sorted(set(triples.tolist()))
    actual = set(zip(folds.tolist(), triples.tolist(), strict=True))
    expected = {(fold, triple) for fold in fold_set for triple in triple_set}
    exact = bool(
        rows == CANONICAL_FOLD_TRIPLES
        and len(fold_set) == CANONICAL_GROUPED_FOLDS
        and len(triple_set) == CANONICAL_TRIPLES
        and len(actual) == rows
        and actual == expected
    )
    return folds, triples, exact


def _all_fold_means_positive(folds: np.ndarray, vectors: Sequence[np.ndarray]) -> bool:
    for fold in sorted(set(folds.tolist())):
        mask = folds == fold
        if any(float(np.mean(vector[mask])) <= 0 for vector in vectors):
            return False
    return True


def grouped_continuous_triple_gates(
    *,
    spearman_delta_vs_ar: Sequence[float] | np.ndarray,
    spearman_delta_vs_best_control: Sequence[float] | np.ndarray,
    top5_delta_vs_ar: Sequence[float] | np.ndarray,
    top5_delta_vs_best_control: Sequence[float] | np.ndarray,
    fold_ids: Sequence[Any] | np.ndarray,
    triple_ids: Sequence[Any] | np.ndarray,
) -> GateResult:
    """Apply canonical Phase-7 grouped continuous fold-triple gates."""

    arrays = _aligned_float_vectors(
        spearman_delta_vs_ar=spearman_delta_vs_ar,
        spearman_delta_vs_best_control=spearman_delta_vs_best_control,
        top5_delta_vs_ar=top5_delta_vs_ar,
        top5_delta_vs_best_control=top5_delta_vs_best_control,
    )
    spearman_ar = arrays["spearman_delta_vs_ar"]
    spearman_control = arrays["spearman_delta_vs_best_control"]
    top5_ar = arrays["top5_delta_vs_ar"]
    top5_control = arrays["top5_delta_vs_best_control"]
    vectors = (spearman_ar, spearman_control, top5_ar, top5_control)
    folds, triples, exact_scope = _fold_triple_scope(
        fold_ids, triple_ids, len(spearman_ar)
    )
    checks = {
        "exact_5_fold_x_3_triple_scope": exact_scope,
        "spearman_mean_delta_vs_ar_at_least_0_002": float(np.mean(spearman_ar))
        >= CANONICAL_CONTINUOUS_SPEARMAN_DELTA,
        "spearman_mean_delta_vs_best_control_at_least_0_002": float(
            np.mean(spearman_control)
        )
        >= CANONICAL_CONTINUOUS_SPEARMAN_DELTA,
        "top5_mean_delta_vs_ar_at_least_0_001": float(np.mean(top5_ar))
        >= CANONICAL_CONTINUOUS_TOP5_DELTA,
        "top5_mean_delta_vs_best_control_at_least_0_001": float(
            np.mean(top5_control)
        )
        >= CANONICAL_CONTINUOUS_TOP5_DELTA,
        "spearman_positive_vs_ar_at_least_12_of_15": int(
            np.count_nonzero(spearman_ar > 0)
        )
        >= CANONICAL_CONTINUOUS_WINS,
        "spearman_positive_vs_best_control_at_least_12_of_15": int(
            np.count_nonzero(spearman_control > 0)
        )
        >= CANONICAL_CONTINUOUS_WINS,
        "top5_positive_vs_ar_at_least_12_of_15": int(np.count_nonzero(top5_ar > 0))
        >= CANONICAL_CONTINUOUS_WINS,
        "top5_positive_vs_best_control_at_least_12_of_15": int(
            np.count_nonzero(top5_control > 0)
        )
        >= CANONICAL_CONTINUOUS_WINS,
        "all_five_fold_means_positive": _all_fold_means_positive(folds, vectors),
        "positive_paired_medians": all(float(np.median(vector)) > 0 for vector in vectors),
        "single_fold_triple_spearman_contribution_at_most_0_25": max_positive_contribution(
            spearman_ar
        )
        <= CANONICAL_GROUPED_CONTRIBUTION_CAP,
        "single_fold_triple_top5_contribution_at_most_0_25": max_positive_contribution(
            top5_ar
        )
        <= CANONICAL_GROUPED_CONTRIBUTION_CAP,
    }
    return _gate_result(
        checks,
        {
            "rows": int(len(spearman_ar)),
            "folds": sorted(set(folds.tolist())),
            "triples": sorted(set(triples.tolist())),
            "mean_spearman_delta_vs_ar": float(np.mean(spearman_ar)),
            "mean_spearman_delta_vs_best_control": float(np.mean(spearman_control)),
            "mean_top5_delta_vs_ar": float(np.mean(top5_ar)),
            "mean_top5_delta_vs_best_control": float(np.mean(top5_control)),
            "wins_spearman_vs_ar": int(np.count_nonzero(spearman_ar > 0)),
            "wins_spearman_vs_best_control": int(np.count_nonzero(spearman_control > 0)),
            "wins_top5_vs_ar": int(np.count_nonzero(top5_ar > 0)),
            "wins_top5_vs_best_control": int(np.count_nonzero(top5_control > 0)),
            "max_spearman_contribution": max_positive_contribution(spearman_ar),
            "max_top5_contribution": max_positive_contribution(top5_ar),
        },
    )


def grouped_event_triple_gates(
    *,
    pr_auc_delta_vs_ar: Sequence[float] | np.ndarray,
    pr_auc_delta_vs_best_control: Sequence[float] | np.ndarray,
    fold_ids: Sequence[Any] | np.ndarray,
    triple_ids: Sequence[Any] | np.ndarray,
) -> GateResult:
    """Apply canonical grouped three-checkpoint event-ranking gates."""

    arrays = _aligned_float_vectors(
        pr_auc_delta_vs_ar=pr_auc_delta_vs_ar,
        pr_auc_delta_vs_best_control=pr_auc_delta_vs_best_control,
    )
    ar = arrays["pr_auc_delta_vs_ar"]
    control = arrays["pr_auc_delta_vs_best_control"]
    folds, triples, exact_scope = _fold_triple_scope(fold_ids, triple_ids, len(ar))
    checks = {
        "exact_5_fold_x_3_triple_scope": exact_scope,
        "mean_delta_vs_ar_at_least_0_005": float(np.mean(ar)) >= CANONICAL_EVENT_DELTA,
        "mean_delta_vs_best_control_at_least_0_005": float(np.mean(control))
        >= CANONICAL_EVENT_DELTA,
        "positive_vs_ar_15_of_15": int(np.count_nonzero(ar > 0))
        == CANONICAL_EVENT_WINS,
        "positive_vs_best_control_15_of_15": int(np.count_nonzero(control > 0))
        == CANONICAL_EVENT_WINS,
        "all_five_fold_means_positive": _all_fold_means_positive(folds, (ar, control)),
        "positive_paired_medians": float(np.median(ar)) > 0
        and float(np.median(control)) > 0,
        "single_fold_triple_contribution_at_most_0_25": max_positive_contribution(control)
        <= CANONICAL_GROUPED_CONTRIBUTION_CAP,
    }
    return _gate_result(
        checks,
        {
            "rows": int(len(ar)),
            "folds": sorted(set(folds.tolist())),
            "triples": sorted(set(triples.tolist())),
            "mean_delta_vs_ar": float(np.mean(ar)),
            "mean_delta_vs_best_control": float(np.mean(control)),
            "wins_vs_ar": int(np.count_nonzero(ar > 0)),
            "wins_vs_best_control": int(np.count_nonzero(control > 0)),
            "max_positive_fold_triple_contribution": max_positive_contribution(control),
        },
    )


def derive_valence_direction(
    rise_magnitude: Sequence[float] | np.ndarray,
    drop_magnitude: Sequence[float] | np.ndarray,
    *,
    tie_tolerance: float = 0.0,
    require_nonnegative: bool = False,
) -> np.ndarray:
    """Derive rise (+1), drop (-1), or unresolved tie (0) from paired values."""

    arrays = _aligned_float_vectors(
        rise_magnitude=rise_magnitude, drop_magnitude=drop_magnitude
    )
    rise = arrays["rise_magnitude"]
    drop = arrays["drop_magnitude"]
    tolerance = float(tie_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tie_tolerance must be finite and non-negative")
    if require_nonnegative and (np.any(rise < 0) or np.any(drop < 0)):
        raise ValueError("True rise/drop magnitudes must be non-negative")
    difference = rise - drop
    direction = np.zeros(len(difference), dtype=np.int8)
    direction[difference > tolerance] = 1
    direction[difference < -tolerance] = -1
    return direction


def _direction_scores(truth: np.ndarray, prediction: np.ndarray) -> DirectionScores:
    if set(np.unique(truth).tolist()) != {-1, 1}:
        raise ValueError("Direction evaluation requires both true rise and drop classes")
    if not set(np.unique(prediction).tolist()).issubset({-1, 0, 1}):
        raise ValueError("Predicted direction contains an unknown class")
    balanced = float(balanced_accuracy_score(truth, prediction))
    macro = float(f1_score(truth, prediction, labels=[-1, 1], average="macro", zero_division=0))
    if not math.isfinite(balanced) or not math.isfinite(macro):
        raise ValueError("Direction evaluation produced a non-finite value")
    return DirectionScores(balanced_accuracy=balanced, macro_f1=macro)


def _direction_bootstrap(
    *,
    truth: np.ndarray,
    primary: np.ndarray,
    control: np.ndarray,
    video_ids: np.ndarray,
    resamples: int,
    seed: int,
) -> Mapping[str, BootstrapDeltaSummary]:
    unique_videos, blocks = _video_blocks(video_ids)
    point_primary = _direction_scores(truth, primary)
    point_control = _direction_scores(truth, control)
    point = {
        "balanced_accuracy": point_primary.balanced_accuracy - point_control.balanced_accuracy,
        "macro_f1": point_primary.macro_f1 - point_control.macro_f1,
    }
    distributions = {
        name: np.empty(resamples, dtype=np.float64) for name in ("balanced_accuracy", "macro_f1")
    }
    rng = np.random.default_rng(int(seed))
    for iteration in range(resamples):
        sampled = rng.choice(unique_videos, size=len(unique_videos), replace=True)
        take = np.concatenate([blocks[str(video)] for video in sampled])
        primary_scores = _direction_scores(truth[take], primary[take])
        control_scores = _direction_scores(truth[take], control[take])
        distributions["balanced_accuracy"][iteration] = (
            primary_scores.balanced_accuracy - control_scores.balanced_accuracy
        )
        distributions["macro_f1"][iteration] = primary_scores.macro_f1 - control_scores.macro_f1
    return {
        name: _bootstrap_summary(point[name], distributions[name]) for name in distributions
    }


def evaluate_valence_direction(
    *,
    true_rise_magnitude: Sequence[float] | np.ndarray,
    true_drop_magnitude: Sequence[float] | np.ndarray,
    primary_rise_prediction: Sequence[float] | np.ndarray,
    primary_drop_prediction: Sequence[float] | np.ndarray,
    control_rise_prediction: Sequence[float] | np.ndarray,
    control_drop_prediction: Sequence[float] | np.ndarray,
    video_ids: Sequence[Any] | np.ndarray,
    fold_ids: Sequence[Any] | np.ndarray,
    tie_tolerance: float = 0.0,
    resamples: int = 5_000,
    seed: int = 20260716,
) -> ValenceDirectionEvaluation:
    """Evaluate paired valence direction and its matched-control delta.

    True ties are excluded because they do not define rise versus drop.  A
    prediction tie on a directional row is retained as class 0 and therefore
    counts as an error/abstention.  Every aggregate, fold, and bootstrap score
    is paired against the same rows from the supplied control.
    """

    arrays = _aligned_float_vectors(
        true_rise_magnitude=true_rise_magnitude,
        true_drop_magnitude=true_drop_magnitude,
        primary_rise_prediction=primary_rise_prediction,
        primary_drop_prediction=primary_drop_prediction,
        control_rise_prediction=control_rise_prediction,
        control_drop_prediction=control_drop_prediction,
    )
    rows = len(arrays["true_rise_magnitude"])
    videos = _id_vector("video_ids", video_ids, rows)
    folds = _id_vector("fold_ids", fold_ids, rows)
    if int(resamples) != resamples or int(resamples) < 100:
        raise ValueError("Direction bootstrap requires an integer resamples >= 100")
    truth = derive_valence_direction(
        arrays["true_rise_magnitude"],
        arrays["true_drop_magnitude"],
        tie_tolerance=tie_tolerance,
        require_nonnegative=True,
    )
    primary = derive_valence_direction(
        arrays["primary_rise_prediction"],
        arrays["primary_drop_prediction"],
        tie_tolerance=tie_tolerance,
    )
    control = derive_valence_direction(
        arrays["control_rise_prediction"],
        arrays["control_drop_prediction"],
        tie_tolerance=tie_tolerance,
    )
    directional = truth != 0
    neutral_count = int(np.count_nonzero(~directional))
    if int(np.count_nonzero(directional)) < 4:
        raise ValueError("Too few directional valence rows after excluding true ties")
    truth = truth[directional]
    primary = primary[directional]
    control = control[directional]
    videos = videos[directional]
    folds = folds[directional]
    primary_scores = _direction_scores(truth, primary)
    control_scores = _direction_scores(truth, control)
    fold_set = sorted(set(folds.tolist()))
    if len(fold_set) != CANONICAL_GROUPED_FOLDS:
        raise ValueError("Canonical valence direction evaluation requires exactly five folds")
    fold_deltas: dict[str, list[float]] = {"balanced_accuracy": [], "macro_f1": []}
    for fold in fold_set:
        mask = folds == fold
        fold_primary = _direction_scores(truth[mask], primary[mask])
        fold_control = _direction_scores(truth[mask], control[mask])
        fold_deltas["balanced_accuracy"].append(
            fold_primary.balanced_accuracy - fold_control.balanced_accuracy
        )
        fold_deltas["macro_f1"].append(fold_primary.macro_f1 - fold_control.macro_f1)
    bootstrap = _direction_bootstrap(
        truth=truth,
        primary=primary,
        control=control,
        video_ids=videos,
        resamples=int(resamples),
        seed=int(seed),
    )
    aggregate = {
        "balanced_accuracy": primary_scores.balanced_accuracy
        - control_scores.balanced_accuracy,
        "macro_f1": primary_scores.macro_f1 - control_scores.macro_f1,
    }
    wins = {
        name: int(np.count_nonzero(np.asarray(values) > 0))
        for name, values in fold_deltas.items()
    }
    checks = {
        "aggregate_balanced_accuracy_delta_positive": aggregate["balanced_accuracy"] > 0,
        "aggregate_macro_f1_delta_positive": aggregate["macro_f1"] > 0,
        "balanced_accuracy_wins_at_least_4_of_5_folds": wins["balanced_accuracy"]
        >= CANONICAL_DIRECTION_FOLD_WINS,
        "macro_f1_wins_at_least_4_of_5_folds": wins["macro_f1"]
        >= CANONICAL_DIRECTION_FOLD_WINS,
        "balanced_accuracy_bootstrap_lower_95_positive": bootstrap[
            "balanced_accuracy"
        ].lower_95_one_sided
        > 0,
        "macro_f1_bootstrap_lower_95_positive": bootstrap["macro_f1"].lower_95_one_sided
        > 0,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    return ValenceDirectionEvaluation(
        primary=primary_scores,
        control=control_scores,
        aggregate_deltas=aggregate,
        fold_deltas={name: tuple(values) for name, values in fold_deltas.items()},
        fold_wins=wins,
        bootstrap=bootstrap,
        directional_rows=int(len(truth)),
        neutral_truth_rows_excluded=neutral_count,
        video_count=int(len(set(videos.tolist()))),
        fold_count=int(len(fold_set)),
        checks=checks,
        passed=not failed,
        failed_gates=failed,
    )


__all__ = [
    "BootstrapDeltaSummary",
    "DirectionScores",
    "END_STATE_METRICS",
    "GateResult",
    "MetricScores",
    "PairedBootstrapResult",
    "SCHEMA_VERSION",
    "SPEARMAN",
    "TOP5_LIFT",
    "TRAIN_Q90_PR_AUC",
    "ValenceDirectionEvaluation",
    "contribution_cap_gate",
    "derive_valence_direction",
    "evaluate_valence_direction",
    "grouped_continuous_triple_gates",
    "grouped_event_triple_gates",
    "grouped_fold_seed_gates",
    "grouped_fold_seed_triple_scope",
    "max_positive_contribution",
    "paired_whole_video_bootstrap",
    "score_end_state_metrics",
    "train_q90_threshold",
]
