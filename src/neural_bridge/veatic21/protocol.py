"""Leakage-safe protocol primitives for fresh VEATIC 2.1 experiments."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from numbers import Integral, Real
from typing import Any, cast

import numpy as np

from .contracts import (
    CandidateSpec,
    CellSpec,
    FeatureRows,
    FrozenRecipe,
    FrozenWinner,
    LabelRows,
    TargetSpec,
    VideoSplit,
)

DEFAULT_OUTER_FOLDS = 5
DEFAULT_INNER_FOLDS = 3
DEFAULT_SPLIT_SEED = 20_260_721
DEFAULT_REFIT_SEED = 20_260_721
DEFAULT_AR_LAGS = (1, 2, 3, 4, 5, 6)
DEFAULT_TIE_BREAK = "candidate_name_ascending"

_NAME_TIE_BREAKS = {
    "candidate_name",
    "candidate_name_asc",
    "candidate_name_ascending",
    "higher_metric_then_candidate_name",
    "mean_desc_candidate_name_asc",
}


def _json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return _json_ready(cast(Any, value).tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("stable JSON mappings require string keys")
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("stable JSON does not permit non-finite floats")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported stable JSON value: {type(value).__name__}")


def stable_json_sha256(value: object) -> str:
    """Return SHA-256 of canonical, finite JSON."""

    encoded = json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _video_ids(values: Sequence[object] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError("video_id must be one-dimensional")
    result = np.asarray([str(value) for value in array], dtype=object)
    if any(not value for value in result):
        raise ValueError("video_id values must be non-empty")
    return result


def _row_indices(values: Sequence[object] | np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("row_index must be one-dimensional")
    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError("row_index values must be integers, not booleans")
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.int64, copy=False)
    if array.dtype == object and all(
        isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) for value in array
    ):
        return np.asarray(array, dtype=np.int64)
    raise ValueError("row_index values must be integers")


def _identity_arrays(
    rows_or_video_id: FeatureRows | LabelRows | Sequence[object] | np.ndarray,
    row_index: Sequence[object] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(rows_or_video_id, (FeatureRows, LabelRows)):
        if row_index is not None:
            raise TypeError("row_index must be omitted when rows are provided")
        videos = _video_ids(rows_or_video_id.video_id)
        indices = _row_indices(rows_or_video_id.row_index)
    else:
        if row_index is None:
            raise TypeError("row_index is required with raw video_id values")
        videos = _video_ids(rows_or_video_id)
        indices = _row_indices(row_index)
    if len(videos) != len(indices):
        raise ValueError("video_id and row_index lengths differ")
    keys = list(zip(videos.tolist(), indices.tolist(), strict=True))
    if len(keys) != len(set(keys)):
        raise ValueError("row identities must be unique")
    return videos, indices


def _stable_video_order(video_ids: Sequence[str], *, seed: int, namespace: str) -> list[str]:
    def key(video_id: str) -> tuple[str, str]:
        digest = hashlib.sha256(f"{seed}:{namespace}:{video_id}".encode()).hexdigest()
        return digest, video_id

    return sorted(video_ids, key=key)


def build_video_splits(
    video_ids: Sequence[object] | np.ndarray,
    *,
    outer_folds: int = DEFAULT_OUTER_FOLDS,
    inner_folds: int = DEFAULT_INNER_FOLDS,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[VideoSplit, ...]:
    """Build deterministic nested splits using video identity only."""

    if outer_folds < 2 or inner_folds < 2:
        raise ValueError("nested discovery requires at least two outer and inner folds")
    unique = sorted(set(_video_ids(video_ids).tolist()))
    if len(unique) < outer_folds:
        raise ValueError("fewer unique videos than outer folds")

    ordered = _stable_video_order(unique, seed=split_seed, namespace="outer")
    outer_groups = [ordered[fold::outer_folds] for fold in range(outer_folds)]
    splits: list[VideoSplit] = []
    for outer_fold, test_group in enumerate(outer_groups):
        test = tuple(sorted(test_group))
        test_set = set(test)
        train = tuple(video_id for video_id in unique if video_id not in test_set)
        if len(train) < inner_folds:
            raise ValueError("fewer outer-training videos than inner folds")

        inner_order = _stable_video_order(
            train,
            seed=split_seed,
            namespace=f"outer-{outer_fold}-inner",
        )
        inner_groups = [inner_order[fold::inner_folds] for fold in range(inner_folds)]
        inner_splits = tuple(
            (
                tuple(sorted(set(train) - set(validation))),
                tuple(sorted(validation)),
            )
            for validation in inner_groups
        )
        payload = {
            "schema": "veatic21-video-split-v1",
            "split_seed": split_seed,
            "outer_folds": outer_folds,
            "inner_folds": inner_folds,
            "outer_fold": outer_fold,
            "train_video_ids": train,
            "test_video_ids": test,
            "inner_splits": inner_splits,
        }
        splits.append(
            VideoSplit(
                outer_fold=outer_fold,
                train_video_ids=train,
                test_video_ids=test,
                inner_splits=inner_splits,
                digest=stable_json_sha256(payload),
            )
        )
    return tuple(splits)


def target_support_mask(
    rows_or_video_id: FeatureRows | LabelRows | Sequence[object] | np.ndarray,
    row_index: Sequence[object] | np.ndarray | TargetSpec | None = None,
    target: TargetSpec | None = None,
) -> np.ndarray:
    """Identify rows having every future horizon using row identities only."""

    if isinstance(row_index, TargetSpec):
        if target is not None:
            raise TypeError("target was provided twice")
        target = row_index
        row_index = None
    if target is None:
        raise TypeError("target is required")
    target.validate()
    videos, indices = _identity_arrays(rows_or_video_id, row_index)
    identities = set(zip(videos.tolist(), indices.tolist(), strict=True))
    return np.fromiter(
        (
            all((video_id, row + horizon) in identities for horizon in target.horizon_rows)
            for video_id, row in zip(videos.tolist(), indices.tolist(), strict=True)
        ),
        dtype=bool,
        count=len(videos),
    )


def assert_row_alignment(features: FeatureRows, labels: LabelRows, *, atol: float = 1e-9) -> None:
    """Reject reordered, missing, duplicated, or time-shifted label rows."""

    features.validate()
    labels.validate()
    feature_videos, feature_indices = _identity_arrays(features)
    label_videos, label_indices = _identity_arrays(labels)
    if not np.array_equal(feature_videos, label_videos) or not np.array_equal(
        feature_indices, label_indices
    ):
        raise ValueError("feature and label row identities are not exactly aligned")
    if not np.allclose(
        np.asarray(features.time_seconds, dtype=np.float64),
        np.asarray(labels.time_seconds, dtype=np.float64),
        rtol=0.0,
        atol=atol,
    ):
        raise ValueError("feature and label timestamps are not aligned")


validate_alignment = assert_row_alignment


def _target_values(labels: LabelRows, target: TargetSpec) -> np.ndarray:
    target.validate()
    if target.label not in {"arousal", "valence"}:
        raise ValueError(f"unsupported target label: {target.label}")
    values = np.asarray(getattr(labels, target.label), dtype=np.float64)
    if values.ndim != 1 or len(values) != len(labels.video_id):
        raise ValueError("target labels must be a row-aligned vector")
    if not np.isfinite(values).all():
        raise ValueError("target labels contain non-finite values")
    return values


def future_target_values(labels: LabelRows, target: TargetSpec) -> np.ndarray:
    """Compute maximum future movement without crossing a video boundary."""

    videos, indices = _identity_arrays(labels)
    values = _target_values(labels, target)
    lookup = {
        (video_id, row): value
        for video_id, row, value in zip(
            videos.tolist(), indices.tolist(), values.tolist(), strict=True
        )
    }
    support = target_support_mask(labels, target)
    result = np.full(len(values), np.nan, dtype=np.float64)
    for position in np.flatnonzero(support):
        video_id = str(videos[position])
        row = int(indices[position])
        deltas = np.asarray(
            [
                lookup[(video_id, row + horizon)] - values[position]
                for horizon in target.horizon_rows
            ]
        )
        if target.transform == "absolute":
            result[position] = float(np.max(np.abs(deltas)))
        elif target.transform == "positive":
            result[position] = float(np.max(np.maximum(deltas, 0.0)))
        else:
            raise ValueError(f"unsupported target transform: {target.transform}")
    return result


def fit_event_threshold(
    target_values: Sequence[float] | np.ndarray,
    train_mask: Sequence[bool] | np.ndarray,
    target: TargetSpec,
) -> float:
    """Fit an event threshold from explicitly train-owned target rows."""

    target.validate()
    values = np.asarray(target_values, dtype=np.float64)
    mask = np.asarray(train_mask)
    if values.ndim != 1 or mask.shape != values.shape:
        raise ValueError("target_values and train_mask must be aligned vectors")
    if not np.issubdtype(mask.dtype, np.bool_):
        raise ValueError("train_mask must be boolean")
    owned = values[mask]
    if not len(owned):
        raise ValueError("train_mask owns no target rows")
    if not np.isfinite(owned).all():
        raise ValueError("train-owned target rows must all have future support")
    return float(np.quantile(owned, target.quantile, method="linear"))


def event_labels(
    target_values: Sequence[float] | np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Return binary labels; unsupported target rows remain false."""

    values = np.asarray(target_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("target_values must be one-dimensional")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    result = np.zeros(values.shape, dtype=bool)
    available = np.isfinite(values)
    result[available] = values[available] >= threshold
    return result


binary_event_labels = event_labels


def causal_ar_features(
    labels: LabelRows,
    target: TargetSpec,
    *,
    lag_rows: Sequence[int] = DEFAULT_AR_LAGS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return past-label values and per-value availability without dropping cold starts."""

    lags = tuple(lag_rows)
    if (
        not lags
        or any(isinstance(lag, bool) or not isinstance(lag, Integral) for lag in lags)
        or any(lag <= 0 for lag in lags)
        or tuple(sorted(set(lags))) != lags
    ):
        raise ValueError("lag_rows must be unique, increasing positive integers")
    videos, indices = _identity_arrays(labels)
    target_values = _target_values(labels, target)
    lookup = {
        (video_id, row): value
        for video_id, row, value in zip(
            videos.tolist(), indices.tolist(), target_values.tolist(), strict=True
        )
    }
    values = np.zeros((len(videos), len(lags)), dtype=np.float64)
    available = np.zeros(values.shape, dtype=bool)
    for position, (video_id, row) in enumerate(
        zip(videos.tolist(), indices.tolist(), strict=True)
    ):
        for column, lag in enumerate(lags):
            key = (video_id, row - lag)
            if key in lookup:
                values[position, column] = lookup[key]
                available[position, column] = True
    return values, available


def _validated_candidates(candidates: Sequence[CandidateSpec]) -> tuple[CandidateSpec, ...]:
    result = tuple(candidates)
    if not result:
        raise ValueError("candidate grid is empty")
    for candidate in result:
        candidate.validate()
    names = [candidate.name for candidate in result]
    if len(names) != len(set(names)):
        raise ValueError("candidate names must be unique")
    return tuple(sorted(result, key=lambda candidate: candidate.name))


def _validate_selection(selection_metric: str, tie_break: str) -> None:
    if not selection_metric or any(char.isspace() for char in selection_metric):
        raise ValueError("selection_metric must be a non-empty token")
    lowered = selection_metric.lower()
    if any(token in lowered for token in ("heldout", "holdout", "confirmation", "outer")):
        raise ValueError("selection_metric must be discovery-owned")
    if tie_break not in _NAME_TIE_BREAKS:
        raise ValueError("tie_break must declare ascending candidate-name ordering")


def _reject_confirmation_fields(row: Mapping[str, object], *, allow_outer_fold: bool) -> None:
    for key in row:
        lowered = key.lower()
        forbidden = any(token in lowered for token in ("heldout", "holdout", "confirmation"))
        forbidden = forbidden or lowered.startswith("test_") or lowered.endswith("_test")
        forbidden = forbidden or (lowered.startswith("outer_") and lowered != "outer_fold")
        if forbidden or (lowered == "outer_fold" and not allow_outer_fold):
            raise ValueError(f"selection rows cannot contain held-out result field {key!r}")


def _score_grid(
    candidates: tuple[CandidateSpec, ...],
    rows: Sequence[Mapping[str, object]],
    *,
    fold_key: str,
    fold_count: int,
    selection_metric: str,
    allow_outer_fold: bool,
    require_discovery_digest: bool,
) -> tuple[tuple[Mapping[str, object], ...], dict[str, list[float]]]:
    expected = {
        (candidate.name, fold)
        for candidate in candidates
        for fold in range(fold_count)
    }
    seen: set[tuple[str, int]] = set()
    scores = {candidate.name: [] for candidate in candidates}
    canonical: list[dict[str, object]] = []
    for source in rows:
        if not isinstance(source, Mapping) or not all(isinstance(key, str) for key in source):
            raise TypeError("selection rows must be string-keyed mappings")
        _reject_confirmation_fields(source, allow_outer_fold=allow_outer_fold)
        candidate_name = source.get("candidate")
        fold = source.get(fold_key)
        score = source.get(selection_metric)
        if not isinstance(candidate_name, str) or candidate_name not in scores:
            raise ValueError("selection row names an undeclared candidate")
        if isinstance(fold, bool) or not isinstance(fold, Integral):
            raise ValueError(f"{fold_key} must be an integer")
        cell = (candidate_name, int(fold))
        if cell not in expected:
            raise ValueError(f"selection row is outside declared grid: {cell}")
        if cell in seen:
            raise ValueError(f"duplicate selection row: {cell}")
        if (
            isinstance(score, bool)
            or not isinstance(score, Real)
            or not math.isfinite(float(score))
        ):
            raise ValueError(f"{selection_metric} must be finite")
        if require_discovery_digest:
            _require_digest("discovery_digest", source.get("discovery_digest"))
        seen.add(cell)
        scores[candidate_name].append(float(score))
        ready = _json_ready(dict(source))
        if not isinstance(ready, dict):
            raise TypeError("selection row did not normalize to a mapping")
        normalized = cast(dict[str, object], ready)
        normalized["candidate"] = candidate_name
        normalized[fold_key] = int(fold)
        normalized[selection_metric] = float(score)
        canonical.append(normalized)
    if seen != expected:
        missing = sorted(expected - seen)
        raise ValueError(f"candidate grid is incomplete; missing {missing[:3]}")
    canonical.sort(
        key=lambda row: (str(row["candidate"]), cast(int, row[fold_key]))
    )
    return tuple(canonical), scores


def _selected_candidate(
    candidates: tuple[CandidateSpec, ...], scores: Mapping[str, Sequence[float]]
) -> CandidateSpec:
    means = {name: math.fsum(values) / len(values) for name, values in scores.items()}
    return min(candidates, key=lambda candidate: (-means[candidate.name], candidate.name))


def _require_digest(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 hex digest") from error
    return value.lower()


def freeze_winner(
    candidates: Sequence[CandidateSpec],
    inner_scores: Sequence[Mapping[str, object]],
    *,
    cell: CellSpec,
    split_digest: str,
    selection_metric: str = "pooled_pr_auc",
    tie_break: str = DEFAULT_TIE_BREAK,
) -> FrozenWinner:
    """Freeze one candidate from the complete inner grid before any outer result exists."""

    cell.validate()
    candidate_grid = _validated_candidates(candidates)
    _validate_selection(selection_metric, tie_break)
    split_digest = _require_digest("split_digest", split_digest)
    canonical_scores, scores = _score_grid(
        candidate_grid,
        inner_scores,
        fold_key="inner_fold",
        fold_count=cell.inner_folds,
        selection_metric=selection_metric,
        allow_outer_fold=False,
        require_discovery_digest=False,
    )
    candidate = _selected_candidate(candidate_grid, scores)
    payload = {
        "schema": "veatic21-frozen-winner-v1",
        "candidate": candidate,
        "cell": cell,
        "split_digest": split_digest,
        "selection_metric": selection_metric,
        "tie_break": tie_break,
        "inner_scores": canonical_scores,
    }
    return FrozenWinner(
        candidate=candidate,
        cell=cell,
        split_digest=split_digest,
        selection_metric=selection_metric,
        tie_break=tie_break,
        inner_scores=canonical_scores,
        digest=stable_json_sha256(payload),
    )


def freeze_final_recipe(
    candidates: Sequence[CandidateSpec],
    discovery_scores: Sequence[Mapping[str, object]],
    *,
    outer_fold_count: int = DEFAULT_OUTER_FOLDS,
    selection_metric: str = "pooled_pr_auc",
    tie_break: str = DEFAULT_TIE_BREAK,
    refit_seed: int = DEFAULT_REFIT_SEED,
    promotable: bool = False,
) -> FrozenRecipe:
    """Freeze a non-promotable recipe from complete inner-discovery summaries."""

    if outer_fold_count < 2:
        raise ValueError("final recipe requires at least two outer discovery folds")
    if isinstance(refit_seed, bool) or not isinstance(refit_seed, Integral) or refit_seed < 0:
        raise ValueError("refit_seed must be a non-negative integer")
    refit_seed = int(refit_seed)
    if promotable is not False:
        raise ValueError(
            "VEATIC 2.1 foundation has no preregistered promotion gate"
        )
    candidate_grid = _validated_candidates(candidates)
    _validate_selection(selection_metric, tie_break)
    rows, scores = _score_grid(
        candidate_grid,
        discovery_scores,
        fold_key="outer_fold",
        fold_count=outer_fold_count,
        selection_metric=selection_metric,
        allow_outer_fold=True,
        require_discovery_digest=True,
    )
    discovery_digests: list[str] = []
    for outer_fold in range(outer_fold_count):
        fold_digests = {
            _require_digest("discovery_digest", row["discovery_digest"])
            for row in rows
            if cast(int, row["outer_fold"]) == outer_fold
        }
        if len(fold_digests) != 1:
            raise ValueError("each outer fold must reference one frozen discovery digest")
        discovery_digests.append(fold_digests.pop())
    if len(set(discovery_digests)) != outer_fold_count:
        raise ValueError("outer discovery folds must have distinct frozen digests")

    candidate = _selected_candidate(candidate_grid, scores)
    payload = {
        "schema": "veatic21-frozen-recipe-v1",
        "candidate": candidate,
        "discovery_digests": tuple(discovery_digests),
        "outer_fold_count": outer_fold_count,
        "selection_metric": selection_metric,
        "tie_break": tie_break,
        "refit_seed": refit_seed,
        "promotable": False,
    }
    return FrozenRecipe(
        candidate=candidate,
        discovery_digests=tuple(discovery_digests),
        outer_fold_count=outer_fold_count,
        selection_metric=selection_metric,
        tie_break=tie_break,
        refit_seed=refit_seed,
        promotable=False,
        digest=stable_json_sha256(payload),
    )


__all__ = [
    "DEFAULT_AR_LAGS",
    "DEFAULT_INNER_FOLDS",
    "DEFAULT_OUTER_FOLDS",
    "DEFAULT_REFIT_SEED",
    "DEFAULT_SPLIT_SEED",
    "DEFAULT_TIE_BREAK",
    "assert_row_alignment",
    "binary_event_labels",
    "build_video_splits",
    "causal_ar_features",
    "event_labels",
    "fit_event_threshold",
    "freeze_final_recipe",
    "freeze_winner",
    "future_target_values",
    "stable_json_sha256",
    "target_support_mask",
    "validate_alignment",
]
