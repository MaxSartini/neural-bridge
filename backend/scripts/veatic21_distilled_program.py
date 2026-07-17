"""Shared contracts for the distilled VEATIC 2.1 research programme.

This module deliberately does not preserve the historical VEATIC benchmark
schema.  It supplies the dataset-independent pieces that were proven useful on
AGAIN: deterministic video ownership, train-only autoregressive fitting,
matched video-level controls, and ranking/event metrics.  The source cache is
read elsewhere and is never modified by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score

from backend.scripts import run_again_dense_2hz_phase5_continuous_residual_blocked as continuous


SCHEMA_VERSION = "veatic21_distilled_program_v2"
ROW_HZ = 2.0
PCA_WIDTH = 256
WINDOW_ROWS = 5
DIAGNOSTIC_WIDTH = 53
DEVELOPMENT_FOLDS = 5
DEVELOPMENT_SEEDS = (20260716, 20260717, 20260718)
RESERVED_VIDEO_COUNT = 0
EVENT_QUANTILE = 0.90
FIRST_30_SECONDS = 30.0
RIDGE_ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)


def canonical_digest(value: Any, *, digest_size: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=digest_size).hexdigest()


def array_digest(values: np.ndarray, *, digest_size: int = 16) -> str:
    arr = np.ascontiguousarray(values)
    digest = hashlib.blake2b(digest_size=digest_size)
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(json.dumps(list(arr.shape)).encode("ascii"))
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def _video_hash(namespace: str, video_id: str) -> str:
    return hashlib.blake2b(
        f"{namespace}|{video_id}".encode("utf-8"), digest_size=16
    ).hexdigest()


@dataclass(frozen=True)
class VideoPanel:
    development_videos: tuple[str, ...]
    reserved_videos: tuple[str, ...]
    namespace: str
    digest: str


def deterministic_video_panel(
    row_counts: Mapping[str, int],
    *,
    reserved_count: int = RESERVED_VIDEO_COUNT,
    namespace: str = "veatic21_distilled_reserved_panel_v1",
) -> VideoPanel:
    """Create the stable all-video CV inventory and any explicitly requested reserve.

    The assignment depends only on video identifiers and the frozen namespace,
    never labels or features.  Row counts are accepted so the resulting
    contract can prove it covers the complete source inventory.
    """

    videos = sorted({str(video_id) for video_id in row_counts}, key=lambda value: int(value))
    if not videos or any(int(row_counts[video_id]) <= 0 for video_id in videos):
        raise ValueError("Video panel requires positive row counts")
    if reserved_count < 0 or reserved_count >= len(videos):
        raise ValueError("Reserved video count must be non-negative and leave training videos")
    ranked = sorted(videos, key=lambda value: (_video_hash(namespace, value), int(value)))
    reserved_set = set(ranked[: int(reserved_count)])
    development = tuple(video for video in videos if video not in reserved_set)
    reserved = tuple(video for video in videos if video in reserved_set)
    payload = {
        "namespace": namespace,
        "development_videos": development,
        "reserved_videos": reserved,
        "row_counts": {video: int(row_counts[video]) for video in videos},
    }
    return VideoPanel(
        development_videos=development,
        reserved_videos=reserved,
        namespace=namespace,
        digest=canonical_digest(payload),
    )


@dataclass(frozen=True)
class GroupedFold:
    fold: int
    train_videos: tuple[str, ...]
    test_videos: tuple[str, ...]
    train_rows: int
    test_rows: int
    digest: str


def balanced_grouped_video_folds(
    row_counts: Mapping[str, int],
    videos: Sequence[str],
    *,
    fold_count: int = DEVELOPMENT_FOLDS,
    namespace: str = "veatic21_distilled_grouped_folds_v1",
) -> tuple[GroupedFold, ...]:
    """Assign whole videos to deterministic row-balanced folds."""

    selected = tuple(sorted({str(video) for video in videos}, key=lambda value: int(value)))
    if fold_count < 2 or fold_count > len(selected):
        raise ValueError("Grouped fold count is incompatible with the video panel")
    if any(video not in row_counts or int(row_counts[video]) <= 0 for video in selected):
        raise ValueError("Grouped folds require a positive row count for every video")
    buckets: list[list[str]] = [[] for _ in range(int(fold_count))]
    totals = [0 for _ in range(int(fold_count))]
    ordered = sorted(
        selected,
        key=lambda value: (-int(row_counts[value]), _video_hash(namespace, value), int(value)),
    )
    for video in ordered:
        destination = min(range(int(fold_count)), key=lambda index: (totals[index], index))
        buckets[destination].append(video)
        totals[destination] += int(row_counts[video])
    folds: list[GroupedFold] = []
    all_set = set(selected)
    for index, bucket in enumerate(buckets, start=1):
        test = tuple(sorted(bucket, key=lambda value: int(value)))
        test_set = set(test)
        train = tuple(video for video in selected if video not in test_set)
        if not test or set(train) & test_set or set(train) | test_set != all_set:
            raise RuntimeError("Grouped video ownership is incomplete or overlapping")
        payload = {
            "namespace": namespace,
            "fold": index,
            "train_videos": train,
            "test_videos": test,
        }
        folds.append(
            GroupedFold(
                fold=index,
                train_videos=train,
                test_videos=test,
                train_rows=sum(int(row_counts[video]) for video in train),
                test_rows=sum(int(row_counts[video]) for video in test),
                digest=canonical_digest(payload),
            )
        )
    return tuple(folds)


def indices_for_videos(video_ids: np.ndarray, videos: Sequence[str]) -> np.ndarray:
    wanted = {str(value) for value in videos}
    values = np.asarray(video_ids, dtype=str)
    return np.flatnonzero(np.isin(values, list(wanted))).astype(np.int64)


def observed_history_features(
    signal: np.ndarray,
    video_ids: np.ndarray,
    time_seconds: np.ndarray,
    *,
    training_initial_value: float,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Build the AGAIN-style privileged history lane for either affect axis."""

    values = np.asarray(signal, dtype=np.float32)
    videos = np.asarray(video_ids, dtype=str)
    times = np.asarray(time_seconds, dtype=np.float32)
    if not (len(values) == len(videos) == len(times)):
        raise ValueError("Observed-history arrays are not row aligned")
    if not math.isfinite(float(training_initial_value)):
        raise ValueError("Observed-history initialization must be train-derived and finite")
    output = np.zeros((len(values), 13), dtype=np.float32)
    resets = 0
    previous_video: str | None = None
    history: list[float] = []
    durations = {
        video: float(np.max(times[videos == video], initial=0.5))
        for video in np.unique(videos)
    }
    for index, (value, video, seconds) in enumerate(zip(values, videos, times, strict=True)):
        if video != previous_video:
            history = []
            previous_video = video
            resets += 1
        lag1 = history[-1] if len(history) >= 1 else float(training_initial_value)
        lag2 = history[-2] if len(history) >= 2 else float(training_initial_value)
        lag4 = history[-4] if len(history) >= 4 else float(training_initial_value)
        recent = np.asarray((history[-4:] or [float(training_initial_value)]), dtype=np.float32)
        fraction = float(np.clip(seconds / max(0.5, durations[video]), 0.0, 1.0))
        output[index] = (
            value,
            lag1,
            lag2,
            lag4,
            value - lag1,
            lag1 - lag2,
            value - lag4,
            float(np.mean(recent)),
            float(np.std(recent)),
            float(np.min(recent)),
            float(np.max(recent)),
            float(np.log1p(max(0.0, seconds))),
            fraction,
        )
        history.append(float(value))
    return output, {
        "feature_width": int(output.shape[1]),
        "state_reset_per_video": True,
        "cross_video_state_carry": 0,
        "video_resets": resets,
        "initialization": "training_median",
        "all_finite": bool(np.isfinite(output).all()),
    }


def canonical_ar_history_features(
    signal: np.ndarray,
    video_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    """Build the exact seven-feature AGAIN privileged AR contract.

    The first four rows of every video lack full lag-1/2/4 ownership and are
    therefore excluded from privileged AR fitting and scoring.  They are not
    median-filled.  Zero-filled/masked history is reserved for the video-only
    cold-start lane, whose inputs never contain the response signal.
    """

    values = np.asarray(signal, dtype=np.float32)
    videos = np.asarray(video_ids, dtype=str)
    if values.ndim != 1 or videos.ndim != 1 or len(values) != len(videos):
        raise ValueError("Canonical AR arrays must be aligned non-empty 1D vectors")
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Canonical AR signal must be non-empty and finite")
    output = np.zeros((len(values), 7), dtype=np.float32)
    context_valid = np.zeros(len(values), dtype=bool)
    starts = 0
    previous_video: str | None = None
    history: list[float] = []
    for index, (value, video) in enumerate(zip(values, videos, strict=True)):
        if video != previous_video:
            history = []
            previous_video = video
            starts += 1
        if len(history) >= 4:
            lag1 = history[-1]
            lag2 = history[-2]
            lag4 = history[-4]
            output[index] = (
                value,
                lag1,
                lag2,
                lag4,
                value - lag1,
                value - lag2,
                value - lag4,
            )
            context_valid[index] = True
        history.append(float(value))
    return output, context_valid, {
        "feature_names": [
            "current",
            "lag_1row",
            "lag_2row",
            "lag_4row",
            "delta_prev_1row",
            "delta_prev_2row",
            "delta_prev_4row",
        ],
        "feature_width": 7,
        "full_context_required": True,
        "invalid_prefix_rows_per_video": 4,
        "state_reset_per_video": True,
        "cross_video_state_carry": 0,
        "video_resets": starts,
        "context_valid_rows": int(np.count_nonzero(context_valid)),
        "context_valid_digest": array_digest(context_valid.astype(np.uint8)),
        "all_exposed_features_finite": bool(np.isfinite(output).all()),
    }


def _train_only_standardize(
    train_x: np.ndarray, test_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(train_x, axis=0).astype(np.float32)
    std = np.nanstd(train_x, axis=0).astype(np.float32)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    train = ((np.nan_to_num(train_x, nan=0.0) - mean) / std).astype(np.float32)
    test = ((np.nan_to_num(test_x, nan=0.0) - mean) / std).astype(np.float32)
    return train, test, mean, std


@dataclass(frozen=True)
class InnerVideoOwnership:
    """One auditable inner split shared by AR, residual, and control heads.

    Ownership is over whole videos and is independent of labels, features, and
    target-validity masks.  ``outer_row_video_digest`` binds the ownership to an
    exact outer-training row order, so a caller cannot silently reuse it after
    filtering or reordering rows.
    """

    namespace: str
    inner_train_videos: tuple[str, ...]
    inner_validation_videos: tuple[str, ...]
    all_videos: tuple[str, ...]
    validation_fraction: float
    outer_row_video_digest: str
    digest: str

    def row_masks(self, video_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return row-aligned train/validation masks after exact ownership checks."""

        videos = np.asarray(video_ids, dtype=str)
        if canonical_digest(videos.tolist()) != self.outer_row_video_digest:
            raise ValueError("Inner-video ownership does not match the outer row inventory/order")
        train = np.isin(videos, list(self.inner_train_videos))
        validation = np.isin(videos, list(self.inner_validation_videos))
        if np.any(train & validation) or not np.all(train | validation):
            raise RuntimeError("Inner-video ownership is incomplete or overlapping")
        return train.astype(bool), validation.astype(bool)

    def eligible_indices(
        self,
        video_ids: np.ndarray,
        eligible_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return eligible inner-train and inner-validation row indices."""

        train, validation = self.row_masks(video_ids)
        eligible = np.asarray(eligible_mask, dtype=bool)
        if eligible.shape != train.shape:
            raise ValueError("Eligibility mask is not aligned to inner ownership")
        return (
            np.flatnonzero(train & eligible).astype(np.int64),
            np.flatnonzero(validation & eligible).astype(np.int64),
        )

    def audit_manifest(self) -> Mapping[str, Any]:
        return {
            "namespace": self.namespace,
            "digest": self.digest,
            "validation_fraction": self.validation_fraction,
            "outer_row_video_digest": self.outer_row_video_digest,
            "inner_train_videos": list(self.inner_train_videos),
            "inner_validation_videos": list(self.inner_validation_videos),
            "all_videos": list(self.all_videos),
            "inner_train_video_count": len(self.inner_train_videos),
            "inner_validation_video_count": len(self.inner_validation_videos),
        }


def member_inner_namespace(*, outer_fold: int, target_name: str, seed: int) -> str:
    """Canonical identity for one separately trained VEATIC 2.1 model member."""

    if int(outer_fold) < 1 or not str(target_name):
        raise ValueError("Member ownership requires a positive fold and non-empty target")
    return (
        f"veatic21|outer_fold{int(outer_fold)}|target={str(target_name)}"
        f"|matched_inner|seed={int(seed)}"
    )


def build_member_inner_video_ownership(
    video_ids: np.ndarray,
    *,
    outer_fold: int,
    target_name: str,
    seed: int,
    validation_fraction: float = 0.20,
) -> InnerVideoOwnership:
    """Build the common ownership for one target/fold/seed neural member."""

    return build_inner_video_ownership(
        video_ids,
        namespace=member_inner_namespace(
            outer_fold=int(outer_fold), target_name=str(target_name), seed=int(seed)
        ),
        validation_fraction=float(validation_fraction),
    )


def build_inner_video_ownership(
    video_ids: np.ndarray,
    *,
    namespace: str,
    validation_fraction: float = 0.20,
) -> InnerVideoOwnership:
    """Freeze common fold/target/seed inner-video ownership.

    ``namespace`` should identify the outer fold, target, and member seed.  The
    same returned object (or at minimum its namespace and digest) must be used
    by the frozen AR lane and every matched residual/control head.
    """

    videos = np.asarray(video_ids, dtype=str)
    if videos.ndim != 1 or not len(videos):
        raise ValueError("Inner ownership requires a non-empty 1D video-id array")
    if not namespace:
        raise ValueError("Inner ownership requires a non-empty namespace")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("Inner validation fraction must be strictly between zero and one")
    unique = sorted(
        np.unique(videos).astype(str),
        key=lambda value: (_video_hash(namespace, str(value)), str(value)),
    )
    # Three videos is the minimum that permits a held-out inner validation
    # video and honest cross-fitting within the remaining inner-training set.
    if len(unique) < 3:
        raise ValueError("Inner validation and AR cross-fitting require at least three videos")
    validation_count = max(1, int(math.ceil(len(unique) * float(validation_fraction))))
    if validation_count >= len(unique) - 1:
        validation_count = len(unique) - 2
    validation_videos = tuple(unique[:validation_count])
    validation_set = set(validation_videos)
    training_videos = tuple(video for video in unique if video not in validation_set)
    row_digest = canonical_digest(videos.tolist())
    payload = {
        "namespace": namespace,
        "validation_fraction": float(validation_fraction),
        "outer_row_video_digest": row_digest,
        "inner_train_videos": training_videos,
        "inner_validation_videos": validation_videos,
        "all_videos": tuple(unique),
    }
    return InnerVideoOwnership(
        namespace=namespace,
        inner_train_videos=training_videos,
        inner_validation_videos=validation_videos,
        all_videos=tuple(unique),
        validation_fraction=float(validation_fraction),
        outer_row_video_digest=row_digest,
        digest=canonical_digest(payload),
    )


def _inner_grouped_split(video_ids: np.ndarray, namespace: str) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper; new code should retain the ownership object."""

    ownership = build_inner_video_ownership(video_ids, namespace=namespace)
    train, validation = ownership.row_masks(video_ids)
    return np.flatnonzero(train).astype(np.int64), np.flatnonzero(validation).astype(np.int64)


@dataclass(frozen=True)
class FrozenArResult:
    # Honest outer-training predictions used to construct residual targets.
    # Every row is predicted by an AR model that excluded its whole video.
    train_prediction: np.ndarray
    # Final outer-test predictions from a separate AR fit on all eligible
    # outer-training rows.  That final fit is never used for residual targets.
    test_prediction: np.ndarray
    selected_alpha: float
    inner_validation_mse: float
    # ``mean``/``std`` retain compatibility and describe the final outer fit.
    mean: np.ndarray
    std: np.ndarray
    ownership: InnerVideoOwnership
    inner_mean: np.ndarray
    inner_std: np.ndarray
    inner_coef: np.ndarray
    inner_intercept: float
    final_coef: np.ndarray
    final_intercept: float
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class _RidgeFit:
    model: Ridge
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray
    intercept: float
    fit_digest: str


@dataclass(frozen=True)
class ArCrossfitVideoFold:
    """Backend-neutral whole-video fit/prediction ownership for stacked AR."""

    fold: int
    fit_videos: tuple[str, ...]
    prediction_videos: tuple[str, ...]
    digest: str


def _fit_ridge(
    x: np.ndarray,
    target: np.ndarray,
    fit_rows: np.ndarray,
    *,
    alpha: float,
) -> _RidgeFit:
    fit_x = np.asarray(x, dtype=np.float32)[fit_rows]
    fit_target = np.asarray(target, dtype=np.float32)[fit_rows]
    standardized, _, mean, std = _train_only_standardize(fit_x, fit_x[:0])
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(standardized, fit_target)
    coef = np.asarray(model.coef_, dtype=np.float32).reshape(-1)
    intercept = float(np.asarray(model.intercept_).reshape(()))
    payload = {
        "alpha": float(alpha),
        "fit_rows": int(len(fit_rows)),
        "fit_x_digest": array_digest(fit_x),
        "fit_target_digest": array_digest(fit_target),
        "mean_digest": array_digest(mean),
        "std_digest": array_digest(std),
        "coef_digest": array_digest(coef),
        "intercept": intercept,
    }
    return _RidgeFit(
        model=model,
        mean=mean,
        std=std,
        coef=coef,
        intercept=intercept,
        fit_digest=canonical_digest(payload),
    )


def _ridge_predict(fit: _RidgeFit, values: np.ndarray) -> np.ndarray:
    standardized = (
        (np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0) - fit.mean) / fit.std
    ).astype(np.float32)
    return fit.model.predict(standardized).astype(np.float32)


def build_ar_crossfit_video_folds(
    ownership: InnerVideoOwnership,
    *,
    fold_count: int = 5,
) -> tuple[ArCrossfitVideoFold, ...]:
    """Plan honest AR predictions for the inner-training side.

    A neural AR backend can train one model per returned scope and predict only
    ``prediction_videos``.  Inner-validation videos are absent from every
    scope; they are predicted by the primary model trained on all
    ``ownership.inner_train_videos``.
    """

    ordered = sorted(
        {str(video) for video in ownership.inner_train_videos},
        key=lambda value: (
            _video_hash(f"{ownership.namespace}|frozen_ar_crossfit", value),
            value,
        ),
    )
    count = min(int(fold_count), len(ordered))
    if count < 2:
        raise ValueError("AR cross-fitting requires at least two inner-training videos")
    buckets: list[list[str]] = [[] for _ in range(count)]
    for index, video in enumerate(ordered):
        buckets[index % count].append(video)
    all_inner_train = set(ownership.inner_train_videos)
    folds: list[ArCrossfitVideoFold] = []
    for fold_index, bucket in enumerate(buckets, start=1):
        prediction_videos = tuple(bucket)
        prediction_set = set(prediction_videos)
        fit_videos = tuple(sorted(all_inner_train - prediction_set))
        payload = {
            "ownership_digest": ownership.digest,
            "fold": fold_index,
            "fit_videos": fit_videos,
            "prediction_videos": prediction_videos,
        }
        folds.append(
            ArCrossfitVideoFold(
                fold=fold_index,
                fit_videos=fit_videos,
                prediction_videos=prediction_videos,
                digest=canonical_digest(payload),
            )
        )
    return tuple(folds)


def frozen_ar_prediction_identity(
    *,
    ownership: InnerVideoOwnership,
    outer_fold: int,
    target_name: str,
    seed: int,
    model_family: str,
    checkpoint_digest: str,
    train_prediction: np.ndarray,
    test_prediction: np.ndarray,
) -> Mapping[str, Any]:
    """Seal one neural AR member for exact reuse by real and control heads."""

    expected_namespace = member_inner_namespace(
        outer_fold=int(outer_fold), target_name=str(target_name), seed=int(seed)
    )
    if ownership.namespace != expected_namespace:
        raise ValueError("Frozen AR identity does not match target/fold/seed ownership")
    if not str(model_family) or not str(checkpoint_digest):
        raise ValueError("Frozen AR identity requires model-family and checkpoint provenance")
    train = np.asarray(train_prediction, dtype=np.float32)
    test = np.asarray(test_prediction, dtype=np.float32)
    if train.ndim != 1 or test.ndim != 1 or not len(train) or not len(test):
        raise ValueError("Frozen AR identity requires non-empty 1D predictions")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("Frozen AR identity cannot seal non-finite predictions")
    payload = {
        "outer_fold": int(outer_fold),
        "target_name": str(target_name),
        "seed": int(seed),
        "model_family": str(model_family),
        "checkpoint_digest": str(checkpoint_digest),
        "ownership_digest": ownership.digest,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_prediction_digest": array_digest(train),
        "test_prediction_digest": array_digest(test),
    }
    return {**payload, "identity_digest": canonical_digest(payload)}


def require_shared_frozen_ar_identity(
    lane_identities: Mapping[str, Mapping[str, Any]],
) -> str:
    """Fail unless every matched real/control lane reuses one frozen AR member."""

    if not lane_identities:
        raise ValueError("Frozen AR identity audit requires at least one lane")
    missing = [
        str(lane)
        for lane, identity in lane_identities.items()
        if not str(identity.get("identity_digest", ""))
    ]
    if missing:
        raise ValueError(f"Frozen AR identity digest missing for lanes: {missing}")
    digests = {str(identity["identity_digest"]) for identity in lane_identities.values()}
    if len(digests) != 1:
        raise RuntimeError("Matched real/control lanes do not share one frozen AR identity")
    return next(iter(digests))


def fit_frozen_ar_train_only(
    *,
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_target: np.ndarray,
    train_valid: np.ndarray,
    train_video_ids: np.ndarray,
    namespace: str | None = None,
    ownership: InnerVideoOwnership | None = None,
    alpha_grid: Sequence[float] = RIDGE_ALPHA_GRID,
    crossfit_folds: int = 5,
) -> FrozenArResult:
    """Fit an honest staged *reference Ridge* AR and final outer-test AR.

    This is a fast executable reference for validating the ownership and
    stacking contract.  It is deliberately **not** the promotable VEATIC 2.1
    baseline: the real development runner must train a target/fold/seed-
    specific neural AR under the same ownership scopes and reuse that member's
    exact frozen predictions across the real lane and all controls.

    Alpha selection and residual checkpoint selection share ``ownership``.
    Candidate AR coefficients and scalers are fit strictly on inner-training
    videos.  The selected candidate predicts inner-validation rows without a
    refit.  Inner-training rows are video-cross-fitted, so every value in
    ``train_prediction`` is out-of-video-fit and safe to use when constructing
    residual targets.  Only ``test_prediction`` uses the separate final model
    fit on all eligible outer-training rows.
    """

    train_array = np.asarray(train_x, dtype=np.float32)
    test_array = np.asarray(test_x, dtype=np.float32)
    target = np.asarray(train_target, dtype=np.float32)
    valid_input = np.asarray(train_valid, dtype=bool)
    videos = np.asarray(train_video_ids, dtype=str)
    if train_array.ndim != 2 or test_array.ndim != 2:
        raise ValueError("Frozen AR features must be 2D")
    if train_array.shape[1] != test_array.shape[1]:
        raise ValueError("Frozen AR train/test feature widths differ")
    if not (len(train_array) == len(target) == len(valid_input) == len(videos)):
        raise ValueError("Frozen AR outer-training arrays are not row aligned")
    if not np.isfinite(train_array).all() or not np.isfinite(test_array).all():
        raise ValueError("Frozen AR features must be finite")
    valid = valid_input & np.isfinite(target)
    eligible = np.flatnonzero(valid).astype(np.int64)
    if len(eligible) < 4:
        raise ValueError("Frozen AR requires at least four eligible rows")
    if ownership is None:
        if namespace is None:
            raise ValueError("Frozen AR requires a namespace or explicit inner ownership")
        ownership = build_inner_video_ownership(videos, namespace=namespace)
    elif namespace is not None and namespace != ownership.namespace:
        raise ValueError("Frozen AR namespace disagrees with explicit inner ownership")
    ownership_train_mask, ownership_val_mask = ownership.row_masks(videos)
    inner_train, inner_val = ownership.eligible_indices(videos, valid)
    if len(np.unique(videos[inner_train])) < 2:
        raise ValueError("Frozen AR needs at least two eligible inner-training videos")
    if not len(inner_val):
        raise ValueError("Frozen AR inner validation has no eligible rows")

    alphas = tuple(float(alpha) for alpha in alpha_grid)
    if not alphas or len(set(alphas)) != len(alphas):
        raise ValueError("Frozen AR alpha grid must be non-empty and unique")
    if any(not math.isfinite(alpha) or alpha <= 0.0 for alpha in alphas):
        raise ValueError("Frozen AR alpha values must be positive and finite")
    best_alpha: float | None = None
    best_mse = math.inf
    selection_rows: list[dict[str, Any]] = []
    candidate_fits: dict[float, _RidgeFit] = {}
    for alpha in alphas:
        fit = _fit_ridge(train_array, target, inner_train, alpha=alpha)
        prediction = _ridge_predict(fit, train_array[inner_val])
        mse = float(np.mean(np.square(prediction - target[inner_val])))
        selection_rows.append(
            {
                "alpha": alpha,
                "inner_validation_mse": mse,
                "fit_digest": fit.fit_digest,
                "coef_digest": array_digest(fit.coef),
                "intercept": fit.intercept,
                "inner_validation_prediction_digest": array_digest(prediction),
            }
        )
        candidate_fits[alpha] = fit
        if mse < best_mse:
            best_alpha = alpha
            best_mse = mse
    if best_alpha is None:
        raise RuntimeError("Frozen AR alpha selection failed")

    # Do not refit the selected model on inner validation before the residual
    # checkpoint is selected.  It supplies the honest inner-validation AR
    # predictions, while inner-training videos receive cross-fitted values.
    inner_fit = candidate_fits[best_alpha]
    train_prediction = np.full(len(train_array), np.nan, dtype=np.float32)
    train_prediction[ownership_val_mask] = _ridge_predict(
        inner_fit, train_array[ownership_val_mask]
    )
    crossfit_rows: list[dict[str, Any]] = []
    crossfit_scopes = build_ar_crossfit_video_folds(
        ownership,
        fold_count=int(crossfit_folds),
    )
    for scope in crossfit_scopes:
        held_out_set = set(scope.prediction_videos)
        fit_videos = set(scope.fit_videos)
        prediction_mask = ownership_train_mask & np.isin(videos, list(held_out_set))
        fit_mask = valid & np.isin(videos, list(fit_videos))
        fit_rows = np.flatnonzero(fit_mask).astype(np.int64)
        if not len(fit_rows) or not np.count_nonzero(prediction_mask):
            raise RuntimeError("Frozen AR cross-fit produced an empty fit or prediction side")
        crossfit_fit = _fit_ridge(train_array, target, fit_rows, alpha=best_alpha)
        fold_prediction = _ridge_predict(crossfit_fit, train_array[prediction_mask])
        train_prediction[prediction_mask] = fold_prediction
        crossfit_rows.append(
            {
                "fold": scope.fold,
                "fit_videos": sorted(fit_videos),
                "prediction_videos": list(scope.prediction_videos),
                "scope_digest": scope.digest,
                "fit_eligible_rows": int(len(fit_rows)),
                "prediction_rows": int(np.count_nonzero(prediction_mask)),
                "fit_digest": crossfit_fit.fit_digest,
                "mean": crossfit_fit.mean.astype(float).tolist(),
                "std": crossfit_fit.std.astype(float).tolist(),
                "coef": crossfit_fit.coef.astype(float).tolist(),
                "intercept": crossfit_fit.intercept,
                "prediction_digest": array_digest(fold_prediction),
            }
        )
    if not np.isfinite(train_prediction).all():
        raise RuntimeError("Frozen AR failed to produce complete honest outer-train predictions")

    # This final fit is intentionally separate and is used only for outer-test
    # baseline predictions.  Its predictions must never form residual labels.
    final_fit = _fit_ridge(train_array, target, eligible, alpha=best_alpha)
    test_prediction = _ridge_predict(final_fit, test_array)
    inner_val_prediction = train_prediction[ownership_val_mask]
    honest_train_digest = array_digest(train_prediction)
    test_digest = array_digest(test_prediction)
    eligible_split_payload = {
        "ownership_digest": ownership.digest,
        "inner_train_eligible_rows": inner_train.tolist(),
        "inner_validation_eligible_rows": inner_val.tolist(),
    }
    return FrozenArResult(
        train_prediction=train_prediction,
        test_prediction=test_prediction,
        selected_alpha=best_alpha,
        inner_validation_mse=best_mse,
        mean=final_fit.mean,
        std=final_fit.std,
        ownership=ownership,
        inner_mean=inner_fit.mean,
        inner_std=inner_fit.std,
        inner_coef=inner_fit.coef,
        inner_intercept=inner_fit.intercept,
        final_coef=final_fit.coef,
        final_intercept=final_fit.intercept,
        audit={
            "namespace": ownership.namespace,
            "model_family": "ridge_reference_contract_only",
            "promotable_veatic21_ar": False,
            "fit_scope": "honest_video_crossfit_for_residuals_plus_outer_train_only_final_test_fit",
            "ownership": ownership.audit_manifest(),
            "ownership_digest": ownership.digest,
            "eligible_split_digest": canonical_digest(eligible_split_payload),
            "selected_alpha": best_alpha,
            "alpha_selection": selection_rows,
            "eligible_train_rows": int(len(eligible)),
            "inner_train_eligible_rows": int(len(inner_train)),
            "inner_validation_eligible_rows": int(len(inner_val)),
            "inner_train_videos": int(len(np.unique(videos[inner_train]))),
            "inner_validation_videos": int(len(np.unique(videos[inner_val]))),
            "inner_fit_excludes_validation_videos": True,
            "inner_validation_rows_in_inner_fit": 0,
            "inner_validation_refit_before_residual_checkpoint_selection": False,
            "alpha_selected_on_common_inner_validation": True,
            "residual_targets_use_final_outer_fit": False,
            "all_outer_train_predictions_out_of_video_fit": True,
            "inner_mean": inner_fit.mean.astype(float).tolist(),
            "inner_std": inner_fit.std.astype(float).tolist(),
            "inner_coef": inner_fit.coef.astype(float).tolist(),
            "inner_intercept": inner_fit.intercept,
            "inner_fit_digest": inner_fit.fit_digest,
            "inner_validation_prediction_digest": array_digest(inner_val_prediction),
            "crossfit": crossfit_rows,
            "final_fit_scope": "all_eligible_outer_train_rows_for_outer_test_predictions_only",
            "final_mean": final_fit.mean.astype(float).tolist(),
            "final_std": final_fit.std.astype(float).tolist(),
            "final_coef": final_fit.coef.astype(float).tolist(),
            "final_intercept": final_fit.intercept,
            "final_fit_digest": final_fit.fit_digest,
            "train_prediction_digest": honest_train_digest,
            "test_prediction_digest": test_digest,
        },
    )


def whole_video_reassignment(
    values: np.ndarray,
    video_ids: np.ndarray,
    *,
    namespace: str,
) -> tuple[np.ndarray, Mapping[str, str]]:
    """Break video/label alignment without splitting temporal sequences."""

    data = np.asarray(values)
    videos = np.asarray(video_ids, dtype=str)
    unique = sorted(np.unique(videos), key=lambda value: int(value))
    row_counts = {video: int(np.sum(videos == video)) for video in unique}
    donors = sorted(
        unique,
        key=lambda value: (row_counts[value], _video_hash(namespace, str(value)), int(value)),
    )
    mapping = {video: donors[(index + 1) % len(donors)] for index, video in enumerate(donors)}
    if any(recipient == donor for recipient, donor in mapping.items()):
        raise RuntimeError("Whole-video control contains an identity assignment")
    output = np.empty_like(data)
    for recipient, donor in mapping.items():
        recipient_rows = np.flatnonzero(videos == recipient)
        donor_rows = np.flatnonzero(videos == donor)
        sample = np.rint(
            np.linspace(0, len(donor_rows) - 1, len(recipient_rows))
        ).astype(np.int64)
        output[recipient_rows] = data[donor_rows[sample]]
    return output, mapping


def train_matched_random_features(
    train_values: np.ndarray,
    test_values: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    """Generate a dimensionality- and scale-matched random feature control."""

    train = np.asarray(train_values, dtype=np.float32)
    test = np.asarray(test_values, dtype=np.float32)
    mean = np.mean(train, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(train, axis=0, dtype=np.float64).astype(np.float32)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    rng = np.random.default_rng(int(seed))
    random_train = rng.normal(size=train.shape).astype(np.float32) * std + mean
    random_test = rng.normal(size=test.shape).astype(np.float32) * std + mean
    return random_train, random_test, {
        "seed": int(seed),
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "train_digest": array_digest(random_train),
        "test_digest": array_digest(random_test),
    }


def top_lift(y_true: np.ndarray, prediction: np.ndarray, fraction: float) -> float:
    values = np.asarray(y_true, dtype=np.float64)
    scores = np.asarray(prediction, dtype=np.float64)
    count = max(1, int(math.ceil(len(values) * float(fraction))))
    chosen = np.argsort(-scores, kind="mergesort")[:count]
    return float(np.mean(values[chosen]) - np.mean(values))


def _event_pr_auc(
    train_values: np.ndarray,
    test_values: np.ndarray,
    prediction: np.ndarray,
) -> tuple[float, float, float]:
    finite_train = np.asarray(train_values)[np.isfinite(train_values)]
    if not len(finite_train):
        raise ValueError("Event threshold requires finite training targets")
    threshold = float(np.quantile(finite_train, EVENT_QUANTILE))
    events = np.asarray(test_values) >= threshold
    if len(np.unique(events)) < 2:
        raise ValueError("Event PR-AUC is undefined for a one-class evaluation panel")
    return float(average_precision_score(events, prediction)), threshold, float(np.mean(events))


def score_prediction(
    *,
    train_values: np.ndarray,
    test_values: np.ndarray,
    prediction: np.ndarray,
    time_seconds: np.ndarray,
) -> Mapping[str, Any]:
    target = np.asarray(test_values, dtype=np.float64)
    score = np.asarray(prediction, dtype=np.float64)
    times = np.asarray(time_seconds, dtype=np.float64)
    if not (len(target) == len(score) == len(times)) or not len(target):
        raise ValueError("Prediction scorer arrays are empty or misaligned")
    if not np.isfinite(target).all() or not np.isfinite(score).all():
        raise ValueError("Prediction scorer received non-finite values")
    pr_auc, threshold, prevalence = _event_pr_auc(train_values, target, score)
    first30 = times < FIRST_30_SECONDS
    first30_metrics: dict[str, float | None] = {
        "first30_pooled_continuous_spearman": None,
        "first30_top_5pct_true_future_movement_lift": None,
        "first30_training_q90_future_event_pr_auc": None,
    }
    if np.count_nonzero(first30) >= 4:
        first30_metrics["first30_pooled_continuous_spearman"] = float(
            continuous.spearman(target[first30], score[first30])
        )
        first30_metrics["first30_top_5pct_true_future_movement_lift"] = top_lift(
            target[first30], score[first30], 0.05
        )
        try:
            first30_metrics["first30_training_q90_future_event_pr_auc"] = _event_pr_auc(
                train_values, target[first30], score[first30]
            )[0]
        except ValueError:
            pass
    return {
        "pooled_continuous_spearman": float(continuous.spearman(target, score)),
        "top_1pct_true_future_movement_lift": top_lift(target, score, 0.01),
        "top_5pct_true_future_movement_lift": top_lift(target, score, 0.05),
        "top_10pct_true_future_movement_lift": top_lift(target, score, 0.10),
        "training_q90_future_event_pr_auc": pr_auc,
        "event_threshold_train_only": threshold,
        "event_prevalence": prevalence,
        "prediction_rows": int(len(score)),
        "prediction_finite_fraction": float(np.mean(np.isfinite(score))),
        "prediction_digest": array_digest(score.astype(np.float32)),
        **first30_metrics,
    }


def contract_manifest() -> Mapping[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "row_hz": ROW_HZ,
        "pca_width": PCA_WIDTH,
        "causal_window_rows": WINDOW_ROWS,
        "causal_window_seconds": (WINDOW_ROWS - 1) / ROW_HZ,
        "diagnostic_width": DIAGNOSTIC_WIDTH,
        "development_folds": DEVELOPMENT_FOLDS,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "reserved_video_count": RESERVED_VIDEO_COUNT,
        "event_quantile_train_only": EVENT_QUANTILE,
        "ridge_alpha_grid": list(RIDGE_ALPHA_GRID),
        "inner_video_ownership": "common_per_outer_fold_target_seed",
        "frozen_ar_residual_target_policy": "whole_video_crossfit",
        "frozen_ar_inner_validation_refit_before_checkpoint_selection": False,
        "frozen_ar_outer_test_policy": "separate_all_outer_train_final_fit",
        "promoted_ar_model_family": "target_fold_seed_specific_trained_neural_ar",
        "ridge_reference_ar_promotable": False,
        "matched_control_frozen_ar_identity_required": True,
        "privileged_ar_feature_contract": (
            "same_axis_current_lag1_lag2_lag4_and_current_minus_each_lag"
        ),
        "privileged_ar_context_policy": "full_lag4_context_required_no_median_fill",
        "historical_veatic_schema_authoritative": False,
        "again_methods_cherry_picked_not_phase_replayed": True,
        "cold_start_primary": True,
    }
