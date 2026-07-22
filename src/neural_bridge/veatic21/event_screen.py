"""Benchmark-train-only VEATIC event target and representation screening.

This stage uses labels fully inside the preregistered benchmark-train pool. It never accepts
benchmark-test videos, and every scaler/PCA/model is fitted inside the current grouped fold.
AR is a strong comparator, not a production input; primary lanes are video-only.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .contracts import FeatureRows, LabelRows, TargetSpec
from .evidence import atomic_write_json, average_precision_skill, digest_json, pooled_pr_auc
from .protocol import (
    assert_row_alignment,
    event_labels,
    fit_event_threshold,
    future_target_values,
    target_support_mask,
)

_ALPHAS = (0.1, 1.0, 10.0, 100.0)
_VARIANCE_TARGET = 0.95
_MAX_COMPACT_COMPONENTS = 512
_MAX_CORTICAL_COMPONENTS = 256
_BATCH_ROWS = 1_024


@dataclass(frozen=True)
class FittedProjection:
    scaler: StandardScaler
    pca: PCA | IncrementalPCA
    width: int
    variance_reached: float

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        output = np.empty((len(matrix), self.width), dtype=np.float32)
        for start in range(0, len(matrix), _BATCH_ROWS):
            stop = min(start + _BATCH_ROWS, len(matrix))
            scaled = self.scaler.transform(matrix[start:stop])
            output[start:stop] = self.pca.transform(scaled)[:, : self.width]
        return output


def targets_from_calibration(calibration: Mapping[str, Any]) -> tuple[TargetSpec, ...]:
    if calibration.get("schema") != "veatic21_event_calibration_v12":
        raise ValueError("unsupported event calibration schema")
    if calibration.get("benchmark_test_labels_accessed") is not False:
        raise ValueError("event screen requires sealed benchmark-test labels")
    output = []
    for row in calibration["target_hypotheses"]:
        label = row["label"]
        transform = row["transform"]
        if label not in {"arousal", "valence"}:
            raise ValueError(f"unsupported calibrated label: {label!r}")
        if transform not in {"absolute", "positive"}:
            raise ValueError(f"unsupported calibrated transform: {transform!r}")
        output.append(
            TargetSpec(
                name=str(row["name"]),
                label=label,
                horizon_rows=tuple(int(value) for value in row["horizon_rows"]),
                quantile=float(row["train_quantile"]),
                transform=transform,
            )
        )
    for target in output:
        target.validate()
    return tuple(output)


def _batches(indices: np.ndarray, minimum: int = 1) -> Iterable[np.ndarray]:
    if not len(indices):
        return
    batch_count = max(1, math.ceil(len(indices) / _BATCH_ROWS))
    while batch_count > 1 and len(indices) // batch_count < minimum:
        batch_count -= 1
    yield from np.array_split(indices, batch_count)


def _fit_projection(
    matrix: np.ndarray,
    train_mask: np.ndarray,
    *,
    source: str,
    seed: int,
) -> FittedProjection:
    indices = np.flatnonzero(train_mask)
    if not len(indices):
        raise ValueError("projection training scope is empty")
    maximum = min(
        _MAX_CORTICAL_COMPONENTS if source == "tribe_cortical" else _MAX_COMPACT_COMPONENTS,
        len(indices) - 1,
        matrix.shape[1],
    )
    if maximum <= 0:
        raise ValueError("projection training scope cannot support PCA")
    scaler = StandardScaler()
    for rows in _batches(indices):
        scaler.partial_fit(matrix[rows])
    if source == "tribe_cortical":
        pca: PCA | IncrementalPCA = IncrementalPCA(
            n_components=maximum,
            batch_size=max(_BATCH_ROWS, maximum),
        )
        for rows in _batches(indices, maximum):
            pca.partial_fit(scaler.transform(matrix[rows]))
    else:
        pca = PCA(n_components=maximum, svd_solver="randomized", random_state=seed)
        pca.fit(scaler.transform(matrix[indices]))
    explained = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    cumulative = np.cumsum(explained)
    width = min(maximum, int(np.searchsorted(cumulative, _VARIANCE_TARGET) + 1))
    return FittedProjection(
        scaler=scaler,
        pca=pca,
        width=width,
        variance_reached=float(cumulative[width - 1]),
    )


def _causal_forms(
    projected: np.ndarray,
    video_id: np.ndarray,
    row_index: np.ndarray,
    windows: Sequence[int],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build causal projected forms without crossing videos."""

    forms: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "current": (projected, np.ones(len(projected), dtype=np.bool_))
    }
    videos = video_id.astype(str)
    for window in sorted(set(windows)):
        if window <= 0:
            raise ValueError("temporal windows must be positive")
        mean = np.zeros_like(projected)
        delta = np.zeros_like(projected)
        difference = np.zeros_like(projected)
        mean_available = np.zeros(len(projected), dtype=np.bool_)
        delta_available = np.zeros(len(projected), dtype=np.bool_)
        for video in np.unique(videos):
            positions = np.flatnonzero(videos == video)
            positions = positions[np.argsort(row_index[positions])]
            values = projected[positions]
            rows = row_index[positions]
            lookup = {int(row): offset for offset, row in enumerate(rows)}
            prefix = np.vstack(
                [np.zeros((1, projected.shape[1]), dtype=np.float32), np.cumsum(values, axis=0)]
            )
            for local, (position, row) in enumerate(zip(positions, rows, strict=True)):
                start_row = int(row) - window + 1
                if start_row in lookup:
                    start = lookup[start_row]
                    if local - start + 1 == window:
                        mean[position] = (prefix[local + 1] - prefix[start]) / window
                        mean_available[position] = True
                past_row = int(row) - window
                if past_row in lookup:
                    past = lookup[past_row]
                    if local - past == window:
                        history_mean = (prefix[local] - prefix[past]) / window
                        delta[position] = values[local] - history_mean
                        difference[position] = values[local] - values[past]
                        delta_available[position] = True
        forms[f"causal_mean_w{window}"] = (mean, mean_available)
        forms[f"current_minus_past_mean_w{window}"] = (delta, delta_available)
        forms[f"first_difference_w{window}"] = (difference, delta_available.copy())
    return forms


def _ar_features(labels: LabelRows, maximum_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Strong current-plus-history baseline calculated from VEATIC labels."""

    videos = labels.video_id.astype(str)
    current = labels.arousal.astype(np.float64)
    columns = [current[:, None]]
    all_available = np.ones(len(labels.video_id), dtype=np.bool_)
    for lag in range(1, maximum_lag + 1):
        past = np.zeros(len(current), dtype=np.float64)
        available = np.zeros(len(current), dtype=np.bool_)
        for video in np.unique(videos):
            positions = np.flatnonzero(videos == video)
            lookup = {int(labels.row_index[position]): position for position in positions}
            for position in positions:
                previous = lookup.get(int(labels.row_index[position]) - lag)
                if previous is not None:
                    past[position] = current[previous]
                    available[position] = True
        columns.extend(
            [
                past[:, None],
                available.astype(np.float64)[:, None],
                np.where(available, current - past, 0.0)[:, None],
            ]
        )
        all_available &= available
    return np.concatenate(columns, axis=1), all_available


def _ridge_scores(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    train_y: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    model = Ridge(alpha=alpha, solver="lsqr", tol=1e-6, max_iter=10_000)
    model.fit(train_scaled, train_y)
    return model.predict(scaler.transform(validation_x)).astype(np.float64)


def run_event_target_screen(
    features: FeatureRows,
    labels: LabelRows,
    preregistration: Mapping[str, Any],
    calibration: Mapping[str, Any],
    *,
    sources: Sequence[str],
) -> dict[str, Any]:
    """Screen VEATIC targets and direct video representations on benchmark-train folds."""

    features.validate()
    labels.validate()
    assert_row_alignment(features, labels)
    split = preregistration["split"]
    expected_videos = set(split["video_ids"])
    if (
        set(labels.video_id.astype(str)) != expected_videos
        or len(labels.video_id) != int(split["benchmark_train_rows"])
        or not features.quality_eligible.all()
    ):
        raise ValueError("event screen may access exactly the eligible benchmark-train rows")
    if set(sources) != set(features.representations):
        raise ValueError("loaded representations must exactly match declared screen sources")
    if calibration.get("preregistration_sha256") != preregistration.get(
        "preregistration_sha256"
    ):
        raise ValueError("calibration does not belong to this preregistration")
    targets = targets_from_calibration(calibration)
    movement_rows = tuple(int(value) for value in calibration["movement_curve"]["milestone_rows"])
    maximum_lag = max(movement_rows)
    ar_matrix, _ = _ar_features(labels, maximum_lag)
    target_groups: dict[tuple[str, tuple[int, ...], str], list[TargetSpec]] = {}
    for target in targets:
        key = (target.label, target.horizon_rows, target.transform)
        target_groups.setdefault(key, []).append(target)
    group_values = {
        key: future_target_values(labels, grouped_targets[0])
        for key, grouped_targets in target_groups.items()
    }
    group_support = {
        key: target_support_mask(features, grouped_targets[0])
        for key, grouped_targets in target_groups.items()
    }
    records: list[dict[str, Any]] = []
    projection_audits: list[dict[str, Any]] = []
    videos = features.video_id.astype(str)
    grouped_folds = split["inner_grouped_video_folds"]

    for fold, validation_videos in enumerate(grouped_folds):
        validation_video_mask = np.isin(videos, validation_videos)
        train_video_mask = ~validation_video_mask
        projection_train = train_video_mask & features.quality_eligible

        ar_metrics: dict[tuple[str, float], dict[str, Any]] = {}
        for key, grouped_targets in target_groups.items():
            train_mask = projection_train & group_support[key]
            validation_mask = (
                validation_video_mask & features.quality_eligible & group_support[key]
            )
            for alpha in _ALPHAS:
                score = _ridge_scores(
                    ar_matrix[train_mask],
                    ar_matrix[validation_mask],
                    group_values[key][train_mask],
                    alpha=alpha,
                )
                for target in grouped_targets:
                    threshold = fit_event_threshold(group_values[key], train_mask, target)
                    binary = event_labels(group_values[key], threshold)
                    ar_metrics[(target.name, alpha)] = {
                        "pr_auc": pooled_pr_auc(binary[validation_mask], score),
                        "skill": average_precision_skill(binary[validation_mask], score),
                        "threshold": threshold,
                        "binary": binary,
                    }

        for source in sources:
            matrix = features.representations[source]
            projection = _fit_projection(
                matrix,
                projection_train,
                source=source,
                seed=int(preregistration["split"]["seed"]) + fold,
            )
            projected = projection.transform(matrix)
            forms = _causal_forms(
                projected,
                features.video_id,
                features.row_index,
                movement_rows,
            )
            forms = {
                name: value
                for name, value in forms.items()
                if name == "current" or name.startswith("current_minus_past_mean")
            }
            projection_audits.append(
                {
                    "fold": fold,
                    "source": source,
                    "fit_rows": int(projection_train.sum()),
                    "maximum_components": int(
                        len(np.asarray(projection.pca.explained_variance_ratio_))
                    ),
                    "selected_width": projection.width,
                    "variance_reached": projection.variance_reached,
                    "fit_scope": "benchmark_train_fold_fit_quality_eligible_only",
                }
            )
            for key, grouped_targets in target_groups.items():
                train_mask = projection_train & group_support[key]
                validation_mask = (
                    validation_video_mask & features.quality_eligible & group_support[key]
                )
                for alpha in _ALPHAS:
                    for form_name, (form_values, form_available) in forms.items():
                        form_train = train_mask & form_available
                        form_validation = validation_mask & form_available
                        score = _ridge_scores(
                            form_values[form_train],
                            form_values[form_validation],
                            group_values[key][form_train],
                            alpha=alpha,
                        )
                        for target in grouped_targets:
                            baseline = ar_metrics[(target.name, alpha)]
                            binary = baseline["binary"]
                            pr_auc = pooled_pr_auc(binary[form_validation], score)
                            skill = average_precision_skill(binary[form_validation], score)
                            records.append(
                                {
                                    "fold": fold,
                                    "target": target.name,
                                    "source": source,
                                    "form": form_name,
                                    "pca_variance_target": _VARIANCE_TARGET,
                                    "pca_width": projection.width,
                                    "ridge_alpha": alpha,
                                    "train_rows": int(form_train.sum()),
                                    "validation_rows": int(form_validation.sum()),
                                    "threshold": baseline["threshold"],
                                    "event_prevalence": float(
                                        np.mean(binary[form_validation])
                                    ),
                                    "pooled_pr_auc": pr_auc,
                                    "average_precision_skill": skill,
                                    "ar_pr_auc": baseline["pr_auc"],
                                    "ar_average_precision_skill": baseline["skill"],
                                    "skill_delta_vs_ar": skill - baseline["skill"],
                                }
                            )

    result: dict[str, Any] = {
        "schema": "veatic21_event_target_screen_v12",
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "calibration_sha256": calibration["calibration_sha256"],
        "benchmark_test_labels_accessed": False,
        "benchmark_train_video_count": len(expected_videos),
        "benchmark_train_row_count": len(labels.video_id),
        "sources": list(sources),
        "target_count": len(targets),
        "fold_count": len(grouped_folds),
        "alphas": list(_ALPHAS),
        "pca_variance_target": _VARIANCE_TARGET,
        "projection_audits": projection_audits,
        "records": records,
        "selection_status": "benchmark_train_diagnostic_not_frozen",
        "production_input": "video_features_only_ar_is_comparator",
    }
    result["screen_sha256"] = digest_json(result)
    return result


def write_event_target_screen(path: Path, result: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(result))


__all__ = [
    "run_event_target_screen",
    "targets_from_calibration",
    "write_event_target_screen",
]
