"""Explicit, portable replay of one published AGAIN fold/seed/lane."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error

from .checkpoints import score_gated_ar, score_temporal_residual
from .contracts import FUTURE_EVENT_TARGET, RESIDUAL_CONTINUOUS_TARGET, Protocol, TargetSpec
from .data import (
    AR_FEATURE_COLUMNS,
    add_redesigned_targets,
    add_targets_and_ar_features,
    ar_matrix,
    build_splits,
    causal_history,
)
from .metrics import top_fraction_lift

PUBLISHED_LANES = (
    "frozen_ar_only",
    "real_residual",
    "shuffled_pca_residual",
    "random_pca_residual",
    "label_permutation_residual",
    "train_only_video_mean_residual",
    "diagnostics_only_residual",
)


@dataclass(frozen=True)
class ReplayRoots:
    dense: Path
    pca: Path
    run: Path


@dataclass(frozen=True)
class ReplaySpec:
    protocol: Protocol
    fold: int
    seed: int
    lane: str
    split_target: str
    target_type: Literal["event", "continuous"]
    continuous_value_column: str
    continuous_transform: Literal["identity", "positive_delta"]
    labels: str
    diagnostics: str
    pca_scores: str
    pca_rows: str
    ar_checkpoint: str
    residual_checkpoint: str | None
    evidence_rows: str
    expected_sha256: dict[str, str]

    @classmethod
    def load(cls, path: Path) -> ReplaySpec:
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec = cls(**payload)
        if spec.lane not in PUBLISHED_LANES:
            raise ValueError(f"unsupported published lane: {spec.lane}")
        for value in (
            spec.labels,
            spec.diagnostics,
            spec.pca_scores,
            spec.pca_rows,
            spec.ar_checkpoint,
            spec.evidence_rows,
            spec.residual_checkpoint,
        ):
            if value is None:
                continue
            relative = Path(value)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("replay paths must be explicit root-relative paths")
        return spec


def _path(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_digest(values: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.ascontiguousarray(values).view(np.uint8))
    return digest.hexdigest()


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.nan_to_num(train.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    test = np.nan_to_num(test.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return ((train - mean) / std).astype(np.float32), ((test - mean) / std).astype(np.float32)


def _target(name: str) -> TargetSpec:
    targets = {
        FUTURE_EVENT_TARGET.name: FUTURE_EVENT_TARGET,
        RESIDUAL_CONTINUOUS_TARGET.name: RESIDUAL_CONTINUOUS_TARGET,
    }
    try:
        return targets[name]
    except KeyError as exc:
        raise ValueError(f"unsupported replay split target: {name}") from exc


def _load_rows(path: Path) -> pd.DataFrame:
    rows = pd.read_parquet(path).sort_values(["video_id", "row_index"]).reset_index(drop=True)
    if not set(AR_FEATURE_COLUMNS) <= set(rows.columns):
        rows = add_targets_and_ar_features(rows)
    rows, _ = add_redesigned_targets(rows)
    return rows


def _load_pca(
    score_path: Path, row_path: Path, train_idx: np.ndarray, test_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_table = pd.read_csv(row_path)
    observed_train = row_table.loc[row_table["split"] == "train", "row_id"].to_numpy(dtype=np.int64)
    observed_test = row_table.loc[row_table["split"] == "test", "row_id"].to_numpy(dtype=np.int64)
    expected = np.concatenate([train_idx, test_idx]).astype(np.int64)
    observed = np.concatenate([observed_train, observed_test])
    if not np.array_equal(np.sort(observed), np.sort(expected)):
        expected_only = len(np.setdiff1d(expected, observed))
        observed_only = len(np.setdiff1d(observed, expected))
        raise ValueError(
            "PCA row identities do not match the declared outer split: "
            f"expected={len(expected)}, observed={len(observed)}, "
            f"expected_only={expected_only}, observed_only={observed_only}"
        )
    scores = np.load(score_path, mmap_mode="r", allow_pickle=False)
    expected_rows = len(observed_train) + len(observed_test)
    if scores.shape != (expected_rows, 256) or scores.dtype != np.float32:
        raise ValueError(f"unexpected PCA score contract: {scores.shape}/{scores.dtype}")
    cutoff = len(observed_train)
    return (
        np.asarray(scores[:cutoff]),
        np.asarray(scores[cutoff:]),
        observed_train,
        observed_test,
    )


def _control_pca(
    train: np.ndarray,
    test: np.ndarray,
    train_video: np.ndarray,
    test_video: np.ndarray,
    *,
    lane: str,
    seed: int,
    fold: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if lane == "diagnostics_only_residual":
        return None, None
    rng = np.random.default_rng(seed + 1009 * (PUBLISHED_LANES.index(lane) + 1) + 17 * fold)
    if lane == "shuffled_pca_residual":
        return train[rng.permutation(len(train))], test[rng.permutation(len(test))]
    if lane == "random_pca_residual":
        return (
            rng.normal(0.0, 1.0, size=train.shape).astype(np.float32),
            rng.normal(0.0, 1.0, size=test.shape).astype(np.float32),
        )
    if lane == "train_only_video_mean_residual":
        means = {
            video: train[train_video == video].mean(axis=0) for video in np.unique(train_video)
        }
        fallback = train.mean(axis=0)
        return (
            np.vstack([means.get(video, fallback) for video in train_video]).astype(np.float32),
            np.vstack([means.get(video, fallback) for video in test_video]).astype(np.float32),
        )
    return train, test


def _representation(
    pca_train: np.ndarray,
    pca_test: np.ndarray,
    diagnostics: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    train_video: np.ndarray,
    test_video: np.ndarray,
    *,
    lane: str,
    seed: int,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    controlled_train, controlled_test = _control_pca(
        pca_train,
        pca_test,
        train_video,
        test_video,
        lane=lane,
        seed=seed,
        fold=fold,
    )
    diag_train = np.asarray(diagnostics[train_idx], dtype=np.float32)
    diag_test = np.asarray(diagnostics[test_idx], dtype=np.float32)
    if controlled_train is None or controlled_test is None:
        train_sequence, _ = causal_history(diag_train, train_idx, train_video, window_rows=5)
        test_sequence, _ = causal_history(diag_test, test_idx, test_video, window_rows=5)
        return (
            *_standardize(
                train_sequence.reshape(len(train_idx), -1), test_sequence.reshape(len(test_idx), -1)
            ),
            diag_train.shape[1],
        )
    train_sequence, _ = causal_history(controlled_train, train_idx, train_video, window_rows=5)
    test_sequence, _ = causal_history(controlled_test, test_idx, test_video, window_rows=5)
    train = np.concatenate([train_sequence.reshape(len(train_idx), -1), diag_train], axis=1)
    test = np.concatenate([test_sequence.reshape(len(test_idx), -1), diag_test], axis=1)
    train, test = _standardize(train, test)
    return train, test, controlled_train.shape[1]


def _metrics(
    y_true: np.ndarray,
    y_event: np.ndarray,
    score: np.ndarray,
    continuous: np.ndarray,
    *,
    target_type: str,
) -> dict[str, float]:
    pearson = float(np.corrcoef(y_true, continuous)[0, 1])
    result = {
        "continuous_pearson": pearson,
        "continuous_spearman": _published_spearman(y_true, continuous),
        "continuous_mae": float(mean_absolute_error(y_true, continuous)),
        "continuous_rmse": float(math.sqrt(mean_squared_error(y_true, continuous))),
        "continuous_bias": float(np.mean(continuous - y_true)),
    }
    if target_type == "event":
        result["pr_auc"] = float(average_precision_score(y_event, score))
        result["top_5pct_avg_true_movement_lift"] = top_fraction_lift(y_true, score, 0.05)
    else:
        result["binary_pr_auc"] = float(average_precision_score(y_event, score))
        result["binary_pr_auc_from_continuous_prediction"] = float(
            average_precision_score(y_event, continuous)
        )
        result["top_5pct_continuous_lift"] = top_fraction_lift(y_true, continuous, 0.05)
    return result


def _published_spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Reproduce AGAIN's stable ordinal-rank definition, including its tie policy."""

    def ordinal_rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=np.float64)
        ranks[order] = np.arange(len(values), dtype=np.float64)
        return ranks

    return float(np.corrcoef(ordinal_rank(left), ordinal_rank(right))[0, 1])


def replay(spec: ReplaySpec, roots: ReplayRoots) -> dict[str, object]:
    """Replay one exact member checkpoint and compare it with its published row."""

    paths = {
        "labels": _path(roots.dense, spec.labels),
        "diagnostics": _path(roots.dense, spec.diagnostics),
        "pca_scores": _path(roots.pca, spec.pca_scores),
        "pca_rows": _path(roots.pca, spec.pca_rows),
        "ar_checkpoint": _path(roots.run, spec.ar_checkpoint),
        "evidence_rows": _path(roots.run, spec.evidence_rows),
    }
    if spec.residual_checkpoint is not None:
        paths["residual_checkpoint"] = _path(roots.run, spec.residual_checkpoint)
    for name, expected in spec.expected_sha256.items():
        if name not in paths or _sha256(paths[name]) != expected:
            raise ValueError(f"artifact checksum mismatch: {name}")

    rows = _load_rows(paths["labels"])
    splits = build_splits(
        rows, target=_target(spec.split_target), protocols=(spec.protocol,), n_splits=5
    )
    split = next((item for item in splits if item.fold == spec.fold), None)
    if split is None:
        raise ValueError("declared split was not reconstructed")
    pca_train, pca_test, artifact_train_idx, artifact_test_idx = _load_pca(
        paths["pca_scores"], paths["pca_rows"], split.train_idx, split.test_idx
    )
    target = _target(spec.split_target)
    values = rows[target.value_column].to_numpy(dtype=np.float64)
    transformed = np.maximum(values, 0.0) if target.transform == "positive_delta" else values
    threshold = float(np.quantile(transformed[artifact_train_idx], target.quantile))
    split = replace(
        split,
        train_idx=artifact_train_idx,
        test_idx=artifact_test_idx,
        train_y=(transformed[artifact_train_idx] >= threshold).astype(np.int8),
        test_y=(transformed[artifact_test_idx] >= threshold).astype(np.int8),
        threshold=threshold,
    )
    if spec.protocol == "grouped_video":
        train_groups = set(rows.loc[split.train_idx, "video_id"].astype(str))
        test_groups = set(rows.loc[split.test_idx, "video_id"].astype(str))
        if train_groups & test_groups:
            raise ValueError("published grouped split leaks video identities")
    train_video = rows.loc[split.train_idx, "video_id"].astype(str).to_numpy()
    test_video = rows.loc[split.test_idx, "video_id"].astype(str).to_numpy()
    ar_train, ar_test = _standardize(
        ar_matrix(rows, split.train_idx), ar_matrix(rows, split.test_idx)
    )
    ar_train_score, ar_train_continuous = score_gated_ar(ar_train, paths["ar_checkpoint"])
    ar_test_score, ar_test_continuous = score_gated_ar(ar_test, paths["ar_checkpoint"])

    if spec.lane == "frozen_ar_only" or spec.residual_checkpoint is None:
        train_score, test_score = ar_train_score, ar_test_score
        test_continuous = ar_test_continuous
        sequence_channels = 0
    else:
        diagnostics = np.load(paths["diagnostics"], mmap_mode="r", allow_pickle=False)
        if diagnostics.shape[0] != len(rows) or diagnostics.dtype != np.float32:
            raise ValueError("diagnostic feature cache is not aligned to the dense table")
        train_x, test_x, sequence_channels = _representation(
            pca_train,
            pca_test,
            diagnostics,
            split.train_idx,
            split.test_idx,
            train_video,
            test_video,
            lane=spec.lane,
            seed=spec.seed,
            fold=spec.fold,
        )
        train_score, _, _ = score_temporal_residual(
            train_x,
            ar_train_score,
            ar_train_continuous,
            paths["residual_checkpoint"],
            sequence_window=5,
            sequence_channels=sequence_channels,
            target_type=spec.target_type,
        )
        test_score, test_continuous, _ = score_temporal_residual(
            test_x,
            ar_test_score,
            ar_test_continuous,
            paths["residual_checkpoint"],
            sequence_window=5,
            sequence_channels=sequence_channels,
            target_type=spec.target_type,
        )

    true_continuous = rows.loc[split.test_idx, spec.continuous_value_column].to_numpy(
        dtype=np.float32
    )
    if spec.continuous_transform == "positive_delta":
        true_continuous = np.maximum(true_continuous, 0.0)
    observed = _metrics(
        true_continuous,
        split.test_y,
        test_score,
        test_continuous,
        target_type=spec.target_type,
    )
    evidence = pd.read_csv(paths["evidence_rows"])
    lane_column = "lane" if "lane" in evidence else "control_type"
    mask = (evidence["seed"] == spec.seed) & (evidence[lane_column] == spec.lane)
    if "row_type" in evidence:
        mask &= evidence["row_type"] == "member"
    if "fold" in evidence:
        mask &= evidence["fold"] == spec.fold
    if mask.sum() != 1:
        raise ValueError("published evidence row identity is not unique")
    published = evidence.loc[mask].iloc[0]
    differences = {
        name: abs(value - float(published[name]))
        for name, value in observed.items()
        if name in evidence
    }
    if not differences:
        raise ValueError("published evidence row has no comparable metrics")
    return {
        "protocol": spec.protocol,
        "fold": spec.fold,
        "seed": spec.seed,
        "lane": spec.lane,
        "split_target": spec.split_target,
        "continuous_value_column": spec.continuous_value_column,
        "continuous_transform": spec.continuous_transform,
        "rows": {"train": len(split.train_idx), "test": len(split.test_idx)},
        "ar_train_digest": _array_digest(ar_train_score),
        "ar_test_digest": _array_digest(ar_test_score),
        "sequence_channels": sequence_channels,
        "observed": observed,
        "absolute_difference": differences,
        "max_absolute_difference": max(differences.values()),
    }
