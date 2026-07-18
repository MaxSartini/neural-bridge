"""One public runner for coherent, apples-to-apples AGAIN fold evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .contracts import SPIKE_TARGET, FoldData, FrozenArScores, Protocol, TargetSpec
from .data import ar_matrix, build_splits, fold_safe_pca, inner_validation
from .metrics import continuous_metrics, event_metrics, top_fraction_lift
from .models import (
    ResidualConfig,
    ResidualTrainingData,
    fit_frozen_ar,
    fit_ridge_residual,
    train_residual_head,
)

CONTROLS = ("real", "shuffled", "random", "train_video_mean")
Head = Literal["ridge_sanity", "learned_residual"]


@dataclass(frozen=True)
class RunConfig:
    target: TargetSpec = SPIKE_TARGET
    protocols: tuple[Protocol, ...] = ("grouped_video", "blocked_temporal_70_30")
    n_splits: int = 5
    pca_width: int | None = None
    seeds: tuple[int, ...] = (20260625, 20260626, 20260627)
    controls: tuple[str, ...] = CONTROLS
    ridge_alpha: float = 10.0
    head: Head = "ridge_sanity"
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    checkpoint_ensembles: tuple[tuple[int, ...], ...] = ()


def _digest_ar(ar: FrozenArScores) -> str:
    digest = hashlib.sha256()
    for values in (ar.train_score, ar.test_score, ar.train_continuous, ar.test_continuous):
        digest.update(np.ascontiguousarray(values).view(np.uint8))
    return digest.hexdigest()


def _control(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_video: np.ndarray,
    test_video: np.ndarray,
    *,
    name: str,
    seed: int,
    diagnostics_train: np.ndarray | None = None,
    diagnostics_test: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if name == "real":
        return train_x, test_x
    if name == "shuffled":
        return train_x[rng.permutation(len(train_x))], test_x[rng.permutation(len(test_x))]
    if name == "random":
        return (
            rng.normal(size=train_x.shape).astype(np.float32),
            rng.normal(size=test_x.shape).astype(np.float32),
        )
    if name == "train_video_mean":
        means = {
            video: train_x[train_video == video].mean(axis=0) for video in np.unique(train_video)
        }
        global_mean = train_x.mean(axis=0)
        return (
            np.vstack([means.get(video, global_mean) for video in train_video]).astype(np.float32),
            np.vstack([means.get(video, global_mean) for video in test_video]).astype(np.float32),
        )
    if name == "diagnostics_only":
        if diagnostics_train is None or diagnostics_test is None:
            raise ValueError("diagnostics_only control requires diagnostic matrices")
        return diagnostics_train, diagnostics_test
    if name == "label_permutation":
        return train_x, test_x
    raise ValueError(f"unknown control: {name}")


def _score_result(
    fold: FoldData,
    *,
    train_score: np.ndarray,
    test_score: np.ndarray,
    test_continuous: np.ndarray,
    ranking_score: np.ndarray | None = None,
) -> dict[str, float | int]:
    result = event_metrics(fold.split.train_y, train_score, fold.split.test_y, test_score)
    result.update(
        {
            f"continuous_{key}": value
            for key, value in continuous_metrics(fold.test_continuous, test_continuous).items()
        }
    )
    result["top_5pct_continuous_lift"] = top_fraction_lift(
        fold.test_continuous,
        test_score if ranking_score is None else ranking_score,
        0.05,
    )
    return result


def evaluate_fold_sanity(
    fold: FoldData, config: RunConfig, *, seed: int
) -> list[dict[str, object]]:
    """Compare the same frozen AR against real and matched-control ridge residuals."""

    fold.validate()
    ar_digest = _digest_ar(fold.frozen_ar)
    rows: list[dict[str, object]] = []
    for offset, control in enumerate(config.controls):
        train_x, test_x = _control(
            fold.train_x,
            fold.test_x,
            fold.train_video_id,
            fold.test_video_id,
            name=control,
            seed=seed + 1009 * (offset + 1),
        )
        train_score, test_score = fit_ridge_residual(
            train_x,
            test_x,
            fold.split.train_y,
            fold.frozen_ar.train_score,
            fold.frozen_ar.test_score,
            alpha=config.ridge_alpha,
        )
        train_reg, test_reg = fit_ridge_residual(
            train_x,
            test_x,
            fold.train_continuous,
            fold.frozen_ar.train_continuous,
            fold.frozen_ar.test_continuous,
            alpha=config.ridge_alpha,
        )
        if _digest_ar(fold.frozen_ar) != ar_digest:
            raise RuntimeError("a residual lane mutated the frozen AR scores")
        result: dict[str, object] = {
            "protocol": fold.split.protocol,
            "fold": fold.split.fold,
            "target": fold.split.target.name,
            "target_threshold_train_only": fold.split.threshold,
            "seed": seed,
            "lane": control,
            "head": "ridge_residual_sanity",
            "frozen_ar_sha256": ar_digest,
        }
        result.update(
            _score_result(
                fold,
                train_score=train_score,
                test_score=test_score,
                test_continuous=test_reg,
            )
        )
        rows.append(result)
    return rows


def evaluate_prepared_fold(
    folds_by_seed: Mapping[int, FoldData],
    config: RunConfig,
    *,
    inner_train: np.ndarray,
    inner_val: np.ndarray,
    checkpoint_root: Path,
    target_type: Literal["event", "continuous"] = "event",
) -> pd.DataFrame:
    """Train matched heads with each seed's own target-specific frozen AR."""

    if config.head != "learned_residual":
        raise ValueError("evaluate_prepared_fold requires head='learned_residual'")
    if set(folds_by_seed) != set(config.seeds):
        raise ValueError("fold bundles must match the prospectively declared seed set exactly")
    ensemble_seeds = [seed for group in config.checkpoint_ensembles for seed in group]
    if len(ensemble_seeds) != len(set(ensemble_seeds)) or not set(ensemble_seeds) <= set(
        config.seeds
    ):
        raise ValueError("checkpoint ensembles must be disjoint groups of declared seeds")
    reference = folds_by_seed[config.seeds[0]]
    reference.validate()
    for seed, fold in folds_by_seed.items():
        fold.validate()
        if not np.array_equal(
            fold.split.train_idx, reference.split.train_idx
        ) or not np.array_equal(fold.split.test_idx, reference.split.test_idx):
            raise ValueError(f"seed {seed} uses a different outer split")
        for name in (
            "train_x",
            "test_x",
            "train_continuous",
            "test_continuous",
            "train_video_id",
            "test_video_id",
        ):
            if not np.array_equal(getattr(fold, name), getattr(reference, name)):
                raise ValueError(f"seed {seed} changes non-AR fold data: {name}")

    rows: list[dict[str, object]] = []
    ensembles: dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for lane_offset, lane in enumerate(config.controls):
        ensembles[lane] = {}
        for seed in config.seeds:
            fold = folds_by_seed[seed]
            ar_digest = _digest_ar(fold.frozen_ar)
            label_policy = "true_outer_train_labels"
            if lane == "frozen_ar_only":
                train_score = fold.frozen_ar.train_score
                test_score = fold.frozen_ar.test_score
                test_continuous = fold.frozen_ar.test_continuous
                backend = "frozen"
                best_epoch = 0
                checkpoint_path = None
                training_audit: dict[str, object] = {"residual_trained": False}
            else:
                train_x, test_x = _control(
                    fold.train_x,
                    fold.test_x,
                    fold.train_video_id,
                    fold.test_video_id,
                    name=lane,
                    seed=seed + 1009 * (lane_offset + 1),
                    diagnostics_train=fold.diagnostics_train,
                    diagnostics_test=fold.diagnostics_test,
                )
                train_y = fold.split.train_y.copy()
                train_continuous = fold.train_continuous.copy()
                if lane == "label_permutation":
                    rng = np.random.default_rng(seed + 503 * (lane_offset + 1))
                    permutation = rng.permutation(len(train_y))
                    train_y = train_y[permutation]
                    train_continuous = train_continuous[permutation]
                    label_policy = "permuted_outer_train_and_inner_selection_labels"
                training = ResidualTrainingData(
                    train_x=train_x,
                    test_x=test_x,
                    train_y=train_y,
                    train_continuous=train_continuous,
                    ar_train_score=fold.frozen_ar.train_score,
                    ar_test_score=fold.frozen_ar.test_score,
                    ar_train_continuous=fold.frozen_ar.train_continuous,
                    ar_test_continuous=fold.frozen_ar.test_continuous,
                    inner_train=inner_train,
                    inner_val=inner_val,
                    target_type=target_type,
                )
                result = train_residual_head(
                    training,
                    config.residual,
                    seed=seed,
                    checkpoint_dir=(
                        checkpoint_root
                        / fold.split.target.name
                        / fold.split.protocol
                        / f"fold-{fold.split.fold}"
                        / lane
                    ),
                )
                train_score = result.train_score
                test_score = result.test_score
                test_continuous = result.test_continuous
                backend = result.backend
                best_epoch = result.best_epoch
                checkpoint_path = result.checkpoint_path
                training_audit = result.audit
            if _digest_ar(fold.frozen_ar) != ar_digest:
                raise RuntimeError("a learned lane mutated the frozen AR scores")
            row: dict[str, object] = {
                "protocol": fold.split.protocol,
                "fold": fold.split.fold,
                "target": fold.split.target.name,
                "target_threshold_train_only": fold.split.threshold,
                "seed": seed,
                "lane": lane,
                "head": config.residual.architecture,
                "backend": backend,
                "label_policy": label_policy,
                "frozen_ar_sha256": ar_digest,
                "best_epoch": best_epoch,
                "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                "training_audit": training_audit,
            }
            row.update(
                _score_result(
                    fold,
                    train_score=train_score,
                    test_score=test_score,
                    test_continuous=test_continuous,
                    ranking_score=test_continuous if target_type == "continuous" else test_score,
                )
            )
            rows.append(row)
            ensembles[lane][seed] = (train_score, test_score, test_continuous)

        for group_index, group in enumerate(config.checkpoint_ensembles, start=1):
            members = [ensembles[lane][seed] for seed in group]
            train_score = np.mean([member[0] for member in members], axis=0)
            test_score = np.mean([member[1] for member in members], axis=0)
            test_continuous = np.mean([member[2] for member in members], axis=0)
            row = {
                "protocol": reference.split.protocol,
                "fold": reference.split.fold,
                "target": reference.split.target.name,
                "target_threshold_train_only": reference.split.threshold,
                "seed": f"declared_ensemble_{group_index}",
                "ensemble_seeds": list(group),
                "lane": lane,
                "head": config.residual.architecture,
                "backend": config.residual.backend,
                "frozen_ar_sha256_by_seed": {
                    seed: _digest_ar(folds_by_seed[seed].frozen_ar) for seed in config.seeds
                },
            }
            row.update(
                _score_result(
                    reference,
                    train_score=train_score,
                    test_score=test_score,
                    test_continuous=test_continuous,
                    ranking_score=test_continuous if target_type == "continuous" else test_score,
                )
            )
            rows.append(row)
    return pd.DataFrame(rows)


def run_sanity_benchmark(
    rows: pd.DataFrame,
    representation: np.ndarray,
    *,
    config: RunConfig | None = None,
) -> pd.DataFrame:
    """Run the freshly trained linear sanity benchmark through one explicit API."""

    config = config or RunConfig()
    if config.head != "ridge_sanity":
        raise ValueError(
            "learned heads require explicit FoldData provenance; call evaluate_prepared_fold"
        )
    if len(rows) != len(representation):
        raise ValueError("row table and representation lengths differ")
    output: list[dict[str, object]] = []
    for split in build_splits(
        rows, target=config.target, protocols=config.protocols, n_splits=config.n_splits
    ):
        inner_train, inner_val, strategy = inner_validation(rows, split)
        train_x = representation[split.train_idx]
        test_x = representation[split.test_idx]
        pca_audit: dict[str, object] = {"fit_scope": "not_requested"}
        if config.pca_width is not None:
            train_x, test_x, pca_audit = fold_safe_pca(
                train_x,
                test_x,
                width=config.pca_width,
                seed=config.seeds[0] + split.fold,
            )
        ar_train, ar_test = ar_matrix(rows, split.train_idx), ar_matrix(rows, split.test_idx)
        continuous = rows[config.target.value_column].to_numpy(dtype=np.float32)
        frozen_ar = fit_frozen_ar(
            ar_train,
            ar_test,
            split,
            continuous[split.train_idx],
            inner_train=inner_train,
            inner_val=inner_val,
            selection_strategy=strategy,
        )
        fold = FoldData(
            split=split,
            train_x=train_x,
            test_x=test_x,
            train_continuous=continuous[split.train_idx],
            test_continuous=continuous[split.test_idx],
            train_video_id=rows.loc[split.train_idx, "video_id"].astype(str).to_numpy(),
            test_video_id=rows.loc[split.test_idx, "video_id"].astype(str).to_numpy(),
            frozen_ar=frozen_ar,
        )
        for seed in config.seeds:
            fold_rows = evaluate_fold_sanity(fold, config, seed=seed)
            for result in fold_rows:
                result["pca"] = pca_audit
                result["ar_provenance"] = frozen_ar.provenance
            output.extend(fold_rows)
    return pd.DataFrame(output)
