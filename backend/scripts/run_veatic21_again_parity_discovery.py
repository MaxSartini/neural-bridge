#!/usr/bin/env python3
"""VEATIC-only inner discovery using the proven AGAIN dual-task method.

"AGAIN parity" describes the training algorithm only.  This executor rejects
all fitted cross-dataset reuse: inputs, labels, PCA parents, AR scores,
normalizers, checkpoints, and weights are freshly VEATIC-scoped.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score

from backend.scripts import run_again_dense_2hz_phase5_learned_heads as base
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_blocked as again_temporal
from backend.scripts import run_veatic21_endstate as runner
from backend.scripts import veatic21_discovery as discovery
from backend.scripts import veatic21_distilled_program as program
from backend.scripts import veatic21_execution as execution
from backend.scripts import veatic21_modeling as modeling


SCHEMA_VERSION = "veatic21_again_method_parity_inner_discovery_v1"
PREREGISTRATION = Path("docs/veatic21_again_parity_arousal_event_preregistration_20260717.md")
TARGET_NAME = "future_arousal_max_delta_rows_4_10"
SEEDS = (20260716, 20260717, 20260718)
OUTER_FOLDS = (1, 2, 3, 4, 5)
INNER_FOLDS = (1, 2, 3)
CHECKPOINT_ELIGIBLE_EPOCH = 1
LAMBDA_BINARY = 0.5
ALPHA_PENALTY = 0.01
AR_HIDDEN = 256
AR_DROPOUT = 0.1
RESIDUAL_HIDDEN = 64
EXPECTED_MEMBER_ROWS = 270
EXPECTED_ENSEMBLE_ROWS = 90
EVENT_METRIC_POLICY = "pooled_valid_heldout_rows_zero_event_videos_not_score_filled"


class ParityDiscoveryError(RuntimeError):
    pass


@dataclass(frozen=True)
class ParityRecipe:
    name: str
    feature_family: str
    pca_width: int
    input_variant: str

    @property
    def digest(self) -> str:
        return runner.canonical_digest(asdict(self))


RECIPES = (
    ParityRecipe("delta_pca64_again_clean_joint", "delta", 64, "again_clean"),
    ParityRecipe("delta_pca64_veatic_enriched_joint", "delta", 64, "veatic_enriched"),
    ParityRecipe("delta_pca128_again_clean_joint", "delta", 128, "again_clean"),
    ParityRecipe("delta_pca128_veatic_enriched_joint", "delta", 128, "veatic_enriched"),
    ParityRecipe(
        "temporal_mean_2s_pca256_again_clean_joint",
        "temporal_mean_2s",
        256,
        "again_clean",
    ),
    ParityRecipe(
        "temporal_mean_2s_pca256_veatic_enriched_joint",
        "temporal_mean_2s",
        256,
        "veatic_enriched",
    ),
)


@dataclass(frozen=True)
class TrainingSettings:
    batch_size: int = 1024
    max_epochs: int = 5000
    min_epochs: int = 50
    patience: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    selection_min_delta: float = 1e-6
    grad_clip: float = 1.0


@dataclass(frozen=True)
class ResidualHyperparameters:
    hidden: int = RESIDUAL_HIDDEN
    alpha_initial_logit: float = -4.0
    alpha_cap: float = 0.12
    gate_bias: float = 4.0
    lambda_binary: float = LAMBDA_BINARY


DEFAULT_RESIDUAL_HYPERPARAMETERS = ResidualHyperparameters()


@dataclass(frozen=True)
class DualScores:
    train_event_logit: np.ndarray
    train_continuous: np.ndarray
    test_event_logit: np.ndarray
    test_continuous: np.ndarray
    best_epoch: int
    epochs_run: int
    suppressed: bool
    checkpoint_path: Path | None
    cache_hit: bool
    identity_digest: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sigmoid(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))).astype(np.float32)


def pr_auc(labels: np.ndarray, logits: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int64)
    if np.unique(y).size != 2:
        raise ParityDiscoveryError("event PR-AUC requires both classes")
    value = float(average_precision_score(y, sigmoid(np.asarray(logits))))
    if not math.isfinite(value):
        raise ParityDiscoveryError("event PR-AUC is non-finite")
    return value


def event_panel_stats(
    *, labels: np.ndarray, valid: np.ndarray, videos: np.ndarray, global_rows: np.ndarray
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    video = np.asarray(videos, dtype=str)
    rows = np.asarray(global_rows, dtype=np.int64)
    if y.shape != mask.shape or video.shape != mask.shape or rows.shape != mask.shape:
        raise ParityDiscoveryError("event panel arrays are not row aligned")
    pooled = y[mask]
    if np.unique(pooled).size != 2:
        raise ParityDiscoveryError("pooled event panel requires both classes")
    positive_by_video = {
        item: int(np.count_nonzero(y[mask & (video == item)] > 0.5))
        for item in sorted(set(video[mask].tolist()))
    }
    return {
        "event_metric_policy": EVENT_METRIC_POLICY,
        "pooled_valid_rows": int(np.count_nonzero(mask)),
        "pooled_positive_rows": int(np.count_nonzero(pooled > 0.5)),
        "pooled_prevalence": float(np.mean(pooled)),
        "heldout_video_count": len(positive_by_video),
        "event_video_count": int(sum(count > 0 for count in positive_by_video.values())),
        "zero_event_video_count": int(sum(count == 0 for count in positive_by_video.values())),
        "zero_event_video_ids": [
            item for item, count in positive_by_video.items() if count == 0
        ],
        "heldout_label_digest": runner.array_digest(pooled.astype(np.float32)),
        "heldout_valid_row_digest": runner.array_digest(rows[mask]),
        "undefined_per_video_pr_auc_score_filled": False,
        "zero_event_videos_excluded_from_pooled_negatives": False,
    }


def preflight_quality_alignment_audit(
    cache: Any, dataset: Any, plan: discovery.NestedDiscoveryPlan
) -> dict[str, Any]:
    totals = {
        "rows": 0,
        "black_rows": 0,
        "duplicate_rows": 0,
        "excluded_rows": 0,
    }
    checks = {
        "exact_124_videos": True,
        "exact_20657_rows": True,
        "quality_flag_formulas_exact": True,
        "exclusion_is_black_or_duplicate": True,
        "source_arousal_lane_exact": True,
        "source_valence_lane_exact": True,
        "exact_2hz_time_grid": True,
        "source_annotation_brackets_exact": True,
        "source_annotation_positions_exact": True,
        "dataset_quality_mask_exact": True,
        "all_15_inner_panels_have_both_classes": True,
        "zero_event_videos_not_score_filled": True,
    }
    concatenated_exclusion: list[np.ndarray] = []
    video_count = 0
    for block in cache.iter_videos():
        video_count += 1
        arrays = block.columns
        n = block.row_count
        black_fraction = np.asarray(arrays["black_frame_fraction"], dtype=np.float32)
        duplicate_fraction = np.asarray(arrays["duplicate_frame_fraction"], dtype=np.float32)
        black = np.asarray(arrays["quality_black_frame_flag"], dtype=bool)
        duplicate = np.asarray(arrays["quality_duplicate_frame_flag"], dtype=bool)
        excluded = np.asarray(arrays["quality_exclusion_flag"], dtype=bool)
        concatenated_exclusion.append(excluded)
        totals["rows"] += n
        totals["black_rows"] += int(np.count_nonzero(black))
        totals["duplicate_rows"] += int(np.count_nonzero(duplicate))
        totals["excluded_rows"] += int(np.count_nonzero(excluded))
        checks["quality_flag_formulas_exact"] &= bool(
            np.array_equal(black, black_fraction >= 0.5)
            and np.array_equal(duplicate, duplicate_fraction >= 0.95)
        )
        checks["exclusion_is_black_or_duplicate"] &= bool(
            np.array_equal(excluded, black | duplicate)
        )
        checks["source_arousal_lane_exact"] &= bool(
            np.array_equal(arrays["source_arousal"], arrays["arousal"])
        )
        checks["source_valence_lane_exact"] &= bool(
            np.array_equal(arrays["source_valence"], arrays["valence"])
        )
        expected_time = np.arange(n, dtype=np.float64) / 2.0
        checks["exact_2hz_time_grid"] &= bool(
            np.allclose(
                np.asarray(arrays["time_seconds"], dtype=np.float64),
                expected_time,
                rtol=0.0,
                atol=1e-7,
            )
        )
        floor = np.asarray(arrays["source_floor_frame_index"], dtype=np.int32)
        ceil = np.asarray(arrays["source_ceil_frame_index"], dtype=np.int32)
        alpha = np.asarray(arrays["source_interp_alpha"], dtype=np.float32)
        position = np.asarray(arrays["source_frame_position"], dtype=np.float32)
        checks["source_annotation_brackets_exact"] &= bool(
            np.all(ceil >= floor)
            and np.all((ceil - floor) <= 1)
            and np.all(alpha >= -1e-7)
            and np.all(alpha <= 1.0 + 1e-7)
        )
        checks["source_annotation_positions_exact"] &= bool(
            np.allclose(position, floor.astype(np.float32) + alpha, rtol=0.0, atol=2e-5)
        )
    checks["exact_124_videos"] = video_count == 124
    checks["exact_20657_rows"] = totals["rows"] == 20_657
    exclusion_all = np.concatenate(concatenated_exclusion)
    checks["dataset_quality_mask_exact"] = bool(
        np.array_equal(np.asarray(dataset.quality_valid, dtype=bool), ~exclusion_all)
    )

    target = np.maximum(
        np.asarray(dataset.target_values[TARGET_NAME], dtype=np.float32), 0.0
    )
    target_valid = np.asarray(dataset.target_valid[TARGET_NAME], dtype=bool)
    _, ar_context, _ = program.canonical_ar_history_features(
        dataset.arousal, dataset.video_id
    )
    panels: list[dict[str, Any]] = []
    for outer in plan.outer_folds:
        for inner in outer.inner_folds:
            train_mask = (
                np.isin(dataset.video_id, list(inner.train_videos))
                & target_valid
                & dataset.quality_valid
                & ar_context
            )
            test_mask = (
                np.isin(dataset.video_id, list(inner.validation_videos))
                & target_valid
                & dataset.quality_valid
                & ar_context
            )
            threshold = float(np.quantile(target[train_mask], 0.90))
            labels = (target >= threshold).astype(np.float32)
            panel = event_panel_stats(
                labels=labels,
                valid=test_mask,
                videos=dataset.video_id,
                global_rows=dataset.row_idx,
            )
            panels.append(
                {
                    "outer_fold": outer.outer_fold,
                    "inner_fold": inner.fold,
                    "event_threshold": threshold,
                    **panel,
                }
            )
    checks["all_15_inner_panels_have_both_classes"] = bool(
        len(panels) == 15
        and all(
            0 < row["pooled_positive_rows"] < row["pooled_valid_rows"]
            for row in panels
        )
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "checks": checks,
        "passed": all(checks.values()),
        "quality_totals": totals,
        "quality_policy": {
            "black_fraction_threshold": 0.5,
            "duplicate_fraction_threshold": 0.95,
            "applies_to_pca_fit_training_selection_and_scoring": True,
            "label_or_outcome_based_exclusion": False,
        },
        "event_metric_policy": EVENT_METRIC_POLICY,
        "event_panels": panels,
    }
    return {**audit, "digest": runner.canonical_digest(audit)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shared-derived-root", type=Path, required=True)
    parser.add_argument(
        "--identity-manifest", type=Path, default=runner.compact.DEFAULT_IDENTITY_MANIFEST
    )
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=5000)
    parser.add_argument("--min-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--selection-min-delta", type=float, default=1e-6)
    return parser


def validate_args(args: argparse.Namespace) -> TrainingSettings:
    if args.dry_run and args.audit_only:
        raise ParityDiscoveryError("--dry-run and --audit-only are mutually exclusive")
    settings = TrainingSettings(
        batch_size=int(args.batch_size),
        max_epochs=int(args.max_epochs),
        min_epochs=int(args.min_epochs),
        patience=int(args.patience),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        selection_min_delta=float(args.selection_min_delta),
    )
    if settings.batch_size < 1 or settings.max_epochs < 1:
        raise ParityDiscoveryError("batch size and max epochs must be positive")
    if not 1 <= settings.min_epochs <= settings.max_epochs:
        raise ParityDiscoveryError("min epochs must be in [1, max epochs]")
    if settings.patience < 1 or settings.min_epochs + settings.patience > settings.max_epochs:
        raise ParityDiscoveryError("invalid patience/min/max epoch relationship")
    if settings.learning_rate <= 0 or settings.weight_decay < 0:
        raise ParityDiscoveryError("invalid optimizer settings")
    if settings.selection_min_delta < 0 or not math.isfinite(settings.selection_min_delta):
        raise ParityDiscoveryError("selection minimum delta must be finite and non-negative")
    return settings


def member_keys(*, smoke: bool = False) -> tuple[tuple[int, int, str, int], ...]:
    outers = OUTER_FOLDS[:1] if smoke else OUTER_FOLDS
    inners = INNER_FOLDS[:1] if smoke else INNER_FOLDS
    recipes = RECIPES[:1] if smoke else RECIPES
    seeds = SEEDS[:1] if smoke else SEEDS
    return tuple(
        (outer, inner, recipe.name, seed)
        for outer in outers
        for inner in inners
        for recipe in recipes
        for seed in seeds
    )


def ensemble_keys(*, smoke: bool = False) -> tuple[tuple[int, int, str], ...]:
    if smoke:
        return ()
    return tuple(
        (outer, inner, recipe.name)
        for outer in OUTER_FOLDS
        for inner in INNER_FOLDS
        for recipe in RECIPES
    )


def clean_or_enriched_view(
    temporal: np.ndarray, *, pca_width: int, input_variant: str
) -> np.ndarray:
    values = np.asarray(temporal, dtype=np.float32)
    clean_width = 5 * int(pca_width) + 53
    enriched_width = clean_width + 5 + 2
    if values.ndim != 2 or values.shape[1] != enriched_width:
        raise ParityDiscoveryError("VEATIC temporal feature width drift")
    if input_variant == "again_clean":
        return values[:, :clean_width].copy()
    if input_variant == "veatic_enriched":
        return values.copy()
    raise ParityDiscoveryError(f"unknown residual input variant {input_variant!r}")


def validate_veatic_pca_provenance(
    *,
    prepared: execution.PreparedFeatures,
    dataset: Any,
    plan: discovery.NestedDiscoveryPlan,
    shared_derived_root: Path,
) -> dict[str, Any]:
    component_path = Path(prepared.provenance["pca_component_path"]).resolve()
    metadata_path = Path(prepared.provenance["pca_metadata_path"]).resolve()
    expected_root = (Path(shared_derived_root).resolve() / "pca")
    try:
        component_path.relative_to(expected_root)
        metadata_path.relative_to(expected_root)
    except ValueError as exc:
        raise ParityDiscoveryError("PCA artifact escaped the VEATIC shared PCA root") from exc
    if not component_path.is_file() or not metadata_path.is_file():
        raise ParityDiscoveryError("VEATIC PCA artifact pair is incomplete")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    identity = metadata.get("identity")
    if not isinstance(identity, Mapping):
        raise ParityDiscoveryError("VEATIC PCA metadata has no identity")
    checks = {
        "dataset_id_veatic_124_v21": identity.get("dataset_id") == "veatic-124-v2.1",
        "fresh_veatic_only": identity.get("fresh_veatic_only") is True,
        "cache_digest_matches_veatic21_seal": identity.get("cache_digest")
        == dataset.dataset_seal_digest,
        "contract_digest_matches_plan": identity.get("contract_digest") == plan.digest,
        "fit_quality_valid_only": metadata.get("fit_quality_valid_only") is True,
        "no_heldout_participation": metadata.get("no_held_out_participation_audit")
        is True,
        "component_checksum_exact": metadata.get("component_file_sha256")
        == runner.file_sha256(component_path),
        "prepared_identity_matches_metadata": prepared.provenance.get(
            "pca_parent_identity"
        )
        == metadata.get("identity_sha256"),
    }
    if not all(checks.values()):
        raise ParityDiscoveryError(f"VEATIC PCA provenance failed: {checks}")
    payload = {
        "checks": checks,
        "pca_parent_identity": metadata["identity_sha256"],
        "pca_component_sha256": metadata["component_file_sha256"],
        "pca_base_family": identity["base_family"],
        "pca_fit_train_row_digest": identity["fit_train_row_digest"],
        "pca_fit_train_video_digest": identity["fit_train_video_digest"],
        "pca_held_out_row_digest": identity["held_out_row_digest"],
        "pca_dataset_id": identity["dataset_id"],
        "pca_fresh_veatic_only": True,
        "pca_from_again": False,
        "pca_from_original_veatic": False,
    }
    return {**payload, "pca_provenance_digest": runner.canonical_digest(payload)}


def _feature_recipe(recipe: ParityRecipe) -> discovery.RecipeSpec:
    payload = {
        "name": recipe.name,
        "feature_family": recipe.feature_family,
        "pca_width": recipe.pca_width,
        "head": "short_temporal_conv_residual",
        "causal_rows": 5,
        "complexity_score": 1,
    }
    return discovery.RecipeSpec(
        order=list(RECIPES).index(recipe),
        name=recipe.name,
        feature_family=recipe.feature_family,
        pca_width=recipe.pca_width,
        head="short_temporal_conv_residual",
        causal_rows=5,
        complexity_score=1,
        payload_json=json.dumps(payload, sort_keys=True, separators=(",", ":")),
        digest=runner.canonical_digest(payload),
    )


def _fit_standardization(x: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(x[rows], dtype=np.float32)
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ParityDiscoveryError("invalid train-only standardization")
    return mean, std


def _standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = ((np.asarray(x, dtype=np.float32) - mean) / std).astype(np.float32)
    if not np.isfinite(out).all():
        raise ParityDiscoveryError("standardized model input is non-finite")
    return out


def _ar_factory() -> Any:
    return base.GatedArPcaMlp(
        7, 0, 0, hidden=AR_HIDDEN, dual_output=True, dropout=AR_DROPOUT
    )


def _residual_factory(
    input_dim: int,
    pca_width: int,
    hyperparameters: ResidualHyperparameters = DEFAULT_RESIDUAL_HYPERPARAMETERS,
) -> Any:
    return again_temporal.TemporalResidualHead(
        input_dim,
        "short_temporal_conv_residual",
        hidden=hyperparameters.hidden,
        sequence_window=5,
        sequence_channels=int(pca_width),
        alpha_initial_logit=hyperparameters.alpha_initial_logit,
        alpha_cap=hyperparameters.alpha_cap,
        gate_bias=hyperparameters.gate_bias,
    )


def _forward_ar(model: Any, x: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(model, "eval"):
        model.eval()
    logits: list[np.ndarray] = []
    continuous: list[np.ndarray] = []
    for start in range(0, len(x), int(batch_size)):
        output = model(base.mx.array(x[start : start + batch_size], dtype=base.mx.float32))
        base.mx.eval(output)
        values = np.asarray(output, dtype=np.float32)
        continuous.append(values[:, 0])
        logits.append(values[:, 1])
    return np.concatenate(logits), np.concatenate(continuous)


def _forward_residual(
    model: Any,
    x: np.ndarray,
    ar_logit: np.ndarray,
    ar_continuous: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(model, "eval"):
        model.eval()
    logits: list[np.ndarray] = []
    continuous: list[np.ndarray] = []
    for start in range(0, len(x), int(batch_size)):
        xb = base.mx.array(x[start : start + batch_size], dtype=base.mx.float32)
        ab = base.mx.array(ar_logit[start : start + batch_size], dtype=base.mx.float32)
        rb = base.mx.array(ar_continuous[start : start + batch_size], dtype=base.mx.float32)
        output = model(xb, ab, rb)
        base.mx.eval(output)
        values = np.asarray(output, dtype=np.float32)
        continuous.append(values[:, 0])
        logits.append(values[:, 1])
    return np.concatenate(logits), np.concatenate(continuous)


def _train_ar_model(
    *,
    x: np.ndarray,
    event: np.ndarray,
    continuous: np.ndarray,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray | None,
    seed: int,
    epochs: int,
    settings: TrainingSettings,
    checkpoint: Path,
) -> tuple[Any, list[dict[str, Any]], int]:
    base.mx.random.seed(int(seed))
    model = _ar_factory()
    _ = model(base.mx.array(x[: min(2, len(x))], dtype=base.mx.float32))
    optimizer = base.optim.AdamW(
        learning_rate=settings.learning_rate, weight_decay=settings.weight_decay
    )
    rng = np.random.default_rng(int(seed) + 70001)

    def loss_fn(model_obj: Any, xb: Any, yb: Any, yr: Any) -> Any:
        out = model_obj(xb)
        reg = base.mx.mean(base.nn.losses.huber_loss(out[:, 0:1], yr, delta=1.0))
        bce = base.mx.mean(
            base.nn.losses.binary_cross_entropy(out[:, 1:2], yb, with_logits=True)
        )
        return reg + LAMBDA_BINARY * bce

    loss_and_grad = base.nn.value_and_grad(model, loss_fn)
    best_value = -math.inf
    best_epoch = 0
    stale = 0
    curves: list[dict[str, Any]] = []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(epochs) + 1):
        if hasattr(model, "train"):
            model.train()
        order = rng.permutation(fit_rows)
        losses: list[float] = []
        for start in range(0, len(order), settings.batch_size):
            rows = order[start : start + settings.batch_size]
            loss, grads = loss_and_grad(
                model,
                base.mx.array(x[rows], dtype=base.mx.float32),
                base.mx.array(event[rows, None], dtype=base.mx.float32),
                base.mx.array(continuous[rows, None], dtype=base.mx.float32),
            )
            grads, _ = base.optim.clip_grad_norm(grads, settings.grad_clip)
            optimizer.update(model, grads)
            base.mx.eval(model.parameters(), optimizer.state)
            losses.append(float(np.asarray(loss)))
        if validation_rows is None:
            best_epoch = epoch
            continue
        val_logit, _ = _forward_ar(model, x[validation_rows], settings.batch_size)
        value = pr_auc(event[validation_rows], val_logit)
        curves.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "inner_validation_pr_auc": value,
                "checkpoint_eligible": epoch >= CHECKPOINT_ELIGIBLE_EPOCH,
                "early_stopping_eligible": epoch >= settings.min_epochs,
            }
        )
        if value > best_value + settings.selection_min_delta:
            model.save_weights(str(checkpoint))
            best_value = value
            best_epoch = epoch
            stale = 0
        elif epoch >= settings.min_epochs:
            stale += 1
        if epoch >= settings.min_epochs and stale >= settings.patience:
            break
    if validation_rows is None:
        model.save_weights(str(checkpoint))
    if best_epoch < 1 or not checkpoint.is_file():
        raise ParityDiscoveryError("dual-task AR training produced no checkpoint")
    restored = _ar_factory()
    _ = restored(base.mx.array(x[: min(2, len(x))], dtype=base.mx.float32))
    restored.load_weights(str(checkpoint))
    if hasattr(restored, "eval"):
        restored.eval()
    return restored, curves, best_epoch


def _train_residual_model(
    *,
    x: np.ndarray,
    event: np.ndarray,
    continuous: np.ndarray,
    ar_logit: np.ndarray,
    ar_continuous: np.ndarray,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray | None,
    pca_width: int,
    seed: int,
    epochs: int,
    settings: TrainingSettings,
    hyperparameters: ResidualHyperparameters,
    checkpoint: Path,
) -> tuple[Any | None, list[dict[str, Any]], int, bool]:
    base.mx.random.seed(int(seed))
    model = _residual_factory(x.shape[1], pca_width, hyperparameters)
    _ = model(
        base.mx.array(x[: min(2, len(x))], dtype=base.mx.float32),
        base.mx.array(ar_logit[: min(2, len(x))], dtype=base.mx.float32),
        base.mx.array(ar_continuous[: min(2, len(x))], dtype=base.mx.float32),
    )
    optimizer = base.optim.AdamW(
        learning_rate=settings.learning_rate, weight_decay=settings.weight_decay
    )
    rng = np.random.default_rng(int(seed) + 90001 + int(pca_width))

    def loss_fn(model_obj: Any, xb: Any, ab: Any, rb: Any, yb: Any, yr: Any) -> Any:
        out = model_obj(xb, ab, rb)
        reg = base.mx.mean(base.nn.losses.huber_loss(out[:, 0:1], yr, delta=1.0))
        bce = base.mx.mean(
            base.nn.losses.binary_cross_entropy(out[:, 1:2], yb, with_logits=True)
        )
        return reg + hyperparameters.lambda_binary * bce + ALPHA_PENALTY * base.mx.mean(
            model_obj.alpha_value() * model_obj.alpha_value()
        )

    loss_and_grad = base.nn.value_and_grad(model, loss_fn)
    baseline = (
        pr_auc(event[validation_rows], ar_logit[validation_rows])
        if validation_rows is not None
        else None
    )
    best_delta = 0.0
    best_epoch = 0
    stale = 0
    suppressed = validation_rows is not None
    curves: list[dict[str, Any]] = []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(epochs) + 1):
        if hasattr(model, "train"):
            model.train()
        order = rng.permutation(fit_rows)
        losses: list[float] = []
        for start in range(0, len(order), settings.batch_size):
            rows = order[start : start + settings.batch_size]
            loss, grads = loss_and_grad(
                model,
                base.mx.array(x[rows], dtype=base.mx.float32),
                base.mx.array(ar_logit[rows], dtype=base.mx.float32),
                base.mx.array(ar_continuous[rows], dtype=base.mx.float32),
                base.mx.array(event[rows, None], dtype=base.mx.float32),
                base.mx.array(continuous[rows, None], dtype=base.mx.float32),
            )
            grads, _ = base.optim.clip_grad_norm(grads, settings.grad_clip)
            optimizer.update(model, grads)
            base.mx.eval(model.parameters(), optimizer.state)
            losses.append(float(np.asarray(loss)))
        if validation_rows is None:
            best_epoch = epoch
            continue
        val_logit, _ = _forward_residual(
            model,
            x[validation_rows],
            ar_logit[validation_rows],
            ar_continuous[validation_rows],
            settings.batch_size,
        )
        absolute = pr_auc(event[validation_rows], val_logit)
        delta = absolute - float(baseline)
        curves.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "inner_validation_pr_auc": absolute,
                "inner_validation_delta_vs_frozen_ar": delta,
                "alpha": float(np.asarray(model.alpha_value())[0]),
                "zero_correction_selected_if_stopped_now": delta <= 0.0,
                "checkpoint_eligible": epoch >= CHECKPOINT_ELIGIBLE_EPOCH,
                "early_stopping_eligible": epoch >= settings.min_epochs,
            }
        )
        if delta > best_delta + settings.selection_min_delta:
            model.save_weights(str(checkpoint))
            best_delta = delta
            best_epoch = epoch
            stale = 0
            suppressed = False
        elif epoch >= settings.min_epochs:
            stale += 1
        if epoch >= settings.min_epochs and stale >= settings.patience:
            break
    if validation_rows is None:
        model.save_weights(str(checkpoint))
    if validation_rows is not None and suppressed:
        return None, curves, 0, True
    if best_epoch < 1 or not checkpoint.is_file():
        raise ParityDiscoveryError("dual-task residual training produced no checkpoint")
    restored = _residual_factory(x.shape[1], pca_width, hyperparameters)
    _ = restored(
        base.mx.array(x[: min(2, len(x))], dtype=base.mx.float32),
        base.mx.array(ar_logit[: min(2, len(x))], dtype=base.mx.float32),
        base.mx.array(ar_continuous[: min(2, len(x))], dtype=base.mx.float32),
    )
    restored.load_weights(str(checkpoint))
    if hasattr(restored, "eval"):
        restored.eval()
    return restored, curves, best_epoch, False


def _load_score_bundle(path: Path, manifest_path: Path, identity: Mapping[str, Any]) -> DualScores | None:
    if not path.exists() and not manifest_path.exists():
        return None
    if not path.is_file() or not manifest_path.is_file():
        raise ParityDiscoveryError("incomplete cached dual-score artifact")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("identity") != identity:
        raise ParityDiscoveryError("cached dual-score identity mismatch")
    if manifest.get("score_sha256") != runner.file_sha256(path):
        raise ParityDiscoveryError("cached dual-score checksum drift")
    with np.load(path, allow_pickle=False) as bundle:
        arrays = {name: np.asarray(bundle[name], dtype=np.float32) for name in bundle.files}
    required = {"train_event_logit", "train_continuous", "test_event_logit", "test_continuous"}
    if set(arrays) != required:
        raise ParityDiscoveryError("cached dual-score keys drifted")
    return DualScores(
        **arrays,
        best_epoch=int(manifest["best_epoch"]),
        epochs_run=int(manifest["epochs_run"]),
        suppressed=bool(manifest["suppressed"]),
        checkpoint_path=(Path(manifest["checkpoint_path"]) if manifest.get("checkpoint_path") else None),
        cache_hit=True,
        identity_digest=str(manifest["identity_digest"]),
    )


def _persist_score_bundle(
    *,
    path: Path,
    manifest_path: Path,
    identity: Mapping[str, Any],
    scores: DualScores,
    curves: Sequence[Mapping[str, Any]],
    extra: Mapping[str, Any],
) -> DualScores:
    runner.atomic_npz(
        path,
        train_event_logit=scores.train_event_logit,
        train_continuous=scores.train_continuous,
        test_event_logit=scores.test_event_logit,
        test_continuous=scores.test_continuous,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "identity": dict(identity),
        "identity_digest": runner.canonical_digest(identity),
        "score_path": str(path.resolve()),
        "score_sha256": runner.file_sha256(path),
        "best_epoch": scores.best_epoch,
        "epochs_run": scores.epochs_run,
        "suppressed": scores.suppressed,
        "checkpoint_path": str(scores.checkpoint_path.resolve()) if scores.checkpoint_path else None,
        "checkpoint_sha256": (
            runner.file_sha256(scores.checkpoint_path) if scores.checkpoint_path else None
        ),
        "eval_mode_scoring": True,
        "curves": list(curves),
        **dict(extra),
    }
    runner.atomic_json(manifest_path, payload)
    return DualScores(**{**asdict(scores), "cache_hit": False})


def fit_frozen_dual_ar(
    *,
    x_train: np.ndarray,
    x_test: np.ndarray,
    event: np.ndarray,
    continuous: np.ndarray,
    valid: np.ndarray,
    videos: np.ndarray,
    ownership: program.InnerVideoOwnership,
    seed: int,
    output_root: Path,
    identity_base: Mapping[str, Any],
    settings: TrainingSettings,
) -> DualScores:
    identity = {
        **dict(identity_base),
        "kind": "veatic_only_dual_task_frozen_ar7",
        "seed": int(seed),
        "ownership_digest": ownership.digest,
        "train_x_digest": runner.array_digest(x_train),
        "test_x_digest": runner.array_digest(x_test),
        "event_digest": runner.array_digest(event),
        "continuous_digest": runner.array_digest(continuous),
        "valid_digest": runner.array_digest(valid.astype(np.uint8)),
        "settings": asdict(settings),
    }
    scores_path = output_root / "scores.npz"
    manifest_path = output_root / "scores.json"
    cached = _load_score_bundle(scores_path, manifest_path, identity)
    if cached is not None:
        return cached
    inner_train, inner_val = ownership.eligible_indices(videos, valid)
    select_mean, select_std = _fit_standardization(x_train, inner_train)
    select_x = _standardize(x_train, select_mean, select_std)
    selection_checkpoint = output_root / "selection" / "ar.npz"
    primary, curves, best_epoch = _train_ar_model(
        x=select_x,
        event=event,
        continuous=continuous,
        fit_rows=inner_train,
        validation_rows=inner_val,
        seed=seed,
        epochs=settings.max_epochs,
        settings=settings,
        checkpoint=selection_checkpoint,
    )
    primary_logit, primary_cont = _forward_ar(primary, select_x, settings.batch_size)
    honest_logit = np.full(len(x_train), np.nan, dtype=np.float32)
    honest_cont = np.full(len(x_train), np.nan, dtype=np.float32)
    ownership_train, ownership_val = ownership.row_masks(videos)
    honest_logit[ownership_val] = primary_logit[ownership_val]
    honest_cont[ownership_val] = primary_cont[ownership_val]
    crossfit_rows: list[dict[str, Any]] = []
    for scope in program.build_ar_crossfit_video_folds(ownership, fold_count=5):
        fit_mask = valid & np.isin(videos, list(scope.fit_videos))
        prediction_mask = ownership_train & np.isin(videos, list(scope.prediction_videos))
        fit_rows = np.flatnonzero(fit_mask).astype(np.int64)
        mean, std = _fit_standardization(x_train, fit_rows)
        standardized = _standardize(x_train, mean, std)
        checkpoint = output_root / "crossfit" / f"fold_{scope.fold}.npz"
        model, _, _ = _train_ar_model(
            x=standardized,
            event=event,
            continuous=continuous,
            fit_rows=fit_rows,
            validation_rows=None,
            seed=seed + scope.fold * 100_003,
            epochs=best_epoch,
            settings=settings,
            checkpoint=checkpoint,
        )
        logits, cont = _forward_ar(model, standardized, settings.batch_size)
        honest_logit[prediction_mask] = logits[prediction_mask]
        honest_cont[prediction_mask] = cont[prediction_mask]
        crossfit_rows.append(
            {
                "scope_digest": scope.digest,
                "prediction_rows": int(np.count_nonzero(prediction_mask)),
                "checkpoint_sha256": runner.file_sha256(checkpoint),
            }
        )
    if not np.isfinite(honest_logit).all() or not np.isfinite(honest_cont).all():
        raise ParityDiscoveryError("dual-task AR cross-fitting left missing predictions")
    eligible = np.flatnonzero(valid).astype(np.int64)
    final_mean, final_std = _fit_standardization(x_train, eligible)
    final_train_x = _standardize(x_train, final_mean, final_std)
    final_test_x = _standardize(x_test, final_mean, final_std)
    combined_x = np.concatenate([final_train_x, final_test_x], axis=0)
    combined_event = np.concatenate([event, np.zeros(len(x_test), dtype=np.float32)])
    combined_cont = np.concatenate([continuous, np.zeros(len(x_test), dtype=np.float32)])
    final_checkpoint = output_root / "final" / "ar.npz"
    final, _, _ = _train_ar_model(
        x=combined_x,
        event=combined_event,
        continuous=combined_cont,
        fit_rows=eligible,
        validation_rows=None,
        seed=seed,
        epochs=best_epoch,
        settings=settings,
        checkpoint=final_checkpoint,
    )
    final_logit, final_cont = _forward_ar(final, combined_x, settings.batch_size)
    scores = DualScores(
        train_event_logit=honest_logit,
        train_continuous=honest_cont,
        test_event_logit=final_logit[len(x_train) :],
        test_continuous=final_cont[len(x_train) :],
        best_epoch=best_epoch,
        epochs_run=len(curves),
        suppressed=False,
        checkpoint_path=final_checkpoint,
        cache_hit=False,
        identity_digest=runner.canonical_digest(identity),
    )
    return _persist_score_bundle(
        path=scores_path,
        manifest_path=manifest_path,
        identity=identity,
        scores=scores,
        curves=curves,
        extra={
            "all_outer_train_predictions_out_of_video_fit": True,
            "final_test_fit_uses_outer_train_only": True,
            "crossfit": crossfit_rows,
            "no_again_artifact_reuse": True,
        },
    )


def fit_dual_residual(
    *,
    x_train: np.ndarray,
    x_test: np.ndarray,
    event: np.ndarray,
    continuous: np.ndarray,
    valid: np.ndarray,
    ownership: program.InnerVideoOwnership,
    frozen_ar: DualScores,
    pca_width: int,
    seed: int,
    output_root: Path,
    identity_base: Mapping[str, Any],
    settings: TrainingSettings,
    hyperparameters: ResidualHyperparameters = DEFAULT_RESIDUAL_HYPERPARAMETERS,
) -> DualScores:
    identity = {
        **dict(identity_base),
        "kind": "veatic_only_joint_short_conv_residual",
        "seed": int(seed),
        "pca_width": int(pca_width),
        "ownership_digest": ownership.digest,
        "frozen_ar_identity_digest": frozen_ar.identity_digest,
        "train_x_digest": runner.array_digest(x_train),
        "test_x_digest": runner.array_digest(x_test),
        "event_digest": runner.array_digest(event),
        "continuous_digest": runner.array_digest(continuous),
        "valid_digest": runner.array_digest(valid.astype(np.uint8)),
        "settings": asdict(settings),
        "ar_is_additive_floor_not_network_input": True,
        "alpha_penalty": ALPHA_PENALTY,
    }
    if hyperparameters != DEFAULT_RESIDUAL_HYPERPARAMETERS:
        identity["residual_hyperparameters"] = asdict(hyperparameters)
    scores_path = output_root / "predictions.npz"
    manifest_path = output_root / "predictions.json"
    cached = _load_score_bundle(scores_path, manifest_path, identity)
    if cached is not None:
        return cached
    inner_train, inner_val = ownership.eligible_indices(
        np.asarray(identity_base["train_video_ids"], dtype=str), valid
    )
    mean, std = _fit_standardization(x_train, inner_train)
    select_x = _standardize(x_train, mean, std)
    selection_checkpoint = output_root / "selection" / "residual.npz"
    _, curves, best_epoch, suppressed = _train_residual_model(
        x=select_x,
        event=event,
        continuous=continuous,
        ar_logit=frozen_ar.train_event_logit,
        ar_continuous=frozen_ar.train_continuous,
        fit_rows=inner_train,
        validation_rows=inner_val,
        pca_width=pca_width,
        seed=seed,
        epochs=settings.max_epochs,
        settings=settings,
        hyperparameters=hyperparameters,
        checkpoint=selection_checkpoint,
    )
    eligible = np.flatnonzero(valid).astype(np.int64)
    if suppressed:
        train_logit = frozen_ar.train_event_logit.copy()
        train_cont = frozen_ar.train_continuous.copy()
        test_logit = frozen_ar.test_event_logit.copy()
        test_cont = frozen_ar.test_continuous.copy()
        final_checkpoint = None
    else:
        final_mean, final_std = _fit_standardization(x_train, eligible)
        final_train_x = _standardize(x_train, final_mean, final_std)
        final_test_x = _standardize(x_test, final_mean, final_std)
        combined_x = np.concatenate([final_train_x, final_test_x], axis=0)
        combined_event = np.concatenate([event, np.zeros(len(x_test), dtype=np.float32)])
        combined_cont = np.concatenate([continuous, np.zeros(len(x_test), dtype=np.float32)])
        combined_ar_logit = np.concatenate(
            [frozen_ar.train_event_logit, frozen_ar.test_event_logit]
        )
        combined_ar_cont = np.concatenate(
            [frozen_ar.train_continuous, frozen_ar.test_continuous]
        )
        final_checkpoint = output_root / "final" / "residual.npz"
        final, _, _, _ = _train_residual_model(
            x=combined_x,
            event=combined_event,
            continuous=combined_cont,
            ar_logit=combined_ar_logit,
            ar_continuous=combined_ar_cont,
            fit_rows=eligible,
            validation_rows=None,
            pca_width=pca_width,
            seed=seed,
            epochs=best_epoch,
            settings=settings,
            hyperparameters=hyperparameters,
            checkpoint=final_checkpoint,
        )
        if final is None:
            raise ParityDiscoveryError("active residual refit unexpectedly suppressed")
        logits, cont = _forward_residual(
            final,
            combined_x,
            combined_ar_logit,
            combined_ar_cont,
            settings.batch_size,
        )
        train_logit = logits[: len(x_train)]
        train_cont = cont[: len(x_train)]
        test_logit = logits[len(x_train) :]
        test_cont = cont[len(x_train) :]
    scores = DualScores(
        train_event_logit=train_logit,
        train_continuous=train_cont,
        test_event_logit=test_logit,
        test_continuous=test_cont,
        best_epoch=best_epoch,
        epochs_run=len(curves),
        suppressed=suppressed,
        checkpoint_path=final_checkpoint,
        cache_hit=False,
        identity_digest=runner.canonical_digest(identity),
    )
    return _persist_score_bundle(
        path=scores_path,
        manifest_path=manifest_path,
        identity=identity,
        scores=scores,
        curves=curves,
        extra={
            "zero_correction_was_selection_candidate": True,
            "joint_continuous_event_training": True,
            "ar_is_additive_floor_not_network_input": True,
            "no_again_artifact_reuse": True,
        },
    )


def _recipe_summary(
    member_rows: Sequence[Mapping[str, Any]], ensemble_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for recipe in RECIPES:
        members = [row for row in member_rows if row["recipe"] == recipe.name]
        ensembles = [row for row in ensemble_rows if row["recipe"] == recipe.name]
        member_delta = np.asarray([row["delta_vs_ar_pr_auc"] for row in members])
        ensemble_delta = np.asarray([row["delta_vs_ar_pr_auc"] for row in ensembles])
        member_outer = [
            float(np.mean([row["delta_vs_ar_pr_auc"] for row in members if row["outer_fold"] == outer]))
            for outer in OUTER_FOLDS
        ]
        ensemble_outer = [
            float(np.mean([row["delta_vs_ar_pr_auc"] for row in ensembles if row["outer_fold"] == outer]))
            for outer in OUTER_FOLDS
        ]
        member_credible = bool(
            np.mean(member_delta) >= 0.001
            and np.median(member_delta) > 0
            and np.count_nonzero(member_delta > 0) >= 30
            and np.count_nonzero(np.asarray(member_outer) > 0) >= 4
        )
        ensemble_credible = bool(
            np.mean(ensemble_delta) >= 0.001
            and np.median(ensemble_delta) > 0
            and np.count_nonzero(ensemble_delta > 0) >= 10
            and np.count_nonzero(np.asarray(ensemble_outer) > 0) >= 4
        )
        summaries[recipe.name] = {
            "member_real_pr_auc": float(np.mean([row["real_pr_auc"] for row in members])),
            "member_ar_pr_auc": float(np.mean([row["ar_pr_auc"] for row in members])),
            "member_mean_delta": float(np.mean(member_delta)),
            "member_median_delta": float(np.median(member_delta)),
            "member_positive": int(np.count_nonzero(member_delta > 0)),
            "member_ties": int(np.count_nonzero(member_delta == 0)),
            "member_positive_outer_means": int(np.count_nonzero(np.asarray(member_outer) > 0)),
            "member_credible": member_credible,
            "ensemble_real_pr_auc": float(np.mean([row["real_pr_auc"] for row in ensembles])),
            "ensemble_ar_pr_auc": float(np.mean([row["ar_pr_auc"] for row in ensembles])),
            "ensemble_mean_delta": float(np.mean(ensemble_delta)),
            "ensemble_median_delta": float(np.median(ensemble_delta)),
            "ensemble_positive": int(np.count_nonzero(ensemble_delta > 0)),
            "ensemble_positive_outer_means": int(np.count_nonzero(np.asarray(ensemble_outer) > 0)),
            "ensemble_uplift_over_member_mean": float(
                np.mean([row["ensemble_uplift_over_member_mean"] for row in ensembles])
            ),
            "ensemble_credible": ensemble_credible,
            "strong": bool(
                member_credible
                and ensemble_credible
                and np.count_nonzero(member_delta > 0) >= 36
                and np.count_nonzero(ensemble_delta > 0) >= 12
                and np.count_nonzero(np.asarray(member_outer) > 0) == 5
                and np.count_nonzero(np.asarray(ensemble_outer) > 0) == 5
            ),
        }
    return summaries


def audit_outputs(output_root: Path, *, smoke: bool = False) -> dict[str, Any]:
    member_path = output_root / "member_rows.json"
    ensemble_path = output_root / "ensemble_rows.json"
    if not member_path.is_file() or (not smoke and not ensemble_path.is_file()):
        raise ParityDiscoveryError("parity output matrix is incomplete")
    members = json.loads(member_path.read_text(encoding="utf-8"))
    ensembles = [] if smoke else json.loads(ensemble_path.read_text(encoding="utf-8"))
    expected_member = member_keys(smoke=smoke)
    expected_ensemble = ensemble_keys(smoke=smoke)
    observed_member = [
        (row["outer_fold"], row["inner_fold"], row["recipe"], row["seed"])
        for row in members
    ]
    observed_ensemble = [
        (row["outer_fold"], row["inner_fold"], row["recipe"]) for row in ensembles
    ]
    all_rows = members + ensembles
    panel_groups: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in all_rows:
        panel_groups.setdefault((int(row["outer_fold"]), int(row["inner_fold"])), []).append(row)
    checks = {
        "member_complete": sorted(observed_member) == sorted(expected_member),
        "member_unique": len(set(observed_member)) == len(observed_member),
        "ensemble_complete": sorted(observed_ensemble) == sorted(expected_ensemble),
        "ensemble_unique": len(set(observed_ensemble)) == len(observed_ensemble),
        "outer_test_scores_used_false": all(not row["outer_test_scores_used"] for row in all_rows),
        "no_again_artifact_reuse": all(row["no_again_artifact_reuse"] for row in all_rows),
        "veatic21_pca_provenance_passed": all(
            row.get("pca_dataset_id") == "veatic-124-v2.1"
            and row.get("pca_fresh_veatic_only") is True
            and row.get("pca_from_again") is False
            and row.get("pca_from_original_veatic") is False
            for row in all_rows
        ),
        "pooled_zero_event_safe_metric_policy": all(
            row.get("event_metric_policy") == EVENT_METRIC_POLICY
            and row.get("undefined_per_video_pr_auc_score_filled") is False
            and row.get("zero_event_videos_excluded_from_pooled_negatives") is False
            for row in all_rows
        ),
        "heldout_label_digest_aligned_by_panel": all(
            len({str(row["heldout_label_digest"]) for row in rows}) == 1
            for rows in panel_groups.values()
        ),
        "heldout_valid_row_digest_aligned_by_panel": all(
            len({str(row["heldout_valid_row_digest"]) for row in rows}) == 1
            for rows in panel_groups.values()
        ),
        "event_counts_aligned_by_panel": all(
            len(
                {
                    (
                        int(row["pooled_valid_rows"]),
                        int(row["pooled_positive_rows"]),
                        int(row["event_video_count"]),
                        int(row["zero_event_video_count"]),
                    )
                    for row in rows
                }
            )
            == 1
            for rows in panel_groups.values()
        ),
        "finite_metrics": all(
            math.isfinite(float(row[key]))
            for row in all_rows
            for key in ("real_pr_auc", "ar_pr_auc", "delta_vs_ar_pr_auc")
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "checks": checks,
        "passed": all(checks.values()),
        "member_rows": len(members),
        "ensemble_rows": len(ensembles),
        "expected_member_rows": len(expected_member),
        "expected_ensemble_rows": len(expected_ensemble),
    }


def _run_identity(
    *,
    args: argparse.Namespace,
    seal: runner.DatasetSeal,
    plan: discovery.NestedDiscoveryPlan,
    settings: TrainingSettings,
    preflight_digest: str,
) -> dict[str, Any]:
    prereg = PREREGISTRATION.resolve()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_seal_digest": runner.canonical_digest(seal.manifest()),
        "plan_digest": plan.digest,
        "recipes": [asdict(recipe) | {"digest": recipe.digest} for recipe in RECIPES],
        "seeds": list(SEEDS),
        "settings": asdict(settings),
        "quality_alignment_zero_event_preflight_digest": preflight_digest,
        "member_rows": EXPECTED_MEMBER_ROWS,
        "ensemble_rows": EXPECTED_ENSEMBLE_ROWS,
        "preregistration": str(prereg),
        "preregistration_sha256": runner.file_sha256(prereg),
        "executor_sha256": runner.file_sha256(Path(__file__)),
        "again_method_source_sha256": runner.file_sha256(Path(again_temporal.__file__)),
        "cache_root": str(Path(args.cache_root).resolve()),
        "upstream_root": str(Path(args.upstream_root).resolve()),
        "shared_derived_root": str(Path(args.shared_derived_root).resolve()),
        "no_again_artifact_reuse": True,
        "explicitly_nonpromotable": True,
        "outer_test_scores_used": False,
    }
    return {**payload, "digest": runner.canonical_digest(payload)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = validate_args(args)
    cache = runner.compact.Veatic21CompactCache(
        args.cache_root,
        upstream_root=args.upstream_root,
        identity_manifest_path=args.identity_manifest,
        verify_checksums=not bool(args.skip_checksums),
    )
    report = cache.validate()
    seal = runner.dataset_seal_from_report(report)
    plan = runner.build_plan(seal)
    dataset = runner.materialize_dense_dataset(
        cache=cache, seal=seal, shared_root=args.shared_derived_root
    )
    preflight = preflight_quality_alignment_audit(cache, dataset, plan)
    if not preflight["passed"]:
        raise ParityDiscoveryError(
            f"quality/alignment/zero-event preflight failed: {preflight['checks']}"
        )
    identity = _run_identity(
        args=args,
        seal=seal,
        plan=plan,
        settings=settings,
        preflight_digest=str(preflight["digest"]),
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    runner.atomic_json(output_root / "quality_alignment_zero_event_preflight.json", preflight)
    identity_path = output_root / "run_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ParityDiscoveryError("parity run identity mismatch")
    else:
        runner.atomic_json(identity_path, identity)
    schedule = {
        "schema_version": SCHEMA_VERSION,
        "member_keys": [list(key) for key in member_keys(smoke=bool(args.smoke))],
        "ensemble_keys": [list(key) for key in ensemble_keys(smoke=bool(args.smoke))],
        "outer_test_scores_used": False,
        "explicitly_nonpromotable": True,
        "no_again_artifact_reuse": True,
        "event_metric_policy": EVENT_METRIC_POLICY,
        "quality_alignment_zero_event_preflight_digest": preflight["digest"],
    }
    runner.atomic_json(output_root / "schedule.json", schedule)
    if args.dry_run:
        return {"status": "dry_run", "run_identity_digest": identity["digest"], **schedule}
    if args.audit_only:
        audit = audit_outputs(output_root, smoke=bool(args.smoke))
        if not audit["passed"]:
            raise ParityDiscoveryError(f"parity audit failed: {audit['checks']}")
        return {"status": "audited", "run_identity_digest": identity["digest"], "audit": audit}

    modeling.require_mlx_gpu()
    target_all = np.asarray(dataset.target_values[TARGET_NAME], dtype=np.float32)
    target_valid_all = np.asarray(dataset.target_valid[TARGET_NAME], dtype=bool)
    ar_x_all, ar_context_all, _ = program.canonical_ar_history_features(dataset.arousal, dataset.video_id)
    outers = plan.outer_folds[:1] if args.smoke else plan.outer_folds
    inner_numbers = INNER_FOLDS[:1] if args.smoke else INNER_FOLDS
    recipes = RECIPES[:1] if args.smoke else RECIPES
    seeds = SEEDS[:1] if args.smoke else SEEDS
    member_rows: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[int, int, str, int], tuple[np.ndarray, np.ndarray, float]] = {}

    for outer in outers:
        inner_by_number = {item.fold: item for item in outer.inner_folds}
        for inner_number in inner_numbers:
            inner = inner_by_number[inner_number]
            prepared_by_representation: dict[tuple[str, int], execution.PreparedFeatures] = {}
            pca_audit_by_representation: dict[tuple[str, int], dict[str, Any]] = {}
            for recipe in recipes:
                key = (recipe.feature_family, recipe.pca_width)
                if key not in prepared_by_representation:
                    prepared_by_representation[key] = execution._prepare_features(
                        dataset=dataset,
                        plan=plan,
                        recipe=_feature_recipe(recipe),
                        outer_fold=outer.outer_fold,
                        inner_fold=inner.fold,
                        args=args,
                    )
                    pca_audit_by_representation[key] = validate_veatic_pca_provenance(
                        prepared=prepared_by_representation[key],
                        dataset=dataset,
                        plan=plan,
                        shared_derived_root=args.shared_derived_root,
                    )
            reference = next(iter(prepared_by_representation.values()))
            train_rows = reference.train_rows
            test_rows = reference.test_rows
            for prepared in prepared_by_representation.values():
                if not np.array_equal(prepared.train_rows, train_rows) or not np.array_equal(prepared.test_rows, test_rows):
                    raise ParityDiscoveryError("recipe representations do not share row ownership")
            train_cont = np.maximum(target_all[train_rows], 0.0).astype(np.float32)
            test_cont = np.maximum(target_all[test_rows], 0.0).astype(np.float32)
            train_valid = (
                target_valid_all[train_rows]
                & dataset.quality_valid[train_rows]
                & ar_context_all[train_rows]
            )
            test_valid = (
                target_valid_all[test_rows]
                & dataset.quality_valid[test_rows]
                & ar_context_all[test_rows]
            )
            threshold = float(np.quantile(train_cont[train_valid], 0.90))
            if threshold <= 0:
                raise ParityDiscoveryError("event q90 must be positive for parity equivalence")
            train_event = (train_cont >= threshold).astype(np.float32)
            test_event = (test_cont >= threshold).astype(np.float32)
            if np.unique(train_event[train_valid]).size != 2 or np.unique(test_event[test_valid]).size != 2:
                raise ParityDiscoveryError("parity event panel lacks both classes")
            panel_stats = event_panel_stats(
                labels=test_event,
                valid=test_valid,
                videos=reference.test.video_id,
                global_rows=test_rows,
            )
            expected_panel = next(
                row
                for row in preflight["event_panels"]
                if row["outer_fold"] == outer.outer_fold
                and row["inner_fold"] == inner.fold
            )
            if (
                not math.isclose(
                    threshold,
                    float(expected_panel["event_threshold"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or panel_stats["heldout_label_digest"]
                != expected_panel["heldout_label_digest"]
                or panel_stats["heldout_valid_row_digest"]
                != expected_panel["heldout_valid_row_digest"]
            ):
                raise ParityDiscoveryError("runtime event panel drifted from preflight")

            for seed in seeds:
                namespace = (
                    f"veatic21|again-method-parity|outer{outer.outer_fold}|inner{inner.fold}|seed{seed}"
                )
                ownership = program.build_inner_video_ownership(
                    reference.train.video_id, namespace=namespace
                )
                ar_root = output_root / "ar" / f"outer_{outer.outer_fold}" / f"inner_{inner.fold}" / f"seed_{seed}"
                ar = fit_frozen_dual_ar(
                    x_train=ar_x_all[train_rows],
                    x_test=ar_x_all[test_rows],
                    event=train_event,
                    continuous=train_cont,
                    valid=train_valid,
                    videos=reference.train.video_id,
                    ownership=ownership,
                    seed=seed,
                    output_root=ar_root,
                    identity_base={
                        "run_identity_digest": identity["digest"],
                        "outer_fold": outer.outer_fold,
                        "inner_fold": inner.fold,
                        "target": TARGET_NAME,
                    },
                    settings=settings,
                )
                ar_score = pr_auc(test_event[test_valid], ar.test_event_logit[test_valid])
                for recipe in recipes:
                    prepared = prepared_by_representation[(recipe.feature_family, recipe.pca_width)]
                    pca_audit = pca_audit_by_representation[
                        (recipe.feature_family, recipe.pca_width)
                    ]
                    train_x = clean_or_enriched_view(
                        prepared.train.x_temporal,
                        pca_width=recipe.pca_width,
                        input_variant=recipe.input_variant,
                    )
                    test_x = clean_or_enriched_view(
                        prepared.test.x_temporal,
                        pca_width=recipe.pca_width,
                        input_variant=recipe.input_variant,
                    )
                    model_root = (
                        output_root
                        / "models"
                        / f"outer_{outer.outer_fold}"
                        / f"inner_{inner.fold}"
                        / recipe.name
                        / f"seed_{seed}"
                    )
                    residual = fit_dual_residual(
                        x_train=train_x,
                        x_test=test_x,
                        event=train_event,
                        continuous=train_cont,
                        valid=train_valid,
                        ownership=ownership,
                        frozen_ar=ar,
                        pca_width=recipe.pca_width,
                        seed=seed,
                        output_root=model_root,
                        identity_base={
                            "run_identity_digest": identity["digest"],
                            "outer_fold": outer.outer_fold,
                            "inner_fold": inner.fold,
                            "target": TARGET_NAME,
                            "recipe": asdict(recipe),
                            "recipe_digest": recipe.digest,
                            "train_video_ids": reference.train.video_id.tolist(),
                            "pca_provenance": dict(prepared.provenance),
                        },
                        settings=settings,
                    )
                    real_score = pr_auc(
                        test_event[test_valid], residual.test_event_logit[test_valid]
                    )
                    row = {
                        "outer_fold": outer.outer_fold,
                        "inner_fold": inner.fold,
                        "recipe": recipe.name,
                        "seed": seed,
                        "real_pr_auc": real_score,
                        "ar_pr_auc": ar_score,
                        "delta_vs_ar_pr_auc": real_score - ar_score,
                        "event_threshold": threshold,
                        "test_prevalence": panel_stats["pooled_prevalence"],
                        "test_rows": panel_stats["pooled_valid_rows"],
                        **panel_stats,
                        "quality_alignment_zero_event_preflight_digest": preflight["digest"],
                        **pca_audit,
                        "residual_suppressed": residual.suppressed,
                        "residual_best_epoch": residual.best_epoch,
                        "residual_epochs_run": residual.epochs_run,
                        "ar_best_epoch": ar.best_epoch,
                        "ar_epochs_run": ar.epochs_run,
                        "frozen_ar_identity_digest": ar.identity_digest,
                        "residual_identity_digest": residual.identity_digest,
                        "prediction_digest": runner.array_digest(residual.test_event_logit[test_valid]),
                        "ar_prediction_digest": runner.array_digest(ar.test_event_logit[test_valid]),
                        "outer_test_scores_used": False,
                        "explicitly_nonpromotable": True,
                        "no_again_artifact_reuse": True,
                    }
                    member_rows.append(row)
                    prediction_cache[(outer.outer_fold, inner.fold, recipe.name, seed)] = (
                        residual.test_event_logit[test_valid].copy(),
                        ar.test_event_logit[test_valid].copy(),
                        real_score,
                    )
                    runner.atomic_json(output_root / "member_rows.partial.json", member_rows)

    runner.atomic_json(output_root / "member_rows.json", member_rows)
    ensemble_rows: list[dict[str, Any]] = []
    if not args.smoke:
        for outer, inner, recipe_name in ensemble_keys():
            members = [prediction_cache[(outer, inner, recipe_name, seed)] for seed in SEEDS]
            real_logit = np.mean(np.stack([item[0] for item in members]), axis=0)
            ar_logit = np.mean(np.stack([item[1] for item in members]), axis=0)
            reference_rows = [
                row
                for row in member_rows
                if row["outer_fold"] == outer and row["inner_fold"] == inner and row["recipe"] == recipe_name
            ]
            threshold = float(reference_rows[0]["event_threshold"])
            # Reconstruct the same held-out labels from any representation.
            outer_obj = plan.outer(outer)
            inner_obj = {item.fold: item for item in outer_obj.inner_folds}[inner]
            recipe = next(item for item in RECIPES if item.name == recipe_name)
            prepared = execution._prepare_features(
                dataset=dataset,
                plan=plan,
                recipe=_feature_recipe(recipe),
                outer_fold=outer,
                inner_fold=inner_obj.fold,
                args=args,
            )
            valid = (
                target_valid_all[prepared.test_rows]
                & dataset.quality_valid[prepared.test_rows]
                & ar_context_all[prepared.test_rows]
            )
            labels = (np.maximum(target_all[prepared.test_rows], 0.0) >= threshold).astype(np.float32)[valid]
            real_score = pr_auc(labels, real_logit)
            ar_score = pr_auc(labels, ar_logit)
            ensemble_rows.append(
                {
                    "outer_fold": outer,
                    "inner_fold": inner,
                    "recipe": recipe_name,
                    "member_seeds": list(SEEDS),
                    "real_pr_auc": real_score,
                    "ar_pr_auc": ar_score,
                    "delta_vs_ar_pr_auc": real_score - ar_score,
                    "event_threshold": threshold,
                    "test_prevalence": reference_rows[0]["test_prevalence"],
                    "test_rows": reference_rows[0]["test_rows"],
                    **{
                        key: reference_rows[0][key]
                        for key in (
                            "event_metric_policy",
                            "pooled_valid_rows",
                            "pooled_positive_rows",
                            "pooled_prevalence",
                            "heldout_video_count",
                            "event_video_count",
                            "zero_event_video_count",
                            "zero_event_video_ids",
                            "heldout_label_digest",
                            "heldout_valid_row_digest",
                            "undefined_per_video_pr_auc_score_filled",
                            "zero_event_videos_excluded_from_pooled_negatives",
                            "quality_alignment_zero_event_preflight_digest",
                            "pca_parent_identity",
                            "pca_component_sha256",
                            "pca_base_family",
                            "pca_fit_train_row_digest",
                            "pca_fit_train_video_digest",
                            "pca_held_out_row_digest",
                            "pca_dataset_id",
                            "pca_fresh_veatic_only",
                            "pca_from_again",
                            "pca_from_original_veatic",
                            "pca_provenance_digest",
                        )
                    },
                    "member_mean_real_pr_auc": float(np.mean([item[2] for item in members])),
                    "ensemble_uplift_over_member_mean": real_score
                    - float(np.mean([item[2] for item in members])),
                    "prediction_digest": runner.array_digest(real_logit),
                    "ar_prediction_digest": runner.array_digest(ar_logit),
                    "outer_test_scores_used": False,
                    "explicitly_nonpromotable": True,
                    "no_again_artifact_reuse": True,
                }
            )
        runner.atomic_json(output_root / "ensemble_rows.json", ensemble_rows)
    audit = audit_outputs(output_root, smoke=bool(args.smoke))
    if not audit["passed"]:
        raise ParityDiscoveryError(f"completed matrix failed audit: {audit['checks']}")
    summaries = {} if args.smoke else _recipe_summary(member_rows, ensemble_rows)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "run_identity_digest": identity["digest"],
        "member_rows": len(member_rows),
        "ensemble_rows": len(ensemble_rows),
        "audit": audit,
        "recipe_summaries": summaries,
        "any_credible_recipe": any(
            row["member_credible"] and row["ensemble_credible"] for row in summaries.values()
        ),
        "outer_test_scores_used": False,
        "explicitly_nonpromotable": True,
        "no_again_artifact_reuse": True,
        "completed_at": utc_now(),
    }
    runner.atomic_json(output_root / "result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        print(f"VEATIC 2.1 AGAIN-method parity discovery failed: {exc}", file=os.sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
