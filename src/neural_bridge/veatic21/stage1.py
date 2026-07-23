"""Executable VEATIC 2.1 Stage-1 training contract."""

from __future__ import annotations

import importlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .contracts import TargetSpec
from .data import CanonicalSubstrate
from .evidence import (
    atomic_save_npz,
    atomic_write_json,
    average_precision_skill,
    digest_json,
    pooled_pr_auc,
    row_identity_digest,
    sha256_file,
)
from .pca_cache import load_event_pca_projection
from .preregistration import benchmark_partition_mask, targets_from_calibration
from .protocol import (
    causal_ar_features,
    event_labels,
    fit_event_threshold,
    future_target_values,
    target_support_mask,
)

_SCHEMA = "veatic21_stage1_child_plan_v2"
_HEADS = (
    "frozen_ar_plus_causal_temporal_residual",
    "frozen_ar_plus_gated_multiscale_temporal_residual",
)


@dataclass(frozen=True)
class Stage1CellConfig:
    """One explicit VEATIC-only learned spike discovery cell."""

    target_name: str
    fold: int
    seed: int
    pca_width: int
    head_family: Literal[
        "frozen_ar_plus_causal_temporal_residual",
        "frozen_ar_plus_gated_multiscale_temporal_residual",
    ]
    hidden_width: int
    learning_rate: float
    weight_decay: float
    residual_logit_cap: float
    batch_rows: int
    context_rows: tuple[int, ...] = (1, 2, 4, 6, 10)
    minimum_epochs: int = 50
    plateau_patience: int = 50
    nonconvergence_patience: int = 400

    def validate(self, plan: Mapping[str, Any]) -> None:
        matrix = plan["matrix"]
        if self.target_name not in {str(row["name"]) for row in matrix["targets"]}:
            raise ValueError("Stage-1 target is not registered")
        folds = {int(row["fold"]): row for row in matrix["folds"]}
        if self.fold not in folds:
            raise ValueError("Stage-1 fold is not registered")
        if self.pca_width not in {int(value) for value in folds[self.fold]["candidate_pca_widths"]}:
            raise ValueError("Stage-1 PCA width is not registered for this fold")
        if self.head_family not in _HEADS or self.head_family not in matrix["head_families"]:
            raise ValueError("Stage-1 head family is not registered")
        if self.seed not in {int(value) for value in matrix["comparison_seeds"]}:
            raise ValueError("Stage-1 comparison seed is not registered")
        if self.hidden_width <= 0 or self.batch_rows <= 0:
            raise ValueError("Stage-1 widths and batch size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.residual_logit_cap <= 0:
            raise ValueError("Stage-1 optimizer and residual values are invalid")
        if (
            tuple(sorted(set(self.context_rows))) != self.context_rows
            or min(self.context_rows) <= 0
        ):
            raise ValueError("Stage-1 context rows must be unique increasing positive integers")
        policy = plan["checkpoint_policy"]
        if (
            self.minimum_epochs != int(policy["minimum_epochs_before_termination"])
            or self.plateau_patience != int(policy["plateau_patience_epochs"])
            or self.nonconvergence_patience != int(policy["nonconvergence_patience_epochs"])
            or policy["eligible_from_epoch"] != 1
            or policy["maximum_epochs"] is not None
        ):
            raise ValueError("Stage-1 cell does not match the checkpoint contract")
        capacity_key = f"{self.head_family}:{self.hidden_width}"
        safe_batches = plan["capacity"]["safe_batch_rows_by_head_hidden_width"]
        if capacity_key not in safe_batches or self.batch_rows > int(safe_batches[capacity_key]):
            raise ValueError("Stage-1 cell exceeds measured MLX capacity")


@dataclass
class CheckpointSelector:
    """Select any improving checkpoint while enforcing convergence before stopping."""

    minimum_epochs: int = 50
    plateau_patience: int = 50
    best_epoch: int | None = None
    best_metric: float = -math.inf
    stale_epochs: int = 0

    def observe(self, epoch: int, metric: float) -> bool:
        if epoch < 1 or not math.isfinite(metric):
            raise ValueError("checkpoint observations require a positive epoch and finite metric")
        if metric > self.best_metric:
            self.best_epoch = epoch
            self.best_metric = metric
            self.stale_epochs = 0
            return True
        self.stale_epochs += 1
        return False

    def should_stop(self, epoch: int, *, optimizer_converged: bool) -> bool:
        return (
            epoch >= self.minimum_epochs
            and self.stale_epochs >= self.plateau_patience
            and optimizer_converged
        )

    def should_fail_nonconvergence(self, epoch: int, *, patience: int) -> bool:
        return epoch >= self.minimum_epochs and self.stale_epochs >= patience


def _physical_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (OSError, ValueError):
        return None


def probe_stage1_capacity(
    pca_manifest: Mapping[str, Any],
    *,
    batch_candidates: Sequence[int] = (32, 64, 128, 256, 512, 1024),
) -> dict[str, Any]:
    """Measure feasible MLX batches for the registered maximum multiscale input."""

    mx = importlib.import_module("mlx.core")
    nn = importlib.import_module("mlx.nn")
    optim = importlib.import_module("mlx.optimizers")
    folds = pca_manifest.get("folds", ())
    widths = sorted({int(width) for fold in folds for width in fold["candidate_widths"]})
    if not widths:
        raise ValueError("PCA manifest has no candidate widths")
    maximum_width = max(widths)
    hidden_candidates = tuple(width for width in (64, 128, 256, 512) if width <= maximum_width)
    context_count = 5
    input_width_by_head = {
        "frozen_ar_plus_causal_temporal_residual": maximum_width * (1 + context_count)
        + context_count,
        "frozen_ar_plus_gated_multiscale_temporal_residual": maximum_width * (1 + 2 * context_count)
        + context_count,
    }
    total_memory = _physical_memory_bytes()
    memory_limit = int(total_memory * 0.25) if total_memory else None
    measurements: list[dict[str, Any]] = []

    class ProbeHead(nn.Module):
        def __init__(self, input_width: int, hidden: int) -> None:
            super().__init__()
            self.input = nn.Linear(input_width, hidden)
            self.gate = nn.Linear(input_width, 1)
            self.output = nn.Linear(hidden, 1)

        def __call__(self, values):
            residual = self.output(nn.gelu(self.input(values)))
            return mx.sigmoid(self.gate(values)) * residual

    def loss_fn(model_obj, values, targets):
        logits = model_obj(values)
        return nn.losses.binary_cross_entropy(logits, targets, with_logits=True, reduction="mean")

    for head_family, input_width in input_width_by_head.items():
        for hidden in hidden_candidates:
            for batch_rows in batch_candidates:
                reset_peak = getattr(mx, "reset_peak_memory", None)
                if reset_peak:
                    reset_peak()
                started = time.perf_counter()
                try:
                    mx.random.seed(20_260_722 + hidden + batch_rows)
                    model = ProbeHead(input_width, hidden)
                    loss_and_grad = nn.value_and_grad(model, loss_fn)
                    values = mx.random.normal((batch_rows, input_width))
                    targets = mx.zeros((batch_rows, 1))
                    loss, gradients = loss_and_grad(model, values, targets)
                    gradients, _ = optim.clip_grad_norm(gradients, 1.0)
                    mx.eval(loss, gradients)
                    peak_fn = getattr(mx, "get_peak_memory", None)
                    peak_bytes = int(peak_fn()) if peak_fn else None
                    feasible = (
                        memory_limit is None or peak_bytes is None or peak_bytes <= memory_limit
                    )
                    error = None
                except (MemoryError, RuntimeError, ValueError) as exc:
                    peak_bytes = None
                    feasible = False
                    error = type(exc).__name__
                measurements.append(
                    {
                        "batch_rows": batch_rows,
                        "elapsed_seconds": time.perf_counter() - started,
                        "error": error,
                        "feasible": feasible,
                        "head_family": head_family,
                        "hidden_width": hidden,
                        "input_width": input_width,
                        "peak_bytes": peak_bytes,
                    }
                )

    feasible = [row for row in measurements if row["feasible"]]
    if not feasible:
        raise RuntimeError("MLX capacity probe found no feasible Stage-1 configuration")
    return {
        "backend": "mlx",
        "input_width_by_head_family": input_width_by_head,
        "maximum_pca_width": maximum_width,
        "memory_limit_bytes": memory_limit,
        "measurements": measurements,
        "physical_memory_bytes": total_memory,
        "safe_batch_rows_by_head_hidden_width": {
            f"{head_family}:{hidden}": max(
                row["batch_rows"]
                for row in feasible
                if row["head_family"] == head_family and row["hidden_width"] == hidden
            )
            for head_family in input_width_by_head
            for hidden in hidden_candidates
            if any(
                row["head_family"] == head_family and row["hidden_width"] == hidden
                for row in feasible
            )
        },
    }


def build_stage1_plan(
    preregistration: Mapping[str, Any],
    calibration: Mapping[str, Any],
    pca_manifest: Mapping[str, Any],
    capacity: Mapping[str, Any],
    ar_benchmark: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind current VEATIC artifacts to one executable, lazily expanded matrix."""

    preregistration_sha = preregistration.get("preregistration_sha256")
    if calibration.get("preregistration_sha256") != preregistration_sha:
        raise ValueError("calibration does not belong to the preregistration")
    if pca_manifest.get("preregistration_sha256") != preregistration_sha:
        raise ValueError("PCA cache does not belong to the preregistration")
    training = preregistration["training"]
    if int(training["minimum_epochs_before_termination"]) != 50:
        raise ValueError("Stage-1 requires the registered 50-epoch minimum")
    if training["checkpoint_eligibility"] != "every_completed_validation_from_epoch_1":
        raise ValueError("Stage-1 requires checkpoint eligibility from epoch 1")
    if training["last_checkpoint_preference"] is not False:
        raise ValueError("Stage-1 cannot prefer the final checkpoint")
    if int(training["plateau_patience_epochs"]) != 50:
        raise ValueError("Stage-1 requires the registered validation plateau")
    if int(training["nonconvergence_patience_epochs"]) != 400:
        raise ValueError("Stage-1 requires the registered nonconvergence rule")

    targets = targets_from_calibration(calibration)
    if not targets:
        raise ValueError("Stage-1 plan requires registered targets")
    folds = sorted(pca_manifest["folds"], key=lambda row: int(row["fold"]))
    artifacts = {
        "calibration_sha256": calibration["calibration_sha256"],
        "pca_manifest_sha256": pca_manifest["manifest_sha256"],
        "preregistration_sha256": preregistration_sha,
        "stage1_code_sha256": sha256_file(Path(__file__)),
    }
    purpose = "executor_validation"
    if ar_benchmark is not None:
        _require_self_digest(ar_benchmark, "summary_sha256")
        if (
            ar_benchmark.get("request_sha256") is None
            or ar_benchmark.get("target_count") != len(targets)
            or ar_benchmark.get("expected_cells")
            != len(targets) * len(folds) * len(training["comparison_seed_panel"])
            or ar_benchmark.get("completed_cells") != ar_benchmark.get("expected_cells")
            or ar_benchmark.get("invalid_cells") != 0
        ):
            raise ValueError("AR benchmark does not cover the registered Stage-1 matrix")
        artifacts["ar_benchmark_sha256"] = ar_benchmark["summary_sha256"]
        purpose = "spike_discovery"
    plan: dict[str, Any] = {
        "artifacts": artifacts,
        "capacity": {
            "backend": capacity["backend"],
            "input_width_by_head_family": dict(capacity["input_width_by_head_family"]),
            "maximum_pca_width": int(capacity["maximum_pca_width"]),
            "safe_batch_rows_by_head_hidden_width": dict(
                capacity["safe_batch_rows_by_head_hidden_width"]
            ),
        },
        "checkpoint_policy": {
            "eligible_from_epoch": 1,
            "maximum_epochs": None,
            "minimum_epochs_before_termination": 50,
            "nonconvergence_patience_epochs": int(training["nonconvergence_patience_epochs"]),
            "optimizer_convergence_required": True,
            "plateau_patience_epochs": int(training["plateau_patience_epochs"]),
            "selection_metric": training["checkpoint_metric"],
            "tie_break": "earliest_checkpoint",
        },
        "matrix": {
            "comparison_seeds": list(training["comparison_seed_panel"]),
            "folds": [
                {
                    "candidate_pca_widths": list(fold["candidate_widths"]),
                    "directory": fold["directory"],
                    "fold": int(fold["fold"]),
                }
                for fold in folds
            ],
            "head_families": list(preregistration["heads"]["label_assisted_discovery"]),
            "representations": ["tribe_cortical"],
            "targets": [
                {
                    "horizon_rows": list(target.horizon_rows),
                    "label": target.label,
                    "name": target.name,
                    "quantile": target.quantile,
                    "transform": target.transform,
                }
                for target in targets
            ],
        },
        "promotion": {
            "ar_fallback_scope": "whole_fold_seed_from_inner_validation",
            "requires_positive_delta_vs_frozen_ar": True,
            "sealed_tail_labels": True,
        },
        "purpose": purpose,
        "schema": _SCHEMA,
    }
    plan["plan_sha256"] = digest_json(plan)
    return plan


def _owned_rows(
    video_id: np.ndarray,
    row_index: np.ndarray,
    mask: np.ndarray,
) -> dict[str, list[int]]:
    rows = {video: [] for video in sorted(set(video_id.astype(str)), key=int)}
    for video, row, owned in zip(video_id, row_index, mask, strict=True):
        if owned:
            rows[str(video)].append(int(row))
    return rows


def _target(calibration: Mapping[str, Any], name: str) -> TargetSpec:
    matches = [target for target in targets_from_calibration(calibration) if target.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one calibrated target named {name!r}")
    return matches[0]


def _require_self_digest(record: Mapping[str, Any], field: str) -> None:
    expected = record.get(field)
    if not isinstance(expected, str):
        raise ValueError(f"artifact is missing {field}")
    payload = dict(record)
    payload.pop(field)
    if digest_json(payload) != expected:
        raise ValueError(f"artifact failed its {field} integrity check")


def _fit_fresh_ar(
    matrix: np.ndarray,
    targets: np.ndarray,
    future_values: np.ndarray,
    target: TargetSpec,
    train_mask: np.ndarray,
    video_id: np.ndarray,
    *,
    seed: int,
    c_grid: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    """Select AR regularization inside the owning training videos, then refit once."""

    train_videos = np.asarray(sorted(set(video_id[train_mask].astype(str)), key=int))
    rng = np.random.default_rng(seed)
    panels = np.array_split(train_videos[rng.permutation(len(train_videos))], 3)
    best: tuple[float, float] | None = None
    selected_c = 0.0
    for c_value in c_grid:
        fold_scores = []
        for panel in panels:
            inner_validation = train_mask & np.isin(video_id.astype(str), panel)
            inner_train = train_mask & ~np.isin(video_id.astype(str), panel)
            inner_threshold = fit_event_threshold(future_values, inner_train, target)
            inner_targets = event_labels(future_values, inner_threshold)
            if (
                len(np.unique(inner_targets[inner_train])) != 2
                or len(np.unique(inner_targets[inner_validation])) != 2
            ):
                raise ValueError("fresh AR inner folds must contain both event classes")
            scaler = StandardScaler().fit(matrix[inner_train])
            model = LogisticRegression(
                C=float(c_value),
                class_weight="balanced",
                max_iter=5_000,
                random_state=seed,
                solver="lbfgs",
                tol=1e-6,
            ).fit(scaler.transform(matrix[inner_train]), inner_targets[inner_train])
            scores = model.decision_function(scaler.transform(matrix[inner_validation]))
            fold_scores.append(average_precision_skill(inner_targets[inner_validation], scores))
        rank = (float(np.mean(fold_scores)), -float(c_value))
        if best is None or rank > best:
            best = rank
            selected_c = float(c_value)
    scaler = StandardScaler().fit(matrix[train_mask])
    model = LogisticRegression(
        C=selected_c,
        class_weight="balanced",
        max_iter=5_000,
        random_state=seed,
        solver="lbfgs",
        tol=1e-6,
    ).fit(scaler.transform(matrix[train_mask]), targets[train_mask])
    return (
        model.decision_function(scaler.transform(matrix)).astype(np.float32),
        selected_c,
        {
            "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
            "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
            "coefficient": np.asarray(model.coef_[0], dtype=np.float64),
            "intercept": np.asarray(model.intercept_, dtype=np.float64),
        },
    )


def run_stage1_ar_benchmark(
    substrate: CanonicalSubstrate,
    preregistration: Mapping[str, Any],
    calibration: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Benchmark fresh AR for every registered target, fold, and comparison seed."""

    if calibration.get("preregistration_sha256") != preregistration.get("preregistration_sha256"):
        raise ValueError("AR benchmark artifacts do not share one preregistration")
    targets = targets_from_calibration(calibration)
    if not targets:
        raise ValueError("AR benchmark requires registered targets")
    folds = preregistration["split"]["inner_grouped_video_folds"]
    seeds = tuple(int(seed) for seed in preregistration["training"]["comparison_seed_panel"])
    request = {
        "schema": "veatic21_stage1_ar_benchmark_request_v1",
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "calibration_sha256": calibration["calibration_sha256"],
        "targets": [target.name for target in targets],
        "folds": len(folds),
        "seeds": list(seeds),
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    request_sha256 = digest_json(request)
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "request.json"
    if request_path.is_file():
        saved = json.loads(request_path.read_text(encoding="utf-8"))
        if digest_json(saved) != request_sha256:
            raise RuntimeError("refusing AR benchmark resume because the request changed")
    else:
        atomic_write_json(request_path, request)

    all_features = substrate.load_features(substrate.video_ids, ("diagnostics_only",))
    development_mask = benchmark_partition_mask(all_features, preregistration["split"], "train")
    features = all_features.subset(development_mask)
    labels = substrate.load_labels(
        substrate.video_ids,
        row_indices=_owned_rows(all_features.video_id, all_features.row_index, development_mask),
        stage="stage1_ar_benchmark_train_labels_only",
    )
    records: list[dict[str, Any]] = []
    target_root = output_dir / "targets"
    target_root.mkdir(exist_ok=True)
    for target in targets:
        target_path = target_root / f"{target.name}.json"
        if target_path.is_file():
            saved = json.loads(target_path.read_text(encoding="utf-8"))
            _require_self_digest(saved, "target_record_sha256")
            if saved.get("request_sha256") != request_sha256:
                raise RuntimeError(f"refusing changed AR target resume: {target.name}")
            records.extend(saved["records"])
            continue
        future = future_target_values(labels, target)
        support = target_support_mask(features, target)
        target_ar, available = causal_ar_features(labels, target)
        ar_matrix = np.concatenate([target_ar, available.astype(np.float64)], axis=1)
        target_records: list[dict[str, Any]] = []
        for fold, validation_videos in enumerate(folds):
            validation_mask = np.isin(features.video_id.astype(str), validation_videos) & support
            train_mask = ~np.isin(features.video_id.astype(str), validation_videos) & support
            threshold = fit_event_threshold(future, train_mask, target)
            binary = event_labels(future, threshold)
            validation = np.flatnonzero(validation_mask)
            prevalence = float(np.mean(binary[validation]))
            for seed in seeds:
                record: dict[str, Any] = {
                    "target": target.name,
                    "fold": fold,
                    "seed": seed,
                    "train_rows": int(train_mask.sum()),
                    "validation_rows": int(validation_mask.sum()),
                    "event_threshold": threshold,
                    "event_prevalence": prevalence,
                }
                try:
                    logits, selected_c, _ = _fit_fresh_ar(
                        ar_matrix,
                        binary,
                        future,
                        target,
                        train_mask,
                        features.video_id,
                        seed=seed,
                    )
                    pr_auc = pooled_pr_auc(binary[validation], logits[validation])
                    record.update(
                        {
                            "status": "complete",
                            "fresh_ar_c": selected_c,
                            "fresh_ar_pr_auc": pr_auc,
                            "fresh_ar_average_precision_skill": average_precision_skill(
                                binary[validation], logits[validation]
                            ),
                        }
                    )
                except (RuntimeError, ValueError) as exc:
                    record.update(
                        {
                            "status": "invalid_support_or_fit",
                            "failure": str(exc),
                        }
                    )
                target_records.append(record)
        target_record = {
            "schema": "veatic21_stage1_ar_target_v1",
            "request_sha256": request_sha256,
            "target": target.name,
            "records": target_records,
        }
        target_record["target_record_sha256"] = digest_json(target_record)
        atomic_write_json(target_path, target_record)
        records.extend(target_records)

    complete = [record for record in records if record["status"] == "complete"]
    summary: dict[str, Any] = {
        "schema": "veatic21_stage1_ar_benchmark_v1",
        "request_sha256": request_sha256,
        "benchmark_test_labels_accessed": False,
        "promotable": False,
        "target_count": len(targets),
        "expected_cells": len(targets) * len(folds) * len(seeds),
        "completed_cells": len(complete),
        "invalid_cells": len(records) - len(complete),
        "records": records,
    }
    summary["summary_sha256"] = digest_json(summary)
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def _causal_design(
    projected: np.ndarray,
    video_id: np.ndarray,
    row_index: np.ndarray,
    *,
    family: str,
    context_rows: Sequence[int],
) -> np.ndarray:
    """Build causal temporal inputs without crossing a video or dropping cold-start rows."""

    current = np.asarray(projected, dtype=np.float32)
    components: list[np.ndarray] = [current]
    availability: list[np.ndarray] = []
    videos = video_id.astype(str)
    for window in context_rows:
        past = np.zeros_like(current)
        mean = np.zeros_like(current)
        available = np.zeros(len(current), dtype=np.float32)
        for video in np.unique(videos):
            positions = np.flatnonzero(videos == video)
            positions = positions[np.argsort(row_index[positions])]
            rows = row_index[positions]
            lookup = {int(row): local for local, row in enumerate(rows)}
            prefix = np.vstack(
                [
                    np.zeros((1, current.shape[1]), dtype=np.float32),
                    np.cumsum(current[positions], axis=0),
                ]
            )
            for local, (position, row) in enumerate(zip(positions, rows, strict=True)):
                prior = lookup.get(int(row) - window)
                if prior is None or local - prior != window:
                    continue
                past[position] = current[positions[prior]]
                mean[position] = (prefix[local + 1] - prefix[prior]) / (local - prior + 1)
                available[position] = 1.0
        components.append(current - past)
        if family == "frozen_ar_plus_gated_multiscale_temporal_residual":
            components.append(mean)
        availability.append(available[:, None])
    return np.concatenate([*components, *availability], axis=1).astype(np.float32)


def _optimizer_converged(losses: Sequence[float], *, window: int = 10) -> bool:
    """Detect a stable optimizer plateau without demanding noise-free mini-batch loss."""

    if len(losses) < window * 2 or not np.isfinite(losses[-window * 2 :]).all():
        return False
    previous = float(np.mean(losses[-window * 2 : -window]))
    recent = float(np.mean(losses[-window:]))
    relative_improvement = (previous - recent) / max(abs(previous), 1e-8)
    return relative_improvement <= 1e-3


def _train_mlx_residual(
    values: np.ndarray,
    targets: np.ndarray,
    ar_logits: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    config: Stage1CellConfig,
    checkpoint: Path,
    state_path: Path,
    request_sha256: str,
) -> tuple[np.ndarray, list[dict[str, float | int | bool]], CheckpointSelector]:
    mx = importlib.import_module("mlx.core")
    nn = importlib.import_module("mlx.nn")
    optim = importlib.import_module("mlx.optimizers")
    mx.random.seed(config.seed)

    class ResidualHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(values.shape[1], config.hidden_width)
            self.second = nn.Linear(config.hidden_width, config.hidden_width)
            self.output = nn.Linear(config.hidden_width, 1)
            self.gate = nn.Linear(values.shape[1], 1)

        def __call__(self, batch, floor):
            hidden = nn.gelu(self.first(batch))
            hidden = nn.gelu(self.second(hidden))
            residual = mx.tanh(self.output(hidden)) * config.residual_logit_cap
            if config.head_family == "frozen_ar_plus_gated_multiscale_temporal_residual":
                residual = residual * mx.sigmoid(self.gate(batch))
            return floor[:, None] + residual

    model = ResidualHead()
    optimizer = optim.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    prevalence = float(np.mean(targets[train_mask]))
    if not 0.0 < prevalence < 1.0:
        raise ValueError("Stage-1 training rows must contain both event classes")

    def loss_fn(model_obj, batch, floor, truth, weights):
        logits = model_obj(batch, floor)
        loss = nn.losses.binary_cross_entropy(logits, truth[:, None], with_logits=True)
        return mx.mean(loss * weights[:, None])

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    rng = np.random.default_rng(config.seed)
    selector = CheckpointSelector(
        minimum_epochs=config.minimum_epochs,
        plateau_patience=config.plateau_patience,
    )
    curve: list[dict[str, float | int | bool]] = []
    losses: list[float] = []

    def predict(indices: np.ndarray) -> np.ndarray:
        output = []
        for start in range(0, len(indices), config.batch_rows):
            rows = indices[start : start + config.batch_rows]
            logits = model(mx.array(values[rows]), mx.array(ar_logits[rows]))
            mx.eval(logits)
            output.append(np.asarray(logits).reshape(-1))
        return np.concatenate(output).astype(np.float32)

    epoch = 0
    while True:
        epoch += 1
        order = rng.permutation(train_indices)
        epoch_losses = []
        for start in range(0, len(order), config.batch_rows):
            rows = order[start : start + config.batch_rows]
            truth = targets[rows].astype(np.float32)
            weights = np.where(truth > 0.5, 0.5 / prevalence, 0.5 / (1.0 - prevalence)).astype(
                np.float32
            )
            loss, gradients = loss_and_grad(
                model,
                mx.array(values[rows]),
                mx.array(ar_logits[rows]),
                mx.array(truth),
                mx.array(weights),
            )
            gradients, _ = optim.clip_grad_norm(gradients, 1.0)
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
            epoch_losses.append(float(np.asarray(loss)))
        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)
        validation_scores = predict(validation_indices)
        model_skill = average_precision_skill(targets[validation_indices], validation_scores)
        ar_skill = average_precision_skill(
            targets[validation_indices], ar_logits[validation_indices]
        )
        metric = model_skill - ar_skill
        improved = selector.observe(epoch, metric)
        if improved:
            temporary = checkpoint.with_name(f".{checkpoint.stem}.tmp.npz")
            model.save_weights(str(temporary))
            temporary.replace(checkpoint)
        converged = _optimizer_converged(losses)
        curve.append(
            {
                "epoch": epoch,
                "improved": improved,
                "inner_skill_delta_vs_ar": metric,
                "train_loss": mean_loss,
                "optimizer_converged": converged,
            }
        )
        atomic_write_json(
            state_path,
            {
                "schema": "veatic21_stage1_cell_state_v1",
                "status": "training",
                "request_sha256": request_sha256,
                "epoch": epoch,
                "best_epoch": selector.best_epoch,
                "stale_epochs": selector.stale_epochs,
                "optimizer_converged": converged,
            },
        )
        if selector.should_stop(epoch, optimizer_converged=converged):
            break
        if selector.should_fail_nonconvergence(
            epoch,
            patience=config.nonconvergence_patience,
        ):
            raise RuntimeError(
                "Stage-1 candidate failed to converge after its declared validation plateau"
            )

    if selector.best_epoch is None:
        raise RuntimeError("Stage-1 training completed without an eligible checkpoint")
    model.load_weights(str(checkpoint))
    return predict(np.arange(len(values))), curve, selector


def run_stage1_discovery_cell(
    substrate: CanonicalSubstrate,
    preregistration: Mapping[str, Any],
    calibration: Mapping[str, Any],
    pca_manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    pca_root: Path,
    output_dir: Path,
    config: Stage1CellConfig,
) -> dict[str, Any]:
    """Train one non-sealed VEATIC learned residual cell on benchmark-train folds only."""

    if plan.get("schema") != _SCHEMA:
        raise ValueError("Stage-1 cell requires the current child-plan schema")
    _require_self_digest(preregistration, "preregistration_sha256")
    _require_self_digest(calibration, "calibration_sha256")
    _require_self_digest(pca_manifest, "manifest_sha256")
    _require_self_digest(plan, "plan_sha256")
    preregistration_sha256 = preregistration["preregistration_sha256"]
    if calibration.get("preregistration_sha256") != preregistration_sha256:
        raise ValueError("calibration does not belong to the Stage-1 preregistration")
    if pca_manifest.get("preregistration_sha256") != preregistration_sha256:
        raise ValueError("PCA manifest does not belong to the Stage-1 preregistration")
    required_artifacts = {
        "calibration_sha256": calibration["calibration_sha256"],
        "pca_manifest_sha256": pca_manifest["manifest_sha256"],
        "preregistration_sha256": preregistration_sha256,
        "stage1_code_sha256": sha256_file(Path(__file__)),
    }
    if any(
        plan.get("artifacts", {}).get(key) != value for key, value in required_artifacts.items()
    ):
        raise ValueError("Stage-1 plan does not bind the supplied artifacts")
    config.validate(plan)
    request = {
        "schema": "veatic21_stage1_cell_request_v1",
        "config": {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in config.__dict__.items()
        },
        "plan_sha256": plan["plan_sha256"],
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "calibration_sha256": calibration["calibration_sha256"],
        "pca_manifest_sha256": pca_manifest["manifest_sha256"],
        "stage1_code_sha256": sha256_file(Path(__file__)),
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    request_sha256 = digest_json(request)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file():
        saved_request = json.loads((output_dir / "request.json").read_text(encoding="utf-8"))
        if digest_json(saved_request) != request_sha256:
            raise RuntimeError("refusing Stage-1 resume because the request changed")
        state = json.loads((output_dir / "state.json").read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if state.get("status") != "complete" or state.get("request_sha256") != request_sha256:
            raise RuntimeError("refusing incomplete Stage-1 result reuse")
        if state.get("metrics_sha256") != sha256_file(metrics_path):
            raise RuntimeError("refusing tampered Stage-1 metrics")
        for name, expected in metrics.get("artifact_sha256", {}).items():
            path = output_dir / name
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"refusing missing or changed Stage-1 artifact: {name}")
        return metrics

    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("refusing to overwrite a partial or unrecognized Stage-1 run")

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "request.json", request)
    atomic_write_json(
        output_dir / "state.json",
        {
            "schema": "veatic21_stage1_cell_state_v1",
            "status": "training",
            "request_sha256": request_sha256,
        },
    )
    all_features = substrate.load_features(substrate.video_ids, ("tribe_cortical",))
    development_mask = benchmark_partition_mask(all_features, preregistration["split"], "train")
    features = all_features.subset(development_mask)
    labels = substrate.load_labels(
        substrate.video_ids,
        row_indices=_owned_rows(all_features.video_id, all_features.row_index, development_mask),
        stage="stage1_benchmark_train_labels_only",
    )
    projected = load_event_pca_projection(
        features,
        preregistration,
        pca_manifest,
        pca_root,
        fold=config.fold,
        width=config.pca_width,
    )
    target = _target(calibration, config.target_name)
    future = future_target_values(labels, target)
    support = target_support_mask(features, target)
    validation_videos = preregistration["split"]["inner_grouped_video_folds"][config.fold]
    validation_mask = np.isin(features.video_id.astype(str), validation_videos) & support
    train_mask = ~np.isin(features.video_id.astype(str), validation_videos) & support
    threshold = fit_event_threshold(future, train_mask, target)
    binary = event_labels(future, threshold)
    ar_values, ar_available = causal_ar_features(labels, target)
    ar_matrix = np.concatenate([ar_values, ar_available.astype(np.float64)], axis=1)
    ar_logits, ar_c, ar_artifact = _fit_fresh_ar(
        ar_matrix,
        binary,
        future,
        target,
        train_mask,
        features.video_id,
        seed=config.seed,
    )
    design = _causal_design(
        projected,
        features.video_id,
        features.row_index,
        family=config.head_family,
        context_rows=config.context_rows,
    )
    scaler = StandardScaler().fit(design[train_mask])
    design = scaler.transform(design).astype(np.float32)
    atomic_save_npz(
        output_dir / "preprocessing.npz",
        {
            **ar_artifact,
            "design_scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
            "design_scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
        },
    )
    checkpoint = output_dir / "best-checkpoint.npz"
    try:
        model_scores, curve, selector = _train_mlx_residual(
            design,
            binary,
            ar_logits,
            train_mask,
            validation_mask,
            config,
            checkpoint,
            output_dir / "state.json",
            request_sha256,
        )
    except Exception as exc:
        atomic_write_json(
            output_dir / "state.json",
            {
                "schema": "veatic21_stage1_cell_state_v1",
                "status": "failed",
                "request_sha256": request_sha256,
                "failure": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    validation = np.flatnonzero(validation_mask)
    ar_pr_auc = pooled_pr_auc(binary[validation], ar_logits[validation])
    model_pr_auc = pooled_pr_auc(binary[validation], model_scores[validation])
    delta = average_precision_skill(
        binary[validation], model_scores[validation]
    ) - average_precision_skill(binary[validation], ar_logits[validation])
    use_residual = delta > 0.0
    selected_scores = model_scores if use_residual else ar_logits
    atomic_save_npz(
        output_dir / "validation-predictions.npz",
        {
            "video_id": features.video_id[validation].astype("U3"),
            "row_index": features.row_index[validation].astype(np.int32),
            "target": binary[validation].astype(np.int8),
            "ar_score": ar_logits[validation],
            "model_score": model_scores[validation],
            "selected_score": selected_scores[validation],
        },
    )
    atomic_write_json(output_dir / "training-curve.json", {"records": curve})
    artifact_sha256 = {
        name: sha256_file(output_dir / name)
        for name in (
            "best-checkpoint.npz",
            "preprocessing.npz",
            "training-curve.json",
            "validation-predictions.npz",
        )
    }
    metrics: dict[str, Any] = {
        "schema": "veatic21_stage1_cell_metrics_v1",
        "promotable": False,
        "benchmark_test_labels_accessed": False,
        "target": target.name,
        "fold": config.fold,
        "seed": config.seed,
        "pca_width": config.pca_width,
        "head_family": config.head_family,
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "event_threshold": threshold,
        "event_prevalence": float(np.mean(binary[validation])),
        "fresh_ar_c": ar_c,
        "fresh_ar_pr_auc": ar_pr_auc,
        "learned_head_pr_auc": model_pr_auc,
        "inner_average_precision_skill_delta_vs_frozen_ar": delta,
        "best_epoch": selector.best_epoch,
        "epochs_completed": len(curve),
        "whole_fold_seed_uses_residual": use_residual,
        "selected_pr_auc": pooled_pr_auc(binary[validation], selected_scores[validation]),
        "checkpoint_sha256": sha256_file(checkpoint),
        "train_row_sha256": row_identity_digest(
            features.video_id[train_mask], features.row_index[train_mask]
        ),
        "validation_row_sha256": row_identity_digest(
            features.video_id[validation_mask], features.row_index[validation_mask]
        ),
        "design_width": int(design.shape[1]),
        "artifact_sha256": artifact_sha256,
        "request_sha256": request_sha256,
    }
    atomic_write_json(metrics_path, metrics)
    atomic_write_json(
        output_dir / "state.json",
        {
            "schema": "veatic21_stage1_cell_state_v1",
            "status": "complete",
            "request_sha256": request_sha256,
            "metrics_sha256": sha256_file(metrics_path),
        },
    )
    return metrics


def write_stage1_plan(path: Path, plan: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(plan))


__all__ = [
    "CheckpointSelector",
    "Stage1CellConfig",
    "build_stage1_plan",
    "probe_stage1_capacity",
    "run_stage1_ar_benchmark",
    "run_stage1_discovery_cell",
    "write_stage1_plan",
]
