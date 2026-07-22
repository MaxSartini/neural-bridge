"""Executable VEATIC 2.1 Stage-1 training contract."""

from __future__ import annotations

import importlib
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .event_screen import targets_from_calibration
from .evidence import atomic_write_json, digest_json

_SCHEMA = "veatic21_stage1_child_plan_v1"


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
    input_width = maximum_width * 4 + 1
    total_memory = _physical_memory_bytes()
    memory_limit = int(total_memory * 0.25) if total_memory else None
    measurements: list[dict[str, Any]] = []

    class ProbeHead(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.input = nn.Linear(input_width, hidden)
            self.gate = nn.Linear(input_width, 1)
            self.output = nn.Linear(hidden, 1)

        def __call__(self, values):
            residual = self.output(nn.gelu(self.input(values)))
            return mx.sigmoid(self.gate(values)) * residual

    for hidden in hidden_candidates:
        for batch_rows in batch_candidates:
            reset_peak = getattr(mx, "reset_peak_memory", None)
            if reset_peak:
                reset_peak()
            started = time.perf_counter()
            try:
                mx.random.seed(20_260_722 + hidden + batch_rows)
                model = ProbeHead(hidden)

                def loss_fn(model_obj, values, targets):
                    logits = model_obj(values)
                    return nn.losses.binary_cross_entropy(
                        logits, targets, with_logits=True, reduction="mean"
                    )

                loss_and_grad = nn.value_and_grad(model, loss_fn)
                values = mx.random.normal((batch_rows, input_width))
                targets = mx.zeros((batch_rows, 1))
                loss, gradients = loss_and_grad(model, values, targets)
                gradients, _ = optim.clip_grad_norm(gradients, 1.0)
                mx.eval(loss, gradients)
                peak_fn = getattr(mx, "get_peak_memory", None)
                peak_bytes = int(peak_fn()) if peak_fn else None
                feasible = memory_limit is None or peak_bytes is None or peak_bytes <= memory_limit
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
                    "hidden_width": hidden,
                    "peak_bytes": peak_bytes,
                }
            )

    feasible = [row for row in measurements if row["feasible"]]
    if not feasible:
        raise RuntimeError("MLX capacity probe found no feasible Stage-1 configuration")
    return {
        "backend": "mlx",
        "input_width": input_width,
        "maximum_pca_width": maximum_width,
        "memory_limit_bytes": memory_limit,
        "measurements": measurements,
        "physical_memory_bytes": total_memory,
        "safe_batch_rows_by_hidden_width": {
            str(hidden): max(
                row["batch_rows"]
                for row in feasible
                if row["hidden_width"] == hidden
            )
            for hidden in hidden_candidates
            if any(row["hidden_width"] == hidden for row in feasible)
        },
    }


def build_stage1_plan(
    preregistration: Mapping[str, Any],
    calibration: Mapping[str, Any],
    pca_manifest: Mapping[str, Any],
    capacity: Mapping[str, Any],
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

    targets = targets_from_calibration(calibration)
    folds = sorted(pca_manifest["folds"], key=lambda row: int(row["fold"]))
    plan: dict[str, Any] = {
        "artifacts": {
            "calibration_sha256": calibration["calibration_sha256"],
            "pca_manifest_sha256": pca_manifest["manifest_sha256"],
            "preregistration_sha256": preregistration_sha,
        },
        "capacity": dict(capacity),
        "checkpoint_policy": {
            "eligible_from_epoch": 1,
            "maximum_epochs": None,
            "minimum_epochs_before_termination": 50,
            "optimizer_convergence_required": True,
            "plateau_patience_epochs": 50,
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
        "schema": _SCHEMA,
    }
    plan["plan_sha256"] = digest_json(plan)
    return plan


def write_stage1_plan(path: Path, plan: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(plan))


__all__ = [
    "CheckpointSelector",
    "build_stage1_plan",
    "probe_stage1_capacity",
    "write_stage1_plan",
]
