"""Matched PCA-512 versus supervised-bottleneck discovery for VEATIC 2.1."""

from __future__ import annotations

import importlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.preprocessing import StandardScaler

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
from .pca_cache import load_event_pca_projection, load_event_pca_scaler
from .preregistration import benchmark_partition_mask
from .protocol import (
    causal_ar_features,
    event_labels,
    fit_event_threshold,
    future_target_values,
    target_support_mask,
)
from .stage1 import (
    CheckpointSelector,
    _causal_design,
    _fit_fresh_ar,
    _optimizer_converged,
    _owned_rows,
    _target,
)

_SCHEMA = "veatic21_supervised_projection_screen_v1"
_LANES = ("fixed_pca512", "supervised_bottleneck512")


def _require_self_digest(record: Mapping[str, Any], field: str) -> None:
    expected = record.get(field)
    if not isinstance(expected, str):
        raise ValueError(f"artifact is missing {field}")
    payload = dict(record)
    payload.pop(field)
    if digest_json(payload) != expected:
        raise ValueError(f"artifact failed its {field} integrity check")


def causal_context_indices(
    video_id: np.ndarray,
    row_index: np.ndarray,
    *,
    context_rows: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return causal same-video source indices and availability flags."""

    if len(video_id) != len(row_index):
        raise ValueError("causal context identities are not aligned")
    if tuple(sorted(set(context_rows))) != tuple(context_rows) or min(context_rows) <= 0:
        raise ValueError("causal context rows must be unique increasing positive integers")
    lookup = {
        (str(video), int(row)): index
        for index, (video, row) in enumerate(zip(video_id, row_index, strict=True))
    }
    indices = np.empty((len(video_id), len(context_rows)), dtype=np.int32)
    available = np.zeros((len(video_id), len(context_rows)), dtype=np.float32)
    for position, (video, row) in enumerate(zip(video_id, row_index, strict=True)):
        for context_index, lag in enumerate(context_rows):
            source = lookup.get((str(video), int(row) - int(lag)))
            indices[position, context_index] = position if source is None else source
            available[position, context_index] = float(source is not None)
    return indices, available


def probe_supervised_projection_capacity(
    *,
    source_width: int,
    projection_width: int = 512,
    context_count: int = 5,
    hidden_width: int = 64,
    batch_candidates: Sequence[int] = (512, 1024, 2048, 4096),
) -> dict[str, Any]:
    """Measure one-worker MLX capacity without imposing an artificial memory fraction."""

    mx = importlib.import_module("mlx.core")
    nn = importlib.import_module("mlx.nn")
    optim = importlib.import_module("mlx.optimizers")

    class Probe(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(source_width, projection_width, bias=False)
            design_width = projection_width * (1 + context_count) + context_count
            self.first = nn.Linear(design_width, hidden_width)
            self.second = nn.Linear(hidden_width, hidden_width)
            self.output = nn.Linear(hidden_width, 1)

        def __call__(self, current, past, availability, floor):
            projected = self.projection(current)
            projected_past = self.projection(past).reshape(
                current.shape[0], context_count, projection_width
            )
            design = mx.concatenate(
                [
                    projected,
                    (projected[:, None, :] - projected_past).reshape(
                        current.shape[0], context_count * projection_width
                    ),
                    availability,
                ],
                axis=1,
            )
            hidden = nn.gelu(self.first(design))
            hidden = nn.gelu(self.second(hidden))
            return floor[:, None] + 0.5 * mx.tanh(self.output(hidden))

    def loss_fn(model, current, past, availability, floor, target):
        logits = model(current, past, availability, floor)
        return mx.mean(nn.losses.binary_cross_entropy(logits, target[:, None], with_logits=True))

    measurements: list[dict[str, Any]] = []
    for batch_rows in batch_candidates:
        try:
            mx.reset_peak_memory()
            mx.random.seed(20_260_723 + batch_rows)
            model = Probe()
            loss_and_grad = nn.value_and_grad(model, loss_fn)
            current = mx.random.normal((batch_rows, source_width))
            past = mx.random.normal((batch_rows * context_count, source_width))
            availability = mx.ones((batch_rows, context_count))
            floor = mx.zeros((batch_rows,))
            target = mx.zeros((batch_rows,))
            started = time.perf_counter()
            loss, gradients = loss_and_grad(model, current, past, availability, floor, target)
            gradients, _ = optim.clip_grad_norm(gradients, 1.0)
            mx.eval(loss, gradients)
            elapsed = time.perf_counter() - started
            measurements.append(
                {
                    "batch_rows": batch_rows,
                    "elapsed_seconds": elapsed,
                    "feasible": True,
                    "peak_bytes": int(mx.get_peak_memory()),
                    "rows_per_second": batch_rows / elapsed,
                }
            )
        except (MemoryError, RuntimeError, ValueError) as exc:
            measurements.append(
                {
                    "batch_rows": batch_rows,
                    "elapsed_seconds": None,
                    "error": type(exc).__name__,
                    "feasible": False,
                    "peak_bytes": None,
                    "rows_per_second": 0.0,
                }
            )
        finally:
            mx.clear_cache()
    feasible = [row for row in measurements if row["feasible"]]
    if not feasible:
        raise RuntimeError("no supervised projection batch is feasible on this MLX host")
    selected = max(
        feasible, key=lambda row: (float(row["rows_per_second"]), int(row["batch_rows"]))
    )
    return {
        "backend": "mlx",
        "worker_count": 1,
        "memory_fraction_cap": None,
        "source_width": source_width,
        "projection_width": projection_width,
        "context_count": context_count,
        "hidden_width": hidden_width,
        "measurements": measurements,
        "selected_batch_rows": int(selected["batch_rows"]),
        "selection_rule": "maximum_measured_rows_per_second_then_larger_batch",
    }


def build_supervised_projection_screen(
    pca_selection: Mapping[str, Any],
    pca_summary: Mapping[str, Any],
    pca_screen: Mapping[str, Any],
    plan: Mapping[str, Any],
    pca_manifest: Mapping[str, Any],
    capacity: Mapping[str, Any],
) -> dict[str, Any]:
    """Register the matched PCA versus supervised-bottleneck representation screen."""

    _require_self_digest(pca_selection, "selection_sha256")
    _require_self_digest(pca_summary, "summary_sha256")
    _require_self_digest(pca_screen, "screen_sha256")
    _require_self_digest(plan, "plan_sha256")
    _require_self_digest(pca_manifest, "manifest_sha256")
    if int(pca_selection.get("selected_pca_width", -1)) != 512:
        raise ValueError("supervised representation matching requires selected PCA-512")
    if pca_selection.get("summary_sha256") != pca_summary.get("summary_sha256"):
        raise ValueError("PCA selection does not belong to the supplied summary")
    if pca_summary.get("screen_sha256") != pca_screen.get("screen_sha256"):
        raise ValueError("PCA summary does not belong to the supplied screen")
    if pca_screen.get("artifacts", {}).get("stage1_plan_sha256") != plan.get("plan_sha256"):
        raise ValueError("representation screen artifacts do not share one Stage-1 plan")
    if pca_manifest.get("manifest_sha256") != plan.get("artifacts", {}).get("pca_manifest_sha256"):
        raise ValueError("representation screen does not bind the current PCA manifest")
    if int(capacity["projection_width"]) != 512 or int(capacity["context_count"]) != 5:
        raise ValueError("supervised capacity does not match the representation contract")

    nuisance = pca_screen["fixed_nuisance_recipe"]
    targets = [str(value) for value in pca_screen["matrix"]["targets"]]
    folds = [int(value) for value in pca_screen["matrix"]["folds"]]
    seeds = [int(value) for value in pca_screen["matrix"]["comparison_seeds"]]
    expected_cells = len(targets) * len(folds) * len(seeds) * len(_LANES)
    screen: dict[str, Any] = {
        "schema": _SCHEMA,
        "purpose": "matched_supervised_projection_vs_pca512",
        "artifacts": {
            "pca_manifest_sha256": pca_manifest["manifest_sha256"],
            "pca_screen_sha256": pca_screen["screen_sha256"],
            "pca_selection_sha256": pca_selection["selection_sha256"],
            "pca_summary_sha256": pca_summary["summary_sha256"],
            "stage1_plan_sha256": plan["plan_sha256"],
            "supervised_projection_code_sha256": sha256_file(Path(__file__)),
        },
        "architecture": {
            "source": "fold_scaler_standardized_tribe_cortical",
            "source_width": int(capacity["source_width"]),
            "projection": "shared_bias_free_linear_applied_to_current_and_five_causal_past_rows",
            "projection_width": 512,
            "causal_composition": (
                "projected_current_plus_five_projected_current_minus_past_blocks_plus_availability"
            ),
            "head": "same_two_layer_causal_temporal_residual_as_pca_lane",
            "future_features_forbidden": True,
            "video_boundary_crossing_forbidden": True,
        },
        "capacity": dict(capacity),
        "matched_recipe": {
            "batch_rows": int(capacity["selected_batch_rows"]),
            "context_rows": list(nuisance["context_rows"]),
            "hidden_width": int(nuisance["hidden_width"]),
            "learning_rate": float(nuisance["learning_rate"]),
            "weight_decay": float(nuisance["weight_decay"]),
            "residual_logit_cap": float(nuisance["residual_logit_cap"]),
            "minimum_epochs": int(nuisance["minimum_epochs"]),
            "plateau_patience": int(nuisance["plateau_patience"]),
            "nonconvergence_patience": int(nuisance["nonconvergence_patience"]),
            "pca_and_supervised_lanes_use_identical_recipe": True,
        },
        "matrix": {
            "lanes": list(_LANES),
            "targets": targets,
            "folds": folds,
            "comparison_seeds": seeds,
            "expected_cells": expected_cells,
            "worker_count": 1,
            "sealed_tail_labels": True,
        },
        "selection_after_completion": {
            "primary_key": (
                "paired_mean_inner_average_precision_skill_delta_supervised_minus_pca512"
            ),
            "supervised_projection_selected_only_if_positive": True,
            "tie_or_nonpositive_result_keeps_pca512": True,
        },
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    screen["screen_sha256"] = digest_json(screen)
    return screen


def write_supervised_projection_screen(path: Path, screen: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(screen))


def select_supervised_projection(
    summary: Mapping[str, Any],
    screen: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the registered paired representation gate after all cells complete."""

    _require_self_digest(summary, "summary_sha256")
    _require_self_digest(screen, "screen_sha256")
    if summary.get("screen_sha256") != screen.get("screen_sha256"):
        raise ValueError("representation summary does not belong to the registered screen")
    if summary.get("completed_cells") != summary.get("expected_cells"):
        raise ValueError("representation selection requires every registered cell")
    if summary.get("benchmark_test_labels_accessed") is not False:
        raise ValueError("representation selection cannot use benchmark-test labels")

    pairs: dict[tuple[str, int, int], dict[str, float]] = {}
    for record in summary["records"]:
        key = (str(record["target"]), int(record["fold"]), int(record["seed"]))
        pairs.setdefault(key, {})[str(record["lane"])] = float(
            record["inner_average_precision_skill_delta_vs_frozen_ar"]
        )
    expected_pairs = int(summary["expected_cells"]) // len(_LANES)
    if len(pairs) != expected_pairs or any(set(values) != set(_LANES) for values in pairs.values()):
        raise ValueError("representation summary does not contain exact matched pairs")

    paired_deltas = np.asarray(
        [values["supervised_bottleneck512"] - values["fixed_pca512"] for values in pairs.values()]
    )
    lane_summaries = {}
    for lane in _LANES:
        values = np.asarray([pair[lane] for pair in pairs.values()])
        lane_summaries[lane] = {
            "mean_inner_average_precision_skill_delta_vs_frozen_ar": float(np.mean(values)),
            "median_inner_average_precision_skill_delta_vs_frozen_ar": float(np.median(values)),
            "positive_residual_pairs": int(np.sum(values > 0.0)),
            "whole_fold_seed_ar_fallback_pairs": int(np.sum(values <= 0.0)),
        }
    by_target = {}
    for target in screen["matrix"]["targets"]:
        values = np.asarray(
            [delta for key, delta in zip(pairs, paired_deltas, strict=True) if key[0] == target]
        )
        by_target[str(target)] = {
            "pair_count": len(values),
            "mean_supervised_minus_pca512": float(np.mean(values)),
            "pca512_wins": int(np.sum(values < 0.0)),
            "supervised_wins": int(np.sum(values > 0.0)),
            "ties": int(np.sum(values == 0.0)),
        }
    mean_paired_delta = float(np.mean(paired_deltas))
    selected = "supervised_bottleneck512" if mean_paired_delta > 0.0 else "fixed_pca512"
    selection: dict[str, Any] = {
        "schema": "veatic21_supervised_projection_selection_v1",
        "screen_sha256": screen["screen_sha256"],
        "summary_sha256": summary["summary_sha256"],
        "primary_key": ("paired_mean_inner_average_precision_skill_delta_supervised_minus_pca512"),
        "paired_mean_supervised_minus_pca512": mean_paired_delta,
        "paired_median_supervised_minus_pca512": float(np.median(paired_deltas)),
        "pair_count": len(paired_deltas),
        "pca512_wins": int(np.sum(paired_deltas < 0.0)),
        "supervised_wins": int(np.sum(paired_deltas > 0.0)),
        "ties": int(np.sum(paired_deltas == 0.0)),
        "lane_summaries": lane_summaries,
        "by_target": by_target,
        "selected_representation": selected,
        "rejected_representation": (
            "supervised_bottleneck512" if selected == "fixed_pca512" else "fixed_pca512"
        ),
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    selection["selection_sha256"] = digest_json(selection)
    return selection


def write_supervised_projection_selection(path: Path, selection: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(selection))


@dataclass(frozen=True)
class SupervisedProjectionCellConfig:
    lane: Literal["fixed_pca512", "supervised_bottleneck512"]
    target_name: str
    fold: int
    seed: int


def _train_lane(
    raw_values: np.ndarray,
    pca_design: np.ndarray | None,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    context_indices: np.ndarray,
    context_available: np.ndarray,
    targets: np.ndarray,
    ar_logits: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    screen: Mapping[str, Any],
    config: SupervisedProjectionCellConfig,
    checkpoint: Path,
    state_path: Path,
    request_sha256: str,
) -> tuple[np.ndarray, list[dict[str, float | int | bool]], CheckpointSelector]:
    mx = importlib.import_module("mlx.core")
    nn = importlib.import_module("mlx.nn")
    optim = importlib.import_module("mlx.optimizers")
    recipe = screen["matched_recipe"]
    projection_width = int(screen["architecture"]["projection_width"])
    context_count = len(recipe["context_rows"])
    hidden_width = int(recipe["hidden_width"])
    mx.random.seed(config.seed)

    class FixedPCAHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if pca_design is None:
                raise ValueError("fixed PCA lane requires its causal design")
            self.first = nn.Linear(pca_design.shape[1], hidden_width)
            self.second = nn.Linear(hidden_width, hidden_width)
            self.output = nn.Linear(hidden_width, 1)

        def __call__(self, batch, floor):
            hidden = nn.gelu(self.first(batch))
            hidden = nn.gelu(self.second(hidden))
            residual = mx.tanh(self.output(hidden)) * float(recipe["residual_logit_cap"])
            return floor[:, None] + residual

    class SupervisedBottleneckHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(raw_values.shape[1], projection_width, bias=False)
            design_width = projection_width * (1 + context_count) + context_count
            self.first = nn.Linear(design_width, hidden_width)
            self.second = nn.Linear(hidden_width, hidden_width)
            self.output = nn.Linear(hidden_width, 1)

        def __call__(self, current, past, availability, floor, mean, scale):
            current = (current.astype(mx.float32) - mean) / scale
            past = (past.astype(mx.float32) - mean) / scale
            projected = self.projection(current)
            projected_past = self.projection(past).reshape(
                current.shape[0], context_count, projection_width
            )
            design = mx.concatenate(
                [
                    projected,
                    (projected[:, None, :] - projected_past).reshape(
                        current.shape[0], context_count * projection_width
                    ),
                    availability,
                ],
                axis=1,
            )
            hidden = nn.gelu(self.first(design))
            hidden = nn.gelu(self.second(hidden))
            residual = mx.tanh(self.output(hidden)) * float(recipe["residual_logit_cap"])
            return floor[:, None] + residual

    model: Any = FixedPCAHead() if config.lane == "fixed_pca512" else SupervisedBottleneckHead()
    optimizer = optim.AdamW(
        learning_rate=float(recipe["learning_rate"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    prevalence = float(np.mean(targets[train_mask]))
    if not 0.0 < prevalence < 1.0:
        raise ValueError("supervised representation training requires both event classes")
    mean = mx.array(scaler_mean)
    scale = mx.array(scaler_scale)

    def pca_loss(model_obj, batch, floor, truth, weights):
        logits = model_obj(batch, floor)
        loss = nn.losses.binary_cross_entropy(logits, truth[:, None], with_logits=True)
        return mx.mean(loss * weights[:, None])

    def supervised_loss(
        model_obj, current, past, availability, floor, truth, weights, source_mean, source_scale
    ):
        logits = model_obj(current, past, availability, floor, source_mean, source_scale)
        loss = nn.losses.binary_cross_entropy(logits, truth[:, None], with_logits=True)
        return mx.mean(loss * weights[:, None])

    loss_and_grad = nn.value_and_grad(
        model, pca_loss if config.lane == "fixed_pca512" else supervised_loss
    )
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    batch_rows = int(recipe["batch_rows"])
    rng = np.random.default_rng(config.seed)
    selector = CheckpointSelector(
        minimum_epochs=int(recipe["minimum_epochs"]),
        plateau_patience=int(recipe["plateau_patience"]),
    )
    curve: list[dict[str, float | int | bool]] = []
    losses: list[float] = []

    def predict(indices: np.ndarray) -> np.ndarray:
        output = []
        for start in range(0, len(indices), batch_rows):
            rows = indices[start : start + batch_rows]
            if config.lane == "fixed_pca512":
                assert pca_design is not None
                logits = model(mx.array(pca_design[rows]), mx.array(ar_logits[rows]))
            else:
                past_rows = context_indices[rows].reshape(-1)
                logits = model(
                    mx.array(np.asarray(raw_values[rows], dtype=np.float16)),
                    mx.array(np.asarray(raw_values[past_rows], dtype=np.float16)),
                    mx.array(context_available[rows]),
                    mx.array(ar_logits[rows]),
                    mean,
                    scale,
                )
            mx.eval(logits)
            output.append(np.asarray(logits).reshape(-1))
        return np.concatenate(output).astype(np.float32)

    for epoch in range(1, 1_000_000):
        order = rng.permutation(train_indices)
        epoch_losses = []
        for start in range(0, len(order), batch_rows):
            rows = order[start : start + batch_rows]
            truth = targets[rows].astype(np.float32)
            weights = np.where(truth > 0.5, 0.5 / prevalence, 0.5 / (1.0 - prevalence)).astype(
                np.float32
            )
            if config.lane == "fixed_pca512":
                assert pca_design is not None
                loss, gradients = loss_and_grad(
                    model,
                    mx.array(pca_design[rows]),
                    mx.array(ar_logits[rows]),
                    mx.array(truth),
                    mx.array(weights),
                )
            else:
                past_rows = context_indices[rows].reshape(-1)
                loss, gradients = loss_and_grad(
                    model,
                    mx.array(np.asarray(raw_values[rows], dtype=np.float16)),
                    mx.array(np.asarray(raw_values[past_rows], dtype=np.float16)),
                    mx.array(context_available[rows]),
                    mx.array(ar_logits[rows]),
                    mx.array(truth),
                    mx.array(weights),
                    mean,
                    scale,
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
                "schema": "veatic21_supervised_projection_cell_state_v1",
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
            epoch, patience=int(recipe["nonconvergence_patience"])
        ):
            raise RuntimeError("supervised representation candidate failed to converge")

    if selector.best_epoch is None:
        raise RuntimeError("supervised representation training produced no checkpoint")
    model.load_weights(str(checkpoint))
    return predict(np.arange(len(raw_values))), curve, selector


def run_supervised_projection_cell(
    substrate: CanonicalSubstrate,
    preregistration: Mapping[str, Any],
    calibration: Mapping[str, Any],
    pca_manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    screen: Mapping[str, Any],
    pca_root: Path,
    output_dir: Path,
    config: SupervisedProjectionCellConfig,
) -> dict[str, Any]:
    """Train one matched representation cell on benchmark-train folds only."""

    _require_self_digest(screen, "screen_sha256")
    if screen.get("schema") != _SCHEMA or config.lane not in _LANES:
        raise ValueError("invalid supervised representation cell")
    if config.target_name not in screen["matrix"]["targets"]:
        raise ValueError("supervised representation target is not registered")
    if (
        config.fold not in screen["matrix"]["folds"]
        or config.seed not in screen["matrix"]["comparison_seeds"]
    ):
        raise ValueError("supervised representation fold or seed is not registered")
    request = {
        "schema": "veatic21_supervised_projection_cell_request_v1",
        "screen_sha256": screen["screen_sha256"],
        "config": dict(config.__dict__),
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    request_sha256 = digest_json(request)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file():
        saved = json.loads((output_dir / "request.json").read_text(encoding="utf-8"))
        state = json.loads((output_dir / "state.json").read_text(encoding="utf-8"))
        if digest_json(saved) != request_sha256 or state.get("status") != "complete":
            raise RuntimeError("refusing changed or incomplete representation cell reuse")
        if state.get("metrics_sha256") != sha256_file(metrics_path):
            raise RuntimeError("refusing changed representation metrics")
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("refusing to overwrite a partial representation cell")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "request.json", request)
    atomic_write_json(
        output_dir / "state.json",
        {
            "schema": "veatic21_supervised_projection_cell_state_v1",
            "status": "training",
            "request_sha256": request_sha256,
        },
    )

    all_features = substrate.load_features(substrate.video_ids, ("tribe_cortical",))
    development_mask = benchmark_partition_mask(all_features, preregistration["split"], "train")
    features = all_features.subset(development_mask)
    raw_values = features.representations["tribe_cortical"]
    if raw_values.shape[1] != int(screen["architecture"]["source_width"]):
        raise ValueError("supervised representation source width changed")
    labels = substrate.load_labels(
        substrate.video_ids,
        row_indices=_owned_rows(all_features.video_id, all_features.row_index, development_mask),
        stage="supervised_projection_benchmark_train_labels_only",
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
        ar_matrix, binary, future, target, train_mask, features.video_id, seed=config.seed
    )
    context_rows = tuple(int(value) for value in screen["matched_recipe"]["context_rows"])
    context_indices, context_available = causal_context_indices(
        features.video_id, features.row_index, context_rows=context_rows
    )
    scaler_mean, scaler_scale, scaler_basis_sha256 = load_event_pca_scaler(
        preregistration, pca_manifest, pca_root, fold=config.fold
    )
    pca_design = None
    design_scaler = None
    if config.lane == "fixed_pca512":
        projected = load_event_pca_projection(
            features, preregistration, pca_manifest, pca_root, fold=config.fold, width=512
        )
        pca_design = _causal_design(
            projected,
            features.video_id,
            features.row_index,
            family="frozen_ar_plus_causal_temporal_residual",
            context_rows=context_rows,
        )
        design_scaler = StandardScaler().fit(pca_design[train_mask])
        pca_design = design_scaler.transform(pca_design).astype(np.float32)
    preprocessing = {
        **ar_artifact,
        "source_scaler_mean": scaler_mean.astype(np.float32),
        "source_scaler_scale": scaler_scale.astype(np.float32),
    }
    if design_scaler is not None:
        preprocessing.update(
            {
                "design_scaler_mean": np.asarray(design_scaler.mean_, dtype=np.float64),
                "design_scaler_scale": np.asarray(design_scaler.scale_, dtype=np.float64),
            }
        )
    atomic_save_npz(output_dir / "preprocessing.npz", preprocessing)
    checkpoint = output_dir / "best-checkpoint.npz"
    try:
        scores, curve, selector = _train_lane(
            raw_values,
            pca_design,
            scaler_mean,
            scaler_scale,
            context_indices,
            context_available,
            binary,
            ar_logits,
            train_mask,
            validation_mask,
            screen,
            config,
            checkpoint,
            output_dir / "state.json",
            request_sha256,
        )
    except Exception as exc:
        atomic_write_json(
            output_dir / "state.json",
            {
                "schema": "veatic21_supervised_projection_cell_state_v1",
                "status": "failed",
                "request_sha256": request_sha256,
                "failure": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    validation = np.flatnonzero(validation_mask)
    delta = average_precision_skill(
        binary[validation], scores[validation]
    ) - average_precision_skill(binary[validation], ar_logits[validation])
    use_residual = delta > 0.0
    selected = scores if use_residual else ar_logits
    atomic_save_npz(
        output_dir / "validation-predictions.npz",
        {
            "video_id": features.video_id[validation].astype("U3"),
            "row_index": features.row_index[validation].astype(np.int32),
            "target": binary[validation].astype(np.int8),
            "ar_score": ar_logits[validation],
            "model_score": scores[validation],
            "selected_score": selected[validation],
        },
    )
    atomic_write_json(output_dir / "training-curve.json", {"records": curve})
    metrics: dict[str, Any] = {
        "schema": "veatic21_supervised_projection_cell_metrics_v1",
        "lane": config.lane,
        "target": config.target_name,
        "fold": config.fold,
        "seed": config.seed,
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "fresh_ar_c": ar_c,
        "fresh_ar_pr_auc": pooled_pr_auc(binary[validation], ar_logits[validation]),
        "learned_head_pr_auc": pooled_pr_auc(binary[validation], scores[validation]),
        "inner_average_precision_skill_delta_vs_frozen_ar": delta,
        "whole_fold_seed_uses_residual": use_residual,
        "selected_pr_auc": pooled_pr_auc(binary[validation], selected[validation]),
        "best_epoch": selector.best_epoch,
        "epochs_completed": len(curve),
        "source_scaler_basis_sha256": scaler_basis_sha256,
        "train_row_sha256": row_identity_digest(
            features.video_id[train_mask], features.row_index[train_mask]
        ),
        "validation_row_sha256": row_identity_digest(
            features.video_id[validation_mask], features.row_index[validation_mask]
        ),
        "checkpoint_sha256": sha256_file(checkpoint),
        "request_sha256": request_sha256,
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    atomic_write_json(metrics_path, metrics)
    atomic_write_json(
        output_dir / "state.json",
        {
            "schema": "veatic21_supervised_projection_cell_state_v1",
            "status": "complete",
            "request_sha256": request_sha256,
            "metrics_sha256": sha256_file(metrics_path),
        },
    )
    return metrics


def run_supervised_projection_screen(
    substrate: CanonicalSubstrate,
    preregistration: Mapping[str, Any],
    calibration: Mapping[str, Any],
    pca_manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    screen: Mapping[str, Any],
    pca_root: Path,
    output_dir: Path,
    *,
    progress=None,
) -> dict[str, Any]:
    """Run the complete matched representation screen with one sequential MLX worker."""

    _require_self_digest(screen, "screen_sha256")
    request = {
        "schema": "veatic21_supervised_projection_run_request_v1",
        "screen_sha256": screen["screen_sha256"],
        "worker_count": 1,
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    request_sha256 = digest_json(request)
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "request.json"
    if request_path.is_file():
        if digest_json(json.loads(request_path.read_text(encoding="utf-8"))) != request_sha256:
            raise RuntimeError("refusing supervised projection resume because request changed")
    else:
        atomic_write_json(request_path, request)
    records = []
    for target in screen["matrix"]["targets"]:
        for fold in screen["matrix"]["folds"]:
            for seed in screen["matrix"]["comparison_seeds"]:
                for lane in screen["matrix"]["lanes"]:
                    config = SupervisedProjectionCellConfig(
                        lane=lane, target_name=str(target), fold=int(fold), seed=int(seed)
                    )
                    cell_dir = (
                        output_dir
                        / "targets"
                        / str(target)
                        / f"fold-{int(fold)}"
                        / f"seed-{int(seed)}"
                        / str(lane)
                    )
                    metrics = run_supervised_projection_cell(
                        substrate,
                        preregistration,
                        calibration,
                        pca_manifest,
                        plan,
                        screen,
                        pca_root,
                        cell_dir,
                        config,
                    )
                    record = {
                        "lane": str(lane),
                        "target": str(target),
                        "fold": int(fold),
                        "seed": int(seed),
                        "inner_average_precision_skill_delta_vs_frozen_ar": metrics[
                            "inner_average_precision_skill_delta_vs_frozen_ar"
                        ],
                        "whole_fold_seed_uses_residual": metrics["whole_fold_seed_uses_residual"],
                        "best_epoch": metrics["best_epoch"],
                        "cell_metrics_sha256": sha256_file(cell_dir / "metrics.json"),
                        "cell_directory": str(cell_dir.relative_to(output_dir)),
                    }
                    records.append(record)
                    progress_record = {
                        "schema": "veatic21_supervised_projection_progress_v1",
                        "request_sha256": request_sha256,
                        "completed_cells": len(records),
                        "expected_cells": int(screen["matrix"]["expected_cells"]),
                        "last_cell": record,
                        "benchmark_test_labels_accessed": False,
                    }
                    atomic_write_json(output_dir / "progress.json", progress_record)
                    if progress is not None:
                        progress(progress_record)
    summary: dict[str, Any] = {
        "schema": "veatic21_supervised_projection_summary_v1",
        "request_sha256": request_sha256,
        "screen_sha256": screen["screen_sha256"],
        "completed_cells": len(records),
        "expected_cells": int(screen["matrix"]["expected_cells"]),
        "records": records,
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    summary["summary_sha256"] = digest_json(summary)
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


__all__ = [
    "SupervisedProjectionCellConfig",
    "build_supervised_projection_screen",
    "causal_context_indices",
    "probe_supervised_projection_capacity",
    "run_supervised_projection_cell",
    "run_supervised_projection_screen",
    "select_supervised_projection",
    "write_supervised_projection_selection",
    "write_supervised_projection_screen",
]
