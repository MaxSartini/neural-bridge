"""Resumable inner-only Stage A screen for the frozen VEATIC Phase 02 AR search."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np

from neural_bridge.veatic21.contracts import (
    PHASE01_ROOT,
    PHASE02_REGISTRATION_ROOT,
    PHASE02_REGISTRATION_SHA256,
    PHASE02_STAGE_A_ROOT,
    REPOSITORY_ROOT,
)
from neural_bridge.veatic21.data import load_json, reject_forbidden_runtime_path, sha256_file
from neural_bridge.veatic21.phase00 import _write_json, digest_json
from neural_bridge.veatic21.phase01 import _array_digest
from neural_bridge.veatic21.phase02_features import (
    CausalHistory,
    build_causal_history,
    build_feature_matrix,
    feature_names,
    standardize_from_owner,
)
from neural_bridge.veatic21.phase02_metrics import binary_ranking_metrics, probability_metrics
from neural_bridge.veatic21.phase02_registration import (
    _blocked_row_masks,
    verify_phase02_registration,
)

mx = importlib.import_module("mlx.core")

REGULARIZATION_MULTIPLIERS = np.asarray(
    [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0],
    dtype=np.float32,
)
STAGE_A_FAMILIES = ("continuous_ridge", "event_logistic_l2")
STAGE_A_SOURCE_FILES = (
    "contracts.py",
    "data.py",
    "phase00.py",
    "phase01.py",
    "phase02_features.py",
    "phase02_metrics.py",
    "phase02_registration.py",
    "phase02_stage_a.py",
)


@dataclass(frozen=True)
class StageAWorkUnit:
    unit_id: str
    sequence: int
    protocol: str
    split_index: int
    repeat: int | None
    outer_fold: int
    inner_fold: int
    feature_form: str
    history_depth: int
    model_family: str


@dataclass(frozen=True)
class StageAInputs:
    arousal: np.ndarray
    video_id: np.ndarray
    row_index: np.ndarray
    time_seconds: np.ndarray
    active_values: np.ndarray
    active_masks: np.ndarray
    candidate_ids: tuple[str, ...]
    target_ends: tuple[int, ...]
    history: CausalHistory
    grouped_splits: tuple[dict[str, object], ...]
    blocked_splits: tuple[dict[str, object], ...]


def _stage_a_code_identity() -> str:
    digest = hashlib.sha256()
    package = REPOSITORY_ROOT / "src/neural_bridge/veatic21"
    for filename in STAGE_A_SOURCE_FILES:
        path = package / filename
        digest.update(filename.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _next_power_of_two(value: float) -> int:
    if value <= 1:
        return 1
    return 1 << math.ceil(math.log2(value))


def enumerate_stage_a_work_units(
    registration: dict[str, object], split_registry: dict[str, object]
) -> tuple[StageAWorkUnit, ...]:
    search = cast(dict[str, object], registration["search"])
    forms = cast(list[str], search["feature_forms"])
    depths = cast(list[int], search["history_depth_rows"])
    grouped = cast(list[dict[str, object]], split_registry["grouped"])
    blocked = cast(list[dict[str, object]], split_registry["blocked"])
    units: list[StageAWorkUnit] = []

    def add(
        *, protocol: str, split_index: int, repeat: int | None, outer_fold: int,
        inner_fold: int, form: str, depth: int, family: str
    ) -> None:
        sequence = len(units)
        repeat_code = "na" if repeat is None else f"{repeat:02d}"
        unit_id = (
            f"{sequence:05d}_{protocol}_r{repeat_code}_o{outer_fold:02d}_i{inner_fold:02d}_"
            f"{form}_d{depth:02d}_{family}"
        )
        units.append(
            StageAWorkUnit(
                unit_id=unit_id,
                sequence=sequence,
                protocol=protocol,
                split_index=split_index,
                repeat=repeat,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                feature_form=form,
                history_depth=depth,
                model_family=family,
            )
        )

    for split_index, split in enumerate(blocked):
        for form in forms:
            for depth in depths:
                for family in STAGE_A_FAMILIES:
                    add(
                        protocol="blocked",
                        split_index=split_index,
                        repeat=None,
                        outer_fold=cast(int, split["outer_fold"]),
                        inner_fold=0,
                        form=form,
                        depth=depth,
                        family=family,
                    )
    for split_index, split in enumerate(grouped):
        inner = cast(list[list[int]], split["inner_validation_video_folds"])
        for inner_fold in range(len(inner)):
            for form in forms:
                for depth in depths:
                    for family in STAGE_A_FAMILIES:
                        add(
                            protocol="grouped",
                            split_index=split_index,
                            repeat=cast(int, split["repeat"]),
                            outer_fold=cast(int, split["outer_fold"]),
                            inner_fold=inner_fold,
                            form=form,
                            depth=depth,
                            family=family,
                        )
    return tuple(units)


def _load_inputs() -> StageAInputs:
    with np.load(PHASE01_ROOT / "aligned-labels.npz", allow_pickle=False) as payload:
        arousal = payload["arousal"].astype(np.float32)
        video_id = payload["video_id"].astype(np.int16)
        row_index = payload["row_index"].astype(np.int32)
        time_seconds = payload["time_seconds"].astype(np.float64)
    with np.load(PHASE01_ROOT / "target-substrate.npz", allow_pickle=False) as payload:
        starts = payload["candidate_start_rows"].astype(int)
        indices = np.flatnonzero(starts == 1)
        active_values = payload["continuous_future_maximum_increase"][indices].astype(
            np.float32
        )
        active_masks = payload["valid_mask"][indices].astype(bool)
        target_ends = tuple(int(value) for value in payload["candidate_end_rows"][indices])
    registration = load_json(PHASE02_REGISTRATION_ROOT / "experiment-registration.json")
    split_registry = load_json(PHASE02_REGISTRATION_ROOT / "split-registry.json")
    candidate_ids = tuple(cast(list[str], registration["targets"]))
    if candidate_ids != tuple(f"s01_e{end:02d}" for end in range(1, 22)):
        raise ValueError("Stage A target registry mismatch")
    history = build_causal_history(
        arousal, video_id, row_index, max_depth=max(target_ends)
    )
    return StageAInputs(
        arousal=arousal,
        video_id=video_id,
        row_index=row_index,
        time_seconds=time_seconds,
        active_values=active_values,
        active_masks=active_masks,
        candidate_ids=candidate_ids,
        target_ends=target_ends,
        history=history,
        grouped_splits=tuple(cast(list[dict[str, object]], split_registry["grouped"])),
        blocked_splits=tuple(cast(list[dict[str, object]], split_registry["blocked"])),
    )


def _split_masks(
    inputs: StageAInputs, unit: StageAWorkUnit
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    rows, targets = inputs.active_masks.T.shape
    train_masks = np.zeros((rows, targets), dtype=bool)
    validation_masks = np.zeros((rows, targets), dtype=bool)
    if unit.protocol == "grouped":
        split = inputs.grouped_splits[unit.split_index]
        outer_train = set(cast(list[int], split["train_videos"]))
        inner_folds = cast(list[list[int]], split["inner_validation_video_folds"])
        validation_videos = set(inner_folds[unit.inner_fold])
        training_videos = outer_train - validation_videos
        train_owner = np.isin(inputs.video_id, list(training_videos))
        validation_owner = np.isin(inputs.video_id, list(validation_videos))
        train_masks = train_owner[:, None] & inputs.active_masks.T
        validation_masks = validation_owner[:, None] & inputs.active_masks.T
        split_digest = digest_json(
            {
                "protocol": unit.protocol,
                "repeat": unit.repeat,
                "outer_fold": unit.outer_fold,
                "inner_fold": unit.inner_fold,
                "training_videos": sorted(training_videos),
                "validation_videos": sorted(validation_videos),
            }
        )
        return train_owner, train_masks, validation_masks, split_digest

    split = inputs.blocked_splits[unit.split_index]
    test_block = cast(int, split["test_block_index"])
    block_count = cast(int, split["block_count"])
    row_counts = {
        int(video): int(np.sum(inputs.video_id == video)) for video in np.unique(inputs.video_id)
    }
    for target, target_end in enumerate(inputs.target_ends):
        masks = _blocked_row_masks(
            inputs.video_id,
            inputs.row_index,
            row_counts,
            target_end,
            test_block,
            block_count,
        )
        train_masks[:, target] = masks["inner_train"] & inputs.active_masks[target]
        validation_masks[:, target] = (
            masks["inner_validation"] & inputs.active_masks[target]
        )
    train_owner = np.zeros(len(inputs.video_id), dtype=bool)
    for video, count in row_counts.items():
        owned = inputs.video_id == video
        inner_boundary = math.floor(count * (test_block - 1) / block_count)
        train_owner[owned] = inputs.row_index[owned] < inner_boundary
    split_digest = digest_json(
        {
            "protocol": unit.protocol,
            "outer_fold": unit.outer_fold,
            "test_block": test_block,
            "block_count": block_count,
            "target_boundary_purge": True,
        }
    )
    return train_owner, train_masks, validation_masks, split_digest


def _thresholds_and_labels(
    values: np.ndarray, train_masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    targets = values.shape[0]
    thresholds = np.empty(targets, dtype=np.float32)
    labels = np.zeros((values.shape[1], targets), dtype=np.float32)
    for target in range(targets):
        owned = train_masks[:, target]
        thresholds[target] = np.quantile(values[target, owned], 0.90)
        labels[:, target] = values[target] >= thresholds[target]
    return thresholds, labels


def _regularization_scales(x: np.ndarray, train_masks: np.ndarray) -> np.ndarray:
    feature_count = max(x.shape[1] - 1, 1)
    scales = np.empty(train_masks.shape[1], dtype=np.float32)
    for target in range(train_masks.shape[1]):
        owned = train_masks[:, target]
        scales[target] = np.sum(x[owned, :-1].astype(np.float64) ** 2) / (
            feature_count * int(owned.sum())
        )
    return np.maximum(scales, np.finfo(np.float32).eps)


def _masked_operator(
    x: mx.array,
    masks: mx.array,
    counts: mx.array,
    penalties: mx.array,
    regularization: mx.array,
    weights: mx.array,
) -> mx.array:
    predictions = mx.einsum("np,apt->nat", x, weights)
    weighted = predictions * masks[:, None, :]
    gram = mx.einsum("np,nat->apt", x, weighted) / counts[None, None, :]
    return gram + weights * penalties[None, :, None] * regularization[:, None, :]


def _ridge_screen(
    x: np.ndarray,
    values: np.ndarray,
    train_masks: np.ndarray,
    validation_masks: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    y = values.T.astype(np.float32)
    y[~train_masks] = 0.0
    counts_np = train_masks.sum(axis=0).astype(np.float32)
    scales = _regularization_scales(x, train_masks)
    regularization_np = REGULARIZATION_MULTIPLIERS[:, None] * scales[None, :]
    penalties_np = np.ones(x.shape[1], dtype=np.float32)
    penalties_np[-1] = 0.0

    x_mx = mx.array(x)
    masks_mx = mx.array(train_masks.astype(np.float32))
    counts_mx = mx.array(counts_np)
    penalties_mx = mx.array(penalties_np)
    regularization_mx = mx.array(regularization_np)
    target_mx = mx.array(y)
    right = mx.einsum("np,nt->pt", x_mx, target_mx) / counts_mx[None, :]
    right = mx.broadcast_to(
        right[None, :, :], (len(REGULARIZATION_MULTIPLIERS), x.shape[1], y.shape[1])
    )
    weights = mx.zeros_like(right)
    residual = right - _masked_operator(
        x_mx, masks_mx, counts_mx, penalties_mx, regularization_mx, weights
    )
    direction = residual
    residual_square = mx.sum(residual * residual, axis=1)
    right_norm = mx.sqrt(mx.sum(right * right, axis=1)) + np.finfo(np.float32).eps
    budget = _next_power_of_two(math.sqrt(float(np.min(counts_np))))
    tolerance = 1 / np.sqrt(counts_np)
    iterations = 0
    relative_np = np.full(residual_square.shape, np.inf, dtype=np.float32)
    for iteration in range(budget):
        applied = _masked_operator(
            x_mx, masks_mx, counts_mx, penalties_mx, regularization_mx, direction
        )
        denominator = mx.sum(direction * applied, axis=1)
        step = residual_square / mx.maximum(denominator, np.finfo(np.float32).eps)
        weights = weights + step[:, None, :] * direction
        residual = residual - step[:, None, :] * applied
        next_square = mx.sum(residual * residual, axis=1)
        beta = next_square / mx.maximum(residual_square, np.finfo(np.float32).eps)
        direction = residual + beta[:, None, :] * direction
        residual_square = next_square
        iterations = iteration + 1
        if iterations % 8 == 0 or iterations == budget:
            relative_np = np.asarray(mx.sqrt(residual_square) / right_norm)
            if np.all(relative_np <= tolerance[None, :]):
                break
    weight_np = np.asarray(weights)
    predictions = np.einsum("np,apt->nat", x, weight_np, optimize=True)
    predictions[:, :, :][~validation_masks[:, None, :].repeat(len(weight_np), axis=1)] = np.nan
    return predictions, {
        "backend": "mlx_gpu_primal_conjugate_gradient",
        "iterations": iterations,
        "budget": budget,
        "relative_residual_max": float(np.max(relative_np)),
        "target_tolerance_max": float(np.max(tolerance)),
        "converged_cells": int(np.sum(relative_np <= tolerance[None, :])),
        "total_cells": int(relative_np.size),
        "regularization_scales": scales.tolist(),
    }


def _logistic_loss_summary(
    x: mx.array, labels: mx.array, masks: mx.array, counts: mx.array, weights: mx.array
) -> float:
    logits = mx.einsum("np,apt->nat", x, weights)
    loss = mx.maximum(logits, 0) - logits * labels[:, None, :] + mx.log1p(
        mx.exp(-mx.abs(logits))
    )
    per_cell = mx.sum(loss * masks[:, None, :], axis=0) / counts[None, :]
    return float(np.asarray(mx.mean(per_cell)))


def _logistic_screen(
    x: np.ndarray,
    labels: np.ndarray,
    train_masks: np.ndarray,
    validation_masks: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    counts_np = train_masks.sum(axis=0).astype(np.float32)
    scales = _regularization_scales(x, train_masks)
    regularization_np = REGULARIZATION_MULTIPLIERS[:, None] * scales[None, :]
    penalties_np = np.ones(x.shape[1], dtype=np.float32)
    penalties_np[-1] = 0.0
    feature_trace = np.empty(train_masks.shape[1], dtype=np.float32)
    for target in range(train_masks.shape[1]):
        owned = train_masks[:, target]
        feature_trace[target] = np.sum(x[owned].astype(np.float64) ** 2) / int(owned.sum())
    lipschitz = 0.25 * feature_trace[None, :] + regularization_np
    step_np = 1 / np.maximum(lipschitz, np.finfo(np.float32).eps)

    x_mx = mx.array(x)
    labels_mx = mx.array(labels.astype(np.float32))
    masks_mx = mx.array(train_masks.astype(np.float32))
    counts_mx = mx.array(counts_np)
    regularization_mx = mx.array(regularization_np)
    penalties_mx = mx.array(penalties_np)
    step_mx = mx.array(step_np)
    shape = (len(REGULARIZATION_MULTIPLIERS), x.shape[1], labels.shape[1])
    weights = mx.zeros(shape, dtype=mx.float32)
    accelerated = weights
    momentum_state = 1.0
    budget = _next_power_of_two(math.sqrt(float(np.min(counts_np))))
    tolerance = 1 / np.sqrt(counts_np)
    learning_curve: list[dict[str, float | int]] = []
    gradient_relative_np = np.full(
        (len(REGULARIZATION_MULTIPLIERS), labels.shape[1]), np.inf, dtype=np.float32
    )
    for iteration in range(budget):
        logits = mx.einsum("np,apt->nat", x_mx, accelerated)
        error = (mx.sigmoid(logits) - labels_mx[:, None, :]) * masks_mx[:, None, :]
        gradient = mx.einsum("np,nat->apt", x_mx, error) / counts_mx[None, None, :]
        gradient = gradient + (
            accelerated
            * penalties_mx[None, :, None]
            * regularization_mx[:, None, :]
        )
        next_weights = accelerated - step_mx[:, None, :] * gradient
        next_momentum = (1 + math.sqrt(1 + 4 * momentum_state**2)) / 2
        accelerated = next_weights + ((momentum_state - 1) / next_momentum) * (
            next_weights - weights
        )
        weights = next_weights
        momentum_state = next_momentum
        completed = iteration + 1
        if completed % 8 == 0 or completed == budget:
            gradient_norm = mx.sqrt(mx.sum(gradient * gradient, axis=1))
            weight_norm = mx.sqrt(mx.sum(weights * weights, axis=1)) + 1.0
            gradient_relative_np = np.asarray(gradient_norm / weight_norm)
            learning_curve.append(
                {
                    "update": completed,
                    "mean_log_loss": _logistic_loss_summary(
                        x_mx, labels_mx, masks_mx, counts_mx, weights
                    ),
                    "max_relative_gradient": float(np.max(gradient_relative_np)),
                }
            )
            if np.all(gradient_relative_np <= tolerance[None, :]):
                break
    logits = np.einsum("np,apt->nat", x, np.asarray(weights), optimize=True)
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    probabilities[~validation_masks[:, None, :].repeat(len(probabilities[0]), axis=1)] = np.nan
    return probabilities, {
        "backend": "mlx_gpu_full_batch_accelerated_gradient",
        "iterations": learning_curve[-1]["update"],
        "budget": budget,
        "max_relative_gradient": float(np.max(gradient_relative_np)),
        "target_tolerance_max": float(np.max(tolerance)),
        "converged_cells": int(np.sum(gradient_relative_np <= tolerance[None, :])),
        "total_cells": int(gradient_relative_np.size),
        "regularization_scales": scales.tolist(),
        "learning_curve": learning_curve,
    }


def _records_from_predictions(
    unit: StageAWorkUnit,
    candidate_ids: tuple[str, ...],
    thresholds: np.ndarray,
    labels: np.ndarray,
    train_masks: np.ndarray,
    validation_masks: np.ndarray,
    predictions: np.ndarray,
    solver: dict[str, object],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    scales = cast(list[float], solver["regularization_scales"])
    for alpha_index, multiplier in enumerate(REGULARIZATION_MULTIPLIERS):
        for target, candidate_id in enumerate(candidate_ids):
            owned = validation_masks[:, target]
            score = predictions[owned, alpha_index, target]
            target_labels = labels[owned, target]
            metrics = binary_ranking_metrics(target_labels, score)
            probability = (
                probability_metrics(target_labels, score)
                if unit.model_family == "event_logistic_l2"
                else {"brier": None}
            )
            configuration_id = (
                f"{unit.unit_id}__{candidate_id}__reg{alpha_index:02d}"
            )
            records.append(
                {
                    "configuration_id": configuration_id,
                    "status": "completed",
                    "disposition": "eligible_for_inner_aggregation",
                    "model_family": unit.model_family,
                    "candidate_id": candidate_id,
                    "feature_form": unit.feature_form,
                    "history_depth_rows": unit.history_depth,
                    "history_depth_seconds": unit.history_depth / 2,
                    "regularization_multiplier": float(multiplier),
                    "regularization_scale": scales[target],
                    "regularization_value": float(multiplier * scales[target]),
                    "train_threshold_q90": float(thresholds[target]),
                    "train_rows": int(train_masks[:, target].sum()),
                    "validation_rows": int(owned.sum()),
                    **metrics,
                    **probability,
                }
            )
    return records


def execute_stage_a_unit(inputs: StageAInputs, unit: StageAWorkUnit) -> dict[str, object]:
    started = time.monotonic()
    train_owner, train_masks, validation_masks, split_digest = _split_masks(inputs, unit)
    raw_features = build_feature_matrix(
        inputs.history, unit.feature_form, unit.history_depth
    )
    standardized, mean, std = standardize_from_owner(raw_features, train_owner)
    x = np.column_stack([standardized, np.ones(len(standardized), dtype=np.float32)])
    thresholds, labels = _thresholds_and_labels(inputs.active_values, train_masks)
    if unit.model_family == "continuous_ridge":
        predictions, solver = _ridge_screen(
            x, inputs.active_values, train_masks, validation_masks
        )
    elif unit.model_family == "event_logistic_l2":
        predictions, solver = _logistic_screen(x, labels, train_masks, validation_masks)
    else:
        raise ValueError(f"unsupported Stage A family: {unit.model_family}")
    records = _records_from_predictions(
        unit,
        inputs.candidate_ids,
        thresholds,
        labels,
        train_masks,
        validation_masks,
        predictions,
        solver,
    )
    return {
        "schema_version": "veatic21_phase02_stage_a_unit_v2",
        "unit": asdict(unit),
        "registration_sha256": PHASE02_REGISTRATION_SHA256,
        "stage_a_code_sha256": _stage_a_code_identity(),
        "split_sha256": split_digest,
        "feature_names": list(feature_names(unit.feature_form, unit.history_depth)),
        "feature_count": raw_features.shape[1],
        "feature_matrix_sha256": _array_digest({"features": raw_features}),
        "scaler_sha256": _array_digest({"mean": mean, "std": std}),
        "target_thresholds_sha256": _array_digest({"thresholds": thresholds}),
        "train_row_counts": train_masks.sum(axis=0).astype(int).tolist(),
        "validation_row_counts": validation_masks.sum(axis=0).astype(int).tolist(),
        "solver": solver,
        "configuration_count": len(records),
        "records": records,
        "runtime_seconds": time.monotonic() - started,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
    }


def _append_ledger(path: Path, unit_result_path: Path, result: dict[str, object]) -> None:
    records = cast(list[dict[str, object]], result["records"])
    entry = {
        "schema_version": "veatic21_phase02_append_only_ledger_entry_v2",
        "unit_id": cast(dict[str, object], result["unit"])["unit_id"],
        "unit_result_path": str(unit_result_path),
        "unit_result_sha256": sha256_file(unit_result_path),
        "configuration_count": len(records),
        "configuration_ids": [record["configuration_id"] for record in records],
        "statuses": sorted({str(record["status"]) for record in records}),
        "outer_test_scores_opened": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _ledger_unit_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"corrupt Stage A ledger line {line_number}") from error
            unit_id = entry.get("unit_id")
            if not isinstance(unit_id, str) or unit_id in ids:
                raise ValueError("invalid or duplicate Stage A ledger unit")
            ids.add(unit_id)
    return ids


def run_phase02_stage_a(
    *, max_units: int | None = None, output_root: Path = PHASE02_STAGE_A_ROOT
) -> dict[str, object]:
    """Run or resume registered inner-only Stage A work units on one MLX GPU worker."""

    output_root = reject_forbidden_runtime_path(output_root)
    if output_root != PHASE02_STAGE_A_ROOT:
        raise ValueError(f"Phase 02 Stage A must use the canonical root: {output_root}")
    if max_units is not None and max_units < 1:
        raise ValueError("max_units must be positive")
    verified = verify_phase02_registration(PHASE02_REGISTRATION_ROOT)
    if verified["registration_sha256"] != PHASE02_REGISTRATION_SHA256:
        raise ValueError("Phase 02 registration identity changed")
    registration = load_json(PHASE02_REGISTRATION_ROOT / "experiment-registration.json")
    split_registry = load_json(PHASE02_REGISTRATION_ROOT / "split-registry.json")
    search = cast(dict[str, object], registration["search"])
    regularization = cast(dict[str, object], search["regularization"])
    registered_multipliers = np.asarray(
        cast(list[float], regularization["linear_multipliers"]), dtype=np.float32
    )
    if not np.array_equal(registered_multipliers, REGULARIZATION_MULTIPLIERS):
        raise ValueError("Stage A regularization grid differs from the frozen registration")
    units = enumerate_stage_a_work_units(registration, split_registry)
    code_sha256 = _stage_a_code_identity()
    request = {
        "schema_version": "veatic21_phase02_stage_a_request_v2",
        "stage": "A_complete_linear_screen",
        "registration_sha256": PHASE02_REGISTRATION_SHA256,
        "stage_a_code_sha256": code_sha256,
        "work_unit_count": len(units),
        "configuration_count": len(units) * len(REGULARIZATION_MULTIPLIERS) * 21,
        "families": list(STAGE_A_FAMILIES),
        "backend": "single MLX GPU worker",
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
    }
    request_path = output_root / "request.json"
    registry_path = output_root / "work-unit-registry.json"
    if request_path.exists():
        if load_json(request_path) != request:
            raise ValueError("Stage A request/code/registration identity changed")
    else:
        _write_json(request_path, request)
        _write_json(
            registry_path,
            {
                "schema_version": "veatic21_phase02_stage_a_work_registry_v2",
                "units": [asdict(unit) for unit in units],
            },
        )
    unit_root = output_root / "units"
    unit_root.mkdir(parents=True, exist_ok=True)
    ledger_path = output_root / "append-only-experiment-ledger.jsonl"
    ledger_ids = _ledger_unit_ids(ledger_path)
    inputs: StageAInputs | None = None
    completed_this_call = 0
    completed_total = 0
    for unit in units:
        unit_path = unit_root / f"{unit.unit_id}.json"
        if unit_path.exists():
            stored = load_json(unit_path)
            if (
                cast(dict[str, object], stored.get("unit", {})).get("unit_id") != unit.unit_id
                or stored.get("stage_a_code_sha256") != code_sha256
                or stored.get("registration_sha256") != PHASE02_REGISTRATION_SHA256
            ):
                raise ValueError(f"Stage A unit identity mismatch: {unit_path}")
            if unit.unit_id not in ledger_ids:
                _append_ledger(ledger_path, unit_path, stored)
                ledger_ids.add(unit.unit_id)
            completed_total += 1
            continue
        if max_units is not None and completed_this_call >= max_units:
            break
        if inputs is None:
            inputs = _load_inputs()
        unit_result = execute_stage_a_unit(inputs, unit)
        _write_json(unit_path, unit_result)
        _append_ledger(ledger_path, unit_path, unit_result)
        ledger_ids.add(unit.unit_id)
        completed_this_call += 1
        completed_total += 1
        _write_json(
            output_root / "run-state.json",
            {
                "schema_version": "veatic21_phase02_stage_a_state_v2",
                "status": "RUNNING" if completed_total < len(units) else "COMPLETE",
                "registration_sha256": PHASE02_REGISTRATION_SHA256,
                "stage_a_code_sha256": code_sha256,
                "work_units_total": len(units),
                "work_units_completed": completed_total,
                "work_units_remaining": len(units) - completed_total,
                "configurations_completed": completed_total
                * len(REGULARIZATION_MULTIPLIERS)
                * 21,
                "outer_test_scores_opened": False,
                "cortical_values_opened": False,
                "last_unit_id": unit.unit_id,
            },
        )
    completed_units = [
        unit for unit in units if (unit_root / f"{unit.unit_id}.json").exists()
    ]
    completed_total = len(completed_units)
    state_path = output_root / "run-state.json"
    _write_json(
        state_path,
        {
            "schema_version": "veatic21_phase02_stage_a_state_v2",
            "status": "RUNNING" if completed_total < len(units) else "COMPLETE",
            "registration_sha256": PHASE02_REGISTRATION_SHA256,
            "stage_a_code_sha256": code_sha256,
            "work_units_total": len(units),
            "work_units_completed": completed_total,
            "work_units_remaining": len(units) - completed_total,
            "configurations_completed": completed_total
            * len(REGULARIZATION_MULTIPLIERS)
            * 21,
            "outer_test_scores_opened": False,
            "cortical_values_opened": False,
            "last_unit_id": completed_units[-1].unit_id if completed_units else None,
        },
    )
    return {
        **load_json(state_path),
        "completed_this_call": completed_this_call,
        "request_sha256": sha256_file(request_path),
        "work_registry_sha256": sha256_file(registry_path),
        "ledger_sha256": sha256_file(ledger_path) if ledger_path.exists() else None,
    }
