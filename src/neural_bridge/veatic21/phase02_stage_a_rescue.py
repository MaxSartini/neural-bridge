"""Sparse, immutable convergence rescue for registered VEATIC Stage A cells."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import time
from concurrent.futures import Executor
from dataclasses import asdict, dataclass
from typing import cast

import numpy as np

from neural_bridge.veatic21.contracts import (
    PHASE02_REGISTRATION_ROOT,
    PHASE02_REGISTRATION_SHA256,
    REPOSITORY_ROOT,
)
from neural_bridge.veatic21.data import load_json, sha256_file
from neural_bridge.veatic21.phase01 import _array_digest
from neural_bridge.veatic21.phase02_metrics import (
    binary_ranking_and_probability_metrics_fast,
    binary_ranking_metrics,
    probability_metrics,
)
from neural_bridge.veatic21.phase02_stage_a import (
    REGULARIZATION_MULTIPLIERS,
    StageAInputs,
    StageAPrepared,
    StageAWorkUnit,
    _load_inputs,
    _next_power_of_two,
    _regularization_scales,
    _stage_a_code_identity,
    prepare_stage_a_unit,
)
from neural_bridge.veatic21.phase02_stage_a_rescue_registration import (
    RESCUE_CELL_REGISTRY,
    RESCUE_REGISTRATION_REQUEST,
    RESCUE_REGISTRATION_SUMMARY,
    RESCUE_REGISTRATION_VERIFICATION,
    RESCUE_UNIT_REGISTRY,
)

mx = importlib.import_module("mlx.core")

COMPACT_RESCUE_REGISTRATION = REPOSITORY_ROOT / (
    "internal/active/veatic21-phase02-registration/convergence-rescue-registration.json"
)
RESCUE_SOLVER_SOURCE_FILES = (
    "phase02_stage_a.py",
    "phase02_stage_a_rescue_registration.py",
    "phase02_stage_a_rescue.py",
)


@dataclass(frozen=True)
class RescueCell:
    rescue_sequence: int
    rescue_cell_identity_sha256: str
    original_configuration_id: str
    original_unit_id: str
    original_unit_sequence: int
    original_unit_result_sha256: str
    protocol: str
    split_index: int
    repeat: int | None
    outer_fold: int
    inner_fold: int
    feature_form: str
    history_depth: int
    model_family: str
    candidate_id: str
    target_index: int
    regularization_index: int
    regularization_multiplier: float
    regularization_scale: float
    regularization_value: float
    train_rows: int
    validation_rows: int
    train_threshold_q90: float
    split_sha256: str
    feature_matrix_sha256: str
    scaler_sha256: str
    target_thresholds_sha256: str
    original_base_budget: int
    original_maximum_budget: int
    rescue_maximum_budget: int
    convergence_tolerance: float
    original_solver_diagnostic: float


@dataclass(frozen=True)
class RescueUnit:
    rescue_unit_sequence: int
    original_unit_sequence: int
    original_unit_id: str
    original_unit_result_sha256: str
    model_family: str
    cells: tuple[RescueCell, ...]


def rescue_solver_code_identity() -> str:
    digest = hashlib.sha256()
    package = REPOSITORY_ROOT / "src/neural_bridge/veatic21"
    for filename in RESCUE_SOLVER_SOURCE_FILES:
        path = package / filename
        digest.update(filename.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _strict_json_object(line: str) -> dict[str, object]:
    def reject(value: str) -> object:
        raise ValueError(f"non-finite JSON constant: {value}")

    result = json.loads(line, parse_constant=reject)
    if not isinstance(result, dict):
        raise TypeError("rescue registry row is not an object")
    return cast(dict[str, object], result)


def _cell_from_dict(value: dict[str, object]) -> RescueCell:
    return RescueCell(
        rescue_sequence=int(cast(int, value["rescue_sequence"])),
        rescue_cell_identity_sha256=str(value["rescue_cell_identity_sha256"]),
        original_configuration_id=str(value["original_configuration_id"]),
        original_unit_id=str(value["original_unit_id"]),
        original_unit_sequence=int(cast(int, value["original_unit_sequence"])),
        original_unit_result_sha256=str(value["original_unit_result_sha256"]),
        protocol=str(value["protocol"]),
        split_index=int(cast(int, value["split_index"])),
        repeat=(None if value["repeat"] is None else int(cast(int, value["repeat"]))),
        outer_fold=int(cast(int, value["outer_fold"])),
        inner_fold=int(cast(int, value["inner_fold"])),
        feature_form=str(value["feature_form"]),
        history_depth=int(cast(int, value["history_depth"])),
        model_family=str(value["model_family"]),
        candidate_id=str(value["candidate_id"]),
        target_index=int(cast(int, value["target_index"])),
        regularization_index=int(cast(int, value["regularization_index"])),
        regularization_multiplier=float(cast(float, value["regularization_multiplier"])),
        regularization_scale=float(cast(float, value["regularization_scale"])),
        regularization_value=float(cast(float, value["regularization_value"])),
        train_rows=int(cast(int, value["train_rows"])),
        validation_rows=int(cast(int, value["validation_rows"])),
        train_threshold_q90=float(cast(float, value["train_threshold_q90"])),
        split_sha256=str(value["split_sha256"]),
        feature_matrix_sha256=str(value["feature_matrix_sha256"]),
        scaler_sha256=str(value["scaler_sha256"]),
        target_thresholds_sha256=str(value["target_thresholds_sha256"]),
        original_base_budget=int(cast(int, value["original_base_budget"])),
        original_maximum_budget=int(cast(int, value["original_maximum_budget"])),
        rescue_maximum_budget=int(cast(int, value["rescue_maximum_budget"])),
        convergence_tolerance=float(cast(float, value["convergence_tolerance"])),
        original_solver_diagnostic=float(cast(float, value["original_solver_diagnostic"])),
    )


def verify_rescue_registry_inputs() -> dict[str, object]:
    """Verify the compact freeze and every external rescue registry artifact."""

    compact = load_json(COMPACT_RESCUE_REGISTRATION)
    external = cast(dict[str, object], compact["external_registration"])
    expected = {
        RESCUE_REGISTRATION_REQUEST: external["request_sha256"],
        RESCUE_CELL_REGISTRY: external["undertrained_cell_registry_sha256"],
        RESCUE_UNIT_REGISTRY: external["affected_unit_registry_sha256"],
        RESCUE_REGISTRATION_SUMMARY: external["summary_sha256"],
        RESCUE_REGISTRATION_VERIFICATION: external["verification_sha256"],
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"rescue registry artifact identity changed: {path}")
    if cast(dict[str, object], compact["coverage"])["undertrained_cells"] != 113_392:
        raise ValueError("compact rescue cell count changed")
    if compact["rescue_execution_authorized"] is not False:
        raise ValueError("registration must not itself authorize the main rescue")
    return compact


def load_rescue_registry() -> tuple[RescueUnit, ...]:
    """Load the exact registered cells grouped by their immutable Stage A unit."""

    verify_rescue_registry_inputs()
    cells_by_id: dict[str, RescueCell] = {}
    with RESCUE_CELL_REGISTRY.open(encoding="utf-8") as handle:
        for line in handle:
            cell = _cell_from_dict(_strict_json_object(line))
            if cell.rescue_cell_identity_sha256 in cells_by_id:
                raise ValueError("duplicate rescue-cell identity")
            cells_by_id[cell.rescue_cell_identity_sha256] = cell

    units: list[RescueUnit] = []
    consumed: set[str] = set()
    with RESCUE_UNIT_REGISTRY.open(encoding="utf-8") as handle:
        for line in handle:
            value = _strict_json_object(line)
            identities = cast(list[str], value["rescue_cell_identity_sha256s"])
            cells = tuple(cells_by_id[identity] for identity in identities)
            if not cells or len(cells) != int(cast(int, value["undertrained_cell_count"])):
                raise ValueError("rescue unit cell count changed")
            unit = RescueUnit(
                rescue_unit_sequence=int(cast(int, value["rescue_unit_sequence"])),
                original_unit_sequence=int(cast(int, value["original_unit_sequence"])),
                original_unit_id=str(value["original_unit_id"]),
                original_unit_result_sha256=str(value["original_unit_result_sha256"]),
                model_family=str(value["model_family"]),
                cells=cells,
            )
            if any(
                cell.original_unit_id != unit.original_unit_id
                or cell.model_family != unit.model_family
                or cell.original_unit_result_sha256 != unit.original_unit_result_sha256
                for cell in cells
            ):
                raise ValueError("rescue cell linkage changed")
            units.append(unit)
            consumed.update(identities)
    if len(units) != 14_465 or len(cells_by_id) != 113_392 or consumed != set(cells_by_id):
        raise ValueError("rescue registry coverage changed")
    if [unit.rescue_unit_sequence for unit in units] != list(range(len(units))):
        raise ValueError("rescue unit ordering changed")
    return tuple(units)


def stage_a_units_by_sequence(inputs: StageAInputs) -> dict[int, StageAWorkUnit]:
    """Recreate the frozen Stage A registry without opening any outcome."""

    del inputs
    from neural_bridge.veatic21.phase02_stage_a import enumerate_stage_a_work_units

    registration = load_json(PHASE02_REGISTRATION_ROOT / "experiment-registration.json")
    split_registry = load_json(PHASE02_REGISTRATION_ROOT / "split-registry.json")
    units = enumerate_stage_a_work_units(registration, split_registry)
    return {unit.sequence: unit for unit in units}


def _validate_prepared_identity(
    inputs: StageAInputs,
    unit: StageAWorkUnit,
    rescue: RescueUnit,
    prepared: StageAPrepared,
) -> None:
    if unit.unit_id != rescue.original_unit_id or unit.model_family != rescue.model_family:
        raise ValueError("Stage A and rescue unit identity mismatch")
    if prepared.split_digest != rescue.cells[0].split_sha256:
        raise ValueError("rescue split identity changed")
    expected = {
        "feature_matrix_sha256": _array_digest({"features": prepared.raw_features}),
        "scaler_sha256": _array_digest({"mean": prepared.mean, "std": prepared.std}),
        "target_thresholds_sha256": _array_digest({"thresholds": prepared.thresholds}),
    }
    scales = _regularization_scales(prepared.x, prepared.train_masks)
    unit_base_budget = _next_power_of_two(
        math.sqrt(float(np.min(prepared.train_masks.sum(axis=0))))
    )
    for cell in rescue.cells:
        if (
            cell.protocol != unit.protocol
            or cell.split_index != unit.split_index
            or cell.repeat != unit.repeat
            or cell.outer_fold != unit.outer_fold
            or cell.inner_fold != unit.inner_fold
            or cell.feature_form != unit.feature_form
            or cell.history_depth != unit.history_depth
        ):
            raise ValueError("rescue Stage A semantic linkage changed")
        if any(getattr(cell, name) != digest for name, digest in expected.items()):
            raise ValueError(
                f"rescue prepared-data identity changed: {cell.original_configuration_id}"
            )
        target = cell.target_index
        scale = float(scales[target])
        if int(prepared.train_masks[:, target].sum()) != cell.train_rows:
            raise ValueError("rescue training-row ownership changed")
        if int(prepared.validation_masks[:, target].sum()) != cell.validation_rows:
            raise ValueError("rescue validation-row ownership changed")
        if float(prepared.thresholds[target]) != cell.train_threshold_q90:
            raise ValueError("rescue threshold changed")
        if scale != cell.regularization_scale:
            raise ValueError("rescue regularization scale changed")
        if float(REGULARIZATION_MULTIPLIERS[cell.regularization_index]) != (
            cell.regularization_multiplier
        ):
            raise ValueError("rescue regularization multiplier changed")
        if float(REGULARIZATION_MULTIPLIERS[cell.regularization_index] * scale) != (
            cell.regularization_value
        ):
            raise ValueError("rescue regularization value changed")
        if cell.original_base_budget != unit_base_budget:
            raise ValueError("rescue base budget changed")
        if cell.rescue_maximum_budget != 16 * cell.original_base_budget:
            raise ValueError("rescue maximum budget changed")
        if cell.convergence_tolerance != 1 / math.sqrt(cell.train_rows):
            raise ValueError("rescue convergence tolerance changed")
    if inputs.candidate_ids != tuple(f"s01_e{index:02d}" for index in range(1, 22)):
        raise ValueError("Stage A target registry changed")


def _ridge_update_block_sparse(
    x: mx.array,
    masks: mx.array,
    counts: mx.array,
    penalties: mx.array,
    regularization: mx.array,
    weights: mx.array,
    residual: mx.array,
    direction: mx.array,
    residual_square: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    for _ in range(8):
        predictions = mx.matmul(x, direction)
        applied = mx.matmul(x.T, predictions * masks) / counts[None, :]
        applied = applied + direction * penalties[:, None] * regularization[None, :]
        denominator = mx.sum(direction * applied, axis=0)
        step = residual_square / mx.maximum(denominator, np.finfo(np.float32).eps)
        weights = weights + direction * step[None, :]
        residual = residual - applied * step[None, :]
        next_square = mx.sum(residual * residual, axis=0)
        beta = next_square / mx.maximum(residual_square, np.finfo(np.float32).eps)
        direction = residual + direction * beta[None, :]
        residual_square = next_square
    return weights, residual, direction, residual_square


_compiled_ridge_update_block_sparse = mx.compile(_ridge_update_block_sparse)


def _logistic_update_block_sparse(
    x: mx.array,
    labels: mx.array,
    masks: mx.array,
    counts: mx.array,
    regularization: mx.array,
    penalties: mx.array,
    step: mx.array,
    coefficients: mx.array,
    weights: mx.array,
    accelerated: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    gradient = mx.zeros_like(weights)
    for index in range(8):
        logits = mx.matmul(x, accelerated)
        error = (mx.sigmoid(logits) - labels) * masks
        gradient = mx.matmul(x.T, error) / counts[None, :]
        gradient = gradient + accelerated * penalties[:, None] * regularization[None, :]
        next_weights = accelerated - gradient * step[None, :]
        accelerated = next_weights + coefficients[index] * (next_weights - weights)
        weights = next_weights
    return weights, accelerated, gradient


_compiled_logistic_update_block_sparse = mx.compile(_logistic_update_block_sparse)


def _cell_batches(
    cells: tuple[RescueCell, ...], batch_size: int
) -> tuple[tuple[RescueCell, ...], ...]:
    if batch_size < 1:
        raise ValueError("cell batch size must be positive")
    return tuple(cells[start : start + batch_size] for start in range(0, len(cells), batch_size))


def _record_learning_checkpoint(
    completed: int, cell: RescueCell, converged_now: bool
) -> bool:
    """Keep the registered budget landmarks plus the exact stopping checkpoint."""

    landmarks = {
        cell.original_maximum_budget,
        2 * cell.original_base_budget,
        4 * cell.original_base_budget,
        8 * cell.original_base_budget,
        12 * cell.original_base_budget,
        cell.rescue_maximum_budget,
    }
    return completed in landmarks or converged_now


def _solve_ridge_batch(
    inputs: StageAInputs,
    prepared: StageAPrepared,
    cells: tuple[RescueCell, ...],
    *,
    compiled_update_blocks: bool,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    target_indices = np.asarray([cell.target_index for cell in cells], dtype=int)
    masks_np = prepared.train_masks[:, target_indices].astype(np.float32)
    counts_np = masks_np.sum(axis=0).astype(np.float32)
    y_np = inputs.active_values[target_indices].T.astype(np.float32)
    y_np[masks_np == 0] = 0
    regularization_np = np.asarray([cell.regularization_value for cell in cells], np.float32)
    penalties_np = np.ones(prepared.x.shape[1], dtype=np.float32)
    penalties_np[-1] = 0

    x = mx.array(prepared.x)
    masks = mx.array(masks_np)
    counts = mx.array(counts_np)
    y = mx.array(y_np)
    regularization = mx.array(regularization_np)
    penalties = mx.array(penalties_np)
    right = mx.matmul(x.T, y) / counts[None, :]
    weights = mx.zeros_like(right)
    residual = right
    direction = residual
    residual_square = mx.sum(residual * residual, axis=0)
    right_norm = mx.sqrt(mx.sum(right * right, axis=0)) + np.finfo(np.float32).eps
    tolerances = np.asarray([cell.convergence_tolerance for cell in cells], np.float32)
    minimums = np.asarray([cell.original_maximum_budget for cell in cells], int)
    maximums = np.asarray([cell.rescue_maximum_budget for cell in cells], int)
    iterations = np.zeros(len(cells), dtype=int)
    diagnostics = np.full(len(cells), np.inf, dtype=np.float32)
    converged = np.zeros(len(cells), dtype=bool)
    final_weights = np.zeros((prepared.x.shape[1], len(cells)), dtype=np.float32)
    learning: list[list[dict[str, float | int]]] = [[] for _ in cells]
    update = (
        _compiled_ridge_update_block_sparse
        if compiled_update_blocks
        else _ridge_update_block_sparse
    )
    for block_start in range(0, int(max(maximums)), 8):
        active_np = (~converged) & (block_start < maximums)
        if not np.any(active_np):
            break
        previous = (weights, residual, direction, residual_square)
        next_state = update(
            x,
            masks,
            counts,
            penalties,
            regularization,
            weights,
            residual,
            direction,
            residual_square,
        )
        active = mx.array(active_np)
        weights = mx.where(active[None, :], next_state[0], previous[0])
        residual = mx.where(active[None, :], next_state[1], previous[1])
        direction = mx.where(active[None, :], next_state[2], previous[2])
        residual_square = mx.where(active, next_state[3], previous[3])
        completed = block_start + 8
        current = np.asarray(mx.sqrt(residual_square) / right_norm)
        weight_snapshot = np.asarray(weights)
        for index in np.flatnonzero(active_np):
            iterations[index] = completed
            diagnostics[index] = current[index]
            converged_now = completed >= minimums[index] and current[index] <= tolerances[index]
            if _record_learning_checkpoint(completed, cells[index], bool(converged_now)):
                learning[index].append(
                    {"update": completed, "relative_residual": float(current[index])}
                )
            if converged_now:
                converged[index] = True
                final_weights[:, index] = weight_snapshot[:, index]
            elif completed >= maximums[index]:
                final_weights[:, index] = weight_snapshot[:, index]
    predictions = prepared.x @ final_weights
    records: list[dict[str, object]] = [
        {
            "iterations": int(iterations[index]),
            "diagnostic_name": "relative_residual",
            "final_diagnostic": float(diagnostics[index]),
            "converged": bool(converged[index]),
            "learning_curve": learning[index],
        }
        for index in range(len(cells))
    ]
    return predictions, records


def _solve_logistic_batch(
    prepared: StageAPrepared,
    cells: tuple[RescueCell, ...],
    *,
    compiled_update_blocks: bool,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    target_indices = np.asarray([cell.target_index for cell in cells], dtype=int)
    masks_np = prepared.train_masks[:, target_indices].astype(np.float32)
    labels_np = prepared.labels[:, target_indices].astype(np.float32)
    counts_np = masks_np.sum(axis=0).astype(np.float32)
    regularization_np = np.asarray([cell.regularization_value for cell in cells], np.float32)
    penalties_np = np.ones(prepared.x.shape[1], dtype=np.float32)
    penalties_np[-1] = 0
    traces = np.asarray(
        [
            np.sum(prepared.x[masks_np[:, index].astype(bool)].astype(np.float64) ** 2)
            / counts_np[index]
            for index in range(len(cells))
        ],
        dtype=np.float32,
    )
    step_np = 1 / np.maximum(0.25 * traces + regularization_np, np.finfo(np.float32).eps)

    x = mx.array(prepared.x)
    labels = mx.array(labels_np)
    masks = mx.array(masks_np)
    counts = mx.array(counts_np)
    regularization = mx.array(regularization_np)
    penalties = mx.array(penalties_np)
    step = mx.array(step_np)
    weights = mx.zeros((prepared.x.shape[1], len(cells)), dtype=mx.float32)
    accelerated = weights
    tolerances = np.asarray([cell.convergence_tolerance for cell in cells], np.float32)
    minimums = np.asarray([cell.original_maximum_budget for cell in cells], int)
    maximums = np.asarray([cell.rescue_maximum_budget for cell in cells], int)
    coefficients: list[float] = []
    momentum = 1.0
    for _ in range(int(max(maximums))):
        next_momentum = (1 + math.sqrt(1 + 4 * momentum**2)) / 2
        coefficients.append((momentum - 1) / next_momentum)
        momentum = next_momentum
    coefficient_mx = mx.array(np.asarray(coefficients, dtype=np.float32))
    converged = np.zeros(len(cells), dtype=bool)
    iterations = np.zeros(len(cells), dtype=int)
    diagnostics = np.full(len(cells), np.inf, dtype=np.float32)
    final_weights = np.zeros((prepared.x.shape[1], len(cells)), dtype=np.float32)
    learning: list[list[dict[str, float | int]]] = [[] for _ in cells]
    update = (
        _compiled_logistic_update_block_sparse
        if compiled_update_blocks
        else _logistic_update_block_sparse
    )
    for block_start in range(0, int(max(maximums)), 8):
        active_np = (~converged) & (block_start < maximums)
        if not np.any(active_np):
            break
        previous_weights, previous_accelerated = weights, accelerated
        next_weights, next_accelerated, gradient = update(
            x,
            labels,
            masks,
            counts,
            regularization,
            penalties,
            step,
            coefficient_mx[block_start : block_start + 8],
            weights,
            accelerated,
        )
        active = mx.array(active_np)
        weights = mx.where(active[None, :], next_weights, previous_weights)
        accelerated = mx.where(active[None, :], next_accelerated, previous_accelerated)
        relative = mx.sqrt(mx.sum(gradient * gradient, axis=0)) / (
            mx.sqrt(mx.sum(weights * weights, axis=0)) + 1.0
        )
        current = np.asarray(relative)
        weight_snapshot = np.asarray(weights)
        completed = block_start + 8
        for index in np.flatnonzero(active_np):
            iterations[index] = completed
            diagnostics[index] = current[index]
            converged_now = completed >= minimums[index] and current[index] <= tolerances[index]
            if _record_learning_checkpoint(completed, cells[index], bool(converged_now)):
                learning[index].append(
                    {"update": completed, "relative_gradient": float(current[index])}
                )
            if converged_now:
                converged[index] = True
                final_weights[:, index] = weight_snapshot[:, index]
            elif completed >= maximums[index]:
                final_weights[:, index] = weight_snapshot[:, index]
    logits = prepared.x @ final_weights
    predictions = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    records: list[dict[str, object]] = [
        {
            "iterations": int(iterations[index]),
            "diagnostic_name": "relative_gradient",
            "final_diagnostic": float(diagnostics[index]),
            "converged": bool(converged[index]),
            "learning_curve": learning[index],
        }
        for index in range(len(cells))
    ]
    return predictions, records


def solve_rescue_unit(
    inputs: StageAInputs,
    stage_a_unit: StageAWorkUnit,
    rescue_unit: RescueUnit,
    prepared: StageAPrepared,
    *,
    cell_batch_size: int,
    compiled_update_blocks: bool,
) -> tuple[np.ndarray, list[dict[str, object]], float]:
    """Solve only registered undertrained cells, with converged cells frozen per checkpoint."""

    started = time.monotonic()
    _validate_prepared_identity(inputs, stage_a_unit, rescue_unit, prepared)
    predictions = np.empty((len(prepared.x), len(rescue_unit.cells)), dtype=np.float32)
    diagnostics: list[dict[str, object]] = []
    offset = 0
    for batch in _cell_batches(rescue_unit.cells, cell_batch_size):
        if rescue_unit.model_family == "continuous_ridge":
            batch_predictions, batch_diagnostics = _solve_ridge_batch(
                inputs, prepared, batch, compiled_update_blocks=compiled_update_blocks
            )
        elif rescue_unit.model_family == "event_logistic_l2":
            batch_predictions, batch_diagnostics = _solve_logistic_batch(
                prepared, batch, compiled_update_blocks=compiled_update_blocks
            )
        else:
            raise ValueError(f"unsupported rescue family: {rescue_unit.model_family}")
        predictions[:, offset : offset + len(batch)] = batch_predictions
        diagnostics.extend(batch_diagnostics)
        offset += len(batch)
    return predictions, diagnostics, time.monotonic() - started


def finalize_rescue_unit(
    inputs: StageAInputs,
    stage_a_unit: StageAWorkUnit,
    rescue_unit: RescueUnit,
    prepared: StageAPrepared,
    predictions: np.ndarray,
    diagnostics: list[dict[str, object]],
    *,
    solve_seconds: float,
    metric_executor: Executor | None,
    fast_metrics: bool,
    execution_provenance: dict[str, object],
) -> dict[str, object]:
    """Create linked cell records without mutating or copying the full Stage A result."""

    started = time.monotonic()

    def build(index: int) -> dict[str, object]:
        cell = rescue_unit.cells[index]
        owned = prepared.validation_masks[:, cell.target_index]
        score = predictions[owned, index]
        labels = prepared.labels[owned, cell.target_index]
        if fast_metrics:
            metrics = binary_ranking_and_probability_metrics_fast(
                labels, score, probability=cell.model_family == "event_logistic_l2"
            )
        else:
            metrics = {
                **binary_ranking_metrics(labels, score),
                **(
                    probability_metrics(labels, score)
                    if cell.model_family == "event_logistic_l2"
                    else {"brier": None}
                ),
            }
        diagnostic = diagnostics[index]
        converged = bool(diagnostic["converged"])
        return {
            "schema_version": "veatic21_phase02_stage_a_rescue_cell_result_v1",
            "rescue_sequence": cell.rescue_sequence,
            "rescue_cell_identity_sha256": cell.rescue_cell_identity_sha256,
            "original_configuration_id": cell.original_configuration_id,
            "original_unit_id": cell.original_unit_id,
            "original_unit_result_sha256": cell.original_unit_result_sha256,
            "status": "completed" if converged else "invalid",
            "disposition": (
                "eligible_for_inner_aggregation"
                if converged
                else "invalid_nonconverged_after_registered_maximum_budget"
            ),
            "converged": converged,
            "model_family": cell.model_family,
            "candidate_id": cell.candidate_id,
            "feature_form": cell.feature_form,
            "history_depth_rows": cell.history_depth,
            "history_depth_seconds": cell.history_depth / 2,
            "regularization_index": cell.regularization_index,
            "regularization_multiplier": cell.regularization_multiplier,
            "regularization_scale": cell.regularization_scale,
            "regularization_value": cell.regularization_value,
            "train_threshold_q90": cell.train_threshold_q90,
            "train_rows": cell.train_rows,
            "validation_rows": cell.validation_rows,
            "original_base_budget": cell.original_base_budget,
            "original_maximum_budget": cell.original_maximum_budget,
            "rescue_maximum_budget": cell.rescue_maximum_budget,
            "convergence_tolerance": cell.convergence_tolerance,
            "original_solver_diagnostic": cell.original_solver_diagnostic,
            **diagnostic,
            **metrics,
        }

    indices = range(len(rescue_unit.cells))
    records = (
        [build(index) for index in indices]
        if metric_executor is None
        else list(metric_executor.map(build, indices))
    )
    return {
        "schema_version": "veatic21_phase02_stage_a_rescue_unit_result_v1",
        "rescue_unit": {
            **asdict(rescue_unit),
            "cells": [cell.rescue_cell_identity_sha256 for cell in rescue_unit.cells],
        },
        "stage_a_unit": asdict(stage_a_unit),
        "scientific_registration_sha256": PHASE02_REGISTRATION_SHA256,
        "stage_a_solver_code_sha256": _stage_a_code_identity(),
        "rescue_solver_code_sha256": rescue_solver_code_identity(),
        "rescue_cell_registry_sha256": sha256_file(RESCUE_CELL_REGISTRY),
        "rescue_unit_registry_sha256": sha256_file(RESCUE_UNIT_REGISTRY),
        "split_sha256": prepared.split_digest,
        "feature_matrix_sha256": _array_digest({"features": prepared.raw_features}),
        "scaler_sha256": _array_digest({"mean": prepared.mean, "std": prepared.std}),
        "target_thresholds_sha256": _array_digest({"thresholds": prepared.thresholds}),
        "cell_count": len(records),
        "records": records,
        "runtime_seconds": prepared.preparation_seconds
        + solve_seconds
        + (time.monotonic() - started),
        "execution_provenance": execution_provenance,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "aggregation_or_pruning_performed": False,
    }


def prepare_registered_rescue_unit(
    inputs: StageAInputs,
    stage_a_by_sequence: dict[int, StageAWorkUnit],
    rescue_unit: RescueUnit,
) -> tuple[StageAWorkUnit, StageAPrepared]:
    stage_a_unit = stage_a_by_sequence[rescue_unit.original_unit_sequence]
    prepared = prepare_stage_a_unit(inputs, stage_a_unit)
    _validate_prepared_identity(inputs, stage_a_unit, rescue_unit, prepared)
    return stage_a_unit, prepared


def load_rescue_runtime_inputs() -> tuple[StageAInputs, dict[int, StageAWorkUnit]]:
    inputs = _load_inputs()
    return inputs, stage_a_units_by_sequence(inputs)
