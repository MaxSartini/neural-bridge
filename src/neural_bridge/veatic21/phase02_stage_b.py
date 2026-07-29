"""Fresh VEATIC-only Stage B fitting from the immutable verified work registry."""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, cast

import numpy as np

from neural_bridge.veatic21.contracts import (
    PHASE02_BENCHMARK_ROOT,
    PHASE02_REGISTRATION_SHA256,
    REPOSITORY_ROOT,
)
from neural_bridge.veatic21.data import load_json, sha256_file
from neural_bridge.veatic21.phase00 import canonical_json_bytes
from neural_bridge.veatic21.phase01 import _array_digest
from neural_bridge.veatic21.phase02_features import feature_names
from neural_bridge.veatic21.phase02_metrics import binary_ranking_and_probability_metrics_fast
from neural_bridge.veatic21.phase02_stage_a import (
    StageAWorkUnit,
    _load_inputs,
    _next_power_of_two,
    _regularization_scales,
    prepare_stage_a_unit,
)

AGGREGATION_ROOT = PHASE02_BENCHMARK_ROOT / "stage-a-aggregation-stage-b-registration"
WORK_REGISTRY = AGGREGATION_ROOT / "stage-b-work-registry.jsonl.gz"
AGGREGATION_VERIFICATION = AGGREGATION_ROOT / "verification.json"
STAGE_B_EXECUTION_REGISTRATION = REPOSITORY_ROOT / (
    "internal/active/veatic21-phase02-registration/stage-b-execution-registration.json"
)
STAGE_B_BACKTEST_ROOT = PHASE02_BENCHMARK_ROOT / "stage-b-executor-backtest"
STAGE_B_MAIN_ROOT = PHASE02_BENCHMARK_ROOT / "stage-b-family-expansion"
EXPECTED_WORK_REGISTRY_SHA256 = "045e86dcf756d070aa285c2a6a4d0351914b4328441fd57eecdcc5a12ca567c4"
EXPECTED_AGGREGATION_VERIFICATION_SHA256 = (
    "1c1a9a40c202ee3573cc34121c447c5836fb0938b94706bfefcd73092ffeac22"
)
MODEL_SEED_LITERAL = "stage_b_model_seed_v1"
FLOAT_EPSILON = float(np.finfo(np.float32).eps)
mx = importlib.import_module("mlx.core")

STAGE_B_SOURCE_FILES = (
    "contracts.py",
    "data.py",
    "phase00.py",
    "phase01.py",
    "phase02_features.py",
    "phase02_metrics.py",
    "phase02_registration.py",
    "phase02_stage_a.py",
    "phase02_stage_b.py",
    "phase02_stage_b_executor.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@cache
def stage_b_code_identity() -> str:
    digest = hashlib.sha256()
    package = REPOSITORY_ROOT / "src/neural_bridge/veatic21"
    for filename in STAGE_B_SOURCE_FILES:
        path = package / filename
        digest.update(filename.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _strict_jsonl_gzip(path: Path) -> Iterator[dict[str, Any]]:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON value {value}: {path}")

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line, parse_constant=reject)
            _require(isinstance(value, dict), f"invalid object at {path}:{line_number}")
            yield cast(dict[str, Any], value)


@dataclass(frozen=True)
class StageBWorkUnit:
    work_unit_id: str
    sequence: int
    scope_id: str
    protocol: str
    repeat: int | None
    outer_fold: int
    inner_fold: int
    candidate_id: str
    feature_set_id: str
    feature_form: str
    history_depth_rows: int
    feature_count: int
    train_rows: int
    validation_rows: int
    split_sha256: str
    regularization_scale: float
    candidate_ids_sha256: str
    candidates: tuple[dict[str, Any], ...]

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> StageBWorkUnit:
        _require(
            value.get("schema_version") == "veatic21_phase02_stage_b_work_unit_v1",
            "Stage B work-unit schema changed",
        )
        candidates = tuple(cast(list[dict[str, Any]], value["candidates"]))
        unit = cls(
            work_unit_id=str(value["work_unit_id"]),
            sequence=int(value["sequence"]),
            scope_id=str(value["scope_id"]),
            protocol=str(value["protocol"]),
            repeat=None if value["repeat"] is None else int(value["repeat"]),
            outer_fold=int(value["outer_fold"]),
            inner_fold=int(value["inner_fold"]),
            candidate_id=str(value["candidate_id"]),
            feature_set_id=str(value["feature_set_id"]),
            feature_form=str(value["feature_form"]),
            history_depth_rows=int(value["history_depth_rows"]),
            feature_count=int(value["feature_count"]),
            train_rows=int(value["train_rows"]),
            validation_rows=int(value["validation_rows"]),
            split_sha256=str(value["split_sha256"]),
            regularization_scale=float(value["regularization_scale"]),
            candidate_ids_sha256=str(value["candidate_ids_sha256"]),
            candidates=candidates,
        )
        unit.validate()
        return unit

    def validate(self) -> None:
        _require(self.protocol in {"blocked", "grouped"}, "invalid Stage B protocol")
        _require(self.candidate_id.startswith("s01_e"), "invalid Stage B target")
        _require(self.history_depth_rows in range(1, 22), "invalid history depth")
        _require(
            self.feature_count == len(feature_names(self.feature_form, self.history_depth_rows)),
            "Stage B feature count changed",
        )
        _require(self.train_rows > 0 and self.validation_rows > 0, "invalid owned row counts")
        ids = [candidate_cell_id(self, candidate) for candidate in self.candidates]
        digest_state = hashlib.sha256()
        for identity in ids:
            digest_state.update(identity.encode())
            digest_state.update(b"\n")
        digest = digest_state.hexdigest()
        _require(digest == self.candidate_ids_sha256, "Stage B candidate identity changed")


@dataclass(frozen=True)
class PreparedStageBUnit:
    unit: StageBWorkUnit
    target_index: int
    x_vector: np.ndarray
    x_linear: np.ndarray
    x_sequence: np.ndarray | None
    target_values: np.ndarray
    labels: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    threshold: float
    feature_matrix_sha256: str
    scaler_sha256: str
    target_thresholds_sha256: str
    preparation_seconds: float


def iter_work_units() -> Iterator[StageBWorkUnit]:
    _require(sha256_file(WORK_REGISTRY) == EXPECTED_WORK_REGISTRY_SHA256, "work registry changed")
    _require(
        sha256_file(AGGREGATION_VERIFICATION) == EXPECTED_AGGREGATION_VERIFICATION_SHA256,
        "aggregation verification changed",
    )
    verification = load_json(AGGREGATION_VERIFICATION)
    _require(verification.get("status") == "PASS", "aggregation verification did not pass")
    _require(verification.get("stage_b_executed") is False, "Stage B registry was mutated")
    for value in _strict_jsonl_gzip(WORK_REGISTRY):
        yield StageBWorkUnit.from_json(value)


def candidate_cell_id(unit: StageBWorkUnit, candidate: dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "feature_set_id": unit.feature_set_id,
                "inner_fold": unit.inner_fold,
                "candidate": candidate,
            }
        )
    ).hexdigest()


def _largest_power_of_two_not_exceeding(value: float) -> int:
    return 1 if value < 1 else 1 << math.floor(math.log2(value))


def logical_candidate(candidate: dict[str, Any], unit: StageBWorkUnit) -> dict[str, Any]:
    """Normalize VEATIC-derived numeric formulas into a cross-inner-fold identity."""

    family = str(candidate["family"])
    result: dict[str, Any] = {
        "family": family,
        "search_role": candidate["search_role"],
    }
    if family in {"continuous_ridge", "event_logistic_l2"}:
        result["regularization_multiplier"] = float(candidate["regularization_multiplier"])
    elif family == "event_elastic_net":
        result.update(
            regularization_multiplier=float(candidate["regularization_multiplier"]),
            l1_ratio=float(candidate["l1_ratio"]),
        )
    else:
        base_width = _next_power_of_two(math.sqrt(unit.feature_count))
        base_batch = _largest_power_of_two_not_exceeding(math.sqrt(unit.train_rows))
        result.update(
            width=int(candidate["width"]),
            width_factor=float(candidate["width"]) / base_width,
            layers=int(candidate["layers"]),
            activation=str(candidate["activation"]),
            dropout=float(candidate["dropout"]),
            dropout_factor=float(candidate["dropout"]) * math.sqrt(unit.feature_count),
            optimizer=str(candidate["optimizer"]),
            learning_rate_factor=float(candidate["learning_rate"]) * math.sqrt(unit.train_rows),
            batch_factor=float(candidate["batch_size"]) / base_batch,
        )
    return result


def logical_candidate_id(candidate: dict[str, Any], unit: StageBWorkUnit) -> str:
    return hashlib.sha256(canonical_json_bytes(logical_candidate(candidate, unit))).hexdigest()


def model_seed(unit: StageBWorkUnit, candidate: dict[str, Any]) -> int:
    payload = "\0".join(
        (
            PHASE02_REGISTRATION_SHA256,
            unit.work_unit_id,
            candidate_cell_id(unit, candidate),
            MODEL_SEED_LITERAL,
        )
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big", signed=False)


def _stage_a_unit_for(unit: StageBWorkUnit, inputs: Any) -> StageAWorkUnit:
    if unit.protocol == "blocked":
        matches = [
            index
            for index, split in enumerate(inputs.blocked_splits)
            if int(split["outer_fold"]) == unit.outer_fold
        ]
    else:
        matches = [
            index
            for index, split in enumerate(inputs.grouped_splits)
            if int(split["outer_fold"]) == unit.outer_fold and int(split["repeat"]) == unit.repeat
        ]
    _require(len(matches) == 1, "Stage B split linkage is not unique")
    return StageAWorkUnit(
        unit_id=f"stage_b_prepare__{unit.work_unit_id}",
        sequence=unit.sequence,
        protocol=unit.protocol,
        split_index=matches[0],
        repeat=unit.repeat,
        outer_fold=unit.outer_fold,
        inner_fold=unit.inner_fold,
        feature_form=unit.feature_form,
        history_depth=unit.history_depth_rows,
        model_family="event_logistic_l2",
    )


def prepare_stage_b_unit(unit: StageBWorkUnit, *, inputs: Any | None = None) -> PreparedStageBUnit:
    started = time.monotonic()
    inputs = _load_inputs() if inputs is None else inputs
    stage_a_unit = _stage_a_unit_for(unit, inputs)
    prepared = prepare_stage_a_unit(inputs, stage_a_unit)
    _require(prepared.split_digest == unit.split_sha256, "Stage B split digest changed")
    target_index = inputs.candidate_ids.index(unit.candidate_id)
    train_indices = np.flatnonzero(prepared.train_masks[:, target_index])
    validation_indices = np.flatnonzero(prepared.validation_masks[:, target_index])
    _require(len(train_indices) == unit.train_rows, "Stage B training rows changed")
    _require(len(validation_indices) == unit.validation_rows, "Stage B validation rows changed")
    scales = _regularization_scales(prepared.x, prepared.train_masks)
    _require(float(scales[target_index]) == unit.regularization_scale, "regularization changed")
    x_vector = np.ascontiguousarray(prepared.x[:, :-1], dtype=np.float32)
    x_linear = np.ascontiguousarray(prepared.x, dtype=np.float32)
    sequence: np.ndarray | None = None
    if unit.feature_form == "raw_sequence_with_availability_mask":
        depth = unit.history_depth_rows
        levels = x_vector[:, : depth + 1]
        availability = x_vector[:, depth + 1 :]
        current_availability = np.zeros((len(x_vector), 1), dtype=np.float32)
        channels = np.stack(
            [levels, np.column_stack([current_availability, availability])], axis=-1
        )
        sequence = np.ascontiguousarray(channels[:, ::-1, :], dtype=np.float32)
    return PreparedStageBUnit(
        unit=unit,
        target_index=target_index,
        x_vector=x_vector,
        x_linear=x_linear,
        x_sequence=sequence,
        target_values=np.ascontiguousarray(inputs.active_values[target_index], dtype=np.float32),
        labels=np.ascontiguousarray(prepared.labels[:, target_index], dtype=np.float32),
        train_indices=train_indices,
        validation_indices=validation_indices,
        threshold=float(prepared.thresholds[target_index]),
        feature_matrix_sha256=_array_digest({"features": prepared.raw_features}),
        scaler_sha256=_array_digest({"mean": prepared.mean, "std": prepared.std}),
        target_thresholds_sha256=_array_digest({"thresholds": prepared.thresholds}),
        preparation_seconds=time.monotonic() - started,
    )


def _ridge_block(
    x: mx.array,
    right: mx.array,
    regularization: mx.array,
    penalties: mx.array,
    weights: mx.array,
    residual: mx.array,
    direction: mx.array,
    residual_square: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    rows = x.shape[0]
    for _ in range(8):
        applied = (x.T @ (x @ direction)) / rows + regularization * penalties * direction
        step = residual_square / mx.maximum(mx.sum(direction * applied), FLOAT_EPSILON)
        weights = weights + step * direction
        residual = residual - step * applied
        next_square = mx.sum(residual * residual)
        direction = residual + next_square / mx.maximum(residual_square, FLOAT_EPSILON) * direction
        residual_square = next_square
    return weights, residual, direction, residual_square


_compiled_ridge_block = mx.compile(_ridge_block)


def _proximal_block(
    x: mx.array,
    y: mx.array,
    l2: mx.array,
    l1: mx.array,
    step: mx.array,
    coefficients: mx.array,
    weights: mx.array,
    accelerated: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    mapping = mx.zeros_like(weights)
    rows = x.shape[0]
    for index in range(8):
        logits = x @ accelerated
        gradient = x.T @ (mx.sigmoid(logits) - y) / rows
        gradient = gradient + mx.concatenate([l2 * accelerated[:-1], mx.zeros((1,))])
        gradient_step = accelerated - step * gradient
        thresholded = mx.sign(gradient_step[:-1]) * mx.maximum(
            mx.abs(gradient_step[:-1]) - step * l1, 0
        )
        next_weights = mx.concatenate([thresholded, gradient_step[-1:]])
        mapping = (accelerated - next_weights) / step
        accelerated = next_weights + coefficients[index] * (next_weights - weights)
        weights = next_weights
    return weights, accelerated, mapping


_compiled_proximal_block = mx.compile(_proximal_block)


def _metric(labels: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    return binary_ranking_and_probability_metrics_fast(labels, probability, probability=True)


def _linear_cell(
    prepared: PreparedStageBUnit, candidate: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    family = str(candidate["family"])
    train = prepared.train_indices
    valid = prepared.validation_indices
    x_np = prepared.x_linear[train]
    y_np = prepared.labels[train]
    x = mx.array(x_np)
    y = mx.array(y_np)
    base = int(candidate["initial_update_budget"])
    maximum_value = candidate.get("maximum_convergence_budget")
    if maximum_value is None:
        maximum_value = candidate["undertraining_recovery_maximum_budget"]
    maximum = int(maximum_value)
    tolerance = 1 / math.sqrt(len(train))
    cadence = max(8, base // 16)
    learning: list[dict[str, Any]] = []
    iterations = 0
    converged = False

    if family == "continuous_ridge":
        target = mx.array(prepared.target_values[train])
        right = x.T @ target / len(train)
        penalties = mx.concatenate([mx.ones((x.shape[1] - 1,)), mx.zeros((1,))])
        weights = mx.zeros_like(right)
        residual = right
        direction = residual
        residual_square = mx.sum(residual * residual)
        right_norm = float(np.asarray(mx.sqrt(mx.sum(right * right)))) + FLOAT_EPSILON
        diagnostic = math.inf
        for start in range(0, maximum, 8):
            weights, residual, direction, residual_square = _compiled_ridge_block(
                x,
                right,
                mx.array(float(candidate["regularization_value"])),
                penalties,
                weights,
                residual,
                direction,
                residual_square,
            )
            iterations = start + 8
            diagnostic = float(np.asarray(mx.sqrt(residual_square))) / right_norm
            if iterations % cadence == 0 or iterations == maximum:
                score = np.asarray(mx.array(prepared.x_linear[valid]) @ weights)
                metrics = binary_ranking_and_probability_metrics_fast(
                    prepared.labels[valid], score, probability=False
                )
                learning.append({"update": iterations, "relative_residual": diagnostic, **metrics})
            if iterations >= base and diagnostic <= tolerance:
                converged = True
                break
        prediction = np.asarray(mx.array(prepared.x_linear[valid]) @ weights).astype(np.float32)
        checkpoint = {"weights": np.asarray(weights).astype(np.float32)}
        metrics = binary_ranking_and_probability_metrics_fast(
            prepared.labels[valid], prediction, probability=False
        )
        diagnostic_name = "relative_residual"
    else:
        regularization = float(candidate["regularization_value"])
        l1_ratio = float(candidate.get("l1_ratio", 0.0))
        l2 = regularization * (1 - l1_ratio)
        l1 = regularization * l1_ratio
        trace = float(np.sum(x_np.astype(np.float64) ** 2) / len(train))
        step_value = 1 / max(0.25 * trace + l2, FLOAT_EPSILON)
        weights = mx.zeros((x.shape[1],), dtype=mx.float32)
        accelerated = weights
        momentum = 1.0
        coefficients: list[float] = []
        for _ in range(maximum):
            next_momentum = (1 + math.sqrt(1 + 4 * momentum**2)) / 2
            coefficients.append((momentum - 1) / next_momentum)
            momentum = next_momentum
        coefficient_array = mx.array(np.asarray(coefficients, dtype=np.float32))
        diagnostic = math.inf
        for start in range(0, maximum, 8):
            weights, accelerated, mapping = _compiled_proximal_block(
                x,
                y,
                mx.array(l2),
                mx.array(l1),
                mx.array(step_value),
                coefficient_array[start : start + 8],
                weights,
                accelerated,
            )
            iterations = start + 8
            diagnostic = float(
                np.asarray(
                    mx.sqrt(mx.sum(mapping * mapping)) / (1 + mx.sqrt(mx.sum(weights * weights)))
                )
            )
            if iterations % cadence == 0 or iterations == maximum:
                logits = np.asarray(mx.array(prepared.x_linear[valid]) @ weights)
                probability = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
                learning.append(
                    {
                        "update": iterations,
                        "proximal_gradient_relative": diagnostic,
                        **_metric(prepared.labels[valid], probability),
                    }
                )
            if iterations >= base and diagnostic <= tolerance:
                converged = True
                break
        logits = np.asarray(mx.array(prepared.x_linear[valid]) @ weights)
        prediction = (1 / (1 + np.exp(-np.clip(logits, -30, 30)))).astype(np.float32)
        checkpoint = {"weights": np.asarray(weights).astype(np.float32)}
        metrics = _metric(prepared.labels[valid], prediction)
        diagnostic_name = "proximal_gradient_relative"
    status = "eligible_for_stage_b_aggregation" if converged else "invalid_incomplete_not_negative"
    solver = {
        "backend": "mlx_gpu_float32",
        "iterations": iterations,
        "initial_update_budget": base,
        "maximum_update_budget": maximum,
        "convergence_tolerance": tolerance,
        "diagnostic_name": diagnostic_name,
        "final_diagnostic": diagnostic,
        "converged": converged,
        "budget_recovered": iterations > base,
        "learning_curve": learning,
        "disposition": status,
    }
    return {"solver": solver, "metrics": metrics}, prediction, checkpoint


def _parameter_seed(seed: int, name: str) -> int:
    payload = seed.to_bytes(4, "big") + b"\0" + name.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _glorot(seed: int, name: str, fan_in: int, fan_out: int) -> mx.array:
    bound = math.sqrt(6 / (fan_in + fan_out))
    rng = np.random.default_rng(_parameter_seed(seed, name))
    values = rng.uniform(-bound, bound, size=(fan_in, fan_out)).astype(np.float32)
    return mx.array(values)


def _initialize_neural(
    family: str, input_dim: int, width: int, layers: int, seed: int
) -> dict[str, mx.array]:
    params: dict[str, mx.array] = {}
    if family == "event_mlp":
        incoming = input_dim
        for layer in range(layers):
            params[f"layer{layer}.w"] = _glorot(seed, f"layer{layer}.w", incoming, width)
            params[f"layer{layer}.b"] = mx.zeros((width,), dtype=mx.float32)
            incoming = width
        params["output.w"] = _glorot(seed, "output.w", incoming, 1)
        params["output.b"] = mx.zeros((1,), dtype=mx.float32)
    else:
        incoming = 2
        for layer in range(layers):
            params[f"gru{layer}.wx"] = _glorot(seed, f"gru{layer}.wx", incoming, 3 * width)
            params[f"gru{layer}.wh"] = _glorot(seed, f"gru{layer}.wh", width, 3 * width)
            params[f"gru{layer}.bx"] = mx.zeros((3 * width,), dtype=mx.float32)
            params[f"gru{layer}.bh"] = mx.zeros((3 * width,), dtype=mx.float32)
            incoming = width
        params["output.w"] = _glorot(seed, "output.w", width, 1)
        params["output.b"] = mx.zeros((1,), dtype=mx.float32)
    return params


def _activation(value: mx.array, name: str) -> mx.array:
    if name == "relu":
        return mx.maximum(value, 0)
    if name == "gelu":
        return value * (1 + mx.erf(value / math.sqrt(2))) / 2
    if name == "tanh":
        return mx.tanh(value)
    raise ValueError(f"unsupported activation: {name}")


def _dropout(value: mx.array, probability: float, key: mx.array) -> mx.array:
    if probability == 0:
        return value
    keep = 1 - probability
    return value * mx.random.bernoulli(keep, shape=value.shape, key=key) / keep


def _neural_forward(
    params: dict[str, mx.array],
    x: mx.array,
    *,
    family: str,
    layers: int,
    activation: str,
    dropout: float,
    key: mx.array,
    training: bool,
) -> mx.array:
    keys = mx.random.split(key, layers + 1)
    if family == "event_mlp":
        hidden = x
        for layer in range(layers):
            hidden = _activation(
                hidden @ params[f"layer{layer}.w"] + params[f"layer{layer}.b"], activation
            )
            if training:
                hidden = _dropout(hidden, dropout, keys[layer])
    else:
        hidden = x
        for layer in range(layers):
            width = params[f"gru{layer}.wh"].shape[0]
            state = mx.zeros((hidden.shape[0], width), dtype=mx.float32)
            outputs: list[mx.array] = []
            for step in range(hidden.shape[1]):
                gx = hidden[:, step] @ params[f"gru{layer}.wx"] + params[f"gru{layer}.bx"]
                gh = state @ params[f"gru{layer}.wh"] + params[f"gru{layer}.bh"]
                xr, xz, xn = mx.split(gx, 3, axis=-1)
                hr, hz, hn = mx.split(gh, 3, axis=-1)
                reset = mx.sigmoid(xr + hr)
                update = mx.sigmoid(xz + hz)
                candidate = mx.tanh(xn + reset * hn)
                state = (1 - update) * candidate + update * state
                outputs.append(state)
            hidden = mx.stack(outputs, axis=1)
            if training and layer + 1 < layers:
                hidden = _dropout(hidden, dropout, keys[layer])
        hidden = hidden[:, -1]
        if training:
            hidden = _dropout(hidden, dropout, keys[-1])
    return (hidden @ params["output.w"] + params["output.b"]).squeeze(-1)


@cache
def _compiled_neural_step(
    family: str, layers: int, activation: str, dropout: float, optimizer: str
) -> Any:
    def loss_fn(params: dict[str, mx.array], x: mx.array, y: mx.array, key: mx.array) -> mx.array:
        logits = _neural_forward(
            params,
            x,
            family=family,
            layers=layers,
            activation=activation,
            dropout=dropout,
            key=key,
            training=True,
        )
        return mx.mean(mx.maximum(logits, 0) - logits * y + mx.log1p(mx.exp(-mx.abs(logits))))

    value_and_grad = mx.value_and_grad(loss_fn)

    def step(
        params: dict[str, mx.array],
        first: dict[str, mx.array],
        second: dict[str, mx.array],
        x: mx.array,
        y: mx.array,
        key: mx.array,
        learning_rate: mx.array,
        weight_decay: mx.array,
    ) -> tuple[mx.array, dict[str, mx.array], dict[str, mx.array], dict[str, mx.array]]:
        loss, gradients = value_and_grad(params, x, y, key)
        next_params: dict[str, mx.array] = {}
        next_first: dict[str, mx.array] = {}
        next_second: dict[str, mx.array] = {}
        for name, parameter in params.items():
            gradient = gradients[name]
            decay = weight_decay if name.endswith((".w", ".wx", ".wh")) else 0.0
            if optimizer == "adamw":
                m = 0.9 * first[name] + 0.1 * gradient
                v = 0.999 * second[name] + 0.001 * gradient * gradient
                next_params[name] = parameter - learning_rate * (
                    m / (mx.sqrt(v) + FLOAT_EPSILON) + decay * parameter
                )
                next_first[name], next_second[name] = m, v
            else:
                adjusted = gradient + decay * parameter
                velocity = 0.9 * first[name] + adjusted
                next_params[name] = parameter - learning_rate * (adjusted + 0.9 * velocity)
                next_first[name], next_second[name] = velocity, second[name]
        return loss, next_params, next_first, next_second

    return mx.compile(step)


def _minibatch_indices(rows: np.ndarray, batch_size: int, updates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    batches = np.empty((updates, batch_size), dtype=np.int32)
    permutation = rng.permutation(rows)
    position = 0
    for update in range(updates):
        pieces: list[np.ndarray] = []
        remaining = batch_size
        while remaining:
            available = len(permutation) - position
            take = min(remaining, available)
            pieces.append(permutation[position : position + take])
            position += take
            remaining -= take
            if position == len(permutation):
                permutation = rng.permutation(rows)
                position = 0
        batches[update] = np.concatenate(pieces)
    return batches


def _better_checkpoint(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> bool:
    if incumbent is None:
        return True
    raw, old_raw = candidate["raw_pr_auc"], incumbent["raw_pr_auc"]
    if raw is None:
        return False
    if old_raw is None or raw > old_raw:
        return True
    if raw < old_raw:
        return False
    brier, old_brier = candidate["brier"], incumbent["brier"]
    return brier is not None and (old_brier is None or brier < old_brier)


def _neural_cell(
    prepared: PreparedStageBUnit, candidate: dict[str, Any], seed: int
) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    family = str(candidate["family"])
    _require(family in {"event_mlp", "event_gru"}, "invalid nonlinear family")
    if family == "event_gru":
        _require(prepared.x_sequence is not None, "GRU received a nonsequence feature")
        features = cast(np.ndarray, prepared.x_sequence)
    else:
        features = prepared.x_vector
    width = int(candidate["width"])
    layers = int(candidate["layers"])
    activation = str(candidate["activation"])
    dropout = float(candidate["dropout"])
    optimizer = str(candidate["optimizer"])
    params = _initialize_neural(family, prepared.unit.feature_count, width, layers, seed)
    first = {name: mx.zeros_like(value) for name, value in params.items()}
    second = {name: mx.zeros_like(value) for name, value in params.items()}
    base = int(candidate["initial_update_budget"])
    maximum = int(candidate["undertraining_recovery_maximum_budget"])
    cadence = max(1, base // 16)
    patience = base // 4
    batches = _minibatch_indices(
        prepared.train_indices, int(candidate["batch_size"]), maximum, seed
    )
    step = _compiled_neural_step(family, layers, activation, dropout, optimizer)
    best_metrics: dict[str, Any] | None = None
    best_params: dict[str, np.ndarray] | None = None
    best_prediction: np.ndarray | None = None
    best_update = 0
    last_improvement = 0
    learning: list[dict[str, Any]] = []
    recovered = True
    weight_decay = 1 / math.sqrt(prepared.unit.train_rows)
    eval_key = mx.random.key(_parameter_seed(seed, "eval") & 0xFFFFFFFF)
    for update in range(1, maximum + 1):
        indices = batches[update - 1]
        x_batch = mx.array(features[indices])
        y_batch = mx.array(prepared.labels[indices])
        update_key = mx.random.key(_parameter_seed(seed, f"update:{update}") & 0xFFFFFFFF)
        loss, params, first, second = step(
            params,
            first,
            second,
            x_batch,
            y_batch,
            update_key,
            mx.array(float(candidate["learning_rate"])),
            mx.array(weight_decay),
        )
        mx.eval(loss, params, first, second)
        if update % cadence:
            continue
        validation_logits = _neural_forward(
            params,
            mx.array(features[prepared.validation_indices]),
            family=family,
            layers=layers,
            activation=activation,
            dropout=dropout,
            key=eval_key,
            training=False,
        )
        validation_probability = np.asarray(mx.sigmoid(validation_logits)).astype(np.float32)
        metrics = _metric(prepared.labels[prepared.validation_indices], validation_probability)
        train_logits = _neural_forward(
            params,
            mx.array(features[prepared.train_indices]),
            family=family,
            layers=layers,
            activation=activation,
            dropout=dropout,
            key=eval_key,
            training=False,
        )
        train_values = np.asarray(train_logits).astype(np.float64)
        train_labels = prepared.labels[prepared.train_indices].astype(np.float64)
        train_loss = float(
            np.mean(
                np.maximum(train_values, 0)
                - train_values * train_labels
                + np.log1p(np.exp(-np.abs(train_values)))
            )
        )
        checkpoint = {
            "update": update,
            "minibatch_loss": float(np.asarray(loss)),
            "full_training_log_loss": train_loss,
            **metrics,
        }
        learning.append(checkpoint)
        if _better_checkpoint(checkpoint, best_metrics):
            best_metrics = checkpoint
            best_params = {
                name: np.asarray(value).astype(np.float32) for name, value in params.items()
            }
            best_prediction = validation_probability
            best_update = update
            last_improvement = update
        if update == base:
            recovered = base - last_improvement < patience
            if not recovered:
                break
    _require(best_metrics is not None and best_params is not None, "no nonlinear checkpoint")
    _require(best_prediction is not None, "no nonlinear validation prediction")
    selected_metrics = cast(dict[str, Any], best_metrics)
    selected_params = cast(dict[str, np.ndarray], best_params)
    selected_prediction = cast(np.ndarray, best_prediction)
    iterations = int(learning[-1]["update"])
    plateaued = iterations - last_improvement >= patience
    disposition = (
        "eligible_for_stage_b_aggregation"
        if plateaued
        else "valid_unplateaued_at_2b_protected_from_negative_claim"
    )
    solver = {
        "backend": "mlx_gpu_float32_compiled_functional",
        "iterations": iterations,
        "initial_update_budget": base,
        "maximum_update_budget": maximum,
        "learning_curve_cadence": cadence,
        "plateau_patience_updates": patience,
        "last_improvement_update": last_improvement,
        "plateaued": plateaued,
        "budget_recovered": iterations > base,
        "selected_checkpoint_update": best_update,
        "weight_decay": weight_decay,
        "learning_curve": learning,
        "disposition": disposition,
    }
    excluded = {"update", "minibatch_loss", "full_training_log_loss"}
    metrics = {key: value for key, value in selected_metrics.items() if key not in excluded}
    return {"solver": solver, "metrics": metrics}, selected_prediction, selected_params


def execute_stage_b_cell(
    prepared: PreparedStageBUnit, candidate: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    """Execute one exact registered Stage B cell without outer or cortical access."""

    started = time.monotonic()
    unit = prepared.unit
    cell_id = candidate_cell_id(unit, candidate)
    seed = model_seed(unit, candidate)
    if candidate["family"] in {"continuous_ridge", "event_logistic_l2", "event_elastic_net"}:
        execution, prediction, checkpoint = _linear_cell(prepared, candidate)
    else:
        execution, prediction, checkpoint = _neural_cell(prepared, candidate, seed)
    checkpoint = {
        **checkpoint,
        "validation_row_indices": prepared.validation_indices.astype(np.int32),
    }
    checkpoint_sha256 = _array_digest(checkpoint)
    prediction_sha256 = _array_digest(
        {
            "validation_indices": prepared.validation_indices.astype(np.int32),
            "prediction": prediction.astype(np.float32),
        }
    )
    record = {
        "schema_version": "veatic21_phase02_stage_b_cell_v1",
        "work_unit_id": unit.work_unit_id,
        "work_unit_sequence": unit.sequence,
        "candidate_cell_id": cell_id,
        "logical_candidate_id": logical_candidate_id(candidate, unit),
        "logical_candidate": logical_candidate(candidate, unit),
        "candidate": candidate,
        "model_seed": seed,
        "phase02_registration_sha256": PHASE02_REGISTRATION_SHA256,
        "work_registry_sha256": EXPECTED_WORK_REGISTRY_SHA256,
        "aggregation_verification_sha256": EXPECTED_AGGREGATION_VERIFICATION_SHA256,
        "stage_b_code_sha256": stage_b_code_identity(),
        "split_sha256": unit.split_sha256,
        "feature_matrix_sha256": prepared.feature_matrix_sha256,
        "scaler_sha256": prepared.scaler_sha256,
        "target_thresholds_sha256": prepared.target_thresholds_sha256,
        "train_threshold_q90": prepared.threshold,
        "train_rows": unit.train_rows,
        "validation_rows": unit.validation_rows,
        "checkpoint_sha256": checkpoint_sha256,
        "validation_prediction_sha256": prediction_sha256,
        **execution,
        "runtime_seconds": time.monotonic() - started,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "prospective_washout_candidates_opened": False,
    }
    return record, prediction, checkpoint
