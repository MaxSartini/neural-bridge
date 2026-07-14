"""Leakage-resistant Optuna studies whose objectives must use MLX or MPS."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ._optional import require_upstream
from .acceleration import require_accelerator


@dataclass(frozen=True)
class AcceleratedObjectiveResult:
    value: float
    accelerator_backend: str


Objective = Callable[
    [Any, tuple[int, ...], tuple[int, ...]], AcceleratedObjectiveResult
]


@dataclass(frozen=True)
class TrainOnlyStudySpec:
    study_name: str
    n_trials: int
    sampler_seed: int
    accelerator_backend: str
    direction: str = "maximize"
    storage: str | None = None
    load_if_exists: bool = True
    timeout_seconds: float | None = None
    initial_trials: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.study_name.strip():
            raise ValueError("study_name is required")
        if self.n_trials < 1:
            raise ValueError("n_trials must be positive")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("direction must be 'maximize' or 'minimize'")
        if self.accelerator_backend not in {"mlx", "mps"}:
            raise ValueError("accelerator_backend must be 'mlx' or 'mps'")


def _index_digest(indices: tuple[int, ...]) -> str:
    encoded = json.dumps(indices, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_indices(
    inner_train_indices: Sequence[int], inner_validation_indices: Sequence[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    train = tuple(int(value) for value in inner_train_indices)
    validation = tuple(int(value) for value in inner_validation_indices)
    if not train or not validation:
        raise ValueError("Both inner-train and inner-validation indices are required")
    if len(train) != len(set(train)) or len(validation) != len(set(validation)):
        raise ValueError("Study indices must be unique within each split")
    overlap = set(train).intersection(validation)
    if overlap:
        raise ValueError(f"Inner-train and inner-validation overlap at {len(overlap)} rows")
    return train, validation


def run_train_only_study(
    spec: TrainOnlyStudySpec,
    objective: Objective,
    *,
    inner_train_indices: Sequence[int],
    inner_validation_indices: Sequence[int],
) -> Any:
    """Run official Optuna while rejecting CPU or held-out-test objectives."""

    accelerator = require_accelerator(spec.accelerator_backend)
    train, validation = _validated_indices(inner_train_indices, inner_validation_indices)
    optuna = require_upstream("optuna")
    sampler = optuna.samplers.TPESampler(seed=spec.sampler_seed)
    study = optuna.create_study(
        study_name=spec.study_name,
        direction=spec.direction,
        sampler=sampler,
        storage=spec.storage,
        load_if_exists=spec.load_if_exists,
    )
    study.set_user_attr("neural_bridge.scope", "inner_train_validation_only")
    study.set_user_attr("neural_bridge.canonical_evidence", False)
    study.set_user_attr("neural_bridge.accelerator_backend", accelerator.backend)
    study.set_user_attr("neural_bridge.accelerator_detail", accelerator.detail)
    study.set_user_attr("neural_bridge.inner_train_sha256", _index_digest(train))
    study.set_user_attr("neural_bridge.inner_validation_sha256", _index_digest(validation))
    if not study.trials:
        for params in spec.initial_trials:
            study.enqueue_trial(dict(params))

    def guarded_objective(trial: Any) -> float:
        result = objective(trial, train, validation)
        if not isinstance(result, AcceleratedObjectiveResult):
            raise TypeError("Objective must return AcceleratedObjectiveResult")
        if result.accelerator_backend != spec.accelerator_backend:
            raise RuntimeError(
                "Objective accelerator does not match the required study accelerator: "
                f"{result.accelerator_backend!r} != {spec.accelerator_backend!r}"
            )
        value = float(result.value)
        if not math.isfinite(value):
            raise ValueError("Objective value must be finite")
        trial.set_user_attr("neural_bridge.accelerator_backend", result.accelerator_backend)
        trial.set_user_attr("neural_bridge.inner_train_sha256", _index_digest(train))
        trial.set_user_attr("neural_bridge.inner_validation_sha256", _index_digest(validation))
        return value

    study.optimize(
        guarded_objective,
        n_trials=spec.n_trials,
        timeout=spec.timeout_seconds,
        gc_after_trial=True,
    )
    return study
