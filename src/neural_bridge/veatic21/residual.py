"""VEATIC-specific frozen-AR residual heads for Phase 05."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np
from sklearn.metrics import average_precision_score

from neural_bridge.veatic21.ar import select_decision_threshold


@dataclass(frozen=True)
class ResidualRecipe:
    """One preregistered head and optimizer recipe derived from input width."""

    name: str
    hidden_width: int
    learning_rate: float
    max_steps: int
    checkpoint_interval: int

    def parameter_count(self, input_width: int) -> int:
        if self.hidden_width == 0:
            return input_width + 1
        return input_width * self.hidden_width + self.hidden_width * 2 + 1


@dataclass(frozen=True)
class ResidualCheckpoint:
    """Fully serializable best inner-validation residual checkpoint."""

    recipe: ResidualRecipe
    input_mean: np.ndarray
    input_scale: np.ndarray
    direct_weight: np.ndarray
    direct_bias: float
    hidden_weight: np.ndarray
    hidden_bias: np.ndarray
    output_weight: np.ndarray
    output_bias: float
    regularization: float
    seed: int
    best_step: int
    best_validation_pr_auc: float
    baseline_validation_pr_auc: float
    validation_delta: float
    active: bool
    training_pr_auc: float
    decision_threshold: float
    checkpoint_evaluations: int
    eval_mode: bool
    device: str


def derive_phase05_seed(source_digest: str, label: str) -> int:
    """Derive a fresh Phase 05 uint32 seed from the sealed Phase 04 identity."""

    payload = f"veatic21-phase05\0{source_digest}\0{label}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def derive_residual_recipes(input_width: int) -> tuple[ResidualRecipe, ...]:
    """Derive a compact linear/dyadic-bottleneck family from the sealed VEATIC width."""

    if input_width < 16 or input_width & (input_width - 1):
        raise ValueError("residual input width must be a power of two of at least 16")
    learning_rate = 1.0 / input_width
    max_steps = 2 * input_width
    checkpoint_interval = max(1, input_width // 16)
    hidden_widths = (0, input_width // 8, input_width // 4)
    return tuple(
        ResidualRecipe(
            name="linear" if hidden_width == 0 else f"relu-bottleneck-{hidden_width}",
            hidden_width=hidden_width,
            learning_rate=learning_rate,
            max_steps=max_steps,
            checkpoint_interval=checkpoint_interval,
        )
        for hidden_width in hidden_widths
    )


def _ar_logit(probability: np.ndarray | mx.array) -> mx.array:
    value = mx.array(probability, dtype=mx.float32)
    value = mx.clip(value, 1e-6, 1.0 - 1e-6)
    return mx.log(value) - mx.log1p(-value)


def _forward(
    parameters: dict[str, mx.array],
    features: mx.array,
    ar_logit: mx.array,
    *,
    hidden_width: int,
) -> mx.array:
    if hidden_width == 0:
        residual = features @ parameters["direct_weight"] + parameters["direct_bias"]
    else:
        hidden = mx.maximum(features @ parameters["hidden_weight"] + parameters["hidden_bias"], 0.0)
        residual = hidden @ parameters["output_weight"] + parameters["output_bias"]
    return mx.clip(ar_logit + residual, -30.0, 30.0)


def _initialize_parameters(input_width: int, hidden_width: int, seed: int) -> dict[str, mx.array]:
    if hidden_width == 0:
        return {
            "direct_weight": mx.zeros((input_width,), dtype=mx.float32),
            "direct_bias": mx.zeros((), dtype=mx.float32),
        }
    key = mx.random.key(seed)
    limit = np.sqrt(6.0 / (input_width + hidden_width))
    hidden_weight = mx.random.uniform(
        low=-limit,
        high=limit,
        shape=(input_width, hidden_width),
        key=key,
    ).astype(mx.float32)
    return {
        "hidden_weight": hidden_weight,
        "hidden_bias": mx.zeros((hidden_width,), dtype=mx.float32),
        "output_weight": mx.zeros((hidden_width,), dtype=mx.float32),
        "output_bias": mx.zeros((), dtype=mx.float32),
    }


def _numpy_parameters(parameters: dict[str, mx.array]) -> dict[str, np.ndarray]:
    return {name: np.asarray(value, dtype=np.float32).copy() for name, value in parameters.items()}


def _validation_probability(
    parameters: dict[str, mx.array],
    features: mx.array,
    ar_logit: mx.array,
    *,
    hidden_width: int,
) -> np.ndarray:
    probability = mx.sigmoid(_forward(parameters, features, ar_logit, hidden_width=hidden_width))
    mx.eval(probability)
    return np.asarray(probability, dtype=np.float64)


def fit_residual_checkpoint_mlx(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_ar_probability: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    validation_ar_probability: np.ndarray,
    *,
    recipe: ResidualRecipe,
    seed: int,
) -> tuple[ResidualCheckpoint, dict[str, Any]]:
    """Fit a residual head and select/restore its checkpoint on owned validation rows."""

    train_features = np.asarray(train_features, dtype=np.float32)
    validation_features = np.asarray(validation_features, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.int8)
    validation_labels = np.asarray(validation_labels, dtype=np.int8)
    train_ar_probability = np.asarray(train_ar_probability, dtype=np.float64)
    validation_ar_probability = np.asarray(validation_ar_probability, dtype=np.float64)
    if train_features.ndim != 2 or validation_features.ndim != 2:
        raise ValueError("residual features must be matrices")
    if train_features.shape[1] != validation_features.shape[1]:
        raise ValueError("residual train/validation widths differ")
    if train_labels.shape != (len(train_features),) or validation_labels.shape != (
        len(validation_features),
    ):
        raise ValueError("residual labels do not align")
    if train_ar_probability.shape != train_labels.shape or validation_ar_probability.shape != (
        len(validation_features),
    ):
        raise ValueError("frozen AR probabilities do not align")
    if not (
        np.isfinite(train_features).all()
        and np.isfinite(validation_features).all()
        and np.isfinite(train_ar_probability).all()
        and np.isfinite(validation_ar_probability).all()
    ):
        raise ValueError("residual inputs must be finite")
    if len(np.unique(train_labels)) != 2 or len(np.unique(validation_labels)) != 2:
        raise ValueError("residual fitting requires both classes in each owned partition")

    mx.set_default_device(mx.gpu)
    train_x = mx.array(train_features)
    validation_x = mx.array(validation_features)
    mean = mx.mean(train_x, axis=0)
    scale = mx.maximum(mx.sqrt(mx.mean(mx.square(train_x - mean), axis=0)), 1e-6)
    train_x = (train_x - mean) / scale
    validation_x = (validation_x - mean) / scale
    train_y = mx.array(train_labels, dtype=mx.float32)
    train_offset = _ar_logit(train_ar_probability)
    validation_offset = _ar_logit(validation_ar_probability)
    parameters = _initialize_parameters(train_features.shape[1], recipe.hidden_width, seed)
    regularization = 1.0 / len(train_features)
    weight_names = {name for name in parameters if "weight" in name}

    def objective(values: dict[str, mx.array]) -> mx.array:
        logits = _forward(values, train_x, train_offset, hidden_width=recipe.hidden_width)
        cross_entropy = mx.mean(mx.logaddexp(mx.zeros_like(logits), logits) - train_y * logits)
        penalty = sum(mx.sum(mx.square(values[name])) for name in weight_names)
        return cross_entropy + 0.5 * regularization * penalty

    objective_and_gradient = mx.value_and_grad(objective)
    first_moment = {name: mx.zeros_like(value) for name, value in parameters.items()}
    second_moment = {name: mx.zeros_like(value) for name, value in parameters.items()}
    beta1 = 0.9
    beta2 = 0.999
    baseline_validation_pr_auc = float(
        average_precision_score(validation_labels, validation_ar_probability)
    )
    best_validation_pr_auc = baseline_validation_pr_auc
    best_step = 0
    best_parameters = _numpy_parameters(parameters)
    checkpoint_metrics = [
        {
            "step": 0,
            "validation_pr_auc": baseline_validation_pr_auc,
            "training_objective": float(objective(parameters).item()),
        }
    ]
    for step in range(1, recipe.max_steps + 1):
        loss, gradients = objective_and_gradient(parameters)
        for name in parameters:
            first_moment[name] = beta1 * first_moment[name] + (1.0 - beta1) * gradients[name]
            second_moment[name] = beta2 * second_moment[name] + (1.0 - beta2) * mx.square(
                gradients[name]
            )
            corrected_first = first_moment[name] / (1.0 - beta1**step)
            corrected_second = second_moment[name] / (1.0 - beta2**step)
            parameters[name] = parameters[name] - recipe.learning_rate * corrected_first / (
                mx.sqrt(corrected_second) + 1e-8
            )
        mx.eval(parameters, first_moment, second_moment, loss)
        if step % recipe.checkpoint_interval == 0 or step == recipe.max_steps:
            validation_probability = _validation_probability(
                parameters,
                validation_x,
                validation_offset,
                hidden_width=recipe.hidden_width,
            )
            validation_pr_auc = float(
                average_precision_score(validation_labels, validation_probability)
            )
            checkpoint_metrics.append(
                {
                    "step": step,
                    "validation_pr_auc": validation_pr_auc,
                    "training_objective": float(loss.item()),
                }
            )
            if validation_pr_auc > best_validation_pr_auc + 1e-12:
                best_validation_pr_auc = validation_pr_auc
                best_step = step
                best_parameters = _numpy_parameters(parameters)

    restored = {name: mx.array(value) for name, value in best_parameters.items()}
    train_probability_raw = _validation_probability(
        restored, train_x, train_offset, hidden_width=recipe.hidden_width
    )
    active = best_validation_pr_auc > baseline_validation_pr_auc + 1e-12
    train_probability = train_probability_raw if active else train_ar_probability.copy()
    training_pr_auc = float(average_precision_score(train_labels, train_probability))
    threshold = select_decision_threshold(train_labels, train_probability)
    empty = np.empty((0,), dtype=np.float32)
    checkpoint = ResidualCheckpoint(
        recipe=recipe,
        input_mean=np.asarray(mean, dtype=np.float32),
        input_scale=np.asarray(scale, dtype=np.float32),
        direct_weight=best_parameters.get("direct_weight", empty),
        direct_bias=float(best_parameters.get("direct_bias", np.asarray(0.0)).item()),
        hidden_weight=best_parameters.get(
            "hidden_weight", np.empty((train_features.shape[1], 0), dtype=np.float32)
        ),
        hidden_bias=best_parameters.get("hidden_bias", empty),
        output_weight=best_parameters.get("output_weight", empty),
        output_bias=float(best_parameters.get("output_bias", np.asarray(0.0)).item()),
        regularization=regularization,
        seed=seed,
        best_step=best_step,
        best_validation_pr_auc=best_validation_pr_auc,
        baseline_validation_pr_auc=baseline_validation_pr_auc,
        validation_delta=best_validation_pr_auc - baseline_validation_pr_auc,
        active=active,
        training_pr_auc=training_pr_auc,
        decision_threshold=threshold,
        checkpoint_evaluations=len(checkpoint_metrics),
        eval_mode=True,
        device="gpu:0",
    )
    mx.clear_cache()
    return checkpoint, {
        "checkpoints": checkpoint_metrics,
        "best_step": best_step,
        "best_validation_pr_auc": best_validation_pr_auc,
        "baseline_validation_pr_auc": baseline_validation_pr_auc,
        "validation_delta": checkpoint.validation_delta,
        "active": active,
        "restored_eval_mode": True,
    }


def predict_residual_mlx(
    checkpoint: ResidualCheckpoint,
    features: np.ndarray,
    ar_probability: np.ndarray,
) -> np.ndarray:
    """Score a restored checkpoint, or return exact AR when the no-harm gate suppressed it."""

    features = np.asarray(features, dtype=np.float32)
    ar_probability = np.asarray(ar_probability, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(checkpoint.input_mean):
        raise ValueError("residual prediction width mismatch")
    if ar_probability.shape != (len(features),):
        raise ValueError("residual prediction AR rows mismatch")
    if not checkpoint.active:
        return ar_probability.copy()
    mx.set_default_device(mx.gpu)
    x = (mx.array(features) - mx.array(checkpoint.input_mean)) / mx.array(checkpoint.input_scale)
    offset = _ar_logit(ar_probability)
    if checkpoint.recipe.hidden_width == 0:
        parameters = {
            "direct_weight": mx.array(checkpoint.direct_weight),
            "direct_bias": mx.array(checkpoint.direct_bias),
        }
    else:
        parameters = {
            "hidden_weight": mx.array(checkpoint.hidden_weight),
            "hidden_bias": mx.array(checkpoint.hidden_bias),
            "output_weight": mx.array(checkpoint.output_weight),
            "output_bias": mx.array(checkpoint.output_bias),
        }
    probability = mx.sigmoid(
        _forward(parameters, x, offset, hidden_width=checkpoint.recipe.hidden_width)
    )
    mx.eval(probability)
    output = np.asarray(probability, dtype=np.float64)
    mx.clear_cache()
    if output.shape != ar_probability.shape or not np.isfinite(output).all():
        raise ValueError("invalid residual probability output")
    return output
