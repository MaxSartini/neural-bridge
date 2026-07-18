"""MLX adapter for the same backend-neutral AGAIN residual configuration."""

from __future__ import annotations

import hashlib
import importlib
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from .metrics import top_fraction_lift
from .models import ResidualConfig, ResidualResult, ResidualTrainingData


def train(
    data: ResidualTrainingData,
    config: ResidualConfig,
    *,
    seed: int,
    checkpoint_dir: Path,
) -> ResidualResult:
    mx = importlib.import_module("mlx.core")
    nn = importlib.import_module("mlx.nn")
    optim = importlib.import_module("mlx.optimizers")

    mx.random.seed(seed)

    class Head(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = data.train_x.shape[1]
            self.alpha = mx.array([config.alpha_initial_logit], dtype=mx.float32)
            self.gate = nn.Linear(width, 1)
            self.conf_gate = nn.Linear(3, 1)
            if config.architecture == "short_temporal_conv_residual":
                sequence_width = config.sequence_window * config.sequence_channels
                if not sequence_width or sequence_width > width:
                    raise ValueError("short temporal head requires valid sequence dimensions")
                self.conv = nn.Linear(config.sequence_channels * 3, config.hidden)
                self.post = nn.Linear(config.hidden + width - sequence_width, config.hidden)
            else:
                self.layers = [
                    nn.Linear(width, config.hidden),
                    nn.Linear(config.hidden, config.hidden),
                ]
            self.out = nn.Linear(config.hidden, 2)

        def alpha_value(self):
            return mx.sigmoid(self.alpha) * config.alpha_cap

        def gate_value(self, x, ar_score):
            gate = mx.sigmoid(self.gate(x) - config.gate_bias)
            if config.architecture == "low_ar_confidence_temporal_residual":
                probability = mx.sigmoid(ar_score[:, None])
                confidence = mx.concatenate(
                    [ar_score[:, None], mx.abs(ar_score[:, None]), probability * (1 - probability)],
                    axis=1,
                )
                gate = gate * mx.sigmoid(self.conf_gate(confidence))
            return gate

        def hidden(self, x):
            if config.architecture != "short_temporal_conv_residual":
                for layer in self.layers:
                    x = nn.gelu(layer(x))
                return x
            sequence_width = config.sequence_window * config.sequence_channels
            sequence = x[:, :sequence_width].reshape(
                (x.shape[0], config.sequence_window, config.sequence_channels)
            )
            extra = x[:, sequence_width:]
            padded = mx.concatenate(
                [mx.zeros((x.shape[0], 2, config.sequence_channels)), sequence], axis=1
            )
            hidden = None
            for position in range(config.sequence_window):
                window = padded[:, position : position + 3].reshape(
                    (x.shape[0], config.sequence_channels * 3)
                )
                hidden = nn.gelu(self.conv(window))
            if extra.shape[1]:
                hidden = mx.concatenate([hidden, extra], axis=1)
            return nn.gelu(self.post(hidden))

        def __call__(self, x, ar_score, ar_continuous):
            residual = self.out(self.hidden(x))
            scale = self.alpha_value() * self.gate_value(x, ar_score)
            continuous = ar_continuous[:, None] + scale * residual[:, :1]
            event = ar_score[:, None] + scale * residual[:, 1:]
            return mx.concatenate([continuous, event], axis=1)

    model = Head()
    optimizer = optim.AdamW(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    inner_train, inner_val = np.asarray(data.inner_train), np.asarray(data.inner_val)
    q80, q90 = np.quantile(data.train_continuous[inner_train], [0.8, 0.9])
    rng = np.random.default_rng(seed)

    def loss_fn(model_obj, x, ar_score, ar_continuous, y, continuous, weights):
        output = model_obj(x, ar_score, ar_continuous)
        regression = mx.mean(nn.losses.huber_loss(output[:, :1], continuous, delta=1.0) * weights)
        if data.target_type == "event":
            loss = regression + config.lambda_binary * mx.mean(
                nn.losses.binary_cross_entropy(output[:, 1:], y, with_logits=True)
            )
        else:
            loss = regression + 0.03 * mx.mean((output[:, :1] - ar_continuous[:, None]) ** 2)
        return loss + 0.01 * mx.mean(model_obj.alpha_value() ** 2)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def predict(values, event_floor, continuous_floor):
        parts = []
        for start in range(0, len(values), config.batch_size):
            output = model(
                mx.array(values[start : start + config.batch_size], dtype=mx.float32),
                mx.array(event_floor[start : start + config.batch_size], dtype=mx.float32),
                mx.array(continuous_floor[start : start + config.batch_size], dtype=mx.float32),
            )
            mx.eval(output)
            parts.append(np.asarray(output))
        return np.concatenate(parts)

    checkpoint = checkpoint_dir / f"residual__{config.architecture}__seed{seed}__mlx.npz"
    best_key, best_epoch, stale = 0.0, 0, 0
    for epoch in range(1, config.max_epochs + 1):
        order = rng.permutation(inner_train)
        for start in range(0, len(order), config.batch_size):
            relative = order[start : start + config.batch_size]
            continuous_np = data.train_continuous[relative].astype(np.float32)[:, None]
            weights = np.ones_like(continuous_np)
            if data.target_type == "continuous":
                weights += (continuous_np >= q80) + 2 * (continuous_np >= q90)
            loss, gradients = loss_and_grad(
                model,
                mx.array(data.train_x[relative]),
                mx.array(data.ar_train_score[relative]),
                mx.array(data.ar_train_continuous[relative]),
                mx.array(data.train_y[relative].astype(np.float32)[:, None]),
                mx.array(continuous_np),
                mx.array(weights.astype(np.float32)),
            )
            gradients, _ = optim.clip_grad_norm(gradients, 1.0)
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
        val = predict(
            data.train_x[inner_val],
            data.ar_train_score[inner_val],
            data.ar_train_continuous[inner_val],
        )
        if data.target_type == "event":
            base = average_precision_score(data.train_y[inner_val], data.ar_train_score[inner_val])
            value = average_precision_score(data.train_y[inner_val], val[:, 1]) - base
        else:
            base = top_fraction_lift(
                data.train_continuous[inner_val], data.ar_train_continuous[inner_val], 0.05
            )
            value = top_fraction_lift(data.train_continuous[inner_val], val[:, 0], 0.05) - base
        if math.isfinite(value) and value > best_key:
            best_key, best_epoch, stale = float(value), epoch, 0
            model.save_weights(str(checkpoint))
        else:
            stale += 1
        if stale >= config.patience:
            break

    if best_epoch:
        model.load_weights(str(checkpoint))
        train_output = predict(data.train_x, data.ar_train_score, data.ar_train_continuous)
        test_output = predict(data.test_x, data.ar_test_score, data.ar_test_continuous)
    else:
        checkpoint = None
        train_output = np.column_stack([data.ar_train_continuous, data.ar_train_score])
        test_output = np.column_stack([data.ar_test_continuous, data.ar_test_score])
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None
    score_column = 0 if data.target_type == "continuous" and best_epoch else 1
    return ResidualResult(
        train_score=train_output[:, score_column].astype(np.float32),
        test_score=test_output[:, score_column].astype(np.float32),
        train_continuous=train_output[:, 0].astype(np.float32),
        test_continuous=test_output[:, 0].astype(np.float32),
        backend="mlx",
        best_epoch=best_epoch,
        checkpoint_path=checkpoint,
        audit={
            "backend": "mlx",
            "best_inner_delta_vs_frozen_ar": best_key,
            "residual_suppressed": best_epoch == 0,
            "score_policy": (
                "continuous_output" if score_column == 0 else "frozen_ar_or_event_output"
            ),
            "checkpoint_sha256": checksum,
        },
    )
