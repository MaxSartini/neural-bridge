"""PyTorch adapter for identical AGAIN residual training on CPU or CUDA."""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from typing import Literal

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
    device: Literal["cpu", "cuda"],
) -> ResidualResult:
    import torch
    from torch import nn
    from torch.nn import functional as F

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    class Head(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = data.train_x.shape[1]
            self.alpha = nn.Parameter(torch.tensor([config.alpha_initial_logit]))
            self.gate = nn.Linear(width, 1)
            self.conf_gate = nn.Linear(3, 1)
            if config.architecture == "short_temporal_conv_residual":
                sequence_width = config.sequence_window * config.sequence_channels
                if not sequence_width or sequence_width > width:
                    raise ValueError("short temporal head requires valid sequence dimensions")
                self.conv = nn.Linear(config.sequence_channels * 3, config.hidden)
                self.post = nn.Linear(config.hidden + width - sequence_width, config.hidden)
            else:
                self.layers = nn.ModuleList(
                    [nn.Linear(width, config.hidden), nn.Linear(config.hidden, config.hidden)]
                )
            self.out = nn.Linear(config.hidden, 2)

        def alpha_value(self):
            return torch.sigmoid(self.alpha) * config.alpha_cap

        def gate_value(self, x, ar_score):
            gate = torch.sigmoid(self.gate(x) - config.gate_bias)
            if config.architecture == "low_ar_confidence_temporal_residual":
                probability = torch.sigmoid(ar_score[:, None])
                confidence = torch.cat(
                    [ar_score[:, None], ar_score.abs()[:, None], probability * (1 - probability)],
                    dim=1,
                )
                gate = gate * torch.sigmoid(self.conf_gate(confidence))
            return gate

        def hidden(self, x):
            if config.architecture != "short_temporal_conv_residual":
                for layer in self.layers:
                    x = F.gelu(layer(x))
                return x
            sequence_width = config.sequence_window * config.sequence_channels
            sequence = x[:, :sequence_width].reshape(
                len(x), config.sequence_window, config.sequence_channels
            )
            extra = x[:, sequence_width:]
            padded = F.pad(sequence, (0, 0, 2, 0))
            hidden = None
            for position in range(config.sequence_window):
                window = padded[:, position : position + 3].reshape(
                    len(x), config.sequence_channels * 3
                )
                hidden = F.gelu(self.conv(window))
            assert hidden is not None
            if extra.shape[1]:
                hidden = torch.cat([hidden, extra], dim=1)
            return F.gelu(self.post(hidden))

        def forward(self, x, ar_score, ar_continuous):
            residual = self.out(self.hidden(x))
            scale = self.alpha_value() * self.gate_value(x, ar_score)
            continuous = ar_continuous[:, None] + scale * residual[:, :1]
            event = ar_score[:, None] + scale * residual[:, 1:]
            return torch.cat([continuous, event], dim=1)

    torch_device = torch.device(device)
    model = Head().to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    x = torch.as_tensor(data.train_x, dtype=torch.float32, device=torch_device)
    y = torch.as_tensor(data.train_y, dtype=torch.float32, device=torch_device)
    continuous = torch.as_tensor(data.train_continuous, dtype=torch.float32, device=torch_device)
    ar_score = torch.as_tensor(data.ar_train_score, dtype=torch.float32, device=torch_device)
    ar_continuous = torch.as_tensor(
        data.ar_train_continuous, dtype=torch.float32, device=torch_device
    )
    inner_train = np.asarray(data.inner_train)
    inner_val = np.asarray(data.inner_val)
    q80, q90 = np.quantile(data.train_continuous[inner_train], [0.8, 0.9])
    rng = np.random.default_rng(seed)
    best_key = 0.0
    best_epoch = 0
    best_state = None
    stale = 0

    def predict(values, event_floor, continuous_floor):
        model.eval()
        parts = []
        with torch.no_grad():
            for start in range(0, len(values), config.batch_size):
                output = model(
                    torch.as_tensor(
                        values[start : start + config.batch_size],
                        dtype=torch.float32,
                        device=torch_device,
                    ),
                    torch.as_tensor(
                        event_floor[start : start + config.batch_size],
                        dtype=torch.float32,
                        device=torch_device,
                    ),
                    torch.as_tensor(
                        continuous_floor[start : start + config.batch_size],
                        dtype=torch.float32,
                        device=torch_device,
                    ),
                )
                parts.append(output.cpu().numpy())
        return np.concatenate(parts)

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = rng.permutation(inner_train)
        for start in range(0, len(order), config.batch_size):
            idx = torch.as_tensor(order[start : start + config.batch_size], device=torch_device)
            output = model(x[idx], ar_score[idx], ar_continuous[idx])
            weights = torch.ones_like(y[idx])
            if data.target_type == "continuous":
                weights = (
                    1 + (continuous[idx] >= q80).float() + 2 * (continuous[idx] >= q90).float()
                )
            regression = (
                F.huber_loss(output[:, 0], continuous[idx], reduction="none") * weights
            ).mean()
            if data.target_type == "event":
                loss = regression + config.lambda_binary * F.binary_cross_entropy_with_logits(
                    output[:, 1], y[idx]
                )
            else:
                loss = regression + 0.03 * ((output[:, 0] - ar_continuous[idx]) ** 2).mean()
            loss = loss + 0.01 * model.alpha_value().square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
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
            best_key, best_epoch = float(value), epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break

    checkpoint = None
    if best_state is not None:
        model.load_state_dict(best_state)
        checkpoint = checkpoint_dir / f"residual__{config.architecture}__seed{seed}__{device}.pt"
        torch.save(best_state, checkpoint)
        train_output = predict(data.train_x, data.ar_train_score, data.ar_train_continuous)
        test_output = predict(data.test_x, data.ar_test_score, data.ar_test_continuous)
    else:
        train_output = np.column_stack([data.ar_train_continuous, data.ar_train_score])
        test_output = np.column_stack([data.ar_test_continuous, data.ar_test_score])
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None
    score_column = 0 if data.target_type == "continuous" and best_state is not None else 1
    return ResidualResult(
        train_score=train_output[:, score_column].astype(np.float32),
        test_score=test_output[:, score_column].astype(np.float32),
        train_continuous=train_output[:, 0].astype(np.float32),
        test_continuous=test_output[:, 0].astype(np.float32),
        backend=device,
        best_epoch=best_epoch,
        checkpoint_path=checkpoint,
        audit={
            "backend": device,
            "best_inner_delta_vs_frozen_ar": best_key,
            "residual_suppressed": best_state is None,
            "score_policy": (
                "continuous_output" if score_column == 0 else "frozen_ar_or_event_output"
            ),
            "checkpoint_sha256": checksum,
            "deterministic_algorithms_requested": True,
        },
    )
