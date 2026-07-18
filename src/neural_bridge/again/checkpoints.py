"""Portable NumPy replay of the final AGAIN MLX checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.special import erf, expit

Weights = Mapping[str, np.ndarray]


def load_mlx_weights(path: Path) -> dict[str, np.ndarray]:
    """Load a named MLX ``.npz`` without requiring Apple hardware or MLX."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as bundle:
        return {name: np.asarray(bundle[name], dtype=np.float32) for name in bundle.files}


def _linear(values: np.ndarray, weights: Weights, name: str) -> np.ndarray:
    weight = weights[f"{name}.weight"]
    bias = weights[f"{name}.bias"]
    if values.shape[1] != weight.shape[1]:
        raise ValueError(f"{name} expects width {weight.shape[1]}, received {values.shape[1]}")
    return values @ weight.T + bias


def _gelu(values: np.ndarray) -> np.ndarray:
    return 0.5 * values * (1.0 + erf(values / np.sqrt(2.0)))


def score_gated_ar(
    values: np.ndarray,
    checkpoint: Path,
    *,
    batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay the eval-mode dual-output target-specific AR checkpoint."""

    weights = load_mlx_weights(checkpoint)
    required = {
        "ar_proj.weight",
        "ar_proj.bias",
        "pca_proj.weight",
        "pca_proj.bias",
        "diag_proj.weight",
        "diag_proj.bias",
        "gate.weight",
        "gate.bias",
        "out.weight",
        "out.bias",
    }
    if set(weights) != required:
        raise ValueError(f"unexpected AR checkpoint keys: {sorted(set(weights) ^ required)}")
    outputs: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        batch = np.asarray(values[start : start + batch_size], dtype=np.float32)
        zeros = np.zeros((len(batch), 1), dtype=np.float32)
        ar = _gelu(_linear(batch, weights, "ar_proj"))
        pca = _gelu(_linear(zeros, weights, "pca_proj"))
        diagnostics = _gelu(_linear(zeros, weights, "diag_proj"))
        gate = expit(_linear(np.concatenate([ar, pca, diagnostics], axis=1), weights, "gate"))
        fused = gate * (pca + diagnostics) + (1.0 - gate) * ar
        outputs.append(_linear(fused, weights, "out").astype(np.float32))
    output = np.concatenate(outputs)
    return output[:, 1], output[:, 0]


def score_temporal_residual(
    values: np.ndarray,
    ar_score: np.ndarray,
    ar_continuous: np.ndarray,
    checkpoint: Path,
    *,
    sequence_window: int,
    sequence_channels: int,
    alpha_cap: float = 0.12,
    gate_bias: float = 4.0,
    target_type: str,
    batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay the selected short-causal residual head in eval mode."""

    weights = load_mlx_weights(checkpoint)
    required = {
        "alpha",
        "conv.weight",
        "conv.bias",
        "post.weight",
        "post.bias",
        "out.weight",
        "out.bias",
        "gate.weight",
        "gate.bias",
        "conf_gate.weight",
        "conf_gate.bias",
    }
    if set(weights) != required:
        raise ValueError(f"unexpected residual checkpoint keys: {sorted(set(weights) ^ required)}")
    sequence_width = sequence_window * sequence_channels
    if values.shape[1] != weights["gate.weight"].shape[1]:
        raise ValueError("prepared representation does not match checkpoint input width")
    if sequence_width > values.shape[1]:
        raise ValueError("sequence dimensions exceed prepared representation width")
    alpha = float(expit(weights["alpha"])[0] * alpha_cap)
    scores: list[np.ndarray] = []
    continuous: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        batch = np.asarray(values[start : start + batch_size], dtype=np.float32)
        event_floor = np.asarray(ar_score[start : start + batch_size], dtype=np.float32)
        continuous_floor = np.asarray(ar_continuous[start : start + batch_size], dtype=np.float32)
        sequence = batch[:, :sequence_width].reshape(len(batch), sequence_window, sequence_channels)
        padded = np.concatenate(
            [np.zeros((len(batch), 2, sequence_channels), dtype=np.float32), sequence],
            axis=1,
        )
        final_window = padded[:, sequence_window - 1 : sequence_window + 2].reshape(
            len(batch), sequence_channels * 3
        )
        hidden = _gelu(_linear(final_window, weights, "conv"))
        extra = batch[:, sequence_width:]
        if extra.shape[1]:
            hidden = np.concatenate([hidden, extra], axis=1)
        hidden = _gelu(_linear(hidden, weights, "post"))
        residual = _linear(hidden, weights, "out")
        gate = expit(_linear(batch, weights, "gate") - gate_bias).reshape(-1)
        scale = alpha * gate
        continuous_output = continuous_floor + scale * residual[:, 0]
        event_output = event_floor + scale * residual[:, 1]
        scores.append(
            continuous_output.astype(np.float32)
            if target_type == "continuous"
            else event_output.astype(np.float32)
        )
        continuous.append(continuous_output.astype(np.float32))
        gates.append(gate.astype(np.float32))
    return np.concatenate(scores), np.concatenate(continuous), np.concatenate(gates)
