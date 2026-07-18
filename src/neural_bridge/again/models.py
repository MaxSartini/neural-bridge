"""Target-specific frozen AR and deliberately simple residual sanity models."""

from __future__ import annotations

import importlib.util
import platform
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .contracts import FrozenArScores, Split
from .metrics import continuous_metrics, event_metrics

Backend = Literal["auto", "cpu", "cuda", "mlx"]
Architecture = Literal[
    "current_row_mlp_residual",
    "delta_feature_mlp_residual",
    "short_temporal_conv_residual",
    "low_ar_confidence_temporal_residual",
]
TargetType = Literal["event", "continuous"]


@dataclass(frozen=True)
class ResidualConfig:
    """One backend-neutral definition of the final AGAIN residual family."""

    backend: Backend = "auto"
    architecture: Architecture = "short_temporal_conv_residual"
    hidden: int = 64
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    alpha_initial_logit: float = -4.0
    alpha_cap: float = 0.12
    gate_bias: float = 4.0
    lambda_binary: float = 0.5
    batch_size: int = 8192
    max_epochs: int = 40
    patience: int = 8
    sequence_window: int = 0
    sequence_channels: int = 0


@dataclass(frozen=True)
class ResidualTrainingData:
    train_x: np.ndarray
    test_x: np.ndarray
    train_y: np.ndarray
    train_continuous: np.ndarray
    ar_train_score: np.ndarray
    ar_test_score: np.ndarray
    ar_train_continuous: np.ndarray
    ar_test_continuous: np.ndarray
    inner_train: np.ndarray
    inner_val: np.ndarray
    target_type: TargetType

    def validate(self) -> None:
        n_train, n_test = len(self.train_x), len(self.test_x)
        for name in (
            "train_y",
            "train_continuous",
            "ar_train_score",
            "ar_train_continuous",
        ):
            if len(getattr(self, name)) != n_train:
                raise ValueError(f"{name} does not match train_x")
        for name in ("ar_test_score", "ar_test_continuous"):
            if len(getattr(self, name)) != n_test:
                raise ValueError(f"{name} does not match test_x")
        if self.train_x.ndim != 2 or self.test_x.ndim != 2:
            raise ValueError("residual inputs must be matrices")
        if self.train_x.shape[1] != self.test_x.shape[1]:
            raise ValueError("residual train/test widths differ")
        if not set(self.inner_train).isdisjoint(set(self.inner_val)):
            raise ValueError("inner train/validation rows overlap")


@dataclass(frozen=True)
class ResidualResult:
    train_score: np.ndarray
    test_score: np.ndarray
    train_continuous: np.ndarray
    test_continuous: np.ndarray
    backend: Literal["cpu", "cuda", "mlx"]
    best_epoch: int
    checkpoint_path: Path | None
    audit: dict[str, object] = field(default_factory=dict)


def resolve_backend(requested: Backend = "auto") -> Literal["cpu", "cuda", "mlx"]:
    """Resolve an explicit portable backend without importing it at module import time."""

    torch_available = importlib.util.find_spec("torch") is not None
    mlx_available = importlib.util.find_spec("mlx") is not None
    if requested == "cuda":
        if not torch_available:
            raise RuntimeError("CUDA backend requires the optional 'training' dependencies")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return "cuda"
    if requested == "mlx":
        if platform.system() != "Darwin" or not mlx_available:
            raise RuntimeError("MLX was requested but is unavailable on this host")
        return "mlx"
    if requested == "cpu":
        if not torch_available:
            raise RuntimeError(
                "CPU learned-head training requires the optional 'training' dependencies"
            )
        return "cpu"
    if torch_available:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    if platform.system() == "Darwin" and mlx_available:
        return "mlx"
    if torch_available:
        return "cpu"
    raise RuntimeError("No learned-head backend is installed; install the 'training' extra")


def train_residual_head(
    data: ResidualTrainingData,
    config: ResidualConfig,
    *,
    seed: int,
    checkpoint_dir: Path,
) -> ResidualResult:
    """Train the same residual family on CPU, CUDA, or MLX."""

    data.validate()
    backend = resolve_backend(config.backend)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if backend == "mlx":
        from .mlx_backend import train

        return train(data, config, seed=seed, checkpoint_dir=checkpoint_dir)
    from .torch_backend import train

    return train(data, config, seed=seed, checkpoint_dir=checkpoint_dir, device=backend)


@dataclass(frozen=True)
class FittedRidge:
    scaler: StandardScaler
    model: Ridge
    alpha: float
    selection_strategy: str

    def predict(self, values: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(values)).astype(np.float32)


def _fit_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    inner_train: np.ndarray,
    inner_val: np.ndarray,
    *,
    task: str,
    alpha_grid: Sequence[float],
    selection_strategy: str,
) -> FittedRidge:
    best: tuple[float, float] | None = None
    for alpha in alpha_grid:
        scaler = StandardScaler().fit(train_x[inner_train])
        model = Ridge(alpha=alpha).fit(scaler.transform(train_x[inner_train]), train_y[inner_train])
        prediction = model.predict(scaler.transform(train_x[inner_val]))
        if task == "event":
            score = event_metrics(
                train_y[inner_train],
                model.predict(scaler.transform(train_x[inner_train])),
                train_y[inner_val],
                prediction,
            )["pr_auc"]
            key = float(score)
        else:
            key = -continuous_metrics(train_y[inner_val], prediction)["mae"]
        candidate = (key, -float(alpha))
        if best is None or candidate > best:
            best = candidate
            selected = float(alpha)
    scaler = StandardScaler().fit(train_x)
    model = Ridge(alpha=selected).fit(scaler.transform(train_x), train_y)
    return FittedRidge(scaler, model, selected, selection_strategy)


def fit_frozen_ar(
    ar_train: np.ndarray,
    ar_test: np.ndarray,
    split: Split,
    train_continuous: np.ndarray,
    *,
    inner_train: np.ndarray,
    inner_val: np.ndarray,
    selection_strategy: str,
    alpha_grid: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
) -> FrozenArScores:
    """Train target-specific AR once; downstream lanes receive only its frozen scores."""

    event = _fit_ridge(
        ar_train,
        split.train_y,
        inner_train,
        inner_val,
        task="event",
        alpha_grid=alpha_grid,
        selection_strategy=selection_strategy,
    )
    continuous = _fit_ridge(
        ar_train,
        train_continuous,
        inner_train,
        inner_val,
        task="continuous",
        alpha_grid=alpha_grid,
        selection_strategy=selection_strategy,
    )
    return FrozenArScores(
        train_score=event.predict(ar_train),
        test_score=event.predict(ar_test),
        train_continuous=continuous.predict(ar_train),
        test_continuous=continuous.predict(ar_test),
        provenance={
            "fit_scope": "outer_train_only",
            "event_alpha": event.alpha,
            "continuous_alpha": continuous.alpha,
            "selection_strategy": selection_strategy,
        },
    )


def fit_ridge_residual(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_target: np.ndarray,
    frozen_train: np.ndarray,
    frozen_test: np.ndarray,
    *,
    alpha: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Freshly trained sanity baseline; this is not the proposed learned AGAIN head."""

    scaler = StandardScaler().fit(train_x)
    model = Ridge(alpha=alpha).fit(scaler.transform(train_x), train_target - frozen_train)
    train = frozen_train + model.predict(scaler.transform(train_x))
    test = frozen_test + model.predict(scaler.transform(test_x))
    return train.astype(np.float32), test.astype(np.float32)
