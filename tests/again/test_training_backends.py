from __future__ import annotations

import importlib.util
import platform
from typing import cast

import numpy as np
import pytest

from neural_bridge.again.models import (
    Backend,
    ResidualConfig,
    ResidualTrainingData,
    train_residual_head,
)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            "cpu",
            marks=pytest.mark.skipif(
                importlib.util.find_spec("torch") is None, reason="training extra not installed"
            ),
        ),
        pytest.param(
            "mlx",
            marks=pytest.mark.skipif(
                platform.system() != "Darwin" or importlib.util.find_spec("mlx") is None,
                reason="MLX is available only with the training extra on macOS",
            ),
        ),
    ],
)
def test_portable_backend_smoke(backend: str, tmp_path) -> None:
    rng = np.random.default_rng(9)
    train_rows, test_rows = 48, 16
    data = ResidualTrainingData(
        train_x=rng.normal(size=(train_rows, 4)).astype(np.float32),
        test_x=rng.normal(size=(test_rows, 4)).astype(np.float32),
        train_y=(np.arange(train_rows) % 2).astype(np.float32),
        train_continuous=rng.normal(size=train_rows).astype(np.float32),
        ar_train_score=rng.normal(size=train_rows).astype(np.float32),
        ar_test_score=rng.normal(size=test_rows).astype(np.float32),
        ar_train_continuous=rng.normal(size=train_rows).astype(np.float32),
        ar_test_continuous=rng.normal(size=test_rows).astype(np.float32),
        inner_train=np.arange(32),
        inner_val=np.arange(32, 48),
        target_type="event",
    )
    config = ResidualConfig(
        backend=cast(Backend, backend),
        architecture="current_row_mlp_residual",
        hidden=8,
        batch_size=16,
        max_epochs=1,
        patience=1,
    )

    result = train_residual_head(data, config, seed=7, checkpoint_dir=tmp_path)

    assert result.backend == backend
    assert result.test_score.shape == (test_rows,)
    assert result.test_continuous.shape == (test_rows,)
