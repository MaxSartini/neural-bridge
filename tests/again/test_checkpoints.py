from __future__ import annotations

from typing import Any, cast

import numpy as np

from neural_bridge.again.checkpoints import score_gated_ar, score_temporal_residual


def test_portable_ar_checkpoint_replay(tmp_path) -> None:
    checkpoint = tmp_path / "ar.npz"
    np.savez(
        checkpoint,
        **cast(
            Any,
            {
                "ar_proj.weight": np.zeros((2, 1), dtype=np.float32),
                "ar_proj.bias": np.zeros(2, dtype=np.float32),
                "pca_proj.weight": np.zeros((2, 1), dtype=np.float32),
                "pca_proj.bias": np.zeros(2, dtype=np.float32),
                "diag_proj.weight": np.zeros((2, 1), dtype=np.float32),
                "diag_proj.bias": np.zeros(2, dtype=np.float32),
                "gate.weight": np.zeros((2, 6), dtype=np.float32),
                "gate.bias": np.zeros(2, dtype=np.float32),
                "out.weight": np.zeros((2, 2), dtype=np.float32),
                "out.bias": np.array([1.0, 2.0], dtype=np.float32),
            },
        ),
    )

    event, continuous = score_gated_ar(np.ones((3, 1), dtype=np.float32), checkpoint)

    np.testing.assert_array_equal(event, np.full(3, 2.0, dtype=np.float32))
    np.testing.assert_array_equal(continuous, np.full(3, 1.0, dtype=np.float32))


def test_continuous_residual_replay_uses_continuous_score(tmp_path) -> None:
    checkpoint = tmp_path / "residual.npz"
    np.savez(
        checkpoint,
        **cast(
            Any,
            {
                "alpha": np.zeros(1, dtype=np.float32),
                "conv.weight": np.zeros((2, 3), dtype=np.float32),
                "conv.bias": np.zeros(2, dtype=np.float32),
                "post.weight": np.zeros((2, 2), dtype=np.float32),
                "post.bias": np.zeros(2, dtype=np.float32),
                "out.weight": np.zeros((2, 2), dtype=np.float32),
                "out.bias": np.array([2.0, 3.0], dtype=np.float32),
                "gate.weight": np.zeros((1, 3), dtype=np.float32),
                "gate.bias": np.zeros(1, dtype=np.float32),
                "conf_gate.weight": np.zeros((1, 3), dtype=np.float32),
                "conf_gate.bias": np.zeros(1, dtype=np.float32),
            },
        ),
    )
    values = np.ones((4, 3), dtype=np.float32)
    event_floor = np.full(4, -1.0, dtype=np.float32)
    continuous_floor = np.full(4, 0.5, dtype=np.float32)

    score, continuous, gate = score_temporal_residual(
        values,
        event_floor,
        continuous_floor,
        checkpoint,
        sequence_window=3,
        sequence_channels=1,
        target_type="continuous",
    )

    np.testing.assert_array_equal(score, continuous)
    np.testing.assert_allclose(gate, 1 / (1 + np.exp(4)), rtol=0, atol=1e-7)
