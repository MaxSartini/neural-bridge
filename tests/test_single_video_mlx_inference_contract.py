from __future__ import annotations

import numpy as np

from backend.app.services.mlx_tribe_encoder import MlxTribeEncoder
from backend.app.services.mlx_vjepa21_cortical import MlxVjepa21Encoder
from tools.run_single_video_neural_bridge_mlx import (
    causal_trailing_mean_2s,
    compact_temporal_diagnostics,
    diagnostic_matrix,
)


def test_mlx_tribe_can_preserve_input_time_grid() -> None:
    import mlx.core as mx

    encoder = object.__new__(MlxTribeEncoder)
    encoder.config = {"depth": 0, "n_output_timesteps": 2}
    encoder.weights = {
        "time_pos_embed": mx.zeros((1, 4, 1)),
        "encoder.final_norm.g": mx.ones((1,)),
        "low_rank_head.weight": mx.ones((1, 1)),
        "predictor.weights": mx.ones((1, 1, 1)),
        "predictor.bias": mx.zeros((1, 1)),
    }
    encoder._aggregate_features = lambda _features: mx.arange(4, dtype=mx.float32).reshape(1, 4, 1)
    encoder._scale_norm = lambda value, _weight: value
    encoder._linear = lambda value, _weight: value

    unpooled = encoder.predict({"video": np.zeros((1, 1, 1, 4), dtype=np.float32)}, pool_outputs=False)
    pooled = encoder.predict({"video": np.zeros((1, 1, 1, 4), dtype=np.float32)})

    assert unpooled.shape == (1, 1, 4)
    assert pooled.shape == (1, 1, 2)


def test_vjepa_temporal_summary_uses_population_spatial_std() -> None:
    import mlx.core as mx

    state = mx.array([[[1.0], [3.0], [2.0], [6.0]]])
    token_mean, temporal_mean, temporal_std = MlxVjepa21Encoder._state_summaries(
        state,
        batch=1,
        temporal=2,
        spatial_tokens=2,
    )
    mx.eval(token_mean, temporal_mean, temporal_std)

    np.testing.assert_allclose(np.asarray(temporal_mean), [[[2.0], [4.0]]])
    np.testing.assert_allclose(np.asarray(temporal_std), [[[1.0], [2.0]]])
    np.testing.assert_allclose(np.asarray(token_mean), [[3.0]])


def test_single_video_diagnostic_and_causal_shapes_match_frozen_head() -> None:
    temporal_std = np.ones((3, 20, 32, 1408), dtype=np.float32)
    compact = compact_temporal_diagnostics(temporal_std)
    diagnostics = diagnostic_matrix(compact)
    cortical = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
    trailing = causal_trailing_mean_2s(cortical)

    assert diagnostics.shape == (3, 53)
    np.testing.assert_allclose(diagnostics, 1.0)
    np.testing.assert_allclose(trailing[0], cortical[0])
    np.testing.assert_allclose(trailing[4], cortical[1:5].mean(axis=0))
