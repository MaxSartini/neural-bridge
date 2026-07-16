from __future__ import annotations

import numpy as np

from backend.app.services.mlx_tribe_encoder import MlxTribeEncoder
from backend.app.services.mlx_vjepa21_cortical import MlxVjepa21Encoder
from tools.run_single_video_neural_bridge_mlx import (
    causal_trailing_mean_2s,
    compact_temporal_diagnostics,
    diagnostic_matrix,
)
from tools.benchmark_veatic_vjepa21_only_one_video import (
    exact_2hz_time_seconds,
    sample_plan,
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


def test_vjepa_compact_temporal_diagnostics_match_legacy_reduction() -> None:
    import mlx.core as mx

    rng = np.random.default_rng(20260715)
    temporal_std = rng.random((2, 20, 32, 17), dtype=np.float32)
    compact_mlx = MlxVjepa21Encoder._compact_temporal_diagnostics(mx.array(temporal_std))
    mx.eval(compact_mlx)
    cached_std = temporal_std.astype(np.float16).astype(np.float32)
    legacy = np.concatenate(
        [
            cached_std.mean(axis=(1, 2, 3), dtype=np.float32)[:, None],
            cached_std.mean(axis=(2, 3), dtype=np.float32).astype(np.float16).astype(np.float32),
            cached_std.mean(axis=3, dtype=np.float32)
            .astype(np.float16)
            .astype(np.float32)
            .mean(axis=1, dtype=np.float32),
        ],
        axis=1,
    )

    assert np.asarray(compact_mlx).shape == (2, 53)
    np.testing.assert_allclose(np.asarray(compact_mlx), legacy, rtol=1e-6, atol=1e-6)


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


def test_veatic_vjepa_only_probe_uses_exact_again_2hz_grid() -> None:
    times = exact_2hz_time_seconds(10.56)
    sample_times, indices = sample_plan(times, decoded_frame_count=169)

    np.testing.assert_allclose(times, np.arange(22, dtype=np.float32) / 2.0)
    assert sample_times.shape == (22, 64)
    assert indices.shape == (22, 64)
    np.testing.assert_array_equal(indices[0], 0)
    assert float(sample_times[-1, 0]) == 6.5625
    assert float(sample_times[-1, -1]) == 10.5
