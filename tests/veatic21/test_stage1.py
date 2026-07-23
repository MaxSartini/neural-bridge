from __future__ import annotations

import numpy as np

from neural_bridge.veatic21.stage1 import (
    CheckpointSelector,
    _causal_design,
    _optimizer_converged,
)


def test_checkpoint_one_can_win_on_merit_after_training_completes() -> None:
    selector = CheckpointSelector(minimum_epochs=50, plateau_patience=50)

    assert selector.observe(1, 0.25)
    for epoch in range(2, 52):
        assert not selector.observe(epoch, 0.25)

    assert selector.best_epoch == 1
    assert selector.should_stop(51, optimizer_converged=True)


def test_checkpoint_ties_keep_the_earliest_epoch() -> None:
    selector = CheckpointSelector()

    assert selector.observe(1, 0.1)
    assert not selector.observe(2, 0.1)
    assert selector.best_epoch == 1


def test_checkpoint_cannot_stop_before_minimum_or_without_convergence() -> None:
    selector = CheckpointSelector(minimum_epochs=50, plateau_patience=2)
    selector.observe(1, 0.1)
    selector.observe(2, 0.0)
    selector.observe(3, 0.0)

    assert not selector.should_stop(3, optimizer_converged=True)
    assert not selector.should_stop(50, optimizer_converged=False)
    assert selector.should_stop(50, optimizer_converged=True)


def test_optimizer_convergence_tolerates_minibatch_noise_but_not_active_improvement() -> None:
    noisy_plateau = [1.0 + ((index % 3) - 1) * 0.001 for index in range(20)]
    improving = list(np.linspace(1.0, 0.5, 20))

    assert _optimizer_converged(noisy_plateau)
    assert not _optimizer_converged(improving)


def test_causal_design_cannot_see_future_rows_or_cross_videos() -> None:
    projected = np.arange(24, dtype=np.float32).reshape(8, 3)
    video_id = np.asarray(["0"] * 4 + ["1"] * 4)
    row_index = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
    original = _causal_design(
        projected,
        video_id,
        row_index,
        family="frozen_ar_plus_gated_multiscale_temporal_residual",
        context_rows=(1, 2),
    )

    changed = projected.copy()
    changed[3] += 10_000
    mutated = _causal_design(
        changed,
        video_id,
        row_index,
        family="frozen_ar_plus_gated_multiscale_temporal_residual",
        context_rows=(1, 2),
    )

    np.testing.assert_array_equal(mutated[:3], original[:3])
    np.testing.assert_array_equal(mutated[4:], original[4:])
