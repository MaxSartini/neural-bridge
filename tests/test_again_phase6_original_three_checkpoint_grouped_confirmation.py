from backend.scripts import run_again_dense_2hz_phase6_original_three_checkpoint_grouped_confirmation as runner


def test_locked_grouped_scope_is_exactly_420_rows():
    assert len(runner.FOLDS) == 5
    assert len(runner.SEEDS) == 9
    assert runner.GROUPS == (
        (20260675, 20260676, 20260677),
        (20260678, 20260679, 20260680),
        (20260681, 20260682, 20260683),
    )
    assert runner.EXPECTED_MEMBER_ROWS == 5 * 9 * 7
    assert runner.EXPECTED_ENSEMBLE_ROWS == 5 * 3 * 7
    assert runner.EXPECTED_ROWS == 420


def test_grouped_confirmation_keeps_all_primary_controls():
    assert set(runner.PRIMARY_CONTROLS) == {
        "shuffled_pca_residual",
        "random_pca_residual",
        "label_permutation_residual",
        "train_only_video_mean_residual",
    }
    assert "real_residual" in runner.RESIDUAL_CONTROLS
    assert "diagnostics_only_residual" in runner.RESIDUAL_CONTROLS
