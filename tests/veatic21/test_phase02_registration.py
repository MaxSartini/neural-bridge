from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from neural_bridge.veatic21.phase02_registration import (
    _blocked_row_masks,
    balanced_group_folds,
    derive_protocol_counts,
    deterministic_seed,
    run_phase02_registration,
    verify_phase02_registration,
)


def test_protocol_counts_are_derived_from_veatic_support() -> None:
    assert derive_protocol_counts(124, 21) == {
        "minimum_test_videos": 12,
        "grouped_outer_folds": 10,
        "grouped_repeats": 4,
        "grouped_inner_folds": 4,
        "blocked_time_blocks": 4,
        "blocked_forward_folds": 2,
        "finalist_seeds": 5,
    }


def test_grouped_folds_are_complete_disjoint_and_balanced() -> None:
    video_ids = tuple(range(124))
    row_counts = {video: 22 + video for video in video_ids}
    folds = balanced_group_folds(video_ids, row_counts, fold_count=10, seed=123)

    flattened = [video for fold in folds for video in fold]
    assert sorted(flattened) == list(video_ids)
    assert len(flattened) == len(set(flattened))
    assert sorted(map(len, folds)) == [12] * 6 + [13] * 4


def test_grouped_seed_is_identity_derived_and_reproducible() -> None:
    assert deterministic_seed("a" * 64, "grouped", 0) == deterministic_seed(
        "a" * 64, "grouped", 0
    )
    assert deterministic_seed("a" * 64, "grouped", 0) != deterministic_seed(
        "a" * 64, "grouped", 1
    )


def test_blocked_masks_purge_training_targets_that_cross_boundaries() -> None:
    video_id = np.zeros(100, dtype=np.int16)
    row_index = np.arange(100, dtype=np.int32)
    masks = _blocked_row_masks(
        video_id,
        row_index,
        {0: 100},
        target_end=5,
        test_block=8,
        block_count=10,
    )

    assert np.flatnonzero(masks["inner_train"])[-1] == 64
    assert np.flatnonzero(masks["inner_validation"])[0] == 70
    assert np.flatnonzero(masks["inner_validation"])[-1] == 74
    assert np.flatnonzero(masks["outer_train"])[-1] == 74
    assert np.array_equal(np.flatnonzero(masks["outer_test"]), np.arange(80, 90))
    assert set(np.flatnonzero(masks["purged"])) >= set(range(65, 70))


def test_phase02_registration_refuses_noncanonical_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="canonical root"):
        run_phase02_registration(tmp_path)


def test_phase02_registration_verifier_refuses_missing_output(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_phase02_registration(tmp_path)


def test_registration_source_has_no_cortical_or_again_runtime_access() -> None:
    source = inspect.getsource(run_phase02_registration)
    assert "TRIBE_PER_VIDEO_ROOT" not in source
    assert "cortical_prediction" not in source
    assert "neural_bridge.again" not in source
