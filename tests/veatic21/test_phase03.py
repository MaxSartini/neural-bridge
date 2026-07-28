from __future__ import annotations

from neural_bridge.veatic21.phase03 import FAMILY_NAMES, PHASE03_CHECKS, phase04_authorized


def test_phase03_registers_complete_raw_and_applicable_control_families() -> None:
    assert FAMILY_NAMES == (
        "real_cortical",
        "shuffled_cortical",
        "shape_matched_random",
        "train_only_video_mean",
        "diagnostics_only",
        "time_video_time_only",
        "quality_motion_luma_only",
        "label_permutation_cortical",
    )


def test_phase04_authorization_requires_exact_complete_phase03_matrix() -> None:
    complete = dict.fromkeys(PHASE03_CHECKS, True)
    assert phase04_authorized(complete)
    complete[PHASE03_CHECKS[0]] = False
    assert not phase04_authorized(complete)
    assert not phase04_authorized(dict.fromkeys(PHASE03_CHECKS[:-1], True))
