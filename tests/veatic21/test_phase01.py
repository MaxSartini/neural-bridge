from __future__ import annotations

import numpy as np
import pytest

from neural_bridge.veatic21.phase01 import (
    PHASE01_CHECKS,
    _pacf_ols,
    derive_washout_starts,
    future_max_increase,
    phase02_authorized,
    select_initial_no_washout,
)


def test_future_max_increase_uses_only_inclusive_registered_future_window() -> None:
    values, valid = future_max_increase(np.asarray((0.0, 1.0, 0.0, 3.0)), 1, 2)

    assert valid.tolist() == [True, True, False, False]
    assert values[:2].tolist() == [1.0, 2.0]
    assert np.isnan(values[2:]).all()


def test_future_target_rejects_current_or_reversed_offsets() -> None:
    with pytest.raises(ValueError, match="1 <= start <= end"):
        future_max_increase(np.asarray((0.0, 1.0)), 0, 1)
    with pytest.raises(ValueError, match="1 <= start <= end"):
        future_max_increase(np.asarray((0.0, 1.0)), 2, 1)


def test_initial_selection_uses_smallest_candidate_that_clears_every_frozen_gate() -> None:
    candidates = [
        {
            "end_row": 1,
            "coverage_fraction": 0.99,
            "median_per_video_acf": 0.95,
            "support_video_fraction": 0.95,
        },
        {
            "end_row": 2,
            "coverage_fraction": 0.95,
            "median_per_video_acf": 0.89,
            "support_video_fraction": 0.85,
        },
        {
            "end_row": 3,
            "coverage_fraction": 0.94,
            "median_per_video_acf": 0.80,
            "support_video_fraction": 0.90,
        },
    ]

    assert select_initial_no_washout(candidates)["end_row"] == 2


def test_initial_selection_fails_instead_of_weakening_frozen_gate() -> None:
    with pytest.raises(ValueError, match="no no-washout candidate"):
        select_initial_no_washout(
            [
                {
                    "end_row": 1,
                    "coverage_fraction": 0.89,
                    "median_per_video_acf": 0.89,
                    "support_video_fraction": 0.90,
                }
            ]
        )


def test_washout_family_is_derived_from_veatic_landmarks_and_deduplicated() -> None:
    assert derive_washout_starts(
        pacf_decay_lag=4,
        rise_duration_q90_rows=5.0,
        event_duration_median_rows=4.0,
    ) == (5, 6)


def test_pacf_is_undefined_only_when_a_video_lacks_estimation_degrees_of_freedom() -> None:
    values = np.asarray((0.0, 0.8, 0.1, 0.7, -0.2, 0.4, 0.3, 0.9))

    assert _pacf_ols(values, 3) is not None
    assert _pacf_ols(values, 4) is None


def test_phase02_authorization_requires_exact_complete_check_matrix() -> None:
    complete = dict.fromkeys(PHASE01_CHECKS, True)
    assert phase02_authorized(complete)
    complete[PHASE01_CHECKS[0]] = False
    assert not phase02_authorized(complete)
    incomplete = dict.fromkeys(PHASE01_CHECKS[:-1], True)
    assert not phase02_authorized(incomplete)
