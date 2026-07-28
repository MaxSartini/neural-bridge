from __future__ import annotations

import numpy as np

from neural_bridge.veatic21.phase04 import (
    PHASE04_CHECKS,
    TEMPORAL_DEPTHS,
    aggregate_causal_scores,
    phase05_authorized,
)


def test_temporal_candidates_are_veatic_label_landmarks() -> None:
    assert TEMPORAL_DEPTHS == (0, 4, 6)


def test_causal_score_aggregation_uses_only_current_and_past_same_video_rows() -> None:
    scores = np.arange(24, dtype=np.float32).reshape(12, 2)
    video_id = np.repeat(np.arange(2), 6)
    indices = np.asarray((2, 3, 8, 9))

    output = aggregate_causal_scores(scores, indices, video_id, depth=2)

    expected = np.asarray([scores[index - 2 : index + 1].mean(axis=0) for index in indices])
    assert np.array_equal(output, expected)


def test_phase05_authorization_requires_exact_complete_phase04_matrix() -> None:
    complete = dict.fromkeys(PHASE04_CHECKS, True)
    assert phase05_authorized(complete)
    complete[PHASE04_CHECKS[0]] = False
    assert not phase05_authorized(complete)
    assert not phase05_authorized(dict.fromkeys(PHASE04_CHECKS[:-1], True))
