from __future__ import annotations

import numpy as np

from neural_bridge.veatic21.innovation import _strict_prior_center


def test_strict_prior_center_is_causal_and_video_bounded() -> None:
    values = np.asarray([[1.0], [3.0], [5.0], [10.0], [14.0]], dtype=np.float32)
    video_id = np.asarray(["1", "1", "1", "2", "2"])
    row_index = np.asarray([0, 1, 2, 0, 1])

    centered, available = _strict_prior_center(values, video_id, row_index)

    assert np.allclose(centered[:, 0], [1.0, 2.0, 3.0, 10.0, 4.0])
    assert np.array_equal(available[:, 0], [0.0, 1.0, 1.0, 0.0, 1.0])


def test_future_values_do_not_change_earlier_centered_rows() -> None:
    values = np.asarray([[2.0], [4.0], [8.0]], dtype=np.float32)
    video_id = np.asarray(["1", "1", "1"])
    row_index = np.asarray([0, 1, 2])
    original, _ = _strict_prior_center(values, video_id, row_index)
    changed = values.copy()
    changed[2] = 1_000.0

    replayed, _ = _strict_prior_center(changed, video_id, row_index)

    assert np.array_equal(original[:2], replayed[:2])
