from __future__ import annotations

import numpy as np
import pandas as pd

from neural_bridge.again import RunConfig, add_targets_and_ar_features, run_sanity_benchmark
from neural_bridge.again.data import build_splits, causal_history, causal_summary


def _rows(videos: int = 6, length: int = 40) -> pd.DataFrame:
    result = []
    for video in range(videos):
        for row in range(length):
            result.append(
                {
                    "video_id": f"v{video}",
                    "row_index": row,
                    "time_seconds": row * 0.5,
                    "label_available": True,
                    "arousal": (row % 8) / 8.0 + 0.02 * video,
                }
            )
    return add_targets_and_ar_features(pd.DataFrame(result))


def test_targets_are_future_labels_but_ar_features_are_past_only() -> None:
    rows = _rows(videos=1, length=10)
    assert np.isclose(
        rows.loc[0, "future_arousal_delta_p2rows"], rows.loc[2, "arousal"] - rows.loc[0, "arousal"]
    )
    assert np.isclose(rows.loc[4, "arousal_lag_4row"], rows.loc[0, "arousal"])
    assert not bool(rows.loc[0, "ar_context_available"])
    assert bool(rows.loc[4, "ar_context_available"])


def test_outer_protocols_keep_their_distinct_leakage_contracts() -> None:
    rows = _rows()
    splits = build_splits(rows, n_splits=3)
    assert {split.protocol for split in splits} == {"grouped_video", "blocked_temporal_70_30"}
    for split in splits:
        assert not np.intersect1d(split.train_idx, split.test_idx).size
        if split.protocol == "grouped_video":
            train_videos = set(rows.loc[split.train_idx, "video_id"])
            test_videos = set(rows.loc[split.test_idx, "video_id"])
            assert train_videos.isdisjoint(test_videos)
        else:
            for video in rows.video_id.unique():
                train = rows.loc[
                    np.intersect1d(split.train_idx, rows.index[rows.video_id == video]), "row_index"
                ]
                test = rows.loc[
                    np.intersect1d(split.test_idx, rows.index[rows.video_id == video]), "row_index"
                ]
                assert train.max() < test.min()


def test_causal_history_never_crosses_video_or_uses_future_rows() -> None:
    current = np.arange(12, dtype=np.float32).reshape(6, 2)
    row_index = np.array([0, 1, 2, 0, 1, 2])
    video_id = np.array(["a", "a", "a", "b", "b", "b"])
    sequence, available = causal_history(current, row_index, video_id, window_rows=3)
    summary = causal_summary(sequence, available)
    assert np.array_equal(sequence[3, -1], current[3])
    assert not available[3, :-1].any()
    assert summary.shape == (6, current.shape[1] * 4 + 1)


def test_one_runner_reuses_identical_ar_for_real_and_controls() -> None:
    rows = _rows()
    rng = np.random.default_rng(7)
    representation = rng.normal(size=(len(rows), 12)).astype(np.float32)
    config = RunConfig(
        protocols=("grouped_video",),
        n_splits=3,
        pca_width=6,
        seeds=(11,),
    )
    result = run_sanity_benchmark(rows, representation, config=config)
    assert not result.empty
    assert set(result.lane) == {"real", "shuffled", "random", "train_video_mean"}
    per_fold = result.groupby(["protocol", "fold"])["frozen_ar_sha256"].nunique()
    assert (per_fold == 1).all()
    assert result["target_threshold_train_only"].notna().all()
