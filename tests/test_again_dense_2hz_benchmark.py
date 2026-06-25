import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.scripts.again_dense_2hz_benchmark import (
    add_dense_2hz_targets_and_ar_features,
    blocked_temporal_split,
    build_ar_feature_map,
    evaluate_lanes,
    grouped_video_splits,
    read_row_index,
    validate_split,
)


def test_dense_2hz_target_construction_uses_future_rows_without_future_features():
    rows = []
    for t in range(8):
        rows.append(
            {
                "video_id": "v1",
                "row_index": t,
                "time_seconds": t * 0.5,
                "label_available": True,
                "arousal": float(t) / 10.0,
                "valence": np.nan,
                "black_frame_fraction": 0.0,
                "duplicate_frame_fraction": 0.0,
                "quality_black_frame_flag": 0,
                "quality_duplicate_frame_flag": 0,
                "quality_exclusion_flag": 0,
                "quality_weight_suggested": 1.0,
                "motion_absdiff_mean": 0.1,
                "luma_mean": 100.0,
                "luma_std": 10.0,
                "frame_luma_std_mean": 20.0,
            }
        )
    df = add_dense_2hz_targets_and_ar_features(pd.DataFrame(rows))

    assert df.loc[0, "future_arousal_delta_p1rows"] == 0.1
    assert df.loc[0, "future_arousal_delta_p2rows"] == 0.2
    assert df.loc[4, "arousal_lag_4row"] == 0.0
    assert df.loc[0, "ar_context_available"] is False or not bool(df.loc[0, "ar_context_available"])
    assert bool(df.loc[4, "ar_context_available"])
    assert bool(df.loc[1, "target_mask_arousal_delta_p1rows"])
    assert not bool(df.loc[7, "target_mask_arousal_delta_p1rows"])


def test_grouped_video_split_keeps_groups_disjoint():
    df = pd.DataFrame(
        {
            "video_id": ["a"] * 5 + ["b"] * 5 + ["c"] * 5,
            "row_index": list(range(5)) * 3,
            "time_seconds": [0.5 * i for i in range(5)] * 3,
        }
    )
    mask = np.ones(len(df), dtype=bool)
    splits = grouped_video_splits(df, mask, n_splits=3)
    assert splits
    for protocol, _fold, train_idx, test_idx in splits:
        validate_split(df, protocol, train_idx, test_idx)


def test_blocked_temporal_split_keeps_time_order_inside_videos():
    df = pd.DataFrame(
        {
            "video_id": ["a"] * 10 + ["b"] * 10,
            "row_index": list(range(10)) * 2,
            "time_seconds": [0.5 * i for i in range(10)] * 2,
        }
    )
    split = blocked_temporal_split(df, np.ones(len(df), dtype=bool))[0]
    _protocol, _fold, train_idx, test_idx = split
    for video_id in df["video_id"].unique():
        train_times = df.loc[np.intersect1d(train_idx, df.index[df["video_id"] == video_id]), "time_seconds"]
        test_times = df.loc[np.intersect1d(test_idx, df.index[df["video_id"] == video_id]), "time_seconds"]
        assert train_times.max() < test_times.min()


def test_row_index_reader_accepts_parquet(tmp_path: Path):
    root = tmp_path / "dense"
    root.mkdir()
    (root / "per_video").mkdir()
    (root / "global_run_metadata.json").write_text(json.dumps({"cache_only": True, "forbid_vjepa": True}))
    pd.DataFrame(
        {
            "video_id": ["v1", "v1"],
            "row_index": [0, 1],
            "time_seconds": [0.0, 0.5],
            "clip_window_start_seconds": [0.0, 0.0],
            "clip_window_end_seconds": [0.0, 0.5],
        }
    ).to_parquet(root / "row_index.parquet", index=False)

    df = read_row_index(root)

    assert len(df) == 2
    assert list(df["time_seconds"]) == [0.0, 0.5]


def test_ar_smoke_benchmark_uses_train_only_thresholds():
    rows = []
    for video_index in range(4):
        for row_index in range(12):
            arousal = (row_index % 6) / 10.0 + video_index * 0.01
            rows.append(
                {
                    "video_id": f"v{video_index}",
                    "row_index": row_index,
                    "time_seconds": row_index * 0.5,
                    "label_available": True,
                    "arousal": arousal,
                    "valence": np.nan,
                    "black_frame_fraction": 0.0,
                    "duplicate_frame_fraction": 0.0,
                    "quality_black_frame_flag": 0,
                    "quality_duplicate_frame_flag": 0,
                    "quality_exclusion_flag": 0,
                    "quality_weight_suggested": 1.0,
                    "motion_absdiff_mean": 0.1,
                    "luma_mean": 100.0,
                    "luma_std": 10.0,
                    "frame_luma_std_mean": 20.0,
                }
            )
    df = add_dense_2hz_targets_and_ar_features(pd.DataFrame(rows))
    feature_map = build_ar_feature_map(df)
    fold_df, summary_df, gates = evaluate_lanes(df, feature_map, n_splits=2)

    assert not fold_df.empty
    assert set(fold_df["model_lane"]) == {"AR_only"}
    assert bool(fold_df["uses_train_only_transform"].all())
    assert "selected_ridge_alpha_train_only" in fold_df.columns
    assert "inner_validation_strategy" in fold_df.columns
    assert set(fold_df["ridge_backend"]).issubset({"mlx_primal_conjugate_gradient", "sklearn_ridge_lsqr_cpu_fallback"})
    assert "targets" in gates
    assert not summary_df.empty
