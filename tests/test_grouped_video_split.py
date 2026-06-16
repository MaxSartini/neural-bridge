from backend.scripts.run_veatic_temporal_fairness_benchmark import grouped_video_folds


def test_grouped_video_folds_keep_videos_disjoint():
    rows = []
    for video_id in range(10):
        for t in range(3):
            rows.append({"video_id": str(video_id), "time_start_seconds": float(t)})
    folds = grouped_video_folds(rows, 5)
    assert len(folds) == 5
    for _fold, held, train_rows, test_rows in folds:
        train_ids = {row["video_id"] for row in train_rows}
        test_ids = {row["video_id"] for row in test_rows}
        assert test_ids == set(held)
        assert train_ids.isdisjoint(test_ids)
