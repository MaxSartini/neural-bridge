import numpy as np
import pytest

from backend.scripts.again_native_temporal_alignment import (
    TribeTimeGrid,
    align_arousal_to_tribe_grid,
    build_native_grid_manifest,
    build_seconds_based_future_targets,
    infer_tribe_time_grid,
    require_benchmarkable_timing,
)


def test_seconds_based_future_windows_on_non_1hz_grid():
    annotation_times = np.arange(0.0, 6.1, 0.5)
    arousal = np.zeros_like(annotation_times)
    arousal[np.isclose(annotation_times, 2.5)] = 0.8
    centers = np.array([0.0, 0.5, 1.0, 1.5])
    current = np.zeros_like(centers)

    targets = build_seconds_based_future_targets(
        annotation_times,
        arousal,
        centers,
        current,
        spike_threshold=0.5,
        change_threshold=0.5,
    )

    assert targets["future_spike_1_3s"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert targets["future_spike_1_3s_feasible"].all()


def test_future_targets_do_not_use_row_offsets():
    annotation_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    arousal = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    centers = np.array([0.0, 0.25, 0.5])
    current = np.zeros_like(centers)

    targets = build_seconds_based_future_targets(
        annotation_times,
        arousal,
        centers,
        current,
        spike_threshold=0.5,
        change_threshold=0.5,
    )

    assert targets["future_spike_1_3s"].tolist() == [1.0, 1.0, 1.0]
    assert targets["target_window_units"][0] == "seconds"


def test_interpolation_aligns_without_future_feature_flag():
    annotation_times = [0.0, 1.0, 2.0]
    arousal = [0.0, 1.0, 0.0]
    centers = [0.5, 1.5]

    aligned = align_arousal_to_tribe_grid(annotation_times, arousal, centers, method="linear")

    assert np.allclose(aligned["arousal"], [0.5, 0.5])
    assert aligned["valid"].tolist() == [True, True]
    assert aligned["uses_future_labels_as_features"] is False


def test_target_feasibility_near_aligned_context_end():
    annotation_times = np.arange(0.0, 10.1, 1.0)
    arousal = np.linspace(0.0, 1.0, len(annotation_times))
    centers = np.array([5.0, 7.0, 8.0])
    current = np.interp(centers, annotation_times, arousal)

    targets = build_seconds_based_future_targets(
        annotation_times,
        arousal,
        centers,
        current,
        spike_threshold=0.05,
        change_threshold=0.05,
        aligned_context_end_seconds=9.0,
    )

    assert targets["future_spike_1_3s_feasible"].tolist() == [True, False, False]
    assert targets["future_change_p3s_feasible"].tolist() == [True, False, False]


def test_inferred_timing_confidence_recorded_for_native_grid_manifest():
    rows, grid = build_native_grid_manifest(
        dataset_name="AGAIN_cleaned",
        video_id="v1",
        video_path="/tmp/v1.webm",
        prediction_shape=(4, 8),
        aligned_video_duration=8.0,
        annotation_times=[0.0, 2.0, 4.0, 6.0, 8.0],
        arousal_values=[0.0, 0.1, 0.2, 0.3, 0.4],
        alignment_policy="drop_last_3s_video_keep_annotation_start",
    )

    assert grid.timing_source == "inferred_duration_even_spacing"
    assert grid.timing_confidence == "medium"
    assert rows[0]["timing_source"] == "inferred_duration_even_spacing"
    assert rows[0]["timing_confidence"] == "medium"


def test_post_roll_context_rows_are_marked_dropped():
    rows, _grid = build_native_grid_manifest(
        dataset_name="AGAIN_cleaned",
        video_id="v1",
        video_path="/tmp/v1.webm",
        prediction_shape=(5, 8),
        aligned_video_duration=10.0,
        annotation_times=[0.0, 2.0, 4.0, 6.0],
        arousal_values=[0.0, 0.2, 0.4, 0.6],
        alignment_policy="drop_last_3s_video_keep_annotation_start",
        aligned_context_end_seconds=6.0,
    )

    assert rows[-1]["tribe_time_center_seconds"] > 6.0
    assert rows[-1]["dropped_by_alignment"] is True
    assert rows[-1]["future_spike_1_3s_feasible"] is False


def test_require_benchmarkable_timing_rejects_unknown():
    grid = TribeTimeGrid(
        row_start_times=np.array([0.0]),
        row_end_times=np.array([1.0]),
        row_center_times=np.array([0.5]),
        timing_source="unknown",
        timing_confidence="low",
        seconds_per_prediction=None,
        prediction_rate_hz=None,
    )

    with pytest.raises(ValueError, match="unknown TRIBE timing"):
        require_benchmarkable_timing(grid)


def test_model_tr_grid_is_not_forced_to_1hz():
    grid = infer_tribe_time_grid(
        prediction_shape=(4, 16),
        aligned_video_duration=2.0,
        model_tr=0.5,
    )

    assert np.allclose(grid.row_center_times, [0.25, 0.75, 1.25, 1.75])
    assert grid.prediction_rate_hz == 2.0
    assert grid.timing_source == "model_tr"
