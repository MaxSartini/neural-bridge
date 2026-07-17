from __future__ import annotations

import copy

import numpy as np
import pytest

from backend.scripts import veatic21_targets as targets


def _video_rows(
    video_id: str,
    arousal: list[float],
    valence: list[float],
    *,
    split: str,
    ticks: list[int] | None = None,
) -> list[dict[str, object]]:
    assert len(arousal) == len(valence)
    resolved_ticks = ticks if ticks is not None else list(range(len(arousal)))
    assert len(resolved_ticks) == len(arousal)
    return [
        {
            "schema_version": "veatic_temporal_window_v1",
            "dataset": "veatic",
            "stimulus_id": f"{video_id}:{tick:06d}",
            "video_id": video_id,
            "frame_index": tick,
            "time_start_seconds": tick / 2.0,
            "time_end_seconds": (tick + 1) / 2.0,
            "sampling_frequency_hz": 2.0,
            "split": split,
            "splits": {
                "official_70_30": split,
                "blocked_temporal_gap": split,
                "leave_video_out_group": video_id,
            },
            "targets": {
                "arousal": arousal[index],
                "valence": valence[index],
                "native_marker": f"keep-{video_id}-{tick}",
            },
            "media_path": f"videos/{video_id}.mp4",
        }
        for index, tick in enumerate(resolved_ticks)
    ]


def _mask_name(target_name: str) -> str:
    return f"target_mask_{target_name}"


def test_dense_rows_4_10_washout_targets_preserve_native_annotations() -> None:
    arousal = [0.0, 99.0, 99.0, 99.0, 0.1, 0.3, 0.2, 0.8, 0.4, 0.5, 0.6]
    valence = [0.1, 99.0, 99.0, 99.0, 0.5, -0.6, 0.2, 0.0, 0.3, -0.1, 0.4]
    source_rows = _video_rows("video-a", arousal, valence, split="train")
    source_snapshot = copy.deepcopy(source_rows)

    result = targets.build_veatic21_targets(source_rows)
    first = result.rows[0]

    assert source_rows == source_snapshot
    assert first["targets"]["arousal"] == 0.0
    assert first["targets"]["valence"] == 0.1
    assert first["targets"]["native_marker"] == "keep-video-a-0"
    assert first["targets"][targets.AROUSAL_FUTURE_MAX_DELTA] == pytest.approx(0.8)
    assert first["targets"][targets.VALENCE_FUTURE_SIGNED_RISE] == pytest.approx(0.4)
    assert first["targets"][targets.VALENCE_FUTURE_SIGNED_DROP] == pytest.approx(-0.7)
    assert first["targets"][targets.VALENCE_FUTURE_RISE_MAGNITUDE] == pytest.approx(0.4)
    assert first["targets"][targets.VALENCE_FUTURE_DROP_MAGNITUDE] == pytest.approx(0.7)
    assert first["targets"][targets.VALENCE_FUTURE_MAX_ABS_MOVEMENT] == pytest.approx(0.7)

    assert result.contract["future_row_offsets"] == [4, 5, 6, 7, 8, 9, 10]
    assert result.contract["future_second_offsets"] == [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    assert result.contract["washout_seconds"] == [0.5, 1.0, 1.5]
    assert result.contract["again_timing_equivalence"] == "rows_4_10_at_2hz"


def test_end_rows_missing_future_points_and_video_boundaries_are_masked() -> None:
    first_video = _video_rows("video-a", [0.0] * 11, [0.0] * 11, split="train")
    second_video = _video_rows("video-b", [100.0] * 11, [100.0] * 11, split="train")
    result = targets.build_veatic21_targets(first_video + second_video)

    assert result.rows[0]["target_masks"][_mask_name(targets.AROUSAL_FUTURE_MAX_DELTA)] is True
    for row in result.rows[1:11]:
        assert row["targets"][targets.AROUSAL_FUTURE_MAX_DELTA] is None
        assert row["target_masks"][_mask_name(targets.AROUSAL_FUTURE_MAX_DELTA)] is False

    missing_tick_rows = _video_rows(
        "video-gap",
        [0.0] * 10,
        [0.0] * 10,
        split="train",
        ticks=[0, 1, 2, 3, 4, 5, 6, 8, 9, 10],
    )
    gap_result = targets.build_veatic21_targets(missing_tick_rows + second_video)
    assert gap_result.rows[0]["targets"][targets.AROUSAL_FUTURE_MAX_DELTA] is None
    assert gap_result.rows[0]["target_masks"][_mask_name(targets.AROUSAL_FUTURE_MAX_DELTA)] is False


def test_event_thresholds_are_fit_from_training_rows_only() -> None:
    train_arousal = [float((tick * 3) % 7) / 10.0 for tick in range(24)]
    train_valence = [float((tick * 5) % 11) / 10.0 - 0.5 for tick in range(24)]
    eval_arousal = [float(tick * tick) for tick in range(24)]
    eval_valence = [float((-1) ** tick * tick) for tick in range(24)]
    train_rows = _video_rows("train-video", train_arousal, train_valence, split="train")
    eval_rows = _video_rows("eval-video", eval_arousal, eval_valence, split="test")

    first_result = targets.build_veatic21_targets(train_rows + eval_rows)
    extreme_eval_rows = copy.deepcopy(eval_rows)
    for row in extreme_eval_rows:
        row["targets"]["arousal"] *= 1_000_000.0
        row["targets"]["valence"] *= 1_000_000.0
    second_result = targets.build_veatic21_targets(train_rows + extreme_eval_rows)

    for event_name in targets.EVENT_TARGET_NAMES:
        first_threshold = first_result.contract["event_thresholds"][event_name]
        second_threshold = second_result.contract["event_thresholds"][event_name]
        assert first_threshold["threshold"] == pytest.approx(second_threshold["threshold"])
        assert first_threshold["fit_partition"] == {
            "source": "splits.blocked_temporal_gap",
            "train_value": "train",
        }

    source_name = targets.AROUSAL_FUTURE_MAX_DELTA
    train_scores = [
        max(float(row["targets"][source_name]), 0.0)
        for row in first_result.rows[: len(train_rows)]
        if row["target_masks"][_mask_name(source_name)]
    ]
    expected = float(np.quantile(np.asarray(train_scores), 0.90))
    actual = first_result.contract["event_thresholds"][targets.AROUSAL_FUTURE_MAX_DELTA_EVENT]
    assert actual["threshold"] == pytest.approx(expected)
    assert actual["fit_row_count"] == len(train_scores)


def test_valence_drop_event_uses_magnitude_without_destroying_signed_target() -> None:
    arousal = [0.0] * 11
    valence = [0.5, 9.0, 9.0, 9.0, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2]
    result = targets.build_veatic21_targets(_video_rows("video-a", arousal, valence, split="train"))
    first = result.rows[0]

    assert first["targets"][targets.VALENCE_FUTURE_SIGNED_RISE] == pytest.approx(-0.1)
    assert first["targets"][targets.VALENCE_FUTURE_SIGNED_DROP] == pytest.approx(-0.7)
    assert first["targets"][targets.VALENCE_FUTURE_RISE_MAGNITUDE] == pytest.approx(0.0)
    assert first["targets"][targets.VALENCE_FUTURE_DROP_MAGNITUDE] == pytest.approx(0.7)
    assert first["targets"][targets.VALENCE_FUTURE_MAX_ABS_MOVEMENT] == pytest.approx(0.7)
    drop_contract = result.contract["event_thresholds"][targets.VALENCE_FUTURE_DROP_EVENT]
    rise_contract = result.contract["event_thresholds"][targets.VALENCE_FUTURE_RISE_EVENT]
    assert drop_contract["score_transform"] == "negative_delta_magnitude"
    assert drop_contract["threshold"] == pytest.approx(0.7)
    assert rise_contract["score_transform"] == "positive_delta"
    assert rise_contract["threshold"] == pytest.approx(0.0)
    assert first["targets"][targets.VALENCE_FUTURE_DROP_EVENT] == 1


def test_contract_rejects_non_2hz_rows_and_supports_explicit_fold_train_mask() -> None:
    rows = _video_rows("video-a", [0.0] * 11, [0.0] * 11, split="test")
    with pytest.raises(ValueError, match="no valid training rows"):
        targets.build_veatic21_targets(rows)

    result = targets.build_veatic21_targets(rows, train_mask=[True] * len(rows))
    threshold = result.contract["event_thresholds"][targets.AROUSAL_FUTURE_MAX_DELTA_EVENT]
    assert threshold["fit_partition"] == {"source": "explicit_train_mask"}

    off_grid = copy.deepcopy(rows)
    off_grid[0]["time_start_seconds"] = 0.25
    with pytest.raises(ValueError, match="not on the exact 2 Hz grid"):
        targets.build_veatic21_targets(off_grid, train_mask=[True] * len(off_grid))


def test_continuous_only_mode_defers_event_thresholds_to_outer_fold() -> None:
    rows = _video_rows("video-a", [0.0] * 11, [0.0] * 11, split="test")
    result = targets.build_veatic21_targets(rows, build_events=False)

    assert result.contract["event_targets_built"] is False
    assert result.contract["event_threshold_fit_scope"] == "deferred_to_outer_training_fold"
    assert result.contract["event_thresholds"] == {}
    assert set(result.contract["target_contracts"]) == set(targets.CONTINUOUS_TARGET_NAMES)
    for row in result.rows:
        assert targets.AROUSAL_FUTURE_MAX_DELTA_EVENT not in row["targets"]
