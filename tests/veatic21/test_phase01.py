from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from neural_bridge.veatic21.phase01 import (
    ALLOWED_AUDIT_ARRAY_KEYS,
    FORBIDDEN_ARRAY_KEYS,
    Phase01Video,
    blocked_membership,
    derive_trajectory_family,
    load_phase01_video,
    shared_computation_contract,
)


def _synthetic_video(length: int = 20) -> Phase01Video:
    values = np.linspace(0.0, 1.0, length)
    return Phase01Video(
        video_id="0",
        row_index=np.arange(length, dtype=np.int64),
        time_seconds=np.arange(length, dtype=np.float64) / 2,
        arousal=values,
        valence=values[::-1],
        audit_arrays={},
    )


def test_trajectory_is_same_video_future_only() -> None:
    values = np.arange(12, dtype=np.float64)
    target = derive_trajectory_family(values, washout_rows=2, horizon_rows=3)
    assert target.eligible.tolist() == [True] * 7 + [False] * 5
    assert target.endpoint_delta[0] == 5
    assert target.max_positive_delta[0] == 5
    assert target.total_variation[0] == 3


def test_blocked_membership_uses_all_eligible_rows_in_forward_order() -> None:
    eligible = np.asarray(
        [False, True, True, True, True, True, True, True, True, True], dtype=np.bool_
    )
    membership = blocked_membership(eligible, outer_train_fraction=0.70, inner_train_fraction=0.80)
    train = np.flatnonzero(membership == 1)
    validation = np.flatnonzero(membership == 2)
    test = np.flatnonzero(membership == 3)
    assert len(train) + len(validation) + len(test) == int(np.sum(eligible))
    assert train[-1] < validation[0] < test[0]
    assert not np.any(membership[~eligible])


def test_allowlist_excludes_every_forbidden_array() -> None:
    assert FORBIDDEN_ARRAY_KEYS.isdisjoint(ALLOWED_AUDIT_ARRAY_KEYS)


def test_loader_reads_allowlist_and_checks_label_equality(tmp_path: Path) -> None:
    directory = tmp_path / "per_video/0"
    directory.mkdir(parents=True)
    length = 8
    times = np.arange(length, dtype=np.float64) / 2
    arousal = np.linspace(0.1, 0.8, length)
    valence = np.linspace(0.8, 0.1, length)
    with (directory / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("video_id", "row_index", "time_seconds", "arousal", "valence"),
        )
        writer.writeheader()
        for index in range(length):
            writer.writerow(
                {
                    "video_id": "0",
                    "row_index": index,
                    "time_seconds": times[index],
                    "arousal": arousal[index],
                    "valence": valence[index],
                }
            )
    arrays: dict[str, np.ndarray] = {}
    for key in ALLOWED_AUDIT_ARRAY_KEYS:
        if key == "arousal":
            value = arousal
        elif key == "valence":
            value = valence
        elif key == "time_seconds":
            value = times
        elif key in {"sample_frame_indices", "sample_time_seconds"}:
            value = np.zeros((length, 4))
        else:
            value = np.zeros(length)
        arrays[key] = np.asarray(value)
    arrays["cortical_prediction"] = np.full((length, 3), np.nan)
    np.savez(
        directory / "tribe_v2_cortical_predictions.npz",
        **arrays,  # ty: ignore[invalid-argument-type]
    )

    video = load_phase01_video("0", bundle_root=tmp_path)
    assert video.row_count == length
    assert set(video.audit_arrays) == set(ALLOWED_AUDIT_ARRAY_KEYS)
    assert "cortical_prediction" not in video.audit_arrays


def test_combined_and_zero_label_are_gated_after_specialists() -> None:
    contract = shared_computation_contract()
    assert "independently confirmed" in contract["combined_challenger_gate"]
    assert "combined challenger" in contract["zero_label_gate"]


def test_synthetic_video_fixture_is_well_formed() -> None:
    video = _synthetic_video()
    assert video.row_count == 20
    assert video.time_seconds[-1] == 9.5
