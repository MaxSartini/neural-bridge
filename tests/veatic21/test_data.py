from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest

from neural_bridge.veatic21.contracts import ROWS_CSV_COLUMNS
from neural_bridge.veatic21.data import (
    allowlisted_tree_identity,
    read_row_identity,
    validate_cortical_array,
    validate_quality_arrays,
    validate_row_count_identity,
    validate_video_inventory,
    verify_allowed_payload_record,
)


def _row(
    row_index: int,
    *,
    time_seconds: float | None = None,
    arousal: str = "not-read",
    valence: str = "also-not-read",
) -> list[str]:
    time_seconds = row_index * 0.5 if time_seconds is None else time_seconds
    source_position = time_seconds * 25
    source_floor = int(np.floor(source_position))
    source_ceil = int(np.ceil(source_position))
    source_alpha = source_position - source_floor
    values = {
        "video_id": "0",
        "video_name": "0.mp4",
        "video_relpath": "videos/0.mp4",
        "arousal_relpath": "rating_averaged/0_arousal.csv",
        "valence_relpath": "rating_averaged/0_valence.csv",
        "row_index": str(row_index),
        "time_seconds": str(time_seconds),
        "row_hz": "2",
        "clip_start_seconds": "0",
        "clip_end_seconds": str(time_seconds),
        "native_label_fps": "25",
        "native_label_frame_count": "100",
        "source_frame_position": str(source_position),
        "source_floor_frame_index": str(source_floor),
        "source_ceil_frame_index": str(source_ceil),
        "source_interp_alpha": str(source_alpha),
        "source_arousal": arousal,
        "source_valence": valence,
        "source_match_quality": ("native_exact" if source_alpha == 0 else "linear_native_frames"),
        "encode_policy": "exact_2hz_native_label_support_no_extrapolation",
        "arousal": arousal,
        "valence": valence,
    }
    return [values[column] for column in ROWS_CSV_COLUMNS]


def _write_rows(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(ROWS_CSV_COLUMNS)
        writer.writerows(rows)


def _quality_arrays() -> dict[str, np.ndarray]:
    return {
        "black_frame_fraction": np.asarray((0.0, 0.5, 0.2), dtype=np.float32),
        "duplicate_frame_fraction": np.asarray((0.95, 0.1, 0.2), dtype=np.float32),
        "quality_black_frame_flag": np.asarray((0, 1, 0), dtype=np.uint8),
        "quality_duplicate_frame_flag": np.asarray((1, 0, 0), dtype=np.uint8),
        "quality_exclusion_flag": np.asarray((1, 1, 0), dtype=np.uint8),
        "quality_weight_suggested": np.asarray((0.05, 0.5, 0.8), dtype=np.float32),
    }


def test_video_inventory_rejects_missing_and_extra_videos() -> None:
    expected = ("0", "1")
    with pytest.raises(ValueError, match="missing"):
        validate_video_inventory(("0",), expected)
    with pytest.raises(ValueError, match="extra"):
        validate_video_inventory(("0", "1", "2"), expected)


def test_rows_reader_rejects_time_grid_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    _write_rows(path, [_row(0), _row(1, time_seconds=0.75)])

    with pytest.raises(ValueError, match="time-grid mismatch"):
        read_row_identity(path, "0")


def test_label_values_do_not_enter_phase00_row_identity(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_rows(first, [_row(0, arousal="-100", valence="nan"), _row(1)])
    _write_rows(second, [_row(0, arousal="100", valence="inf"), _row(1)])

    first_identity = read_row_identity(first, "0")
    second_identity = read_row_identity(second, "0")

    assert np.array_equal(first_identity.row_index, second_identity.row_index)
    assert np.array_equal(first_identity.time_seconds, second_identity.time_seconds)
    assert first_identity.source_match_quality == second_identity.source_match_quality


def test_row_count_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="row-count identity mismatch"):
        validate_row_count_identity(3, 2, 3)


@pytest.mark.parametrize(
    "array, message",
    (
        (np.zeros((2, 10), dtype=np.float16), "shape"),
        (np.zeros((2, 20_484), dtype=np.float32), "dtype"),
        (
            np.full((2, 20_484), np.nan, dtype=np.float16),
            "nonfinite",
        ),
    ),
)
def test_cortical_layout_dtype_and_finiteness_fail_closed(array: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_cortical_array(array, 2)


def test_quality_union_mismatch_is_rejected() -> None:
    arrays = _quality_arrays()
    arrays["quality_exclusion_flag"] = np.zeros(3, dtype=np.uint8)

    with pytest.raises(ValueError, match="quality union mismatch"):
        validate_quality_arrays(arrays, 3)


def test_allowlisted_tree_digest_never_hashes_unlisted_file(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed.json"
    forbidden = tmp_path / "vjepa21_hidden_states.npz"
    allowed.write_bytes(b"allowed")
    forbidden.write_bytes(b"must-not-contribute")
    expected = hashlib.sha256()
    allowed_sha = hashlib.sha256(b"allowed").hexdigest()
    expected.update(f"F\0allowed.json\0{len(b'allowed')}\0{allowed_sha}\n".encode())

    identity = allowlisted_tree_identity(tmp_path, (Path("allowed.json"),))

    assert identity.sha256_tree == expected.hexdigest()
    assert identity.files == 1


def test_allowed_file_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_bytes(b"canonical")
    record = {
        "bytes": len(b"canonical"),
        "sha256": hashlib.sha256(b"tampered").hexdigest(),
    }

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_allowed_payload_record(path, record)
