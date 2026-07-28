from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from neural_bridge.veatic21.contracts import (
    ENCODE_POLICY,
    EXPECTED_CORTICAL_WIDTH,
    ROWS_CSV_COLUMNS,
    TRIBE_KEY_SCHEMA,
)
from neural_bridge.veatic21.data import (
    RowIdentity,
    allowlisted_tree_identity,
    assert_tree_identity,
    audit_tribe_payload,
    read_row_identity,
    read_tribe_row_metadata,
    validate_row_count_identity,
    validate_video_inventory,
    verify_vjepa_payload_records,
)


def _write_rows(path: Path, *, times: tuple[float, ...] = (0.0, 0.5)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROWS_CSV_COLUMNS)
        writer.writeheader()
        for index, time_seconds in enumerate(times):
            source_position = time_seconds * 25.0
            floor = int(np.floor(source_position))
            ceil = int(np.ceil(source_position))
            writer.writerow(
                {
                    "video_id": "0",
                    "video_name": "0.mp4",
                    "video_relpath": "videos/0.mp4",
                    "arousal_relpath": "rating_averaged/0_arousal.csv",
                    "valence_relpath": "rating_averaged/0_valence.csv",
                    "row_index": index,
                    "time_seconds": time_seconds,
                    "row_hz": 2,
                    "clip_start_seconds": 0,
                    "clip_end_seconds": time_seconds,
                    "native_label_fps": 25,
                    "native_label_frame_count": 100,
                    "source_frame_position": source_position,
                    "source_floor_frame_index": floor,
                    "source_ceil_frame_index": ceil,
                    "source_interp_alpha": source_position - floor,
                    "source_arousal": 0.1,
                    "source_valence": 0.2,
                    "source_match_quality": (
                        "native_exact" if source_position == floor else "linear_native_frames"
                    ),
                    "encode_policy": ENCODE_POLICY,
                    "arousal": 0.1,
                    "valence": 0.2,
                }
            )


def _payload_arrays(
    *,
    cortical_dtype: np.dtype[np.generic] | None = None,
    nonfinite: bool = False,
    bad_union: bool = False,
) -> dict[str, np.ndarray]:
    rows = 2
    cortical_dtype = cortical_dtype or np.dtype(np.float16)
    arrays: dict[str, np.ndarray] = {}
    for key in TRIBE_KEY_SCHEMA:
        arrays[key] = np.zeros(rows, dtype=np.float32)
    arrays["time_seconds"] = np.asarray([0.0, 0.5], dtype=np.float32)
    arrays["sample_frame_indices"] = np.zeros((rows, 64), dtype=np.int32)
    arrays["sample_time_seconds"] = np.zeros((rows, 64), dtype=np.float32)
    arrays["selected_state_indices"] = np.arange(20, dtype=np.int32)
    cortical = np.zeros((rows, EXPECTED_CORTICAL_WIDTH), dtype=cortical_dtype)
    if nonfinite:
        cortical[0, 0] = np.nan
    arrays["cortical_prediction"] = cortical
    arrays["tribe_grouped_video_feature"] = np.zeros((rows, 2, 1408), dtype=np.float16)
    arrays["temporal_diagnostics53"] = np.zeros((rows, 53), dtype=np.float32)
    arrays["black_frame_fraction"] = np.asarray([0.0, 0.75], dtype=np.float32)
    arrays["duplicate_frame_fraction"] = np.asarray([0.0, 0.0], dtype=np.float32)
    arrays["quality_black_frame_flag"] = np.asarray([0, 1], dtype=np.uint8)
    arrays["quality_duplicate_frame_flag"] = np.asarray([0, 0], dtype=np.uint8)
    arrays["quality_exclusion_flag"] = np.asarray([0, 0 if bad_union else 1], dtype=np.uint8)
    return arrays


def _identity() -> RowIdentity:
    return RowIdentity(
        video_id="0",
        row_index=np.asarray([0, 1], dtype=np.int32),
        time_seconds=np.asarray([0.0, 0.5], dtype=np.float64),
        source_match_counts={"native_exact": 1, "linear_native_frames": 1},
    )


def test_video_inventory_fails_on_missing_or_extra_video() -> None:
    with pytest.raises(ValueError, match="missing"):
        validate_video_inventory(("0",), ("0", "1"))
    with pytest.raises(ValueError, match="extra"):
        validate_video_inventory(("0", "1"), ("0",))


def test_rows_reader_returns_identity_without_labels(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    _write_rows(path)
    identity = read_row_identity(path, "0")

    assert identity.row_count == 2
    assert np.array_equal(identity.time_seconds, [0.0, 0.5])
    assert not hasattr(identity, "arousal")
    assert not hasattr(identity, "valence")


def test_rows_reader_fails_on_time_grid_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    _write_rows(path, times=(0.0, 0.6))
    with pytest.raises(ValueError, match="time-grid mismatch"):
        read_row_identity(path, "0")


def test_row_count_identity_fails_closed() -> None:
    with pytest.raises(ValueError, match="row-count identity mismatch"):
        validate_row_count_identity(2, 3, 2)


def test_payload_audit_accepts_only_declared_nonlabel_arrays(tmp_path: Path) -> None:
    path = tmp_path / "payload.npz"
    np.savez(path, **_payload_arrays())  # ty: ignore[invalid-argument-type]
    audit = audit_tribe_payload(path, _identity())

    assert audit.cortical_shape == (2, EXPECTED_CORTICAL_WIDTH)
    assert not set(audit.accessed_arrays) & {"arousal", "valence"}
    assert audit.union_rows == 1


def test_phase01_tribe_reader_preserves_quality_without_cortical_access(tmp_path: Path) -> None:
    path = tmp_path / "payload.npz"
    np.savez(path, **_payload_arrays(nonfinite=True))  # ty: ignore[invalid-argument-type]
    metadata = read_tribe_row_metadata(path, _identity())

    assert metadata.row_count == 2
    assert np.array_equal(metadata.quality_exclusion_flag, [0, 1])
    assert "cortical_prediction" not in metadata.accessed_arrays
    assert not set(metadata.accessed_arrays) & {"arousal", "valence"}


@pytest.mark.parametrize(
    ("arrays", "message"),
    (
        (_payload_arrays(cortical_dtype=np.dtype(np.float32)), "cortical layout mismatch"),
        (_payload_arrays(nonfinite=True), "nonfinite cortical value"),
        (_payload_arrays(bad_union=True), "quality union mismatch"),
    ),
)
def test_payload_audit_fails_on_corruptions(
    tmp_path: Path, arrays: dict[str, np.ndarray], message: str
) -> None:
    path = tmp_path / "payload.npz"
    np.savez(path, **arrays)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match=message):
        audit_tribe_payload(path, _identity())


def test_allowlisted_tree_digest_detects_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "allowed.json"
    path.write_text("{}", encoding="utf-8")
    identity = allowlisted_tree_identity(tmp_path, (Path("allowed.json"),))
    with pytest.raises(ValueError, match="tree identity mismatch"):
        assert_tree_identity(
            identity,
            sha256="0" * 64,
            files=1,
            size_bytes=2,
            name="fixture",
        )


def test_vjepa_payload_verification_never_hashes_hidden_state(tmp_path: Path) -> None:
    verified = ("manifest.json", "preprocessing.json", "rows.csv", "status.json")
    records = []
    for filename in verified:
        path = tmp_path / filename
        path.write_text("{}", encoding="utf-8")
        records.append(
            {
                "path": filename,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    records.append(
        {
            "path": "vjepa21_hidden_states.npz",
            "bytes": 999,
            "sha256": "f" * 64,
        }
    )
    payload = {
        "video_id": "0",
        "file_count": 5,
        "total_bytes": sum(int(record["bytes"]) for record in records),
        "files": records,
    }
    payload_path = tmp_path / "_PAYLOAD_SHA256.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    upload = {
        "status": "complete",
        "video_id": "0",
        "payload_manifest": payload_path.name,
        "payload_manifest_sha256": hashlib.sha256(payload_path.read_bytes()).hexdigest(),
        "payload_manifest_bytes": payload_path.stat().st_size,
        "payload_file_count": 5,
        "payload_total_bytes": payload["total_bytes"],
    }
    (tmp_path / "_UPLOAD_COMPLETE.json").write_text(json.dumps(upload), encoding="utf-8")

    verify_vjepa_payload_records(tmp_path, "0")
