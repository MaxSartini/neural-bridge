"""Fail-closed readers for the fresh VEATIC 2.1 Phase 00 audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neural_bridge.veatic21.contracts import (
    BLACK_FRACTION_THRESHOLD,
    DUPLICATE_FRACTION_THRESHOLD,
    ENCODE_POLICY,
    EXPECTED_CORTICAL_DTYPE,
    EXPECTED_CORTICAL_WIDTH,
    EXPECTED_DIAGNOSTIC_WIDTH,
    EXPECTED_NATIVE_LABEL_FPS,
    EXPECTED_ROW_HZ,
    EXPECTED_TIME_STEP_SECONDS,
    EXPECTED_VIDEO_IDS,
    FORBIDDEN_AGAIN_RUNTIME_ROOTS,
    FORBIDDEN_HIDDEN_STATE_FILENAME,
    PHASE00_ACCESSED_TRIBE_ARRAYS,
    ROWS_CSV_COLUMNS,
    TRIBE_KEY_SCHEMA,
    TRIBE_VIDEO_FILENAMES,
    VJEPA_ALLOWED_FILENAMES,
    VJEPA_PAYLOAD_VERIFIED_FILENAMES,
)


@dataclass(frozen=True)
class TreeIdentity:
    root: str
    sha256: str
    files: int
    size_bytes: int
    entries: tuple[dict[str, str | int], ...]

    def compact(self) -> dict[str, str | int]:
        return {
            "root": self.root,
            "sha256": self.sha256,
            "files": self.files,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class RowIdentity:
    video_id: str
    row_index: np.ndarray
    time_seconds: np.ndarray
    source_match_counts: dict[str, int]

    @property
    def row_count(self) -> int:
        return int(self.row_index.size)


@dataclass(frozen=True)
class PayloadAudit:
    video_id: str
    row_count: int
    key_schema: tuple[str, ...]
    cortical_shape: tuple[int, int]
    cortical_dtype: str
    black_rows: int
    duplicate_rows: int
    both_rows: int
    union_rows: int
    accessed_arrays: tuple[str, ...]


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def reject_forbidden_runtime_path(path: Path) -> Path:
    """Reject hidden states and AGAIN runtime roots before any filesystem read."""

    candidate = _resolved(path)
    if FORBIDDEN_HIDDEN_STATE_FILENAME in candidate.parts:
        raise ValueError(f"forbidden V-JEPA hidden-state path: {candidate}")
    for root in FORBIDDEN_AGAIN_RUNTIME_ROOTS:
        if candidate == root or candidate.is_relative_to(root):
            raise ValueError(f"forbidden AGAIN runtime path: {candidate}")
    return candidate


def sha256_file(path: Path) -> str:
    path = reject_forbidden_runtime_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    path = reject_forbidden_runtime_path(path)
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def discover_numeric_video_ids(root: Path) -> tuple[str, ...]:
    root = reject_forbidden_runtime_path(root)
    numeric: list[str] = []
    unexpected: list[str] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.isdecimal() and str(int(entry.name)) == entry.name:
            numeric.append(entry.name)
        else:
            unexpected.append(entry.name)
    if unexpected:
        raise ValueError(f"unexpected nonnumeric video directories: {unexpected}")
    return tuple(sorted(numeric, key=int))


def validate_video_inventory(
    found: Sequence[str], expected: Sequence[str] = EXPECTED_VIDEO_IDS
) -> None:
    missing = sorted(set(expected) - set(found), key=int)
    extra = sorted(set(found) - set(expected), key=int)
    if missing or extra or len(found) != len(set(found)):
        raise ValueError(f"video inventory mismatch: missing={missing}, extra={extra}")


def validate_tribe_layout(root: Path, video_ids: Sequence[str]) -> None:
    root = reject_forbidden_runtime_path(root)
    ignored = {".DS_Store"}
    actual_root = {entry.name for entry in root.iterdir() if entry.name not in ignored}
    expected_root = {"run_status.json", *video_ids}
    if actual_root != expected_root:
        raise ValueError("TRIBE per-video root layout mismatch")
    if (root / "run_status.json").is_symlink():
        raise ValueError("TRIBE run status must be a regular file")
    for video_id in video_ids:
        video_root = root / video_id
        actual = {entry.name for entry in video_root.iterdir() if entry.name not in ignored}
        if actual != TRIBE_VIDEO_FILENAMES:
            raise ValueError(f"TRIBE file layout mismatch for video {video_id}: {actual}")
        for filename in TRIBE_VIDEO_FILENAMES:
            path = video_root / filename
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"TRIBE input is not a regular file: {path}")


def validate_vjepa_layout(root: Path, video_ids: Sequence[str]) -> None:
    root = reject_forbidden_runtime_path(root)
    ignored = {".DS_Store"}
    actual_root = {entry.name for entry in root.iterdir() if entry.name not in ignored}
    if actual_root != set(video_ids):
        raise ValueError("V-JEPA root layout mismatch")
    allowed_names = set(VJEPA_ALLOWED_FILENAMES)
    expected_names = allowed_names | {FORBIDDEN_HIDDEN_STATE_FILENAME}
    for video_id in video_ids:
        video_root = root / video_id
        actual = {entry.name for entry in video_root.iterdir() if entry.name not in ignored}
        if actual != expected_names:
            raise ValueError(f"V-JEPA file layout mismatch for video {video_id}: {actual}")
        for filename in VJEPA_ALLOWED_FILENAMES:
            path = video_root / filename
            reject_forbidden_runtime_path(path)
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"allowed V-JEPA input is not a regular file: {path}")


def allowlisted_tree_identity(root: Path, relative_paths: Iterable[Path]) -> TreeIdentity:
    root = reject_forbidden_runtime_path(root)
    digest = hashlib.sha256()
    entries: list[dict[str, str | int]] = []
    size_bytes = 0
    for relative in sorted(relative_paths, key=lambda value: value.as_posix()):
        path = reject_forbidden_runtime_path(root / relative)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"allowlisted input is not a regular file: {path}")
        size = path.stat().st_size
        content_sha256 = sha256_file(path)
        relative_text = relative.as_posix()
        digest.update(f"F\0{relative_text}\0{size}\0{content_sha256}\n".encode())
        size_bytes += size
        entries.append({"path": relative_text, "bytes": size, "sha256": content_sha256})
    return TreeIdentity(
        root=str(root),
        sha256=digest.hexdigest(),
        files=len(entries),
        size_bytes=size_bytes,
        entries=tuple(entries),
    )


def expected_tribe_paths(video_ids: Sequence[str]) -> tuple[Path, ...]:
    paths = [Path("run_status.json")]
    for video_id in video_ids:
        paths.extend(Path(video_id) / filename for filename in sorted(TRIBE_VIDEO_FILENAMES))
    return tuple(paths)


def expected_vjepa_paths(video_ids: Sequence[str]) -> tuple[Path, ...]:
    return tuple(
        Path(video_id) / filename
        for video_id in video_ids
        for filename in sorted(VJEPA_ALLOWED_FILENAMES)
    )


def assert_tree_identity(
    actual: TreeIdentity, *, sha256: str, files: int, size_bytes: int, name: str
) -> None:
    expected = {"sha256": sha256, "files": files, "size_bytes": size_bytes}
    observed = actual.compact()
    mismatch = {
        key: (expected[key], observed[key])
        for key in expected
        if expected[key] != observed[key]
    }
    if mismatch:
        raise ValueError(f"{name} tree identity mismatch: {mismatch}")


def _parse_int(value: str, *, field: str, path: Path, row_number: int) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{path}:{row_number}: invalid {field}") from error


def _parse_float(value: str, *, field: str, path: Path, row_number: int) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"{path}:{row_number}: invalid {field}") from error
    if not math.isfinite(result):
        raise ValueError(f"{path}:{row_number}: nonfinite {field}")
    return result


def read_row_identity(path: Path, expected_video_id: str) -> RowIdentity:
    """Read and return identity/provenance only; label values never enter the result."""

    path = reject_forbidden_runtime_path(path)
    row_indices: list[int] = []
    times: list[float] = []
    source_matches: Counter[str] = Counter()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as error:
            raise ValueError(f"empty rows.csv: {path}") from error
        if header != ROWS_CSV_COLUMNS:
            raise ValueError(f"rows.csv schema mismatch: {path}")
        column = {name: header.index(name) for name in header}
        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"{path}:{row_number}: column-count mismatch")
            if row[column["video_id"]] != expected_video_id:
                raise ValueError(f"{path}:{row_number}: video_id mismatch")
            index = _parse_int(
                row[column["row_index"]], field="row_index", path=path, row_number=row_number
            )
            if index != len(row_indices):
                raise ValueError(f"{path}:{row_number}: nonsequential row_index")
            time_seconds = _parse_float(
                row[column["time_seconds"]],
                field="time_seconds",
                path=path,
                row_number=row_number,
            )
            if time_seconds != index * EXPECTED_TIME_STEP_SECONDS:
                raise ValueError(f"{path}:{row_number}: time-grid mismatch")
            row_hz = _parse_float(
                row[column["row_hz"]], field="row_hz", path=path, row_number=row_number
            )
            if row_hz != EXPECTED_ROW_HZ:
                raise ValueError(f"{path}:{row_number}: row_hz mismatch")
            native_fps = _parse_float(
                row[column["native_label_fps"]],
                field="native_label_fps",
                path=path,
                row_number=row_number,
            )
            if native_fps != EXPECTED_NATIVE_LABEL_FPS:
                raise ValueError(f"{path}:{row_number}: native label FPS mismatch")
            if row[column["encode_policy"]] != ENCODE_POLICY:
                raise ValueError(f"{path}:{row_number}: encode policy mismatch")

            position = _parse_float(
                row[column["source_frame_position"]],
                field="source_frame_position",
                path=path,
                row_number=row_number,
            )
            floor = _parse_int(
                row[column["source_floor_frame_index"]],
                field="source_floor_frame_index",
                path=path,
                row_number=row_number,
            )
            ceil = _parse_int(
                row[column["source_ceil_frame_index"]],
                field="source_ceil_frame_index",
                path=path,
                row_number=row_number,
            )
            alpha = _parse_float(
                row[column["source_interp_alpha"]],
                field="source_interp_alpha",
                path=path,
                row_number=row_number,
            )
            if floor > ceil or not 0.0 <= alpha <= 1.0:
                raise ValueError(f"{path}:{row_number}: invalid interpolation provenance")
            if not math.isclose(position, floor + alpha, abs_tol=1e-9):
                raise ValueError(f"{path}:{row_number}: source position mismatch")
            source_match = row[column["source_match_quality"]]
            if source_match not in {"native_exact", "linear_native_frames"}:
                raise ValueError(f"{path}:{row_number}: source match quality mismatch")
            source_matches[source_match] += 1
            row_indices.append(index)
            times.append(time_seconds)
    if not row_indices:
        raise ValueError(f"rows.csv contains no data rows: {path}")
    return RowIdentity(
        video_id=expected_video_id,
        row_index=np.asarray(row_indices, dtype=np.int32),
        time_seconds=np.asarray(times, dtype=np.float64),
        source_match_counts=dict(source_matches),
    )


def verify_vjepa_payload_records(video_root: Path, video_id: str) -> None:
    video_root = reject_forbidden_runtime_path(video_root)
    payload_path = video_root / "_PAYLOAD_SHA256.json"
    upload_path = video_root / "_UPLOAD_COMPLETE.json"
    payload = load_json(payload_path)
    upload = load_json(upload_path)

    records = payload.get("files")
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"invalid V-JEPA payload records for video {video_id}")
    by_name = {record.get("path"): record for record in records}
    expected_record_names = set(VJEPA_PAYLOAD_VERIFIED_FILENAMES) | {
        FORBIDDEN_HIDDEN_STATE_FILENAME
    }
    if set(by_name) != expected_record_names or len(by_name) != len(records):
        raise ValueError(f"unexpected V-JEPA payload record set for video {video_id}")
    if payload.get("video_id") != video_id or payload.get("file_count") != len(records):
        raise ValueError(f"V-JEPA payload summary mismatch for video {video_id}")
    if payload.get("total_bytes") != sum(int(record["bytes"]) for record in records):
        raise ValueError(f"V-JEPA payload total mismatch for video {video_id}")

    for filename in VJEPA_PAYLOAD_VERIFIED_FILENAMES:
        record = by_name[filename]
        path = video_root / filename
        if record.get("bytes") != path.stat().st_size:
            raise ValueError(f"V-JEPA allowed size mismatch: {path}")
        if record.get("sha256") != sha256_file(path):
            raise ValueError(f"V-JEPA allowed SHA-256 mismatch: {path}")

    payload_sha256 = sha256_file(payload_path)
    if not (
        upload.get("status") == "complete"
        and upload.get("video_id") == video_id
        and upload.get("payload_manifest") == payload_path.name
        and upload.get("payload_manifest_sha256") == payload_sha256
        and upload.get("payload_manifest_bytes") == payload_path.stat().st_size
        and upload.get("payload_file_count") == payload.get("file_count")
        and upload.get("payload_total_bytes") == payload.get("total_bytes")
    ):
        raise ValueError(f"V-JEPA upload marker mismatch for video {video_id}")


def validate_row_count_identity(tribe: object, vjepa: object, rows: int) -> None:
    if tribe != rows or vjepa != rows:
        raise ValueError(f"row-count identity mismatch: TRIBE={tribe}, V-JEPA={vjepa}, CSV={rows}")


def audit_tribe_payload(path: Path, row_identity: RowIdentity) -> PayloadAudit:
    path = reject_forbidden_runtime_path(path)
    accessed: list[str] = []

    def read(npz: Mapping[str, np.ndarray], key: str) -> np.ndarray:
        if key not in PHASE00_ACCESSED_TRIBE_ARRAYS:
            raise AssertionError(f"Phase 00 attempted undeclared TRIBE array access: {key}")
        accessed.append(key)
        return npz[key]

    with np.load(path, allow_pickle=False) as payload:
        key_schema = tuple(payload.files)
        if key_schema != TRIBE_KEY_SCHEMA:
            raise ValueError(f"TRIBE key schema mismatch for video {row_identity.video_id}")
        cortical = read(payload, "cortical_prediction")
        expected_shape = (row_identity.row_count, EXPECTED_CORTICAL_WIDTH)
        if cortical.shape != expected_shape or str(cortical.dtype) != EXPECTED_CORTICAL_DTYPE:
            raise ValueError(f"cortical layout mismatch for video {row_identity.video_id}")
        if not np.isfinite(cortical).all():
            raise ValueError(f"nonfinite cortical value for video {row_identity.video_id}")

        time_seconds = read(payload, "time_seconds")
        if time_seconds.shape != (row_identity.row_count,) or not np.isfinite(time_seconds).all():
            raise ValueError(f"TRIBE time layout mismatch for video {row_identity.video_id}")
        if not np.array_equal(time_seconds.astype(np.float64), row_identity.time_seconds):
            raise ValueError(f"TRIBE/rows.csv time mismatch for video {row_identity.video_id}")

        black_fraction = read(payload, "black_frame_fraction")
        duplicate_fraction = read(payload, "duplicate_frame_fraction")
        black_raw = read(payload, "quality_black_frame_flag")
        duplicate_raw = read(payload, "quality_duplicate_frame_flag")
        exclusion_raw = read(payload, "quality_exclusion_flag")
        for name, array in (
            ("black_frame_fraction", black_fraction),
            ("duplicate_frame_fraction", duplicate_fraction),
            ("quality_black_frame_flag", black_raw),
            ("quality_duplicate_frame_flag", duplicate_raw),
            ("quality_exclusion_flag", exclusion_raw),
        ):
            if array.shape != (row_identity.row_count,):
                raise ValueError(f"{name} shape mismatch for video {row_identity.video_id}")
        if not np.isfinite(black_fraction).all() or not np.isfinite(duplicate_fraction).all():
            raise ValueError(f"nonfinite quality fraction for video {row_identity.video_id}")
        for name, array in (
            ("black", black_raw),
            ("duplicate", duplicate_raw),
            ("exclusion", exclusion_raw),
        ):
            if not np.isin(array, (0, 1)).all():
                raise ValueError(f"nonbinary {name} flag for video {row_identity.video_id}")
        black = black_raw.astype(bool)
        duplicate = duplicate_raw.astype(bool)
        exclusion = exclusion_raw.astype(bool)
        if not np.array_equal(black, black_fraction >= BLACK_FRACTION_THRESHOLD):
            raise ValueError(f"black threshold mismatch for video {row_identity.video_id}")
        if not np.array_equal(duplicate, duplicate_fraction >= DUPLICATE_FRACTION_THRESHOLD):
            raise ValueError(f"duplicate threshold mismatch for video {row_identity.video_id}")
        if not np.array_equal(exclusion, black | duplicate):
            raise ValueError(f"quality union mismatch for video {row_identity.video_id}")

        diagnostics = read(payload, "temporal_diagnostics53")
        if diagnostics.shape != (row_identity.row_count, EXPECTED_DIAGNOSTIC_WIDTH):
            raise ValueError(f"diagnostics layout mismatch for video {row_identity.video_id}")
        if str(diagnostics.dtype) != "float32" or not np.isfinite(diagnostics).all():
            raise ValueError(f"diagnostics dtype/finiteness mismatch for {row_identity.video_id}")

    return PayloadAudit(
        video_id=row_identity.video_id,
        row_count=row_identity.row_count,
        key_schema=key_schema,
        cortical_shape=expected_shape,
        cortical_dtype=EXPECTED_CORTICAL_DTYPE,
        black_rows=int(black.sum()),
        duplicate_rows=int(duplicate.sum()),
        both_rows=int((black & duplicate).sum()),
        union_rows=int(exclusion.sum()),
        accessed_arrays=tuple(accessed),
    )
