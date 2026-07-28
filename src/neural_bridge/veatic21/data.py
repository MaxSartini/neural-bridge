"""Phase 00-only VEATIC input readers and deterministic validators."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping
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
    EXPECTED_NATIVE_LABEL_FPS,
    EXPECTED_ROW_HZ,
    EXPECTED_TIME_STEP_SECONDS,
    EXPECTED_VIDEO_IDS,
    FORBIDDEN_HIDDEN_STATE_FILENAME,
    LABEL_ARRAY_NAMES,
    PHASE00_FEATURE_ARRAYS,
    ROWS_CSV_COLUMNS,
    VJEPA_ALLOWED_FILENAMES,
    reject_forbidden_runtime_path,
)


@dataclass(frozen=True)
class RowIdentity:
    video_id: str
    row_index: np.ndarray
    time_seconds: np.ndarray
    source_match_quality: tuple[str, ...]

    @property
    def row_count(self) -> int:
        return len(self.row_index)


@dataclass(frozen=True)
class TreeIdentity:
    path: str
    sha256_tree: str
    files: int
    symlinks: int
    size_bytes: int
    entries: tuple[dict[str, str | int], ...]

    def compact(self) -> dict[str, str | int]:
        return {
            "path": self.path,
            "sha256_tree": self.sha256_tree,
            "files": self.files,
            "symlinks": self.symlinks,
            "size_bytes": self.size_bytes,
        }


def safe_sha256_file(path: Path) -> str:
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
        raise ValueError(f"expected JSON object: {path}")
    return value


def discover_numeric_video_ids(root: Path) -> tuple[str, ...]:
    reject_forbidden_runtime_path(root)
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
        raise ValueError(f"unexpected nonnumeric video directories under {root}: {unexpected}")
    return tuple(sorted(numeric, key=int))


def validate_video_inventory(
    found: tuple[str, ...], expected: tuple[str, ...] = EXPECTED_VIDEO_IDS
) -> None:
    missing = sorted(set(expected) - set(found), key=int)
    extra = sorted(set(found) - set(expected), key=int)
    if missing or extra or len(found) != len(set(found)):
        raise ValueError(f"video inventory mismatch: missing={missing}, extra={extra}")


def validate_allowed_vjepa_filenames(video_root: Path) -> None:
    for filename in VJEPA_ALLOWED_FILENAMES:
        path = video_root / filename
        reject_forbidden_runtime_path(path)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or non-regular allowed V-JEPA file: {path}")
    if FORBIDDEN_HIDDEN_STATE_FILENAME in VJEPA_ALLOWED_FILENAMES:
        raise AssertionError("hidden-state payload entered the V-JEPA allowlist")


def validate_row_count_identity(tribe_count: object, vjepa_count: object, csv_count: int) -> None:
    if tribe_count != csv_count or vjepa_count != csv_count:
        raise ValueError(
            "row-count identity mismatch: "
            f"TRIBE={tribe_count}, V-JEPA={vjepa_count}, rows.csv={csv_count}"
        )


def verify_allowed_payload_record(path: Path, record: Mapping[str, object]) -> None:
    path = reject_forbidden_runtime_path(path)
    if record.get("bytes") != path.stat().st_size:
        raise ValueError(f"allowed payload size mismatch: {path}")
    if record.get("sha256") != safe_sha256_file(path):
        raise ValueError(f"allowed payload hash mismatch: {path}")


def allowlisted_tree_identity(root: Path, relative_paths: tuple[Path, ...]) -> TreeIdentity:
    """Hash only constructed allowlisted files using the canonical tree algorithm."""

    reject_forbidden_runtime_path(root)
    digest = hashlib.sha256()
    size_bytes = 0
    entries: list[dict[str, str | int]] = []
    for relative in sorted(relative_paths, key=lambda value: value.as_posix()):
        path = reject_forbidden_runtime_path(root / relative)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"allowlisted input is not a regular file: {path}")
        size = path.stat().st_size
        content_sha256 = safe_sha256_file(path)
        relative_text = relative.as_posix()
        digest.update(f"F\0{relative_text}\0{size}\0{content_sha256}\n".encode())
        size_bytes += size
        entries.append({"path": relative_text, "bytes": size, "sha256": content_sha256})
    return TreeIdentity(
        path=str(root.resolve()),
        sha256_tree=digest.hexdigest(),
        files=len(entries),
        symlinks=0,
        size_bytes=size_bytes,
        entries=tuple(entries),
    )


def _parse_int(value: str, *, field: str, path: Path, row_number: int) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{path}:{row_number}: invalid {field}") from error


def _parse_float(value: str, *, field: str, path: Path, row_number: int) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{path}:{row_number}: invalid {field}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{path}:{row_number}: nonfinite {field}")
    return parsed


def read_row_identity(path: Path, expected_video_id: str) -> RowIdentity:
    """Read identity/provenance columns only; label values never enter the result."""

    path = reject_forbidden_runtime_path(path)
    row_indices: list[int] = []
    times: list[float] = []
    match_quality: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as error:
            raise ValueError(f"empty rows.csv: {path}") from error
        if header != ROWS_CSV_COLUMNS:
            raise ValueError(f"rows.csv schema mismatch: {path}")
        index = {name: header.index(name) for name in header}
        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"{path}:{row_number}: column-count mismatch")
            if row[index["video_id"]] != expected_video_id:
                raise ValueError(f"{path}:{row_number}: video_id mismatch")
            row_index = _parse_int(
                row[index["row_index"]], field="row_index", path=path, row_number=row_number
            )
            if row_index != len(row_indices):
                raise ValueError(f"{path}:{row_number}: nonsequential row_index")
            time_seconds = _parse_float(
                row[index["time_seconds"]],
                field="time_seconds",
                path=path,
                row_number=row_number,
            )
            if time_seconds != row_index * EXPECTED_TIME_STEP_SECONDS:
                raise ValueError(f"{path}:{row_number}: time-grid mismatch")
            row_hz = _parse_float(
                row[index["row_hz"]], field="row_hz", path=path, row_number=row_number
            )
            if row_hz != EXPECTED_ROW_HZ:
                raise ValueError(f"{path}:{row_number}: row_hz mismatch")
            native_fps = _parse_float(
                row[index["native_label_fps"]],
                field="native_label_fps",
                path=path,
                row_number=row_number,
            )
            if native_fps != EXPECTED_NATIVE_LABEL_FPS:
                raise ValueError(f"{path}:{row_number}: native label FPS mismatch")
            source_position = _parse_float(
                row[index["source_frame_position"]],
                field="source_frame_position",
                path=path,
                row_number=row_number,
            )
            source_floor = _parse_int(
                row[index["source_floor_frame_index"]],
                field="source_floor_frame_index",
                path=path,
                row_number=row_number,
            )
            source_ceil = _parse_int(
                row[index["source_ceil_frame_index"]],
                field="source_ceil_frame_index",
                path=path,
                row_number=row_number,
            )
            source_alpha = _parse_float(
                row[index["source_interp_alpha"]],
                field="source_interp_alpha",
                path=path,
                row_number=row_number,
            )
            if source_position != time_seconds * native_fps:
                raise ValueError(f"{path}:{row_number}: source-frame position mismatch")
            if source_floor != math.floor(source_position) or source_ceil != math.ceil(
                source_position
            ):
                raise ValueError(f"{path}:{row_number}: source-frame bounds mismatch")
            if source_alpha != source_position - source_floor:
                raise ValueError(f"{path}:{row_number}: interpolation alpha mismatch")
            quality = row[index["source_match_quality"]]
            expected_quality = "native_exact" if source_alpha == 0.0 else "linear_native_frames"
            if quality != expected_quality:
                raise ValueError(f"{path}:{row_number}: source match quality mismatch")
            if row[index["encode_policy"]] != ENCODE_POLICY:
                raise ValueError(f"{path}:{row_number}: encode policy mismatch")
            row_indices.append(row_index)
            times.append(time_seconds)
            match_quality.append(quality)
    if not row_indices:
        raise ValueError(f"rows.csv contains no data rows: {path}")
    return RowIdentity(
        video_id=expected_video_id,
        row_index=np.asarray(row_indices, dtype=np.int64),
        time_seconds=np.asarray(times, dtype=np.float64),
        source_match_quality=tuple(match_quality),
    )


def load_phase00_tribe_arrays(
    path: Path, requested: tuple[str, ...]
) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    """Load only explicitly permitted non-label arrays from the final TRIBE payload."""

    path = reject_forbidden_runtime_path(path)
    requested_set = set(requested)
    forbidden = sorted(requested_set & LABEL_ARRAY_NAMES)
    unsupported = sorted(requested_set - PHASE00_FEATURE_ARRAYS)
    if forbidden:
        raise ValueError(f"Phase 00 cannot request label arrays: {forbidden}")
    if unsupported:
        raise ValueError(f"Phase 00 array request is not allowlisted: {unsupported}")
    with np.load(path, allow_pickle=False) as archive:
        schema = tuple(archive.files)
        missing = sorted(requested_set - set(schema))
        if missing:
            raise ValueError(f"TRIBE payload lacks required arrays: {missing}")
        arrays = {name: archive[name] for name in requested}
    return schema, arrays


def validate_cortical_array(cortical: np.ndarray, row_count: int) -> None:
    if cortical.shape != (row_count, EXPECTED_CORTICAL_WIDTH):
        raise ValueError(f"cortical shape mismatch: {cortical.shape}")
    if cortical.dtype.name != EXPECTED_CORTICAL_DTYPE:
        raise ValueError(f"cortical dtype mismatch: {cortical.dtype}")
    if not np.isfinite(cortical).all():
        raise ValueError("cortical array contains nonfinite values")


def validate_quality_arrays(arrays: dict[str, np.ndarray], row_count: int) -> dict[str, int]:
    black_fraction = arrays["black_frame_fraction"]
    duplicate_fraction = arrays["duplicate_frame_fraction"]
    black = arrays["quality_black_frame_flag"]
    duplicate = arrays["quality_duplicate_frame_flag"]
    union = arrays["quality_exclusion_flag"]
    weights = arrays["quality_weight_suggested"]
    for name, values in (
        ("black_frame_fraction", black_fraction),
        ("duplicate_frame_fraction", duplicate_fraction),
        ("quality_weight_suggested", weights),
    ):
        if values.shape != (row_count,) or not np.issubdtype(values.dtype, np.floating):
            raise ValueError(f"{name} layout mismatch")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains nonfinite values")
    for name, values in (
        ("quality_black_frame_flag", black),
        ("quality_duplicate_frame_flag", duplicate),
        ("quality_exclusion_flag", union),
    ):
        if values.shape != (row_count,) or values.dtype != np.dtype("uint8"):
            raise ValueError(f"{name} layout mismatch")
        if not np.isin(values, (0, 1)).all():
            raise ValueError(f"{name} is not binary")
    expected_black = (black_fraction >= BLACK_FRACTION_THRESHOLD).astype(np.uint8)
    expected_duplicate = (duplicate_fraction >= DUPLICATE_FRACTION_THRESHOLD).astype(np.uint8)
    expected_union = np.maximum(black, duplicate)
    if not np.array_equal(black, expected_black):
        raise ValueError("black-frame flag threshold mismatch")
    if not np.array_equal(duplicate, expected_duplicate):
        raise ValueError("duplicate-frame flag threshold mismatch")
    if not np.array_equal(union, expected_union):
        raise ValueError("quality union mismatch")
    return {
        "black_rows": int(black.sum()),
        "duplicate_rows": int(duplicate.sum()),
        "both_rows": int(np.logical_and(black, duplicate).sum()),
        "union_rows": int(union.sum()),
        "unflagged_rows": int(row_count - union.sum()),
        "total_rows": row_count,
    }
