"""Assemble and verify the single canonical VEATIC 2.1 downstream input bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_TRIBE_ROOT = Path(
    "/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/tribe-v2/"
    "veatic 2.1 raw cortical predictions/per_video"
)
DEFAULT_VJEPA_ROOT = Path(
    "/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/vjepa-2.1/"
    "veatic 2.1 v jepa 2.1 stuff"
)
DEFAULT_BUNDLE_ROOT = Path(
    "/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/neural-bridge-input"
)

EXPECTED_VIDEO_IDS = tuple(str(value) for value in range(124))
FORBIDDEN_NAME = "vjepa21_hidden_states.npz"
PROTECTED_ROOTS = (
    Path("/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/tribe-v2"),
    Path("/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/vjepa-2.1"),
    DEFAULT_BUNDLE_ROOT,
    Path("/Volumes/onn. Drive/Neural Bridge Artifacts/runs/again"),
)

TRIBE_FILES = {
    "tribe_v2_cortical_predictions.npz": "tribe_v2_cortical_predictions.npz",
    "manifest.json": "cortical_manifest.json",
    "status.json": "cortical_status.json",
}

ALIGNMENT_FILES = {
    "rows.csv": "rows.csv",
    "manifest.json": "alignment_manifest.json",
    "preprocessing.json": "alignment_preprocessing.json",
    "status.json": "alignment_status.json",
}

REQUIRED_NPZ_KEYS = (
    "time_seconds",
    "source_frame_position",
    "source_floor_frame_index",
    "source_ceil_frame_index",
    "source_interp_alpha",
    "arousal",
    "valence",
    "cortical_prediction",
    "temporal_diagnostics53",
)

CSV_NUMERIC_MATCHES = {
    "time_seconds": "time_seconds",
    "source_frame_position": "source_frame_position",
    "source_floor_frame_index": "source_floor_frame_index",
    "source_ceil_frame_index": "source_ceil_frame_index",
    "source_interp_alpha": "source_interp_alpha",
    "arousal": "arousal",
    "valence": "valence",
}


class BundleError(RuntimeError):
    """Raised when the input bundle contract is not satisfied."""


def assert_safe_delete_target(target: Path) -> None:
    """Fail closed when a deletion boundary touches any literal protected root."""

    resolved = target.resolve()
    for protected in PROTECTED_ROOTS:
        protected = protected.resolve()
        if (
            resolved == protected
            or resolved.is_relative_to(protected)
            or protected.is_relative_to(resolved)
        ):
            raise BundleError(
                f"refusing deletion boundary that touches protected root: "
                f"target={resolved} protected={protected}"
            )


def _sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _numeric_directories(root: Path) -> tuple[str, ...]:
    if not root.is_dir():
        raise BundleError(f"missing source root: {root}")
    values = sorted(
        (child.name for child in root.iterdir() if child.is_dir() and child.name.isdigit()),
        key=int,
    )
    return tuple(values)


def _require_video_ids(root: Path, expected_ids: Sequence[str]) -> None:
    observed = _numeric_directories(root)
    expected = tuple(expected_ids)
    if observed != expected:
        raise BundleError(
            f"video-ID mismatch for {root}: expected={expected!r} observed={observed!r}"
        )


def _read_rows(path: Path, video_id: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise BundleError(f"rows.csv is empty for video {video_id}")
    required = {
        "video_id",
        "row_index",
        "time_seconds",
        "row_hz",
        "source_frame_position",
        "source_floor_frame_index",
        "source_ceil_frame_index",
        "source_interp_alpha",
        "arousal",
        "valence",
    }
    missing = required.difference(rows[0])
    if missing:
        raise BundleError(f"rows.csv missing columns for video {video_id}: {sorted(missing)}")
    for index, row in enumerate(rows):
        if row["video_id"] != video_id:
            raise BundleError(
                f"video identity mismatch in {path}: expected={video_id} got={row['video_id']}"
            )
        if int(row["row_index"]) != index:
            raise BundleError(
                f"noncontiguous row index in {path}: expected={index} got={row['row_index']}"
            )
        if float(row["row_hz"]) != 2.0:
            raise BundleError(f"non-2Hz row in {path} at index {index}")
    times = np.asarray([float(row["time_seconds"]) for row in rows], dtype=np.float64)
    if len(times) > 1 and not np.allclose(np.diff(times), 0.5, rtol=0.0, atol=1e-7):
        raise BundleError(f"non-0.5-second cadence in {path}")
    return rows


def _read_complete_status(path: Path, video_id: str, row_count: int) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("status") != "complete":
        raise BundleError(f"upstream status is not complete for video {video_id}: {path}")
    if str(value.get("video_id")) != video_id:
        raise BundleError(f"upstream status video mismatch for video {video_id}: {path}")
    if int(value.get("row_count", -1)) != row_count:
        raise BundleError(f"upstream status row mismatch for video {video_id}: {path}")
    return {
        "status": "complete",
        "video_id": video_id,
        "row_count": row_count,
    }


def _validate_npz(
    path: Path,
    rows: Sequence[dict[str, str]],
    video_id: str,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        keys = tuple(payload.files)
        missing = sorted(set(REQUIRED_NPZ_KEYS).difference(keys))
        if missing:
            raise BundleError(f"TRIBE payload missing keys for video {video_id}: {missing}")
        row_count = len(rows)
        cortical = payload["cortical_prediction"]
        if cortical.ndim != 2 or cortical.shape[0] != row_count:
            raise BundleError(
                f"cortical row mismatch for video {video_id}: "
                f"rows={row_count} shape={cortical.shape}"
            )
        if cortical.shape[1] != 20_484:
            raise BundleError(
                f"unexpected cortical width for video {video_id}: {cortical.shape[1]}"
            )
        if not np.isfinite(cortical).all():
            raise BundleError(f"nonfinite cortical value for video {video_id}")
        diagnostics = payload["temporal_diagnostics53"]
        if diagnostics.shape != (row_count, 53) or not np.isfinite(diagnostics).all():
            raise BundleError(
                f"invalid temporal diagnostics for video {video_id}: {diagnostics.shape}"
            )
        for csv_name, npz_name in CSV_NUMERIC_MATCHES.items():
            expected = np.asarray([float(row[csv_name]) for row in rows], dtype=np.float64)
            observed = np.asarray(payload[npz_name], dtype=np.float64)
            if observed.shape != expected.shape or not np.allclose(
                observed,
                expected,
                rtol=1e-6,
                atol=1e-6,
                equal_nan=False,
            ):
                raise BundleError(
                    f"row mapping mismatch for video {video_id}: {csv_name} vs {npz_name}"
                )
        arrays = {
            key: {"shape": list(payload[key].shape), "dtype": str(payload[key].dtype)}
            for key in keys
        }
    return {
        "video_id": video_id,
        "row_count": len(rows),
        "npz_keys": list(keys),
        "arrays": arrays,
        "row_mapping_verified": True,
        "cortical_finite": True,
        "temporal_diagnostics_finite": True,
    }


def _copy_verified(source: Path, destination: Path) -> dict[str, Any]:
    if source.name == FORBIDDEN_NAME:
        raise BundleError(f"forbidden source requested: {source}")
    if not source.is_file():
        raise BundleError(f"missing required source file: {source}")
    source_hash = _sha256(source)
    shutil.copy2(source, destination)
    destination_hash = _sha256(destination)
    if source_hash != destination_hash or source.stat().st_size != destination.stat().st_size:
        raise BundleError(f"copy verification failed: {source} -> {destination}")
    return {
        "source": str(source),
        "destination_name": destination.name,
        "bytes": destination.stat().st_size,
        "source_sha256": source_hash,
        "destination_sha256": destination_hash,
    }


def _assemble_video(
    arguments: tuple[str, str, str, str],
) -> dict[str, Any]:
    tribe_root_text, vjepa_root_text, per_video_text, video_id = arguments
    tribe_dir = Path(tribe_root_text) / video_id
    alignment_dir = Path(vjepa_root_text) / video_id
    destination = Path(per_video_text) / video_id
    destination.mkdir()

    rows = _read_rows(alignment_dir / "rows.csv", video_id)
    tribe_status = _read_complete_status(tribe_dir / "status.json", video_id, len(rows))
    alignment_status = _read_complete_status(alignment_dir / "status.json", video_id, len(rows))
    npz_validation = _validate_npz(
        tribe_dir / "tribe_v2_cortical_predictions.npz",
        rows,
        video_id,
    )
    copied: list[dict[str, Any]] = []
    for source_name, destination_name in TRIBE_FILES.items():
        copied.append(_copy_verified(tribe_dir / source_name, destination / destination_name))
    for source_name, destination_name in ALIGNMENT_FILES.items():
        copied.append(_copy_verified(alignment_dir / source_name, destination / destination_name))
    record = {
        **npz_validation,
        "tribe_status": tribe_status,
        "alignment_status": alignment_status,
        "identity_key": ["video_id", "row_index"],
        "position_contract": "rows.csv row_index i equals every row-shaped NPZ position i",
        "copied_files": copied,
        "forbidden_hidden_state_read": False,
    }
    _write_json(destination / "input-manifest.json", record)
    record["input_manifest_sha256"] = _sha256(destination / "input-manifest.json")
    return record


def _verify_video(arguments: tuple[str, str]) -> dict[str, Any]:
    root_text, video_id = arguments
    directory = Path(root_text) / "per_video" / video_id
    if (directory / FORBIDDEN_NAME).exists():
        raise BundleError(f"forbidden hidden-state file present for video {video_id}")
    record = json.loads((directory / "input-manifest.json").read_text())
    if record.get("video_id") != video_id:
        raise BundleError(f"input manifest identity mismatch for video {video_id}")
    rows = _read_rows(directory / "rows.csv", video_id)
    validation = _validate_npz(
        directory / "tribe_v2_cortical_predictions.npz",
        rows,
        video_id,
    )
    if validation["row_count"] != record.get("row_count"):
        raise BundleError(f"row-count manifest mismatch for video {video_id}")
    for copied in record.get("copied_files", []):
        destination = directory / copied["destination_name"]
        if _sha256(destination) != copied["destination_sha256"]:
            raise BundleError(f"bundle file hash mismatch: {destination}")
        source = Path(copied["source"])
        if _sha256(source) != copied["source_sha256"]:
            raise BundleError(f"protected source changed during construction: {source}")
    return {
        "video_id": video_id,
        "row_count": len(rows),
        "npz_schema": validation["npz_keys"],
        "input_manifest_sha256": _sha256(directory / "input-manifest.json"),
    }


def _seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(
                stat.S_IRUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )
    root.chmod(
        stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    )


def _aggregate_digest(records: Iterable[dict[str, Any]]) -> str:
    normalized = json.dumps(list(records), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode()).hexdigest()


def assemble_bundle(
    *,
    tribe_root: Path = DEFAULT_TRIBE_ROOT,
    vjepa_root: Path = DEFAULT_VJEPA_ROOT,
    output_root: Path = DEFAULT_BUNDLE_ROOT,
    expected_ids: Sequence[str] = EXPECTED_VIDEO_IDS,
    workers: int = 1,
) -> dict[str, Any]:
    """Build a new immutable bundle without modifying either protected source root."""

    tribe_root = tribe_root.resolve()
    vjepa_root = vjepa_root.resolve()
    output_root = output_root.resolve()
    expected = tuple(expected_ids)
    if workers < 1:
        raise BundleError("workers must be at least one")
    if output_root.exists():
        raise BundleError(f"refusing to overwrite existing bundle: {output_root}")
    if output_root in (tribe_root, vjepa_root):
        raise BundleError("output root must differ from both protected source roots")
    if output_root.is_relative_to(tribe_root) or output_root.is_relative_to(vjepa_root):
        raise BundleError("output root must not be inside a protected source root")

    _require_video_ids(tribe_root, expected)
    _require_video_ids(vjepa_root, expected)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent)
    )
    records: list[dict[str, Any]] = []
    schema: tuple[str, ...] | None = None
    try:
        per_video = temporary / "per_video"
        per_video.mkdir()
        arguments = [
            (str(tribe_root), str(vjepa_root), str(per_video), video_id) for video_id in expected
        ]
        if workers == 1:
            records = [_assemble_video(argument) for argument in arguments]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                records = list(executor.map(_assemble_video, arguments))
        records.sort(key=lambda value: int(value["video_id"]))
        for record in records:
            observed_schema = tuple(record["npz_keys"])
            if schema is None:
                schema = observed_schema
            elif observed_schema != schema:
                raise BundleError(f"NPZ schema differs for video {record['video_id']}")

        bundle_manifest = {
            "schema_version": "veatic21_neural_bridge_input_bundle_v1",
            "created_at": datetime.now(UTC).isoformat(),
            "video_ids": list(expected),
            "video_count": len(expected),
            "total_rows": sum(record["row_count"] for record in records),
            "source_tribe_root": str(tribe_root),
            "source_alignment_root": str(vjepa_root),
            "downstream_feature_source": "tribe_v2_cortical_predictions.npz:cortical_prediction",
            "row_and_label_authority": "rows.csv",
            "identity_key": ["video_id", "row_index"],
            "vjepa_hidden_states_copied": False,
            "vjepa_hidden_states_read_or_hashed": False,
            "vjepa_and_tribe_rerun": False,
            "assembly_workers": workers,
            "npz_schema": list(schema or ()),
            "records_digest_sha256": _aggregate_digest(records),
            "per_video": records,
        }
        _write_json(temporary / "bundle-manifest.json", bundle_manifest)
        (temporary / "README.md").write_text(
            "# VEATIC 2.1 Neural Bridge Input\n\n"
            "This sealed bundle is the sole downstream Phase 00+ input. Each numeric "
            "per-video folder co-locates one final TRIBE-v2 cortical prediction payload "
            "with its matching authoritative 2 Hz rows.csv and small allowlisted alignment "
            "metadata. V-JEPA hidden-state files are absent and forbidden. The protected "
            "staging roots remain unchanged.\n"
        )
        bundle_manifest["bundle_manifest_sha256"] = _sha256(temporary / "bundle-manifest.json")
        _write_json(
            temporary / "construction-audit.json",
            {
                "bundle_manifest_sha256": bundle_manifest["bundle_manifest_sha256"],
                "source_destination_hash_match": True,
                "video_id_match": True,
                "row_mapping_match": True,
                "cortical_finite": True,
                "forbidden_hidden_state_read_or_hash": False,
                "forbidden_hidden_state_copied": False,
                "protected_sources_modified": False,
            },
        )
        prepublication = verify_bundle(
            output_root=temporary,
            expected_ids=expected,
            workers=workers,
        )
        os.replace(temporary, output_root)
        _seal_tree(output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        **prepublication,
        "bundle_root": str(output_root),
        "atomic_publication": True,
        "sealed_read_only": True,
    }


def verify_bundle(
    *,
    output_root: Path = DEFAULT_BUNDLE_ROOT,
    expected_ids: Sequence[str] = EXPECTED_VIDEO_IDS,
    workers: int = 1,
) -> dict[str, Any]:
    """Verify bundle contents and re-hash each allowlisted protected source file."""

    root = output_root.resolve()
    manifest_path = root / "bundle-manifest.json"
    if not manifest_path.is_file():
        raise BundleError(f"missing bundle manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected = tuple(expected_ids)
    if workers < 1:
        raise BundleError("workers must be at least one")
    if tuple(manifest.get("video_ids", ())) != expected:
        raise BundleError("bundle manifest video IDs do not match expected IDs")
    _require_video_ids(root / "per_video", expected)
    arguments = [(str(root), video_id) for video_id in expected]
    if workers == 1:
        records = [_verify_video(argument) for argument in arguments]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(_verify_video, arguments))
    verified_by_id = {record["video_id"]: record for record in records}
    for manifest_record in manifest.get("per_video", []):
        video_id = manifest_record.get("video_id")
        if video_id not in verified_by_id:
            raise BundleError(f"unexpected per-video manifest record: {video_id}")
        if verified_by_id[video_id]["input_manifest_sha256"] != manifest_record.get(
            "input_manifest_sha256"
        ):
            raise BundleError(f"input-manifest hash mismatch for video {video_id}")
    schemas = {tuple(record["npz_schema"]) for record in records}
    if len(schemas) != 1 or list(next(iter(schemas))) != manifest.get("npz_schema"):
        raise BundleError("bundle NPZ schema is not globally consistent")
    total_rows = sum(int(record["row_count"]) for record in records)
    if total_rows != manifest.get("total_rows"):
        raise BundleError(
            f"bundle total-row mismatch: verified={total_rows} "
            f"manifest={manifest.get('total_rows')}"
        )
    return {
        "status": "pass",
        "bundle_root": str(root),
        "video_count": len(expected),
        "total_rows": total_rows,
        "bundle_manifest_sha256": _sha256(manifest_path),
        "forbidden_hidden_state_present": False,
        "protected_source_hashes_unchanged": True,
        "verification_workers": workers,
    }
