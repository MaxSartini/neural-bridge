"""Read-only adapter for the completed VEATIC V-JEPA 2.1/TRIBE v2 cache.

The compact cache is the authority for dense 2 Hz rows, labels, timestamps,
quality signals, and cortical predictions.  The historical VEATIC manifest is
used only for stable video/media/source-annotation identity.  This module never
resamples, rewrites, re-encodes, or repairs cache data.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IDENTITY_MANIFEST = (
    REPO_ROOT / "benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl"
)

EXPECTED_VIDEO_IDS = tuple(str(index) for index in range(124))
EXPECTED_TOTAL_ROWS = 20_657
ROW_HZ = 2.0
PREDICTION_WIDTH = 20_484
ROW_PLAN_SHA256 = "81a7491ab7653eb15dafc93ea9f31cd80a336bab614e6bec182b465f51e803b1"
MODEL_SHA256 = "ded8a1375bf118a74230ba6f2baef924e2cdbd508870fcddc7dd950293ba156a"

RUN_SCHEMA = "veatic_compact_mlx_tribe_v2_run_v1"
POSTPASS_SCHEMA = "veatic_compact_mlx_tribe_v2_postpass_v1"
UPSTREAM_SCHEMA = "veatic_dense_vjepa21_vitg_compact_temporal_v1"
UPLOAD_SCHEMA = "neural_bridge_h100_per_video_upload_v1"
PAYLOAD_SCHEMA = "neural_bridge_h100_per_video_payload_v1"
ENCODE_POLICY = "exact_2hz_native_label_support_no_extrapolation"
SELECTED_STATE_INDICES = (0, 2, 4, 6, 8, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 34, 36, 38, 40)

ROW_ARRAY_KEYS = (
    "time_seconds",
    "sample_frame_indices",
    "sample_time_seconds",
    "source_frame_position",
    "source_floor_frame_index",
    "source_ceil_frame_index",
    "source_interp_alpha",
    "source_arousal",
    "source_valence",
    "arousal",
    "valence",
    "luma_mean",
    "luma_std",
    "frame_luma_std_mean",
    "motion_absdiff_mean",
    "black_frame_fraction",
    "duplicate_frame_fraction",
    "quality_black_frame_flag",
    "quality_duplicate_frame_flag",
    "quality_exclusion_flag",
    "quality_weight_suggested",
    "cortical_prediction",
    "tribe_grouped_video_feature",
    "temporal_diagnostics53",
)
PASSTHROUGH_KEYS = tuple(
    key
    for key in ROW_ARRAY_KEYS
    if key not in {"cortical_prediction", "tribe_grouped_video_feature"}
)

# These are the scalar, row-aligned arrays emitted by the sealed VEATIC 2.1
# encoder/postpass pipeline.  Keep this contract explicit: accepting an
# arbitrary numeric dtype or an extra trailing dimension here can silently
# change downstream pandas/numpy broadcasting and feature construction.
ROW_SCALAR_DTYPES = MappingProxyType(
    {
        "time_seconds": np.dtype(np.float32),
        "source_frame_position": np.dtype(np.float32),
        "source_floor_frame_index": np.dtype(np.int32),
        "source_ceil_frame_index": np.dtype(np.int32),
        "source_interp_alpha": np.dtype(np.float32),
        "source_arousal": np.dtype(np.float32),
        "source_valence": np.dtype(np.float32),
        "arousal": np.dtype(np.float32),
        "valence": np.dtype(np.float32),
        "luma_mean": np.dtype(np.float32),
        "luma_std": np.dtype(np.float32),
        "frame_luma_std_mean": np.dtype(np.float32),
        "motion_absdiff_mean": np.dtype(np.float32),
        "black_frame_fraction": np.dtype(np.float32),
        "duplicate_frame_fraction": np.dtype(np.float32),
        "quality_black_frame_flag": np.dtype(np.uint8),
        "quality_duplicate_frame_flag": np.dtype(np.uint8),
        "quality_exclusion_flag": np.dtype(np.uint8),
        "quality_weight_suggested": np.dtype(np.float32),
    }
)

# Inclusive physical/semantic domains.  Luma and absolute-difference signals
# are calculated on 8-bit decoded frames, so [0, 255] is the conservative
# closed domain even where a tighter theoretical standard-deviation bound
# exists.  ``None`` means that only the finite lower bound is prescribed.
ROW_SCALAR_BOUNDS = MappingProxyType(
    {
        "time_seconds": (0.0, None),
        "source_frame_position": (0.0, None),
        "source_floor_frame_index": (0, None),
        "source_ceil_frame_index": (0, None),
        "source_interp_alpha": (0.0, 1.0),
        "source_arousal": (-1.0, 1.0),
        "source_valence": (-1.0, 1.0),
        "arousal": (-1.0, 1.0),
        "valence": (-1.0, 1.0),
        "luma_mean": (0.0, 255.0),
        "luma_std": (0.0, 255.0),
        "frame_luma_std_mean": (0.0, 255.0),
        "motion_absdiff_mean": (0.0, 255.0),
        "black_frame_fraction": (0.0, 1.0),
        "duplicate_frame_fraction": (0.0, 1.0),
        "quality_black_frame_flag": (0, 1),
        "quality_duplicate_frame_flag": (0, 1),
        "quality_exclusion_flag": (0, 1),
        "quality_weight_suggested": (0.0, 1.0),
    }
)
QUALITY_FLAG_KEYS = (
    "quality_black_frame_flag",
    "quality_duplicate_frame_flag",
    "quality_exclusion_flag",
)


class Veatic21CompactCacheError(RuntimeError):
    """Raised when any cache or provenance gate fails."""


@dataclass(frozen=True)
class Veatic21VideoIdentity:
    video_id: str
    media_path: str
    source_annotation: Mapping[str, str]


@dataclass(frozen=True)
class Veatic21VideoProvenance:
    postpass_manifest_path: Path
    postpass_npz_path: Path
    postpass_npz_sha256: str
    upstream_manifest_path: Path
    upstream_npz_path: Path
    upstream_npz_sha256: str
    row_plan_sha256: str
    model_sha256: str
    video_sha256: str


@dataclass(frozen=True)
class Veatic21DenseVideo:
    """One authoritative dense-row video block.

    ``columns["predictions"]`` and :attr:`predictions` are aliases of the
    physical ``cortical_prediction`` array.  No dtype conversion or duplicate
    prediction array is made.
    """

    video_id: str
    columns: Mapping[str, np.ndarray]
    identity: Veatic21VideoIdentity
    provenance: Veatic21VideoProvenance

    @property
    def row_count(self) -> int:
        return int(self.columns["time_seconds"].shape[0])

    @property
    def predictions(self) -> np.ndarray:
        return self.columns["cortical_prediction"]

    @property
    def cortical_prediction(self) -> np.ndarray:
        return self.columns["cortical_prediction"]

    @property
    def time_seconds(self) -> np.ndarray:
        return self.columns["time_seconds"]

    @property
    def arousal(self) -> np.ndarray:
        return self.columns["arousal"]

    @property
    def valence(self) -> np.ndarray:
        return self.columns["valence"]


@dataclass(frozen=True)
class Veatic21ValidationReport:
    status: str
    video_ids: tuple[str, ...]
    video_count: int
    total_rows: int
    row_hz: float
    prediction_width: int
    row_plan_sha256: str
    model_sha256: str
    dataset_fingerprint_sha256: str
    row_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "video_ids": list(self.video_ids),
            "video_count": self.video_count,
            "total_rows": self.total_rows,
            "row_hz": self.row_hz,
            "prediction_width": self.prediction_width,
            "row_plan_sha256": self.row_plan_sha256,
            "model_sha256": self.model_sha256,
            "dataset_fingerprint_sha256": self.dataset_fingerprint_sha256,
            "row_counts": dict(self.row_counts),
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Veatic21CompactCacheError(message)


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required JSON file is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Veatic21CompactCacheError(f"invalid JSON at {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError as exc:
        raise Veatic21CompactCacheError(f"could not hash {path}: {exc}") from exc
    return digest.hexdigest()


def _natural_video_id(video_id: str) -> tuple[int, str]:
    try:
        return int(video_id), video_id
    except ValueError:
        return 2**63 - 1, video_id


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


class Veatic21CompactCache:
    """Fail-closed, read-only view over the compact VEATIC 2.1 cache."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        upstream_root: str | Path,
        identity_manifest_path: str | Path = DEFAULT_IDENTITY_MANIFEST,
        expected_video_ids: Sequence[str] = EXPECTED_VIDEO_IDS,
        expected_total_rows: int = EXPECTED_TOTAL_ROWS,
        verify_checksums: bool = True,
    ) -> None:
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.per_video_root = self.cache_root / "per_video"
        self.upstream_root = Path(upstream_root).expanduser().resolve()
        self.identity_manifest_path = Path(identity_manifest_path).expanduser().resolve()
        self.expected_video_ids = tuple(str(value) for value in expected_video_ids)
        self.expected_total_rows = int(expected_total_rows)
        self.verify_checksums = bool(verify_checksums)
        _require(bool(self.expected_video_ids), "expected_video_ids cannot be empty")
        _require(
            len(set(self.expected_video_ids)) == len(self.expected_video_ids),
            "expected_video_ids contains duplicates",
        )
        _require(self.expected_total_rows > 0, "expected_total_rows must be positive")

    def validate(self) -> Veatic21ValidationReport:
        """Validate the full cache and return an immutable dataset seal."""
        try:
            self._validate_run_and_ids()
            identities = self._load_identities()
            total_rows = 0
            row_counts: dict[str, int] = {}
            fingerprint = hashlib.sha256()
            for video_id in self.expected_video_ids:
                block = self._load_video(video_id, identities[video_id])
                row_counts[video_id] = block.row_count
                total_rows += block.row_count
                fingerprint.update(video_id.encode("utf-8"))
                fingerprint.update(b"\0")
                fingerprint.update(block.provenance.postpass_npz_sha256.encode("ascii"))
                fingerprint.update(b"\0")
                fingerprint.update(block.provenance.upstream_npz_sha256.encode("ascii"))
                fingerprint.update(b"\n")
                del block
            _require(
                total_rows == self.expected_total_rows,
                f"dense row total {total_rows} != expected {self.expected_total_rows}",
            )
            return Veatic21ValidationReport(
                status="pass",
                video_ids=self.expected_video_ids,
                video_count=len(self.expected_video_ids),
                total_rows=total_rows,
                row_hz=ROW_HZ,
                prediction_width=PREDICTION_WIDTH,
                row_plan_sha256=ROW_PLAN_SHA256,
                model_sha256=MODEL_SHA256,
                dataset_fingerprint_sha256=fingerprint.hexdigest(),
                row_counts=MappingProxyType(row_counts),
            )
        except Veatic21CompactCacheError:
            raise
        except Exception as exc:
            raise Veatic21CompactCacheError(f"cache validation failed closed: {exc}") from exc

    def load_video(self, video_id: str) -> Veatic21DenseVideo:
        """Load and validate one video without resampling or dtype conversion."""
        video_id = str(video_id)
        try:
            _require(video_id in self.expected_video_ids, f"unexpected video_id: {video_id}")
            self._validate_run_and_ids()
            identities = self._load_identities()
            return self._load_video(video_id, identities[video_id])
        except Veatic21CompactCacheError:
            raise
        except Exception as exc:
            raise Veatic21CompactCacheError(
                f"video {video_id} failed closed during compact-cache load: {exc}"
            ) from exc

    def iter_videos(self) -> Iterator[Veatic21DenseVideo]:
        """Stream validated video blocks in numeric video-id order."""
        self._validate_run_and_ids()
        identities = self._load_identities()
        for video_id in self.expected_video_ids:
            yield self._load_video(video_id, identities[video_id])

    def _validate_run_and_ids(self) -> None:
        _require(self.cache_root.is_dir(), f"compact cache root is missing: {self.cache_root}")
        _require(self.per_video_root.is_dir(), f"per_video directory is missing: {self.per_video_root}")
        _require(self.upstream_root.is_dir(), f"upstream V-JEPA cache root is missing: {self.upstream_root}")
        run = _read_json(self.per_video_root / "run_status.json")
        expected_count = len(self.expected_video_ids)
        _require(run.get("schema_version") == RUN_SCHEMA, "unexpected compact run schema")
        _require(run.get("expected_videos") == expected_count, "run expected_videos mismatch")
        _require(run.get("completed_videos") == expected_count, "compact run is not complete")
        _require(run.get("failures") == {}, "compact run reports failures")
        _require(isinstance(run.get("updated_at"), str) and run["updated_at"], "run timestamp is missing")

        actual_ids = {
            item.name
            for item in self.per_video_root.iterdir()
            if item.is_dir()
        }
        expected_ids = set(self.expected_video_ids)
        missing = sorted(expected_ids - actual_ids, key=_natural_video_id)
        extra = sorted(actual_ids - expected_ids, key=_natural_video_id)
        _require(not missing and not extra, f"video-id coverage mismatch: missing={missing} extra={extra}")

        # Do not trust the aggregate run marker alone.  Gate every per-video
        # completion record and the canonical total before exposing even one
        # video block.
        manifest_row_total = 0
        for video_id in self.expected_video_ids:
            video_root = self.per_video_root / video_id
            manifest = _read_json(video_root / "manifest.json")
            status = _read_json(video_root / "status.json")
            _require(manifest == status, f"postpass manifest/status mismatch for {video_id}")
            _require(manifest.get("schema_version") == POSTPASS_SCHEMA, f"postpass schema mismatch for {video_id}")
            _require(manifest.get("status") == "complete", f"postpass is incomplete for {video_id}")
            _require(str(manifest.get("video_id")) == video_id, f"postpass video_id mismatch for {video_id}")
            row_count = manifest.get("row_count")
            _require(
                isinstance(row_count, int) and not isinstance(row_count, bool) and row_count > 0,
                f"invalid postpass row_count for {video_id}",
            )
            manifest_row_total += row_count
        _require(
            manifest_row_total == self.expected_total_rows,
            f"postpass manifest row total {manifest_row_total} != expected {self.expected_total_rows}",
        )

    def _load_identities(self) -> dict[str, Veatic21VideoIdentity]:
        _require(
            self.identity_manifest_path.is_file(),
            f"VEATIC identity manifest is missing: {self.identity_manifest_path}",
        )
        identities: dict[str, Veatic21VideoIdentity] = {}
        try:
            with self.identity_manifest_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    _require(isinstance(row, dict), f"identity row {line_number} is not an object")
                    _require(str(row.get("dataset", "")).lower() == "veatic", f"identity row {line_number} is not VEATIC")
                    video_id = str(row.get("video_id", ""))
                    _require(video_id in self.expected_video_ids, f"unexpected identity video_id {video_id!r}")
                    media_path = row.get("media_path")
                    source = row.get("source_annotation")
                    _require(isinstance(media_path, str) and media_path, f"missing media_path for {video_id}")
                    _require(isinstance(source, dict), f"missing source_annotation for {video_id}")
                    source_strings = {
                        str(key): str(value)
                        for key, value in source.items()
                        if isinstance(value, str) and value
                    }
                    _require(
                        {"arousal", "valence"} <= set(source_strings),
                        f"source annotation identity is incomplete for {video_id}",
                    )
                    candidate = Veatic21VideoIdentity(
                        video_id=video_id,
                        media_path=media_path,
                        source_annotation=MappingProxyType(source_strings),
                    )
                    previous = identities.get(video_id)
                    _require(previous is None or previous == candidate, f"unstable identity metadata for {video_id}")
                    identities[video_id] = candidate
        except Veatic21CompactCacheError:
            raise
        except (OSError, ValueError, TypeError) as exc:
            raise Veatic21CompactCacheError(
                f"invalid VEATIC identity manifest {self.identity_manifest_path}: {exc}"
            ) from exc
        missing = sorted(set(self.expected_video_ids) - set(identities), key=_natural_video_id)
        extra = sorted(set(identities) - set(self.expected_video_ids), key=_natural_video_id)
        _require(not missing and not extra, f"identity coverage mismatch: missing={missing} extra={extra}")
        return identities

    def _load_video(
        self,
        video_id: str,
        identity: Veatic21VideoIdentity,
    ) -> Veatic21DenseVideo:
        video_root = self.per_video_root / video_id
        npz_path = video_root / "tribe_v2_cortical_predictions.npz"
        manifest_path = video_root / "manifest.json"
        status_path = video_root / "status.json"
        manifest = _read_json(manifest_path)
        status = _read_json(status_path)
        _require(manifest == status, f"postpass manifest/status mismatch for {video_id}")
        _require(manifest.get("schema_version") == POSTPASS_SCHEMA, f"postpass schema mismatch for {video_id}")
        _require(manifest.get("status") == "complete", f"postpass is incomplete for {video_id}")
        _require(str(manifest.get("video_id")) == video_id, f"postpass video_id mismatch for {video_id}")
        _require(npz_path.is_file(), f"postpass NPZ is missing for {video_id}")

        expected_upstream_npz = self.upstream_root / video_id / "vjepa21_hidden_states.npz"
        self._require_declared_path(manifest, "input_cache", expected_upstream_npz, video_id)
        self._require_declared_path(manifest, "output_cache", npz_path, video_id)

        arrays = self._load_arrays(npz_path, video_id)
        row_count = self._validate_arrays(arrays, video_id)
        _require(manifest.get("row_count") == row_count, f"postpass row_count mismatch for {video_id}")
        _require(manifest.get("cortical_shape") == [row_count, PREDICTION_WIDTH], f"postpass cortical_shape mismatch for {video_id}")
        _require(manifest.get("grouped_shape") == [row_count, 2, 1408], f"postpass grouped_shape mismatch for {video_id}")
        runtime = manifest.get("runtime_seconds")
        _require(isinstance(runtime, (int, float)) and math.isfinite(runtime) and runtime >= 0, f"invalid runtime provenance for {video_id}")
        _require(isinstance(manifest.get("finished_at"), str) and manifest["finished_at"], f"missing finish provenance for {video_id}")

        upstream_manifest, upstream_sha = self._validate_upstream(
            video_id=video_id,
            row_count=row_count,
            arrays=arrays,
        )
        output_sha = _sha256_file(npz_path) if self.verify_checksums else "checksum_not_requested"

        row_index = _readonly(np.arange(row_count, dtype=np.int32))
        columns: dict[str, np.ndarray] = {"row_index": row_index, **arrays}
        # Logical compatibility alias: exactly the same ndarray object.
        columns["predictions"] = columns["cortical_prediction"]
        provenance = Veatic21VideoProvenance(
            postpass_manifest_path=manifest_path,
            postpass_npz_path=npz_path,
            postpass_npz_sha256=output_sha,
            upstream_manifest_path=self.upstream_root / video_id / "manifest.json",
            upstream_npz_path=expected_upstream_npz,
            upstream_npz_sha256=upstream_sha,
            row_plan_sha256=str(upstream_manifest["row_plan_sha256"]),
            model_sha256=str(upstream_manifest["model_sha256"]),
            video_sha256=str(upstream_manifest["video_sha256"]),
        )
        return Veatic21DenseVideo(
            video_id=video_id,
            columns=MappingProxyType(columns),
            identity=identity,
            provenance=provenance,
        )

    @staticmethod
    def _require_declared_path(
        manifest: Mapping[str, Any],
        key: str,
        expected: Path,
        video_id: str,
    ) -> None:
        declared = manifest.get(key)
        _require(isinstance(declared, str) and declared, f"missing {key} provenance for {video_id}")
        declared_path = Path(declared).expanduser()
        _require(declared_path.is_absolute(), f"{key} provenance is not absolute for {video_id}")
        _require(
            declared_path.resolve() == expected.resolve(),
            f"{key} provenance mismatch for {video_id}: {declared_path} != {expected}",
        )

    @staticmethod
    def _load_arrays(path: Path, video_id: str) -> dict[str, np.ndarray]:
        try:
            with np.load(path, allow_pickle=False) as bundle:
                missing = sorted((set(ROW_ARRAY_KEYS) | {"selected_state_indices"}) - set(bundle.files))
                _require(not missing, f"postpass NPZ keys missing for {video_id}: {missing}")
                arrays = {key: _readonly(bundle[key]) for key in bundle.files}
        except Veatic21CompactCacheError:
            raise
        except Exception as exc:
            raise Veatic21CompactCacheError(f"could not load postpass NPZ for {video_id}: {exc}") from exc
        return arrays

    @staticmethod
    def _validate_arrays(arrays: Mapping[str, np.ndarray], video_id: str) -> int:
        time_seconds = arrays["time_seconds"]
        _require(
            time_seconds.ndim == 1 and time_seconds.shape[0] > 0,
            f"row scalar shape mismatch for {video_id}:time_seconds",
        )
        _require(time_seconds.dtype == np.float32, f"time_seconds dtype mismatch for {video_id}")
        row_count = int(time_seconds.shape[0])
        for key in ROW_ARRAY_KEYS:
            value = arrays[key]
            _require(value.ndim >= 1 and value.shape[0] == row_count, f"row count mismatch for {video_id}:{key}")
            _require(np.issubdtype(value.dtype, np.number), f"non-numeric array for {video_id}:{key}")
            _require(np.isfinite(value).all(), f"non-finite values for {video_id}:{key}")

        for key, expected_dtype in ROW_SCALAR_DTYPES.items():
            value = arrays[key]
            _require(
                value.shape == (row_count,),
                f"row scalar shape mismatch for {video_id}:{key}; "
                f"expected {(row_count,)}, got {value.shape}",
            )
            _require(
                value.dtype == expected_dtype,
                f"row scalar dtype mismatch for {video_id}:{key}; "
                f"expected {expected_dtype}, got {value.dtype}",
            )
            if key in QUALITY_FLAG_KEYS:
                _require(
                    bool(np.all((value == np.uint8(0)) | (value == np.uint8(1)))),
                    f"quality flag boolean-domain mismatch for {video_id}:{key}",
                )
            lower, upper = ROW_SCALAR_BOUNDS[key]
            _require(
                lower is None or bool(np.all(value >= lower)),
                f"row scalar domain mismatch for {video_id}:{key}; values below {lower}",
            )
            _require(
                upper is None or bool(np.all(value <= upper)),
                f"row scalar domain mismatch for {video_id}:{key}; values above {upper}",
            )

        expected_times = np.arange(row_count, dtype=np.float64) / ROW_HZ
        _require(
            np.allclose(time_seconds.astype(np.float64), expected_times, rtol=0.0, atol=1e-7),
            f"timestamps are not the exact 2 Hz grid for {video_id}",
        )
        _require(arrays["sample_frame_indices"].shape == (row_count, 64), f"sample frame shape mismatch for {video_id}")
        _require(arrays["sample_frame_indices"].dtype == np.int32, f"sample frame dtype mismatch for {video_id}")
        _require(np.all(arrays["sample_frame_indices"] >= 0), f"negative sample frame index for {video_id}")
        sample_times = arrays["sample_time_seconds"]
        _require(sample_times.shape == (row_count, 64), f"sample time shape mismatch for {video_id}")
        _require(sample_times.dtype == np.float32, f"sample time dtype mismatch for {video_id}")
        _require(np.all(sample_times >= -1e-7), f"negative sample time for {video_id}")
        _require(np.all(sample_times <= time_seconds[:, None] + 1e-7), f"future sample time for {video_id}")
        _require(np.allclose(sample_times[:, -1], time_seconds, rtol=0.0, atol=1e-7), f"causal window endpoint mismatch for {video_id}")
        _require(np.all(np.diff(sample_times, axis=1) >= -1e-7), f"sample times are not monotonic for {video_id}")

        cortical = arrays["cortical_prediction"]
        _require(cortical.shape == (row_count, PREDICTION_WIDTH), f"cortical width mismatch for {video_id}")
        _require(cortical.dtype == np.float16, f"cortical dtype mismatch for {video_id}")
        grouped = arrays["tribe_grouped_video_feature"]
        _require(grouped.shape == (row_count, 2, 1408), f"grouped feature shape mismatch for {video_id}")
        _require(grouped.dtype == np.float16, f"grouped feature dtype mismatch for {video_id}")
        diagnostics = arrays["temporal_diagnostics53"]
        _require(diagnostics.shape == (row_count, 53), f"diagnostic shape mismatch for {video_id}")
        _require(diagnostics.dtype == np.float32, f"diagnostic dtype mismatch for {video_id}")
        selected = arrays["selected_state_indices"]
        _require(selected.shape == (20,), f"selected-state shape mismatch for {video_id}")
        _require(selected.dtype == np.int16, f"selected-state dtype mismatch for {video_id}")
        _require(tuple(int(value) for value in selected) == SELECTED_STATE_INDICES, f"selected states mismatch for {video_id}")

        _require(np.array_equal(arrays["source_arousal"], arrays["arousal"]), f"arousal alias mismatch for {video_id}")
        _require(np.array_equal(arrays["source_valence"], arrays["valence"]), f"valence alias mismatch for {video_id}")
        floor = arrays["source_floor_frame_index"]
        ceil = arrays["source_ceil_frame_index"]
        alpha = arrays["source_interp_alpha"]
        position = arrays["source_frame_position"]
        _require(
            np.all(ceil >= floor) and np.all((ceil - floor) <= 1),
            f"source frame bracket mismatch for {video_id}",
        )
        _require(np.all((alpha >= -1e-7) & (alpha <= 1.0 + 1e-7)), f"source interpolation alpha mismatch for {video_id}")
        _require(
            np.allclose(position, floor.astype(np.float32) + alpha, rtol=0.0, atol=2e-5),
            f"source frame position provenance mismatch for {video_id}",
        )
        return row_count

    def _validate_upstream(
        self,
        *,
        video_id: str,
        row_count: int,
        arrays: Mapping[str, np.ndarray],
    ) -> tuple[dict[str, Any], str]:
        root = self.upstream_root / video_id
        manifest_path = root / "manifest.json"
        status_path = root / "status.json"
        npz_path = root / "vjepa21_hidden_states.npz"
        payload_path = root / "_PAYLOAD_SHA256.json"
        marker_path = root / "_UPLOAD_COMPLETE.json"
        manifest = _read_json(manifest_path)
        status = _read_json(status_path)
        _require(manifest == status, f"upstream manifest/status mismatch for {video_id}")
        _require(manifest.get("schema") == UPSTREAM_SCHEMA, f"upstream schema mismatch for {video_id}")
        _require(manifest.get("status") == "complete", f"upstream cache is incomplete for {video_id}")
        _require(str(manifest.get("video_id")) == video_id, f"upstream video_id mismatch for {video_id}")
        _require(manifest.get("video_name") == f"{video_id}.mp4", f"upstream video name mismatch for {video_id}")
        _require(manifest.get("row_count") == row_count, f"upstream row_count mismatch for {video_id}")
        _require(manifest.get("row_hz") == ROW_HZ, f"upstream row rate mismatch for {video_id}")
        _require(manifest.get("image_size") == 256, f"upstream image size mismatch for {video_id}")
        _require(manifest.get("decode_hz") == 16.0, f"upstream decode rate mismatch for {video_id}")
        _require(manifest.get("frames_per_clip") == 64, f"upstream frame-window mismatch for {video_id}")
        _require(manifest.get("causal_sample_span_seconds") == 3.9375, f"upstream causal span mismatch for {video_id}")
        _require(manifest.get("dtype") == "float16", f"upstream dtype mismatch for {video_id}")
        _require(manifest.get("feature_shape") == [row_count, 20, 1, 1408], f"upstream feature shape mismatch for {video_id}")
        _require(manifest.get("temporal_diagnostics53_shape") == [row_count, 53], f"upstream diagnostic shape mismatch for {video_id}")
        _require(manifest.get("selected_state_indices") == list(SELECTED_STATE_INDICES), f"upstream selected-state mismatch for {video_id}")
        _require(manifest.get("row_plan_sha256") == ROW_PLAN_SHA256, f"row-plan provenance mismatch for {video_id}")
        _require(manifest.get("model_sha256") == MODEL_SHA256, f"model provenance mismatch for {video_id}")
        _require(manifest.get("encode_policy") == ENCODE_POLICY, f"encode policy mismatch for {video_id}")
        _require(manifest.get("persisted_full_temporal_tensors") is False, f"upstream is not compact for {video_id}")
        _require(manifest.get("vjepa_only") is True, f"upstream V-JEPA-only provenance mismatch for {video_id}")
        _require(manifest.get("tribe_imported_or_run") is False, f"upstream TRIBE provenance mismatch for {video_id}")
        _require(manifest.get("time_start_seconds") == 0.0, f"upstream start time mismatch for {video_id}")
        _require(
            abs(float(manifest.get("time_end_seconds")) - (row_count - 1) / ROW_HZ) <= 1e-7,
            f"upstream end time mismatch for {video_id}",
        )
        _require(isinstance(manifest.get("video_sha256"), str) and len(manifest["video_sha256"]) == 64, f"video provenance missing for {video_id}")

        upstream_sha = self._validate_payload_commit(
            video_id=video_id,
            root=root,
            payload_path=payload_path,
            marker_path=marker_path,
            npz_path=npz_path,
        )
        _require(manifest.get("cache_sha256") == upstream_sha, f"upstream NPZ checksum mismatch for {video_id}")

        try:
            with np.load(npz_path, allow_pickle=False) as upstream:
                missing = sorted((set(PASSTHROUGH_KEYS) | {"selected_state_indices"}) - set(upstream.files))
                _require(not missing, f"upstream passthrough keys missing for {video_id}: {missing}")
                for key in (*PASSTHROUGH_KEYS, "selected_state_indices"):
                    _require(
                        np.array_equal(upstream[key], arrays[key]),
                        f"postpass changed authoritative upstream array for {video_id}:{key}",
                    )
        except Veatic21CompactCacheError:
            raise
        except Exception as exc:
            raise Veatic21CompactCacheError(f"could not validate upstream NPZ for {video_id}: {exc}") from exc
        return manifest, upstream_sha

    def _validate_payload_commit(
        self,
        *,
        video_id: str,
        root: Path,
        payload_path: Path,
        marker_path: Path,
        npz_path: Path,
    ) -> str:
        payload = _read_json(payload_path)
        marker = _read_json(marker_path)
        _require(payload.get("schema_version") == PAYLOAD_SCHEMA, f"payload schema mismatch for {video_id}")
        _require(str(payload.get("video_id")) == video_id, f"payload video_id mismatch for {video_id}")
        files = payload.get("files")
        _require(isinstance(files, list) and files, f"payload file list is missing for {video_id}")
        _require(payload.get("file_count") == len(files), f"payload file_count mismatch for {video_id}")

        seen: set[str] = set()
        total_bytes = 0
        npz_sha = ""
        for entry in files:
            _require(isinstance(entry, dict), f"invalid payload entry for {video_id}")
            relative = entry.get("path")
            _require(isinstance(relative, str), f"invalid payload path for {video_id}")
            pure = PurePosixPath(relative)
            _require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe payload path for {video_id}: {relative}")
            _require(relative not in seen, f"duplicate payload path for {video_id}: {relative}")
            seen.add(relative)
            path = root.joinpath(*pure.parts)
            _require(path.is_file(), f"payload file is missing for {video_id}: {relative}")
            size = path.stat().st_size
            _require(entry.get("bytes") == size, f"payload byte count mismatch for {video_id}:{relative}")
            total_bytes += size
            declared_sha = entry.get("sha256")
            _require(isinstance(declared_sha, str) and len(declared_sha) == 64, f"payload checksum missing for {video_id}:{relative}")
            actual_sha = _sha256_file(path) if self.verify_checksums else declared_sha
            _require(actual_sha == declared_sha, f"payload checksum mismatch for {video_id}:{relative}")
            if path.resolve() == npz_path.resolve():
                npz_sha = actual_sha
        _require(
            {"manifest.json", "preprocessing.json", "rows.csv", "status.json", "vjepa21_hidden_states.npz"} <= seen,
            f"upstream payload is incomplete for {video_id}",
        )
        _require(payload.get("total_bytes") == total_bytes, f"payload total_bytes mismatch for {video_id}")
        _require(bool(npz_sha), f"upstream NPZ is not sealed by payload manifest for {video_id}")

        payload_bytes = payload_path.read_bytes()
        payload_sha = hashlib.sha256(payload_bytes).hexdigest()
        _require(marker.get("schema_version") == UPLOAD_SCHEMA, f"upload marker schema mismatch for {video_id}")
        _require(marker.get("status") == "complete", f"upload marker is incomplete for {video_id}")
        _require(str(marker.get("video_id")) == video_id, f"upload marker video_id mismatch for {video_id}")
        _require(marker.get("payload_manifest") == "_PAYLOAD_SHA256.json", f"upload marker payload name mismatch for {video_id}")
        _require(marker.get("payload_manifest_bytes") == len(payload_bytes), f"upload marker byte count mismatch for {video_id}")
        _require(marker.get("payload_manifest_sha256") == payload_sha, f"upload marker checksum mismatch for {video_id}")
        _require(marker.get("payload_file_count") == len(files), f"upload marker file count mismatch for {video_id}")
        _require(marker.get("payload_total_bytes") == total_bytes, f"upload marker total bytes mismatch for {video_id}")
        return npz_sha


__all__ = [
    "DEFAULT_IDENTITY_MANIFEST",
    "EXPECTED_TOTAL_ROWS",
    "EXPECTED_VIDEO_IDS",
    "MODEL_SHA256",
    "PREDICTION_WIDTH",
    "ROW_HZ",
    "ROW_PLAN_SHA256",
    "Veatic21CompactCache",
    "Veatic21CompactCacheError",
    "Veatic21DenseVideo",
    "Veatic21ValidationReport",
    "Veatic21VideoIdentity",
    "Veatic21VideoProvenance",
]
