"""Strict access to the canonical VEATIC 2.1 substrate.

Feature loading deliberately never reads label arrays.  Labels cross their
own auditable boundary and come only from each canonical ``rows.csv`` file.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ..provenance import TreeDigest
from .contracts import CANONICAL_DATASET, FeatureRows, LabelRows, SubstrateIdentity

_VIDEO_IDS = tuple(str(video_id) for video_id in range(CANONICAL_DATASET.video_count))
_ROW_PLAN_SHA256 = "81a7491ab7653eb15dafc93ea9f31cd80a336bab614e6bec182b465f51e803b1"
_SOURCE_TREE_SHA256 = "ec04ac1341d9ffcfc50ca8f13ff88cf0d427eb36f1d033dc2486dbb1b368b174"
_ENCODER_MODEL_SHA256 = "ded8a1375bf118a74230ba6f2baef924e2cdbd508870fcddc7dd950293ba156a"

_ROW_COLUMNS = (
    "video_id",
    "video_name",
    "video_relpath",
    "arousal_relpath",
    "valence_relpath",
    "row_index",
    "time_seconds",
    "row_hz",
    "clip_start_seconds",
    "clip_end_seconds",
    "native_label_fps",
    "native_label_frame_count",
    "source_frame_position",
    "source_floor_frame_index",
    "source_ceil_frame_index",
    "source_interp_alpha",
    "source_arousal",
    "source_valence",
    "source_match_quality",
    "encode_policy",
    "arousal",
    "valence",
)

_LABEL_KEYS = frozenset({"arousal", "valence", "source_arousal", "source_valence"})
_SHARED_METADATA_KEYS = (
    "time_seconds",
    "sample_frame_indices",
    "sample_time_seconds",
    "selected_state_indices",
    "source_frame_position",
    "source_floor_frame_index",
    "source_ceil_frame_index",
    "source_interp_alpha",
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
    "temporal_diagnostics53",
)
_REPRESENTATION_LAYOUTS: Mapping[str, tuple[int, np.dtype[Any]]] = MappingProxyType(
    {
        "tribe_cortical": (20_484, np.dtype(np.float16)),
        "diagnostics_only": (53, np.dtype(np.float32)),
    }
)


@dataclass(frozen=True)
class _ArtifactSpec:
    artifact_id: str
    files: int
    relative_path: str
    sha256_tree: str
    size_bytes: int


_VJEPA = _ArtifactSpec(
    artifact_id="veatic-2.1-vjepa-2.1-compact-20260716",
    files=868,
    relative_path="features/veatic-2.1/vjepa-2.1/compact-20260716",
    sha256_tree="cccc46f6559f8fadf83d0e1e140426349406a5778c5ccb039f782e4dcb808311",
    size_bytes=1_102_348_297,
)
_TRIBE = _ArtifactSpec(
    artifact_id="veatic-2.1-tribe-v2-compact-20260716",
    files=373,
    relative_path="features/veatic-2.1/tribe-v2/compact-20260716",
    sha256_tree="0d4adc27dd9d226de87d0cfc4df92de14cb7450de6671857e0665418ad26f6dd",
    size_bytes=866_111_964,
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"required VEATIC substrate file is missing: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def _require_equal(actual: object, expected: object, field: str, path: Path) -> None:
    if actual != expected:
        raise ValueError(f"{path}: expected {field}={expected!r}, found {actual!r}")


def _canonical_root(artifact_root: Path, spec: _ArtifactSpec) -> tuple[Path, TreeDigest]:
    root = artifact_root / spec.relative_path
    if not root.is_dir():
        raise FileNotFoundError(f"canonical VEATIC artifact is unavailable: {root}")
    return root, {
        "files": spec.files,
        "path": str(root),
        "sha256_tree": spec.sha256_tree,
        "size_bytes": spec.size_bytes,
        "symlinks": 0,
    }


def _feature_array(bundle: Any, key: str) -> np.ndarray:
    if key in _LABEL_KEYS:
        raise RuntimeError(f"feature access attempted to cross the label boundary: {key}")
    if key not in bundle.files:
        raise ValueError(f"canonical feature bundle is missing {key}")
    return np.asarray(bundle[key])


def _normalise_video_ids(
    video_ids: Iterable[str | int] | str | int, available: tuple[str, ...]
) -> tuple[str, ...]:
    if isinstance(video_ids, (str, int, np.integer)):
        text_ids = (str(video_ids),)
    else:
        text_ids = tuple(str(video_id) for video_id in video_ids)
    normalised: list[str] = []
    for text in text_ids:
        if not text.isdigit() or str(int(text)) != text:
            raise ValueError(f"invalid canonical video id: {text!r}")
        normalised.append(text)
    result = tuple(normalised)
    if not result:
        raise ValueError("at least one video id is required")
    if len(set(result)) != len(result):
        raise ValueError("video ids must be unique")
    unknown = sorted(set(result).difference(available), key=int)
    if unknown:
        raise ValueError(f"unknown canonical video ids: {unknown}")
    return result


def _normalise_representations(representations: Iterable[str]) -> tuple[str, ...]:
    result = (representations,) if isinstance(representations, str) else tuple(representations)
    if not result:
        raise ValueError("at least one representation is required")
    if len(set(result)) != len(result):
        raise ValueError("representations must be unique")
    unknown = sorted(set(result).difference(_REPRESENTATION_LAYOUTS))
    if unknown:
        raise ValueError(f"unsupported VEATIC representations: {unknown}")
    return result


def _validate_row_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"canonical VEATIC rows are missing: {path}")
    # Do not open the file here: rows.csv is the label boundary. Its identity
    # and values are validated only at the caller-declared access stage.


def _validate_manifests(
    video_id: str,
    vjepa_root: Path,
    tribe_root: Path,
) -> tuple[int, int]:
    vjepa_dir = vjepa_root / video_id
    tribe_dir = tribe_root / "per_video" / video_id
    vjepa_manifest_path = vjepa_dir / "manifest.json"
    tribe_manifest_path = tribe_dir / "manifest.json"
    vjepa = _read_json(vjepa_manifest_path)
    tribe = _read_json(tribe_manifest_path)

    _require_equal(vjepa.get("status"), "complete", "status", vjepa_manifest_path)
    _require_equal(vjepa.get("video_id"), video_id, "video_id", vjepa_manifest_path)
    _require_equal(
        vjepa.get("schema"),
        "veatic_dense_vjepa21_vitg_compact_temporal_v1",
        "schema",
        vjepa_manifest_path,
    )
    _require_equal(vjepa.get("row_hz"), CANONICAL_DATASET.row_hz, "row_hz", vjepa_manifest_path)
    _require_equal(
        vjepa.get("row_plan_sha256"), _ROW_PLAN_SHA256, "row_plan_sha256", vjepa_manifest_path
    )
    _require_equal(
        vjepa.get("model_sha256"), _ENCODER_MODEL_SHA256, "model_sha256", vjepa_manifest_path
    )
    row_count = int(vjepa.get("row_count", -1))
    _require_equal(
        vjepa.get("feature_shape"), [row_count, 20, 1, 1_408], "feature_shape", vjepa_manifest_path
    )
    _require_equal(
        vjepa.get("temporal_diagnostics53_shape"),
        [row_count, 53],
        "temporal_diagnostics53_shape",
        vjepa_manifest_path,
    )
    _require_equal(vjepa.get("dtype"), "float16", "dtype", vjepa_manifest_path)

    _require_equal(tribe.get("status"), "complete", "status", tribe_manifest_path)
    _require_equal(tribe.get("video_id"), video_id, "video_id", tribe_manifest_path)
    _require_equal(
        tribe.get("schema_version"),
        "veatic_compact_mlx_tribe_v2_postpass_v1",
        "schema_version",
        tribe_manifest_path,
    )
    _require_equal(tribe.get("row_count"), row_count, "row_count", tribe_manifest_path)
    _require_equal(
        tribe.get("cortical_shape"), [row_count, 20_484], "cortical_shape", tribe_manifest_path
    )
    _require_equal(
        tribe.get("grouped_shape"), [row_count, 2, 1_408], "grouped_shape", tribe_manifest_path
    )

    _validate_row_file(vjepa_dir / "rows.csv")
    vjepa_npz = vjepa_dir / "vjepa21_hidden_states.npz"
    tribe_npz = tribe_dir / "tribe_v2_cortical_predictions.npz"
    if not vjepa_npz.is_file() or not tribe_npz.is_file():
        raise FileNotFoundError(
            f"canonical VEATIC feature payload is incomplete for video {video_id}"
        )

    with (
        np.load(vjepa_npz, allow_pickle=False) as vjepa_bundle,
        np.load(tribe_npz, allow_pickle=False) as tribe_bundle,
    ):
        for key in (*_SHARED_METADATA_KEYS, "features"):
            if key not in vjepa_bundle.files:
                raise ValueError(f"{vjepa_npz}: missing {key}")
        for key in (*_SHARED_METADATA_KEYS, "cortical_prediction", "tribe_grouped_video_feature"):
            if key not in tribe_bundle.files:
                raise ValueError(f"{tribe_npz}: missing {key}")

        time_seconds = _feature_array(vjepa_bundle, "time_seconds")
        expected_time = np.arange(row_count, dtype=np.float32) / np.float32(
            CANONICAL_DATASET.row_hz
        )
        if time_seconds.dtype != np.float32 or not np.array_equal(time_seconds, expected_time):
            raise ValueError(f"{vjepa_npz}: rows are not the exact sequential 2 Hz grid")

        black = _feature_array(vjepa_bundle, "quality_black_frame_flag")
        duplicate = _feature_array(vjepa_bundle, "quality_duplicate_frame_flag")
        exclusion = _feature_array(vjepa_bundle, "quality_exclusion_flag")
        for name, values in (
            ("quality_black_frame_flag", black),
            ("quality_duplicate_frame_flag", duplicate),
            ("quality_exclusion_flag", exclusion),
        ):
            if values.dtype != np.uint8 or values.shape != (row_count,):
                raise ValueError(f"{vjepa_npz}: {name} must be uint8[{row_count}]")
        if not np.array_equal(exclusion, np.logical_or(black, duplicate).astype(np.uint8)):
            raise ValueError(f"{vjepa_npz}: quality exclusion is not black OR duplicate")
        exclusion_count = int(exclusion.sum())
        _require_equal(
            vjepa.get("quality_exclusion_rows"),
            exclusion_count,
            "quality_exclusion_rows",
            vjepa_manifest_path,
        )

        for key in _SHARED_METADATA_KEYS:
            left = _feature_array(vjepa_bundle, key)
            right = _feature_array(tribe_bundle, key)
            if not np.array_equal(left, right):
                raise ValueError(f"video {video_id}: V-JEPA/TRIBE metadata differs for {key}")
    return row_count, exclusion_count


@dataclass(frozen=True)
class CanonicalSubstrate:
    """Validated paths and row layout for the fresh VEATIC 2.1 programme."""

    repo_root: Path
    artifact_root: Path
    vjepa_root: Path
    tribe_root: Path
    identity: SubstrateIdentity
    _row_counts: Mapping[str, int]

    @classmethod
    def from_repo(cls, repo_root: str | Path | None = None) -> CanonicalSubstrate:
        root = (
            Path(repo_root).expanduser().resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[3]
        )
        artifact_root = Path("/Volumes/onn. Drive/Neural Bridge Artifacts")
        if not artifact_root.is_dir():
            raise FileNotFoundError(f"canonical artifact root is unavailable: {artifact_root}")
        vjepa_root, vjepa_tree = _canonical_root(artifact_root, _VJEPA)
        tribe_root, tribe_tree = _canonical_root(artifact_root, _TRIBE)

        vjepa_ids = tuple(
            sorted((path.name for path in vjepa_root.iterdir() if path.is_dir()), key=int)
        )
        tribe_per_video = tribe_root / "per_video"
        tribe_ids = tuple(
            sorted((path.name for path in tribe_per_video.iterdir() if path.is_dir()), key=int)
        )
        if vjepa_ids != _VIDEO_IDS or tribe_ids != _VIDEO_IDS:
            raise ValueError("canonical feature caches must contain exactly video ids 0..123")

        row_counts: dict[str, int] = {}
        row_count = 0
        exclusion_count = 0
        for video_id in _VIDEO_IDS:
            video_rows, video_exclusions = _validate_manifests(video_id, vjepa_root, tribe_root)
            row_counts[video_id] = video_rows
            row_count += video_rows
            exclusion_count += video_exclusions

        identity = SubstrateIdentity(
            video_ids=_VIDEO_IDS,
            row_count=row_count,
            exclusion_count=exclusion_count,
            row_hz=CANONICAL_DATASET.row_hz,
            vjepa_artifact_id=_VJEPA.artifact_id,
            vjepa_sha256_tree=vjepa_tree["sha256_tree"],
            vjepa_file_count=vjepa_tree["files"],
            vjepa_size_bytes=vjepa_tree["size_bytes"],
            tribe_artifact_id=_TRIBE.artifact_id,
            tribe_sha256_tree=tribe_tree["sha256_tree"],
            tribe_file_count=tribe_tree["files"],
            tribe_size_bytes=tribe_tree["size_bytes"],
            row_plan_sha256=_ROW_PLAN_SHA256,
            source_tree_sha256=_SOURCE_TREE_SHA256,
            encoder_model_sha256=_ENCODER_MODEL_SHA256,
        )
        identity.validate()
        return cls(
            repo_root=root,
            artifact_root=artifact_root,
            vjepa_root=vjepa_root,
            tribe_root=tribe_root,
            identity=identity,
            _row_counts=MappingProxyType(row_counts),
        )

    @property
    def video_ids(self) -> tuple[str, ...]:
        return self.identity.video_ids

    def load_features(
        self,
        video_ids: Iterable[str | int] | str | int,
        representations: Iterable[str],
    ) -> FeatureRows:
        selected_videos = _normalise_video_ids(video_ids, self.video_ids)
        selected_representations = _normalise_representations(representations)
        total_rows = sum(self._row_counts[video_id] for video_id in selected_videos)

        video_values = np.empty(total_rows, dtype="U3")
        row_values = np.empty(total_rows, dtype=np.int32)
        time_values = np.empty(total_rows, dtype=np.float32)
        eligible_values = np.empty(total_rows, dtype=np.bool_)
        matrices: dict[str, np.ndarray] = {}
        for name in selected_representations:
            width, dtype = _REPRESENTATION_LAYOUTS[name]
            matrices[name] = np.empty((total_rows, width), dtype=dtype)

        offset = 0
        tribe_requested = "tribe_cortical" in selected_representations
        for video_id in selected_videos:
            count = self._row_counts[video_id]
            destination = slice(offset, offset + count)
            vjepa_path = self.vjepa_root / video_id / "vjepa21_hidden_states.npz"
            with np.load(vjepa_path, allow_pickle=False) as vjepa_bundle:
                time_seconds = _feature_array(vjepa_bundle, "time_seconds")
                exclusion = _feature_array(vjepa_bundle, "quality_exclusion_flag")
                video_values[destination] = video_id
                row_values[destination] = np.arange(count, dtype=np.int32)
                time_values[destination] = time_seconds
                eligible_values[destination] = ~exclusion.astype(np.bool_, copy=False)

                if "diagnostics_only" in matrices:
                    diagnostics = _feature_array(vjepa_bundle, "temporal_diagnostics53")
                    if diagnostics.shape != (count, 53):
                        raise ValueError(f"{vjepa_path}: unexpected diagnostics shape")
                    matrices["diagnostics_only"][destination] = diagnostics

            if tribe_requested:
                tribe_path = (
                    self.tribe_root / "per_video" / video_id / "tribe_v2_cortical_predictions.npz"
                )
                with np.load(tribe_path, allow_pickle=False) as tribe_bundle:
                    if "tribe_cortical" in matrices:
                        cortical = _feature_array(tribe_bundle, "cortical_prediction")
                        if cortical.shape != (count, 20_484) or cortical.dtype != np.float16:
                            raise ValueError(f"{tribe_path}: unexpected cortical feature layout")
                        matrices["tribe_cortical"][destination] = cortical
            offset += count

        rows = FeatureRows(
            video_id=video_values,
            row_index=row_values,
            time_seconds=time_values,
            quality_eligible=eligible_values,
            representations=matrices,
        )
        rows.validate()
        return rows

    def load_labels(
        self,
        video_ids: Iterable[str | int] | str | int,
        *,
        row_indices: Mapping[str, Iterable[int]] | None = None,
        access_callback: Callable[[str], None] | None = None,
        stage: str = "load_labels",
    ) -> LabelRows:
        selected_videos = _normalise_video_ids(video_ids, self.video_ids)
        if not stage:
            raise ValueError("label access stage must be non-empty")
        if access_callback is not None:
            access_callback(stage)

        selected_rows: dict[str, set[int]] | None = None
        if row_indices is not None:
            selected_rows = {
                str(video): {int(row) for row in rows} for video, rows in row_indices.items()
            }
            if set(selected_rows) != set(selected_videos) or any(
                not rows or min(rows) < 0 or max(rows) >= self._row_counts[video]
                for video, rows in selected_rows.items()
            ):
                raise ValueError("label row selection must contain valid rows for every video")
        total_rows = (
            sum(self._row_counts[video_id] for video_id in selected_videos)
            if selected_rows is None
            else sum(len(selected_rows[video_id]) for video_id in selected_videos)
        )
        video_values = np.empty(total_rows, dtype="U3")
        row_values = np.empty(total_rows, dtype=np.int32)
        time_values = np.empty(total_rows, dtype=np.float32)
        arousal_values = np.empty(total_rows, dtype=np.float32)
        valence_values = np.empty(total_rows, dtype=np.float32)

        offset = 0
        for video_id in selected_videos:
            expected_rows = self._row_counts[video_id]
            owned_rows = None if selected_rows is None else selected_rows[video_id]
            written = 0
            rows_path = self.vjepa_root / video_id / "rows.csv"
            with rows_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != _ROW_COLUMNS:
                    raise ValueError(f"{rows_path}: unexpected canonical row schema")
                loaded = 0
                for loaded, row in enumerate(reader, start=1):
                    local_index = loaded - 1
                    if row["video_id"] != video_id or int(row["row_index"]) != local_index:
                        raise ValueError(
                            f"{rows_path}: non-sequential row identity at {local_index}"
                        )
                    expected_time = local_index / CANONICAL_DATASET.row_hz
                    time_seconds = float(row["time_seconds"])
                    if not np.isclose(time_seconds, expected_time, rtol=0.0, atol=1e-6):
                        raise ValueError(f"{rows_path}: non-2 Hz timestamp at row {local_index}")
                    if owned_rows is not None and local_index not in owned_rows:
                        continue
                    destination = offset + written
                    video_values[destination] = row["video_id"]
                    row_values[destination] = local_index
                    time_values[destination] = time_seconds
                    arousal_values[destination] = float(row["arousal"])
                    valence_values[destination] = float(row["valence"])
                    written += 1
            if loaded != expected_rows:
                raise ValueError(f"{rows_path}: expected {expected_rows} rows, found {loaded}")
            expected_written = expected_rows if owned_rows is None else len(owned_rows)
            if written != expected_written:
                raise ValueError(f"{rows_path}: expected {expected_written} selected rows")
            offset += written

        rows = LabelRows(
            video_id=video_values,
            row_index=row_values,
            time_seconds=time_values,
            arousal=arousal_values,
            valence=valence_values,
        )
        rows.validate()
        return rows


__all__ = ["CanonicalSubstrate"]
