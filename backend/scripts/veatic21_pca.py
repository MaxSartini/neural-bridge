"""Fresh, fold-safe PCA for the sealed VEATIC 2.1 cortical cache.

This module deliberately owns a new artifact schema.  It cannot resume an
AGAIN or historical VEATIC PCA artifact: every fit is bound to VEATIC 2.1,
the compact-cache digest, the end-state contract digest, the exact candidate
and quality-valid fit rows, and the grouped-video ownership audit.

PCA fitting uses the same causal base families and deterministic randomized
SVD recipe as the dense AGAIN substrate, while remaining CPU-only and
independent of the older Phase 4 runner.  Quality-excluded rows can still be
transformed after fitting (for causal feature alignment), but they never
participate in the fit.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    import mlx.core as mx
except Exception:  # pragma: no cover - exercised only off Apple Silicon/without MLX.
    mx = None


SCHEMA_VERSION = "veatic21_fold_safe_pca_v1"
DATASET_ID = "veatic-124-v2.1"
ROW_HZ = 2.0
TEMPORAL_MEAN_ROWS = 4
ALLOWED_BASE_FAMILIES = ("temporal_mean_2s", "current", "delta")
ALLOWED_WIDTHS = (64, 128, 256)
PCA_ALGORITHM = "deterministic_streaming_randomized_svd_train_only_zscore"
PCA_MATMUL_BACKEND = "mlx_gpu" if mx is not None else "numpy_cpu_fallback"


class Veatic21PcaError(RuntimeError):
    """Raised when a PCA fit, resume, transform, or provenance gate fails."""


def canonical_digest(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible scientific identity."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_digest(array: np.ndarray) -> str:
    """Hash an array with its dtype and shape, not only its raw bytes."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_identity_token(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Veatic21PcaError(f"{name} must be a non-empty string")
    return value.strip()


def _coerce_row_indices(
    values: Sequence[int] | np.ndarray,
    *,
    row_count: int,
    name: str,
    require_strictly_increasing: bool,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise Veatic21PcaError(f"{name} must be a one-dimensional integer array")
    indices = np.asarray(array, dtype=np.int64)
    if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= row_count):
        raise Veatic21PcaError(f"{name} contains an out-of-range row index")
    if require_strictly_increasing and indices.size > 1 and np.any(np.diff(indices) <= 0):
        raise Veatic21PcaError(f"{name} must be unique and strictly increasing")
    return indices


def _ordered_unique_strings(values: np.ndarray) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values.tolist()))


def _video_digest(video_ids: Sequence[str]) -> str:
    return canonical_digest(list(video_ids))


class Veatic21PcaAccessor:
    """Row accessor for the three causal VEATIC 2.1 PCA base families.

    ``cortical`` may be an ndarray or read-only memmap.  Videos must be stored
    in contiguous row blocks, matching the compact-cache dense table.  The
    cache digest binds the accessor to a verified VEATIC cache seal.
    """

    def __init__(
        self,
        cortical: np.ndarray,
        video_ids: Sequence[str] | np.ndarray,
        *,
        base_family: str,
        cache_digest: str,
    ) -> None:
        matrix = np.asanyarray(cortical)
        if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
            raise Veatic21PcaError("cortical must be a non-empty two-dimensional matrix")
        if matrix.dtype.kind not in "fiu":
            raise Veatic21PcaError(f"cortical must be numeric, got {matrix.dtype}")
        if base_family not in ALLOWED_BASE_FAMILIES:
            raise Veatic21PcaError(
                f"base_family must be one of {ALLOWED_BASE_FAMILIES}, got {base_family!r}"
            )
        raw_video_ids = np.asarray(video_ids)
        if raw_video_ids.ndim != 1 or raw_video_ids.shape[0] != matrix.shape[0]:
            raise Veatic21PcaError("video_ids must be one-dimensional and row-aligned")
        canonical_video_ids = np.asarray([str(value) for value in raw_video_ids], dtype=np.str_)
        if any(not value for value in canonical_video_ids.tolist()):
            raise Veatic21PcaError("video_ids cannot contain empty identifiers")

        seen: set[str] = set()
        previous: str | None = None
        for value in canonical_video_ids.tolist():
            if value != previous:
                if value in seen:
                    raise Veatic21PcaError(
                        f"video {value!r} is not stored in one contiguous row block"
                    )
                seen.add(value)
                previous = value

        boundaries = np.empty(matrix.shape[0], dtype=bool)
        boundaries[0] = True
        boundaries[1:] = canonical_video_ids[1:] != canonical_video_ids[:-1]
        starts = np.where(boundaries, np.arange(matrix.shape[0], dtype=np.int64), 0)
        starts = np.maximum.accumulate(starts)
        valid_mask = np.ones(matrix.shape[0], dtype=bool)
        if base_family == "delta":
            valid_mask[boundaries] = False

        self.cortical = matrix
        self.video_ids = canonical_video_ids
        self.base_family = base_family
        self.cache_digest = _require_identity_token(cache_digest, name="cache_digest")
        self.video_starts = starts
        self.family_valid_mask = valid_mask
        self.row_video_layout_digest = array_digest(canonical_video_ids)

    @property
    def row_count(self) -> int:
        return int(self.cortical.shape[0])

    @property
    def feature_width(self) -> int:
        return int(self.cortical.shape[1])

    @property
    def description(self) -> str:
        descriptions = {
            "current": "current-row cortical_prediction",
            "delta": "current minus previous same-video cortical_prediction",
            "temporal_mean_2s": "causal current-plus-three-previous-row cortical mean",
        }
        return descriptions[self.base_family]

    def batch(self, row_indices: Sequence[int] | np.ndarray) -> np.ndarray:
        """Materialize family-valid rows in caller-supplied order."""

        indices = _coerce_row_indices(
            row_indices,
            row_count=self.row_count,
            name="row_indices",
            require_strictly_increasing=False,
        )
        if indices.size and not np.all(self.family_valid_mask[indices]):
            bad = indices[~self.family_valid_mask[indices]][:5].tolist()
            raise Veatic21PcaError(
                f"{self.base_family} cannot materialize family-invalid rows: {bad}"
            )
        if not indices.size:
            return np.empty((0, self.feature_width), dtype=np.float32)
        if self.base_family == "current":
            return np.asarray(self.cortical[indices], dtype=np.float32)
        if self.base_family == "delta":
            return (
                np.asarray(self.cortical[indices], dtype=np.float32)
                - np.asarray(self.cortical[indices - 1], dtype=np.float32)
            ).astype(np.float32, copy=False)

        rows: list[np.ndarray] = []
        for row_index in indices.tolist():
            start = max(int(self.video_starts[row_index]), row_index - (TEMPORAL_MEAN_ROWS - 1))
            rows.append(
                np.asarray(self.cortical[start : row_index + 1], dtype=np.float32).mean(
                    axis=0, dtype=np.float32
                )
            )
        return np.stack(rows, axis=0).astype(np.float32, copy=False)


@dataclass(frozen=True)
class PcaTransform:
    """A row-aligned PCA transform; invalid delta starts remain NaN."""

    row_indices: np.ndarray
    values: np.ndarray
    family_valid_mask: np.ndarray
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class Veatic21PcaFit:
    """Loaded or freshly fitted immutable VEATIC 2.1 PCA state."""

    components: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    singular_values: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    fit_row_indices: np.ndarray
    train_video_ids: tuple[str, ...]
    component_path: Path
    metadata_path: Path
    metadata: Mapping[str, Any]
    cache_hit: bool

    @property
    def base_family(self) -> str:
        return str(self.metadata["identity"]["base_family"])

    @property
    def width(self) -> int:
        return int(self.components.shape[0])

    def transform(
        self,
        accessor: Veatic21PcaAccessor,
        row_indices: Sequence[int] | np.ndarray,
        *,
        batch_size: int = 384,
    ) -> PcaTransform:
        return transform_rows(self, accessor, row_indices, batch_size=batch_size)


def _row_batches(indices: np.ndarray, *, batch_size: int) -> Iterable[np.ndarray]:
    if not isinstance(batch_size, int) or batch_size < 1:
        raise Veatic21PcaError("batch_size must be a positive integer")
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def _fit_mean_scale(
    accessor: Veatic21PcaAccessor,
    fit_rows: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    sum_x = np.zeros(accessor.feature_width, dtype=np.float64)
    sum_x2 = np.zeros(accessor.feature_width, dtype=np.float64)
    observed = 0
    for batch_rows in _row_batches(fit_rows, batch_size=batch_size):
        values = accessor.batch(batch_rows).astype(np.float32, copy=False)
        if not np.isfinite(values).all():
            raise Veatic21PcaError(
                f"non-finite {accessor.base_family} values in PCA fit rows"
            )
        sum_x += values.sum(axis=0, dtype=np.float64)
        sum_x2 += np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)
        observed += int(values.shape[0])
    if observed != len(fit_rows):
        raise Veatic21PcaError("PCA statistics did not consume the exact fit rows")
    mean64 = sum_x / float(observed)
    variance64 = np.maximum(sum_x2 / float(observed) - np.square(mean64), 1e-6)
    mean = mean64.astype(np.float32)
    scale = np.sqrt(variance64).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise Veatic21PcaError("invalid train-only PCA mean/scale")
    return mean, scale


def _standardized_batch(
    accessor: Veatic21PcaAccessor,
    rows: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    values = accessor.batch(rows).astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise Veatic21PcaError(
            f"non-finite {accessor.base_family} values while applying PCA"
        )
    standardized = ((values - mean) / scale).astype(np.float32, copy=False)
    if not np.isfinite(standardized).all():
        raise Veatic21PcaError("non-finite standardized PCA input")
    return standardized


def _matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Run the large randomized-SVD products on MLX when it is available."""

    left32 = np.asarray(left, dtype=np.float32)
    right32 = np.asarray(right, dtype=np.float32)
    if mx is None:
        return (left32 @ right32).astype(np.float32, copy=False)
    result = mx.array(left32, dtype=mx.float32) @ mx.array(right32, dtype=mx.float32)
    mx.eval(result)
    return np.asarray(result, dtype=np.float32)


def _canonicalize_component_signs(components: np.ndarray) -> np.ndarray:
    canonical = np.asarray(components, dtype=np.float32).copy()
    for component in canonical:
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0:
            component *= np.float32(-1.0)
    return canonical


def _rank_tolerance(singular_values: np.ndarray, rows: int, columns: int) -> float:
    if singular_values.size == 0:
        return math.inf
    return float(np.finfo(np.float32).eps * max(rows, columns) * singular_values[0])


def _validate_numeric_state(
    *,
    components: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    singular_values: np.ndarray,
    explained_variance: np.ndarray,
    explained_variance_ratio: np.ndarray,
    width: int,
    feature_width: int,
    fit_row_count: int,
) -> None:
    expected = {
        "components": ((width, feature_width), np.dtype(np.float32)),
        "mean": ((feature_width,), np.dtype(np.float32)),
        "scale": ((feature_width,), np.dtype(np.float32)),
        "singular_values": ((width,), np.dtype(np.float32)),
        "explained_variance": ((width,), np.dtype(np.float32)),
        "explained_variance_ratio": ((width,), np.dtype(np.float32)),
    }
    arrays = {
        "components": components,
        "mean": mean,
        "scale": scale,
        "singular_values": singular_values,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
    }
    for name, array in arrays.items():
        shape, dtype = expected[name]
        if tuple(array.shape) != shape or array.dtype != dtype:
            raise Veatic21PcaError(
                f"invalid {name} shape/dtype: {array.shape}/{array.dtype}, expected {shape}/{dtype}"
            )
        if not np.isfinite(array).all():
            raise Veatic21PcaError(f"non-finite persisted PCA {name}")
    if np.any(scale <= 0):
        raise Veatic21PcaError("persisted PCA scale is not strictly positive")
    if np.any(singular_values < 0) or np.any(np.diff(singular_values) > 1e-4):
        raise Veatic21PcaError("persisted PCA singular values are invalid or unordered")
    tolerance = _rank_tolerance(singular_values, fit_row_count, feature_width)
    if singular_values[-1] <= tolerance:
        raise Veatic21PcaError(
            f"insufficient numerical rank for PCA width {width}: "
            f"smallest singular value {singular_values[-1]:.8g} <= {tolerance:.8g}"
        )
    gram = components @ components.T
    if not np.allclose(gram, np.eye(width, dtype=np.float32), atol=2e-3, rtol=2e-3):
        raise Veatic21PcaError("persisted PCA components are not orthonormal")
    if np.any(explained_variance < 0) or np.any(explained_variance_ratio < 0):
        raise Veatic21PcaError("persisted PCA variance metadata is negative")


def _artifact_paths(
    output_root: Path,
    *,
    artifact_key: str,
    base_family: str,
    width: int,
) -> tuple[Path, Path]:
    original = _require_identity_token(artifact_key, name="artifact_key")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", original).strip("._-") or "pca"
    stem = stem[:80]
    key_suffix = hashlib.sha256(original.encode("utf-8")).hexdigest()[:12]
    directory = Path(output_root) / "veatic21_pca"
    directory.mkdir(parents=True, exist_ok=True)
    base = f"{stem}__{key_suffix}__{base_family}__pca{width}"
    return directory / f"{base}.npz", directory / f"{base}.json"


def _ownership_identity(
    accessor: Veatic21PcaAccessor,
    *,
    train_rows: np.ndarray,
    held_out_rows: np.ndarray,
    quality_mask: np.ndarray,
    width: int,
    seed: int,
    oversampling: int,
    power_iterations: int,
    artifact_key: str,
    contract_digest: str,
) -> tuple[dict[str, Any], np.ndarray, tuple[str, ...]]:
    if np.intersect1d(train_rows, held_out_rows, assume_unique=True).size:
        raise Veatic21PcaError("training and held-out PCA rows overlap")

    train_videos = _ordered_unique_strings(accessor.video_ids[train_rows])
    held_out_videos = _ordered_unique_strings(accessor.video_ids[held_out_rows])
    video_overlap = sorted(set(train_videos) & set(held_out_videos))
    if video_overlap:
        raise Veatic21PcaError(
            f"held-out videos participate in PCA training rows: {video_overlap[:5]}"
        )

    fit_mask = quality_mask[train_rows] & accessor.family_valid_mask[train_rows]
    fit_rows = train_rows[fit_mask]
    fit_videos = _ordered_unique_strings(accessor.video_ids[fit_rows])
    if set(fit_videos) & set(held_out_videos):
        raise Veatic21PcaError("held-out video leaked into the quality-valid PCA fit rows")
    if len(fit_rows) <= width or accessor.feature_width < width:
        raise Veatic21PcaError(
            f"insufficient PCA rank envelope: fit_rows={len(fit_rows)} "
            f"features={accessor.feature_width} width={width}"
        )

    identity = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "fresh_veatic_only": True,
        "upstream_fitted_pca_reused": False,
        "artifact_key": artifact_key,
        "base_family": accessor.base_family,
        "family_description": accessor.description,
        "pca_width": int(width),
        "random_seed": int(seed),
        "oversampling": int(oversampling),
        "power_iterations": int(power_iterations),
        "algorithm": PCA_ALGORITHM,
        "matmul_backend": PCA_MATMUL_BACKEND,
        "centering_scaling_policy": "quality_valid_outer_train_rows_only_zscore",
        "row_hz": ROW_HZ,
        "temporal_mean_rows": TEMPORAL_MEAN_ROWS,
        "source_row_count": accessor.row_count,
        "source_feature_width": accessor.feature_width,
        "source_dtype": str(accessor.cortical.dtype),
        "cache_digest": accessor.cache_digest,
        "contract_digest": contract_digest,
        "row_video_layout_digest": accessor.row_video_layout_digest,
        "quality_mask_digest": array_digest(quality_mask),
        "family_valid_mask_digest": array_digest(accessor.family_valid_mask),
        "supplied_train_row_count": int(len(train_rows)),
        "supplied_train_row_digest": array_digest(train_rows),
        "fit_train_row_count": int(len(fit_rows)),
        "fit_train_row_digest": array_digest(fit_rows),
        "quality_excluded_train_row_count": int(np.count_nonzero(~quality_mask[train_rows])),
        "family_excluded_train_row_count": int(
            np.count_nonzero(~accessor.family_valid_mask[train_rows])
        ),
        "train_video_ids": list(train_videos),
        "train_video_digest": _video_digest(train_videos),
        "fit_train_video_ids": list(fit_videos),
        "fit_train_video_digest": _video_digest(fit_videos),
        "held_out_row_count": int(len(held_out_rows)),
        "held_out_row_digest": array_digest(held_out_rows),
        "held_out_video_ids": list(held_out_videos),
        "held_out_video_digest": _video_digest(held_out_videos),
        "held_out_row_overlap_count": 0,
        "held_out_video_overlap_count": 0,
        "no_held_out_participation_audit": True,
    }
    return identity, fit_rows, train_videos


def _load_exact_fit(
    *,
    component_path: Path,
    metadata_path: Path,
    expected_identity: Mapping[str, Any],
    expected_fit_rows: np.ndarray,
    expected_train_videos: tuple[str, ...],
) -> Veatic21PcaFit:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Veatic21PcaError(f"cannot read PCA metadata: {metadata_path}") from exc
    expected_identity_digest = canonical_digest(expected_identity)
    if metadata.get("identity_sha256") != expected_identity_digest:
        raise Veatic21PcaError(
            f"PCA resume identity mismatch for {metadata_path.name}; refusing stale reuse"
        )
    if metadata.get("identity") != dict(expected_identity):
        raise Veatic21PcaError("PCA resume metadata differs despite its identity digest")
    if metadata.get("component_file_sha256") != _file_digest(component_path):
        raise Veatic21PcaError("PCA component file checksum drift")

    expected_keys = {
        "components",
        "mean",
        "scale",
        "singular_values",
        "explained_variance",
        "explained_variance_ratio",
        "fit_row_indices",
        "supplied_train_row_indices",
        "train_video_ids",
    }
    try:
        with np.load(component_path, allow_pickle=False) as payload:
            if set(payload.files) != expected_keys:
                raise Veatic21PcaError(
                    f"PCA component keys drifted: {sorted(payload.files)}"
                )
            arrays = {name: np.asarray(payload[name]) for name in expected_keys}
    except (OSError, ValueError, KeyError) as exc:
        raise Veatic21PcaError(f"cannot load PCA components: {component_path}") from exc

    fit_rows = np.asarray(arrays["fit_row_indices"], dtype=np.int64)
    supplied_train_rows = np.asarray(arrays["supplied_train_row_indices"], dtype=np.int64)
    train_video_ids = tuple(str(value) for value in arrays["train_video_ids"].tolist())
    if not np.array_equal(fit_rows, expected_fit_rows):
        raise Veatic21PcaError("persisted PCA fit rows differ from exact current ownership")
    identity = dict(expected_identity)
    if array_digest(supplied_train_rows) != identity["supplied_train_row_digest"]:
        raise Veatic21PcaError("persisted supplied PCA train rows have drifted")
    if train_video_ids != expected_train_videos:
        raise Veatic21PcaError("persisted PCA training videos have drifted")

    components = np.asarray(arrays["components"])
    mean = np.asarray(arrays["mean"])
    scale = np.asarray(arrays["scale"])
    singular_values = np.asarray(arrays["singular_values"])
    explained_variance = np.asarray(arrays["explained_variance"])
    explained_variance_ratio = np.asarray(arrays["explained_variance_ratio"])
    width = int(identity["pca_width"])
    feature_width = int(identity["source_feature_width"])
    _validate_numeric_state(
        components=components,
        mean=mean,
        scale=scale,
        singular_values=singular_values,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        width=width,
        feature_width=feature_width,
        fit_row_count=len(fit_rows),
    )
    digests = metadata.get("array_sha256", {})
    for name, array in (
        ("components", components),
        ("mean", mean),
        ("scale", scale),
        ("singular_values", singular_values),
        ("fit_row_indices", fit_rows),
    ):
        if digests.get(name) != array_digest(array):
            raise Veatic21PcaError(f"persisted PCA {name} checksum drift")

    return Veatic21PcaFit(
        components=components,
        mean=mean,
        scale=scale,
        singular_values=singular_values,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        fit_row_indices=fit_rows,
        train_video_ids=train_video_ids,
        component_path=component_path,
        metadata_path=metadata_path,
        metadata=metadata,
        cache_hit=True,
    )


def fit_or_load_pca(
    accessor: Veatic21PcaAccessor,
    *,
    train_row_indices: Sequence[int] | np.ndarray,
    held_out_row_indices: Sequence[int] | np.ndarray,
    quality_mask: np.ndarray,
    output_root: Path,
    artifact_key: str,
    width: int,
    seed: int,
    contract_digest: str,
    batch_size: int = 384,
    oversampling: int = 32,
    power_iterations: int = 1,
) -> Veatic21PcaFit:
    """Fit or exactly resume one fresh, grouped-fold-safe VEATIC 2.1 PCA.

    The fit set is precisely ``train_row_indices`` intersected with the caller's
    boolean ``quality_mask`` and the family-valid mask (delta video starts are
    invalid).  Held-out rows and every video represented by them are audited
    out of the supplied training ownership before any artifact is read or fit.
    """

    if not isinstance(accessor, Veatic21PcaAccessor):
        raise Veatic21PcaError("accessor must be a Veatic21PcaAccessor")
    if width not in ALLOWED_WIDTHS:
        raise Veatic21PcaError(f"width must be one of {ALLOWED_WIDTHS}, got {width}")
    if not isinstance(seed, (int, np.integer)):
        raise Veatic21PcaError("seed must be an integer")
    if not isinstance(oversampling, int) or oversampling < 0:
        raise Veatic21PcaError("oversampling must be a non-negative integer")
    if not isinstance(power_iterations, int) or power_iterations < 0:
        raise Veatic21PcaError("power_iterations must be a non-negative integer")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise Veatic21PcaError("batch_size must be a positive integer")
    contract_digest = _require_identity_token(contract_digest, name="contract_digest")
    artifact_key = _require_identity_token(artifact_key, name="artifact_key")

    train_rows = _coerce_row_indices(
        train_row_indices,
        row_count=accessor.row_count,
        name="train_row_indices",
        require_strictly_increasing=True,
    )
    held_out_rows = _coerce_row_indices(
        held_out_row_indices,
        row_count=accessor.row_count,
        name="held_out_row_indices",
        require_strictly_increasing=True,
    )
    quality = np.asarray(quality_mask)
    if quality.ndim != 1 or quality.shape[0] != accessor.row_count or quality.dtype != np.bool_:
        raise Veatic21PcaError("quality_mask must be a row-aligned boolean array")

    identity, fit_rows, train_videos = _ownership_identity(
        accessor,
        train_rows=train_rows,
        held_out_rows=held_out_rows,
        quality_mask=quality,
        width=width,
        seed=int(seed),
        oversampling=oversampling,
        power_iterations=power_iterations,
        artifact_key=artifact_key,
        contract_digest=contract_digest,
    )
    component_path, metadata_path = _artifact_paths(
        Path(output_root),
        artifact_key=artifact_key,
        base_family=accessor.base_family,
        width=width,
    )
    if component_path.exists() != metadata_path.exists():
        raise Veatic21PcaError(
            f"incomplete PCA artifact pair for {component_path.stem}; refusing implicit repair"
        )
    if component_path.exists():
        return _load_exact_fit(
            component_path=component_path,
            metadata_path=metadata_path,
            expected_identity=identity,
            expected_fit_rows=fit_rows,
            expected_train_videos=train_videos,
        )

    mean, scale = _fit_mean_scale(accessor, fit_rows, batch_size=batch_size)
    projection_width = min(
        accessor.feature_width,
        len(fit_rows),
        int(width + oversampling),
    )
    if projection_width < width:
        raise Veatic21PcaError("randomized SVD projection width is below requested PCA width")
    rng = np.random.default_rng(int(seed))
    omega = rng.normal(size=(accessor.feature_width, projection_width)).astype(np.float32)
    projected = np.empty((len(fit_rows), projection_width), dtype=np.float32)
    total_standardized_sum_squares = 0.0
    cursor = 0
    for batch_rows in _row_batches(fit_rows, batch_size=batch_size):
        standardized = _standardized_batch(accessor, batch_rows, mean, scale)
        projected[cursor : cursor + len(batch_rows)] = _matmul(standardized, omega)
        total_standardized_sum_squares += float(
            np.square(standardized, dtype=np.float64).sum(dtype=np.float64)
        )
        cursor += len(batch_rows)
    q_matrix, _ = np.linalg.qr(projected, mode="reduced")
    q_matrix = q_matrix.astype(np.float32, copy=False)
    del projected

    for _ in range(power_iterations):
        z_matrix = np.zeros((accessor.feature_width, projection_width), dtype=np.float32)
        cursor = 0
        for batch_rows in _row_batches(fit_rows, batch_size=batch_size):
            standardized = _standardized_batch(accessor, batch_rows, mean, scale)
            q_batch = q_matrix[cursor : cursor + len(batch_rows)]
            z_matrix += _matmul(standardized.T, q_batch)
            cursor += len(batch_rows)
        projected = np.empty((len(fit_rows), projection_width), dtype=np.float32)
        cursor = 0
        for batch_rows in _row_batches(fit_rows, batch_size=batch_size):
            standardized = _standardized_batch(accessor, batch_rows, mean, scale)
            projected[cursor : cursor + len(batch_rows)] = _matmul(
                standardized, z_matrix
            )
            cursor += len(batch_rows)
        q_matrix, _ = np.linalg.qr(projected, mode="reduced")
        q_matrix = q_matrix.astype(np.float32, copy=False)
        del projected, z_matrix

    compressed = np.zeros((projection_width, accessor.feature_width), dtype=np.float32)
    cursor = 0
    for batch_rows in _row_batches(fit_rows, batch_size=batch_size):
        standardized = _standardized_batch(accessor, batch_rows, mean, scale)
        q_batch = q_matrix[cursor : cursor + len(batch_rows)]
        compressed += _matmul(q_batch.T, standardized)
        cursor += len(batch_rows)
    _left, singular_values_full, right_t = np.linalg.svd(compressed, full_matrices=False)
    components = _canonicalize_component_signs(right_t[:width])
    singular_values = singular_values_full[:width].astype(np.float32, copy=False)
    explained_variance = (
        np.square(singular_values.astype(np.float64)) / float(len(fit_rows) - 1)
    ).astype(np.float32)
    total_variance = total_standardized_sum_squares / float(len(fit_rows) - 1)
    if not math.isfinite(total_variance) or total_variance <= 0:
        raise Veatic21PcaError("PCA fit has no finite positive total variance")
    explained_variance_ratio = (
        explained_variance.astype(np.float64) / total_variance
    ).astype(np.float32)
    _validate_numeric_state(
        components=components,
        mean=mean,
        scale=scale,
        singular_values=singular_values,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        width=width,
        feature_width=accessor.feature_width,
        fit_row_count=len(fit_rows),
    )

    component_tmp = component_path.with_suffix(".tmp.npz")
    metadata_tmp = metadata_path.with_suffix(".tmp.json")
    try:
        np.savez(
            component_tmp,
            components=components,
            mean=mean,
            scale=scale,
            singular_values=singular_values,
            explained_variance=explained_variance,
            explained_variance_ratio=explained_variance_ratio,
            fit_row_indices=fit_rows.astype(np.int64, copy=False),
            supplied_train_row_indices=train_rows.astype(np.int64, copy=False),
            train_video_ids=np.asarray(train_videos, dtype=np.str_),
        )
        component_tmp.replace(component_path)
        identity_sha256 = canonical_digest(identity)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "identity_sha256": identity_sha256,
            "identity": identity,
            "component_file_sha256": _file_digest(component_path),
            "array_sha256": {
                "components": array_digest(components),
                "mean": array_digest(mean),
                "scale": array_digest(scale),
                "singular_values": array_digest(singular_values),
                "fit_row_indices": array_digest(fit_rows),
            },
            "smallest_retained_singular_value": float(singular_values[-1]),
            "largest_singular_value": float(singular_values[0]),
            "singular_values_top10": [
                float(value) for value in singular_values[: min(10, len(singular_values))]
            ],
            "explained_variance_ratio_sum": float(explained_variance_ratio.sum()),
            "no_held_out_participation_audit": True,
            "fit_quality_valid_only": True,
            "transform_quality_excluded_rows_allowed_for_alignment_only": True,
        }
        metadata_tmp.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        metadata_tmp.replace(metadata_path)
    finally:
        component_tmp.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)

    return Veatic21PcaFit(
        components=components,
        mean=mean,
        scale=scale,
        singular_values=singular_values,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        fit_row_indices=fit_rows.copy(),
        train_video_ids=train_videos,
        component_path=component_path,
        metadata_path=metadata_path,
        metadata=metadata,
        cache_hit=False,
    )


def transform_rows(
    fitted: Veatic21PcaFit,
    accessor: Veatic21PcaAccessor,
    row_indices: Sequence[int] | np.ndarray,
    *,
    batch_size: int = 384,
) -> PcaTransform:
    """Transform arbitrary rows while preserving exact caller alignment.

    Family-invalid delta video starts are represented by NaN score rows and a
    false ``family_valid_mask``.  Quality exclusions are intentionally not
    applied here: they were excluded from fitting, but can be transformed to
    preserve the causal row grid for downstream sequence construction.
    """

    if not isinstance(fitted, Veatic21PcaFit) or not isinstance(
        accessor, Veatic21PcaAccessor
    ):
        raise Veatic21PcaError("transform requires Veatic21PcaFit and Veatic21PcaAccessor")
    identity = fitted.metadata.get("identity")
    if not isinstance(identity, Mapping):
        raise Veatic21PcaError("PCA fit is missing immutable identity metadata")
    binding_checks = {
        "dataset_id": DATASET_ID,
        "base_family": accessor.base_family,
        "source_row_count": accessor.row_count,
        "source_feature_width": accessor.feature_width,
        "source_dtype": str(accessor.cortical.dtype),
        "cache_digest": accessor.cache_digest,
        "row_video_layout_digest": accessor.row_video_layout_digest,
    }
    for key, expected in binding_checks.items():
        if identity.get(key) != expected:
            raise Veatic21PcaError(
                f"PCA transform provenance drift for {key}: "
                f"fit={identity.get(key)!r} current={expected!r}"
            )
    indices = _coerce_row_indices(
        row_indices,
        row_count=accessor.row_count,
        name="row_indices",
        require_strictly_increasing=False,
    )
    valid = accessor.family_valid_mask[indices].astype(bool, copy=True)
    values = np.full((len(indices), fitted.width), np.nan, dtype=np.float32)
    valid_positions = np.flatnonzero(valid)
    for position_batch in _row_batches(valid_positions, batch_size=batch_size):
        rows = indices[position_batch]
        standardized = _standardized_batch(accessor, rows, fitted.mean, fitted.scale)
        projected = _matmul(standardized, fitted.components.T)
        if tuple(projected.shape) != (len(rows), fitted.width) or not np.isfinite(
            projected
        ).all():
            raise Veatic21PcaError("invalid PCA transform shape or values")
        values[position_batch] = projected.astype(np.float32, copy=False)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "pca_identity_sha256": fitted.metadata["identity_sha256"],
        "row_count": int(len(indices)),
        "row_index_digest": array_digest(indices),
        "family_valid_row_count": int(np.count_nonzero(valid)),
        "alignment_preserved": True,
        "quality_mask_applied_to_transform": False,
    }
    return PcaTransform(
        row_indices=indices.copy(),
        values=values,
        family_valid_mask=valid,
        metadata=metadata,
    )


__all__ = [
    "ALLOWED_BASE_FAMILIES",
    "ALLOWED_WIDTHS",
    "DATASET_ID",
    "PcaTransform",
    "PCA_MATMUL_BACKEND",
    "SCHEMA_VERSION",
    "Veatic21PcaAccessor",
    "Veatic21PcaError",
    "Veatic21PcaFit",
    "array_digest",
    "canonical_digest",
    "fit_or_load_pca",
    "transform_rows",
]
