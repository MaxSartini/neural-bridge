"""Fold-owned reusable PCA projections for VEATIC 2.1 discovery."""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

from .contracts import FeatureRows
from .evidence import atomic_write_json, digest_json, load_json
from .preregistration import benchmark_partition_mask

_SCHEMA = "veatic21_cortical_pca_cache_v1"
_SOURCE = "tribe_cortical"
_NUMERICAL_IMPLEMENTATION = "standard_scaler_incremental_pca_v1"
_LEGACY_IMPLEMENTATION_SHA256 = "9bdc08ab3bd2737024d92e9dab542990b3b2a5eadb6148756f2142272ae07c0e"


def _batches(indices: np.ndarray, batch_rows: int, minimum_rows: int = 1) -> list[np.ndarray]:
    if batch_rows < minimum_rows:
        raise ValueError("PCA batch size must cover its component count")
    count = (len(indices) + batch_rows - 1) // batch_rows
    if not count or len(indices) // count < minimum_rows:
        raise ValueError("training rows cannot form valid balanced PCA batches")
    base, larger = divmod(len(indices), count)
    result = []
    start = 0
    for index in range(count):
        stop = start + base + int(index < larger)
        result.append(indices[start:stop])
        start = stop
    return result


def _row_digest(features: FeatureRows, indices: np.ndarray) -> str:
    digest = hashlib.sha256()
    for video, row in zip(
        features.video_id[indices], features.row_index[indices], strict=True
    ):
        digest.update(f"{video}:{int(row)}\n".encode())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)  # ty: ignore[invalid-argument-type]
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _project_to_atomic_npy(
    path: Path,
    matrix: np.ndarray,
    indices: np.ndarray,
    scaler: StandardScaler,
    pca: IncrementalPCA,
    batch_rows: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        projected = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=(len(indices), int(pca.n_components_)),
        )
        offset = 0
        for batch in _batches(indices, batch_rows):
            values = np.asarray(matrix[batch], dtype=np.float32)
            count = len(batch)
            projected[offset : offset + count] = pca.transform(scaler.transform(values))
            offset += count
        projected.flush()
        del projected
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _fit_basis(
    matrix: np.ndarray,
    fit_indices: np.ndarray,
    maximum_components: int,
    batch_rows: int,
) -> tuple[StandardScaler, IncrementalPCA]:
    scaler = StandardScaler()
    batches = _batches(fit_indices, batch_rows, maximum_components)
    for batch in batches:
        scaler.partial_fit(np.asarray(matrix[batch], dtype=np.float32))
    pca = IncrementalPCA(n_components=maximum_components, batch_size=batch_rows)
    for batch in batches:
        values = np.asarray(matrix[batch], dtype=np.float32)
        pca.partial_fit(scaler.transform(values))
    return scaler, pca


def _variance_widths(cumulative: np.ndarray, targets: Sequence[float]) -> dict[str, int | None]:
    widths: dict[str, int | None] = {}
    for target in targets:
        index = int(np.searchsorted(cumulative, target))
        widths[str(target)] = index + 1 if index < len(cumulative) else None
    return widths


def _read_metadata(directory: Path) -> dict[str, Any] | None:
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        return None
    return load_json(metadata_path)


def _verify_payload(directory: Path, metadata: Mapping[str, Any]) -> None:
    for name, expected in metadata["file_sha256"].items():
        path = directory / name
        if not path.is_file() or _file_sha256(path) != expected:
            raise RuntimeError(f"PCA cache payload verification failed: {path}")


def _compatible_cache(
    output_root: Path,
    fold: int,
    ownership: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]] | None:
    comparison = {
        key: value for key, value in ownership.items() if key != "numerical_implementation"
    }
    for directory in sorted(output_root.glob(f"fold-{fold}-*")):
        metadata = _read_metadata(directory)
        if metadata is None or any(metadata.get(key) != value for key, value in comparison.items()):
            continue
        implementation = metadata.get("numerical_implementation")
        if implementation is None:
            if metadata.get("code_sha256") != _LEGACY_IMPLEMENTATION_SHA256:
                continue
        elif implementation != _NUMERICAL_IMPLEMENTATION:
            continue
        _verify_payload(directory, metadata)
        return directory, metadata
    return None


def fit_event_pca_cache(
    features: FeatureRows,
    preregistration: Mapping[str, Any],
    output_root: Path,
    *,
    folds: Sequence[int] | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Fit or verify one maximum PCA basis per requested benchmark-train fold."""

    features.validate()
    if preregistration.get("schema") != "veatic21_event_preregistration_v12":
        raise ValueError("PCA cache requires the VEATIC event preregistration v12")
    if set(features.representations) != {_SOURCE}:
        raise ValueError(f"PCA cache requires exactly the {_SOURCE} representation")
    specification = preregistration["representations"]["pca"]
    maximum = int(specification["maximum_components"])
    batch_rows = maximum * 2
    if specification["batch_rows"] != (
        "2_x_maximum_components_balanced_without_short_final_batch"
    ):
        raise ValueError("unsupported PCA batch contract")

    benchmark_mask = benchmark_partition_mask(features, preregistration["split"], "train")
    benchmark_indices = np.flatnonzero(benchmark_mask)
    grouped_folds = preregistration["split"]["inner_grouped_video_folds"]
    selected_folds = tuple(range(len(grouped_folds))) if folds is None else tuple(folds)
    if not selected_folds or len(set(selected_folds)) != len(selected_folds):
        raise ValueError("PCA folds must be a non-empty unique sequence")
    if min(selected_folds) < 0 or max(selected_folds) >= len(grouped_folds):
        raise ValueError("PCA fold is outside the preregistered fold range")

    matrix = features.representations[_SOURCE]
    if maximum > min(len(benchmark_indices) - 1, matrix.shape[1]):
        raise ValueError("PCA maximum exceeds the benchmark-train rank ceiling")
    benchmark_row_sha256 = _row_digest(features, benchmark_indices)
    fold_records = []
    output_root.mkdir(parents=True, exist_ok=True)

    for fold in selected_folds:
        validation_videos = [str(value) for value in grouped_folds[fold]]
        fit_indices = benchmark_indices[
            ~np.isin(features.video_id[benchmark_indices].astype(str), validation_videos)
        ]
        fit_row_sha256 = _row_digest(features, fit_indices)
        ownership = {
            "schema": _SCHEMA,
            "substrate": preregistration["substrate"],
            "split_sha256": preregistration["split"]["split_sha256"],
            "source": _SOURCE,
            "source_dtype": str(matrix.dtype),
            "source_width": int(matrix.shape[1]),
            "fold": fold,
            "validation_videos": validation_videos,
            "fit_row_sha256": fit_row_sha256,
            "benchmark_row_sha256": benchmark_row_sha256,
            "maximum_components": maximum,
            "scaler": specification["scaler"],
            "solver": specification["solver"],
            "batch_rows": batch_rows,
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
            "numerical_implementation": _NUMERICAL_IMPLEMENTATION,
        }
        ownership_key = digest_json(ownership)
        compatible = _compatible_cache(output_root, fold, ownership)
        cache_hit = compatible is not None
        if compatible is None:
            directory = output_root / f"fold-{fold}-{ownership_key[:12]}"
            directory.mkdir(parents=True, exist_ok=True)
            scaler, pca = _fit_basis(matrix, fit_indices, maximum, batch_rows)
            components_path = directory / "basis.npz"
            projection_path = directory / "benchmark-train-projection.npy"
            _atomic_savez(
                components_path,
                scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
                scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
                pca_mean=np.asarray(pca.mean_, dtype=np.float64),
                pca_components=np.asarray(pca.components_, dtype=np.float32),
                explained_variance=np.asarray(pca.explained_variance_, dtype=np.float64),
                explained_variance_ratio=np.asarray(
                    pca.explained_variance_ratio_, dtype=np.float64
                ),
            )
            _project_to_atomic_npy(
                projection_path,
                matrix,
                benchmark_indices,
                scaler,
                pca,
                batch_rows,
            )
            cumulative = np.cumsum(np.asarray(pca.explained_variance_ratio_, dtype=np.float64))
            variance_widths = _variance_widths(cumulative, specification["variance_targets"])
            fixed_widths = [
                int(width) for width in specification["fixed_width_candidates"] if width <= maximum
            ]
            candidate_widths = sorted(
                set(fixed_widths).union(
                    width for width in variance_widths.values() if width is not None
                )
            )
            metadata = {
                **ownership,
                "cache_key": ownership_key,
                "ownership_key": ownership_key,
                "invocation_preregistration_sha256": preregistration[
                    "preregistration_sha256"
                ],
                "label_values_accessed": False,
                "fit_rows": int(len(fit_indices)),
                "projected_rows": int(len(benchmark_indices)),
                "variance_captured_at_maximum": float(cumulative[-1]),
                "variance_widths": variance_widths,
                "candidate_widths": candidate_widths,
                "projection_shape": [int(len(benchmark_indices)), maximum],
                "file_sha256": {
                    components_path.name: _file_sha256(components_path),
                    projection_path.name: _file_sha256(projection_path),
                },
            }
            metadata["metadata_sha256"] = digest_json(metadata)
            atomic_write_json(directory / "metadata.json", metadata)
        else:
            directory, metadata = compatible
        record = {
            "fold": fold,
            "ownership_key": ownership_key,
            "cache_hit": cache_hit,
            "directory": directory.name,
            "fit_rows": metadata["fit_rows"],
            "candidate_widths": metadata["candidate_widths"],
            "variance_captured_at_maximum": metadata["variance_captured_at_maximum"],
        }
        fold_records.append(record)
        if progress is not None:
            progress(record)

    manifest: dict[str, Any] = {
        "schema": _SCHEMA,
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "label_values_accessed": False,
        "source": _SOURCE,
        "benchmark_row_sha256": benchmark_row_sha256,
        "benchmark_rows": int(len(benchmark_indices)),
        "folds": fold_records,
    }
    manifest["manifest_sha256"] = digest_json(manifest)
    atomic_write_json(output_root / "manifest.json", manifest)
    return manifest


__all__ = ["fit_event_pca_cache"]
