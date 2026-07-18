"""Train-only representation builders for cached VEATIC/TRIBE raw cortex.

These builders are deliberately split out from the audit runner so tests can
check leakage-sensitive behavior without loading the external VEATIC cache.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from backend.app.services.cortical_roi_mapper import (
    CorticalRoiMapper,
    FSAVERAGE5_VERTICES_PER_HEMI,
    TRIBE_CORTICAL_VERTICES,
)
from backend.scripts import run_veatic_neuro_benchmark as bench


TEMPORAL_PREFIXES = bench.TEMPORAL_PREFIXES


@dataclass
class RepresentationMatrix:
    rows: list[dict[str, Any]]
    idx: np.ndarray
    values: np.ndarray
    keep_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class FittedRepresentation(Protocol):
    name: str

    def transform(self, rows: list[dict[str, Any]], idx: np.ndarray) -> RepresentationMatrix:
        ...

    def metadata(self) -> dict[str, Any]:
        ...


class RepresentationBuilder:
    name: str
    family: str
    uses_labels_for_fit: bool = False
    uses_future_features: bool = False
    compression_type: str = "none"
    feature_width: int | None = None

    def fit(
        self,
        train_rows: list[dict[str, Any]],
        train_idx: np.ndarray,
        all_rows: list[dict[str, Any]],
        base_feature_sets: dict[str, np.ndarray],
        y_train: np.ndarray | None = None,
        inner_validation: dict[str, Any] | None = None,
    ) -> FittedRepresentation:
        raise NotImplementedError


def _base_metadata(
    *,
    name: str,
    family: str,
    compression_type: str,
    train_rows: list[dict[str, Any]],
    train_idx: np.ndarray,
    feature_width: int,
    uses_labels_for_fit: bool,
    uses_future_features: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "representation_name": name,
        "family": family,
        "source": "raw cortical predictions from tribe_raw_output.npz",
        "fit_scope": "train_rows_only",
        "train_row_count": int(len(train_rows)),
        "train_index_count": int(train_idx.size),
        "feature_width": int(feature_width),
        "compression_type": compression_type,
        "transform_metadata": {},
        "leakage_contract": {
            "fit_on_train_rows_only": True,
            "uses_labels_for_fit": bool(uses_labels_for_fit),
            "uses_future_features": bool(uses_future_features),
            "crosses_video_boundaries": False,
            "test_labels_used_for_fit": False,
        },
    }
    if extra:
        metadata.update(extra)
    return metadata


def _fit_cache(inner_validation: dict[str, Any] | None) -> dict[str, Any] | None:
    if inner_validation is None:
        return None
    return inner_validation.setdefault("fit_cache", {})


def _fit_cache_dir(inner_validation: dict[str, Any] | None) -> Path | None:
    if inner_validation is None or not inner_validation.get("fit_cache_dir"):
        return None
    path = Path(inner_validation["fit_cache_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def _train_idx_digest(train_idx: np.ndarray) -> str:
    idx = np.ascontiguousarray(train_idx, dtype=np.int64)
    return hashlib.blake2b(idx.view(np.uint8), digest_size=12).hexdigest()


def _pca_cache_path(
    cache_dir: Path,
    *,
    raw: np.ndarray,
    train_idx: np.ndarray,
    components: int,
) -> Path:
    key = f"pca_projection:{int(components)}:{tuple(raw.shape)}:{bench.PCA_BACKEND}:{_train_idx_digest(train_idx)}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()
    return cache_dir / f"{digest}.npz"


def _read_pca_cache(cache_path: Path) -> tuple[np.ndarray, dict[str, Any]] | None:
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as bundle:
            projected_all = np.asarray(bundle["projected_all"], dtype=np.float32)
            pca_meta = json.loads(str(np.asarray(bundle["pca_meta_json"]).item()))
    except Exception:
        return None
    pca_meta = dict(pca_meta)
    pca_meta["cache_hit"] = True
    pca_meta["disk_cache_hit"] = True
    pca_meta["disk_cache_path"] = str(cache_path)
    return projected_all, pca_meta


def _write_pca_cache(cache_path: Path, projected_all: np.ndarray, pca_meta: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp")
    payload_meta = dict(pca_meta)
    payload_meta.pop("cache_hit", None)
    payload_meta.pop("disk_cache_hit", None)
    payload_meta.pop("disk_cache_path", None)
    with tmp_path.open("wb") as handle:
        np.savez(
            handle,
            projected_all=np.asarray(projected_all, dtype=np.float32),
            pca_meta_json=json.dumps(bench.json_safe(payload_meta), sort_keys=True),
        )
    tmp_path.replace(cache_path)


def _cached_pca_projection(
    raw: np.ndarray,
    train_idx: np.ndarray,
    components: int,
    inner_validation: dict[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache = _fit_cache(inner_validation)
    key = f"pca_projection:{int(components)}:{tuple(raw.shape)}:{bench.PCA_BACKEND}:{_train_idx_digest(train_idx)}"
    if cache is not None and key in cache:
        projected_all, pca_meta = cache[key]
        meta = dict(pca_meta)
        meta["cache_hit"] = True
        return projected_all, meta
    disk_cache_dir = _fit_cache_dir(inner_validation)
    disk_cache_path = None
    if disk_cache_dir is not None:
        disk_cache_path = _pca_cache_path(
            disk_cache_dir,
            raw=raw,
            train_idx=train_idx,
            components=components,
        )
        cached = _read_pca_cache(disk_cache_path)
        if cached is not None:
            projected_all, meta = cached
            if cache is not None:
                cache[key] = (projected_all, meta)
            return projected_all, meta
    projected_all, pca_meta = bench.pca_fit_transform(raw[train_idx], raw, components)
    projected_all = projected_all.astype(np.float32, copy=False)
    meta = dict(pca_meta)
    meta["cache_hit"] = False
    meta["fit_cache_key"] = key
    if disk_cache_path is not None:
        meta["disk_cache_path"] = str(disk_cache_path)
        _write_pca_cache(disk_cache_path, projected_all, meta)
    if cache is not None:
        cache[key] = (projected_all, meta)
    return projected_all, meta


@dataclass
class MatrixBackedRepresentation:
    name: str
    matrix: np.ndarray
    meta: dict[str, Any]
    valid_mask_all: np.ndarray | None = None
    auxiliary_matrices: dict[str, np.ndarray] = field(default_factory=dict)

    def transform(self, rows: list[dict[str, Any]], idx: np.ndarray) -> RepresentationMatrix:
        idx = np.asarray(idx, dtype=np.int64)
        if self.valid_mask_all is None:
            keep_mask = np.ones(idx.shape[0], dtype=bool)
        else:
            keep_mask = self.valid_mask_all[idx].astype(bool)
        kept_idx = idx[keep_mask]
        kept_rows = [row for row, keep in zip(rows, keep_mask) if keep]
        values = self.matrix[kept_idx].astype(np.float32, copy=False)
        meta = dict(self.meta)
        meta["test_row_count"] = int(len(kept_rows))
        return RepresentationMatrix(
            rows=kept_rows,
            idx=kept_idx,
            values=values,
            keep_mask=keep_mask,
            metadata=meta,
        )

    def metadata(self) -> dict[str, Any]:
        return dict(self.meta)


class ExistingFeatureBuilder(RepresentationBuilder):
    def __init__(self, name: str):
        self.name = name
        self.family = "frozen_reference"
        self.uses_labels_for_fit = False
        self.uses_future_features = False
        self.compression_type = "existing_benchmark_feature"
        self.feature_width = None

    def fit(
        self,
        train_rows: list[dict[str, Any]],
        train_idx: np.ndarray,
        all_rows: list[dict[str, Any]],
        base_feature_sets: dict[str, np.ndarray],
        y_train: np.ndarray | None = None,
        inner_validation: dict[str, Any] | None = None,
    ) -> FittedRepresentation:
        if self.name == "cortical_global":
            matrix = base_feature_sets["cortical_global"]
            meta_extra: dict[str, Any] = {"reference_behavior": "static cached global features"}
        elif self.name == "cortical_global_delta":
            matrix = base_feature_sets.get("cortical_global_delta")
            if matrix is None:
                matrix = bench.temporal_dynamics_features(
                    all_rows,
                    base_feature_sets["cortical_global"],
                    include_base=True,
                )
            meta_extra = {
                "reference_behavior": "existing cortical_global_delta dynamics",
                "temporal_dynamics": list(TEMPORAL_PREFIXES),
            }
        elif self.name in {"cortical_pca_64", "cortical_pca64_delta"}:
            raw = base_feature_sets["cortical_raw"]
            components = 64
            if self.name == "cortical_pca64_delta":
                projected_all, pca_meta = _cached_pca_projection(raw, train_idx, components, inner_validation)
                matrix = bench.temporal_dynamics_features(all_rows, projected_all, include_base=True)
                meta_extra = {
                    "reference_behavior": "existing cortical_pca64_delta benchmark path",
                    "pca": pca_meta,
                    "temporal_dynamics": list(TEMPORAL_PREFIXES),
                }
            else:
                projected_all, pca_meta = _cached_pca_projection(raw, train_idx, components, inner_validation)
                matrix = projected_all
                meta_extra = {
                    "reference_behavior": "existing cortical_pca_64 benchmark path",
                    "pca": pca_meta,
                }
        else:
            raise ValueError(f"Unsupported existing feature builder: {self.name}")
        meta = _base_metadata(
            name=self.name,
            family=self.family,
            compression_type=self.compression_type,
            train_rows=train_rows,
            train_idx=train_idx,
            feature_width=int(matrix.shape[1]),
            uses_labels_for_fit=False,
            uses_future_features=False,
            extra=meta_extra,
        )
        return MatrixBackedRepresentation(self.name, matrix.astype(np.float32, copy=False), meta)


class RawCurrentBuilder(RepresentationBuilder):
    name = "raw_current_ridge"
    family = "raw_direct"
    uses_labels_for_fit = False
    uses_future_features = False
    compression_type = "none"

    def fit(
        self,
        train_rows: list[dict[str, Any]],
        train_idx: np.ndarray,
        all_rows: list[dict[str, Any]],
        base_feature_sets: dict[str, np.ndarray],
        y_train: np.ndarray | None = None,
        inner_validation: dict[str, Any] | None = None,
    ) -> FittedRepresentation:
        matrix = base_feature_sets["cortical_raw"]
        meta = _base_metadata(
            name=self.name,
            family=self.family,
            compression_type=self.compression_type,
            train_rows=train_rows,
            train_idx=train_idx,
            feature_width=int(matrix.shape[1]),
            uses_labels_for_fit=False,
            uses_future_features=False,
            extra={"standardization": "head_train_only"},
        )
        return MatrixBackedRepresentation(self.name, matrix.astype(np.float32, copy=False), meta)


class PcaCurrentBuilder(RepresentationBuilder):
    family = "unsupervised_compression"
    uses_labels_for_fit = False
    uses_future_features = False
    compression_type = "pca_current"

    def __init__(self, components: int):
        self.components = int(components)
        self.name = f"pca_current_{self.components}"
        self.feature_width = self.components

    def fit(
        self,
        train_rows: list[dict[str, Any]],
        train_idx: np.ndarray,
        all_rows: list[dict[str, Any]],
        base_feature_sets: dict[str, np.ndarray],
        y_train: np.ndarray | None = None,
        inner_validation: dict[str, Any] | None = None,
    ) -> FittedRepresentation:
        raw = base_feature_sets["cortical_raw"]
        projected_all, pca_meta = _cached_pca_projection(raw, train_idx, self.components, inner_validation)
        meta = _base_metadata(
            name=self.name,
            family=self.family,
            compression_type=self.compression_type,
            train_rows=train_rows,
            train_idx=train_idx,
            feature_width=int(projected_all.shape[1]),
            uses_labels_for_fit=False,
            uses_future_features=False,
            extra={"pca": pca_meta},
        )
        return MatrixBackedRepresentation(self.name, projected_all.astype(np.float32, copy=False), meta)


class PcaDeltaBuilder(RepresentationBuilder):
    family = "unsupervised_compression"
    uses_labels_for_fit = False
    uses_future_features = False
    compression_type = "pca_delta"

    def __init__(self, components: int):
        self.components = int(components)
        self.name = f"pca_delta_{self.components}"
        self.feature_width = self.components * (len(TEMPORAL_PREFIXES) + 1)

    def fit(
        self,
        train_rows: list[dict[str, Any]],
        train_idx: np.ndarray,
        all_rows: list[dict[str, Any]],
        base_feature_sets: dict[str, np.ndarray],
        y_train: np.ndarray | None = None,
        inner_validation: dict[str, Any] | None = None,
    ) -> FittedRepresentation:
        raw = base_feature_sets["cortical_raw"]
        projected_all, pca_meta = _cached_pca_projection(raw, train_idx, self.components, inner_validation)
        matrix = bench.temporal_dynamics_features(all_rows, projected_all, include_base=True)
        meta = _base_metadata(
            name=self.name,
            family=self.family,
            compression_type=self.compression_type,
            train_rows=train_rows,
            train_idx=train_idx,
            feature_width=int(matrix.shape[1]),
            uses_labels_for_fit=False,
            uses_future_features=False,
            extra={"pca": pca_meta, "temporal_dynamics": list(TEMPORAL_PREFIXES)},
        )
        return MatrixBackedRepresentation(self.name, matrix.astype(np.float32, copy=False), meta)


class RandomProjectionBuilder(RepresentationBuilder):
    family = "random_projection_control"
    uses_labels_for_fit = False
    uses_future_features = False
    compression_type = "random_projection"

    def __init__(self, components: int, seed: int = 17):
        self.components = int(components)
        self.seed = int(seed)
        self.name = f"random_projection_{self.components}"
        self.feature_width = self.components

    def fit(
        self,
        train_rows: list[dict[str, Any]],
        train_idx: np.ndarray,
        all_rows: list[dict[str, Any]],
        base_feature_sets: dict[str, np.ndarray],
        y_train: np.ndarray | None = None,
        inner_validation: dict[str, Any] | None = None,
    ) -> FittedRepresentation:
        raw = base_feature_sets["cortical_raw"].astype(np.float32, copy=False)
        rng = np.random.default_rng(self.seed)
        projection = rng.normal(
            0.0,
            1.0 / math.sqrt(max(raw.shape[1], 1)),
            size=(raw.shape[1], self.components),
        ).astype(np.float32)
        matrix = raw @ projection
        meta = _base_metadata(
            name=self.name,
            family=self.family,
            compression_type=self.compression_type,
            train_rows=train_rows,
            train_idx=train_idx,
            feature_width=int(matrix.shape[1]),
            uses_labels_for_fit=False,
            uses_future_features=False,
            extra={"random_seed": self.seed, "projection_distribution": "normal_scaled_by_input_width"},
        )
        return MatrixBackedRepresentation(self.name, matrix.astype(np.float32, copy=False), meta)


def _video_sorted_indices(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(str(row["video_id"]), []).append(index)
    for indices in grouped.values():
        indices.sort(key=lambda item: float(rows[item]["time_start_seconds"]))
    return grouped


def causal_window_matrix(
    rows: list[dict[str, Any]],
    matrix: np.ndarray,
    *,
    window_seconds: float,
    aggregation: str,
    require_full_window: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    grouped = _video_sorted_indices(rows)
    width = int(matrix.shape[1])
    out_width = width if aggregation in {"mean", "last"} else width * 4
    output = np.zeros((matrix.shape[0], out_width), dtype=np.float32)
    valid = np.zeros(matrix.shape[0], dtype=bool)
    for indices in grouped.values():
        times = np.asarray([float(rows[index]["time_start_seconds"]) for index in indices], dtype=np.float64)
        values = matrix[np.asarray(indices, dtype=np.int64)].astype(np.float32, copy=False)
        first_time = float(times[0]) if times.size else 0.0
        for local_index, global_index in enumerate(indices):
            anchor = float(times[local_index])
            start = anchor - float(window_seconds)
            if require_full_window and start < first_time - 1e-9:
                continue
            start_pos = int(np.searchsorted(times, start, side="left"))
            window = values[start_pos : local_index + 1]
            if window.size == 0:
                continue
            last = values[local_index]
            if aggregation == "last":
                features = last
            elif aggregation == "mean":
                features = window.mean(axis=0)
            elif aggregation == "mean_std_last_slope":
                first = window[0]
                duration = max(anchor - float(times[start_pos]), 1.0)
                slope = (last - first) / duration
                features = np.concatenate([window.mean(axis=0), window.std(axis=0), last, slope], axis=0)
            else:
                raise ValueError(f"Unsupported causal aggregation: {aggregation}")
            output[global_index] = features.astype(np.float32, copy=False)
            valid[global_index] = True
    return output, valid, {
        "window_seconds": float(window_seconds),
        "aggregation": aggregation,
        "require_full_window": bool(require_full_window),
        "dropped_rows": int(np.sum(~valid)),
        "kept_rows": int(np.sum(valid)),
    }


class PcaSequenceBuilder(RepresentationBuilder):
    family = "causal_sequence_compression"
    uses_labels_for_fit = False
    uses_future_features = False
    compression_type = "pca_then_causal_window"

    def __init__(self, components: int, window_seconds: float, aggregation: str):
        self.components = int(components)
        self.window_seconds = float(window_seconds)
        self.aggregation = aggregation
        window_label = f"{int(self.window_seconds)}s" if self.window_seconds.is_integer() else f"{self.window_seconds:g}s"
        self.name = f"pca_sequence_{self.components}_causal_past_{window_label}_{aggregation}"
        multiplier = 1 if aggregation in {"last", "mean"} else 4
        self.feature_width = self.components * multiplier

    def fit(
        self,
        train_rows: list[dict[str, Any]],
        train_idx: np.ndarray,
        all_rows: list[dict[str, Any]],
        base_feature_sets: dict[str, np.ndarray],
        y_train: np.ndarray | None = None,
        inner_validation: dict[str, Any] | None = None,
    ) -> FittedRepresentation:
        raw = base_feature_sets["cortical_raw"]
        projected_all, pca_meta = _cached_pca_projection(raw, train_idx, self.components, inner_validation)
        matrix, valid, window_meta = causal_window_matrix(
            all_rows,
            projected_all,
            window_seconds=self.window_seconds,
            aggregation=self.aggregation,
            require_full_window=True,
        )
        meta = _base_metadata(
            name=self.name,
            family=self.family,
            compression_type=self.compression_type,
            train_rows=train_rows,
            train_idx=train_idx,
            feature_width=int(matrix.shape[1]),
            uses_labels_for_fit=False,
            uses_future_features=False,
            extra={
                "pca": pca_meta,
                "causal_window": window_meta,
                "leakage_contract": {
                    "fit_on_train_rows_only": True,
                    "uses_labels_for_fit": False,
                    "uses_future_features": False,
                    "crosses_video_boundaries": False,
                    "test_labels_used_for_fit": False,
                    "drops_rows_without_full_history": True,
                },
            },
        )
        return MatrixBackedRepresentation(
            self.name,
            matrix.astype(np.float32, copy=False),
            meta,
            valid,
            {"matched_current": projected_all.astype(np.float32, copy=False)},
        )


class RawCausalMeanBuilder(RepresentationBuilder):
    family = "raw_temporal"
    uses_labels_for_fit = False
    uses_future_features = False
    compression_type = "raw_causal_window"

    def __init__(self, window_seconds: float = 2.0):
        self.window_seconds = float(window_seconds)
        self.name = f"raw_causal_mean_{int(window_seconds)}s"

    def fit(
        self,
        train_rows: list[dict[str, Any]],
        train_idx: np.ndarray,
        all_rows: list[dict[str, Any]],
        base_feature_sets: dict[str, np.ndarray],
        y_train: np.ndarray | None = None,
        inner_validation: dict[str, Any] | None = None,
    ) -> FittedRepresentation:
        raw = base_feature_sets["cortical_raw"]
        matrix, valid, window_meta = causal_window_matrix(
            all_rows,
            raw,
            window_seconds=self.window_seconds,
            aggregation="mean",
            require_full_window=True,
        )
        meta = _base_metadata(
            name=self.name,
            family=self.family,
            compression_type=self.compression_type,
            train_rows=train_rows,
            train_idx=train_idx,
            feature_width=int(matrix.shape[1]),
            uses_labels_for_fit=False,
            uses_future_features=False,
            extra={"causal_window": window_meta},
        )
        return MatrixBackedRepresentation(
            self.name,
            matrix.astype(np.float32, copy=False),
            meta,
            valid,
            {"matched_current": raw.astype(np.float32, copy=False)},
        )


class RawCausalLastSlopeBuilder(RepresentationBuilder):
    family = "raw_temporal"
    uses_labels_for_fit = False
    uses_future_features = False
    compression_type = "raw_causal_last_slope"

    def __init__(self, window_seconds: float = 2.0):
        self.window_seconds = float(window_seconds)
        self.name = f"raw_causal_last_slope_{int(window_seconds)}s"

    def fit(
        self,
        train_rows: list[dict[str, Any]],
        train_idx: np.ndarray,
        all_rows: list[dict[str, Any]],
        base_feature_sets: dict[str, np.ndarray],
        y_train: np.ndarray | None = None,
        inner_validation: dict[str, Any] | None = None,
    ) -> FittedRepresentation:
        raw = base_feature_sets["cortical_raw"]
        matrix, valid, window_meta = causal_window_matrix(
            all_rows,
            raw,
            window_seconds=self.window_seconds,
            aggregation="mean_std_last_slope",
            require_full_window=True,
        )
        # Keep only last+slope for the named raw variant.
        width = raw.shape[1]
        matrix = np.concatenate([matrix[:, width * 2 : width * 3], matrix[:, width * 3 : width * 4]], axis=1)
        meta = _base_metadata(
            name=self.name,
            family=self.family,
            compression_type=self.compression_type,
            train_rows=train_rows,
            train_idx=train_idx,
            feature_width=int(matrix.shape[1]),
            uses_labels_for_fit=False,
            uses_future_features=False,
            extra={"causal_window": window_meta, "output_blocks": ["last", "slope"]},
        )
        return MatrixBackedRepresentation(
            self.name,
            matrix.astype(np.float32, copy=False),
            meta,
            valid,
            {"matched_current": raw.astype(np.float32, copy=False)},
        )


class PlsBuilder(RepresentationBuilder):
    family = "supervised_compression"
    uses_labels_for_fit = True
    uses_future_features = False
    compression_type = "pls_regression"

    def __init__(self, components: int):
        self.components = int(components)
        self.name = f"pls_{self.components}"
        self.feature_width = self.components

    def fit(
        self,
        train_rows: list[dict[str, Any]],
        train_idx: np.ndarray,
        all_rows: list[dict[str, Any]],
        base_feature_sets: dict[str, np.ndarray],
        y_train: np.ndarray | None = None,
        inner_validation: dict[str, Any] | None = None,
    ) -> FittedRepresentation:
        if y_train is None:
            raise ValueError(f"{self.name} requires y_train")
        from sklearn.cross_decomposition import PLSRegression

        raw = base_feature_sets["cortical_raw"].astype(np.float32, copy=False)
        train_x = raw[train_idx]
        mean = train_x.mean(axis=0, keepdims=True)
        std = train_x.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        max_components = max(1, min(self.components, train_x.shape[0] - 1, train_x.shape[1]))
        model = PLSRegression(n_components=max_components, scale=False)
        model.fit((train_x - mean) / std, y_train.reshape(-1, 1))
        matrix = model.transform((raw - mean) / std).astype(np.float32)
        meta = _base_metadata(
            name=self.name,
            family=self.family,
            compression_type=self.compression_type,
            train_rows=train_rows,
            train_idx=train_idx,
            feature_width=int(matrix.shape[1]),
            uses_labels_for_fit=True,
            uses_future_features=False,
            extra={
                "requested_components": self.components,
                "actual_components": int(max_components),
                "supervised_target": "outer_train_target_only",
                "selected_hyperparameters": {"components": int(max_components), "selection": "fixed_candidate"},
            },
        )
        return MatrixBackedRepresentation(self.name, matrix.astype(np.float32, copy=False), meta)


class TopKVerticesBuilder(RepresentationBuilder):
    family = "supervised_feature_selection"
    uses_labels_for_fit = True
    uses_future_features = False
    compression_type = "train_only_topk_vertices"

    def __init__(self, k: int):
        self.k = int(k)
        self.name = f"topk_vertices_{self.k}"
        self.feature_width = self.k

    def fit(
        self,
        train_rows: list[dict[str, Any]],
        train_idx: np.ndarray,
        all_rows: list[dict[str, Any]],
        base_feature_sets: dict[str, np.ndarray],
        y_train: np.ndarray | None = None,
        inner_validation: dict[str, Any] | None = None,
    ) -> FittedRepresentation:
        if y_train is None:
            raise ValueError(f"{self.name} requires y_train")
        raw = base_feature_sets["cortical_raw"].astype(np.float32, copy=False)
        train_x = raw[train_idx]
        y = np.asarray(y_train, dtype=np.float64)
        y_std = float(np.std(y))
        if y_std < 1e-12:
            scores = np.zeros(train_x.shape[1], dtype=np.float64)
            score_method = "constant_target"
        else:
            x_mean = train_x.mean(axis=0)
            x_centered = train_x - x_mean
            x_std = train_x.std(axis=0)
            y_centered = y - float(np.mean(y))
            denom = np.maximum(x_std * y_std * max(train_x.shape[0] - 1, 1), 1e-12)
            scores = np.abs((x_centered * y_centered[:, None]).sum(axis=0) / denom)
            score_method = "absolute_pearson_or_point_biserial"
        actual_k = min(self.k, raw.shape[1])
        selected = np.argsort(-scores, kind="mergesort")[:actual_k]
        matrix = raw[:, selected]
        digest = hashlib.blake2b(selected.astype(np.int64).tobytes(), digest_size=12).hexdigest()
        meta = _base_metadata(
            name=self.name,
            family=self.family,
            compression_type=self.compression_type,
            train_rows=train_rows,
            train_idx=train_idx,
            feature_width=int(matrix.shape[1]),
            uses_labels_for_fit=True,
            uses_future_features=False,
            extra={
                "score_method": score_method,
                "selected_vertex_count": int(actual_k),
                "selected_vertices_digest": digest,
                "selected_vertices": selected.astype(int).tolist()[: min(100, actual_k)],
                "selected_hyperparameters": {"top_k": int(actual_k), "selection": "fixed_candidate"},
            },
        )
        return MatrixBackedRepresentation(self.name, matrix.astype(np.float32, copy=False), meta)


class SupervisedPcaAfterTopKBuilder(RepresentationBuilder):
    family = "supervised_feature_selection"
    uses_labels_for_fit = True
    uses_future_features = False
    compression_type = "topk_then_pca"

    def __init__(self, top_k: int, components: int):
        self.top_k = int(top_k)
        self.components = int(components)
        self.name = f"supervised_pca_topk_{self.top_k}_pca_{self.components}"
        self.feature_width = self.components

    def fit(
        self,
        train_rows: list[dict[str, Any]],
        train_idx: np.ndarray,
        all_rows: list[dict[str, Any]],
        base_feature_sets: dict[str, np.ndarray],
        y_train: np.ndarray | None = None,
        inner_validation: dict[str, Any] | None = None,
    ) -> FittedRepresentation:
        topk = TopKVerticesBuilder(self.top_k).fit(train_rows, train_idx, all_rows, base_feature_sets, y_train)
        selected_matrix = topk.matrix if isinstance(topk, MatrixBackedRepresentation) else topk.transform(all_rows, np.arange(len(all_rows))).values
        projected_all, pca_meta = bench.pca_fit_transform(selected_matrix[train_idx], selected_matrix, self.components)
        meta = _base_metadata(
            name=self.name,
            family=self.family,
            compression_type=self.compression_type,
            train_rows=train_rows,
            train_idx=train_idx,
            feature_width=int(projected_all.shape[1]),
            uses_labels_for_fit=True,
            uses_future_features=False,
            extra={"topk_metadata": topk.metadata(), "pca": pca_meta},
        )
        return MatrixBackedRepresentation(self.name, projected_all.astype(np.float32, copy=False), meta)


class RoiParcelBuilder(RepresentationBuilder):
    name = "roi_parcel_features"
    family = "atlas_compression"
    uses_labels_for_fit = False
    uses_future_features = False
    compression_type = "destrieux_parcel_mean"

    def fit(
        self,
        train_rows: list[dict[str, Any]],
        train_idx: np.ndarray,
        all_rows: list[dict[str, Any]],
        base_feature_sets: dict[str, np.ndarray],
        y_train: np.ndarray | None = None,
        inner_validation: dict[str, Any] | None = None,
    ) -> FittedRepresentation:
        raw = base_feature_sets["cortical_raw"].astype(np.float32, copy=False)
        if raw.shape[1] != TRIBE_CORTICAL_VERTICES:
            raise RuntimeError(f"ROI atlas expects {TRIBE_CORTICAL_VERTICES} vertices, got {raw.shape[1]}")
        atlas = CorticalRoiMapper().load_destrieux_atlas()
        if not atlas:
            raise RuntimeError("Destrieux atlas unavailable")
        labels = np.concatenate([np.asarray(atlas["left"]), np.asarray(atlas["right"])])
        if labels.shape[0] != raw.shape[1]:
            raise RuntimeError(f"Atlas label count {labels.shape[0]} does not match raw width {raw.shape[1]}")
        parcels = [int(label) for label in sorted(set(labels.tolist())) if int(label) >= 0]
        columns = []
        parcel_sizes = {}
        for label in parcels:
            mask = labels == label
            if not np.any(mask):
                continue
            parcel_sizes[str(label)] = int(np.sum(mask))
            values = raw[:, mask]
            columns.append(values.mean(axis=1))
        if not columns:
            raise RuntimeError("No atlas parcels could be mapped")
        matrix = np.stack(columns, axis=1).astype(np.float32)
        meta = _base_metadata(
            name=self.name,
            family=self.family,
            compression_type=self.compression_type,
            train_rows=train_rows,
            train_idx=train_idx,
            feature_width=int(matrix.shape[1]),
            uses_labels_for_fit=False,
            uses_future_features=False,
            extra={
                "atlas": "destrieux_surface",
                "hemi_vertices": FSAVERAGE5_VERTICES_PER_HEMI,
                "parcel_count": int(matrix.shape[1]),
                "parcel_sizes": parcel_sizes,
            },
        )
        return MatrixBackedRepresentation(self.name, matrix, meta)


def builder_from_name(name: str, *, seed: int = 17) -> RepresentationBuilder:
    if name in {"cortical_global", "cortical_global_delta", "cortical_pca_64", "cortical_pca64_delta"}:
        return ExistingFeatureBuilder(name)
    if name == "raw_current_ridge":
        return RawCurrentBuilder()
    if name == "raw_causal_mean_2s":
        return RawCausalMeanBuilder(2.0)
    if name == "raw_causal_last_slope_2s":
        return RawCausalLastSlopeBuilder(2.0)
    if name.startswith("pca_current_"):
        return PcaCurrentBuilder(int(name.removeprefix("pca_current_")))
    if name.startswith("pca_delta_"):
        return PcaDeltaBuilder(int(name.removeprefix("pca_delta_")))
    if name.startswith("random_projection_"):
        return RandomProjectionBuilder(int(name.removeprefix("random_projection_")), seed=seed)
    if name.startswith("pls_"):
        return PlsBuilder(int(name.removeprefix("pls_")))
    if name.startswith("topk_vertices_"):
        return TopKVerticesBuilder(int(name.removeprefix("topk_vertices_")))
    if name == "roi_parcel_features":
        return RoiParcelBuilder()
    if name.startswith("pca_sequence_"):
        parts = name.split("_")
        # pca_sequence_128_causal_past_2s_mean or ..._mean_std_last_slope
        components = int(parts[2])
        window = float(parts[5].removesuffix("s"))
        aggregation = "_".join(parts[6:])
        return PcaSequenceBuilder(components, window, aggregation)
    if name.startswith("supervised_pca_topk_"):
        parts = name.split("_")
        return SupervisedPcaAfterTopKBuilder(int(parts[3]), int(parts[5]))
    raise ValueError(f"Unknown representation candidate: {name}")


SMOKE_CANDIDATES = (
    "cortical_pca64_delta",
    "cortical_pca_64",
    "pca_current_32",
    "pca_delta_64",
)
PRIMARY_CANDIDATES = (
    "cortical_pca64_delta",
    "cortical_pca_64",
    "raw_current_ridge",
    "pca_current_128",
    "pca_current_256",
    "pca_delta_128",
    "pca_delta_256",
    "pca_sequence_128_causal_past_2s_mean",
    "pca_sequence_128_causal_past_2s_mean_std_last_slope",
    "pca_sequence_256_causal_past_2s_mean",
    "pls_32",
    "pls_64",
    "topk_vertices_512",
    "roi_parcel_features",
)
FULL_CANDIDATES = (
    "cortical_global",
    "cortical_global_delta",
    "cortical_pca_64",
    "cortical_pca64_delta",
    "raw_current_ridge",
    "raw_causal_mean_2s",
    "raw_causal_last_slope_2s",
    "pca_current_32",
    "pca_current_64",
    "pca_current_96",
    "pca_current_128",
    "pca_current_192",
    "pca_current_256",
    "pca_current_384",
    "pca_current_512",
    "pca_delta_64",
    "pca_delta_128",
    "pca_delta_256",
    "random_projection_64",
    "random_projection_128",
    "random_projection_256",
    "random_projection_512",
    "roi_parcel_features",
    "pls_8",
    "pls_16",
    "pls_32",
    "pls_64",
    "pls_128",
    "topk_vertices_128",
    "topk_vertices_256",
    "topk_vertices_512",
    "topk_vertices_1024",
    "supervised_pca_topk_512_pca_32",
    "supervised_pca_topk_512_pca_64",
    "supervised_pca_topk_1024_pca_128",
    "pca_sequence_64_causal_past_1s_last",
    "pca_sequence_64_causal_past_2s_mean",
    "pca_sequence_64_causal_past_3s_mean_std_last_slope",
    "pca_sequence_128_causal_past_1s_last",
    "pca_sequence_128_causal_past_2s_mean",
    "pca_sequence_128_causal_past_2s_mean_std_last_slope",
    "pca_sequence_256_causal_past_2s_mean",
)


def candidate_names_for_mode(mode: str) -> tuple[str, ...]:
    if mode == "smoke":
        return SMOKE_CANDIDATES
    if mode == "primary-audit":
        return PRIMARY_CANDIDATES
    if mode == "full-audit":
        return FULL_CANDIDATES
    raise ValueError(f"Unknown candidate mode: {mode}")


def describe_candidates(names: tuple[str, ...], *, seed: int = 17) -> list[dict[str, Any]]:
    rows = []
    for name in names:
        builder = builder_from_name(name, seed=seed)
        rows.append(
            {
                "name": builder.name,
                "family": builder.family,
                "uses_labels_for_fit": bool(builder.uses_labels_for_fit),
                "uses_future_features": bool(builder.uses_future_features),
                "compression_type": builder.compression_type,
                "feature_width": builder.feature_width,
            }
        )
    return rows
