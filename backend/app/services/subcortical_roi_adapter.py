"""Exact Harvard-Oxford adapter for the released TRIBE v2 subcortical head."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Tuple

import numpy as np


TRIBE_SUBCORTICAL_VOXELS = 8808
ATLAS_NAME = "HarvardOxford sub-maxprob-thr50-2mm"
NON_NEURAL_LABEL_KEYWORDS = ("ventricle",)


@lru_cache(maxsize=1)
def _atlas_contract() -> Tuple[np.ndarray, list[str], np.ndarray]:
    """Return the exact mask order used by TRIBE and Homunculus.

    TRIBE flattens positive voxels from the 2 mm Harvard-Oxford subcortical
    mask in NumPy C-order. Keeping this implementation independent of the
    optional TRIBE plotting stack avoids pulling GUI dependencies into
    inference.
    """
    from nilearn import datasets

    atlas = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr50-2mm", verbose=0)
    labels = list(atlas.labels)
    excluded = ("cortex", "white", "stem", "background")
    excluded_indices = [
        index
        for index, label in enumerate(labels)
        if any(term in label.lower() for term in excluded)
    ]
    data = np.asarray(atlas.maps.get_fdata()).copy()
    data[np.isin(data, excluded_indices)] = 0
    flat_region_ids = data[data > 0].astype(np.int16)
    if flat_region_ids.size != TRIBE_SUBCORTICAL_VOXELS:
        raise RuntimeError(
            f"Expected {TRIBE_SUBCORTICAL_VOXELS} Harvard-Oxford voxels, "
            f"got {flat_region_ids.size}"
        )
    included_labels = [
        label
        for index, label in enumerate(labels)
        if index in set(int(value) for value in np.unique(flat_region_ids))
    ]
    return flat_region_ids, included_labels, np.asarray(data.shape, dtype=np.int32)


class SubcorticalRoiAdapter:
    """Project exact TRIBE subcortical voxel outputs into named ROI traces."""

    SCHEMA_VERSION = "tribe_subcortical_harvard_oxford_v1"

    def project(self, predictions: Any) -> Dict[str, Any]:
        voxels = self._time_by_voxel(predictions)
        flat_region_ids, labels, atlas_shape = _atlas_contract()

        trajectories = []
        region_rows = []
        for label in labels:
            region_id = self._region_id(label)
            indices = np.flatnonzero(flat_region_ids == region_id)
            trajectory = voxels[:, indices].mean(axis=1).astype(np.float32)
            trajectories.append(trajectory)
            values = voxels[:, indices]
            region_rows.append(
                {
                    "label": label,
                    "region_id": region_id,
                    "voxel_count": int(indices.size),
                    "interpretation_eligible": not self._is_non_neural(label),
                    "mean": float(np.mean(values)),
                    "mean_abs": float(np.mean(np.abs(values))),
                    "std": float(np.std(values)),
                    "peak_abs": float(np.max(np.abs(values))),
                    "temporal_variance": float(np.var(trajectory)),
                }
            )

        return {
            "schema_version": self.SCHEMA_VERSION,
            "atlas": ATLAS_NAME,
            "atlas_shape": atlas_shape.tolist(),
            "voxel_order": "mask_data[mask_data > 0] using NumPy C-order",
            "voxel_count": int(voxels.shape[1]),
            "timepoints": int(voxels.shape[0]),
            "region_labels": labels,
            "region_trajectories": np.stack(trajectories, axis=1).astype(np.float32),
            "region_metrics": region_rows,
            "interpretation_exclusions": [
                row["label"] for row in region_rows if not row["interpretation_eligible"]
            ],
            "limitations": [
                "Region trajectories are predicted BOLD proxies, not emotion labels.",
                "The released head is subject-specific and trained on Lahner/BOLD Moments.",
                "Ventricles remain in the source vector to preserve exact ordering but are excluded from interpretation.",
            ],
        }

    def feature_vector(self, projection: Dict[str, Any]) -> Tuple[np.ndarray, list[str]]:
        """Build stable, label-free summary features for supervised calibration."""
        trajectories = np.asarray(projection["region_trajectories"], dtype=np.float32)
        labels = list(projection["region_labels"])
        features = []
        names = []
        for index, label in enumerate(labels):
            trace = trajectories[:, index]
            for statistic, value in (
                ("mean", np.mean(trace)),
                ("mean_abs", np.mean(np.abs(trace))),
                ("std", np.std(trace)),
                ("peak_abs", np.max(np.abs(trace))),
                ("delta", trace[-1] - trace[0] if trace.size > 1 else 0.0),
            ):
                names.append(f"subcortical::{label}::{statistic}")
                features.append(float(value))
        return np.asarray(features, dtype=np.float32), names

    @staticmethod
    def _time_by_voxel(predictions: Any) -> np.ndarray:
        array = np.nan_to_num(np.asarray(predictions, dtype=np.float32))
        if array.ndim == 1:
            array = array.reshape(1, -1)
        elif array.ndim > 2:
            array = array.reshape(-1, array.shape[-1])
        if array.ndim != 2:
            raise ValueError(f"Unsupported subcortical prediction shape: {array.shape}")
        if array.shape[0] == TRIBE_SUBCORTICAL_VOXELS and array.shape[1] != TRIBE_SUBCORTICAL_VOXELS:
            array = array.T
        if array.shape[1] != TRIBE_SUBCORTICAL_VOXELS:
            raise ValueError(
                f"Expected {TRIBE_SUBCORTICAL_VOXELS} subcortical voxels, got {array.shape[1]}"
            )
        return array

    @staticmethod
    def _region_id(label: str) -> int:
        from nilearn import datasets

        atlas = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr50-2mm", verbose=0)
        return int(list(atlas.labels).index(label))

    @staticmethod
    def _is_non_neural(label: str) -> bool:
        return any(keyword in label.lower() for keyword in NON_NEURAL_LABEL_KEYWORDS)
