"""Canonical, loss-aware intermediate representation for TRIBE responses."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .neuro_roi_calibrator import NeuroRoiCalibrator, TRIBE_CORTICAL_VERTICES
from .subcortical_roi_adapter import SubcorticalRoiAdapter


class NeuroResponseIRBuilder:
    """Translate raw TRIBE output into reusable spatial-temporal vocabularies."""

    SCHEMA_VERSION = "neuro_response_ir_v2"

    def __init__(self, sampling_hz: float = 2.0, atlas_dir: str | None = None):
        self.sampling_hz = float(sampling_hz)
        self.calibrator = NeuroRoiCalibrator(atlas_dir=atlas_dir)
        project_root = Path(__file__).resolve().parents[3]
        self.schaefer_dir = project_root / "models" / "neuro_atlases" / "schaefer2018"

    def build_from_npz(self, raw_path: str, output_dir: str) -> Dict[str, Any]:
        source = Path(raw_path).expanduser().resolve()
        with np.load(source) as bundle:
            predictions = np.asarray(bundle["predictions"], dtype=np.float32)
            subcortical = (
                np.asarray(bundle["subcortical_predictions"], dtype=np.float32)
                if "subcortical_predictions" in bundle
                else None
            )
            modality_missing_flags = (
                np.asarray(bundle["modality_missing_flags"], dtype=np.float32)
                if "modality_missing_flags" in bundle
                else None
            )
            segment_retention_features = (
                np.asarray(bundle["segment_retention_features"], dtype=np.float32)
                if "segment_retention_features" in bundle
                else None
            )
        return self.build(
            predictions,
            output_dir,
            source_path=source,
            subcortical_predictions=subcortical,
            modality_missing_flags=modality_missing_flags,
            segment_retention_features=segment_retention_features,
        )

    def build(
        self,
        predictions: np.ndarray,
        output_dir: str,
        source_path: Path | None = None,
        subcortical_predictions: np.ndarray | None = None,
        modality_missing_flags: np.ndarray | None = None,
        segment_retention_features: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        time_by_vertex = self._time_by_vertex(predictions)
        if time_by_vertex.shape[1] != TRIBE_CORTICAL_VERTICES:
            raise ValueError(
                f"Expected {TRIBE_CORTICAL_VERTICES} fsaverage5 vertices, "
                f"got {time_by_vertex.shape[1]}"
            )

        parcel_trajectories, parcel_labels, reconstruction = self._destrieux_projection(time_by_vertex)
        schaefer_trajectories, schaefer_labels, schaefer_reconstruction = self._schaefer400_projection(
            time_by_vertex
        )
        global_trajectories = np.stack(
            [
                time_by_vertex.mean(axis=1),
                np.abs(time_by_vertex).mean(axis=1),
                time_by_vertex.std(axis=1),
                np.abs(time_by_vertex).max(axis=1),
            ],
            axis=1,
        ).astype(np.float32)
        subcortical_projection = None
        subcortical_trajectories = None
        subcortical_features = None
        subcortical_feature_names: list[str] = []
        if subcortical_predictions is not None:
            subcortical_projection = SubcorticalRoiAdapter().project(subcortical_predictions)
            subcortical_trajectories = np.asarray(
                subcortical_projection.pop("region_trajectories"), dtype=np.float32
            )
            if subcortical_trajectories.shape[0] != time_by_vertex.shape[0]:
                raise ValueError(
                    "Cortical and subcortical predictions must have aligned timepoints: "
                    f"{time_by_vertex.shape[0]} != {subcortical_trajectories.shape[0]}"
                )
            feature_projection = dict(subcortical_projection)
            feature_projection["region_trajectories"] = subcortical_trajectories
            subcortical_features, subcortical_feature_names = (
                SubcorticalRoiAdapter().feature_vector(feature_projection)
            )
        calibration_features, calibration_feature_names = self._calibration_features(
            parcel_trajectories,
            parcel_labels,
            global_trajectories,
            subcortical_features,
            subcortical_feature_names,
            modality_missing_flags,
            segment_retention_features,
        )

        target_dir = Path(output_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        arrays_path = target_dir / "neuro_response_ir.npz"
        metadata_path = target_dir / "neuro_response_ir.json"
        arrays: Dict[str, np.ndarray] = {
            "parcel_trajectories": parcel_trajectories,
            "schaefer400_trajectories": schaefer_trajectories,
            "global_trajectories": global_trajectories,
            "timestamps_seconds": np.arange(time_by_vertex.shape[0], dtype=np.float32) / self.sampling_hz,
            "calibration_feature_vector": calibration_features,
        }
        if modality_missing_flags is not None:
            arrays["modality_missing_flags"] = np.asarray(modality_missing_flags, dtype=np.float32)
        if segment_retention_features is not None:
            arrays["segment_retention_features"] = np.asarray(segment_retention_features, dtype=np.float32)
        if subcortical_trajectories is not None and subcortical_features is not None:
            arrays["subcortical_roi_trajectories"] = subcortical_trajectories
            arrays["subcortical_summary_features"] = subcortical_features
        np.savez_compressed(
            arrays_path,
            **arrays,
        )

        metadata: Dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "source": {
                "model": "TRIBE v2",
                "representation": "population-average predicted cortical BOLD proxy",
                "space": "fsaverage5",
                "shape": list(time_by_vertex.shape),
                "sampling_hz": self.sampling_hz,
                "source_path": str(source_path) if source_path else "",
                "source_sha256": self._sha256(source_path) if source_path else "",
            },
            "translations": {
                "destrieux_parcels": {
                    "atlas": "nilearn.fetch_atlas_surf_destrieux",
                    "labels": parcel_labels,
                    "shape": list(parcel_trajectories.shape),
                    "method": "per-timepoint mean across vertices in each surface parcel",
                    "reconstruction_mae": reconstruction["mae"],
                    "reconstruction_rmse": reconstruction["rmse"],
                    "explained_variance": reconstruction["explained_variance"],
                },
                "global_trajectories": {
                    "labels": ["mean", "mean_abs", "std", "peak_abs"],
                    "shape": list(global_trajectories.shape),
                },
                "schaefer400_cortical": {
                    "atlas": "Schaefer2018 400Parcels 17Networks fsaverage5",
                    "labels": schaefer_labels,
                    "shape": list(schaefer_trajectories.shape),
                    "method": "per-timepoint mean across registered fsaverage5 vertices in each hemisphere-specific parcel",
                    "reconstruction_mae": schaefer_reconstruction["mae"],
                    "reconstruction_rmse": schaefer_reconstruction["rmse"],
                    "explained_variance": schaefer_reconstruction["explained_variance"],
                    "vertex_coverage": schaefer_reconstruction["vertex_coverage"],
                    "brain_jepa_status": "cortical 400/450 channels available; Tian Scale III subcortical 50 channels must be supplied or masked",
                },
            },
            "semantics": {
                "preserved": [
                    "temporal order",
                    "sampling interval",
                    "signed cortical response within parcel means",
                    "population-average scope",
                ],
                "not_inferred": [
                    "emotion labels",
                    "individual response",
                    "subgroup response",
                    "causal behavioral effect",
                ],
            },
            "feature_contract": {
                "schema_version": "neuro_calibration_features_v2",
                "feature_names": calibration_feature_names,
                "feature_count": len(calibration_feature_names),
                "includes_subcortical": subcortical_features is not None,
                "includes_missingness_masks": modality_missing_flags is not None,
                "includes_segment_retention_features": segment_retention_features is not None,
                "behavior_labels_inferred": False,
                "intended_use": "Input features for held-out supervised calibration and ablation tests.",
            },
            "arrays_path": str(arrays_path),
        }
        if subcortical_projection is not None and subcortical_trajectories is not None:
            metadata["translations"]["harvard_oxford_subcortical"] = {
                **subcortical_projection,
                "shape": list(subcortical_trajectories.shape),
                "method": "exact ROI means over the released head's Harvard-Oxford flat voxel order",
            }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    @staticmethod
    def _calibration_features(
        parcel_trajectories: np.ndarray,
        parcel_labels: list[str],
        global_trajectories: np.ndarray,
        subcortical_features: np.ndarray | None,
        subcortical_feature_names: list[str],
        modality_missing_flags: np.ndarray | None = None,
        segment_retention_features: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, list[str]]:
        """Create a stable label-free vector for downstream supervised models."""
        values = []
        names = []
        for index, label in enumerate(parcel_labels):
            trace = parcel_trajectories[:, index]
            for statistic, value in (
                ("mean", np.mean(trace)),
                ("mean_abs", np.mean(np.abs(trace))),
                ("std", np.std(trace)),
                ("peak_abs", np.max(np.abs(trace))),
                ("delta", trace[-1] - trace[0] if trace.size > 1 else 0.0),
            ):
                names.append(f"cortical::{label}::{statistic}")
                values.append(float(value))
        for index, label in enumerate(("mean", "mean_abs", "std", "peak_abs")):
            trace = global_trajectories[:, index]
            for statistic, value in (
                ("mean", np.mean(trace)),
                ("std", np.std(trace)),
                ("peak_abs", np.max(np.abs(trace))),
                ("delta", trace[-1] - trace[0] if trace.size > 1 else 0.0),
            ):
                names.append(f"global::{label}::{statistic}")
                values.append(float(value))
        if subcortical_features is not None:
            values.extend(float(value) for value in subcortical_features)
            names.extend(subcortical_feature_names)
        if modality_missing_flags is not None:
            labels = ("text", "audio", "video")
            for label, value in zip(labels, np.asarray(modality_missing_flags).reshape(-1)):
                names.append(f"missingness::{label}::is_missing")
                values.append(float(value))
        if segment_retention_features is not None:
            labels = (
                "retention_ratio",
                "kept_segments",
                "dropped_segments",
                "word_duration_repairs",
                "null_word_durations_after_repair",
            )
            for label, value in zip(labels, np.asarray(segment_retention_features).reshape(-1)):
                names.append(f"quality::{label}")
                values.append(float(value))
        return np.asarray(values, dtype=np.float32), names

    def _destrieux_projection(self, time_by_vertex: np.ndarray) -> Tuple[np.ndarray, list[str], Dict[str, float]]:
        atlas = self.calibrator._load_destrieux_atlas()
        if atlas is None:
            raise RuntimeError("Destrieux fsaverage5 atlas is unavailable")
        labels = []
        trajectories = []
        reconstructed = np.zeros_like(time_by_vertex)

        offset = 0
        for hemisphere, label_map in (("L", atlas["left"]), ("R", atlas["right"])):
            for label_id, label in enumerate(atlas["labels"]):
                if label_id == 0:
                    continue
                local_indices = np.where(label_map == label_id)[0]
                if not local_indices.size:
                    continue
                indices = local_indices + offset
                trajectory = time_by_vertex[:, indices].mean(axis=1)
                trajectories.append(trajectory)
                labels.append(f"{hemisphere}:{label}")
                reconstructed[:, indices] = trajectory[:, None]
            offset += len(label_map)

        error = time_by_vertex - reconstructed
        variance = float(np.var(time_by_vertex))
        return (
            np.stack(trajectories, axis=1).astype(np.float32),
            labels,
            self._reconstruction_metrics(error, variance),
        )

    def _schaefer400_projection(self, time_by_vertex: np.ndarray) -> Tuple[np.ndarray, list[str], Dict[str, float]]:
        from nibabel.freesurfer.io import read_annot

        atlas_root = self.schaefer_dir / "atl-schaefer2018" / "fsaverage5"
        files = [
            ("L", atlas_root / "atl-Schaefer2018_space-fsaverage5_hemi-L_desc-400Parcels17Networks_deterministic.annot"),
            ("R", atlas_root / "atl-Schaefer2018_space-fsaverage5_hemi-R_desc-400Parcels17Networks_deterministic.annot"),
        ]
        if not all(path.exists() for _, path in files):
            raise RuntimeError("Schaefer-400 fsaverage5 atlas is unavailable")

        trajectories = []
        names = []
        reconstructed = np.zeros_like(time_by_vertex)
        covered = np.zeros(time_by_vertex.shape[1], dtype=bool)
        offset = 0
        for hemisphere, path in files:
            vertex_labels, _, region_names = read_annot(str(path))
            for label_id, raw_name in enumerate(region_names):
                if label_id == 0:
                    continue
                indices_local = np.where(vertex_labels == label_id)[0]
                if not indices_local.size:
                    continue
                indices = indices_local + offset
                trajectory = time_by_vertex[:, indices].mean(axis=1)
                trajectories.append(trajectory)
                decoded = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
                names.append(f"{hemisphere}:{decoded}")
                reconstructed[:, indices] = trajectory[:, None]
                covered[indices] = True
            offset += len(vertex_labels)

        error = time_by_vertex[:, covered] - reconstructed[:, covered]
        source_variance = float(np.var(time_by_vertex[:, covered]))
        metrics = self._reconstruction_metrics(error, source_variance)
        metrics["vertex_coverage"] = float(np.mean(covered))
        return (
            np.stack(trajectories, axis=1).astype(np.float32),
            names,
            metrics,
        )

    @staticmethod
    def _reconstruction_metrics(error: np.ndarray, source_variance: float) -> Dict[str, float]:
        return {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "explained_variance": (
                float(1.0 - (np.var(error) / source_variance)) if source_variance > 0 else 1.0
            ),
        }

    @staticmethod
    def _time_by_vertex(predictions: np.ndarray) -> np.ndarray:
        array = np.nan_to_num(np.asarray(predictions, dtype=np.float32))
        if array.ndim == 1:
            array = array.reshape(1, -1)
        elif array.ndim > 2:
            array = array.reshape(-1, array.shape[-1])
        if array.ndim != 2:
            raise ValueError(f"Unsupported TRIBE prediction shape: {array.shape}")
        if array.shape[0] == TRIBE_CORTICAL_VERTICES and array.shape[1] != TRIBE_CORTICAL_VERTICES:
            array = array.T
        return array

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
