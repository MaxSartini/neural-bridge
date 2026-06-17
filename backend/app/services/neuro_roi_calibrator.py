"""ROI-based calibration for TRIBE cortical predictions.

This module turns raw TRIBE cortical BOLD predictions into transparent,
conservative behavioural axes. It is not a validated neuroscience model; it is
an auditable adapter layer that makes the BOLD-to-behaviour mapping explicit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from ..utils.logger import get_logger

logger = get_logger("neural_bridge.neuro_roi_calibrator")


FSAVERAGE5_VERTICES_PER_HEMI = 10242
TRIBE_CORTICAL_VERTICES = FSAVERAGE5_VERTICES_PER_HEMI * 2


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _nan_safe(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def _percentile_score(value: float, population: np.ndarray) -> float:
    pop = np.asarray(population, dtype=float)
    if pop.size == 0:
        return 0.5
    return _clamp01(float(np.mean(pop <= value)))


@dataclass(frozen=True)
class RoiGroup:
    name: str
    keywords: tuple[str, ...]
    interpretation: str


ROI_GROUPS: tuple[RoiGroup, ...] = (
    RoiGroup(
        name="salience_attention_proxy",
        keywords=("insula", "cingul-ant", "cingul-mid-ant", "front_inf", "supramar"),
        interpretation="Cortical proxy for salience, attention capture, and conflict monitoring.",
    ),
    RoiGroup(
        name="threat_avoidance_proxy",
        keywords=("insula", "cingul-ant", "subcallosal", "orbital", "front_inf-orbital"),
        interpretation="Weak cortical proxy for defensive, aversive, or risk-sensitive appraisal.",
    ),
    RoiGroup(
        name="reward_approach_proxy",
        keywords=("orbital", "rectus", "frontomargin", "subcallosal"),
        interpretation="Weak cortical proxy for valuation, approach, and positive appraisal.",
    ),
    RoiGroup(
        name="memory_context_proxy",
        keywords=("parahip", "precuneus", "temporal_middle", "temporal_inf", "pole_temporal"),
        interpretation="Proxy for contextual memory, narrative association, and recall relevance.",
    ),
    RoiGroup(
        name="uncertainty_control_proxy",
        keywords=("cingul-mid-ant", "front_middle", "front_sup", "intrapariet", "parietal"),
        interpretation="Proxy for ambiguity handling, cognitive control, and deliberative uncertainty.",
    ),
    RoiGroup(
        name="social_semantic_proxy",
        keywords=("temporal_sup", "temporal_middle", "angular", "precuneus", "front_inf-triangul"),
        interpretation="Proxy for social-semantic interpretation and narrative framing.",
    ),
)


class NeuroRoiCalibrator:
    """Summarise TRIBE fsaverage5 cortical predictions by surface parcels."""

    def __init__(self, atlas_dir: Optional[str] = None):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        self.atlas_dir = atlas_dir or os.path.join(project_root, "models", "neuro_atlases")

    def calibrate_predictions(self, predictions: Any) -> Dict[str, Any]:
        arr = _nan_safe(np.asarray(predictions))
        if arr.size == 0:
            return self._empty("No TRIBE predictions supplied.")

        arr = self._flatten_to_time_by_vertex(arr)
        global_metrics = self._global_metrics(arr)

        if arr.shape[-1] != TRIBE_CORTICAL_VERTICES:
            result = self._global_only(global_metrics, f"Expected {TRIBE_CORTICAL_VERTICES} cortical vertices, got {arr.shape[-1]}.")
            return result

        atlas = self._load_destrieux_atlas()
        if atlas is None:
            return self._global_only(global_metrics, "Surface atlas unavailable; used global TRIBE activation proxies only.")

        parcel_rows = self._parcel_metrics(arr, atlas)
        axis_scores = self._axis_scores(parcel_rows, global_metrics)
        profile = self._profile_from_axes(axis_scores, global_metrics)

        top_parcels = sorted(parcel_rows, key=lambda row: row["mean_abs"], reverse=True)[:12]
        profile.update({
            "roi_summary": {
                "atlas": "nilearn.fetch_atlas_surf_destrieux",
                "space": "fsaverage5",
                "top_parcels": top_parcels,
                "global_metrics": global_metrics,
            },
            "behavioural_axes": axis_scores,
            "calibration_trace": {
                "method": "destrieux_surface_percentile_adapter_v1",
                "vertex_count": int(arr.shape[-1]),
                "timepoints": int(arr.shape[0]),
                "roi_group_definitions": [
                    {
                        "name": group.name,
                        "keywords": list(group.keywords),
                        "interpretation": group.interpretation,
                    }
                    for group in ROI_GROUPS
                ],
            },
            "limitations": [
                "ROI-to-behaviour mapping is heuristic and requires empirical validation.",
                "Destrieux cortical parcels are coarse population-level BOLD proxies.",
                "TRIBE predicts population-average BOLD proxies, not emotion, intent, or individual behaviour.",
            ],
        })
        return profile

    def _flatten_to_time_by_vertex(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.ndim == 2:
            return arr
        return arr.reshape(-1, arr.shape[-1])

    def _global_metrics(self, arr: np.ndarray) -> Dict[str, float]:
        abs_arr = np.abs(arr)
        return {
            "mean_abs": float(np.mean(abs_arr)),
            "std": float(np.std(arr)),
            "temporal_variance": float(np.var(arr)),
            "peak_abs": float(np.max(abs_arr) if abs_arr.size else 0.0),
            "p95_abs": float(np.percentile(abs_arr, 95) if abs_arr.size else 0.0),
        }

    def _load_destrieux_atlas(self) -> Optional[Dict[str, Any]]:
        try:
            from nilearn import datasets

            atlas = datasets.fetch_atlas_surf_destrieux(data_dir=self.atlas_dir, verbose=0)
            return {
                "labels": list(atlas.labels),
                "left": np.asarray(atlas.map_left),
                "right": np.asarray(atlas.map_right),
            }
        except Exception as exc:
            logger.warning(f"Failed to load Destrieux surface atlas: {exc}")
            return None

    def _parcel_metrics(self, arr: np.ndarray, atlas: Dict[str, Any]) -> List[Dict[str, Any]]:
        labels = atlas["labels"]
        abs_arr = np.abs(arr)
        rows: List[Dict[str, Any]] = []

        offset = 0
        for hemisphere, label_map in (("L", atlas["left"]), ("R", atlas["right"])):
            for label_id, label_name in enumerate(labels):
                if label_id == 0:
                    continue
                local_idx = np.where(label_map == label_id)[0]
                if local_idx.size == 0:
                    continue
                idx = local_idx + offset
                values = arr[:, idx]
                abs_values = abs_arr[:, idx]
                rows.append({
                    "label_id": int(label_id),
                    "hemisphere": hemisphere,
                    "label": f"{hemisphere}:{label_name}",
                    "vertex_count": int(idx.size),
                    "mean_abs": float(np.mean(abs_values)),
                    "peak_abs": float(np.max(abs_values)),
                    "std": float(np.std(values)),
                    "temporal_variance": float(np.var(values)),
                })
            offset += len(label_map)
        return rows

    def _axis_scores(self, parcel_rows: List[Dict[str, Any]], global_metrics: Dict[str, float]) -> Dict[str, Any]:
        parcel_means = np.asarray([row["mean_abs"] for row in parcel_rows], dtype=float)
        axes: Dict[str, Any] = {}

        for group in ROI_GROUPS:
            matches = [
                row for row in parcel_rows
                if self._matches_keywords(str(row["label"]), group.keywords)
            ]
            if matches:
                mean_abs = float(np.mean([row["mean_abs"] for row in matches]))
                peak_abs = float(np.max([row["peak_abs"] for row in matches]))
                percentile = _percentile_score(mean_abs, parcel_means)
            else:
                mean_abs = 0.0
                peak_abs = 0.0
                percentile = 0.5
            axes[group.name] = {
                "score": _clamp01(percentile),
                "mean_abs": mean_abs,
                "peak_abs": peak_abs,
                "matched_parcels": [row["label"] for row in matches],
                "interpretation": group.interpretation,
            }

        axes["global_activation"] = {
            "score": _clamp01(global_metrics["p95_abs"] / (global_metrics["p95_abs"] + 1.0)),
            "metrics": global_metrics,
            "interpretation": "Global response magnitude proxy.",
        }
        return axes

    def _matches_keywords(self, label: str, keywords: Iterable[str]) -> bool:
        normalized = label.lower().replace("_", "-")
        return any(keyword.lower().replace("_", "-") in normalized for keyword in keywords)

    def _profile_from_axes(self, axes: Dict[str, Any], global_metrics: Dict[str, float]) -> Dict[str, Any]:
        salience = axes["salience_attention_proxy"]["score"]
        threat = axes["threat_avoidance_proxy"]["score"]
        reward = axes["reward_approach_proxy"]["score"]
        memory = axes["memory_context_proxy"]["score"]
        uncertainty = axes["uncertainty_control_proxy"]["score"]
        social_semantic = axes["social_semantic_proxy"]["score"]
        global_activation = axes["global_activation"]["score"]

        arousal = _clamp01(0.55 * salience + 0.25 * global_activation + 0.20 * _clamp01(global_metrics["std"]))
        approach = _clamp01(0.65 * reward + 0.20 * social_semantic + 0.15 * (1.0 - threat))
        avoidance = _clamp01(0.65 * threat + 0.25 * uncertainty + 0.10 * salience)
        polarisation = _clamp01(0.45 * threat + 0.35 * uncertainty + 0.20 * salience)
        virality = _clamp01(0.45 * salience + 0.30 * arousal + 0.25 * social_semantic)

        return {
            "salience_score": salience,
            "threat_score": threat,
            "reward_score": reward,
            "arousal_score": arousal,
            "uncertainty_score": uncertainty,
            "memory_relevance_score": memory,
            "approach_bias": approach,
            "avoidance_bias": avoidance,
            "polarisation_risk": polarisation,
            "virality_pressure": virality,
            "confidence": 0.48,
            "dominant_neural_interpretation": (
                "Destrieux surface-parcel TRIBE summary: salience, threat/avoidance, "
                "reward/approach, memory/context, uncertainty/control, and social-semantic proxies."
            ),
            "behavioural_prior_summary": (
                "TRIBE cortical predictions were mapped through named surface parcels into conservative "
                "population-level behavioural priors. This improves auditability over global activation only."
            ),
        }

    def _global_only(self, global_metrics: Dict[str, float], reason: str) -> Dict[str, Any]:
        mean_activation = _clamp01(global_metrics["mean_abs"])
        volatility = _clamp01(global_metrics["std"])
        peak_response = _clamp01(global_metrics["peak_abs"])
        temporal_variance = _clamp01(global_metrics["temporal_variance"])
        return {
            "salience_score": max(mean_activation, peak_response * 0.8),
            "threat_score": _clamp01(0.5 + (volatility - 0.5) * 0.4),
            "reward_score": 0.5,
            "arousal_score": max(temporal_variance, volatility),
            "uncertainty_score": volatility,
            "memory_relevance_score": max(mean_activation, temporal_variance),
            "approach_bias": 0.5,
            "avoidance_bias": _clamp01(0.5 + (volatility - 0.5) * 0.3),
            "polarisation_risk": _clamp01(0.5 + (volatility - 0.5) * 0.3),
            "virality_pressure": max(peak_response, mean_activation),
            "confidence": 0.35,
            "dominant_neural_interpretation": "TRIBE response summary using global activation proxies.",
            "behavioural_prior_summary": "TRIBE produced a population-level brain-response proxy; behavioural mapping is conservative.",
            "roi_summary": {"global_metrics": global_metrics},
            "behavioural_axes": {},
            "calibration_trace": {"method": "global_activation_proxy_v1", "fallback_reason": reason},
            "limitations": [reason, "No validated ROI-to-behaviour mapping implemented yet."],
        }

    def _empty(self, reason: str) -> Dict[str, Any]:
        return {
            "salience_score": 0.5,
            "threat_score": 0.5,
            "reward_score": 0.5,
            "arousal_score": 0.5,
            "uncertainty_score": 0.5,
            "memory_relevance_score": 0.5,
            "approach_bias": 0.5,
            "avoidance_bias": 0.5,
            "polarisation_risk": 0.5,
            "virality_pressure": 0.5,
            "confidence": 0.0,
            "limitations": [reason],
            "calibration_trace": {"method": "empty_neutral"},
        }
