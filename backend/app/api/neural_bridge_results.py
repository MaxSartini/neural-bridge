"""Bridge-native, read-only API for frozen Neural Bridge result bundles.

The legacy neuro viewer intentionally exposes TRIBE cortical diagnostics.  This
module instead serves the final Neural Bridge timeline and keeps run-level,
implementation-reproduction, and model-validation evidence in separate scopes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from flask import jsonify, request, send_file

from . import neural_bridge_results_bp
from ..utils.logger import get_logger


logger = get_logger("neural_bridge.api.neural_bridge_results")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

MANIFEST_NAME = "run_manifest.json"
PREDICTIONS_NAME = "neural_bridge_predictions.csv"
REQUIRED_PREDICTION_COLUMNS = (
    "time_seconds",
    "future_arousal_movement_score",
    "within_video_percentile",
    "relative_top_5pct_spike_candidate",
)
MEMBER_PREFIX = "member_seed_"
FORECAST_START_SECONDS = 2.0
FORECAST_END_SECONDS = 5.0
HISTORY_ROWS = 5
UPSTREAM_WINDOW_SECONDS = 4.0

MODEL_VALIDATION_REFERENCE = {
    "evidence_id": "again_zero_label_direct_supervised_locked_confirmation_20260715",
    "evidence_scope": "same_frozen_lane_validation_context",
    "dataset_id": "AGAIN",
    "training_video_count": 696,
    "locked_video_count": 299,
    "matrix_rows": 140,
    "matrix_complete": True,
    "full_video_panel_wins": {"spearman": "5/5", "top_5pct_lift": "5/5", "event_pr_auc": "5/5"},
    "metrics": {
        "spearman": {
            "real": 0.1785132961,
            "strongest_control": 0.1004882655,
            "delta": 0.0780250306,
        },
        "top_5pct_lift": {
            "real": 0.0766079674,
            "strongest_control": 0.0448520122,
            "delta": 0.0317559552,
        },
        "event_pr_auc": {
            "real": 0.1710622218,
            "strongest_control": 0.1352295369,
            "delta": 0.0358326849,
        },
    },
    "whole_video_bootstrap_lower_95pct": {
        "spearman": 0.0606787212,
        "top_5pct_lift": 0.0187740072,
        "event_pr_auc": 0.0235455194,
    },
    "applies_to_this_analysis": False,
    "applicability_note": (
        "This is locked AGAIN validation context for the same frozen lane, not a per-video "
        "control result or external validation of this analysis."
    ),
}

UNSUPPORTED_OUTPUTS = {
    "arousal_dropoff": {
        "available": False,
        "reason": "The deployed target ranks future positive movement; low scores are not validated drop-offs.",
    },
    "valence": {
        "available": False,
        "reason": "This frozen video-only lane was not trained or validated for valence.",
    },
    "exact_arousal_level": {
        "available": False,
        "reason": "The score is an uncalibrated relative ranking output, not an exact arousal value.",
    },
    "individual_response": {
        "available": False,
        "reason": "The result is population-level response intelligence, not individual profiling.",
    },
    "causal_creative_explanation": {
        "available": False,
        "reason": "The output locates review candidates but does not establish why a response occurs.",
    },
}


class BundleProblem(Exception):
    """Expected bundle or request failure with a stable API error code."""

    def __init__(self, code: str, message: str, status: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


@dataclass(frozen=True)
class TimelineData:
    timestamps: np.ndarray
    scores: np.ndarray
    percentiles: np.ndarray
    candidates: np.ndarray
    members: dict[str, np.ndarray]


@dataclass(frozen=True)
class AnalysisBundle:
    analysis_id: str
    root: Path
    registry_root: Path
    manifest: dict[str, Any]
    timeline: TimelineData
    manifest_sha256: str
    predictions_sha256: str
    revision: str


def _success(schema_version: str, data: Any):
    return jsonify({"success": True, "schema_version": schema_version, "data": data})


@neural_bridge_results_bp.errorhandler(BundleProblem)
def _bundle_problem(error: BundleProblem):
    return (
        jsonify(
            {
                "success": False,
                "schema_version": "neural_bridge.error.v1",
                "error": {"code": error.code, "message": error.message},
            }
        ),
        error.status,
    )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def _path_list(value: str | None) -> list[Path]:
    if not value:
        return []
    return [Path(item).expanduser() for item in value.split(os.pathsep) if item.strip()]


def _external_root() -> Path:
    return Path(
        os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT", str(PROJECT_ROOT / "external_assets"))
    ).expanduser()


def _registry_roots() -> list[Path]:
    candidates = [_external_root() / "outputs"]
    candidates.extend(_path_list(os.environ.get("NEURAL_BRIDGE_RESULTS_ROOTS")))
    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        key = str(resolved)
        if key in seen or not resolved.is_dir():
            continue
        seen.add(key)
        roots.append(resolved)
    return roots


def _media_roots(bundle: AnalysisBundle | None = None) -> list[Path]:
    candidates = [PROJECT_ROOT / "data", _external_root() / "data"]
    candidates.extend(_path_list(os.environ.get("NEURAL_BRIDGE_MEDIA_ROOTS")))
    if bundle is not None:
        candidates.append(bundle.root)
    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        key = str(resolved)
        if key in seen or not resolved.is_dir():
            continue
        seen.add(key)
        roots.append(resolved)
    return roots


def _candidate_directories() -> Iterable[tuple[str, Path, Path]]:
    seen_ids: set[str] = set()
    for registry_root in _registry_roots():
        if (registry_root / MANIFEST_NAME).is_file() or (registry_root / PREDICTIONS_NAME).is_file():
            children = [registry_root]
            parent = registry_root
        else:
            try:
                children = sorted(
                    (item for item in registry_root.iterdir() if item.is_dir()),
                    key=lambda item: item.name,
                )
            except OSError:
                continue
            parent = registry_root
        for child in children:
            analysis_id = child.name
            if analysis_id in seen_ids or ANALYSIS_ID_RE.fullmatch(analysis_id) is None:
                continue
            try:
                resolved = child.resolve()
            except OSError:
                continue
            if not _is_within(resolved, parent):
                continue
            seen_ids.add(analysis_id)
            yield analysis_id, resolved, parent.resolve()


def _bundle_location(analysis_id: str) -> tuple[Path, Path]:
    if ANALYSIS_ID_RE.fullmatch(analysis_id) is None:
        raise BundleProblem("invalid_analysis_id", "Invalid analysis identifier.", 400)
    for candidate_id, root, registry_root in _candidate_directories():
        if candidate_id == analysis_id:
            return root, registry_root
    raise BundleProblem("analysis_not_found", "Analysis not found.", 404)


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BundleProblem("bundle_incomplete", "Analysis manifest is missing.", 409)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleProblem("invalid_manifest", "Analysis manifest is not valid JSON.", 422) from exc
    if not isinstance(payload, dict):
        raise BundleProblem("invalid_manifest", "Analysis manifest must be a JSON object.", 422)
    return payload


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(value)


def _read_timeline(path: Path) -> TimelineData:
    if not path.is_file():
        raise BundleProblem("bundle_incomplete", "Neural Bridge predictions CSV is missing.", 409)
    timestamps: list[float] = []
    scores: list[float] = []
    percentiles: list[float] = []
    candidates: list[bool] = []
    member_rows: dict[str, list[float]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            missing = [name for name in REQUIRED_PREDICTION_COLUMNS if name not in fieldnames]
            if missing:
                raise BundleProblem(
                    "invalid_predictions_schema",
                    "Predictions CSV is missing required columns.",
                    422,
                )
            member_names = sorted(
                (name for name in fieldnames if name.startswith(MEMBER_PREFIX)),
                key=lambda name: name[len(MEMBER_PREFIX) :],
            )
            member_rows = {name: [] for name in member_names}
            for row in reader:
                timestamps.append(float(row["time_seconds"]))
                scores.append(float(row["future_arousal_movement_score"]))
                percentiles.append(float(row["within_video_percentile"]))
                candidates.append(_parse_bool(row["relative_top_5pct_spike_candidate"]))
                for name in member_names:
                    member_rows[name].append(float(row[name]))
    except BundleProblem:
        raise
    except (OSError, UnicodeDecodeError, csv.Error, TypeError, ValueError) as exc:
        raise BundleProblem(
            "invalid_predictions_schema", "Predictions CSV contains invalid values.", 422
        ) from exc
    return TimelineData(
        timestamps=np.asarray(timestamps, dtype=np.float64),
        scores=np.asarray(scores, dtype=np.float64),
        percentiles=np.asarray(percentiles, dtype=np.float64),
        candidates=np.asarray(candidates, dtype=bool),
        members={name: np.asarray(values, dtype=np.float64) for name, values in member_rows.items()},
    )


def _positive_float(manifest: dict[str, Any], field: str) -> float:
    try:
        value = float(manifest[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise BundleProblem("invalid_manifest", f"Manifest field {field} is invalid.", 422) from exc
    if not math.isfinite(value) or value <= 0:
        raise BundleProblem("invalid_manifest", f"Manifest field {field} is invalid.", 422)
    return value


def _validate_bundle(manifest: dict[str, Any], timeline: TimelineData) -> None:
    status = str(manifest.get("status", ""))
    if status != "complete":
        raise BundleProblem("bundle_incomplete", "Analysis run is not complete.", 409)
    row_hz = _positive_float(manifest, "row_hz")
    duration = _positive_float(manifest, "duration_seconds")
    try:
        row_count = int(manifest["row_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BundleProblem("invalid_manifest", "Manifest row_count is invalid.", 422) from exc
    if row_count <= 0 or row_count != len(timeline.timestamps):
        raise BundleProblem(
            "row_count_mismatch", "Manifest and predictions row counts do not match.", 422
        )
    if not timeline.members:
        raise BundleProblem("invalid_predictions_schema", "Checkpoint member columns are missing.", 422)
    arrays = [timeline.timestamps, timeline.scores, timeline.percentiles, *timeline.members.values()]
    if any(len(array) != row_count or not np.isfinite(array).all() for array in arrays):
        raise BundleProblem("invalid_predictions_values", "Prediction values must be finite.", 422)
    if np.any(timeline.timestamps < 0) or np.any(np.diff(timeline.timestamps) <= 0):
        raise BundleProblem(
            "invalid_time_grid", "Prediction timestamps must be unique and increasing.", 422
        )
    expected_step = 1.0 / row_hz
    if row_count > 1 and not np.allclose(
        np.diff(timeline.timestamps), expected_step, rtol=0.0, atol=1e-5
    ):
        raise BundleProblem("invalid_time_grid", "Prediction timestamps do not match row_hz.", 422)
    if float(timeline.timestamps[-1]) > duration + expected_step:
        raise BundleProblem("invalid_time_grid", "Prediction timeline exceeds media duration.", 422)
    if np.any((timeline.percentiles < 0.0) | (timeline.percentiles > 1.0)):
        raise BundleProblem("invalid_predictions_values", "Percentiles must be in [0, 1].", 422)
    expected_candidates = max(1, int(math.ceil(0.05 * row_count)))
    if int(timeline.candidates.sum()) != expected_candidates:
        raise BundleProblem(
            "invalid_candidate_policy", "Candidate flags do not match the top-5% policy.", 422
        )
    candidate_scores = timeline.scores[timeline.candidates]
    noncandidate_scores = timeline.scores[~timeline.candidates]
    if len(noncandidate_scores) and float(candidate_scores.min()) < float(noncandidate_scores.max()) - 1e-9:
        raise BundleProblem(
            "invalid_candidate_policy", "Candidate flags are not the highest ranked scores.", 422
        )
    expected_percentiles = _average_percentile_rank(timeline.scores)
    if not np.allclose(timeline.percentiles, expected_percentiles, rtol=0.0, atol=1e-6):
        raise BundleProblem(
            "invalid_percentiles", "Within-video percentiles do not match score ranks.", 422
        )
    member_matrix = np.column_stack(list(timeline.members.values()))
    ensemble_delta = np.abs(member_matrix.mean(axis=1) - timeline.scores)
    if float(ensemble_delta.max(initial=0.0)) > 1e-6:
        raise BundleProblem(
            "invalid_ensemble", "Ensemble scores do not match the checkpoint-member mean.", 422
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _average_percentile_rank(values: np.ndarray) -> np.ndarray:
    """Match pandas rank(method='average', pct=True) without a pandas dependency."""
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_one_based_rank = ((start + 1) + end) / 2.0
        result[order[start:end]] = average_one_based_rank / len(values)
        start = end
    return result


def _load_bundle(analysis_id: str) -> AnalysisBundle:
    root, registry_root = _bundle_location(analysis_id)
    manifest_path = root / MANIFEST_NAME
    predictions_path = root / PREDICTIONS_NAME
    manifest = _read_manifest(manifest_path)
    timeline = _read_timeline(predictions_path)
    _validate_bundle(manifest, timeline)
    manifest_sha = _file_sha256(manifest_path)
    predictions_sha = _file_sha256(predictions_path)
    revision = hashlib.sha256(f"{manifest_sha}:{predictions_sha}".encode("ascii")).hexdigest()
    return AnalysisBundle(
        analysis_id=analysis_id,
        root=root,
        registry_root=registry_root,
        manifest=manifest,
        timeline=timeline,
        manifest_sha256=manifest_sha,
        predictions_sha256=predictions_sha,
        revision=revision,
    )


def _dataset_id(manifest: dict[str, Any]) -> str | None:
    explicit = manifest.get("dataset_id")
    if explicit:
        return str(explicit).upper()
    schema = str(manifest.get("schema_version", "")).lower()
    if schema.startswith("beat_"):
        return "BEAT"
    return None


def _source_kind(manifest: dict[str, Any]) -> str:
    explicit = manifest.get("source_kind")
    if explicit:
        return str(explicit)
    scope = str(manifest.get("scope", "")).lower()
    if "external" in scope or "unlabeled" in scope:
        return "external_experiment"
    if "benchmark" in scope:
        return "benchmark_video"
    return "analysis_bundle"


def _labels_available(manifest: dict[str, Any]) -> bool:
    if "labels_available" in manifest:
        return bool(manifest["labels_available"])
    return False


def _model_lane(manifest: dict[str, Any]) -> str:
    return str(manifest.get("neural_bridge_lane", "unknown"))


def _zero_label_lane(manifest: dict[str, Any]) -> bool:
    return _model_lane(manifest) == "video_supervised_temporal"


def _observed_arousal_used_at_inference(manifest: dict[str, Any]) -> bool | None:
    explicit = manifest.get("observed_arousal_used_at_inference")
    if isinstance(explicit, bool):
        return explicit
    lane = _model_lane(manifest)
    if lane == "video_supervised_temporal":
        return False
    if lane in {"short_temporal_conv_residual", "phase7_continuous_checkpoint_ensemble"}:
        return True
    return None


def _resolve_media(bundle: AnalysisBundle) -> Path | None:
    raw = str(bundle.manifest.get("input_video", "")).strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = bundle.root / candidate
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    if not resolved.is_file():
        return None
    if not any(_is_within(resolved, root) for root in _media_roots(bundle)):
        return None
    return resolved


def _evidence_scopes(bundle: AnalysisBundle) -> dict[str, Any]:
    reproduction = bundle.manifest.get("locked_prediction_reproduction")
    if not isinstance(reproduction, dict):
        reproduction = {"passed": False, "status": "not_recorded"}
    else:
        per_seed = reproduction.get("per_seed")
        if not isinstance(per_seed, dict):
            per_seed = {}
        reproduction = {
            "evidence_scope": "implementation_reproduction",
            "passed": bool(reproduction.get("passed", False)),
            "feature_width": reproduction.get("feature_width"),
            "train_rows": reproduction.get("train_rows"),
            "locked_rows": reproduction.get("locked_rows"),
            "locked_reproduction_skipped": bool(
                reproduction.get("locked_reproduction_skipped", False)
            ),
            "per_seed": {
                str(seed): {
                    "checkpoint_sha256": record.get("checkpoint_sha256"),
                    "max_abs_error": record.get("max_abs_error"),
                    "mean_abs_error": record.get("mean_abs_error"),
                    "passed": bool(record.get("passed", False)),
                }
                for seed, record in per_seed.items()
                if isinstance(record, dict)
            },
            "validates": "frozen head/scaler/checkpoint loading against sealed predictions",
            "does_not_validate": [
            "external predictive validity",
            "new-video correctness",
            "MLX versus H100 upstream feature parity",
            ],
        }
    labels_available = _labels_available(bundle.manifest)
    return {
        "implementation_reproduction": reproduction,
        "model_validation_reference": json.loads(json.dumps(MODEL_VALIDATION_REFERENCE)),
        "run_level_validation": {
            "evidence_scope": "this_analysis",
            "labels_available": labels_available,
            "controls_run": False,
            "validation_status": str(
                bundle.manifest.get("validation_status", "not_external_validation_without_labels")
            ),
            "external_validity": "not_established" if not labels_available else "not_recorded",
        },
        "upstream_backend_parity": {
            "evidence_scope": "implementation_compatibility",
            "status": "not_evaluated",
            "note": (
                "Frozen-head reproduction does not establish MLX V-JEPA/TRIBE parity with the "
                "upstream backend used to create locked validation caches."
            ),
        },
    }


def _video_id(bundle: AnalysisBundle) -> str:
    explicit = bundle.manifest.get("video_id")
    if explicit:
        return str(explicit)
    raw = str(bundle.manifest.get("input_video", "")).strip()
    return Path(raw).stem if raw else bundle.analysis_id


def _analysis_summary(bundle: AnalysisBundle) -> dict[str, Any]:
    manifest = bundle.manifest
    timeline = bundle.timeline
    media = _resolve_media(bundle)
    modalities = manifest.get("modalities")
    if not isinstance(modalities, list):
        modalities = []
    evidence = _evidence_scopes(bundle)
    return {
        "analysis_id": bundle.analysis_id,
        "analysis_revision": bundle.revision,
        "status": "complete",
        "video": {
            "video_id": _video_id(bundle),
            "display_name": _video_id(bundle),
            "duration_seconds": float(manifest["duration_seconds"]),
            "media_available": media is not None,
        },
        "source": {
            "kind": _source_kind(manifest),
            "dataset_id": _dataset_id(manifest),
            "labels_available": _labels_available(manifest),
        },
        "grid": {
            "row_hz": float(manifest["row_hz"]),
            "row_count": int(manifest["row_count"]),
            "start_seconds": float(timeline.timestamps[0]),
            "end_seconds": float(timeline.timestamps[-1]),
        },
        "inference": {
            "lane": _model_lane(manifest),
            "target": str(manifest.get("neural_bridge_target", "unknown")),
            "interpretation": str(manifest.get("neural_bridge_interpretation", "")),
            "training_supervision": "supervised" if _zero_label_lane(manifest) else "not_recorded",
            "observed_arousal_used_at_inference": _observed_arousal_used_at_inference(manifest),
            "modalities_used": [str(item) for item in modalities],
            "upstream_backend": {
                "vjepa": manifest.get("vjepa_backend"),
                "tribe": manifest.get("tribe_backend"),
            },
        },
        "outputs": {
            "candidate_count": int(timeline.candidates.sum()),
            "timeline": True,
            "media": media is not None,
            "report": True,
            "predictions_csv": True,
            "reference_labels": _labels_available(manifest),
            "raw_cortical_diagnostics": False,
        },
        "evidence": {
            "run_scope": str(manifest.get("scope", "not_recorded")),
            "validation_status": evidence["run_level_validation"]["validation_status"],
            "external_validity": evidence["run_level_validation"]["external_validity"],
            "exact_value_calibrated": False,
            "event_policy_status": "provisional_relative_rank",
        },
    }


def _parse_include_members() -> bool:
    value = request.args.get("include_members", "false")
    try:
        return _parse_bool(value)
    except ValueError as exc:
        raise BundleProblem(
            "invalid_query", "include_members must be true or false.", 400
        ) from exc


def _quality_arrays(bundle: AnalysisBundle) -> dict[str, list[Any]]:
    count = len(bundle.timeline.timestamps)
    duration = float(bundle.manifest["duration_seconds"])
    available = [min(index + 1, HISTORY_ROWS) for index in range(count)]
    full_bridge_history = [value == HISTORY_ROWS for value in available]
    full_upstream_window = [
        bool(float(timestamp) >= UPSTREAM_WINDOW_SECONDS - 1e-6)
        for timestamp in bundle.timeline.timestamps
    ]
    full_horizon = [
        bool(float(timestamp) + FORECAST_END_SECONDS <= duration + 1e-6)
        for timestamp in bundle.timeline.timestamps
    ]
    return {
        "bridge_history_rows_available": available,
        "full_bridge_history_context": full_bridge_history,
        "full_upstream_window_context": full_upstream_window,
        "full_forecast_window_in_media": full_horizon,
    }


def _event_items(bundle: AnalysisBundle, quality: dict[str, list[Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index in np.flatnonzero(bundle.timeline.candidates):
        anchor = float(bundle.timeline.timestamps[index])
        items.append(
            {
                "event_id": f"candidate-{len(items) + 1:04d}",
                "row_index": int(index),
                "anchor_time_seconds": anchor,
                "forecast_window_start_seconds": anchor + FORECAST_START_SECONDS,
                "forecast_window_end_seconds": anchor + FORECAST_END_SECONDS,
                "score": float(bundle.timeline.scores[index]),
                "percentile": float(bundle.timeline.percentiles[index]),
                "full_bridge_history_context": bool(
                    quality["full_bridge_history_context"][index]
                ),
                "full_upstream_window_context": bool(
                    quality["full_upstream_window_context"][index]
                ),
                "cold_start_context": not bool(
                    quality["full_upstream_window_context"][index]
                ),
                "full_forecast_window_in_media": bool(
                    quality["full_forecast_window_in_media"][index]
                ),
            }
        )
    return items


def _timeline_payload(bundle: AnalysisBundle, *, include_members: bool) -> dict[str, Any]:
    quality = _quality_arrays(bundle)
    payload: dict[str, Any] = {
        "analysis_id": bundle.analysis_id,
        "analysis_revision": bundle.revision,
        "grid": {
            "row_hz": float(bundle.manifest["row_hz"]),
            "row_count": len(bundle.timeline.timestamps),
            "timestamps_seconds": bundle.timeline.timestamps.tolist(),
        },
        "target": {
            "id": str(bundle.manifest.get("neural_bridge_target", "unknown")),
            "anchor_semantics": (
                "A score at time t ranks maximum positive arousal movement relative to t over "
                "the future t+2 to t+5 second window."
            ),
            "forecast_window_offset_seconds": [FORECAST_START_SECONDS, FORECAST_END_SECONDS],
            "scale": "uncalibrated_model_score",
            "higher_means": "higher relative rank of future positive movement",
        },
        "series": {
            "future_arousal_movement_score": {
                "values": bundle.timeline.scores.tolist(),
                "calibrated_exact_value": False,
            },
            "within_video_percentile": {
                "values": bundle.timeline.percentiles.tolist(),
                "scale": [0.0, 1.0],
                "reference_population": "rows within this video only",
            },
        },
        "events": {
            "policy": {
                "id": "artifact_within_video_top_5pct_v1",
                "percentile": 0.95,
                "provisional": True,
                "calibrated": False,
                "source_field": "relative_top_5pct_spike_candidate",
            },
            "items": _event_items(bundle, quality),
        },
        "row_quality": quality,
        "context_windows": {
            "bridge_history_rows": HISTORY_ROWS,
            "bridge_history_span_seconds_including_current": HISTORY_ROWS
            / float(bundle.manifest["row_hz"]),
            "upstream_vjepa_window_seconds": UPSTREAM_WINDOW_SECONDS,
        },
        "reference": {
            "available": _labels_available(bundle.manifest),
            "channels": [],
            "note": "Reference labels, when present, are display-only and separate from predictions.",
        },
        "unsupported_outputs": json.loads(json.dumps(UNSUPPORTED_OUTPUTS)),
        "evidence_scopes": _evidence_scopes(bundle),
    }
    if include_members:
        payload["diagnostics"] = {
            "member_scores": {
                name[len(MEMBER_PREFIX) :]: values.tolist()
                for name, values in bundle.timeline.members.items()
            },
            "interpretation": "Checkpoint-member diagnostic; not calibrated predictive uncertainty.",
        }
    return payload


def _report_payload(bundle: AnalysisBundle) -> dict[str, Any]:
    quality = _quality_arrays(bundle)
    events = sorted(
        _event_items(bundle, quality),
        key=lambda item: (-item["score"], item["anchor_time_seconds"]),
    )
    findings = []
    for item in events:
        anchor = item["anchor_time_seconds"]
        start = item["forecast_window_start_seconds"]
        end = item["forecast_window_end_seconds"]
        flags = []
        if item["cold_start_context"]:
            flags.append("cold_start_context")
        if not item["full_forecast_window_in_media"]:
            flags.append("incomplete_media_horizon")
        findings.append(
            {
                **item,
                "kind": "relative_future_arousal_movement_candidate",
                "basis_fields": [
                    "future_arousal_movement_score",
                    "within_video_percentile",
                    "relative_top_5pct_spike_candidate",
                ],
                "evidence_scope": "experimental_model_output_for_editorial_review",
                "quality_flags": flags,
                "wording_status": "bounded",
                "review_guidance": (
                    f"Review content at {anchor:.1f}s; the model ranks a potential relative response "
                    f"increase for {start:.1f}-{end:.1f}s in this video's provisional top 5%."
                ),
            }
        )
    manifest = bundle.manifest
    checkpoint_hashes = manifest.get("checkpoint_hashes")
    if not isinstance(checkpoint_hashes, dict):
        checkpoint_hashes = {}
    runtime = manifest.get("runtime_seconds")
    if not isinstance(runtime, dict):
        runtime = {}
    return {
        "report_id": f"{bundle.analysis_id}:{bundle.revision}:bounded-v1",
        "analysis_id": bundle.analysis_id,
        "analysis_revision": bundle.revision,
        "report_schema": "neural_bridge.report.v1",
        "title": f"Neural Bridge response review — {_video_id(bundle)}",
        "executive_summary": {
            "statement": (
                "This analysis ranks relative future arousal-movement candidates for editorial "
                "review. It does not measure an individual viewer or provide calibrated exact arousal."
            ),
            "candidate_count": len(findings),
            "interpretation": str(manifest.get("neural_bridge_interpretation", "")),
        },
        "media": {
            "video_id": _video_id(bundle),
            "duration_seconds": float(manifest["duration_seconds"]),
            "row_hz": float(manifest["row_hz"]),
            "row_count": int(manifest["row_count"]),
        },
        "findings": findings,
        "controls_and_validation": _evidence_scopes(bundle),
        "claim_boundaries": {
            "external_validity_established": False,
            "calibrated_exact_arousal": False,
            "event_threshold_calibrated": False,
            "arousal_dropoff_available": False,
            "valence_available": False,
            "population_level_decision_support_only": True,
            "individual_or_medical_inference": False,
        },
        "unsupported_outputs": json.loads(json.dumps(UNSUPPORTED_OUTPUTS)),
        "provenance": {
            "manifest_sha256": bundle.manifest_sha256,
            "predictions_sha256": bundle.predictions_sha256,
            "input_sha256": manifest.get("input_sha256"),
            "schema_version": manifest.get("schema_version"),
            "created_at": manifest.get("created_at"),
            "scope": manifest.get("scope"),
            "modalities_used": manifest.get("modalities", []),
            "vjepa_backend": manifest.get("vjepa_backend"),
            "vjepa_weights_sha256": manifest.get("vjepa_weights_sha256"),
            "tribe_backend": manifest.get("tribe_backend"),
            "neural_bridge_lane": _model_lane(manifest),
            "neural_bridge_target": manifest.get("neural_bridge_target"),
            "checkpoint_hashes": checkpoint_hashes,
            "runtime_seconds": runtime,
        },
    }


def _bounded_int_query(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = request.args.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise BundleProblem("invalid_query", f"{name} must be an integer.", 400) from exc
    if value < minimum or value > maximum:
        raise BundleProblem(
            "invalid_query", f"{name} must be between {minimum} and {maximum}.", 400
        )
    return value


@neural_bridge_results_bp.route("/analyses", methods=["GET"])
def list_analyses():
    items: list[dict[str, Any]] = []
    for analysis_id, root, _registry_root in _candidate_directories():
        if not (root / MANIFEST_NAME).is_file() or not (root / PREDICTIONS_NAME).is_file():
            continue
        try:
            items.append(_analysis_summary(_load_bundle(analysis_id)))
        except BundleProblem as exc:
            logger.warning("Skipping invalid Neural Bridge bundle %s: %s", analysis_id, exc.code)
    filters = {
        "dataset_id": request.args.get("dataset_id"),
        "lane": request.args.get("lane"),
        "status": request.args.get("status"),
        "source_kind": request.args.get("source_kind"),
    }
    if filters["dataset_id"]:
        value = str(filters["dataset_id"]).upper()
        items = [item for item in items if item["source"]["dataset_id"] == value]
    if filters["lane"]:
        items = [item for item in items if item["inference"]["lane"] == filters["lane"]]
    if filters["status"]:
        items = [item for item in items if item["status"] == filters["status"]]
    if filters["source_kind"]:
        items = [item for item in items if item["source"]["kind"] == filters["source_kind"]]
    items.sort(key=lambda item: item["analysis_id"])
    cursor = request.args.get("cursor")
    if cursor:
        items = [item for item in items if item["analysis_id"] > cursor]
    limit = _bounded_int_query("limit", 50, 1, 200)
    page = items[:limit]
    next_cursor = page[-1]["analysis_id"] if len(items) > limit and page else None
    return _success(
        "neural_bridge.analysis_list.v1",
        {"items": page, "next_cursor": next_cursor, "count": len(page)},
    )


@neural_bridge_results_bp.route("/analyses/<analysis_id>/timeline", methods=["GET"])
def analysis_timeline(analysis_id: str):
    bundle = _load_bundle(analysis_id)
    return _success(
        "neural_bridge.timeline.v1",
        _timeline_payload(bundle, include_members=_parse_include_members()),
    )


@neural_bridge_results_bp.route("/analyses/<analysis_id>/report", methods=["GET"])
def analysis_report(analysis_id: str):
    requested_format = request.args.get("format", "json").lower()
    if requested_format != "json":
        raise BundleProblem(
            "unsupported_report_format", "Only the deterministic JSON report is available.", 406
        )
    bundle = _load_bundle(analysis_id)
    return _success("neural_bridge.report.v1", _report_payload(bundle))


def _safe_etag(value: Any, fallback_path: Path) -> str:
    candidate = str(value or "").lower()
    return candidate if SHA256_RE.fullmatch(candidate) else _file_sha256(fallback_path)


@neural_bridge_results_bp.route(
    "/analyses/<analysis_id>/predictions.csv", methods=["GET", "HEAD"]
)
def analysis_predictions_csv(analysis_id: str):
    bundle = _load_bundle(analysis_id)
    path = bundle.root / PREDICTIONS_NAME
    return send_file(
        path,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"{analysis_id}-neural-bridge-predictions.csv",
        conditional=True,
        etag=bundle.predictions_sha256,
        max_age=0,
    )


@neural_bridge_results_bp.route("/analyses/<analysis_id>/media", methods=["GET", "HEAD"])
def analysis_media(analysis_id: str):
    bundle = _load_bundle(analysis_id)
    path = _resolve_media(bundle)
    if path is None:
        raise BundleProblem("media_not_found", "Registered media file is unavailable.", 404)
    mimetype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    response = send_file(
        path,
        mimetype=mimetype,
        as_attachment=False,
        download_name=path.name,
        conditional=True,
        etag=_safe_etag(bundle.manifest.get("input_sha256"), path),
        max_age=0,
    )
    response.headers["Accept-Ranges"] = "bytes"
    return response
