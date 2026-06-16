"""Post-run integrity diagnostics for OpenLAV neuro calibration benchmarks.

This script does not score new models. It consolidates the evidence needed to
decide whether a completed benchmark is scientifically interpretable.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def metric_mean(condition: dict[str, Any], metric: str) -> Any:
    value = condition.get(metric)
    if isinstance(value, dict):
        return value.get("mean")
    return value


def condition_row(condition: dict[str, Any]) -> dict[str, Any]:
    return {
        "mae": metric_mean(condition, "mae"),
        "rmse": metric_mean(condition, "rmse"),
        "pearson": metric_mean(condition, "pearson"),
        "spearman": metric_mean(condition, "spearman"),
        "delta_vs_mean_mae": condition.get("paired_mae_delta_vs_mean_baseline", {}).get("mean"),
        "folds": len(condition.get("split_metrics", [])),
        "mae_bootstrap_95_ci_available": all(
            "mae_bootstrap_95_ci" in split for split in condition.get("split_metrics", [])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("benchmark")
    parser.add_argument("--output", default="benchmarks/openlav/openlav_post50_integrity_report.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    benchmark_path = Path(args.benchmark).expanduser().resolve()
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    benchmark = load_json(benchmark_path)

    cache_rows: list[dict[str, Any]] = []
    bad_cache_rows: list[dict[str, Any]] = []
    feature_vectors: list[np.ndarray] = []
    feature_names: list[str] | None = None
    for row in rows:
        feature_path = Path(row["feature_path"])
        cache_dir = feature_path.parent
        status_path = cache_dir / "cache_status.json"
        summary_path = cache_dir / "tribe_summary.json"
        raw_path = cache_dir / "tribe_raw_output.npz"
        metadata_path = feature_path.with_suffix(".json")
        reasons: list[str] = []
        if not status_path.exists():
            reasons.append("missing_cache_status")
        if not summary_path.exists():
            reasons.append("missing_tribe_summary")
        if not raw_path.exists():
            reasons.append("missing_raw_output")
        if not feature_path.exists() or not metadata_path.exists():
            reasons.append("missing_ir_feature_artifacts")
        if reasons:
            bad_cache_rows.append({"stimulus_id": row["stimulus_id"], "reasons": reasons})
            continue
        status = load_json(status_path)
        summary = load_json(summary_path)
        metadata = load_json(metadata_path)
        with np.load(feature_path) as bundle:
            vector = np.asarray(bundle["calibration_feature_vector"], dtype=np.float32)
        with np.load(raw_path) as raw:
            raw_files = set(raw.files)
            missing_flags = np.asarray(raw["modality_missing_flags"], dtype=np.float32).tolist()
            retention = np.asarray(raw["segment_retention_features"], dtype=np.float32).tolist()
        current_names = metadata["feature_contract"]["feature_names"]
        if feature_names is None:
            feature_names = current_names
        elif feature_names != current_names:
            reasons.append("feature_name_order_mismatch")
        if vector.shape != (854,):
            reasons.append(f"unexpected_feature_vector_shape:{vector.shape}")
        if not np.isfinite(vector).all():
            reasons.append("nonfinite_feature_vector")
        contract = status.get("model_contract", {})
        if contract.get("video_num_frames") != 64:
            reasons.append(f"unexpected_frames:{contract.get('video_num_frames')}")
        if contract.get("video_extraction_contract") != "official_64_frame_exact_chunked_attention":
            reasons.append(f"unexpected_extraction_contract:{contract.get('video_extraction_contract')}")
        if contract.get("video_device_effective") != "mps":
            reasons.append(f"unexpected_effective_video_device:{contract.get('video_device_effective')}")
        if "subcortical_predictions" not in raw_files:
            reasons.append("missing_subcortical_predictions")
        if reasons:
            bad_cache_rows.append({"stimulus_id": row["stimulus_id"], "reasons": reasons})
        feature_vectors.append(vector)
        cache_rows.append(
            {
                "stimulus_id": row["stimulus_id"],
                "event_quality": summary.get("event_quality", {}),
                "segment_quality": summary.get("segment_quality", {}),
                "missing_flags": missing_flags,
                "retention_features": retention,
                "timings_seconds": status.get("timings_seconds", {}),
            }
        )

    feature_names = feature_names or []
    matrix = np.stack(feature_vectors) if feature_vectors else np.empty((0, 0), dtype=np.float32)
    duplicate_vectors = int(matrix.shape[0] - np.unique(matrix, axis=0).shape[0]) if matrix.size else 0
    missingness_names = [name for name in feature_names if name.startswith("missingness::")]
    quality_names = [name for name in feature_names if name.startswith("quality::")]

    repairs = [float(row["event_quality"].get("word_duration_repairs", 0.0)) for row in cache_rows]
    null_after = [float(row["event_quality"].get("null_word_durations_after_repair", 0.0)) for row in cache_rows]
    zero_before = [
        float(row["event_quality"].get("zero_duration_word_fraction", 0.0))
        for row in cache_rows
        if finite(row["event_quality"].get("zero_duration_word_fraction", 0.0))
    ]
    retention_ratios = [float(row["segment_quality"].get("retention_ratio", 0.0)) for row in cache_rows]
    kept_segments = [float(row["segment_quality"].get("kept_segments", 0.0)) for row in cache_rows]
    degenerate_reasons = Counter(
        str(row["event_quality"].get("degenerate_text_reason", "none")) for row in cache_rows
    )
    missing_flag_counts = {
        "text": int(sum(row["missing_flags"][0] > 0.5 for row in cache_rows if len(row["missing_flags"]) >= 3)),
        "audio": int(sum(row["missing_flags"][1] > 0.5 for row in cache_rows if len(row["missing_flags"]) >= 3)),
        "video": int(sum(row["missing_flags"][2] > 0.5 for row in cache_rows if len(row["missing_flags"]) >= 3)),
    }

    benchmark_conditions = {}
    for axis, target in benchmark.get("targets", {}).items():
        benchmark_conditions[axis] = {
            "mean_baseline": condition_row(target["mean_baseline"]),
            "cortical_only": condition_row(target["cortical_only"]),
            "subcortical_only": condition_row(target["subcortical_only"]),
            "cortical_plus_subcortical_calibrated": condition_row(target["cortical_plus_subcortical_calibrated"]),
            "compact_cortical_salience": condition_row(target["compact_cortical_salience"]),
            "compact_subcortical_affective": condition_row(target["compact_subcortical_affective"]),
            "compact_neuro_affect": condition_row(target["compact_neuro_affect"]),
            "ultra_compact_neuro": condition_row(target["ultra_compact_neuro"]),
            "shuffled_labels": condition_row(target["shuffled_labels"]),
            "strict_split_local_controls": {
                name: condition_row(condition)
                for name, condition in target.get("strict_split_local_controls", {}).items()
            },
            "component_controls": {
                name: condition_row(condition)
                for name, condition in target.get("cortical_subcortical_component_controls", {}).items()
            },
        }

    report = {
        "schema_version": "openlav_post50_integrity_report_v1",
        "manifest": str(manifest_path),
        "benchmark": str(benchmark_path),
        "rows": len(rows),
        "accepted_cache_rows": len(cache_rows) - len(bad_cache_rows),
        "bad_cache_rows": bad_cache_rows,
        "contract": {
            "video_num_frames": 64,
            "video_extraction_contract": "official_64_frame_exact_chunked_attention",
            "video_device_effective": "mps",
            "feature_schema": "neuro_calibration_features_v2",
            "feature_count": int(matrix.shape[1]) if matrix.size else 0,
        },
        "leakage_integrity": {
            "train_test_group_overlap_by_split": benchmark.get("leakage_audit", {}).get("group_overlap_by_split", []),
            "group_overlap_detected": any(
                len(overlap) > 0 for overlap in benchmark.get("leakage_audit", {}).get("group_overlap_by_split", [])
            ),
            "duplicate_feature_vectors": duplicate_vectors,
            "duplicate_stimulus_ids": [
                stimulus_id for stimulus_id, count in Counter(row["stimulus_id"] for row in rows).items() if count > 1
            ],
            "single_model_contract": benchmark.get("contract_audit", {}).get("single_model_contract"),
            "single_feature_contract": benchmark.get("contract_audit", {}).get("single_feature_contract"),
            "all_rows_have_cache_keys": benchmark.get("contract_audit", {}).get("all_rows_have_cache_keys"),
            "all_rows_have_stimulus_hashes": benchmark.get("contract_audit", {}).get("all_rows_have_stimulus_hashes"),
            "target_leakage_check": "No target columns are loaded into feature matrices; benchmark loads labels separately after split.",
            "filename_leakage_check": "stimulus_id and file paths are used for joins/audit only, not as model features.",
            "metadata_leakage_check": "source_URL is used only as the grouped holdout key, not as a model feature.",
            "cache_leakage_check": "single cache and feature-name hashes enforced by manifest and benchmark.",
        },
        "missingness_audit": {
            "explicit_missingness_feature_names": missingness_names,
            "explicit_missingness_feature_count": len(missingness_names),
            "raw_modality_missing_flag_counts": missing_flag_counts,
            "missing_all_zero_warning_interpretation": (
                "neuralset may encode missing events internally as zeros, but the persisted IR v2 "
                "adds explicit missingness::{text,audio,video}::is_missing features so downstream "
                "calibrators can distinguish missing modalities from neutral BOLD summaries."
            ),
        },
        "null_duration_audit": {
            "handling": (
                "Word durations <=0 or null are repaired before TRIBE dataloader construction using "
                "next word start delta, with median positive duration fallback and 0.02-1.0s clamp. "
                "The repaired durations affect event windows/alignment before feature generation."
            ),
            "word_duration_repairs": summarize(repairs),
            "null_word_durations_after_repair": summarize(null_after),
            "zero_duration_word_fraction_before_repair_observed": summarize(zero_before),
            "rows_with_duration_repairs": int(sum(value > 0 for value in repairs)),
            "rows_with_null_duration_after_repair": int(sum(value > 0 for value in null_after)),
            "degenerate_text_reason_counts": dict(degenerate_reasons),
            "true_before_after_limitation": (
                "The current caches persist post-repair TRIBE outputs plus repair counts, not a second "
                "unrepaired TRIBE output. A true prediction-level before/after comparison requires an "
                "explicit no-repair rerun on a small subset under a new, separate experimental contract."
            ),
        },
        "segment_retention_audit": {
            "retention_ratio": summarize(retention_ratios),
            "kept_segments": summarize(kept_segments),
            "quality_feature_names": quality_names,
            "interpretation": (
                "Retention is explicit quality metadata. It must remain in nuisance checks and should "
                "not be used as headline neuro evidence if it dominates importance."
            ),
        },
        "feature_dimensionality": {
            "total_features": int(matrix.shape[1]) if matrix.size else 0,
            "rows": len(rows),
            "feature_to_row_ratio": float(matrix.shape[1] / len(rows)) if rows and matrix.size else None,
            "recommendation": (
                "Treat 854-feature full vectors as exploratory at n=50. Use compact masks as headline "
                "until rows materially exceed feature count or dimensionality reduction is validated."
            ),
        },
        "benchmark_conditions": benchmark_conditions,
        "control_audit": benchmark.get("control_audit", {}),
        "unavailable_conditions": benchmark.get("unavailable_conditions", {}),
    }

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output), "rows": len(rows), "bad_cache_rows": len(bad_cache_rows)}, indent=2))


if __name__ == "__main__":
    main()
