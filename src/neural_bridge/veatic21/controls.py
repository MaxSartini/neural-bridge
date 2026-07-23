"""Lifecycle-complete matched controls for the VEATIC 2.1 causal bridge."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from .data import CanonicalSubstrate
from .evidence import (
    atomic_save_npz,
    atomic_write_json,
    average_precision_skill,
    digest_json,
    pooled_pr_auc,
    row_identity_digest,
    sha256_file,
)
from .pca_cache import load_event_pca_projection
from .preregistration import benchmark_partition_mask
from .protocol import (
    causal_ar_features,
    event_labels,
    fit_event_threshold,
    future_target_values,
    target_support_mask,
)
from .runner import (
    _causal_video_means,
    _circular_permute_labels,
    _lane_rng,
    _permute_within_video,
)
from .stage1 import _causal_design, _owned_rows, _target
from .supervised_projection import (
    SupervisedProjectionCellConfig,
    _train_lane,
)

_SCHEMA = "veatic21_control_plan_v1"
_REAL = "real_causal_residual"
_MATCHED_CONTROLS = (
    "sequence_shuffled_residual",
    "random_pca_residual",
    "causal_prefix_video_mean_residual",
    "diagnostics_only_residual",
    "label_permutation_residual",
)
_ABLATIONS = ("current_row_only_residual",)
_TRAINED_LANES = (*_MATCHED_CONTROLS, *_ABLATIONS)


def _require_self_digest(record: Mapping[str, Any], field: str) -> None:
    expected = record.get(field)
    if not isinstance(expected, str):
        raise ValueError(f"artifact is missing {field}")
    payload = dict(record)
    payload.pop(field)
    if digest_json(payload) != expected:
        raise ValueError(f"artifact failed its {field} integrity check")


def build_control_plan(
    preregistration: Mapping[str, Any],
    recipe_plan: Mapping[str, Any],
    recipe_selection: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    pca_manifest: Mapping[str, Any],
    lifecycle_crosswalk: Path,
) -> dict[str, Any]:
    """Register all matched residual controls before further stability work."""

    _require_self_digest(preregistration, "preregistration_sha256")
    _require_self_digest(recipe_plan, "plan_sha256")
    _require_self_digest(recipe_selection, "resolution_sha256")
    _require_self_digest(baseline_summary, "summary_sha256")
    _require_self_digest(pca_manifest, "manifest_sha256")
    if recipe_selection.get("plan_sha256") != recipe_plan.get("plan_sha256"):
        raise ValueError("control selection does not resolve the supplied recipe plan")
    source_summary_sha256 = recipe_plan.get("artifacts", {}).get(
        "representation_summary_sha256"
    )
    if source_summary_sha256 != baseline_summary.get("summary_sha256"):
        raise ValueError("control plan does not bind the real comparison summary")
    if pca_manifest.get("manifest_sha256") != recipe_plan.get("artifacts", {}).get(
        "pca_manifest_sha256"
    ):
        raise ValueError("control artifacts do not share the PCA manifest")
    if baseline_summary.get("completed_cells") != baseline_summary.get("expected_cells"):
        raise ValueError("controls require a complete real comparison summary")
    real_records = [row for row in baseline_summary["records"] if row.get("lane") == "fixed_pca512"]
    cells = int(recipe_plan["matrix"]["cells_per_candidate"])
    if len(real_records) != cells:
        raise ValueError("controls require the exact real PCA-512 causal panel")
    if not lifecycle_crosswalk.is_file():
        raise ValueError("lifecycle control crosswalk is missing")

    plan: dict[str, Any] = {
        "schema": _SCHEMA,
        "purpose": "lifecycle_complete_matched_controls_before_stability",
        "artifacts": {
            "baseline_summary_sha256": baseline_summary["summary_sha256"],
            "control_code_sha256": sha256_file(Path(__file__)),
            "lifecycle_crosswalk_sha256": sha256_file(lifecycle_crosswalk),
            "pca_manifest_sha256": pca_manifest["manifest_sha256"],
            "preregistration_sha256": preregistration["preregistration_sha256"],
            "recipe_plan_sha256": recipe_plan["plan_sha256"],
            "recipe_selection_sha256": recipe_selection["resolution_sha256"],
        },
        "selected_recipe": dict(recipe_selection["selected_recipe"]),
        "selected_representation": "fixed_pca512",
        "selected_head_family": "frozen_ar_plus_causal_temporal_residual",
        "execution_plan": dict(recipe_plan["execution_plan"]),
        "matrix": {
            "targets": list(recipe_plan["matrix"]["targets"]),
            "folds": list(recipe_plan["matrix"]["folds"]),
            "comparison_seeds": list(recipe_plan["matrix"]["comparison_seeds"]),
            "reused_lanes": ["target_specific_frozen_ar", _REAL],
            "matched_control_lanes": list(_MATCHED_CONTROLS),
            "ablation_lanes": list(_ABLATIONS),
            "trained_lanes": list(_TRAINED_LANES),
            "cells_per_lane": cells,
            "expected_new_cells": cells * len(_TRAINED_LANES),
            "worker_count": 1,
            "backend": "mlx",
            "sealed_tail_labels": True,
        },
        "control_semantics": {
            "sequence_shuffled_residual": "within_video_pca_row_permutation_independently_trained",
            "random_pca_residual": "seed_namespaced_shape_matched_gaussian_independently_trained",
            "causal_prefix_video_mean_residual": "causal_prefix_mean_only_independently_trained",
            "diagnostics_only_residual": "canonical_video_diagnostics_independently_trained",
            "label_permutation_residual": (
                "nonzero_within_video_circular_residual_label_shift_with_identical_ar_floor"
            ),
            "current_row_only_residual": "current_pca_row_without_temporal_difference_blocks",
        },
        "gates": {
            "real_mean_delta_vs_ar_positive": True,
            "real_mean_minus_strongest_matched_control_positive": True,
            "paired_median_real_minus_strongest_control_positive": True,
            "every_target_mean_real_minus_strongest_control_positive": True,
            "every_fold_mean_real_minus_strongest_control_positive": True,
            "real_mean_minus_current_row_ablation_positive": True,
            "label_permutation_mean_delta_vs_ar_nonpositive": True,
            "exact_matrix_and_artifact_audits_pass": True,
            "numeric_margins_imported_from_again": False,
        },
        "stability_must_not_resume_before_all_gates_pass": True,
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    plan["plan_sha256"] = digest_json(plan)
    return plan


def write_control_plan(path: Path, plan: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(plan))


def _replay_frozen_ar(
    ar_matrix: np.ndarray,
    source_dir: Path,
    validation_mask: np.ndarray,
    binary: np.ndarray,
    video_id: np.ndarray,
    row_index: np.ndarray,
) -> np.ndarray:
    preprocessing_path = source_dir / "preprocessing.npz"
    predictions_path = source_dir / "validation-predictions.npz"
    with np.load(preprocessing_path, allow_pickle=False) as arrays:
        mean = np.asarray(arrays["scaler_mean"], dtype=np.float64)
        scale = np.asarray(arrays["scaler_scale"], dtype=np.float64)
        coefficient = np.asarray(arrays["coefficient"], dtype=np.float64)
        intercept = float(np.asarray(arrays["intercept"], dtype=np.float64)[0])
    logits = (((ar_matrix - mean) / scale) @ coefficient + intercept).astype(np.float32)
    validation = np.flatnonzero(validation_mask)
    with np.load(predictions_path, allow_pickle=False) as arrays:
        if not (
            np.array_equal(arrays["video_id"].astype(str), video_id[validation].astype(str))
            and np.array_equal(arrays["row_index"], row_index[validation])
            and np.array_equal(arrays["target"], binary[validation])
            and np.allclose(arrays["ar_score"], logits[validation], rtol=0.0, atol=1e-6)
        ):
            raise ValueError("saved real lane does not replay its frozen AR or validation rows")
    return logits


def _lane_design_and_targets(
    lane: str,
    projected: np.ndarray,
    diagnostics: np.ndarray,
    binary: np.ndarray,
    video_id: np.ndarray,
    row_index: np.ndarray,
    context_rows: tuple[int, ...],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    lane_targets = binary
    if lane == "current_row_only_residual":
        return projected, lane_targets
    if lane == "sequence_shuffled_residual":
        values = _permute_within_video(
            projected,
            video_id,
            _lane_rng(seed, lane),
        )
    elif lane == "random_pca_residual":
        values = _lane_rng(seed, lane).standard_normal(projected.shape).astype(np.float32)
    elif lane == "causal_prefix_video_mean_residual":
        values = _causal_video_means(projected, video_id, row_index).astype(np.float32)
    elif lane == "diagnostics_only_residual":
        values = diagnostics
    elif lane == "label_permutation_residual":
        values = projected
        lane_targets = _circular_permute_labels(
            binary,
            video_id,
            row_index,
            _lane_rng(seed, lane),
        )
    else:
        raise ValueError(f"unknown control lane: {lane}")
    return (
        _causal_design(
            values,
            video_id,
            row_index,
            family="frozen_ar_plus_causal_temporal_residual",
            context_rows=context_rows,
        ),
        lane_targets,
    )


def _quarantine_partial(cell_dir: Path, output_dir: Path, plan_sha256: str) -> None:
    if not cell_dir.exists() or (cell_dir / "metrics.json").is_file():
        return
    request_path = cell_dir / "request.json"
    state_path = cell_dir / "state.json"
    if not request_path.is_file() or not state_path.is_file():
        raise RuntimeError("refusing unrecognized partial control cell")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if request.get("plan_sha256") != plan_sha256 or state.get("status") not in {
        "training",
        "failed",
    }:
        raise RuntimeError("refusing changed partial control cell")
    quarantine = output_dir / "quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    base = "__".join(cell_dir.parts[-4:])
    index = 1
    destination = quarantine / f"{base}__attempt-{index}"
    while destination.exists():
        index += 1
        destination = quarantine / f"{base}__attempt-{index}"
    cell_dir.replace(destination)


def _run_control_cell(
    *,
    projected: np.ndarray,
    diagnostics: np.ndarray,
    binary: np.ndarray,
    ar_logits: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    video_id: np.ndarray,
    row_index: np.ndarray,
    plan: Mapping[str, Any],
    output_dir: Path,
    lane: str,
    target: str,
    fold: int,
    seed: int,
    source_metrics_sha256: str,
    source_artifact_sha256: Mapping[str, str],
) -> dict[str, Any]:
    request = {
        "schema": "veatic21_control_cell_request_v1",
        "plan_sha256": plan["plan_sha256"],
        "lane": lane,
        "target": target,
        "fold": fold,
        "seed": seed,
        "source_real_metrics_sha256": source_metrics_sha256,
        "source_real_artifact_sha256": dict(source_artifact_sha256),
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    request_sha256 = digest_json(request)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file():
        saved_request = json.loads((output_dir / "request.json").read_text(encoding="utf-8"))
        state = json.loads((output_dir / "state.json").read_text(encoding="utf-8"))
        if (
            digest_json(saved_request) != request_sha256
            or state.get("status") != "complete"
            or state.get("request_sha256") != request_sha256
            or state.get("metrics_sha256") != sha256_file(metrics_path)
        ):
            raise RuntimeError("refusing changed control-cell reuse")
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("refusing unrecognized partial control cell")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "request.json", request)
    atomic_write_json(
        output_dir / "state.json",
        {
            "schema": "veatic21_control_cell_state_v1",
            "status": "training",
            "request_sha256": request_sha256,
        },
    )
    recipe = plan["selected_recipe"]
    context_rows = tuple(int(value) for value in recipe["context_rows"])
    design, lane_targets = _lane_design_and_targets(
        lane,
        projected,
        diagnostics,
        binary,
        video_id,
        row_index,
        context_rows,
        seed,
    )
    scaler = StandardScaler().fit(design[train_mask])
    design = scaler.transform(design).astype(np.float32)
    atomic_save_npz(
        output_dir / "preprocessing.npz",
        {
            "design_scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
            "design_scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
        },
    )
    checkpoint = output_dir / "best-checkpoint.npz"
    dummy_indices = np.zeros((len(design), len(context_rows)), dtype=np.int32)
    dummy_available = np.zeros((len(design), len(context_rows)), dtype=np.float32)
    training_screen = {
        "architecture": {"projection_width": 512},
        "matched_recipe": recipe,
    }
    try:
        scores, curve, selector = _train_lane(
            design,
            design,
            np.zeros(design.shape[1], dtype=np.float32),
            np.ones(design.shape[1], dtype=np.float32),
            dummy_indices,
            dummy_available,
            lane_targets,
            ar_logits,
            train_mask,
            validation_mask,
            training_screen,
            SupervisedProjectionCellConfig(
                lane="fixed_pca512",
                target_name=target,
                fold=fold,
                seed=seed,
            ),
            checkpoint,
            output_dir / "state.json",
            request_sha256,
        )
    except Exception as exc:
        atomic_write_json(
            output_dir / "state.json",
            {
                "schema": "veatic21_control_cell_state_v1",
                "status": "failed",
                "request_sha256": request_sha256,
                "failure": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    validation = np.flatnonzero(validation_mask)
    delta = average_precision_skill(
        binary[validation], scores[validation]
    ) - average_precision_skill(binary[validation], ar_logits[validation])
    atomic_save_npz(
        output_dir / "validation-predictions.npz",
        {
            "video_id": video_id[validation].astype("U3"),
            "row_index": row_index[validation].astype(np.int32),
            "target": binary[validation].astype(np.int8),
            "ar_score": ar_logits[validation],
            "control_score": scores[validation],
        },
    )
    atomic_write_json(output_dir / "training-curve.json", {"records": curve})
    metrics: dict[str, Any] = {
        "schema": "veatic21_control_cell_metrics_v1",
        "lane": lane,
        "target": target,
        "fold": fold,
        "seed": seed,
        "inner_average_precision_skill_delta_vs_frozen_ar": delta,
        "fresh_ar_pr_auc": pooled_pr_auc(binary[validation], ar_logits[validation]),
        "control_pr_auc": pooled_pr_auc(binary[validation], scores[validation]),
        "best_epoch": selector.best_epoch,
        "epochs_completed": len(curve),
        "train_row_sha256": row_identity_digest(video_id[train_mask], row_index[train_mask]),
        "validation_row_sha256": row_identity_digest(
            video_id[validation_mask], row_index[validation_mask]
        ),
        "request_sha256": request_sha256,
        "checkpoint_sha256": sha256_file(checkpoint),
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    atomic_write_json(metrics_path, metrics)
    atomic_write_json(
        output_dir / "state.json",
        {
            "schema": "veatic21_control_cell_state_v1",
            "status": "complete",
            "request_sha256": request_sha256,
            "metrics_sha256": sha256_file(metrics_path),
        },
    )
    return metrics


def _lane_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = np.asarray(
        [float(row["inner_average_precision_skill_delta_vs_frozen_ar"]) for row in records]
    )
    return {
        "cell_count": len(values),
        "mean_inner_average_precision_skill_delta_vs_frozen_ar": float(np.mean(values)),
        "median_inner_average_precision_skill_delta_vs_frozen_ar": float(np.median(values)),
        "positive_cells": int(np.sum(values > 0.0)),
        "nonpositive_cells": int(np.sum(values <= 0.0)),
    }


def run_control_program(
    substrate: CanonicalSubstrate,
    preregistration: Mapping[str, Any],
    calibration: Mapping[str, Any],
    pca_manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    pca_root: Path,
    baseline_root: Path,
    output_dir: Path,
    *,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Backfill every lifecycle control against the verified comparison panel."""

    _require_self_digest(plan, "plan_sha256")
    _require_self_digest(baseline_summary, "summary_sha256")
    if plan.get("schema") != _SCHEMA:
        raise ValueError("control runner requires the current plan schema")
    if plan.get("artifacts", {}).get("control_code_sha256") != sha256_file(Path(__file__)):
        raise ValueError("control plan does not bind the current runner code")
    if plan.get("artifacts", {}).get("baseline_summary_sha256") != baseline_summary.get(
        "summary_sha256"
    ):
        raise ValueError("control plan does not bind the supplied real summary")
    request = {
        "schema": "veatic21_control_run_request_v1",
        "plan_sha256": plan["plan_sha256"],
        "worker_count": 1,
        "backend": "mlx",
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    request_sha256 = digest_json(request)
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "request.json"
    if request_path.is_file():
        if digest_json(json.loads(request_path.read_text(encoding="utf-8"))) != request_sha256:
            raise RuntimeError("refusing control resume because request changed")
    else:
        atomic_write_json(request_path, request)

    matrix = plan["matrix"]
    expected_keys = {
        (str(target), int(fold), int(seed))
        for target in matrix["targets"]
        for fold in matrix["folds"]
        for seed in matrix["comparison_seeds"]
    }
    real = {
        (str(row["target"]), int(row["fold"]), int(row["seed"])): row
        for row in baseline_summary["records"]
        if row.get("lane") == "fixed_pca512"
    }
    if set(real) != expected_keys:
        raise ValueError("real comparison records do not cover the exact control matrix")

    all_features = substrate.load_features(
        substrate.video_ids,
        ("tribe_cortical", "diagnostics_only"),
    )
    development_mask = benchmark_partition_mask(all_features, preregistration["split"], "train")
    features = all_features.subset(development_mask)
    labels = substrate.load_labels(
        substrate.video_ids,
        row_indices=_owned_rows(all_features.video_id, all_features.row_index, development_mask),
        stage="matched_controls_benchmark_train_labels_only",
    )
    diagnostics = features.representations["diagnostics_only"]
    projected_by_fold = {
        int(fold): load_event_pca_projection(
            features,
            preregistration,
            pca_manifest,
            pca_root,
            fold=int(fold),
            width=512,
        )
        for fold in matrix["folds"]
    }
    records: list[dict[str, Any]] = []
    for target_name in matrix["targets"]:
        target = _target(calibration, str(target_name))
        future = future_target_values(labels, target)
        support = target_support_mask(features, target)
        ar_values, ar_available = causal_ar_features(labels, target)
        ar_matrix = np.concatenate([ar_values, ar_available.astype(np.float64)], axis=1)
        for fold in matrix["folds"]:
            validation_videos = preregistration["split"]["inner_grouped_video_folds"][int(fold)]
            validation_mask = np.isin(features.video_id.astype(str), validation_videos) & support
            train_mask = ~np.isin(features.video_id.astype(str), validation_videos) & support
            threshold = fit_event_threshold(future, train_mask, target)
            binary = event_labels(future, threshold)
            projected = projected_by_fold[int(fold)]
            for seed in matrix["comparison_seeds"]:
                key = (str(target_name), int(fold), int(seed))
                source_record = real[key]
                source_dir = baseline_root / str(source_record["cell_directory"])
                metrics_path = source_dir / "metrics.json"
                if sha256_file(metrics_path) != source_record["cell_metrics_sha256"]:
                    raise ValueError("real comparison cell metrics changed")
                source_artifacts = {
                    name: sha256_file(source_dir / name)
                    for name in (
                        "best-checkpoint.npz",
                        "preprocessing.npz",
                        "training-curve.json",
                        "validation-predictions.npz",
                    )
                }
                ar_logits = _replay_frozen_ar(
                    ar_matrix,
                    source_dir,
                    validation_mask,
                    binary,
                    features.video_id,
                    features.row_index,
                )
                for lane in matrix["trained_lanes"]:
                    cell_dir = (
                        output_dir
                        / "targets"
                        / str(target_name)
                        / f"fold-{int(fold)}"
                        / f"seed-{int(seed)}"
                        / str(lane)
                    )
                    _quarantine_partial(cell_dir, output_dir, plan["plan_sha256"])
                    metrics = _run_control_cell(
                        projected=projected,
                        diagnostics=diagnostics,
                        binary=binary,
                        ar_logits=ar_logits,
                        train_mask=train_mask,
                        validation_mask=validation_mask,
                        video_id=features.video_id,
                        row_index=features.row_index,
                        plan=plan,
                        output_dir=cell_dir,
                        lane=str(lane),
                        target=str(target_name),
                        fold=int(fold),
                        seed=int(seed),
                        source_metrics_sha256=str(source_record["cell_metrics_sha256"]),
                        source_artifact_sha256=source_artifacts,
                    )
                    record = {
                        "lane": str(lane),
                        "target": str(target_name),
                        "fold": int(fold),
                        "seed": int(seed),
                        "inner_average_precision_skill_delta_vs_frozen_ar": metrics[
                            "inner_average_precision_skill_delta_vs_frozen_ar"
                        ],
                        "best_epoch": metrics["best_epoch"],
                        "cell_metrics_sha256": sha256_file(cell_dir / "metrics.json"),
                        "cell_directory": str(cell_dir.relative_to(output_dir)),
                    }
                    records.append(record)
                    if progress is not None:
                        progress(
                            {
                                "schema": "veatic21_control_progress_v1",
                                "completed_new_cells": len(records),
                                "expected_new_cells": int(matrix["expected_new_cells"]),
                                "last_cell": record,
                                "worker_count": 1,
                                "benchmark_test_labels_accessed": False,
                            }
                        )

    if len(records) != int(matrix["expected_new_cells"]):
        raise RuntimeError("control program did not complete its exact registered matrix")
    records_by_lane = {
        lane: [row for row in records if row["lane"] == lane] for lane in matrix["trained_lanes"]
    }
    real_delta = {
        key: float(row["inner_average_precision_skill_delta_vs_frozen_ar"])
        for key, row in real.items()
    }
    control_delta = {
        str(lane): {
            (str(row["target"]), int(row["fold"]), int(row["seed"])): float(
                row["inner_average_precision_skill_delta_vs_frozen_ar"]
            )
            for row in records_by_lane[str(lane)]
        }
        for lane in matrix["trained_lanes"]
    }
    lane_summaries = {
        str(lane): _lane_summary(records_by_lane[str(lane)]) for lane in matrix["trained_lanes"]
    }
    strongest = max(
        matrix["matched_control_lanes"],
        key=lambda lane: lane_summaries[str(lane)][
            "mean_inner_average_precision_skill_delta_vs_frozen_ar"
        ],
    )
    paired = np.asarray(
        [real_delta[key] - control_delta[str(strongest)][key] for key in sorted(real)]
    )
    target_pass = {}
    for target_name in matrix["targets"]:
        key_subset = [key for key in sorted(real) if key[0] == target_name]
        best_mean = max(
            float(np.mean([control_delta[str(lane)][key] for key in key_subset]))
            for lane in matrix["matched_control_lanes"]
        )
        target_pass[str(target_name)] = float(
            np.mean([real_delta[key] for key in key_subset])
        ) - best_mean > 0.0
    fold_pass = {}
    for fold in matrix["folds"]:
        key_subset = [key for key in sorted(real) if key[1] == int(fold)]
        best_mean = max(
            float(np.mean([control_delta[str(lane)][key] for key in key_subset]))
            for lane in matrix["matched_control_lanes"]
        )
        fold_pass[str(fold)] = (
            float(np.mean([real_delta[key] for key in key_subset])) - best_mean > 0.0
        )
    real_mean = float(np.mean(list(real_delta.values())))
    strongest_mean = float(np.mean(list(control_delta[str(strongest)].values())))
    current_row_mean = float(
        np.mean(list(control_delta["current_row_only_residual"].values()))
    )
    label_permutation_mean = float(
        np.mean(list(control_delta["label_permutation_residual"].values()))
    )
    gate_results = {
        "real_mean_delta_vs_ar_positive": real_mean > 0.0,
        "real_mean_minus_strongest_matched_control_positive": real_mean - strongest_mean > 0.0,
        "paired_median_real_minus_strongest_control_positive": float(np.median(paired)) > 0.0,
        "every_target_mean_real_minus_strongest_control_positive": all(target_pass.values()),
        "every_fold_mean_real_minus_strongest_control_positive": all(fold_pass.values()),
        "real_mean_minus_current_row_ablation_positive": real_mean - current_row_mean > 0.0,
        "label_permutation_mean_delta_vs_ar_nonpositive": label_permutation_mean <= 0.0,
        "exact_matrix_and_artifact_audits_pass": True,
    }
    summary: dict[str, Any] = {
        "schema": "veatic21_control_summary_v1",
        "request_sha256": request_sha256,
        "plan_sha256": plan["plan_sha256"],
        "completed_new_cells": len(records),
        "expected_new_cells": int(matrix["expected_new_cells"]),
        "reused_real_cells": len(real),
        "real_mean_inner_average_precision_skill_delta_vs_frozen_ar": real_mean,
        "lane_summaries": lane_summaries,
        "strongest_matched_control": str(strongest),
        "real_minus_strongest_control": real_mean - strongest_mean,
        "paired_median_real_minus_strongest_control": float(np.median(paired)),
        "target_gate_results": target_pass,
        "fold_gate_results": fold_pass,
        "gate_results": gate_results,
        "all_gates_pass": all(gate_results.values()),
        "records": records,
        "worker_count": 1,
        "backend": "mlx",
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    summary["summary_sha256"] = digest_json(summary)
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


__all__ = ["build_control_plan", "run_control_program", "write_control_plan"]
