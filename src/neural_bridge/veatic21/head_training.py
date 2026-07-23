"""Matched causal versus gated head-family discovery for VEATIC 2.1."""

from __future__ import annotations

import json
from collections.abc import Mapping
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
from .stage1 import (
    Stage1CellConfig,
    _causal_design,
    _fit_fresh_ar,
    _owned_rows,
    _target,
    _train_mlx_residual,
)

_SCHEMA = "veatic21_head_family_screen_v1"
_BASELINE = "frozen_ar_plus_causal_temporal_residual"
_CANDIDATE = "frozen_ar_plus_gated_multiscale_temporal_residual"


def _require_self_digest(record: Mapping[str, Any], field: str) -> None:
    expected = record.get(field)
    if not isinstance(expected, str):
        raise ValueError(f"artifact is missing {field}")
    payload = dict(record)
    payload.pop(field)
    if digest_json(payload) != expected:
        raise ValueError(f"artifact failed its {field} integrity check")


def build_head_family_screen(
    representation_selection: Mapping[str, Any],
    representation_summary: Mapping[str, Any],
    representation_screen: Mapping[str, Any],
    plan: Mapping[str, Any],
    pca_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Register only the missing gated cells against verified causal-head evidence."""

    _require_self_digest(representation_selection, "selection_sha256")
    _require_self_digest(representation_summary, "summary_sha256")
    _require_self_digest(representation_screen, "screen_sha256")
    _require_self_digest(plan, "plan_sha256")
    _require_self_digest(pca_manifest, "manifest_sha256")
    if representation_selection.get("selected_representation") != "fixed_pca512":
        raise ValueError("head discovery requires selected fixed PCA-512")
    if representation_selection.get("summary_sha256") != representation_summary.get(
        "summary_sha256"
    ):
        raise ValueError("representation selection does not bind the supplied summary")
    if representation_summary.get("screen_sha256") != representation_screen.get("screen_sha256"):
        raise ValueError("representation summary does not bind the supplied screen")
    if pca_manifest.get("manifest_sha256") != plan.get("artifacts", {}).get("pca_manifest_sha256"):
        raise ValueError("head screen does not bind the current PCA manifest")

    recipe = dict(representation_screen["matched_recipe"])
    targets = [str(value) for value in representation_screen["matrix"]["targets"]]
    folds = [int(value) for value in representation_screen["matrix"]["folds"]]
    seeds = [int(value) for value in representation_screen["matrix"]["comparison_seeds"]]
    screen: dict[str, Any] = {
        "schema": _SCHEMA,
        "purpose": "matched_gated_multiscale_vs_causal_temporal_head",
        "artifacts": {
            "baseline_representation_summary_sha256": representation_summary["summary_sha256"],
            "pca_manifest_sha256": pca_manifest["manifest_sha256"],
            "representation_selection_sha256": representation_selection["selection_sha256"],
            "representation_screen_sha256": representation_screen["screen_sha256"],
            "stage1_code_sha256": sha256_file(Path(__file__).with_name("stage1.py")),
            "stage1_plan_sha256": plan["plan_sha256"],
            "head_training_code_sha256": sha256_file(Path(__file__)),
        },
        "selected_representation": "fixed_pca512",
        "baseline_head": _BASELINE,
        "candidate_head": _CANDIDATE,
        "matched_recipe": recipe,
        "matrix": {
            "targets": targets,
            "folds": folds,
            "comparison_seeds": seeds,
            "expected_candidate_cells": len(targets) * len(folds) * len(seeds),
            "worker_count": 1,
            "sealed_tail_labels": True,
        },
        "selection_after_completion": {
            "primary_key": "paired_mean_inner_average_precision_skill_delta_gated_minus_causal",
            "gated_selected_only_if_positive": True,
            "tie_or_nonpositive_result_keeps_causal": True,
        },
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    screen["screen_sha256"] = digest_json(screen)
    return screen


def write_head_family_screen(path: Path, screen: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(screen))


def run_gated_head_cell(
    substrate: CanonicalSubstrate,
    preregistration: Mapping[str, Any],
    calibration: Mapping[str, Any],
    pca_manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    screen: Mapping[str, Any],
    pca_root: Path,
    output_dir: Path,
    *,
    target_name: str,
    fold: int,
    seed: int,
) -> dict[str, Any]:
    """Train one gated multiscale cell on development ownership only."""

    _require_self_digest(screen, "screen_sha256")
    if screen.get("schema") != _SCHEMA:
        raise ValueError("gated head cell requires the current screen")
    if (
        target_name not in screen["matrix"]["targets"]
        or fold not in screen["matrix"]["folds"]
        or seed not in screen["matrix"]["comparison_seeds"]
    ):
        raise ValueError("gated head cell is not registered")
    recipe = screen["matched_recipe"]
    config = Stage1CellConfig(
        target_name=target_name,
        fold=fold,
        seed=seed,
        pca_width=512,
        head_family=_CANDIDATE,
        hidden_width=int(recipe["hidden_width"]),
        learning_rate=float(recipe["learning_rate"]),
        weight_decay=float(recipe["weight_decay"]),
        residual_logit_cap=float(recipe["residual_logit_cap"]),
        batch_rows=int(recipe["batch_rows"]),
        context_rows=tuple(int(value) for value in recipe["context_rows"]),
        minimum_epochs=int(recipe["minimum_epochs"]),
        plateau_patience=int(recipe["plateau_patience"]),
        nonconvergence_patience=int(recipe["nonconvergence_patience"]),
    )
    request = {
        "schema": "veatic21_gated_head_cell_request_v1",
        "screen_sha256": screen["screen_sha256"],
        "config": {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in config.__dict__.items()
        },
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    request_sha256 = digest_json(request)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file():
        saved = json.loads((output_dir / "request.json").read_text(encoding="utf-8"))
        state = json.loads((output_dir / "state.json").read_text(encoding="utf-8"))
        if digest_json(saved) != request_sha256 or state.get("status") != "complete":
            raise RuntimeError("refusing changed or incomplete gated cell reuse")
        if state.get("metrics_sha256") != sha256_file(metrics_path):
            raise RuntimeError("refusing changed gated cell metrics")
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("refusing to overwrite a partial gated cell")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "request.json", request)
    atomic_write_json(
        output_dir / "state.json",
        {
            "schema": "veatic21_gated_head_cell_state_v1",
            "status": "training",
            "request_sha256": request_sha256,
        },
    )

    all_features = substrate.load_features(substrate.video_ids, ("tribe_cortical",))
    development_mask = benchmark_partition_mask(all_features, preregistration["split"], "train")
    features = all_features.subset(development_mask)
    labels = substrate.load_labels(
        substrate.video_ids,
        row_indices=_owned_rows(all_features.video_id, all_features.row_index, development_mask),
        stage="gated_head_benchmark_train_labels_only",
    )
    target = _target(calibration, target_name)
    future = future_target_values(labels, target)
    support = target_support_mask(features, target)
    validation_videos = preregistration["split"]["inner_grouped_video_folds"][fold]
    validation_mask = np.isin(features.video_id.astype(str), validation_videos) & support
    train_mask = ~np.isin(features.video_id.astype(str), validation_videos) & support
    threshold = fit_event_threshold(future, train_mask, target)
    binary = event_labels(future, threshold)
    ar_values, ar_available = causal_ar_features(labels, target)
    ar_matrix = np.concatenate([ar_values, ar_available.astype(np.float64)], axis=1)
    ar_logits, ar_c, ar_artifact = _fit_fresh_ar(
        ar_matrix, binary, future, target, train_mask, features.video_id, seed=seed
    )
    projected = load_event_pca_projection(
        features, preregistration, pca_manifest, pca_root, fold=fold, width=512
    )
    design = _causal_design(
        projected,
        features.video_id,
        features.row_index,
        family=_CANDIDATE,
        context_rows=config.context_rows,
    )
    scaler = StandardScaler().fit(design[train_mask])
    design = scaler.transform(design).astype(np.float32)
    atomic_save_npz(
        output_dir / "preprocessing.npz",
        {
            **ar_artifact,
            "design_scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
            "design_scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
        },
    )
    checkpoint = output_dir / "best-checkpoint.npz"
    try:
        scores, curve, selector = _train_mlx_residual(
            design,
            binary,
            ar_logits,
            train_mask,
            validation_mask,
            config,
            checkpoint,
            output_dir / "state.json",
            request_sha256,
        )
    except Exception as exc:
        atomic_write_json(
            output_dir / "state.json",
            {
                "schema": "veatic21_gated_head_cell_state_v1",
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
    use_residual = delta > 0.0
    selected = scores if use_residual else ar_logits
    atomic_save_npz(
        output_dir / "validation-predictions.npz",
        {
            "video_id": features.video_id[validation].astype("U3"),
            "row_index": features.row_index[validation].astype(np.int32),
            "target": binary[validation].astype(np.int8),
            "ar_score": ar_logits[validation],
            "model_score": scores[validation],
            "selected_score": selected[validation],
        },
    )
    atomic_write_json(output_dir / "training-curve.json", {"records": curve})
    metrics: dict[str, Any] = {
        "schema": "veatic21_gated_head_cell_metrics_v1",
        "head_family": _CANDIDATE,
        "target": target_name,
        "fold": fold,
        "seed": seed,
        "fresh_ar_c": ar_c,
        "fresh_ar_pr_auc": pooled_pr_auc(binary[validation], ar_logits[validation]),
        "learned_head_pr_auc": pooled_pr_auc(binary[validation], scores[validation]),
        "inner_average_precision_skill_delta_vs_frozen_ar": delta,
        "whole_fold_seed_uses_residual": use_residual,
        "selected_pr_auc": pooled_pr_auc(binary[validation], selected[validation]),
        "best_epoch": selector.best_epoch,
        "epochs_completed": len(curve),
        "train_row_sha256": row_identity_digest(
            features.video_id[train_mask], features.row_index[train_mask]
        ),
        "validation_row_sha256": row_identity_digest(
            features.video_id[validation_mask], features.row_index[validation_mask]
        ),
        "checkpoint_sha256": sha256_file(checkpoint),
        "request_sha256": request_sha256,
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    atomic_write_json(metrics_path, metrics)
    atomic_write_json(
        output_dir / "state.json",
        {
            "schema": "veatic21_gated_head_cell_state_v1",
            "status": "complete",
            "request_sha256": request_sha256,
            "metrics_sha256": sha256_file(metrics_path),
        },
    )
    return metrics


def run_head_family_screen(
    substrate: CanonicalSubstrate,
    preregistration: Mapping[str, Any],
    calibration: Mapping[str, Any],
    pca_manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    screen: Mapping[str, Any],
    pca_root: Path,
    output_dir: Path,
    *,
    progress=None,
) -> dict[str, Any]:
    """Run every missing gated-head cell with one sequential worker."""

    _require_self_digest(screen, "screen_sha256")
    request = {
        "schema": "veatic21_head_family_run_request_v1",
        "screen_sha256": screen["screen_sha256"],
        "worker_count": 1,
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    request_sha256 = digest_json(request)
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "request.json"
    if request_path.is_file():
        if digest_json(json.loads(request_path.read_text(encoding="utf-8"))) != request_sha256:
            raise RuntimeError("refusing head-family resume because request changed")
    else:
        atomic_write_json(request_path, request)
    records = []
    for target in screen["matrix"]["targets"]:
        for fold in screen["matrix"]["folds"]:
            for seed in screen["matrix"]["comparison_seeds"]:
                cell_dir = (
                    output_dir / "targets" / str(target) / f"fold-{int(fold)}" / f"seed-{int(seed)}"
                )
                metrics = run_gated_head_cell(
                    substrate,
                    preregistration,
                    calibration,
                    pca_manifest,
                    plan,
                    screen,
                    pca_root,
                    cell_dir,
                    target_name=str(target),
                    fold=int(fold),
                    seed=int(seed),
                )
                record = {
                    "target": str(target),
                    "fold": int(fold),
                    "seed": int(seed),
                    "inner_average_precision_skill_delta_vs_frozen_ar": metrics[
                        "inner_average_precision_skill_delta_vs_frozen_ar"
                    ],
                    "whole_fold_seed_uses_residual": metrics["whole_fold_seed_uses_residual"],
                    "best_epoch": metrics["best_epoch"],
                    "cell_metrics_sha256": sha256_file(cell_dir / "metrics.json"),
                    "cell_directory": str(cell_dir.relative_to(output_dir)),
                }
                records.append(record)
                progress_record = {
                    "schema": "veatic21_head_family_progress_v1",
                    "request_sha256": request_sha256,
                    "completed_cells": len(records),
                    "expected_cells": int(screen["matrix"]["expected_candidate_cells"]),
                    "last_cell": record,
                    "benchmark_test_labels_accessed": False,
                }
                atomic_write_json(output_dir / "progress.json", progress_record)
                if progress is not None:
                    progress(progress_record)
    summary: dict[str, Any] = {
        "schema": "veatic21_head_family_summary_v1",
        "request_sha256": request_sha256,
        "screen_sha256": screen["screen_sha256"],
        "completed_cells": len(records),
        "expected_cells": int(screen["matrix"]["expected_candidate_cells"]),
        "records": records,
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    summary["summary_sha256"] = digest_json(summary)
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def select_head_family(
    gated_summary: Mapping[str, Any],
    screen: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the paired gated-minus-causal head-family gate."""

    _require_self_digest(gated_summary, "summary_sha256")
    _require_self_digest(screen, "screen_sha256")
    _require_self_digest(baseline_summary, "summary_sha256")
    if gated_summary.get("screen_sha256") != screen.get("screen_sha256"):
        raise ValueError("gated summary does not belong to the head screen")
    if gated_summary.get("completed_cells") != gated_summary.get("expected_cells"):
        raise ValueError("head selection requires every gated cell")
    baseline = {
        (str(row["target"]), int(row["fold"]), int(row["seed"])): float(
            row["inner_average_precision_skill_delta_vs_frozen_ar"]
        )
        for row in baseline_summary["records"]
        if row.get("lane") == "fixed_pca512"
    }
    gated = {
        (str(row["target"]), int(row["fold"]), int(row["seed"])): float(
            row["inner_average_precision_skill_delta_vs_frozen_ar"]
        )
        for row in gated_summary["records"]
    }
    if set(baseline) != set(gated):
        raise ValueError("head selection requires exact causal/gated pairs")
    paired = np.asarray([gated[key] - baseline[key] for key in gated])
    mean_delta = float(np.mean(paired))
    selected = _CANDIDATE if mean_delta > 0.0 else _BASELINE
    selection: dict[str, Any] = {
        "schema": "veatic21_head_family_selection_v1",
        "screen_sha256": screen["screen_sha256"],
        "gated_summary_sha256": gated_summary["summary_sha256"],
        "baseline_summary_sha256": baseline_summary["summary_sha256"],
        "pair_count": len(paired),
        "paired_mean_gated_minus_causal": mean_delta,
        "paired_median_gated_minus_causal": float(np.median(paired)),
        "gated_wins": int(np.sum(paired > 0.0)),
        "causal_wins": int(np.sum(paired < 0.0)),
        "ties": int(np.sum(paired == 0.0)),
        "causal_mean_delta_vs_ar": float(np.mean(list(baseline.values()))),
        "gated_mean_delta_vs_ar": float(np.mean(list(gated.values()))),
        "selected_head_family": selected,
        "benchmark_test_labels_accessed": False,
        "promotable": False,
    }
    selection["selection_sha256"] = digest_json(selection)
    return selection


def write_head_family_selection(path: Path, selection: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(selection))


__all__ = [
    "build_head_family_screen",
    "run_gated_head_cell",
    "run_head_family_screen",
    "select_head_family",
    "write_head_family_screen",
    "write_head_family_selection",
]
