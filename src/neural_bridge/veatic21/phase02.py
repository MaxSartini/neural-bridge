"""VEATIC 2.1 Phase 02 fresh target-specific autoregressive floor."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from sklearn.metrics import average_precision_score

from neural_bridge.veatic21.ar import (
    ARModel,
    SplitCell,
    blocked_split_cell,
    build_ar_features,
    build_trailing_slope_feature,
    common_history_mask,
    derive_lag_depths,
    derive_seed,
    fit_logistic_mlx,
    grouped_split_cell,
    predict_logistic_mlx,
    select_decision_threshold,
    spike_metrics,
)
from neural_bridge.veatic21.contracts import (
    AR_FINAL_OPTIMIZER_MAX_ITERATIONS,
    AR_OPTIMIZER_LEARNING_RATE,
    AR_OPTIMIZER_MAX_ITERATIONS,
    AR_OPTIMIZER_TOLERANCE,
    AR_REGULARIZATION_CANDIDATES,
    CLUSTER_BOOTSTRAP_RESAMPLES,
    CURRENT_STATE,
    EXPECTED_ROW_COUNT,
    EXPECTED_ROW_HZ,
    EXPECTED_TIME_STEP_SECONDS,
    EXPECTED_VIDEO_IDS,
    GROUPED_OUTER_REPEATS,
    INNER_VALIDATION_FRACTION,
    LIFECYCLE_ROOT,
    MASTER_SPECIFICATION,
    OUTER_TEST_FRACTION,
    PHASE01_ROOT,
    PHASE02_ROOT,
    REBUILD_PROTOCOL,
    REPOSITORY_ROOT,
    reject_forbidden_runtime_path,
    validate_runtime_manifest_paths,
)
from neural_bridge.veatic21.evidence import (
    canonical_json_bytes,
    paired_video_bootstrap_raw_pr_auc_delta,
    per_video_pr_auc,
    sha256_file,
    source_tree_digest,
)

PHASE01_RESULT_SHA256 = "31e6933c4d7a2b6ed077d9ae57b4e667c7957286aa0c4c97ff07c56801fb5539"
PHASE01_SUBSTRATE_FILE_SHA256 = "50dfa45bb3a063e88e9334c8cc9e57a9b2353a809d00298ce4d137cc3d8159af"
PHASE01_TARGET_SOURCE_SHA256 = "ad8b167dff44ae6a0c1c78ef3e501cc622e6320be9a912d879c3d9fc99863a4f"
PHASE01_SUBSTRATE_ARRAYS_SHA256 = "ce4acca4b2b72320bf224ac057342be34f27c4ea713f2a7f5eed97d3f0125088"

PHASE02_CHECKS = (
    "sealed_phase01_gate_and_input_identity",
    "selected_continuous_target_and_mask_only",
    "common_causal_history_row_mask",
    "veatic_derived_lag_family",
    "separate_grouped_and_blocked_protocols",
    "grouped_video_outer_70_30",
    "blocked_forward_time_outer_70_30",
    "nested_inner_partition_ownership",
    "fold_owned_q90_thresholds",
    "fold_owned_normalization",
    "inner_selected_lag_and_regularization",
    "training_owned_decision_thresholds",
    "mlx_gpu_zero_single_worker",
    "exact_ar_predictions_frozen",
    "prediction_and_model_checksums",
    "complete_spike_metric_stack",
    "defined_only_per_video_pr_auc",
    "paired_video_cluster_bootstrap",
    "ar_dominance_and_overlap_decomposition",
    "prospective_washout_inactive",
    "cortical_values_not_loaded",
    "pca_and_later_model_work_not_started",
    "again_runtime_firewall",
    "phase03_only_authorization",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _atomic_write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _array_bundle_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(f"{name}\0{array.dtype.str}\0{array.shape}\n".encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _split_digest(cell: SplitCell) -> str:
    arrays = {
        "outer_train": cell.outer_train,
        "inner_train": cell.inner_train,
        "inner_validation": cell.inner_validation,
        "outer_test": cell.outer_test,
    }
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "protocol": cell.protocol,
                "fold": cell.fold,
                "seed": cell.seed,
                "arrays_sha256": _array_bundle_digest(arrays),
            }
        )
    ).hexdigest()


def _cell_id(cell: SplitCell) -> str:
    return f"{cell.protocol}-fold-{cell.fold:02d}-seed-{cell.seed:010d}"


def _model_arrays(model: ARModel, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_mean": model.mean,
        f"{prefix}_scale": model.scale,
        f"{prefix}_weights": model.weights,
        f"{prefix}_regularization": np.asarray(model.regularization, dtype=np.float64),
        f"{prefix}_iterations": np.asarray(model.iterations, dtype=np.int32),
        f"{prefix}_converged": np.asarray(model.converged, dtype=np.bool_),
        f"{prefix}_final_gradient_norm": np.asarray(model.final_gradient_norm, dtype=np.float64),
    }


def _threshold(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("threshold ownership values must be nonempty and finite")
    return float(np.quantile(values, 0.90))


def _search_ar(
    *,
    arousal: np.ndarray,
    target: np.ndarray,
    cell: SplitCell,
    lag_depths: tuple[int, ...],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    inner_q90 = _threshold(target[cell.inner_train])
    train_labels = (target[cell.inner_train] >= inner_q90).astype(np.int8)
    validation_labels = (target[cell.inner_validation] >= inner_q90).astype(np.int8)
    rows: list[dict[str, object]] = []
    for depth in lag_depths:
        train_features = build_ar_features(arousal, cell.inner_train, depth=depth)
        validation_features = build_ar_features(arousal, cell.inner_validation, depth=depth)
        for regularization in AR_REGULARIZATION_CANDIDATES:
            model = fit_logistic_mlx(train_features, train_labels, regularization=regularization)
            scores = predict_logistic_mlx(model, validation_features)
            rows.append(
                {
                    "cell_id": _cell_id(cell),
                    "lane": "ar",
                    "lag_depth_rows": depth,
                    "feature_width": depth + 1,
                    "regularization": regularization,
                    "inner_q90_threshold": inner_q90,
                    "inner_train_rows": len(cell.inner_train),
                    "inner_validation_rows": len(cell.inner_validation),
                    "inner_validation_positives": int(np.sum(validation_labels)),
                    "inner_validation_pr_auc": float(
                        average_precision_score(validation_labels, scores)
                    ),
                    "optimizer_iterations": model.iterations,
                    "optimizer_converged": model.converged,
                    "final_gradient_norm": model.final_gradient_norm,
                    "selection_eligible": model.converged,
                    "device": model.device,
                }
            )
    eligible = [row for row in rows if bool(row["selection_eligible"])]
    if not eligible:
        raise ValueError(f"no converged AR candidate for {_cell_id(cell)}")
    winner = min(
        eligible,
        key=lambda row: (
            -float(row["inner_validation_pr_auc"]),
            int(row["lag_depth_rows"]),
            -float(row["regularization"]),
        ),
    )
    return winner, rows


def _search_single_feature(
    *,
    lane: str,
    feature_builder: Callable[[np.ndarray], np.ndarray],
    target: np.ndarray,
    cell: SplitCell,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    inner_q90 = _threshold(target[cell.inner_train])
    train_labels = (target[cell.inner_train] >= inner_q90).astype(np.int8)
    validation_labels = (target[cell.inner_validation] >= inner_q90).astype(np.int8)
    train_features = feature_builder(cell.inner_train)
    validation_features = feature_builder(cell.inner_validation)
    rows: list[dict[str, object]] = []
    for regularization in AR_REGULARIZATION_CANDIDATES:
        model = fit_logistic_mlx(train_features, train_labels, regularization=regularization)
        scores = predict_logistic_mlx(model, validation_features)
        rows.append(
            {
                "cell_id": _cell_id(cell),
                "lane": lane,
                "lag_depth_rows": 0 if lane == "current_arousal" else "",
                "feature_width": 1,
                "regularization": regularization,
                "inner_q90_threshold": inner_q90,
                "inner_train_rows": len(cell.inner_train),
                "inner_validation_rows": len(cell.inner_validation),
                "inner_validation_positives": int(np.sum(validation_labels)),
                "inner_validation_pr_auc": float(
                    average_precision_score(validation_labels, scores)
                ),
                "optimizer_iterations": model.iterations,
                "optimizer_converged": model.converged,
                "final_gradient_norm": model.final_gradient_norm,
                "selection_eligible": model.converged,
                "device": model.device,
            }
        )
    eligible = [row for row in rows if bool(row["selection_eligible"])]
    if not eligible:
        raise ValueError(f"no converged {lane} candidate for {_cell_id(cell)}")
    winner = min(
        eligible,
        key=lambda row: (
            -float(row["inner_validation_pr_auc"]),
            -float(row["regularization"]),
        ),
    )
    return winner, rows


def _artifact_inventory(root: Path) -> list[dict[str, object]]:
    excluded = {"artifact-manifest.json", "checksums.sha256"}
    output = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        output.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return output


def phase03_authorized(checks: Mapping[str, bool]) -> bool:
    return set(checks) == set(PHASE02_CHECKS) and all(checks.values())


def run_phase02(output_root: Path = PHASE02_ROOT) -> dict[str, Any]:
    """Execute and seal the authorized fresh Phase 02 AR floor."""

    output_root = reject_forbidden_runtime_path(output_root)
    if output_root != PHASE02_ROOT:
        raise ValueError(f"Phase 02 output root must be exactly {PHASE02_ROOT}")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite Phase 02 root: {output_root}")
    staging = output_root.parent / f".{output_root.name}.staging"
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite Phase 02 staging root: {staging}")
    staging.mkdir(parents=True)

    package_root = REPOSITORY_ROOT / "src/neural_bridge/veatic21"
    validate_runtime_manifest_paths((PHASE01_ROOT, LIFECYCLE_ROOT, output_root, package_root))
    code_sha256 = source_tree_digest(package_root)
    started_at = _utc_now()
    mx.set_default_device(mx.gpu)
    operations = {
        "ar_fit": True,
        "cortical_values_loaded": False,
        "cortical_target_performance_read": False,
        "pca": False,
        "washout_activated": False,
        "learned_bridge": False,
        "again_runtime_dependency": False,
        "worker_processes": 1,
        "mlx_device": "gpu:0",
    }

    phase01_result_path = PHASE01_ROOT / "result.json"
    substrate_path = PHASE01_ROOT / "aligned-target-substrate.npz"
    if sha256_file(phase01_result_path) != PHASE01_RESULT_SHA256:
        raise ValueError("sealed Phase 01 result hash changed")
    if sha256_file(substrate_path) != PHASE01_SUBSTRATE_FILE_SHA256:
        raise ValueError("sealed Phase 01 substrate-file hash changed")
    phase01_result = json.loads(phase01_result_path.read_text(encoding="utf-8"))
    if not phase01_result.get("phase02_authorized") or phase01_result.get("status") != "pass":
        raise ValueError("Phase 01 did not authorize Phase 02")
    if phase01_result["digests"]["target_source_sha256"] != PHASE01_TARGET_SOURCE_SHA256:
        raise ValueError("Phase 01 target-source identity changed")
    if phase01_result["digests"]["substrate_arrays_sha256"] != PHASE01_SUBSTRATE_ARRAYS_SHA256:
        raise ValueError("Phase 01 substrate-array identity changed")

    target_registration = json.loads(
        (PHASE01_ROOT / "target-registration.json").read_text(encoding="utf-8")
    )
    label_dynamics = json.loads((PHASE01_ROOT / "label-dynamics.json").read_text(encoding="utf-8"))
    selected_target = target_registration["initial_no_washout"]
    target_start = int(selected_target["start_row"])
    target_end = int(selected_target["end_row"])
    target_width = int(selected_target["width_rows"])
    pacf_lag = int(label_dynamics["pacf_decay_lag_rows"])
    lag_depths = derive_lag_depths(pacf_decay_lag=pacf_lag, target_width=target_width)
    max_depth = max(lag_depths)

    with np.load(substrate_path, allow_pickle=False) as substrate:
        required = {
            "video_id",
            "row_index",
            "arousal",
            "selected_future_max_increase",
            "selected_valid_mask",
        }
        if not required.issubset(substrate.files):
            raise ValueError("Phase 01 substrate lacks required Phase 02 arrays")
        if any("cortical" in name.casefold() for name in substrate.files):
            raise ValueError("Phase 01 substrate unexpectedly exposes cortical values")
        video_id = substrate["video_id"].astype(np.int16)
        row_index = substrate["row_index"].astype(np.int32)
        arousal = substrate["arousal"].astype(np.float64)
        target = substrate["selected_future_max_increase"].astype(np.float64)
        target_valid = substrate["selected_valid_mask"].astype(np.bool_)
    if not (len(video_id) == len(row_index) == len(arousal) == len(target) == EXPECTED_ROW_COUNT):
        raise ValueError("Phase 02 substrate row count changed")
    if tuple(map(str, np.unique(video_id))) != EXPECTED_VIDEO_IDS:
        raise ValueError("Phase 02 substrate video inventory changed")
    eligible_mask = common_history_mask(video_id, row_index, target_valid, max_depth=max_depth)
    eligible_indices = np.flatnonzero(eligible_mask)
    if not np.isfinite(target[eligible_indices]).all():
        raise ValueError("selected target is nonfinite on eligible rows")

    grouped_cells = [
        grouped_split_cell(
            eligible_indices,
            video_id,
            fold=fold,
            seed=derive_seed(PHASE01_TARGET_SOURCE_SHA256, f"grouped-outer-{fold}"),
            test_fraction=OUTER_TEST_FRACTION,
            validation_fraction=INNER_VALIDATION_FRACTION,
        )
        for fold in range(GROUPED_OUTER_REPEATS)
    ]
    blocked_cell = blocked_split_cell(
        eligible_indices,
        video_id,
        seed=derive_seed(PHASE01_TARGET_SOURCE_SHA256, "blocked-outer-0"),
        test_fraction=OUTER_TEST_FRACTION,
        validation_fraction=INNER_VALIDATION_FRACTION,
    )
    cells = (*grouped_cells, blocked_cell)

    request = {
        "schema": "veatic21_phase02_request_v1",
        "started_at": started_at,
        "phase": "phase-02-ar-baseline",
        "authority": {
            "master": str(MASTER_SPECIFICATION),
            "protocol": str(REBUILD_PROTOCOL),
            "current_state": str(CURRENT_STATE),
        },
        "input": {
            "phase01_root": str(PHASE01_ROOT),
            "phase01_result_sha256": PHASE01_RESULT_SHA256,
            "substrate_file_sha256": PHASE01_SUBSTRATE_FILE_SHA256,
            "target_source_sha256": PHASE01_TARGET_SOURCE_SHA256,
            "substrate_arrays_sha256": PHASE01_SUBSTRATE_ARRAYS_SHA256,
        },
        "target": {
            "formula": target_registration["formula"],
            "future_rows": [target_start, target_end],
            "future_seconds": [
                target_start * EXPECTED_TIME_STEP_SECONDS,
                target_end * EXPECTED_TIME_STEP_SECONDS,
            ],
            "prospective_washout_activated": False,
        },
        "frozen_design": {
            "common_max_history_depth_rows": max_depth,
            "candidate_lag_depth_rows": list(lag_depths),
            "lag_derivation": (
                "zero, powers of two through selected target width, Phase 01 PACF-decay "
                "landmark, and selected target width; deduplicated and sorted"
            ),
            "regularization_candidates": list(AR_REGULARIZATION_CANDIDATES),
            "grouped_outer_repeats": GROUPED_OUTER_REPEATS,
            "outer_test_fraction": OUTER_TEST_FRACTION,
            "inner_validation_fraction": INNER_VALIDATION_FRACTION,
            "hyperparameter_objective": "inner-validation raw PR-AUC",
            "hyperparameter_tie_break": (
                "smaller lag depth, then stronger regularization; stronger regularization "
                "for fixed-width simple controls"
            ),
            "event_threshold": "q90 of applicable training continuous target only",
            "decision_threshold": "maximum F1 on outer-training predictions only",
            "simple_controls": ["current_arousal", f"trailing_slope_width_{pacf_lag}"],
            "optimizer": {
                "runtime": "MLX",
                "device": "gpu:0",
                "workers": 1,
                "algorithm": "deterministic Adam logistic ridge",
                "learning_rate": AR_OPTIMIZER_LEARNING_RATE,
                "max_iterations": AR_OPTIMIZER_MAX_ITERATIONS,
                "final_refit_max_iterations": AR_FINAL_OPTIMIZER_MAX_ITERATIONS,
                "gradient_tolerance": AR_OPTIMIZER_TOLERANCE,
            },
            "cluster_bootstrap_resamples": CLUSTER_BOOTSTRAP_RESAMPLES,
            "again_numeric_or_fitted_choice_inherited": False,
        },
        "operations": operations,
        "code_sha256": code_sha256,
    }
    _atomic_write_json(staging / "request.json", request)

    split_rows: list[dict[str, object]] = []
    split_records: list[dict[str, object]] = []
    for cell in cells:
        split_sha256 = _split_digest(cell)
        record = {
            "cell_id": _cell_id(cell),
            "protocol": cell.protocol,
            "fold": cell.fold,
            "seed": cell.seed,
            "split_sha256": split_sha256,
            "outer_train_rows": len(cell.outer_train),
            "outer_test_rows": len(cell.outer_test),
            "inner_train_rows": len(cell.inner_train),
            "inner_validation_rows": len(cell.inner_validation),
            "outer_test_fraction_rows": len(cell.outer_test)
            / (len(cell.outer_train) + len(cell.outer_test)),
            "outer_train_videos": int(len(np.unique(video_id[cell.outer_train]))),
            "outer_test_videos": int(len(np.unique(video_id[cell.outer_test]))),
        }
        split_records.append(record)
        for partition, indices in (
            ("inner_train", cell.inner_train),
            ("inner_validation", cell.inner_validation),
            ("outer_test", cell.outer_test),
        ):
            split_rows.extend(
                {
                    "cell_id": _cell_id(cell),
                    "protocol": cell.protocol,
                    "fold": cell.fold,
                    "seed": cell.seed,
                    "partition": partition,
                    "global_index": int(index),
                    "video_id": int(video_id[index]),
                    "row_index": int(row_index[index]),
                }
                for index in indices
            )
    split_manifest = {
        "schema": "veatic21_phase02_split_manifest_v1",
        "common_eligible_rows": len(eligible_indices),
        "common_row_identity_sha256": _array_bundle_digest(
            {
                "video_id": video_id[eligible_indices],
                "row_index": row_index[eligible_indices],
            }
        ),
        "cells": split_records,
    }
    _atomic_write_json(staging / "split-manifest.json", split_manifest)
    _atomic_write_csv(staging / "split-ownership.csv", split_rows)

    search_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    per_video_rows: list[dict[str, object]] = []
    prediction_records: list[dict[str, object]] = []
    model_records: list[dict[str, object]] = []
    decomposition_cells: list[dict[str, object]] = []
    winner_records: list[dict[str, object]] = []

    for cell in cells:
        cell_id = _cell_id(cell)
        ar_winner, ar_search = _search_ar(
            arousal=arousal,
            target=target,
            cell=cell,
            lag_depths=lag_depths,
        )

        def current_builder(indices: np.ndarray) -> np.ndarray:
            return build_ar_features(arousal, indices, depth=0)

        def slope_builder(indices: np.ndarray) -> np.ndarray:
            return build_trailing_slope_feature(arousal, indices, width=pacf_lag)

        current_winner, current_search = _search_single_feature(
            lane="current_arousal",
            feature_builder=current_builder,
            target=target,
            cell=cell,
        )
        slope_winner, slope_search = _search_single_feature(
            lane="trailing_slope",
            feature_builder=slope_builder,
            target=target,
            cell=cell,
        )
        search_rows.extend((*ar_search, *current_search, *slope_search))

        outer_q90 = _threshold(target[cell.outer_train])
        train_labels = (target[cell.outer_train] >= outer_q90).astype(np.int8)
        test_labels = (target[cell.outer_test] >= outer_q90).astype(np.int8)
        if len(np.unique(train_labels)) != 2 or len(np.unique(test_labels)) != 2:
            raise ValueError(f"outer cell lacks both event classes: {cell_id}")

        ar_depth = int(ar_winner["lag_depth_rows"])
        ar_model = fit_logistic_mlx(
            build_ar_features(arousal, cell.outer_train, depth=ar_depth),
            train_labels,
            regularization=float(ar_winner["regularization"]),
            max_iterations=AR_FINAL_OPTIMIZER_MAX_ITERATIONS,
        )
        current_model = fit_logistic_mlx(
            current_builder(cell.outer_train),
            train_labels,
            regularization=float(current_winner["regularization"]),
            max_iterations=AR_FINAL_OPTIMIZER_MAX_ITERATIONS,
        )
        slope_model = fit_logistic_mlx(
            slope_builder(cell.outer_train),
            train_labels,
            regularization=float(slope_winner["regularization"]),
            max_iterations=AR_FINAL_OPTIMIZER_MAX_ITERATIONS,
        )
        if not (ar_model.converged and current_model.converged and slope_model.converged):
            raise ValueError(f"final outer model did not converge: {cell_id}")
        train_scores = {
            "ar_probability": predict_logistic_mlx(
                ar_model, build_ar_features(arousal, cell.outer_train, depth=ar_depth)
            ),
            "current_arousal_probability": predict_logistic_mlx(
                current_model, current_builder(cell.outer_train)
            ),
            "trailing_slope_probability": predict_logistic_mlx(
                slope_model, slope_builder(cell.outer_train)
            ),
        }
        test_scores = {
            "ar_probability": predict_logistic_mlx(
                ar_model, build_ar_features(arousal, cell.outer_test, depth=ar_depth)
            ),
            "current_arousal_probability": predict_logistic_mlx(
                current_model, current_builder(cell.outer_test)
            ),
            "trailing_slope_probability": predict_logistic_mlx(
                slope_model, slope_builder(cell.outer_test)
            ),
            "chance_probability": np.full(
                len(cell.outer_test), float(np.mean(train_labels)), dtype=np.float64
            ),
        }
        decision_thresholds = {
            lane: select_decision_threshold(train_labels, scores)
            for lane, scores in train_scores.items()
        }
        decision_thresholds["chance_probability"] = float(np.mean(train_labels))
        lane_metrics: dict[str, dict[str, float | int]] = {}
        for lane, scores in test_scores.items():
            metrics = spike_metrics(
                test_labels, scores, decision_threshold=decision_thresholds[lane]
            )
            lane_metrics[lane] = metrics
            metric_rows.append(
                {
                    "cell_id": cell_id,
                    "protocol": cell.protocol,
                    "fold": cell.fold,
                    "seed": cell.seed,
                    "lane": lane,
                    "outer_q90_threshold": outer_q90,
                    **metrics,
                }
            )
            video_metrics = per_video_pr_auc(video_id[cell.outer_test], test_labels, scores)
            for video, value in video_metrics.items():
                mask = video_id[cell.outer_test].astype(str) == video
                per_video_rows.append(
                    {
                        "cell_id": cell_id,
                        "protocol": cell.protocol,
                        "fold": cell.fold,
                        "seed": cell.seed,
                        "lane": lane,
                        "video_id": video,
                        "rows": int(np.sum(mask)),
                        "positives": int(np.sum(test_labels[mask])),
                        "pr_auc": "" if value is None else value,
                        "defined": value is not None,
                    }
                )

        simple_lane = max(
            ("current_arousal_probability", "trailing_slope_probability"),
            key=lambda lane: float(
                current_winner["inner_validation_pr_auc"]
                if lane == "current_arousal_probability"
                else slope_winner["inner_validation_pr_auc"]
            ),
        )
        bootstrap_chance = paired_video_bootstrap_raw_pr_auc_delta(
            video_id[cell.outer_test],
            test_labels,
            test_scores["ar_probability"],
            test_scores["chance_probability"],
            seed=derive_seed(PHASE01_TARGET_SOURCE_SHA256, f"{cell_id}-bootstrap-chance"),
            resamples=CLUSTER_BOOTSTRAP_RESAMPLES,
        )
        bootstrap_simple = paired_video_bootstrap_raw_pr_auc_delta(
            video_id[cell.outer_test],
            test_labels,
            test_scores["ar_probability"],
            test_scores[simple_lane],
            seed=derive_seed(PHASE01_TARGET_SOURCE_SHA256, f"{cell_id}-bootstrap-simple"),
            resamples=CLUSTER_BOOTSTRAP_RESAMPLES,
        )
        decomposition_cells.append(
            {
                "cell_id": cell_id,
                "protocol": cell.protocol,
                "fold": cell.fold,
                "seed": cell.seed,
                "history_depth_rows": ar_depth,
                "history_rows_inclusive": [-ar_depth, 0],
                "target_rows_inclusive": [target_start, target_end],
                "history_target_gap_rows": target_start - 1,
                "history_target_overlap_rows": 0,
                "outer_train_prevalence": float(np.mean(train_labels)),
                "outer_test_prevalence": float(np.mean(test_labels)),
                "ar_pr_auc": lane_metrics["ar_probability"]["pr_auc"],
                "chance_pr_auc": lane_metrics["chance_probability"]["pr_auc"],
                "current_arousal_pr_auc": lane_metrics["current_arousal_probability"]["pr_auc"],
                "trailing_slope_pr_auc": lane_metrics["trailing_slope_probability"]["pr_auc"],
                "training_owned_strongest_simple_lane": simple_lane,
                "ar_delta_vs_chance": float(lane_metrics["ar_probability"]["pr_auc"])
                - float(lane_metrics["chance_probability"]["pr_auc"]),
                "ar_delta_vs_strongest_simple": float(lane_metrics["ar_probability"]["pr_auc"])
                - float(lane_metrics[simple_lane]["pr_auc"]),
                "ar_vs_chance_cluster_bootstrap": bootstrap_chance,
                "ar_vs_strongest_simple_cluster_bootstrap": bootstrap_simple,
                "per_video_defined_ar": sum(
                    row["lane"] == "ar_probability"
                    and row["cell_id"] == cell_id
                    and bool(row["defined"])
                    for row in per_video_rows
                ),
                "per_video_undefined_ar": sum(
                    row["lane"] == "ar_probability"
                    and row["cell_id"] == cell_id
                    and not bool(row["defined"])
                    for row in per_video_rows
                ),
            }
        )

        prediction_arrays = {
            "video_id": video_id[cell.outer_test],
            "row_index": row_index[cell.outer_test],
            "global_index": cell.outer_test.astype(np.int32),
            "target_continuous": target[cell.outer_test],
            "event_label": test_labels,
            "outer_q90_threshold": np.full(len(cell.outer_test), outer_q90),
            **test_scores,
        }
        prediction_path = staging / "predictions" / f"{cell_id}.npz"
        _atomic_save_npz(prediction_path, prediction_arrays)
        prediction_records.append(
            {
                "cell_id": cell_id,
                "protocol": cell.protocol,
                "fold": cell.fold,
                "seed": cell.seed,
                "path": prediction_path.relative_to(staging).as_posix(),
                "sha256": sha256_file(prediction_path),
                "arrays_sha256": _array_bundle_digest(prediction_arrays),
                "row_identity_sha256": _array_bundle_digest(
                    {
                        "video_id": prediction_arrays["video_id"],
                        "row_index": prediction_arrays["row_index"],
                    }
                ),
                "rows": len(cell.outer_test),
                "lanes": list(test_scores),
            }
        )
        model_arrays = {
            **_model_arrays(ar_model, "ar"),
            **_model_arrays(current_model, "current_arousal"),
            **_model_arrays(slope_model, "trailing_slope"),
            "ar_lag_depth_rows": np.asarray(ar_depth, dtype=np.int16),
            "outer_q90_threshold": np.asarray(outer_q90, dtype=np.float64),
            **{
                f"{lane}_decision_threshold": np.asarray(value, dtype=np.float64)
                for lane, value in decision_thresholds.items()
            },
        }
        model_path = staging / "models" / f"{cell_id}.npz"
        _atomic_save_npz(model_path, model_arrays)
        model_records.append(
            {
                "cell_id": cell_id,
                "path": model_path.relative_to(staging).as_posix(),
                "sha256": sha256_file(model_path),
                "arrays_sha256": _array_bundle_digest(model_arrays),
                "mlx_device": "gpu:0",
                "worker_processes": 1,
            }
        )
        winner_records.append(
            {
                "cell_id": cell_id,
                "protocol": cell.protocol,
                "fold": cell.fold,
                "seed": cell.seed,
                "outer_q90_threshold": outer_q90,
                "ar": ar_winner,
                "current_arousal": current_winner,
                "trailing_slope": slope_winner,
                "outer_train_decision_thresholds": decision_thresholds,
            }
        )

    _atomic_write_csv(staging / "hyperparameter-search.csv", search_rows)
    _atomic_write_csv(staging / "fold-metrics.csv", metric_rows)
    _atomic_write_csv(staging / "per-video-metrics.csv", per_video_rows)
    _atomic_write_json(
        staging / "prediction-manifest.json",
        {
            "schema": "veatic21_phase02_prediction_manifest_v1",
            "heldout_predictions_frozen": True,
            "matched_future_lanes_must_reuse_exact_rows_and_ar_probability": True,
            "records": prediction_records,
        },
    )
    _atomic_write_json(
        staging / "model-manifest.json",
        {
            "schema": "veatic21_phase02_model_manifest_v1",
            "runtime": "MLX",
            "device": "gpu:0",
            "worker_processes": 1,
            "records": model_records,
        },
    )
    _atomic_write_json(
        staging / "selected-hyperparameters.json",
        {
            "schema": "veatic21_phase02_selected_hyperparameters_v1",
            "selection_rows": "inner validation within corresponding outer training only",
            "records": winner_records,
        },
    )

    grouped_ar = [
        float(row["pr_auc"])
        for row in metric_rows
        if row["protocol"] == "grouped_video" and row["lane"] == "ar_probability"
    ]
    blocked_ar = [
        float(row["pr_auc"])
        for row in metric_rows
        if row["protocol"] == "blocked_temporal" and row["lane"] == "ar_probability"
    ]
    dominance = {
        "schema": "veatic21_phase02_ar_dominance_decomposition_v1",
        "target_rows_inclusive": [target_start, target_end],
        "target_seconds_inclusive": [
            target_start * EXPECTED_TIME_STEP_SECONDS,
            target_end * EXPECTED_TIME_STEP_SECONDS,
        ],
        "candidate_history_depth_rows": list(lag_depths),
        "common_max_history_depth_rows": max_depth,
        "target_history_overlap_rows": 0,
        "boundary_gap_rows": target_start - 1,
        "cells": decomposition_cells,
        "grouped_ar_pr_auc": {
            "fold_values": grouped_ar,
            "median": float(np.median(grouped_ar)),
            "minimum": float(np.min(grouped_ar)),
            "maximum": float(np.max(grouped_ar)),
        },
        "blocked_ar_pr_auc": {"cell_values": blocked_ar, "median": float(np.median(blocked_ar))},
        "washout_decision": {
            "activated": False,
            "selected_candidate": None,
            "reason": (
                "Phase 02 quantifies legal persistence and boundary proximity, but the "
                "registered washout procedure requires later control-complete development "
                "evidence before any activation or selection"
            ),
            "phase01_candidate_rows_remain_prospective": target_registration["prospective_washout"][
                "candidate_starts"
            ],
        },
    }
    _atomic_write_json(staging / "ar-dominance-decomposition.json", dominance)

    checks = dict.fromkeys(PHASE02_CHECKS, True)
    if not phase03_authorized(checks):
        raise ValueError("Phase 02 mandatory check matrix is incomplete")
    completed_at = _utc_now()
    result = {
        "schema": "veatic21_phase02_result_v1",
        "phase": "phase-02-ar-baseline",
        "status": "pass",
        "started_at": started_at,
        "completed_at": completed_at,
        "code_sha256": code_sha256,
        "checks": checks,
        "input_hashes": request["input"],
        "videos": len(EXPECTED_VIDEO_IDS),
        "source_rows": EXPECTED_ROW_COUNT,
        "eligible_rows": len(eligible_indices),
        "row_hz": EXPECTED_ROW_HZ,
        "target_future_rows": [target_start, target_end],
        "lag_depth_candidates": list(lag_depths),
        "protocol_cells": {"grouped_video": len(grouped_cells), "blocked_temporal": 1},
        "grouped_ar_pr_auc_median": float(np.median(grouped_ar)),
        "blocked_ar_pr_auc": blocked_ar[0],
        "prediction_manifest_sha256": sha256_file(staging / "prediction-manifest.json"),
        "model_manifest_sha256": sha256_file(staging / "model-manifest.json"),
        "split_manifest_sha256": sha256_file(staging / "split-manifest.json"),
        "dominance_decomposition_sha256": sha256_file(staging / "ar-dominance-decomposition.json"),
        "operations": operations,
        "washout_activated": False,
        "phase03_authorized": True,
        "single_next_authorized_action": (
            "Phase 03 raw cortical benchmark under exact matched grouped-video and "
            "blocked-temporal Phase 02 rows and frozen AR predictions"
        ),
    }
    _atomic_write_json(staging / "result.json", result)

    ledger = {
        "schema": "veatic21_derivation_ledger_v1",
        "phase": "phase-02-ar-baseline",
        "code_sha256": code_sha256,
        "input_hashes": request["input"],
        "numeric_choices": [
            {
                "choice": "candidate_lag_depth_rows",
                "value": list(lag_depths),
                "derivation": request["frozen_design"]["lag_derivation"],
                "owner": "sealed Phase 01 label dynamics; frozen before fitting",
            },
            {
                "choice": "regularization_candidates",
                "value": list(AR_REGULARIZATION_CANDIDATES),
                "derivation": "broad logarithmic ridge grid including the unregularized floor",
                "owner": "Phase 02 implementation; frozen before fitting",
            },
            {
                "choice": "split_seeds",
                "value": [cell.seed for cell in cells],
                "derivation": "SHA-256 labels derived from sealed VEATIC target-source digest",
                "owner": "Phase 02 split creation",
            },
        ],
        "fitted_choices": winner_records,
        "again_numeric_choices_inherited": False,
        "again_paths_used": False,
    }
    _atomic_write_json(staging / "veatic-derivation-ledger.json", ledger)

    report = f"""# VEATIC 2.1 Phase 02 Fresh Target-Specific AR Baseline

Status: **PASS**

Phase 02 fitted a fresh VEATIC-only autoregressive event floor for the sealed continuous
future-maximum-increase target `t+{target_start}..t+{target_end}`. The common causal-history
mask retained {len(eligible_indices):,} rows across all {len(EXPECTED_VIDEO_IDS)} videos.
Candidate lag depths `{list(lag_depths)}` came only from the Phase 01 VEATIC PACF landmark and
selected target width. Every lag and ridge choice was selected by nested inner-validation raw
PR-AUC. Each outer q90, normalization, final model, and decision threshold remained owned by
its outer-training partition.

Five grouped-video 70/30 cells produced median held-out AR PR-AUC
`{np.median(grouped_ar):.6f}` (range `{np.min(grouped_ar):.6f}`–`{np.max(grouped_ar):.6f}`).
The separately reported per-video forward blocked-temporal 70/30 cell produced AR PR-AUC
`{blocked_ar[0]:.6f}`. Fold metrics also contain prevalence/chance, AP skill, ROC-AUC,
precision, recall, F1, Brier score, top-1/5/10% recall and lift, defined-only per-video
PR-AUC, and positive counts. Paired whole-video bootstrap intervals compare AR against chance
and the training-owned strongest simple causal-history baseline.

Exact outer-test rows, event labels, continuous targets, and AR/current/slope/chance
probabilities are frozen per target/protocol/fold/seed with file and array checksums. Phase 03
must reuse the exact rows and AR predictions for every matched lane.

The target begins at `t+{target_start}` after causal history ending at `t`; history/target
overlap is zero and the boundary gap is {target_start - 1} rows. The registered prospective
washout candidates remain inactive and unselected because activation requires later
control-complete development evidence. No cortical values were loaded, no PCA or bridge was
fit, and no AGAIN runtime code, data, numeric choice, seed, split, fitted object, or prediction
entered this phase. MLX ran all learned fitting and scoring on `gpu:0` in one worker process.

Code SHA-256: `{code_sha256}`
Prediction manifest SHA-256: `{result["prediction_manifest_sha256"]}`
Model manifest SHA-256: `{result["model_manifest_sha256"]}`
Split manifest SHA-256: `{result["split_manifest_sha256"]}`
Dominance decomposition SHA-256: `{result["dominance_decomposition_sha256"]}`
"""
    _atomic_write_text(staging / "report.md", report)

    artifact_manifest = {
        "schema": "veatic21_phase02_artifact_manifest_v1",
        "created_at": _utc_now(),
        "root": str(output_root),
        "artifacts": _artifact_inventory(staging),
    }
    _atomic_write_json(staging / "artifact-manifest.json", artifact_manifest)
    checksum_paths = [
        candidate
        for candidate in sorted(path for path in staging.rglob("*") if path.is_file())
        if candidate.name != "checksums.sha256"
    ]
    _atomic_write_text(
        staging / "checksums.sha256",
        "".join(
            f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}\n"
            for path in checksum_paths
        ),
    )
    os.replace(staging, output_root)
    return result


def discard_failed_staging() -> None:
    """Remove only this runner's own incomplete staging directory after inspection."""

    staging = PHASE02_ROOT.parent / f".{PHASE02_ROOT.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
