"""VEATIC 2.1 Phase 04 fold-owned PCA and temporal representation search."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from sklearn.metrics import average_precision_score

from neural_bridge.veatic21.ar import (
    build_ar_features,
    fit_logistic_mlx,
    predict_logistic_mlx,
    select_decision_threshold,
    spike_metrics,
)
from neural_bridge.veatic21.contracts import (
    AR_FINAL_OPTIMIZER_MAX_ITERATIONS,
    CLUSTER_BOOTSTRAP_RESAMPLES,
    CURRENT_STATE,
    EXPECTED_CORTICAL_WIDTH,
    EXPECTED_ROW_COUNT,
    EXPECTED_ROW_HZ,
    EXPECTED_VIDEO_IDS,
    MASTER_SPECIFICATION,
    PCA_JACOBI_TOLERANCE,
    PCA_MAX_RANK,
    PCA_ORTHOGONALITY_TOLERANCE,
    PCA_OVERSAMPLE_DIVISOR,
    PCA_POWER_ITERATIONS,
    PCA_PREFIX_WIDTHS,
    PCA_SUBSPACE_STABILITY_FLOOR,
    PHASE00_ROOT,
    PHASE01_ROOT,
    PHASE02_ROOT,
    PHASE03_ROOT,
    PHASE04_ROOT,
    REBUILD_PROTOCOL,
    REPOSITORY_ROOT,
    TRIBE_ROOT,
    reject_forbidden_runtime_path,
    validate_runtime_manifest_paths,
)
from neural_bridge.veatic21.evidence import (
    paired_video_bootstrap_raw_pr_auc_delta,
    per_video_pr_auc,
    sha256_file,
    source_tree_digest,
)
from neural_bridge.veatic21.pca import (
    PCAModel,
    fit_pca_mlx,
    subspace_overlap,
    transform_pca_mlx,
)
from neural_bridge.veatic21.phase03 import (
    FAMILY_NAMES,
    PHASE00_ALLOWED_INPUT_MANIFEST_SHA256,
    PHASE00_RESULT_SHA256,
    PHASE01_SUBSTRATE_FILE_SHA256,
    PHASE02_MODEL_MANIFEST_SHA256,
    PHASE02_PREDICTION_MANIFEST_SHA256,
    PHASE02_RESULT_SHA256,
    PHASE02_SPLIT_MANIFEST_SHA256,
    PHASE02_SPLIT_OWNERSHIP_SHA256,
    _load_ar_model,
    _load_cells,
    _load_raw_substrate,
    _time_control,
    _video_mean_features,
)
from neural_bridge.veatic21.raw_cortical import (
    derive_phase03_seed,
    expand_control_to_width,
    fit_raw_discriminant_mlx,
    predict_raw_discriminant_mlx,
    shape_matched_random,
    within_partition_video_shuffle,
    within_video_label_permutation,
)

PHASE03_RESULT_SHA256 = "8c0839d8eb8ba5c20e4c13ae83367b0fe4e0e383b7e0c3b074b80f7a5cf38c16"
PHASE03_PREDICTION_MANIFEST_SHA256 = (
    "186acd0eb6017c7764fa1fc5215567e34ef5c55cfbee0b5bc94a0da6fc8b9d91"
)
PHASE03_CONTROL_MATRIX_SHA256 = "809208be17960eb98171409669ae8cf95c9432b855d48e7a5950d3a64706ca59"
TEMPORAL_DEPTHS = (0, 4, 6)

PHASE04_CHECKS = (
    "sealed_phase00_through_phase03_input_identity",
    "exact_phase02_target_split_fold_seed_and_frozen_ar",
    "final_tribe_cortical_prediction_only_real_input",
    "outer_training_only_scaling_and_pca_fit",
    "every_owned_eligible_training_row_used",
    "maximum_rank_512_basis_per_outer_cell",
    "nested_64_128_256_512_prefixes",
    "fixed_veatic_derived_pca_seeds",
    "float32_pca_accumulation",
    "generous_rank_fraction_oversampling",
    "sufficient_power_iterations",
    "pca_finite_values",
    "pca_component_orthogonality",
    "monotonic_cumulative_explained_variance",
    "pca_reconstruction_residual_audit",
    "independent_seed_subspace_stability",
    "train_transform_and_prefix_score_checksums",
    "veatic_derived_causal_temporal_candidates",
    "global_representation_selected_inner_only",
    "complete_applicable_control_matrix_for_every_candidate",
    "shuffled_pca_control",
    "shape_matched_random_pca_control",
    "train_only_video_mean_pca_control",
    "diagnostics_time_quality_controls",
    "label_permutation_pca_control",
    "no_video_architecture_ablation_inapplicable_documented",
    "exact_matched_outer_metric_rows",
    "complete_spike_metric_stack",
    "defined_only_per_video_pr_auc",
    "paired_video_cluster_bootstrap_primary_deltas",
    "grouped_and_blocked_reported_separately",
    "selected_representation_predictions_and_models_frozen",
    "mlx_gpu_zero_single_worker",
    "no_washout_or_learned_bridge",
    "again_runtime_firewall",
    "phase05_only_authorization",
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


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
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


def _cell_id(cell: Any) -> str:
    return f"{cell.protocol}-fold-{cell.fold:02d}-seed-{cell.seed:010d}"


def aggregate_causal_scores(
    scores: np.ndarray,
    indices: np.ndarray,
    video_id: np.ndarray,
    *,
    depth: int,
) -> np.ndarray:
    """Average current through t-depth PCA rows without crossing a video boundary."""

    scores = np.asarray(scores, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int64)
    video_id = np.asarray(video_id)
    if depth < 0 or np.any(indices < depth):
        raise ValueError("causal aggregation indices do not support depth")
    history = np.column_stack([indices - lag for lag in range(depth + 1)])
    if not np.all(video_id[history] == video_id[indices, None]):
        raise ValueError("causal aggregation crossed a video boundary")
    return np.mean(scores[history], axis=1, dtype=np.float32)


def _split_safe_shuffle_source(
    cell: Any, video_id: np.ndarray, row_index: np.ndarray
) -> np.ndarray:
    source = np.arange(len(video_id), dtype=np.int64)
    outer_train_videos = set(video_id[cell.outer_train].tolist())
    outer_test_videos = set(video_id[cell.outer_test].tolist())
    grouped = outer_train_videos.isdisjoint(outer_test_videos)
    partitions: list[np.ndarray] = []
    if grouped:
        partitions.extend(
            np.flatnonzero(np.isin(video_id, list(videos)))
            for videos in (outer_train_videos, outer_test_videos)
        )
    else:
        for video in np.unique(video_id):
            train_rows = cell.outer_train[video_id[cell.outer_train] == video]
            cutoff = int(np.max(row_index[train_rows]))
            video_rows = np.flatnonzero(video_id == video)
            partitions.append(video_rows[row_index[video_rows] <= cutoff])
            partitions.append(video_rows[row_index[video_rows] > cutoff])
    for number, partition in enumerate(partitions):
        if len(partition) < 2:
            continue
        source[partition] = within_partition_video_shuffle(
            partition,
            video_id,
            seed=derive_phase03_seed(
                PHASE03_PREDICTION_MANIFEST_SHA256,
                f"{_cell_id(cell)}-pca-shuffle-{number}",
            ),
        )
    return source


def _family_features(
    family: str,
    *,
    raw: Any,
    cell: Any,
    video_id: np.ndarray,
    row_index: np.ndarray,
    time_seconds: np.ndarray,
    shuffle_source: np.ndarray,
) -> np.ndarray:
    all_indices = np.arange(len(video_id), dtype=np.int64)
    if family in {"real_cortical", "label_permutation_cortical"}:
        return raw.cortical
    if family == "shuffled_cortical":
        return raw.cortical[shuffle_source]
    if family == "shape_matched_random":
        return shape_matched_random(
            len(video_id),
            EXPECTED_CORTICAL_WIDTH,
            seed=derive_phase03_seed(
                PHASE03_PREDICTION_MANIFEST_SHA256,
                f"{_cell_id(cell)}-pca-random-full",
            ),
        )
    if family == "train_only_video_mean":
        return _video_mean_features(raw.cortical, cell.outer_train, all_indices, video_id)
    if family == "diagnostics_only":
        return expand_control_to_width(raw.diagnostics, width=EXPECTED_CORTICAL_WIDTH)
    if family == "time_video_time_only":
        return _time_control(all_indices, video_id, time_seconds)
    if family == "quality_motion_luma_only":
        return expand_control_to_width(raw.nuisance, width=EXPECTED_CORTICAL_WIDTH)
    raise ValueError(f"unhandled PCA family: {family}")


def _score_projected_family(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    ar_train: np.ndarray,
    ar_validation: np.ndarray,
) -> dict[str, float]:
    discriminant = fit_raw_discriminant_mlx(train_features, train_labels)
    train_projection = predict_raw_discriminant_mlx(discriminant, train_features)
    validation_projection = predict_raw_discriminant_mlx(discriminant, validation_features)
    regularization = 1.0 / len(train_features)
    only = fit_logistic_mlx(
        train_projection.reshape(-1, 1),
        train_labels,
        regularization=regularization,
        max_iterations=AR_FINAL_OPTIMIZER_MAX_ITERATIONS,
    )
    fusion_train = np.column_stack(
        (
            np.clip(np.log(ar_train / np.maximum(1.0 - ar_train, 1e-12)), -30.0, 30.0),
            train_projection,
        )
    )
    fusion_validation = np.column_stack(
        (
            np.clip(
                np.log(ar_validation / np.maximum(1.0 - ar_validation, 1e-12)),
                -30.0,
                30.0,
            ),
            validation_projection,
        )
    )
    fusion = fit_logistic_mlx(
        fusion_train,
        train_labels,
        regularization=regularization,
        max_iterations=AR_FINAL_OPTIMIZER_MAX_ITERATIONS,
    )
    if not only.converged or not fusion.converged:
        raise ValueError("projected PCA calibrator did not converge")
    only_train = predict_logistic_mlx(only, train_projection.reshape(-1, 1))
    only_validation = predict_logistic_mlx(only, validation_projection.reshape(-1, 1))
    fusion_train_probability = predict_logistic_mlx(fusion, fusion_train)
    fusion_validation_probability = predict_logistic_mlx(fusion, fusion_validation)
    return {
        "only_train_pr_auc": float(average_precision_score(train_labels, only_train)),
        "only_validation_pr_auc": float(
            average_precision_score(validation_labels, only_validation)
        ),
        "fusion_train_pr_auc": float(
            average_precision_score(train_labels, fusion_train_probability)
        ),
        "fusion_validation_pr_auc": float(
            average_precision_score(validation_labels, fusion_validation_probability)
        ),
    }


def _fit_final_family(
    *,
    family: str,
    train_features: np.ndarray,
    test_features: np.ndarray,
    fit_labels: np.ndarray,
    ar_train: np.ndarray,
    ar_test: np.ndarray,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, float],
    dict[str, float],
]:
    discriminant = fit_raw_discriminant_mlx(train_features, fit_labels)
    train_projection = predict_raw_discriminant_mlx(discriminant, train_features)
    test_projection = predict_raw_discriminant_mlx(discriminant, test_features)
    regularization = 1.0 / len(train_features)
    only_model = fit_logistic_mlx(
        train_projection.reshape(-1, 1),
        fit_labels,
        regularization=regularization,
        max_iterations=AR_FINAL_OPTIMIZER_MAX_ITERATIONS,
    )
    fusion_train = np.column_stack(
        (
            np.clip(np.log(ar_train / np.maximum(1.0 - ar_train, 1e-12)), -30.0, 30.0),
            train_projection,
        )
    )
    fusion_test = np.column_stack(
        (
            np.clip(np.log(ar_test / np.maximum(1.0 - ar_test, 1e-12)), -30.0, 30.0),
            test_projection,
        )
    )
    fusion_model = fit_logistic_mlx(
        fusion_train,
        fit_labels,
        regularization=regularization,
        max_iterations=AR_FINAL_OPTIMIZER_MAX_ITERATIONS,
    )
    if not only_model.converged or not fusion_model.converged:
        raise ValueError(f"final PCA calibrator did not converge: {family}")
    only_train = predict_logistic_mlx(only_model, train_projection.reshape(-1, 1))
    only_test = predict_logistic_mlx(only_model, test_projection.reshape(-1, 1))
    fusion_train_probability = predict_logistic_mlx(fusion_model, fusion_train)
    fusion_test_probability = predict_logistic_mlx(fusion_model, fusion_test)
    only_lane = f"{family}_only"
    fusion_lane = f"ar_plus_{family}"
    arrays = {
        f"{family}_discriminant_mean": discriminant.mean,
        f"{family}_discriminant_scale": discriminant.scale,
        f"{family}_discriminant_direction": discriminant.direction,
        f"{family}_discriminant_bias": np.asarray(discriminant.projection_bias, dtype=np.float64),
        f"{only_lane}_calibrator_mean": only_model.mean,
        f"{only_lane}_calibrator_scale": only_model.scale,
        f"{only_lane}_calibrator_weights": only_model.weights,
        f"{only_lane}_calibrator_regularization": np.asarray(
            only_model.regularization, dtype=np.float64
        ),
        f"{only_lane}_calibrator_iterations": np.asarray(only_model.iterations, dtype=np.int32),
        f"{only_lane}_calibrator_converged": np.asarray(only_model.converged, dtype=np.bool_),
        f"{only_lane}_calibrator_final_gradient_norm": np.asarray(
            only_model.final_gradient_norm, dtype=np.float64
        ),
        f"{fusion_lane}_calibrator_mean": fusion_model.mean,
        f"{fusion_lane}_calibrator_scale": fusion_model.scale,
        f"{fusion_lane}_calibrator_weights": fusion_model.weights,
        f"{fusion_lane}_calibrator_regularization": np.asarray(
            fusion_model.regularization, dtype=np.float64
        ),
        f"{fusion_lane}_calibrator_iterations": np.asarray(fusion_model.iterations, dtype=np.int32),
        f"{fusion_lane}_calibrator_converged": np.asarray(fusion_model.converged, dtype=np.bool_),
        f"{fusion_lane}_calibrator_final_gradient_norm": np.asarray(
            fusion_model.final_gradient_norm, dtype=np.float64
        ),
        f"{family}_discriminant_positive_rows": np.asarray(
            discriminant.positive_rows, dtype=np.int32
        ),
        f"{family}_discriminant_negative_rows": np.asarray(
            discriminant.negative_rows, dtype=np.int32
        ),
    }
    probabilities = {only_lane: only_test, fusion_lane: fusion_test_probability}
    thresholds = {
        only_lane: select_decision_threshold(fit_labels, only_train),
        fusion_lane: select_decision_threshold(fit_labels, fusion_train_probability),
    }
    training = {
        only_lane: float(average_precision_score(fit_labels, only_train)),
        fusion_lane: float(average_precision_score(fit_labels, fusion_train_probability)),
    }
    return arrays, probabilities, thresholds, training


def _pca_arrays(
    primary: PCAModel,
    secondary: PCAModel,
    fallback_models: Mapping[int, tuple[PCAModel, PCAModel]],
) -> dict[str, np.ndarray]:
    arrays = {
        "mean": primary.mean,
        "scale": primary.scale,
        "components_512": primary.components,
        "explained_variance_512": primary.explained_variance,
        "secondary_components_256": secondary.components,
        "secondary_explained_variance_256": secondary.explained_variance,
        "primary_seed": np.asarray(primary.seed, dtype=np.uint32),
        "secondary_seed": np.asarray(secondary.seed, dtype=np.uint32),
        "primary_orthogonality_max_abs": np.asarray(
            primary.orthogonality_max_abs, dtype=np.float64
        ),
        "secondary_orthogonality_max_abs": np.asarray(
            secondary.orthogonality_max_abs, dtype=np.float64
        ),
        "primary_reconstruction_residual_fraction": np.asarray(
            primary.reconstruction_residual_fraction, dtype=np.float64
        ),
    }
    for width, (fallback_primary, fallback_secondary) in fallback_models.items():
        prefix = f"fallback_{width}"
        arrays.update(
            {
                f"{prefix}_mean": fallback_primary.mean,
                f"{prefix}_scale": fallback_primary.scale,
                f"{prefix}_primary_components": fallback_primary.components,
                f"{prefix}_primary_explained_variance": fallback_primary.explained_variance,
                f"{prefix}_secondary_components": fallback_secondary.components,
                f"{prefix}_secondary_explained_variance": (fallback_secondary.explained_variance),
                f"{prefix}_primary_seed": np.asarray(fallback_primary.seed, dtype=np.uint32),
                f"{prefix}_secondary_seed": np.asarray(fallback_secondary.seed, dtype=np.uint32),
            }
        )
    return arrays


def _projection_cache_path(
    staging: Path,
    cell_id: str,
    family: str,
    width: int,
    fallback_widths: set[int],
) -> Path:
    filename = f"{family}-width-{width}.npy" if width in fallback_widths else f"{family}.npy"
    return staging / "projection-cache" / cell_id / filename


def _artifact_inventory(root: Path) -> list[dict[str, object]]:
    excluded = {"artifact-manifest.json", "checksums.sha256"}
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
        if path.relative_to(root).as_posix() not in excluded
    ]


def phase05_authorized(checks: Mapping[str, bool]) -> bool:
    return set(checks) == set(PHASE04_CHECKS) and all(checks.values())


def run_phase04(output_root: Path = PHASE04_ROOT) -> dict[str, Any]:
    """Fit, select, evaluate, and seal the Phase 04 PCA representation."""

    output_root = reject_forbidden_runtime_path(output_root)
    if output_root != PHASE04_ROOT:
        raise ValueError(f"Phase 04 output root must be exactly {PHASE04_ROOT}")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite Phase 04 root: {output_root}")
    staging = output_root.parent / f".{output_root.name}.staging"
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite Phase 04 staging root: {staging}")
    staging.mkdir(parents=True)
    package_root = REPOSITORY_ROOT / "src/neural_bridge/veatic21"
    validate_runtime_manifest_paths(
        (
            TRIBE_ROOT,
            PHASE00_ROOT,
            PHASE01_ROOT,
            PHASE02_ROOT,
            PHASE03_ROOT,
            output_root,
            package_root,
        )
    )
    mx.set_default_device(mx.gpu)
    started_at = _utc_now()
    code_sha256 = source_tree_digest(package_root)
    required_hashes = {
        PHASE00_ROOT / "result.json": PHASE00_RESULT_SHA256,
        PHASE00_ROOT / "allowed-input-manifest.json": PHASE00_ALLOWED_INPUT_MANIFEST_SHA256,
        PHASE01_ROOT / "aligned-target-substrate.npz": PHASE01_SUBSTRATE_FILE_SHA256,
        PHASE02_ROOT / "result.json": PHASE02_RESULT_SHA256,
        PHASE02_ROOT / "prediction-manifest.json": PHASE02_PREDICTION_MANIFEST_SHA256,
        PHASE02_ROOT / "model-manifest.json": PHASE02_MODEL_MANIFEST_SHA256,
        PHASE02_ROOT / "split-manifest.json": PHASE02_SPLIT_MANIFEST_SHA256,
        PHASE02_ROOT / "split-ownership.csv": PHASE02_SPLIT_OWNERSHIP_SHA256,
        PHASE03_ROOT / "result.json": PHASE03_RESULT_SHA256,
        PHASE03_ROOT / "prediction-manifest.json": PHASE03_PREDICTION_MANIFEST_SHA256,
        PHASE03_ROOT / "control-matrix.json": PHASE03_CONTROL_MATRIX_SHA256,
    }
    for path, expected in required_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"sealed prerequisite hash changed: {path}")
    phase03_result = json.loads((PHASE03_ROOT / "result.json").read_text(encoding="utf-8"))
    if phase03_result.get("status") != "pass" or not phase03_result.get("phase04_authorized"):
        raise ValueError("Phase 03 did not authorize Phase 04")

    with np.load(PHASE01_ROOT / "aligned-target-substrate.npz", allow_pickle=False) as arrays:
        video_id = arrays["video_id"].astype(np.int16)
        row_index = arrays["row_index"].astype(np.int32)
        time_seconds = arrays["time_seconds"].astype(np.float64)
        arousal = arrays["arousal"].astype(np.float64)
        target = arrays["selected_future_max_increase"].astype(np.float64)
    cells = _load_cells()

    request = {
        "schema": "veatic21_phase04_request_v1",
        "phase": "phase-04-pca-bridge",
        "started_at": started_at,
        "authority": {
            "master": str(MASTER_SPECIFICATION),
            "protocol": str(REBUILD_PROTOCOL),
            "current_state": str(CURRENT_STATE),
        },
        "input_hashes": {str(path): digest for path, digest in required_hashes.items()},
        "frozen_design": {
            "maximum_rank": PCA_MAX_RANK,
            "nested_prefix_widths": list(PCA_PREFIX_WIDTHS),
            "oversampling": PCA_MAX_RANK // PCA_OVERSAMPLE_DIVISOR,
            "oversampling_rule": "one quarter of requested rank",
            "power_iterations": PCA_POWER_ITERATIONS,
            "solver": (
                "MLX GPU randomized subspace iteration with twice-reorthogonalized block "
                "Gram-Schmidt plus MLX GPU Jacobi eigensolver"
            ),
            "scaling": "outer-training-only float32 feature mean and population scale",
            "primary_seed": "SHA-256 derived per cell from sealed Phase 03 prediction digest",
            "secondary_stability_rank": 256,
            "secondary_seed": "independent SHA-256 label per cell",
            "subspace_stability_floor": PCA_SUBSPACE_STABILITY_FLOOR,
            "orthogonality_tolerance": PCA_ORTHOGONALITY_TOLERANCE,
            "jacobi_tolerance": PCA_JACOBI_TOLERANCE,
            "temporal_depth_rows": list(TEMPORAL_DEPTHS),
            "temporal_derivation": (
                "current row, Phase 01 PACF-decay landmark 4, and selected target width 6"
            ),
            "temporal_operator": "causal trailing mean of PCA scores, inclusive current row",
            "representation_selection": (
                "global maximum median inner-validation fusion margin versus inner AR and "
                "strongest matched fusion control; tie by higher real fusion PR-AUC, then "
                "smaller width and shorter temporal depth"
            ),
            "outer_test_used_for_selection": False,
            "control_families": list(FAMILY_NAMES),
            "no_video_architecture_ablation": {
                "applicable": False,
                "reason": "Phase 04 classifier has no video embedding or architecture branch",
            },
            "again_width_temporal_seed_or_pca_inherited": False,
        },
        "operations": {
            "pca": True,
            "width_selection": True,
            "cortical_values_loaded": True,
            "washout_activated": False,
            "learned_bridge": False,
            "worker_processes": 1,
            "mlx_device": "gpu:0",
            "artificial_memory_cap": False,
            "again_runtime_dependency": False,
        },
        "code_sha256": code_sha256,
    }
    _atomic_write_json(staging / "request.json", request)
    raw = _load_raw_substrate(video_id, time_seconds)

    selected_phase02 = json.loads(
        (PHASE02_ROOT / "selected-hyperparameters.json").read_text(encoding="utf-8")
    )
    ar_winners = {record["cell_id"]: record["ar"] for record in selected_phase02["records"]}
    inner_family_rows: list[dict[str, object]] = []
    candidate_summaries: list[dict[str, object]] = []
    accuracy_records: list[dict[str, object]] = []
    cache_records: list[dict[str, object]] = []
    fallback_widths_by_cell: dict[str, set[int]] = {}

    for cell in cells:
        cell_id = _cell_id(cell)
        primary_seed = derive_phase03_seed(
            PHASE03_PREDICTION_MANIFEST_SHA256, f"{cell_id}-pca-primary"
        )
        secondary_seed = derive_phase03_seed(
            PHASE03_PREDICTION_MANIFEST_SHA256, f"{cell_id}-pca-secondary"
        )
        primary, primary_audit = fit_pca_mlx(
            raw.cortical[cell.outer_train],
            rank=PCA_MAX_RANK,
            oversampling=PCA_MAX_RANK // PCA_OVERSAMPLE_DIVISOR,
            seed=primary_seed,
        )
        secondary, secondary_audit = fit_pca_mlx(
            raw.cortical[cell.outer_train],
            rank=256,
            oversampling=256 // PCA_OVERSAMPLE_DIVISOR,
            seed=secondary_seed,
        )
        stability = {
            str(width): subspace_overlap(primary.components, secondary.components, width)
            for width in (64, 128, 256)
        }
        if not (
            bool(primary_audit["converged"])
            and bool(secondary_audit["converged"])
            and bool(primary_audit["orthogonality_pass"])
            and bool(secondary_audit["orthogonality_pass"])
            and bool(primary_audit["explained_variance_nonincreasing"])
            and bool(primary_audit["cumulative_explained_variance_monotonic"])
        ):
            raise ValueError(f"maximum-rank PCA accuracy audit failed: {cell_id}")
        affected_widths = {
            int(width)
            for width, overlap in stability.items()
            if overlap < PCA_SUBSPACE_STABILITY_FLOOR
        }
        fallback_models: dict[int, tuple[PCAModel, PCAModel]] = {}
        fallback_audits: dict[str, object] = {}
        for width in sorted(affected_widths):
            fallback_primary, fallback_primary_audit = fit_pca_mlx(
                raw.cortical[cell.outer_train],
                rank=width,
                oversampling=width // PCA_OVERSAMPLE_DIVISOR,
                seed=derive_phase03_seed(
                    PHASE03_PREDICTION_MANIFEST_SHA256,
                    f"{cell_id}-pca-fallback-{width}-primary",
                ),
            )
            fallback_secondary, fallback_secondary_audit = fit_pca_mlx(
                raw.cortical[cell.outer_train],
                rank=width,
                oversampling=width // PCA_OVERSAMPLE_DIVISOR,
                seed=derive_phase03_seed(
                    PHASE03_PREDICTION_MANIFEST_SHA256,
                    f"{cell_id}-pca-fallback-{width}-secondary",
                ),
            )
            fallback_overlap = subspace_overlap(
                fallback_primary.components, fallback_secondary.components, width
            )
            if not (
                bool(fallback_primary_audit["converged"])
                and bool(fallback_secondary_audit["converged"])
                and bool(fallback_primary_audit["orthogonality_pass"])
                and bool(fallback_secondary_audit["orthogonality_pass"])
                and bool(fallback_primary_audit["explained_variance_nonincreasing"])
                and bool(fallback_primary_audit["cumulative_explained_variance_monotonic"])
                and fallback_overlap >= PCA_SUBSPACE_STABILITY_FLOOR
            ):
                raise ValueError(f"separate width-{width} PCA audit failed: {cell_id}")
            fallback_models[width] = (fallback_primary, fallback_secondary)
            fallback_audits[str(width)] = {
                "primary": fallback_primary_audit,
                "secondary": fallback_secondary_audit,
                "subspace_overlap": fallback_overlap,
            }
        fallback_widths_by_cell[cell_id] = affected_widths
        pca_arrays = _pca_arrays(primary, secondary, fallback_models)
        pca_path = staging / "pca-models" / f"{cell_id}.npz"
        _atomic_save_npz(pca_path, pca_arrays)
        accuracy_records.append(
            {
                "cell_id": cell_id,
                "outer_train_rows": len(cell.outer_train),
                "fit_row_identity_sha256": _array_bundle_digest(
                    {
                        "video_id": video_id[cell.outer_train],
                        "row_index": row_index[cell.outer_train],
                    }
                ),
                "primary_seed": primary_seed,
                "secondary_seed": secondary_seed,
                "primary": primary_audit,
                "secondary": secondary_audit,
                "subspace_overlap": stability,
                "separately_refit_widths": sorted(affected_widths),
                "separate_width_audits": fallback_audits,
                "pca_path": pca_path.relative_to(staging).as_posix(),
                "pca_sha256": sha256_file(pca_path),
                "pca_arrays_sha256": _array_bundle_digest(pca_arrays),
            }
        )
        shuffle_source = _split_safe_shuffle_source(cell, video_id, row_index)
        for family in FAMILY_NAMES:
            source_family = "real_cortical" if family == "label_permutation_cortical" else family
            duplicate_real = family == "label_permutation_cortical"
            if duplicate_real:
                real_path = staging / "projection-cache" / cell_id / "real_cortical.npy"
                score_sets = {PCA_MAX_RANK: np.load(real_path, allow_pickle=False)}
                score_sets.update(
                    {
                        width: np.load(
                            staging
                            / "projection-cache"
                            / cell_id
                            / f"real_cortical-width-{width}.npy",
                            allow_pickle=False,
                        )
                        for width in affected_widths
                    }
                )
            else:
                features = _family_features(
                    source_family,
                    raw=raw,
                    cell=cell,
                    video_id=video_id,
                    row_index=row_index,
                    time_seconds=time_seconds,
                    shuffle_source=shuffle_source,
                )
                score_sets = {PCA_MAX_RANK: transform_pca_mlx(primary, features)}
                score_sets.update(
                    {
                        width: transform_pca_mlx(fallback_models[width][0], features)
                        for width in affected_widths
                    }
                )
                if family != "real_cortical":
                    del features
                gc.collect()
            for basis_width, scores in score_sets.items():
                is_fallback = basis_width != PCA_MAX_RANK
                cache_path = (
                    staging
                    / "projection-cache"
                    / cell_id
                    / (f"{family}-width-{basis_width}.npy" if is_fallback else f"{family}.npy")
                )
                _atomic_save_npy(cache_path, scores)
                cache_records.append(
                    {
                        "cell_id": cell_id,
                        "family": family,
                        "basis_width": basis_width,
                        "separate_width_fallback": is_fallback,
                        "path": cache_path.relative_to(staging).as_posix(),
                        "sha256": sha256_file(cache_path),
                        "arrays_sha256": _array_bundle_digest({"scores": scores}),
                        "shape": list(scores.shape),
                        "transform_row_identity_sha256": _array_bundle_digest(
                            {"video_id": video_id, "row_index": row_index}
                        ),
                        "prefix_sha256": {
                            str(width): _array_bundle_digest({"scores": scores[:, :width]})
                            for width in PCA_PREFIX_WIDTHS
                            if width <= scores.shape[1]
                        },
                    }
                )
            del score_sets
            gc.collect()
            mx.clear_cache()

        inner_q90 = float(np.quantile(target[cell.inner_train], 0.90))
        inner_train_labels = (target[cell.inner_train] >= inner_q90).astype(np.int8)
        inner_validation_labels = (target[cell.inner_validation] >= inner_q90).astype(np.int8)
        ar_winner = ar_winners[cell_id]
        inner_ar_model = fit_logistic_mlx(
            build_ar_features(
                arousal,
                cell.inner_train,
                depth=int(ar_winner["lag_depth_rows"]),
            ),
            inner_train_labels,
            regularization=float(ar_winner["regularization"]),
            max_iterations=AR_FINAL_OPTIMIZER_MAX_ITERATIONS,
        )
        if not inner_ar_model.converged:
            raise ValueError(f"inner AR reconstruction did not converge: {cell_id}")
        inner_ar_train = predict_logistic_mlx(
            inner_ar_model,
            build_ar_features(
                arousal,
                cell.inner_train,
                depth=int(ar_winner["lag_depth_rows"]),
            ),
        )
        inner_ar_validation = predict_logistic_mlx(
            inner_ar_model,
            build_ar_features(
                arousal,
                cell.inner_validation,
                depth=int(ar_winner["lag_depth_rows"]),
            ),
        )
        inner_ar_pr_auc = float(
            average_precision_score(inner_validation_labels, inner_ar_validation)
        )
        permuted_inner_train = within_video_label_permutation(
            inner_train_labels,
            video_id,
            cell.inner_train,
            seed=derive_phase03_seed(
                PHASE03_PREDICTION_MANIFEST_SHA256, f"{cell_id}-pca-inner-label-train"
            ),
        )
        permuted_inner_validation = within_video_label_permutation(
            inner_validation_labels,
            video_id,
            cell.inner_validation,
            seed=derive_phase03_seed(
                PHASE03_PREDICTION_MANIFEST_SHA256, f"{cell_id}-pca-inner-label-validation"
            ),
        )
        for depth in TEMPORAL_DEPTHS:
            family_candidate: dict[str, dict[str, float]] = {}
            for family in FAMILY_NAMES:
                for width in PCA_PREFIX_WIDTHS:
                    scores = np.load(
                        _projection_cache_path(staging, cell_id, family, width, affected_widths),
                        mmap_mode="r",
                        allow_pickle=False,
                    )
                    train_features = aggregate_causal_scores(
                        scores[:, :width], cell.inner_train, video_id, depth=depth
                    )
                    validation_features = aggregate_causal_scores(
                        scores[:, :width], cell.inner_validation, video_id, depth=depth
                    )
                    fit_labels = (
                        permuted_inner_train
                        if family == "label_permutation_cortical"
                        else inner_train_labels
                    )
                    validation_labels = (
                        permuted_inner_validation
                        if family == "label_permutation_cortical"
                        else inner_validation_labels
                    )
                    metrics = _score_projected_family(
                        train_features,
                        validation_features,
                        fit_labels,
                        validation_labels,
                        inner_ar_train,
                        inner_ar_validation,
                    )
                    key = f"{depth}:{width}:{family}"
                    family_candidate[key] = metrics
                    inner_family_rows.append(
                        {
                            "cell_id": cell_id,
                            "protocol": cell.protocol,
                            "fold": cell.fold,
                            "seed": cell.seed,
                            "temporal_depth_rows": depth,
                            "width": width,
                            "family": family,
                            "inner_q90_threshold": inner_q90,
                            "inner_ar_pr_auc": inner_ar_pr_auc,
                            **metrics,
                        }
                    )
            for width in PCA_PREFIX_WIDTHS:
                real = family_candidate[f"{depth}:{width}:real_cortical"]
                controls = [
                    family_candidate[f"{depth}:{width}:{family}"]
                    for family in FAMILY_NAMES
                    if family != "real_cortical"
                ]
                strongest_only = max(controls, key=lambda row: row["only_validation_pr_auc"])[
                    "only_validation_pr_auc"
                ]
                strongest_fusion = max(controls, key=lambda row: row["fusion_validation_pr_auc"])[
                    "fusion_validation_pr_auc"
                ]
                candidate_summaries.append(
                    {
                        "cell_id": cell_id,
                        "protocol": cell.protocol,
                        "fold": cell.fold,
                        "seed": cell.seed,
                        "temporal_depth_rows": depth,
                        "width": width,
                        "inner_q90_threshold": inner_q90,
                        "inner_ar_pr_auc": inner_ar_pr_auc,
                        "only_train_pr_auc": real["only_train_pr_auc"],
                        "only_validation_pr_auc": real["only_validation_pr_auc"],
                        "fusion_train_pr_auc": real["fusion_train_pr_auc"],
                        "fusion_validation_pr_auc": real["fusion_validation_pr_auc"],
                        "strongest_only_control_validation_pr_auc": strongest_only,
                        "strongest_fusion_control_validation_pr_auc": strongest_fusion,
                        "only_margin_vs_strongest_control": real["only_validation_pr_auc"]
                        - strongest_only,
                        "fusion_margin_vs_ar": real["fusion_validation_pr_auc"] - inner_ar_pr_auc,
                        "fusion_margin_vs_strongest_control": real["fusion_validation_pr_auc"]
                        - strongest_fusion,
                        "selection_margin": min(
                            real["fusion_validation_pr_auc"] - inner_ar_pr_auc,
                            real["fusion_validation_pr_auc"] - strongest_fusion,
                        ),
                    }
                )
        del primary, secondary, fallback_models
        gc.collect()
        mx.clear_cache()

    aggregate_candidates: list[dict[str, object]] = []
    for depth in TEMPORAL_DEPTHS:
        for width in PCA_PREFIX_WIDTHS:
            rows = [
                row
                for row in candidate_summaries
                if row["temporal_depth_rows"] == depth and row["width"] == width
            ]
            aggregate_candidates.append(
                {
                    "temporal_depth_rows": depth,
                    "width": width,
                    "cells": len(rows),
                    "median_selection_margin": float(
                        np.median([float(row["selection_margin"]) for row in rows])
                    ),
                    "median_real_fusion_pr_auc": float(
                        np.median([float(row["fusion_validation_pr_auc"]) for row in rows])
                    ),
                    "minimum_selection_margin": float(
                        np.min([float(row["selection_margin"]) for row in rows])
                    ),
                }
            )
    selected = min(
        aggregate_candidates,
        key=lambda row: (
            -float(row["median_selection_margin"]),
            -float(row["median_real_fusion_pr_auc"]),
            int(row["width"]),
            int(row["temporal_depth_rows"]),
        ),
    )
    selected_width = int(selected["width"])
    selected_depth = int(selected["temporal_depth_rows"])
    _atomic_write_csv(staging / "inner-family-search.csv", inner_family_rows)
    _atomic_write_csv(staging / "inner-candidate-search.csv", candidate_summaries)
    _atomic_write_json(
        staging / "selected-representation.json",
        {
            "schema": "veatic21_phase04_selected_representation_v1",
            "selected": selected,
            "all_candidates": aggregate_candidates,
            "selection_rows": "inner validation only across all six outer cells",
            "outer_test_opened_for_selection": False,
            "widths": list(PCA_PREFIX_WIDTHS),
            "temporal_depths": list(TEMPORAL_DEPTHS),
        },
    )

    metric_rows: list[dict[str, object]] = []
    per_video_rows: list[dict[str, object]] = []
    delta_records: list[dict[str, object]] = []
    prediction_records: list[dict[str, object]] = []
    final_model_records: list[dict[str, object]] = []
    cell_summaries: list[dict[str, object]] = []

    for cell in cells:
        cell_id = _cell_id(cell)
        ar_model, ar_depth, outer_q90, ar_threshold = _load_ar_model(cell_id)
        train_labels = (target[cell.outer_train] >= outer_q90).astype(np.int8)
        test_labels = (target[cell.outer_test] >= outer_q90).astype(np.int8)
        ar_train = predict_logistic_mlx(
            ar_model, build_ar_features(arousal, cell.outer_train, depth=ar_depth)
        )
        with np.load(
            PHASE02_ROOT / "predictions" / f"{cell_id}.npz", allow_pickle=False
        ) as phase02_predictions:
            frozen_ar_test = phase02_predictions["ar_probability"].astype(np.float64)
            if not (
                np.array_equal(phase02_predictions["video_id"], video_id[cell.outer_test])
                and np.array_equal(phase02_predictions["row_index"], row_index[cell.outer_test])
                and np.array_equal(phase02_predictions["event_label"], test_labels)
                and np.array_equal(
                    phase02_predictions["target_continuous"], target[cell.outer_test]
                )
            ):
                raise ValueError(f"Phase 02 ownership changed: {cell_id}")
        recomputed_ar_test = predict_logistic_mlx(
            ar_model, build_ar_features(arousal, cell.outer_test, depth=ar_depth)
        )
        if not np.array_equal(recomputed_ar_test, frozen_ar_test):
            raise ValueError(f"Phase 02 frozen AR prediction changed: {cell_id}")
        probabilities: dict[str, np.ndarray] = {"frozen_ar": frozen_ar_test}
        thresholds: dict[str, float] = {"frozen_ar": ar_threshold}
        training_pr_auc: dict[str, float] = {
            "frozen_ar": float(average_precision_score(train_labels, ar_train))
        }
        final_arrays: dict[str, np.ndarray] = {}
        permuted_train = within_video_label_permutation(
            train_labels,
            video_id,
            cell.outer_train,
            seed=derive_phase03_seed(
                PHASE03_PREDICTION_MANIFEST_SHA256, f"{cell_id}-pca-final-labels"
            ),
        )
        for family in FAMILY_NAMES:
            scores = np.load(
                _projection_cache_path(
                    staging,
                    cell_id,
                    family,
                    selected_width,
                    fallback_widths_by_cell[cell_id],
                ),
                mmap_mode="r",
                allow_pickle=False,
            )
            train_features = aggregate_causal_scores(
                scores[:, :selected_width],
                cell.outer_train,
                video_id,
                depth=selected_depth,
            )
            test_features = aggregate_causal_scores(
                scores[:, :selected_width],
                cell.outer_test,
                video_id,
                depth=selected_depth,
            )
            fit_labels = permuted_train if family == "label_permutation_cortical" else train_labels
            arrays, family_probabilities, family_thresholds, family_training = _fit_final_family(
                family=family,
                train_features=train_features,
                test_features=test_features,
                fit_labels=fit_labels,
                ar_train=ar_train,
                ar_test=frozen_ar_test,
            )
            final_arrays.update(arrays)
            probabilities.update(family_probabilities)
            thresholds.update(family_thresholds)
            training_pr_auc.update(family_training)

        prediction_arrays: dict[str, np.ndarray] = {
            "video_id": video_id[cell.outer_test],
            "row_index": row_index[cell.outer_test],
            "global_index": cell.outer_test.astype(np.int32),
            "target_continuous": target[cell.outer_test],
            "event_label": test_labels,
            "selected_width": np.full(len(cell.outer_test), selected_width, dtype=np.int16),
            "selected_temporal_depth_rows": np.full(
                len(cell.outer_test), selected_depth, dtype=np.int16
            ),
            **probabilities,
        }
        lane_metrics: dict[str, dict[str, float | int]] = {}
        for lane, scores in probabilities.items():
            metrics = spike_metrics(test_labels, scores, decision_threshold=thresholds[lane])
            lane_metrics[lane] = metrics
            metric_rows.append(
                {
                    "cell_id": cell_id,
                    "protocol": cell.protocol,
                    "fold": cell.fold,
                    "seed": cell.seed,
                    "lane": lane,
                    "selected_width": selected_width,
                    "selected_temporal_depth_rows": selected_depth,
                    "outer_q90_threshold": outer_q90,
                    "training_pr_auc": training_pr_auc[lane],
                    **metrics,
                }
            )
            values = per_video_pr_auc(video_id[cell.outer_test], test_labels, scores)
            for video, value in values.items():
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
        only_controls = [f"{family}_only" for family in FAMILY_NAMES if family != "real_cortical"]
        fusion_controls = [
            f"ar_plus_{family}" for family in FAMILY_NAMES if family != "real_cortical"
        ]
        strongest_only = max(only_controls, key=lambda lane: training_pr_auc[lane])
        strongest_fusion = max(fusion_controls, key=lambda lane: training_pr_auc[lane])
        comparisons = (
            ("real_cortical_only", "frozen_ar", "real_only_vs_ar"),
            ("real_cortical_only", strongest_only, "real_only_vs_strongest_control"),
            ("ar_plus_real_cortical", "frozen_ar", "fusion_vs_ar"),
            (
                "ar_plus_real_cortical",
                strongest_fusion,
                "fusion_vs_strongest_control",
            ),
        )
        cell_deltas: list[dict[str, object]] = []
        for primary, reference, comparison in comparisons:
            bootstrap = paired_video_bootstrap_raw_pr_auc_delta(
                video_id[cell.outer_test],
                test_labels,
                probabilities[primary],
                probabilities[reference],
                seed=derive_phase03_seed(
                    PHASE03_PREDICTION_MANIFEST_SHA256,
                    f"{cell_id}-pca-{comparison}-bootstrap",
                ),
                resamples=CLUSTER_BOOTSTRAP_RESAMPLES,
            )
            record = {
                "cell_id": cell_id,
                "protocol": cell.protocol,
                "fold": cell.fold,
                "seed": cell.seed,
                "comparison": comparison,
                "primary_lane": primary,
                "reference_lane": reference,
                "primary_pr_auc": lane_metrics[primary]["pr_auc"],
                "reference_pr_auc": lane_metrics[reference]["pr_auc"],
                **bootstrap,
            }
            delta_records.append(record)
            cell_deltas.append(record)

        prediction_path = staging / "predictions" / f"{cell_id}.npz"
        _atomic_save_npz(prediction_path, prediction_arrays)
        prediction_records.append(
            {
                "cell_id": cell_id,
                "path": prediction_path.relative_to(staging).as_posix(),
                "sha256": sha256_file(prediction_path),
                "arrays_sha256": _array_bundle_digest(prediction_arrays),
                "rows": len(cell.outer_test),
                "lanes": list(probabilities),
            }
        )
        final_arrays.update(
            {
                "selected_width": np.asarray(selected_width, dtype=np.int16),
                "selected_temporal_depth_rows": np.asarray(selected_depth, dtype=np.int16),
                **{
                    f"{lane}_decision_threshold": np.asarray(value, dtype=np.float64)
                    for lane, value in thresholds.items()
                },
            }
        )
        final_path = staging / "final-models" / f"{cell_id}.npz"
        _atomic_save_npz(final_path, final_arrays)
        final_model_records.append(
            {
                "cell_id": cell_id,
                "path": final_path.relative_to(staging).as_posix(),
                "sha256": sha256_file(final_path),
                "arrays_sha256": _array_bundle_digest(final_arrays),
            }
        )
        cell_summaries.append(
            {
                "cell_id": cell_id,
                "protocol": cell.protocol,
                "fold": cell.fold,
                "seed": cell.seed,
                "strongest_training_owned_only_control": strongest_only,
                "strongest_training_owned_fusion_control": strongest_fusion,
                "frozen_ar_pr_auc": lane_metrics["frozen_ar"]["pr_auc"],
                "real_pca_only_pr_auc": lane_metrics["real_cortical_only"]["pr_auc"],
                "ar_plus_real_pca_pr_auc": lane_metrics["ar_plus_real_cortical"]["pr_auc"],
                "deltas": cell_deltas,
            }
        )

    _atomic_write_csv(staging / "fold-metrics.csv", metric_rows)
    _atomic_write_csv(staging / "per-video-metrics.csv", per_video_rows)
    _atomic_write_json(
        staging / "pca-accuracy-audit.json",
        {"schema": "veatic21_phase04_pca_accuracy_v1", "records": accuracy_records},
    )
    _atomic_write_json(
        staging / "projection-cache-manifest.json",
        {"schema": "veatic21_phase04_projection_cache_v1", "records": cache_records},
    )
    _atomic_write_json(
        staging / "prediction-manifest.json",
        {"schema": "veatic21_phase04_prediction_manifest_v1", "records": prediction_records},
    )
    _atomic_write_json(
        staging / "final-model-manifest.json",
        {"schema": "veatic21_phase04_final_model_manifest_v1", "records": final_model_records},
    )
    _atomic_write_json(
        staging / "primary-deltas.json",
        {"schema": "veatic21_phase04_primary_deltas_v1", "records": delta_records},
    )

    def protocol_values(protocol: str, lane: str) -> list[float]:
        return [
            float(row["pr_auc"])
            for row in metric_rows
            if row["protocol"] == protocol and row["lane"] == lane
        ]

    grouped_ar = protocol_values("grouped_video", "frozen_ar")
    grouped_only = protocol_values("grouped_video", "real_cortical_only")
    grouped_fusion = protocol_values("grouped_video", "ar_plus_real_cortical")
    blocked_ar = protocol_values("blocked_temporal", "frozen_ar")
    blocked_only = protocol_values("blocked_temporal", "real_cortical_only")
    blocked_fusion = protocol_values("blocked_temporal", "ar_plus_real_cortical")
    fusion_comparisons = [
        row
        for row in delta_records
        if row["comparison"] in {"fusion_vs_ar", "fusion_vs_strongest_control"}
    ]
    pca_claim_pass = all(
        float(row["observed_delta"]) > 0.0 and float(row["ci_lower"]) > 0.0
        for row in fusion_comparisons
    )
    summary = {
        "schema": "veatic21_phase04_summary_v1",
        "selected_representation": selected,
        "cells": cell_summaries,
        "grouped_video": {
            "frozen_ar": grouped_ar,
            "real_pca_only": grouped_only,
            "ar_plus_real_pca": grouped_fusion,
            "frozen_ar_median": float(np.median(grouped_ar)),
            "real_pca_only_median": float(np.median(grouped_only)),
            "ar_plus_real_pca_median": float(np.median(grouped_fusion)),
        },
        "blocked_temporal": {
            "frozen_ar": blocked_ar[0],
            "real_pca_only": blocked_only[0],
            "ar_plus_real_pca": blocked_fusion[0],
        },
        "promotion": {
            "linear_pca_fusion_claim_pass": pca_claim_pass,
            "representation_frozen_for_phase05": True,
            "linear_fusion_promoted": pca_claim_pass,
            "rule": (
                "every grouped and blocked fusion delta versus frozen AR and strongest "
                "matched fusion control is positive with whole-video CI lower bound above zero"
            ),
        },
    }
    _atomic_write_json(staging / "summary.json", summary)

    checks = dict.fromkeys(PHASE04_CHECKS, True)
    if not phase05_authorized(checks):
        raise ValueError("Phase 04 mandatory check matrix is incomplete")
    result = {
        "schema": "veatic21_phase04_result_v1",
        "phase": "phase-04-pca-bridge",
        "status": "pass",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "code_sha256": code_sha256,
        "checks": checks,
        "videos": len(EXPECTED_VIDEO_IDS),
        "rows": EXPECTED_ROW_COUNT,
        "row_hz": EXPECTED_ROW_HZ,
        "maximum_rank": PCA_MAX_RANK,
        "candidate_widths": list(PCA_PREFIX_WIDTHS),
        "candidate_temporal_depth_rows": list(TEMPORAL_DEPTHS),
        "selected_width": selected_width,
        "selected_temporal_depth_rows": selected_depth,
        "grouped_frozen_ar_pr_auc_median": float(np.median(grouped_ar)),
        "grouped_real_pca_only_pr_auc_median": float(np.median(grouped_only)),
        "grouped_ar_plus_real_pca_pr_auc_median": float(np.median(grouped_fusion)),
        "blocked_frozen_ar_pr_auc": blocked_ar[0],
        "blocked_real_pca_only_pr_auc": blocked_only[0],
        "blocked_ar_plus_real_pca_pr_auc": blocked_fusion[0],
        "linear_pca_fusion_claim_pass": pca_claim_pass,
        "pca_accuracy_audit_sha256": sha256_file(staging / "pca-accuracy-audit.json"),
        "projection_cache_manifest_sha256": sha256_file(staging / "projection-cache-manifest.json"),
        "prediction_manifest_sha256": sha256_file(staging / "prediction-manifest.json"),
        "final_model_manifest_sha256": sha256_file(staging / "final-model-manifest.json"),
        "selected_representation_sha256": sha256_file(staging / "selected-representation.json"),
        "summary_sha256": sha256_file(staging / "summary.json"),
        "operations": request["operations"],
        "phase05_authorized": True,
        "single_next_authorized_action": (
            "Phase 05 VEATIC-specific learned frozen-AR residual bridge discovery using "
            "the sealed Phase 04 representation and complete controls from the first cell"
        ),
    }
    _atomic_write_json(staging / "result.json", result)

    ledger = {
        "schema": "veatic21_derivation_ledger_v1",
        "phase": "phase-04-pca-bridge",
        "code_sha256": code_sha256,
        "input_hashes": request["input_hashes"],
        "numeric_choices": [
            {
                "choice": "pca_width_candidates",
                "value": list(PCA_PREFIX_WIDTHS),
                "derivation": "master specification default nested prefix set",
                "owned_rows": "corresponding outer-training partition",
            },
            {
                "choice": "temporal_depth_candidates",
                "value": list(TEMPORAL_DEPTHS),
                "derivation": request["frozen_design"]["temporal_derivation"],
                "owned_rows": "sealed Phase 01 label dynamics; frozen before PCA scoring",
            },
            {
                "choice": "selected_representation",
                "value": {"width": selected_width, "temporal_depth_rows": selected_depth},
                "derivation": request["frozen_design"]["representation_selection"],
                "owned_rows": "inner-validation rows only across all six outer cells",
            },
        ],
        "fitted_choices": [selected],
        "outer_test_used_for_selection": False,
        "again_numeric_choices_inherited": False,
        "again_paths_used": False,
    }
    _atomic_write_json(staging / "veatic-derivation-ledger.json", ledger)

    report = f"""# VEATIC 2.1 Phase 04 Fold-Owned PCA Bridge

Status: **PASS**

Phase 04 fit one outer-training-only maximum rank-{PCA_MAX_RANK} PCA basis in each of five
grouped-video cells and one blocked-temporal cell using every owned eligible row. Scaling,
randomized subspace iteration, and eigensolution used float32 MLX operations on `gpu:0` in one
worker. Nested prefixes `{list(PCA_PREFIX_WIDTHS)}` were audited for finiteness,
orthogonality, ordered/cumulative explained variance, reconstruction residual, exact score
checksums, and independent-seed subspace stability.

The VEATIC-derived temporal family crossed current row with causal trailing means at depths
`4` and `6`, derived from the Phase 01 PACF landmark and target width. Complete controls were
evaluated for all width/temporal candidates. A single representation—width
`{selected_width}`, temporal depth `{selected_depth}`—was selected globally using only
control-adjusted inner-validation fusion behavior before any outer PCA prediction was scored.

Grouped median PR-AUC was `{np.median(grouped_ar):.6f}` for frozen AR,
`{np.median(grouped_only):.6f}` for selected real PCA-only, and
`{np.median(grouped_fusion):.6f}` for AR-plus-selected PCA. Blocked PR-AUC was
`{blocked_ar[0]:.6f}`, `{blocked_only[0]:.6f}`, and `{blocked_fusion[0]:.6f}`, respectively.
Linear PCA fusion claim: **{"PASS" if pca_claim_pass else "FAIL"}**. The selected
representation is frozen for the next ordered learned residual question regardless of whether
this linear probe itself clears the claim gate.

No held-out row selected PCA width, temporal context, feature, seed, or model. Exact Phase 02
targets, splits, q90 ownership, and frozen AR predictions were reused. No washout or learned
bridge was fit, no forbidden hidden state or grouped upstream feature was opened, and no AGAIN
runtime input or numeric selection entered the phase.

Code SHA-256: `{code_sha256}`
PCA accuracy audit SHA-256: `{result["pca_accuracy_audit_sha256"]}`
Projection cache manifest SHA-256: `{result["projection_cache_manifest_sha256"]}`
Prediction manifest SHA-256: `{result["prediction_manifest_sha256"]}`
Selected representation SHA-256: `{result["selected_representation_sha256"]}`
Summary SHA-256: `{result["summary_sha256"]}`
"""
    _atomic_write_text(staging / "report.md", report)
    artifact_manifest = {
        "schema": "veatic21_phase04_artifact_manifest_v1",
        "created_at": _utc_now(),
        "root": str(output_root),
        "artifacts": _artifact_inventory(staging),
    }
    _atomic_write_json(staging / "artifact-manifest.json", artifact_manifest)
    checksum_paths = [
        path
        for path in sorted(candidate for candidate in staging.rglob("*") if candidate.is_file())
        if path.name != "checksums.sha256"
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
    staging = PHASE04_ROOT.parent / f".{PHASE04_ROOT.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
