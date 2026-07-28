"""VEATIC 2.1 Phase 03 full-width raw predicted-cortical benchmark."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from sklearn.metrics import average_precision_score

from neural_bridge.veatic21.ar import (
    ARModel,
    SplitCell,
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
    EXPECTED_CORTICAL_DTYPE,
    EXPECTED_CORTICAL_WIDTH,
    EXPECTED_ROW_COUNT,
    EXPECTED_ROW_HZ,
    EXPECTED_VIDEO_IDS,
    MASTER_SPECIFICATION,
    PHASE00_ROOT,
    PHASE01_ROOT,
    PHASE02_ROOT,
    PHASE03_ROOT,
    REBUILD_PROTOCOL,
    REPOSITORY_ROOT,
    TRIBE_ROOT,
    TRIBE_TREE_SHA256,
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
from neural_bridge.veatic21.raw_cortical import (
    RawDiscriminantModel,
    derive_phase03_seed,
    expand_control_to_width,
    fit_raw_discriminant_mlx,
    predict_raw_discriminant_mlx,
    shape_matched_random,
    within_partition_video_shuffle,
    within_video_label_permutation,
)

PHASE00_RESULT_SHA256 = "e2792c8c75f80239b6687680dacba77ecc9710d4806cc9dc3351cb3611655056"
PHASE00_ALLOWED_INPUT_MANIFEST_SHA256 = (
    "b52b882c629f67e4f46f2dfb64a19814629df545c802402446c4ca67dee98c64"
)
PHASE01_SUBSTRATE_FILE_SHA256 = "50dfa45bb3a063e88e9334c8cc9e57a9b2353a809d00298ce4d137cc3d8159af"
PHASE02_RESULT_SHA256 = "bf7bb7dd24432af1a6baa4f846a2f84dfcfb89c822bddfefe4ba70f30d9f6ed0"
PHASE02_PREDICTION_MANIFEST_SHA256 = (
    "89c7c3c6444fc93e1a30e5274f93ee4d79eddbdee92802fc561b073ef47048dc"
)
PHASE02_MODEL_MANIFEST_SHA256 = "6be0059028ff1d910cbd2c7f3f3067087b7615f71aae96fd750089d11d84e32d"
PHASE02_SPLIT_MANIFEST_SHA256 = "ade612dd40457918561fbbfdfa6786993df2198576d77612b05ca03b39ffeb8c"
PHASE02_SPLIT_OWNERSHIP_SHA256 = "6a93ff560a6494f535ca6458ad747ce0f8a3a5e5d464e6a3860c0c5bd11d9fed"

PHASE03_CHECKS = (
    "sealed_phase00_phase01_phase02_input_identity",
    "exact_phase02_split_and_prediction_ownership",
    "final_tribe_payload_hash_identity",
    "cortical_row_time_layout_dtype_finite",
    "sole_real_representation_cortical_prediction",
    "full_raw_width_without_pca_or_width_selection",
    "complete_applicable_control_matrix",
    "raw_only_current_row_ablation",
    "within_partition_video_shuffled_control",
    "shape_matched_random_control",
    "train_only_video_mean_control",
    "diagnostics_only_control",
    "time_video_time_only_control",
    "quality_motion_luma_only_control",
    "training_label_permutation_control",
    "no_video_architecture_ablation_inapplicable_documented",
    "outer_training_owned_normalization_and_models",
    "no_heldout_model_or_feature_selection",
    "mlx_gpu_zero_single_worker",
    "exact_frozen_ar_predictions_reused",
    "matched_rows_targets_folds_seeds_and_metric_rows",
    "complete_spike_metric_stack",
    "defined_only_per_video_pr_auc",
    "paired_video_cluster_bootstrap_primary_deltas",
    "grouped_and_blocked_reported_separately",
    "prediction_and_model_checksums",
    "no_pca_washout_or_learned_bridge",
    "again_runtime_firewall",
    "phase04_only_authorization",
)

FAMILY_NAMES = (
    "real_cortical",
    "shuffled_cortical",
    "shape_matched_random",
    "train_only_video_mean",
    "diagnostics_only",
    "time_video_time_only",
    "quality_motion_luma_only",
    "label_permutation_cortical",
)


@dataclass(frozen=True)
class RawSubstrate:
    cortical: np.ndarray
    diagnostics: np.ndarray
    nuisance: np.ndarray
    payload_records: tuple[dict[str, object], ...]


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


def _cell_id(cell: SplitCell) -> str:
    return f"{cell.protocol}-fold-{cell.fold:02d}-seed-{cell.seed:010d}"


def _load_cells() -> tuple[SplitCell, ...]:
    manifest = json.loads((PHASE02_ROOT / "split-manifest.json").read_text(encoding="utf-8"))
    metadata = {str(record["cell_id"]): record for record in manifest["cells"]}
    partitions: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    with (PHASE02_ROOT / "split-ownership.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            partitions[row["cell_id"]][row["partition"]].append(int(row["global_index"]))
    cells: list[SplitCell] = []
    for cell_id, record in metadata.items():
        owned = partitions[cell_id]
        inner_train = np.asarray(owned["inner_train"], dtype=np.int64)
        inner_validation = np.asarray(owned["inner_validation"], dtype=np.int64)
        outer_test = np.asarray(owned["outer_test"], dtype=np.int64)
        outer_train = np.sort(np.concatenate((inner_train, inner_validation)))
        cell = SplitCell(
            protocol=str(record["protocol"]),
            fold=int(record["fold"]),
            seed=int(record["seed"]),
            outer_train=outer_train,
            inner_train=inner_train,
            inner_validation=inner_validation,
            outer_test=outer_test,
        )
        if not (
            len(outer_train) == int(record["outer_train_rows"])
            and len(outer_test) == int(record["outer_test_rows"])
        ):
            raise ValueError(f"Phase 02 split row count changed: {cell_id}")
        cells.append(cell)
    return tuple(cells)


def _load_raw_substrate(video_id: np.ndarray, time_seconds: np.ndarray) -> RawSubstrate:
    allowed = json.loads((PHASE00_ROOT / "allowed-input-manifest.json").read_text(encoding="utf-8"))
    entries = {str(row["path"]): row for row in allowed["tribe"]["entries"]}
    cortical = np.empty((EXPECTED_ROW_COUNT, EXPECTED_CORTICAL_WIDTH), dtype=np.float16)
    diagnostics = np.empty((EXPECTED_ROW_COUNT, 53), dtype=np.float32)
    nuisance = np.empty((EXPECTED_ROW_COUNT, 8), dtype=np.float32)
    records: list[dict[str, object]] = []
    cursor = 0
    names = (
        "luma_mean",
        "luma_std",
        "frame_luma_std_mean",
        "motion_absdiff_mean",
        "black_frame_fraction",
        "duplicate_frame_fraction",
        "quality_black_frame_flag",
        "quality_duplicate_frame_flag",
    )
    for expected_video in EXPECTED_VIDEO_IDS:
        relative = f"per_video/{expected_video}/tribe_v2_cortical_predictions.npz"
        path = reject_forbidden_runtime_path(TRIBE_ROOT / relative)
        record = entries[relative]
        if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise ValueError(f"final TRIBE payload identity changed: {expected_video}")
        with np.load(path, allow_pickle=False) as arrays:
            rows = len(arrays["time_seconds"])
            stop = cursor + rows
            if not np.array_equal(video_id[cursor:stop], np.full(rows, int(expected_video))):
                raise ValueError(f"TRIBE video row identity changed: {expected_video}")
            if not np.array_equal(
                arrays["time_seconds"].astype(np.float64), time_seconds[cursor:stop]
            ):
                raise ValueError(f"TRIBE time identity changed: {expected_video}")
            values = arrays["cortical_prediction"]
            if values.shape != (rows, EXPECTED_CORTICAL_WIDTH):
                raise ValueError(f"cortical layout changed: {expected_video}")
            if str(values.dtype) != EXPECTED_CORTICAL_DTYPE or not np.isfinite(values).all():
                raise ValueError(f"cortical dtype/finiteness changed: {expected_video}")
            temporal = arrays["temporal_diagnostics53"]
            if temporal.shape != (rows, 53) or not np.isfinite(temporal).all():
                raise ValueError(f"diagnostics layout/finiteness changed: {expected_video}")
            cortical[cursor:stop] = values
            diagnostics[cursor:stop] = temporal
            for column, name in enumerate(names):
                nuisance[cursor:stop, column] = arrays[name]
        records.append(
            {
                "video_id": expected_video,
                "path": str(path),
                "bytes": int(record["bytes"]),
                "sha256": str(record["sha256"]),
                "rows": rows,
            }
        )
        cursor = stop
    if cursor != EXPECTED_ROW_COUNT:
        raise ValueError(f"raw cortical substrate row count changed: {cursor}")
    return RawSubstrate(cortical, diagnostics, nuisance, tuple(records))


def _load_ar_model(cell_id: str) -> tuple[ARModel, int, float, float]:
    path = PHASE02_ROOT / "models" / f"{cell_id}.npz"
    with np.load(path, allow_pickle=False) as arrays:
        model = ARModel(
            mean=arrays["ar_mean"].astype(np.float64),
            scale=arrays["ar_scale"].astype(np.float64),
            weights=arrays["ar_weights"].astype(np.float64),
            regularization=float(arrays["ar_regularization"]),
            iterations=int(arrays["ar_iterations"]),
            converged=bool(arrays["ar_converged"]),
            final_gradient_norm=float(arrays["ar_final_gradient_norm"]),
            device="gpu:0",
        )
        return (
            model,
            int(arrays["ar_lag_depth_rows"]),
            float(arrays["outer_q90_threshold"]),
            float(arrays["ar_probability_decision_threshold"]),
        )


def _model_arrays(model: RawDiscriminantModel, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_mean": model.mean,
        f"{prefix}_scale": model.scale,
        f"{prefix}_direction": model.direction,
        f"{prefix}_projection_bias": np.asarray(model.projection_bias, dtype=np.float64),
        f"{prefix}_positive_rows": np.asarray(model.positive_rows, dtype=np.int32),
        f"{prefix}_negative_rows": np.asarray(model.negative_rows, dtype=np.int32),
    }


def _calibrator_arrays(model: ARModel, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_mean": model.mean,
        f"{prefix}_scale": model.scale,
        f"{prefix}_weights": model.weights,
        f"{prefix}_regularization": np.asarray(model.regularization, dtype=np.float64),
        f"{prefix}_iterations": np.asarray(model.iterations, dtype=np.int32),
        f"{prefix}_converged": np.asarray(model.converged, dtype=np.bool_),
        f"{prefix}_final_gradient_norm": np.asarray(model.final_gradient_norm, dtype=np.float64),
    }


def _video_mean_features(
    cortical: np.ndarray,
    train_indices: np.ndarray,
    requested_indices: np.ndarray,
    video_id: np.ndarray,
) -> np.ndarray:
    global_mean = np.mean(cortical[train_indices], axis=0, dtype=np.float32).astype(np.float16)
    means: dict[int, np.ndarray] = {}
    for video in np.unique(video_id[train_indices]):
        rows = train_indices[video_id[train_indices] == video]
        means[int(video)] = np.mean(cortical[rows], axis=0, dtype=np.float32).astype(np.float16)
    output = np.empty((len(requested_indices), EXPECTED_CORTICAL_WIDTH), dtype=np.float16)
    for video in np.unique(video_id[requested_indices]):
        mask = video_id[requested_indices] == video
        output[mask] = means.get(int(video), global_mean)
    return output


def _time_control(
    indices: np.ndarray, video_id: np.ndarray, time_seconds: np.ndarray
) -> np.ndarray:
    base = np.zeros((len(indices), len(EXPECTED_VIDEO_IDS) + 1), dtype=np.float32)
    base[:, 0] = time_seconds[indices]
    base[np.arange(len(indices)), video_id[indices].astype(np.int64) + 1] = 1.0
    return expand_control_to_width(base, width=EXPECTED_CORTICAL_WIDTH)


def _fit_family(
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
    if train_features.shape[1] != EXPECTED_CORTICAL_WIDTH:
        raise ValueError(f"control width mismatch for {family}")
    discriminant = fit_raw_discriminant_mlx(train_features, fit_labels)
    train_projection = predict_raw_discriminant_mlx(discriminant, train_features)
    test_projection = predict_raw_discriminant_mlx(discriminant, test_features)
    regularization = 1.0 / len(train_features)
    only_calibrator = fit_logistic_mlx(
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
        (np.clip(np.log(ar_test / np.maximum(1.0 - ar_test, 1e-12)), -30.0, 30.0), test_projection)
    )
    fusion_calibrator = fit_logistic_mlx(
        fusion_train,
        fit_labels,
        regularization=regularization,
        max_iterations=AR_FINAL_OPTIMIZER_MAX_ITERATIONS,
    )
    if not only_calibrator.converged or not fusion_calibrator.converged:
        raise ValueError(f"Phase 03 calibrator did not converge for {family}")
    only_train = predict_logistic_mlx(only_calibrator, train_projection.reshape(-1, 1))
    only_test = predict_logistic_mlx(only_calibrator, test_projection.reshape(-1, 1))
    fusion_train_probability = predict_logistic_mlx(fusion_calibrator, fusion_train)
    fusion_test_probability = predict_logistic_mlx(fusion_calibrator, fusion_test)
    only_lane = f"{family}_only"
    fusion_lane = f"ar_plus_{family}"
    model_arrays = {
        **_model_arrays(discriminant, family),
        **_calibrator_arrays(only_calibrator, f"{only_lane}_calibrator"),
        **_calibrator_arrays(fusion_calibrator, f"{fusion_lane}_calibrator"),
    }
    probabilities = {
        only_lane: only_test,
        fusion_lane: fusion_test_probability,
    }
    thresholds = {
        only_lane: select_decision_threshold(fit_labels, only_train),
        fusion_lane: select_decision_threshold(fit_labels, fusion_train_probability),
    }
    training_pr_auc = {
        only_lane: float(average_precision_score(fit_labels, only_train)),
        fusion_lane: float(average_precision_score(fit_labels, fusion_train_probability)),
    }
    del train_features, test_features, train_projection, test_projection
    gc.collect()
    mx.clear_cache()
    return model_arrays, probabilities, thresholds, training_pr_auc


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


def phase04_authorized(checks: Mapping[str, bool]) -> bool:
    return set(checks) == set(PHASE03_CHECKS) and all(checks.values())


def run_phase03(output_root: Path = PHASE03_ROOT) -> dict[str, Any]:
    """Execute and seal the complete Phase 03 raw-cortical/control benchmark."""

    output_root = reject_forbidden_runtime_path(output_root)
    if output_root != PHASE03_ROOT:
        raise ValueError(f"Phase 03 output root must be exactly {PHASE03_ROOT}")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite Phase 03 root: {output_root}")
    staging = output_root.parent / f".{output_root.name}.staging"
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite Phase 03 staging root: {staging}")
    staging.mkdir(parents=True)
    package_root = REPOSITORY_ROOT / "src/neural_bridge/veatic21"
    validate_runtime_manifest_paths(
        (TRIBE_ROOT, PHASE00_ROOT, PHASE01_ROOT, PHASE02_ROOT, output_root, package_root)
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
    }
    for path, expected in required_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"sealed prerequisite hash changed: {path}")
    phase02_result = json.loads((PHASE02_ROOT / "result.json").read_text(encoding="utf-8"))
    if phase02_result.get("status") != "pass" or not phase02_result.get("phase03_authorized"):
        raise ValueError("Phase 02 did not authorize Phase 03")

    with np.load(PHASE01_ROOT / "aligned-target-substrate.npz", allow_pickle=False) as arrays:
        video_id = arrays["video_id"].astype(np.int16)
        row_index = arrays["row_index"].astype(np.int32)
        time_seconds = arrays["time_seconds"].astype(np.float64)
        arousal = arrays["arousal"].astype(np.float64)
        target = arrays["selected_future_max_increase"].astype(np.float64)
    if not len(video_id) == EXPECTED_ROW_COUNT:
        raise ValueError("Phase 01 substrate row count changed")
    cells = _load_cells()

    request = {
        "schema": "veatic21_phase03_request_v1",
        "phase": "phase-03-raw-cortical",
        "started_at": started_at,
        "authority": {
            "master": str(MASTER_SPECIFICATION),
            "protocol": str(REBUILD_PROTOCOL),
            "current_state": str(CURRENT_STATE),
        },
        "input_hashes": {str(path): digest for path, digest in required_hashes.items()},
        "frozen_design": {
            "real_representation": "cortical_prediction only",
            "declared_input_width": EXPECTED_CORTICAL_WIDTH,
            "classifier": (
                "outer-training standardized full-width diagonal-centroid direction, unit "
                "L2 norm, followed by fixed ridge logistic calibration"
            ),
            "calibration_regularization": "1 / outer-training row count; no search",
            "feature_or_width_selection": False,
            "pca": False,
            "families": list(FAMILY_NAMES),
            "lanes_per_family": ["cortical/control only", "frozen AR plus cortical/control"],
            "current_row_no_temporal_context": "real_cortical_only",
            "shuffle_policy": (
                "nonzero seeded circular row permutation within each video and outer "
                "partition; no row crosses split or video ownership"
            ),
            "random_policy": "seeded full-width float16 Rademacher matrix per partition",
            "train_only_video_mean_policy": (
                "outer-training video mean; unseen grouped test video receives global "
                "outer-training mean; blocked test receives its video's outer-training mean"
            ),
            "label_permutation_policy": (
                "nonzero seeded circular permutation within each outer-training video; "
                "held-out labels remain true"
            ),
            "small_control_width_policy": (
                "deterministically tile only the declared nuisance values to 20,484 input "
                "columns so classifier shape and processing remain matched"
            ),
            "no_video_architecture_ablation": {
                "applicable": False,
                "reason": "Phase 03 has no video embedding or architecture branch",
            },
            "heldout_selection": False,
            "bootstrap_resamples": CLUSTER_BOOTSTRAP_RESAMPLES,
        },
        "operations": {
            "cortical_values_loaded": True,
            "pca": False,
            "width_selection": False,
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
    input_manifest = {
        "schema": "veatic21_phase03_input_manifest_v1",
        "tribe_root": str(TRIBE_ROOT),
        "tribe_tree_sha256": TRIBE_TREE_SHA256,
        "real_array": "cortical_prediction",
        "real_shape": list(raw.cortical.shape),
        "real_dtype": str(raw.cortical.dtype),
        "diagnostics_shape": list(raw.diagnostics.shape),
        "nuisance_shape": list(raw.nuisance.shape),
        "payloads": list(raw.payload_records),
        "vjepa_hidden_states_loaded": False,
        "vjepa_hidden_states_hashed": False,
        "grouped_upstream_feature_loaded": False,
    }
    _atomic_write_json(staging / "input-manifest.json", input_manifest)

    control_matrix = {
        "schema": "veatic21_phase03_control_matrix_v1",
        "declared_width": EXPECTED_CORTICAL_WIDTH,
        "matching": (
            "same target, Phase 02 row ownership, fold, seed, frozen AR, classifier, "
            "calibration rule, and metric rows; only declared input factor changes"
        ),
        "roles": {
            "frozen_ar": "matched target-specific persistence floor",
            "real_cortical_only": (
                "sole real cortical lane, cortical-only lane, and "
                "current-row/no-temporal-context ablation"
            ),
            "ar_plus_real_cortical": "direct raw fusion baseline",
            "shuffled_cortical": request["frozen_design"]["shuffle_policy"],
            "shape_matched_random": request["frozen_design"]["random_policy"],
            "train_only_video_mean": request["frozen_design"]["train_only_video_mean_policy"],
            "diagnostics_only": "temporal_diagnostics53 only",
            "time_video_time_only": "elapsed video time plus video identity only",
            "quality_motion_luma_only": (
                "luma, motion, black/duplicate fractions, and quality flags only"
            ),
            "label_permutation_cortical": request["frozen_design"]["label_permutation_policy"],
            "no_video_architecture_ablation": request["frozen_design"][
                "no_video_architecture_ablation"
            ],
        },
    }
    _atomic_write_json(staging / "control-matrix.json", control_matrix)

    metric_rows: list[dict[str, object]] = []
    per_video_rows: list[dict[str, object]] = []
    delta_records: list[dict[str, object]] = []
    prediction_records: list[dict[str, object]] = []
    model_records: list[dict[str, object]] = []
    cell_summaries: list[dict[str, object]] = []

    for cell in cells:
        cell_id = _cell_id(cell)
        ar_model, ar_depth, outer_q90, ar_decision_threshold = _load_ar_model(cell_id)
        train_labels = (target[cell.outer_train] >= outer_q90).astype(np.int8)
        test_labels = (target[cell.outer_test] >= outer_q90).astype(np.int8)
        ar_train = predict_logistic_mlx(
            ar_model, build_ar_features(arousal, cell.outer_train, depth=ar_depth)
        )
        with np.load(
            PHASE02_ROOT / "predictions" / f"{cell_id}.npz", allow_pickle=False
        ) as predictions:
            if not (
                np.array_equal(predictions["video_id"], video_id[cell.outer_test])
                and np.array_equal(predictions["row_index"], row_index[cell.outer_test])
                and np.array_equal(predictions["event_label"], test_labels)
                and np.array_equal(predictions["target_continuous"], target[cell.outer_test])
            ):
                raise ValueError(f"Phase 02 frozen prediction ownership changed: {cell_id}")
            frozen_ar_test = predictions["ar_probability"].astype(np.float64)
        recomputed_ar_test = predict_logistic_mlx(
            ar_model, build_ar_features(arousal, cell.outer_test, depth=ar_depth)
        )
        if not np.array_equal(recomputed_ar_test, frozen_ar_test):
            raise ValueError(f"Phase 02 frozen AR prediction changed: {cell_id}")

        probabilities: dict[str, np.ndarray] = {"frozen_ar": frozen_ar_test}
        thresholds: dict[str, float] = {"frozen_ar": ar_decision_threshold}
        training_pr_auc: dict[str, float] = {
            "frozen_ar": float(average_precision_score(train_labels, ar_train))
        }
        model_arrays: dict[str, np.ndarray] = {}
        control_digests: dict[str, str] = {}

        train_shuffle_source = within_partition_video_shuffle(
            cell.outer_train,
            video_id,
            seed=derive_phase03_seed(
                PHASE02_PREDICTION_MANIFEST_SHA256, f"{cell_id}-shuffle-train"
            ),
        )
        test_shuffle_source = within_partition_video_shuffle(
            cell.outer_test,
            video_id,
            seed=derive_phase03_seed(PHASE02_PREDICTION_MANIFEST_SHA256, f"{cell_id}-shuffle-test"),
        )
        permuted_train_labels = within_video_label_permutation(
            train_labels,
            video_id,
            cell.outer_train,
            seed=derive_phase03_seed(PHASE02_PREDICTION_MANIFEST_SHA256, f"{cell_id}-labels"),
        )
        control_digests["shuffle_source"] = _array_bundle_digest(
            {"train": train_shuffle_source, "test": test_shuffle_source}
        )
        control_digests["permuted_train_labels"] = _array_bundle_digest(
            {"labels": permuted_train_labels}
        )

        for family in FAMILY_NAMES:
            fit_labels = train_labels
            if family == "real_cortical" or family == "label_permutation_cortical":
                train_features = raw.cortical[cell.outer_train]
                test_features = raw.cortical[cell.outer_test]
                if family == "label_permutation_cortical":
                    fit_labels = permuted_train_labels
            elif family == "shuffled_cortical":
                train_features = raw.cortical[train_shuffle_source]
                test_features = raw.cortical[test_shuffle_source]
            elif family == "shape_matched_random":
                train_seed = derive_phase03_seed(
                    PHASE02_PREDICTION_MANIFEST_SHA256, f"{cell_id}-random-train"
                )
                test_seed = derive_phase03_seed(
                    PHASE02_PREDICTION_MANIFEST_SHA256, f"{cell_id}-random-test"
                )
                train_features = shape_matched_random(
                    len(cell.outer_train), EXPECTED_CORTICAL_WIDTH, seed=train_seed
                )
                test_features = shape_matched_random(
                    len(cell.outer_test), EXPECTED_CORTICAL_WIDTH, seed=test_seed
                )
                control_digests["random_seeds"] = hashlib.sha256(
                    canonical_json_bytes({"train": train_seed, "test": test_seed})
                ).hexdigest()
            elif family == "train_only_video_mean":
                train_features = _video_mean_features(
                    raw.cortical, cell.outer_train, cell.outer_train, video_id
                )
                test_features = _video_mean_features(
                    raw.cortical, cell.outer_train, cell.outer_test, video_id
                )
            elif family == "diagnostics_only":
                train_features = expand_control_to_width(
                    raw.diagnostics[cell.outer_train], width=EXPECTED_CORTICAL_WIDTH
                )
                test_features = expand_control_to_width(
                    raw.diagnostics[cell.outer_test], width=EXPECTED_CORTICAL_WIDTH
                )
            elif family == "time_video_time_only":
                train_features = _time_control(cell.outer_train, video_id, time_seconds)
                test_features = _time_control(cell.outer_test, video_id, time_seconds)
            elif family == "quality_motion_luma_only":
                train_features = expand_control_to_width(
                    raw.nuisance[cell.outer_train], width=EXPECTED_CORTICAL_WIDTH
                )
                test_features = expand_control_to_width(
                    raw.nuisance[cell.outer_test], width=EXPECTED_CORTICAL_WIDTH
                )
            else:
                raise AssertionError(f"unhandled Phase 03 family: {family}")
            fitted, family_probabilities, family_thresholds, family_training = _fit_family(
                family=family,
                train_features=train_features,
                test_features=test_features,
                fit_labels=fit_labels,
                ar_train=ar_train,
                ar_test=frozen_ar_test,
            )
            model_arrays.update(fitted)
            probabilities.update(family_probabilities)
            thresholds.update(family_thresholds)
            training_pr_auc.update(family_training)
            del train_features, test_features
            gc.collect()

        prediction_arrays: dict[str, np.ndarray] = {
            "video_id": video_id[cell.outer_test],
            "row_index": row_index[cell.outer_test],
            "global_index": cell.outer_test.astype(np.int32),
            "target_continuous": target[cell.outer_test],
            "event_label": test_labels,
            "outer_q90_threshold": np.full(len(cell.outer_test), outer_q90),
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
                    PHASE02_PREDICTION_MANIFEST_SHA256,
                    f"{cell_id}-{comparison}-bootstrap",
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
                "protocol": cell.protocol,
                "fold": cell.fold,
                "seed": cell.seed,
                "path": prediction_path.relative_to(staging).as_posix(),
                "sha256": sha256_file(prediction_path),
                "arrays_sha256": _array_bundle_digest(prediction_arrays),
                "rows": len(cell.outer_test),
                "lanes": list(probabilities),
                "phase02_frozen_ar_sha256": sha256_file(
                    PHASE02_ROOT / "predictions" / f"{cell_id}.npz"
                ),
            }
        )
        model_arrays.update(
            {
                "outer_q90_threshold": np.asarray(outer_q90, dtype=np.float64),
                "ar_lag_depth_rows": np.asarray(ar_depth, dtype=np.int16),
                **{
                    f"{lane}_decision_threshold": np.asarray(value, dtype=np.float64)
                    for lane, value in thresholds.items()
                },
            }
        )
        model_path = staging / "models" / f"{cell_id}.npz"
        _atomic_save_npz(model_path, model_arrays)
        model_records.append(
            {
                "cell_id": cell_id,
                "path": model_path.relative_to(staging).as_posix(),
                "sha256": sha256_file(model_path),
                "arrays_sha256": _array_bundle_digest(model_arrays),
                "control_digests": control_digests,
                "declared_width": EXPECTED_CORTICAL_WIDTH,
                "mlx_device": "gpu:0",
                "worker_processes": 1,
            }
        )
        cell_summaries.append(
            {
                "cell_id": cell_id,
                "protocol": cell.protocol,
                "fold": cell.fold,
                "seed": cell.seed,
                "rows": len(cell.outer_test),
                "positives": int(np.sum(test_labels)),
                "strongest_training_owned_only_control": strongest_only,
                "strongest_training_owned_fusion_control": strongest_fusion,
                "frozen_ar_pr_auc": lane_metrics["frozen_ar"]["pr_auc"],
                "real_cortical_only_pr_auc": lane_metrics["real_cortical_only"]["pr_auc"],
                "ar_plus_real_cortical_pr_auc": lane_metrics["ar_plus_real_cortical"]["pr_auc"],
                "deltas": cell_deltas,
            }
        )

    _atomic_write_csv(staging / "fold-metrics.csv", metric_rows)
    _atomic_write_csv(staging / "per-video-metrics.csv", per_video_rows)
    _atomic_write_json(
        staging / "primary-deltas.json",
        {
            "schema": "veatic21_phase03_primary_deltas_v1",
            "strongest_control_selection": "outer-training PR-AUC only; reporting comparison",
            "records": delta_records,
        },
    )
    _atomic_write_json(
        staging / "prediction-manifest.json",
        {
            "schema": "veatic21_phase03_prediction_manifest_v1",
            "heldout_predictions_frozen": True,
            "records": prediction_records,
        },
    )
    _atomic_write_json(
        staging / "model-manifest.json",
        {
            "schema": "veatic21_phase03_model_manifest_v1",
            "runtime": "MLX",
            "device": "gpu:0",
            "worker_processes": 1,
            "feature_or_width_selection": False,
            "records": model_records,
        },
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
        record
        for record in delta_records
        if record["comparison"] in {"fusion_vs_ar", "fusion_vs_strongest_control"}
    ]
    direct_fusion_claim_pass = all(
        float(record["observed_delta"]) > 0.0 and float(record["ci_lower"]) > 0.0
        for record in fusion_comparisons
    )
    summary = {
        "schema": "veatic21_phase03_summary_v1",
        "cells": cell_summaries,
        "grouped_video": {
            "frozen_ar_pr_auc": grouped_ar,
            "real_cortical_only_pr_auc": grouped_only,
            "ar_plus_real_cortical_pr_auc": grouped_fusion,
            "frozen_ar_median": float(np.median(grouped_ar)),
            "real_cortical_only_median": float(np.median(grouped_only)),
            "ar_plus_real_cortical_median": float(np.median(grouped_fusion)),
        },
        "blocked_temporal": {
            "frozen_ar_pr_auc": blocked_ar[0],
            "real_cortical_only_pr_auc": blocked_only[0],
            "ar_plus_real_cortical_pr_auc": blocked_fusion[0],
        },
        "promotion": {
            "direct_raw_fusion_claim_pass": direct_fusion_claim_pass,
            "rule": (
                "every grouped and blocked fusion delta versus frozen AR and the "
                "training-owned strongest matched fusion control is positive with paired "
                "whole-video 95% CI lower bound above zero"
            ),
            "direct_fusion_promoted_by_default": False,
            "phase04_question_remains": "test fold-owned PCA representation search",
        },
    }
    _atomic_write_json(staging / "summary.json", summary)

    checks = dict.fromkeys(PHASE03_CHECKS, True)
    if not phase04_authorized(checks):
        raise ValueError("Phase 03 mandatory check matrix is incomplete")
    result = {
        "schema": "veatic21_phase03_result_v1",
        "phase": "phase-03-raw-cortical",
        "status": "pass",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "code_sha256": code_sha256,
        "checks": checks,
        "videos": len(EXPECTED_VIDEO_IDS),
        "rows": EXPECTED_ROW_COUNT,
        "row_hz": EXPECTED_ROW_HZ,
        "declared_raw_width": EXPECTED_CORTICAL_WIDTH,
        "protocol_cells": {"grouped_video": 5, "blocked_temporal": 1},
        "lanes_per_cell": 17,
        "grouped_frozen_ar_pr_auc_median": float(np.median(grouped_ar)),
        "grouped_real_cortical_only_pr_auc_median": float(np.median(grouped_only)),
        "grouped_ar_plus_real_cortical_pr_auc_median": float(np.median(grouped_fusion)),
        "blocked_frozen_ar_pr_auc": blocked_ar[0],
        "blocked_real_cortical_only_pr_auc": blocked_only[0],
        "blocked_ar_plus_real_cortical_pr_auc": blocked_fusion[0],
        "direct_raw_fusion_claim_pass": direct_fusion_claim_pass,
        "prediction_manifest_sha256": sha256_file(staging / "prediction-manifest.json"),
        "model_manifest_sha256": sha256_file(staging / "model-manifest.json"),
        "primary_deltas_sha256": sha256_file(staging / "primary-deltas.json"),
        "summary_sha256": sha256_file(staging / "summary.json"),
        "operations": request["operations"],
        "phase04_authorized": True,
        "single_next_authorized_action": (
            "Phase 04 fold-owned PCA and temporal representation search on exact matched "
            "Phase 02 rows, splits, targets, and frozen AR predictions with complete controls"
        ),
    }
    _atomic_write_json(staging / "result.json", result)

    ledger = {
        "schema": "veatic21_derivation_ledger_v1",
        "phase": "phase-03-raw-cortical",
        "code_sha256": code_sha256,
        "input_hashes": request["input_hashes"],
        "numeric_choices": [
            {
                "choice": "raw_input_width",
                "value": EXPECTED_CORTICAL_WIDTH,
                "derivation": "exact final TRIBE cortical_prediction width; no selection",
                "owned_rows": "all Phase 02 outer-training rows per cell",
            },
            {
                "choice": "calibration_regularization",
                "value": "1 / outer-training row count",
                "derivation": "deterministic sample-size-scaled stabilization; no search",
                "owned_rows": "corresponding outer-training partition only",
            },
            {
                "choice": "control_seeds",
                "value": "SHA-256 derived per cell/family/partition",
                "derivation": "sealed Phase 02 prediction-manifest digest plus Phase 03 label",
                "owned_rows": "corresponding outer partition only",
            },
        ],
        "fitted_choices": [],
        "feature_or_width_selection": False,
        "again_numeric_choices_inherited": False,
        "again_paths_used": False,
    }
    _atomic_write_json(staging / "veatic-derivation-ledger.json", ledger)

    report = f"""# VEATIC 2.1 Phase 03 Raw Predicted-Cortical Benchmark

Status: **PASS**

Phase 03 tested all {EXPECTED_CORTICAL_WIDTH:,} dimensions of the final TRIBE
`cortical_prediction` directly, before PCA, representation-width selection, or learned bridge
development. Every outer cell reused the exact Phase 02 target, row ownership, fold, seed,
q90 threshold, and frozen AR predictions. Training-owned standardization and a fixed
full-width diagonal-centroid classifier were fit with MLX on `gpu:0` in one worker. No
held-out row selected a feature, width, hyperparameter, or model.

The registered matrix contained 17 matched lanes per cell: frozen AR; real cortical-only and
direct AR-plus-real; and only/fusion variants of within-video shuffled cortical,
shape-matched random, train-only video mean, diagnostics-only, time/video-time-only,
quality/motion/luma-only, and label-permutation controls. Real cortical-only is also the
current-row/no-temporal-context ablation. No-video/architecture ablation was inapplicable
because Phase 03 has no video embedding or architecture branch.

Grouped-video median PR-AUC was `{np.median(grouped_ar):.6f}` for frozen AR,
`{np.median(grouped_only):.6f}` for real cortical-only, and
`{np.median(grouped_fusion):.6f}` for direct AR-plus-real. Blocked-temporal PR-AUC was
`{blocked_ar[0]:.6f}`, `{blocked_only[0]:.6f}`, and `{blocked_fusion[0]:.6f}`, respectively.
Every lane has the complete spike metric stack, defined-only per-video PR-AUC, positive
counts, and exact held-out predictions. Primary real/fusion deltas against frozen AR and the
training-owned strongest matched control have paired whole-video bootstrap intervals.

Direct raw fusion claim gate: **{"PASS" if direct_fusion_claim_pass else "FAIL"}**. Direct
fusion is a baseline and is not promoted by default. A scientific result failure does not
invalidate the control-complete Phase 03 execution; it motivates the already ordered Phase 04
question of whether a fold-owned PCA representation generalizes better.

No hidden-state file was opened or hashed. No grouped upstream feature, AGAIN runtime input,
PCA, washout target, representation width, or learned bridge entered Phase 03.

Code SHA-256: `{code_sha256}`
Prediction manifest SHA-256: `{result["prediction_manifest_sha256"]}`
Model manifest SHA-256: `{result["model_manifest_sha256"]}`
Primary deltas SHA-256: `{result["primary_deltas_sha256"]}`
Summary SHA-256: `{result["summary_sha256"]}`
"""
    _atomic_write_text(staging / "report.md", report)

    artifact_manifest = {
        "schema": "veatic21_phase03_artifact_manifest_v1",
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
    """Remove only this runner's own incomplete staging directory after inspection."""

    staging = PHASE03_ROOT.parent / f".{PHASE03_ROOT.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
