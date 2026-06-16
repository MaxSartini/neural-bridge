"""Benchmark cached VEATIC TRIBE temporal features against affect traces."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.subcortical_roi_adapter import SubcorticalRoiAdapter  # noqa: E402

RUN_MODES = ("cortical_fast_default", "full_research", "subcortical_ablation")
FEATURE_MODES = (
    "cortical_global",
    "cortical_global_delta",
    "cortical_pca_64",
    "cortical_pca_128",
    "cortical_pca64_delta",
    "cortical_raw_ridge",
)
TARGETS = ("valence", "arousal")
DERIVED_TARGETS = (
    "raw",
    "delta_prev_1s",
    "residual_after_persistence",
    "future_state_p1s",
    "future_state_p2s",
    "future_state_p3s",
    "future_change_p1s",
    "future_change_p2s",
    "future_change_p3s",
    "residual_future_p1s_persistence",
    "residual_future_p2s_persistence",
    "residual_future_p3s_persistence",
    "residual_future_p1s_rolling3",
    "residual_future_p2s_rolling3",
    "residual_future_p3s_rolling3",
    "residual_future_p1s_time_only",
    "residual_future_p2s_time_only",
    "residual_future_p3s_time_only",
    "event_future_spike_1_3s",
    "event_future_drop_1_3s",
    "event_future_rise_1_3s",
    "event_trend_reversal_1_3s",
    "event_peak_onset_1_3s",
    "event_recovery_onset_1_3s",
)
EVENT_THRESHOLD = 0.05
FEATURE_SETS_BY_RUN_MODE = {
    "cortical_fast_default": ("cortical_global",),
    "full_research": ("cortical_global", "subcortical_roi", "combined"),
    "subcortical_ablation": ("cortical_global", "subcortical_roi", "combined"),
}
AR_FEATURE_NAMES = (
    "ar_current_or_previous",
    "ar_lag_1s",
    "ar_lag_2s",
    "ar_lag_3s",
    "ar_history_mean",
    "ar_history_std",
    "ar_recent_slope",
    "ar_momentum",
    "ar_history_min",
    "ar_history_max",
    "ar_seconds",
    "ar_video_progress",
    "ar_video_progress_squared",
    "ar_video_progress_sin",
    "ar_video_progress_cos",
)
CORTICAL_FEATURE_NAMES = (
    "cortical_mean",
    "cortical_mean_abs",
    "cortical_std",
    "cortical_peak_abs",
    "cortical_p95_abs",
    "cortical_positive_fraction",
)
TEMPORAL_PREFIXES = ("delta1", "accel", "rollmean3", "slope3", "slope5")
PCA_BACKEND = os.environ.get("VEATIC_PCA_BACKEND", "auto")
RIDGE_BACKEND = os.environ.get("VEATIC_RIDGE_BACKEND", "auto")
RIDGE_MPS_MIN_FEATURES = int(os.environ.get("VEATIC_RIDGE_MPS_MIN_FEATURES", "2048"))
DIAGNOSTIC_TARGETS = (
    "arousal__future_change_p1s",
    "arousal__residual_future_p1s_persistence",
    "arousal__event_future_spike_1_3s",
    "valence__future_change_p1s",
    "valence__residual_future_p1s_persistence",
    "valence__event_future_drop_1_3s",
)
DIAGNOSTIC_CONDITIONS = (
    "autoregressive_plus_cortical_global",
    "autoregressive_plus_subcortical_roi",
    "autoregressive_plus_combined",
    "residualized_autoregressive_plus_cortical_global",
    "residualized_autoregressive_plus_subcortical_roi",
    "residualized_autoregressive_plus_combined",
)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rows_by_video(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["video_id"])].append(row)
    for video_rows in grouped.values():
        video_rows.sort(key=lambda item: int(item["frame_index"]))
    return dict(grouped)


def load_cached_video_features(
    cache_dir: Path,
    video_id: str,
    expected_rows: int,
    *,
    include_subcortical: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw_path = cache_dir / video_id / "tribe_raw_output.npz"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    with np.load(raw_path) as bundle:
        cortical = np.asarray(bundle["predictions"], dtype=np.float32)
        subcortical = (
            np.asarray(bundle["subcortical_predictions"], dtype=np.float32)
            if include_subcortical and "subcortical_predictions" in bundle.files
            else None
        )
    if cortical.shape[0] != expected_rows:
        cortical = resample_rows(cortical, expected_rows)
        cortical_alignment = "linear_resampled"
    else:
        cortical_alignment = "exact"

    features = {
        "cortical_global": cortical_global_features(cortical),
        "cortical_raw": cortical,
    }
    if subcortical is not None:
        if subcortical.shape[0] != expected_rows:
            subcortical = resample_rows(subcortical, expected_rows)
            subcortical_alignment = "linear_resampled"
        else:
            subcortical_alignment = "exact"
        features["subcortical_roi"] = subcortical_roi_features(subcortical)
        features["combined"] = np.concatenate([features["cortical_global"], features["subcortical_roi"]], axis=1)
    else:
        subcortical_alignment = "disabled_by_run_mode" if not include_subcortical else "missing"
        features["subcortical_roi"] = np.zeros((expected_rows, 0), dtype=np.float32)
        features["combined"] = features["cortical_global"]
    metadata = {
        "video_id": video_id,
        "raw_path": str(raw_path),
        "expected_rows": expected_rows,
        "cortical_alignment": cortical_alignment,
        "subcortical_alignment": subcortical_alignment,
        "feature_counts": {key: int(value.shape[1]) for key, value in features.items()},
    }
    return features, metadata


def resample_rows(values: np.ndarray, expected_rows: int) -> np.ndarray:
    if values.shape[0] == expected_rows:
        return values
    if values.shape[0] == 0:
        raise ValueError("Cannot resample empty feature matrix")
    source_x = np.linspace(0.0, 1.0, values.shape[0])
    target_x = np.linspace(0.0, 1.0, expected_rows)
    columns = [np.interp(target_x, source_x, values[:, index]) for index in range(values.shape[1])]
    return np.stack(columns, axis=1).astype(np.float32)


def cortical_global_features(cortical: np.ndarray) -> np.ndarray:
    abs_values = np.abs(cortical)
    return np.stack(
        [
            cortical.mean(axis=1),
            abs_values.mean(axis=1),
            cortical.std(axis=1),
            abs_values.max(axis=1),
            np.percentile(abs_values, 95, axis=1),
            (cortical > 0).mean(axis=1),
        ],
        axis=1,
    ).astype(np.float32)


def slope_window(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.zeros(values.shape[1], dtype=np.float32)
    x = np.arange(values.shape[0], dtype=np.float32)
    x = x - x.mean()
    denom = float(np.sum(x * x))
    if denom < 1e-8:
        return np.zeros(values.shape[1], dtype=np.float32)
    centered = values - values.mean(axis=0, keepdims=True)
    return (x[:, None] * centered).sum(axis=0).astype(np.float32) / denom


def temporal_dynamics_features(
    rows: list[dict[str, Any]],
    matrix: np.ndarray,
    *,
    include_base: bool = True,
) -> np.ndarray:
    """Create within-video dynamics without crossing stimulus boundaries."""
    output = []
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["video_id"])].append(index)
    dynamics = np.zeros((matrix.shape[0], matrix.shape[1] * len(TEMPORAL_PREFIXES)), dtype=np.float32)
    for indices in grouped.values():
        indices.sort(key=lambda item: float(rows[item]["time_start_seconds"]))
        video_values = matrix[np.asarray(indices, dtype=np.int64)].astype(np.float32, copy=False)
        previous = np.vstack([video_values[:1], video_values[:-1]])
        delta1 = video_values - previous
        prev_delta = np.vstack([delta1[:1], delta1[:-1]])
        accel = delta1 - prev_delta
        for local_index, global_index in enumerate(indices):
            start3 = max(0, local_index - 2)
            start5 = max(0, local_index - 4)
            window3 = video_values[start3 : local_index + 1]
            window5 = video_values[start5 : local_index + 1]
            pieces = [
                delta1[local_index],
                accel[local_index],
                window3.mean(axis=0),
                slope_window(window3),
                slope_window(window5),
            ]
            dynamics[global_index] = np.concatenate(pieces, axis=0)
    if include_base:
        output.append(matrix.astype(np.float32, copy=False))
    output.append(dynamics)
    return np.concatenate(output, axis=1).astype(np.float32)


def pca_fit_transform(
    train_x: np.ndarray,
    apply_x: np.ndarray,
    components: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train-only PCA using SVD; callers pass test rows only for transformation."""
    train_x = np.asarray(train_x, dtype=np.float32)
    apply_x = np.asarray(apply_x, dtype=np.float32)
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    train_z = (train_x - mean) / std
    apply_z = (apply_x - mean) / std
    max_components = max(1, min(components, train_z.shape[0] - 1, train_z.shape[1]))
    if PCA_BACKEND in {"auto", "mps_gram"}:
        try:
            return pca_fit_transform_mps_gram(train_z, apply_z, components, max_components)
        except Exception as exc:
            if PCA_BACKEND == "mps_gram":
                raise
            print(f"[WARN] MPS Gram PCA failed; trying MPS power PCA: {type(exc).__name__}: {exc}", file=sys.stderr)
    if PCA_BACKEND in {"auto", "mps_power"}:
        try:
            return pca_fit_transform_mps_power(train_z, apply_z, components, max_components)
        except Exception as exc:
            if PCA_BACKEND == "mps_power":
                raise
            print(f"[WARN] MPS power PCA failed; falling back to CPU SVD: {type(exc).__name__}: {exc}", file=sys.stderr)
    _, singular_values, vt = np.linalg.svd(train_z, full_matrices=False)
    basis = vt[:max_components].T.astype(np.float32)
    projected = apply_z @ basis
    denom = float(np.sum(singular_values**2))
    explained = (
        (singular_values[:max_components] ** 2 / denom).astype(float).tolist()
        if denom > 0
        else []
    )
    return projected.astype(np.float32), {
        "backend": "cpu_svd",
        "requested_components": components,
        "actual_components": int(max_components),
        "explained_variance_ratio_sum": float(sum(explained)),
        "explained_variance_ratio": explained,
    }


def mps_modified_gram_schmidt(matrix: Any, eps: float = 1e-6) -> Any:
    import torch

    columns = []
    for index in range(matrix.shape[1]):
        vector = matrix[:, index]
        for basis in columns:
            vector = vector - torch.sum(vector * basis) * basis
        norm = torch.linalg.norm(vector).clamp_min(eps)
        columns.append(vector / norm)
    return torch.stack(columns, dim=1)


def pca_fit_transform_mps_power(
    train_z: np.ndarray,
    apply_z: np.ndarray,
    requested_components: int,
    max_components: int,
    oversample: int = 16,
    power_iterations: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Approximate PCA using MPS-native matmul/reduction operations.

    This avoids PyTorch/MLX SVD/eigh GPU gaps by using subspace iteration.
    A tiny Rayleigh-Ritz eigensolve over <= components+oversample dimensions is
    done on CPU to rotate the MPS subspace; the expensive cortical products are MPS.
    """
    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    device = torch.device("mps")
    rank = max_components
    subspace = min(train_z.shape[0] - 1, train_z.shape[1], rank + oversample)
    if subspace < rank:
        rank = subspace
    train_t = torch.as_tensor(train_z, dtype=torch.float32, device=device)
    apply_t = torch.as_tensor(apply_z, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)
    q = torch.randn(train_z.shape[1], subspace, generator=generator, dtype=torch.float32).to(device)
    q = mps_modified_gram_schmidt(q)
    for _ in range(power_iterations):
        sample = train_t @ q
        q = train_t.T @ sample
        q = mps_modified_gram_schmidt(q)
    projected_train = train_t @ q
    small_cov = projected_train.T @ projected_train
    torch.mps.synchronize()
    small_cov_np = small_cov.detach().cpu().numpy()
    eigenvalues, eigenvectors = np.linalg.eigh(small_cov_np)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order][:rank]
    eigenvectors = eigenvectors[:, order][:, :rank]
    rotation = torch.as_tensor(eigenvectors.astype(np.float32), device=device)
    basis = q @ rotation
    projected = apply_t @ basis
    torch.mps.synchronize()
    total_variance = float(np.sum(np.square(train_z)))
    explained = (
        (np.maximum(eigenvalues, 0.0) / total_variance).astype(float).tolist()
        if total_variance > 0
        else []
    )
    return projected.detach().cpu().numpy().astype(np.float32), {
        "backend": "mps_power_cpu_tiny_eigh",
        "requested_components": requested_components,
        "actual_components": int(rank),
        "oversample": int(oversample),
        "power_iterations": int(power_iterations),
        "explained_variance_ratio_sum": float(sum(explained)),
        "explained_variance_ratio": explained,
        "large_matrix_products": "mps",
        "orthogonalization": "mps_modified_gram_schmidt",
        "small_eigensolve": "cpu",
        "approximate": True,
        "train_rows": int(train_z.shape[0]),
        "vertices": int(train_z.shape[1]),
    }


def pca_fit_transform_mps_gram(
    train_z: np.ndarray,
    apply_z: np.ndarray,
    requested_components: int,
    max_components: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Exact train-fit PCA with large matrix products on MPS and small eigensolve on CPU.

    PyTorch MPS currently falls back or errors for SVD/eigh/QR. The heavy operation
    here is the cortical row-space Gram product; that runs on MPS. The CPU only
    solves the much smaller train_rows x train_rows eigensystem.
    """
    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    device = torch.device("mps")
    train_t = torch.as_tensor(train_z, dtype=torch.float32, device=device)
    apply_t = torch.as_tensor(apply_z, dtype=torch.float32, device=device)
    gram = train_t @ train_t.T
    torch.mps.synchronize()
    gram_np = gram.detach().cpu().numpy()
    rank = min(max_components, gram_np.shape[0])
    try:
        from scipy import linalg as scipy_linalg

        # We only need the leading components, not the full row-space eigensystem.
        eigenvalues, eigenvectors = scipy_linalg.eigh(
            gram_np,
            subset_by_index=(gram_np.shape[0] - rank, gram_np.shape[0] - 1),
            check_finite=False,
            overwrite_a=True,
        )
        eigensolve_backend = "scipy_lapack_subset_cpu"
    except Exception as exc:
        if PCA_BACKEND == "mps_gram":
            print(
                f"[WARN] SciPy subset eigensolve failed; falling back to NumPy full eigh: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        eigenvalues, eigenvectors = np.linalg.eigh(gram_np)
        eigensolve_backend = "numpy_full_eigh_cpu"
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    valid = eigenvalues > 1e-6
    eigenvalues = eigenvalues[valid][:max_components]
    eigenvectors = eigenvectors[:, valid][:, :max_components]
    if eigenvectors.shape[1] == 0:
        raise RuntimeError("PCA eigensystem has no positive components")
    eigvec_t = torch.as_tensor(eigenvectors, dtype=torch.float32, device=device)
    scale_t = torch.as_tensor(np.sqrt(eigenvalues), dtype=torch.float32, device=device).clamp_min(1e-6)
    basis = (train_t.T @ eigvec_t) / scale_t.unsqueeze(0)
    projected = apply_t @ basis
    torch.mps.synchronize()
    all_total = float(np.trace(gram_np))
    explained = (
        (np.maximum(eigenvalues, 0.0) / all_total).astype(float).tolist()
        if all_total > 0
        else []
    )
    return projected.detach().cpu().numpy().astype(np.float32), {
        "backend": "mps_gram_cpu_eigh",
        "requested_components": requested_components,
        "actual_components": int(eigenvectors.shape[1]),
        "explained_variance_ratio_sum": float(sum(explained)),
        "explained_variance_ratio": explained,
        "large_matrix_products": "mps",
        "small_eigensolve": "cpu",
        "eigensolve_backend": eigensolve_backend,
        "subset_eigensolve": eigensolve_backend == "scipy_lapack_subset_cpu",
        "train_rows": int(train_z.shape[0]),
        "vertices": int(train_z.shape[1]),
    }


def subcortical_roi_features(subcortical: np.ndarray) -> np.ndarray:
    projection = SubcorticalRoiAdapter().project(subcortical)
    trajectories = np.asarray(projection["region_trajectories"], dtype=np.float32)
    abs_trajectories = np.abs(trajectories)
    global_features = np.stack(
        [
            trajectories.mean(axis=1),
            abs_trajectories.mean(axis=1),
            trajectories.std(axis=1),
            abs_trajectories.max(axis=1),
        ],
        axis=1,
    )
    return np.concatenate([trajectories, abs_trajectories, global_features], axis=1).astype(np.float32)


def feature_names(feature_set: str, width: int) -> list[str]:
    if feature_set == "cortical_global":
        return list(CORTICAL_FEATURE_NAMES[:width])
    if feature_set == "cortical_global_delta":
        base = list(CORTICAL_FEATURE_NAMES)
        names = base[:]
        for prefix in TEMPORAL_PREFIXES:
            names.extend(f"{prefix}_{name}" for name in base)
        return names[:width]
    if feature_set == "cortical_pca_64":
        return [f"pca64_component_{index}" for index in range(width)]
    if feature_set == "cortical_pca_128":
        return [f"pca128_component_{index}" for index in range(width)]
    if feature_set == "cortical_pca64_delta":
        base = [f"pca64_component_{index}" for index in range(max(1, width // (len(TEMPORAL_PREFIXES) + 1)))]
        names = base[:]
        for prefix in TEMPORAL_PREFIXES:
            names.extend(f"{prefix}_{name}" for name in base)
        return names[:width]
    if feature_set == "cortical_raw_ridge":
        return [f"raw_vertex_{index}" for index in range(width)]
    if feature_set == "subcortical_roi":
        roi_width = max(0, (width - 4) // 2)
        names = [f"subcortical_roi_{index}" for index in range(roi_width)]
        names.extend(f"subcortical_roi_{index}_abs" for index in range(roi_width))
        names.extend(["subcortical_mean", "subcortical_mean_abs", "subcortical_std", "subcortical_peak_abs"])
        return names[:width]
    if feature_set == "combined":
        cortical = list(CORTICAL_FEATURE_NAMES)
        subcortical = feature_names("subcortical_roi", max(0, width - len(cortical)))
        return (cortical + subcortical)[:width]
    return [f"{feature_set}_{index}" for index in range(width)]


def time_features(rows: list[dict[str, Any]]) -> np.ndarray:
    grouped = rows_by_video(rows)
    max_time = {
        video_id: max(float(row["time_start_seconds"]) for row in video_rows) or 1.0
        for video_id, video_rows in grouped.items()
    }
    values = []
    for row in rows:
        seconds = float(row["time_start_seconds"])
        frac = seconds / max_time[str(row["video_id"])]
        values.append([1.0, seconds, frac, frac * frac, np.sin(np.pi * 2.0 * frac), np.cos(np.pi * 2.0 * frac)])
    return np.asarray(values, dtype=np.float64)


def autoregressive_features(
    all_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    target: str,
    *,
    include_current: bool,
) -> np.ndarray:
    """Build leak-safe affect-history features for temporal forecasting.

    For future/delta/event targets, y_t is known at forecast time and is allowed.
    For raw y_t prediction, include_current must be false to avoid target leakage.
    """
    lookup = temporal_value_lookup(all_rows, target)
    grouped = rows_by_video(all_rows)
    max_time = {
        video_id: max(float(row["time_start_seconds"]) for row in video_rows) or 1.0
        for video_id, video_rows in grouped.items()
    }
    features = []
    global_values = np.asarray(list(lookup.values()), dtype=np.float64)
    global_mean = float(np.mean(global_values)) if global_values.size else 0.0
    for row in rows:
        video_id = str(row["video_id"])
        second = int(round(float(row["time_start_seconds"])))
        current = lookup.get((video_id, second), global_mean)
        lag_values = [
            lookup.get((video_id, second - lag), global_mean)
            for lag in (1, 2, 3)
        ]
        history = []
        if include_current:
            history.append(current)
        history.extend(lag_values)
        history_arr = np.asarray(history, dtype=np.float64)
        recent_slope = current - lag_values[0]
        momentum = (current - lag_values[0]) - (lag_values[0] - lag_values[1])
        seconds = float(row["time_start_seconds"])
        frac = seconds / max_time.get(video_id, max(seconds, 1.0))
        features.append(
            [
                current if include_current else lag_values[0],
                *lag_values,
                float(np.mean(history_arr)),
                float(np.std(history_arr)),
                recent_slope,
                momentum,
                float(np.min(history_arr)),
                float(np.max(history_arr)),
                seconds,
                frac,
                frac * frac,
                np.sin(np.pi * 2.0 * frac),
                np.cos(np.pi * 2.0 * frac),
            ]
        )
    return np.asarray(features, dtype=np.float64)


def target_array(rows: list[dict[str, Any]], target: str) -> np.ndarray:
    return np.asarray([row["targets"][target] for row in rows], dtype=np.float64)


def temporal_value_lookup(rows: list[dict[str, Any]], target: str) -> dict[tuple[str, int], float]:
    return {
        (str(row["video_id"]), int(round(float(row["time_start_seconds"])))): float(row["targets"][target])
        for row in rows
    }


def derived_target_rows(
    all_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    target: str,
    derived: str,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if derived == "raw":
        return rows, target_array(rows, target)
    lookup = temporal_value_lookup(all_rows, target)
    selected = []
    values = []
    for row in rows:
        key = (str(row["video_id"]), int(round(float(row["time_start_seconds"]))))
        current = lookup.get(key)
        if current is None:
            continue
        if derived == "delta_prev_1s":
            previous = lookup.get((key[0], key[1] - 1))
            if previous is None:
                continue
            value = current - previous
        elif derived.startswith("future_state_p"):
            horizon = int(derived.removeprefix("future_state_p").removesuffix("s"))
            future = lookup.get((key[0], key[1] + horizon))
            if future is None:
                continue
            value = future
        elif derived.startswith("future_change_p"):
            horizon = int(derived.removeprefix("future_change_p").removesuffix("s"))
            future = lookup.get((key[0], key[1] + horizon))
            if future is None:
                continue
            value = future - current
        elif derived.startswith("residual_future_p") and derived.endswith("_persistence"):
            horizon = int(derived.removeprefix("residual_future_p").removesuffix("s_persistence"))
            future = lookup.get((key[0], key[1] + horizon))
            if future is None:
                continue
            value = future - current
        elif derived.startswith("residual_future_p") and derived.endswith("_rolling3"):
            horizon = int(derived.removeprefix("residual_future_p").removesuffix("s_rolling3"))
            future = lookup.get((key[0], key[1] + horizon))
            history = [
                lookup.get((key[0], key[1] - offset))
                for offset in range(0, 3)
            ]
            history = [item for item in history if item is not None]
            if future is None or not history:
                continue
            value = future - float(np.mean(history))
        elif derived == "event_future_spike_1_3s":
            if target != "arousal":
                continue
            futures = [lookup.get((key[0], key[1] + horizon)) for horizon in (1, 2, 3)]
            futures = [item for item in futures if item is not None]
            if not futures:
                continue
            value = float(max(futures) - current >= EVENT_THRESHOLD)
        elif derived == "event_future_drop_1_3s":
            futures = [lookup.get((key[0], key[1] + horizon)) for horizon in (1, 2, 3)]
            futures = [item for item in futures if item is not None]
            if not futures:
                continue
            value = float(min(futures) - current <= -EVENT_THRESHOLD)
        elif derived == "event_future_rise_1_3s":
            futures = [lookup.get((key[0], key[1] + horizon)) for horizon in (1, 2, 3)]
            futures = [item for item in futures if item is not None]
            if not futures:
                continue
            value = float(max(futures) - current >= EVENT_THRESHOLD)
        elif derived == "event_trend_reversal_1_3s":
            previous = lookup.get((key[0], key[1] - 1))
            future = lookup.get((key[0], key[1] + 3))
            if previous is None or future is None:
                continue
            past_delta = current - previous
            future_delta = future - current
            value = float(abs(past_delta) >= EVENT_THRESHOLD / 2 and abs(future_delta) >= EVENT_THRESHOLD / 2 and np.sign(past_delta) != np.sign(future_delta))
        elif derived == "event_peak_onset_1_3s":
            previous = lookup.get((key[0], key[1] - 1))
            future_values = [lookup.get((key[0], key[1] + horizon)) for horizon in (1, 2, 3)]
            future_values = [item for item in future_values if item is not None]
            if previous is None or len(future_values) < 3:
                continue
            value = float(
                current - previous >= EVENT_THRESHOLD / 2
                and max(future_values) - current >= EVENT_THRESHOLD
                and future_values[-1] < max(future_values)
            )
        elif derived == "event_recovery_onset_1_3s":
            previous = lookup.get((key[0], key[1] - 1))
            futures = [lookup.get((key[0], key[1] + horizon)) for horizon in (1, 2, 3)]
            futures = [item for item in futures if item is not None]
            if previous is None or not futures:
                continue
            value = float(current - previous <= -EVENT_THRESHOLD / 2 and max(futures) - current >= EVENT_THRESHOLD)
        else:
            raise ValueError(f"Unsupported derived target: {derived}")
        selected.append(row)
        values.append(value)
    return selected, np.asarray(values, dtype=np.float64)


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (train - mean) / std, (test - mean) / std


def ridge_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    if RIDGE_BACKEND in {"auto", "mps_solve"} and train_x.shape[1] >= RIDGE_MPS_MIN_FEATURES:
        try:
            return ridge_fit_predict_mps(train_x, train_y, test_x, alpha=alpha)
        except Exception as exc:
            if RIDGE_BACKEND == "mps_solve":
                raise
            print(f"[WARN] MPS ridge failed; falling back to CPU pinv: {type(exc).__name__}: {exc}", file=sys.stderr)
    train_x, test_x = standardize(train_x, test_x)
    train_x = np.concatenate([np.ones((train_x.shape[0], 1)), train_x], axis=1)
    test_x = np.concatenate([np.ones((test_x.shape[0], 1)), test_x], axis=1)
    penalty = np.eye(train_x.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(train_x.T @ train_x + penalty) @ train_x.T @ train_y
    return test_x @ beta, beta


def ridge_fit_predict_mps(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    train_x, test_x = standardize(train_x, test_x)
    device = torch.device("mps")
    train_t = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    test_t = torch.as_tensor(test_x, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(train_y.reshape(-1, 1), dtype=torch.float32, device=device)
    ones_train = torch.ones((train_t.shape[0], 1), dtype=torch.float32, device=device)
    ones_test = torch.ones((test_t.shape[0], 1), dtype=torch.float32, device=device)
    train_design = torch.cat([ones_train, train_t], dim=1)
    test_design = torch.cat([ones_test, test_t], dim=1)
    if train_design.shape[1] > train_design.shape[0]:
        penalty = torch.eye(train_design.shape[0], dtype=torch.float32, device=device) * float(alpha)
        system = train_design @ train_design.T + penalty
        dual = torch.linalg.solve(system, y_t)
        beta = train_design.T @ dual
    else:
        penalty = torch.eye(train_design.shape[1], dtype=torch.float32, device=device) * float(alpha)
        penalty[0, 0] = 0.0
        system = train_design.T @ train_design + penalty
        rhs = train_design.T @ y_t
        beta = torch.linalg.solve(system, rhs)
    pred = test_design @ beta
    torch.mps.synchronize()
    return pred.detach().cpu().numpy().reshape(-1).astype(np.float64), beta.detach().cpu().numpy().reshape(-1).astype(np.float64)


def ridge(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return ridge_fit_predict(train_x, train_y, test_x, alpha=alpha)[0]


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if len(unique) != len(values):
        sums = np.bincount(inverse, weights=ranks)
        ranks = sums[inverse] / counts[inverse]
    return ranks


def corr(y_true: np.ndarray, y_pred: np.ndarray, spearman: bool = False) -> float | None:
    if len(y_true) < 2:
        return None
    left = rankdata(y_true) if spearman else y_true
    right = rankdata(y_pred) if spearman else y_pred
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    return {
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean(np.square(y_true - y_pred)))),
        "pearson": corr(y_true, y_pred),
        "spearman": corr(y_true, y_pred, spearman=True),
    }


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, train_prevalence: float) -> dict[str, float | None]:
    if len(y_true) == 0:
        return {"accuracy": None, "balanced_accuracy": None, "precision": None, "recall": None, "f1": None, "event_rate": None}
    prevalence = min(0.95, max(0.05, float(train_prevalence)))
    threshold = float(np.quantile(y_score, 1.0 - prevalence))
    y_pred = (y_score >= threshold).astype(np.float64)
    tp = float(np.sum((y_pred == 1.0) & (y_true == 1.0)))
    fp = float(np.sum((y_pred == 1.0) & (y_true == 0.0)))
    fn = float(np.sum((y_pred == 0.0) & (y_true == 1.0)))
    tn = float(np.sum((y_pred == 0.0) & (y_true == 0.0)))
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    balanced_accuracy = (
        (recall + specificity) / 2.0
        if recall is not None and specificity is not None
        else None
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall)
        else None
    )
    return {
        "accuracy": (tp + tn) / len(y_true),
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "event_rate": float(np.mean(y_true)),
        "threshold": threshold,
    }


def mean_predict(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], target: str) -> np.ndarray:
    return np.full(len(test_rows), float(np.mean(target_array(train_rows, target))), dtype=np.float64)


def persistence_predict(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], target: str) -> np.ndarray:
    global_mean = float(np.mean(target_array(train_rows, target)))
    history: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in train_rows:
        history[str(row["video_id"])].append((int(row["frame_index"]), float(row["targets"][target])))
    for values in history.values():
        values.sort()
    preds = []
    for row in test_rows:
        values = history.get(str(row["video_id"]), [])
        position = bisect.bisect_left(values, (int(row["frame_index"]), -math.inf))
        preds.append(values[position - 1][1] if position else global_mean)
    return np.asarray(preds, dtype=np.float64)


def persistence_predict_within_rows(rows: list[dict[str, Any]], target: str) -> np.ndarray:
    global_mean = float(np.mean(target_array(rows, target)))
    history: dict[str, list[tuple[int, float]]] = defaultdict(list)
    preds = []
    for row in rows:
        video_id = str(row["video_id"])
        frame = int(row["frame_index"])
        values = history[video_id]
        position = bisect.bisect_left(values, (frame, -math.inf))
        preds.append(values[position - 1][1] if position else global_mean)
        bisect.insort(values, (frame, float(row["targets"][target])))
    return np.asarray(preds, dtype=np.float64)


def row_indices(all_rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> np.ndarray:
    lookup = {id(row): index for index, row in enumerate(all_rows)}
    return np.asarray([lookup[id(row)] for row in selected], dtype=np.int64)


def build_split_feature_sets(
    all_rows: list[dict[str, Any]],
    base_feature_sets: dict[str, np.ndarray],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_mode: str,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Return split-local train/test feature matrices.

    PCA modes fit the basis on train rows only. Static/global modes reuse
    no-reencode matrices generated from cached TRIBE outputs.
    """
    if feature_mode == "cortical_global":
        matrix = base_feature_sets["cortical_global"]
        return {"cortical_global": (matrix[train_idx], matrix[test_idx])}, {}
    if feature_mode == "cortical_global_delta":
        matrix = base_feature_sets["cortical_global_delta"]
        return {"cortical_global_delta": (matrix[train_idx], matrix[test_idx])}, {}
    if feature_mode == "cortical_raw_ridge":
        matrix = base_feature_sets["cortical_raw"]
        return {"cortical_raw_ridge": (matrix[train_idx], matrix[test_idx])}, {
            "experimental": True,
            "warning": "Full raw cortical ridge is high-dimensional and not a default evidence mode.",
        }
    if feature_mode in {"cortical_pca_64", "cortical_pca_128", "cortical_pca64_delta"}:
        components = 128 if feature_mode == "cortical_pca_128" else 64
        raw = base_feature_sets["cortical_raw"]
        if feature_mode == "cortical_pca64_delta":
            projected_all, metadata = pca_fit_transform(raw[train_idx], raw, components)
            matrix = temporal_dynamics_features(all_rows, projected_all, include_base=True)
            return {"cortical_pca64_delta": (matrix[train_idx], matrix[test_idx])}, {
                "cortical_pca64_delta": metadata,
                "temporal_dynamics": list(TEMPORAL_PREFIXES),
                "leakage_contract": "PCA basis fit on train rows only; all rows transformed using train-fitted basis; dynamics are within-video only.",
            }
        train_test = np.concatenate([raw[train_idx], raw[test_idx]], axis=0)
        projected, metadata = pca_fit_transform(raw[train_idx], train_test, components)
        train_projected = projected[: len(train_idx)]
        test_projected = projected[len(train_idx) :]
        name = f"cortical_pca_{components}"
        return {name: (train_projected, test_projected)}, {
            name: metadata,
            "leakage_contract": "PCA basis fit on train rows only; test rows transformed using train-fitted basis.",
        }
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def cache_feature_keys_for(feature_mode: str, run_mode: str) -> tuple[str, ...]:
    if feature_mode in {"cortical_global", "cortical_global_delta"}:
        return ("cortical_global",)
    if feature_mode in {"cortical_pca_64", "cortical_pca_128", "cortical_pca64_delta", "cortical_raw_ridge"}:
        return ("cortical_raw",)
    return FEATURE_SETS_BY_RUN_MODE[run_mode]


def report_feature_sets_for(
    feature_mode: str,
    feature_sets: dict[str, np.ndarray],
) -> dict[str, int | str]:
    if feature_mode == "cortical_global":
        return {"cortical_global": int(feature_sets["cortical_global"].shape[1])}
    if feature_mode == "cortical_global_delta":
        return {"cortical_global_delta": int(feature_sets["cortical_global_delta"].shape[1])}
    if feature_mode == "cortical_pca_64":
        return {"cortical_pca_64": 64}
    if feature_mode == "cortical_pca_128":
        return {"cortical_pca_128": 128}
    if feature_mode == "cortical_pca64_delta":
        return {"cortical_pca64_delta": 64 * (len(TEMPORAL_PREFIXES) + 1)}
    if feature_mode == "cortical_raw_ridge":
        return {"cortical_raw_ridge": int(feature_sets["cortical_raw"].shape[1])}
    return {key: int(value.shape[1]) for key, value in feature_sets.items() if key != "cortical_raw"}


def lead_lag_feature_sets(
    all_rows: list[dict[str, Any]],
    base_feature_sets: dict[str, np.ndarray],
    feature_mode: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if feature_mode == "cortical_global":
        return {"cortical_global": base_feature_sets["cortical_global"]}, {"basis": "static"}
    if feature_mode == "cortical_global_delta":
        return {"cortical_global_delta": base_feature_sets["cortical_global_delta"]}, {"basis": "static"}
    if feature_mode == "cortical_raw_ridge":
        return {}, {"skipped": "raw 20484-column lead/lag omitted to avoid excessive diagnostic cost"}
    if feature_mode in {"cortical_pca_64", "cortical_pca_128", "cortical_pca64_delta"}:
        components = 128 if feature_mode == "cortical_pca_128" else 64
        raw = base_feature_sets["cortical_raw"]
        projected, metadata = pca_fit_transform(raw, raw, components)
        if feature_mode == "cortical_pca64_delta":
            return {"cortical_pca64_delta": temporal_dynamics_features(all_rows, projected, include_base=True)}, {
                "basis": "diagnostic_all_rows_fit",
                "leakage_warning": "Lead/lag PCA diagnostic fits unsupervised PCA on all accepted rows only for alignment scanning; split metrics use train-only PCA.",
                **metadata,
            }
        name = f"cortical_pca_{components}"
        return {name: projected}, {
            "basis": "diagnostic_all_rows_fit",
            "leakage_warning": "Lead/lag PCA diagnostic fits unsupervised PCA on all accepted rows only for alignment scanning; split metrics use train-only PCA.",
            **metadata,
        }
    return {}, {"skipped": f"unsupported feature mode {feature_mode}"}


def eval_split(
    all_rows: list[dict[str, Any]],
    feature_sets: dict[str, np.ndarray],
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    seed: int,
    feature_mode: str = "cortical_global",
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    train_idx = row_indices(all_rows, train_rows)
    test_idx = row_indices(all_rows, test_rows)
    split_feature_sets, split_feature_metadata = build_split_feature_sets(
        all_rows,
        feature_sets,
        train_idx,
        test_idx,
        feature_mode,
    )
    train_position = {int(global_index): position for position, global_index in enumerate(train_idx)}
    test_position = {int(global_index): position for position, global_index in enumerate(test_idx)}
    result: dict[str, Any] = {
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_videos": len({row["video_id"] for row in train_rows}),
        "test_videos": len({row["video_id"] for row in test_rows}),
        "feature_mode": feature_mode,
        "split_feature_metadata": split_feature_metadata,
        "targets": {},
    }
    for target in TARGETS:
        for derived in DERIVED_TARGETS:
            target_name = f"{target}__{derived}"
            if derived == "residual_after_persistence":
                train_base = persistence_predict_within_rows(train_rows, target)
                test_base = persistence_predict(train_rows, test_rows, target)
                train_selected = train_rows
                test_selected = test_rows
                train_y = target_array(train_rows, target) - train_base
                test_y = target_array(test_rows, target) - test_base
            elif derived.startswith("residual_future_p") and derived.endswith("_time_only"):
                horizon = int(derived.removeprefix("residual_future_p").removesuffix("s_time_only"))
                future_name = f"future_state_p{horizon}s"
                train_selected, train_future_y = derived_target_rows(all_rows, train_rows, target, future_name)
                test_selected, test_future_y = derived_target_rows(all_rows, test_rows, target, future_name)
                if len(train_selected) >= 4 and len(test_selected) >= 2:
                    train_time_pred = ridge(time_features(train_selected), train_future_y, time_features(train_selected))
                    test_time_pred = ridge(time_features(train_selected), train_future_y, time_features(test_selected))
                    train_y = train_future_y - train_time_pred
                    test_y = test_future_y - test_time_pred
                else:
                    train_y = np.asarray([], dtype=np.float64)
                    test_y = np.asarray([], dtype=np.float64)
            else:
                train_selected, train_y = derived_target_rows(all_rows, train_rows, target, derived)
                test_selected, test_y = derived_target_rows(all_rows, test_rows, target, derived)
            if len(train_selected) < 4 or len(test_selected) < 2:
                result["targets"][target_name] = {
                    "skipped": {
                        "reason": "insufficient_rows",
                        "train_rows": len(train_selected),
                        "test_rows": len(test_selected),
                    }
                }
                continue
            target_train_idx = row_indices(all_rows, train_selected)
            target_test_idx = row_indices(all_rows, test_selected)
            target_train_pos = np.asarray([train_position[int(index)] for index in target_train_idx], dtype=np.int64)
            target_test_pos = np.asarray([test_position[int(index)] for index in target_test_idx], dtype=np.int64)
            include_current_in_ar = derived != "raw"
            train_ar = autoregressive_features(
                all_rows, train_selected, target, include_current=include_current_in_ar
            )
            test_ar = autoregressive_features(
                all_rows, test_selected, target, include_current=include_current_in_ar
            )
            ar_train_pred = ridge(train_ar, train_y, train_ar)
            ar_test_pred = ridge(train_ar, train_y, test_ar)
            predictions = {
                "mean_train": np.full(len(test_selected), float(np.mean(train_y)), dtype=np.float64),
                "time_ridge": ridge(time_features(train_selected), train_y, time_features(test_selected)),
                "autoregressive": ar_test_pred,
            }
            if derived == "raw":
                predictions["persistence_previous_known"] = persistence_predict(train_selected, test_selected, target)
            elif derived.startswith("future_state_p"):
                predictions["persistence_current_value"] = target_array(test_selected, target)
            elif derived.startswith("future_change_p") or derived.startswith("residual_future_p"):
                predictions["zero_change_or_residual"] = np.zeros(len(test_selected), dtype=np.float64)
            elif derived == "residual_after_persistence":
                predictions["zero_residual"] = np.zeros(len(test_selected), dtype=np.float64)
            for name, (train_matrix, test_matrix) in split_feature_sets.items():
                train_feature_matrix = train_matrix[target_train_pos]
                test_feature_matrix = test_matrix[target_test_pos]
                predictions[name] = ridge(train_feature_matrix, train_y, test_feature_matrix)
                train_perm = rng.permutation(len(target_train_idx))
                test_perm = rng.permutation(len(target_test_idx))
                predictions[f"shuffled_{name}"] = ridge(
                    train_feature_matrix[train_perm],
                    train_y,
                    test_feature_matrix[test_perm],
                )
                predictions[f"random_gaussian_{name}"] = ridge(
                    rng.normal(size=train_feature_matrix.shape),
                    train_y,
                    rng.normal(size=test_feature_matrix.shape),
                )
                ar_train_x = np.concatenate([train_ar, train_feature_matrix], axis=1)
                ar_test_x = np.concatenate([test_ar, test_feature_matrix], axis=1)
                predictions[f"autoregressive_plus_{name}"] = ridge(ar_train_x, train_y, ar_test_x)
                predictions[f"autoregressive_plus_shuffled_{name}"] = ridge(
                    np.concatenate([train_ar, train_feature_matrix[train_perm]], axis=1),
                    train_y,
                    np.concatenate([test_ar, test_feature_matrix[test_perm]], axis=1),
                )
                predictions[f"autoregressive_plus_random_gaussian_{name}"] = ridge(
                    np.concatenate([train_ar, rng.normal(size=train_feature_matrix.shape)], axis=1),
                    train_y,
                    np.concatenate([test_ar, rng.normal(size=test_feature_matrix.shape)], axis=1),
                )
                neuro_residual_train = train_y - ar_train_pred
                residual_pred = ridge(train_feature_matrix, neuro_residual_train, test_feature_matrix)
                predictions[f"residualized_autoregressive_plus_{name}"] = ar_test_pred + residual_pred
                residual_pred_shuffled = ridge(
                    train_feature_matrix[train_perm],
                    neuro_residual_train,
                    test_feature_matrix[test_perm],
                )
                predictions[f"residualized_autoregressive_plus_shuffled_{name}"] = ar_test_pred + residual_pred_shuffled
                residual_pred_random = ridge(
                    rng.normal(size=train_feature_matrix.shape),
                    neuro_residual_train,
                    rng.normal(size=test_feature_matrix.shape),
                )
                predictions[f"residualized_autoregressive_plus_random_gaussian_{name}"] = ar_test_pred + residual_pred_random
            if "combined" in split_feature_sets:
                shuffled_y = rng.permutation(train_y)
                combined_train, combined_test = split_feature_sets["combined"]
                predictions["shuffled_labels_autoregressive_plus_combined"] = ridge(
                    np.concatenate([train_ar, combined_train[target_train_pos]], axis=1),
                    shuffled_y,
                    np.concatenate([test_ar, combined_test[target_test_pos]], axis=1),
                )
            is_event_target = derived.startswith("event_")
            if is_event_target:
                metric_fn = lambda truth, pred: binary_metrics(truth, pred, float(np.mean(train_y)))
            else:
                metric_fn = metrics
            result["targets"][target_name] = {
                "target_contract": {
                    "base_target": target,
                    "derived_target": derived,
                    "train_rows": len(train_selected),
                    "test_rows": len(test_selected),
                    "metric_family": "classification" if is_event_target else "regression",
                },
                **{key: metric_fn(test_y, pred) for key, pred in predictions.items()},
            }
    return result


def aggregate_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "folds": len(folds),
        "test_rows_total": int(sum(fold["test_rows"] for fold in folds)),
        "targets": {},
    }
    target_names = sorted({target for fold in folds for target in fold["targets"]})
    for target in target_names:
        output["targets"][target] = {}
        conditions = sorted(
            {
                condition
                for fold in folds
                if target in fold["targets"]
                for condition in fold["targets"][target]
            }
        )
        for condition in conditions:
            if condition == "target_contract" or condition == "skipped":
                continue
            if not any(
                isinstance(fold.get("targets", {}).get(target), dict)
                and condition in fold["targets"][target]
                for fold in folds
            ):
                continue
            output["targets"][target][condition] = {}
            metric_names = sorted(
                {
                    metric
                    for fold in folds
                    if isinstance(fold.get("targets", {}).get(target), dict)
                    and condition in fold["targets"][target]
                    and isinstance(fold["targets"][target][condition], dict)
                    for metric in fold["targets"][target][condition]
                }
            )
            for metric_name in metric_names:
                values = [
                    fold.get("targets", {}).get(target, {}).get(condition, {}).get(metric_name)
                    for fold in folds
                    if isinstance(fold.get("targets", {}).get(target), dict)
                    and isinstance(fold["targets"][target].get(condition), dict)
                    and fold["targets"][target][condition].get(metric_name) is not None
                ]
                output["targets"][target][condition][metric_name] = float(np.mean(values)) if values else None
    return output


def split_target_name(target_name: str) -> tuple[str, str]:
    target, derived = target_name.split("__", 1)
    return target, derived


def diagnostic_target_data(
    all_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    target_name: str,
) -> tuple[str, str, list[dict[str, Any]], np.ndarray, list[dict[str, Any]], np.ndarray]:
    target, derived = split_target_name(target_name)
    if derived == "residual_after_persistence":
        train_base = persistence_predict_within_rows(train_rows, target)
        test_base = persistence_predict(train_rows, test_rows, target)
        return target, derived, train_rows, target_array(train_rows, target) - train_base, test_rows, target_array(test_rows, target) - test_base
    train_selected, train_y = derived_target_rows(all_rows, train_rows, target, derived)
    test_selected, test_y = derived_target_rows(all_rows, test_rows, target, derived)
    return target, derived, train_selected, train_y, test_selected, test_y


def metric_value(y_true: np.ndarray, y_pred: np.ndarray, is_event_target: bool, train_y: np.ndarray) -> tuple[str, float | None, dict[str, Any]]:
    if is_event_target:
        result = binary_metrics(y_true, y_pred, float(np.mean(train_y)))
        return "f1", result.get("f1"), result
    result = metrics(y_true, y_pred)
    return "mae", result.get("mae"), result


def diagnostic_condition_matrix(
    condition: str,
    train_ar: np.ndarray,
    test_ar: np.ndarray,
    feature_sets: dict[str, np.ndarray],
    target_train_idx: np.ndarray,
    target_test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str], str, bool]:
    is_residualized = condition.startswith("residualized_autoregressive_plus_")
    feature_set = condition.removeprefix("residualized_autoregressive_plus_").removeprefix("autoregressive_plus_")
    if feature_set not in feature_sets:
        raise KeyError(feature_set)
    neuro_train = feature_sets[feature_set][target_train_idx]
    neuro_test = feature_sets[feature_set][target_test_idx]
    names = list(AR_FEATURE_NAMES[: train_ar.shape[1]]) + feature_names(feature_set, neuro_train.shape[1])
    return (
        np.concatenate([train_ar, neuro_train], axis=1),
        np.concatenate([test_ar, neuro_test], axis=1),
        names,
        feature_set,
        is_residualized,
    )


def permutation_importance_for_split(
    all_rows: list[dict[str, Any]],
    feature_sets: dict[str, np.ndarray],
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    output: dict[str, Any] = {}
    for target_name in DIAGNOSTIC_TARGETS:
        target, derived, train_selected, train_y, test_selected, test_y = diagnostic_target_data(
            all_rows, train_rows, test_rows, target_name
        )
        if len(train_selected) < 8 or len(test_selected) < 4:
            output[target_name] = {"skipped": "insufficient_rows"}
            continue
        target_train_idx = row_indices(all_rows, train_selected)
        target_test_idx = row_indices(all_rows, test_selected)
        include_current_in_ar = derived != "raw"
        train_ar = autoregressive_features(all_rows, train_selected, target, include_current=include_current_in_ar)
        test_ar = autoregressive_features(all_rows, test_selected, target, include_current=include_current_in_ar)
        ar_train_pred = ridge(train_ar, train_y, train_ar)
        ar_test_pred = ridge(train_ar, train_y, test_ar)
        is_event = derived.startswith("event_")
        target_output: dict[str, Any] = {}
        for condition in DIAGNOSTIC_CONDITIONS:
            feature_set_name = condition.removeprefix("residualized_autoregressive_plus_").removeprefix("autoregressive_plus_")
            if feature_set_name not in feature_sets:
                continue
            train_x, test_x, names, feature_set, is_residualized = diagnostic_condition_matrix(
                condition, train_ar, test_ar, feature_sets, target_train_idx, target_test_idx
            )
            fit_y = train_y - ar_train_pred if is_residualized else train_y
            pred, beta = ridge_fit_predict(train_x, fit_y, test_x)
            pred = ar_test_pred + pred if is_residualized else pred
            metric_name, base_score, base_metrics = metric_value(test_y, pred, is_event, train_y)
            if base_score is None:
                continue
            standardized_train, standardized_test = standardize(train_x, test_x)
            coefficients = np.asarray(beta[1:], dtype=np.float64)
            top_coefficients = sorted(
                (
                    {
                        "feature": names[index] if index < len(names) else f"feature_{index}",
                        "abs_coefficient": float(abs(value)),
                        "coefficient": float(value),
                        "group": "autoregressive" if index < train_ar.shape[1] else feature_set,
                    }
                    for index, value in enumerate(coefficients)
                ),
                key=lambda item: item["abs_coefficient"],
                reverse=True,
            )[:12]
            permutation_rows = []
            for index, name in enumerate(names):
                permuted = standardized_test.copy()
                permuted[:, index] = rng.permutation(permuted[:, index])
                permuted_x = np.concatenate([np.ones((permuted.shape[0], 1)), permuted], axis=1)
                permuted_pred = permuted_x @ beta
                permuted_pred = ar_test_pred + permuted_pred if is_residualized else permuted_pred
                _, permuted_score, _ = metric_value(test_y, permuted_pred, is_event, train_y)
                if permuted_score is None:
                    continue
                importance_delta = (
                    float(base_score - permuted_score)
                    if is_event
                    else float(permuted_score - base_score)
                )
                permutation_rows.append(
                    {
                        "feature": name,
                        "group": "autoregressive" if index < train_ar.shape[1] else feature_set,
                        "importance_delta": importance_delta,
                        "permuted_score": float(permuted_score),
                    }
                )
            top_permutation = sorted(
                permutation_rows,
                key=lambda item: item["importance_delta"],
                reverse=True,
            )[:12]
            group_importance: dict[str, float] = defaultdict(float)
            for row in permutation_rows:
                if row["importance_delta"] > 0:
                    group_importance[row["group"]] += float(row["importance_delta"])
            target_output[condition] = {
                "metric": metric_name,
                "base_score": float(base_score),
                "base_metrics": base_metrics,
                "top_coefficients": top_coefficients,
                "top_permutation_importance": top_permutation,
                "positive_permutation_importance_by_group": dict(sorted(group_importance.items())),
            }
        output[target_name] = target_output
    return output


def lead_lag_analysis(
    all_rows: list[dict[str, Any]],
    feature_sets: dict[str, np.ndarray],
) -> dict[str, Any]:
    index_by_time = {
        (str(row["video_id"]), int(round(float(row["time_start_seconds"])))): index
        for index, row in enumerate(all_rows)
    }
    output: dict[str, Any] = {
        "contract": {
            "offset_seconds": "feature_time = target_time + offset_seconds",
            "negative_offset_interpretation": "feature occurs before the target/change row, so the feature leads the annotation target",
            "positive_offset_interpretation": "feature occurs after the target/change row, so the feature lags the annotation target",
        },
        "targets": {},
    }
    for target_name in (
        "arousal__future_change_p1s",
        "arousal__future_change_p2s",
        "arousal__future_change_p3s",
        "valence__future_change_p1s",
        "valence__future_change_p2s",
        "valence__future_change_p3s",
    ):
        target, derived = split_target_name(target_name)
        selected, y = derived_target_rows(all_rows, all_rows, target, derived)
        target_result: dict[str, Any] = {}
        for feature_set, matrix in feature_sets.items():
            names = feature_names(feature_set, matrix.shape[1])
            offset_rows = []
            for offset in range(-3, 4):
                feature_indices = []
                target_values = []
                for row, value in zip(selected, y):
                    key = (str(row["video_id"]), int(round(float(row["time_start_seconds"]))) + offset)
                    feature_index = index_by_time.get(key)
                    if feature_index is None:
                        continue
                    feature_indices.append(feature_index)
                    target_values.append(float(value))
                if len(feature_indices) < 8:
                    offset_rows.append(
                        {
                            "offset_seconds": offset,
                            "aligned_rows": len(feature_indices),
                            "mean_abs_pearson": None,
                            "max_abs_pearson": None,
                            "best_feature": None,
                            "best_pearson": None,
                        }
                    )
                    continue
                feature_matrix = matrix[np.asarray(feature_indices, dtype=np.int64)]
                y_values = np.asarray(target_values, dtype=np.float64)
                correlations = []
                for column in range(feature_matrix.shape[1]):
                    value = corr(y_values, feature_matrix[:, column])
                    if value is not None:
                        correlations.append((column, float(value)))
                if not correlations:
                    offset_rows.append(
                        {
                            "offset_seconds": offset,
                            "aligned_rows": len(feature_indices),
                            "mean_abs_pearson": None,
                            "max_abs_pearson": None,
                            "best_feature": None,
                            "best_pearson": None,
                        }
                    )
                    continue
                best_column, best_corr = max(correlations, key=lambda item: abs(item[1]))
                offset_rows.append(
                    {
                        "offset_seconds": offset,
                        "aligned_rows": len(feature_indices),
                        "mean_abs_pearson": float(np.mean([abs(item[1]) for item in correlations])),
                        "max_abs_pearson": float(abs(best_corr)),
                        "best_feature": names[best_column] if best_column < len(names) else f"feature_{best_column}",
                        "best_pearson": best_corr,
                    }
                )
            best = max(
                [row for row in offset_rows if row["max_abs_pearson"] is not None],
                key=lambda row: row["max_abs_pearson"],
                default=None,
            )
            target_result[feature_set] = {
                "offsets": offset_rows,
                "best_offset": best,
            }
        output["targets"][target_name] = target_result
    return output


def best_conditions(target_table: dict[str, Any], metric: str, limit: int = 8) -> list[dict[str, Any]]:
    rows = []
    for condition, values in target_table.items():
        if not isinstance(values, dict) or values.get(metric) is None:
            continue
        rows.append({"condition": condition, metric: values[metric]})
    reverse = metric in {"pearson", "spearman", "accuracy", "f1", "balanced_accuracy", "precision", "recall"}
    return sorted(rows, key=lambda item: item[metric], reverse=reverse)[:limit]


def write_markdown_summary(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# VEATIC Temporal Dynamics Benchmark",
        "",
        "## How To Read This",
        "This benchmark predicts VEATIC human valence/arousal annotations from cached TRIBE features and controls.",
        "Raw state prediction is expected to be dominated by temporal persistence. The stronger test is whether real neuro features add value beyond autoregressive affect-history baselines and beyond shuffled/random controls.",
        "",
        "## Dataset / Cache",
        f"- Accepted videos: {report['accepted_videos']}",
        f"- Accepted rows: {report['accepted_rows']}",
        f"- Rejected cache entries: {len(report['rejected'])}",
        f"- Feature sets: {report['feature_sets']}",
        f"- Run mode: {report['run_mode']}",
        f"- Feature mode: {report.get('feature_mode', 'cortical_global')}",
        f"- Subcortical enabled: {report['subcortical_enabled']}",
        "",
        "## Scientific Contract",
        f"- TRIBE extraction contract unchanged.",
        f"- Default subcortical policy: {report['subcortical_policy']}",
        "- Subcortical remains available for explicit `full_research` and `subcortical_ablation` runs.",
        "- Subcortical is disabled in the default run because current OpenLAV/VEATIC evidence does not show stable additive lift over compact cortical features, while it adds inference time, memory pressure, crash risk, and benchmark complexity.",
        "- Expected benefit: lower runtime and memory pressure by skipping the separate subcortical model branch and ROI projection; exact speedup depends on video length and cache state.",
        f"- Event threshold: {report['target_contract']['event_threshold']}",
        "- Autoregressive features use only current/past labels relative to the prediction horizon.",
        "- Residualization is fit inside each split/fold only.",
        "",
    ]
    for mode_name, mode in report["modes"].items():
        aggregate = mode.get("aggregate", mode)
        lines.extend([f"## {mode_name}", ""])
        if "gap_rows" in mode:
            lines.append(f"- Gap rows: {mode['gap_rows']}")
        targets = aggregate.get("targets", {})
        focus_targets = [
            "arousal__future_state_p1s",
            "arousal__future_change_p1s",
            "arousal__residual_future_p1s_persistence",
            "arousal__event_future_spike_1_3s",
            "arousal__event_future_drop_1_3s",
            "arousal__event_trend_reversal_1_3s",
            "arousal__event_peak_onset_1_3s",
            "arousal__event_recovery_onset_1_3s",
            "valence__future_state_p1s",
            "valence__future_change_p1s",
            "valence__residual_future_p1s_persistence",
            "valence__event_future_drop_1_3s",
        ]
        for target_name in focus_targets:
            table = targets.get(target_name)
            if not table:
                continue
            metric = "f1" if "event_" in target_name else "mae"
            lines.append(f"### {target_name}")
            lines.append("")
            lines.append(f"Top conditions by `{metric}`:")
            for row in best_conditions(table, metric, limit=6):
                lines.append(f"- {row['condition']}: {row[metric]:.4f}")
            for condition in (
                "autoregressive",
                "autoregressive_plus_cortical_global",
                "autoregressive_plus_subcortical_roi",
                "autoregressive_plus_combined",
                "autoregressive_plus_shuffled_combined",
                "autoregressive_plus_random_gaussian_combined",
                "residualized_autoregressive_plus_combined",
                "residualized_autoregressive_plus_shuffled_combined",
            ):
                if condition in table and isinstance(table[condition], dict):
                    values = table[condition]
                    compact = ", ".join(
                        f"{key}={value:.4f}"
                        for key, value in values.items()
                        if value is not None and key in {"mae", "rmse", "pearson", "spearman", "accuracy", "f1"}
                    )
            lines.append(f"- {condition}: {compact}")
            lines.append("")
    lines.extend(["## Lead / Lag Diagnostics", ""])
    lead_lag = report.get("lead_lag_analysis", {}).get("targets", {})
    for target_name in (
        "arousal__future_change_p1s",
        "arousal__future_change_p2s",
        "arousal__future_change_p3s",
        "valence__future_change_p1s",
        "valence__future_change_p2s",
        "valence__future_change_p3s",
    ):
        target_result = lead_lag.get(target_name, {})
        if not target_result:
            continue
        lines.append(f"### {target_name}")
        for feature_set, values in target_result.items():
            best = values.get("best_offset")
            if not best:
                continue
            lines.append(
                "- "
                f"{feature_set}: best_offset={best['offset_seconds']}s, "
                f"max_abs_pearson={best['max_abs_pearson']:.4f}, "
                f"best_feature={best['best_feature']}, "
                f"best_pearson={best['best_pearson']:.4f}"
            )
        lines.append("")

    lines.extend(["## Feature / Permutation Importance Diagnostics", ""])
    importance = report.get("feature_importance", {})
    for mode_name in ("mode_a_official_veatic_70_30", "mode_b_blocked_temporal_gap"):
        mode_importance = importance.get(mode_name, {})
        if not mode_importance:
            continue
        lines.extend([f"### {mode_name}", ""])
        for target_name in (
            "arousal__event_future_spike_1_3s",
            "arousal__event_future_drop_1_3s",
            "arousal__event_trend_reversal_1_3s",
            "arousal__event_peak_onset_1_3s",
            "arousal__event_recovery_onset_1_3s",
            "valence__event_future_drop_1_3s",
            "arousal__residual_future_p1s_persistence",
            "valence__residual_future_p1s_persistence",
        ):
            target_importance = mode_importance.get(target_name, {})
            if not target_importance:
                continue
            lines.append(f"#### {target_name}")
            for condition in (
                "autoregressive_plus_cortical_global",
                "autoregressive_plus_subcortical_roi",
                "autoregressive_plus_combined",
                "residualized_autoregressive_plus_combined",
            ):
                values = target_importance.get(condition)
                if not isinstance(values, dict):
                    continue
                top_perm = values.get("top_permutation_importance", [])
                top_feature = top_perm[0] if top_perm else None
                group_importance = values.get("positive_permutation_importance_by_group", {})
                if top_feature:
                    lines.append(
                        "- "
                        f"{condition}: metric={values['metric']}, base={values['base_score']:.4f}, "
                        f"top_perm={top_feature['feature']} ({top_feature['importance_delta']:.4f}), "
                        f"group_importance={group_importance}"
                    )
                else:
                    lines.append(
                        "- "
                        f"{condition}: metric={values['metric']}, base={values['base_score']:.4f}, "
                        f"group_importance={group_importance}"
                    )
            lines.append("")
    lines.extend(
        [
            "## Interpretation Guardrails",
            "- Do not claim neuro-additive value unless real neuro beats autoregressive-only and shuffled/random controls.",
            "- Do not make anatomical emotion claims from feature importance; use feature contribution language only.",
            "- This 20-video run is a gated diagnostic, not investor-grade proof.",
            "",
        ]
    )
    output.with_suffix(".summary.md").write_text("\n".join(lines), encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        return f"<callable:{getattr(value, '__name__', type(value).__name__)}>"
    return value


def fixed_rows(rows: list[dict[str, Any]], split_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    train = [row for row in rows if row["splits"][split_name] == "train"]
    test = [row for row in rows if row["splits"][split_name] == "test"]
    gap = sum(1 for row in rows if row["splits"][split_name] == "gap")
    return train, test, gap


def grouped_video_folds(rows: list[dict[str, Any]], fold_count: int) -> list[tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]]:
    grouped = rows_by_video(rows)
    video_ids = sorted(grouped, key=lambda item: int(item))
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    fold_count = min(fold_count, len(video_ids))
    folds = []
    for fold_index in range(fold_count):
        held_out = [
            video_id
            for offset, video_id in enumerate(video_ids)
            if offset % fold_count == fold_index
        ]
        held_out_set = set(held_out)
        test_rows = [row for video_id in held_out for row in grouped[video_id]]
        train_rows = [
            row
            for video_id, video_rows in grouped.items()
            if video_id not in held_out_set
            for row in video_rows
        ]
        folds.append((held_out, train_rows, test_rows))
    return folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="benchmarks/veatic/veatic_manifest_1hz.jsonl")
    parser.add_argument("--cache-dir", default="/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache")
    parser.add_argument("--output", default="benchmarks/veatic/veatic_neuro_benchmark_small.json")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--run-mode",
        choices=RUN_MODES,
        default="cortical_fast_default",
        help=(
            "cortical_fast_default evaluates compact cortical features only. "
            "full_research includes cortical+subcortical. "
            "subcortical_ablation preserves cortical/subcortical comparisons."
        ),
    )
    parser.add_argument(
        "--feature-mode",
        choices=FEATURE_MODES,
        default="cortical_global",
        help=(
            "No-reencode feature mode built from cached TRIBE outputs. "
            "cortical_global preserves the original 6-feature benchmark contract."
        ),
    )
    parser.add_argument(
        "--pca-backend",
        choices=("auto", "mps_power", "mps_gram", "cpu_svd"),
        default=os.environ.get("VEATIC_PCA_BACKEND", "auto"),
        help=(
            "PCA implementation for no-reencode PCA feature modes. "
            "mps_power uses MPS-native subspace iteration; mps_gram runs exact large "
            "cortical matrix products on MPS and the row-space eigensolve on CPU."
        ),
    )
    parser.add_argument(
        "--ridge-backend",
        choices=("auto", "mps_solve", "cpu_pinv"),
        default=os.environ.get("VEATIC_RIDGE_BACKEND", "auto"),
        help=(
            "Ridge implementation. mps_solve uses PyTorch MPS normal-equation solve; "
            "cpu_pinv preserves the original NumPy pseudo-inverse path."
        ),
    )
    parser.add_argument(
        "--grouped-video-folds",
        type=int,
        default=0,
        help=(
            "Use grouped video K-fold validation instead of full leave-video-out. "
            "Set to at least 5 for the confirmatory 124-video fallback when full "
            "leave-video-out is too expensive."
        ),
    )
    args = parser.parse_args()
    global PCA_BACKEND, RIDGE_BACKEND
    PCA_BACKEND = args.pca_backend
    RIDGE_BACKEND = args.ridge_backend

    manifest_rows = load_manifest(Path(args.manifest).expanduser().resolve())
    grouped_manifest = rows_by_video(manifest_rows)
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    complete_statuses = sorted(cache_dir.glob("*/cache_status.json"))
    accepted_rows: list[dict[str, Any]] = []
    feature_blocks: dict[str, list[np.ndarray]] = defaultdict(list)
    feature_metadata = []
    rejected = []
    selected_feature_sets = cache_feature_keys_for(args.feature_mode, args.run_mode)
    include_subcortical = args.run_mode in {"full_research", "subcortical_ablation"}
    for status_path in complete_statuses:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        video_id = str(status.get("video_id") or status_path.parent.name)
        if not status.get("complete"):
            rejected.append({"video_id": video_id, "reason": "cache_incomplete"})
            continue
        video_rows = grouped_manifest.get(video_id)
        if not video_rows:
            rejected.append({"video_id": video_id, "reason": "not_in_manifest"})
            continue
        try:
            feature_sets, metadata = load_cached_video_features(
                cache_dir,
                video_id,
                len(video_rows),
                include_subcortical=include_subcortical,
            )
        except Exception as exc:
            rejected.append({"video_id": video_id, "reason": str(exc)})
            continue
        accepted_rows.extend(video_rows)
        feature_metadata.append(metadata)
        for key in selected_feature_sets:
            value = feature_sets[key]
            if value.shape[1] == 0:
                continue
            feature_blocks[key].append(value)

    if not accepted_rows:
        raise RuntimeError("No complete VEATIC TRIBE cache entries were available for benchmarking.")
    feature_sets = {key: np.concatenate(value, axis=0) for key, value in feature_blocks.items()}
    if args.feature_mode == "cortical_global_delta":
        feature_sets["cortical_global_delta"] = temporal_dynamics_features(
            accepted_rows,
            feature_sets["cortical_global"],
            include_base=True,
        )
    if not feature_sets:
        raise RuntimeError(f"No feature sets available for run mode {args.run_mode}")
    official_train, official_test, official_gap = fixed_rows(accepted_rows, "official_70_30")
    blocked_train, blocked_test, blocked_gap = fixed_rows(accepted_rows, "blocked_temporal_gap")

    grouped_accepted = rows_by_video(accepted_rows)
    lvo_folds = []
    if args.grouped_video_folds:
        fold_specs = grouped_video_folds(accepted_rows, args.grouped_video_folds)
        video_generalization = {
            "validation_type": "grouped_video_k_fold",
            "requested_folds": int(args.grouped_video_folds),
            "actual_folds": len(fold_specs),
            "contract": "No train/test rows from the same video appear in the same fold.",
        }
    else:
        fold_specs = [
            (
                [video_id],
                [row for other_id, rows in grouped_accepted.items() if other_id != video_id for row in rows],
                grouped_accepted[video_id],
            )
            for video_id in sorted(grouped_accepted, key=lambda item: int(item))
        ]
        video_generalization = {
            "validation_type": "leave_video_out",
            "actual_folds": len(fold_specs),
            "contract": "Each fold holds out one complete video.",
        }
    for offset, (held_out_video_ids, train_rows, test_rows) in enumerate(fold_specs):
        fold = eval_split(
            accepted_rows,
            feature_sets,
            train_rows,
            test_rows,
            args.seed + 100 + offset,
            feature_mode=args.feature_mode,
        )
        fold["held_out_video_ids"] = held_out_video_ids
        if len(held_out_video_ids) == 1:
            fold["held_out_video_id"] = held_out_video_ids[0]
        lvo_folds.append(fold)
    lead_lag_sets, lead_lag_metadata = lead_lag_feature_sets(accepted_rows, feature_sets, args.feature_mode)

    report = {
        "schema_version": "veatic_neuro_temporal_dynamics_benchmark_v2",
        "run_mode": args.run_mode,
        "feature_mode": args.feature_mode,
        "subcortical_enabled": include_subcortical,
        "subcortical_policy": (
            "Subcortical disabled for cortical_fast_default. It remains available as explicit "
            "full_research/subcortical_ablation, but current OpenLAV/VEATIC evidence does not "
            "justify it as default compute."
            if not include_subcortical
            else "Subcortical explicitly enabled for research/ablation mode."
        ),
        "default_mode_rationale": {
            "why_disabled": (
                "Compact cortical features have been the most consistent useful signal so far; "
                "subcortical has not shown stable additive lift over cortical-only."
            ),
            "expected_speedup_runtime_reduction": (
                "Avoids the separate subcortical checkpoint pass and ROI projection. "
                "This should materially reduce wall time on uncached runs; cached benchmark-only reruns "
                "mainly save subcortical ROI projection time."
            ),
            "memory_benefit": (
                "Keeps only the cortical branch active during extraction, reducing MPS memory pressure "
                "and crash risk on Apple Silicon."
            ),
            "stability_expectation": (
                "Cortical-only default should improve run stability by removing the highest-risk second "
                "model stage from the default path."
            ),
        },
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "cache_dir": str(cache_dir),
        "backend_policy": {
            "pca_backend": PCA_BACKEND,
            "ridge_backend": RIDGE_BACKEND,
            "device_policy": (
                "PCA backend may use MPS when explicitly selected; ridge/logistic-style "
                "benchmark scoring follows the selected ridge backend."
            ),
            "seed": int(args.seed),
            "dtype": "float32 cached cortical features; NumPy float64 ridge metrics",
        },
        "accepted_videos": len({row["video_id"] for row in accepted_rows}),
        "accepted_rows": len(accepted_rows),
        "target_contract": {
            "base_targets": list(TARGETS),
            "derived_targets": list(DERIVED_TARGETS),
            "residual_after_persistence": (
                "Split-local residuals. Train residuals use only previous training "
                "labels from the same video; test residuals subtract persistence "
                "predictions fit from train rows only."
            ),
            "future_change": "y(t+horizon)-y(t), within the same video only.",
            "future_state": "y(t+horizon), within the same video only.",
            "delta_prev_1s": "y(t)-y(t-1), within the same video only.",
            "event_threshold": EVENT_THRESHOLD,
            "classification": (
                "Event predictions use train-prevalence thresholding on model scores "
                "inside each split."
            ),
            "autoregressive_features": [
                "y_t for future/delta/event targets only",
                "y_t-1",
                "y_t-2",
                "y_t-3",
                "rolling mean/std over available current/past values",
                "recent slope",
                "momentum",
                "recent min/max",
                "time index",
                "normalized video progress",
                "cyclical time features",
            ],
            "success_condition": (
                "Meaningful signal requires autoregressive_plus_real_neuro or "
                "residualized_autoregressive_plus_real_neuro to beat autoregressive-only "
                "and shuffled/random controls."
            ),
        },
        "rejected": rejected,
        "feature_sets": report_feature_sets_for(args.feature_mode, feature_sets),
        "cache_feature_sets_loaded": {key: int(value.shape[1]) for key, value in feature_sets.items()},
        "feature_metadata": feature_metadata,
        "lead_lag_analysis": {
            **lead_lag_analysis(accepted_rows, lead_lag_sets),
            "feature_mode_metadata": lead_lag_metadata,
        },
        "feature_importance": {
            "contract": {
                "method": "single split-local ridge coefficient magnitude plus one-pass test-set permutation importance",
                "positive_importance_delta": "For regression, permuted MAE - base MAE. For events, base F1 - permuted F1. Positive means the feature helped the score.",
                "scope": "diagnostic only; no feature selection is fit from test labels and no TRIBE extraction is rerun.",
            },
            "mode_a_official_veatic_70_30": permutation_importance_for_split(
                accepted_rows, feature_sets, official_train, official_test, args.seed + 300
            ),
            "mode_b_blocked_temporal_gap": permutation_importance_for_split(
                accepted_rows, feature_sets, blocked_train, blocked_test, args.seed + 301
            ),
        },
        "modes": {
            "mode_a_official_veatic_70_30": {
                "gap_rows": official_gap,
                **eval_split(
                    accepted_rows,
                    feature_sets,
                    official_train,
                    official_test,
                    args.seed,
                    feature_mode=args.feature_mode,
                ),
            },
            "mode_b_blocked_temporal_gap": {
                "gap_rows": blocked_gap,
                **eval_split(
                    accepted_rows,
                    feature_sets,
                    blocked_train,
                    blocked_test,
                    args.seed + 1,
                    feature_mode=args.feature_mode,
                ),
            },
            "mode_c_leave_video_out": {
                **video_generalization,
                "aggregate": aggregate_folds(lvo_folds),
                "folds": lvo_folds,
            },
        },
        "interpretation_warning": (
            "This is a small gated run. Treat it as pipeline validation and control "
            "diagnostics, not evidence of general neuro-additive performance."
        ),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = json_safe(report)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_summary(report, output)
    print(json.dumps({"output": str(output), "accepted_videos": report["accepted_videos"], "accepted_rows": report["accepted_rows"]}, indent=2))


if __name__ == "__main__":
    main()
