"""Phase 4 dense AGAIN 2Hz train-only PCA bridge benchmark.

This module consumes the completed H100 V-JEPA 2.1 / TRIBE v2 cache only. It
never decodes raw videos, never runs V-JEPA, and never runs TRIBE. PCA is fit
inside each target/protocol/fold row set and then applied to that fold's rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

try:
    import mlx.core as mx
except Exception:  # pragma: no cover - only exercised where MLX is absent.
    mx = None

from backend.scripts.again_dense_2hz_benchmark import (
    AR_FEATURE_COLUMNS,
    DEFAULT_DENSE_ROOT,
    DEFAULT_RIDGE_ALPHA_GRID,
    QUALITY_FEATURE_COLUMNS,
    ROW_RATE_HZ,
    TARGET_SPECS,
    TIME_FEATURE_COLUMNS,
    TargetSpec,
    blocked_temporal_split,
    clean_json,
    default_output_root,
    feature_matrix,
    grouped_video_splits,
    inner_validation_relative_split,
    load_labels,
    load_or_build_temporal_diagnostic_features,
    local_npz_path,
    metric_row,
    parse_alpha_grid,
    regression_metric_row,
    mlx_ridge_fit_predict,
    target_base_mask,
    threshold_labels,
    utc_stamp,
    validate_split,
    write_csv,
    write_json,
)


PHASE4_SCHEMA_VERSION = "again_dense_2hz_phase4_pca_bridge_v1"
DEFAULT_WIDTHS = (64, 128, 192, 256)
DEFAULT_FEATURE_FAMILIES = ("current", "delta", "pca_then_temporal", "temporal_then_pca")
DEFAULT_PROTOCOLS = ("grouped_video", "blocked_temporal_70_30")
DEFAULT_RANDOM_SEED = 20260625
CORTICAL_WIDTH = 20484
TEMPORAL_MEAN_WINDOWS = {
    "causal_past_0p5s_mean": 1,
    "causal_past_1s_mean": 2,
    "causal_past_2s_mean": 4,
    "causal_past_3s_mean": 6,
}
TEMPORAL_SLOPE_WINDOWS = {
    "causal_past_1s_slope": 2,
    "causal_past_2s_slope": 4,
    "causal_past_3s_slope": 6,
}
TEMPORAL_STD_WINDOWS = {
    "causal_past_1s_std": 2,
    "causal_past_2s_std": 4,
    "causal_past_3s_std": 6,
}
TEMPORAL_THEN_PCA_WINDOWS = {"temporal_mean_2s_then_pca": 4}
PHASE4_MLX_CG_MIN_WIDTH = 96


def assert_again_only_output_path(path: Path) -> None:
    """Keep dense AGAIN outputs out of VEATIC or ambiguous directories."""
    parts = {part.lower() for part in path.expanduser().parts}
    if "veatic" in parts:
        raise ValueError(f"AGAIN pipeline output cannot target a VEATIC path: {path}")
    if "again" not in str(path).lower():
        raise ValueError(f"AGAIN pipeline output path must be clearly AGAIN scoped: {path}")


@dataclass(frozen=True)
class SplitSpec:
    target: TargetSpec
    protocol: str
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    target_threshold: float

    @property
    def key(self) -> str:
        return f"{self.target.name}__{self.protocol}__fold{self.fold}"


@dataclass(frozen=True)
class PcaFitSpec:
    split: SplitSpec
    base_family: str
    source_family: str
    valid_mask: np.ndarray
    width: int
    seed: int

    @property
    def fit_key(self) -> str:
        return f"{self.split.key}__{self.source_family}__pca{self.width}"


@dataclass
class PcaFitResult:
    spec: PcaFitSpec
    score_path: Path
    component_path: Path
    meta_path: Path
    scores: np.ndarray
    metadata: dict[str, Any]


def external_phase4_root() -> Path | None:
    candidate = Path("/Volumes/onn. Drive/Neural Bridge/outputs")
    if candidate.exists() and os.access(candidate, os.W_OK):
        return candidate
    return None


def phase4_output_root() -> Path:
    external = external_phase4_root()
    if external is not None:
        return external / f"again_dense_2hz_phase4_pca_bridge_{utc_stamp()}"
    return default_output_root("again_dense_2hz_phase4_pca_bridge")


def parse_csv_set(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def parse_widths(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"Invalid PCA widths: {text}")
    return tuple(sorted(set(values)))


def git_metadata() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = run(["git", "status", "--short"])
    return {
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_commit_short": run(["git", "rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(status),
        "git_status_short": status or "",
    }


def array_digest(arr: np.ndarray, *, digest_size: int = 16) -> str:
    contiguous = np.ascontiguousarray(arr)
    return hashlib.blake2b(contiguous.view(np.uint8), digest_size=digest_size).hexdigest()


def text_digest(text: str, *, digest_size: int = 16) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=digest_size).hexdigest()


def file_digest(path: Path, *, max_bytes: int | None = None, digest_size: int = 16) -> str:
    digest = hashlib.blake2b(digest_size=digest_size)
    with path.open("rb") as handle:
        remaining = max_bytes
        while True:
            size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            if size <= 0:
                break
            chunk = handle.read(size)
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return digest.hexdigest()


def ensure_phase4_cache_only(dense_root: Path) -> None:
    required = [
        dense_root / "row_index.parquet",
        dense_root / "labels_aligned_2hz.parquet",
        dense_root / "per_video",
        dense_root / "global_run_metadata.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Dense H100 cache is incomplete; missing {missing}")
    metadata = json.loads((dense_root / "global_run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("forbid_vjepa") is not True:
        raise ValueError("Dense metadata does not carry forbid_vjepa=true")
    if metadata.get("cache_only") is not True:
        raise ValueError("Dense metadata does not carry cache_only=true")


def phase4_dirs(output_root: Path) -> dict[str, Path]:
    dirs = {
        "features": output_root / "features",
        "components": output_root / "pca_components",
        "score_parts": output_root / "score_parts",
        "metrics": output_root / "metrics",
        "promotion": output_root / "promotion",
        "diagnostics": output_root / "diagnostics",
        "plots": output_root / "plots",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_phase3_summary(report_path: Path | None = None) -> dict[str, dict[str, float]]:
    if report_path is None:
        candidates = sorted(Path("reports").glob("again_dense_2hz_raw_cortical_vs_ar_*.md"))
        report_path = candidates[-1] if candidates else None
    if report_path is None or not report_path.exists():
        return {}
    out: dict[str, dict[str, float]] = {}
    for line in report_path.read_text(encoding="utf-8").splitlines():
        if " / `grouped_video` / `" not in line or "PR-AUC `" not in line:
            continue
        try:
            parts = line.split("`")
            target = parts[1]
            lane = parts[5]
            pr_text = line.split("PR-AUC `", 1)[1].split("%`", 1)[0]
            out.setdefault(target, {})[lane] = float(pr_text) / 100.0
        except Exception:
            continue
    return out


def build_split_specs(
    df: pd.DataFrame,
    *,
    protocols: Sequence[str],
    n_splits: int,
    target_specs: Sequence[TargetSpec] = TARGET_SPECS,
) -> list[SplitSpec]:
    selected_protocols = set(protocols)
    specs: list[SplitSpec] = []
    for target in target_specs:
        base_mask = target_base_mask(df, target)
        raw_splits: list[tuple[str, int, np.ndarray, np.ndarray]] = []
        if "grouped_video" in selected_protocols:
            raw_splits.extend(grouped_video_splits(df, base_mask, n_splits=n_splits))
        if "blocked_temporal_70_30" in selected_protocols:
            raw_splits.extend(blocked_temporal_split(df, base_mask))
        values = df[target.value_column].to_numpy(dtype=np.float64)
        for protocol, fold, train_idx, test_idx in raw_splits:
            validate_split(df, protocol, train_idx, test_idx)
            train_mask = np.zeros(len(df), dtype=bool)
            test_mask = np.zeros(len(df), dtype=bool)
            train_mask[train_idx] = True
            test_mask[test_idx] = True
            y_train, y_test, target_threshold = threshold_labels(values, train_mask, test_mask, target)
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            specs.append(
                SplitSpec(
                    target=target,
                    protocol=protocol,
                    fold=fold,
                    train_idx=np.asarray(train_idx, dtype=np.int64),
                    test_idx=np.asarray(test_idx, dtype=np.int64),
                    y_train=y_train,
                    y_test=y_test,
                    target_threshold=target_threshold,
                )
            )
    return specs


def split_fingerprint(split: SplitSpec) -> dict[str, Any]:
    return {
        "target_name": split.target.name,
        "protocol": split.protocol,
        "fold": split.fold,
        "train_rows": int(split.train_idx.size),
        "test_rows": int(split.test_idx.size),
        "train_idx_digest": array_digest(split.train_idx),
        "test_idx_digest": array_digest(split.test_idx),
        "target_threshold_train_only": split.target_threshold,
    }


def score_part_metadata(
    split: SplitSpec,
    *,
    widths: Sequence[int],
    families: Sequence[str],
    ridge_alpha_grid: Sequence[float] | None,
    ridge_alpha: float,
    random_seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": PHASE4_SCHEMA_VERSION,
        "kind": "phase4_split_score_part",
        "split": split_fingerprint(split),
        "widths": [int(width) for width in widths],
        "feature_families": list(families),
        "ridge_alpha_grid": [float(alpha) for alpha in ridge_alpha_grid] if ridge_alpha_grid is not None else None,
        "ridge_alpha": float(ridge_alpha),
        "ridge_alpha_selection": "train_only_inner_validation" if ridge_alpha_grid is not None else "fixed_cli_alpha",
        "random_seed": int(random_seed),
    }


def score_part_paths(output_root: Path, split: SplitSpec) -> tuple[Path, Path]:
    score_dir = output_root / "score_parts"
    return score_dir / f"{split.key}__fold_metrics.csv", score_dir / f"{split.key}__fold_metrics_metadata.json"


def load_score_part_if_valid(output_root: Path, split: SplitSpec, expected_metadata: dict[str, Any]) -> list[dict[str, Any]] | None:
    csv_path, meta_path = score_part_paths(output_root, split)
    if not csv_path.exists() or not meta_path.exists():
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if metadata != expected_metadata:
        return None
    part = pd.read_csv(csv_path)
    if part.empty:
        return None
    required = {"target_name", "validation_protocol", "fold", "model_lane", "feature_name", "pca_width", "pr_auc"}
    if missing := sorted(required - set(part.columns)):
        raise ValueError(f"Score part {csv_path} is missing columns: {missing}")
    return part.to_dict(orient="records")


def write_score_part(output_root: Path, split: SplitSpec, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    csv_path, meta_path = score_part_paths(output_root, split)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp_csv, index=False)
    tmp_meta.write_text(json.dumps(clean_json(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_csv.replace(csv_path)
    tmp_meta.replace(meta_path)


def video_start_indices(df: pd.DataFrame) -> np.ndarray:
    starts = np.zeros(len(df), dtype=np.int64)
    for _video, group in df.groupby("video_id", sort=False):
        idx = group.index.to_numpy(dtype=np.int64)
        starts[idx] = idx[0]
    return starts


def same_video_previous_valid(df: pd.DataFrame) -> np.ndarray:
    valid = np.zeros(len(df), dtype=bool)
    for _video, group in df.groupby("video_id", sort=False):
        idx = group.index.to_numpy(dtype=np.int64)
        if len(idx) > 1:
            valid[idx[1:]] = True
    return valid


def load_or_build_cortical_memmap(
    dense_root: Path,
    df: pd.DataFrame,
    *,
    output_root: Path | None = None,
    force: bool = False,
) -> np.ndarray:
    if output_root is None:
        derived = dense_root / "_derived"
    else:
        derived = output_root / "cache"
    derived.mkdir(parents=True, exist_ok=True)
    path = derived / "cortical_prediction_rows_fp16.npy"
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("rows") == len(df) and meta.get("width") == CORTICAL_WIDTH:
            return np.load(path, mmap_mode="r")
    tmp_path = path.with_suffix(".tmp.npy")
    arr = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float16, shape=(len(df), CORTICAL_WIDTH))
    for video_id, group in df.groupby("video_id", sort=False):
        npz_path = local_npz_path(dense_root, str(video_id), "tribe_v2_cortical_predictions.npz")
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing cortical cache for {video_id}: {npz_path}")
        with np.load(npz_path) as npz:
            cortical = np.asarray(npz["cortical_prediction"], dtype=np.float16)
            times = np.asarray(npz["time_seconds"], dtype=np.float64)
        if cortical.shape != (len(group), CORTICAL_WIDTH):
            raise ValueError(f"Cortical shape mismatch for {video_id}: {cortical.shape}")
        if not np.allclose(times, group["time_seconds"].to_numpy(dtype=np.float64), atol=1e-6):
            raise ValueError(f"Time mismatch for {video_id}")
        arr[group.index.to_numpy(dtype=np.int64)] = cortical
    arr.flush()
    tmp_path.replace(path)
    write_json(
        meta_path,
        {
            "schema_version": PHASE4_SCHEMA_VERSION,
            "source": "per_video/<video_id>/tribe_v2_cortical_predictions.npz:cortical_prediction",
            "rows": len(df),
            "width": CORTICAL_WIDTH,
            "dtype": "float16",
            "cache_only": True,
            "vjepa_encoding_run": False,
            "tribe_encoding_run": False,
        },
    )
    return np.load(path, mmap_mode="r")


class CorticalVariantAccessor:
    def __init__(self, cortical: np.ndarray, df: pd.DataFrame, *, base_family: str):
        self.cortical = cortical
        self.df = df
        self.base_family = base_family
        self.starts = video_start_indices(df)
        self.prev_valid = same_video_previous_valid(df)
        if base_family == "current":
            self.valid_mask = np.ones(len(df), dtype=bool)
            self.description = "current-row cortical_prediction"
        elif base_family == "delta":
            self.valid_mask = self.prev_valid.copy()
            self.description = "current row minus previous valid same-video row"
        elif base_family == "temporal_mean_2s":
            self.valid_mask = np.ones(len(df), dtype=bool)
            self.description = "raw cortical causal trailing 2s mean before PCA"
        else:
            raise ValueError(f"Unknown cortical base family: {base_family}")

    def batch(self, indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(indices, dtype=np.int64)
        if self.base_family == "current":
            return np.asarray(self.cortical[idx], dtype=np.float32)
        if self.base_family == "delta":
            prev = idx - 1
            out = np.asarray(self.cortical[idx], dtype=np.float32) - np.asarray(self.cortical[prev], dtype=np.float32)
            return out
        rows: list[np.ndarray] = []
        for row_idx in idx:
            start = max(int(self.starts[row_idx]), int(row_idx) - 3)
            rows.append(np.asarray(self.cortical[start : row_idx + 1], dtype=np.float32).mean(axis=0))
        return np.stack(rows, axis=0).astype(np.float32, copy=False)


def row_batches(indices: np.ndarray, *, batch_size: int) -> Iterable[np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64)
    for start in range(0, len(idx), batch_size):
        yield idx[start : start + batch_size]


def streaming_mean_std(
    accessor: CorticalVariantAccessor,
    train_idx: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sum_x = np.zeros(CORTICAL_WIDTH, dtype=np.float64)
    sum_x2 = np.zeros(CORTICAL_WIDTH, dtype=np.float64)
    n = 0
    nan_count = 0
    inf_count = 0
    for batch_idx in row_batches(train_idx, batch_size=batch_size):
        x = accessor.batch(batch_idx).astype(np.float32, copy=False)
        nan_count += int(np.isnan(x).sum())
        inf_count += int(np.isinf(x).sum())
        if nan_count or inf_count:
            raise ValueError(f"Non-finite cortical values in {accessor.base_family}: nan={nan_count} inf={inf_count}")
        sum_x += x.sum(axis=0, dtype=np.float64)
        sum_x2 += np.square(x, dtype=np.float64).sum(axis=0, dtype=np.float64)
        n += x.shape[0]
    if n < 2:
        raise ValueError(f"Need at least 2 train rows for PCA; got {n}")
    mean = (sum_x / n).astype(np.float32)
    var = np.maximum(sum_x2 / n - np.square(mean.astype(np.float64)), 1e-6)
    std = np.sqrt(var).astype(np.float32)
    return mean, std, {"train_rows_for_stats": n, "input_nan_count": nan_count, "input_inf_count": inf_count}


def standardize_batch(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float32, copy=False) - mean) / std).astype(np.float32, copy=False)


def matmul_gpu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if mx is None:
        return (a @ b).astype(np.float32, copy=False)
    aa = mx.array(a.astype(np.float32, copy=False), dtype=mx.float32)
    bb = mx.array(b.astype(np.float32, copy=False), dtype=mx.float32)
    out = aa @ bb
    mx.eval(out)
    return np.asarray(out, dtype=np.float32)


def decision_threshold_from_train(y_train: np.ndarray, train_scores: np.ndarray) -> float:
    """Train-only decision threshold selected for F1, matching Phase 1-3 semantics."""
    if len(np.unique(y_train)) < 2:
        return float(np.median(train_scores))
    candidates = np.unique(np.quantile(train_scores, np.linspace(0.05, 0.95, 19)))
    best = (float("-inf"), float(candidates[0]))
    for threshold in candidates:
        pred = (train_scores >= threshold).astype(int)
        score = f1_score(y_train, pred, zero_division=0)
        if score > best[0]:
            best = (float(score), float(threshold))
    return best[1]


def phase4_scale_fit_predict(train_x: np.ndarray, test_x: np.ndarray, train_y: np.ndarray, *, alpha: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Hybrid ridge solve for Phase 4 compressed features.

    For larger compressed feature matrices, MLX conjugate gradient is faster on
    this machine than a direct NumPy solve while preserving the same ridge
    objective up to tiny floating-point differences. For smaller systems the
    direct solve is still cheaper, so we keep a width-based switch.
    """
    train_x = np.nan_to_num(train_x.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    test_x = np.nan_to_num(test_x.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    if mx is not None and train_x.shape[1] >= PHASE4_MLX_CG_MIN_WIDTH:
        train_scores, test_scores, info = mlx_ridge_fit_predict(train_x, test_x, train_y, alpha=alpha)
        info = dict(info)
        info["ridge_backend"] = "phase4_mlx_primal_conjugate_gradient"
        info["phase4_solver_switch_width"] = PHASE4_MLX_CG_MIN_WIDTH
        return train_scores, test_scores, info
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    xtr = (train_x - mean) / std
    xte = (test_x - mean) / std
    y = train_y.astype(np.float32, copy=False)
    y_mean = float(np.mean(y))
    yc = y - y_mean
    solver = "numpy_primal_closed_form_solve"
    try:
        xtx = xtr.T @ xtr
        xtx = xtx.astype(np.float64, copy=False)
        xtx.flat[:: xtx.shape[0] + 1] += float(alpha)
        xty = (xtr.T @ yc).astype(np.float64, copy=False)
        weights = np.linalg.solve(xtx, xty).astype(np.float32)
        train_scores = xtr @ weights + y_mean
        test_scores = xte @ weights + y_mean
    except Exception:
        xtx = xtr.T @ xtr
        xtx = xtx.astype(np.float64, copy=False)
        xtx.flat[:: xtx.shape[0] + 1] += float(alpha)
        xty = (xtr.T @ yc).astype(np.float64, copy=False)
        weights = np.linalg.lstsq(xtx, xty, rcond=None)[0].astype(np.float32)
        train_scores = xtr @ weights + y_mean
        test_scores = xte @ weights + y_mean
        solver = "numpy_primal_closed_form_lstsq_fallback"
    return (
        train_scores.astype(np.float32, copy=False),
        test_scores.astype(np.float32, copy=False),
        {
            "ridge_backend": solver,
            "ridge_alpha": float(alpha),
            "feature_width": int(train_x.shape[1]),
            "ridge_iterations": 0,
            "mlx_available": mx is not None,
            "phase4_solver_switch_width": PHASE4_MLX_CG_MIN_WIDTH,
        },
    )


def phase4_select_alpha_train_only(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha_grid: Sequence[float],
    inner_split_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, str]],
) -> dict[str, Any]:
    valid_grid = [float(alpha) for alpha in alpha_grid if float(alpha) > 0 and math.isfinite(float(alpha))]
    if not valid_grid:
        raise ValueError("Empty ridge alpha grid")
    cache_key = (array_digest(train_idx.astype(np.int64, copy=False)), array_digest(y_train.astype(np.int8, copy=False)))
    cached = inner_split_cache.get(cache_key)
    if cached is None:
        cached = inner_validation_relative_split(df, train_idx, y_train)
        inner_split_cache[cache_key] = cached
    inner_train, inner_val, strategy = cached
    rows = []
    best_alpha = valid_grid[0]
    best_score = float("-inf")
    for alpha in valid_grid:
        _train_scores, val_scores, info = phase4_scale_fit_predict(
            x_train[inner_train],
            x_train[inner_val],
            y_train[inner_train],
            alpha=alpha,
        )
        if len(np.unique(y_train[inner_val])) < 2:
            score = math.nan
        else:
            score = float(average_precision_score(y_train[inner_val], val_scores))
        rows.append({"alpha": alpha, "inner_pr_auc": score, **info})
        comparable = score if math.isfinite(score) else float("-inf")
        if comparable > best_score:
            best_score = comparable
            best_alpha = alpha
    return {
        "selected_alpha": float(best_alpha),
        "inner_validation_pr_auc": best_score if math.isfinite(best_score) else math.nan,
        "inner_validation_strategy": strategy,
        "alpha_grid": valid_grid,
        "alpha_selection_rows": rows,
        "phase4_inner_split_cache": True,
    }


def streaming_randomized_pca_fit(
    accessor: CorticalVariantAccessor,
    train_idx: np.ndarray,
    all_idx: np.ndarray,
    *,
    width: int,
    seed: int,
    output_root: Path,
    fit_key: str,
    batch_size: int,
    oversampling: int,
    power_iterations: int,
) -> PcaFitResult:
    start_time = time.time()
    train_idx = np.asarray(train_idx, dtype=np.int64)
    all_idx = np.asarray(all_idx, dtype=np.int64)
    q = int(min(CORTICAL_WIDTH, width + oversampling))
    component_dir = output_root / "pca_components"
    feature_dir = output_root / "features"
    component_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    score_path = feature_dir / f"{fit_key}__scores_w{width}.npy"
    component_path = component_dir / f"{fit_key}__components_w{width}.npz"
    meta_path = component_dir / f"{fit_key}__metadata.json"
    expected_train_digest = array_digest(train_idx)
    expected_transform_digest = array_digest(all_idx)
    if score_path.exists() and component_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        scores = np.load(score_path, mmap_mode="r")
        cache_valid = (
            meta.get("schema_version") == PHASE4_SCHEMA_VERSION
            and meta.get("pca_width") == width
            and meta.get("source_family") == accessor.base_family
            and meta.get("random_seed") == seed
            and meta.get("oversampling") == oversampling
            and meta.get("power_iterations") == int(power_iterations)
            and meta.get("train_idx_digest") == expected_train_digest
            and meta.get("transform_idx_digest") == expected_transform_digest
            and tuple(scores.shape) == (len(all_idx), width)
        )
        if cache_valid:
            meta = dict(meta)
            meta["cache_hit"] = True
            return PcaFitResult(
                spec=PcaFitSpec(
                    split=SplitSpec(
                        target=TargetSpec("placeholder", "placeholder", "placeholder"),
                        protocol="placeholder",
                        fold=-1,
                        train_idx=train_idx,
                        test_idx=np.array([], dtype=np.int64),
                        y_train=np.array([], dtype=np.int64),
                        y_test=np.array([], dtype=np.int64),
                        target_threshold=math.nan,
                    ),
                    base_family=accessor.base_family,
                    source_family=accessor.base_family,
                    valid_mask=accessor.valid_mask,
                    width=width,
                    seed=seed,
                ),
                score_path=score_path,
                component_path=component_path,
                meta_path=meta_path,
                scores=scores,
                metadata=meta,
            )
    mean, std, stats_meta = streaming_mean_std(accessor, train_idx, batch_size=batch_size)
    rng = np.random.default_rng(seed)
    omega = rng.normal(size=(CORTICAL_WIDTH, q)).astype(np.float32)
    y = np.empty((len(train_idx), q), dtype=np.float32)
    cursor = 0
    for batch_idx in row_batches(train_idx, batch_size=batch_size):
        x = standardize_batch(accessor.batch(batch_idx), mean, std)
        y[cursor : cursor + len(batch_idx)] = matmul_gpu(x, omega)
        cursor += len(batch_idx)
    q_mat, _ = np.linalg.qr(y, mode="reduced")
    del y
    for _ in range(max(0, int(power_iterations))):
        z = np.zeros((CORTICAL_WIDTH, q), dtype=np.float32)
        cursor = 0
        for batch_idx in row_batches(train_idx, batch_size=batch_size):
            x = standardize_batch(accessor.batch(batch_idx), mean, std)
            q_batch = q_mat[cursor : cursor + len(batch_idx)].astype(np.float32, copy=False)
            z += matmul_gpu(x.T, q_batch)
            cursor += len(batch_idx)
        y_power = np.empty((len(train_idx), q), dtype=np.float32)
        cursor = 0
        for batch_idx in row_batches(train_idx, batch_size=batch_size):
            x = standardize_batch(accessor.batch(batch_idx), mean, std)
            y_power[cursor : cursor + len(batch_idx)] = matmul_gpu(x, z)
            cursor += len(batch_idx)
        q_mat, _ = np.linalg.qr(y_power, mode="reduced")
        del y_power, z
    b = np.zeros((q, CORTICAL_WIDTH), dtype=np.float32)
    cursor = 0
    for batch_idx in row_batches(train_idx, batch_size=batch_size):
        x = standardize_batch(accessor.batch(batch_idx), mean, std)
        q_batch = q_mat[cursor : cursor + len(batch_idx)].astype(np.float32, copy=False)
        b += matmul_gpu(q_batch.T, x)
        cursor += len(batch_idx)
    _u, singular_values, vt = np.linalg.svd(b, full_matrices=False)
    components = vt[:width].astype(np.float32, copy=False)
    singular_values = singular_values[:width].astype(np.float32, copy=False)
    scores_tmp = score_path.with_suffix(".tmp.npy")
    scores = np.lib.format.open_memmap(scores_tmp, mode="w+", dtype=np.float32, shape=(len(all_idx), width))
    cursor = 0
    comp_t = components.T.astype(np.float32, copy=False)
    for batch_idx in row_batches(all_idx, batch_size=batch_size):
        x = standardize_batch(accessor.batch(batch_idx), mean, std)
        scores[cursor : cursor + len(batch_idx)] = matmul_gpu(x, comp_t)
        cursor += len(batch_idx)
    scores.flush()
    scores_tmp.replace(score_path)
    explained_variance = np.square(singular_values.astype(np.float64)) / max(1, len(train_idx) - 1)
    total_variance = float(CORTICAL_WIDTH * len(train_idx) / max(1, len(train_idx) - 1))
    explained_ratio = (explained_variance / total_variance).astype(np.float64)
    component_checksum = array_digest(components)
    explained_ratio_list = [float(x) for x in explained_ratio]
    np.savez(
        component_path,
        components=components,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        singular_values=singular_values,
        explained_variance_ratio=explained_ratio.astype(np.float32),
        train_idx=train_idx.astype(np.int64),
    )
    metadata = {
        "schema_version": PHASE4_SCHEMA_VERSION,
        "pca_algorithm": "streaming_randomized_svd_mlx_matmul",
        "pca_backend": "mlx_gpu_matmul_numpy_qr_svd" if mx is not None else "numpy_cpu_fallback",
        "pca_width": width,
        "random_seed": seed,
        "oversampling": oversampling,
        "power_iterations": int(power_iterations),
        "source_family": accessor.base_family,
        "source_description": accessor.description,
        "train_row_count": int(len(train_idx)),
        "transform_row_count": int(len(all_idx)),
        "train_idx_digest": expected_train_digest,
        "transform_idx_digest": expected_transform_digest,
        "centering_scaling_policy": "train_only_zscore_before_pca",
        "explained_variance_ratio_sum": float(np.sum(explained_ratio)),
        "top_explained_variance_ratio": [float(x) for x in explained_ratio[: min(10, len(explained_ratio))]],
        "explained_variance_ratio": explained_ratio_list,
        "component_checksum": component_checksum,
        "component_path": str(component_path),
        "score_path": str(score_path),
        "score_checksum_first_64mb": file_digest(score_path, max_bytes=64 * 1024 * 1024),
        "runtime_seconds": time.time() - start_time,
        "cache_only": True,
        "vjepa_encoding_run": False,
        "tribe_encoding_run": False,
        "pca_run": True,
        "cache_hit": False,
        **stats_meta,
    }
    write_json(meta_path, metadata)
    return PcaFitResult(
        spec=PcaFitSpec(
            split=SplitSpec(
                target=TargetSpec("placeholder", "placeholder", "placeholder"),
                protocol="placeholder",
                fold=-1,
                train_idx=train_idx,
                test_idx=np.array([], dtype=np.int64),
                y_train=np.array([], dtype=np.int64),
                y_test=np.array([], dtype=np.int64),
                target_threshold=math.nan,
            ),
            base_family=accessor.base_family,
            source_family=accessor.base_family,
            valid_mask=accessor.valid_mask,
            width=width,
            seed=seed,
        ),
        score_path=score_path,
        component_path=component_path,
        meta_path=meta_path,
        scores=np.load(score_path, mmap_mode="r"),
        metadata=metadata,
    )


def fit_or_load_pca(
    split: SplitSpec,
    accessor: CorticalVariantAccessor,
    *,
    output_root: Path,
    width: int,
    seed: int,
    batch_size: int,
    oversampling: int,
    power_iterations: int,
) -> PcaFitResult:
    valid_train = split.train_idx[accessor.valid_mask[split.train_idx]]
    valid_test = split.test_idx[accessor.valid_mask[split.test_idx]]
    all_idx = np.concatenate([valid_train, valid_test]).astype(np.int64)
    fit_key = f"{split.key}__{accessor.base_family}"
    result = streaming_randomized_pca_fit(
        accessor,
        valid_train,
        all_idx,
        width=width,
        seed=seed,
        output_root=output_root,
        fit_key=fit_key,
        batch_size=batch_size,
        oversampling=oversampling,
        power_iterations=power_iterations,
    )
    spec = PcaFitSpec(
        split=split,
        base_family=accessor.base_family,
        source_family=accessor.base_family,
        valid_mask=accessor.valid_mask,
        width=width,
        seed=seed,
    )
    result.spec = spec
    result.metadata.update(
        {
            "target_name": split.target.name,
            "validation_protocol": split.protocol,
            "fold": split.fold,
            "eval_row_count": int(valid_test.size),
            "dropped_train_rows": int(split.train_idx.size - valid_train.size),
            "dropped_eval_rows": int(split.test_idx.size - valid_test.size),
            "drop_reason": "invalid first row for delta family" if accessor.base_family == "delta" else "",
        }
    )
    write_json(result.meta_path, result.metadata)
    return result


def score_subset(result: PcaFitResult, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid_train = train_idx[result.spec.valid_mask[train_idx]]
    valid_test = test_idx[result.spec.valid_mask[test_idx]]
    all_idx = np.concatenate([valid_train, valid_test]).astype(np.int64)
    if not np.array_equal(all_idx, np.concatenate([valid_train, valid_test])):
        raise AssertionError("Unexpected row-order mismatch")
    scores = np.asarray(result.scores, dtype=np.float32)
    train_scores = scores[: len(valid_train)]
    test_scores = scores[len(valid_train) :]
    return valid_train, valid_test, train_scores, test_scores


def causal_reduce_scores(
    scores_by_row: dict[int, np.ndarray],
    indices: np.ndarray,
    starts: np.ndarray,
    *,
    width: int,
    window_rows: int,
    mode: str,
) -> np.ndarray:
    out = np.zeros((len(indices), width), dtype=np.float32)
    for i, row_idx in enumerate(indices.astype(np.int64)):
        start = max(int(starts[row_idx]), int(row_idx) - window_rows + 1)
        rows = [scores_by_row[j][:width] for j in range(start, int(row_idx) + 1) if j in scores_by_row]
        if not rows:
            rows = [scores_by_row[int(row_idx)][:width]]
        stack = np.stack(rows, axis=0)
        if mode == "mean":
            out[i] = stack.mean(axis=0)
        elif mode == "slope":
            denom = max(1, stack.shape[0] - 1)
            out[i] = (stack[-1] - stack[0]) / float(denom)
        elif mode == "std":
            out[i] = stack.std(axis=0)
        else:
            raise ValueError(f"Unknown causal reduce mode: {mode}")
    return out


def build_scores_lookup(indices: np.ndarray, scores: np.ndarray) -> dict[int, np.ndarray]:
    return {int(idx): np.asarray(score, dtype=np.float32) for idx, score in zip(indices, scores)}


def fit_and_score_lane(
    df: pd.DataFrame,
    split: SplitSpec,
    lane: str,
    model_train_idx: np.ndarray,
    model_test_idx: np.ndarray,
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
    continuous_test_values: np.ndarray,
    *,
    alpha_grid: Sequence[float] | None,
    alpha: float,
    random_seed: int,
    inner_split_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, str]],
    extra: dict[str, Any],
) -> dict[str, Any]:
    if train_x.shape[0] != train_y.shape[0] or test_x.shape[0] != test_y.shape[0]:
        raise ValueError(f"Lane {lane} row mismatch: train {train_x.shape}/{train_y.shape}, test {test_x.shape}/{test_y.shape}")
    if train_x.shape[0] < 2 or test_x.shape[0] < 2:
        raise ValueError(f"Lane {lane} has insufficient rows")
    if alpha_grid is None:
        selection = {
            "selected_alpha": float(alpha),
            "inner_validation_pr_auc": math.nan,
            "inner_validation_strategy": "fixed_cli_alpha",
            "alpha_grid": [float(alpha)],
            "alpha_selection_rows": [],
        }
    else:
        selection = phase4_select_alpha_train_only(df, model_train_idx, train_x, train_y, alpha_grid, inner_split_cache)
    train_scores, test_scores, fit_info = phase4_scale_fit_predict(train_x, test_x, train_y, alpha=selection["selected_alpha"])
    decision_threshold = decision_threshold_from_train(train_y, train_scores)
    row = {
        "schema_version": PHASE4_SCHEMA_VERSION,
        "target_name": split.target.name,
        "target_value_column": split.target.value_column,
        "target_mask_column": split.target.mask_column,
        "target_threshold_train_only": split.target_threshold,
        "target_threshold_quantile": split.target.quantile,
        "target_transform": split.target.transform,
        "validation_protocol": split.protocol,
        "fold": split.fold,
        "model_lane": lane,
        "n_train": int(train_x.shape[0]),
        "n_test": int(test_x.shape[0]),
        "train_videos": int(df.loc[model_train_idx, "video_id"].nunique()),
        "test_videos": int(df.loc[model_test_idx, "video_id"].nunique()),
        "train_event_count": int(np.sum(train_y)),
        "test_event_count": int(np.sum(test_y)),
        "train_positive_rate": float(np.mean(train_y)),
        "test_positive_rate": float(np.mean(test_y)),
        "feature_width": int(train_x.shape[1]),
        "decision_threshold_train_only": decision_threshold,
        "selected_ridge_alpha_train_only": selection["selected_alpha"],
        "inner_validation_pr_auc": selection["inner_validation_pr_auc"],
        "inner_validation_strategy": selection["inner_validation_strategy"],
        "ridge_alpha_grid_json": json.dumps(selection["alpha_grid"]),
        "ridge_alpha_selection_json": json.dumps(clean_json(selection["alpha_selection_rows"]), sort_keys=True),
        "uses_future_features": False,
        "uses_train_only_transform": True,
        "vjepa_encoding_run": False,
        "tribe_encoding_run": False,
        "pca_run": True,
        "bridge_training_run": False,
        **extra,
        **fit_info,
        **metric_row(test_y, test_scores, decision_threshold),
        **{f"delta_{k}": v for k, v in regression_metric_row(continuous_test_values, test_scores).items()},
    }
    return row


def score_phase4_split(
    df: pd.DataFrame,
    split: SplitSpec,
    pca_results: dict[str, PcaFitResult],
    *,
    widths: Sequence[int],
    families: Sequence[str],
    temporal: np.ndarray,
    starts: np.ndarray,
    alpha_grid: Sequence[float] | None,
    alpha: float,
    random_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values_all = df[split.target.value_column].to_numpy(dtype=np.float64)
    ar = feature_matrix(df, AR_FEATURE_COLUMNS)
    time_features = feature_matrix(df, TIME_FEATURE_COLUMNS)
    quality = feature_matrix(df, QUALITY_FEATURE_COLUMNS)
    seed_offset = int(text_digest(split.key, digest_size=8), 16) % 1_000_000
    rng = np.random.default_rng(random_seed + split.fold + seed_offset)
    invariant_lane_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    ar_base_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]] = {}
    inner_split_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, str]] = {}

    def labels_for(train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        train_mask = np.zeros(len(df), dtype=bool)
        test_mask = np.zeros(len(df), dtype=bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True
        y_train, y_test, _threshold = threshold_labels(values_all, train_mask, test_mask, split.target)
        return y_train, y_test

    def fit_lane_reusing_invariant(
        cache_key: tuple[Any, ...] | None,
        lane: str,
        model_train_idx: np.ndarray,
        model_test_idx: np.ndarray,
        train_x: np.ndarray,
        test_x: np.ndarray,
        train_y: np.ndarray,
        test_y: np.ndarray,
        continuous_test_values: np.ndarray,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        if cache_key is not None and cache_key in invariant_lane_cache:
            row = dict(invariant_lane_cache[cache_key])
            row.update(extra)
            return row
        row = fit_and_score_lane(
            df,
            split,
            lane,
            model_train_idx,
            model_test_idx,
            train_x,
            test_x,
            train_y,
            test_y,
            continuous_test_values,
            alpha_grid=alpha_grid,
            alpha=alpha,
            random_seed=random_seed,
            inner_split_cache=inner_split_cache,
            extra=extra,
        )
        if cache_key is not None:
            invariant_lane_cache[cache_key] = dict(row)
        return row

    def ar_base_for(valid_train: np.ndarray, valid_test: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
        cache_key = ("ar_base", array_digest(valid_train), array_digest(valid_test), array_digest(y_train.astype(np.int8, copy=False)))
        cached = ar_base_cache.get(cache_key)
        if cached is not None:
            return cached
        if alpha_grid is None:
            ar_selection = {
                "selected_alpha": float(alpha),
                "inner_validation_pr_auc": math.nan,
                "inner_validation_strategy": "fixed_cli_alpha",
                "alpha_grid": [float(alpha)],
                "alpha_selection_rows": [],
            }
        else:
            ar_selection = phase4_select_alpha_train_only(df, valid_train, ar[valid_train], y_train, alpha_grid, inner_split_cache)
        ar_train_scores, ar_test_scores, ar_fit = phase4_scale_fit_predict(
            ar[valid_train],
            ar[valid_test],
            y_train,
            alpha=ar_selection["selected_alpha"],
        )
        cached = (ar_train_scores, ar_test_scores, ar_selection, ar_fit)
        ar_base_cache[cache_key] = cached
        return cached

    for family in families:
        source_family = "temporal_mean_2s" if family == "temporal_then_pca" else ("delta" if family == "delta" else "current")
        result = pca_results.get(f"{split.key}::{source_family}")
        if result is None:
            continue
        valid_train, valid_test, pca_train_all, pca_test_all = score_subset(result, split.train_idx, split.test_idx)
        y_train, y_test = labels_for(valid_train, valid_test)
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        all_valid = np.concatenate([valid_train, valid_test]).astype(np.int64)
        all_scores = np.concatenate([pca_train_all, pca_test_all], axis=0).astype(np.float32, copy=False)
        scores_lookup = build_scores_lookup(all_valid, all_scores)
        for width in widths:
            base_features: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
            if family == "current":
                base_features[f"cortical_pca{width}_current"] = (
                    pca_train_all[:, :width],
                    pca_test_all[:, :width],
                    "current-row PCA score",
                )
            elif family == "delta":
                base_features[f"cortical_pca{width}_delta"] = (
                    pca_train_all[:, :width],
                    pca_test_all[:, :width],
                    "PCA of current-minus-previous same-video cortical rows",
                )
            elif family == "pca_then_temporal":
                for name, window_rows in TEMPORAL_MEAN_WINDOWS.items():
                    feat_train = causal_reduce_scores(scores_lookup, valid_train, starts, width=width, window_rows=window_rows, mode="mean")
                    feat_test = causal_reduce_scores(scores_lookup, valid_test, starts, width=width, window_rows=window_rows, mode="mean")
                    base_features[f"cortical_pca{width}_{name}"] = (feat_train, feat_test, "PCA score causal trailing mean")
                for name, window_rows in TEMPORAL_SLOPE_WINDOWS.items():
                    feat_train = causal_reduce_scores(scores_lookup, valid_train, starts, width=width, window_rows=window_rows, mode="slope")
                    feat_test = causal_reduce_scores(scores_lookup, valid_test, starts, width=width, window_rows=window_rows, mode="slope")
                    base_features[f"cortical_pca{width}_{name}"] = (feat_train, feat_test, "PCA score causal trailing slope")
                for name, window_rows in TEMPORAL_STD_WINDOWS.items():
                    feat_train = causal_reduce_scores(scores_lookup, valid_train, starts, width=width, window_rows=window_rows, mode="std")
                    feat_test = causal_reduce_scores(scores_lookup, valid_test, starts, width=width, window_rows=window_rows, mode="std")
                    base_features[f"cortical_pca{width}_{name}"] = (feat_train, feat_test, "PCA score causal trailing std")
            elif family == "temporal_then_pca":
                base_features[f"temporal_mean_2s_then_pca{width}"] = (
                    pca_train_all[:, :width],
                    pca_test_all[:, :width],
                    "raw cortical causal 2s mean followed by train-only PCA",
                )
            for feature_name, (pca_train, pca_test, feature_desc) in base_features.items():
                lane_specs = {
                    "AR_only": (ar[valid_train], ar[valid_test]),
                    "PCA_only": (pca_train, pca_test),
                    "AR_plus_PCA": (np.concatenate([ar[valid_train], pca_train], axis=1), np.concatenate([ar[valid_test], pca_test], axis=1)),
                    "PCA_plus_temporal_diagnostics": (
                        np.concatenate([pca_train, temporal[valid_train]], axis=1),
                        np.concatenate([pca_test, temporal[valid_test]], axis=1),
                    ),
                    "AR_plus_PCA_plus_temporal_diagnostics": (
                        np.concatenate([ar[valid_train], pca_train, temporal[valid_train]], axis=1),
                        np.concatenate([ar[valid_test], pca_test, temporal[valid_test]], axis=1),
                    ),
                    "shuffled_PCA_control": (
                        pca_train[rng.permutation(len(pca_train))],
                        pca_test[rng.permutation(len(pca_test))],
                    ),
                    "shuffled_temporal_diagnostics_control": (
                        temporal[valid_train][rng.permutation(len(valid_train))],
                        temporal[valid_test][rng.permutation(len(valid_test))],
                    ),
                    "random_matched_PCA_control": (
                        rng.normal(size=pca_train.shape).astype(np.float32),
                        rng.normal(size=pca_test.shape).astype(np.float32),
                    ),
                    "timestamp_video_time_only_control": (time_features[valid_train], time_features[valid_test]),
                    "quality_motion_luma_only_control": (quality[valid_train], quality[valid_test]),
                    "AR_plus_shuffled_PCA_control": (
                        np.concatenate([ar[valid_train], pca_train[rng.permutation(len(pca_train))]], axis=1),
                        np.concatenate([ar[valid_test], pca_test[rng.permutation(len(pca_test))]], axis=1),
                    ),
                    "AR_plus_random_matched_PCA_control": (
                        np.concatenate([ar[valid_train], rng.normal(size=pca_train.shape).astype(np.float32)], axis=1),
                        np.concatenate([ar[valid_test], rng.normal(size=pca_test.shape).astype(np.float32)], axis=1),
                    ),
                    "AR_plus_timestamp_video_time_control": (
                        np.concatenate([ar[valid_train], time_features[valid_train]], axis=1),
                        np.concatenate([ar[valid_test], time_features[valid_test]], axis=1),
                    ),
                    "AR_plus_quality_motion_luma_control": (
                        np.concatenate([ar[valid_train], quality[valid_train]], axis=1),
                        np.concatenate([ar[valid_test], quality[valid_test]], axis=1),
                    ),
                }
                valid_key = (array_digest(valid_train), array_digest(valid_test))
                for lane, (train_x, test_x) in lane_specs.items():
                    invariant_key = None
                    if lane in {
                        "AR_only",
                        "timestamp_video_time_only_control",
                        "quality_motion_luma_only_control",
                        "AR_plus_timestamp_video_time_control",
                        "AR_plus_quality_motion_luma_control",
                        "shuffled_temporal_diagnostics_control",
                    }:
                        invariant_key = ("invariant_lane", lane, *valid_key)
                    rows.append(
                        fit_lane_reusing_invariant(
                            invariant_key,
                            lane,
                            valid_train,
                            valid_test,
                            train_x,
                            test_x,
                            y_train,
                            y_test,
                            values_all[valid_test],
                            extra={
                                "pca_width": width,
                                "feature_family": family,
                                "feature_name": feature_name,
                                "feature_description": feature_desc,
                                "pca_source_family": source_family,
                                "pca_component_path": str(result.component_path),
                                "pca_score_path": str(result.score_path),
                                "pca_explained_variance_ratio_sum": result.metadata.get("explained_variance_ratio_sum"),
                            },
                        )
                    )
                # Residualized lanes reuse AR fitted on the exact same rows.
                ar_train_scores, ar_test_scores, _ar_selection, _ar_fit = ar_base_for(valid_train, valid_test, y_train)
                residual_train = y_train.astype(np.float32) - ar_train_scores.astype(np.float32)
                for residual_lane, train_x, test_x in [
                    ("residualized_AR_plus_PCA", pca_train, pca_test),
                    (
                        "residualized_AR_plus_PCA_plus_temporal_diagnostics",
                        np.concatenate([pca_train, temporal[valid_train]], axis=1),
                        np.concatenate([pca_test, temporal[valid_test]], axis=1),
                    ),
                ]:
                    if alpha_grid is None:
                        selection = {
                            "selected_alpha": float(alpha),
                            "inner_validation_pr_auc": math.nan,
                            "inner_validation_strategy": "fixed_cli_alpha",
                            "alpha_grid": [float(alpha)],
                            "alpha_selection_rows": [],
                        }
                    else:
                        selection = phase4_select_alpha_train_only(df, valid_train, train_x, (residual_train > 0).astype(int), alpha_grid, inner_split_cache)
                    res_train_scores, res_test_scores, fit_info = phase4_scale_fit_predict(
                        train_x,
                        test_x,
                        residual_train,
                        alpha=selection["selected_alpha"],
                    )
                    combined_train = ar_train_scores + res_train_scores
                    combined_test = ar_test_scores + res_test_scores
                    decision_threshold = decision_threshold_from_train(y_train, combined_train)
                    rows.append(
                        {
                            "schema_version": PHASE4_SCHEMA_VERSION,
                            "target_name": split.target.name,
                            "target_value_column": split.target.value_column,
                            "target_mask_column": split.target.mask_column,
                            "target_threshold_train_only": split.target_threshold,
                            "target_threshold_quantile": split.target.quantile,
                            "target_transform": split.target.transform,
                            "validation_protocol": split.protocol,
                            "fold": split.fold,
                            "model_lane": residual_lane,
                            "n_train": int(train_x.shape[0]),
                            "n_test": int(test_x.shape[0]),
                            "train_videos": int(df.loc[valid_train, "video_id"].nunique()),
                            "test_videos": int(df.loc[valid_test, "video_id"].nunique()),
                            "train_event_count": int(np.sum(y_train)),
                            "test_event_count": int(np.sum(y_test)),
                            "train_positive_rate": float(np.mean(y_train)),
                            "test_positive_rate": float(np.mean(y_test)),
                            "feature_width": int(train_x.shape[1]),
                            "decision_threshold_train_only": decision_threshold,
                            "selected_ridge_alpha_train_only": selection["selected_alpha"],
                            "inner_validation_pr_auc": selection["inner_validation_pr_auc"],
                            "inner_validation_strategy": selection["inner_validation_strategy"],
                            "ridge_alpha_grid_json": json.dumps(selection["alpha_grid"]),
                            "ridge_alpha_selection_json": json.dumps(clean_json(selection["alpha_selection_rows"]), sort_keys=True),
                            "residualization_math": "fit AR on train; fit PCA lane to train_y - AR_train_scores; eval score = AR_eval_scores + PCA_residual_eval_scores",
                            "uses_future_features": False,
                            "uses_train_only_transform": True,
                            "vjepa_encoding_run": False,
                            "tribe_encoding_run": False,
                            "pca_run": True,
                            "bridge_training_run": False,
                            "pca_width": width,
                            "feature_family": family,
                            "feature_name": feature_name,
                            "feature_description": feature_desc,
                            "pca_source_family": source_family,
                            "pca_component_path": str(result.component_path),
                            "pca_score_path": str(result.score_path),
                            "pca_explained_variance_ratio_sum": result.metadata.get("explained_variance_ratio_sum"),
                            **fit_info,
                            **metric_row(y_test, combined_test, decision_threshold),
                            **{f"delta_{k}": v for k, v in regression_metric_row(values_all[valid_test], combined_test).items()},
                        }
                    )
    return rows


def summarize_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    if fold_df.empty:
        return pd.DataFrame()
    return (
        fold_df.groupby(["target_name", "validation_protocol", "feature_family", "feature_name", "pca_width", "model_lane"], dropna=False)
        .agg(
            folds=("fold", "count"),
            rows_test_total=("n_test", "sum"),
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            min_pr_auc=("pr_auc", "min"),
            max_pr_auc=("pr_auc", "max"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_f1=("f1", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_top_1pct_recall=("top_1pct_recall", "mean"),
            mean_top_5pct_recall=("top_5pct_recall", "mean"),
            mean_top_10pct_recall=("top_10pct_recall", "mean"),
            mean_delta_mae=("delta_mae", "mean"),
            mean_delta_mse=("delta_mse", "mean"),
            mean_delta_pearson=("delta_pearson", "mean"),
        )
        .reset_index()
    )


def add_delta_columns(summary_df: pd.DataFrame, phase3: dict[str, dict[str, float]]) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    out = summary_df.copy()
    keys = ["target_name", "validation_protocol", "feature_family", "feature_name", "pca_width"]
    ar_rows = out[out["model_lane"] == "AR_only"][keys + ["mean_pr_auc"]].rename(columns={"mean_pr_auc": "ar_only_mean_pr_auc"})
    out = out.merge(ar_rows, on=keys, how="left")
    out["delta_vs_AR_only"] = out["mean_pr_auc"] - out["ar_only_mean_pr_auc"]
    out["phase3_raw_cortical_pr_auc"] = out["target_name"].map(lambda t: phase3.get(t, {}).get("raw_cortical_only"))
    out["phase3_AR_plus_raw_cortical_pr_auc"] = out["target_name"].map(lambda t: phase3.get(t, {}).get("AR_plus_raw_cortical"))
    out["delta_vs_phase3_raw_cortical"] = out["mean_pr_auc"] - out["phase3_raw_cortical_pr_auc"]
    out["delta_vs_phase3_AR_plus_raw_cortical"] = out["mean_pr_auc"] - out["phase3_AR_plus_raw_cortical_pr_auc"]
    return out


def promotion_gates_phase4(summary_df: pd.DataFrame, fold_df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    grouped = summary_df[summary_df["validation_protocol"] == "grouped_video"]
    blocked = summary_df[summary_df["validation_protocol"] == "blocked_temporal_70_30"]
    lane_sets = grouped.groupby(["target_name", "feature_name", "pca_width"], dropna=False)
    for (target, feature_name, width), group in lane_sets:
        by_lane = {row["model_lane"]: float(row["mean_pr_auc"]) for _, row in group.iterrows()}
        ar = by_lane.get("AR_only")
        time = by_lane.get("timestamp_video_time_only_control")
        quality = by_lane.get("quality_motion_luma_only_control")
        shuffled = by_lane.get("shuffled_PCA_control")
        random_control = by_lane.get("random_matched_PCA_control")
        ar_shuffle = by_lane.get("AR_plus_shuffled_PCA_control")
        ar_random = by_lane.get("AR_plus_random_matched_PCA_control")
        ar_time = by_lane.get("AR_plus_timestamp_video_time_control")
        ar_quality = by_lane.get("AR_plus_quality_motion_luma_control")
        for lane, gate_name, controls in [
            ("PCA_only", "strict_grouped_pass", [ar, shuffled, random_control, time, quality]),
            ("AR_plus_PCA", "strict_AR_plus_PCA_grouped_pass", [ar, ar_shuffle, ar_random, ar_time, ar_quality]),
            ("residualized_AR_plus_PCA", "residualized_grouped_pass", [ar, random_control, shuffled]),
            ("AR_plus_PCA_plus_temporal_diagnostics", "strict_AR_plus_PCA_temporal_grouped_pass", [ar, ar_shuffle, ar_random, ar_time, ar_quality]),
            ("residualized_AR_plus_PCA_plus_temporal_diagnostics", "residualized_temporal_grouped_pass", [ar, random_control, shuffled]),
        ]:
            candidate = by_lane.get(lane)
            grouped_fold_rows = fold_df[
                (fold_df["validation_protocol"] == "grouped_video")
                & (fold_df["target_name"] == target)
                & (fold_df["feature_name"] == feature_name)
                & (fold_df["pca_width"] == width)
                & (fold_df["model_lane"] == lane)
            ]
            ar_fold_rows = fold_df[
                (fold_df["validation_protocol"] == "grouped_video")
                & (fold_df["target_name"] == target)
                & (fold_df["feature_name"] == feature_name)
                & (fold_df["pca_width"] == width)
                & (fold_df["model_lane"] == "AR_only")
            ][["fold", "pr_auc"]].rename(columns={"pr_auc": "ar_pr_auc"})
            merged = grouped_fold_rows.merge(ar_fold_rows, on="fold", how="left") if not grouped_fold_rows.empty else pd.DataFrame()
            fold_delta = merged["pr_auc"] - merged["ar_pr_auc"] if not merged.empty else pd.Series(dtype=float)
            positive_mean_fold_delta = bool(len(fold_delta) and float(fold_delta.mean()) > 0)
            not_one_fold_only = bool(len(fold_delta) >= 3 and int((fold_delta > 0).sum()) >= 3)
            blocked_candidate = blocked[
                (blocked["target_name"] == target)
                & (blocked["feature_name"] == feature_name)
                & (blocked["pca_width"] == width)
                & (blocked["model_lane"] == lane)
            ]
            blocked_ar = blocked[
                (blocked["target_name"] == target)
                & (blocked["feature_name"] == feature_name)
                & (blocked["pca_width"] == width)
                & (blocked["model_lane"] == "AR_only")
            ]
            blocked_support = (
                not blocked_candidate.empty
                and not blocked_ar.empty
                and float(blocked_candidate.iloc[0]["mean_pr_auc"]) > float(blocked_ar.iloc[0]["mean_pr_auc"])
            )
            comparisons = [candidate is not None and c is not None and candidate > c for c in controls]
            status = bool(candidate is not None and all(comparisons) and positive_mean_fold_delta and not_one_fold_only)
            gates.append(
                {
                    "target_name": target,
                    "feature_name": feature_name,
                    "pca_width": int(width),
                    "model_lane": lane,
                    "gate_name": gate_name,
                    "status": "pass" if status else "fail",
                    "candidate_grouped_pr_auc": candidate,
                    "ar_only_grouped_pr_auc": ar,
                    "delta_vs_AR_only": candidate - ar if candidate is not None and ar is not None else None,
                    "beats_required_controls": bool(all(comparisons)),
                    "positive_mean_fold_delta_vs_AR": positive_mean_fold_delta,
                    "not_driven_by_one_fold_only": not_one_fold_only,
                    "positive_fold_count_vs_AR": int((fold_delta > 0).sum()) if len(fold_delta) else 0,
                    "fold_count": int(len(fold_delta)),
                    "blocked_temporal_support": bool(blocked_support),
                    "blocked_temporal_status": "support" if blocked_support else "mixed_or_fail",
                }
            )
    if not grouped.empty:
        for target, target_df in grouped.groupby("target_name"):
            candidates = target_df[target_df["model_lane"].isin(["PCA_only", "AR_plus_PCA", "residualized_AR_plus_PCA", "AR_plus_PCA_plus_temporal_diagnostics", "residualized_AR_plus_PCA_plus_temporal_diagnostics"])]
            if candidates.empty:
                continue
            best = candidates.sort_values("mean_pr_auc", ascending=False).iloc[0].to_dict()
            best_rows.append(clean_json(best))
    consistency = {
        "grouped_gate_count": len(gates),
        "grouped_gate_pass_count": sum(1 for gate in gates if gate["status"] == "pass"),
        "blocked_temporal_support_count": sum(1 for gate in gates if gate.get("blocked_temporal_support")),
    }
    return gates, best_rows, consistency


def write_phase4_reports(
    output_root: Path,
    manifest: dict[str, Any],
    pca_manifest_rows: list[dict[str, Any]],
    summary_df: pd.DataFrame,
    gates: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    *,
    report_dir: Path = Path("reports"),
) -> dict[str, str]:
    timestamp = utc_stamp()
    report_dir.mkdir(parents=True, exist_ok=True)
    report1 = report_dir / f"again_dense_2hz_phase4_pca_feature_build_{timestamp}.md"
    report2 = report_dir / f"again_dense_2hz_phase4_pca_bridge_benchmark_{timestamp}.md"
    report3 = report_dir / f"again_dense_2hz_phase4_pca_promotion_summary_{timestamp}.md"

    variance_by_width: dict[int, list[float]] = {}
    for row in pca_manifest_rows:
        variance_by_width.setdefault(int(row["pca_width"]), []).append(float(row.get("explained_variance_ratio_sum") or 0.0))
    lines1 = [
        "# AGAIN Dense 2Hz Phase 4 PCA Feature Build",
        "",
        f"- input cache root: `{manifest['dense_root']}`",
        f"- label manifest path: `{manifest['labels_path']}`",
        f"- dense rows: `{manifest['rows']}`",
        f"- labeled rows: `{manifest['labeled_rows']}`",
        f"- PCA widths: `{manifest['pca_widths']}`",
        f"- feature families: `{manifest['feature_families']}`",
        "- PCA policy: train-only inside target/protocol/fold row set; max width fit is sliced for narrower widths.",
        f"- fold-specific PCA fits: `{len(pca_manifest_rows)}`",
        f"- feature artifact root: `{output_root / 'features'}`",
        "",
        "## Explained Variance",
        "",
    ]
    for width, values in sorted(variance_by_width.items()):
        lines1.append(f"- width `{width}`: mean explained variance ratio sum `{np.mean(values):.6f}` across `{len(values)}` fits")
    lines1.extend(
        [
            "",
            "## Guardrails",
            "",
            "- no V-JEPA/TRIBE re-encoding",
            "- no global PCA for promoted claims",
            "- row identity is keyed by `video_id,row_index,time_seconds`",
            "- delta PCA explicitly drops first rows per video",
            "",
        ]
    )
    report1.write_text("\n".join(lines1), encoding="utf-8")

    top_lines: list[str] = []
    if not summary_df.empty:
        grouped = summary_df[summary_df["validation_protocol"] == "grouped_video"]
        for target, target_df in grouped.groupby("target_name"):
            top = target_df.sort_values("mean_pr_auc", ascending=False).head(12)
            top_lines.append(f"### {target}")
            for _, row in top.iterrows():
                top_lines.append(
                    f"- `{row['model_lane']}` / `{row['feature_name']}` / width `{int(row['pca_width'])}`: "
                    f"PR-AUC `{100 * row['mean_pr_auc']:.2f}%`, delta vs AR `{100 * row.get('delta_vs_AR_only', math.nan):.2f} pp`"
                )
    lines2 = [
        "# AGAIN Dense 2Hz Phase 4 PCA Bridge Benchmark",
        "",
        "- Scope: dense true-2Hz AGAIN H100 TRIBE cortical cache.",
        "- No raw video decode, V-JEPA run, or TRIBE run was performed.",
        "- This is PCA bridge benchmarking, not learned-head training.",
        f"- targets: `{manifest['targets']}`",
        f"- validation protocols: `{manifest['validation_protocols']}`",
        f"- model lanes and controls: `{manifest['model_lanes']}`",
        f"- ridge alpha grid: `{manifest['ridge_alpha_grid']}`",
        "",
        "## Top Grouped-Video Rows",
        "",
        *top_lines,
        "",
        "## Limitations",
        "",
        "- PCA uses deterministic randomized SVD with MLX-backed batch matmul where available.",
        "- Blocked temporal support is diagnostic; grouped-video remains the primary gate.",
        "- Phase 3 raw-cortical deltas are joined from the latest tracked report when available.",
        "",
    ]
    report2.write_text("\n".join(lines2), encoding="utf-8")

    gate_lines = [
        "# AGAIN Dense 2Hz Phase 4 PCA Promotion Summary",
        "",
        "## Best Lanes By Target",
        "",
    ]
    for row in best_rows:
        gate_lines.append(
            f"- `{row.get('target_name')}`: `{row.get('model_lane')}` / `{row.get('feature_name')}` / width `{row.get('pca_width')}` "
            f"PR-AUC `{100 * float(row.get('mean_pr_auc') or 0.0):.2f}%`"
        )
    gate_lines.extend(["", "## Gates", ""])
    for gate in gates:
        if gate["status"] == "pass":
            gate_lines.append(
                f"- PASS `{gate['target_name']}` `{gate['gate_name']}` `{gate['model_lane']}` `{gate['feature_name']}` "
                f"width `{gate['pca_width']}` delta vs AR `{100 * float(gate.get('delta_vs_AR_only') or 0.0):.2f} pp`, "
                f"blocked `{gate['blocked_temporal_status']}`"
            )
    if not any(gate["status"] == "pass" for gate in gates):
        gate_lines.append("- No strict grouped promotion gate passed.")
    gate_lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "- Promote only lanes that pass grouped-video controls and have at least mixed-to-positive blocked temporal support.",
            "- Phase 5 should use the best Phase 4 lane as a locked train-only bridge input, not reselect over this whole grid after seeing held-out results.",
            "",
        ]
    )
    report3.write_text("\n".join(gate_lines), encoding="utf-8")
    return {"feature_report": str(report1), "benchmark_report": str(report2), "promotion_report": str(report3)}


def run_phase4(
    *,
    dense_root: Path,
    output_root: Path,
    widths: Sequence[int] = DEFAULT_WIDTHS,
    feature_families: Sequence[str] = DEFAULT_FEATURE_FAMILIES,
    validation_protocols: Sequence[str] = DEFAULT_PROTOCOLS,
    n_splits: int = 5,
    random_seed: int = DEFAULT_RANDOM_SEED,
    ridge_alpha: float = 1.0,
    ridge_alpha_grid: Sequence[float] | None = DEFAULT_RIDGE_ALPHA_GRID,
    batch_size: int = 384,
    oversampling: int = 32,
    power_iterations: int = 1,
    dry_run: bool = False,
    resume: bool = False,
    force_cortical_memmap: bool = False,
    max_fit_count: int | None = None,
    report_dir: Path = Path("reports"),
) -> dict[str, Any]:
    assert_again_only_output_path(output_root)
    ensure_phase4_cache_only(dense_root)
    df = load_labels(dense_root)
    if abs(float(df["row_rate_hz"].dropna().median()) - ROW_RATE_HZ) > 1e-6:
        raise ValueError("Phase 4 requires true 2Hz labels")
    splits = build_split_specs(df, protocols=validation_protocols, n_splits=n_splits)
    if not splits:
        raise ValueError("No valid train/eval splits for Phase 4")
    max_width = max(widths)
    planned_base_families = []
    if "current" in feature_families or "pca_then_temporal" in feature_families:
        planned_base_families.append("current")
    if "delta" in feature_families:
        planned_base_families.append("delta")
    if "temporal_then_pca" in feature_families:
        planned_base_families.append("temporal_mean_2s")
    planned_fits = [(split, base) for split in splits for base in planned_base_families]
    if max_fit_count is not None:
        planned_fits = planned_fits[:max_fit_count]
    manifest_base = {
        "schema_version": PHASE4_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dense_root": str(dense_root),
        "labels_path": str(dense_root / "labels_aligned_2hz.parquet"),
        "output_root": str(output_root),
        "rows": int(len(df)),
        "labeled_rows": int(df["label_available"].sum()),
        "targets": [spec.name for spec in TARGET_SPECS],
        "validation_protocols": list(validation_protocols),
        "pca_widths": list(widths),
        "feature_families": list(feature_families),
        "model_lanes": [
            "AR_only",
            "PCA_only",
            "AR_plus_PCA",
            "residualized_AR_plus_PCA",
            "PCA_plus_temporal_diagnostics",
            "AR_plus_PCA_plus_temporal_diagnostics",
            "residualized_AR_plus_PCA_plus_temporal_diagnostics",
            "shuffled_PCA_control",
            "shuffled_temporal_diagnostics_control",
            "random_matched_PCA_control",
            "timestamp_video_time_only_control",
            "quality_motion_luma_only_control",
            "AR_plus_shuffled_PCA_control",
            "AR_plus_random_matched_PCA_control",
            "AR_plus_timestamp_video_time_control",
            "AR_plus_quality_motion_luma_control",
        ],
        "seeds": {"random_seed": random_seed},
        "ridge_alpha_grid": list(ridge_alpha_grid) if ridge_alpha_grid is not None else [ridge_alpha],
        "ridge_alpha_selection": "train_only_inner_validation" if ridge_alpha_grid is not None else "fixed_cli_alpha",
        "planned_pca_fit_count": len(planned_fits),
        "split_fingerprints": [split_fingerprint(split) for split in splits],
        "no_vjepa_tribe_reencoding": True,
        "vjepa_encoding_run": False,
        "tribe_encoding_run": False,
        "pca_run": True,
        "bridge_training_run": False,
        "hardware_backend": {
            "mlx_available": mx is not None,
            "mlx_default_device": str(mx.default_device()) if mx is not None else None,
            "cpu_count": os.cpu_count(),
        },
        **git_metadata(),
    }
    if dry_run:
        return {
            **manifest_base,
            "dry_run": True,
            "planned_fits": [
                {"target": split.target.name, "protocol": split.protocol, "fold": split.fold, "base_family": base}
                for split, base in planned_fits
            ],
        }
    if output_root.exists() and any(output_root.iterdir()) and not resume:
        raise FileExistsError(f"Output root exists and is non-empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    dirs = phase4_dirs(output_root)
    cortical = load_or_build_cortical_memmap(dense_root, df, output_root=output_root, force=force_cortical_memmap)
    temporal = load_or_build_temporal_diagnostic_features(dense_root, df)
    starts = video_start_indices(df)
    pca_results: dict[str, PcaFitResult] = {}
    pca_manifest_rows: list[dict[str, Any]] = []
    for ordinal, (split, base_family) in enumerate(planned_fits, start=1):
        accessor = CorticalVariantAccessor(cortical, df, base_family=base_family)
        result = fit_or_load_pca(
            split,
            accessor,
            output_root=output_root,
            width=max_width,
            seed=random_seed + ordinal,
            batch_size=batch_size,
            oversampling=oversampling,
            power_iterations=power_iterations,
        )
        pca_results[f"{split.key}::{base_family}"] = result
        for width in widths:
            explained_ratio = result.metadata.get("explained_variance_ratio") or []
            row = {
                "target_name": split.target.name,
                "validation_protocol": split.protocol,
                "fold": split.fold,
                "base_family": base_family,
                "pca_width": int(width),
                "training_row_count": int(result.metadata["train_row_count"]),
                "eval_row_count": int(result.metadata.get("eval_row_count", split.test_idx.size)),
                "pca_algorithm": result.metadata["pca_algorithm"],
                "centering_scaling_policy": result.metadata["centering_scaling_policy"],
                "explained_variance_ratio_sum": float(np.sum(explained_ratio[:width]))
                if width <= len(explained_ratio)
                else result.metadata["explained_variance_ratio_sum"],
                "component_checksum": result.metadata["component_checksum"],
                "train_row_fingerprint": result.metadata["train_idx_digest"],
                "transform_row_fingerprint": result.metadata["transform_idx_digest"],
                "feature_artifact_path": str(result.score_path),
                "feature_artifact_checksum": result.metadata["score_checksum_first_64mb"],
                "dropped_train_rows": result.metadata.get("dropped_train_rows", 0),
                "dropped_eval_rows": result.metadata.get("dropped_eval_rows", 0),
                "drop_reason": result.metadata.get("drop_reason", ""),
            }
            pca_manifest_rows.append(row)
    fold_rows: list[dict[str, Any]] = []
    reused_score_parts = 0
    computed_score_parts = 0
    for split in splits:
        part_metadata = score_part_metadata(
            split,
            widths=widths,
            families=feature_families,
            ridge_alpha_grid=ridge_alpha_grid,
            ridge_alpha=ridge_alpha,
            random_seed=random_seed,
        )
        part_rows = load_score_part_if_valid(output_root, split, part_metadata) if resume else None
        if part_rows is not None:
            reused_score_parts += 1
        else:
            part_rows = score_phase4_split(
                df,
                split,
                pca_results,
                widths=widths,
                families=feature_families,
                temporal=temporal,
                starts=starts,
                alpha_grid=ridge_alpha_grid,
                alpha=ridge_alpha,
                random_seed=random_seed,
            )
            write_score_part(output_root, split, part_rows, part_metadata)
            computed_score_parts += 1
        fold_rows.extend(part_rows)
    fold_df = pd.DataFrame(fold_rows)
    summary_df = add_delta_columns(summarize_metrics(fold_df), load_phase3_summary())
    gates, best_rows, consistency = promotion_gates_phase4(summary_df, fold_df)
    metrics_dir = dirs["metrics"]
    promotion_dir = dirs["promotion"]
    diagnostics_dir = dirs["diagnostics"]
    fold_path = metrics_dir / "phase4_fold_metrics.csv"
    summary_path = metrics_dir / "phase4_summary_metrics.csv"
    control_path = metrics_dir / "phase4_control_metrics.csv"
    grouped_path = metrics_dir / "phase4_grouped_video_metrics.csv"
    blocked_path = metrics_dir / "phase4_blocked_temporal_metrics.csv"
    delta_path = metrics_dir / "phase4_delta_vs_phase3.csv"
    width_path = metrics_dir / "phase4_width_sweep.csv"
    family_path = metrics_dir / "phase4_feature_family_sweep.csv"
    prevalence_path = metrics_dir / "phase4_event_prevalence_by_fold.csv"
    coverage_path = metrics_dir / "phase4_row_video_coverage.csv"
    fold_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    summary_df[summary_df["model_lane"].str.contains("control", na=False)].to_csv(control_path, index=False)
    summary_df[summary_df["validation_protocol"] == "grouped_video"].to_csv(grouped_path, index=False)
    summary_df[summary_df["validation_protocol"] == "blocked_temporal_70_30"].to_csv(blocked_path, index=False)
    summary_df.to_csv(delta_path, index=False)
    summary_df.groupby(["target_name", "validation_protocol", "pca_width", "model_lane"], dropna=False)["mean_pr_auc"].max().reset_index().to_csv(width_path, index=False)
    summary_df.groupby(["target_name", "validation_protocol", "feature_family", "model_lane"], dropna=False)["mean_pr_auc"].max().reset_index().to_csv(family_path, index=False)
    fold_df[
        ["target_name", "validation_protocol", "fold", "model_lane", "feature_name", "pca_width", "train_positive_rate", "test_positive_rate", "train_event_count", "test_event_count"]
    ].to_csv(prevalence_path, index=False)
    pd.DataFrame(
        [
            {
                "target_name": split.target.name,
                "validation_protocol": split.protocol,
                "fold": split.fold,
                "n_train": int(split.train_idx.size),
                "n_test": int(split.test_idx.size),
                "n_train_videos": int(df.loc[split.train_idx, "video_id"].nunique()),
                "n_test_videos": int(df.loc[split.test_idx, "video_id"].nunique()),
            }
            for split in splits
        ]
    ).to_csv(coverage_path, index=False)
    write_csv(output_root / "pca_feature_manifest.csv", pca_manifest_rows)
    write_json(output_root / "pca_feature_manifest.json", {"fits": pca_manifest_rows})
    write_json(promotion_dir / "promotion_gates.json", gates)
    write_csv(promotion_dir / "promotion_gates.csv", gates)
    write_json(promotion_dir / "best_lanes_by_target.json", best_rows)
    write_csv(promotion_dir / "best_lanes_by_target.csv", best_rows)
    write_json(promotion_dir / "grouped_vs_blocked_consistency.json", consistency)
    write_json(promotion_dir / "failure_reasons.json", {"failed_gates": [gate for gate in gates if gate["status"] != "pass"]})
    write_json(diagnostics_dir / "split_leakage_audit.json", {"splits": [split_fingerprint(split) for split in splits], "grouped_video_no_overlap": True})
    write_json(diagnostics_dir / "transform_leakage_audit.json", {"pca_fit_scope": "train_only_per_target_protocol_fold_mask", "global_pca_used": False})
    write_json(diagnostics_dir / "pca_fit_diagnostics.json", {"fits": [result.metadata for result in pca_results.values()]})
    write_json(diagnostics_dir / "solver_diagnostics.json", {"ridge_backend": "mlx_primal_conjugate_gradient_when_available", "alpha_grid": manifest_base["ridge_alpha_grid"]})
    write_json(diagnostics_dir / "nan_inf_audit.json", {"fold_metric_nan_cells": int(fold_df.isna().sum().sum()) if not fold_df.empty else 0})
    write_json(diagnostics_dir / "row_join_integrity_audit.json", {"rows": int(len(df)), "duplicates_video_row": bool(df.duplicated(["video_id", "row_index"]).any())})
    write_json(diagnostics_dir / "target_mask_audit.json", {spec.name: int(target_base_mask(df, spec).sum()) for spec in TARGET_SPECS})
    write_json(
        diagnostics_dir / "quality_flag_audit.json",
        {
            "quality_excluded_rows": int(df.get("quality_exclusion_flag", pd.Series(dtype=int)).sum()) if "quality_exclusion_flag" in df else None,
            "black_frame_fraction_max": float(df["black_frame_fraction"].max()) if "black_frame_fraction" in df else None,
        },
    )
    write_json(diagnostics_dir / "alignment_coverage_reuse_audit.json", {"labels_path": str(dense_root / "labels_aligned_2hz.parquet"), "true_2hz": True})
    write_json(diagnostics_dir / "compute_resource_summary.json", manifest_base["hardware_backend"])
    write_json(
        diagnostics_dir / "resume_reuse_summary.json",
        {
            "pca_fit_count": int(len(pca_results)),
            "score_parts_expected": int(len(splits)),
            "score_parts_reused": int(reused_score_parts),
            "score_parts_computed": int(computed_score_parts),
            "score_part_reuse_policy": "schema/split/width/family/ridge/seed exact metadata match",
        },
    )
    reports = write_phase4_reports(output_root, manifest_base, pca_manifest_rows, summary_df, gates, best_rows, report_dir=report_dir)
    manifest = {
        **manifest_base,
        "output_root": str(output_root),
        "scripts_used": [
            "backend/scripts/again_dense_2hz_phase4_pca_bridge.py",
            "backend/scripts/build_again_dense_2hz_train_only_pca_features.py",
            "backend/scripts/run_again_dense_2hz_pca_bridge_benchmark.py",
            "backend/scripts/summarize_again_dense_2hz_phase4_pca_bridge.py",
        ],
        "pca_fit_count": len(pca_manifest_rows),
        "metrics": {
            "fold_metrics": str(fold_path),
            "summary_metrics": str(summary_path),
            "control_metrics": str(control_path),
            "grouped_video_metrics": str(grouped_path),
            "blocked_temporal_metrics": str(blocked_path),
            "delta_vs_phase3": str(delta_path),
        },
        "promotion": {
            "promotion_gates": str(promotion_dir / "promotion_gates.json"),
            "best_lanes_by_target": str(promotion_dir / "best_lanes_by_target.json"),
        },
        "reports": reports,
    }
    write_json(output_root / "run_manifest.json", manifest)
    write_json(output_root / "summary.json", manifest)
    return manifest


def build_features_cli(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build/run train-only PCA bridge features for dense AGAIN 2Hz.")
    add_common_args(parser)
    args = parser.parse_args(argv)
    manifest = run_phase4_from_args(args, dry_run=args.dry_run)
    print(json.dumps(clean_json(manifest), indent=2, sort_keys=True))


def benchmark_cli(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run dense AGAIN 2Hz Phase 4 PCA bridge benchmark.")
    add_common_args(parser)
    args = parser.parse_args(argv)
    manifest = run_phase4_from_args(args, dry_run=False)
    print(json.dumps(clean_json(manifest), indent=2, sort_keys=True))


def summarize_cli(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize an existing dense AGAIN Phase 4 PCA bridge output root.")
    parser.add_argument("--phase4-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    root = args.phase4_root
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(root / "metrics" / "phase4_summary_metrics.csv")
    gates = json.loads((root / "promotion" / "promotion_gates.json").read_text(encoding="utf-8"))
    best_rows = json.loads((root / "promotion" / "best_lanes_by_target.json").read_text(encoding="utf-8"))
    reports = write_phase4_reports(
        root,
        summary,
        json.loads((root / "pca_feature_manifest.json").read_text(encoding="utf-8")).get("fits", []),
        metrics,
        gates,
        best_rows,
        report_dir=args.report_dir,
    )
    summary["reports"] = reports
    write_json(root / "summary.json", summary)
    print(json.dumps(clean_json({"phase4_root": str(root), "reports": reports}), indent=2, sort_keys=True))


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dense-root", type=Path, default=DEFAULT_DENSE_ROOT)
    parser.add_argument("--labels", type=Path, default=None, help="Accepted for command compatibility; labels are read from dense-root.")
    parser.add_argument("--output-root", "--phase4-feature-root", type=Path, default=None)
    parser.add_argument("--widths", default=",".join(str(x) for x in DEFAULT_WIDTHS))
    parser.add_argument("--feature-families", default=",".join(DEFAULT_FEATURE_FAMILIES))
    parser.add_argument("--validation-protocols", default=",".join(DEFAULT_PROTOCOLS))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--ridge-alpha-grid", default=",".join(str(x) for x in DEFAULT_RIDGE_ALPHA_GRID))
    parser.add_argument("--fixed-ridge-alpha", action="store_true")
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--oversampling", type=int, default=32)
    parser.add_argument("--power-iterations", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-cortical-memmap", action="store_true")
    parser.add_argument("--max-fit-count", type=int, default=None, help="Debug/budget guard. Full Phase 4 should leave this unset.")
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))


def run_phase4_from_args(args: argparse.Namespace, *, dry_run: bool) -> dict[str, Any]:
    widths = parse_widths(args.widths)
    families = parse_csv_set(args.feature_families)
    protocols = parse_csv_set(args.validation_protocols)
    unknown = sorted(set(families) - set(DEFAULT_FEATURE_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown feature families: {unknown}")
    output_root = args.output_root or phase4_output_root()
    alpha_grid = None if args.fixed_ridge_alpha else parse_alpha_grid(args.ridge_alpha_grid)
    if args.labels is not None and args.labels != args.dense_root / "labels_aligned_2hz.parquet":
        raise ValueError("Phase 4 requires labels_aligned_2hz.parquet under --dense-root; do not pass stale labels")
    return run_phase4(
        dense_root=args.dense_root,
        output_root=output_root,
        widths=widths,
        feature_families=families,
        validation_protocols=protocols,
        n_splits=args.n_splits,
        random_seed=args.random_seed,
        ridge_alpha=args.ridge_alpha,
        ridge_alpha_grid=alpha_grid,
        batch_size=args.batch_size,
        oversampling=args.oversampling,
        power_iterations=args.power_iterations,
        dry_run=dry_run,
        resume=args.resume,
        force_cortical_memmap=args.force_cortical_memmap,
        max_fit_count=args.max_fit_count,
        report_dir=args.report_dir,
    )


if __name__ == "__main__":
    benchmark_cli()
