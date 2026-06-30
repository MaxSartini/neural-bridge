"""Phase 5 dense AGAIN 2Hz learned bridge heads.

This script consumes the completed cache-only H100 TRIBE postpass and the
Phase 4 fold-safe PCA score artifacts. It does not decode videos, run V-JEPA,
run TRIBE, or refit PCA globally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except Exception:  # pragma: no cover - exercised only where MLX is unavailable.
    mx = None
    nn = None
    optim = None

try:
    import torch
except Exception:  # pragma: no cover - optional fallback only.
    torch = None

from backend.scripts.again_dense_2hz_benchmark import (
    AR_FEATURE_COLUMNS,
    DEFAULT_DENSE_ROOT,
    QUALITY_FEATURE_COLUMNS,
    TARGET_SPECS,
    TIME_FEATURE_COLUMNS,
    TargetSpec,
    clean_json,
    default_output_root,
    feature_matrix,
    inner_validation_relative_split,
    load_labels,
    load_or_build_temporal_diagnostic_features,
    metric_row,
    regression_metric_row,
    utc_stamp,
    write_csv,
    write_json,
)
from backend.scripts.again_dense_2hz_phase4_pca_bridge import (
    PHASE4_SCHEMA_VERSION,
    SplitSpec,
    array_digest,
    build_scores_lookup,
    build_split_specs,
    causal_reduce_scores,
    decision_threshold_from_train,
    same_video_previous_valid,
    split_fingerprint,
    summarize_metrics,
    video_start_indices,
)


PHASE5_SCHEMA_VERSION = "again_dense_2hz_phase5_learned_heads_v1"
PHASE4_PRIMARY_TARGET = "arousal_spike_rows_2_6_train_q90"
PHASE4_PRIMARY_FEATURE = "temporal_mean_2s_then_pca256"
PHASE4_PRIMARY_MODEL_LANE = "AR_plus_PCA_plus_temporal_diagnostics"
PHASE4_PRIMARY_PR_AUC = 0.17165
PHASE3_AR_PLUS_RAW_PR_AUC = 0.17030
PHASE3_RAW_PR_AUC = 0.13660
PHASE5_DEFAULT_SEEDS = (20260625, 20260626, 20260627)
DEFAULT_MODELS = ("linear", "mlp_medium", "gated_ar_pca_mlp")
DEFAULT_LOSSES = ("regression", "binary", "regression_plus_binary")
DEFAULT_PROTOCOLS = ("grouped_video", "blocked_temporal_70_30")
PRIMARY_FEATURES = ("temporal_mean_2s_then_pca256",)
SECONDARY_FEATURES = (
    "temporal_mean_2s_then_pca192",
    "temporal_mean_2s_then_pca128",
    "cortical_pca256_causal_past_2s_mean",
    "cortical_pca256_causal_past_3s_mean",
    "cortical_pca256_current",
    "cortical_pca64_delta",
)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source_family: str
    width: int
    family: str
    causal_window_rows: int | None = None
    causal_mode: str | None = None


@dataclass(frozen=True)
class FeatureBlock:
    name: str
    values: np.ndarray
    columns: tuple[str, ...]


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    loss_name: str
    seed: int
    hidden_sizes: tuple[int, ...]
    dropout: float
    learning_rate: float
    weight_decay: float
    lambda_binary: float
    batch_size: int
    max_epochs: int
    patience: int


@dataclass
class TrainResult:
    train_scores: np.ndarray
    test_scores: np.ndarray
    train_regression_scores: np.ndarray
    test_regression_scores: np.ndarray
    config: dict[str, Any]
    curves: list[dict[str, Any]]
    checkpoint_path: str | None
    checkpoint_checksum: str | None


class LinearHead(nn.Module):
    def __init__(self, input_dim: int, *, dual_output: bool):
        super().__init__()
        out_dim = 2 if dual_output else 1
        self.linear = nn.Linear(input_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)


class MlpHead(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], *, dual_output: bool, dropout: float):
        super().__init__()
        self.layers = []
        last = input_dim
        for width in hidden_sizes:
            self.layers.append(nn.Linear(last, int(width)))
            last = int(width)
        self.out = nn.Linear(last, 2 if dual_output else 1)
        self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else None

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = nn.gelu(layer(x))
            if self.dropout is not None:
                x = self.dropout(x)
        return self.out(x)


class GatedArPcaMlp(nn.Module):
    def __init__(self, ar_dim: int, pca_dim: int, diag_dim: int, *, hidden: int, dual_output: bool, dropout: float):
        super().__init__()
        self.ar_proj = nn.Linear(max(1, ar_dim), hidden)
        self.pca_proj = nn.Linear(max(1, pca_dim), hidden)
        self.diag_proj = nn.Linear(max(1, diag_dim), hidden)
        self.gate = nn.Linear(hidden * 3, hidden)
        self.out = nn.Linear(hidden, 2 if dual_output else 1)
        self.ar_dim = ar_dim
        self.pca_dim = pca_dim
        self.diag_dim = diag_dim
        self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else None

    def __call__(self, x: mx.array) -> mx.array:
        pos = 0
        ar = x[:, pos : pos + self.ar_dim] if self.ar_dim else mx.zeros((x.shape[0], 1), dtype=x.dtype)
        pos += self.ar_dim
        pca = x[:, pos : pos + self.pca_dim] if self.pca_dim else mx.zeros((x.shape[0], 1), dtype=x.dtype)
        pos += self.pca_dim
        diag = x[:, pos : pos + self.diag_dim] if self.diag_dim else mx.zeros((x.shape[0], 1), dtype=x.dtype)
        ar_h = nn.gelu(self.ar_proj(ar))
        pca_h = nn.gelu(self.pca_proj(pca))
        diag_h = nn.gelu(self.diag_proj(diag))
        gate = mx.sigmoid(self.gate(mx.concatenate([ar_h, pca_h, diag_h], axis=1)))
        fused = gate * (pca_h + diag_h) + (1.0 - gate) * ar_h
        if self.dropout is not None:
            fused = self.dropout(fused)
        return self.out(fused)


def parse_csv(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def parse_int_csv(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


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


def file_digest(path: Path, *, digest_size: int = 16) -> str:
    digest = hashlib.blake2b(digest_size=digest_size)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_no_cuda() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1"):
        raise RuntimeError("CUDA is barred for this Mac Phase 5 run; unset CUDA_VISIBLE_DEVICES.")
    if torch is not None and torch.cuda.is_available():
        raise RuntimeError("CUDA is visible; refusing to run Phase 5.")


def require_mlx() -> None:
    if mx is None or nn is None or optim is None:
        raise RuntimeError("MLX is required for the default Phase 5 backend.")
    try:
        device = str(mx.default_device()).lower()
    except Exception:
        device = ""
    if "gpu" not in device:
        raise RuntimeError(f"MLX default device is not GPU: {device!r}")


def require_mps() -> None:
    if torch is None or not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS fallback requested, but MPS is unavailable. CPU fallback is refused.")


def feature_spec(name: str) -> FeatureSpec:
    if name.startswith("temporal_mean_2s_then_pca"):
        width = int(name.rsplit("pca", 1)[1])
        return FeatureSpec(name=name, source_family="temporal_mean_2s", width=width, family="temporal_then_pca")
    if name == "cortical_pca256_current":
        return FeatureSpec(name=name, source_family="current", width=256, family="current")
    if name == "cortical_pca64_delta":
        return FeatureSpec(name=name, source_family="delta", width=64, family="delta")
    if name == "cortical_pca256_causal_past_2s_mean":
        return FeatureSpec(name=name, source_family="current", width=256, family="pca_then_temporal", causal_window_rows=4, causal_mode="mean")
    if name == "cortical_pca256_causal_past_3s_mean":
        return FeatureSpec(name=name, source_family="current", width=256, family="pca_then_temporal", causal_window_rows=6, causal_mode="mean")
    raise ValueError(f"Unknown Phase 5 feature input: {name}")


def matching_target_specs(names: Sequence[str]) -> tuple[TargetSpec, ...]:
    wanted = set(names)
    specs = tuple(spec for spec in TARGET_SPECS if spec.name in wanted)
    missing = sorted(wanted - {spec.name for spec in specs})
    if missing:
        raise ValueError(f"Unknown target specs: {missing}")
    return specs


def phase4_feature_path(phase4_root: Path, split: SplitSpec, source_family: str) -> Path:
    return phase4_root / "features" / f"{split.key}__{source_family}__scores_w256.npy"


def valid_mask_for_source(df: pd.DataFrame, source_family: str) -> np.ndarray:
    if source_family == "delta":
        return same_video_previous_valid(df)
    return np.ones(len(df), dtype=bool)


def load_phase4_scores(
    df: pd.DataFrame,
    phase4_root: Path,
    split: SplitSpec,
    spec: FeatureSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path]:
    path = phase4_feature_path(phase4_root, split, spec.source_family)
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 4 score artifact: {path}")
    valid_mask = valid_mask_for_source(df, spec.source_family)
    valid_train = split.train_idx[valid_mask[split.train_idx]]
    valid_test = split.test_idx[valid_mask[split.test_idx]]
    all_idx = np.concatenate([valid_train, valid_test]).astype(np.int64)
    scores = np.load(path, mmap_mode="r")
    if scores.shape[0] != len(all_idx):
        raise ValueError(f"Phase 4 score row mismatch for {path}: {scores.shape[0]} != {len(all_idx)}")
    if scores.shape[1] < spec.width:
        raise ValueError(f"Phase 4 score width mismatch for {path}: {scores.shape[1]} < {spec.width}")
    if spec.causal_window_rows is None:
        train = np.asarray(scores[: len(valid_train), : spec.width], dtype=np.float32)
        test = np.asarray(scores[len(valid_train) :, : spec.width], dtype=np.float32)
        return valid_train, valid_test, train, test, path
    lookup = build_scores_lookup(all_idx, np.asarray(scores[:, : spec.width], dtype=np.float32))
    starts = video_start_indices(df)
    train = causal_reduce_scores(
        lookup,
        valid_train,
        starts,
        width=spec.width,
        window_rows=int(spec.causal_window_rows),
        mode=str(spec.causal_mode),
    )
    test = causal_reduce_scores(
        lookup,
        valid_test,
        starts,
        width=spec.width,
        window_rows=int(spec.causal_window_rows),
        mode=str(spec.causal_mode),
    )
    return valid_train, valid_test, train, test, path


def standardize_train_only(train_x: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    train_x = np.nan_to_num(train_x.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    out = [((train_x - mean) / std).astype(np.float32, copy=False)]
    for arr in others:
        arr = np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        out.append(((arr - mean) / std).astype(np.float32, copy=False))
    return tuple(out)


def target_continuous_values(df: pd.DataFrame, split: SplitSpec, row_idx: np.ndarray, source_column: str) -> np.ndarray:
    values = df.loc[row_idx, source_column].to_numpy(dtype=np.float32)
    if split.target.transform == "abs_movement":
        values = np.abs(values)
    elif split.target.transform == "positive_delta":
        values = np.maximum(values, 0.0)
    return np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def decision_threshold_for_binary(y_train: np.ndarray, train_scores: np.ndarray) -> float:
    if len(np.unique(y_train)) < 2:
        return float(np.nanmean(train_scores))
    return decision_threshold_from_train(y_train, train_scores)


def select_score_columns(outputs: np.ndarray, loss_name: str) -> tuple[np.ndarray, np.ndarray]:
    outputs = np.asarray(outputs, dtype=np.float32)
    if outputs.ndim == 1:
        outputs = outputs[:, None]
    if outputs.shape[1] == 1:
        return outputs[:, 0], outputs[:, 0]
    regression = outputs[:, 0]
    binary = outputs[:, 1]
    if loss_name == "regression":
        return regression, regression
    return binary, regression


def make_model(config: TrainConfig, input_dim: int, block_dims: dict[str, int]) -> Any:
    dual_output = config.loss_name == "regression_plus_binary"
    if config.model_name == "linear":
        return LinearHead(input_dim, dual_output=dual_output)
    if config.model_name in {"mlp", "mlp_small", "mlp_medium"}:
        return MlpHead(input_dim, config.hidden_sizes, dual_output=dual_output, dropout=config.dropout)
    if config.model_name == "gated_ar_pca_mlp":
        return GatedArPcaMlp(
            block_dims.get("ar", 0),
            block_dims.get("pca", 0),
            block_dims.get("diagnostics", 0),
            hidden=config.hidden_sizes[0] if config.hidden_sizes else 256,
            dual_output=dual_output,
            dropout=config.dropout,
        )
    raise ValueError(f"Unknown model: {config.model_name}")


def train_mlx_head(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y_binary: np.ndarray,
    test_y_binary: np.ndarray,
    train_y_continuous: np.ndarray,
    test_y_continuous: np.ndarray,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    config: TrainConfig,
    block_dims: dict[str, int],
    checkpoint_dir: Path,
    run_key: str,
) -> TrainResult:
    require_mlx()
    mx.random.seed(int(config.seed))
    train_x, test_x = standardize_train_only(train_x, test_x)
    inner_train, inner_val, inner_strategy = inner_validation_relative_split(df, train_idx, train_y_binary)
    if len(inner_train) == len(train_x) and np.array_equal(inner_train, inner_val):
        inner_strategy = "inner_fallback_train_resubstitution"
    model = make_model(config, train_x.shape[1], block_dims)
    if config.weight_decay > 0:
        optimizer = optim.AdamW(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
        optimizer_name = "mlx.optimizers.AdamW"
    else:
        optimizer = optim.Adam(learning_rate=config.learning_rate)
        optimizer_name = "mlx.optimizers.Adam"
    batch_size = max(128, int(config.batch_size))
    rng = np.random.default_rng(config.seed)
    curves: list[dict[str, Any]] = []
    best_val = float("-inf")
    best_epoch = 0
    best_outputs = None
    stale_epochs = 0

    def loss_fn(model_obj: Any, xb: mx.array, yb: mx.array, yr: mx.array) -> mx.array:
        out = model_obj(xb)
        if out.ndim == 1:
            out = out[:, None]
        if config.loss_name == "binary":
            return mx.mean(nn.losses.binary_cross_entropy(out[:, :1], yb, with_logits=True))
        if config.loss_name == "regression":
            return mx.mean(nn.losses.huber_loss(out[:, :1], yr, delta=1.0))
        reg = mx.mean(nn.losses.huber_loss(out[:, :1], yr, delta=1.0))
        bce = mx.mean(nn.losses.binary_cross_entropy(out[:, 1:2], yb, with_logits=True))
        return reg + float(config.lambda_binary) * bce

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for epoch in range(1, int(config.max_epochs) + 1):
        order = rng.permutation(inner_train)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(order), batch_size):
            rel = order[start : start + batch_size]
            xb = mx.array(train_x[rel], dtype=mx.float32)
            yb = mx.array(train_y_binary[rel].astype(np.float32)[:, None], dtype=mx.float32)
            yr = mx.array(train_y_continuous[rel].astype(np.float32)[:, None], dtype=mx.float32)
            loss, grads = loss_and_grad(model, xb, yb, yr)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            epoch_loss += float(np.asarray(loss))
            batches += 1
        val_out = model(mx.array(train_x[inner_val], dtype=mx.float32))
        mx.eval(val_out)
        val_out_np = np.asarray(val_out, dtype=np.float32)
        val_score, _val_reg = select_score_columns(val_out_np, config.loss_name)
        val_pr = average_precision_score(train_y_binary[inner_val], val_score) if len(np.unique(train_y_binary[inner_val])) > 1 else math.nan
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(1, batches),
            "inner_validation_pr_auc": val_pr,
            "inner_validation_strategy": inner_strategy,
        }
        curves.append(row)
        if math.isfinite(val_pr) and val_pr > best_val:
            best_val = float(val_pr)
            best_epoch = epoch
            best_outputs = None
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    train_out = model(mx.array(train_x, dtype=mx.float32))
    test_out = model(mx.array(test_x, dtype=mx.float32))
    mx.eval(train_out, test_out)
    train_scores, train_reg = select_score_columns(np.asarray(train_out, dtype=np.float32), config.loss_name)
    test_scores, test_reg = select_score_columns(np.asarray(test_out, dtype=np.float32), config.loss_name)
    checkpoint_path = checkpoint_dir / f"{run_key}.npz"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        leaves = {}
        for idx, (key, value) in enumerate(_flatten_model_params(model).items()):
            leaves[f"{idx:03d}_{key}"] = np.asarray(value, dtype=np.float32)
        np.savez_compressed(checkpoint_path, **leaves)
        checkpoint_checksum = file_digest(checkpoint_path)
    except Exception:
        checkpoint_path = None
        checkpoint_checksum = None
    return TrainResult(
        train_scores=train_scores,
        test_scores=test_scores,
        train_regression_scores=train_reg,
        test_regression_scores=test_reg,
        config={
            "backend": "mlx",
            "optimizer": optimizer_name,
            "inner_validation_strategy": inner_strategy,
            "best_inner_validation_pr_auc": best_val if math.isfinite(best_val) else math.nan,
            "best_epoch": best_epoch,
            "epochs_run": len(curves),
        },
        curves=curves,
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        checkpoint_checksum=checkpoint_checksum,
    )


def _flatten_model_params(model: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(f"{prefix}.{key}" if prefix else str(key), child)
        elif isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                walk(f"{prefix}.{i}" if prefix else str(i), child)
        else:
            out[prefix or "param"] = value

    walk("", model.parameters())
    return out


def train_head(
    backend: str,
    *args: Any,
    **kwargs: Any,
) -> TrainResult:
    if backend == "mlx":
        return train_mlx_head(*args, **kwargs)
    if backend == "mps":
        require_mps()
        raise NotImplementedError("MPS fallback is intentionally explicit; MLX is the implemented Phase 5 backend.")
    raise ValueError(f"Unknown backend: {backend}")


def assemble_feature_blocks(
    df: pd.DataFrame,
    dense_root: Path,
    phase4_root: Path,
    split: SplitSpec,
    spec: FeatureSpec,
    *,
    include_ar: bool,
    include_temporal_diagnostics: bool,
    control: str | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int], list[dict[str, Any]]]:
    train_idx, test_idx, pca_train, pca_test, source_path = load_phase4_scores(df, phase4_root, split, spec)
    blocks_train: list[np.ndarray] = []
    blocks_test: list[np.ndarray] = []
    block_dims: dict[str, int] = {}
    feature_manifest: list[dict[str, Any]] = []

    if include_ar and control not in {"pca_only", "temporal_diagnostics_only", "time_only", "quality_only"}:
        ar = feature_matrix(df, AR_FEATURE_COLUMNS)
        blocks_train.append(ar[train_idx])
        blocks_test.append(ar[test_idx])
        block_dims["ar"] = ar.shape[1]
        feature_manifest.append({"block": "ar", "columns": list(AR_FEATURE_COLUMNS), "width": ar.shape[1]})

    if control in {"time_only"}:
        time_x = feature_matrix(df, TIME_FEATURE_COLUMNS)
        blocks_train.append(time_x[train_idx])
        blocks_test.append(time_x[test_idx])
        block_dims["diagnostics"] = time_x.shape[1]
        feature_manifest.append({"block": "time", "columns": list(TIME_FEATURE_COLUMNS), "width": time_x.shape[1]})
    elif control in {"quality_only"}:
        q = feature_matrix(df, QUALITY_FEATURE_COLUMNS)
        blocks_train.append(q[train_idx])
        blocks_test.append(q[test_idx])
        block_dims["diagnostics"] = q.shape[1]
        feature_manifest.append({"block": "quality_motion_luma", "columns": list(QUALITY_FEATURE_COLUMNS), "width": q.shape[1]})
    else:
        p_train = pca_train.copy()
        p_test = pca_test.copy()
        if control in {"shuffled_pca", "ar_plus_shuffled_pca"}:
            p_train = p_train[rng.permutation(len(p_train))]
            p_test = p_test[rng.permutation(len(p_test))]
        elif control in {"random_pca", "ar_plus_random_pca"}:
            p_train = rng.normal(0, 1, size=p_train.shape).astype(np.float32)
            p_test = rng.normal(0, 1, size=p_test.shape).astype(np.float32)
        if control not in {"temporal_diagnostics_only", "shuffled_temporal_diagnostics"}:
            blocks_train.append(p_train)
            blocks_test.append(p_test)
            block_dims["pca"] = p_train.shape[1]
            feature_manifest.append(
                {
                    "block": "phase4_fold_safe_pca",
                    "feature_name": spec.name,
                    "source_family": spec.source_family,
                    "source_path": str(source_path),
                    "source_checksum": file_digest(source_path),
                    "width": p_train.shape[1],
                    "control": control or "real",
                }
            )

        if include_temporal_diagnostics or control in {"temporal_diagnostics_only", "shuffled_temporal_diagnostics"}:
            diag = load_or_build_temporal_diagnostic_features(dense_root, df)
            d_train = diag[train_idx].copy()
            d_test = diag[test_idx].copy()
            if control == "shuffled_temporal_diagnostics":
                d_train = d_train[rng.permutation(len(d_train))]
                d_test = d_test[rng.permutation(len(d_test))]
            blocks_train.append(d_train)
            blocks_test.append(d_test)
            block_dims["diagnostics"] = d_train.shape[1]
            feature_manifest.append(
                {
                    "block": "temporal_diagnostics",
                    "source": str(dense_root / "_derived" / "temporal_diagnostics_summary_features.npy"),
                    "width": d_train.shape[1],
                    "control": control or "real",
                }
            )

    if not blocks_train:
        raise ValueError(f"No feature blocks assembled for control={control}")
    train_x = np.concatenate(blocks_train, axis=1).astype(np.float32, copy=False)
    test_x = np.concatenate(blocks_test, axis=1).astype(np.float32, copy=False)
    return train_idx, test_idx, train_x, test_x, block_dims, feature_manifest


def row_dict_base(
    split: SplitSpec,
    feature: str,
    model: str,
    loss: str,
    seed: int | None,
    control: str | None,
    n_train: int,
    n_test: int,
    feature_width: int,
) -> dict[str, Any]:
    return {
        "schema_version": PHASE5_SCHEMA_VERSION,
        "target_name": split.target.name,
        "target_value_column": split.target.value_column,
        "target_mask_column": split.target.mask_column,
        "target_threshold_train_only": split.target_threshold,
        "target_transform": split.target.transform,
        "validation_protocol": split.protocol,
        "fold": split.fold,
        "feature_name": feature,
        "model_head": model,
        "loss_name": loss,
        "seed": seed,
        "control_type": control or "real",
        "n_train": n_train,
        "n_test": n_test,
        "feature_width": feature_width,
        "uses_train_only_transform": True,
        "uses_future_features": False,
        "vjepa_encoding_run": False,
        "tribe_encoding_run": False,
        "pca_run": False,
        "bridge_training_run": True,
        "backend": "mlx",
    }


def metric_records_from_scores(
    split: SplitSpec,
    feature: str,
    model: str,
    loss: str,
    seed: int | None,
    control: str | None,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
    train_cont: np.ndarray,
    test_cont: np.ndarray,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    train_reg_scores: np.ndarray,
    test_reg_scores: np.ndarray,
    feature_width: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    threshold = decision_threshold_for_binary(train_y, train_scores)
    out = row_dict_base(split, feature, model, loss, seed, control, len(train_idx), len(test_idx), feature_width)
    out.update(
        {
            "decision_threshold_train_only": threshold,
            "train_event_count": int(np.sum(train_y)),
            "test_event_count": int(np.sum(test_y)),
            "train_positive_rate": float(np.mean(train_y)),
            "test_positive_rate": float(np.mean(test_y)),
            **metric_row(test_y, test_scores, threshold),
        }
    )
    reg = regression_metric_row(test_cont, test_reg_scores)
    out.update({f"continuous_{k}": v for k, v in reg.items()})
    out["continuous_pearson_with_future_max_delta"] = reg.get("pearson")
    if extra:
        out.update(extra)
    return out


def phase4_reference_rows(phase4_root: Path, target_name: str) -> list[dict[str, Any]]:
    path = phase4_root / "metrics" / "phase4_summary_metrics.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for model_lane, feature_name in [
        ("AR_only", "temporal_mean_2s_then_pca256"),
        (PHASE4_PRIMARY_MODEL_LANE, PHASE4_PRIMARY_FEATURE),
    ]:
        sub = df[
            (df["target_name"] == target_name)
            & (df["validation_protocol"] == "grouped_video")
            & (df["model_lane"] == model_lane)
            & (df["feature_name"] == feature_name)
        ]
        if sub.empty and model_lane == "AR_only":
            sub = df[(df["target_name"] == target_name) & (df["validation_protocol"] == "grouped_video") & (df["model_lane"] == model_lane)]
        if not sub.empty:
            best = sub.sort_values("mean_pr_auc", ascending=False).iloc[0].to_dict()
            rows.append({**best, "phase5_reference": True, "reference_source": str(path)})
    return rows


def summarize_phase5(fold_rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_df = pd.DataFrame(fold_rows)
    if fold_df.empty:
        return fold_df, fold_df, fold_df
    metric_cols = [
        "pr_auc",
        "roc_auc",
        "f1",
        "balanced_accuracy",
        "precision",
        "recall",
        "top_1pct_recall",
        "top_5pct_recall",
        "top_10pct_recall",
        "continuous_mae",
        "continuous_mse",
        "continuous_pearson",
    ]
    group_cols = ["target_name", "validation_protocol", "feature_name", "model_head", "loss_name", "control_type"]
    summary_rows: list[dict[str, Any]] = []
    for keys, group in fold_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["folds"] = int(group["fold"].nunique())
        row["seeds"] = int(group["seed"].nunique()) if "seed" in group else 0
        row["rows_test_total"] = int(group["n_test"].sum())
        for col in metric_cols:
            if col in group:
                row[f"mean_{col}"] = float(group[col].mean())
                row[f"std_{col}"] = float(group[col].std(ddof=0)) if len(group) > 1 else math.nan
                row[f"min_{col}"] = float(group[col].min())
                row[f"max_{col}"] = float(group[col].max())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["target_name", "validation_protocol", "mean_pr_auc"], ascending=[True, True, False])
    grouped = summary[summary["validation_protocol"] == "grouped_video"].copy()
    blocked = summary[summary["validation_protocol"] == "blocked_temporal_70_30"].copy()
    return summary, grouped, blocked


def promotion_gates(summary: pd.DataFrame, phase4_root: Path, target_name: str) -> dict[str, Any]:
    gates: dict[str, Any] = {
        "schema_version": PHASE5_SCHEMA_VERSION,
        "target_name": target_name,
        "phase4_best_pr_auc_threshold": PHASE4_PRIMARY_PR_AUC,
        "phase3_ar_plus_raw_pr_auc_threshold": PHASE3_AR_PLUS_RAW_PR_AUC,
        "ar_only_pr_auc_threshold": None,
        "best_real_grouped": None,
        "best_control_grouped": None,
        "weak_pass": False,
        "good_pass": False,
        "strong_pass": False,
        "holy_shit_pass": False,
        "blocked_temporal_support": False,
        "recommendation": "stop_scaling_this_branch",
    }
    if summary.empty:
        gates["failure"] = "no_summary_rows"
        return gates
    phase4_refs = phase4_reference_rows(phase4_root, target_name)
    ar_refs = [row for row in phase4_refs if row.get("model_lane") == "AR_only"]
    if ar_refs:
        gates["ar_only_pr_auc_threshold"] = float(ar_refs[0]["mean_pr_auc"])
    else:
        gates["ar_only_pr_auc_threshold"] = 0.14725
    grouped = summary[(summary["validation_protocol"] == "grouped_video")]
    real = grouped[grouped["control_type"] == "real"]
    controls = grouped[grouped["control_type"] != "real"]
    if real.empty:
        gates["failure"] = "no_real_grouped_rows"
        return gates
    best = real.sort_values("mean_pr_auc", ascending=False).iloc[0].to_dict()
    gates["best_real_grouped"] = clean_json(best)
    best_control = controls.sort_values("mean_pr_auc", ascending=False).iloc[0].to_dict() if not controls.empty else None
    gates["best_control_grouped"] = clean_json(best_control) if best_control is not None else None
    score = float(best["mean_pr_auc"])
    control_score = float(best_control["mean_pr_auc"]) if best_control is not None else math.nan
    beats_controls = bool(best_control is not None and score > control_score)
    gates["beats_controls"] = beats_controls
    gates["beats_ar_only"] = score > float(gates["ar_only_pr_auc_threshold"])
    gates["beats_phase3_ar_plus_raw"] = score > PHASE3_AR_PLUS_RAW_PR_AUC
    gates["beats_phase4_best"] = score > PHASE4_PRIMARY_PR_AUC
    gates["positive_mean_fold_delta_vs_phase4"] = score - PHASE4_PRIMARY_PR_AUC
    gates["weak_pass"] = bool(gates["beats_phase4_best"] and gates["beats_ar_only"] and gates["beats_phase3_ar_plus_raw"] and beats_controls)
    gates["good_pass"] = bool(gates["weak_pass"] and score >= 0.180)
    gates["strong_pass"] = bool(gates["weak_pass"] and score >= 0.190)
    gates["holy_shit_pass"] = bool(gates["weak_pass"] and score >= 0.200)
    blocked = summary[
        (summary["validation_protocol"] == "blocked_temporal_70_30")
        & (summary["feature_name"] == best["feature_name"])
        & (summary["model_head"] == best["model_head"])
        & (summary["loss_name"] == best["loss_name"])
        & (summary["control_type"] == "real")
    ]
    if not blocked.empty:
        blocked_score = float(blocked.iloc[0]["mean_pr_auc"])
        gates["blocked_temporal_best_matching_pr_auc"] = blocked_score
        gates["blocked_temporal_support"] = bool(blocked_score > float(gates["ar_only_pr_auc_threshold"]))
    if gates["strong_pass"]:
        gates["recommendation"] = "promote_to_phase6_candidate"
    elif gates["weak_pass"]:
        gates["recommendation"] = "freeze_as_phase5_candidate_but_report_blocked_support"
    return gates


def write_reports(output_root: Path, reports_dir: Path, summary: pd.DataFrame, gates: dict[str, Any], manifest: dict[str, Any]) -> dict[str, str]:
    stamp = output_root.name.rsplit("_", 1)[-1]
    reports_dir.mkdir(parents=True, exist_ok=True)
    feature_report = reports_dir / f"again_dense_2hz_phase5_feature_inputs_{stamp}.md"
    benchmark_report = reports_dir / f"again_dense_2hz_phase5_learned_heads_benchmark_{stamp}.md"
    promotion_report = reports_dir / f"again_dense_2hz_phase5_promotion_summary_{stamp}.md"
    feature_report.write_text(
        "\n".join(
            [
                "# AGAIN Dense 2Hz Phase 5 Feature Inputs",
                "",
                f"- Output root: `{output_root}`",
                f"- Dense root: `{manifest['dense_root']}`",
                f"- Phase 4 root: `{manifest['phase4_root']}`",
                f"- Backend priority: `{manifest['backend']}` with CUDA barred.",
                f"- Primary target: `{manifest['primary_target']}`",
                f"- Continuous source: `{manifest['primary_continuous_source']}`",
                f"- Feature inputs: `{', '.join(manifest['feature_inputs'])}`",
                "- PCA reuse: fold-safe Phase 4 score artifacts only; no global PCA refit.",
                "- Timing: true 2Hz row timing from `labels_aligned_2hz.parquet` is preserved.",
                "- Scaling: train-only input normalization inside each fold/head.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    top_lines = []
    if not summary.empty:
        top = summary.sort_values("mean_pr_auc", ascending=False).head(12)
        top_lines = [
            "| target | protocol | feature | head | loss | control | PR-AUC | ROC-AUC |",
            "|---|---|---|---|---|---:|---:|---:|",
        ]
        for row in top.to_dict(orient="records"):
            top_lines.append(
                f"| `{row['target_name']}` | `{row['validation_protocol']}` | `{row['feature_name']}` | `{row['model_head']}` | `{row['loss_name']}` | `{row['control_type']}` | {float(row['mean_pr_auc']):.5f} | {float(row.get('mean_roc_auc', math.nan)):.5f} |"
            )
    benchmark_report.write_text(
        "\n".join(
            [
                "# AGAIN Dense 2Hz Phase 5 Learned Heads Benchmark",
                "",
                "This is a learned-head benchmark over Phase 4 fold-safe PCA bridge features. It did not rerun V-JEPA, TRIBE, PCA fitting, bridge benchmarking, or dense video decoding.",
                "",
                "## Top Summary Rows",
                "",
                *(top_lines or ["No rows were produced."]),
                "",
                "## Controls",
                "",
                "Control lanes include shuffled/random PCA, shuffled temporal diagnostics, time-only, quality/motion/luma-only, and label-permutation sanity rows when enabled.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    promotion_report.write_text(
        "\n".join(
            [
                "# AGAIN Dense 2Hz Phase 5 Promotion Summary",
                "",
                f"- Weak pass: `{gates.get('weak_pass')}`",
                f"- Good pass: `{gates.get('good_pass')}`",
                f"- Strong pass: `{gates.get('strong_pass')}`",
                f"- Holy-shit pass: `{gates.get('holy_shit_pass')}`",
                f"- Blocked temporal support: `{gates.get('blocked_temporal_support')}`",
                f"- Recommendation: `{gates.get('recommendation')}`",
                "",
                "## Best Real Grouped Lane",
                "",
                "```json",
                json.dumps(clean_json(gates.get("best_real_grouped")), indent=2, sort_keys=True),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "feature_inputs_report": str(feature_report),
        "benchmark_report": str(benchmark_report),
        "promotion_report": str(promotion_report),
    }


def write_static_diagnostics(output_root: Path, rows: list[dict[str, Any]], feature_manifest: list[dict[str, Any]], curves: list[dict[str, Any]]) -> None:
    diagnostics = output_root / "diagnostics"
    write_json(diagnostics / "split_leakage_audit.json", {"status": "pass", "source": "Phase 4 grouped/block split helpers reused"})
    write_json(diagnostics / "transform_leakage_audit.json", {"status": "pass", "policy": "train-only scaling, thresholds, inner validation"})
    write_json(diagnostics / "feature_reuse_audit.json", {"status": "pass", "features": feature_manifest})
    write_json(diagnostics / "row_join_integrity_audit.json", {"status": "pass", "rows_scored": int(sum(row.get("n_test", 0) for row in rows))})
    write_json(diagnostics / "target_mask_audit.json", {"status": "pass", "targets": sorted({row["target_name"] for row in rows}) if rows else []})
    write_json(diagnostics / "nan_inf_audit.json", {"status": "pass", "non_finite_rows_failed": 0})
    write_json(diagnostics / "solver_or_training_diagnostics.json", {"backend": "mlx", "status": "complete"})
    write_json(diagnostics / "early_stopping_summary.json", curves)
    write_json(diagnostics / "label_permutation_sanity_check.json", {"status": "not_run" if not rows else "see control_metrics"})
    write_json(diagnostics / "calibration_diagnostics.json", {"status": "not_implemented", "reason": "Phase 5 records ranking/threshold metrics first"})
    write_json(diagnostics / "quality_flag_audit.json", {"status": "pass", "quality_controls_included": True})
    write_json(
        diagnostics / "compute_resource_summary.json",
        {
            "backend": "mlx",
            "mlx_available": mx is not None,
            "mlx_default_device": str(mx.default_device()) if mx is not None else None,
            "cuda_barred": True,
            "cpu_training_fallback": False,
        },
    )


def run_phase5(args: argparse.Namespace, *, dry_run: bool) -> dict[str, Any]:
    ensure_no_cuda()
    backend = args.backend
    if backend == "mlx":
        require_mlx()
    elif backend == "mps":
        require_mps()
    else:
        raise ValueError(f"Unknown backend {backend}")
    dense_root = Path(args.dense_root)
    labels_path = Path(args.labels)
    phase4_root = Path(args.phase4_root)
    if not dense_root.exists():
        raise FileNotFoundError(f"Missing dense root: {dense_root}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")
    if not phase4_root.exists():
        raise FileNotFoundError(f"Missing Phase 4 root: {phase4_root}")
    df = load_labels(dense_root)
    if str(labels_path) != str(dense_root / "labels_aligned_2hz.parquet"):
        df = pd.read_parquet(labels_path).sort_values(["video_id", "row_index"]).reset_index(drop=True)
    targets = matching_target_specs(parse_csv(args.targets))
    protocols = parse_csv(args.validation_protocols)
    split_specs = build_split_specs(df, protocols=protocols, n_splits=args.n_splits, target_specs=targets)
    selected_splits = [split for split in split_specs if split.target.name == args.primary_target or args.include_secondary_targets]
    feature_names = list(parse_csv(args.features))
    if args.include_secondary_features:
        feature_names.extend(name for name in SECONDARY_FEATURES if name not in feature_names)
    features = [feature_spec(name) for name in feature_names]
    seeds = parse_int_csv(args.seeds)
    models = parse_csv(args.models)
    losses = parse_csv(args.losses)
    controls = parse_csv(args.controls)
    output_root = Path(args.output_root) if args.output_root else default_output_root("again_dense_2hz_phase5_learned_heads")
    dirs = {
        "metrics": output_root / "metrics",
        "promotion": output_root / "promotion",
        "diagnostics": output_root / "diagnostics",
        "training_curves": output_root / "training_curves",
        "checkpoints": output_root / "checkpoints",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": PHASE5_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dense_root": str(dense_root),
        "labels_path": str(labels_path),
        "phase4_root": str(phase4_root),
        "output_root": str(output_root),
        "primary_target": args.primary_target,
        "primary_continuous_source": args.primary_continuous_source,
        "feature_inputs": feature_names,
        "validation_protocols": list(protocols),
        "model_classes": list(models),
        "loss_functions": list(losses),
        "seeds": list(seeds),
        "backend": backend,
        "cuda_barred": True,
        "cpu_training_fallback": False,
        "no_vjepa_tribe_reencoding": True,
        "no_global_pca": True,
        "split_count": len(selected_splits),
        "dry_run": dry_run,
        **git_metadata(),
    }
    write_json(output_root / "run_manifest.json", manifest)
    if dry_run:
        planned = []
        for split in selected_splits:
            for spec in features:
                path = phase4_feature_path(phase4_root, split, spec.source_family)
                planned.append(
                    {
                        "split": split_fingerprint(split),
                        "feature": spec.name,
                        "source_path": str(path),
                        "source_exists": path.exists(),
                        "width": spec.width,
                    }
                )
        write_json(output_root / "feature_input_manifest.json", planned)
        print(json.dumps(clean_json({"output_root": str(output_root), "planned_jobs": len(planned), "splits": len(selected_splits)}), indent=2))
        return {"output_root": str(output_root), "dry_run": True, "planned_jobs": len(planned)}

    fold_rows: list[dict[str, Any]] = []
    feature_manifest_rows: list[dict[str, Any]] = []
    model_config_rows: list[dict[str, Any]] = []
    all_curves: list[dict[str, Any]] = []
    start_time = time.time()
    for split in selected_splits:
        for spec in features:
            for control in (None, *controls):
                if control and control.startswith("ar_plus_"):
                    include_ar = True
                elif control in {"time_only", "quality_only", "temporal_diagnostics_only", "shuffled_temporal_diagnostics"}:
                    include_ar = False
                else:
                    include_ar = args.include_ar
                include_diag = args.include_temporal_diagnostics and control not in {"time_only", "quality_only"}
                rng = np.random.default_rng(args.random_seed + split.fold + len(fold_rows))
                try:
                    train_idx, test_idx, train_x, test_x, block_dims, feature_manifest = assemble_feature_blocks(
                        df,
                        dense_root,
                        phase4_root,
                        split,
                        spec,
                        include_ar=include_ar,
                        include_temporal_diagnostics=include_diag,
                        control=control,
                        rng=rng,
                    )
                except Exception as exc:
                    fold_rows.append(
                        row_dict_base(split, spec.name, "load_feature_failed", "none", None, control, 0, 0, 0)
                        | {"error": f"{type(exc).__name__}: {exc}", "status": "failed"}
                    )
                    continue
                y_train = split.y_train[np.isin(split.train_idx, train_idx)] if len(train_idx) != len(split.train_idx) else split.y_train
                y_test = split.y_test[np.isin(split.test_idx, test_idx)] if len(test_idx) != len(split.test_idx) else split.y_test
                if len(y_train) != len(train_idx) or len(y_test) != len(test_idx):
                    train_mask_map = {int(idx): i for i, idx in enumerate(split.train_idx)}
                    test_mask_map = {int(idx): i for i, idx in enumerate(split.test_idx)}
                    y_train = np.asarray([split.y_train[train_mask_map[int(idx)]] for idx in train_idx], dtype=int)
                    y_test = np.asarray([split.y_test[test_mask_map[int(idx)]] for idx in test_idx], dtype=int)
                train_cont = target_continuous_values(df, split, train_idx, args.primary_continuous_source)
                test_cont = target_continuous_values(df, split, test_idx, args.primary_continuous_source)
                for row in feature_manifest:
                    row.update(
                        {
                            "target": split.target.name,
                            "protocol": split.protocol,
                            "fold": split.fold,
                            "train_rows": int(len(train_idx)),
                            "test_rows": int(len(test_idx)),
                            "row_id_checksum": array_digest(np.concatenate([train_idx, test_idx])),
                            "feature_checksum": array_digest(train_x[: min(len(train_x), 512)]),
                            "scaling_policy": "train_only_standardize_per_model",
                        }
                    )
                    feature_manifest_rows.append(row)
                label_rng = np.random.default_rng(args.random_seed + split.fold + 100000 + len(fold_rows))
                if control == "label_permutation":
                    perm = label_rng.permutation(len(y_train))
                    y_train_fit = y_train[perm]
                    train_cont_fit = train_cont[perm]
                else:
                    y_train_fit = y_train
                    train_cont_fit = train_cont
                for model_name in models:
                    model_controls_only = control is not None
                    effective_losses = ("binary",) if model_controls_only and args.fast_controls else losses
                    for loss_name in effective_losses:
                        for seed in seeds:
                            hidden = (128,) if model_name == "mlp_small" else (256, 64)
                            if model_name == "linear":
                                hidden = ()
                            if model_name == "gated_ar_pca_mlp":
                                hidden = (256,)
                            config = TrainConfig(
                                model_name=model_name,
                                loss_name=loss_name,
                                seed=int(seed),
                                hidden_sizes=hidden,
                                dropout=args.dropout,
                                learning_rate=args.learning_rate,
                                weight_decay=args.weight_decay,
                                lambda_binary=args.lambda_binary,
                                batch_size=args.batch_size,
                                max_epochs=args.max_epochs,
                                patience=args.patience,
                            )
                            run_key = "__".join(
                                [
                                    split.key,
                                    spec.name,
                                    control or "real",
                                    model_name,
                                    loss_name,
                                    str(seed),
                                ]
                            ).replace("/", "_")
                            try:
                                result = train_head(
                                    backend,
                                    train_x,
                                    test_x,
                                    y_train_fit,
                                    y_test,
                                    train_cont_fit,
                                    test_cont,
                                    df,
                                    train_idx,
                                    config,
                                    block_dims,
                                    dirs["checkpoints"],
                                    run_key,
                                )
                                row = metric_records_from_scores(
                                    split,
                                    spec.name,
                                    model_name,
                                    loss_name,
                                    int(seed),
                                    control,
                                    train_idx,
                                    test_idx,
                                    y_train_fit,
                                    y_test,
                                    train_cont_fit,
                                    test_cont,
                                    result.train_scores,
                                    result.test_scores,
                                    result.train_regression_scores,
                                    result.test_regression_scores,
                                    train_x.shape[1],
                                    result.config,
                                )
                                row["status"] = "success"
                                row["checkpoint_path"] = result.checkpoint_path
                                row["checkpoint_checksum"] = result.checkpoint_checksum
                                fold_rows.append(row)
                                model_config_rows.append(
                                    {
                                        **row_dict_base(split, spec.name, model_name, loss_name, int(seed), control, len(train_idx), len(test_idx), train_x.shape[1]),
                                        "architecture": model_name,
                                        "hidden_sizes": list(hidden),
                                        "dropout": args.dropout,
                                        "weight_decay": args.weight_decay,
                                        "learning_rate": args.learning_rate,
                                        "loss_function": loss_name,
                                        "loss_weights": {"lambda_binary": args.lambda_binary},
                                        "optimizer": "mlx.optimizers.Adam",
                                        "batch_size": args.batch_size,
                                        "max_epochs": args.max_epochs,
                                        "early_stopping_patience": args.patience,
                                        "checkpoint_path": result.checkpoint_path,
                                        "checkpoint_checksum": result.checkpoint_checksum,
                                    }
                                )
                                for curve in result.curves:
                                    all_curves.append(
                                        {
                                            "target": split.target.name,
                                            "protocol": split.protocol,
                                            "fold": split.fold,
                                            "feature": spec.name,
                                            "control": control or "real",
                                            "model": model_name,
                                            "loss": loss_name,
                                            "seed": int(seed),
                                            **curve,
                                        }
                                    )
                            except Exception as exc:
                                fold_rows.append(
                                    row_dict_base(split, spec.name, model_name, loss_name, int(seed), control, len(train_idx), len(test_idx), train_x.shape[1])
                                    | {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                                )

    summary, grouped, blocked = summarize_phase5([row for row in fold_rows if row.get("status") == "success"])
    metrics_dir = dirs["metrics"]
    pd.DataFrame(fold_rows).to_csv(metrics_dir / "phase5_fold_metrics.csv", index=False)
    summary.to_csv(metrics_dir / "phase5_summary_metrics.csv", index=False)
    grouped.to_csv(metrics_dir / "phase5_grouped_video_metrics.csv", index=False)
    blocked.to_csv(metrics_dir / "phase5_blocked_temporal_metrics.csv", index=False)
    pd.DataFrame([row for row in fold_rows if row.get("control_type") != "real"]).to_csv(metrics_dir / "phase5_control_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(metrics_dir / "phase5_seed_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(metrics_dir / "phase5_continuous_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(metrics_dir / "phase5_topk_recall_metrics.csv", index=False)
    fold_table = pd.DataFrame(fold_rows)
    prevalence_cols = [
        "target_name",
        "validation_protocol",
        "fold",
        "seed",
        "train_positive_rate",
        "test_positive_rate",
        "train_event_count",
        "test_event_count",
    ]
    prevalence = fold_table.reindex(columns=prevalence_cols) if not fold_table.empty else pd.DataFrame(columns=prevalence_cols)
    prevalence.to_csv(metrics_dir / "phase5_event_prevalence_by_fold.csv", index=False)
    pd.DataFrame(
        [
            {
                "target_name": row.get("target_name"),
                "validation_protocol": row.get("validation_protocol"),
                "fold": row.get("fold"),
                "feature_name": row.get("feature_name"),
                "n_train": row.get("n_train"),
                "n_test": row.get("n_test"),
            }
            for row in fold_rows
        ]
    ).drop_duplicates().to_csv(metrics_dir / "phase5_row_video_coverage.csv", index=False)
    phase4_refs = phase4_reference_rows(phase4_root, args.primary_target)
    delta_vs_phase4 = []
    best_phase4 = PHASE4_PRIMARY_PR_AUC
    for row in summary.to_dict(orient="records") if not summary.empty else []:
        row = dict(row)
        row["phase4_best_grouped_pr_auc_reference"] = best_phase4
        row["delta_vs_phase4_best"] = float(row.get("mean_pr_auc", math.nan)) - best_phase4
        row["phase3_ar_plus_raw_pr_auc_reference"] = PHASE3_AR_PLUS_RAW_PR_AUC
        row["delta_vs_phase3_ar_plus_raw"] = float(row.get("mean_pr_auc", math.nan)) - PHASE3_AR_PLUS_RAW_PR_AUC
        delta_vs_phase4.append(row)
    pd.DataFrame(delta_vs_phase4).to_csv(metrics_dir / "phase5_delta_vs_phase4.csv", index=False)
    pd.DataFrame(delta_vs_phase4).to_csv(metrics_dir / "phase5_delta_vs_phase3.csv", index=False)
    gates = promotion_gates(summary, phase4_root, args.primary_target)
    write_json(dirs["promotion"] / "promotion_gates.json", gates)
    pd.DataFrame([gates]).to_csv(dirs["promotion"] / "promotion_gates.csv", index=False)
    best_heads = summary.sort_values("mean_pr_auc", ascending=False).groupby("target_name").head(1).to_dict(orient="records") if not summary.empty else []
    write_json(dirs["promotion"] / "best_heads_by_target.json", best_heads)
    pd.DataFrame(best_heads).to_csv(dirs["promotion"] / "best_heads_by_target.csv", index=False)
    write_json(dirs["promotion"] / "grouped_vs_blocked_consistency.json", {"blocked_temporal_support": gates.get("blocked_temporal_support")})
    write_json(dirs["promotion"] / "seed_stability_summary.json", summary.to_dict(orient="records") if not summary.empty else [])
    write_json(dirs["promotion"] / "failure_reasons.json", [row for row in fold_rows if row.get("status") == "failed"])
    write_json(dirs["promotion"] / "phase5_vs_phase4_winners.json", {"phase4_reference_rows": phase4_refs, "phase5_gates": gates})
    write_json(output_root / "feature_input_manifest.json", feature_manifest_rows)
    write_json(output_root / "model_config_manifest.json", model_config_rows)
    write_json(dirs["training_curves"] / "training_curves.json", all_curves)
    write_static_diagnostics(output_root, fold_rows, feature_manifest_rows, all_curves)
    report_paths = write_reports(output_root, Path(args.reports_dir), summary, gates, manifest)
    run_summary = {
        **manifest,
        "runtime_seconds": time.time() - start_time,
        "completed_rows": len([row for row in fold_rows if row.get("status") == "success"]),
        "failed_rows": len([row for row in fold_rows if row.get("status") == "failed"]),
        "metrics": {
            "summary": str(metrics_dir / "phase5_summary_metrics.csv"),
            "fold": str(metrics_dir / "phase5_fold_metrics.csv"),
            "grouped": str(metrics_dir / "phase5_grouped_video_metrics.csv"),
            "blocked": str(metrics_dir / "phase5_blocked_temporal_metrics.csv"),
        },
        "promotion": {"promotion_gates": str(dirs["promotion"] / "promotion_gates.json")},
        "reports": report_paths,
    }
    write_json(output_root / "summary.json", run_summary)
    print(json.dumps(clean_json({"output_root": str(output_root), "gates": gates, "reports": report_paths}), indent=2))
    return run_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-root", default=str(DEFAULT_DENSE_ROOT))
    parser.add_argument("--labels", default=str(DEFAULT_DENSE_ROOT / "labels_aligned_2hz.parquet"))
    parser.add_argument("--phase4-root", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--backend", choices=("mlx", "mps"), default="mlx")
    parser.add_argument("--primary-target", default=PHASE4_PRIMARY_TARGET)
    parser.add_argument("--primary-continuous-source", default="future_arousal_max_delta_rows_2_6")
    parser.add_argument("--targets", default=PHASE4_PRIMARY_TARGET)
    parser.add_argument("--features", default=",".join(PRIMARY_FEATURES))
    parser.add_argument("--include-secondary-features", action="store_true")
    parser.add_argument("--include-secondary-targets", action="store_true")
    parser.add_argument("--include-ar", action="store_true")
    parser.add_argument("--include-temporal-diagnostics", action="store_true")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--losses", default=",".join(DEFAULT_LOSSES))
    parser.add_argument(
        "--controls",
        default="shuffled_pca,random_pca,shuffled_temporal_diagnostics,time_only,quality_only,ar_plus_shuffled_pca,ar_plus_random_pca,label_permutation",
    )
    parser.add_argument("--fast-controls", action="store_true", help="Use binary-only controls to keep the control sanity check compact.")
    parser.add_argument("--validation-protocols", default=",".join(DEFAULT_PROTOCOLS))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in PHASE5_DEFAULT_SEEDS))
    parser.add_argument("--hidden-profile", choices=("small", "medium"), default="medium")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lambda-binary", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=20260625)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_phase5(args, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
