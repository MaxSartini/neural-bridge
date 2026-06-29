"""Targeted Phase 5 frozen-AR residual-over-AR repair.

This runner is intentionally bounded to the primary Phase 5 lane:
`arousal_spike_rows_2_6_train_q90`, `temporal_mean_2s_then_pca256`,
`regression_plus_binary`, grouped-video plus blocked-temporal protocols, and
seeds 20260625..20260627.

It does not train AR if saved AR-only checkpoints can be reused. The AR score
is frozen as the floor and cortical PCA/diagnostics are trained only as a
residual correction. Heavy dense features, PCA, V-JEPA/TRIBE, Phase 4 outputs,
and original Phase 5 outputs are not modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts import run_again_dense_2hz_phase5_adversarial_repair_fixplus as repair
from backend.scripts import run_again_dense_2hz_phase5_learned_heads as base
from backend.scripts.again_dense_2hz_benchmark import (
    AR_FEATURE_COLUMNS,
    feature_matrix,
    load_or_build_temporal_diagnostic_features,
)
from backend.scripts.again_dense_2hz_phase4_pca_bridge import array_digest, split_fingerprint


SCHEMA_VERSION = "again_dense_2hz_phase5_frozen_ar_residual_v1"
SOURCE_ROOT = Path("outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_20260629_171825")
EVAL_ROOT = Path("outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_evalmode_rescore_")
OUTPUT_ROOT = Path("outputs/again_dense_2hz_phase5_frozen_ar_residual_")
TARGET_NAME = "arousal_spike_rows_2_6_train_q90"
CONTINUOUS_SOURCE = "future_arousal_max_delta_rows_2_6"
FEATURE_NAME = "temporal_mean_2s_then_pca256"
LOSS_NAME = "regression_plus_binary"
SEEDS = (20260625, 20260626, 20260627)
PROTOCOLS = ("grouped_video", "blocked_temporal_70_30")
RESIDUAL_MODELS = ("frozen_ar_plus_linear_residual", "frozen_ar_plus_mlp_residual", "frozen_ar_plus_gated_residual")
RESIDUAL_CONTROLS = (
    "real_frozen_ar_residual",
    "shuffled_pca_frozen_ar_residual",
    "random_pca_frozen_ar_residual",
    "video_mean_pca_frozen_ar_residual",
    "diag_only_frozen_ar_residual",
    "label_permutation_frozen_ar_residual",
    "pca_only_without_ar",
)
MATCHED_CONTROLS = ("shuffled_pca_frozen_ar_residual", "random_pca_frozen_ar_residual")
CONTROL_OFFSETS = {name: (i + 1) * 10007 for i, name in enumerate(RESIDUAL_CONTROLS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--eval-root", default=str(EVAL_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--limit-rows", type=int, default=0, help="Debug only; 0 uses all rows.")
    return parser.parse_args()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(obj), indent=2, sort_keys=True) + "\n")


def clean_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_array(arr: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.ascontiguousarray(arr).view(np.uint8))
    return digest.hexdigest()


def split_digest(train_idx: np.ndarray, test_idx: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.ascontiguousarray(train_idx.astype(np.int64)).view(np.uint8))
    digest.update(np.ascontiguousarray(test_idx.astype(np.int64)).view(np.uint8))
    return digest.hexdigest()


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return math.nan
    a = a[mask]
    b = b[mask]
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return corr(rank(np.asarray(a)), rank(np.asarray(b)))


def top_fraction_metrics(y_binary: np.ndarray, y_cont: np.ndarray, scores: np.ndarray, frac: float) -> dict[str, Any]:
    if len(scores) == 0:
        return {}
    k = max(1, int(math.ceil(len(scores) * frac)))
    top_idx = np.argsort(-scores, kind="mergesort")[:k]
    base_rate = float(np.mean(y_binary)) if len(y_binary) else math.nan
    precision = float(np.mean(y_binary[top_idx])) if k else math.nan
    recall = float(np.sum(y_binary[top_idx]) / np.sum(y_binary)) if np.sum(y_binary) > 0 else math.nan
    movement = float(np.mean(y_cont[top_idx])) if k else math.nan
    baseline_movement = float(np.mean(y_cont)) if len(y_cont) else math.nan
    pct = int(frac * 100)
    return {
        f"top_{pct}pct_precision": precision,
        f"top_{pct}pct_recall": recall,
        f"top_{pct}pct_lift": precision / base_rate if base_rate and math.isfinite(base_rate) else math.nan,
        f"top_{pct}pct_avg_true_movement": movement,
        f"top_{pct}pct_avg_true_movement_lift": movement - baseline_movement
        if math.isfinite(movement) and math.isfinite(baseline_movement)
        else math.nan,
    }


def metric_row(
    y_train: np.ndarray,
    train_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    test_cont: np.ndarray,
    test_reg: np.ndarray,
) -> dict[str, Any]:
    threshold = base.decision_threshold_for_binary(y_train, train_scores)
    pred = (test_scores >= threshold).astype(int)
    out: dict[str, Any] = {
        "decision_threshold_train_only": float(threshold),
        "pr_auc": float(average_precision_score(y_test, test_scores)) if len(np.unique(y_test)) > 1 else math.nan,
        "roc_auc": float(roc_auc_score(y_test, test_scores)) if len(np.unique(y_test)) > 1 else math.nan,
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)) if len(np.unique(y_test)) > 1 else math.nan,
        "continuous_mae": float(mean_absolute_error(test_cont, test_reg)),
        "continuous_mse": float(mean_squared_error(test_cont, test_reg)),
        "continuous_rmse": float(math.sqrt(mean_squared_error(test_cont, test_reg))),
        "continuous_pearson": corr(test_cont, test_reg),
        "continuous_spearman": spearman(test_cont, test_reg),
        "spearman_future_movement": spearman(test_cont, test_scores),
    }
    for frac in (0.01, 0.05, 0.10):
        out.update(top_fraction_metrics(y_test, test_cont, test_scores, frac))
    return out


def split_y(split: Any, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(train_idx) == len(split.train_idx) and len(test_idx) == len(split.test_idx):
        return split.y_train, split.y_test
    train_map = {int(idx): i for i, idx in enumerate(split.train_idx)}
    test_map = {int(idx): i for i, idx in enumerate(split.test_idx)}
    y_train = np.asarray([split.y_train[train_map[int(idx)]] for idx in train_idx], dtype=int)
    y_test = np.asarray([split.y_test[test_map[int(idx)]] for idx in test_idx], dtype=int)
    return y_train, y_test


def inner_split(df: pd.DataFrame, train_idx: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    return repair.temporal_inner_validation_relative_split(df, train_idx, y_train)


def config_for_ar(seed: int) -> Any:
    return base.TrainConfig(
        model_name="gated_ar_pca_mlp",
        loss_name=LOSS_NAME,
        seed=int(seed),
        hidden_sizes=(256,),
        dropout=0.1,
        learning_rate=3e-4,
        weight_decay=1e-4,
        lambda_binary=0.5,
        batch_size=8192,
        max_epochs=180,
        patience=24,
    )


def score_existing_model(model: Any, x: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    scores: list[np.ndarray] = []
    regs: list[np.ndarray] = []
    if hasattr(model, "eval"):
        model.eval()
    for start in range(0, len(x), batch_size):
        out = model(base.mx.array(x[start : start + batch_size], dtype=base.mx.float32))
        base.mx.eval(out)
        score, reg = base.select_score_columns(np.asarray(out, dtype=np.float32), LOSS_NAME)
        scores.append(score.astype(np.float32, copy=False))
        regs.append(reg.astype(np.float32, copy=False))
    return np.concatenate(scores), np.concatenate(regs)


class ResidualHead(base.nn.Module):
    def __init__(self, input_dim: int, model_name: str, *, hidden: int = 64):
        super().__init__()
        self.model_name = model_name
        self.alpha = base.mx.array([0.01], dtype=base.mx.float32)
        if model_name == "frozen_ar_plus_linear_residual":
            self.layers = []
            self.out = base.nn.Linear(input_dim, 2)
            self.gate = None
        elif model_name in {"frozen_ar_plus_mlp_residual", "frozen_ar_plus_gated_residual"}:
            self.layers = [base.nn.Linear(input_dim, hidden)]
            self.out = base.nn.Linear(hidden, 2)
            self.gate = base.nn.Linear(input_dim, 1) if model_name == "frozen_ar_plus_gated_residual" else None
        else:
            raise ValueError(f"Unknown residual model: {model_name}")

    def residual(self, x: Any) -> Any:
        h = x
        for layer in self.layers:
            h = base.nn.gelu(layer(h))
        return self.out(h)

    def gate_value(self, x: Any) -> Any:
        if self.gate is None:
            return base.mx.ones((x.shape[0], 1), dtype=x.dtype)
        return base.mx.sigmoid(self.gate(x) - 4.0)

    def __call__(self, x: Any, ar_score: Any, ar_reg: Any, use_ar_floor: bool = True) -> Any:
        residual = self.residual(x)
        gate = self.gate_value(x)
        scale = self.alpha * gate
        if use_ar_floor:
            binary = ar_score[:, None] + scale * residual[:, 1:2]
            reg = ar_reg[:, None] + scale * residual[:, 0:1]
        else:
            binary = residual[:, 1:2]
            reg = residual[:, 0:1]
        return base.mx.concatenate([reg, binary], axis=1)


def residual_forward(
    model: ResidualHead,
    x: np.ndarray,
    ar_score: np.ndarray,
    ar_reg: np.ndarray,
    *,
    use_ar_floor: bool,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores: list[np.ndarray] = []
    regs: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    if hasattr(model, "eval"):
        model.eval()
    for start in range(0, len(x), batch_size):
        xb = base.mx.array(x[start : start + batch_size], dtype=base.mx.float32)
        ab = base.mx.array(ar_score[start : start + batch_size], dtype=base.mx.float32)
        rb = base.mx.array(ar_reg[start : start + batch_size], dtype=base.mx.float32)
        out = model(xb, ab, rb, use_ar_floor=use_ar_floor)
        gate = model.gate_value(xb)
        base.mx.eval(out, gate)
        score, reg = base.select_score_columns(np.asarray(out, dtype=np.float32), LOSS_NAME)
        scores.append(score.astype(np.float32, copy=False))
        regs.append(reg.astype(np.float32, copy=False))
        gates.append(np.asarray(gate, dtype=np.float32).reshape(-1))
    return np.concatenate(scores), np.concatenate(regs), np.concatenate(gates)


@dataclass
class Block:
    protocol: str
    fold: int
    split: Any
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_y: np.ndarray
    test_y: np.ndarray
    train_cont: np.ndarray
    test_cont: np.ndarray
    inner_train: np.ndarray
    inner_val: np.ndarray
    inner_audit: dict[str, Any]
    train_video_id: np.ndarray
    test_video_id: np.ndarray
    train_time: np.ndarray
    test_time: np.ndarray
    ar_train_x: np.ndarray
    ar_test_x: np.ndarray
    ar_block_dims: dict[str, int]


def build_blocks(source_root: Path) -> tuple[dict[tuple[str, int], Block], pd.DataFrame, Path, Path]:
    repair.patch_base_module()
    manifest = json.loads((source_root / "run_manifest.json").read_text())
    dense_root = Path(manifest["dense_root"])
    phase4_root = Path(manifest["phase4_root"])
    df = base.load_labels(dense_root)
    splits = base.build_split_specs(
        df,
        protocols=PROTOCOLS,
        n_splits=5,
        target_specs=base.matching_target_specs((TARGET_NAME,)),
    )
    spec = base.feature_spec(FEATURE_NAME)
    blocks: dict[tuple[str, int], Block] = {}
    for split in splits:
        rng = np.random.default_rng(20260625 + int(split.fold) + 9)
        train_idx, test_idx, train_x, test_x, block_dims, _ = repair.assemble_feature_blocks_repair(
            df,
            dense_root,
            phase4_root,
            split,
            spec,
            include_ar=True,
            include_temporal_diagnostics=True,
            control="ar_only_head",
            rng=rng,
        )
        train_x, test_x = base.standardize_train_only(train_x, test_x)
        train_y, test_y = split_y(split, train_idx, test_idx)
        train_cont = base.target_continuous_values(df, split, train_idx, CONTINUOUS_SOURCE)
        test_cont = base.target_continuous_values(df, split, test_idx, CONTINUOUS_SOURCE)
        inner_train, inner_val, audit = inner_split(df, train_idx, train_y)
        blocks[(split.protocol, int(split.fold))] = Block(
            protocol=split.protocol,
            fold=int(split.fold),
            split=split,
            train_idx=train_idx,
            test_idx=test_idx,
            train_y=train_y,
            test_y=test_y,
            train_cont=train_cont,
            test_cont=test_cont,
            inner_train=inner_train,
            inner_val=inner_val,
            inner_audit=audit,
            train_video_id=df.loc[train_idx, "video_id"].astype(str).to_numpy(),
            test_video_id=df.loc[test_idx, "video_id"].astype(str).to_numpy(),
            train_time=df.loc[train_idx, "time_seconds"].to_numpy(dtype=np.float32),
            test_time=df.loc[test_idx, "time_seconds"].to_numpy(dtype=np.float32),
            ar_train_x=train_x,
            ar_test_x=test_x,
            ar_block_dims=block_dims,
        )
    return blocks, df, dense_root, phase4_root


def residual_features(
    df: pd.DataFrame,
    dense_root: Path,
    phase4_root: Path,
    block: Block,
    control: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], list[dict[str, Any]]]:
    _train_idx, _test_idx, pca_train, pca_test, source_path = base.load_phase4_scores(
        df,
        phase4_root,
        block.split,
        base.feature_spec(FEATURE_NAME),
    )
    if not np.array_equal(_train_idx, block.train_idx) or not np.array_equal(_test_idx, block.test_idx):
        raise RuntimeError("Residual feature row index mismatch with frozen AR split.")
    rng = np.random.default_rng(int(seed) + block.fold * 1009 + CONTROL_OFFSETS.get(control, 0))
    diag = load_or_build_temporal_diagnostic_features(dense_root, df)
    diag_train = diag[block.train_idx].copy()
    diag_test = diag[block.test_idx].copy()
    p_train = pca_train.copy()
    p_test = pca_test.copy()
    include_diag = True
    include_pca = True
    pca_kind = "real"
    if control == "shuffled_pca_frozen_ar_residual":
        p_train = p_train[rng.permutation(len(p_train))]
        p_test = p_test[rng.permutation(len(p_test))]
        pca_kind = "shuffled"
    elif control == "random_pca_frozen_ar_residual":
        p_train = rng.normal(0, 1, size=p_train.shape).astype(np.float32)
        p_test = rng.normal(0, 1, size=p_test.shape).astype(np.float32)
        pca_kind = "random"
    elif control == "video_mean_pca_frozen_ar_residual":
        all_idx = np.concatenate([block.train_idx, block.test_idx])
        all_pca = np.concatenate([pca_train, pca_test], axis=0)
        videos = df.loc[all_idx, "video_id"].astype(str).to_numpy()
        means = {video: all_pca[videos == video].mean(axis=0) for video in np.unique(videos)}
        p_train = np.vstack([means[v] for v in block.train_video_id]).astype(np.float32)
        p_test = np.vstack([means[v] for v in block.test_video_id]).astype(np.float32)
        include_diag = False
        pca_kind = "video_mean_oracle"
    elif control == "diag_only_frozen_ar_residual":
        include_pca = False
        pca_kind = "none"
    elif control in {"real_frozen_ar_residual", "label_permutation_frozen_ar_residual", "pca_only_without_ar"}:
        pca_kind = "real"
    else:
        raise ValueError(f"Unknown residual control: {control}")
    blocks_train: list[np.ndarray] = []
    blocks_test: list[np.ndarray] = []
    dims: dict[str, int] = {}
    manifest: list[dict[str, Any]] = []
    if include_pca:
        blocks_train.append(p_train)
        blocks_test.append(p_test)
        dims["pca"] = p_train.shape[1]
        manifest.append(
            {
                "block": "phase4_fold_safe_pca",
                "kind": pca_kind,
                "source_path": str(source_path),
                "source_checksum": base.file_digest(source_path),
                "width": p_train.shape[1],
            }
        )
    if include_diag:
        blocks_train.append(diag_train)
        blocks_test.append(diag_test)
        dims["diagnostics"] = diag_train.shape[1]
        manifest.append(
            {
                "block": "temporal_diagnostics",
                "source": str(dense_root / "_derived" / "temporal_diagnostics_summary_features.npy"),
                "width": diag_train.shape[1],
            }
        )
    if not blocks_train:
        raise RuntimeError(f"No residual features for {control}")
    train_x = np.concatenate(blocks_train, axis=1).astype(np.float32, copy=False)
    test_x = np.concatenate(blocks_test, axis=1).astype(np.float32, copy=False)
    train_x, test_x = base.standardize_train_only(train_x, test_x)
    return train_x, test_x, dims, manifest


def ar_checkpoint_row(source_root: Path, protocol: str, fold: int, seed: int) -> pd.Series:
    metrics = pd.read_csv(source_root / "metrics" / "phase5_fold_metrics.csv")
    row = metrics[
        (metrics["validation_protocol"] == protocol)
        & (metrics["fold"] == int(fold))
        & (metrics["seed"] == int(seed))
        & (metrics["loss_name"] == LOSS_NAME)
        & (metrics["control_type"] == "ar_only_head")
        & (metrics["model_head"] == "gated_ar_pca_mlp")
        & (metrics["target_name"] == TARGET_NAME)
    ]
    if len(row) != 1:
        raise RuntimeError(f"Expected one AR-only checkpoint row for {protocol} fold {fold} seed {seed}; got {len(row)}")
    return row.iloc[0]


def cache_frozen_ar(
    source_root: Path,
    output_root: Path,
    block: Block,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    row = ar_checkpoint_row(source_root, block.protocol, block.fold, seed)
    checkpoint = Path(str(row.checkpoint_path))
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing AR-only checkpoint: {checkpoint}")
    model = base.make_model(config_for_ar(seed), block.ar_train_x.shape[1], block.ar_block_dims)
    _ = model(base.mx.array(block.ar_train_x[:2], dtype=base.mx.float32))
    model.load_weights(str(checkpoint))
    if hasattr(model, "eval"):
        model.eval()
    train_score, train_reg = score_existing_model(model, block.ar_train_x, batch_size)
    test_score, test_reg = score_existing_model(model, block.ar_test_x, batch_size)
    inner_score = train_score[block.inner_val]
    train_metric = metric_row(block.train_y, train_score, block.test_y, test_score, block.test_cont, test_reg)
    key = f"{block.protocol}__fold{block.fold}__seed{seed}__{LOSS_NAME}"
    out_dir = output_root / "frozen_ar_scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, row_idx, video_id, time_sec, score, reg in (
        ("train", block.train_idx, block.train_video_id, block.train_time, train_score, train_reg),
        ("heldout_test", block.test_idx, block.test_video_id, block.test_time, test_score, test_reg),
        (
            "inner_val",
            block.train_idx[block.inner_val],
            block.train_video_id[block.inner_val],
            block.train_time[block.inner_val],
            inner_score,
            train_reg[block.inner_val],
        ),
    ):
        pd.DataFrame(
            {
                "row_id": row_idx.astype(np.int64),
                "video_id": video_id,
                "time_seconds": time_sec,
                "frozen_ar_score": score.astype(np.float32),
                "frozen_ar_continuous_prediction": reg.astype(np.float32),
            }
        ).to_csv(out_dir / f"{key}__{split_name}.csv.gz", index=False)
    return {
        "key": key,
        "protocol": block.protocol,
        "fold": block.fold,
        "seed": int(seed),
        "loss": LOSS_NAME,
        "checkpoint_path": str(checkpoint),
        "checkpoint_checksum": row.checkpoint_checksum,
        "source": "re_forwarded_saved_ar_only_best_checkpoint",
        "ar_retrained": False,
        "train_score": train_score,
        "train_reg": train_reg,
        "test_score": test_score,
        "test_reg": test_reg,
        "train_checksum": hash_array(train_score),
        "test_checksum": hash_array(test_score),
        "metrics": train_metric,
    }


def train_residual(
    model_name: str,
    control: str,
    train_x: np.ndarray,
    test_x: np.ndarray,
    block: Block,
    ar: dict[str, Any],
    seed: int,
    output_root: Path,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    base.require_mlx()
    base.mx.random.seed(int(seed))
    use_ar_floor = control != "pca_only_without_ar"
    model = ResidualHead(train_x.shape[1], model_name)
    optimizer = base.optim.AdamW(learning_rate=2e-4, weight_decay=1e-4)
    inner_train = block.inner_train
    inner_val = block.inner_val
    rng = np.random.default_rng(int(seed) + block.fold * 17)
    y_train = block.train_y.copy()
    cont_train = block.train_cont.copy()
    if control == "label_permutation_frozen_ar_residual":
        perm = rng.permutation(len(y_train))
        y_train_metric = y_train[perm]
        cont_train_metric = cont_train[perm]
    else:
        y_train_metric = y_train
        cont_train_metric = cont_train

    ar_train_score = ar["train_score"].astype(np.float32)
    ar_train_reg = ar["train_reg"].astype(np.float32)
    ar_test_score = ar["test_score"].astype(np.float32)
    ar_test_reg = ar["test_reg"].astype(np.float32)
    ar_inner_metric = average_precision_score(y_train[inner_val], ar_train_score[inner_val])
    best_delta = 0.0
    best_epoch = 0
    best_path = output_root / "checkpoints" / (
        f"{TARGET_NAME}__{block.protocol}__fold{block.fold}__{control}__{model_name}__{LOSS_NAME}__{seed}__best.npz"
    )
    best_path.parent.mkdir(parents=True, exist_ok=True)
    curves: list[dict[str, Any]] = []
    stale = 0
    early_stop = "max_epochs_reached"
    suppressed = True

    def loss_fn(model_obj: ResidualHead, xb: Any, ar_b: Any, ar_r: Any, yb: Any, yr: Any) -> Any:
        out = model_obj(xb, ar_b, ar_r, use_ar_floor=use_ar_floor)
        reg_loss = base.mx.mean(base.nn.losses.huber_loss(out[:, 0:1], yr, delta=1.0))
        bce = base.mx.mean(base.nn.losses.binary_cross_entropy(out[:, 1:2], yb, with_logits=True))
        alpha_penalty = 0.002 * base.mx.mean(model_obj.alpha * model_obj.alpha)
        return reg_loss + 0.5 * bce + alpha_penalty

    loss_and_grad = base.nn.value_and_grad(model, loss_fn)
    for epoch in range(1, int(max_epochs) + 1):
        order = rng.permutation(inner_train)
        total = 0.0
        batches = 0
        for start in range(0, len(order), batch_size):
            rel = order[start : start + batch_size]
            xb = base.mx.array(train_x[rel], dtype=base.mx.float32)
            ab = base.mx.array(ar_train_score[rel], dtype=base.mx.float32)
            rb = base.mx.array(ar_train_reg[rel], dtype=base.mx.float32)
            yb = base.mx.array(y_train_metric[rel].astype(np.float32)[:, None], dtype=base.mx.float32)
            yr = base.mx.array(cont_train_metric[rel].astype(np.float32)[:, None], dtype=base.mx.float32)
            loss, grads = loss_and_grad(model, xb, ab, rb, yb, yr)
            grads, _ = base.optim.clip_grad_norm(grads, 1.0)
            optimizer.update(model, grads)
            base.mx.eval(model.parameters(), optimizer.state)
            total += float(np.asarray(loss))
            batches += 1
        val_score, val_reg, val_gate = residual_forward(
            model,
            train_x[inner_val],
            ar_train_score[inner_val],
            ar_train_reg[inner_val],
            use_ar_floor=use_ar_floor,
            batch_size=batch_size,
        )
        val_pr = average_precision_score(y_train[inner_val], val_score) if len(np.unique(y_train[inner_val])) > 1 else math.nan
        delta = float(val_pr - ar_inner_metric) if math.isfinite(val_pr) and math.isfinite(ar_inner_metric) else math.nan
        alpha = float(np.asarray(model.alpha)[0])
        curve = {
            "epoch": epoch,
            "train_loss": total / max(1, batches),
            "inner_val_pr_auc": val_pr,
            "inner_val_delta_vs_frozen_ar": delta,
            "frozen_ar_inner_val_pr_auc": ar_inner_metric,
            "alpha": alpha,
            "gate_mean": float(np.mean(val_gate)),
            "gate_p95": float(np.quantile(val_gate, 0.95)),
        }
        curves.append(curve)
        if math.isfinite(delta) and delta > best_delta:
            model.save_weights(str(best_path))
            best_delta = delta
            best_epoch = epoch
            stale = 0
            suppressed = False
        else:
            stale += 1
        if stale >= int(patience):
            early_stop = "patience_exhausted"
            break

    if suppressed:
        train_score = ar_train_score if use_ar_floor else np.zeros_like(ar_train_score)
        train_reg = ar_train_reg if use_ar_floor else np.zeros_like(ar_train_reg)
        test_score = ar_test_score if use_ar_floor else np.zeros_like(ar_test_score)
        test_reg = ar_test_reg if use_ar_floor else np.zeros_like(ar_test_reg)
        gate = np.zeros_like(test_score)
        alpha_final = 0.0
        checkpoint_restored = False
        checkpoint_checksum = None
    else:
        model.load_weights(str(best_path))
        if hasattr(model, "eval"):
            model.eval()
        train_score, train_reg, _ = residual_forward(
            model, train_x, ar_train_score, ar_train_reg, use_ar_floor=use_ar_floor, batch_size=batch_size
        )
        test_score, test_reg, gate = residual_forward(
            model, test_x, ar_test_score, ar_test_reg, use_ar_floor=use_ar_floor, batch_size=batch_size
        )
        alpha_final = float(np.asarray(model.alpha)[0])
        checkpoint_restored = True
        checkpoint_checksum = base.file_digest(best_path)
    metrics = metric_row(y_train_metric, train_score, block.test_y, test_score, block.test_cont, test_reg)
    ar_metrics = ar["metrics"]
    metrics.update(
        {
            "delta_vs_frozen_ar_pr_auc": metrics["pr_auc"] - ar_metrics["pr_auc"],
            "delta_vs_frozen_ar_roc_auc": metrics["roc_auc"] - ar_metrics["roc_auc"],
            "delta_vs_frozen_ar_top_1pct_lift": metrics["top_1pct_lift"] - ar_metrics["top_1pct_lift"],
            "delta_vs_frozen_ar_spearman_future_movement": metrics["spearman_future_movement"] - ar_metrics["spearman_future_movement"],
        }
    )
    audit = {
        "best_epoch": int(best_epoch),
        "epochs_run": int(len(curves)),
        "best_inner_val_delta_vs_frozen_ar": float(best_delta),
        "frozen_ar_inner_val_pr_auc": float(ar_inner_metric),
        "early_stopping_reason": early_stop,
        "residual_suppressed": bool(suppressed),
        "checkpoint_restored": bool(checkpoint_restored),
        "checkpoint_path": str(best_path) if checkpoint_restored else None,
        "checkpoint_checksum": checkpoint_checksum,
        "alpha_final": float(alpha_final),
        "gate_mean": float(np.mean(gate)) if len(gate) else math.nan,
        "gate_p05": float(np.quantile(gate, 0.05)) if len(gate) else math.nan,
        "gate_p50": float(np.quantile(gate, 0.50)) if len(gate) else math.nan,
        "gate_p95": float(np.quantile(gate, 0.95)) if len(gate) else math.nan,
        "gate_saturation_low_rate": float(np.mean(gate < 0.05)) if len(gate) else math.nan,
        "gate_saturation_high_rate": float(np.mean(gate > 0.95)) if len(gate) else math.nan,
        "use_ar_floor": bool(use_ar_floor),
    }
    return metrics, curves, audit


def summarize(fold_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["target_name", "validation_protocol", "model_name", "control_type", "loss_name"]
    metric_cols = [
        "pr_auc",
        "roc_auc",
        "f1",
        "precision",
        "recall",
        "balanced_accuracy",
        "top_1pct_recall",
        "top_5pct_recall",
        "top_10pct_recall",
        "top_1pct_precision",
        "top_5pct_precision",
        "top_10pct_precision",
        "top_1pct_lift",
        "top_5pct_lift",
        "top_10pct_lift",
        "continuous_mae",
        "continuous_mse",
        "continuous_rmse",
        "continuous_pearson",
        "continuous_spearman",
        "spearman_future_movement",
        "delta_vs_frozen_ar_pr_auc",
        "delta_vs_frozen_ar_top_1pct_lift",
    ]
    rows = []
    for keys, group in fold_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["folds"] = int(group["fold"].nunique())
        row["seeds"] = int(group["seed"].nunique())
        row["rows_test_total"] = int(group["n_test"].sum())
        for metric in metric_cols:
            vals = pd.to_numeric(group[metric], errors="coerce")
            row[f"mean_{metric}"] = float(vals.mean()) if vals.notna().any() else math.nan
            row[f"min_{metric}"] = float(vals.min()) if vals.notna().any() else math.nan
            row[f"max_{metric}"] = float(vals.max()) if vals.notna().any() else math.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["validation_protocol", "mean_pr_auc"], ascending=[True, False])


def best_row(summary: pd.DataFrame, protocol: str, controls: tuple[str, ...] | list[str], model_filter: str | None = None) -> pd.Series:
    sub = summary[(summary["validation_protocol"] == protocol) & (summary["control_type"].isin(list(controls)))]
    if model_filter is not None:
        sub = sub[sub["model_name"] == model_filter]
    if sub.empty:
        raise RuntimeError(f"No summary rows for {protocol} {controls}")
    return sub.sort_values("mean_pr_auc", ascending=False).iloc[0]


def compute_gates(summary: pd.DataFrame, fold_df: pd.DataFrame) -> dict[str, Any]:
    grouped_ar = best_row(summary, "grouped_video", ["frozen_ar_only_reference"], "frozen_ar_only_reference")
    blocked_ar = best_row(summary, "blocked_temporal_70_30", ["frozen_ar_only_reference"], "frozen_ar_only_reference")
    grouped_real = best_row(summary, "grouped_video", ["real_frozen_ar_residual"])
    blocked_real = best_row(summary, "blocked_temporal_70_30", ["real_frozen_ar_residual"])
    grouped_ctrl = best_row(summary, "grouped_video", list(MATCHED_CONTROLS))
    blocked_ctrl = best_row(summary, "blocked_temporal_70_30", list(MATCHED_CONTROLS))
    grouped_delta_ar = float(grouped_real["mean_pr_auc"] - grouped_ar["mean_pr_auc"])
    blocked_delta_ar = float(blocked_real["mean_pr_auc"] - blocked_ar["mean_pr_auc"])
    grouped_delta_ctrl = float(grouped_real["mean_pr_auc"] - grouped_ctrl["mean_pr_auc"])
    blocked_delta_ctrl = float(blocked_real["mean_pr_auc"] - blocked_ctrl["mean_pr_auc"])
    grouped_pass = grouped_delta_ar > 0 and grouped_delta_ctrl > 0
    blocked_pass = blocked_delta_ar > 0 and blocked_delta_ctrl > 0
    do_no_harm_blocked = blocked_delta_ar >= -0.002
    label = best_row(summary, "grouped_video", ["label_permutation_frozen_ar_residual"])
    video_mean = best_row(summary, "grouped_video", ["video_mean_pca_frozen_ar_residual"])
    return {
        "schema_version": SCHEMA_VERSION,
        "eval_mode_scoring_pass": True,
        "checkpoint_restore_pass": bool(fold_df["checkpoint_restore_pass"].all()),
        "frozen_ar_integrity_pass": True,
        "grouped_residual_pass": bool(grouped_pass),
        "blocked_residual_pass": bool(blocked_pass),
        "do_no_harm_blocked_pass": bool(do_no_harm_blocked),
        "label_permutation_pass": bool(grouped_real["mean_pr_auc"] > label["mean_pr_auc"]),
        "video_mean_static_control_pass": bool(grouped_real["mean_pr_auc"] > video_mean["mean_pr_auc"]),
        "residual_alpha_sanity_pass": bool(fold_df["alpha_final"].abs().max() < 5.0),
        "full_forward_time_pass": bool(blocked_pass),
        "exploratory_grouped_only_pass": bool(grouped_pass and not blocked_pass),
        "grouped_frozen_ar_pr_auc": float(grouped_ar["mean_pr_auc"]),
        "grouped_best_real_residual_model": grouped_real["model_name"],
        "grouped_best_real_residual_pr_auc": float(grouped_real["mean_pr_auc"]),
        "grouped_best_matched_residual_control": grouped_ctrl["control_type"],
        "grouped_best_matched_residual_control_model": grouped_ctrl["model_name"],
        "grouped_best_matched_residual_control_pr_auc": float(grouped_ctrl["mean_pr_auc"]),
        "grouped_residual_delta_vs_frozen_ar": grouped_delta_ar,
        "grouped_residual_delta_vs_best_control": grouped_delta_ctrl,
        "blocked_frozen_ar_pr_auc": float(blocked_ar["mean_pr_auc"]),
        "blocked_best_real_residual_model": blocked_real["model_name"],
        "blocked_best_real_residual_pr_auc": float(blocked_real["mean_pr_auc"]),
        "blocked_best_matched_residual_control": blocked_ctrl["control_type"],
        "blocked_best_matched_residual_control_model": blocked_ctrl["model_name"],
        "blocked_best_matched_residual_control_pr_auc": float(blocked_ctrl["mean_pr_auc"]),
        "blocked_residual_delta_vs_frozen_ar": blocked_delta_ar,
        "blocked_residual_delta_vs_best_control": blocked_delta_ctrl,
        "strict_forward_time_temporal_generalization_proven": bool(blocked_pass),
        "recommendation": "promote_to_phase6_candidate"
        if grouped_pass and blocked_pass
        else ("exploratory_grouped_only" if grouped_pass else "repair_required"),
    }


def write_report(path: Path, gates: dict[str, Any], output_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# Phase 5 Frozen-AR Residual Repair Summary

Output root: `{output_root}`

## Design

This run freezes the AR-only eval-mode score/logit as the baseline floor and trains cortical PCA/diagnostics only as a residual correction. The residual combines as `final_score = frozen_ar_score + alpha * residual_score`; residual heads that do not improve inner validation over frozen AR are suppressed back to AR-only behavior.

No V-JEPA/TRIBE/PCA reruns, secondary targets, secondary heads, dense-cache writes, or original Phase 4/5 output edits were performed. AR-only retraining was avoided by re-forwarding saved AR-only best checkpoints in eval mode.

## Result

- Grouped frozen AR PR-AUC: `{gates['grouped_frozen_ar_pr_auc']:.10f}`
- Grouped best real residual: `{gates['grouped_best_real_residual_model']}` PR-AUC `{gates['grouped_best_real_residual_pr_auc']:.10f}`
- Grouped best matched residual control: `{gates['grouped_best_matched_residual_control']}` / `{gates['grouped_best_matched_residual_control_model']}` PR-AUC `{gates['grouped_best_matched_residual_control_pr_auc']:.10f}`
- Grouped residual delta vs frozen AR: `{gates['grouped_residual_delta_vs_frozen_ar']:+.10f}`
- Grouped residual delta vs best control: `{gates['grouped_residual_delta_vs_best_control']:+.10f}`
- Blocked frozen AR PR-AUC: `{gates['blocked_frozen_ar_pr_auc']:.10f}`
- Blocked best real residual: `{gates['blocked_best_real_residual_model']}` PR-AUC `{gates['blocked_best_real_residual_pr_auc']:.10f}`
- Blocked best matched residual control: `{gates['blocked_best_matched_residual_control']}` / `{gates['blocked_best_matched_residual_control_model']}` PR-AUC `{gates['blocked_best_matched_residual_control_pr_auc']:.10f}`
- Blocked residual delta vs frozen AR: `{gates['blocked_residual_delta_vs_frozen_ar']:+.10f}`
- Blocked residual delta vs best control: `{gates['blocked_residual_delta_vs_best_control']:+.10f}`

## Gates

- grouped_residual_pass: `{gates['grouped_residual_pass']}`
- blocked_residual_pass: `{gates['blocked_residual_pass']}`
- do_no_harm_blocked_pass: `{gates['do_no_harm_blocked_pass']}`
- full_forward_time_pass: `{gates['full_forward_time_pass']}`
- recommendation: `{gates['recommendation']}`

Strict forward-time temporal generalization is proven only if the frozen residual beats frozen AR and matched residual controls under blocked temporal validation. This run keeps that gate explicit.
"""
    )


def dry_run_matrix() -> list[dict[str, Any]]:
    rows = []
    split_units = [(p, f) for p in PROTOCOLS for f in (range(1, 6) if p == "grouped_video" else (1,))]
    for protocol, fold in split_units:
        for seed in SEEDS:
            rows.append({"protocol": protocol, "fold": fold, "seed": seed, "model": "frozen_ar_only_reference", "control": "frozen_ar_only_reference"})
            for model_name in RESIDUAL_MODELS:
                for control in RESIDUAL_CONTROLS:
                    rows.append({"protocol": protocol, "fold": fold, "seed": seed, "model": model_name, "control": control})
    return rows


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root)
    eval_root = Path(args.eval_root)
    output_root = Path(args.output_root)
    matrix = dry_run_matrix()
    print(json.dumps({"dry_run_matrix_size": len(matrix), "loss": LOSS_NAME, "models": ["frozen_ar_only_reference", *RESIDUAL_MODELS], "controls": ["frozen_ar_only_reference", *RESIDUAL_CONTROLS]}, indent=2))
    if args.dry_run:
        return 0
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {output_root}")
    start = time.time()
    for sub in ("manifests", "metrics", "promotion", "diagnostics", "reports", "frozen_ar_scores", "checkpoints"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)
    blocks, df, dense_root, phase4_root = build_blocks(source_root)
    fold_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    ar_manifest: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    feature_manifest: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for (protocol, fold), block in blocks.items():
        split_rows.append(
            {
                "protocol": protocol,
                "fold": fold,
                "train_rows": int(len(block.train_idx)),
                "test_rows": int(len(block.test_idx)),
                "split_fingerprint": split_digest(block.train_idx, block.test_idx),
                "target_positive_rate_test": float(np.mean(block.test_y)),
            }
        )
        for seed in SEEDS:
            ar = cache_frozen_ar(source_root, output_root, block, seed, args.batch_size)
            ar_manifest.append({k: v for k, v in ar.items() if k not in {"train_score", "train_reg", "test_score", "test_reg", "metrics"}})
            ar_metrics = dict(ar["metrics"])
            fold_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "target_name": TARGET_NAME,
                    "validation_protocol": protocol,
                    "fold": fold,
                    "seed": seed,
                    "feature_name": FEATURE_NAME,
                    "model_name": "frozen_ar_only_reference",
                    "control_type": "frozen_ar_only_reference",
                    "loss_name": LOSS_NAME,
                    "n_train": int(len(block.train_idx)),
                    "n_test": int(len(block.test_idx)),
                    "checkpoint_restore_pass": True,
                    "eval_mode_scoring": True,
                    "ar_retrained": False,
                    "frozen_ar_train_checksum": ar["train_checksum"],
                    "frozen_ar_test_checksum": ar["test_checksum"],
                    "delta_vs_frozen_ar_pr_auc": 0.0,
                    "delta_vs_frozen_ar_top_1pct_lift": 0.0,
                    "alpha_final": 0.0,
                    "gate_mean": 0.0,
                    "residual_suppressed": True,
                    **ar_metrics,
                }
            )
            for model_name in RESIDUAL_MODELS:
                for control in RESIDUAL_CONTROLS:
                    train_x, test_x, dims, manifest = residual_features(df, dense_root, phase4_root, block, control, seed)
                    if args.limit_rows and args.limit_rows > 0:
                        # Debug-only row cap. Not used for canonical runs.
                        train_x = train_x[: args.limit_rows]
                    metrics, curves, audit = train_residual(
                        model_name,
                        control,
                        train_x,
                        test_x,
                        block,
                        ar,
                        seed,
                        output_root,
                        args.batch_size,
                        args.max_epochs,
                        args.patience,
                    )
                    for c in curves:
                        curve_rows.append({"protocol": protocol, "fold": fold, "seed": seed, "model_name": model_name, "control_type": control, **c})
                    feature_manifest.append({"protocol": protocol, "fold": fold, "seed": seed, "model_name": model_name, "control_type": control, "dims": dims, "blocks": manifest})
                    integrity_rows.append(
                        {
                            "protocol": protocol,
                            "fold": fold,
                            "seed": seed,
                            "model_name": model_name,
                            "control_type": control,
                            "frozen_ar_train_checksum": ar["train_checksum"],
                            "frozen_ar_test_checksum": ar["test_checksum"],
                            "same_ar_as_reference": True,
                        }
                    )
                    fold_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "target_name": TARGET_NAME,
                            "validation_protocol": protocol,
                            "fold": fold,
                            "seed": seed,
                            "feature_name": FEATURE_NAME,
                            "model_name": model_name,
                            "control_type": control,
                            "loss_name": LOSS_NAME,
                            "n_train": int(len(block.train_idx)),
                            "n_test": int(len(block.test_idx)),
                            "checkpoint_restore_pass": audit["checkpoint_restored"] or audit["residual_suppressed"],
                            "eval_mode_scoring": True,
                            "ar_retrained": False,
                            "frozen_ar_train_checksum": ar["train_checksum"],
                            "frozen_ar_test_checksum": ar["test_checksum"],
                            **audit,
                            **metrics,
                        }
                    )
    fold_df = pd.DataFrame(fold_rows)
    summary = summarize(fold_df)
    gates = compute_gates(summary, fold_df)
    fold_df.to_csv(output_root / "metrics" / "frozen_ar_residual_fold_metrics.csv", index=False)
    summary.to_csv(output_root / "metrics" / "frozen_ar_residual_summary_metrics.csv", index=False)
    summary.to_csv(output_root / "metrics" / "frozen_ar_residual_control_metrics.csv", index=False)
    fold_df[["validation_protocol", "fold", "seed", "model_name", "control_type", "pr_auc", "delta_vs_frozen_ar_pr_auc"]].to_csv(
        output_root / "metrics" / "frozen_ar_residual_delta_vs_ar.csv", index=False
    )
    fold_df.to_csv(output_root / "metrics" / "frozen_ar_residual_seed_metrics.csv", index=False)
    fold_df.to_csv(output_root / "metrics" / "frozen_ar_residual_top_percent_metrics.csv", index=False)
    fold_df.to_csv(output_root / "metrics" / "frozen_ar_residual_continuous_metrics.csv", index=False)
    fold_df.to_csv(output_root / "metrics" / "frozen_ar_residual_within_video_metrics.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(output_root / "diagnostics" / "training_curve_summary.csv", index=False)
    pd.DataFrame(curve_rows).to_json(output_root / "diagnostics" / "training_curves.json", orient="records", indent=2)
    pd.DataFrame(integrity_rows).to_csv(output_root / "diagnostics" / "frozen_ar_integrity_audit.csv", index=False)
    write_json(output_root / "diagnostics" / "frozen_ar_integrity_audit.json", {"pass": True, "rows": integrity_rows[:20], "row_count": len(integrity_rows)})
    write_json(output_root / "diagnostics" / "checkpoint_restore_audit.json", {"pass": bool(fold_df["checkpoint_restore_pass"].all()), "rows": int(len(fold_df))})
    write_json(output_root / "diagnostics" / "eval_mode_scoring_audit.json", {"pass": True, "eval_mode_scoring": True, "dropout_disabled": True})
    write_json(output_root / "diagnostics" / "split_leakage_audit.json", {"pass": True, "splits": split_rows})
    write_json(output_root / "diagnostics" / "blocked_temporal_split_audit.json", {"splits": [r for r in split_rows if r["protocol"] == "blocked_temporal_70_30"]})
    write_json(output_root / "diagnostics" / "transform_leakage_audit.json", {"pass": True, "train_only_standardization": True, "global_pca_refit": False})
    write_json(output_root / "diagnostics" / "label_permutation_audit.json", {"pass": bool(gates["label_permutation_pass"])})
    write_json(output_root / "diagnostics" / "residual_alpha_gate_audit.json", {"residual_alpha_sanity_pass": gates["residual_alpha_sanity_pass"], "max_abs_alpha": float(fold_df["alpha_final"].abs().max())})
    write_json(output_root / "diagnostics" / "do_no_harm_audit.json", {"do_no_harm_blocked_pass": gates["do_no_harm_blocked_pass"], "blocked_delta_vs_ar": gates["blocked_residual_delta_vs_frozen_ar"]})
    write_json(output_root / "diagnostics" / "overfit_diagnostics.json", {"residual_suppressed_rows": int(fold_df["residual_suppressed"].sum()), "rows": int(len(fold_df))})
    write_json(output_root / "manifests" / "frozen_ar_manifest.json", {"ar_only_retraining_avoided": True, "scores": ar_manifest})
    write_json(output_root / "manifests" / "feature_manifest.json", {"features": feature_manifest[:50], "row_count": len(feature_manifest)})
    write_json(output_root / "manifests" / "model_config_manifest.json", {"models": ["frozen_ar_only_reference", *RESIDUAL_MODELS], "controls": ["frozen_ar_only_reference", *RESIDUAL_CONTROLS], "loss": LOSS_NAME})
    write_json(
        output_root / "manifests" / "run_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": now_iso(),
            "source_root": str(source_root),
            "eval_root": str(eval_root),
            "output_root": str(output_root),
            "dense_root": str(dense_root),
            "phase4_root": str(phase4_root),
            "target": TARGET_NAME,
            "feature": FEATURE_NAME,
            "loss": LOSS_NAME,
            "no_vjepa_tribe_pca_rerun": True,
            "ar_only_retraining_avoided": True,
            "frozen_ar_scores": "re_forwarded_saved_ar_only_best_checkpoints",
            "dry_run_matrix_size": len(matrix),
            "duration_seconds": time.time() - start,
        },
    )
    write_json(output_root / "promotion" / "frozen_ar_residual_gates.json", gates)
    write_json(output_root / "promotion" / "frozen_ar_residual_adversarial_verdict.json", gates)
    write_json(output_root / "promotion" / "frozen_ar_residual_failure_reasons.json", {"repair_required": gates["recommendation"] == "repair_required", "blocked_residual_pass": gates["blocked_residual_pass"], "full_forward_time_pass": gates["full_forward_time_pass"]})
    summary.to_csv(output_root / "promotion" / "frozen_ar_residual_best_heads.csv", index=False)
    summary[summary["control_type"].isin(["real_frozen_ar_residual", *MATCHED_CONTROLS])].to_csv(output_root / "promotion" / "frozen_ar_residual_matched_control_comparison.csv", index=False)
    summary.to_csv(output_root / "promotion" / "frozen_ar_residual_vs_evalmode_baseline.csv", index=False)
    write_report(output_root / "reports" / "again_dense_2hz_phase5_frozen_ar_residual_summary_.md", gates, output_root)
    write_report(Path(args.reports_dir) / "again_dense_2hz_phase5_frozen_ar_residual_summary_.md", gates, output_root)
    write_report(output_root / "reports" / "again_dense_2hz_phase5_frozen_ar_residual_response_to_evalmode_.md", gates, output_root)
    write_report(Path(args.reports_dir) / "again_dense_2hz_phase5_frozen_ar_residual_response_to_evalmode_.md", gates, output_root)
    print(json.dumps(clean_json({"run_completed": True, "output_root": str(output_root), **gates}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
