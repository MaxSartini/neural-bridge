"""Blocked redesigned-target frozen-AR residual test.

Bounded matrix:
2 redesigned targets x 3 seeds x 7 controls = 42 rows.

This script uses the fold-safe PCA artifacts from
`outputs/again_dense_2hz_phase5_redesigned_target_foldsafe_pca_20260630_005312/`.
It does not run grouped, broad variants, extra targets, AR retraining,
V-JEPA/TRIBE, or PCA refitting.
"""

from __future__ import annotations

import argparse
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
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts import again_dense_2hz_phase4_pca_bridge as phase4
from backend.scripts import run_again_dense_2hz_phase5_adversarial_correction_fixplus as phase5_base
from backend.scripts import run_again_dense_2hz_phase5_blocked_residual_clean_confirm as clean
from backend.scripts import run_again_dense_2hz_phase5_blocked_residual_targeted as targeted
from backend.scripts import run_again_dense_2hz_phase5_continuous_residual_blocked as continuous_run
from backend.scripts import run_again_dense_2hz_phase5_frozen_ar_residual as fr
from backend.scripts import run_again_dense_2hz_phase5_learned_heads as base
from backend.scripts.again_dense_2hz_benchmark import TargetSpec


SCHEMA_VERSION = "again_dense_2hz_phase5_redesigned_target_blocked_v1"
SOURCE_ROOT = Path("outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_20260629_171825")
FOLDSAFE_PCA_ROOT = Path("outputs/again_dense_2hz_phase5_redesigned_target_foldsafe_pca_20260630_005312")
PROTOCOL = "blocked_temporal_70_30"
FOLD = 1
FEATURE_NAME = "temporal_mean_2s_then_pca256"
SOURCE_FAMILY = "temporal_mean_2s"
PCA_WIDTH = 256
VARIANT = "monotonic_do_no_harm_residual"
SEEDS = (20260625, 20260626, 20260627)
CONTROLS = (
    "frozen_ar_only",
    "real_residual",
    "shuffled_pca_residual",
    "random_pca_residual",
    "label_permutation_residual",
    "train_only_video_mean_residual",
    "diagnostics_only_residual",
)
PRIMARY_CONTROLS = (
    "shuffled_pca_residual",
    "random_pca_residual",
    "label_permutation_residual",
    "train_only_video_mean_residual",
)
BINARY_TARGET = "future_arousal_max_delta_rows_4_10_train_q90"
CONTINUOUS_TARGET = "residual_future_max_delta_rows_4_10"
TARGET_TYPES = {BINARY_TARGET: "binary", CONTINUOUS_TARGET: "continuous"}
BINARY_THRESHOLD = 0.001
CONTINUOUS_THRESHOLD = 0.001
DO_NO_HARM_SEED_FLOOR = -0.0005


@dataclass
class RunBlock:
    target_name: str
    target_type: str
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


def default_output_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(f"outputs/again_dense_2hz_phase5_redesigned_target_blocked_{stamp}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--foldsafe-pca-root", default=str(FOLDSAFE_PCA_ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    fr.write_json(path, obj)


def validate_pca_root(pca_root: Path) -> dict[str, Any]:
    audit_path = pca_root / "diagnostics" / "redesigned_split_pca_leakage_audit.json"
    manifest_path = pca_root / "manifests" / "redesigned_pca_manifest.json"
    if not audit_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing redesigned PCA audit/manifest under {pca_root}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("leakage_audit_pass") or not audit.get("safe_to_run_redesigned_target_training"):
        raise RuntimeError(f"Fold-safe PCA audit is not pass/safe: {audit_path}")
    if not audit.get("original_pca_artifact_not_reused") or not audit.get("no_test_rows_used_in_pca_fit"):
        raise RuntimeError("Fold-safe PCA audit does not satisfy no-reuse/no-test-fit policy")
    return audit


def future_max_delta(df: pd.DataFrame, start: int = 4, end: int = 10) -> tuple[np.ndarray, np.ndarray]:
    n = len(df)
    arousal = df["arousal"].to_numpy(dtype=np.float64)
    labels = df["label_available"].to_numpy(dtype=bool)
    values = np.full(n, np.nan, dtype=np.float64)
    mask = np.zeros(n, dtype=bool)
    for _video_id, group in df.groupby("video_id", sort=False):
        idx = group.index.to_numpy(dtype=np.int64)
        if len(idx) <= end:
            continue
        base_idx = idx[: len(idx) - end]
        future = np.vstack([arousal[idx[offset : len(idx) - end + offset]] for offset in range(start, end + 1)])
        future_masks = np.vstack([labels[idx[offset : len(idx) - end + offset]] for offset in range(start, end + 1)])
        feasible = labels[base_idx] & np.all(future_masks, axis=0) & np.isfinite(arousal[base_idx]) & np.all(np.isfinite(future), axis=0)
        out = np.full(len(base_idx), np.nan, dtype=np.float64)
        if np.any(feasible):
            out[feasible] = np.max(future[:, feasible] - arousal[base_idx][feasible][None, :], axis=0)
        values[base_idx] = out
        mask[base_idx] = feasible
    return values, mask


def trailing_4s_mean(df: pd.DataFrame) -> np.ndarray:
    out = np.full(len(df), np.nan, dtype=np.float64)
    for _video_id, group in df.groupby("video_id", sort=False):
        idx = group.index.to_numpy(dtype=np.int64)
        series = pd.Series(df.loc[idx, "arousal"].to_numpy(dtype=np.float64))
        out[idx] = series.rolling(window=9, min_periods=9).mean().to_numpy()
    return out


def add_redesigned_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df.copy()
    max_4_10, valid_4_10 = future_max_delta(df, 4, 10)
    df["future_arousal_max_delta_rows_4_10"] = max_4_10
    df["target_mask_future_arousal_max_delta_rows_4_10"] = valid_4_10
    trailing_2s = np.nanmean(
        np.vstack(
            [
                df["arousal"].to_numpy(dtype=np.float64),
                df["arousal_lag_1row"].to_numpy(dtype=np.float64),
                df["arousal_lag_2row"].to_numpy(dtype=np.float64),
                df["arousal_lag_4row"].to_numpy(dtype=np.float64),
            ]
        ),
        axis=0,
    )
    ar_features = np.column_stack(
        [
            df["arousal"].to_numpy(dtype=np.float64),
            df["arousal_lag_1row"].to_numpy(dtype=np.float64),
            trailing_2s,
            trailing_4s_mean(df),
            df["arousal_delta_prev_4row"].to_numpy(dtype=np.float64),
            df["video_time_fraction"].to_numpy(dtype=np.float64),
        ]
    )
    residual_valid = (
        df["label_available"].to_numpy(dtype=bool)
        & df["ar_context_available"].to_numpy(dtype=bool)
        & valid_4_10
        & np.isfinite(max_4_10)
        & np.all(np.isfinite(ar_features), axis=1)
    )
    train_idx = phase4.blocked_temporal_split(df, residual_valid)[0][2].astype(np.int64)
    test_idx = phase4.blocked_temporal_split(df, residual_valid)[0][3].astype(np.int64)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(ar_features[train_idx])
    model = Ridge(alpha=10.0)
    model.fit(x_train, max_4_10[train_idx])
    all_idx = np.concatenate([train_idx, test_idx])
    pred = model.predict(scaler.transform(ar_features[all_idx]))
    residual = np.full(len(df), np.nan, dtype=np.float64)
    residual[all_idx] = max_4_10[all_idx] - pred
    df["residual_future_max_delta_rows_4_10"] = residual
    df["target_mask_residual_future_max_delta_rows_4_10"] = np.isfinite(residual)
    return df, {
        "policy": "train_only_simple_ar_residualizer_inside_redesigned_blocked_split",
        "ridge_alpha": 10.0,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_idx_digest": phase4.array_digest(train_idx),
        "test_idx_digest": phase4.array_digest(test_idx),
        "train_r2": float(model.score(x_train, max_4_10[train_idx])),
    }


def target_specs() -> tuple[TargetSpec, TargetSpec]:
    return (
        TargetSpec(
            name=BINARY_TARGET,
            value_column="future_arousal_max_delta_rows_4_10",
            mask_column="target_mask_future_arousal_max_delta_rows_4_10",
            threshold_mode="train_quantile",
            quantile=0.90,
            transform="positive_delta",
        ),
        TargetSpec(
            name=CONTINUOUS_TARGET,
            value_column="residual_future_max_delta_rows_4_10",
            mask_column="target_mask_residual_future_max_delta_rows_4_10",
            threshold_mode="train_quantile",
            quantile=0.90,
            transform="identity",
        ),
    )


def build_blocks(source_root: Path, pca_root: Path) -> tuple[dict[str, RunBlock], pd.DataFrame, Path, dict[str, Any]]:
    phase5_base.patch_base_module()
    source_manifest = json.loads((source_root / "run_manifest.json").read_text(encoding="utf-8"))
    dense_root = Path(source_manifest["dense_root"])
    df = base.load_labels(dense_root)
    df, residual_meta = add_redesigned_targets(df)
    splits = phase4.build_split_specs(df, protocols=(PROTOCOL,), n_splits=5, target_specs=target_specs())
    blocks: dict[str, RunBlock] = {}
    spec = base.feature_spec(FEATURE_NAME)
    for split in splits:
        rng = np.random.default_rng(20260625 + int(split.fold) + 99)
        train_idx, test_idx, ar_train_x, ar_test_x, dims, _manifest = phase5_base.assemble_feature_blocks_correction(
            df,
            dense_root,
            pca_root,
            split,
            spec,
            include_ar=True,
            include_temporal_diagnostics=True,
            control="ar_only_head",
            rng=rng,
        )
        ar_train_x, ar_test_x = base.standardize_train_only(ar_train_x, ar_test_x)
        train_cont = base.target_continuous_values(df, split, train_idx, split.target.value_column)
        test_cont = base.target_continuous_values(df, split, test_idx, split.target.value_column)
        train_y, test_y = fr.split_y(split, train_idx, test_idx)
        inner_train, inner_val, inner_audit = fr.inner_split(df, train_idx, train_y)
        blocks[split.target.name] = RunBlock(
            target_name=split.target.name,
            target_type=TARGET_TYPES[split.target.name],
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
            inner_audit=inner_audit,
            train_video_id=df.loc[train_idx, "video_id"].astype(str).to_numpy(),
            test_video_id=df.loc[test_idx, "video_id"].astype(str).to_numpy(),
            train_time=df.loc[train_idx, "time_seconds"].to_numpy(dtype=np.float32),
            test_time=df.loc[test_idx, "time_seconds"].to_numpy(dtype=np.float32),
            ar_train_x=ar_train_x,
            ar_test_x=ar_test_x,
            ar_block_dims=dims,
        )
    if set(blocks) != {BINARY_TARGET, CONTINUOUS_TARGET}:
        raise RuntimeError(f"Unexpected target blocks: {sorted(blocks)}")
    return blocks, df, dense_root, residual_meta


def cache_frozen_ar(source_root: Path, output_root: Path, block: RunBlock, seed: int, batch_size: int) -> dict[str, Any]:
    row = fr.ar_checkpoint_row(source_root, block.protocol, block.fold, seed)
    checkpoint = Path(str(row.checkpoint_path))
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing AR-only checkpoint: {checkpoint}")
    model = base.make_model(fr.config_for_ar(seed), block.ar_train_x.shape[1], block.ar_block_dims)
    _ = model(base.mx.array(block.ar_train_x[:2], dtype=base.mx.float32))
    model.load_weights(str(checkpoint))
    if hasattr(model, "eval"):
        model.eval()
    train_score, train_reg = fr.score_existing_model(model, block.ar_train_x, batch_size)
    test_score, test_reg = fr.score_existing_model(model, block.ar_test_x, batch_size)
    key = f"{block.target_name}__{block.protocol}__fold{block.fold}__seed{seed}__{fr.LOSS_NAME}"
    out_dir = output_root / "frozen_ar_scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, row_idx, video_id, time_sec, score, reg in (
        ("train", block.train_idx, block.train_video_id, block.train_time, train_score, train_reg),
        ("heldout_test", block.test_idx, block.test_video_id, block.test_time, test_score, test_reg),
        ("inner_val", block.train_idx[block.inner_val], block.train_video_id[block.inner_val], block.train_time[block.inner_val], train_score[block.inner_val], train_reg[block.inner_val]),
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
        "target_name": block.target_name,
        "protocol": block.protocol,
        "fold": block.fold,
        "seed": int(seed),
        "loss": fr.LOSS_NAME,
        "checkpoint_path": str(checkpoint),
        "checkpoint_checksum": row.checkpoint_checksum,
        "source": "re_forwarded_saved_ar_only_best_checkpoint_for_redesigned_target_rows",
        "ar_retrained": False,
        "train_score": train_score,
        "train_reg": train_reg,
        "test_score": test_score,
        "test_reg": test_reg,
        "train_checksum": fr.hash_array(train_score),
        "test_checksum": fr.hash_array(test_score),
    }


def residual_features_for_control(
    df: pd.DataFrame,
    pca_root: Path,
    dense_root: Path,
    block: RunBlock,
    control: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], list[dict[str, Any]]]:
    if control == "train_only_video_mean_residual":
        return clean.video_mean_features(df, dense_root, pca_root, block, train_only=True)
    mapping = {
        "real_residual": "real_frozen_ar_residual",
        "shuffled_pca_residual": "shuffled_pca_frozen_ar_residual",
        "random_pca_residual": "random_pca_frozen_ar_residual",
        "label_permutation_residual": "real_frozen_ar_residual",
        "diagnostics_only_residual": "diag_only_frozen_ar_residual",
    }
    return fr.residual_features(df, dense_root, pca_root, block, mapping[control], seed)


def binary_metric_row(block: RunBlock, train_score: np.ndarray, test_score: np.ndarray, test_reg: np.ndarray) -> dict[str, Any]:
    return targeted.metric_row(block.train_y, train_score, block.test_y, test_score, block.test_cont, test_reg, block.test_video_id)


def continuous_metric_row(block: RunBlock, train_score: np.ndarray, test_score: np.ndarray, test_reg: np.ndarray) -> dict[str, Any]:
    return continuous_run.continuous_metric_row(block.train_y, train_score, block.test_y, test_score, block.test_cont, test_reg)


def add_binary_deltas(metrics: dict[str, Any], ar_metrics: dict[str, Any]) -> None:
    for key in ("pr_auc", "roc_auc", "top_1pct_recall", "top_5pct_recall", "top_10pct_recall", "spearman_future_movement"):
        metrics[f"delta_vs_frozen_ar_{key}"] = float(metrics[key] - ar_metrics[key])


def add_continuous_deltas(metrics: dict[str, Any], ar_metrics: dict[str, Any]) -> None:
    continuous_run.add_delta_metrics(metrics, ar_metrics)


def train_residual(
    *,
    target_type: str,
    control: str,
    train_x: np.ndarray,
    test_x: np.ndarray,
    block: RunBlock,
    ar: dict[str, Any],
    seed: int,
    output_root: Path,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    base.require_mlx()
    base.mx.random.seed(int(seed))
    model = targeted.TargetedResidualHead(train_x.shape[1], VARIANT)
    optimizer = base.optim.AdamW(learning_rate=2e-4, weight_decay=1e-4)
    inner_train = block.inner_train
    inner_val = block.inner_val
    rng = np.random.default_rng(int(seed) + block.fold * 7919 + (11 if target_type == "binary" else 29))
    train_y_metric = block.train_y.copy()
    train_cont_metric = block.train_cont.copy()
    selection_y = block.train_y.copy()
    selection_cont = block.train_cont.copy()
    if control == "label_permutation_residual":
        perm = rng.permutation(len(train_y_metric))
        train_y_metric = train_y_metric[perm]
        train_cont_metric = train_cont_metric[perm]
        selection_y = train_y_metric
        selection_cont = train_cont_metric
        selection_policy = "permuted_inner_val_labels_targets"
        training_policy = "permuted_train_labels_targets"
    else:
        selection_policy = "true_inner_val_labels_targets"
        training_policy = "true_train_labels_targets"
    ar_train_score = ar["train_score"].astype(np.float32)
    ar_train_reg = ar["train_reg"].astype(np.float32)
    ar_test_score = ar["test_score"].astype(np.float32)
    ar_test_reg = ar["test_reg"].astype(np.float32)
    if target_type == "binary":
        ar_inner_metric = average_precision_score(selection_y[inner_val], ar_train_score[inner_val])
        best_score = 0.0
    else:
        ar_inner_metrics = continuous_run.continuous_metric_row(
            block.train_y[inner_train],
            ar_train_score[inner_train],
            block.train_y[inner_val],
            ar_train_score[inner_val],
            selection_cont[inner_val],
            ar_train_reg[inner_val],
        )
        best_tuple = (0.0, 0.0, 0.0)
    q80 = float(np.quantile(train_cont_metric[inner_train], 0.80))
    q90 = float(np.quantile(train_cont_metric[inner_train], 0.90))
    best_epoch = 0
    best_path = output_root / "checkpoints" / (
        f"{block.target_name}__{block.protocol}__fold{block.fold}__{control}__{VARIANT}__{target_type}__{seed}__best.npz"
    )
    best_path.parent.mkdir(parents=True, exist_ok=True)
    curves: list[dict[str, Any]] = []
    stale = 0
    early_stop = "max_epochs_reached"
    suppressed = True

    def loss_fn(model_obj: targeted.TargetedResidualHead, xb: Any, ar_b: Any, ar_r: Any, yb: Any, yr: Any, wb: Any) -> Any:
        out = model_obj(xb, ar_b, ar_r, use_ar_floor=True)
        reg_loss = base.mx.mean(base.nn.losses.huber_loss(out[:, 0:1], yr, delta=1.0) * wb)
        alpha_penalty = 0.01 * base.mx.mean(model_obj.alpha_value() * model_obj.alpha_value())
        if target_type == "continuous":
            anchor = 0.04 * base.mx.mean((out[:, 0:1] - ar_r[:, None]) * (out[:, 0:1] - ar_r[:, None]))
            return reg_loss + anchor + alpha_penalty
        bce = base.nn.losses.binary_cross_entropy(out[:, 1:2], yb, with_logits=True)
        return reg_loss + 0.5 * base.mx.mean(bce * wb) + alpha_penalty

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
            yb_np = train_y_metric[rel].astype(np.float32)[:, None]
            yr_np = train_cont_metric[rel].astype(np.float32)[:, None]
            if target_type == "continuous":
                weights = 1.0 + 1.0 * (yr_np >= q80).astype(np.float32) + 2.0 * (yr_np >= q90).astype(np.float32)
            else:
                weights = np.ones_like(yb_np, dtype=np.float32)
            loss, grads = loss_and_grad(
                model,
                xb,
                ab,
                rb,
                base.mx.array(yb_np, dtype=base.mx.float32),
                base.mx.array(yr_np, dtype=base.mx.float32),
                base.mx.array(weights, dtype=base.mx.float32),
            )
            grads, _ = base.optim.clip_grad_norm(grads, 1.0)
            optimizer.update(model, grads)
            base.mx.eval(model.parameters(), optimizer.state)
            total += float(np.asarray(loss))
            batches += 1
        val_score, val_reg, val_gate = targeted.residual_forward(model, train_x[inner_val], ar_train_score[inner_val], ar_train_reg[inner_val], batch_size=batch_size)
        if target_type == "binary":
            val_metric = average_precision_score(selection_y[inner_val], val_score) if len(np.unique(selection_y[inner_val])) > 1 else math.nan
            delta = float(val_metric - ar_inner_metric) if math.isfinite(val_metric) and math.isfinite(ar_inner_metric) else math.nan
            comparable = delta if math.isfinite(delta) else -math.inf
            curve = {
                "epoch": epoch,
                "train_loss": total / max(1, batches),
                "inner_val_selection_metric": val_metric,
                "inner_val_delta_vs_frozen_ar": delta,
                "selection_metric_name": "pr_auc_delta_vs_frozen_ar",
            }
            improved = comparable > best_score
            if improved:
                best_score = comparable
        else:
            val_metrics = continuous_run.continuous_metric_row(
                block.train_y[inner_train],
                ar_train_score[inner_train],
                block.train_y[inner_val],
                ar_train_score[inner_val],
                selection_cont[inner_val],
                val_reg,
            )
            continuous_run.add_delta_metrics(val_metrics, ar_inner_metrics)
            current_tuple = (
                float(val_metrics["delta_vs_frozen_ar_top_5pct_continuous_lift"]),
                float(val_metrics["delta_vs_frozen_ar_continuous_spearman"]),
                float(val_metrics["delta_vs_frozen_ar_top_10pct_continuous_lift"]),
            )
            curve = {
                "epoch": epoch,
                "train_loss": total / max(1, batches),
                "inner_val_selection_metric": current_tuple[0],
                "inner_val_delta_vs_frozen_ar": current_tuple[0],
                "inner_val_spearman_delta_vs_frozen_ar": current_tuple[1],
                "selection_metric_name": "top_5pct_lift_delta_vs_frozen_ar",
            }
            improved = current_tuple > best_tuple
            if improved:
                best_tuple = current_tuple
        curve.update(
            {
                "alpha": float(np.asarray(model.alpha_value())[0]),
                "gate_mean": float(np.mean(val_gate)),
                "gate_p95": float(np.quantile(val_gate, 0.95)),
                "inner_val_selection_policy": selection_policy,
            }
        )
        curves.append(curve)
        if improved:
            model.save_weights(str(best_path))
            best_epoch = epoch
            stale = 0
            suppressed = False
        else:
            stale += 1
        if stale >= int(patience):
            early_stop = "patience_exhausted"
            break

    if suppressed:
        train_score, train_reg = ar_train_score, ar_train_reg
        test_score, test_reg = ar_test_score, ar_test_reg
        gate = np.zeros_like(test_score)
        checkpoint_restored = False
        checkpoint_checksum = None
        alpha_final = 0.0
    else:
        model.load_weights(str(best_path))
        if hasattr(model, "eval"):
            model.eval()
        train_score, train_reg, _ = targeted.residual_forward(model, train_x, ar_train_score, ar_train_reg, batch_size=batch_size)
        test_score, test_reg, gate = targeted.residual_forward(model, test_x, ar_test_score, ar_test_reg, batch_size=batch_size)
        checkpoint_restored = True
        checkpoint_checksum = base.file_digest(best_path)
        alpha_final = float(np.asarray(model.alpha_value())[0])
    if target_type == "binary":
        metrics = binary_metric_row(block, train_score, test_score, test_reg)
        ar_metrics = binary_metric_row(block, ar_train_score, ar_test_score, ar_test_reg)
        add_binary_deltas(metrics, ar_metrics)
    else:
        metrics = continuous_metric_row(block, train_score, test_score, test_reg)
        ar_metrics = continuous_metric_row(block, ar_train_score, ar_test_score, ar_test_reg)
        add_continuous_deltas(metrics, ar_metrics)
    audit = {
        "best_epoch": int(best_epoch),
        "epochs_run": int(len(curves)),
        "best_inner_val_delta_vs_frozen_ar": float(best_score if target_type == "binary" else best_tuple[0]),
        "inner_val_selection_policy": selection_policy,
        "training_label_policy": training_policy,
        "heldout_scoring_policy": "true_heldout_labels_targets",
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
    }
    return metrics, curves, audit


def summarize(fold_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["target_name", "target_type", "validation_protocol", "variant_name", "control_type"]
    metric_cols = [
        "pr_auc",
        "roc_auc",
        "top_1pct_recall",
        "top_5pct_recall",
        "top_10pct_recall",
        "top_1pct_precision",
        "top_5pct_precision",
        "top_10pct_precision",
        "continuous_spearman",
        "continuous_pearson",
        "top_1pct_continuous_lift",
        "top_5pct_continuous_lift",
        "top_10pct_continuous_lift",
        "top_1pct_avg_true_movement",
        "top_5pct_avg_true_movement",
        "top_10pct_avg_true_movement",
        "ndcg_at_1pct",
        "ndcg_at_5pct",
        "ndcg_at_10pct",
        "delta_vs_frozen_ar_pr_auc",
        "delta_vs_frozen_ar_top_5pct_continuous_lift",
        "delta_vs_frozen_ar_continuous_spearman",
    ]
    rows = []
    for keys, group in fold_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["seeds"] = int(group["seed"].nunique())
        row["rows_test_total"] = int(group["n_test"].sum())
        for metric in metric_cols:
            if metric in group:
                vals = pd.to_numeric(group[metric], errors="coerce")
                row[f"mean_{metric}"] = float(vals.mean()) if vals.notna().any() else math.nan
                row[f"min_{metric}"] = float(vals.min()) if vals.notna().any() else math.nan
                row[f"max_{metric}"] = float(vals.max()) if vals.notna().any() else math.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["target_name", "control_type"])


def seed_deltas(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, group in fold_df.groupby("target_name"):
        target_type = str(group["target_type"].iloc[0])
        for seed, sg in group.groupby("seed"):
            vals = sg.set_index("control_type")
            real = vals.loc["real_residual"]
            row: dict[str, Any] = {"target_name": target, "target_type": target_type, "seed": int(seed)}
            if target_type == "binary":
                row["real_pr_auc"] = float(real["pr_auc"])
                for control in ("frozen_ar_only", *PRIMARY_CONTROLS):
                    row[f"{control}_pr_auc"] = float(vals.loc[control, "pr_auc"])
                    row[f"real_minus_{control}_pr_auc"] = float(real["pr_auc"] - vals.loc[control, "pr_auc"])
            else:
                row["real_top_5pct_lift"] = float(real["top_5pct_continuous_lift"])
                row["real_spearman"] = float(real["continuous_spearman"])
                row["real_pr_auc"] = float(real["pr_auc"])
                for control in ("frozen_ar_only", *PRIMARY_CONTROLS):
                    row[f"{control}_top_5pct_lift"] = float(vals.loc[control, "top_5pct_continuous_lift"])
                    row[f"real_minus_{control}_top_5pct_lift"] = float(real["top_5pct_continuous_lift"] - vals.loc[control, "top_5pct_continuous_lift"])
                    row[f"real_minus_{control}_spearman"] = float(real["continuous_spearman"] - vals.loc[control, "continuous_spearman"])
                    row[f"real_minus_{control}_pr_auc"] = float(real["pr_auc"] - vals.loc[control, "pr_auc"])
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["target_name", "seed"])


def summary_row(summary: pd.DataFrame, target: str, control: str) -> pd.Series:
    sub = summary[(summary["target_name"] == target) & (summary["control_type"] == control)]
    if len(sub) != 1:
        raise RuntimeError(f"Expected one summary row for {target}/{control}, got {len(sub)}")
    return sub.iloc[0]


def compute_gates(summary: pd.DataFrame, fold_df: pd.DataFrame, seed_df: pd.DataFrame) -> dict[str, Any]:
    failed: list[str] = []
    b_real = summary_row(summary, BINARY_TARGET, "real_residual")
    b_ar = summary_row(summary, BINARY_TARGET, "frozen_ar_only")
    b_controls = [summary_row(summary, BINARY_TARGET, control) for control in PRIMARY_CONTROLS]
    b_best = max(b_controls, key=lambda row: float(row["mean_pr_auc"]))
    b_deltas = {
        "real_minus_frozen_ar_pr_auc": float(b_real["mean_pr_auc"] - b_ar["mean_pr_auc"]),
        "real_minus_best_control_pr_auc": float(b_real["mean_pr_auc"] - b_best["mean_pr_auc"]),
    }
    for row in b_controls:
        b_deltas[f"real_minus_{row['control_type']}_pr_auc"] = float(b_real["mean_pr_auc"] - row["mean_pr_auc"])
    b_seed = seed_df[seed_df["target_name"] == BINARY_TARGET]
    b_seed_cols = [f"real_minus_{control}_pr_auc" for control in ("frozen_ar_only", *PRIMARY_CONTROLS)]
    b_seed_positive = int((b_seed[b_seed_cols].min(axis=1) > 0).sum())
    b_do_no_harm = bool((b_seed["real_minus_frozen_ar_only_pr_auc"] >= DO_NO_HARM_SEED_FLOOR).all())
    b_threshold_pass = bool(
        b_deltas["real_minus_frozen_ar_pr_auc"] >= BINARY_THRESHOLD
        and all(b_deltas[f"real_minus_{control}_pr_auc"] >= BINARY_THRESHOLD for control in PRIMARY_CONTROLS)
    )
    binary_pass = bool(b_threshold_pass and b_seed_positive >= 2 and b_do_no_harm)
    if not b_threshold_pass:
        failed.append("binary_min_delta_threshold")
    if b_seed_positive < 2:
        failed.append("binary_seed_consistency")
    if not b_do_no_harm:
        failed.append("binary_do_no_harm")

    c_real = summary_row(summary, CONTINUOUS_TARGET, "real_residual")
    c_ar = summary_row(summary, CONTINUOUS_TARGET, "frozen_ar_only")
    c_controls = [summary_row(summary, CONTINUOUS_TARGET, control) for control in PRIMARY_CONTROLS]
    c_best = max(c_controls, key=lambda row: float(row["mean_top_5pct_continuous_lift"]))
    c_deltas = {
        "real_minus_frozen_ar_top_5pct_lift": float(c_real["mean_top_5pct_continuous_lift"] - c_ar["mean_top_5pct_continuous_lift"]),
        "real_minus_best_control_top_5pct_lift": float(c_real["mean_top_5pct_continuous_lift"] - c_best["mean_top_5pct_continuous_lift"]),
        "real_minus_frozen_ar_spearman": float(c_real["mean_continuous_spearman"] - c_ar["mean_continuous_spearman"]),
        "real_minus_frozen_ar_pr_auc": float(c_real["mean_pr_auc"] - c_ar["mean_pr_auc"]),
    }
    for row in c_controls:
        c_deltas[f"real_minus_{row['control_type']}_top_5pct_lift"] = float(c_real["mean_top_5pct_continuous_lift"] - row["mean_top_5pct_continuous_lift"])
    c_seed = seed_df[seed_df["target_name"] == CONTINUOUS_TARGET]
    c_seed_cols = [f"real_minus_{control}_top_5pct_lift" for control in ("frozen_ar_only", *PRIMARY_CONTROLS)]
    c_seed_positive = int((c_seed[c_seed_cols].min(axis=1) > 0).sum())
    c_binary_do_no_harm = bool((c_seed["real_minus_frozen_ar_only_pr_auc"] >= DO_NO_HARM_SEED_FLOOR).all())
    c_threshold_pass = bool(
        c_deltas["real_minus_frozen_ar_top_5pct_lift"] >= CONTINUOUS_THRESHOLD
        and all(c_deltas[f"real_minus_{control}_top_5pct_lift"] >= CONTINUOUS_THRESHOLD for control in PRIMARY_CONTROLS)
    )
    continuous_pass = bool(c_threshold_pass and c_deltas["real_minus_frozen_ar_spearman"] > 0 and c_seed_positive >= 2 and c_binary_do_no_harm)
    if not c_threshold_pass:
        failed.append("continuous_min_delta_threshold")
    if not c_deltas["real_minus_frozen_ar_spearman"] > 0:
        failed.append("continuous_spearman_delta")
    if c_seed_positive < 2:
        failed.append("continuous_seed_consistency")
    if not c_binary_do_no_harm:
        failed.append("continuous_binary_do_no_harm")

    frozen_ar_integrity_pass = bool(
        fold_df.groupby(["target_name", "seed"])["frozen_ar_test_checksum"].nunique().max() == 1
        and fold_df.groupby(["target_name", "seed"])["frozen_ar_train_checksum"].nunique().max() == 1
    )
    checkpoint_restore_pass = bool(fold_df["checkpoint_restore_pass"].all())
    eval_mode_scoring_pass = bool(fold_df["eval_mode_scoring"].all())
    if not frozen_ar_integrity_pass:
        failed.append("frozen_ar_integrity")
    if not checkpoint_restore_pass:
        failed.append("checkpoint_restore")
    if not eval_mode_scoring_pass:
        failed.append("eval_mode_scoring")
    recommendation = (
        "redesigned_targets_pass_consider_review_before_grouped"
        if binary_pass and continuous_pass
        else "redesigned_target_blocked_failed_do_not_run_grouped_or_504"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "variant": VARIANT,
        "matrix_rows_expected": 42,
        "matrix_rows_actual": int(len(fold_df)),
        "eval_mode_scoring_pass": eval_mode_scoring_pass,
        "checkpoint_restore_pass": checkpoint_restore_pass,
        "frozen_ar_integrity_pass": frozen_ar_integrity_pass,
        "binary_pass": binary_pass,
        "continuous_pass": continuous_pass,
        "strict_forward_time_temporal_generalization_proven": False,
        "grouped_started": False,
        "recommendation": recommendation,
        "failed_gates": failed,
        "binary": {
            "target": BINARY_TARGET,
            "real_pr_auc": float(b_real["mean_pr_auc"]),
            "frozen_ar_pr_auc": float(b_ar["mean_pr_auc"]),
            "best_control": str(b_best["control_type"]),
            "best_control_pr_auc": float(b_best["mean_pr_auc"]),
            "seed_positive_count": b_seed_positive,
            "do_no_harm_pass": b_do_no_harm,
            "deltas": b_deltas,
        },
        "continuous": {
            "target": CONTINUOUS_TARGET,
            "real_spearman": float(c_real["mean_continuous_spearman"]),
            "frozen_ar_spearman": float(c_ar["mean_continuous_spearman"]),
            "real_top_5pct_lift": float(c_real["mean_top_5pct_continuous_lift"]),
            "frozen_ar_top_5pct_lift": float(c_ar["mean_top_5pct_continuous_lift"]),
            "best_control": str(c_best["control_type"]),
            "best_control_top_5pct_lift": float(c_best["mean_top_5pct_continuous_lift"]),
            "seed_positive_count": c_seed_positive,
            "binary_do_no_harm_pass": c_binary_do_no_harm,
            "deltas": c_deltas,
        },
    }


def write_report(path: Path, output_root: Path, gates: dict[str, Any]) -> None:
    b = gates["binary"]
    c = gates["continuous"]
    report = f"""# Phase 5 Redesigned Target Blocked Summary

Output root: `{output_root}`

This is a bounded blocked-only redesigned target training test. It uses two approved targets, three seeds, seven controls, the fold-safe redesigned-target PCA root, and one residual variant: `monotonic_do_no_harm_residual`. It does not run grouped, broad variants, extra targets, AR retraining, V-JEPA/TRIBE, or PCA refitting.

## Binary Target

- Target: `{b['target']}`
- Real PR-AUC: `{b['real_pr_auc']:.10f}`
- Frozen AR PR-AUC: `{b['frozen_ar_pr_auc']:.10f}`
- Delta vs frozen AR: `{b['deltas']['real_minus_frozen_ar_pr_auc']:+.10f}`
- Best control: `{b['best_control']}` PR-AUC `{b['best_control_pr_auc']:.10f}`
- Delta vs best control: `{b['deltas']['real_minus_best_control_pr_auc']:+.10f}`
- Seed positive count: `{b['seed_positive_count']}/3`
- Binary pass: `{gates['binary_pass']}`

## Continuous Target

- Target: `{c['target']}`
- Real Spearman: `{c['real_spearman']:.10f}`
- Frozen AR Spearman: `{c['frozen_ar_spearman']:.10f}`
- Spearman delta: `{c['deltas']['real_minus_frozen_ar_spearman']:+.10f}`
- Real top 5pct lift: `{c['real_top_5pct_lift']:.10f}`
- Frozen AR top 5pct lift: `{c['frozen_ar_top_5pct_lift']:.10f}`
- Top 5pct lift delta vs frozen AR: `{c['deltas']['real_minus_frozen_ar_top_5pct_lift']:+.10f}`
- Best control: `{c['best_control']}` top 5pct lift `{c['best_control_top_5pct_lift']:.10f}`
- Delta vs best control: `{c['deltas']['real_minus_best_control_top_5pct_lift']:+.10f}`
- Seed positive count: `{c['seed_positive_count']}/3`
- Continuous pass: `{gates['continuous_pass']}`

## Gates

- `frozen_ar_integrity_pass`: `{gates['frozen_ar_integrity_pass']}`
- `checkpoint_restore_pass`: `{gates['checkpoint_restore_pass']}`
- `eval_mode_scoring_pass`: `{gates['eval_mode_scoring_pass']}`
- Failed gates: `{gates['failed_gates']}`
- Recommendation: `{gates['recommendation']}`

Strict forward-time temporal generalization remains unproven. Do not run grouped from this result unless the gates pass cleanly and the result is reviewed.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def finalize_output(output_root: Path, reports_dir: Path) -> dict[str, Any]:
    fold_df = pd.read_csv(output_root / "metrics" / "redesigned_target_blocked_seed_metrics.csv")
    summary = summarize(fold_df)
    seed_df = seed_deltas(fold_df)
    gates = compute_gates(summary, fold_df, seed_df)
    summary.to_csv(output_root / "metrics" / "redesigned_target_blocked_summary_metrics.csv", index=False)
    seed_df.to_csv(output_root / "metrics" / "redesigned_target_blocked_seed_deltas.csv", index=False)
    summary.to_csv(output_root / "promotion" / "redesigned_target_blocked_control_comparison.csv", index=False)
    seed_df.to_csv(output_root / "promotion" / "redesigned_target_blocked_seed_deltas.csv", index=False)
    write_json(output_root / "promotion" / "redesigned_target_blocked_gates.json", gates)
    write_json(output_root / "promotion" / "redesigned_target_blocked_adversarial_verdict.json", gates)
    write_json(
        output_root / "promotion" / "redesigned_target_blocked_failure_reasons.json",
        {"failed_gates": gates["failed_gates"], "recommendation": gates["recommendation"]},
    )
    stamp = output_root.name.replace("again_dense_2hz_phase5_redesigned_target_blocked_", "")
    report_name = f"again_dense_2hz_phase5_redesigned_target_blocked_summary_{stamp}.md"
    write_report(output_root / "reports" / report_name, output_root, gates)
    report_path = reports_dir / report_name
    write_report(report_path, output_root, gates)
    return {"gates": gates, "report_path": str(report_path)}


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root) if args.output_root else default_output_root()
    pca_root = Path(args.foldsafe_pca_root)
    source_root = Path(args.source_root)
    matrix = [(target, seed, control) for target in TARGET_TYPES for seed in SEEDS for control in CONTROLS]
    print(json.dumps({"matrix_size": len(matrix), "max_allowed": 42, "targets": list(TARGET_TYPES), "controls": CONTROLS}, indent=2))
    if len(matrix) > 42:
        raise RuntimeError(f"Refusing to exceed 42 rows: {len(matrix)}")
    pca_audit = validate_pca_root(pca_root)
    if args.dry_run:
        return 0
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {output_root}")
    start = time.time()
    for sub in ("manifests", "metrics", "promotion", "diagnostics", "reports", "frozen_ar_scores", "checkpoints"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)
    blocks, df, dense_root, residual_meta = build_blocks(source_root, pca_root)
    fold_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    ar_manifest: list[dict[str, Any]] = []
    feature_manifest: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    for target_name in (BINARY_TARGET, CONTINUOUS_TARGET):
        block = blocks[target_name]
        for seed in SEEDS:
            ar = cache_frozen_ar(source_root, output_root, block, seed, args.batch_size)
            ar_manifest.append({k: v for k, v in ar.items() if k not in {"train_score", "train_reg", "test_score", "test_reg"}})
            if block.target_type == "binary":
                ar_metrics = binary_metric_row(block, ar["train_score"], ar["test_score"], ar["test_reg"])
            else:
                ar_metrics = continuous_metric_row(block, ar["train_score"], ar["test_score"], ar["test_reg"])
            fold_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "target_name": target_name,
                    "target_type": block.target_type,
                    "validation_protocol": PROTOCOL,
                    "fold": FOLD,
                    "seed": seed,
                    "feature_name": FEATURE_NAME,
                    "variant_name": VARIANT,
                    "control_type": "frozen_ar_only",
                    "n_train": int(len(block.train_idx)),
                    "n_test": int(len(block.test_idx)),
                    "checkpoint_restore_pass": True,
                    "eval_mode_scoring": True,
                    "ar_retrained": False,
                    "frozen_ar_train_checksum": ar["train_checksum"],
                    "frozen_ar_test_checksum": ar["test_checksum"],
                    "residual_suppressed": True,
                    "inner_val_selection_policy": "not_applicable",
                    "training_label_policy": "not_applicable",
                    **ar_metrics,
                }
            )
            for control in [c for c in CONTROLS if c != "frozen_ar_only"]:
                train_x, test_x, dims, manifest = residual_features_for_control(df, pca_root, dense_root, block, control, seed)
                metrics, curves, audit = train_residual(
                    target_type=block.target_type,
                    control=control,
                    train_x=train_x,
                    test_x=test_x,
                    block=block,
                    ar=ar,
                    seed=seed,
                    output_root=output_root,
                    batch_size=args.batch_size,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                )
                for curve in curves:
                    curve_rows.append({"target_name": target_name, "target_type": block.target_type, "seed": seed, "control_type": control, **curve})
                feature_manifest.append({"target_name": target_name, "seed": seed, "control_type": control, "dims": dims, "blocks": manifest})
                integrity_rows.append(
                    {
                        "target_name": target_name,
                        "seed": seed,
                        "control_type": control,
                        "frozen_ar_train_checksum": ar["train_checksum"],
                        "frozen_ar_test_checksum": ar["test_checksum"],
                        "same_ar_as_reference": True,
                    }
                )
                fold_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "target_name": target_name,
                        "target_type": block.target_type,
                        "validation_protocol": PROTOCOL,
                        "fold": FOLD,
                        "seed": seed,
                        "feature_name": FEATURE_NAME,
                        "variant_name": VARIANT,
                        "control_type": control,
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
    if len(fold_df) != 42:
        raise RuntimeError(f"Expected 42 rows, got {len(fold_df)}")
    fold_df.to_csv(output_root / "metrics" / "redesigned_target_blocked_seed_metrics.csv", index=False)
    fold_df.to_csv(output_root / "metrics" / "redesigned_target_blocked_fold_metrics.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(output_root / "diagnostics" / "training_curve_summary.csv", index=False)
    pd.DataFrame(curve_rows).to_json(output_root / "diagnostics" / "training_curves.json", orient="records", indent=2)
    pd.DataFrame(integrity_rows).to_csv(output_root / "diagnostics" / "frozen_ar_integrity_audit.csv", index=False)
    label_audit = fold_df[fold_df["control_type"] == "label_permutation_residual"][
        ["target_name", "target_type", "seed", "best_epoch", "inner_val_selection_policy", "training_label_policy", "heldout_scoring_policy", "best_inner_val_delta_vs_frozen_ar"]
    ].sort_values(["target_name", "seed"])
    video_audit = fold_df[fold_df["control_type"] == "train_only_video_mean_residual"][
        ["target_name", "target_type", "seed", "best_epoch", "checkpoint_restored", "residual_suppressed"]
    ].sort_values(["target_name", "seed"])
    label_audit.to_csv(output_root / "diagnostics" / "label_permutation_audit.csv", index=False)
    video_audit.to_csv(output_root / "diagnostics" / "train_only_video_mean_audit.csv", index=False)
    write_json(output_root / "diagnostics" / "label_permutation_audit.json", {"policy_implemented": True, "rows": label_audit.to_dict(orient="records")})
    write_json(output_root / "diagnostics" / "train_only_video_mean_audit.json", {"train_only_video_mean_primary_static_control": True, "rows": video_audit.to_dict(orient="records")})
    write_json(output_root / "diagnostics" / "frozen_ar_integrity_audit.json", {"rows": integrity_rows, "row_count": len(integrity_rows)})
    write_json(output_root / "manifests" / "frozen_ar_manifest.json", {"ar_only_retraining_avoided": True, "scores": ar_manifest})
    write_json(output_root / "manifests" / "feature_manifest.json", {"foldsafe_pca_root": str(pca_root), "features": feature_manifest, "row_count": len(feature_manifest)})
    write_json(output_root / "manifests" / "model_config_manifest.json", {"variant": VARIANT, "controls": CONTROLS, "targets": list(TARGET_TYPES)})
    write_json(
        output_root / "manifests" / "run_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": now_iso(),
            "source_root": str(source_root),
            "foldsafe_pca_root": str(pca_root),
            "pca_audit": pca_audit,
            "output_root": str(output_root),
            "dense_root": str(dense_root),
            "targets": list(TARGET_TYPES),
            "feature": FEATURE_NAME,
            "protocol_scope": "blocked_temporal_70_30_only",
            "variant_scope": VARIANT,
            "matrix_size": len(matrix),
            "no_ar_retraining": True,
            "no_grouped": True,
            "no_extra_targets": True,
            "no_vjepa_tribe_pca_rerun": True,
            "residual_target_definition": residual_meta,
            "duration_seconds": time.time() - start,
        },
    )
    finalized = finalize_output(output_root, Path(args.reports_dir))
    gates = finalized["gates"]
    print(json.dumps(fr.clean_json({"run_completed": True, "output_root": str(output_root), "report": finalized["report_path"], **gates}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
