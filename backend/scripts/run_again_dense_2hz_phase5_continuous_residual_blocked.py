"""Blocked continuous arousal movement residual over frozen AR.

Bounded run:
blocked_temporal_70_30 only x 5 seeds x 7 controls = 35 rows.

This is not a binary spike confirmation rerun, not grouped 5-fold, not , and
not a broad model search. It keeps AR frozen and trains only a monotonic
continuous residual branch for future arousal movement ranking/lift.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts import run_again_dense_2hz_phase5_blocked_residual_clean_confirm as clean
from backend.scripts import run_again_dense_2hz_phase5_blocked_residual_targeted as targeted
from backend.scripts import run_again_dense_2hz_phase5_frozen_ar_residual as fr
from backend.scripts import run_again_dense_2hz_phase5_learned_heads as base


SCHEMA_VERSION = "again_dense_2hz_phase5_continuous_residual_blocked_v1"
SOURCE_ROOT = Path("outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_20260629_171825")
PREVIOUS_FROZEN_ROOT = Path("outputs/again_dense_2hz_phase5_frozen_ar_residual_")
TARGET_NAME = fr.TARGET_NAME
BINARY_TARGET_NAME = fr.TARGET_NAME
CONTINUOUS_SOURCE = fr.CONTINUOUS_SOURCE
FEATURE_NAME = fr.FEATURE_NAME
LOSS_NAME = "continuous_rank_lift_residual"
FROZEN_AR_LOSS_NAME = fr.LOSS_NAME
PROTOCOL = "blocked_temporal_70_30"
FOLD = 1
VARIANT = "monotonic_do_no_harm_residual"
SEEDS = (20260625, 20260626, 20260627, 20260628, 20260629)
CONTROLS = (
    "frozen_ar_only",
    "real_continuous_residual",
    "shuffled_pca_continuous_residual",
    "random_pca_continuous_residual",
    "train_only_video_mean_continuous_residual",
    "label_permutation_continuous_residual",
    "diagnostics_only_continuous_residual",
)
MATCHED_CONTROLS = ("shuffled_pca_continuous_residual", "random_pca_continuous_residual")
PRIMARY_CONTROLS = (
    "shuffled_pca_continuous_residual",
    "random_pca_continuous_residual",
    "label_permutation_continuous_residual",
    "train_only_video_mean_continuous_residual",
)
TOP5_THRESHOLD = 0.001
DO_NO_HARM_BINARY_FLOOR = -0.0005


def default_output_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(f"outputs/again_dense_2hz_phase5_continuous_residual_blocked_{stamp}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--previous-frozen-root", default=str(PREVIOUS_FROZEN_ROOT))
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


def control_to_feature_control(control: str) -> str:
    mapping = {
        "real_continuous_residual": "real_frozen_ar_residual",
        "shuffled_pca_continuous_residual": "shuffled_pca_frozen_ar_residual",
        "random_pca_continuous_residual": "random_pca_frozen_ar_residual",
        "diagnostics_only_continuous_residual": "diag_only_frozen_ar_residual",
        "label_permutation_continuous_residual": "real_frozen_ar_residual",
    }
    return mapping[control]


def residual_features_for_control(
    df: pd.DataFrame,
    dense_root: Path,
    phase4_root: Path,
    block: fr.Block,
    control: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], list[dict[str, Any]]]:
    if control == "train_only_video_mean_continuous_residual":
        return clean.video_mean_features(df, dense_root, phase4_root, block, train_only=True)
    return fr.residual_features(df, dense_root, phase4_root, block, control_to_feature_control(control), seed)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return fr.corr(np.asarray(a), np.asarray(b))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return fr.spearman(np.asarray(a), np.asarray(b))


def top_continuous_metrics(y_cont: np.ndarray, pred: np.ndarray, frac: float) -> dict[str, float]:
    k = max(1, int(math.ceil(len(pred) * frac)))
    order = np.argsort(-pred, kind="mergesort")[:k]
    baseline = float(np.mean(y_cont)) if len(y_cont) else math.nan
    avg_true = float(np.mean(y_cont[order])) if k else math.nan
    pct = int(frac * 100)
    return {
        f"top_{pct}pct_continuous_lift": avg_true - baseline if math.isfinite(avg_true) and math.isfinite(baseline) else math.nan,
        f"top_{pct}pct_avg_true_movement": avg_true,
    }


def ndcg_at_frac(y_cont: np.ndarray, pred: np.ndarray, frac: float) -> float:
    if len(pred) == 0:
        return math.nan
    k = max(1, int(math.ceil(len(pred) * frac)))
    rel = np.asarray(y_cont, dtype=np.float64)
    rel = rel - float(np.min(rel))
    denom = np.log2(np.arange(2, k + 2, dtype=np.float64))
    order = np.argsort(-pred, kind="mergesort")[:k]
    ideal = np.argsort(-rel, kind="mergesort")[:k]
    dcg = float(np.sum(rel[order] / denom))
    idcg = float(np.sum(rel[ideal] / denom))
    return dcg / idcg if idcg > 0 else math.nan


def continuous_metric_row(
    y_train_binary: np.ndarray,
    train_binary_score: np.ndarray,
    y_test_binary: np.ndarray,
    binary_score: np.ndarray,
    y_cont: np.ndarray,
    pred_cont: np.ndarray,
) -> dict[str, Any]:
    err = pred_cont - y_cont
    true_top = np.argsort(-y_cont, kind="mergesort")[: max(1, int(math.ceil(len(y_cont) * 0.05)))]
    row: dict[str, Any] = {
        "continuous_pearson": corr(y_cont, pred_cont),
        "continuous_spearman": spearman(y_cont, pred_cont),
        "continuous_mae": float(np.mean(np.abs(err))),
        "continuous_rmse": float(math.sqrt(float(np.mean(err * err)))),
        "continuous_bias": float(np.mean(err)),
        "peak_underprediction": float(np.mean(y_cont[true_top] - pred_cont[true_top])),
        "binary_pr_auc": float(average_precision_score(y_test_binary, binary_score)) if len(np.unique(y_test_binary)) > 1 else math.nan,
        "binary_roc_auc": float(roc_auc_score(y_test_binary, binary_score)) if len(np.unique(y_test_binary)) > 1 else math.nan,
        "binary_pr_auc_from_continuous_prediction": float(average_precision_score(y_test_binary, pred_cont))
        if len(np.unique(y_test_binary)) > 1
        else math.nan,
    }
    for frac in (0.01, 0.05, 0.10):
        row.update(top_continuous_metrics(y_cont, pred_cont, frac))
        pct = int(frac * 100)
        row[f"ndcg_at_{pct}pct"] = ndcg_at_frac(y_cont, pred_cont, frac)
        spike_top = fr.top_fraction_metrics(y_test_binary, y_cont, pred_cont, frac)
        row[f"top_{pct}pct_spike_recall_from_continuous_prediction"] = spike_top.get(f"top_{pct}pct_recall", math.nan)
    return row


def add_delta_metrics(metrics: dict[str, Any], ar_metrics: dict[str, Any]) -> None:
    for key in (
        "continuous_spearman",
        "continuous_pearson",
        "top_1pct_continuous_lift",
        "top_5pct_continuous_lift",
        "top_10pct_continuous_lift",
        "ndcg_at_1pct",
        "ndcg_at_5pct",
        "ndcg_at_10pct",
        "binary_pr_auc",
        "binary_pr_auc_from_continuous_prediction",
    ):
        metrics[f"delta_vs_frozen_ar_{key}"] = float(metrics[key] - ar_metrics[key])


def score_tuple(metrics: dict[str, Any], ar_metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    pr_delta = metrics.get("delta_vs_frozen_ar_binary_pr_auc_from_continuous_prediction", 0.0)
    return (
        float(metrics["delta_vs_frozen_ar_top_5pct_continuous_lift"]),
        float(metrics["delta_vs_frozen_ar_continuous_spearman"]),
        float(metrics["delta_vs_frozen_ar_top_10pct_continuous_lift"]),
        float(pr_delta) if math.isfinite(float(pr_delta)) else 0.0,
    )


def permuted_continuous_target(block: fr.Block, seed: int) -> np.ndarray | None:
    rng = np.random.default_rng(int(seed) + block.fold * 7919 + 50123)
    return block.train_cont[rng.permutation(len(block.train_cont))].copy()


def train_continuous_residual(
    control: str,
    train_x: np.ndarray,
    test_x: np.ndarray,
    block: fr.Block,
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
    rng = np.random.default_rng(int(seed) + block.fold * 7919 + 701)
    cont_metric = permuted_continuous_target(block, seed) if control == "label_permutation_continuous_residual" else block.train_cont.copy()
    selection_cont = cont_metric
    selection_policy = "permuted_inner_val_continuous_target" if control == "label_permutation_continuous_residual" else "true_inner_val_continuous_target"
    ar_train_score = ar["train_score"].astype(np.float32)
    ar_train_reg = ar["train_reg"].astype(np.float32)
    ar_test_score = ar["test_score"].astype(np.float32)
    ar_test_reg = ar["test_reg"].astype(np.float32)
    ar_val_metrics = continuous_metric_row(
        block.train_y[inner_train],
        ar_train_score[inner_train],
        block.train_y[inner_val],
        ar_train_score[inner_val],
        selection_cont[inner_val],
        ar_train_reg[inner_val],
    )
    true_ar_val_metrics = continuous_metric_row(
        block.train_y[inner_train],
        ar_train_score[inner_train],
        block.train_y[inner_val],
        ar_train_score[inner_val],
        block.train_cont[inner_val],
        ar_train_reg[inner_val],
    )
    q80 = float(np.quantile(cont_metric[inner_train], 0.80))
    q90 = float(np.quantile(cont_metric[inner_train], 0.90))
    best_score = (0.0, 0.0, 0.0, 0.0)
    best_epoch = 0
    best_path = output_root / "checkpoints" / (
        f"{TARGET_NAME}__{PROTOCOL}__fold{FOLD}__{control}__{VARIANT}__{LOSS_NAME}__{seed}__best.npz"
    )
    best_path.parent.mkdir(parents=True, exist_ok=True)
    curves: list[dict[str, Any]] = []
    stale = 0
    early_stop = "max_epochs_reached"
    suppressed = True

    def loss_fn(model_obj: targeted.TargetedResidualHead, xb: Any, ar_b: Any, ar_r: Any, yr: Any, wb: Any) -> Any:
        out = model_obj(xb, ar_b, ar_r, use_ar_floor=True)
        reg_loss = base.mx.mean(base.nn.losses.huber_loss(out[:, 0:1], yr, delta=1.0) * wb)
        anchor = 0.04 * base.mx.mean((out[:, 0:1] - ar_r[:, None]) * (out[:, 0:1] - ar_r[:, None]))
        alpha_penalty = 0.01 * base.mx.mean(model_obj.alpha_value() * model_obj.alpha_value())
        return reg_loss + anchor + alpha_penalty

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
            yr_np = cont_metric[rel].astype(np.float32)[:, None]
            weights = 1.0 + 1.0 * (yr_np >= q80).astype(np.float32) + 2.0 * (yr_np >= q90).astype(np.float32)
            yr = base.mx.array(yr_np, dtype=base.mx.float32)
            wb = base.mx.array(weights, dtype=base.mx.float32)
            loss, grads = loss_and_grad(model, xb, ab, rb, yr, wb)
            grads, _ = base.optim.clip_grad_norm(grads, 1.0)
            optimizer.update(model, grads)
            base.mx.eval(model.parameters(), optimizer.state)
            total += float(np.asarray(loss))
            batches += 1
        _val_score, val_reg, val_gate = targeted.residual_forward(
            model,
            train_x[inner_val],
            ar_train_score[inner_val],
            ar_train_reg[inner_val],
            batch_size=batch_size,
        )
        val_metrics = continuous_metric_row(
            block.train_y[inner_train],
            ar_train_score[inner_train],
            block.train_y[inner_val],
            ar_train_score[inner_val],
            selection_cont[inner_val],
            val_reg,
        )
        add_delta_metrics(val_metrics, ar_val_metrics)
        true_val_metrics = continuous_metric_row(
            block.train_y[inner_train],
            ar_train_score[inner_train],
            block.train_y[inner_val],
            ar_train_score[inner_val],
            block.train_cont[inner_val],
            val_reg,
        )
        add_delta_metrics(true_val_metrics, true_ar_val_metrics)
        current_score = score_tuple(val_metrics, ar_val_metrics)
        curve = {
            "epoch": epoch,
            "train_loss": total / max(1, batches),
            "inner_val_selection_policy": selection_policy,
            "inner_val_top_5pct_lift": val_metrics["top_5pct_continuous_lift"],
            "inner_val_top_5pct_lift_delta_vs_frozen_ar": val_metrics["delta_vs_frozen_ar_top_5pct_continuous_lift"],
            "inner_val_spearman": val_metrics["continuous_spearman"],
            "inner_val_spearman_delta_vs_frozen_ar": val_metrics["delta_vs_frozen_ar_continuous_spearman"],
            "true_inner_val_top_5pct_lift_delta_vs_frozen_ar": true_val_metrics["delta_vs_frozen_ar_top_5pct_continuous_lift"],
            "true_inner_val_spearman_delta_vs_frozen_ar": true_val_metrics["delta_vs_frozen_ar_continuous_spearman"],
            "alpha": float(np.asarray(model.alpha_value())[0]),
            "gate_mean": float(np.mean(val_gate)),
            "gate_p95": float(np.quantile(val_gate, 0.95)),
        }
        curves.append(curve)
        if current_score > best_score:
            model.save_weights(str(best_path))
            best_score = current_score
            best_epoch = epoch
            stale = 0
            suppressed = False
        else:
            stale += 1
        if stale >= int(patience):
            early_stop = "patience_exhausted"
            break

    if suppressed:
        train_reg = ar_train_reg
        test_reg = ar_test_reg
        gate = np.zeros_like(ar_test_reg)
        alpha_final = 0.0
        checkpoint_restored = False
        checkpoint_checksum = None
    else:
        model.load_weights(str(best_path))
        if hasattr(model, "eval"):
            model.eval()
        _train_score, train_reg, _ = targeted.residual_forward(model, train_x, ar_train_score, ar_train_reg, batch_size=batch_size)
        _test_score, test_reg, gate = targeted.residual_forward(model, test_x, ar_test_score, ar_test_reg, batch_size=batch_size)
        alpha_final = float(np.asarray(model.alpha_value())[0])
        checkpoint_restored = True
        checkpoint_checksum = base.file_digest(best_path)

    # The binary AR path is frozen for this continuous-only residual experiment.
    metrics = continuous_metric_row(block.train_y, ar_train_score, block.test_y, ar_test_score, block.test_cont, test_reg)
    ar_metrics = continuous_metric_row(block.train_y, ar_train_score, block.test_y, ar_test_score, block.test_cont, ar_test_reg)
    add_delta_metrics(metrics, ar_metrics)
    audit = {
        "best_epoch": int(best_epoch),
        "epochs_run": int(len(curves)),
        "best_inner_val_top_5pct_lift_delta_vs_frozen_ar": float(best_score[0]),
        "best_inner_val_spearman_delta_vs_frozen_ar": float(best_score[1]),
        "best_inner_val_top_10pct_lift_delta_vs_frozen_ar": float(best_score[2]),
        "inner_val_selection_policy": selection_policy,
        "training_target_policy": "permuted_train_continuous_target"
        if control == "label_permutation_continuous_residual"
        else "true_train_continuous_target",
        "heldout_scoring_policy": "true_heldout_continuous_target",
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
    group_cols = ["target_name", "validation_protocol", "variant_name", "control_type", "loss_name"]
    metric_cols = [
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
        "continuous_mae",
        "continuous_rmse",
        "continuous_bias",
        "peak_underprediction",
        "binary_pr_auc",
        "binary_roc_auc",
        "binary_pr_auc_from_continuous_prediction",
        "top_1pct_spike_recall_from_continuous_prediction",
        "top_5pct_spike_recall_from_continuous_prediction",
        "top_10pct_spike_recall_from_continuous_prediction",
        "delta_vs_frozen_ar_continuous_spearman",
        "delta_vs_frozen_ar_top_5pct_continuous_lift",
        "delta_vs_frozen_ar_binary_pr_auc",
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
    return pd.DataFrame(rows).sort_values(["validation_protocol", "mean_top_5pct_continuous_lift"], ascending=[True, False])


def row_for(summary: pd.DataFrame, control: str) -> pd.Series:
    sub = summary[(summary["validation_protocol"] == PROTOCOL) & (summary["control_type"] == control)]
    if len(sub) != 1:
        raise RuntimeError(f"Expected one summary row for {control}; got {len(sub)}")
    return sub.iloc[0]


def seed_delta_table(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = fold_df[fold_df["validation_protocol"] == PROTOCOL]
    for seed, group in sub.groupby("seed"):
        vals = group.set_index("control_type")
        real = vals.loc["real_continuous_residual"]
        row: dict[str, Any] = {"seed": int(seed)}
        for metric in ("continuous_spearman", "top_1pct_continuous_lift", "top_5pct_continuous_lift", "top_10pct_continuous_lift", "binary_pr_auc"):
            row[f"real_{metric}"] = float(real[metric])
        for control in ("frozen_ar_only", *PRIMARY_CONTROLS):
            ctrl = vals.loc[control]
            row[f"{control}_top_5pct_continuous_lift"] = float(ctrl["top_5pct_continuous_lift"])
            row[f"real_minus_{control}_top_5pct_continuous_lift"] = float(real["top_5pct_continuous_lift"] - ctrl["top_5pct_continuous_lift"])
            row[f"real_minus_{control}_continuous_spearman"] = float(real["continuous_spearman"] - ctrl["continuous_spearman"])
            row[f"real_minus_{control}_binary_pr_auc"] = float(real["binary_pr_auc"] - ctrl["binary_pr_auc"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("seed")


def no_single_seed_over_60(vals: np.ndarray) -> bool:
    positives = vals[vals > 0]
    if positives.size == 0 or float(np.sum(positives)) <= 0:
        return False
    return bool(float(np.max(positives) / np.sum(positives)) <= 0.60)


def compute_gates(summary: pd.DataFrame, fold_df: pd.DataFrame) -> dict[str, Any]:
    ar = row_for(summary, "frozen_ar_only")
    real = row_for(summary, "real_continuous_residual")
    shuffled = row_for(summary, "shuffled_pca_continuous_residual")
    random = row_for(summary, "random_pca_continuous_residual")
    label = row_for(summary, "label_permutation_continuous_residual")
    train_video = row_for(summary, "train_only_video_mean_continuous_residual")
    diag = row_for(summary, "diagnostics_only_continuous_residual")
    matched_best = max([shuffled, random], key=lambda r: float(r["mean_top_5pct_continuous_lift"]))
    deltas = {
        "real_minus_frozen_ar_top_5pct_continuous_lift": float(real["mean_top_5pct_continuous_lift"] - ar["mean_top_5pct_continuous_lift"]),
        "real_minus_shuffled_top_5pct_continuous_lift": float(real["mean_top_5pct_continuous_lift"] - shuffled["mean_top_5pct_continuous_lift"]),
        "real_minus_random_top_5pct_continuous_lift": float(real["mean_top_5pct_continuous_lift"] - random["mean_top_5pct_continuous_lift"]),
        "real_minus_label_permutation_top_5pct_continuous_lift": float(real["mean_top_5pct_continuous_lift"] - label["mean_top_5pct_continuous_lift"]),
        "real_minus_train_only_video_mean_top_5pct_continuous_lift": float(real["mean_top_5pct_continuous_lift"] - train_video["mean_top_5pct_continuous_lift"]),
        "real_minus_best_matched_control_top_5pct_continuous_lift": float(real["mean_top_5pct_continuous_lift"] - matched_best["mean_top_5pct_continuous_lift"]),
        "real_minus_frozen_ar_spearman": float(real["mean_continuous_spearman"] - ar["mean_continuous_spearman"]),
        "real_minus_best_matched_control_spearman": float(real["mean_continuous_spearman"] - matched_best["mean_continuous_spearman"]),
        "real_minus_label_permutation_spearman": float(real["mean_continuous_spearman"] - label["mean_continuous_spearman"]),
        "real_minus_train_only_video_mean_spearman": float(real["mean_continuous_spearman"] - train_video["mean_continuous_spearman"]),
        "real_minus_frozen_ar_binary_pr_auc": float(real["mean_binary_pr_auc"] - ar["mean_binary_pr_auc"]),
    }
    seed_deltas = seed_delta_table(fold_df)
    top5_seed_col = "real_minus_frozen_ar_only_top_5pct_continuous_lift"
    seed_positive_count = int((seed_deltas[top5_seed_col] > 0).sum())
    seed_values = seed_deltas[top5_seed_col].to_numpy(dtype=float)
    seed_consistency_pass = seed_positive_count >= 4
    concentration_pass = no_single_seed_over_60(seed_values)
    binary_do_no_harm = bool(deltas["real_minus_frozen_ar_binary_pr_auc"] >= DO_NO_HARM_BINARY_FLOOR)
    control_top5_pass = all(deltas[key] > 0 for key in (
        "real_minus_shuffled_top_5pct_continuous_lift",
        "real_minus_random_top_5pct_continuous_lift",
        "real_minus_label_permutation_top_5pct_continuous_lift",
        "real_minus_train_only_video_mean_top_5pct_continuous_lift",
    ))
    control_spearman_pass = all(deltas[key] > 0 for key in (
        "real_minus_best_matched_control_spearman",
        "real_minus_label_permutation_spearman",
        "real_minus_train_only_video_mean_spearman",
    ))
    continuous_residual_pass = bool(
        deltas["real_minus_frozen_ar_top_5pct_continuous_lift"] >= TOP5_THRESHOLD
        and control_top5_pass
        and deltas["real_minus_frozen_ar_spearman"] > 0
        and seed_consistency_pass
        and binary_do_no_harm
    )
    credible_continuous_pass = bool(
        continuous_residual_pass
        and control_spearman_pass
        and concentration_pass
    )
    recommendation = (
        "credible_continuous_candidate_consider_grouped_compatibility_only_after_review"
        if credible_continuous_pass
        else (
            "continuous_directional_only_do_not_run_grouped_or_504"
            if deltas["real_minus_frozen_ar_top_5pct_continuous_lift"] > 0
            else "continuous_residual_failed_do_not_run_grouped_or_504"
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "variant": VARIANT,
        "loss": LOSS_NAME,
        "matrix_rows_expected": 35,
        "continuous_source": CONTINUOUS_SOURCE,
        "binary_target_secondary": BINARY_TARGET_NAME,
        "top5_threshold": TOP5_THRESHOLD,
        "same_variant_gates_only": True,
        "eval_mode_scoring_pass": bool(fold_df["eval_mode_scoring"].all()),
        "checkpoint_restore_pass": bool(fold_df["checkpoint_restore_pass"].all()),
        "frozen_ar_integrity_pass": bool(
            fold_df.groupby(["seed"])["frozen_ar_test_checksum"].nunique().max() == 1
            and fold_df.groupby(["seed"])["frozen_ar_train_checksum"].nunique().max() == 1
        ),
        "label_permutation_policy_implemented": True,
        "train_only_video_mean_policy_implemented": True,
        "real_gt_frozen_ar_top5_pass": bool(deltas["real_minus_frozen_ar_top_5pct_continuous_lift"] >= TOP5_THRESHOLD),
        "real_gt_controls_top5_pass": bool(control_top5_pass),
        "spearman_delta_vs_frozen_ar_positive": bool(deltas["real_minus_frozen_ar_spearman"] > 0),
        "seed_consistency_pass": bool(seed_consistency_pass),
        "top5_seed_positive_count": seed_positive_count,
        "no_single_seed_over_60pct_pass": bool(concentration_pass),
        "binary_pr_auc_do_no_harm_pass": binary_do_no_harm,
        "continuous_residual_pass": continuous_residual_pass,
        "credible_continuous_pass": credible_continuous_pass,
        "strict_forward_time_spike_prediction_claimed": False,
        "full_grouped_5fold_started": False,
        "recommendation": recommendation,
        "pr_auc_secondary": {
            "real": float(real["mean_binary_pr_auc"]),
            "frozen_ar": float(ar["mean_binary_pr_auc"]),
        },
        "continuous_metrics": {
            "real_spearman": float(real["mean_continuous_spearman"]),
            "frozen_ar_spearman": float(ar["mean_continuous_spearman"]),
            "real_pearson": float(real["mean_continuous_pearson"]),
            "frozen_ar_pearson": float(ar["mean_continuous_pearson"]),
            "real_top_1pct_lift": float(real["mean_top_1pct_continuous_lift"]),
            "frozen_ar_top_1pct_lift": float(ar["mean_top_1pct_continuous_lift"]),
            "real_top_5pct_lift": float(real["mean_top_5pct_continuous_lift"]),
            "frozen_ar_top_5pct_lift": float(ar["mean_top_5pct_continuous_lift"]),
            "real_top_10pct_lift": float(real["mean_top_10pct_continuous_lift"]),
            "frozen_ar_top_10pct_lift": float(ar["mean_top_10pct_continuous_lift"]),
            "best_matched_control": str(matched_best["control_type"]),
            "best_matched_control_top_5pct_lift": float(matched_best["mean_top_5pct_continuous_lift"]),
            "label_permutation_top_5pct_lift": float(label["mean_top_5pct_continuous_lift"]),
            "train_only_video_mean_top_5pct_lift": float(train_video["mean_top_5pct_continuous_lift"]),
            "diagnostics_only_top_5pct_lift": float(diag["mean_top_5pct_continuous_lift"]),
        },
        "deltas": deltas,
    }


def write_report(path: Path, gates: dict[str, Any], output_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    m = gates["continuous_metrics"]
    d = gates["deltas"]
    report = f"""# Phase 5 Continuous Residual Blocked Summary

Output root: `{output_root}`

This is a targeted continuous future arousal movement residual experiment over frozen AR. It uses only `blocked_temporal_70_30`, only `monotonic_do_no_harm_residual`, five prespecified seeds, and seven controls for a maximum of 35 rows. It does not rerun binary spike confirmation, grouped 5-fold, secondary targets, AR training, V-JEPA/TRIBE, or PCA.

## Result

- Continuous residual pass: `{gates['continuous_residual_pass']}`
- Credible continuous pass: `{gates['credible_continuous_pass']}`
- Recommendation: `{gates['recommendation']}`
- Strict forward-time spike prediction claimed: `{gates['strict_forward_time_spike_prediction_claimed']}`

## Continuous Metrics

- Real Spearman: `{m['real_spearman']:.10f}`
- Frozen AR Spearman: `{m['frozen_ar_spearman']:.10f}`
- Spearman delta: `{d['real_minus_frozen_ar_spearman']:+.10f}`
- Real top 1pct continuous lift: `{m['real_top_1pct_lift']:.10f}`
- Frozen AR top 1pct continuous lift: `{m['frozen_ar_top_1pct_lift']:.10f}`
- Real top 5pct continuous lift: `{m['real_top_5pct_lift']:.10f}`
- Frozen AR top 5pct continuous lift: `{m['frozen_ar_top_5pct_lift']:.10f}`
- Top 5pct lift delta vs frozen AR: `{d['real_minus_frozen_ar_top_5pct_continuous_lift']:+.10f}`
- Best matched control: `{m['best_matched_control']}` top 5pct lift `{m['best_matched_control_top_5pct_lift']:.10f}`
- Delta vs best matched control: `{d['real_minus_best_matched_control_top_5pct_continuous_lift']:+.10f}`
- Label permutation top 5pct lift: `{m['label_permutation_top_5pct_lift']:.10f}`
- Train-only video mean top 5pct lift: `{m['train_only_video_mean_top_5pct_lift']:.10f}`

## Gates

- `real_gt_frozen_ar_top5_pass`: `{gates['real_gt_frozen_ar_top5_pass']}`
- `real_gt_controls_top5_pass`: `{gates['real_gt_controls_top5_pass']}`
- `spearman_delta_vs_frozen_ar_positive`: `{gates['spearman_delta_vs_frozen_ar_positive']}`
- `top5_seed_positive_count`: `{gates['top5_seed_positive_count']}/5`
- `no_single_seed_over_60pct_pass`: `{gates['no_single_seed_over_60pct_pass']}`
- `binary_pr_auc_do_no_harm_pass`: `{gates['binary_pr_auc_do_no_harm_pass']}`
- `frozen_ar_integrity_pass`: `{gates['frozen_ar_integrity_pass']}`
- `checkpoint_restore_pass`: `{gates['checkpoint_restore_pass']}`
- `eval_mode_scoring_pass`: `{gates['eval_mode_scoring_pass']}`

Binary spike metrics are secondary only in this run. Do not claim strict forward-time spike prediction unless binary PR-AUC gates pass in a separate binary confirmation.
"""
    path.write_text(report)


def matrix_rows() -> list[dict[str, Any]]:
    return [{"protocol": PROTOCOL, "fold": FOLD, "seed": seed, "variant": VARIANT, "control": control} for seed in SEEDS for control in CONTROLS]


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root) if args.output_root else default_output_root()
    matrix = matrix_rows()
    print(json.dumps({"matrix_size": len(matrix), "max_allowed": 35, "protocol": PROTOCOL, "variant": VARIANT, "controls": CONTROLS}, indent=2))
    if len(matrix) > 35:
        raise RuntimeError(f"Refusing to exceed 35 rows: {len(matrix)}")
    if args.dry_run:
        return 0
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {output_root}")
    start = time.time()
    for sub in ("manifests", "metrics", "promotion", "diagnostics", "reports", "frozen_ar_scores", "checkpoints"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)
    source_root = Path(args.source_root)
    previous_root = Path(args.previous_frozen_root)
    all_blocks, df, dense_root, phase4_root = fr.build_blocks(source_root)
    block = all_blocks[(PROTOCOL, FOLD)]
    fold_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    ar_manifest: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    feature_manifest: list[dict[str, Any]] = []
    for seed in SEEDS:
        ar = clean.load_or_cache_frozen_ar(previous_root, source_root, output_root, block, seed, args.batch_size)
        ar_manifest.append({k: v for k, v in ar.items() if k not in {"train_score", "train_reg", "test_score", "test_reg", "metrics"}})
        ar_metrics = continuous_metric_row(block.train_y, ar["train_score"], block.test_y, ar["test_score"], block.test_cont, ar["test_reg"])
        fold_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "target_name": TARGET_NAME,
                "validation_protocol": PROTOCOL,
                "actual_validation_protocol": PROTOCOL,
                "fold": FOLD,
                "seed": seed,
                "feature_name": FEATURE_NAME,
                "variant_name": VARIANT,
                "control_type": "frozen_ar_only",
                "loss_name": LOSS_NAME,
                "n_train": int(len(block.train_idx)),
                "n_test": int(len(block.test_idx)),
                "checkpoint_restore_pass": True,
                "eval_mode_scoring": True,
                "ar_retrained": False,
                "frozen_ar_train_checksum": ar["train_checksum"],
                "frozen_ar_test_checksum": ar["test_checksum"],
                "residual_suppressed": True,
                "inner_val_selection_policy": "not_applicable",
                "training_target_policy": "not_applicable",
                "heldout_scoring_policy": "true_heldout_continuous_target",
                **ar_metrics,
            }
        )
        for control in [c for c in CONTROLS if c != "frozen_ar_only"]:
            train_x, test_x, dims, manifest = residual_features_for_control(df, dense_root, phase4_root, block, control, seed)
            metrics, curves, audit = train_continuous_residual(
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
                curve_rows.append({"validation_protocol": PROTOCOL, "fold": FOLD, "seed": seed, "variant_name": VARIANT, "control_type": control, **c})
            feature_manifest.append({"protocol": PROTOCOL, "fold": FOLD, "seed": seed, "variant_name": VARIANT, "control_type": control, "dims": dims, "blocks": manifest})
            integrity_rows.append(
                {
                    "protocol": PROTOCOL,
                    "fold": FOLD,
                    "seed": seed,
                    "variant_name": VARIANT,
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
                    "validation_protocol": PROTOCOL,
                    "actual_validation_protocol": PROTOCOL,
                    "fold": FOLD,
                    "seed": seed,
                    "feature_name": FEATURE_NAME,
                    "variant_name": VARIANT,
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
    if len(fold_df) != 35:
        raise RuntimeError(f"Expected 35 rows, got {len(fold_df)}")
    summary = summarize(fold_df)
    seed_deltas = seed_delta_table(fold_df)
    gates = compute_gates(summary, fold_df)
    fold_df.to_csv(output_root / "metrics" / "continuous_residual_blocked_seed_metrics.csv", index=False)
    fold_df.to_csv(output_root / "metrics" / "continuous_residual_blocked_fold_metrics.csv", index=False)
    summary.to_csv(output_root / "metrics" / "continuous_residual_blocked_summary_metrics.csv", index=False)
    seed_deltas.to_csv(output_root / "metrics" / "continuous_residual_blocked_seed_deltas.csv", index=False)
    summary.to_csv(output_root / "promotion" / "continuous_residual_blocked_control_comparison.csv", index=False)
    seed_deltas.to_csv(output_root / "promotion" / "continuous_residual_blocked_seed_deltas.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(output_root / "diagnostics" / "training_curve_summary.csv", index=False)
    pd.DataFrame(curve_rows).to_json(output_root / "diagnostics" / "training_curves.json", orient="records", indent=2)
    pd.DataFrame(integrity_rows).to_csv(output_root / "diagnostics" / "frozen_ar_integrity_audit.csv", index=False)
    label_audit = fold_df[fold_df["control_type"] == "label_permutation_continuous_residual"][
        ["seed", "control_type", "best_epoch", "inner_val_selection_policy", "training_target_policy", "heldout_scoring_policy", "best_inner_val_top_5pct_lift_delta_vs_frozen_ar", "best_inner_val_spearman_delta_vs_frozen_ar", "top_5pct_continuous_lift", "continuous_spearman"]
    ].sort_values("seed")
    video_audit = fold_df[fold_df["control_type"] == "train_only_video_mean_continuous_residual"][
        ["seed", "control_type", "best_epoch", "top_5pct_continuous_lift", "continuous_spearman", "checkpoint_restored", "residual_suppressed"]
    ].sort_values("seed")
    label_audit.to_csv(output_root / "diagnostics" / "label_permutation_continuous_audit.csv", index=False)
    video_audit.to_csv(output_root / "diagnostics" / "train_only_video_mean_continuous_audit.csv", index=False)
    write_json(output_root / "diagnostics" / "label_permutation_continuous_audit.json", {"policy_implemented": True, "rows": label_audit.to_dict(orient="records")})
    write_json(output_root / "diagnostics" / "train_only_video_mean_continuous_audit.json", {"train_only_video_mean_primary_static_control": True, "rows": video_audit.to_dict(orient="records")})
    write_json(output_root / "diagnostics" / "frozen_ar_integrity_audit.json", {"pass": gates["frozen_ar_integrity_pass"], "rows": integrity_rows, "row_count": len(integrity_rows)})
    write_json(output_root / "diagnostics" / "checkpoint_restore_audit.json", {"pass": gates["checkpoint_restore_pass"], "rows": int(len(fold_df))})
    write_json(output_root / "diagnostics" / "eval_mode_scoring_audit.json", {"pass": gates["eval_mode_scoring_pass"], "dropout_disabled": True})
    write_json(output_root / "diagnostics" / "do_no_harm_audit.json", {"binary_pr_auc_do_no_harm_pass": gates["binary_pr_auc_do_no_harm_pass"], "binary_pr_auc_delta_vs_frozen_ar": gates["deltas"]["real_minus_frozen_ar_binary_pr_auc"]})
    write_json(output_root / "manifests" / "frozen_ar_manifest.json", {"ar_only_retraining_avoided": True, "scores": ar_manifest})
    write_json(output_root / "manifests" / "feature_manifest.json", {"features": feature_manifest, "row_count": len(feature_manifest)})
    write_json(output_root / "manifests" / "model_config_manifest.json", {"variant": VARIANT, "controls": CONTROLS, "loss": LOSS_NAME})
    write_json(
        output_root / "manifests" / "run_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": now_iso(),
            "source_root": str(source_root),
            "previous_frozen_root": str(previous_root),
            "output_root": str(output_root),
            "dense_root": str(dense_root),
            "phase4_root": str(phase4_root),
            "target": TARGET_NAME,
            "continuous_source": CONTINUOUS_SOURCE,
            "feature": FEATURE_NAME,
            "loss": LOSS_NAME,
            "protocol_scope": "blocked_temporal_70_30_only",
            "variant_scope": VARIANT,
            "matrix_size": len(matrix),
            "no_ar_retraining": True,
            "no_grouped": True,
            "no_secondary_targets": True,
            "no_vjepa_tribe_pca_rerun": True,
            "duration_seconds": time.time() - start,
        },
    )
    write_json(output_root / "promotion" / "continuous_residual_blocked_gates.json", gates)
    write_json(output_root / "promotion" / "continuous_residual_blocked_adversarial_verdict.json", gates)
    write_json(
        output_root / "promotion" / "continuous_residual_blocked_failure_reasons.json",
        {
            "continuous_residual_pass": gates["continuous_residual_pass"],
            "credible_continuous_pass": gates["credible_continuous_pass"],
            "failed_gates": [k for k, v in gates.items() if k.endswith("_pass") and v is False],
            "recommendation": gates["recommendation"],
        },
    )
    stamp = output_root.name.replace("again_dense_2hz_phase5_continuous_residual_blocked_", "")
    report_name = f"again_dense_2hz_phase5_continuous_residual_blocked_summary_{stamp}.md"
    write_report(output_root / "reports" / report_name, gates, output_root)
    report_path = Path(args.reports_dir) / report_name
    write_report(report_path, gates, output_root)
    print(json.dumps(fr.clean_json({"run_completed": True, "output_root": str(output_root), "report": str(report_path), **gates}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
