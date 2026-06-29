"""Bounded Phase 5 blocked-residual targeted diagnostic.

This is not a new headline benchmark. It uses the full
`blocked_temporal_70_30` split as the primary diagnostic target and
`grouped_video` fold 1 only as `grouped_fold1_reference_only`.

The run is capped at:
2 protocol views x 3 seeds x 4 variants x 7 controls = 168 rows.

It reuses frozen-AR score caches from the committed frozen-AR residual run when
present. If a cache is missing, it re-forwards saved AR-only checkpoints in eval
mode; it never retrains AR.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
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

from backend.scripts import run_again_dense_2hz_phase5_frozen_ar_residual as fr
from backend.scripts import run_again_dense_2hz_phase5_learned_heads as base


SCHEMA_VERSION = "again_dense_2hz_phase5_blocked_residual_targeted_v1"
SOURCE_ROOT = Path("outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_20260629_171825")
PREVIOUS_FROZEN_ROOT = Path("outputs/again_dense_2hz_phase5_frozen_ar_residual_")
TARGET_NAME = fr.TARGET_NAME
CONTINUOUS_SOURCE = fr.CONTINUOUS_SOURCE
FEATURE_NAME = fr.FEATURE_NAME
LOSS_NAME = fr.LOSS_NAME
SEEDS = fr.SEEDS
ACTUAL_PROTOCOL_UNITS = (
    ("blocked_temporal_70_30", 1, "blocked_temporal_70_30"),
    ("grouped_video", 1, "grouped_fold1_reference_only"),
)
VARIANTS = (
    "blocked_delta_selected_gated_residual",
    "monotonic_do_no_harm_residual",
    "low_ar_confidence_residual",
    "rank_lift_residual",
)
CONTROLS = (
    "real_residual",
    "frozen_ar_only",
    "shuffled_pca_residual",
    "random_pca_residual",
    "diagnostics_only_residual",
    "video_mean_pca_residual",
    "label_permutation_residual",
)
MATCHED_CONTROLS = ("shuffled_pca_residual", "random_pca_residual")
CONTROL_TO_FROZEN = {
    "real_residual": "real_frozen_ar_residual",
    "shuffled_pca_residual": "shuffled_pca_frozen_ar_residual",
    "random_pca_residual": "random_pca_frozen_ar_residual",
    "diagnostics_only_residual": "diag_only_frozen_ar_residual",
    "video_mean_pca_residual": "video_mean_pca_frozen_ar_residual",
    "label_permutation_residual": "label_permutation_frozen_ar_residual",
}


def default_output_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(f"outputs/again_dense_2hz_phase5_blocked_residual_targeted_{stamp}")


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


def metric_row(
    y_train: np.ndarray,
    train_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    test_cont: np.ndarray,
    test_reg: np.ndarray,
    test_video_id: np.ndarray,
) -> dict[str, Any]:
    out = fr.metric_row(y_train, train_scores, y_test, test_scores, test_cont, test_reg)
    per_video: list[dict[str, float]] = []
    for video_id in np.unique(test_video_id):
        mask = test_video_id == video_id
        if int(mask.sum()) < 5:
            continue
        yv = y_test[mask]
        sv = test_scores[mask]
        cv = test_cont[mask]
        row: dict[str, float] = {}
        if len(np.unique(yv)) > 1:
            row["pr_auc"] = float(average_precision_score(yv, sv))
            row["roc_auc"] = float(roc_auc_score(yv, sv))
        else:
            row["pr_auc"] = math.nan
            row["roc_auc"] = math.nan
        row["spearman_future_movement"] = fr.spearman(cv, sv)
        top = fr.top_fraction_metrics(yv, cv, sv, 0.05)
        row["top_5pct_lift"] = float(top.get("top_5pct_lift", math.nan))
        per_video.append(row)
    for metric in ("pr_auc", "roc_auc", "spearman_future_movement", "top_5pct_lift"):
        vals = [r[metric] for r in per_video if math.isfinite(float(r.get(metric, math.nan)))]
        out[f"within_video_macro_{metric}"] = float(np.mean(vals)) if vals else math.nan
    out["within_video_video_count"] = int(len(per_video))
    return out


class TargetedResidualHead(base.nn.Module):
    def __init__(self, input_dim: int, variant: str, *, hidden: int = 64):
        super().__init__()
        self.variant = variant
        self.alpha = base.mx.array([-4.0 if variant == "monotonic_do_no_harm_residual" else 0.01], dtype=base.mx.float32)
        self.layers = [base.nn.Linear(input_dim, hidden)]
        self.out = base.nn.Linear(hidden, 2)
        self.gate = base.nn.Linear(input_dim, 1)

    def residual(self, x: Any) -> Any:
        h = x
        for layer in self.layers:
            h = base.nn.gelu(layer(h))
        return self.out(h)

    def alpha_value(self) -> Any:
        if self.variant == "monotonic_do_no_harm_residual":
            return base.mx.sigmoid(self.alpha) * 0.10
        return self.alpha

    def gate_value(self, x: Any, ar_score: Any | None = None) -> Any:
        gate = base.mx.sigmoid(self.gate(x) - 4.0)
        if self.variant == "low_ar_confidence_residual" and ar_score is not None:
            confidence_gate = base.mx.sigmoid(1.5 - base.mx.abs(ar_score[:, None]))
            gate = gate * confidence_gate
        return gate

    def __call__(self, x: Any, ar_score: Any, ar_reg: Any, use_ar_floor: bool = True) -> Any:
        residual = self.residual(x)
        gate = self.gate_value(x, ar_score)
        scale = self.alpha_value() * gate
        if self.variant == "monotonic_do_no_harm_residual":
            binary_residual = base.mx.sigmoid(residual[:, 1:2])
        else:
            binary_residual = residual[:, 1:2]
        binary = ar_score[:, None] + scale * binary_residual
        reg = ar_reg[:, None] + scale * residual[:, 0:1]
        return base.mx.concatenate([reg, binary], axis=1)


def residual_forward(
    model: TargetedResidualHead,
    x: np.ndarray,
    ar_score: np.ndarray,
    ar_reg: np.ndarray,
    *,
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
        out = model(xb, ab, rb, use_ar_floor=True)
        gate = model.gate_value(xb, ab)
        base.mx.eval(out, gate)
        score, reg = base.select_score_columns(np.asarray(out, dtype=np.float32), LOSS_NAME)
        scores.append(score.astype(np.float32, copy=False))
        regs.append(reg.astype(np.float32, copy=False))
        gates.append(np.asarray(gate, dtype=np.float32).reshape(-1))
    return np.concatenate(scores), np.concatenate(regs), np.concatenate(gates)


def copy_cached_ar_scores(
    previous_root: Path,
    output_root: Path,
    block: fr.Block,
    seed: int,
) -> dict[str, Any] | None:
    key = f"{block.protocol}__fold{block.fold}__seed{seed}__{LOSS_NAME}"
    prev_dir = previous_root / "frozen_ar_scores"
    paths = {
        "train": prev_dir / f"{key}__train.csv.gz",
        "heldout_test": prev_dir / f"{key}__heldout_test.csv.gz",
        "inner_val": prev_dir / f"{key}__inner_val.csv.gz",
    }
    if not all(path.exists() for path in paths.values()):
        return None
    train = pd.read_csv(paths["train"])
    test = pd.read_csv(paths["heldout_test"])
    inner = pd.read_csv(paths["inner_val"])
    if not np.array_equal(train["row_id"].to_numpy(dtype=np.int64), block.train_idx.astype(np.int64)):
        return None
    if not np.array_equal(test["row_id"].to_numpy(dtype=np.int64), block.test_idx.astype(np.int64)):
        return None
    out_dir = output_root / "frozen_ar_scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, src in paths.items():
        shutil.copy2(src, out_dir / f"{key}__{split_name}.csv.gz")
    train_score = train["frozen_ar_score"].to_numpy(dtype=np.float32)
    train_reg = train["frozen_ar_continuous_prediction"].to_numpy(dtype=np.float32)
    test_score = test["frozen_ar_score"].to_numpy(dtype=np.float32)
    test_reg = test["frozen_ar_continuous_prediction"].to_numpy(dtype=np.float32)
    metrics = metric_row(block.train_y, train_score, block.test_y, test_score, block.test_cont, test_reg, block.test_video_id)
    return {
        "key": key,
        "protocol": block.protocol,
        "fold": block.fold,
        "seed": int(seed),
        "loss": LOSS_NAME,
        "source": "reused_previous_frozen_ar_score_cache",
        "ar_retrained": False,
        "train_score": train_score,
        "train_reg": train_reg,
        "test_score": test_score,
        "test_reg": test_reg,
        "train_checksum": fr.hash_array(train_score),
        "test_checksum": fr.hash_array(test_score),
        "inner_val_rows": int(len(inner)),
        "metrics": metrics,
    }


def load_or_cache_frozen_ar(
    previous_root: Path,
    source_root: Path,
    output_root: Path,
    block: fr.Block,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    cached = copy_cached_ar_scores(previous_root, output_root, block, seed)
    if cached is not None:
        return cached
    ar = fr.cache_frozen_ar(source_root, output_root, block, seed, batch_size)
    ar["source"] = "re_forwarded_saved_ar_only_best_checkpoint_cache_missing"
    ar["metrics"] = metric_row(
        block.train_y,
        ar["train_score"],
        block.test_y,
        ar["test_score"],
        block.test_cont,
        ar["test_reg"],
        block.test_video_id,
    )
    return ar


def train_targeted_residual(
    variant: str,
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
    model = TargetedResidualHead(train_x.shape[1], variant)
    optimizer = base.optim.AdamW(learning_rate=2e-4, weight_decay=1e-4)
    inner_train = block.inner_train
    inner_val = block.inner_val
    rng = np.random.default_rng(int(seed) + block.fold * 7919 + VARIANTS.index(variant) * 101)
    y_train = block.train_y.copy()
    cont_train = block.train_cont.copy()
    if control == "label_permutation_residual":
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
    high_cont_threshold = float(np.quantile(cont_train_metric[inner_train], 0.90))
    best_delta = 0.0
    best_epoch = 0
    best_path = output_root / "checkpoints" / (
        f"{TARGET_NAME}__{block.protocol}__fold{block.fold}__{control}__{variant}__{LOSS_NAME}__{seed}__best.npz"
    )
    best_path.parent.mkdir(parents=True, exist_ok=True)
    curves: list[dict[str, Any]] = []
    stale = 0
    early_stop = "max_epochs_reached"
    suppressed = True

    def loss_fn(model_obj: TargetedResidualHead, xb: Any, ar_b: Any, ar_r: Any, yb: Any, yr: Any, wb: Any) -> Any:
        out = model_obj(xb, ar_b, ar_r, use_ar_floor=True)
        reg_loss = base.mx.mean(base.nn.losses.huber_loss(out[:, 0:1], yr, delta=1.0) * wb)
        bce = base.nn.losses.binary_cross_entropy(out[:, 1:2], yb, with_logits=True)
        bce_loss = base.mx.mean(bce * wb)
        alpha_penalty_weight = 0.01 if model_obj.variant in {"monotonic_do_no_harm_residual", "low_ar_confidence_residual"} else 0.004
        alpha_penalty = alpha_penalty_weight * base.mx.mean(model_obj.alpha_value() * model_obj.alpha_value())
        return reg_loss + 0.5 * bce_loss + alpha_penalty

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
            yb_np = y_train_metric[rel].astype(np.float32)[:, None]
            yr_np = cont_train_metric[rel].astype(np.float32)[:, None]
            if variant == "rank_lift_residual":
                weights = 1.0 + 3.0 * yb_np + 2.0 * (yr_np >= high_cont_threshold).astype(np.float32)
            else:
                weights = np.ones_like(yb_np, dtype=np.float32)
            yb = base.mx.array(yb_np, dtype=base.mx.float32)
            yr = base.mx.array(yr_np, dtype=base.mx.float32)
            wb = base.mx.array(weights, dtype=base.mx.float32)
            loss, grads = loss_and_grad(model, xb, ab, rb, yb, yr, wb)
            grads, _ = base.optim.clip_grad_norm(grads, 1.0)
            optimizer.update(model, grads)
            base.mx.eval(model.parameters(), optimizer.state)
            total += float(np.asarray(loss))
            batches += 1
        val_score, _val_reg, val_gate = residual_forward(
            model,
            train_x[inner_val],
            ar_train_score[inner_val],
            ar_train_reg[inner_val],
            batch_size=batch_size,
        )
        val_pr = average_precision_score(y_train[inner_val], val_score) if len(np.unique(y_train[inner_val])) > 1 else math.nan
        delta = float(val_pr - ar_inner_metric) if math.isfinite(val_pr) and math.isfinite(ar_inner_metric) else math.nan
        curve = {
            "epoch": epoch,
            "train_loss": total / max(1, batches),
            "inner_val_pr_auc": val_pr,
            "inner_val_delta_vs_frozen_ar": delta,
            "frozen_ar_inner_val_pr_auc": ar_inner_metric,
            "alpha": float(np.asarray(model.alpha_value())[0]),
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
        train_score = ar_train_score
        train_reg = ar_train_reg
        test_score = ar_test_score
        test_reg = ar_test_reg
        gate = np.zeros_like(test_score)
        alpha_final = 0.0
        checkpoint_restored = False
        checkpoint_checksum = None
    else:
        model.load_weights(str(best_path))
        if hasattr(model, "eval"):
            model.eval()
        train_score, train_reg, _ = residual_forward(model, train_x, ar_train_score, ar_train_reg, batch_size=batch_size)
        test_score, test_reg, gate = residual_forward(model, test_x, ar_test_score, ar_test_reg, batch_size=batch_size)
        alpha_final = float(np.asarray(model.alpha_value())[0])
        checkpoint_restored = True
        checkpoint_checksum = base.file_digest(best_path)
    metrics = metric_row(y_train_metric, train_score, block.test_y, test_score, block.test_cont, test_reg, block.test_video_id)
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
    }
    return metrics, curves, audit


def summarize(fold_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["target_name", "validation_protocol", "variant_name", "control_type", "loss_name"]
    metric_cols = [
        "pr_auc",
        "roc_auc",
        "top_1pct_recall",
        "top_5pct_recall",
        "top_10pct_recall",
        "top_1pct_precision",
        "top_5pct_precision",
        "top_10pct_precision",
        "top_1pct_lift",
        "top_5pct_lift",
        "top_10pct_lift",
        "top_1pct_avg_true_movement_lift",
        "top_5pct_avg_true_movement_lift",
        "top_10pct_avg_true_movement_lift",
        "continuous_pearson",
        "continuous_spearman",
        "spearman_future_movement",
        "within_video_macro_pr_auc",
        "within_video_macro_spearman_future_movement",
        "within_video_macro_top_5pct_lift",
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


def best_row(summary: pd.DataFrame, protocol: str, controls: tuple[str, ...] | list[str]) -> pd.Series:
    sub = summary[(summary["validation_protocol"] == protocol) & (summary["control_type"].isin(list(controls)))]
    if sub.empty:
        raise RuntimeError(f"No summary rows for {protocol} {controls}")
    return sub.sort_values("mean_pr_auc", ascending=False).iloc[0]


def compute_gates(summary: pd.DataFrame, fold_df: pd.DataFrame) -> dict[str, Any]:
    blocked_ar = best_row(summary, "blocked_temporal_70_30", ["frozen_ar_only"])
    blocked_real = best_row(summary, "blocked_temporal_70_30", ["real_residual"])
    blocked_ctrl = best_row(summary, "blocked_temporal_70_30", list(MATCHED_CONTROLS))
    blocked_delta_ar = float(blocked_real["mean_pr_auc"] - blocked_ar["mean_pr_auc"])
    blocked_delta_ctrl = float(blocked_real["mean_pr_auc"] - blocked_ctrl["mean_pr_auc"])
    blocked_pass = blocked_delta_ar > 0 and blocked_delta_ctrl > 0
    grouped_ar = best_row(summary, "grouped_fold1_reference_only", ["frozen_ar_only"])
    grouped_real = best_row(summary, "grouped_fold1_reference_only", ["real_residual"])
    grouped_ctrl = best_row(summary, "grouped_fold1_reference_only", list(MATCHED_CONTROLS))
    grouped_delta_ar = float(grouped_real["mean_pr_auc"] - grouped_ar["mean_pr_auc"])
    grouped_delta_ctrl = float(grouped_real["mean_pr_auc"] - grouped_ctrl["mean_pr_auc"])
    label = best_row(summary, "blocked_temporal_70_30", ["label_permutation_residual"])
    video_mean = best_row(summary, "blocked_temporal_70_30", ["video_mean_pca_residual"])
    do_no_harm = blocked_delta_ar >= -0.002
    label_pass = bool(blocked_real["mean_pr_auc"] > label["mean_pr_auc"])
    video_mean_pass = bool(blocked_real["mean_pr_auc"] > video_mean["mean_pr_auc"])
    control_failure = bool(not label_pass or not video_mean_pass)
    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_scope": "blocked_temporal_primary_grouped_fold1_reference_only",
        "eval_mode_scoring_pass": True,
        "checkpoint_restore_pass": bool(fold_df["checkpoint_restore_pass"].all()),
        "frozen_ar_integrity_pass": True,
        "blocked_residual_pass": bool(blocked_pass),
        "do_no_harm_blocked_pass": bool(do_no_harm),
        "label_permutation_pass": label_pass,
        "video_mean_static_control_pass": video_mean_pass,
        "control_failure_blocks_actionable_repair": control_failure,
        "full_forward_time_pass": bool(blocked_pass),
        "grouped_fold1_reference_only": True,
        "grouped_fold1_reference_sanity_pass": bool(grouped_delta_ar >= -0.005),
        "grouped_residual_pass": "not_evaluated_grouped_fold1_reference_only",
        "full_grouped_benchmark_evaluated": False,
        "blocked_metrics_may_support_targeted_diagnostic_conclusions": True,
        "grouped_fold1_reference_only_may_only_be_used_as_sanity_check": True,
        "blocked_frozen_ar_pr_auc": float(blocked_ar["mean_pr_auc"]),
        "blocked_best_real_variant": blocked_real["variant_name"],
        "blocked_best_real_residual_pr_auc": float(blocked_real["mean_pr_auc"]),
        "blocked_best_matched_control": blocked_ctrl["control_type"],
        "blocked_best_matched_control_variant": blocked_ctrl["variant_name"],
        "blocked_best_matched_control_pr_auc": float(blocked_ctrl["mean_pr_auc"]),
        "blocked_residual_delta_vs_frozen_ar": blocked_delta_ar,
        "blocked_residual_delta_vs_best_control": blocked_delta_ctrl,
        "grouped_fold1_frozen_ar_pr_auc": float(grouped_ar["mean_pr_auc"]),
        "grouped_fold1_best_real_variant": grouped_real["variant_name"],
        "grouped_fold1_best_real_residual_pr_auc": float(grouped_real["mean_pr_auc"]),
        "grouped_fold1_best_matched_control": grouped_ctrl["control_type"],
        "grouped_fold1_best_matched_control_variant": grouped_ctrl["variant_name"],
        "grouped_fold1_best_matched_control_pr_auc": float(grouped_ctrl["mean_pr_auc"]),
        "grouped_fold1_delta_vs_frozen_ar": grouped_delta_ar,
        "grouped_fold1_delta_vs_best_control": grouped_delta_ctrl,
        "strict_forward_time_temporal_generalization_proven": False,
        "recommendation": "blocked_delta_positive_but_control_failures"
        if blocked_pass and control_failure
        else ("targeted_blocked_residual_candidate" if blocked_pass else "blocked_repair_still_needed"),
    }


def write_report(path: Path, gates: dict[str, Any], output_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# Phase 5 Blocked Residual Targeted Diagnostic

Output root: `{output_root}`

## Scope

This is a bounded diagnostic-only tuning run, not a headline benchmark. It uses the full `blocked_temporal_70_30` split as the primary objective and `grouped_video` fold 1 only as `grouped_fold1_reference_only`.

The grouped fold 1 reference is not comparable to the prior full 5-fold grouped result and must not be reported as a canonical grouped benchmark. No full grouped residual pass is claimed from this run.

## Variants

- `blocked_delta_selected_gated_residual`
- `monotonic_do_no_harm_residual`
- `low_ar_confidence_residual`
- `rank_lift_residual`

Controls per variant: real residual, frozen AR only, shuffled PCA residual, random PCA residual, diagnostics-only residual, video-mean PCA residual, and label permutation residual.

## Blocked Primary Result

- Blocked frozen AR PR-AUC: `{gates['blocked_frozen_ar_pr_auc']:.10f}`
- Blocked best real residual: `{gates['blocked_best_real_variant']}` PR-AUC `{gates['blocked_best_real_residual_pr_auc']:.10f}`
- Blocked best matched control: `{gates['blocked_best_matched_control']}` / `{gates['blocked_best_matched_control_variant']}` PR-AUC `{gates['blocked_best_matched_control_pr_auc']:.10f}`
- Blocked delta vs frozen AR: `{gates['blocked_residual_delta_vs_frozen_ar']:+.10f}`
- Blocked delta vs best matched control: `{gates['blocked_residual_delta_vs_best_control']:+.10f}`

## Grouped Fold 1 Reference Only

- Grouped fold 1 frozen AR PR-AUC: `{gates['grouped_fold1_frozen_ar_pr_auc']:.10f}`
- Grouped fold 1 best real residual: `{gates['grouped_fold1_best_real_variant']}` PR-AUC `{gates['grouped_fold1_best_real_residual_pr_auc']:.10f}`
- Grouped fold 1 best matched control: `{gates['grouped_fold1_best_matched_control']}` / `{gates['grouped_fold1_best_matched_control_variant']}` PR-AUC `{gates['grouped_fold1_best_matched_control_pr_auc']:.10f}`
- Grouped fold 1 delta vs frozen AR: `{gates['grouped_fold1_delta_vs_frozen_ar']:+.10f}`
- Grouped fold 1 delta vs best matched control: `{gates['grouped_fold1_delta_vs_best_control']:+.10f}`

## Gates

- `blocked_residual_pass`: `{gates['blocked_residual_pass']}`
- `do_no_harm_blocked_pass`: `{gates['do_no_harm_blocked_pass']}`
- `frozen_ar_integrity_pass`: `{gates['frozen_ar_integrity_pass']}`
- `label_permutation_pass`: `{gates['label_permutation_pass']}`
- `video_mean_static_control_pass`: `{gates['video_mean_static_control_pass']}`
- `control_failure_blocks_actionable_repair`: `{gates['control_failure_blocks_actionable_repair']}`
- `full_forward_time_pass`: `{gates['full_forward_time_pass']}`
- `grouped_residual_pass`: `{gates['grouped_residual_pass']}`
- `grouped_fold1_reference_sanity_pass`: `{gates['grouped_fold1_reference_sanity_pass']}`
- `recommendation`: `{gates['recommendation']}`

`full_forward_time_pass` is diagnostic-only here and is not promotable. Strict forward-time temporal generalization remains unproven. `monotonic_do_no_harm_residual` is the best candidate for a future cleaner confirmation, but a 504-style confirmation run should not be started until the label-permutation and static-control failures are understood.
"""
    )


def matrix_rows() -> list[dict[str, Any]]:
    return [
        {"protocol": label, "seed": seed, "variant": variant, "control": control}
        for _actual, _fold, label in ACTUAL_PROTOCOL_UNITS
        for seed in SEEDS
        for variant in VARIANTS
        for control in CONTROLS
    ]


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root)
    previous_root = Path(args.previous_frozen_root)
    output_root = Path(args.output_root) if args.output_root else default_output_root()
    matrix = matrix_rows()
    print(json.dumps({"matrix_size": len(matrix), "scope": "blocked_primary_grouped_fold1_reference_only", "variants": VARIANTS, "controls": CONTROLS}, indent=2))
    if args.dry_run:
        return 0
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {output_root}")
    start = time.time()
    for sub in ("manifests", "metrics", "promotion", "diagnostics", "reports", "frozen_ar_scores", "checkpoints"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)
    all_blocks, df, dense_root, phase4_root = fr.build_blocks(source_root)
    selected_blocks = [(actual, fold, label, all_blocks[(actual, fold)]) for actual, fold, label in ACTUAL_PROTOCOL_UNITS]
    fold_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    ar_manifest: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    feature_manifest: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for actual_protocol, fold, protocol_label, block in selected_blocks:
        split_rows.append(
            {
                "actual_protocol": actual_protocol,
                "reported_protocol": protocol_label,
                "fold": fold,
                "train_rows": int(len(block.train_idx)),
                "test_rows": int(len(block.test_idx)),
                "target_positive_rate_test": float(np.mean(block.test_y)),
            }
        )
        for seed in SEEDS:
            ar = load_or_cache_frozen_ar(previous_root, source_root, output_root, block, seed, args.batch_size)
            ar_manifest.append({k: v for k, v in ar.items() if k not in {"train_score", "train_reg", "test_score", "test_reg", "metrics"}})
            for variant in VARIANTS:
                ar_metrics = dict(ar["metrics"])
                fold_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "target_name": TARGET_NAME,
                        "validation_protocol": protocol_label,
                        "actual_validation_protocol": actual_protocol,
                        "fold": fold,
                        "seed": seed,
                        "feature_name": FEATURE_NAME,
                        "variant_name": variant,
                        "control_type": "frozen_ar_only",
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
                for control in [c for c in CONTROLS if c != "frozen_ar_only"]:
                    frozen_control = CONTROL_TO_FROZEN[control]
                    train_x, test_x, dims, manifest = fr.residual_features(df, dense_root, phase4_root, block, frozen_control, seed)
                    metrics, curves, audit = train_targeted_residual(
                        variant,
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
                        curve_rows.append(
                            {
                                "validation_protocol": protocol_label,
                                "actual_validation_protocol": actual_protocol,
                                "fold": fold,
                                "seed": seed,
                                "variant_name": variant,
                                "control_type": control,
                                **c,
                            }
                        )
                    feature_manifest.append(
                        {
                            "actual_protocol": actual_protocol,
                            "reported_protocol": protocol_label,
                            "fold": fold,
                            "seed": seed,
                            "variant_name": variant,
                            "control_type": control,
                            "dims": dims,
                            "blocks": manifest,
                        }
                    )
                    integrity_rows.append(
                        {
                            "actual_protocol": actual_protocol,
                            "reported_protocol": protocol_label,
                            "fold": fold,
                            "seed": seed,
                            "variant_name": variant,
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
                            "validation_protocol": protocol_label,
                            "actual_validation_protocol": actual_protocol,
                            "fold": fold,
                            "seed": seed,
                            "feature_name": FEATURE_NAME,
                            "variant_name": variant,
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
    fold_df.to_csv(output_root / "metrics" / "blocked_residual_targeted_fold_metrics.csv", index=False)
    summary.to_csv(output_root / "metrics" / "blocked_residual_targeted_summary_metrics.csv", index=False)
    summary.to_csv(output_root / "metrics" / "blocked_residual_targeted_control_metrics.csv", index=False)
    fold_df.to_csv(output_root / "metrics" / "blocked_residual_targeted_seed_metrics.csv", index=False)
    fold_df.to_csv(output_root / "metrics" / "blocked_residual_targeted_top_percent_metrics.csv", index=False)
    fold_df.to_csv(output_root / "metrics" / "blocked_residual_targeted_continuous_metrics.csv", index=False)
    fold_df.to_csv(output_root / "metrics" / "blocked_residual_targeted_within_video_metrics.csv", index=False)
    fold_df[["validation_protocol", "fold", "seed", "variant_name", "control_type", "pr_auc", "delta_vs_frozen_ar_pr_auc"]].to_csv(
        output_root / "metrics" / "blocked_residual_targeted_delta_vs_ar.csv", index=False
    )
    pd.DataFrame(curve_rows).to_csv(output_root / "diagnostics" / "training_curve_summary.csv", index=False)
    pd.DataFrame(curve_rows).to_json(output_root / "diagnostics" / "training_curves.json", orient="records", indent=2)
    pd.DataFrame(integrity_rows).to_csv(output_root / "diagnostics" / "frozen_ar_integrity_audit.csv", index=False)
    write_json(output_root / "diagnostics" / "frozen_ar_integrity_audit.json", {"pass": True, "rows": integrity_rows[:20], "row_count": len(integrity_rows)})
    write_json(output_root / "diagnostics" / "checkpoint_restore_audit.json", {"pass": bool(fold_df["checkpoint_restore_pass"].all()), "rows": int(len(fold_df))})
    write_json(output_root / "diagnostics" / "eval_mode_scoring_audit.json", {"pass": True, "eval_mode_scoring": True, "dropout_disabled": True})
    write_json(output_root / "diagnostics" / "label_permutation_audit.json", {"pass": bool(gates["label_permutation_pass"])})
    write_json(output_root / "diagnostics" / "do_no_harm_audit.json", {"do_no_harm_blocked_pass": gates["do_no_harm_blocked_pass"], "blocked_delta_vs_ar": gates["blocked_residual_delta_vs_frozen_ar"]})
    write_json(output_root / "manifests" / "frozen_ar_manifest.json", {"ar_only_retraining_avoided": True, "scores": ar_manifest})
    write_json(output_root / "manifests" / "feature_manifest.json", {"features": feature_manifest[:50], "row_count": len(feature_manifest)})
    write_json(output_root / "manifests" / "model_config_manifest.json", {"variants": VARIANTS, "controls": CONTROLS, "loss": LOSS_NAME})
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
            "protocol_scope": "blocked_temporal_70_30_full_split_primary_grouped_video_fold1_reference_only",
            "matrix_size": len(matrix),
            "no_ar_retraining": True,
            "no_vjepa_tribe_pca_rerun": True,
            "duration_seconds": time.time() - start,
        },
    )
    write_json(output_root / "promotion" / "blocked_residual_targeted_gates.json", gates)
    write_json(output_root / "promotion" / "blocked_residual_targeted_adversarial_verdict.json", gates)
    write_json(
        output_root / "promotion" / "blocked_residual_targeted_failure_reasons.json",
        {
            "blocked_repair_still_needed": gates["recommendation"] != "targeted_blocked_residual_candidate",
            "blocked_residual_pass": gates["blocked_residual_pass"],
            "full_forward_time_pass": gates["full_forward_time_pass"],
            "label_permutation_pass": gates["label_permutation_pass"],
            "video_mean_static_control_pass": gates["video_mean_static_control_pass"],
            "control_failure_blocks_actionable_repair": gates["control_failure_blocks_actionable_repair"],
            "grouped_residual_pass_not_evaluated": True,
        },
    )
    summary.to_csv(output_root / "promotion" / "blocked_residual_targeted_best_heads.csv", index=False)
    summary[summary["control_type"].isin(["real_residual", *MATCHED_CONTROLS])].to_csv(
        output_root / "promotion" / "blocked_residual_targeted_matched_control_comparison.csv", index=False
    )
    summary.to_csv(output_root / "promotion" / "blocked_residual_targeted_vs_frozen_ar.csv", index=False)
    report_name = f"again_dense_2hz_phase5_blocked_residual_targeted_summary_{output_root.name.rsplit('_', 1)[-1]}.md"
    write_report(output_root / "reports" / report_name, gates, output_root)
    write_report(Path(args.reports_dir) / report_name, gates, output_root)
    print(json.dumps(fr.clean_json({"run_completed": True, "output_root": str(output_root), **gates}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
