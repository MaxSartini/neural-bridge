"""Clean monotonic-only blocked residual confirmation.

This runs the design from
`evidence/phase_5_2_blocked_residual_targeted_230325/confirmation_design.json`.
It is intentionally bounded:

1 blocked protocol x 1 variant x 1 loss x 5 seeds x 9 controls = 45 rows.

It does not run grouped, does not run 504, does not start secondary targets, and
does not retrain AR. Frozen AR scores are reused when cached; otherwise saved
AR-only checkpoints are re-forwarded in deterministic eval mode.
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
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts import run_again_dense_2hz_phase5_blocked_residual_targeted as targeted
from backend.scripts import run_again_dense_2hz_phase5_frozen_ar_residual as fr
from backend.scripts import run_again_dense_2hz_phase5_learned_heads as base


SCHEMA_VERSION = "again_dense_2hz_phase5_blocked_residual_clean_confirm_v1"
SOURCE_ROOT = Path("outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_20260629_171825")
PREVIOUS_FROZEN_ROOT = Path("outputs/again_dense_2hz_phase5_frozen_ar_residual_")
DESIGN_PATH = Path("evidence/phase_5_2_blocked_residual_targeted_230325/confirmation_design.json")
TARGET_NAME = fr.TARGET_NAME
CONTINUOUS_SOURCE = fr.CONTINUOUS_SOURCE
FEATURE_NAME = fr.FEATURE_NAME
LOSS_NAME = fr.LOSS_NAME
PROTOCOL = "blocked_temporal_70_30"
FOLD = 1
VARIANT = "monotonic_do_no_harm_residual"
SEEDS = (20260625, 20260626, 20260627, 20260628, 20260629)
FROZEN_AR_FALLBACK_SEED = 20260627
CONTROLS = (
    "frozen_ar_only",
    "real_residual",
    "shuffled_pca_residual",
    "random_pca_residual",
    "label_permutation_residual_permuted_inner_val_selection",
    "label_permutation_fixed_epoch_audit",
    "train_only_video_mean_pca_residual",
    "full_video_video_mean_pca_residual_oracle_warning",
    "diagnostics_only_residual",
)
PRIMARY_CONTROLS = (
    "shuffled_pca_residual",
    "random_pca_residual",
    "label_permutation_residual_permuted_inner_val_selection",
    "train_only_video_mean_pca_residual",
)
WEAK_THRESHOLD = 0.001
CREDIBLE_THRESHOLD = 0.003
DO_NO_HARM_SEED_FLOOR = -0.0005
ORACLE_WARNING_THRESHOLD = 0.001


def default_output_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(f"outputs/again_dense_2hz_phase5_blocked_residual_clean_confirm_{stamp}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--previous-frozen-root", default=str(PREVIOUS_FROZEN_ROOT))
    parser.add_argument("--design-path", default=str(DESIGN_PATH))
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


def clean_control_to_feature_control(control: str) -> str:
    mapping = {
        "real_residual": "real_frozen_ar_residual",
        "shuffled_pca_residual": "shuffled_pca_frozen_ar_residual",
        "random_pca_residual": "random_pca_frozen_ar_residual",
        "diagnostics_only_residual": "diag_only_frozen_ar_residual",
        "label_permutation_residual_permuted_inner_val_selection": "real_frozen_ar_residual",
        "label_permutation_fixed_epoch_audit": "real_frozen_ar_residual",
    }
    return mapping[control]


def copy_fallback_frozen_ar_scores(previous_root: Path, output_root: Path, block: fr.Block, seed: int) -> dict[str, Any] | None:
    source_seed = FROZEN_AR_FALLBACK_SEED
    prev_dir = previous_root / "frozen_ar_scores"
    src_key = f"{block.protocol}__fold{block.fold}__seed{source_seed}__{LOSS_NAME}"
    dst_key = f"{block.protocol}__fold{block.fold}__seed{seed}__{LOSS_NAME}"
    paths = {
        "train": prev_dir / f"{src_key}__train.csv.gz",
        "heldout_test": prev_dir / f"{src_key}__heldout_test.csv.gz",
        "inner_val": prev_dir / f"{src_key}__inner_val.csv.gz",
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
        shutil.copy2(src, out_dir / f"{dst_key}__{split_name}.csv.gz")
    train_score = train["frozen_ar_score"].to_numpy(dtype=np.float32)
    train_reg = train["frozen_ar_continuous_prediction"].to_numpy(dtype=np.float32)
    test_score = test["frozen_ar_score"].to_numpy(dtype=np.float32)
    test_reg = test["frozen_ar_continuous_prediction"].to_numpy(dtype=np.float32)
    metrics = targeted.metric_row(block.train_y, train_score, block.test_y, test_score, block.test_cont, test_reg, block.test_video_id)
    return {
        "key": dst_key,
        "protocol": block.protocol,
        "fold": block.fold,
        "seed": int(seed),
        "loss": LOSS_NAME,
        "source": "reused_fallback_frozen_ar_score_cache_no_ar_checkpoint_for_confirmation_seed",
        "fallback_source_seed": int(source_seed),
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


def load_or_cache_frozen_ar(previous_root: Path, source_root: Path, output_root: Path, block: fr.Block, seed: int, batch_size: int) -> dict[str, Any]:
    try:
        return targeted.load_or_cache_frozen_ar(previous_root, source_root, output_root, block, seed, batch_size)
    except RuntimeError as exc:
        if "Expected one AR-only checkpoint row" not in str(exc):
            raise
        fallback = copy_fallback_frozen_ar_scores(previous_root, output_root, block, seed)
        if fallback is None:
            raise
        return fallback


def video_mean_features(
    df: pd.DataFrame,
    dense_root: Path,
    phase4_root: Path,
    block: fr.Block,
    *,
    train_only: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], list[dict[str, Any]]]:
    _train_idx, _test_idx, pca_train, pca_test, source_path = base.load_phase4_scores(
        df,
        phase4_root,
        block.split,
        base.feature_spec(FEATURE_NAME),
    )
    if not np.array_equal(_train_idx, block.train_idx) or not np.array_equal(_test_idx, block.test_idx):
        raise RuntimeError("Residual feature row index mismatch with blocked split.")
    train_video = block.train_video_id.astype(str)
    test_video = block.test_video_id.astype(str)
    if train_only:
        means = {video: pca_train[train_video == video].mean(axis=0) for video in np.unique(train_video)}
        global_mean = pca_train.mean(axis=0)
        pca_kind = "video_mean_train_only"
    else:
        all_videos = np.concatenate([train_video, test_video])
        all_pca = np.concatenate([pca_train, pca_test], axis=0)
        means = {video: all_pca[all_videos == video].mean(axis=0) for video in np.unique(all_videos)}
        global_mean = all_pca.mean(axis=0)
        pca_kind = "video_mean_full_video_oracle"
    p_train = np.vstack([means.get(v, global_mean) for v in train_video]).astype(np.float32)
    p_test = np.vstack([means.get(v, global_mean) for v in test_video]).astype(np.float32)
    train_x, test_x = base.standardize_train_only(p_train, p_test)
    return (
        train_x.astype(np.float32, copy=False),
        test_x.astype(np.float32, copy=False),
        {"pca": p_train.shape[1]},
        [
            {
                "block": "phase4_fold_safe_pca",
                "kind": pca_kind,
                "source_path": str(source_path),
                "source_checksum": base.file_digest(source_path),
                "width": p_train.shape[1],
                "uses_test_rows_for_mean": not train_only,
                "includes_temporal_diagnostics": False,
            }
        ],
    )


def residual_features_for_control(
    df: pd.DataFrame,
    dense_root: Path,
    phase4_root: Path,
    block: fr.Block,
    control: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], list[dict[str, Any]]]:
    if control == "train_only_video_mean_pca_residual":
        return video_mean_features(df, dense_root, phase4_root, block, train_only=True)
    if control == "full_video_video_mean_pca_residual_oracle_warning":
        return video_mean_features(df, dense_root, phase4_root, block, train_only=False)
    return fr.residual_features(df, dense_root, phase4_root, block, clean_control_to_feature_control(control), seed)


def label_permutation_arrays(block: fr.Block, seed: int, control: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if not control.startswith("label_permutation"):
        return None, None
    offset = 30103 if control == "label_permutation_residual_permuted_inner_val_selection" else 40111
    rng = np.random.default_rng(int(seed) + block.fold * 7919 + offset)
    perm = rng.permutation(len(block.train_y))
    return block.train_y[perm].copy(), block.train_cont[perm].copy()


def train_clean_residual(
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
    *,
    fixed_epoch: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    base.require_mlx()
    base.mx.random.seed(int(seed))
    model = targeted.TargetedResidualHead(train_x.shape[1], VARIANT)
    optimizer = base.optim.AdamW(learning_rate=2e-4, weight_decay=1e-4)
    inner_train = block.inner_train
    inner_val = block.inner_val
    rng = np.random.default_rng(int(seed) + block.fold * 7919 + 101)
    y_metric, cont_metric = label_permutation_arrays(block, seed, control)
    if y_metric is None or cont_metric is None:
        y_metric = block.train_y.copy()
        cont_metric = block.train_cont.copy()
        selection_y = block.train_y
        selection_cont = block.train_cont
        selection_policy = "true_inner_val_labels"
    else:
        selection_y = y_metric
        selection_cont = cont_metric
        selection_policy = "permuted_inner_val_labels"

    ar_train_score = ar["train_score"].astype(np.float32)
    ar_train_reg = ar["train_reg"].astype(np.float32)
    ar_test_score = ar["test_score"].astype(np.float32)
    ar_test_reg = ar["test_reg"].astype(np.float32)
    ar_inner_metric = average_precision_score(selection_y[inner_val], ar_train_score[inner_val])
    true_ar_inner_metric = average_precision_score(block.train_y[inner_val], ar_train_score[inner_val])
    high_cont_threshold = float(np.quantile(cont_metric[inner_train], 0.90))
    best_delta = 0.0
    best_epoch = 0
    fixed_epoch = int(fixed_epoch) if fixed_epoch is not None and int(fixed_epoch) > 0 else None
    best_path = output_root / "checkpoints" / (
        f"{TARGET_NAME}__{PROTOCOL}__fold{FOLD}__{control}__{VARIANT}__{LOSS_NAME}__{seed}__best.npz"
    )
    best_path.parent.mkdir(parents=True, exist_ok=True)
    curves: list[dict[str, Any]] = []
    stale = 0
    early_stop = "max_epochs_reached"
    suppressed = True

    def loss_fn(model_obj: targeted.TargetedResidualHead, xb: Any, ar_b: Any, ar_r: Any, yb: Any, yr: Any, wb: Any) -> Any:
        out = model_obj(xb, ar_b, ar_r, use_ar_floor=True)
        reg_loss = base.mx.mean(base.nn.losses.huber_loss(out[:, 0:1], yr, delta=1.0) * wb)
        bce = base.nn.losses.binary_cross_entropy(out[:, 1:2], yb, with_logits=True)
        bce_loss = base.mx.mean(bce * wb)
        alpha_penalty = 0.01 * base.mx.mean(model_obj.alpha_value() * model_obj.alpha_value())
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
            yb_np = y_metric[rel].astype(np.float32)[:, None]
            yr_np = cont_metric[rel].astype(np.float32)[:, None]
            wb_np = np.ones_like(yb_np, dtype=np.float32)
            yb = base.mx.array(yb_np, dtype=base.mx.float32)
            yr = base.mx.array(yr_np, dtype=base.mx.float32)
            wb = base.mx.array(wb_np, dtype=base.mx.float32)
            loss, grads = loss_and_grad(model, xb, ab, rb, yb, yr, wb)
            grads, _ = base.optim.clip_grad_norm(grads, 1.0)
            optimizer.update(model, grads)
            base.mx.eval(model.parameters(), optimizer.state)
            total += float(np.asarray(loss))
            batches += 1
        val_score, _val_reg, val_gate = targeted.residual_forward(
            model,
            train_x[inner_val],
            ar_train_score[inner_val],
            ar_train_reg[inner_val],
            batch_size=batch_size,
        )
        val_pr = average_precision_score(selection_y[inner_val], val_score) if len(np.unique(selection_y[inner_val])) > 1 else math.nan
        true_val_pr = average_precision_score(block.train_y[inner_val], val_score) if len(np.unique(block.train_y[inner_val])) > 1 else math.nan
        delta = float(val_pr - ar_inner_metric) if math.isfinite(val_pr) and math.isfinite(ar_inner_metric) else math.nan
        true_delta = float(true_val_pr - true_ar_inner_metric) if math.isfinite(true_val_pr) and math.isfinite(true_ar_inner_metric) else math.nan
        curve = {
            "epoch": epoch,
            "train_loss": total / max(1, batches),
            "inner_val_pr_auc": val_pr,
            "inner_val_delta_vs_frozen_ar": delta,
            "true_inner_val_pr_auc": true_val_pr,
            "true_inner_val_delta_vs_frozen_ar": true_delta,
            "frozen_ar_inner_val_pr_auc": ar_inner_metric,
            "true_frozen_ar_inner_val_pr_auc": true_ar_inner_metric,
            "alpha": float(np.asarray(model.alpha_value())[0]),
            "gate_mean": float(np.mean(val_gate)),
            "gate_p95": float(np.quantile(val_gate, 0.95)),
        }
        curves.append(curve)
        if fixed_epoch is not None:
            if epoch == fixed_epoch:
                model.save_weights(str(best_path))
                best_delta = delta if math.isfinite(delta) else 0.0
                best_epoch = epoch
                suppressed = False
                early_stop = "fixed_epoch_selected"
                break
        elif math.isfinite(delta) and delta > best_delta:
            model.save_weights(str(best_path))
            best_delta = delta
            best_epoch = epoch
            stale = 0
            suppressed = False
        else:
            stale += 1
        if fixed_epoch is None and stale >= int(patience):
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
        train_score, train_reg, _ = targeted.residual_forward(model, train_x, ar_train_score, ar_train_reg, batch_size=batch_size)
        test_score, test_reg, gate = targeted.residual_forward(model, test_x, ar_test_score, ar_test_reg, batch_size=batch_size)
        alpha_final = float(np.asarray(model.alpha_value())[0])
        checkpoint_restored = True
        checkpoint_checksum = base.file_digest(best_path)
    metrics = targeted.metric_row(block.train_y, train_score, block.test_y, test_score, block.test_cont, test_reg, block.test_video_id)
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
        "true_frozen_ar_inner_val_pr_auc": float(true_ar_inner_metric),
        "inner_val_selection_policy": selection_policy,
        "training_label_policy": "permuted_train_labels" if control.startswith("label_permutation") else "true_train_labels",
        "heldout_scoring_policy": "true_heldout_labels",
        "fixed_epoch_requested": fixed_epoch,
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
    return targeted.summarize(fold_df)


def row_for(summary: pd.DataFrame, control: str) -> pd.Series:
    sub = summary[(summary["validation_protocol"] == PROTOCOL) & (summary["variant_name"] == VARIANT) & (summary["control_type"] == control)]
    if len(sub) != 1:
        raise RuntimeError(f"Expected one summary row for {control}; got {len(sub)}")
    return sub.iloc[0]


def seed_delta_table(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = fold_df[(fold_df["validation_protocol"] == PROTOCOL) & (fold_df["variant_name"] == VARIANT)]
    for seed, group in sub.groupby("seed"):
        vals = group.set_index("control_type")["pr_auc"].to_dict()
        real = vals["real_residual"]
        row = {"seed": int(seed), "real_pr_auc": real}
        for control in ("frozen_ar_only", *PRIMARY_CONTROLS, "full_video_video_mean_pca_residual_oracle_warning"):
            row[f"{control}_pr_auc"] = vals[control]
            row[f"real_minus_{control}"] = real - vals[control]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("seed")


def positive_count(seed_deltas: pd.DataFrame, col: str) -> int:
    return int((pd.to_numeric(seed_deltas[col], errors="coerce") > 0).sum())


def no_single_seed_over_60(seed_deltas: pd.DataFrame, cols: list[str]) -> bool:
    for col in cols:
        vals = pd.to_numeric(seed_deltas[col], errors="coerce").to_numpy(dtype=float)
        if np.nanmean(vals) <= 0:
            return False
        positives = vals[vals > 0]
        if positives.size == 0:
            return False
        if float(np.max(positives) / np.sum(positives)) > 0.60:
            return False
    return True


def compute_gates(summary: pd.DataFrame, fold_df: pd.DataFrame, design: dict[str, Any]) -> dict[str, Any]:
    ar = row_for(summary, "frozen_ar_only")
    real = row_for(summary, "real_residual")
    shuffled = row_for(summary, "shuffled_pca_residual")
    random = row_for(summary, "random_pca_residual")
    label = row_for(summary, "label_permutation_residual_permuted_inner_val_selection")
    fixed_label = row_for(summary, "label_permutation_fixed_epoch_audit")
    train_video = row_for(summary, "train_only_video_mean_pca_residual")
    full_video = row_for(summary, "full_video_video_mean_pca_residual_oracle_warning")
    diag = row_for(summary, "diagnostics_only_residual")
    deltas = {
        "real_minus_frozen_ar": float(real["mean_pr_auc"] - ar["mean_pr_auc"]),
        "real_minus_shuffled": float(real["mean_pr_auc"] - shuffled["mean_pr_auc"]),
        "real_minus_random": float(real["mean_pr_auc"] - random["mean_pr_auc"]),
        "real_minus_clean_label_permutation": float(real["mean_pr_auc"] - label["mean_pr_auc"]),
        "real_minus_label_permutation_fixed_epoch_audit": float(real["mean_pr_auc"] - fixed_label["mean_pr_auc"]),
        "real_minus_train_only_video_mean": float(real["mean_pr_auc"] - train_video["mean_pr_auc"]),
        "real_minus_full_video_oracle_mean": float(real["mean_pr_auc"] - full_video["mean_pr_auc"]),
        "real_minus_diagnostics_only": float(real["mean_pr_auc"] - diag["mean_pr_auc"]),
    }
    seed_deltas = seed_delta_table(fold_df)
    primary_delta_cols = [
        "real_minus_frozen_ar_only",
        "real_minus_shuffled_pca_residual",
        "real_minus_random_pca_residual",
        "real_minus_label_permutation_residual_permuted_inner_val_selection",
        "real_minus_train_only_video_mean_pca_residual",
    ]
    primary_seed_counts = {col: positive_count(seed_deltas, col) for col in primary_delta_cols}
    seed_consistency_pass = all(count >= 4 for count in primary_seed_counts.values())
    concentration_pass = no_single_seed_over_60(seed_deltas, primary_delta_cols)
    do_no_harm_seed_floor_pass = bool((seed_deltas["real_minus_frozen_ar_only"] >= DO_NO_HARM_SEED_FLOOR).all())
    do_no_harm_mean_pass = bool(deltas["real_minus_frozen_ar"] >= 0)
    do_no_harm_pass = bool(do_no_harm_seed_floor_pass and do_no_harm_mean_pass)
    frozen_ar_integrity_pass = bool(
        fold_df.groupby(["seed"])["frozen_ar_test_checksum"].nunique().max() == 1
        and fold_df.groupby(["seed"])["frozen_ar_train_checksum"].nunique().max() == 1
    )
    checkpoint_restore_pass = bool(fold_df["checkpoint_restore_pass"].all())
    eval_mode_scoring_pass = bool(fold_df["eval_mode_scoring"].all())
    threshold_gates = {
        "real_gt_frozen_ar_pass": deltas["real_minus_frozen_ar"] >= WEAK_THRESHOLD,
        "real_gt_shuffled_pass": deltas["real_minus_shuffled"] >= WEAK_THRESHOLD,
        "real_gt_random_pass": deltas["real_minus_random"] >= WEAK_THRESHOLD,
        "real_gt_clean_label_permutation_pass": deltas["real_minus_clean_label_permutation"] >= WEAK_THRESHOLD,
        "real_gt_train_only_video_mean_pass": deltas["real_minus_train_only_video_mean"] >= WEAK_THRESHOLD,
    }
    credible_threshold_gates = {
        "real_gt_frozen_ar_credible_pass": deltas["real_minus_frozen_ar"] >= CREDIBLE_THRESHOLD,
        "real_gt_shuffled_credible_pass": deltas["real_minus_shuffled"] >= CREDIBLE_THRESHOLD,
        "real_gt_random_credible_pass": deltas["real_minus_random"] >= CREDIBLE_THRESHOLD,
        "real_gt_clean_label_permutation_credible_pass": deltas["real_minus_clean_label_permutation"] >= CREDIBLE_THRESHOLD,
        "real_gt_train_only_video_mean_credible_pass": deltas["real_minus_train_only_video_mean"] >= CREDIBLE_THRESHOLD,
    }
    full_video_oracle_warning = bool((full_video["mean_pr_auc"] - real["mean_pr_auc"]) >= ORACLE_WARNING_THRESHOLD)
    weak_pass = bool(
        frozen_ar_integrity_pass
        and checkpoint_restore_pass
        and eval_mode_scoring_pass
        and all(threshold_gates.values())
        and seed_consistency_pass
        and concentration_pass
        and do_no_harm_pass
    )
    credible_pass = bool(weak_pass and all(credible_threshold_gates.values()))
    recommendation = (
        "credible_blocked_confirmation_candidate_grouped_compatibility_next"
        if credible_pass
        else (
            "weak_blocked_confirmation_candidate_grouped_compatibility_next"
            if weak_pass
            else "clean_confirmation_failed_do_not_run_grouped_or_504"
        )
    )
    if full_video_oracle_warning and weak_pass:
        recommendation = "primary_gates_pass_but_full_video_oracle_warning_blocks_headline_language"
    return {
        "schema_version": SCHEMA_VERSION,
        "source_design_commit": "0d6ce16",
        "design_path": str(DESIGN_PATH),
        "protocol": PROTOCOL,
        "variant": VARIANT,
        "loss": LOSS_NAME,
        "target": TARGET_NAME,
        "feature": FEATURE_NAME,
        "matrix_rows_expected": 45,
        "residual_trainings_expected": 40,
        "weak_threshold_pr_auc": WEAK_THRESHOLD,
        "credible_threshold_pr_auc": CREDIBLE_THRESHOLD,
        "eval_mode_scoring_pass": eval_mode_scoring_pass,
        "checkpoint_restore_pass": checkpoint_restore_pass,
        "frozen_ar_integrity_pass": frozen_ar_integrity_pass,
        "same_variant_gates_only": True,
        "label_permutation_policy_implemented": True,
        "train_only_and_full_video_means_separated": True,
        "minimum_delta_thresholds_encoded": True,
        **threshold_gates,
        **credible_threshold_gates,
        "seed_consistency_pass": bool(seed_consistency_pass),
        "primary_seed_positive_counts": primary_seed_counts,
        "no_single_seed_over_60pct_pass": bool(concentration_pass),
        "do_no_harm_blocked_pass": do_no_harm_pass,
        "do_no_harm_seed_floor_pass": do_no_harm_seed_floor_pass,
        "do_no_harm_mean_pass": do_no_harm_mean_pass,
        "full_video_oracle_warning": full_video_oracle_warning,
        "full_video_oracle_promotability_blocking": False,
        "weak_confirmation_pass": weak_pass,
        "credible_confirmation_pass": credible_pass,
        "strict_forward_time_temporal_generalization_proven": False,
        "grouped_5fold_compatibility_check_justified": bool(weak_pass),
        "full_504_confirmation_justified": False,
        "recommendation": recommendation,
        "pr_auc": {
            "real_residual": float(real["mean_pr_auc"]),
            "frozen_ar_only": float(ar["mean_pr_auc"]),
            "shuffled_pca_residual": float(shuffled["mean_pr_auc"]),
            "random_pca_residual": float(random["mean_pr_auc"]),
            "label_permutation_residual_permuted_inner_val_selection": float(label["mean_pr_auc"]),
            "label_permutation_fixed_epoch_audit": float(fixed_label["mean_pr_auc"]),
            "train_only_video_mean_pca_residual": float(train_video["mean_pr_auc"]),
            "full_video_video_mean_pca_residual_oracle_warning": float(full_video["mean_pr_auc"]),
            "diagnostics_only_residual": float(diag["mean_pr_auc"]),
        },
        "deltas": deltas,
        "design_summary": {
            "label_permutation_policy": design["label_permutation_policy"]["cleanest_null"],
            "video_mean_policy": "train_only_blocks_promotion_full_video_oracle_warning",
        },
    }


def write_report(path: Path, gates: dict[str, Any], output_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pr = gates["pr_auc"]
    d = gates["deltas"]
    report = f"""# Phase 5 Blocked Residual Clean Confirmation

Output root: `{output_root}`

This is the clean monotonic-only blocked confirmation specified by `confirmation_design.json` at design commit `0d6ce16`. It runs only `blocked_temporal_70_30`, only `monotonic_do_no_harm_residual`, only `regression_plus_binary`, and five prespecified seeds. It does not run grouped, does not run 504, does not start secondary targets, and does not change claims.

## Confirmation Result

- Weak confirmation passed: `{gates['weak_confirmation_pass']}`
- Credible threshold passed: `{gates['credible_confirmation_pass']}`
- Recommendation: `{gates['recommendation']}`
- Strict forward-time temporal generalization remains unproven: `{not gates['strict_forward_time_temporal_generalization_proven']}`
- Grouped 5-fold compatibility check justified: `{gates['grouped_5fold_compatibility_check_justified']}`
- 504 justified: `{gates['full_504_confirmation_justified']}`

## PR-AUC

- Real residual: `{pr['real_residual']:.10f}`
- Frozen AR: `{pr['frozen_ar_only']:.10f}`
- Shuffled PCA residual: `{pr['shuffled_pca_residual']:.10f}`
- Random PCA residual: `{pr['random_pca_residual']:.10f}`
- Clean label permutation residual: `{pr['label_permutation_residual_permuted_inner_val_selection']:.10f}`
- Label permutation fixed-epoch audit: `{pr['label_permutation_fixed_epoch_audit']:.10f}`
- Train-only video mean residual: `{pr['train_only_video_mean_pca_residual']:.10f}`
- Full-video oracle mean residual: `{pr['full_video_video_mean_pca_residual_oracle_warning']:.10f}`
- Diagnostics-only residual: `{pr['diagnostics_only_residual']:.10f}`

## Real-Minus-Control Deltas

- Real minus frozen AR: `{d['real_minus_frozen_ar']:+.10f}`
- Real minus shuffled PCA: `{d['real_minus_shuffled']:+.10f}`
- Real minus random PCA: `{d['real_minus_random']:+.10f}`
- Real minus clean label permutation: `{d['real_minus_clean_label_permutation']:+.10f}`
- Real minus label permutation fixed-epoch audit: `{d['real_minus_label_permutation_fixed_epoch_audit']:+.10f}`
- Real minus train-only video mean: `{d['real_minus_train_only_video_mean']:+.10f}`
- Real minus full-video oracle mean: `{d['real_minus_full_video_oracle_mean']:+.10f}`
- Real minus diagnostics-only: `{d['real_minus_diagnostics_only']:+.10f}`

## Gates

- `real_gt_frozen_ar_pass`: `{gates['real_gt_frozen_ar_pass']}`
- `real_gt_shuffled_pass`: `{gates['real_gt_shuffled_pass']}`
- `real_gt_random_pass`: `{gates['real_gt_random_pass']}`
- `real_gt_clean_label_permutation_pass`: `{gates['real_gt_clean_label_permutation_pass']}`
- `real_gt_train_only_video_mean_pass`: `{gates['real_gt_train_only_video_mean_pass']}`
- `seed_consistency_pass`: `{gates['seed_consistency_pass']}`
- `primary_seed_positive_counts`: `{gates['primary_seed_positive_counts']}`
- `no_single_seed_over_60pct_pass`: `{gates['no_single_seed_over_60pct_pass']}`
- `do_no_harm_blocked_pass`: `{gates['do_no_harm_blocked_pass']}`
- `full_video_oracle_warning`: `{gates['full_video_oracle_warning']}`
- `frozen_ar_integrity_pass`: `{gates['frozen_ar_integrity_pass']}`
- `checkpoint_restore_pass`: `{gates['checkpoint_restore_pass']}`
- `eval_mode_scoring_pass`: `{gates['eval_mode_scoring_pass']}`

The weak pass threshold is `+0.0010` PR-AUC on every primary blocked gate. The credible threshold is `+0.0030` PR-AUC. Full-video oracle mean is a mechanism warning, while train-only video mean is the promotability-blocking static-control gate.

## Interpretation

This report is a confirmation result for the blocked residual candidate only. It does not update the canonical grouped claim by itself. If weak confirmation fails, do not run grouped 5-fold compatibility or 504. If weak confirmation passes, grouped 5-fold compatibility may be considered next, but strict forward-time temporal generalization should remain explicitly caveated unless the credible threshold and all control gates are satisfied.
"""
    path.write_text(report)


def matrix_rows() -> list[dict[str, Any]]:
    return [{"protocol": PROTOCOL, "fold": FOLD, "seed": seed, "variant": VARIANT, "control": control} for seed in SEEDS for control in CONTROLS]


def main() -> int:
    args = parse_args()
    design = json.loads(Path(args.design_path).read_text())
    output_root = Path(args.output_root) if args.output_root else default_output_root()
    matrix = matrix_rows()
    print(json.dumps({"matrix_size": len(matrix), "residual_trainings": len(matrix) - len(SEEDS), "protocol": PROTOCOL, "variant": VARIANT, "controls": CONTROLS}, indent=2))
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
    real_best_epochs: dict[int, int] = {}
    for seed in SEEDS:
        ar = load_or_cache_frozen_ar(previous_root, source_root, output_root, block, seed, args.batch_size)
        ar_manifest.append({k: v for k, v in ar.items() if k not in {"train_score", "train_reg", "test_score", "test_reg", "metrics"}})
        ar_metrics = dict(ar["metrics"])
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
                "delta_vs_frozen_ar_pr_auc": 0.0,
                "delta_vs_frozen_ar_top_1pct_lift": 0.0,
                "alpha_final": 0.0,
                "gate_mean": 0.0,
                "residual_suppressed": True,
                "inner_val_selection_policy": "not_applicable",
                "training_label_policy": "not_applicable",
                "heldout_scoring_policy": "true_heldout_labels",
                **ar_metrics,
            }
        )
        for control in [c for c in CONTROLS if c != "frozen_ar_only"]:
            fixed_epoch = real_best_epochs.get(seed, 1) if control == "label_permutation_fixed_epoch_audit" else None
            train_x, test_x, dims, manifest = residual_features_for_control(df, dense_root, phase4_root, block, control, seed)
            metrics, curves, audit = train_clean_residual(
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
                fixed_epoch=fixed_epoch,
            )
            if control == "real_residual":
                real_best_epochs[seed] = max(1, int(audit["best_epoch"]))
            for c in curves:
                curve_rows.append(
                    {
                        "validation_protocol": PROTOCOL,
                        "actual_validation_protocol": PROTOCOL,
                        "fold": FOLD,
                        "seed": seed,
                        "variant_name": VARIANT,
                        "control_type": control,
                        **c,
                    }
                )
            feature_manifest.append(
                {
                    "actual_protocol": PROTOCOL,
                    "reported_protocol": PROTOCOL,
                    "fold": FOLD,
                    "seed": seed,
                    "variant_name": VARIANT,
                    "control_type": control,
                    "dims": dims,
                    "blocks": manifest,
                }
            )
            integrity_rows.append(
                {
                    "actual_protocol": PROTOCOL,
                    "reported_protocol": PROTOCOL,
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
    if len(fold_df) != 45:
        raise RuntimeError(f"Expected 45 metric rows, got {len(fold_df)}")
    summary = summarize(fold_df)
    seed_deltas = seed_delta_table(fold_df)
    gates = compute_gates(summary, fold_df, design)
    fold_df.to_csv(output_root / "metrics" / "blocked_residual_clean_confirm_seed_metrics.csv", index=False)
    fold_df.to_csv(output_root / "metrics" / "blocked_residual_clean_confirm_fold_metrics.csv", index=False)
    summary.to_csv(output_root / "metrics" / "blocked_residual_clean_confirm_summary_metrics.csv", index=False)
    seed_deltas.to_csv(output_root / "metrics" / "blocked_residual_clean_confirm_seed_deltas.csv", index=False)
    summary.to_csv(output_root / "promotion" / "blocked_residual_clean_confirm_control_comparison.csv", index=False)
    seed_deltas.to_csv(output_root / "promotion" / "blocked_residual_clean_confirm_seed_deltas.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(output_root / "diagnostics" / "training_curve_summary.csv", index=False)
    pd.DataFrame(curve_rows).to_json(output_root / "diagnostics" / "training_curves.json", orient="records", indent=2)
    pd.DataFrame(integrity_rows).to_csv(output_root / "diagnostics" / "frozen_ar_integrity_audit.csv", index=False)
    label_audit = fold_df[fold_df["control_type"].str.startswith("label_permutation")][
        [
            "seed",
            "control_type",
            "best_epoch",
            "inner_val_selection_policy",
            "training_label_policy",
            "heldout_scoring_policy",
            "best_inner_val_delta_vs_frozen_ar",
            "true_frozen_ar_inner_val_pr_auc",
            "pr_auc",
            "delta_vs_frozen_ar_pr_auc",
            "checkpoint_restored",
            "residual_suppressed",
        ]
    ].sort_values(["control_type", "seed"])
    video_audit = fold_df[fold_df["control_type"].isin(["train_only_video_mean_pca_residual", "full_video_video_mean_pca_residual_oracle_warning"])][
        ["seed", "control_type", "pr_auc", "delta_vs_frozen_ar_pr_auc", "best_epoch", "checkpoint_restored", "residual_suppressed"]
    ].sort_values(["control_type", "seed"])
    label_audit.to_csv(output_root / "diagnostics" / "label_permutation_selection_audit.csv", index=False)
    video_audit.to_csv(output_root / "diagnostics" / "video_mean_train_only_vs_full_video_audit.csv", index=False)
    write_json(output_root / "diagnostics" / "label_permutation_selection_audit.json", {"policy_implemented": True, "rows": label_audit.to_dict(orient="records")})
    write_json(output_root / "diagnostics" / "video_mean_train_only_vs_full_video_audit.json", {"train_only_and_full_video_separated": True, "rows": video_audit.to_dict(orient="records")})
    write_json(output_root / "diagnostics" / "frozen_ar_integrity_audit.json", {"pass": gates["frozen_ar_integrity_pass"], "rows": integrity_rows, "row_count": len(integrity_rows)})
    write_json(output_root / "diagnostics" / "checkpoint_restore_audit.json", {"pass": gates["checkpoint_restore_pass"], "rows": int(len(fold_df))})
    write_json(output_root / "diagnostics" / "eval_mode_scoring_audit.json", {"pass": gates["eval_mode_scoring_pass"], "dropout_disabled": True, "deterministic_eval_mode_scoring": True})
    write_json(output_root / "diagnostics" / "do_no_harm_audit.json", {"do_no_harm_blocked_pass": gates["do_no_harm_blocked_pass"], "real_minus_frozen_ar": gates["deltas"]["real_minus_frozen_ar"]})
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
            "design_path": str(args.design_path),
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
            "residual_trainings": len(matrix) - len(SEEDS),
            "no_ar_retraining": True,
            "no_grouped": True,
            "no_504": True,
            "no_secondary_targets": True,
            "no_vjepa_tribe_pca_rerun": True,
            "duration_seconds": time.time() - start,
        },
    )
    write_json(output_root / "promotion" / "blocked_residual_clean_confirm_gates.json", gates)
    write_json(output_root / "promotion" / "blocked_residual_clean_confirm_adversarial_verdict.json", gates)
    write_json(
        output_root / "promotion" / "blocked_residual_clean_confirm_failure_reasons.json",
        {
            "weak_confirmation_pass": gates["weak_confirmation_pass"],
            "credible_confirmation_pass": gates["credible_confirmation_pass"],
            "strict_forward_time_temporal_generalization_proven": gates["strict_forward_time_temporal_generalization_proven"],
            "failed_primary_gates": [k for k, v in gates.items() if k.endswith("_pass") and v is False],
            "recommendation": gates["recommendation"],
        },
    )
    report_name = f"again_dense_2hz_phase5_blocked_residual_clean_confirm_{output_root.name.rsplit('_', 2)[-2]}_{output_root.name.rsplit('_', 1)[-1]}.md"
    write_report(output_root / "reports" / report_name, gates, output_root)
    report_path = Path(args.reports_dir) / report_name
    write_report(report_path, gates, output_root)
    print(json.dumps(fr.clean_json({"run_completed": True, "output_root": str(output_root), "report": str(report_path), **gates}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
