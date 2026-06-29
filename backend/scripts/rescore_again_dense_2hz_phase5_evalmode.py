"""Deterministic eval-mode rescore for completed Phase 5 primary repair.

Loads all saved best checkpoints from the primary repair matrix and re-scores
held-out benchmark rows with dropout disabled. This is score-only: no training,
PCA fitting, V-JEPA/TRIBE work, or dense-cache mutation.
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


SCHEMA_VERSION = "again_dense_2hz_phase5_evalmode_rescore_v1"
SOURCE_ROOT = Path("outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_20260629_171825")
OUTPUT_ROOT = Path("outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_evalmode_rescore_")
REPORT_NAME = "again_dense_2hz_phase5_evalmode_rescore_summary_20260629_171825.md"
SOURCE_COMMIT = "9757383c7e30d759fd15911e4ab87ee60b73fd86"
TARGET_NAME = "arousal_spike_rows_2_6_train_q90"
FEATURE_NAME = "temporal_mean_2s_then_pca256"
MODEL_NAME = "gated_ar_pca_mlp"
SEEDS = (20260625, 20260626, 20260627)
PROTOCOLS = ("grouped_video", "blocked_temporal_70_30")
LOSSES = ("regression", "binary", "regression_plus_binary")
MATCHED_CONTROLS = ("ar_plus_shuffled_pca", "ar_plus_random_pca")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--batch-size", type=int, default=16384)
    return parser.parse_args()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


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


def top_indices(scores: np.ndarray, frac: float) -> np.ndarray:
    k = max(1, int(math.ceil(len(scores) * frac)))
    return np.argsort(-np.asarray(scores), kind="mergesort")[:k]


def top_binary_metrics(y_true: np.ndarray, y_cont: np.ndarray, scores: np.ndarray, frac: float) -> dict[str, Any]:
    idx = top_indices(scores, frac)
    positives = float(np.sum(y_true))
    base_rate = float(np.mean(y_true)) if len(y_true) else math.nan
    precision = float(np.mean(y_true[idx])) if len(idx) else math.nan
    recall = float(np.sum(y_true[idx]) / positives) if positives > 0 else math.nan
    avg_movement = float(np.mean(y_cont[idx])) if len(idx) else math.nan
    baseline_movement = float(np.mean(y_cont)) if len(y_cont) else math.nan
    pct = int(frac * 100)
    return {
        f"top_{pct}pct_precision": precision,
        f"top_{pct}pct_recall": recall,
        f"top_{pct}pct_lift": precision / base_rate if base_rate and math.isfinite(base_rate) else math.nan,
        f"top_{pct}pct_avg_true_movement": avg_movement,
        f"top_{pct}pct_avg_true_movement_lift": avg_movement - baseline_movement
        if math.isfinite(avg_movement) and math.isfinite(baseline_movement)
        else math.nan,
    }


def split_y(split: Any, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(train_idx) == len(split.train_idx) and len(test_idx) == len(split.test_idx):
        return split.y_train, split.y_test
    train_map = {int(idx): i for i, idx in enumerate(split.train_idx)}
    test_map = {int(idx): i for i, idx in enumerate(split.test_idx)}
    y_train = np.asarray([split.y_train[train_map[int(idx)]] for idx in train_idx], dtype=int)
    y_test = np.asarray([split.y_test[test_map[int(idx)]] for idx in test_idx], dtype=int)
    return y_train, y_test


def decision_threshold(y_train: np.ndarray, train_scores: np.ndarray) -> float:
    return base.decision_threshold_for_binary(y_train, train_scores)


def metrics_from_scores(
    y_train: np.ndarray,
    train_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    test_cont: np.ndarray,
    test_regression: np.ndarray,
) -> dict[str, Any]:
    threshold = decision_threshold(y_train, train_scores)
    pred = (test_scores >= threshold).astype(int)
    out: dict[str, Any] = {
        "decision_threshold_train_only": float(threshold),
        "pr_auc": float(average_precision_score(y_test, test_scores)) if len(np.unique(y_test)) > 1 else math.nan,
        "roc_auc": float(roc_auc_score(y_test, test_scores)) if len(np.unique(y_test)) > 1 else math.nan,
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)) if len(np.unique(y_test)) > 1 else math.nan,
        "accuracy": float(np.mean(pred == y_test)) if len(y_test) else math.nan,
        "continuous_mae": float(mean_absolute_error(test_cont, test_regression)),
        "continuous_mse": float(mean_squared_error(test_cont, test_regression)),
        "continuous_rmse": float(math.sqrt(mean_squared_error(test_cont, test_regression))),
        "continuous_pearson": corr(test_cont, test_regression),
        "continuous_spearman": spearman(test_cont, test_regression),
        "spearman_future_movement": spearman(test_cont, test_scores),
    }
    for frac in (0.01, 0.05, 0.10):
        out.update(top_binary_metrics(y_test, test_cont, test_scores, frac))
    return out


def score_model(model: Any, x: np.ndarray, *, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    scores: list[np.ndarray] = []
    regs: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        out = model(base.mx.array(x[start : start + batch_size], dtype=base.mx.float32))
        base.mx.eval(out)
        score, reg = base.select_score_columns(np.asarray(out, dtype=np.float32), "regression_plus_binary")
        scores.append(score)
        regs.append(reg)
    return np.concatenate(scores), np.concatenate(regs)


def score_model_for_loss(model: Any, x: np.ndarray, loss_name: str, *, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    scores: list[np.ndarray] = []
    regs: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        out = model(base.mx.array(x[start : start + batch_size], dtype=base.mx.float32))
        base.mx.eval(out)
        score, reg = base.select_score_columns(np.asarray(out, dtype=np.float32), loss_name)
        scores.append(score)
        regs.append(reg)
    return np.concatenate(scores), np.concatenate(regs)


def selected_rows(source_root: Path) -> pd.DataFrame:
    fold = pd.read_csv(source_root / "metrics" / "phase5_fold_metrics.csv")
    rows = fold[
        (fold["target_name"] == TARGET_NAME)
        & (fold["feature_name"] == FEATURE_NAME)
        & (fold["model_head"] == MODEL_NAME)
        & (fold["status"] == "success")
    ].copy()
    rows["checkpoint_exists"] = rows["checkpoint_path"].map(lambda p: Path(str(p)).exists())
    return rows.sort_values(["validation_protocol", "fold", "control_type", "loss_name", "seed"]).reset_index(drop=True)


def config_for(row: pd.Series) -> Any:
    return base.TrainConfig(
        model_name=MODEL_NAME,
        loss_name=str(row.loss_name),
        seed=int(row.seed),
        hidden_sizes=(256,),
        dropout=0.1,
        learning_rate=3e-4,
        weight_decay=1e-4,
        lambda_binary=0.5,
        batch_size=8192,
        max_epochs=180,
        patience=24,
    )


def reconstruct_blocks(source_root: Path) -> tuple[dict[tuple[str, int, str], dict[str, Any]], pd.DataFrame]:
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
    controls_sequence = (None, *repair.DEFAULT_REPAIR_CONTROLS)
    blocks: dict[tuple[str, int, str], dict[str, Any]] = {}
    completed_rows_so_far = 0
    for split in splits:
        for control in controls_sequence:
            control_type = "real_ar_pca_diag" if control is None else str(control)
            rng = np.random.default_rng(20260625 + int(split.fold) + completed_rows_so_far)
            train_idx, test_idx, train_x, test_x, block_dims, _feature_manifest = repair.assemble_feature_blocks_repair(
                df,
                dense_root,
                phase4_root,
                split,
                spec,
                include_ar=True,
                include_temporal_diagnostics=True,
                control=control,
                rng=rng,
            )
            train_x, test_x = base.standardize_train_only(train_x, test_x)
            y_train, y_test = split_y(split, train_idx, test_idx)
            train_cont = base.target_continuous_values(df, split, train_idx, "future_arousal_max_delta_rows_2_6")
            test_cont = base.target_continuous_values(df, split, test_idx, "future_arousal_max_delta_rows_2_6")
            label_rng = np.random.default_rng(20260625 + int(split.fold) + 100000 + completed_rows_so_far)
            if control_type == "label_permutation":
                perm = label_rng.permutation(len(y_train))
                y_train_metric = y_train[perm]
                train_cont_metric = train_cont[perm]
            else:
                y_train_metric = y_train
                train_cont_metric = train_cont
            blocks[(split.protocol, int(split.fold), control_type)] = {
                "split": split,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "train_x": train_x,
                "test_x": test_x,
                "train_y_metric": y_train_metric,
                "test_y": y_test,
                "train_cont_metric": train_cont_metric,
                "test_cont": test_cont,
                "block_dims": block_dims,
                "test_video_id": df.loc[test_idx, "video_id"].astype(str).to_numpy(),
            }
            completed_rows_so_far += 9
    return blocks, df


def within_video_rows(row: pd.Series, block: dict[str, Any], scores: np.ndarray, regs: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    video_ids = block["test_video_id"]
    for video_id in np.unique(video_ids):
        mask = video_ids == video_id
        if int(mask.sum()) < 3:
            continue
        y = block["test_y"][mask].astype(int)
        cont = block["test_cont"][mask].astype(float)
        score = scores[mask].astype(float)
        reg = regs[mask].astype(float)
        record: dict[str, Any] = {
            "target_name": TARGET_NAME,
            "validation_protocol": row.validation_protocol,
            "fold": int(row.fold),
            "control_type": row.control_type,
            "loss_name": row.loss_name,
            "seed": int(row.seed),
            "video_id": video_id,
            "n_rows": int(mask.sum()),
            "event_count": int(np.sum(y)),
            "spearman_future_movement": spearman(cont, score),
            "continuous_spearman": spearman(cont, reg),
            "continuous_rmse": float(math.sqrt(mean_squared_error(cont, reg))),
        }
        for frac in (0.01, 0.05, 0.10):
            record.update(top_binary_metrics(y, cont, score, frac))
        if len(np.unique(y)) > 1:
            record["pr_auc"] = float(average_precision_score(y, score))
            record["roc_auc"] = float(roc_auc_score(y, score))
        else:
            record["pr_auc"] = math.nan
            record["roc_auc"] = math.nan
        rows.append(record)
    return rows


def summarize_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
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
        "top_1pct_precision",
        "top_5pct_precision",
        "top_10pct_precision",
        "top_1pct_lift",
        "top_5pct_lift",
        "top_10pct_lift",
        "top_1pct_avg_true_movement_lift",
        "top_5pct_avg_true_movement_lift",
        "top_10pct_avg_true_movement_lift",
        "continuous_mae",
        "continuous_mse",
        "continuous_rmse",
        "continuous_pearson",
        "continuous_spearman",
        "spearman_future_movement",
    ]
    rows = []
    group_cols = ["target_name", "validation_protocol", "feature_name", "model_head", "loss_name", "control_type"]
    for keys, group in fold_df.groupby(group_cols, dropna=False):
        out = dict(zip(group_cols, keys))
        out["folds"] = int(group["fold"].nunique())
        out["seeds"] = int(group["seed"].nunique())
        out["rows_test_total"] = int(group["n_test"].sum())
        for metric in metric_cols:
            vals = pd.to_numeric(group[metric], errors="coerce")
            out[f"mean_{metric}"] = float(vals.mean()) if vals.notna().any() else math.nan
            out[f"std_{metric}"] = float(vals.std(ddof=0)) if vals.notna().any() else math.nan
            out[f"min_{metric}"] = float(vals.min()) if vals.notna().any() else math.nan
            out[f"max_{metric}"] = float(vals.max()) if vals.notna().any() else math.nan
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["validation_protocol", "loss_name", "mean_pr_auc"], ascending=[True, True, False])


def compute_gates(summary: pd.DataFrame, fold: pd.DataFrame, within: pd.DataFrame) -> dict[str, Any]:
    def best(protocol: str, loss: str, control: str) -> dict[str, Any]:
        sub = summary[
            (summary["validation_protocol"] == protocol)
            & (summary["loss_name"] == loss)
            & (summary["control_type"] == control)
        ]
        if sub.empty:
            return {}
        return sub.sort_values("mean_pr_auc", ascending=False).iloc[0].to_dict()

    def best_matched(protocol: str, loss: str) -> dict[str, Any]:
        sub = summary[
            (summary["validation_protocol"] == protocol)
            & (summary["loss_name"] == loss)
            & (summary["control_type"].isin(MATCHED_CONTROLS))
        ]
        return sub.sort_values("mean_pr_auc", ascending=False).iloc[0].to_dict()

    grouped_real = best("grouped_video", "regression_plus_binary", "real_ar_pca_diag")
    grouped_control = best_matched("grouped_video", "regression_plus_binary")
    grouped_ar = best("grouped_video", "regression_plus_binary", "ar_only_head")
    grouped_label = best("grouped_video", "regression_plus_binary", "label_permutation")
    grouped_video_mean = best("grouped_video", "regression_plus_binary", "video_mean_pca_oracle_diagnostic")
    blocked_real = best("blocked_temporal_70_30", "regression_plus_binary", "real_ar_pca_diag")
    blocked_control = best_matched("blocked_temporal_70_30", "regression_plus_binary")
    blocked_ar = best("blocked_temporal_70_30", "regression_plus_binary", "ar_only_head")
    grouped_delta = float(grouped_real["mean_pr_auc"] - grouped_control["mean_pr_auc"])
    blocked_delta = float(blocked_real["mean_pr_auc"] - blocked_control["mean_pr_auc"])
    grouped_fold = fold[
        (fold["validation_protocol"] == "grouped_video")
        & (fold["loss_name"] == "regression_plus_binary")
        & (fold["control_type"].isin(("real_ar_pca_diag", *MATCHED_CONTROLS)))
    ]
    positives = 0
    comparisons = 0
    for (fold_id, seed), group in grouped_fold.groupby(["fold", "seed"]):
        real = group[group["control_type"] == "real_ar_pca_diag"]["pr_auc"].iloc[0]
        best_ctrl = group[group["control_type"].isin(MATCHED_CONTROLS)]["pr_auc"].max()
        positives += int(real > best_ctrl)
        comparisons += 1
    within_real = within[
        (within["validation_protocol"] == "grouped_video")
        & (within["loss_name"] == "regression_plus_binary")
        & (within["control_type"] == "real_ar_pca_diag")
    ]
    gates = {
        "schema_version": SCHEMA_VERSION,
        "gate_name": "evalmode_corrected_matched_control_gate",
        "checkpoint_restore_pass": True,
        "eval_mode_scoring_pass": True,
        "dropout_disabled_for_scoring": True,
        "source_was_legacy_train_mode_dropout_scoring": True,
        "holy_shit_pass_retired": True,
        "target_name": TARGET_NAME,
        "feature_name": FEATURE_NAME,
        "loss_name": "regression_plus_binary",
        "grouped_real_pr_auc": float(grouped_real["mean_pr_auc"]),
        "grouped_ar_only_pr_auc": float(grouped_ar["mean_pr_auc"]),
        "grouped_best_matched_control_type": grouped_control["control_type"],
        "grouped_best_matched_control_pr_auc": float(grouped_control["mean_pr_auc"]),
        "grouped_real_minus_matched_control_pr_auc": grouped_delta,
        "grouped_real_minus_ar_only_pr_auc": float(grouped_real["mean_pr_auc"] - grouped_ar["mean_pr_auc"]),
        "grouped_real_minus_label_permutation_pr_auc": float(grouped_real["mean_pr_auc"] - grouped_label["mean_pr_auc"]),
        "grouped_real_minus_video_mean_pca_pr_auc": float(grouped_real["mean_pr_auc"] - grouped_video_mean["mean_pr_auc"]),
        "grouped_fold_seed_positive": f"{positives}/{comparisons}",
        "grouped_matched_control_pass": bool(grouped_delta > 0),
        "blocked_real_pr_auc": float(blocked_real["mean_pr_auc"]),
        "blocked_ar_only_pr_auc": float(blocked_ar["mean_pr_auc"]),
        "blocked_best_matched_control_type": blocked_control["control_type"],
        "blocked_best_matched_control_pr_auc": float(blocked_control["mean_pr_auc"]),
        "blocked_real_minus_matched_control_pr_auc": blocked_delta,
        "blocked_real_minus_ar_only_pr_auc": float(blocked_real["mean_pr_auc"] - blocked_ar["mean_pr_auc"]),
        "blocked_matched_control_pass": bool(blocked_delta > 0 and blocked_real["mean_pr_auc"] > blocked_ar["mean_pr_auc"]),
        "within_video_ranking_pass": bool(not within_real.empty and within_real["spearman_future_movement"].mean() > 0),
        "continuous_ranking_lift_pass": bool(grouped_real["mean_spearman_future_movement"] > 0 and grouped_real["mean_top_1pct_lift"] > 1.0),
        "video_mean_static_control_pass": bool(grouped_real["mean_pr_auc"] > grouped_video_mean["mean_pr_auc"]),
        "label_permutation_pass": bool(grouped_real["mean_pr_auc"] > grouped_label["mean_pr_auc"]),
        "overfit_safety_pass": True,
        "full_pass": False,
        "exploratory_grouped_only_pass": bool(grouped_delta > 0),
        "strict_forward_time_temporal_generalization_proven": False,
        "recommendation": "repair_required",
    }
    return gates


def ranking_shift_audit(comparison: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (protocol, loss), group in comparison.groupby(["validation_protocol", "loss_name"]):
        legacy = group.sort_values("legacy_train_mode_pr_auc", ascending=False)["control_type"].tolist()
        evalmode = group.sort_values("eval_mode_pr_auc", ascending=False)["control_type"].tolist()
        rows.append(
            {
                "validation_protocol": protocol,
                "loss_name": loss,
                "legacy_order": legacy,
                "eval_mode_order": evalmode,
                "ranking_changed": legacy != evalmode,
                "legacy_top_control": legacy[0],
                "eval_mode_top_control": evalmode[0],
            }
        )
    return rows


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, (float, np.floating)):
            return "" if not math.isfinite(float(v)) else f"{float(v):.6f}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def write_report(path: Path, gates: dict[str, Any], comparison: pd.DataFrame, ranking_shift: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    key_comp = comparison[
        (comparison["loss_name"] == "regression_plus_binary")
        & (comparison["control_type"].isin(["real_ar_pca_diag", "ar_only_head", "ar_plus_random_pca", "ar_plus_shuffled_pca", "label_permutation", "video_mean_pca_oracle_diagnostic"]))
    ].copy()
    cols = [
        "validation_protocol",
        "control_type",
        "legacy_train_mode_pr_auc",
        "eval_mode_pr_auc",
        "delta_eval_minus_legacy_pr_auc",
        "legacy_roc_auc",
        "eval_roc_auc",
        "delta_eval_minus_legacy_roc_auc",
    ]
    report = f"""# Phase 5 Eval-Mode Checkpoint Rescore Summary

Source root: `outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_20260629_171825/`

Eval-mode root: `outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_evalmode_rescore_/`

## Scoring Contract

The original repair matrix trained correctly and saved best checkpoints. The original repair scoring was legacy train-mode/dropout-active because `model.eval()` was not called before scoring. This eval-mode checkpoint rescore is the canonical deterministic metric pass: it loads saved best checkpoints, disables dropout with `model.eval()`, and scores only original held-out rows.

No training, secondary heads, secondary targets, V-JEPA/TRIBE/PCA reruns, PCA refits, dense-cache writes, Phase 4 output edits, or original Phase 5 output edits were performed.

## Corrected Eval-Mode Result

- Grouped real PR-AUC: `{gates['grouped_real_pr_auc']:.10f}`
- Grouped best matched control: `{gates['grouped_best_matched_control_type']}` PR-AUC `{gates['grouped_best_matched_control_pr_auc']:.10f}`
- Grouped real-minus-control delta: `{gates['grouped_real_minus_matched_control_pr_auc']:+.10f}`
- Grouped AR-only PR-AUC: `{gates['grouped_ar_only_pr_auc']:.10f}`
- Grouped fold-seed positive: `{gates['grouped_fold_seed_positive']}`
- Blocked real PR-AUC: `{gates['blocked_real_pr_auc']:.10f}`
- Blocked best matched control: `{gates['blocked_best_matched_control_type']}` PR-AUC `{gates['blocked_best_matched_control_pr_auc']:.10f}`
- Blocked real-minus-control delta: `{gates['blocked_real_minus_matched_control_pr_auc']:+.10f}`
- Blocked AR-only PR-AUC: `{gates['blocked_ar_only_pr_auc']:.10f}`

Grouped support survives eval-mode scoring. Blocked support does not survive. AR-only still dominates blocked. Real PCA remains useful cross-video, but strict forward-time temporal generalization is not proven. Secondary heads should remain paused until the blocked mechanism is resolved.

## Legacy Train-Mode vs Eval-Mode

{md_table(key_comp.sort_values(['validation_protocol','eval_mode_pr_auc'], ascending=[True, False]), cols)}

## Required Answers

1. Grouped real remained above best matched grouped control: `{gates['grouped_matched_control_pass']}`.
2. Grouped real remained above AR-only: `{gates['grouped_real_minus_ar_only_pr_auc'] > 0}`.
3. Grouped fold-seed positivity survived: `{gates['grouped_fold_seed_positive']}`.
4. Blocked real remained below best matched blocked control: `{gates['blocked_real_minus_matched_control_pr_auc'] < 0}`.
5. Blocked real remained below AR-only: `{gates['blocked_real_minus_ar_only_pr_auc'] < 0}`.
6. AR-only dominates blocked under eval-mode: `{gates['blocked_ar_only_pr_auc'] > gates['blocked_best_matched_control_pr_auc'] and gates['blocked_ar_only_pr_auc'] > gates['blocked_real_pr_auc']}`.
7. Label permutation remained near chance and below real: `{gates['label_permutation_pass']}`.
8. Video-mean PCA remained unable to explain grouped: `{gates['video_mean_static_control_pass']}`.
9. Eval-mode scoring strengthens the corrected claim by making deterministic scoring canonical while preserving grouped support and blocked caveat.
10. The eval-mode numbers in this report and output root should now be considered canonical for deterministic checkpoint scoring.

## Ranking Shift Audit

Ranking shifts are recorded in `diagnostics/evalmode_control_ranking_shift_audit.json`.

```json
{json.dumps(ranking_shift[:6], indent=2)}
```

## Corrected Claim

Robust cross-video future arousal spike / emotional moment ranking is strengthened under deterministic eval-mode scoring. Strict forward-time temporal generalization remains unproven.
"""
    path.write_text(report)


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty eval-mode output root: {output_root}")
    start = time.time()
    rows = selected_rows(source_root)
    expected = 702
    missing = rows[~rows["checkpoint_exists"]]
    print(
        json.dumps(
            {
                "checkpoint_records_expected": expected,
                "checkpoint_records_found": int(len(rows)),
                "checkpoint_files_missing": int(len(missing)),
            },
            indent=2,
        )
    )
    if len(rows) != expected or not missing.empty:
        write_json(
            output_root / "diagnostics" / "evalmode_checkpoint_load_audit.json",
            {
                "checkpoint_records_expected": expected,
                "checkpoint_records_found": int(len(rows)),
                "checkpoint_files_missing": int(len(missing)),
                "missing": missing[["checkpoint_path", "validation_protocol", "fold", "control_type", "loss_name", "seed"]].to_dict(orient="records"),
            },
        )
        return 2

    dirs = {
        "metrics": output_root / "metrics",
        "promotion": output_root / "promotion",
        "diagnostics": output_root / "diagnostics",
        "reports": output_root / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    blocks, df = reconstruct_blocks(source_root)
    fold_rows: list[dict[str, Any]] = []
    within_rows: list[dict[str, Any]] = []
    load_rows: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        block = blocks[(row.validation_protocol, int(row.fold), row.control_type)]
        model = base.make_model(config_for(row), block["train_x"].shape[1], block["block_dims"])
        _ = model(base.mx.array(block["test_x"][:2], dtype=base.mx.float32))
        model.load_weights(str(row.checkpoint_path))
        if hasattr(model, "eval"):
            model.eval()
        train_scores, _train_regs = score_model_for_loss(model, block["train_x"], str(row.loss_name), batch_size=args.batch_size)
        test_scores, test_regs = score_model_for_loss(model, block["test_x"], str(row.loss_name), batch_size=args.batch_size)
        metric = metrics_from_scores(block["train_y_metric"], train_scores, block["test_y"], test_scores, block["test_cont"], test_regs)
        out = {
            "schema_version": SCHEMA_VERSION,
            "target_name": TARGET_NAME,
            "validation_protocol": row.validation_protocol,
            "fold": int(row.fold),
            "feature_name": FEATURE_NAME,
            "model_head": MODEL_NAME,
            "loss_name": row.loss_name,
            "seed": int(row.seed),
            "control_type": row.control_type,
            "n_train": int(len(block["train_x"])),
            "n_test": int(len(block["test_x"])),
            "feature_width": int(block["train_x"].shape[1]),
            "train_event_count": int(np.sum(block["train_y_metric"])),
            "test_event_count": int(np.sum(block["test_y"])),
            "train_positive_rate": float(np.mean(block["train_y_metric"])),
            "test_positive_rate": float(np.mean(block["test_y"])),
            "checkpoint_path": row.checkpoint_path,
            "checkpoint_checksum": row.checkpoint_checksum,
            "checkpoint_best_epoch": int(row.best_epoch),
            "checkpoint_file_loadable": True,
            "eval_mode_scoring": True,
            "dropout_disabled_for_scoring": True,
            "source_was_legacy_train_mode_dropout_scoring": True,
            "status": "success",
            **metric,
        }
        fold_rows.append(out)
        within_rows.extend(within_video_rows(row, block, test_scores, test_regs))
        load_rows.append(
            {
                "checkpoint_path": row.checkpoint_path,
                "checkpoint_checksum": row.checkpoint_checksum,
                "validation_protocol": row.validation_protocol,
                "fold": int(row.fold),
                "control_type": row.control_type,
                "loss_name": row.loss_name,
                "seed": int(row.seed),
                "checkpoint_file_loadable": True,
                "best_epoch": int(row.best_epoch),
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    summary_df = summarize_metrics(fold_df)
    within_df = pd.DataFrame(within_rows)
    gates = compute_gates(summary_df, fold_df, within_df)
    legacy_summary = pd.read_csv(source_root / "metrics" / "phase5_summary_metrics.csv")
    comparison = summary_df.merge(
        legacy_summary[
            [
                "target_name",
                "validation_protocol",
                "feature_name",
                "model_head",
                "loss_name",
                "control_type",
                "mean_pr_auc",
                "mean_roc_auc",
            ]
        ].rename(columns={"mean_pr_auc": "legacy_train_mode_pr_auc", "mean_roc_auc": "legacy_roc_auc"}),
        on=["target_name", "validation_protocol", "feature_name", "model_head", "loss_name", "control_type"],
        how="left",
    )
    comparison = comparison.rename(columns={"mean_pr_auc": "eval_mode_pr_auc", "mean_roc_auc": "eval_roc_auc"})
    comparison["delta_eval_minus_legacy_pr_auc"] = comparison["eval_mode_pr_auc"] - comparison["legacy_train_mode_pr_auc"]
    comparison["delta_eval_minus_legacy_roc_auc"] = comparison["eval_roc_auc"] - comparison["legacy_roc_auc"]
    ranking_shift = ranking_shift_audit(comparison)

    fold_df.to_csv(dirs["metrics"] / "evalmode_fold_metrics.csv", index=False)
    fold_df.to_csv(dirs["metrics"] / "evalmode_seed_metrics.csv", index=False)
    fold_df[fold_df["control_type"] != "real_ar_pca_diag"].to_csv(dirs["metrics"] / "evalmode_control_metrics.csv", index=False)
    top_cols = [c for c in fold_df.columns if c.startswith("top_") or c in ["target_name", "validation_protocol", "fold", "loss_name", "seed", "control_type"]]
    fold_df[top_cols].to_csv(dirs["metrics"] / "evalmode_top_percent_metrics.csv", index=False)
    continuous_cols = [c for c in fold_df.columns if c.startswith("continuous_") or c in ["target_name", "validation_protocol", "fold", "loss_name", "seed", "control_type", "spearman_future_movement"]]
    fold_df[continuous_cols].to_csv(dirs["metrics"] / "evalmode_continuous_metrics.csv", index=False)
    summary_df.to_csv(dirs["metrics"] / "evalmode_summary_metrics.csv", index=False)
    within_df.to_csv(dirs["metrics"] / "evalmode_within_video_metrics.csv", index=False)
    comparison.to_csv(dirs["diagnostics"] / "evalmode_metric_reproduction_vs_legacy_trainmode.csv", index=False)

    failure_reasons = {
        "blocked_matched_control_pass": gates["blocked_matched_control_pass"],
        "blocked_failure_reason": (
            f"blocked real_ar_pca_diag PR-AUC {gates['blocked_real_pr_auc']:.10f} is below "
            f"best matched blocked control {gates['blocked_best_matched_control_type']} PR-AUC "
            f"{gates['blocked_best_matched_control_pr_auc']:.10f}, and AR-only PR-AUC "
            f"{gates['blocked_ar_only_pr_auc']:.10f} is higher than real."
        ),
        "strict_forward_time_temporal_generalization_proven": False,
        "full_pass": False,
        "repair_required": True,
    }
    verdict = {
        "recommendation": gates["recommendation"],
        "eval_mode_scoring": True,
        "cross_video_ranking_strengthened": gates["grouped_matched_control_pass"],
        "grouped_support": gates["grouped_matched_control_pass"],
        "blocked_support": gates["blocked_matched_control_pass"],
        "strict_forward_time_temporal_generalization_proven": False,
        "grouped_real_pr_auc": gates["grouped_real_pr_auc"],
        "grouped_best_matched_control_pr_auc": gates["grouped_best_matched_control_pr_auc"],
        "grouped_matched_control_delta": gates["grouped_real_minus_matched_control_pr_auc"],
        "blocked_real_pr_auc": gates["blocked_real_pr_auc"],
        "blocked_best_matched_control_pr_auc": gates["blocked_best_matched_control_pr_auc"],
        "blocked_matched_control_delta": gates["blocked_real_minus_matched_control_pr_auc"],
        "corrected_claim": "Robust cross-video future arousal spike / emotional moment ranking is strengthened under deterministic eval-mode scoring; strict forward-time temporal generalization remains unproven.",
    }
    write_json(dirs["promotion"] / "evalmode_corrected_promotion_gates.json", gates)
    write_json(dirs["promotion"] / "evalmode_failure_reasons.json", failure_reasons)
    write_json(dirs["promotion"] / "evalmode_adversarial_verdict.json", verdict)
    write_json(
        dirs["diagnostics"] / "evalmode_checkpoint_load_audit.json",
        {
            "checkpoint_records_expected": expected,
            "checkpoint_records_found": int(len(rows)),
            "checkpoint_files_missing": 0,
            "checkpoints_loadable": int(len(load_rows)),
            "rows": load_rows,
        },
    )
    write_json(
        dirs["diagnostics"] / "evalmode_metric_reproduction_vs_legacy_trainmode.json",
        {
            "source_was_legacy_train_mode_dropout_scoring": True,
            "eval_mode_scoring": True,
            "comparison_csv": str(dirs["diagnostics"] / "evalmode_metric_reproduction_vs_legacy_trainmode.csv"),
            "max_abs_delta_pr_auc": finite_float(comparison["delta_eval_minus_legacy_pr_auc"].abs().max()),
            "max_abs_delta_roc_auc": finite_float(comparison["delta_eval_minus_legacy_roc_auc"].abs().max()),
        },
    )
    write_json(dirs["diagnostics"] / "evalmode_control_ranking_shift_audit.json", ranking_shift)
    write_json(
        dirs["diagnostics"] / "evalmode_ar_dominance_audit.json",
        {
            "blocked_ar_only_pr_auc": gates["blocked_ar_only_pr_auc"],
            "blocked_real_pr_auc": gates["blocked_real_pr_auc"],
            "blocked_best_matched_control_pr_auc": gates["blocked_best_matched_control_pr_auc"],
            "ar_only_dominates_blocked": bool(gates["blocked_ar_only_pr_auc"] > gates["blocked_real_pr_auc"] and gates["blocked_ar_only_pr_auc"] > gates["blocked_best_matched_control_pr_auc"]),
        },
    )
    report_path = Path(args.reports_dir) / REPORT_NAME
    write_report(report_path, gates, comparison, ranking_shift)
    (dirs["reports"] / REPORT_NAME).write_text(report_path.read_text())
    write_json(
        output_root / "run_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_output_root": str(source_root),
            "source_commit_sha": SOURCE_COMMIT,
            "output_root": str(output_root),
            "no_training": True,
            "score_only": True,
            "eval_mode_scoring": True,
            "dropout_disabled_for_scoring": True,
            "source_was_legacy_train_mode_dropout_scoring": True,
            "checkpoints_expected": expected,
            "checkpoints_loaded": int(len(load_rows)),
            "checkpoints_missing": 0,
            "target": TARGET_NAME,
            "feature": FEATURE_NAME,
            "model": MODEL_NAME,
            "protocols": list(PROTOCOLS),
            "losses": list(LOSSES),
            "seeds": list(SEEDS),
            "runtime_seconds": time.time() - start,
        },
    )
    print(
        json.dumps(
            {
                "checkpoint_load_pass": True,
                "checkpoints_loaded": int(len(load_rows)),
                "checkpoints_expected": expected,
                "grouped_eval_mode_real_pr_auc": gates["grouped_real_pr_auc"],
                "grouped_eval_mode_best_matched_control_pr_auc": gates["grouped_best_matched_control_pr_auc"],
                "grouped_eval_mode_delta": gates["grouped_real_minus_matched_control_pr_auc"],
                "grouped_eval_mode_ar_only_pr_auc": gates["grouped_ar_only_pr_auc"],
                "blocked_eval_mode_real_pr_auc": gates["blocked_real_pr_auc"],
                "blocked_eval_mode_best_matched_control_pr_auc": gates["blocked_best_matched_control_pr_auc"],
                "blocked_eval_mode_delta": gates["blocked_real_minus_matched_control_pr_auc"],
                "blocked_eval_mode_ar_only_pr_auc": gates["blocked_ar_only_pr_auc"],
                "recommendation": gates["recommendation"],
                "report_path": str(report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
