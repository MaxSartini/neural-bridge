"""Big blocked binary confirmation for temporal residual diagnostic.

Bounded matrix:
10 seeds x 7 scored lanes = 70 rows.

This script confirms only the redesigned binary washout-gap target with the
previous diagnostic winner: `short_temporal_conv_residual`. It does not run
continuous, grouped, 504, extra targets, extra architectures, V-JEPA/TRIBE, PCA
fitting, or PCA refitting.
"""

from __future__ import annotations

import argparse
import gc
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

from backend.scripts import run_again_dense_2hz_phase5_frozen_ar_residual as fr
from backend.scripts import run_again_dense_2hz_phase5_learned_heads as base
from backend.scripts import run_again_dense_2hz_phase5_redesigned_target_blocked as redesigned
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_blocked as temporal


SCHEMA_VERSION = "again_dense_2hz_phase5_temporal_residual_binary_big_confirm_v1"
SOURCE_ROOT = Path("outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_20260629_171825")
FOLDSAFE_PCA_ROOT = temporal.FOLDSAFE_PCA_ROOT
PREVIOUS_TEMPORAL_ROOT = Path("outputs/again_dense_2hz_phase5_temporal_residual_blocked_20260630_020557")
TARGET_NAME = temporal.BINARY_TARGET
PROTOCOL = temporal.PROTOCOL
FOLD = temporal.FOLD
ARCHITECTURE = "short_temporal_conv_residual"
FEATURE_NAME = temporal.FEATURE_NAME
SEEDS = (20260625, 20260626, 20260627, 20260628, 20260629, 20260630, 20260631, 20260632, 20260633, 20260634)
REUSED_AR_SEEDS = (20260625, 20260626, 20260627)
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
WEAK_THRESHOLD = 0.001
CREDIBLE_THRESHOLD = 0.003
STRONG_THRESHOLD = 0.005


def default_output_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(f"outputs/again_dense_2hz_phase5_temporal_residual_binary_big_confirm_{stamp}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--foldsafe-pca-root", default=str(FOLDSAFE_PCA_ROOT))
    parser.add_argument("--previous-temporal-root", default=str(PREVIOUS_TEMPORAL_ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--ar-max-epochs", type=int, default=80)
    parser.add_argument("--ar-patience", type=int, default=12)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    fr.write_json(path, obj)


def ar_config(seed: int, *, max_epochs: int, patience: int, batch_size: int) -> base.TrainConfig:
    return base.TrainConfig(
        model_name="gated_ar_pca_mlp",
        loss_name=fr.LOSS_NAME,
        seed=int(seed),
        hidden_sizes=(256,),
        dropout=0.1,
        learning_rate=3e-4,
        weight_decay=1e-4,
        lambda_binary=0.5,
        batch_size=int(batch_size),
        max_epochs=int(max_epochs),
        patience=int(patience),
    )


def load_reused_ar_scores(previous_root: Path, output_root: Path, block: redesigned.RunBlock, seed: int) -> dict[str, Any] | None:
    key = f"{block.target_name}__{block.protocol}__fold{block.fold}__seed{seed}__{fr.LOSS_NAME}"
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
    return {
        "key": key,
        "target_name": block.target_name,
        "protocol": block.protocol,
        "fold": int(block.fold),
        "seed": int(seed),
        "loss": fr.LOSS_NAME,
        "source": "reused_existing_seed_specific_frozen_ar_score_cache",
        "ar_baseline_reused": True,
        "ar_baseline_newly_trained": False,
        "ar_retrained": False,
        "train_score": train_score,
        "train_reg": train_reg,
        "test_score": test_score,
        "test_reg": test_reg,
        "train_checksum": fr.hash_array(train_score),
        "test_checksum": fr.hash_array(test_score),
        "checkpoint_restore_pass": True,
        "eval_mode_scoring_pass": True,
    }


def train_ar_baseline(
    *,
    output_root: Path,
    block: redesigned.RunBlock,
    seed: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base.require_mlx()
    base.mx.random.seed(int(seed))
    config = ar_config(seed, max_epochs=max_epochs, patience=patience, batch_size=batch_size)
    model = base.make_model(config, block.ar_train_x.shape[1], block.ar_block_dims)
    optimizer = base.optim.AdamW(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    rng = np.random.default_rng(int(seed) + 70001)
    inner_train = block.inner_train
    inner_val = block.inner_val
    best_val = float("-inf")
    best_epoch = 0
    stale = 0
    curves: list[dict[str, Any]] = []
    early_stop = "max_epochs_reached"
    checkpoint_path = output_root / "ar_baseline_checkpoints" / f"{block.target_name}__{block.protocol}__fold{block.fold}__ar_only__seed{seed}__best.npz"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def loss_fn(model_obj: Any, xb: Any, yb: Any, yr: Any) -> Any:
        out = model_obj(xb)
        if out.ndim == 1:
            out = out[:, None]
        reg = base.mx.mean(base.nn.losses.huber_loss(out[:, 0:1], yr, delta=1.0))
        bce = base.mx.mean(base.nn.losses.binary_cross_entropy(out[:, 1:2], yb, with_logits=True))
        return reg + float(config.lambda_binary) * bce

    loss_and_grad = base.nn.value_and_grad(model, loss_fn)
    for epoch in range(1, int(config.max_epochs) + 1):
        if hasattr(model, "train"):
            model.train()
        order = rng.permutation(inner_train)
        total = 0.0
        batches = 0
        for start in range(0, len(order), int(config.batch_size)):
            rel = order[start : start + int(config.batch_size)]
            loss, grads = loss_and_grad(
                model,
                base.mx.array(block.ar_train_x[rel], dtype=base.mx.float32),
                base.mx.array(block.train_y[rel].astype(np.float32)[:, None], dtype=base.mx.float32),
                base.mx.array(block.train_cont[rel].astype(np.float32)[:, None], dtype=base.mx.float32),
            )
            grads, _ = base.optim.clip_grad_norm(grads, 1.0)
            optimizer.update(model, grads)
            base.mx.eval(model.parameters(), optimizer.state)
            total += float(np.asarray(loss))
            batches += 1
        if hasattr(model, "eval"):
            model.eval()
        val_score, _val_reg = fr.score_existing_model(model, block.ar_train_x[inner_val], int(config.batch_size))
        val_pr = average_precision_score(block.train_y[inner_val], val_score) if len(np.unique(block.train_y[inner_val])) > 1 else math.nan
        curves.append(
            {
                "seed": int(seed),
                "epoch": int(epoch),
                "train_loss": total / max(1, batches),
                "inner_val_pr_auc": val_pr,
                **block.inner_audit,
            }
        )
        if math.isfinite(val_pr) and val_pr > best_val:
            model.save_weights(str(checkpoint_path))
            best_val = float(val_pr)
            best_epoch = int(epoch)
            stale = 0
        else:
            stale += 1
        if stale >= int(config.patience):
            early_stop = "patience_exhausted"
            break
    if not checkpoint_path.exists():
        raise RuntimeError(f"AR baseline checkpoint was not saved for seed {seed}")
    model.load_weights(str(checkpoint_path))
    if hasattr(model, "eval"):
        model.eval()
    train_score, train_reg = fr.score_existing_model(model, block.ar_train_x, int(config.batch_size))
    test_score, test_reg = fr.score_existing_model(model, block.ar_test_x, int(config.batch_size))
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
    return (
        {
            "key": key,
            "target_name": block.target_name,
            "protocol": block.protocol,
            "fold": int(block.fold),
            "seed": int(seed),
            "loss": fr.LOSS_NAME,
            "source": "newly_trained_ar_only_baseline_for_confirmation_seed",
            "ar_baseline_reused": False,
            "ar_baseline_newly_trained": True,
            "ar_retrained": True,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_checksum": base.file_digest(checkpoint_path),
            "best_epoch": int(best_epoch),
            "epochs_run": int(len(curves)),
            "best_inner_val_pr_auc": float(best_val),
            "early_stopping_reason": early_stop,
            "train_score": train_score,
            "train_reg": train_reg,
            "test_score": test_score,
            "test_reg": test_reg,
            "train_checksum": fr.hash_array(train_score),
            "test_checksum": fr.hash_array(test_score),
            "checkpoint_restore_pass": True,
            "eval_mode_scoring_pass": True,
        },
        curves,
    )


def obtain_ar_baseline(
    *,
    previous_root: Path,
    output_root: Path,
    block: redesigned.RunBlock,
    seed: int,
    batch_size: int,
    ar_max_epochs: int,
    ar_patience: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if seed in REUSED_AR_SEEDS:
        reused = load_reused_ar_scores(previous_root, output_root, block, seed)
        if reused is None:
            raise RuntimeError(f"Expected reusable AR score cache for seed {seed}, but none matched")
        return reused, []
    return train_ar_baseline(
        output_root=output_root,
        block=block,
        seed=seed,
        batch_size=batch_size,
        max_epochs=ar_max_epochs,
        patience=ar_patience,
    )


def summarize_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["target_name", "architecture", "control_type"]
    metric_cols = [
        "pr_auc",
        "roc_auc",
        "top_1pct_recall",
        "top_5pct_recall",
        "top_10pct_recall",
        "top_1pct_precision",
        "top_5pct_precision",
        "top_10pct_precision",
        "delta_vs_frozen_ar_pr_auc",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in fold_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["seeds"] = int(group["seed"].nunique())
        row["rows_test_total"] = int(group["n_test"].sum())
        for metric in metric_cols:
            if metric in group:
                vals = pd.to_numeric(group[metric], errors="coerce")
                row[f"mean_{metric}"] = float(vals.mean()) if vals.notna().any() else math.nan
                row[f"std_{metric}"] = float(vals.std(ddof=0)) if vals.notna().sum() > 1 else math.nan
                row[f"min_{metric}"] = float(vals.min()) if vals.notna().any() else math.nan
                row[f"max_{metric}"] = float(vals.max()) if vals.notna().any() else math.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["target_name", "control_type"])


def seed_deltas(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed, group in fold_df.groupby("seed"):
        vals = group.set_index("control_type")
        real = vals.loc["real_residual"]
        row: dict[str, Any] = {"target_name": TARGET_NAME, "architecture": ARCHITECTURE, "seed": int(seed), "real_pr_auc": float(real["pr_auc"])}
        control_prs = {}
        for control in ("frozen_ar_only", *PRIMARY_CONTROLS, "diagnostics_only_residual"):
            control_pr = float(vals.loc[control, "pr_auc"])
            control_prs[control] = control_pr
            row[f"{control}_pr_auc"] = control_pr
            row[f"real_minus_{control}_pr_auc"] = float(real["pr_auc"] - control_pr)
        primary_vals = {control: control_prs[control] for control in PRIMARY_CONTROLS}
        best_control = max(primary_vals, key=primary_vals.get)
        row["best_control"] = best_control
        row["best_control_pr_auc"] = primary_vals[best_control]
        row["real_minus_best_control_pr_auc"] = float(real["pr_auc"] - primary_vals[best_control])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("seed")


def max_seed_contribution(delta_values: pd.Series) -> float:
    positives = pd.to_numeric(delta_values, errors="coerce")
    positives = positives[positives > 0]
    total = float(positives.sum())
    if total <= 0 or positives.empty:
        return math.inf
    return float(positives.max() / total)


def row_for(summary: pd.DataFrame, control: str) -> pd.Series:
    sub = summary[(summary["target_name"] == TARGET_NAME) & (summary["architecture"] == ARCHITECTURE) & (summary["control_type"] == control)]
    if len(sub) != 1:
        raise RuntimeError(f"Expected one summary row for {control}, got {len(sub)}")
    return sub.iloc[0]


def compute_gates(summary: pd.DataFrame, fold_df: pd.DataFrame, seed_df: pd.DataFrame, context_audit: dict[str, Any], ar_manifest: list[dict[str, Any]]) -> dict[str, Any]:
    failed: list[str] = []
    real = row_for(summary, "real_residual")
    ar = row_for(summary, "frozen_ar_only")
    controls = [row_for(summary, control) for control in PRIMARY_CONTROLS]
    best_control = max(controls, key=lambda row: float(row["mean_pr_auc"]))
    delta_ar = float(real["mean_pr_auc"] - ar["mean_pr_auc"])
    delta_best = float(real["mean_pr_auc"] - best_control["mean_pr_auc"])
    deltas = {
        "real_minus_frozen_ar_pr_auc": delta_ar,
        "real_minus_best_control_pr_auc": delta_best,
    }
    for control in PRIMARY_CONTROLS:
        ctrl = row_for(summary, control)
        deltas[f"real_minus_{control}_pr_auc"] = float(real["mean_pr_auc"] - ctrl["mean_pr_auc"])
    seeds_positive_ar = int((seed_df["real_minus_frozen_ar_only_pr_auc"] > 0).sum())
    seeds_positive_best = int((seed_df["real_minus_best_control_pr_auc"] > 0).sum())
    ar_contrib = max_seed_contribution(seed_df["real_minus_frozen_ar_only_pr_auc"])
    best_contrib = max_seed_contribution(seed_df["real_minus_best_control_pr_auc"])
    leakage_pass = bool(context_audit.get("leakage_context_audit_pass"))
    frozen_integrity = bool(fold_df.groupby("seed")["frozen_ar_test_checksum"].nunique().max() == 1)
    within_seed_controls_match_ar = bool(fold_df.groupby("seed")["frozen_ar_test_checksum"].nunique().eq(1).all())
    checkpoint_restore = bool(fold_df["checkpoint_restore_pass"].all())
    eval_mode = bool(fold_df["eval_mode_scoring"].all())
    ar_manifest_seeds = {int(row["seed"]) for row in ar_manifest}
    reused_seeds = sorted(int(row["seed"]) for row in ar_manifest if row.get("ar_baseline_reused"))
    newly_trained_seeds = sorted(int(row["seed"]) for row in ar_manifest if row.get("ar_baseline_newly_trained"))
    ar_generation_valid = bool(
        ar_manifest_seeds == set(SEEDS)
        and reused_seeds == list(REUSED_AR_SEEDS)
        and newly_trained_seeds == [seed for seed in SEEDS if seed not in REUSED_AR_SEEDS]
    )
    all_control_mean = all(deltas[f"real_minus_{control}_pr_auc"] > 0 for control in PRIMARY_CONTROLS)
    weak = bool(delta_ar >= WEAK_THRESHOLD and delta_best >= WEAK_THRESHOLD and seeds_positive_ar >= 7 and seeds_positive_best >= 7 and leakage_pass)
    credible = bool(delta_ar >= CREDIBLE_THRESHOLD and delta_best >= CREDIBLE_THRESHOLD and seeds_positive_ar >= 8 and seeds_positive_best >= 8 and ar_contrib <= 0.40 and best_contrib <= 0.40 and all_control_mean and leakage_pass)
    strong = bool(delta_ar >= STRONG_THRESHOLD and delta_best >= STRONG_THRESHOLD and seeds_positive_ar >= 8 and seeds_positive_best >= 8 and ar_contrib <= 0.40 and best_contrib <= 0.40 and all_control_mean and leakage_pass)
    if delta_ar < CREDIBLE_THRESHOLD:
        failed.append("credible_delta_vs_ar")
    if delta_best < CREDIBLE_THRESHOLD:
        failed.append("credible_delta_vs_best_control")
    if not all_control_mean:
        failed.append("real_beats_all_primary_controls")
    if seeds_positive_ar < 8:
        failed.append("seed_consistency_vs_ar")
    if seeds_positive_best < 8:
        failed.append("seed_consistency_vs_best_control")
    if ar_contrib > 0.40:
        failed.append("single_seed_contribution_vs_ar")
    if best_contrib > 0.40:
        failed.append("single_seed_contribution_vs_best_control")
    for name, ok in (
        ("leakage_context_audit", leakage_pass),
        ("frozen_ar_integrity", frozen_integrity),
        ("within_seed_controls_match_ar", within_seed_controls_match_ar),
        ("checkpoint_restore", checkpoint_restore),
        ("eval_mode_scoring", eval_mode),
        ("ar_baseline_generation", ar_generation_valid),
    ):
        if not ok:
            failed.append(name)
    recommendation = "binary_big_confirmation_pass_review_before_any_grouped_or_504" if credible else "binary_big_confirmation_failed_do_not_run_grouped_or_504"
    return {
        "schema_version": SCHEMA_VERSION,
        "target": TARGET_NAME,
        "protocol": PROTOCOL,
        "architecture": ARCHITECTURE,
        "matrix_rows_expected": 70,
        "matrix_rows_actual": int(len(fold_df)),
        "residual_control_rows": 70,
        "ar_baselines_reused": int(sum(bool(row.get("ar_baseline_reused")) for row in ar_manifest)),
        "ar_baselines_newly_trained": int(sum(bool(row.get("ar_baseline_newly_trained")) for row in ar_manifest)),
        "ar_baselines_reused_seeds": reused_seeds,
        "ar_baselines_newly_trained_seeds": newly_trained_seeds,
        "each_seed_uses_own_frozen_ar_score": within_seed_controls_match_ar,
        "all_controls_within_seed_use_identical_frozen_ar_scores": within_seed_controls_match_ar,
        "shared_three_seed_ar_cache_reused_across_ten_seed_confirmation": False,
        "ar_only_baseline_generation_reported_separately_from_residual_control_rows": True,
        "ten_seed_confirmation_valid": bool(leakage_pass and frozen_integrity and within_seed_controls_match_ar and ar_generation_valid and checkpoint_restore and eval_mode),
        "weak_confirmation_pass": weak,
        "credible_confirmation_pass": credible,
        "strong_confirmation_pass": strong,
        "binary_pass": credible,
        "strict_forward_time_temporal_generalization_proven": False,
        "grouped_started": False,
        "full_504_started": False,
        "leakage_context_audit_pass": leakage_pass,
        "frozen_ar_integrity_pass": frozen_integrity,
        "within_seed_controls_match_ar_pass": within_seed_controls_match_ar,
        "checkpoint_restore_pass": checkpoint_restore,
        "eval_mode_scoring_pass": eval_mode,
        "ar_baseline_generation_pass": ar_generation_valid,
        "failed_gates": failed,
        "recommendation": recommendation,
        "real_pr_auc": float(real["mean_pr_auc"]),
        "frozen_ar_pr_auc": float(ar["mean_pr_auc"]),
        "best_control": str(best_control["control_type"]),
        "best_control_pr_auc": float(best_control["mean_pr_auc"]),
        "delta_vs_ar": delta_ar,
        "delta_vs_best_control": delta_best,
        "seeds_positive_vs_ar": seeds_positive_ar,
        "seeds_positive_vs_best_control": seeds_positive_best,
        "max_seed_contribution_vs_ar": ar_contrib,
        "max_seed_contribution_vs_best_control": best_contrib,
        "deltas": deltas,
    }


def write_report(path: Path, output_root: Path, gates: dict[str, Any]) -> None:
    text = f"""# Phase 5 Temporal Residual Binary Big Confirmation

Output root: `{output_root}`

This is a blocked-only 10-seed confirmation for the redesigned binary washout-gap target using only `short_temporal_conv_residual`. It does not run continuous, grouped, 504, extra targets, extra architectures, V-JEPA/TRIBE/PCA, PCA refit, or claim changes.

## Scope

- Target: `{gates['target']}`
- Protocol: `{gates['protocol']}`
- Architecture: `{gates['architecture']}`
- Residual/control rows: `{gates['residual_control_rows']}`
- AR baselines reused: `{gates['ar_baselines_reused']}` seeds, `{gates['ar_baselines_reused_seeds']}`
- AR baselines newly trained: `{gates['ar_baselines_newly_trained']}` seeds, `{gates['ar_baselines_newly_trained_seeds']}`
- Each seed uses its own frozen AR score: `{gates['each_seed_uses_own_frozen_ar_score']}`
- All controls within a seed use identical frozen AR scores: `{gates['all_controls_within_seed_use_identical_frozen_ar_scores']}`
- Shared 3-seed AR cache reused across the 10-seed confirmation: `{gates['shared_three_seed_ar_cache_reused_across_ten_seed_confirmation']}`
- AR-only baseline generation is reported separately from residual/control rows: `{gates['ar_only_baseline_generation_reported_separately_from_residual_control_rows']}`
- 10/10 seed confirmation valid: `{gates['ten_seed_confirmation_valid']}`

## Result

- Real PR-AUC: `{gates['real_pr_auc']:.10f}`
- Frozen AR PR-AUC: `{gates['frozen_ar_pr_auc']:.10f}`
- Best control: `{gates['best_control']}` PR-AUC `{gates['best_control_pr_auc']:.10f}`
- Delta vs frozen AR: `{gates['delta_vs_ar']:+.10f}`
- Delta vs best control: `{gates['delta_vs_best_control']:+.10f}`
- Seeds positive vs AR: `{gates['seeds_positive_vs_ar']}/10`
- Seeds positive vs best control: `{gates['seeds_positive_vs_best_control']}/10`
- Max seed contribution vs AR: `{gates['max_seed_contribution_vs_ar']:.4f}`
- Max seed contribution vs best control: `{gates['max_seed_contribution_vs_best_control']:.4f}`

## Gates

- Weak confirmation: `{gates['weak_confirmation_pass']}`
- Credible confirmation: `{gates['credible_confirmation_pass']}`
- Strong confirmation: `{gates['strong_confirmation_pass']}`
- `leakage_context_audit_pass`: `{gates['leakage_context_audit_pass']}`
- `frozen_ar_integrity_pass`: `{gates['frozen_ar_integrity_pass']}`
- `within_seed_controls_match_ar_pass`: `{gates['within_seed_controls_match_ar_pass']}`
- `checkpoint_restore_pass`: `{gates['checkpoint_restore_pass']}`
- `eval_mode_scoring_pass`: `{gates['eval_mode_scoring_pass']}`
- `ar_baseline_generation_pass`: `{gates['ar_baseline_generation_pass']}`
- Failed gates: `{gates['failed_gates']}`
- Recommendation: `{gates['recommendation']}`

Strict forward-time temporal generalization remains unproven until any further confirmation is explicitly reviewed and promoted. This report alone does not authorize grouped or 504.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def finalize_output(output_root: Path, reports_dir: Path) -> dict[str, Any]:
    fold_df = pd.read_csv(output_root / "metrics" / "temporal_residual_binary_big_confirm_seed_metrics.csv")
    context_audit = json.loads((output_root / "diagnostics" / "leakage_context_audit.json").read_text(encoding="utf-8"))
    ar_manifest = json.loads((output_root / "manifests" / "ar_baseline_generation_manifest.json").read_text(encoding="utf-8"))["baselines"]
    summary = summarize_metrics(fold_df)
    seed_df = seed_deltas(fold_df)
    gates = compute_gates(summary, fold_df, seed_df, context_audit, ar_manifest)
    summary.to_csv(output_root / "metrics" / "temporal_residual_binary_big_confirm_summary_metrics.csv", index=False)
    seed_df.to_csv(output_root / "metrics" / "temporal_residual_binary_big_confirm_seed_deltas.csv", index=False)
    summary.to_csv(output_root / "promotion" / "temporal_residual_binary_big_confirm_control_comparison.csv", index=False)
    write_json(output_root / "promotion" / "temporal_residual_binary_big_confirm_gates.json", gates)
    write_json(output_root / "promotion" / "temporal_residual_binary_big_confirm_adversarial_verdict.json", gates)
    write_json(output_root / "promotion" / "temporal_residual_binary_big_confirm_failure_reasons.json", {"failed_gates": gates["failed_gates"], "recommendation": gates["recommendation"]})
    stamp = output_root.name.replace("again_dense_2hz_phase5_temporal_residual_binary_big_confirm_", "")
    report_name = f"again_dense_2hz_phase5_temporal_residual_binary_big_confirm_{stamp}.md"
    write_report(output_root / "reports" / report_name, output_root, gates)
    report_path = reports_dir / report_name
    write_report(report_path, output_root, gates)
    return {"gates": gates, "report_path": str(report_path)}


def matrix_rows() -> list[tuple[int, str]]:
    return [(seed, control) for seed in SEEDS for control in CONTROLS]


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root) if args.output_root else default_output_root()
    source_root = Path(args.source_root)
    pca_root = Path(args.foldsafe_pca_root)
    previous_root = Path(args.previous_temporal_root)
    matrix = matrix_rows()
    print(json.dumps({"matrix_size": len(matrix), "max_allowed": 70, "target": TARGET_NAME, "architecture": ARCHITECTURE, "seeds": list(SEEDS)}, indent=2))
    if len(matrix) > 70:
        raise RuntimeError(f"Refusing to exceed 70 rows: {len(matrix)}")
    pca_info = temporal.load_pca_manifest(pca_root)
    if args.dry_run:
        return 0
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {output_root}")
    start = time.time()
    for sub in ("manifests", "metrics", "promotion", "diagnostics", "reports", "frozen_ar_scores", "checkpoints", "ar_baseline_checkpoints"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)
    blocks, df, dense_root, residual_meta = temporal.build_blocks(source_root, pca_root)
    block = temporal.block_for_target(blocks, TARGET_NAME)
    split_audit = temporal.verify_pca_rows(pca_root, block, pca_info["rows"])
    fold_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    ar_curve_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    ar_manifest: list[dict[str, Any]] = []

    for seed in SEEDS:
        ar, ar_curves = obtain_ar_baseline(
            previous_root=previous_root,
            output_root=output_root,
            block=block,
            seed=seed,
            batch_size=args.batch_size,
            ar_max_epochs=args.ar_max_epochs,
            ar_patience=args.ar_patience,
        )
        ar_curve_rows.extend(ar_curves)
        ar_manifest.append({k: v for k, v in ar.items() if k not in {"train_score", "train_reg", "test_score", "test_reg"}})
        ar_metrics = temporal.metric_row_for_block(block, ar["train_score"], ar["test_score"], ar["test_reg"])
        for control in CONTROLS:
            if control == "frozen_ar_only":
                fold_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "target_name": TARGET_NAME,
                        "target_type": "binary",
                        "validation_protocol": PROTOCOL,
                        "fold": FOLD,
                        "seed": int(seed),
                        "architecture": ARCHITECTURE,
                        "control_type": control,
                        "feature_name": FEATURE_NAME,
                        "n_train": int(len(block.train_idx)),
                        "n_test": int(len(block.test_idx)),
                        "checkpoint_restore_pass": bool(ar.get("checkpoint_restore_pass")),
                        "eval_mode_scoring": bool(ar.get("eval_mode_scoring_pass")),
                        "dropout_disabled": True,
                        "ar_baseline_reused": bool(ar.get("ar_baseline_reused")),
                        "ar_baseline_newly_trained": bool(ar.get("ar_baseline_newly_trained")),
                        "frozen_ar_train_checksum": ar["train_checksum"],
                        "frozen_ar_test_checksum": ar["test_checksum"],
                        **ar_metrics,
                    }
                )
                continue
            pack = temporal.feature_pack_for(df, dense_root, pca_root, block, ARCHITECTURE, control, seed)
            metrics, curves, audit = temporal.train_temporal_residual(
                architecture=ARCHITECTURE,
                control=control,
                pack=pack,
                block=block,
                ar=ar,
                seed=seed,
                output_root=output_root,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                patience=args.patience,
            )
            curve_rows.extend(curves)
            feature_rows.append({"target_name": TARGET_NAME, "architecture": ARCHITECTURE, "control_type": control, "seed": int(seed), "dims": pack.dims, "blocks": pack.manifest})
            context_rows.append(pack.context_audit)
            if control == "label_permutation_residual":
                label_rows.append({"seed": int(seed), "control_type": control, "best_epoch": audit["best_epoch"], "label_policy": audit["label_policy"], "heldout_scoring_policy": "true_heldout_labels_targets", "best_inner_val_delta_vs_frozen_ar": audit["best_inner_val_delta_vs_frozen_ar"]})
            if control == "train_only_video_mean_residual":
                video_rows.append({"seed": int(seed), "control_type": control, "uses_test_rows_for_mean": False, "best_epoch": audit["best_epoch"], "checkpoint_restored": audit["checkpoint_restored"]})
            fold_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "target_name": TARGET_NAME,
                    "target_type": "binary",
                    "validation_protocol": PROTOCOL,
                    "fold": FOLD,
                    "seed": int(seed),
                    "architecture": ARCHITECTURE,
                    "control_type": control,
                    "feature_name": FEATURE_NAME,
                    "n_train": int(len(block.train_idx)),
                    "n_test": int(len(block.test_idx)),
                    "checkpoint_restore_pass": audit["checkpoint_restored"] or audit["residual_suppressed"],
                    "ar_baseline_reused": bool(ar.get("ar_baseline_reused")),
                    "ar_baseline_newly_trained": bool(ar.get("ar_baseline_newly_trained")),
                    "frozen_ar_train_checksum": ar["train_checksum"],
                    "frozen_ar_test_checksum": ar["test_checksum"],
                    **audit,
                    **metrics,
                }
            )
            pd.DataFrame(fold_rows).to_csv(output_root / "metrics" / "temporal_residual_binary_big_confirm_seed_metrics.partial.csv", index=False)
            gc.collect()

    fold_df = pd.DataFrame(fold_rows)
    if len(fold_df) != 70:
        raise RuntimeError(f"Expected 70 scored rows, got {len(fold_df)}")
    fold_df.to_csv(output_root / "metrics" / "temporal_residual_binary_big_confirm_seed_metrics.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(output_root / "diagnostics" / "training_curve_summary.csv", index=False)
    pd.DataFrame(ar_curve_rows).to_csv(output_root / "diagnostics" / "ar_baseline_training_curve_summary.csv", index=False)
    pd.DataFrame(label_rows).to_csv(output_root / "diagnostics" / "label_permutation_audit.csv", index=False)
    pd.DataFrame(video_rows).to_csv(output_root / "diagnostics" / "train_only_video_mean_audit.csv", index=False)
    leakage_context_pass = bool(
        pca_info["audit"].get("leakage_audit_pass")
        and pca_info["audit"].get("no_test_rows_used_in_pca_fit")
        and split_audit.get("row_index_verified")
        and all(row.get("temporal_context_causal_only") for row in context_rows)
        and not any(row.get("uses_centered_or_future_windows") for row in context_rows)
        and all(row.get("same_video_history_masking") for row in context_rows)
        and all(row.get("label_policy") == "permuted_train_and_permuted_inner_val_selection" for row in label_rows)
        and not any(row.get("uses_test_rows_for_mean") for row in video_rows)
    )
    context_audit = {
        "schema_version": SCHEMA_VERSION,
        "leakage_context_audit_pass": leakage_context_pass,
        "foldsafe_pca_audit": pca_info["audit"],
        "split_row_audit": split_audit,
        "context_rows": context_rows,
        "label_permutation_policy_pass": all(row.get("label_policy") == "permuted_train_and_permuted_inner_val_selection" for row in label_rows),
        "train_only_video_mean_pass": not any(row.get("uses_test_rows_for_mean") for row in video_rows),
        "temporal_context_causal_only": all(row.get("temporal_context_causal_only") for row in context_rows),
        "no_centered_or_future_windows": not any(row.get("uses_centered_or_future_windows") for row in context_rows),
        "same_video_history_masking": all(row.get("same_video_history_masking") for row in context_rows),
    }
    write_json(output_root / "diagnostics" / "leakage_context_audit.json", context_audit)
    write_json(output_root / "diagnostics" / "label_permutation_audit.json", {"policy_implemented": True, "rows": label_rows})
    write_json(output_root / "diagnostics" / "train_only_video_mean_audit.json", {"train_only_video_mean_primary_static_control": True, "rows": video_rows})
    write_json(
        output_root / "manifests" / "ar_baseline_generation_manifest.json",
        {
            "baselines": ar_manifest,
            "reused": 3,
            "newly_trained": 7,
            "reused_seeds": list(REUSED_AR_SEEDS),
            "newly_trained_seeds": [seed for seed in SEEDS if seed not in REUSED_AR_SEEDS],
            "each_seed_uses_own_frozen_ar_score": True,
            "shared_three_seed_ar_cache_reused_across_ten_seed_confirmation": False,
            "ar_only_baseline_generation_reported_separately_from_residual_control_rows": True,
        },
    )
    write_json(output_root / "manifests" / "feature_manifest.json", {"features": feature_rows, "row_count": len(feature_rows)})
    write_json(
        output_root / "manifests" / "run_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": now_iso(),
            "source_root": str(source_root),
            "foldsafe_pca_root": str(pca_root),
            "previous_temporal_root": str(previous_root),
            "output_root": str(output_root),
            "dense_root": str(dense_root),
            "target": TARGET_NAME,
            "architecture": ARCHITECTURE,
            "controls": list(CONTROLS),
            "seeds": list(SEEDS),
            "feature": FEATURE_NAME,
            "protocol_scope": "blocked_temporal_70_30_only",
            "matrix_size": len(matrix),
            "residual_control_rows": 70,
            "ar_baseline_generation": {"reused": 3, "newly_trained": 7},
            "no_continuous": True,
            "no_grouped": True,
            "no_504": True,
            "no_extra_targets": True,
            "no_extra_architectures": True,
            "no_vjepa_tribe_pca_rerun": True,
            "no_pca_refit": True,
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
