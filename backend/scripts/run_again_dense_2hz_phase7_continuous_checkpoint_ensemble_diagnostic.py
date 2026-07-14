#!/usr/bin/env python3
"""Locked Phase 7 blocked continuous checkpoint-ensemble diagnostic.

The run uses the washout continuous target, the selected short temporal-conv
residual head, nine untouched seeds in three fixed groups, and all established
matched controls. It is a diagnostic, not a promotion confirmation.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts import run_again_dense_2hz_phase5_frozen_ar_residual as fr  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_learned_heads as base  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_binary_big_confirm as confirm  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_blocked as temporal  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase6_fixed_blend_fresh5 as fixed  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase6_robust_multiseed_optuna as robust  # noqa: E402

SCHEMA_VERSION = "again_dense_2hz_phase7_continuous_checkpoint_ensemble_diagnostic_v1"
TARGET_NAME = temporal.CONTINUOUS_TARGET
ARCHITECTURE = "short_temporal_conv_residual"
SEEDS = tuple(range(20260684, 20260693))
GROUPS = tuple(tuple(SEEDS[i : i + 3]) for i in range(0, 9, 3))
CONTROLS = temporal.CONTROLS
TRAINED_CONTROLS = tuple(control for control in CONTROLS if control != "frozen_ar_only")
PRIMARY_CONTROLS = temporal.PRIMARY_CONTROLS
EXPECTED_MEMBER_ROWS = len(SEEDS) * len(CONTROLS)
EXPECTED_ENSEMBLE_ROWS = len(GROUPS) * len(CONTROLS)
EXPECTED_ROWS = EXPECTED_MEMBER_ROWS + EXPECTED_ENSEMBLE_ROWS
PARAMS = robust.ORIGINAL_PARAMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(confirm.SOURCE_ROOT))
    parser.add_argument("--foldsafe-pca-root", default=str(confirm.FOLDSAFE_PCA_ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_output_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(f"outputs/again_dense_2hz_phase7_continuous_checkpoint_ensemble_diagnostic_{stamp}")


def average_scores(items: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if len(items) != 3:
        raise ValueError("Exactly three checkpoints are required")
    keys = ("train_score", "train_reg", "test_score", "test_reg")
    lengths = {key: {len(item[key]) for item in items} for key in keys}
    if any(len(values) != 1 for values in lengths.values()):
        raise ValueError("Checkpoint score rows are not aligned")
    return {
        key: np.mean(np.stack([item[key] for item in items]), axis=0).astype(np.float32)
        for key in keys
    }


def train_continuous_ar_inner_only(
    *, block: Any, seed: int, output_root: Path, batch_size: int, max_epochs: int, patience: int
) -> list[dict[str, Any]]:
    """Train and select the target-specific AR checkpoint without outer scoring."""
    base.require_mlx()
    base.mx.random.seed(int(seed))
    config = confirm.ar_config(
        seed, max_epochs=max_epochs, patience=patience, batch_size=batch_size
    )
    model = base.make_model(config, block.ar_train_x.shape[1], block.ar_block_dims)
    optimizer = base.optim.AdamW(
        learning_rate=config.learning_rate, weight_decay=config.weight_decay
    )
    rng = np.random.default_rng(int(seed) + 70001)
    checkpoint = output_root / "ar_baseline_checkpoints" / f"seed{seed}__best.npz"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_key = (-math.inf, -math.inf, -math.inf)
    stale = 0
    curves: list[dict[str, Any]] = []
    q80 = float(np.quantile(block.train_cont[block.inner_train], 0.80))
    q90 = float(np.quantile(block.train_cont[block.inner_train], 0.90))

    def loss_fn(model_obj: Any, xb: Any, yr: Any, weights: Any) -> Any:
        out = model_obj(xb)
        if out.ndim == 1:
            out = out[:, None]
        return base.mx.mean(
            base.nn.losses.huber_loss(out[:, 0:1], yr, delta=1.0) * weights
        )

    loss_and_grad = base.nn.value_and_grad(model, loss_fn)
    for epoch in range(1, max_epochs + 1):
        if hasattr(model, "train"):
            model.train()
        order = rng.permutation(block.inner_train)
        for start in range(0, len(order), batch_size):
            rel = order[start : start + batch_size]
            target = block.train_cont[rel].astype(np.float32)[:, None]
            weights = (
                1.0
                + 1.0 * (target >= q80).astype(np.float32)
                + 2.0 * (target >= q90).astype(np.float32)
            )
            loss, grads = loss_and_grad(
                model,
                base.mx.array(block.ar_train_x[rel], dtype=base.mx.float32),
                base.mx.array(target, dtype=base.mx.float32),
                base.mx.array(weights, dtype=base.mx.float32),
            )
            grads, _ = base.optim.clip_grad_norm(grads, 1.0)
            optimizer.update(model, grads)
            base.mx.eval(loss, model.parameters(), optimizer.state)
        if hasattr(model, "eval"):
            model.eval()
        val_score, val_reg = fr.score_existing_model(
            model, block.ar_train_x[block.inner_val], batch_size
        )
        val_metrics = temporal.continuous_run.continuous_metric_row(
            block.train_y[block.inner_train],
            np.zeros(len(block.inner_train), dtype=np.float32),
            block.train_y[block.inner_val],
            val_score,
            block.train_cont[block.inner_val],
            val_reg,
        )
        key = (
            float(val_metrics["top_5pct_continuous_lift"]),
            float(val_metrics["continuous_spearman"]),
            float(val_metrics["top_10pct_continuous_lift"]),
        )
        curves.append(
            {
                "seed": seed,
                "epoch": epoch,
                "inner_val_top_5pct_continuous_lift": key[0],
                "inner_val_continuous_spearman": key[1],
                "inner_val_top_10pct_continuous_lift": key[2],
            }
        )
        if all(math.isfinite(value) for value in key) and key > best_key:
            model.save_weights(str(checkpoint))
            best_key = key
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if not checkpoint.exists():
        raise RuntimeError(f"Target-specific AR did not produce a valid checkpoint for seed {seed}")
    return curves


def load_continuous_ar(
    output_root: Path, block: Any, seed: int, batch_size: int
) -> dict[str, Any]:
    checkpoint = output_root / "ar_baseline_checkpoints" / f"seed{seed}__best.npz"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    config = confirm.ar_config(seed, max_epochs=80, patience=12, batch_size=batch_size)
    model = base.make_model(config, block.ar_train_x.shape[1], block.ar_block_dims)
    _ = model(base.mx.array(block.ar_train_x[:2], dtype=base.mx.float32))
    model.load_weights(str(checkpoint))
    if hasattr(model, "eval"):
        model.eval()
    train_score, train_reg = fr.score_existing_model(model, block.ar_train_x, batch_size)
    test_score, test_reg = fr.score_existing_model(model, block.ar_test_x, batch_size)
    return {
        "seed": seed,
        "source": "target_specific_continuous_inner_only_ar_checkpoint",
        "train_score": train_score,
        "train_reg": train_reg,
        "test_score": test_score,
        "test_reg": test_reg,
        "train_checksum": fr.hash_array(train_score),
        "test_checksum": fr.hash_array(test_score),
        "checkpoint_restore_pass": True,
        "eval_mode_scoring_pass": True,
    }


def metric_means(groups: pd.DataFrame, metric: str) -> dict[str, float]:
    return {
        str(lane): float(values[metric].mean())
        for lane, values in groups.groupby("lane", sort=False)
    }


def max_positive_contribution(values: pd.Series) -> float:
    positive = values[values > 0]
    total = float(positive.sum())
    return float(positive.max() / total) if total > 0 else math.inf


def best_lane(means: dict[str, float], *, higher_is_better: bool) -> tuple[str, float]:
    candidates = {lane: means[lane] for lane in PRIMARY_CONTROLS}
    chooser = max if higher_is_better else min
    lane = chooser(candidates, key=candidates.get)
    return lane, float(candidates[lane])


def compute_verdict(rows: pd.DataFrame, audit_pass: bool) -> tuple[dict[str, Any], pd.DataFrame]:
    members = rows[rows["row_type"] == "member"].copy()
    ensembles = rows[rows["row_type"] == "ensemble"].copy()
    means_by_metric = {
        metric: metric_means(ensembles, metric)
        for metric in (
            "continuous_spearman",
            "continuous_pearson",
            "continuous_mae",
            "continuous_rmse",
            "continuous_bias",
            "peak_underprediction",
            "top_1pct_continuous_lift",
            "top_5pct_continuous_lift",
            "top_10pct_continuous_lift",
        )
    }
    high_metrics = (
        "continuous_spearman",
        "continuous_pearson",
        "top_1pct_continuous_lift",
        "top_5pct_continuous_lift",
        "top_10pct_continuous_lift",
    )
    best_controls = {
        metric: best_lane(means_by_metric[metric], higher_is_better=metric in high_metrics)
        for metric in means_by_metric
        if metric != "continuous_bias"
    }
    group_rows: list[dict[str, Any]] = []
    for group in range(1, len(GROUPS) + 1):
        sub = ensembles[ensembles["group"] == group].set_index("lane")
        row: dict[str, Any] = {"group": group}
        for metric in (
            "continuous_spearman",
            "continuous_mae",
            "continuous_rmse",
            "top_5pct_continuous_lift",
        ):
            best_control = best_controls[metric][0]
            real = float(sub.loc["real_residual", metric])
            ar = float(sub.loc["frozen_ar_only", metric])
            control = float(sub.loc[best_control, metric])
            if metric in high_metrics:
                row[f"{metric}_real_minus_ar"] = real - ar
                row[f"{metric}_real_minus_best_control"] = real - control
            else:
                row[f"{metric}_ar_minus_real"] = ar - real
                row[f"{metric}_best_control_minus_real"] = control - real
        group_rows.append(row)
    group_frame = pd.DataFrame(group_rows)

    spearman = means_by_metric["continuous_spearman"]
    top1 = means_by_metric["top_1pct_continuous_lift"]
    top5 = means_by_metric["top_5pct_continuous_lift"]
    top10 = means_by_metric["top_10pct_continuous_lift"]
    mae = means_by_metric["continuous_mae"]
    rmse = means_by_metric["continuous_rmse"]
    bias = means_by_metric["continuous_bias"]
    peak = means_by_metric["peak_underprediction"]
    best_spearman_lane, best_spearman = best_controls["continuous_spearman"]
    best_top1_lane, best_top1 = best_controls["top_1pct_continuous_lift"]
    best_top5_lane, best_top5 = best_controls["top_5pct_continuous_lift"]
    best_top10_lane, best_top10 = best_controls["top_10pct_continuous_lift"]
    best_mae_lane, best_mae = best_controls["continuous_mae"]
    best_rmse_lane, best_rmse = best_controls["continuous_rmse"]

    real_members = members[members["lane"] == "real_residual"]
    real_ensembles = ensembles[ensembles["lane"] == "real_residual"]
    spearman_ar_deltas = group_frame["continuous_spearman_real_minus_ar"]
    spearman_control_deltas = group_frame["continuous_spearman_real_minus_best_control"]
    top5_ar_deltas = group_frame["top_5pct_continuous_lift_real_minus_ar"]
    top5_control_deltas = group_frame["top_5pct_continuous_lift_real_minus_best_control"]

    scope_pass = bool(
        len(rows) == EXPECTED_ROWS
        and len(members) == EXPECTED_MEMBER_ROWS
        and len(ensembles) == EXPECTED_ENSEMBLE_ROWS
        and set(members["seed"]) == set(SEEDS)
        and set(ensembles["group"]) == set(range(1, len(GROUPS) + 1))
        and set(rows["lane"]) == set(CONTROLS)
    )
    ranking_checks = {
        "exact_scope": scope_pass,
        "audit_pass": bool(audit_pass),
        "spearman_mean_delta_vs_ar_at_least_0_002": spearman["real_residual"] - spearman["frozen_ar_only"] >= 0.002,
        "spearman_mean_delta_vs_best_control_at_least_0_002": spearman["real_residual"] - best_spearman >= 0.002,
        "top5_mean_delta_vs_ar_at_least_0_001": top5["real_residual"] - top5["frozen_ar_only"] >= 0.001,
        "top5_mean_delta_vs_best_control_at_least_0_001": top5["real_residual"] - best_top5 >= 0.001,
        "top1_mean_beats_ar_and_best_control": top1["real_residual"] > max(top1["frozen_ar_only"], best_top1),
        "top10_mean_beats_ar_and_best_control": top10["real_residual"] > max(top10["frozen_ar_only"], best_top10),
        "spearman_positive_vs_ar_all_groups": bool((spearman_ar_deltas > 0).all()),
        "spearman_positive_vs_best_control_all_groups": bool((spearman_control_deltas > 0).all()),
        "top5_positive_vs_ar_all_groups": bool((top5_ar_deltas > 0).all()),
        "top5_positive_vs_best_control_all_groups": bool((top5_control_deltas > 0).all()),
        "ensemble_spearman_uplift_at_least_0_001": float(real_ensembles["continuous_spearman"].mean() - real_members["continuous_spearman"].mean()) >= 0.001,
        "ensemble_top5_uplift_positive": float(real_ensembles["top_5pct_continuous_lift"].mean() - real_members["top_5pct_continuous_lift"].mean()) > 0,
        "real_beats_label_permutation_on_spearman_and_top5": spearman["real_residual"] > spearman["label_permutation_residual"] and top5["real_residual"] > top5["label_permutation_residual"],
        "single_group_spearman_contribution_at_most_0_60": max_positive_contribution(spearman_ar_deltas) <= 0.60,
        "single_group_top5_contribution_at_most_0_60": max_positive_contribution(top5_ar_deltas) <= 0.60,
    }
    exact_checks = {
        "mae_mean_improvement_vs_ar_at_least_0_0005": mae["frozen_ar_only"] - mae["real_residual"] >= 0.0005,
        "mae_mean_improvement_vs_best_control_at_least_0_0005": best_mae - mae["real_residual"] >= 0.0005,
        "rmse_mean_improvement_vs_ar_at_least_0_0005": rmse["frozen_ar_only"] - rmse["real_residual"] >= 0.0005,
        "rmse_mean_improvement_vs_best_control_at_least_0_0005": best_rmse - rmse["real_residual"] >= 0.0005,
        "mae_positive_vs_ar_and_best_control_all_groups": bool(
            (group_frame["continuous_mae_ar_minus_real"] > 0).all()
            and (group_frame["continuous_mae_best_control_minus_real"] > 0).all()
        ),
        "rmse_positive_vs_ar_and_best_control_all_groups": bool(
            (group_frame["continuous_rmse_ar_minus_real"] > 0).all()
            and (group_frame["continuous_rmse_best_control_minus_real"] > 0).all()
        ),
        "absolute_bias_no_worse_than_ar": abs(bias["real_residual"]) <= abs(bias["frozen_ar_only"]),
        "peak_underprediction_better_than_ar": peak["real_residual"] < peak["frozen_ar_only"],
    }
    failed_ranking = [name for name, passed in ranking_checks.items() if not passed]
    failed_exact = [name for name, passed in exact_checks.items() if not passed]
    ranking_pass = not failed_ranking
    exact_pass = ranking_pass and not failed_exact
    result = {
        "schema_version": SCHEMA_VERSION,
        "target": TARGET_NAME,
        "architecture": ARCHITECTURE,
        "rows_expected": EXPECTED_ROWS,
        "rows_actual": int(len(rows)),
        "ranking_lift_diagnostic_pass": ranking_pass,
        "exact_value_candidate_pass": exact_pass,
        "fresh_blocked_ranking_confirmation_authorized": ranking_pass,
        "fresh_exact_value_confirmation_authorized": exact_pass,
        "grouped_followup_authorized": False,
        "failed_ranking_gates": failed_ranking,
        "failed_exact_value_gates": failed_exact,
        "ranking_checks": ranking_checks,
        "exact_value_checks": exact_checks,
        "real_spearman": spearman["real_residual"],
        "ar_spearman": spearman["frozen_ar_only"],
        "best_control_spearman_lane": best_spearman_lane,
        "best_control_spearman": best_spearman,
        "real_minus_ar_spearman": spearman["real_residual"] - spearman["frozen_ar_only"],
        "real_minus_best_control_spearman": spearman["real_residual"] - best_spearman,
        "real_top5_lift": top5["real_residual"],
        "ar_top5_lift": top5["frozen_ar_only"],
        "best_control_top5_lane": best_top5_lane,
        "best_control_top5_lift": best_top5,
        "real_minus_ar_top5_lift": top5["real_residual"] - top5["frozen_ar_only"],
        "real_minus_best_control_top5_lift": top5["real_residual"] - best_top5,
        "real_mae": mae["real_residual"],
        "ar_mae": mae["frozen_ar_only"],
        "best_control_mae_lane": best_mae_lane,
        "best_control_mae": best_mae,
        "real_rmse": rmse["real_residual"],
        "ar_rmse": rmse["frozen_ar_only"],
        "best_control_rmse_lane": best_rmse_lane,
        "best_control_rmse": best_rmse,
        "ensemble_spearman_uplift": float(real_ensembles["continuous_spearman"].mean() - real_members["continuous_spearman"].mean()),
        "ensemble_top5_uplift": float(real_ensembles["top_5pct_continuous_lift"].mean() - real_members["top_5pct_continuous_lift"].mean()),
        "member_spearman_std": float(real_members["continuous_spearman"].std(ddof=0)),
        "ensemble_spearman_std": float(real_ensembles["continuous_spearman"].std(ddof=0)),
    }
    return result, group_frame


def report_text(result: dict[str, Any], root: Path) -> str:
    return f"""# AGAIN Phase 7 blocked continuous checkpoint-ensemble diagnostic

- Output: `{root}`
- Target/head: `{TARGET_NAME}` / `{ARCHITECTURE}`
- Matrix: `{result['rows_actual']}/{result['rows_expected']}` rows
- Real / AR / best-control Spearman: `{result['real_spearman']:.10f}` / `{result['ar_spearman']:.10f}` / `{result['best_control_spearman']:.10f}`
- Real minus AR / best-control Spearman: `{result['real_minus_ar_spearman']:+.10f}` / `{result['real_minus_best_control_spearman']:+.10f}`
- Real / AR / best-control top-5% lift: `{result['real_top5_lift']:.10f}` / `{result['ar_top5_lift']:.10f}` / `{result['best_control_top5_lift']:.10f}`
- Real minus AR / best-control top-5% lift: `{result['real_minus_ar_top5_lift']:+.10f}` / `{result['real_minus_best_control_top5_lift']:+.10f}`
- Ranking/lift diagnostic pass: `{result['ranking_lift_diagnostic_pass']}`
- Exact-value candidate pass: `{result['exact_value_candidate_pass']}`
- Failed ranking gates: `{result['failed_ranking_gates']}`
- Failed exact-value gates: `{result['failed_exact_value_gates']}`

This bounded diagnostic cannot itself promote blocked continuous generalization or exact-value forecasting. A fresh preregistered confirmation is required for any promotion.
"""


def run_manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "target": TARGET_NAME,
        "architecture": ARCHITECTURE,
        "seeds": list(SEEDS),
        "groups": [list(group) for group in GROUPS],
        "controls": list(CONTROLS),
        "primary_controls": list(PRIMARY_CONTROLS),
        "rows": EXPECTED_ROWS,
        "params": PARAMS,
        "member_selection": False,
        "heldout_weight_search": False,
        "heldout_hyperparameter_search": False,
        "optuna_used": False,
        "ar_selection": "inner_only_top5_lift_then_spearman_then_top10_lift",
        "ar_loss": "top20_top10_weighted_continuous_huber",
        "accelerator": "mlx_gpu_mps",
        "no_cpu_fallback": True,
        "no_vjepa_tribe_pca_rerun": True,
        "no_pca_refit": True,
        "no_grouped": True,
        "diagnostic_only": True,
    }


def main() -> int:
    args = parse_args()
    manifest = run_manifest()
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    base.require_mlx()
    root = Path(args.output_root) if args.output_root else default_output_root()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    for sub in ("ar_baseline_checkpoints", "checkpoints", "metrics", "diagnostics", "reports", "manifests"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    started = time.time()
    pca_root = Path(args.foldsafe_pca_root)
    blocks, df, dense_root, residual_meta = temporal.build_blocks(Path(args.source_root), pca_root)
    block = temporal.block_for_target(blocks, TARGET_NAME)
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    scores: dict[int, dict[str, dict[str, Any]]] = {}
    group_of = {seed: index + 1 for index, group in enumerate(GROUPS) for seed in group}

    for seed in SEEDS:
        ar_curves = train_continuous_ar_inner_only(
            block=block,
            seed=seed,
            output_root=root,
            batch_size=args.batch_size,
            max_epochs=80,
            patience=12,
        )
        curves.extend({"lane": "frozen_ar_only", **curve} for curve in ar_curves)
        ar = load_continuous_ar(root, block, seed, args.batch_size)
        ar_metrics = temporal.metric_row_for_block(
            block, ar["train_score"], ar["test_score"], ar["test_reg"]
        )
        rows.append(
            {
                "row_type": "member",
                "group": group_of[seed],
                "seed": seed,
                "lane": "frozen_ar_only",
                "ar_test_checksum": ar["test_checksum"],
                **ar_metrics,
            }
        )
        scores[seed] = {"frozen_ar_only": ar}
        for control in TRAINED_CONTROLS:
            pack = temporal.feature_pack_for(
                df, dense_root, pca_root, block, ARCHITECTURE, control, seed
            )
            metrics, lane_curves, audit = temporal.train_temporal_residual(
                architecture=ARCHITECTURE,
                control=control,
                pack=pack,
                block=block,
                ar=ar,
                seed=seed,
                output_root=root / "checkpoints" / control,
                batch_size=args.batch_size,
                max_epochs=int(PARAMS["max_epochs"]),
                patience=int(PARAMS["patience"]),
                hyperparameters=PARAMS,
            )
            curves.extend(lane_curves)
            audits.append(
                {
                    "seed": seed,
                    "control": control,
                    **audit,
                    "context": pack.context_audit,
                }
            )
            rows.append(
                {
                    "row_type": "member",
                    "group": group_of[seed],
                    "seed": seed,
                    "lane": control,
                    "ar_test_checksum": ar["test_checksum"],
                    **metrics,
                }
            )
            scores[seed][control] = fixed.restored_scores(
                audit=audit,
                params=PARAMS,
                pack=pack,
                block=block,
                ar=ar,
                batch_size=args.batch_size,
            )
        pd.DataFrame(rows).to_csv(root / "metrics/rows.partial.csv", index=False)

    for group_id, group in enumerate(GROUPS, 1):
        for lane in CONTROLS:
            averaged = average_scores([scores[seed][lane] for seed in group])
            metrics = temporal.metric_row_for_block(
                block,
                averaged["train_score"],
                averaged["test_score"],
                averaged["test_reg"],
            )
            rows.append(
                {
                    "row_type": "ensemble",
                    "group": group_id,
                    "seed": 0,
                    "lane": lane,
                    "member_seeds": ",".join(map(str, group)),
                    **metrics,
                }
            )

    frame = pd.DataFrame(rows)
    member_frame = frame[frame["row_type"] == "member"]
    frozen_integrity = bool(
        member_frame.groupby("seed")["ar_test_checksum"].nunique().max() == 1
    )
    context_pass = bool(
        len(audits) == len(SEEDS) * len(TRAINED_CONTROLS)
        and all(
            audit["context"].get("temporal_context_causal_only")
            and audit["context"].get("same_video_history_masking")
            and not audit["context"].get("uses_centered_or_future_windows")
            and (audit.get("checkpoint_restored") or audit.get("residual_suppressed"))
            and audit.get("eval_mode_scoring")
            for audit in audits
        )
    )
    audit_pass = frozen_integrity and context_pass
    result, group_frame = compute_verdict(frame, audit_pass)
    result.update(
        {
            "duration_seconds": time.time() - started,
            "accelerator_detail": "Device(gpu, 0)",
            "frozen_ar_integrity_pass": frozen_integrity,
            "context_checkpoint_eval_audit_pass": context_pass,
        }
    )
    frame.to_csv(root / "metrics/rows.csv", index=False)
    group_frame.to_csv(root / "metrics/group_deltas.csv", index=False)
    pd.DataFrame(curves).to_csv(root / "diagnostics/training_curves.csv", index=False)
    fr.write_json(root / "diagnostics/audits.json", {"rows": audits})
    fr.write_json(root / "metrics/result.json", result)
    fr.write_json(
        root / "manifests/run_manifest.json",
        {
            **manifest,
            "source_root": str(args.source_root),
            "foldsafe_pca_root": str(pca_root),
            "residual_target_definition": residual_meta,
            "duration_seconds": result["duration_seconds"],
        },
    )
    report = report_text(result, root)
    stamp = root.name.removeprefix(
        "again_dense_2hz_phase7_continuous_checkpoint_ensemble_diagnostic_"
    )
    report_name = f"again_dense_2hz_phase7_continuous_checkpoint_ensemble_diagnostic_{stamp}.md"
    (root / "reports" / report_name).write_text(report, encoding="utf-8")
    report_path = Path(args.reports_dir) / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {"run_completed": True, "output_root": str(root), "report": str(report_path), **result},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
