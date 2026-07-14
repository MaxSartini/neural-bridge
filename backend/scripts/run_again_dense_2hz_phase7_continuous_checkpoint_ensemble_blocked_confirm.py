#!/usr/bin/env python3
"""Fresh control-complete Phase 7 blocked continuous ranking/lift confirmation."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts import run_again_dense_2hz_phase5_frozen_ar_residual as fr  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_learned_heads as base  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_binary_big_confirm as binary_confirm  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_blocked as temporal  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase6_fixed_blend_fresh5 as fixed  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase7_continuous_checkpoint_ensemble_diagnostic as diagnostic  # noqa: E402

SCHEMA_VERSION = "again_dense_2hz_phase7_continuous_checkpoint_ensemble_blocked_confirm_v1"
TARGET_NAME = diagnostic.TARGET_NAME
ARCHITECTURE = diagnostic.ARCHITECTURE
PARAMS = diagnostic.PARAMS
CONTROLS = diagnostic.CONTROLS
TRAINED_CONTROLS = diagnostic.TRAINED_CONTROLS
PRIMARY_CONTROLS = diagnostic.PRIMARY_CONTROLS
SEEDS = tuple(range(20260693, 20260708))
GROUPS = tuple(tuple(SEEDS[i : i + 3]) for i in range(0, 15, 3))
EXPECTED_MEMBER_ROWS = len(SEEDS) * len(CONTROLS)
EXPECTED_ENSEMBLE_ROWS = len(GROUPS) * len(CONTROLS)
EXPECTED_ROWS = EXPECTED_MEMBER_ROWS + EXPECTED_ENSEMBLE_ROWS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(binary_confirm.SOURCE_ROOT))
    parser.add_argument("--foldsafe-pca-root", default=str(binary_confirm.FOLDSAFE_PCA_ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_output_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(f"outputs/again_dense_2hz_phase7_continuous_checkpoint_ensemble_blocked_confirm_{stamp}")


def lane_means(frame: pd.DataFrame, metric: str) -> dict[str, float]:
    return {
        str(lane): float(rows[metric].mean())
        for lane, rows in frame.groupby("lane", sort=False)
    }


def best_control(means: dict[str, float], *, higher: bool) -> tuple[str, float]:
    values = {lane: means[lane] for lane in PRIMARY_CONTROLS}
    chooser = max if higher else min
    lane = chooser(values, key=values.get)
    return lane, float(values[lane])


def compute_verdict(rows: pd.DataFrame, audit_pass: bool) -> tuple[dict[str, Any], pd.DataFrame]:
    members = rows[rows["row_type"] == "member"].copy()
    ensembles = rows[rows["row_type"] == "ensemble"].copy()
    metrics = (
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
    means = {metric: lane_means(ensembles, metric) for metric in metrics}
    best_spearman_lane, best_spearman = best_control(
        means["continuous_spearman"], higher=True
    )
    best_top1_lane, best_top1 = best_control(
        means["top_1pct_continuous_lift"], higher=True
    )
    best_top5_lane, best_top5 = best_control(
        means["top_5pct_continuous_lift"], higher=True
    )
    best_top10_lane, best_top10 = best_control(
        means["top_10pct_continuous_lift"], higher=True
    )
    best_mae_lane, best_mae = best_control(means["continuous_mae"], higher=False)
    best_rmse_lane, best_rmse = best_control(means["continuous_rmse"], higher=False)

    group_rows: list[dict[str, Any]] = []
    for group in range(1, len(GROUPS) + 1):
        sub = ensembles[ensembles["group"] == group].set_index("lane")
        group_rows.append(
            {
                "group": group,
                "spearman_real_minus_ar": float(
                    sub.loc["real_residual", "continuous_spearman"]
                    - sub.loc["frozen_ar_only", "continuous_spearman"]
                ),
                "spearman_real_minus_best_control": float(
                    sub.loc["real_residual", "continuous_spearman"]
                    - sub.loc[best_spearman_lane, "continuous_spearman"]
                ),
                "top5_real_minus_ar": float(
                    sub.loc["real_residual", "top_5pct_continuous_lift"]
                    - sub.loc["frozen_ar_only", "top_5pct_continuous_lift"]
                ),
                "top5_real_minus_best_control": float(
                    sub.loc["real_residual", "top_5pct_continuous_lift"]
                    - sub.loc[best_top5_lane, "top_5pct_continuous_lift"]
                ),
            }
        )
    group_frame = pd.DataFrame(group_rows)

    spearman = means["continuous_spearman"]
    top1 = means["top_1pct_continuous_lift"]
    top5 = means["top_5pct_continuous_lift"]
    top10 = means["top_10pct_continuous_lift"]
    real_members = members[members["lane"] == "real_residual"]
    real_ensembles = ensembles[ensembles["lane"] == "real_residual"]
    member_std = float(real_members["continuous_spearman"].std(ddof=0))
    ensemble_std = float(real_ensembles["continuous_spearman"].std(ddof=0))
    scope_pass = bool(
        len(rows) == EXPECTED_ROWS
        and len(members) == EXPECTED_MEMBER_ROWS
        and len(ensembles) == EXPECTED_ENSEMBLE_ROWS
        and set(members["seed"]) == set(SEEDS)
        and set(ensembles["group"]) == set(range(1, len(GROUPS) + 1))
        and set(rows["lane"]) == set(CONTROLS)
    )
    checks = {
        "exact_scope": scope_pass,
        "audit_pass": bool(audit_pass),
        "spearman_mean_delta_vs_ar_at_least_0_002": spearman["real_residual"] - spearman["frozen_ar_only"] >= 0.002,
        "spearman_mean_delta_vs_best_control_at_least_0_002": spearman["real_residual"] - best_spearman >= 0.002,
        "top5_mean_delta_vs_ar_at_least_0_001": top5["real_residual"] - top5["frozen_ar_only"] >= 0.001,
        "top5_mean_delta_vs_best_control_at_least_0_001": top5["real_residual"] - best_top5 >= 0.001,
        "top1_mean_beats_ar_and_best_control": top1["real_residual"] > max(top1["frozen_ar_only"], best_top1),
        "top10_mean_beats_ar_and_best_control": top10["real_residual"] > max(top10["frozen_ar_only"], best_top10),
        "spearman_positive_vs_ar_5_of_5": int((group_frame["spearman_real_minus_ar"] > 0).sum()) == 5,
        "spearman_positive_vs_best_control_5_of_5": int((group_frame["spearman_real_minus_best_control"] > 0).sum()) == 5,
        "top5_positive_vs_ar_5_of_5": int((group_frame["top5_real_minus_ar"] > 0).sum()) == 5,
        "top5_positive_vs_best_control_5_of_5": int((group_frame["top5_real_minus_best_control"] > 0).sum()) == 5,
        "spearman_group_median_delta_vs_ar_at_least_0_002": float(group_frame["spearman_real_minus_ar"].median()) >= 0.002,
        "top5_group_median_delta_vs_ar_at_least_0_001": float(group_frame["top5_real_minus_ar"].median()) >= 0.001,
        "ensemble_spearman_uplift_at_least_0_001": float(real_ensembles["continuous_spearman"].mean() - real_members["continuous_spearman"].mean()) >= 0.001,
        "ensemble_top5_uplift_positive": float(real_ensembles["top_5pct_continuous_lift"].mean() - real_members["top_5pct_continuous_lift"].mean()) > 0,
        "ensemble_spearman_std_at_least_20pct_below_members": ensemble_std <= 0.80 * member_std,
        "real_beats_label_permutation_on_spearman_and_top5": spearman["real_residual"] > spearman["label_permutation_residual"] and top5["real_residual"] > top5["label_permutation_residual"],
        "single_group_spearman_contribution_at_most_0_50": diagnostic.max_positive_contribution(group_frame["spearman_real_minus_ar"]) <= 0.50,
        "single_group_top5_contribution_at_most_0_50": diagnostic.max_positive_contribution(group_frame["top5_real_minus_ar"]) <= 0.50,
    }
    failed = [name for name, passed in checks.items() if not passed]
    passed = not failed
    result = {
        "schema_version": SCHEMA_VERSION,
        "target": TARGET_NAME,
        "architecture": ARCHITECTURE,
        "rows_expected": EXPECTED_ROWS,
        "rows_actual": int(len(rows)),
        "blocked_continuous_ranking_lift_confirmation_pass": passed,
        "blocked_continuous_ranking_lift_proven": passed,
        "exact_continuous_value_forecasting_proven": False,
        "grouped_continuous_confirmation_authorized": passed,
        "checks": checks,
        "failed_gates": failed,
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
        "ensemble_spearman_uplift": float(real_ensembles["continuous_spearman"].mean() - real_members["continuous_spearman"].mean()),
        "ensemble_top5_uplift": float(real_ensembles["top_5pct_continuous_lift"].mean() - real_members["top_5pct_continuous_lift"].mean()),
        "member_spearman_std": member_std,
        "ensemble_spearman_std": ensemble_std,
        "real_mae": means["continuous_mae"]["real_residual"],
        "ar_mae": means["continuous_mae"]["frozen_ar_only"],
        "best_control_mae_lane": best_mae_lane,
        "best_control_mae": best_mae,
        "real_rmse": means["continuous_rmse"]["real_residual"],
        "ar_rmse": means["continuous_rmse"]["frozen_ar_only"],
        "best_control_rmse_lane": best_rmse_lane,
        "best_control_rmse": best_rmse,
    }
    return result, group_frame


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
        "accelerator": "mlx_gpu_mps",
        "no_cpu_fallback": True,
        "no_vjepa_tribe_pca_rerun": True,
        "no_pca_refit": True,
        "no_grouped": True,
        "confirmation_scope": "blocked_continuous_ranking_lift_only",
        "exact_value_promotion_forbidden": True,
    }


def report_text(result: dict[str, Any], root: Path) -> str:
    return f"""# AGAIN Phase 7 blocked continuous checkpoint-ensemble confirmation

- Output: `{root}`
- Target/head: `{TARGET_NAME}` / `{ARCHITECTURE}`
- Matrix: `{result['rows_actual']}/{result['rows_expected']}` rows
- Real / AR / best-control Spearman: `{result['real_spearman']:.10f}` / `{result['ar_spearman']:.10f}` / `{result['best_control_spearman']:.10f}`
- Real minus AR / best-control Spearman: `{result['real_minus_ar_spearman']:+.10f}` / `{result['real_minus_best_control_spearman']:+.10f}`
- Real / AR / best-control top-5% lift: `{result['real_top5_lift']:.10f}` / `{result['ar_top5_lift']:.10f}` / `{result['best_control_top5_lift']:.10f}`
- Real minus AR / best-control top-5% lift: `{result['real_minus_ar_top5_lift']:+.10f}` / `{result['real_minus_best_control_top5_lift']:+.10f}`
- Blocked continuous ranking/lift confirmation pass: `{result['blocked_continuous_ranking_lift_confirmation_pass']}`
- Failed gates: `{result['failed_gates']}`

This confirmation is scoped to blocked continuous future-movement ranking/lift. Exact continuous values remain unproven regardless of this verdict.
"""


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
        ar_curves = diagnostic.train_continuous_ar_inner_only(
            block=block,
            seed=seed,
            output_root=root,
            batch_size=args.batch_size,
            max_epochs=80,
            patience=12,
        )
        curves.extend({"lane": "frozen_ar_only", **curve} for curve in ar_curves)
        ar = diagnostic.load_continuous_ar(root, block, seed, args.batch_size)
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
            averaged = diagnostic.average_scores([scores[seed][lane] for seed in group])
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
    result, group_frame = compute_verdict(frame, frozen_integrity and context_pass)
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
        "again_dense_2hz_phase7_continuous_checkpoint_ensemble_blocked_confirm_"
    )
    report_name = f"again_dense_2hz_phase7_continuous_checkpoint_ensemble_blocked_confirm_{stamp}.md"
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
