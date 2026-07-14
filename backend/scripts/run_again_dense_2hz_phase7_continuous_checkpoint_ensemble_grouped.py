#!/usr/bin/env python3
"""User-authorized grouped Phase 7 continuous ranking/lift validation."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts import run_again_dense_2hz_phase5_frozen_ar_residual as fr  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_learned_heads as base  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_blocked as temporal  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_grouped_compat as grouped  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase6_fixed_blend_fresh5 as fixed  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase7_continuous_checkpoint_ensemble_diagnostic as diagnostic  # noqa: E402

SCHEMA_VERSION = "again_dense_2hz_phase7_continuous_checkpoint_ensemble_grouped_v1"
TARGET_NAME = diagnostic.TARGET_NAME
ARCHITECTURE = diagnostic.ARCHITECTURE
PARAMS = diagnostic.PARAMS
FOLDS = tuple(grouped.FOLDS)
SEEDS = tuple(range(20260708, 20260717))
GROUPS = tuple(tuple(SEEDS[i : i + 3]) for i in range(0, 9, 3))
CONTROLS = diagnostic.CONTROLS
TRAINED_CONTROLS = diagnostic.TRAINED_CONTROLS
PRIMARY_CONTROLS = diagnostic.PRIMARY_CONTROLS
EXPECTED_MEMBER_ROWS = len(FOLDS) * len(SEEDS) * len(CONTROLS)
EXPECTED_ENSEMBLE_ROWS = len(FOLDS) * len(GROUPS) * len(CONTROLS)
EXPECTED_ROWS = EXPECTED_MEMBER_ROWS + EXPECTED_ENSEMBLE_ROWS
GROUPED_PCA_ROOT = Path(
    "outputs/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520/foldsafe_grouped_pca"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(grouped.SOURCE_ROOT))
    parser.add_argument("--grouped-pca-root", default=str(GROUPED_PCA_ROOT))
    parser.add_argument("--external-phase4-root", default=str(grouped.EXTERNAL_PHASE4_ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_output_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(f"outputs/again_dense_2hz_phase7_continuous_checkpoint_ensemble_grouped_{stamp}")


def lane_means(frame: pd.DataFrame, metric: str) -> dict[str, float]:
    return {
        str(lane): float(rows[metric].mean())
        for lane, rows in frame.groupby("lane", sort=False)
    }


def best_control(means: dict[str, float]) -> tuple[str, float]:
    values = {lane: means[lane] for lane in PRIMARY_CONTROLS}
    lane = max(values, key=values.get)
    return lane, float(values[lane])


def compute_verdict(rows: pd.DataFrame, audit_pass: bool) -> tuple[dict[str, Any], pd.DataFrame]:
    members = rows[rows["row_type"] == "member"].copy()
    ensembles = rows[rows["row_type"] == "ensemble"].copy()
    spearman = lane_means(ensembles, "continuous_spearman")
    top1 = lane_means(ensembles, "top_1pct_continuous_lift")
    top5 = lane_means(ensembles, "top_5pct_continuous_lift")
    top10 = lane_means(ensembles, "top_10pct_continuous_lift")
    mae = lane_means(ensembles, "continuous_mae")
    rmse = lane_means(ensembles, "continuous_rmse")
    best_spearman_lane, best_spearman = best_control(spearman)
    best_top1_lane, best_top1 = best_control(top1)
    best_top5_lane, best_top5 = best_control(top5)
    best_top10_lane, best_top10 = best_control(top10)

    delta_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        for group in range(1, len(GROUPS) + 1):
            sub = ensembles[
                (ensembles["fold"] == fold) & (ensembles["group"] == group)
            ].set_index("lane")
            delta_rows.append(
                {
                    "fold": fold,
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
    deltas = pd.DataFrame(delta_rows)
    fold_means = deltas.groupby("fold")[
        [
            "spearman_real_minus_ar",
            "spearman_real_minus_best_control",
            "top5_real_minus_ar",
            "top5_real_minus_best_control",
        ]
    ].mean()
    real_members = members[members["lane"] == "real_residual"]
    real_ensembles = ensembles[ensembles["lane"] == "real_residual"]
    scope_pass = bool(
        len(rows) == EXPECTED_ROWS
        and len(members) == EXPECTED_MEMBER_ROWS
        and len(ensembles) == EXPECTED_ENSEMBLE_ROWS
        and set(members["seed"]) == set(SEEDS)
        and set(rows["fold"]) == set(FOLDS)
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
        "spearman_positive_vs_ar_at_least_12_of_15": int((deltas["spearman_real_minus_ar"] > 0).sum()) >= 12,
        "spearman_positive_vs_best_control_at_least_12_of_15": int((deltas["spearman_real_minus_best_control"] > 0).sum()) >= 12,
        "top5_positive_vs_ar_at_least_12_of_15": int((deltas["top5_real_minus_ar"] > 0).sum()) >= 12,
        "top5_positive_vs_best_control_at_least_12_of_15": int((deltas["top5_real_minus_best_control"] > 0).sum()) >= 12,
        "all_five_fold_means_positive": bool((fold_means > 0).all().all()),
        "positive_paired_medians": bool(
            (deltas[
                [
                    "spearman_real_minus_ar",
                    "spearman_real_minus_best_control",
                    "top5_real_minus_ar",
                    "top5_real_minus_best_control",
                ]
            ].median() > 0).all()
        ),
        "ensemble_spearman_uplift_at_least_0_001": float(real_ensembles["continuous_spearman"].mean() - real_members["continuous_spearman"].mean()) >= 0.001,
        "ensemble_top5_uplift_positive": float(real_ensembles["top_5pct_continuous_lift"].mean() - real_members["top_5pct_continuous_lift"].mean()) > 0,
        "real_beats_label_permutation_on_spearman_and_top5": spearman["real_residual"] > spearman["label_permutation_residual"] and top5["real_residual"] > top5["label_permutation_residual"],
        "single_fold_group_spearman_contribution_at_most_0_25": diagnostic.max_positive_contribution(deltas["spearman_real_minus_ar"]) <= 0.25,
        "single_fold_group_top5_contribution_at_most_0_25": diagnostic.max_positive_contribution(deltas["top5_real_minus_ar"]) <= 0.25,
    }
    failed = [name for name, passed in checks.items() if not passed]
    passed = not failed
    result = {
        "schema_version": SCHEMA_VERSION,
        "target": TARGET_NAME,
        "architecture": ARCHITECTURE,
        "rows_expected": EXPECTED_ROWS,
        "rows_actual": int(len(rows)),
        "member_rows": int(len(members)),
        "ensemble_rows": int(len(ensembles)),
        "grouped_continuous_ranking_lift_pass": passed,
        "exact_continuous_value_forecasting_proven": False,
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
        "wins_spearman_vs_ar": int((deltas["spearman_real_minus_ar"] > 0).sum()),
        "wins_spearman_vs_best_control": int((deltas["spearman_real_minus_best_control"] > 0).sum()),
        "wins_top5_vs_ar": int((deltas["top5_real_minus_ar"] > 0).sum()),
        "wins_top5_vs_best_control": int((deltas["top5_real_minus_best_control"] > 0).sum()),
        "positive_fold_means": int((fold_means > 0).all(axis=1).sum()),
        "ensemble_spearman_uplift": float(real_ensembles["continuous_spearman"].mean() - real_members["continuous_spearman"].mean()),
        "ensemble_top5_uplift": float(real_ensembles["top_5pct_continuous_lift"].mean() - real_members["top_5pct_continuous_lift"].mean()),
        "real_mae": mae["real_residual"],
        "ar_mae": mae["frozen_ar_only"],
        "real_rmse": rmse["real_residual"],
        "ar_rmse": rmse["frozen_ar_only"],
    }
    return result, deltas


def run_manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "target": TARGET_NAME,
        "architecture": ARCHITECTURE,
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "groups": [list(group) for group in GROUPS],
        "controls": list(CONTROLS),
        "rows": EXPECTED_ROWS,
        "params": PARAMS,
        "authorization": "explicit_user_authorization_after_blocked_4_of_5_near_confirmation",
        "blocked_verdict_relabelled": False,
        "member_selection": False,
        "weight_search": False,
        "heldout_hyperparameter_search": False,
        "accelerator": "mlx_gpu_mps",
        "no_cpu_fallback": True,
        "no_vjepa_tribe_rerun": True,
        "no_pca_refit": True,
        "exact_value_promotion_forbidden": True,
    }


def report_text(result: dict[str, Any], root: Path) -> str:
    return f"""# AGAIN Phase 7 grouped continuous checkpoint-ensemble validation

- Output: `{root}`
- Matrix: `{result['rows_actual']}/{result['rows_expected']}` (`{result['member_rows']}` member + `{result['ensemble_rows']}` ensemble)
- Target/head: `{TARGET_NAME}` / `{ARCHITECTURE}`
- Real / AR / best-control Spearman: `{result['real_spearman']:.10f}` / `{result['ar_spearman']:.10f}` / `{result['best_control_spearman']:.10f}`
- Real minus AR / best-control Spearman: `{result['real_minus_ar_spearman']:+.10f}` / `{result['real_minus_best_control_spearman']:+.10f}`
- Real / AR / best-control top-5% lift: `{result['real_top5_lift']:.10f}` / `{result['ar_top5_lift']:.10f}` / `{result['best_control_top5_lift']:.10f}`
- Real minus AR / best-control top-5% lift: `{result['real_minus_ar_top5_lift']:+.10f}` / `{result['real_minus_best_control_top5_lift']:+.10f}`
- Grouped continuous ranking/lift pass: `{result['grouped_continuous_ranking_lift_pass']}`
- Failed gates: `{result['failed_gates']}`

This user-authorized grouped test does not relabel the prior blocked `4/5` verdict and cannot prove exact continuous values.
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
    for sub in ("metrics", "diagnostics", "reports", "manifests", "models"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    started = time.time()
    df, dense_root, splits, residual_meta = grouped.load_df_and_splits(Path(args.source_root))
    pca_args = argparse.Namespace(
        grouped_pca_root=args.grouped_pca_root,
        external_phase4_root=args.external_phase4_root,
        pca_batch_size=384,
        pca_oversampling=32,
        pca_power_iterations=1,
    )
    pca_root, pca_info = grouped.ensure_grouped_pca(pca_args, root, df, dense_root, splits)
    raw_blocks = grouped.build_blocks(df, dense_root, splits)
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    group_of = {seed: group_id for group_id, group in enumerate(GROUPS, 1) for seed in group}
    for fold in FOLDS:
        block = replace(
            raw_blocks[fold],
            target_name=f"{TARGET_NAME}__grouped_fold{fold}",
            target_type="continuous",
        )
        fold_root = root / f"fold{fold}"
        scores: dict[int, dict[str, dict[str, Any]]] = {}
        for seed in SEEDS:
            ar_curves = diagnostic.train_continuous_ar_inner_only(
                block=block,
                seed=seed,
                output_root=fold_root,
                batch_size=args.batch_size,
                max_epochs=80,
                patience=12,
            )
            curves.extend({"fold": fold, "lane": "frozen_ar_only", **curve} for curve in ar_curves)
            ar = diagnostic.load_continuous_ar(fold_root, block, seed, args.batch_size)
            ar_metrics = temporal.metric_row_for_block(
                block, ar["train_score"], ar["test_score"], ar["test_reg"]
            )
            rows.append(
                {
                    "row_type": "member",
                    "fold": fold,
                    "group": group_of[seed],
                    "seed": seed,
                    "lane": "frozen_ar_only",
                    "frozen_ar_train_checksum": ar["train_checksum"],
                    "frozen_ar_test_checksum": ar["test_checksum"],
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
                    output_root=root / "models" / f"fold{fold}" / control,
                    batch_size=args.batch_size,
                    max_epochs=int(PARAMS["max_epochs"]),
                    patience=int(PARAMS["patience"]),
                    hyperparameters=PARAMS,
                )
                curves.extend({"fold": fold, **curve} for curve in lane_curves)
                audits.append(
                    {"fold": fold, "seed": seed, "control": control, **audit, "context": pack.context_audit}
                )
                rows.append(
                    {
                        "row_type": "member",
                        "fold": fold,
                        "group": group_of[seed],
                        "seed": seed,
                        "lane": control,
                        "frozen_ar_train_checksum": ar["train_checksum"],
                        "frozen_ar_test_checksum": ar["test_checksum"],
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
        for group_id, members in enumerate(GROUPS, 1):
            for lane in CONTROLS:
                averaged = diagnostic.average_scores([scores[seed][lane] for seed in members])
                metrics = temporal.metric_row_for_block(
                    block,
                    averaged["train_score"],
                    averaged["test_score"],
                    averaged["test_reg"],
                )
                rows.append(
                    {
                        "row_type": "ensemble",
                        "fold": fold,
                        "group": group_id,
                        "seed": 0,
                        "lane": lane,
                        "member_seeds": ",".join(map(str, members)),
                        **metrics,
                    }
                )
        pd.DataFrame(rows).to_csv(root / "metrics/rows.partial.csv", index=False)
        del scores
        gc.collect()

    frame = pd.DataFrame(rows)
    member_frame = frame[frame["row_type"] == "member"]
    ar_identity = bool(
        member_frame.groupby(["fold", "seed"])[
            ["frozen_ar_train_checksum", "frozen_ar_test_checksum"]
        ].nunique(dropna=True).fillna(1).le(1).all().all()
    )
    context_pass = bool(
        len(audits) == len(FOLDS) * len(SEEDS) * len(TRAINED_CONTROLS)
        and all(
            audit["context"].get("temporal_context_causal_only")
            and audit["context"].get("same_video_history_masking")
            and not audit["context"].get("uses_centered_or_future_windows")
            and (audit.get("checkpoint_restored") or audit.get("residual_suppressed"))
            and audit.get("eval_mode_scoring")
            for audit in audits
        )
    )
    pca_audit = pca_info["audit"]
    audit_pass = bool(
        ar_identity
        and context_pass
        and pca_audit.get("leakage_audit_pass")
        and pca_audit.get("no_test_rows_used_in_pca_fit")
    )
    result, deltas = compute_verdict(frame, audit_pass)
    result.update(
        {
            "duration_seconds": time.time() - started,
            "accelerator_detail": "Device(gpu, 0)",
            "frozen_ar_identity_pass": ar_identity,
            "context_checkpoint_eval_audit_pass": context_pass,
            "pca_audit_pass": bool(pca_audit.get("leakage_audit_pass")),
        }
    )
    frame.to_csv(root / "metrics/rows.csv", index=False)
    deltas.to_csv(root / "metrics/fold_group_deltas.csv", index=False)
    pd.DataFrame(curves).to_csv(root / "diagnostics/training_curves.csv", index=False)
    fr.write_json(root / "diagnostics/audit.json", {"audit_pass": audit_pass, "ar_identity": ar_identity, "context_pass": context_pass, "pca_audit": pca_audit})
    fr.write_json(root / "metrics/result.json", result)
    fr.write_json(
        root / "manifests/run_manifest.json",
        {
            **manifest,
            "source_root": args.source_root,
            "grouped_pca_root": str(pca_root),
            "residual_target_definition": residual_meta,
            "duration_seconds": result["duration_seconds"],
        },
    )
    report = report_text(result, root)
    stamp = root.name.removeprefix(
        "again_dense_2hz_phase7_continuous_checkpoint_ensemble_grouped_"
    )
    report_name = f"again_dense_2hz_phase7_continuous_checkpoint_ensemble_grouped_{stamp}.md"
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
