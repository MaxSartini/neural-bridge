#!/usr/bin/env python3
"""Fresh grouped-video control-complete confirmation of original 3-checkpoint ensembles."""

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

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts import run_again_dense_2hz_phase5_frozen_ar_residual as fr  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_learned_heads as base  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_binary_big_confirm as confirm  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_blocked as temporal  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_grouped_compat as grouped  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase6_fixed_blend_fresh5 as fixed  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase6_robust_multiseed_optuna as robust  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase6_trial4_three_checkpoint_fresh15 as ens  # noqa: E402

SCHEMA_VERSION = "again_dense_2hz_phase6_original_three_checkpoint_grouped_confirmation_v1"
SEEDS = tuple(range(20260675, 20260684))
GROUPS = tuple(tuple(SEEDS[i : i + 3]) for i in range(0, 9, 3))
FOLDS = grouped.FOLDS
RESIDUAL_CONTROLS = grouped.RESIDUAL_CONTROLS
PRIMARY_CONTROLS = grouped.PRIMARY_CONTROLS
EXPECTED_MEMBER_ROWS = 315
EXPECTED_ENSEMBLE_ROWS = 105
EXPECTED_ROWS = 420


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(grouped.SOURCE_ROOT))
    parser.add_argument("--grouped-pca-root", default=None)
    parser.add_argument("--external-phase4-root", default=str(grouped.EXTERNAL_PHASE4_ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_output_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(f"outputs/again_dense_2hz_phase6_original_three_checkpoint_grouped_confirmation_{stamp}")


def compute_verdict(rows: pd.DataFrame, audit_pass: bool) -> tuple[dict[str, Any], pd.DataFrame]:
    members = rows[rows.row_type.eq("member")]
    ensembles = rows[rows.row_type.eq("ensemble")]
    pivot = ensembles.pivot(index=["fold", "group"], columns="lane", values="pr_auc").reset_index()
    control_columns = [f"{name}_ensemble" for name in PRIMARY_CONTROLS]
    aggregate_controls = {name: float(pivot[f"{name}_ensemble"].mean()) for name in PRIMARY_CONTROLS}
    best_control = max(aggregate_controls, key=aggregate_controls.get)
    pivot["best_matched_control"] = pivot[control_columns].max(axis=1)
    pivot["real_minus_ar"] = pivot.real_residual_ensemble - pivot.frozen_ar_ensemble
    pivot["real_minus_best_control"] = pivot.real_residual_ensemble - pivot.best_matched_control
    member_real = members[members.lane.eq("real_residual_member")]
    member_means = member_real.groupby(["fold", "group"]).pr_auc.mean()
    pivot["member_real_mean"] = [member_means.loc[(row.fold, row.group)] for row in pivot.itertuples()]
    pivot["real_minus_member_mean"] = pivot.real_residual_ensemble - pivot.member_real_mean
    real = float(pivot.real_residual_ensemble.mean())
    ar = float(pivot.frozen_ar_ensemble.mean())
    best = aggregate_controls[best_control]
    label = float(pivot.label_permutation_residual_ensemble.mean())
    fold_means = pivot.groupby("fold")[["real_minus_ar", "real_minus_best_control"]].mean()
    contribution = fixed.max_positive_contribution(pivot.real_minus_best_control)
    checks = {
        "delta_vs_ar_at_least_0_005": real - ar >= 0.005,
        "delta_vs_best_control_at_least_0_005": real - best >= 0.005,
        "positive_vs_ar_15_of_15": int((pivot.real_minus_ar > 0).sum()) == 15,
        "positive_vs_best_control_15_of_15": int((pivot.real_minus_best_control > 0).sum()) == 15,
        "all_five_fold_means_positive": bool((fold_means > 0).all().all()),
        "positive_paired_medians": float(pivot.real_minus_ar.median()) > 0 and float(pivot.real_minus_best_control.median()) > 0,
        "ensemble_uplift_over_members_at_least_0_001": real - float(member_real.pr_auc.mean()) >= 0.001,
        "ensemble_beats_group_member_mean_at_least_12_of_15": int((pivot.real_minus_member_mean > 0).sum()) >= 12,
        "label_permutation_minus_ar_at_most_0_001": label - ar <= 0.001,
        "single_fold_group_contribution_at_most_0_25": contribution <= 0.25,
        "exact_scope": len(rows) == EXPECTED_ROWS and len(members) == EXPECTED_MEMBER_ROWS and len(ensembles) == EXPECTED_ENSEMBLE_ROWS and set(members.seed) == set(SEEDS) and set(rows.fold) == set(FOLDS),
        "audit_pass": audit_pass,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": SCHEMA_VERSION,
        "rows_actual": int(len(rows)),
        "rows_expected": EXPECTED_ROWS,
        "member_rows": int(len(members)),
        "ensemble_rows": int(len(ensembles)),
        "real_ensemble_pr_auc": real,
        "ar_ensemble_pr_auc": ar,
        "best_control": best_control,
        "best_control_pr_auc": best,
        "real_minus_ar": real - ar,
        "real_minus_best_control": real - best,
        "real_minus_member_mean": real - float(member_real.pr_auc.mean()),
        "wins_vs_ar": int((pivot.real_minus_ar > 0).sum()),
        "wins_vs_best_control": int((pivot.real_minus_best_control > 0).sum()),
        "wins_vs_member_mean": int((pivot.real_minus_member_mean > 0).sum()),
        "positive_fold_means_vs_ar": int((fold_means.real_minus_ar > 0).sum()),
        "positive_fold_means_vs_best_control": int((fold_means.real_minus_best_control > 0).sum()),
        "label_permutation_minus_ar": label - ar,
        "max_positive_fold_group_contribution": contribution,
        "checks": checks,
        "failed_gates": failed,
        "grouped_control_complete_pass": not failed,
    }, pivot


def report_text(result: dict[str, Any], root: Path) -> str:
    return f"""# Phase 6 Original Three-Checkpoint Grouped Confirmation

Output root: `{root}`

- rows: `{result['rows_actual']}/{EXPECTED_ROWS}` (`{result['member_rows']}` member + `{result['ensemble_rows']}` ensemble)
- real / AR / best-control PR-AUC: `{result['real_ensemble_pr_auc']:.10f}` / `{result['ar_ensemble_pr_auc']:.10f}` / `{result['best_control_pr_auc']:.10f}`
- real minus AR / best control / member mean: `{result['real_minus_ar']:+.10f}` / `{result['real_minus_best_control']:+.10f}` / `{result['real_minus_member_mean']:+.10f}`
- wins vs AR / best control / member mean: `{result['wins_vs_ar']}/15` / `{result['wins_vs_best_control']}/15` / `{result['wins_vs_member_mean']}/15`
- positive fold means vs AR / best control: `{result['positive_fold_means_vs_ar']}/5` / `{result['positive_fold_means_vs_best_control']}/5`
- grouped control-complete pass: `{result['grouped_control_complete_pass']}`
- failed gates: `{result['failed_gates']}`
"""


def main() -> int:
    args = parse_args()
    prereg = {
        "schema_version": SCHEMA_VERSION,
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "groups": [list(group) for group in GROUPS],
        "lanes": ["frozen_ar", *RESIDUAL_CONTROLS],
        "rows": EXPECTED_ROWS,
        "params": robust.ORIGINAL_PARAMS,
        "member_selection": False,
        "weight_search": False,
        "accelerator": "mlx",
    }
    print(json.dumps(prereg, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    base.require_mlx()
    root = Path(args.output_root) if args.output_root else default_output_root()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    for sub in ("ar_baseline_checkpoints", "frozen_ar_scores", "metrics", "diagnostics", "reports", "manifests", "models"):
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
    blocks = grouped.build_blocks(df, dense_root, splits)
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    ar_manifest: list[dict[str, Any]] = []
    group_of = {seed: group_id for group_id, group in enumerate(GROUPS, 1) for seed in group}
    for fold in FOLDS:
        block = blocks[fold]
        scores: dict[int, dict[str, dict[str, Any]]] = {}
        for seed in SEEDS:
            ar, ar_curves = confirm.train_ar_baseline(output_root=root, block=block, seed=seed, batch_size=args.batch_size, max_epochs=80, patience=12)
            curves.extend({"fold": fold, "model": "frozen_ar", **item} for item in ar_curves)
            ar_manifest.append({key: value for key, value in ar.items() if key not in {"train_score", "train_reg", "test_score", "test_reg"}})
            scores[seed] = {"frozen_ar": ar}
            ar_metrics = temporal.metric_row_for_block(block, ar["train_score"], ar["test_score"], ar["test_reg"])
            rows.append({"row_type": "member", "fold": fold, "group": group_of[seed], "seed": seed, "lane": "frozen_ar_member", **ar_metrics})
            for control in RESIDUAL_CONTROLS:
                pack = temporal.feature_pack_for(df, dense_root, pca_root, block, confirm.ARCHITECTURE, control, seed)
                train_block = replace(block, target_name=f"{grouped.TARGET_NAME}__grouped_fold{fold}")
                metrics, lane_curves, audit = temporal.train_temporal_residual(
                    architecture=confirm.ARCHITECTURE,
                    control=control,
                    pack=pack,
                    block=train_block,
                    ar=ar,
                    seed=seed,
                    output_root=root / "models",
                    batch_size=args.batch_size,
                    max_epochs=int(robust.ORIGINAL_PARAMS["max_epochs"]),
                    patience=int(robust.ORIGINAL_PARAMS["patience"]),
                    hyperparameters=robust.ORIGINAL_PARAMS,
                )
                curves.extend({"fold": fold, "model": control, **item} for item in lane_curves)
                audits.append({"fold": fold, "seed": seed, "control": control, **audit, "context": pack.context_audit})
                rows.append({"row_type": "member", "fold": fold, "group": group_of[seed], "seed": seed, "lane": f"{control}_member", "frozen_ar_train_checksum": ar["train_checksum"], "frozen_ar_test_checksum": ar["test_checksum"], **metrics})
                scores[seed][control] = fixed.restored_scores(audit=audit, params=robust.ORIGINAL_PARAMS, pack=pack, block=train_block, ar=ar, batch_size=args.batch_size)
            pd.DataFrame(rows).to_csv(root / "metrics/rows.partial.csv", index=False)
        for group_id, members in enumerate(GROUPS, 1):
            for key in ("frozen_ar", *RESIDUAL_CONTROLS):
                averaged = ens.average_scores([scores[seed][key] for seed in members])
                metrics = temporal.metric_row_for_block(block, averaged["train_score"], averaged["test_score"], averaged["test_reg"])
                rows.append({"row_type": "ensemble", "fold": fold, "group": group_id, "seed": 0, "lane": f"{key}_ensemble", "member_seeds": ",".join(map(str, members)), **metrics})
        pd.DataFrame(rows).to_csv(root / "metrics/rows.partial.csv", index=False)
        del scores
        gc.collect()
    frame = pd.DataFrame(rows)
    context_pass = all(a["context"].get("temporal_context_causal_only") and a["context"].get("same_video_history_masking") and not a["context"].get("uses_centered_or_future_windows") for a in audits)
    checkpoint_pass = all(a.get("checkpoint_restored") or a.get("residual_suppressed") for a in audits)
    ar_identity_pass = bool(frame[frame.row_type.eq("member")].groupby(["fold", "seed"])[["frozen_ar_train_checksum", "frozen_ar_test_checksum"]].nunique(dropna=True).fillna(1).le(1).all().all())
    pca_audit = pca_info["audit"]
    audit_pass = bool(len(audits) == 270 and context_pass and checkpoint_pass and ar_identity_pass and pca_audit.get("leakage_audit_pass") and pca_audit.get("no_test_rows_used_in_pca_fit"))
    result, deltas = compute_verdict(frame, audit_pass)
    result.update({"duration_seconds": time.time() - started, "accelerator_detail": "Device(gpu, 0)"})
    frame.to_csv(root / "metrics/rows.csv", index=False)
    deltas.to_csv(root / "metrics/fold_group_deltas.csv", index=False)
    pd.DataFrame(curves).to_csv(root / "diagnostics/training_curves.csv", index=False)
    fr.write_json(root / "diagnostics/audit.json", {"audit_pass": audit_pass, "context_pass": context_pass, "checkpoint_pass": checkpoint_pass, "frozen_ar_identity_pass": ar_identity_pass, "pca_audit": pca_audit})
    fr.write_json(root / "metrics/result.json", result)
    fr.write_json(root / "manifests/ar_baseline_generation_manifest.json", {"baselines": ar_manifest, "newly_trained": len(ar_manifest), "each_fold_seed_uses_own_frozen_ar_score": True})
    fr.write_json(root / "manifests/run_manifest.json", {**prereg, "source_root": args.source_root, "grouped_pca_root": str(pca_root), "residual_target_definition": residual_meta, "duration_seconds": result["duration_seconds"]})
    report = report_text(result, root)
    stamp = root.name.rsplit("_", 2)[-2] + "_" + root.name.rsplit("_", 1)[-1]
    name = f"again_dense_2hz_phase6_original_three_checkpoint_grouped_confirmation_{stamp}.md"
    (root / "reports" / name).write_text(report, encoding="utf-8")
    report_path = Path(args.reports_dir) / name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"run_completed": True, "output_root": str(root), "report": str(report_path), **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
