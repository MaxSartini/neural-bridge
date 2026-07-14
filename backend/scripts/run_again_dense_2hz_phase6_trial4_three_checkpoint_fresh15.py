#!/usr/bin/env python3
"""Fresh-15 larger retraining with five locked three-checkpoint ensembles."""

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
from backend.scripts import run_again_dense_2hz_phase6_trial4_blocked_15seed as blocked15  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase6_trial4_fresh_seed_validation as fresh  # noqa: E402

SCHEMA_VERSION = "again_dense_2hz_phase6_trial4_three_checkpoint_fresh15_v1"
SEEDS = tuple(range(20260645, 20260660))
GROUPS = tuple(tuple(SEEDS[i : i + 3]) for i in range(0, 15, 3))
EXPECTED_ROWS = 60


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", default=str(confirm.SOURCE_ROOT))
    p.add_argument("--foldsafe-pca-root", default=str(confirm.FOLDSAFE_PCA_ROOT))
    p.add_argument("--output-root", default=None)
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def default_output_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(f"outputs/again_dense_2hz_phase6_trial4_three_checkpoint_fresh15_{stamp}")


def average_scores(items: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if len(items) != 3:
        raise ValueError("Exactly three checkpoints are required")
    keys = ("train_score", "train_reg", "test_score", "test_reg")
    lengths = {key: {len(item[key]) for item in items} for key in keys}
    if any(len(values) != 1 for values in lengths.values()):
        raise ValueError("Checkpoint score rows are not aligned")
    return {key: np.mean(np.stack([item[key] for item in items]), axis=0).astype(np.float32) for key in keys}


def compute_verdict(rows: pd.DataFrame, audit_pass: bool) -> tuple[dict[str, Any], pd.DataFrame]:
    members = rows[rows["row_type"] == "member"]
    groups = rows[rows["row_type"] == "ensemble"]
    pivot = groups.pivot(index="group", columns="lane", values="pr_auc").reset_index()
    pivot["trial4_minus_original"] = pivot["trial4_checkpoint_ensemble"] - pivot["original_checkpoint_ensemble"]
    pivot["trial4_minus_ar"] = pivot["trial4_checkpoint_ensemble"] - pivot["ar_checkpoint_ensemble"]
    member_trial = members[members["lane"] == "trial4_member"]
    group_member_mean = member_trial.groupby("group")["pr_auc"].mean()
    pivot["trial4_member_mean"] = pivot["group"].map(group_member_mean)
    pivot["trial4_ensemble_minus_member_mean"] = pivot["trial4_checkpoint_ensemble"] - pivot["trial4_member_mean"]
    candidate = pivot["trial4_checkpoint_ensemble"]
    original = pivot["original_checkpoint_ensemble"]
    ar = pivot["ar_checkpoint_ensemble"]
    member_std = float(member_trial["pr_auc"].std(ddof=0))
    ensemble_std = float(candidate.std(ddof=0))
    contribution = fixed.max_positive_contribution(pivot["trial4_minus_original"])
    checks = {
        "candidate_mean_exceeds_original_by_0_0005": float(candidate.mean() - original.mean()) >= 0.0005,
        "candidate_median_exceeds_original": float(candidate.median()) > float(original.median()),
        "candidate_beats_original_at_least_4_of_5": int((pivot["trial4_minus_original"] > 0).sum()) >= 4,
        "candidate_mean_exceeds_member_mean_by_0_0005": float(candidate.mean() - member_trial["pr_auc"].mean()) >= 0.0005,
        "candidate_beats_group_member_mean_at_least_4_of_5": int((pivot["trial4_ensemble_minus_member_mean"] > 0).sum()) >= 4,
        "candidate_std_at_least_20pct_below_member_std": ensemble_std <= 0.80 * member_std,
        "candidate_mean_delta_vs_ar_at_least_0_003": float(candidate.mean() - ar.mean()) >= 0.003,
        "candidate_positive_vs_ar_5_of_5": int((pivot["trial4_minus_ar"] > 0).sum()) == 5,
        "single_group_contribution_at_most_0_50": contribution <= 0.50,
        "exact_scope": len(rows) == EXPECTED_ROWS and set(members["seed"]) == set(SEEDS) and len(pivot) == 5,
        "audit_pass": audit_pass,
    }
    failed = [k for k, v in checks.items() if not v]
    return {
        "schema_version": SCHEMA_VERSION,
        "rows_actual": int(len(rows)),
        "rows_expected": EXPECTED_ROWS,
        "candidate_pr_auc": float(candidate.mean()),
        "original_ensemble_pr_auc": float(original.mean()),
        "ar_ensemble_pr_auc": float(ar.mean()),
        "candidate_minus_original": float(candidate.mean() - original.mean()),
        "candidate_minus_member_mean": float(candidate.mean() - member_trial["pr_auc"].mean()),
        "candidate_minus_ar": float(candidate.mean() - ar.mean()),
        "wins_vs_original": int((pivot["trial4_minus_original"] > 0).sum()),
        "wins_vs_member_mean": int((pivot["trial4_ensemble_minus_member_mean"] > 0).sum()),
        "wins_vs_ar": int((pivot["trial4_minus_ar"] > 0).sum()),
        "trial4_member_std": member_std,
        "candidate_group_std": ensemble_std,
        "std_reduction": 1.0 - ensemble_std / member_std if member_std else -math.inf,
        "max_positive_group_contribution": contribution,
        "checks": checks,
        "failed_gates": failed,
        "pilot_pass": not failed,
        "control_complete_confirmation_authorized": not failed,
    }, pivot


def report_text(r: dict[str, Any], root: Path) -> str:
    return f"""# Phase 6 Trial-4 Three-Checkpoint Fresh-15 Result

Output root: `{root}`

- rows: `{r['rows_actual']}/{EXPECTED_ROWS}`
- Trial-4/original/AR ensemble PR-AUC: `{r['candidate_pr_auc']:.10f}` / `{r['original_ensemble_pr_auc']:.10f}` / `{r['ar_ensemble_pr_auc']:.10f}`
- candidate minus original/member mean/AR: `{r['candidate_minus_original']:+.10f}` / `{r['candidate_minus_member_mean']:+.10f}` / `{r['candidate_minus_ar']:+.10f}`
- wins vs original/member mean/AR: `{r['wins_vs_original']}/5` / `{r['wins_vs_member_mean']}/5` / `{r['wins_vs_ar']}/5`
- checkpoint-ensemble standard-deviation reduction: `{r['std_reduction']:+.2%}`
- pass: `{r['pilot_pass']}`
- failed gates: `{r['failed_gates']}`
"""


def main() -> int:
    args = parse_args()
    dry = {"schema_version": SCHEMA_VERSION, "seeds": list(SEEDS), "groups": [list(g) for g in GROUPS], "rows": EXPECTED_ROWS, "candidate_params": fresh.TRIAL4_PARAMS, "original_params": robust.ORIGINAL_PARAMS, "member_selection": False, "heldout_weight_search": False, "accelerator": "mlx"}
    print(json.dumps(dry, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    base.require_mlx()
    root = Path(args.output_root) if args.output_root else default_output_root()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    for sub in ("ar_baseline_checkpoints", "metrics", "diagnostics", "reports", "manifests", "original", "trial4"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    started = time.time()
    pca_root = Path(args.foldsafe_pca_root)
    blocks, df, dense_root, _ = temporal.build_blocks(Path(args.source_root), pca_root)
    block = temporal.block_for_target(blocks, confirm.TARGET_NAME)
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    scores: dict[int, dict[str, dict[str, Any]]] = {}
    group_of = {seed: i + 1 for i, group in enumerate(GROUPS) for seed in group}
    for seed in SEEDS:
        _, ar_curves = fresh.train_ar_inner_only(block=block, seed=seed, output_root=root, batch_size=args.batch_size, max_epochs=80, patience=12)
        curves.extend({"model": "ar", **x} for x in ar_curves)
        ar = blocked15.load_fresh_ar(root, block, seed, args.batch_size)
        pack = temporal.feature_pack_for(df, dense_root, pca_root, block, confirm.ARCHITECTURE, "real_residual", seed)
        ar_metrics = temporal.metric_row_for_block(block, ar["train_score"], ar["test_score"], ar["test_reg"])
        rows.append({"row_type": "member", "group": group_of[seed], "seed": seed, "lane": "ar_member", **ar_metrics})
        scores[seed] = {"ar": ar}
        for name, params, output in (("original", robust.ORIGINAL_PARAMS, root / "original"), ("trial4", fresh.TRIAL4_PARAMS, root / "trial4")):
            metrics, lane_curves, audit = temporal.train_temporal_residual(architecture=confirm.ARCHITECTURE, control="real_residual", pack=pack, block=block, ar=ar, seed=seed, output_root=output, batch_size=args.batch_size, max_epochs=int(params["max_epochs"]), patience=int(params["patience"]), hyperparameters=params)
            curves.extend({"model": name, **x} for x in lane_curves)
            audits.append({"seed": seed, "model": name, **audit, "context": pack.context_audit})
            rows.append({"row_type": "member", "group": group_of[seed], "seed": seed, "lane": f"{name}_member", **metrics})
            scores[seed][name] = fixed.restored_scores(audit=audit, params=params, pack=pack, block=block, ar=ar, batch_size=args.batch_size)
        pd.DataFrame(rows).to_csv(root / "metrics/rows.partial.csv", index=False)
    for group_id, group in enumerate(GROUPS, 1):
        for name, lane in (("ar", "ar_checkpoint_ensemble"), ("original", "original_checkpoint_ensemble"), ("trial4", "trial4_checkpoint_ensemble")):
            avg = average_scores([scores[s][name] for s in group])
            metrics = temporal.metric_row_for_block(block, avg["train_score"], avg["test_score"], avg["test_reg"])
            rows.append({"row_type": "ensemble", "group": group_id, "seed": 0, "lane": lane, "member_seeds": ",".join(map(str, group)), **metrics})
    frame = pd.DataFrame(rows)
    audit_pass = bool(len(audits) == 30 and all(a["context"].get("temporal_context_causal_only") and a["context"].get("same_video_history_masking") and not a["context"].get("uses_centered_or_future_windows") and (a.get("checkpoint_restored") or a.get("residual_suppressed")) for a in audits))
    result, group_frame = compute_verdict(frame, audit_pass)
    result.update({"duration_seconds": time.time() - started, "accelerator_detail": "Device(gpu, 0)"})
    frame.to_csv(root / "metrics/rows.csv", index=False)
    group_frame.to_csv(root / "metrics/group_deltas.csv", index=False)
    pd.DataFrame(curves).to_csv(root / "diagnostics/training_curves.csv", index=False)
    fr.write_json(root / "metrics/result.json", result)
    fr.write_json(root / "manifests/run_manifest.json", {**dry, "duration_seconds": result["duration_seconds"]})
    report = report_text(result, root)
    name = f"again_dense_2hz_phase6_trial4_three_checkpoint_fresh15_{root.name.rsplit('_', 2)[-2]}_{root.name.rsplit('_', 1)[-1]}.md"
    (root / "reports" / name).write_text(report, encoding="utf-8")
    report_path = Path(args.reports_dir) / name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"run_completed": True, "output_root": str(root), "report": str(report_path), **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
