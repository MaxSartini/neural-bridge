#!/usr/bin/env python3
"""Post-hoc convergence diagnostic for the Phase 6 seed-20260627 outlier.

This diagnostic extends only the epoch/patience ceiling for the already-locked
Optuna real-residual configuration. It is explanatory and cannot alter the
preregistered 10-seed verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_binary_big_confirm as confirm  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_blocked as temporal  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase6_optuna_locked_10seed_confirm as locked  # noqa: E402


SEED = 20260627
STANDARD_ROOT = REPO_ROOT / "outputs/again_dense_2hz_phase6_optuna_locked_10seed_confirm_20260714_141457"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_epochs <= 40:
        raise ValueError("Diagnostic must extend beyond the original 40-epoch ceiling")
    params, winner = locked.load_locked_winner(locked.LOCKED_WINNER)
    dry = {
        "posthoc": True,
        "claim_bearing": False,
        "seed": SEED,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "locked_trial": winner["trial_number"],
        "locked_params": params,
        "control": "real_residual",
        "accelerator": "mlx",
    }
    print(json.dumps(dry, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else Path(
        f"outputs/again_dense_2hz_phase6_seed27_convergence_diagnostic_{stamp}"
    )
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(output_root)
    (output_root / "metrics").mkdir(parents=True, exist_ok=True)
    (output_root / "diagnostics").mkdir(parents=True, exist_ok=True)
    (output_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_root / "frozen_ar_scores").mkdir(parents=True, exist_ok=True)

    pca_root = Path(confirm.FOLDSAFE_PCA_ROOT)
    blocks, df, dense_root, _meta = temporal.build_blocks(confirm.SOURCE_ROOT, pca_root)
    block = temporal.block_for_target(blocks, confirm.TARGET_NAME)
    ar = confirm.load_reused_ar_scores(
        locked.CANONICAL_ROOT, output_root, block, SEED
    )
    if ar is None:
        raise RuntimeError("Canonical seed-20260627 frozen AR cache is unavailable")
    pack = temporal.feature_pack_for(
        df, dense_root, pca_root, block, confirm.ARCHITECTURE, "real_residual", SEED
    )
    metrics, curves, audit = temporal.train_temporal_residual(
        architecture=confirm.ARCHITECTURE,
        control="real_residual",
        pack=pack,
        block=block,
        ar=ar,
        seed=SEED,
        output_root=output_root,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        hyperparameters=params,
    )
    standard = pd.read_csv(STANDARD_ROOT / "metrics/locked_10seed_metrics.csv")
    standard_row = standard[
        (standard["seed"] == SEED) & (standard["control_type"] == "real_residual")
    ].iloc[0]
    canonical = locked.load_canonical_metrics(locked.CANONICAL_METRICS)
    original_row = canonical[
        (canonical["seed"] == SEED) & (canonical["control_type"] == "real_residual")
    ].iloc[0]
    result = {
        **dry,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "standard_40_epoch_pr_auc": float(standard_row["pr_auc"]),
        "standard_40_epoch_best_epoch": int(standard_row["best_epoch"]),
        "extended_pr_auc": float(metrics["pr_auc"]),
        "extended_best_epoch": int(audit["best_epoch"]),
        "extended_best_inner_val_delta": float(audit["best_inner_val_delta_vs_frozen_ar"]),
        "extended_minus_standard_pr_auc": float(metrics["pr_auc"] - standard_row["pr_auc"]),
        "extended_minus_original_pr_auc": float(metrics["pr_auc"] - original_row["pr_auc"]),
        "canonical_original_pr_auc": float(original_row["pr_auc"]),
        "hit_epoch_ceiling": int(audit["best_epoch"]) >= args.max_epochs - 2,
        "interpretation_boundary": "posthoc convergence diagnostic; cannot change locked 10-seed verdict",
    }
    pd.DataFrame(curves).to_csv(output_root / "diagnostics/extended_training_curve.csv", index=False)
    (output_root / "metrics/result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_root": str(output_root), **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
