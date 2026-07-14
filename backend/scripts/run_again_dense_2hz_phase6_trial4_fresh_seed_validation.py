#!/usr/bin/env python3
"""Stage A2 fresh-seed validation for post-hoc sensitivity trial 4.

The candidate is checksum/literal pinned before evaluation. Only outer-train
rows are used: AR and residual models train/select on inner train/validation,
and no blocked held-out or grouped test arrays are scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from backend.scripts import run_again_dense_2hz_phase5_frozen_ar_residual as fr  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_learned_heads as base  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_binary_big_confirm as confirm  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_blocked as temporal  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase6_optuna_selected_head_pilot as pilot  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase6_robust_multiseed_optuna as robust  # noqa: E402


SCHEMA_VERSION = "again_dense_2hz_phase6_trial4_fresh_seed_validation_v1"
FRESH_SEEDS = (20260635, 20260636, 20260637, 20260638, 20260639)
TRIAL4_PARAMS: dict[str, float | int] = {
    "alpha_cap": 0.08,
    "alpha_initial_logit": -3.0,
    "gate_bias": 4.0,
    "hidden": 96,
    "lambda_binary": 0.8,
    "learning_rate": 0.0002796515604869101,
    "max_epochs": 60,
    "patience": 8,
    "weight_decay": 0.00015731894101489417,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(confirm.SOURCE_ROOT))
    parser.add_argument("--foldsafe-pca-root", default=str(confirm.FOLDSAFE_PCA_ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--ar-max-epochs", type=int, default=80)
    parser.add_argument("--ar-patience", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_output_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(f"outputs/again_dense_2hz_phase6_trial4_fresh_seed_validation_{stamp}")


def train_ar_inner_only(
    *, block: Any, seed: int, output_root: Path, batch_size: int, max_epochs: int, patience: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Train/select frozen AR without ever scoring outer held-out arrays."""
    base.require_mlx()
    base.mx.random.seed(int(seed))
    config = confirm.ar_config(seed, max_epochs=max_epochs, patience=patience, batch_size=batch_size)
    model = base.make_model(config, block.ar_train_x.shape[1], block.ar_block_dims)
    optimizer = base.optim.AdamW(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    rng = np.random.default_rng(int(seed) + 70001)
    checkpoint = output_root / "ar_baseline_checkpoints" / f"seed{seed}__best.npz"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("-inf")
    best_epoch = 0
    stale = 0
    curves: list[dict[str, Any]] = []

    def loss_fn(model_obj: Any, xb: Any, yb: Any, yr: Any) -> Any:
        out = model_obj(xb)
        if out.ndim == 1:
            out = out[:, None]
        reg = base.mx.mean(base.nn.losses.huber_loss(out[:, 0:1], yr, delta=1.0))
        bce = base.mx.mean(base.nn.losses.binary_cross_entropy(out[:, 1:2], yb, with_logits=True))
        return reg + float(config.lambda_binary) * bce

    loss_and_grad = base.nn.value_and_grad(model, loss_fn)
    for epoch in range(1, max_epochs + 1):
        if hasattr(model, "train"):
            model.train()
        order = rng.permutation(block.inner_train)
        for start in range(0, len(order), batch_size):
            rel = order[start : start + batch_size]
            loss, grads = loss_and_grad(
                model,
                base.mx.array(block.ar_train_x[rel], dtype=base.mx.float32),
                base.mx.array(block.train_y[rel].astype(np.float32)[:, None], dtype=base.mx.float32),
                base.mx.array(block.train_cont[rel].astype(np.float32)[:, None], dtype=base.mx.float32),
            )
            grads, _ = base.optim.clip_grad_norm(grads, 1.0)
            optimizer.update(model, grads)
            base.mx.eval(loss, model.parameters(), optimizer.state)
        if hasattr(model, "eval"):
            model.eval()
        val_score, _ = fr.score_existing_model(model, block.ar_train_x[block.inner_val], batch_size)
        val_pr = float(average_precision_score(block.train_y[block.inner_val], val_score))
        curves.append({"seed": seed, "epoch": epoch, "inner_val_pr_auc": val_pr})
        if math.isfinite(val_pr) and val_pr > best_val:
            model.save_weights(str(checkpoint))
            best_val = val_pr
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_weights(str(checkpoint))
    if hasattr(model, "eval"):
        model.eval()
    train_score, train_reg = fr.score_existing_model(model, block.ar_train_x, batch_size)
    return {
        "seed": seed,
        "train_score": train_score,
        "train_reg": train_reg,
        "train_checksum": fr.hash_array(train_score),
        "best_epoch": best_epoch,
        "best_inner_val_pr_auc": best_val,
        "heldout_scored": False,
    }, curves


def report_text(result: dict[str, Any], output_root: Path) -> str:
    gate = result["gate"]
    return f"""# Phase 6 Trial 4 Fresh-Seed Validation — Stage A2

Output root: `{output_root}`

Trial 4 was locked after a post-hoc development sensitivity and evaluated on
five new seeds using inner train/validation only. No blocked held-out or grouped
test array was scored.

- fresh seeds: `{list(FRESH_SEEDS)}`
- paired wins: `{gate['paired_wins']}/5`
- candidate/original mean delta vs AR: `{gate['candidate_mean_delta_vs_ar']:.10f}` / `{gate['original_mean_delta_vs_ar']:.10f}`
- robust-objective gain: `{gate['robust_objective_gain']:+.10f}`
- Stage A2 pass: `{gate['stage_a_pass']}`
- failed gates: `{gate['failed_gates']}`

A pass authorizes only the preregistered 15-seed blocked Stage B, where seed
`20260627` remains included as a stress-test.
"""


def main() -> int:
    args = parse_args()
    dry = {
        "schema_version": SCHEMA_VERSION,
        "candidate": "posthoc_sensitivity_trial_4_locked_before_fresh_seed_evaluation",
        "fresh_seeds": list(FRESH_SEEDS),
        "params": TRIAL4_PARAMS,
        "heldout_scores_read": False,
        "grouped_scores_read": False,
        "seed_20260627_deleted": False,
        "accelerator": "mlx",
    }
    print(json.dumps(dry, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    output_root = Path(args.output_root) if args.output_root else default_output_root()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(output_root)
    for sub in ("ar_baseline_checkpoints", "metrics", "diagnostics", "reports", "manifests"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)
    started = time.time()
    blocks, df, dense_root, _meta = temporal.build_blocks(Path(args.source_root), Path(args.foldsafe_pca_root))
    block = temporal.block_for_target(blocks, confirm.TARGET_NAME)
    real = temporal.feature_pack_for(df, dense_root, Path(args.foldsafe_pca_root), block, confirm.ARCHITECTURE, "real_residual", FRESH_SEEDS[0])
    pack = pilot.InnerPack(real.train_x, real.dims)
    ar_by_seed: dict[int, dict[str, Any]] = {}
    ar_curves: list[dict[str, Any]] = []
    for seed in FRESH_SEEDS:
        ar, curves = train_ar_inner_only(
            block=block,
            seed=seed,
            output_root=output_root,
            batch_size=args.batch_size,
            max_epochs=args.ar_max_epochs,
            patience=args.ar_patience,
        )
        ar_by_seed[seed] = ar
        ar_curves.extend(curves)
    candidate = robust.evaluate_params(TRIAL4_PARAMS, seeds=FRESH_SEEDS, pack=pack, block=block, ar_by_seed=ar_by_seed, batch_size=args.batch_size)
    original = robust.evaluate_params(robust.ORIGINAL_PARAMS, seeds=FRESH_SEEDS, pack=pack, block=block, ar_by_seed=ar_by_seed, batch_size=args.batch_size)
    gate = robust.validation_gate(candidate, original, seeds=FRESH_SEEDS)
    result = {
        **dry,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_deltas": {str(k): v for k, v in candidate.items()},
        "original_deltas": {str(k): v for k, v in original.items()},
        "gate": gate,
        "duration_seconds": time.time() - started,
        "next_stage_authorized": bool(gate["stage_a_pass"]),
        "accelerator_detail": "Device(gpu, 0)",
    }
    pd.DataFrame(ar_curves).to_csv(output_root / "diagnostics/ar_inner_training_curves.csv", index=False)
    (output_root / "metrics/result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "manifests/locked_candidate.json").write_text(json.dumps({"params": TRIAL4_PARAMS, "source_trial": 4, "source_stage": "posthoc_seed27_exclusion_sensitivity", "locked_before_fresh_seed_evaluation": True}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = report_text(result, output_root)
    name = f"again_dense_2hz_phase6_trial4_fresh_seed_validation_{output_root.name.rsplit('_', 2)[-2]}_{output_root.name.rsplit('_', 1)[-1]}.md"
    (output_root / "reports" / name).write_text(report, encoding="utf-8")
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / name
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"run_completed": True, "output_root": str(output_root), "report": str(report_path), **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
