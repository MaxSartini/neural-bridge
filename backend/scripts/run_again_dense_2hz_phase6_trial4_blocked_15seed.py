#!/usr/bin/env python3
"""Stage B: locked trial-4 blocked confirmation over 15 seeds and 8 lanes."""

from __future__ import annotations

import argparse
import gc
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
from backend.scripts import run_again_dense_2hz_phase6_optuna_locked_10seed_confirm as locked10  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase6_robust_multiseed_optuna as robust  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase6_trial4_fresh_seed_validation as fresh  # noqa: E402


SCHEMA_VERSION = "again_dense_2hz_phase6_trial4_blocked_15seed_v1"
SEEDS = tuple(range(20260625, 20260640))
OLD_SEEDS = SEEDS[:10]
FRESH_SEEDS = SEEDS[10:]
CONTROLS = fresh_controls = (
    "real_residual",
    "shuffled_pca_residual",
    "random_pca_residual",
    "label_permutation_residual",
    "train_only_video_mean_residual",
    "diagnostics_only_residual",
)
PRIMARY_CONTROLS = confirm.PRIMARY_CONTROLS
A2_ROOT = REPO_ROOT / "outputs/again_dense_2hz_phase6_trial4_fresh_seed_validation_20260714_145120"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(confirm.SOURCE_ROOT))
    parser.add_argument("--foldsafe-pca-root", default=str(confirm.FOLDSAFE_PCA_ROOT))
    parser.add_argument("--canonical-root", default=str(locked10.CANONICAL_ROOT))
    parser.add_argument("--canonical-metrics", default=str(locked10.CANONICAL_METRICS))
    parser.add_argument("--a2-root", default=str(A2_ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_output_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(f"outputs/again_dense_2hz_phase6_trial4_blocked_15seed_{stamp}")


def load_fresh_ar(a2_root: Path, block: Any, seed: int, batch_size: int) -> dict[str, Any]:
    checkpoint = a2_root / "ar_baseline_checkpoints" / f"seed{seed}__best.npz"
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
        "source": "stage_a2_inner_only_ar_checkpoint_now_scored_for_stage_b",
        "train_score": train_score,
        "train_reg": train_reg,
        "test_score": test_score,
        "test_reg": test_reg,
        "train_checksum": fr.hash_array(train_score),
        "test_checksum": fr.hash_array(test_score),
        "checkpoint_restore_pass": True,
        "eval_mode_scoring_pass": True,
    }


def max_positive_contribution(values: pd.Series) -> float:
    positive = pd.to_numeric(values, errors="coerce")
    positive = positive[positive > 0]
    total = float(positive.sum())
    return math.inf if total <= 0 else float(positive.max() / total)


def compute_verdict(rows: pd.DataFrame, audit_pass: bool) -> tuple[dict[str, Any], pd.DataFrame]:
    pivot = rows.pivot(index="seed", columns="lane", values="pr_auc").reset_index()
    primary_lanes = [f"candidate_{name}" for name in PRIMARY_CONTROLS]
    pivot["best_control_pr_auc"] = pivot[primary_lanes].max(axis=1)
    pivot["candidate_minus_original"] = pivot["candidate_real_residual"] - pivot["original_real_residual"]
    pivot["candidate_minus_ar"] = pivot["candidate_real_residual"] - pivot["frozen_ar_only"]
    pivot["candidate_minus_best_control"] = pivot["candidate_real_residual"] - pivot["best_control_pr_auc"]
    candidate = float(pivot["candidate_real_residual"].mean())
    original = float(pivot["original_real_residual"].mean())
    ar = float(pivot["frozen_ar_only"].mean())
    best_control_means = {lane: float(pivot[lane].mean()) for lane in primary_lanes}
    best_control = max(best_control_means, key=best_control_means.get)
    deltas = pivot["candidate_minus_original"]
    stable = pivot[pivot["seed"] != 20260627]
    stable_deltas = stable["candidate_minus_original"]
    fresh_panel = pivot[pivot["seed"].isin(FRESH_SEEDS)]
    fresh_deltas = fresh_panel["candidate_minus_original"]
    checks = {
        "fresh_mean_candidate_beats_original": float(fresh_deltas.mean()) > 0,
        "fresh_median_candidate_beats_original": float(fresh_deltas.median()) > 0,
        "fresh_paired_wins_at_least_4_of_5": int((fresh_deltas > 0).sum()) >= 4,
        "stable14_mean_candidate_beats_original": float(stable_deltas.mean()) > 0,
        "stable14_median_candidate_beats_original": float(stable_deltas.median()) > 0,
        "stable14_paired_wins_at_least_9_of_14": int((stable_deltas > 0).sum()) >= 9,
        "all15_median_candidate_beats_original": float(deltas.median()) > 0,
        "all15_paired_wins_at_least_10_of_15": int((deltas > 0).sum()) >= 10,
        "delta_vs_ar_at_least_0_003": candidate - ar >= 0.003,
        "delta_vs_best_control_at_least_0_003": candidate - best_control_means[best_control] >= 0.003,
        "positive_vs_ar_at_least_12_of_15": int((pivot["candidate_minus_ar"] > 0).sum()) >= 12,
        "positive_vs_best_control_at_least_12_of_15": int((pivot["candidate_minus_best_control"] > 0).sum()) >= 12,
        "single_seed_contribution_at_most_0_40": max_positive_contribution(deltas) <= 0.40,
        "seed_20260627_retained": 20260627 in set(pivot["seed"]),
        "audit_pass": audit_pass,
    }
    failed = [name for name, value in checks.items() if not value]
    return {
        "schema_version": SCHEMA_VERSION,
        "rows_expected": 120,
        "rows_actual": int(len(rows)),
        "seeds": list(SEEDS),
        "candidate_pr_auc": candidate,
        "original_pr_auc": original,
        "frozen_ar_pr_auc": ar,
        "best_control": best_control,
        "best_control_pr_auc": best_control_means[best_control],
        "candidate_minus_original_mean": float(deltas.mean()),
        "candidate_minus_original_median": float(deltas.median()),
        "candidate_minus_original_trimmed_mean": float(deltas.sort_values().iloc[1:-1].mean()),
        "stable14_candidate_minus_original_mean": float(stable_deltas.mean()),
        "stable14_candidate_minus_original_median": float(stable_deltas.median()),
        "stable14_paired_wins": int((stable_deltas > 0).sum()),
        "fresh5_candidate_minus_original_mean": float(fresh_deltas.mean()),
        "fresh5_candidate_minus_original_median": float(fresh_deltas.median()),
        "fresh5_paired_wins": int((fresh_deltas > 0).sum()),
        "candidate_minus_ar": candidate - ar,
        "candidate_minus_best_control": candidate - best_control_means[best_control],
        "paired_wins_vs_original": int((deltas > 0).sum()),
        "positive_vs_ar": int((pivot["candidate_minus_ar"] > 0).sum()),
        "positive_vs_best_control": int((pivot["candidate_minus_best_control"] > 0).sum()),
        "seed_20260627_delta": float(pivot.loc[pivot["seed"] == 20260627, "candidate_minus_original"].iloc[0]),
        "max_positive_seed_contribution": max_positive_contribution(deltas),
        "checks": checks,
        "failed_gates": failed,
        "stage_b_pass": not failed,
        "stage_c_authorized": not failed,
    }, pivot


def report_text(result: dict[str, Any], output_root: Path) -> str:
    return f"""# Phase 6 Trial 4 Blocked 15-Seed Confirmation — Stage B

Output root: `{output_root}`

- rows: `{result['rows_actual']}/120`
- candidate/original PR-AUC: `{result['candidate_pr_auc']:.10f}` / `{result['original_pr_auc']:.10f}`
- mean / median candidate-minus-original: `{result['candidate_minus_original_mean']:+.10f}` / `{result['candidate_minus_original_median']:+.10f}`
- paired wins vs original: `{result['paired_wins_vs_original']}/15`
- stable 14-seed mean / median / wins: `{result['stable14_candidate_minus_original_mean']:+.10f}` / `{result['stable14_candidate_minus_original_median']:+.10f}` / `{result['stable14_paired_wins']}/14`
- fresh 5-seed mean / median / wins: `{result['fresh5_candidate_minus_original_mean']:+.10f}` / `{result['fresh5_candidate_minus_original_median']:+.10f}` / `{result['fresh5_paired_wins']}/5`
- candidate-minus-AR / best-control: `{result['candidate_minus_ar']:+.10f}` / `{result['candidate_minus_best_control']:+.10f}`
- positives vs AR / best control: `{result['positive_vs_ar']}/15` / `{result['positive_vs_best_control']}/15`
- seed 20260627 candidate-minus-original: `{result['seed_20260627_delta']:+.10f}`
- Stage B pass: `{result['stage_b_pass']}`
- failed gates: `{result['failed_gates']}`

Stage C grouped evaluation is authorized only on a Stage B pass.
"""


def main() -> int:
    args = parse_args()
    dry = {
        "schema_version": SCHEMA_VERSION,
        "seeds": list(SEEDS),
        "fresh_seeds": list(FRESH_SEEDS),
        "lanes_per_seed": 8,
        "rows": 120,
        "candidate_params": fresh.TRIAL4_PARAMS,
        "seed_20260627_retained": True,
        "accelerator": "mlx",
    }
    print(json.dumps(dry, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    output_root = Path(args.output_root) if args.output_root else default_output_root()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(output_root)
    for sub in ("metrics", "diagnostics", "reports", "manifests", "checkpoints", "frozen_ar_scores"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)
    started = time.time()
    pca_root = Path(args.foldsafe_pca_root)
    blocks, df, dense_root, _meta = temporal.build_blocks(Path(args.source_root), pca_root)
    block = temporal.block_for_target(blocks, confirm.TARGET_NAME)
    canonical = locked10.load_canonical_metrics(Path(args.canonical_metrics))
    canonical_original = canonical[canonical["control_type"] == "real_residual"].set_index("seed")
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    ar_manifest: list[dict[str, Any]] = []
    for seed in SEEDS:
        if seed in OLD_SEEDS:
            ar = confirm.load_reused_ar_scores(Path(args.canonical_root), output_root, block, seed)
            if ar is None:
                raise RuntimeError(f"Missing canonical AR for seed {seed}")
        else:
            ar = load_fresh_ar(Path(args.a2_root), block, seed, args.batch_size)
        ar_manifest.append({k: v for k, v in ar.items() if k not in {"train_score", "train_reg", "test_score", "test_reg"}})
        ar_metrics = temporal.metric_row_for_block(block, ar["train_score"], ar["test_score"], ar["test_reg"])
        rows.append({"seed": seed, "lane": "frozen_ar_only", **ar_metrics})
        real_pack = temporal.feature_pack_for(df, dense_root, pca_root, block, confirm.ARCHITECTURE, "real_residual", seed)
        if seed in OLD_SEEDS:
            old = canonical_original.loc[seed].to_dict()
            rows.append({"seed": seed, "lane": "original_real_residual", **old})
        else:
            metrics, lane_curves, audit = temporal.train_temporal_residual(
                architecture=confirm.ARCHITECTURE, control="real_residual", pack=real_pack,
                block=block, ar=ar, seed=seed, output_root=output_root / "original",
                batch_size=args.batch_size, max_epochs=40, patience=8,
                hyperparameters=robust.ORIGINAL_PARAMS,
            )
            curves.extend(lane_curves)
            rows.append({"seed": seed, "lane": "original_real_residual", **audit, **metrics})
        for control in CONTROLS:
            pack = real_pack if control == "real_residual" else temporal.feature_pack_for(
                df, dense_root, pca_root, block, confirm.ARCHITECTURE, control, seed
            )
            metrics, lane_curves, audit = temporal.train_temporal_residual(
                architecture=confirm.ARCHITECTURE, control=control, pack=pack,
                block=block, ar=ar, seed=seed, output_root=output_root / "candidate",
                batch_size=args.batch_size, max_epochs=int(fresh.TRIAL4_PARAMS["max_epochs"]),
                patience=int(fresh.TRIAL4_PARAMS["patience"]), hyperparameters=fresh.TRIAL4_PARAMS,
            )
            curves.extend(lane_curves)
            audits.append({"seed": seed, "control": control, **audit, "context": pack.context_audit})
            rows.append({"seed": seed, "lane": f"candidate_{control}", **audit, **metrics})
            pd.DataFrame(rows).to_csv(output_root / "metrics/blocked_15seed_rows.partial.csv", index=False)
            gc.collect()
    frame = pd.DataFrame(rows)
    audit_pass = bool(
        len(frame) == 120
        and all(row["context"].get("temporal_context_causal_only") for row in audits)
        and not any(row["context"].get("uses_centered_or_future_windows") for row in audits)
        and all(row["context"].get("same_video_history_masking") for row in audits)
        and all(row.get("checkpoint_restored") or row.get("residual_suppressed") for row in audits)
    )
    result, seed_df = compute_verdict(frame, audit_pass)
    result["duration_seconds"] = time.time() - started
    result["accelerator_detail"] = "Device(gpu, 0)"
    frame.to_csv(output_root / "metrics/blocked_15seed_rows.csv", index=False)
    seed_df.to_csv(output_root / "metrics/blocked_15seed_seed_deltas.csv", index=False)
    pd.DataFrame(curves).to_csv(output_root / "diagnostics/training_curves.csv", index=False)
    fr.write_json(output_root / "metrics/result.json", result)
    fr.write_json(output_root / "manifests/ar_manifest.json", {"rows": ar_manifest})
    fr.write_json(output_root / "manifests/run_manifest.json", {**dry, "duration_seconds": result["duration_seconds"], "canonical_rows_reused": 10, "fresh_original_rows_trained": 5})
    report = report_text(result, output_root)
    name = f"again_dense_2hz_phase6_trial4_blocked_15seed_{output_root.name.rsplit('_', 2)[-2]}_{output_root.name.rsplit('_', 1)[-1]}.md"
    (output_root / "reports" / name).write_text(report, encoding="utf-8")
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / name
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"run_completed": True, "output_root": str(output_root), "report": str(report_path), **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
