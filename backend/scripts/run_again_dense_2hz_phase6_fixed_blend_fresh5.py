#!/usr/bin/env python3
"""Preregistered fixed 50/50 original + Trial 4 fresh-five blocked pilot."""

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
from backend.scripts import run_again_dense_2hz_phase6_robust_multiseed_optuna as robust  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase6_trial4_blocked_15seed as blocked15  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase6_trial4_fresh_seed_validation as fresh  # noqa: E402


SCHEMA_VERSION = "again_dense_2hz_phase6_fixed_blend_fresh5_v1"
FRESH_SEEDS = (20260640, 20260641, 20260642, 20260643, 20260644)
BLEND_WEIGHTS = {"original": 0.5, "trial4": 0.5}
EXPECTED_ROWS = 20


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
    return Path(f"outputs/again_dense_2hz_phase6_fixed_blend_fresh5_{stamp}")


def max_positive_contribution(values: pd.Series) -> float:
    positive = pd.to_numeric(values, errors="coerce")
    positive = positive[positive > 0]
    total = float(positive.sum())
    return math.inf if total <= 0 else float(positive.max() / total)


def restored_scores(
    *,
    audit: dict[str, Any],
    params: dict[str, float | int],
    pack: temporal.FeaturePack,
    block: Any,
    ar: dict[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    """Restore one selected checkpoint and return aligned eval-mode scores."""
    if audit.get("residual_suppressed"):
        return {
            "train_score": ar["train_score"].astype(np.float32, copy=True),
            "train_reg": ar["train_reg"].astype(np.float32, copy=True),
            "test_score": ar["test_score"].astype(np.float32, copy=True),
            "test_reg": ar["test_reg"].astype(np.float32, copy=True),
            "checkpoint_restored": False,
            "residual_suppressed": True,
        }
    checkpoint = Path(str(audit["checkpoint_path"]))
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model = temporal.TemporalResidualHead(
        pack.train_x.shape[1],
        confirm.ARCHITECTURE,
        hidden=int(params["hidden"]),
        sequence_window=int(pack.dims.get("sequence_window", 0)),
        sequence_channels=int(pack.dims.get("sequence_channels", 0)),
        alpha_initial_logit=float(params["alpha_initial_logit"]),
        alpha_cap=float(params["alpha_cap"]),
        gate_bias=float(params["gate_bias"]),
    )
    _ = model(
        base.mx.array(pack.train_x[:2], dtype=base.mx.float32),
        base.mx.array(ar["train_score"][:2], dtype=base.mx.float32),
        base.mx.array(ar["train_reg"][:2], dtype=base.mx.float32),
    )
    model.load_weights(str(checkpoint))
    if hasattr(model, "eval"):
        model.eval()
    train_score, train_reg, _ = temporal.forward_residual(
        model,
        pack.train_x,
        ar["train_score"],
        ar["train_reg"],
        batch_size=batch_size,
        target_type=block.target_type,
    )
    test_score, test_reg, _ = temporal.forward_residual(
        model,
        pack.test_x,
        ar["test_score"],
        ar["test_reg"],
        batch_size=batch_size,
        target_type=block.target_type,
    )
    return {
        "train_score": train_score,
        "train_reg": train_reg,
        "test_score": test_score,
        "test_reg": test_reg,
        "checkpoint_restored": True,
        "residual_suppressed": False,
        "checkpoint_checksum": base.file_digest(checkpoint),
    }


def compute_verdict(rows: pd.DataFrame, audit_pass: bool) -> tuple[dict[str, Any], pd.DataFrame]:
    pivot = rows.pivot(index="seed", columns="lane", values="pr_auc").reset_index()
    component_means = {
        "original_real_residual": float(pivot["original_real_residual"].mean()),
        "trial4_real_residual": float(pivot["trial4_real_residual"].mean()),
    }
    stronger = max(component_means, key=component_means.get)
    ensemble = pivot["fixed_50_50_blend"]
    original = pivot["original_real_residual"]
    trial4 = pivot["trial4_real_residual"]
    ar = pivot["frozen_ar_only"]
    pivot["ensemble_minus_original"] = ensemble - original
    pivot["ensemble_minus_trial4"] = ensemble - trial4
    pivot["ensemble_minus_ar"] = ensemble - ar
    pivot["ensemble_minus_stronger_component"] = ensemble - pivot[stronger]
    stds = {
        "ensemble": float(ensemble.std(ddof=0)),
        "original": float(original.std(ddof=0)),
        "trial4": float(trial4.std(ddof=0)),
    }
    min_component_std = min(stds["original"], stds["trial4"])
    checks = {
        "ensemble_mean_exceeds_higher_component_by_0_0005": float(ensemble.mean()) - max(component_means.values()) >= 0.0005,
        "ensemble_median_exceeds_higher_component_median": float(ensemble.median()) > max(float(original.median()), float(trial4.median())),
        "ensemble_beats_original_at_least_3_of_5": int((pivot["ensemble_minus_original"] > 0).sum()) >= 3,
        "ensemble_beats_trial4_at_least_3_of_5": int((pivot["ensemble_minus_trial4"] > 0).sum()) >= 3,
        "ensemble_std_at_least_5pct_below_lower_component_std": stds["ensemble"] <= 0.95 * min_component_std,
        "ensemble_mean_delta_vs_ar_at_least_0_003": float(ensemble.mean() - ar.mean()) >= 0.003,
        "ensemble_positive_vs_ar_at_least_4_of_5": int((pivot["ensemble_minus_ar"] > 0).sum()) >= 4,
        "single_seed_positive_gain_contribution_at_most_0_50": max_positive_contribution(pivot["ensemble_minus_stronger_component"]) <= 0.50,
        "exact_five_fresh_seeds": set(pivot["seed"]) == set(FRESH_SEEDS),
        "exact_twenty_rows": len(rows) == EXPECTED_ROWS,
        "audit_pass": audit_pass,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": SCHEMA_VERSION,
        "rows_expected": EXPECTED_ROWS,
        "rows_actual": int(len(rows)),
        "seeds": list(FRESH_SEEDS),
        "stronger_component_by_mean": stronger,
        "ensemble_pr_auc": float(ensemble.mean()),
        "original_pr_auc": float(original.mean()),
        "trial4_pr_auc": float(trial4.mean()),
        "frozen_ar_pr_auc": float(ar.mean()),
        "ensemble_minus_stronger_component": float(ensemble.mean() - max(component_means.values())),
        "ensemble_minus_original": float(ensemble.mean() - original.mean()),
        "ensemble_minus_trial4": float(ensemble.mean() - trial4.mean()),
        "ensemble_minus_ar": float(ensemble.mean() - ar.mean()),
        "ensemble_wins_vs_original": int((pivot["ensemble_minus_original"] > 0).sum()),
        "ensemble_wins_vs_trial4": int((pivot["ensemble_minus_trial4"] > 0).sum()),
        "ensemble_wins_vs_ar": int((pivot["ensemble_minus_ar"] > 0).sum()),
        "seed_pr_auc_std": stds,
        "ensemble_std_reduction_vs_lower_component": 1.0 - stds["ensemble"] / min_component_std if min_component_std > 0 else -math.inf,
        "max_positive_seed_contribution": max_positive_contribution(pivot["ensemble_minus_stronger_component"]),
        "checks": checks,
        "failed_gates": failed,
        "pilot_pass": not failed,
        "control_complete_confirmation_authorized": not failed,
    }, pivot


def report_text(result: dict[str, Any], output_root: Path) -> str:
    return f"""# Phase 6 Fixed 50/50 Blend — Fresh-Five Blocked Pilot

Output root: `{output_root}`

- rows: `{result['rows_actual']}/{EXPECTED_ROWS}`
- ensemble/original/Trial 4/AR PR-AUC: `{result['ensemble_pr_auc']:.10f}` / `{result['original_pr_auc']:.10f}` / `{result['trial4_pr_auc']:.10f}` / `{result['frozen_ar_pr_auc']:.10f}`
- ensemble minus stronger component: `{result['ensemble_minus_stronger_component']:+.10f}`
- ensemble minus AR: `{result['ensemble_minus_ar']:+.10f}`
- paired wins vs original / Trial 4 / AR: `{result['ensemble_wins_vs_original']}/5` / `{result['ensemble_wins_vs_trial4']}/5` / `{result['ensemble_wins_vs_ar']}/5`
- ensemble seed-standard-deviation reduction vs lower-variance component: `{result['ensemble_std_reduction_vs_lower_component']:+.2%}`
- pilot pass: `{result['pilot_pass']}`
- failed gates: `{result['failed_gates']}`

A pass authorizes only a separately preregistered control-complete confirmation.
The canonical 420-row result is unchanged.
"""


def main() -> int:
    args = parse_args()
    dry = {
        "schema_version": SCHEMA_VERSION,
        "plan": "docs/phase6_fixed_blend_fresh5_pilot_plan.md",
        "fresh_seeds": list(FRESH_SEEDS),
        "rows": EXPECTED_ROWS,
        "lanes": ["frozen_ar_only", "original_real_residual", "trial4_real_residual", "fixed_50_50_blend"],
        "blend_weights": BLEND_WEIGHTS,
        "weight_search": False,
        "viewed_seed_scores_reused": False,
        "matched_controls_deferred_until_pilot_pass": True,
        "accelerator": "mlx",
    }
    print(json.dumps(dry, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    base.require_mlx()
    output_root = Path(args.output_root) if args.output_root else default_output_root()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(output_root)
    for sub in ("ar_baseline_checkpoints", "metrics", "diagnostics", "reports", "manifests", "original", "trial4"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)
    started = time.time()
    pca_root = Path(args.foldsafe_pca_root)
    blocks, df, dense_root, _meta = temporal.build_blocks(Path(args.source_root), pca_root)
    block = temporal.block_for_target(blocks, confirm.TARGET_NAME)
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    score_manifest: list[dict[str, Any]] = []

    for seed in FRESH_SEEDS:
        _inner_ar, ar_curves = fresh.train_ar_inner_only(
            block=block,
            seed=seed,
            output_root=output_root,
            batch_size=args.batch_size,
            max_epochs=80,
            patience=12,
        )
        curves.extend({"model": "frozen_ar", **row} for row in ar_curves)
        ar = blocked15.load_fresh_ar(output_root, block, seed, args.batch_size)
        pack = temporal.feature_pack_for(
            df, dense_root, pca_root, block, confirm.ARCHITECTURE, "real_residual", seed
        )
        ar_metrics = temporal.metric_row_for_block(block, ar["train_score"], ar["test_score"], ar["test_reg"])
        rows.append({"seed": seed, "lane": "frozen_ar_only", **ar_metrics})

        component_scores: dict[str, dict[str, Any]] = {}
        for name, params, root in (
            ("original", robust.ORIGINAL_PARAMS, output_root / "original"),
            ("trial4", fresh.TRIAL4_PARAMS, output_root / "trial4"),
        ):
            metrics, lane_curves, audit = temporal.train_temporal_residual(
                architecture=confirm.ARCHITECTURE,
                control="real_residual",
                pack=pack,
                block=block,
                ar=ar,
                seed=seed,
                output_root=root,
                batch_size=args.batch_size,
                max_epochs=int(params["max_epochs"]),
                patience=int(params["patience"]),
                hyperparameters=params,
            )
            curves.extend({"model": name, **row} for row in lane_curves)
            audits.append({"seed": seed, "model": name, **audit, "context": pack.context_audit})
            rows.append({"seed": seed, "lane": f"{name}_real_residual", **audit, **metrics})
            component_scores[name] = restored_scores(
                audit=audit,
                params=params,
                pack=pack,
                block=block,
                ar=ar,
                batch_size=args.batch_size,
            )

        original = component_scores["original"]
        trial4 = component_scores["trial4"]
        blended = {
            key: 0.5 * original[key] + 0.5 * trial4[key]
            for key in ("train_score", "train_reg", "test_score", "test_reg")
        }
        blend_metrics = temporal.metric_row_for_block(
            block, blended["train_score"], blended["test_score"], blended["test_reg"]
        )
        temporal.add_deltas(blend_metrics, ar_metrics, block.target_type)
        rows.append({
            "seed": seed,
            "lane": "fixed_50_50_blend",
            "blend_original_weight": 0.5,
            "blend_trial4_weight": 0.5,
            "eval_mode_scoring": True,
            **blend_metrics,
        })
        score_manifest.append({
            "seed": seed,
            "row_count": int(len(blended["test_score"])),
            "original_test_score_checksum": fr.hash_array(original["test_score"]),
            "trial4_test_score_checksum": fr.hash_array(trial4["test_score"]),
            "blend_test_score_checksum": fr.hash_array(blended["test_score"]),
            "row_alignment_pass": len(original["test_score"]) == len(trial4["test_score"]) == len(block.test_y),
        })
        pd.DataFrame(rows).to_csv(output_root / "metrics/fixed_blend_rows.partial.csv", index=False)

    frame = pd.DataFrame(rows)
    audit_pass = bool(
        len(frame) == EXPECTED_ROWS
        and len(audits) == 10
        and all(row["context"].get("temporal_context_causal_only") for row in audits)
        and not any(row["context"].get("uses_centered_or_future_windows") for row in audits)
        and all(row["context"].get("same_video_history_masking") for row in audits)
        and all(row.get("checkpoint_restored") or row.get("residual_suppressed") for row in audits)
        and all(row["row_alignment_pass"] for row in score_manifest)
    )
    result, seed_frame = compute_verdict(frame, audit_pass)
    result.update({
        "duration_seconds": time.time() - started,
        "accelerator_detail": "Device(gpu, 0)",
        "blend_weights": BLEND_WEIGHTS,
        "weight_search": False,
        "viewed_seed_scores_reused": False,
    })
    frame.to_csv(output_root / "metrics/fixed_blend_rows.csv", index=False)
    seed_frame.to_csv(output_root / "metrics/fixed_blend_seed_deltas.csv", index=False)
    pd.DataFrame(curves).to_csv(output_root / "diagnostics/training_curves.csv", index=False)
    fr.write_json(output_root / "metrics/result.json", result)
    fr.write_json(output_root / "manifests/score_manifest.json", {"rows": score_manifest})
    fr.write_json(output_root / "manifests/run_manifest.json", {**dry, "duration_seconds": result["duration_seconds"]})
    report = report_text(result, output_root)
    name = f"again_dense_2hz_phase6_fixed_blend_fresh5_{output_root.name.rsplit('_', 2)[-2]}_{output_root.name.rsplit('_', 1)[-1]}.md"
    (output_root / "reports" / name).write_text(report, encoding="utf-8")
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / name
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"run_completed": True, "output_root": str(output_root), "report": str(report_path), **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
