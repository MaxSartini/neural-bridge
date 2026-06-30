"""Repair grouped compatibility verdict for frozen-AR residual label null.

This script reads the completed grouped compatibility CSV/JSON artifacts and
rewrites only repaired verdict/report artifacts. It does not train, score,
generate PCA, or modify the original output root.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts import run_again_dense_2hz_phase5_frozen_ar_residual as fr


SCHEMA_VERSION = "again_dense_2hz_phase5_temporal_residual_grouped_compat_repaired_verdict_v1"
DEFAULT_BUNDLE = Path("evidence_bundle_phase5_temporal_residual_grouped_compat_20260630_033520")
DEFAULT_REPORT = Path("reports/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520_REPAIRED_VERDICT.md")
TARGET = "future_arousal_max_delta_rows_4_10_train_q90"
ARCHITECTURE = "short_temporal_conv_residual"
PROTOCOL = "grouped_video"
PRIMARY_CONTROLS = (
    "shuffled_pca_residual",
    "random_pca_residual",
    "label_permutation_residual",
    "train_only_video_mean_residual",
)
DELTA_THRESHOLD = 0.003
FOLD_SEED_POSITIVE_THRESHOLD = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def write_json(path: Path, obj: Any) -> None:
    fr.write_json(path, obj)


def row_for(summary: pd.DataFrame, control: str) -> pd.Series:
    sub = summary[summary["control_type"] == control]
    if len(sub) != 1:
        raise RuntimeError(f"Expected one summary row for {control}, got {len(sub)}")
    return sub.iloc[0]


def load_inputs(bundle: Path) -> dict[str, Any]:
    metrics = pd.read_csv(bundle / "metrics" / "temporal_residual_grouped_compat_seed_metrics.csv")
    summary = pd.read_csv(bundle / "metrics" / "temporal_residual_grouped_compat_summary_metrics.csv")
    deltas = pd.read_csv(bundle / "metrics" / "temporal_residual_grouped_compat_fold_seed_deltas.csv")
    old_gates = json.loads((bundle / "promotion" / "temporal_residual_grouped_compat_gates.json").read_text(encoding="utf-8"))
    ar_manifest = json.loads((bundle / "manifests" / "ar_baseline_generation_manifest.json").read_text(encoding="utf-8"))
    context_audit = json.loads((bundle / "diagnostics" / "leakage_context_audit.json").read_text(encoding="utf-8"))
    return {
        "metrics": metrics,
        "summary": summary,
        "deltas": deltas,
        "old_gates": old_gates,
        "ar_manifest": ar_manifest,
        "context_audit": context_audit,
    }


def compute_repaired_gates(bundle: Path) -> dict[str, Any]:
    data = load_inputs(bundle)
    metrics: pd.DataFrame = data["metrics"]
    summary: pd.DataFrame = data["summary"]
    deltas: pd.DataFrame = data["deltas"]
    old_gates: dict[str, Any] = data["old_gates"]
    ar_manifest: dict[str, Any] = data["ar_manifest"]
    context_audit: dict[str, Any] = data["context_audit"]

    real = row_for(summary, "real_residual")
    ar = row_for(summary, "frozen_or_ar_only")
    label = row_for(summary, "label_permutation_residual")
    control_rows = [row_for(summary, control) for control in PRIMARY_CONTROLS]
    best_control = max(control_rows, key=lambda row: float(row["mean_pr_auc"]))

    real_pr = float(real["mean_pr_auc"])
    ar_pr = float(ar["mean_pr_auc"])
    label_pr = float(label["mean_pr_auc"])
    best_control_pr = float(best_control["mean_pr_auc"])
    mean_test_prev = float(pd.to_numeric(metrics["test_positive_rate"], errors="coerce").mean())
    legacy_near_chance_threshold = mean_test_prev + 0.02

    control_deltas = {
        f"real_minus_{row['control_type']}_pr_auc": float(real_pr - float(row["mean_pr_auc"]))
        for row in control_rows
    }
    real_minus_ar = real_pr - ar_pr
    real_minus_best_control = real_pr - best_control_pr
    real_minus_label = real_pr - label_pr
    label_minus_ar = label_pr - ar_pr

    positives_vs_best = int((deltas["real_minus_best_control_pr_auc"] > 0).sum())
    positives_vs_ar = int((deltas["real_minus_frozen_or_ar_only_pr_auc"] > 0).sum())
    positives_vs_label = int((deltas["real_minus_label_permutation_residual_pr_auc"] > 0).sum())

    mean_delta_vs_ar_pass = bool(real_minus_ar >= DELTA_THRESHOLD)
    mean_delta_vs_best_control_pass = bool(real_minus_best_control >= DELTA_THRESHOLD)
    real_above_all_primary_controls_pass = bool(all(delta >= DELTA_THRESHOLD for delta in control_deltas.values()))
    fold_seed_consistency_vs_best_control_pass = bool(positives_vs_best >= FOLD_SEED_POSITIVE_THRESHOLD)
    label_real_margin_pass = bool(real_minus_label >= DELTA_THRESHOLD)
    label_fold_seed_pass = bool(positives_vs_label >= FOLD_SEED_POSITIVE_THRESHOLD)
    label_not_above_ar_pass = bool(label_minus_ar < DELTA_THRESHOLD)
    repaired_label_permutation_pass = bool(label_real_margin_pass and label_fold_seed_pass and label_not_above_ar_pass)

    leakage_pass = bool(context_audit.get("leakage_context_audit_pass"))
    frozen_integrity_pass = bool(metrics.groupby(["fold", "seed"])["frozen_ar_test_checksum"].nunique().eq(1).all())
    checkpoint_restore_pass = bool(metrics["checkpoint_restore_pass"].all())
    eval_mode_scoring_pass = bool(metrics["eval_mode_scoring"].all())
    baselines = ar_manifest.get("baselines", [])
    ar_baseline_generation_pass = bool(
        int(ar_manifest.get("newly_trained", -1)) == 50
        and int(ar_manifest.get("reused", -1)) == 0
        and len(baselines) == 50
        and all(row.get("ar_baseline_newly_trained") for row in baselines)
    )

    gate_checks = {
        "mean_delta_vs_ar_pass": mean_delta_vs_ar_pass,
        "mean_delta_vs_best_primary_control_pass": mean_delta_vs_best_control_pass,
        "real_above_all_primary_controls_by_0p003_pass": real_above_all_primary_controls_pass,
        "fold_seed_consistency_vs_best_control_pass": fold_seed_consistency_vs_best_control_pass,
        "frozen_ar_residual_aware_label_permutation_pass": repaired_label_permutation_pass,
        "leakage_context_audit_pass": leakage_pass,
        "frozen_ar_integrity_pass": frozen_integrity_pass,
        "checkpoint_restore_pass": checkpoint_restore_pass,
        "eval_mode_scoring_pass": eval_mode_scoring_pass,
        "ar_baseline_generation_pass": ar_baseline_generation_pass,
    }
    failed = [name for name, ok in gate_checks.items() if not ok]
    grouped_pass = bool(not failed)

    return {
        "schema_version": SCHEMA_VERSION,
        "source_old_schema_version": old_gates.get("schema_version"),
        "verdict_repair_policy": "frozen_ar_residual_aware_label_permutation_null",
        "no_retraining": True,
        "no_new_scoring": True,
        "no_pca_generation": True,
        "no_504_run": True,
        "target": TARGET,
        "protocol": PROTOCOL,
        "architecture": ARCHITECTURE,
        "matrix_rows_actual": int(len(metrics)),
        "matrix_rows_expected": 350,
        "real_pr_auc": real_pr,
        "ar_frozen_pr_auc": ar_pr,
        "best_control": str(best_control["control_type"]),
        "best_control_pr_auc": best_control_pr,
        "label_permutation_pr_auc": label_pr,
        "delta_vs_ar": real_minus_ar,
        "delta_vs_best_control": real_minus_best_control,
        "real_minus_label_permutation_pr_auc": real_minus_label,
        "label_permutation_minus_ar_pr_auc": label_minus_ar,
        "fold_seed_positives_vs_ar": positives_vs_ar,
        "fold_seed_positives_vs_best_control": positives_vs_best,
        "fold_seed_positives_vs_label_permutation": positives_vs_label,
        "control_deltas": control_deltas,
        "legacy_label_permutation_near_chance": {
            "legacy_gate_name": "label_permutation_near_chance_pass",
            "legacy_gate_pass": bool(old_gates.get("label_permutation_near_chance_pass")),
            "legacy_gate_failed": "label_permutation_not_near_chance" in old_gates.get("failed_gates", []),
            "legacy_gate_status_for_frozen_ar_residual": "recorded_inapplicable_not_promotability_blocking",
            "legacy_threshold": legacy_near_chance_threshold,
            "mean_test_positive_rate": mean_test_prev,
            "label_permutation_pr_auc": label_pr,
            "reason_inapplicable": "label_permutation_residual includes the same frozen AR floor and only permutes residual train/inner-val labels",
        },
        "repaired_label_permutation_policy": {
            "real_minus_label_permutation_threshold": DELTA_THRESHOLD,
            "real_minus_label_permutation": real_minus_label,
            "real_minus_label_permutation_pass": label_real_margin_pass,
            "fold_seed_positives_threshold": FOLD_SEED_POSITIVE_THRESHOLD,
            "fold_seed_positives_vs_label_permutation": positives_vs_label,
            "fold_seed_positives_vs_label_permutation_pass": label_fold_seed_pass,
            "label_permutation_minus_ar_must_be_less_than": DELTA_THRESHOLD,
            "label_permutation_minus_ar": label_minus_ar,
            "label_permutation_not_above_ar_by_0p003_pass": label_not_above_ar_pass,
            "frozen_ar_residual_aware_label_permutation_pass": repaired_label_permutation_pass,
        },
        **gate_checks,
        "grouped_compatibility_pass": grouped_pass,
        "failed_gates": failed,
        "recommendation": "grouped_compatibility_pass_review_before_any_504" if grouped_pass else "grouped_compatibility_failed_do_not_run_504",
        "strict_broad_temporal_generalization_proven": False,
        "strict_forward_time_temporal_generalization_proven": False,
    }


def write_report(path: Path, gates: dict[str, Any]) -> None:
    text = f"""# Phase 5 Temporal Residual Grouped Compatibility Repaired Verdict

This report repairs only the grouped compatibility verdict logic for the completed frozen-AR residual grouped run. No training, scoring, PCA generation, grouped rerun, or 504 run was performed.

## Source

- Evidence bundle: `evidence_bundle_phase5_temporal_residual_grouped_compat_20260630_033520/`
- Original output root: `outputs/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520`
- Target: `{gates['target']}`
- Protocol: `{gates['protocol']}`
- Architecture: `{gates['architecture']}`
- Rows: `{gates['matrix_rows_actual']}` / `{gates['matrix_rows_expected']}`

## Original Gate Issue

The original verdict failed only because `label_permutation_near_chance_pass` required label permutation PR-AUC to be less than mean test prevalence plus 0.02. That is inappropriate for this frozen-AR residual design because the label-permutation residual lane includes the same frozen AR floor and only permutes residual train/inner-val labels.

- Legacy near-chance pass: `{gates['legacy_label_permutation_near_chance']['legacy_gate_pass']}`
- Legacy near-chance status: `{gates['legacy_label_permutation_near_chance']['legacy_gate_status_for_frozen_ar_residual']}`
- Label permutation PR-AUC: `{gates['label_permutation_pr_auc']:.10f}`
- Mean test positive rate: `{gates['legacy_label_permutation_near_chance']['mean_test_positive_rate']:.10f}`
- Legacy threshold: `{gates['legacy_label_permutation_near_chance']['legacy_threshold']:.10f}`

## Repaired Label Null

The repaired frozen-AR-residual-aware policy requires:

- real mean PR-AUC beats label permutation by at least `0.003`
- real beats label permutation in at least `40/50` fold-seed comparisons
- label permutation does not beat frozen AR by at least `0.003` mean PR-AUC

Observed:

- Real PR-AUC: `{gates['real_pr_auc']:.10f}`
- AR/frozen PR-AUC: `{gates['ar_frozen_pr_auc']:.10f}`
- Label permutation PR-AUC: `{gates['label_permutation_pr_auc']:.10f}`
- Real - label permutation: `{gates['real_minus_label_permutation_pr_auc']:+.10f}`
- Label permutation - AR: `{gates['label_permutation_minus_ar_pr_auc']:+.10f}`
- Fold-seed positives vs label permutation: `{gates['fold_seed_positives_vs_label_permutation']}/50`
- Repaired label permutation pass: `{gates['frozen_ar_residual_aware_label_permutation_pass']}`

## Repaired Verdict

- Real PR-AUC: `{gates['real_pr_auc']:.10f}`
- AR/frozen PR-AUC: `{gates['ar_frozen_pr_auc']:.10f}`
- Best matched control: `{gates['best_control']}`, PR-AUC `{gates['best_control_pr_auc']:.10f}`
- Delta vs AR/frozen: `{gates['delta_vs_ar']:+.10f}`
- Delta vs best control: `{gates['delta_vs_best_control']:+.10f}`
- Fold-seed positives vs best control: `{gates['fold_seed_positives_vs_best_control']}/50`
- Leakage/context audit pass: `{gates['leakage_context_audit_pass']}`
- Frozen AR integrity pass: `{gates['frozen_ar_integrity_pass']}`
- Checkpoint restore pass: `{gates['checkpoint_restore_pass']}`
- Eval-mode scoring pass: `{gates['eval_mode_scoring_pass']}`
- AR baseline generation pass: `{gates['ar_baseline_generation_pass']}`
- Repaired grouped compatibility pass: `{gates['grouped_compatibility_pass']}`
- Failed repaired gates: `{gates['failed_gates']}`
- Recommendation: `{gates['recommendation']}`

This repaired verdict remains a grouped compatibility result only. It is not a 504 run, not a broad claim change, and strict broad temporal generalization still requires explicit later confirmation.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    bundle = Path(args.evidence_bundle)
    report_path = Path(args.report_path)
    gates = compute_repaired_gates(bundle)
    promotion = bundle / "promotion"
    reports = bundle / "reports"
    write_json(promotion / "temporal_residual_grouped_compat_repaired_gates.json", gates)
    write_json(promotion / "temporal_residual_grouped_compat_repaired_adversarial_verdict.json", gates)
    write_json(
        promotion / "temporal_residual_grouped_compat_repaired_failure_reasons.json",
        {"failed_gates": gates["failed_gates"], "recommendation": gates["recommendation"]},
    )
    bundle_report = reports / "again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520_REPAIRED_VERDICT.md"
    write_report(bundle_report, gates)
    write_report(report_path, gates)
    print(
        json.dumps(
            fr.clean_json(
                {
                    "repaired_grouped_compatibility_pass": gates["grouped_compatibility_pass"],
                    "failed_gates": gates["failed_gates"],
                    "real_pr_auc": gates["real_pr_auc"],
                    "ar_pr_auc": gates["ar_frozen_pr_auc"],
                    "best_control_pr_auc": gates["best_control_pr_auc"],
                    "label_permutation_pr_auc": gates["label_permutation_pr_auc"],
                    "real_minus_label_permutation": gates["real_minus_label_permutation_pr_auc"],
                    "label_permutation_minus_ar": gates["label_permutation_minus_ar_pr_auc"],
                    "fold_seed_positives_vs_label_permutation": gates["fold_seed_positives_vs_label_permutation"],
                    "report": str(report_path),
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
