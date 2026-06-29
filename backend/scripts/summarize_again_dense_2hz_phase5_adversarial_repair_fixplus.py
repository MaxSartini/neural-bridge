"""Summarize canonical Phase 5 adversarial repair + fix-plus outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


REPAIR_SCHEMA_VERSION = "again_dense_2hz_phase5_adversarial_repair_fixplus_summary_v1"
REAL_CONTROL = "real_ar_pca_diag"
MATCHED_CONTROLS = ("ar_plus_shuffled_pca", "ar_plus_random_pca")


def clean_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def matched_comparisons(summary: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    real_rows = summary[summary["control_type"] == REAL_CONTROL]
    keys = ["target_name", "validation_protocol", "feature_name", "model_head", "loss_name"]
    for _, real in real_rows.iterrows():
        mask = np.ones(len(summary), dtype=bool)
        for key in keys:
            mask &= summary[key].astype(str).to_numpy() == str(real[key])
        controls = summary[mask & summary["control_type"].isin(MATCHED_CONTROLS)].copy()
        if controls.empty:
            continue
        best = controls.sort_values("mean_pr_auc", ascending=False).iloc[0]
        rows.append(
            {
                **{key: real[key] for key in keys},
                "real_control_type": REAL_CONTROL,
                "real_pr_auc": float(real["mean_pr_auc"]),
                "best_matched_control_type": str(best["control_type"]),
                "best_matched_control_pr_auc": float(best["mean_pr_auc"]),
                "real_minus_matched_control_pr_auc": float(real["mean_pr_auc"]) - float(best["mean_pr_auc"]),
                "real_beats_matched_control": bool(float(real["mean_pr_auc"]) > float(best["mean_pr_auc"])),
            }
        )
    return rows


def paired_fold_deltas(fold: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    real = fold[(fold["status"] == "success") & (fold["control_type"] == REAL_CONTROL)]
    keys = ["target_name", "validation_protocol", "feature_name", "model_head", "loss_name", "seed", "fold"]
    for _, real_row in real.iterrows():
        mask = np.ones(len(fold), dtype=bool)
        for key in keys:
            mask &= fold[key].astype(str).to_numpy() == str(real_row[key])
        controls = fold[mask & fold["control_type"].isin(MATCHED_CONTROLS)].copy()
        if controls.empty:
            continue
        best = controls.sort_values("pr_auc", ascending=False).iloc[0]
        rows.append(
            {
                **{key: real_row[key] for key in keys},
                "real_pr_auc": float(real_row["pr_auc"]),
                "best_matched_control_type": str(best["control_type"]),
                "best_matched_control_pr_auc": float(best["pr_auc"]),
                "delta_pr_auc": float(real_row["pr_auc"]) - float(best["pr_auc"]),
            }
        )
    return rows


def summarize(output_root: Path, reports_dir: Path) -> dict[str, Any]:
    metrics = output_root / "metrics"
    promotion = output_root / "promotion"
    summary_path = metrics / "phase5_summary_metrics.csv"
    fold_path = metrics / "phase5_fold_metrics.csv"
    if not summary_path.exists() or not fold_path.exists():
        raise FileNotFoundError(f"Missing repair metrics under {metrics}")
    summary = pd.read_csv(summary_path)
    fold = pd.read_csv(fold_path)
    comparisons = matched_comparisons(summary)
    fold_deltas = paired_fold_deltas(fold)
    comparison_df = pd.DataFrame(comparisons)
    fold_delta_df = pd.DataFrame(fold_deltas)
    comparison_path = metrics / "repair_matched_control_comparisons.csv"
    fold_delta_path = metrics / "repair_paired_fold_deltas.csv"
    comparison_df.to_csv(comparison_path, index=False)
    fold_delta_df.to_csv(fold_delta_path, index=False)
    gate_path = promotion / "promotion_gates.json"
    gates = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else {}
    label_perm = summary[summary["control_type"] == "label_permutation"].sort_values("mean_pr_auc", ascending=True)
    oracle = summary[summary["control_type"] == "video_mean_pca_oracle_diagnostic"].sort_values("mean_pr_auc", ascending=False)
    grouped = comparison_df[comparison_df["validation_protocol"] == "grouped_video"] if not comparison_df.empty else pd.DataFrame()
    blocked = comparison_df[comparison_df["validation_protocol"] == "blocked_temporal_70_30"] if not comparison_df.empty else pd.DataFrame()
    grouped_best = grouped.sort_values("real_minus_matched_control_pr_auc", ascending=False).head(1).to_dict(orient="records")
    blocked_best = blocked.sort_values("real_minus_matched_control_pr_auc", ascending=False).head(1).to_dict(orient="records")
    payload = {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "output_root": str(output_root),
        "corrected_claim": "cross-video future arousal spike / emotional moment ranking"
        if grouped_best and grouped_best[0]["real_minus_matched_control_pr_auc"] > 0
        else "no corrected positive matched-control claim",
        "strict_forward_time_temporal_generalization_proven": bool(
            grouped_best
            and blocked_best
            and grouped_best[0]["real_minus_matched_control_pr_auc"] > 0
            and blocked_best[0]["real_minus_matched_control_pr_auc"] > 0
        ),
        "grouped_best_matched_comparison": grouped_best[0] if grouped_best else None,
        "blocked_best_matched_comparison": blocked_best[0] if blocked_best else None,
        "label_permutation_best": label_perm.head(3).to_dict(orient="records") if not label_perm.empty else [],
        "video_mean_oracle_best": oracle.head(3).to_dict(orient="records") if not oracle.empty else [],
        "corrected_gates": gates,
        "matched_control_comparisons_csv": str(comparison_path),
        "paired_fold_deltas_csv": str(fold_delta_path),
    }
    write_json(output_root / "repair_fixplus_corrected_claim_summary.json", payload)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = output_root.name.rsplit("_", 1)[-1]
    report = reports_dir / f"again_dense_2hz_phase5_adversarial_repair_fixplus_summary_{stamp}.md"
    lines = [
        "# AGAIN Dense 2Hz Phase 5 Adversarial Repair Fix-Plus Summary",
        "",
        f"- Output root: `{output_root}`",
        f"- Corrected claim: `{payload['corrected_claim']}`",
        f"- Strict forward-time temporal generalization proven: `{payload['strict_forward_time_temporal_generalization_proven']}`",
        "",
        "## Matched-Control Result",
    ]
    if grouped_best:
        row = grouped_best[0]
        lines.append(
            f"- Grouped real-minus-matched-control PR-AUC: `{row['real_minus_matched_control_pr_auc']:.5f}` "
            f"(real `{row['real_pr_auc']:.5f}`, control `{row['best_matched_control_type']}` `{row['best_matched_control_pr_auc']:.5f}`)."
        )
    if blocked_best:
        row = blocked_best[0]
        lines.append(
            f"- Blocked real-minus-matched-control PR-AUC: `{row['real_minus_matched_control_pr_auc']:.5f}` "
            f"(real `{row['real_pr_auc']:.5f}`, control `{row['best_matched_control_type']}` `{row['best_matched_control_pr_auc']:.5f}`)."
        )
    lines.extend(
        [
            "",
            "## Repair Notes",
            "",
            "- `holy_shit_pass` is retired.",
            "- Headline effect is real-minus-matched-control.",
            "- Continuous and product-facing metrics are in the fold metrics and within-video metrics files.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload["report"] = str(report)
    write_json(output_root / "repair_fixplus_corrected_claim_summary.json", payload)
    print(json.dumps(clean_json(payload), indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root")
    parser.add_argument("--reports-dir", default="reports")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summarize(Path(args.output_root), Path(args.reports_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
