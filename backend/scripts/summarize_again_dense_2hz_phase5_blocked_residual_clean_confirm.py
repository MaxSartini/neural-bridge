"""Regenerate clean blocked residual confirmation gates/report from an output root.

This summarizer is no-training only. It reads existing clean-confirm metrics and
writes promotion JSON plus the tracked-style report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts import run_again_dense_2hz_phase5_blocked_residual_clean_confirm as run
from backend.scripts import run_again_dense_2hz_phase5_frozen_ar_residual as fr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root")
    parser.add_argument("--design-path", default=str(run.DESIGN_PATH))
    parser.add_argument("--reports-dir", default="reports")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    fold_path = output_root / "metrics" / "blocked_residual_clean_confirm_fold_metrics.csv"
    if not fold_path.exists():
        raise FileNotFoundError(f"Missing clean-confirm fold metrics: {fold_path}")
    design = json.loads(Path(args.design_path).read_text())
    fold_df = pd.read_csv(fold_path)
    summary = run.summarize(fold_df)
    seed_deltas = run.seed_delta_table(fold_df)
    gates = run.compute_gates(summary, fold_df, design)
    summary.to_csv(output_root / "metrics" / "blocked_residual_clean_confirm_summary_metrics.csv", index=False)
    seed_deltas.to_csv(output_root / "metrics" / "blocked_residual_clean_confirm_seed_deltas.csv", index=False)
    summary.to_csv(output_root / "promotion" / "blocked_residual_clean_confirm_control_comparison.csv", index=False)
    seed_deltas.to_csv(output_root / "promotion" / "blocked_residual_clean_confirm_seed_deltas.csv", index=False)
    fr.write_json(output_root / "promotion" / "blocked_residual_clean_confirm_gates.json", gates)
    fr.write_json(output_root / "promotion" / "blocked_residual_clean_confirm_adversarial_verdict.json", gates)
    fr.write_json(
        output_root / "promotion" / "blocked_residual_clean_confirm_failure_reasons.json",
        {
            "weak_confirmation_pass": gates["weak_confirmation_pass"],
            "credible_confirmation_pass": gates["credible_confirmation_pass"],
            "strict_forward_time_temporal_generalization_proven": gates["strict_forward_time_temporal_generalization_proven"],
            "failed_primary_gates": [k for k, v in gates.items() if k.endswith("_pass") and v is False],
            "recommendation": gates["recommendation"],
        },
    )
    report_name = f"again_dense_2hz_phase5_blocked_residual_clean_confirm_{output_root.name.rsplit('_', 2)[-2]}_{output_root.name.rsplit('_', 1)[-1]}.md"
    run.write_report(output_root / "reports" / report_name, gates, output_root)
    report_path = Path(args.reports_dir) / report_name
    run.write_report(report_path, gates, output_root)
    print(json.dumps(fr.clean_json({"output_root": str(output_root), "report": str(report_path), **gates}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
