"""Regenerate blocked continuous residual summary from an existing output root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts import run_again_dense_2hz_phase5_continuous_residual_blocked as run
from backend.scripts import run_again_dense_2hz_phase5_frozen_ar_residual as fr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root")
    parser.add_argument("--reports-dir", default="reports")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    fold_path = output_root / "metrics" / "continuous_residual_blocked_fold_metrics.csv"
    if not fold_path.exists():
        raise FileNotFoundError(f"Missing fold metrics: {fold_path}")
    fold_df = pd.read_csv(fold_path)
    summary = run.summarize(fold_df)
    seed_deltas = run.seed_delta_table(fold_df)
    gates = run.compute_gates(summary, fold_df)
    summary.to_csv(output_root / "metrics" / "continuous_residual_blocked_summary_metrics.csv", index=False)
    seed_deltas.to_csv(output_root / "metrics" / "continuous_residual_blocked_seed_deltas.csv", index=False)
    summary.to_csv(output_root / "promotion" / "continuous_residual_blocked_control_comparison.csv", index=False)
    seed_deltas.to_csv(output_root / "promotion" / "continuous_residual_blocked_seed_deltas.csv", index=False)
    fr.write_json(output_root / "promotion" / "continuous_residual_blocked_gates.json", gates)
    fr.write_json(output_root / "promotion" / "continuous_residual_blocked_adversarial_verdict.json", gates)
    fr.write_json(
        output_root / "promotion" / "continuous_residual_blocked_failure_reasons.json",
        {
            "continuous_residual_pass": gates["continuous_residual_pass"],
            "credible_continuous_pass": gates["credible_continuous_pass"],
            "failed_gates": [k for k, v in gates.items() if k.endswith("_pass") and v is False],
            "recommendation": gates["recommendation"],
        },
    )
    stamp = output_root.name.replace("again_dense_2hz_phase5_continuous_residual_blocked_", "")
    report_name = f"again_dense_2hz_phase5_continuous_residual_blocked_summary_{stamp}.md"
    run.write_report(output_root / "reports" / report_name, gates, output_root)
    report_path = Path(args.reports_dir) / report_name
    run.write_report(report_path, gates, output_root)
    print(json.dumps(fr.clean_json({"output_root": str(output_root), "report": str(report_path), **gates}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
