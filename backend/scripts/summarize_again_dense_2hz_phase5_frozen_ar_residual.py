"""Summarize the Phase 5 frozen-AR residual design output root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_ROOT = Path("outputs/again_dense_2hz_phase5_frozen_ar_residual_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.output_root)
    gates_path = root / "promotion" / "frozen_ar_residual_gates.json"
    summary_path = root / "metrics" / "frozen_ar_residual_summary_metrics.csv"
    if not gates_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"Missing frozen-AR residual summary artifacts under {root}")
    gates = json.loads(gates_path.read_text())
    summary = pd.read_csv(summary_path)
    print(json.dumps(gates, indent=2, sort_keys=True))
    cols = ["validation_protocol", "model_name", "control_type", "mean_pr_auc", "mean_delta_vs_frozen_ar_pr_auc"]
    present = [c for c in cols if c in summary.columns]
    print(summary.sort_values(["validation_protocol", "mean_pr_auc"], ascending=[True, False])[present].head(40).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
