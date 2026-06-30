"""Regenerate redesigned-target blocked summary from an existing output root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts import run_again_dense_2hz_phase5_redesigned_target_blocked as run
from backend.scripts import run_again_dense_2hz_phase5_frozen_ar_residual as fr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root")
    parser.add_argument("--reports-dir", default="reports")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    metrics_path = output_root / "metrics" / "redesigned_target_blocked_seed_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {metrics_path}")
    result = run.finalize_output(output_root, Path(args.reports_dir))
    print(json.dumps(fr.clean_json({"output_root": str(output_root), **result}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
