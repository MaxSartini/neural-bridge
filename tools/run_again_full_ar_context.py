#!/usr/bin/env python3
"""Run the full-AGAIN AR-only context baseline."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.scripts.again_full_ar_context import DEFAULT_MANIFEST_PATH, FullArContextConfig, run_full_ar_context  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-AGAIN AR-only context baseline.")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root or Path("outputs") / f"again_full_ar_context_{args.timestamp}"
    manifest = run_full_ar_context(
        output_root=output_root,
        config=FullArContextConfig(
            manifest_path=args.manifest_path,
            report_date=args.timestamp,
            n_splits=args.n_splits,
            ridge_alpha=args.ridge_alpha,
        ),
    )
    print(
        json.dumps(
            {
                "benchmark_mode": manifest["benchmark_mode"],
                "output_root": manifest["output_root"],
                "manifest_videos": manifest["manifest_videos"],
                "manifest_rows": manifest["manifest_rows"],
                "direct_sparse_pca128_comparison_made": manifest["direct_sparse_pca128_comparison_made"],
                "tribe_encoding_run": manifest["tribe_encoding_run"],
                "veatic_outputs_modified": manifest["veatic_outputs_modified"],
                "report_path": manifest["report_path"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
