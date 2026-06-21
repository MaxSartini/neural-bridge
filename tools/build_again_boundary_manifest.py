#!/usr/bin/env python3
"""Build a boundary-audited 1Hz AGAIN manifest.

This script consumes the all-video boundary audit and the source annotation
manifest proposal. It does not run TRIBE, train models, or modify VEATIC
outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.again_boundary_manifest import run_builder  # noqa: E402


DEFAULT_BOUNDARY_RECOMMENDATIONS = Path(
    "outputs/again_video_boundary_audit_20260621_204520/again_video_boundary_recommendations.csv"
)
DEFAULT_MANIFEST_PROPOSAL = Path("outputs/again_cleaned_inventory_audit_20260621_123531/again_manifest_proposal.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AGAIN 1Hz manifest from audited video boundaries.")
    parser.add_argument("--boundary-recommendations", type=Path, default=DEFAULT_BOUNDARY_RECOMMENDATIONS)
    parser.add_argument("--manifest-proposal", type=Path, default=DEFAULT_MANIFEST_PROPOSAL)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    args = parse_args()
    output_root = args.output_root or Path("outputs") / f"again_boundary_aligned_1hz_manifest_{now_stamp()}"
    manifest = run_builder(
        boundary_recommendations_path=args.boundary_recommendations,
        manifest_proposal_path=args.manifest_proposal,
        output_root=output_root,
    )
    print(
        json.dumps(
            {
                "output_root": manifest["output_root"],
                "alignment_policy": manifest["alignment_policy"],
                "videos_in_manifest": manifest["videos_in_manifest"],
                "manifest_rows": manifest["manifest_rows"],
                "target_feasible_rows_future_spike_1_3s": manifest[
                    "target_feasible_rows_future_spike_1_3s"
                ],
                "tribe_encoding_run": manifest["tribe_encoding_run"],
                "models_trained": manifest["models_trained"],
                "veatic_outputs_modified": manifest["veatic_outputs_modified"],
                "final_benchmark_manifest_created": manifest["final_benchmark_manifest_created"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
