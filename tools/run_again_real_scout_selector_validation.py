#!/usr/bin/env python3
"""Run real AGAIN V-JEPA 2.1 ViT-B scout selector validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.scripts.again_real_scout_selector_validation import (  # noqa: E402
    RealScoutConfig,
    now_stamp,
    render_report,
    run_validation,
)
from backend.scripts.again_scout_sparse_pipeline import (  # noqa: E402
    default_again_dataset_root,
    default_boundary_manifest_root,
    external_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-videos", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--external-cache-root", type=Path, default=None)
    parser.add_argument("--again-root", type=Path, default=default_again_dataset_root())
    parser.add_argument("--manifest-root", type=Path, default=default_boundary_manifest_root())
    parser.add_argument("--scout-stride-seconds", type=float, default=4.0)
    parser.add_argument("--scout-frame-count", type=int, default=16)
    parser.add_argument("--scout-batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp = now_stamp()
    output_root = args.output_root or Path("outputs") / f"again_real_scout_selector_validation_{stamp}_n{args.limit_videos}"
    external_cache_root = (
        args.external_cache_root
        or external_root() / "benchmarks" / "again" / f"real_scout_selector_validation_{stamp}"
    )
    config = RealScoutConfig(
        limit_videos=args.limit_videos,
        scout_stride_seconds=args.scout_stride_seconds,
        scout_frame_count=args.scout_frame_count,
        scout_batch_size=args.scout_batch_size,
    )
    results = run_validation(
        output_root=output_root,
        external_cache_root=external_cache_root,
        again_root=args.again_root,
        manifest_root=args.manifest_root,
        config=config,
    )
    report_path = Path("reports") / f"again_real_scout_selector_validation_{stamp}_n{args.limit_videos}.md"
    render_report(results, report_path=report_path)
    print(f"output_root={output_root}")
    print(f"external_cache_root={external_cache_root}")
    print(f"report={report_path}")
    print(f"videos={len(results['throughput_rows'])}")
    print("dense_vitg_tribe_run=false")
    print("models_trained=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
