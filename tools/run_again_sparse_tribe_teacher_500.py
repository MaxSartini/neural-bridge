#!/usr/bin/env python3
"""Run an AGAIN sparse ViT-G/TRIBE teacher pilot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.scripts.again_sparse_tribe_teacher_500 import (  # noqa: E402
    SparseTeacherConfig,
    external_root,
    now_stamp,
    run_sparse_teacher_500,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", default=now_stamp())
    parser.add_argument("--max-actual-windows", type=int, default=500)
    parser.add_argument("--selector-validation-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--external-cache-root", type=Path, default=None)
    parser.add_argument("--run-label", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    default_run_label = f"again_sparse_tribe_teacher_{args.max_actual_windows}"
    run_label = args.run_label or default_run_label
    run_title = f"AGAIN Sparse TRIBE Teacher {args.max_actual_windows}"
    output_root = args.output_root or Path("outputs") / f"{run_label}_{args.timestamp}"
    cache_root = args.external_cache_root or external_root() / "benchmarks" / "again" / f"{run_label}_{args.timestamp}"
    config_kwargs = {
        "max_actual_windows": args.max_actual_windows,
        "report_date": args.timestamp,
        "run_label": run_label,
        "run_title": run_title,
    }
    if args.selector_validation_root is not None:
        config_kwargs["selector_validation_root"] = args.selector_validation_root
    result = run_sparse_teacher_500(
        output_root=output_root,
        external_cache_root=cache_root,
        config=SparseTeacherConfig(**config_kwargs),
    )
    manifest = result["manifest"]
    runtime = result["runtime_summary"]
    print(f"benchmark_mode={manifest['benchmark_mode']}")
    print(f"output_root={manifest['output_root']}")
    print(f"external_cache_root={manifest['external_cache_root']}")
    print(f"actual_unique_windows_queued={manifest['actual_unique_windows_queued']}")
    print(f"actual_successful_windows={manifest['actual_successful_windows']}")
    print(f"cache_hits={runtime['cache_hits']}")
    print(f"failed_windows={runtime['failed_windows']}")
    print(f"dense_again_vitg_encoding_run={str(manifest['dense_again_vitg_encoding_run']).lower()}")
    print(f"models_trained={str(manifest['models_trained']).lower()}")
    print(f"results_report={manifest['results_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
