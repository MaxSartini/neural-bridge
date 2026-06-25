"""Thin runner for trained heads on frozen VEATIC tensor contracts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.scripts import run_veatic_neuro_benchmark as bench  # noqa: E402
from backend.scripts.veatic_frozen_tensor_adapter import SUMMARY_ROOT, FrozenTensorFeatureProvider  # noqa: E402
from backend.scripts.veatic_frozen_tensor_trained_heads import (  # noqa: E402
    TRAINED_BENCHMARK_MODE,
    score_trained_heads,
    trained_head_dry_run_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run trained heads through frozen tensor VEATIC lanes.")
    parser.add_argument("--manifest", default="benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl")
    parser.add_argument("--tensor-root", default=None)
    parser.add_argument("--summary-root", default=str(SUMMARY_ROOT))
    parser.add_argument(
        "--output-dir",
        default=f"outputs/veatic_124_frozen_tensor_trained_heads_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--fresh-run-id", default=None)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without model fitting or scoring.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    provider = FrozenTensorFeatureProvider(
        tensor_root=Path(args.tensor_root).expanduser() if args.tensor_root else None,
        summary_root=Path(args.summary_root).expanduser(),
    )
    fresh_run_id = args.fresh_run_id or f"trained_heads_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.dry_run:
        plan = trained_head_dry_run_plan(provider, fresh_run_id=fresh_run_id)
        print(json.dumps(plan, indent=2))
        if plan["preflight_status"] != "pass":
            raise SystemExit(1)
        return
    all_rows = bench.load_manifest(Path(args.manifest).expanduser())
    manifest = score_trained_heads(
        provider,
        all_rows=all_rows,
        output_dir=Path(args.output_dir).expanduser(),
        fresh_run_id=fresh_run_id,
        seed=args.seed,
    )
    print(json.dumps({"benchmark_mode": TRAINED_BENCHMARK_MODE, **manifest}, indent=2))


if __name__ == "__main__":
    main()
