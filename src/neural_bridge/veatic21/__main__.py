from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_bridge.veatic21.benchmark import benchmark_bundle_topologies
from neural_bridge.veatic21.bundle import (
    DEFAULT_BUNDLE_ROOT,
    DEFAULT_TRIBE_ROOT,
    DEFAULT_VJEPA_ROOT,
    assemble_bundle,
    verify_bundle,
)
from neural_bridge.veatic21.phase00 import (
    DEFAULT_PHASE00_ROOT,
    run_phase00,
    verify_phase00,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="VEATIC 2.1 input-bundle tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    assemble = subparsers.add_parser("assemble-input-bundle")
    assemble.add_argument("--tribe-root", type=Path, default=DEFAULT_TRIBE_ROOT)
    assemble.add_argument("--vjepa-root", type=Path, default=DEFAULT_VJEPA_ROOT)
    assemble.add_argument("--output-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    assemble.add_argument("--workers", type=int, default=1)

    verify = subparsers.add_parser("verify-input-bundle")
    verify.add_argument("--output-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    verify.add_argument("--workers", type=int, default=1)

    phase00 = subparsers.add_parser("run-phase00")
    phase00.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    phase00.add_argument("--output-root", type=Path, default=DEFAULT_PHASE00_ROOT)
    phase00.add_argument("--workers", type=int, default=12)
    phase00.add_argument("--registration-path", type=Path, required=True)

    verify_phase = subparsers.add_parser("verify-phase00")
    verify_phase.add_argument("--output-root", type=Path, default=DEFAULT_PHASE00_ROOT)

    benchmark = subparsers.add_parser("benchmark-input-bundle")
    benchmark.add_argument("--workers", type=int, nargs="+", required=True)
    benchmark.add_argument("--repeats", type=int, default=3)

    args = parser.parse_args()
    if args.command == "assemble-input-bundle":
        result = assemble_bundle(
            tribe_root=args.tribe_root,
            vjepa_root=args.vjepa_root,
            output_root=args.output_root,
            workers=args.workers,
        )
    elif args.command == "verify-input-bundle":
        result = verify_bundle(output_root=args.output_root, workers=args.workers)
    elif args.command == "run-phase00":
        result = run_phase00(
            bundle_root=args.bundle_root,
            output_root=args.output_root,
            workers=args.workers,
            registration_path=args.registration_path,
        )
    elif args.command == "verify-phase00":
        result = verify_phase00(output_root=args.output_root)
    else:
        result = benchmark_bundle_topologies(
            worker_counts=tuple(args.workers),
            repeats=args.repeats,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
