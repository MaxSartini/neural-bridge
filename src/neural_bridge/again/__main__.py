"""Portable command-line entrypoint for the canonical AGAIN package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from .evidence import EvidenceName, verify_evidence
from .replay import ReplayRoots, ReplaySpec, replay


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m neural_bridge.again")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-evidence")
    verify.add_argument("endpoint", choices=("phase5-selected", "phase7-blocked", "phase7-grouped"))
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--tolerance", type=float, default=1e-12)
    replay_parser = subparsers.add_parser("replay-checkpoint")
    replay_parser.add_argument("--spec", type=Path, required=True)
    replay_parser.add_argument("--dense-root", type=Path, required=True)
    replay_parser.add_argument("--pca-root", type=Path, required=True)
    replay_parser.add_argument("--run-root", type=Path, required=True)
    replay_parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    if args.command == "verify-evidence":
        result = verify_evidence(
            cast(EvidenceName, args.endpoint), args.root, tolerance=args.tolerance
        )
        passed = bool(result["verification_pass"])
    else:
        result = replay(
            ReplaySpec.load(args.spec),
            ReplayRoots(args.dense_root, args.pca_root, args.run_root),
        )
        maximum_difference = cast(float, result["max_absolute_difference"])
        passed = maximum_difference <= args.tolerance
        result["verification_pass"] = passed
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
