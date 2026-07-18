"""Portable command-line entrypoint for zero-label evidence verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evidence import verify_locked_confirmation


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m neural_bridge.zero_label")
    parser.add_argument("root", type=Path, help="Compact locked-confirmation directory")
    parser.add_argument("--external-root", type=Path, help="Optional full canonical run directory")
    parser.add_argument("--registry", type=Path, help="Registry record for --external-root")
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    result = verify_locked_confirmation(
        args.root,
        external_root=args.external_root,
        registry_path=args.registry,
        tolerance=args.tolerance,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verification_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
