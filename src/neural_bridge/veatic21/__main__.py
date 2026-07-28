"""Command-line entry point for fresh VEATIC 2.1 work."""

from __future__ import annotations

import argparse
import json

from neural_bridge.veatic21.phase00 import run_phase00, verify_phase00_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Fresh VEATIC 2.1 scientific runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("phase00", help="run the complete fresh Phase 00 audit")
    subparsers.add_parser("verify-phase00", help="verify the sealed Phase 00 outputs")
    arguments = parser.parse_args()

    result = run_phase00() if arguments.command == "phase00" else verify_phase00_output()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
