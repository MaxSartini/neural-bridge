"""Command-line entry point for fresh VEATIC 2.1 work."""

from __future__ import annotations

import argparse
import json

from neural_bridge.veatic21.phase00 import run_phase00, verify_phase00_output
from neural_bridge.veatic21.phase01 import run_phase01, verify_phase01_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Fresh VEATIC 2.1 scientific runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("phase00", help="run the complete fresh Phase 00 audit")
    subparsers.add_parser("verify-phase00", help="verify the sealed Phase 00 outputs")
    subparsers.add_parser("phase01", help="run fresh Phase 01 label alignment")
    subparsers.add_parser("verify-phase01", help="verify the sealed Phase 01 outputs")
    arguments = parser.parse_args()

    if arguments.command == "phase00":
        result = run_phase00()
    elif arguments.command == "phase01":
        result = run_phase01()
    elif arguments.command == "verify-phase01":
        result = verify_phase01_output()
    else:
        result = verify_phase00_output()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
