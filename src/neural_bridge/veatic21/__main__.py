"""Command-line entrypoint for authorized VEATIC 2.1 actions."""

from __future__ import annotations

import argparse
import json

from neural_bridge.veatic21.phase00 import run_phase00


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("phase00",))
    args = parser.parse_args()
    if args.action == "phase00":
        print(json.dumps(run_phase00(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
