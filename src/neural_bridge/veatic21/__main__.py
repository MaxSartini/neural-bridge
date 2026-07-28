"""Command-line entrypoint for authorized VEATIC 2.1 actions."""

from __future__ import annotations

import argparse
import json

from neural_bridge.veatic21.phase00 import run_phase00
from neural_bridge.veatic21.phase01 import run_phase01
from neural_bridge.veatic21.phase02 import run_phase02
from neural_bridge.veatic21.phase03 import run_phase03
from neural_bridge.veatic21.phase04 import run_phase04
from neural_bridge.veatic21.phase05 import run_phase05


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("phase00", "phase01", "phase02", "phase03", "phase04", "phase05")
    )
    args = parser.parse_args()
    if args.action == "phase00":
        print(json.dumps(run_phase00(), indent=2, sort_keys=True))
    elif args.action == "phase01":
        print(json.dumps(run_phase01(), indent=2, sort_keys=True))
    elif args.action == "phase02":
        print(json.dumps(run_phase02(), indent=2, sort_keys=True))
    elif args.action == "phase03":
        print(json.dumps(run_phase03(), indent=2, sort_keys=True))
    elif args.action == "phase04":
        print(json.dumps(run_phase04(), indent=2, sort_keys=True))
    elif args.action == "phase05":
        print(json.dumps(run_phase05(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
