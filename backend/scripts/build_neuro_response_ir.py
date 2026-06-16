"""Build the canonical neuro-response IR from a TRIBE raw-output archive."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.neuro_response_ir import NeuroResponseIRBuilder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="TRIBE raw-output NPZ")
    parser.add_argument("output_dir", help="Destination directory")
    parser.add_argument("--sampling-hz", type=float, default=2.0)
    args = parser.parse_args()
    result = NeuroResponseIRBuilder(sampling_hz=args.sampling_hz).build_from_npz(
        args.input,
        args.output_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
