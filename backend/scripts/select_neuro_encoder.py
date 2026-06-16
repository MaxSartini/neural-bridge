"""Select a fidelity-compatible neuro-response encoder."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.neuro_encoder_router import NeuroEncoderRouter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ir_path")
    args = parser.parse_args()
    print(json.dumps(NeuroEncoderRouter().inspect(args.ir_path), indent=2))


if __name__ == "__main__":
    main()
