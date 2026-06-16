"""Build a masked Brain-JEPA input from canonical neuro-response IR."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.brain_jepa_adapter import BrainJepaAdapter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ir_path")
    parser.add_argument("output_path")
    parser.add_argument("--safetensors-path")
    args = parser.parse_args()
    print(
        json.dumps(
            BrainJepaAdapter().build(args.ir_path, args.output_path, args.safetensors_path),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
