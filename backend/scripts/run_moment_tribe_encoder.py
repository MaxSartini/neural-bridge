"""Encode a saved TRIBE raw-output archive using MOMENT."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.moment_tribe_encoder import MomentTribeEncoder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="TRIBE raw output NPZ containing predictions")
    parser.add_argument("output", help="Destination NPZ for MOMENT embedding")
    parser.add_argument(
        "--model-dir",
        default=os.environ.get(
            "MOMENT_MODEL_DIR",
            "/Volumes/onn. Drive/Neural Bridge/models/MOMENT-1-small",
        ),
    )
    parser.add_argument("--device", default=os.environ.get("MOMENT_DEVICE", "auto"))
    parser.add_argument("--channels", type=int, default=400)
    args = parser.parse_args()

    metadata = MomentTribeEncoder(args.model_dir, args.device, args.channels).encode_npz(
        args.input,
        args.output,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
