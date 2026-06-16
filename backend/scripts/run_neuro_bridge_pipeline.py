"""Run raw TRIBE output through the guarded neuro-response bridge."""

import argparse
import json
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.neuro_bridge_pipeline import NeuroBridgePipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_tribe_path")
    parser.add_argument("output_dir")
    parser.add_argument("--sampling-hz", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--moment-model-dir",
        default=os.environ.get(
            "MOMENT_MODEL_DIR",
            "/Volumes/onn. Drive/Neural Bridge/models/MOMENT-1-small",
        ),
    )
    args = parser.parse_args()
    result = NeuroBridgePipeline(str(BACKEND_ROOT), args.moment_model_dir).run(
        args.raw_tribe_path,
        args.output_dir,
        args.sampling_hz,
        args.dry_run,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
