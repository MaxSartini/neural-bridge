"""Run guarded Brain-JEPA Rust/Metal inference from canonical TRIBE IR."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.brain_jepa_runner import BrainJepaRunner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ir_path")
    parser.add_argument("output_path")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument(
        "--binary",
        default=os.environ.get(
            "BRAIN_JEPA_BINARY",
            "/Volumes/onn. Drive/Neural Bridge/sources/brainjepa-rs/target/release/infer",
        ),
    )
    parser.add_argument(
        "--weights",
        default=os.environ.get(
            "BRAIN_JEPA_WEIGHTS",
            "/Volumes/onn. Drive/Neural Bridge/models/Brain-JEPA/brainjepa.safetensors",
        ),
    )
    parser.add_argument(
        "--gradient",
        default=os.environ.get(
            "BRAIN_JEPA_GRADIENT",
            "/Volumes/onn. Drive/Neural Bridge/sources/brainjepa-rs/data/gradient_mapping_450.csv",
        ),
    )
    parser.add_argument("--allow-incomplete-research-input", action="store_true")
    args = parser.parse_args()

    result = BrainJepaRunner(args.binary, args.weights, args.gradient).run(
        args.ir_path,
        args.output_path,
        args.work_dir,
        args.allow_incomplete_research_input,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
