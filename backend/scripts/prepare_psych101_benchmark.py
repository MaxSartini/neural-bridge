"""Prepare held-out Psych-101 next-choice benchmark cases."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.psych101_benchmark import Psych101BenchmarkBuilder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_path")
    parser.add_argument("output_path")
    parser.add_argument("--max-cases", type=int, default=100)
    parser.add_argument(
        "--split",
        choices=["participant_holdout", "experiment_holdout"],
        default="participant_holdout",
    )
    parser.add_argument("--max-context-chars", type=int, default=32000)
    parser.add_argument("--max-cases-per-group", type=int, default=5)
    args = parser.parse_args()
    result = Psych101BenchmarkBuilder(args.max_context_chars).build(
        args.source_path,
        args.output_path,
        args.max_cases,
        args.split,
        args.max_cases_per_group,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
