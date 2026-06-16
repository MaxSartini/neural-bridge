"""Score local neuro-conditioned simulation predictions against outcomes."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.neuro_validation_harness import NeuroValidationHarness  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Score baseline vs neuro-conditioned prediction records.")
    parser.add_argument("records", help="Path to JSON, JSONL, or CSV validation records")
    parser.add_argument("--out", help="Optional output JSON path")
    parser.add_argument("--axes", nargs="+", help="Axes to score")
    parser.add_argument("--baseline", default="baseline", help="Condition to use as baseline")
    args = parser.parse_args()

    harness = NeuroValidationHarness()
    records = harness.load_records(args.records)
    result = harness.score_records(records, axes=args.axes, baseline_condition=args.baseline)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
