"""Create a local validation-record template from a registered benchmark."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "benchmarks" / "benchmark_registry.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create JSONL template records for validation harness scoring.")
    parser.add_argument("--benchmark", required=True, help="Benchmark id from benchmarks/benchmark_registry.json")
    parser.add_argument("--stimulus-id", default="case_001")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    benchmark = next((item for item in registry["benchmarks"] if item["id"] == args.benchmark), None)
    if not benchmark:
        valid = ", ".join(item["id"] for item in registry["benchmarks"])
        raise SystemExit(f"Unknown benchmark id: {args.benchmark}. Valid: {valid}")

    axes = benchmark["observed_axes"]
    conditions = list(benchmark["baseline_conditions"]) + [benchmark["neuro_condition"]]
    records = []
    for condition in conditions:
        records.append({
            "stimulus_id": args.stimulus_id,
            "condition": condition,
            "predicted": {axis: 0.5 for axis in axes},
            "observed": {axis: 0.5 for axis in axes},
            "metadata": {
                "benchmark_id": benchmark["id"],
                "domain": benchmark["domain"],
                "prediction_target": benchmark["prediction_target"],
                "fill_me": "Replace predicted/observed values with normalized 0-1 scores from a real backtest case."
            }
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n", encoding="utf-8")
    print({"created": str(out), "records": len(records), "benchmark": benchmark["id"]})


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "backend"))
    main()
