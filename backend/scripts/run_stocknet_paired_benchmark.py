"""Run a small paired StockNet benchmark against the configured local model."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.neuro_validation_harness import NeuroValidationHarness  # noqa: E402
from app.services.stocknet_paired_benchmark import StockNetPairedBenchmark  # noqa: E402


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, rows: list) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="benchmarks/data/stocknet/train.parquet")
    parser.add_argument("--cases", type=int, default=3)
    parser.add_argument("--min-tweets", type=int, default=2)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sampling", choices=["spaced", "sequential"], default="spaced")
    parser.add_argument("--run-id", help="Stable run name. Reusing it resumes completed cases.")
    parser.add_argument("--out-dir", default="benchmarks/results")
    args = parser.parse_args()

    runner = StockNetPairedBenchmark()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / (args.run_id or f"stocknet_paired_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    audit_path = out_dir / "audit.jsonl"
    existing_records = load_jsonl(records_path)
    completed = {
        row["stimulus_id"]
        for row in existing_records
        if row.get("condition") == "neuro_conditioned"
    }
    cases = runner.load_cases(
        args.dataset,
        count=args.cases,
        min_tweets=args.min_tweets,
        holdout_fraction=args.holdout_fraction,
        offset=args.offset,
        sampling=args.sampling,
    )
    for index, case in enumerate(cases, start=1):
        if case.stimulus_id in completed:
            print(f"[{index}/{len(cases)}] skip completed {case.stimulus_id}", flush=True)
            continue
        print(f"[{index}/{len(cases)}] run {case.stimulus_id}", flush=True)
        result = runner.run([case])
        append_jsonl(records_path, result["records"])
        append_jsonl(audit_path, result["cases"])
        existing_records.extend(result["records"])
        score = NeuroValidationHarness().score_records(
            [NeuroValidationHarness()._record_from_dict(row) for row in existing_records],
            axes=["probability_up"],
            baseline_condition="llm_only",
        )
        (out_dir / "score.json").write_text(json.dumps(score, indent=2), encoding="utf-8")
    score = NeuroValidationHarness().score_records(
        [NeuroValidationHarness()._record_from_dict(row) for row in load_jsonl(records_path)],
        axes=["probability_up"],
        baseline_condition="llm_only",
    )
    print(json.dumps({"output_dir": str(out_dir), "score": score}, indent=2))


if __name__ == "__main__":
    main()
