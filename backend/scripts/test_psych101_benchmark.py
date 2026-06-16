"""Smoke-test leakage-aware Psych-101 case preparation."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.psych101_benchmark import Psych101BenchmarkBuilder  # noqa: E402


def main() -> None:
    builder = Psych101BenchmarkBuilder(max_context_chars=1000)
    record = {
        "text": "Choose A or B. You press <<A>>. Then you press <<B>>.",
        "experiment": "experiment",
        "participant": "participant",
    }
    cases = list(builder._cases(record, "participant_holdout"))
    assert [case["observed_choice"] for case in cases] == ["A", "B"]
    assert cases[1]["prompt"].endswith("Then you press ")
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "source.jsonl"
        output = Path(temporary) / "cases.jsonl"
        source.write_text(json.dumps(record) + "\n", encoding="utf-8")
        builder.build(str(source), str(output), max_cases=10, max_cases_per_group=1)
        assert len(output.read_text(encoding="utf-8").splitlines()) <= 1
    print(json.dumps({"psych101_benchmark_builder_ok": True}))


if __name__ == "__main__":
    main()
